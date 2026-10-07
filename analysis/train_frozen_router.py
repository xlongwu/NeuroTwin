# coding=utf-8
"""S3：冻结专家库的路由可学习性检验（RECC 执行计划 §7）。

固定 backbone / shared 分支 / decoder / 全部专家与原 Router，仅训练一个小型
Router：输入为**推理时可得**的全脑状态描述符（Router 原输入 [μ_t,σ_t,μ_g,σ_g]
经 LayerNorm，4D+C 维）与病理条件，目标为逐样本 competence 分布 q（Huber 口径，
stop-gradient，与训练监督同口径）。校准/验证按**被试级不相交**二分（seed 固定），
避免训练集乐观 competence（§7.3）。

对照策略（验证半区，subject 聚合 MAE）：
  - 固定专家（校准半区选择）、均匀混合、原 Router（mixture）、新 Router、
    单专家 oracle / 连续 pair oracle（事后上界）。

使用边界：本检验的前提是 S2/G2 判定“存在有实践意义的可行收益空间”（§7）。
当前专家库 G2 未通过时，本脚本仅作方法储备与吞吐验证，不投入完整研究。

用法::

    python -m analysis.train_frozen_router --ckpt <finetuned_best.pt> \
        --out_dir analysis_results/recc_repair_20261007/router_smoke \
        [--limit_subjects 8] [--epochs 300] [--hidden 32]
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.evaluate_variant import build_model_from_args, load_train_args
from utils.common import set_seed


class SmallRouter(nn.Module):
    """冻结库 Router：LayerNorm 输入 → Linear → GELU → Linear → E logits。"""

    def __init__(self, in_dim: int, num_experts: int, hidden: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, num_experts))

    def forward(self, x):
        return self.net(self.norm(x))


@torch.no_grad()
def collect_features(model, data_loader, device, huber_delta: float,
                     limit_subjects: int = 0):
    """收集 (router 输入特征, 逐专家 Huber, subject)。

    特征取原 Router 的 input_norm 输出（[μ_t,σ_t,μ_g,σ_g] ⊕ cond 经 LayerNorm），
    全部为推理时可得信息，不含未来真值；Huber 仅用于构造训练目标 q。
    """
    was_training = model.training
    model.eval()
    feats, hub_all, subj_all = [], [], []
    seen = set()
    stop = False
    hook_out = {}

    def hook(_module, _inp, out):
        hook_out['x'] = out.detach()

    router = model.one_step_head.pathology_moe.router
    handle = router.input_norm.register_forward_hook(hook)
    try:
        for batch in data_loader:
            if stop:
                break
            x = batch['x'].to(device, non_blocking=True)
            sc = batch['sc'].to(device, non_blocking=True)
            score = batch.get('pathology_score', None)
            if score is not None:
                score = score.to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            target2d = y.reshape(y.shape[0], y.shape[1], -1)[:, :, 0]
            _, aux = model(x, sc, score, return_expert_pred=True)
            cand = aux['tfm_expert_pred'].float()
            a = (cand - target2d.unsqueeze(1)).abs()
            hub_b = torch.where(a <= huber_delta, 0.5 * a ** 2,
                                huber_delta * (a - 0.5 * huber_delta)
                                ).mean(dim=-1).cpu().numpy()
            s = np.array([str(v) for v in batch['subj_id']])
            feats.append(hook_out['x'].float().cpu().numpy())
            hub_all.append(hub_b)
            subj_all.append(s)
            seen.update(s.tolist())
            if limit_subjects and len(seen) >= limit_subjects:
                stop = True
    finally:
        handle.remove()
        model.train(was_training)
    return (np.concatenate(feats), np.concatenate(hub_all),
            np.concatenate(subj_all))


def train_small_router(feats, hub_losses, tau, n_experts, device,
                       epochs=300, hidden=32, lr=1e-2, weight_decay=1e-4):
    """KL(sg(q) ‖ p_new) 训练小 Router（与 RECC 监督同口径，q stop-gradient）。"""
    torch.manual_seed(2024)
    router = SmallRouter(feats.shape[1], n_experts, hidden).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.from_numpy(feats).float().to(device)
    ell = torch.from_numpy(hub_losses).float().to(device)
    mu = ell.mean(-1, keepdim=True)
    sigma = ell.std(-1, unbiased=False, keepdim=True)
    log_q = F.log_softmax(-(ell - mu) / (sigma + 1e-8) / tau, dim=-1)
    q = log_q.exp()                                    # sg（常量，不建梯度）
    history = []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        log_p = F.log_softmax(router(x), dim=-1)
        loss = (q * (log_q - log_p)).sum(dim=-1).mean()
        loss.backward()
        opt.step()
        history.append(float(loss))
    return router, history


def _subject_mean(values, subj):
    uniq = np.unique(subj)
    return np.array([values[subj == s].mean() for s in uniq], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description='S3 冻结专家 Router 可学习性检验')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--limit_subjects', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--calib_frac', type=float, default=0.5,
                    help='被试级校准半区比例')
    ap.add_argument('--device', default='')
    ap.add_argument('--refresh_split_manifest', action='store_true')
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device
                          or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    train_args = load_train_args(args_cli.ckpt)
    if str(getattr(train_args, 'tfm_patho_mode', 'film')) != 'moe':
        raise ValueError('仅支持 tfm_patho_mode=moe 的 checkpoint')
    model = build_model_from_args(train_args, device)
    raw = torch.load(args_cli.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(raw['state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)                      # 冻结专家库（§7.1）

    from analysis import next_point_eval as npe
    data_loader, subjects, _ = npe.build_next_point_loader(
        train_args, args_cli.split,
        refresh_split_manifest=args_cli.refresh_split_manifest)
    n_experts = int(getattr(train_args, 'tfm_num_experts', 4))
    tau = float(getattr(train_args, 'tfm_competence_temperature', 1.0))
    huber_delta = float(getattr(train_args, 'cpm_huber_delta', 1.0))
    set_seed(2024)

    print(f'[frozen-router] ckpt={args_cli.ckpt} split={args_cli.split} '
          f'device={device}')
    feats, hub, subj = collect_features(
        model, data_loader, device, huber_delta, args_cli.limit_subjects)
    if args_cli.limit_subjects:
        seen = []
        for s in subj:
            if s not in seen:
                seen.append(s)
        keep = set(seen[:args_cli.limit_subjects])
        m = np.array([s in keep for s in subj])
        feats, hub, subj = feats[m], hub[m], subj[m]

    # 被试级不相交二分（seed 固定；§7.3 独立 calibration）
    uniq = sorted(np.unique(subj))
    rng = np.random.RandomState(2024)
    perm = rng.permutation(len(uniq))
    n_calib = max(1, int(len(uniq) * args_cli.calib_frac))
    calib_subjects = set(uniq[i] for i in perm[:n_calib])
    is_calib = np.array([s in calib_subjects for s in subj])
    is_val = ~is_calib

    router, history = train_small_router(
        feats[is_calib], hub[is_calib], tau, n_experts, device,
        epochs=args_cli.epochs, hidden=args_cli.hidden)
    with torch.no_grad():
        logits = router(torch.from_numpy(feats).float().to(device))
        w_new = F.softmax(logits, dim=-1).cpu().numpy()

    cand = None
    # candidate 已在 collect 中丢GPU引用；这里用损失矩阵直接重组 mixture 误差：
    # MAE 不可由逐专家 MAE 线性重组，需要逐 anchor candidate → 重新前向收集
    from analysis.reeval_tfm_experts import collect as reeval_collect
    records = reeval_collect(model, data_loader, device, huber_delta, tau,
                             args_cli.limit_subjects)
    if args_cli.limit_subjects:
        keep = set(seen[:args_cli.limit_subjects])
        records = [r for r in records if r['subject'] in keep]
    r_subj = np.array([r['subject'] for r in records])
    r_cand = np.stack([r['cand'] for r in records])
    r_target = np.stack([r['target'] for r in records])
    r_probs = np.stack([r['probs'] for r in records])
    r_mae = np.stack([r['ell_mae'] for r in records])
    r_mix = np.array([r['mix_mae'] for r in records])
    # 对齐两轮收集的 anchor 顺序（同 loader、同顺序 → 逐条一致）
    assert r_subj.tolist() == subj.tolist(), '两轮收集顺序不一致'

    is_calib_r = np.array([s in calib_subjects for s in r_subj])
    is_val_r = ~is_calib_r
    mix_new = (w_new[:, :, None] * r_cand).sum(axis=1)          # 新 Router mixture
    mae_new = np.abs(mix_new - r_target).mean(axis=1)
    mix_uni = (np.full((len(r_subj), n_experts), 1.0 / n_experts)[:, :, None]
               * r_cand).sum(axis=1)
    mae_uni = np.abs(mix_uni - r_target).mean(axis=1)
    mix_old = (r_probs[:, :, None] * r_cand).sum(axis=1)        # 原 Router（soft）
    mae_old = np.abs(mix_old - r_target).mean(axis=1)

    # 固定专家：校准半区选择 → 验证半区评估（可部署对照，§7.3）
    best_j = int(r_mae[is_calib_r].mean(axis=0).argmin())
    mae_fixed = r_mae[:, best_j]
    single_oracle = r_mae.min(axis=1)

    def agg(v, mask):
        return float(_subject_mean(v[mask], r_subj[mask]).mean())

    report = {
        'checkpoint': args_cli.ckpt,
        'checkpoint_sha256_16': hashlib.sha256(
            Path(args_cli.ckpt).read_bytes()).hexdigest()[:16],
        'split': args_cli.split,
        'n_anchors': int(len(r_subj)),
        'n_subjects_calib': int(len(set(r_subj[is_calib_r]))),
        'n_subjects_val': int(len(set(r_subj[is_val_r]))),
        'router_input': 'input_norm 输出（全脑描述子+条件，推理时可得）',
        'router_hidden': args_cli.hidden, 'epochs': args_cli.epochs,
        'competence_tau': tau, 'huber_delta': huber_delta,
        'final_kl_train': history[-1],
        'val_subject_mae': {
            'new_router': agg(mae_new, is_val_r),
            'old_router_soft': agg(mae_old, is_val_r),
            'uniform_mix': agg(mae_uni, is_val_r),
            f'fixed_expert_{best_j}_calib_selected': agg(mae_fixed, is_val_r),
            'single_expert_oracle(posthoc)': agg(single_oracle, is_val_r),
        },
        'gate_note': 'G3 判据：新 Router 需超过最强可部署对照并达到预设收益门槛；'
                     'oracle 为事后上界不参与门判定（§7/G3）。',
    }
    vr = report['val_subject_mae']
    deployable = {k: v for k, v in vr.items() if 'posthoc' not in k}
    other_dep = {k: v for k, v in deployable.items() if k != 'new_router'}
    best_other = min(other_dep, key=other_dep.get)
    report['best_deployable_excl_new_router'] = best_other
    report['new_router_gain_vs_best_other_pct'] = \
        100.0 * (other_dep[best_other] - vr['new_router']) / other_dep[best_other]

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'frozen_router_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(out_dir / 'frozen_router_arrays.npz',
                        w_new=w_new, subj=r_subj, mae=r_mae, probs=r_probs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
