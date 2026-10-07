# coding=utf-8
"""S4 路线 C 前置检验：多步冻结异构预测器库的 oracle 空间与可学习路由
（RECC 执行计划 §8.3 / §6.2；不训练任何新预测器）。

one-step 专家库 G2 未通过后的指定廉价检验：把**已存在的冻结预测器**组成库，
在同一批 val anchors/masks 上输出同一完整 horizon，检验：

1. 逐 anchor 逐预测器误差（原始空间，masked）；competence 用预设的
   预设窗口（默认 H4–H8，可配 --comp_lo/--comp_hi）等权轨迹 Huber（§6.2）；
2. 单预测器 oracle（事后上界）相对最强可部署固定预测器的收益空间 →
   G2-multi 判定（1% 门槛）；
3. 可学习路由：被试级不相交校准/验证，仅用推理时可得输入（backbone 状态
   描述符 ⊕ 条件）训练小型 Router 做**整段轨迹选择**（§8.3 第一版口径，
   不逐 horizon 切换）。

冻结库（全部推理时可得，无重训）：
  - cpm_k64：--ckpt 的 CPM 头（完整 context）
  - cpm_k{L}：同一 checkpoint、截断 context 至最近 L 个 TR（不同历史视野）
  - rollout：--ckpt 的 one-step 自回归 8 步（中间不用真值）
  - cpm_{name}：--extra_ckpt 提供的其它冻结模型（如历史 FiLM 参考）的 CPM 头
  - persistence / trend / ar1：三条解析基线（AR(1) 仅用 train subjects 拟合）

产物：multistep_summary.json / multistep_per_anchor.csv /
multistep_per_predictor.csv / multistep_arrays.npz（仅损失矩阵，受控存储）。

用法::

    python -m analysis.reeval_multistep_library \
        --ckpt checkpoints/neurotwin_tfm_finetune_moe/finetuned_best.pt \
        --extra_ckpt checkpoints/neurotwin_tfm_finetune/finetuned_best.pt \
        --out_dir analysis_results/recc_repair_20261007/multistep_B --split val
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.metrics import NETWORK_GROUPS
from analysis.next_point_eval import fit_ar1_from_train
from experiments.evaluate_variant import build_model_from_args, load_train_args
from utils.common import set_seed


def _subject_mean(values, subj):
    uniq = np.unique(subj)
    return np.array([values[subj == s].mean() for s in uniq], dtype=np.float64)


def _huber(diff, delta):
    a = np.abs(diff)
    return np.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def _network_roi_groups(n_rois: int):
    """NETWORK_GROUPS（1 基 AAL）→ 0 基 ROI 索引组（含 Other），顺序固定。"""
    named = [sorted(r for r in rois if 1 <= r <= n_rois)
             for rois in NETWORK_GROUPS.values()]
    covered = {r for rois in named for r in rois}
    other = [r for r in range(1, n_rois + 1) if r not in covered]
    return [[r - 1 for r in rois] for rois in named + ([other] if other else [])]


class ExtraPredictor:
    """额外冻结模型（只跑 CPM 头）。"""

    def __init__(self, ckpt_path, device):
        from experiments.evaluate_variant import build_model_from_args
        args = load_train_args(ckpt_path)
        self.model = build_model_from_args(args, device)
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(raw['state_dict'], strict=True)
        self.model.eval().to(device)
        parent = Path(ckpt_path).parents[0].name
        tag = parent.replace('neurotwin_tfm_finetune', 'film').strip('_') \
            or 'extra'
        self.name = f'cpm_{tag}'

    @torch.no_grad()
    def cpm(self, x, sc, score):
        _, aux = self.model(x, sc, score)
        return aux['cpm_pred']


@torch.no_grad()
def collect(model, data_loader, device, extra_models, ctx_lengths, n_horizon,
            ar1, limit_subjects=0, need_features=True,
            router_features='global', cpm_horizon=8):
    """单轮遍历：收集库内全部预测器的 [B,H,F] 原始空间预测与目标/掩码。

    router_features='global'：原 Router 的 input_norm 输出（全脑描述子⊕条件）；
    router_features='network'：预设 network 统计——z_t/z_g 按预设网络组内
    mean/std（D 维向量），拼接条件（§7.3：仅在已证实能力差异后检验的信息扩展）。
    n_horizon > cpm_horizon 时额外提供 cpm_rollout（CPM 分块自回归，把已预测
    chunk 接回 context 滑窗，推理时可得、无真值泄漏）。
    """
    model.eval()
    preds = {}                                    # name -> [N,H,F]（np，逐 batch 拼接）
    targets, masks = [], []
    subj_all, cutoff_all = [], []
    feats = [] if need_features else None
    hook_out = {}
    roi_groups = (_network_roi_groups(int(getattr(model, 'features', 116)))
                  if need_features and router_features == 'network' else None)

    if need_features and router_features == 'global':
        def hook(_m, _i, out):
            hook_out['x'] = out.detach()
        handle = model.one_step_head.pathology_moe.router.input_norm \
            .register_forward_hook(hook)
    elif need_features and router_features == 'network':
        def hook(module, inp, out):
            hook_out['z_t'] = inp[0].detach()
            hook_out['z_g'] = inp[1].detach()
            hook_out['cond'] = inp[2].detach() if len(inp) > 2 and inp[2] is not None \
                else None
        handle = model.one_step_head.pathology_moe.register_forward_hook(hook)
    else:
        handle = None

    seen = set()
    stop = False
    try:
        for batch in data_loader:
            if stop:
                break
            x = batch['x'].to(device, non_blocking=True)
            sc = batch['sc'].to(device, non_blocking=True)
            score = batch.get('pathology_score', None)
            if score is not None:
                score = score.to(device, non_blocking=True)
            fut = batch['future'].to(device, non_blocking=True)[:, :, :n_horizon]
            fmask = batch['future_mask'].to(device, non_blocking=True)[:, :n_horizon]
            b, f, _, k = x.shape
            chunk = {}

            # 主 checkpoint：CPM（完整 context）+ one-step rollout + 状态特征
            # 统一约定：chunk 内全部预测为 [B,H,F]
            _, aux = model(x, sc, score)
            cpm_width = int(getattr(model, 'cpm_horizon', 8))
            # 完整 horizon 约束（§8.3）：直出 CPM 只有在 cpm_width >= n_horizon
            # 时才能进入库；否则由 cpm_rollout（分块自回归）代表 CPM 家族
            if n_horizon <= cpm_width:
                chunk['cpm_k64'] = aux['cpm_pred'][:, :, :n_horizon].transpose(1, 2)
            if need_features:
                if router_features == 'network':
                    zt, zg = hook_out['z_t'], hook_out['z_g']       # [B,F,D]
                    parts = []
                    for rois in roi_groups:
                        # unbiased=False：单 ROI 网络（如 VN）std 定义为 0 而非 NaN
                        parts += [zt[:, rois].mean(dim=1),
                                  zt[:, rois].std(dim=1, unbiased=False),
                                  zg[:, rois].mean(dim=1),
                                  zg[:, rois].std(dim=1, unbiased=False)]
                    if hook_out['cond'] is not None:
                        parts.append(hook_out['cond'].float())
                    feats.append(torch.cat(parts, dim=-1).float().cpu().numpy())
                else:
                    feats.append(hook_out['x'].float().cpu().numpy())
            hist = x[:, :, 0, :].transpose(1, 2).contiguous()          # [B,K,F]
            chunk['rollout'] = model.rollout_next_states(
                hist, sc, score, steps=n_horizon)                      # [B,H,F]
            if n_horizon <= cpm_width:
                for L in ctx_lengths:
                    _, aux_l = model(x[:, :, :, -L:], sc, score)
                    chunk[f'cpm_k{L}'] = \
                        aux_l['cpm_pred'][:, :, :n_horizon].transpose(1, 2)
            if n_horizon > cpm_horizon:
                # CPM 分块自回归：已预测 chunk 接回 context 滑窗（无真值泄漏）
                cur = x
                pieces = []
                for c in range((n_horizon + cpm_horizon - 1) // cpm_horizon):
                    _, aux_c = model(cur, sc, score)
                    w = min(cpm_horizon, n_horizon - c * cpm_horizon)
                    p = aux_c['cpm_pred'][:, :, :w]                    # [B,F,w]
                    pieces.append(p)
                    if c * cpm_horizon + w < n_horizon:
                        grid = cur[:, :, 0, :]
                        grid = torch.cat([grid[:, :, w:], p], dim=2)
                        cur = grid.unsqueeze(2)
                chunk['cpm_rollout'] = torch.cat(pieces, dim=2).transpose(1, 2)
            # 额外冻结模型（同样受完整 horizon 约束）
            if n_horizon <= cpm_width:
                for extra in extra_models:
                    chunk[extra.name] = \
                        extra.cpm(x, sc, score)[:, :, :n_horizon].transpose(1, 2)
            # 解析基线（与 next_point_eval 同一口径）
            x_last = x[:, :, 0, -1]
            x_prev = x[:, :, 0, -2]
            k_step = torch.arange(1, n_horizon + 1, device=device,
                                  dtype=x_last.dtype).view(1, -1, 1)
            chunk['persistence'] = x_last.unsqueeze(1).expand(
                b, n_horizon, f).contiguous()
            chunk['trend'] = (x_last.unsqueeze(1)
                              + k_step * (x_last - x_prev).unsqueeze(1))
            a = torch.as_tensor(ar1['a'], device=device, dtype=x_last.dtype
                                ).view(1, -1)
            bb = torch.as_tensor(ar1['b'], device=device, dtype=x_last.dtype
                                 ).view(1, -1)
            cur = x_last
            ar_steps = []
            for _ in range(n_horizon):
                cur = a * cur + bb
                ar_steps.append(cur)
            chunk['ar1'] = torch.stack(ar_steps, dim=1)                # [B,H,F]

            for name, p in chunk.items():
                preds.setdefault(name, []).append(
                    p.float().cpu().numpy())                           # [B,H,F]
            targets.append(fut.float().cpu().numpy().transpose(0, 2, 1))  # [B,H,F]
            masks.append(fmask.float().cpu().numpy())                  # [B,H]
            s = np.array([str(v) for v in batch['subj_id']])
            subj_all.append(s)
            cutoff_all.append(batch['cutoff'].numpy())
            seen.update(s.tolist())
            if limit_subjects and len(seen) >= limit_subjects:
                stop = True
    finally:
        if handle is not None:
            handle.remove()
    out = {name: np.concatenate(v, axis=0) for name, v in preds.items()}
    data = {
        'preds': out,
        'target': np.concatenate(targets, axis=0),                     # [N,F,H]
        'mask': np.concatenate(masks, axis=0),                         # [N,H]
        'subj': np.concatenate(subj_all),
        'cutoff': np.concatenate(cutoff_all),
    }
    if need_features:
        data['feats'] = np.concatenate(feats, axis=0)
    return data


def main():
    ap = argparse.ArgumentParser(
        description='S4 路线C前置：多步冻结预测器库 oracle 与可学习路由检验')
    ap.add_argument('--ckpt', required=True,
                    help='主 checkpoint（提供 backbone 特征/CPM/rollout）')
    ap.add_argument('--extra_ckpt', nargs='*', default=[],
                    help='其它冻结模型的 checkpoint（各贡献一个 CPM 预测器）')
    ap.add_argument('--ctx_lengths', default='32,16',
                    help='截断 context 的 CPM 变体长度列表（逗号分隔）')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--limit_subjects', type=int, default=0)
    ap.add_argument('--horizons', type=int, default=0,
                    help='评估 horizon 数；0=取 checkpoint 的 cpm_horizon；'
                         '>cpm_horizon 时自动加入 cpm_rollout（分块自回归）')
    ap.add_argument('--comp_lo', type=int, default=4,
                    help='competence 窗口起点 horizon（含），默认 4')
    ap.add_argument('--comp_hi', type=int, default=8,
                    help='competence 窗口终点 horizon（含），默认 8')
    ap.add_argument('--router_features', default='global',
                    choices=['global', 'network'],
                    help='Router 输入特征：global=全脑描述子（input_norm 输出）；'
                         'network=预设网络组内 z_t/z_g mean/std（§7.3 信息扩展）')
    ap.add_argument('--router_epochs', type=int, default=400)
    ap.add_argument('--router_hidden', type=int, default=32)
    ap.add_argument('--skip_router', action='store_true')
    ap.add_argument('--device', default='')
    ap.add_argument('--refresh_split_manifest', action='store_true')
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device
                          or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    train_args = load_train_args(args_cli.ckpt)
    n_horizon = int(args_cli.horizons
                    or getattr(train_args, 'cpm_horizon', 8))
    comp_lo = max(1, int(args_cli.comp_lo))
    comp_hi = min(n_horizon, int(args_cli.comp_hi))
    if comp_lo > comp_hi:
        raise ValueError(f'competence 窗口非法：H{comp_lo}-H{comp_hi}')
    huber_delta = float(getattr(train_args, 'cpm_huber_delta', 1.0))
    model = build_model_from_args(train_args, device)
    raw = torch.load(args_cli.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(raw['state_dict'], strict=True)
    model.eval().to(device)
    extra_models = [ExtraPredictor(p, device) for p in args_cli.extra_ckpt]
    ctx_lengths = [int(v) for v in args_cli.ctx_lengths.split(',') if v]

    from analysis import next_point_eval as npe
    data_loader, subjects, loader_obj = npe.build_next_point_loader(
        train_args, args_cli.split,
        refresh_split_manifest=args_cli.refresh_split_manifest)
    ar1 = fit_ar1_from_train(loader_obj.base_dataset,
                             loader_obj.get_train_subjects())
    set_seed(2024)
    print(f'[multistep] ckpt={args_cli.ckpt} split={args_cli.split} '
          f'H={n_horizon} comp=H{comp_lo}-H{comp_hi} ctx={ctx_lengths} '
          f'extras={len(extra_models)} router_feat={args_cli.router_features} '
          f'device={device}')

    data = collect(model, data_loader, device, extra_models, ctx_lengths,
                   n_horizon, ar1, args_cli.limit_subjects,
                   need_features=not args_cli.skip_router,
                   router_features=args_cli.router_features,
                   cpm_horizon=int(getattr(train_args, 'cpm_horizon', 8)))
    if args_cli.limit_subjects:
        # 精确截断到先出现的 N 个被试（batch 粒度停止会超采）
        seen = []
        for s in data['subj']:
            if s not in seen:
                seen.append(s)
        keep = set(seen[:args_cli.limit_subjects])
        m = np.array([s in keep for s in data['subj']])
        data['subj'] = data['subj'][m]
        data['cutoff'] = data['cutoff'][m]
        data['target'] = data['target'][m]
        data['mask'] = data['mask'][m]
        if 'feats' in data:
            data['feats'] = data['feats'][m]
        data['preds'] = {name: p[m] for name, p in data['preds'].items()}

    names = sorted(data['preds'].keys())
    subj = data['subj']
    target = data['target']                                   # [N,F,H]
    mask = data['mask']                                       # [N,H]
    n = len(subj)

    # 逐 (anchor, predictor) 损失：逐 horizon masked MAE + 预设窗口 Huber competence
    # 维度约定：pred/target [N,H,F]，mask [N,H]
    mae = {}      # name -> [N,H]
    comp = {}     # name -> [N]
    w_mask = mask[:, :, None]                                  # [N,H,1]
    comp_win = slice(comp_lo - 1, min(comp_hi, n_horizon))
    n_f = target.shape[2]
    # mask 广播到 [N,H,F] / [N,W,F]：逐元素平均的分母必须含 F
    w3 = np.broadcast_to(w_mask, (n, n_horizon, n_f))
    mw = np.broadcast_to(mask[:, comp_win, None],
                         (n, comp_win.stop - comp_win.start, n_f))
    for name, pred in data['preds'].items():
        err = np.abs(pred - target)                            # [N,H,F]
        mae[name] = (err * w3).sum(axis=2) / np.maximum(
            w3.sum(axis=2), 1e-8)                              # 逐 horizon [N,H]
        hub = _huber(pred[:, comp_win, :] - target[:, comp_win, :],
                     huber_delta)                              # [N,W,F]
        comp[name] = (hub * mw).sum(axis=(1, 2)) / np.maximum(
            mw.sum(axis=(1, 2)), 1e-8)

    comp_mat = np.stack([comp[name] for name in names], axis=1)   # [N,P]
    mae_mat = np.stack([mae[name] for name in names], axis=1)     # [N,P,H]

    def agg(v):
        return float(_subject_mean(v, subj).mean())

    # 可部署策略：单一固定预测器（cross-fit：被试级 LOSO）
    best_other = {}
    for s in np.unique(subj):
        m_ = subj != s
        best_other[s] = int(comp_mat[m_].mean(axis=0).argmin())
    fixed_cf_idx = np.array([best_other[s] for s in subj])
    fixed_cf_comp = comp_mat[np.arange(n), fixed_cf_idx]
    fixed_cf_mae_h = mae_mat[np.arange(n), fixed_cf_idx]          # [N,H]
    # 均匀平均（库平均，可部署参照）
    avg_pred = np.mean(np.stack([data['preds'][nm] for nm in names]), axis=0)
    err_avg = np.abs(avg_pred - target)
    mae_avg = (err_avg * w3).sum(axis=2) / np.maximum(
        w3.sum(axis=2), 1e-8)                                     # [N,H]
    hub_avg = _huber(avg_pred[:, comp_win, :] - target[:, comp_win, :],
                     huber_delta)
    comp_avg = (hub_avg * mw).sum(axis=(1, 2)) / np.maximum(
        mw.sum(axis=(1, 2)), 1e-8)

    # oracle（事后）：逐 anchor 选 competence 最小的预测器（整段轨迹，§8.3）
    oracle_idx = comp_mat.argmin(axis=1)
    oracle_comp = comp_mat[np.arange(n), oracle_idx]
    oracle_mae_h = mae_mat[np.arange(n), oracle_idx]
    switch = float(np.mean([
        1.0 if i == 0 or oracle_idx[i] != oracle_idx[i - 1] else 0.0
        for i in range(1, n) if subj[i] == subj[i - 1]] or [np.nan]))

    best_fixed_comp = agg(fixed_cf_comp)
    oracle_comp_mean = agg(oracle_comp)
    gap = best_fixed_comp - oracle_comp_mean
    threshold = 0.01 * best_fixed_comp

    summary = {
        'checkpoint': args_cli.ckpt,
        'extra_ckpts': args_cli.extra_ckpt,
        'checkpoint_sha256_16': hashlib.sha256(
            Path(args_cli.ckpt).read_bytes()).hexdigest()[:16],
        'split': args_cli.split,
        'n_anchors': int(n),
        'n_subjects': int(len(np.unique(subj))),
        'n_horizon': int(n_horizon),
        'competence_window': f'H{comp_lo}-H{min(comp_hi, n_horizon)} '
                             '等权轨迹 Huber（δ=%.1f）' % huber_delta,
        'predictors': names,
        'per_predictor_subject_mae': {
            name: dict(
                H1=agg(mae[name][:, 0]),
                H4=agg(mae[name][:, min(3, n_horizon - 1)]),
                H8=agg(mae[name][:, min(7, n_horizon - 1)]),
                **({f'H{n_horizon}': agg(mae[name][:, n_horizon - 1])}
                   if n_horizon > 8 else {}),
                **{f'mean_H1_H{n_horizon}': agg(mae[name].mean(axis=1)),
                   f'competence_H{comp_lo}_H{comp_hi}': agg(comp[name])})
            for name in names},
        'deployable_policies': {
            'fixed_crossfit_best': {'competence': best_fixed_comp,
                                    'mae_per_horizon': [
                                        agg(fixed_cf_mae_h[:, h])
                                        for h in range(n_horizon)]},
            'uniform_library_average': {'competence': agg(comp_avg),
                                        'mae_per_horizon': [
                                            agg(mae_avg[:, h])
                                            for h in range(n_horizon)]},
        },
        'single_predictor_oracle_posthoc': {
            'competence': oracle_comp_mean,
            'mae_per_horizon': [agg(oracle_mae_h[:, h])
                                for h in range(n_horizon)],
            'selected_share': [float((oracle_idx == j).mean())
                               for j in range(len(names))],
        },
        'oracle_gap_competence': gap,
        'oracle_gap_pct_of_best_fixed': 100.0 * gap / best_fixed_comp,
        'g2_multi_threshold_1pct': threshold,
        'g2_multi_pass': bool(gap >= threshold),
        'oracle_switch_rate_adjacent_anchors': switch,
        'note': 'oracle 使用未来真值，仅作事后上界（§6.2）；'
                'oracle_gap ≥ 1% 才进入 Router 可学习性检验（§8.3）。',
    }

    # ---- 可学习路由（被试级不相交；仅 oracle 空间过门槛时有意义）----
    if not args_cli.skip_router:
        import torch.nn.functional as F
        from analysis.train_frozen_router import SmallRouter

        feats = data['feats']
        uniq = sorted(np.unique(subj))
        rng = np.random.RandomState(2024)
        perm = rng.permutation(len(uniq))
        n_calib = max(1, int(len(uniq) * 0.5))
        calib = set(uniq[i] for i in perm[:n_calib])
        is_calib = np.array([s in calib for s in subj])

        torch.manual_seed(2024)
        router = SmallRouter(feats.shape[1], len(names),
                             args_cli.router_hidden).to(device)
        opt = torch.optim.AdamW(router.parameters(), lr=1e-2, weight_decay=1e-4)
        xt = torch.from_numpy(feats[is_calib]).float().to(device)
        ct = torch.from_numpy(comp_mat[is_calib]).float().to(device)
        mu, sigma = ct.mean(-1, keepdim=True), ct.std(-1, unbiased=False,
                                                      keepdim=True)
        log_q = F.log_softmax(-(ct - mu) / (sigma + 1e-8) / 1.0, dim=-1)
        q = log_q.exp()
        for _ in range(args_cli.router_epochs):
            opt.zero_grad(set_to_none=True)
            log_p = F.log_softmax(router(xt), dim=-1)
            loss = (q * (log_q - log_p)).sum(dim=-1).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = router(torch.from_numpy(feats).float().to(device))
        route_idx = logits.argmax(dim=-1).cpu().numpy()           # 整段轨迹选择
        routed_pred = np.stack([
            data['preds'][names[route_idx[i]]][i] for i in range(n)])
        err_r = np.abs(routed_pred - target)
        mae_r = (err_r * w3).sum(axis=2) / np.maximum(
            w3.sum(axis=2), 1e-8)                              # [N,H]
        hub_r = _huber(routed_pred[:, comp_win, :] - target[:, comp_win, :],
                       huber_delta)
        comp_r = (hub_r * mw).sum(axis=(1, 2)) / np.maximum(
            mw.sum(axis=(1, 2)), 1e-8)
        is_val = ~is_calib
        subj_val = subj[is_val]

        def agg_val(v):
            return float(_subject_mean(v, subj_val).mean())

        summary['router_learnability'] = {
            'protocol': '被试级 50/50 校准/验证；输入=%s（推理时可得）；'
                        '整段轨迹 argmax 选择' % (
                            'backbone 状态描述符⊕条件' if args_cli.router_features == 'global'
                            else '预设网络组内 z_t/z_g mean/std ⊕条件'),
            'router_features': args_cli.router_features,
            'n_subjects_calib': int(len(set(subj[is_calib]))),
            'n_subjects_val': int(len(set(subj_val))),
            'val_competence': {
                'new_router': agg_val(comp_r[is_val]),
                'fixed_crossfit_best': float(
                    _subject_mean(fixed_cf_comp[is_val], subj_val).mean()),
                'uniform_library_average': float(
                    _subject_mean(comp_avg[is_val], subj_val).mean()),
                'single_predictor_oracle_posthoc': float(
                    _subject_mean(oracle_comp[is_val], subj_val).mean()),
            },
            'val_mae_per_horizon_new_router': [
                agg_val(mae_r[is_val][:, h]) for h in range(n_horizon)],
        }

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'multistep_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    with open(out_dir / 'multistep_per_predictor.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        cols = (['predictor', 'H1', 'H4', 'H8']
                + ([f'H{n_horizon}'] if n_horizon > 8 else [])
                + [f'mean_H1_H{n_horizon}', f'competence_H{comp_lo}_H{comp_hi}'])
        w.writerow(cols)
        for name in names:
            d = summary['per_predictor_subject_mae'][name]
            row = [f"{d[c]:.6f}" for c in cols[1:]]
            w.writerow([name] + row)

    with open(out_dir / 'multistep_per_anchor.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'cutoff', 'oracle_pick', 'fixed_cf_pick']
                   + [f'comp_{nm}' for nm in names]
                   + [f'mae_H1_{nm}' for nm in names]
                   + [f'mae_H8_{nm}' for nm in names])
        for i in range(n):
            w.writerow([subj[i], int(data['cutoff'][i]),
                        names[oracle_idx[i]], names[fixed_cf_idx[i]]]
                       + [f"{comp_mat[i, j]:.6f}" for j in range(len(names))]
                       + [f"{mae_mat[i, j, 0]:.6f}" for j in range(len(names))]
                       + [f"{mae_mat[i, j, min(7, n_horizon - 1)]:.6f}"
                          for j in range(len(names))])

    np.savez_compressed(out_dir / 'multistep_arrays.npz',
                        competence=comp_mat, mae=mae_mat,
                        subj=subj, mask=mask,
                        oracle_idx=oracle_idx)
    print(json.dumps({k: summary[k] for k in (
        'n_anchors', 'oracle_gap_competence', 'oracle_gap_pct_of_best_fixed',
        'g2_multi_threshold_1pct', 'g2_multi_pass',
        'oracle_switch_rate_adjacent_anchors')}, ensure_ascii=False, indent=2))
    print('[multistep] per-predictor competence:',
          json.dumps({nm: summary['per_predictor_subject_mae'][nm]
                      [f'competence_H{comp_lo}_H{comp_hi}'] for nm in names},
                     ensure_ascii=False, indent=2))
    if 'router_learnability' in summary:
        print('[multistep] router val competence:',
              json.dumps(summary['router_learnability']['val_competence'],
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
