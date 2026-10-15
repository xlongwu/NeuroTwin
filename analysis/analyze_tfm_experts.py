# coding=utf-8
"""TFM 病理专家机制分析（RECC 方案 §72–§77/§94–§95）。

从 moe 模式 finetune checkpoint 重建模型，在评估协议（与 evaluate_variant 同一
被试级切分与确定性 anchor）上对每个 anchor 收集：

- Router 概率（含 argmax / Top-2）、逐专家 candidate 误差（competence）、
  oracle best expert、router regret；
- 逐专家 latent 修正 Δz_i 经共享 decode 的线性映射得到的逐 ROI 修正量
  （方案 §75：decode(z+Δz_i) − decode(z) = W·Δz_i），按 AAL 网络聚合（§76）。

输出（out_dir，方案 §95）：
  tfm_expert_summary.json        汇总指标（top1 agreement / top2 recall /
                                 regret / spearman / competence entropy /
                                 oracle gap / 专家使用率 / competence matrix）
  tfm_expert_per_anchor.csv      逐 anchor 明细
  tfm_expert_per_subject.csv     逐被试聚合（含 HAMD 与 intra-subject 路由一致性）
  tfm_competence_matrix.csv      C_ij = E[ℓ_j | argmax(p)=i]（§71）
  tfm_competence_matrix.png / tfm_router_hamd.png / tfm_router_regret.png /
  tfm_expert_network_profile.png（matplotlib 缺失时跳过图像）

用法::

    python -m analysis.analyze_tfm_experts --ckpt <finetuned_best.pt> \
        --out_dir results/tfm_experts --split test
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoint import load_checkpoint_compat
from analysis.metrics import NETWORK_GROUPS
from experiments.evaluate_variant import build_model_from_args, load_train_args
from train.tfm_coupling import compute_expert_competence_losses
from utils.common import set_seed

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def _spearman_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐行 Spearman 秩相关（按值秩，并列取顺序秩），a/b 同形 [N, E]。"""
    ra = a.argsort(axis=1).argsort(axis=1).astype(np.float64)
    rb = b.argsort(axis=1).argsort(axis=1).astype(np.float64)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1)) + eps
    return num / den


def _network_labels(n_rois: int):
    """ROI 索引（0 基）→ 网络标签（未命名脑区归 Other）。"""
    labels = ['Other'] * n_rois
    for net, regions in NETWORK_GROUPS.items():
        for r in regions:
            if 1 <= r <= n_rois:
                labels[r - 1] = net
    return labels


@torch.no_grad()
def collect_anchor_stats(model, data_loader, device, huber_delta=1.0,
                         comp_tau=1.0):
    """遍历评估集，返回逐 anchor 记录与逐专家逐 ROI 修正量聚合。"""
    model.eval()
    records = []
    roi_corr_sum = None        # [E, F] 逐专家逐 ROI 平均 |W·Δz_i|（归一化空间）
    roi_corr_n = 0
    decode_w = None            # 共享 decode 的 [D] 权重（Δz→Δx 的线性映射）
    for batch in data_loader:
        x = batch['x'].to(device, non_blocking=True)
        sc = batch['sc'].to(device, non_blocking=True)
        score = batch.get('pathology_score', None)
        if score is not None:
            score = score.to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)              # [B,F,H=1]
        # one-step 目标取第一个偏移（forecast_offsets=[1]）→ [B,F]
        target2d = y.reshape(y.shape[0], y.shape[1], -1)[:, :, 0]
        pred, aux = model(x, sc, score, return_expert_pred=True)
        expert_pred = aux['tfm_expert_pred']                      # [B,E,F] 原始空间
        probs = aux['tfm_router_probs'].float().cpu().numpy()     # [B,E]
        ell = compute_expert_competence_losses(
            expert_pred, target2d, huber_delta=huber_delta).cpu().numpy()  # [B,E]

        b, e = probs.shape
        router_top1 = probs.argmax(axis=1)
        oracle_top1 = ell.argmin(axis=1)
        router_top2 = np.argsort(-probs, axis=1)[:, :2]
        oracle_top2 = np.argsort(ell, axis=1)[:, :2]
        regret = ell[np.arange(b), router_top1] - ell.min(axis=1)
        actual_mae = (pred[:, :, 0, 0] - target2d).abs() \
            .mean(dim=1).cpu().numpy()
        q = torch.softmax(-torch.from_numpy(
            (ell - ell.mean(axis=1, keepdims=True))
            / (ell.std(axis=1, keepdims=True) + 1e-8)) / comp_tau, dim=-1).numpy()

        # 逐专家 latent 修正 → 共享 decode 线性映射到预测空间（§75）
        if 'tfm_expert_delta_z_norm' in aux:
            if decode_w is None:
                decode_w = model.one_step_head.decode.weight.detach() \
                    .view(-1).cpu().numpy()                       # [D]
            delta_z = aux['tfm_expert_delta_z_norm'].float()       # [B,E,F,D]
            delta_x = torch.einsum('d,befd->bef', torch.from_numpy(decode_w),
                                   delta_z.cpu()).numpy()          # [B,E,F]
            corr = np.abs(delta_x).mean(axis=0)                    # [E,F]
            roi_corr_sum = corr if roi_corr_sum is None else roi_corr_sum + corr
            roi_corr_n += 1

        subj_ids = batch['subj_id']
        cutoffs = batch['cutoff'].numpy() if torch.is_tensor(batch['cutoff']) \
            else np.asarray(batch['cutoff'])
        hammad = score.squeeze(-1).cpu().numpy() if score is not None \
            else np.full(b, np.nan)
        for i in range(b):
            records.append({
                'subject_id': subj_ids[i],
                'cutoff': int(cutoffs[i]),
                'hamd': float(hammad[i]),
                'router_probs': probs[i].tolist(),
                'expert_losses': ell[i].tolist(),
                'router_top1': int(router_top1[i]),
                'oracle_best_expert': int(oracle_top1[i]),
                'router_top2': sorted(int(v) for v in router_top2[i]),
                'oracle_top2': sorted(int(v) for v in oracle_top2[i]),
                'router_regret': float(regret[i]),
                'actual_mae': float(actual_mae[i]),
                'competence_entropy': float(-(q[i] * np.log(q[i] + 1e-12)).sum()),
                'top1_agree': int(router_top1[i] == oracle_top1[i]),
                'top2_overlap': float(len(set(router_top2[i].tolist())
                                         & set(oracle_top2[i].tolist())) / 2.0),
            })
    return records, (None if roi_corr_sum is None else roi_corr_sum / max(1, roi_corr_n))


def summarize(records, n_experts):
    """汇总指标（方案 §63–§71）+ competence matrix。"""
    n = len(records)
    probs = np.array([r['router_probs'] for r in records])        # [N,E]
    ell = np.array([r['expert_losses'] for r in records])
    top1 = np.array([r['router_top1'] for r in records])
    hamd = np.array([r['hamd'] for r in records])

    comp_matrix = np.zeros((n_experts, n_experts))
    for i in range(n_experts):
        mask = top1 == i
        comp_matrix[i] = ell[mask].mean(axis=0) if mask.any() else np.nan

    summary = {
        'n_anchors': n,
        'n_subjects': len({r['subject_id'] for r in records}),
        'router_top1_agreement': float(np.mean([r['top1_agree'] for r in records])),
        'router_top2_recall': float(np.mean([r['top2_overlap'] for r in records])),
        'router_regret_mean': float(np.mean([r['router_regret'] for r in records])),
        'router_competence_spearman_mean': float(
            _spearman_rows(probs, -ell).mean()),
        'competence_entropy_mean': float(
            np.mean([r['competence_entropy'] for r in records])),
        'actual_mae_mean': float(np.mean([r['actual_mae'] for r in records])),
        'oracle_best_expert_gap': float(np.mean(
            [r['actual_mae'] - min(r['expert_losses']) for r in records])),
        'expert_top1_share': [float((top1 == i).mean()) for i in range(n_experts)],
        'expert_mean_prob': probs.mean(axis=0).tolist(),
        'expert_mean_loss': ell.mean(axis=0).tolist(),
        'competence_matrix': comp_matrix.tolist(),
    }
    # Router 是否按 HAMD 分箱（§73/§80）：HAMD 与各专家概率在 anchor 维的
    # 秩相关（每专家一个标量；|ρ| 高 → 该专家概率主要由严重程度决定）
    if np.isfinite(hamd).all() and np.std(hamd) > 0:
        hr = np.empty(len(hamd), dtype=np.float64)
        hr[np.argsort(hamd)] = np.arange(len(hamd))
        hr = hr - hr.mean()
        pr = probs.argsort(axis=0).argsort(axis=0).astype(np.float64)
        pr = pr - pr.mean(axis=0, keepdims=True)
        num = (pr * hr[:, None]).sum(axis=0)
        den = np.sqrt((pr ** 2).sum(axis=0) * (hr ** 2).sum()) + 1e-8
        summary['hamd_expert_prob_spearman'] = (num / den).tolist()
    return summary


def write_outputs(out_dir, records, summary, roi_corr, n_experts):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'tfm_expert_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / 'tfm_expert_per_anchor.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subject_id', 'cutoff', 'hamd', 'router_top1',
                    'oracle_best_expert', 'router_top2', 'oracle_top2',
                    'router_regret', 'actual_mae', 'competence_entropy']
                   + [f'prob_e{i}' for i in range(n_experts)]
                   + [f'loss_e{i}' for i in range(n_experts)])
        for r in records:
            w.writerow([r['subject_id'], r['cutoff'], r['hamd'],
                        r['router_top1'], r['oracle_best_expert'],
                        '|'.join(map(str, r['router_top2'])),
                        '|'.join(map(str, r['oracle_top2'])),
                        r['router_regret'], r['actual_mae'],
                        r['competence_entropy']]
                       + r['router_probs'] + r['expert_losses'])

    # 逐被试聚合 + intra-subject 路由一致性（§77：与 inter-subject 对比解读）
    subjects = sorted({r['subject_id'] for r in records})
    inter_sim = None
    all_probs = np.array([r['router_probs'] for r in records])
    if len(all_probs) > 1:
        pn = all_probs / np.clip(np.linalg.norm(all_probs, axis=1,
                                                keepdims=True), 1e-8, None)
        inter_sim = float((pn @ pn.T)[np.triu_indices(len(pn), 1)].mean())
    with open(out_dir / 'tfm_expert_per_subject.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subject_id', 'hamd_mean', 'n_anchors', 'router_top1_mode',
                    'router_regret_mean', 'top1_agree_rate',
                    'intra_subject_routing_sim', 'inter_subject_routing_sim']
                   + [f'prob_e{i}_mean' for i in range(n_experts)])
        for sid in subjects:
            rows = [r for r in records if r['subject_id'] == sid]
            p = np.array([r['router_probs'] for r in rows])
            intra = None
            if len(p) > 1:
                pn = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True),
                                 1e-8, None)
                intra = float((pn @ pn.T)[np.triu_indices(len(pn), 1)].mean())
            top1s = [r['router_top1'] for r in rows]
            w.writerow([sid, float(np.mean([r['hamd'] for r in rows])),
                        len(rows), max(set(top1s), key=top1s.count),
                        float(np.mean([r['router_regret'] for r in rows])),
                        float(np.mean([r['top1_agree'] for r in rows])),
                        intra, inter_sim]
                       + p.mean(axis=0).tolist())

    comp = np.array(summary['competence_matrix'])
    with open(out_dir / 'tfm_competence_matrix.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['router\\loss'] + [f'loss_e{j}' for j in range(n_experts)]
                   + ['n_anchors'])
        for i in range(n_experts):
            w.writerow([f'expert_{i}'] + comp[i].tolist()
                       + [int(sum(1 for r in records
                                  if r['router_top1'] == i))])

    if roi_corr is not None:
        np.savetxt(out_dir / 'tfm_expert_roi_correction.csv',
                   roi_corr, delimiter=',',
                   header=','.join(f'roi_{i}' for i in range(roi_corr.shape[1])),
                   comments='')
    _write_plots(out_dir, records, summary, roi_corr, n_experts)


def _write_plots(out_dir, records, summary, roi_corr, n_experts):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning('matplotlib 不可用，跳过 PNG 输出')
        return

    comp = np.array(summary['competence_matrix'], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(comp, cmap='viridis_r')
    ax.set_xticks(range(n_experts), [f'E{i}' for i in range(n_experts)])
    ax.set_yticks(range(n_experts), [f'router=E{i}' for i in range(n_experts)])
    for i in range(n_experts):
        for j in range(n_experts):
            if np.isfinite(comp[i, j]):
                ax.text(j, i, f'{comp[i, j]:.3g}', ha='center', va='center',
                        color='w', fontsize=8)
    ax.set_title('Competence matrix C_ij = E[loss_j | argmax p = i]\n'
                 '(ideal: diagonal lowest, §71)')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / 'tfm_competence_matrix.png', dpi=160)
    plt.close(fig)

    hamd = np.array([r['hamd'] for r in records])
    probs = np.array([r['router_probs'] for r in records])
    if np.isfinite(hamd).all() and np.std(hamd) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        order = np.argsort(hamd)
        for i in range(n_experts):
            ax.plot(hamd[order], probs[order, i], '.', ms=4,
                    alpha=0.6, label=f'E{i}')
        ax.set_xlabel('raw HAMD')
        ax.set_ylabel('router prob')
        ax.set_title('Router probability vs HAMD (§73)')
        ax.legend(ncol=n_experts, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / 'tfm_router_hamd.png', dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist([r['router_regret'] for r in records], bins=40)
    ax.set_xlabel(f"router regret (mean={summary['router_regret_mean']:.3g})")
    ax.set_ylabel('#anchors')
    ax.set_title('Router regret distribution (§66)')
    fig.tight_layout()
    fig.savefig(out_dir / 'tfm_router_regret.png', dpi=160)
    plt.close(fig)

    if roi_corr is not None:
        labels = _network_labels(roi_corr.shape[1])
        nets = sorted(set(labels) - {'Other'})
        mat = np.stack([roi_corr[:, [i for i, lb in enumerate(labels)
                                   if lb == net]].mean(axis=1)
                        for net in nets])                     # [nets, E]
        fig, ax = plt.subplots(figsize=(1.2 * n_experts + 3.5,
                                        0.45 * len(nets) + 2.2))
        im = ax.imshow(mat, cmap='magma', aspect='auto')
        ax.set_yticks(range(len(nets)), nets)
        ax.set_xticks(range(n_experts), [f'E{i}' for i in range(n_experts)])
        ax.set_title('Per-expert correction by network (|Δx|, §76)')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / 'tfm_expert_network_profile.png', dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='TFM 病理专家机制分析（RECC 方案 §94–§95；仅 tfm_patho_mode=moe 的 checkpoint）')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    ap.add_argument('--device', default='',
                    help='空 = 自动（有 GPU 用 cuda:0）')
    ap.add_argument('--refresh_split_manifest', action='store_true')
    args_cli = ap.parse_args()

    set_seed(2024)
    device = torch.device(args_cli.device
                          or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    train_args = load_train_args(args_cli.ckpt)
    if str(getattr(train_args, 'tfm_patho_mode', 'film')) != 'moe':
        raise ValueError(
            "该 checkpoint 不是 tfm_patho_mode=moe 训练的（无病理专家可分析）；"
            "请改用 tfm_patho_mode=moe 训练的 checkpoint（main.py --tfm_patho_mode moe）。")
    model = build_model_from_args(train_args, device)
    model, compat = load_checkpoint_compat(model, args_cli.ckpt, device,
                                           strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')

    from analysis import next_point_eval as npe
    data_loader, subjects, _ = npe.build_next_point_loader(
        train_args, args_cli.split,
        refresh_split_manifest=args_cli.refresh_split_manifest)
    log.info('[tfm_experts] split=%s | subjects=%d | tasks=%d',
             args_cli.split, len(subjects), len(data_loader.dataset))

    n_experts = int(getattr(train_args, 'tfm_num_experts', 4))
    records, roi_corr = collect_anchor_stats(model, data_loader, device)
    summary = summarize(records, n_experts)
    summary['checkpoint'] = str(args_cli.ckpt)
    summary['split'] = args_cli.split
    # candidate 口径标记（执行计划 §6.3）：本脚本自 2026-10-07 起基于
    # state_anchored_v2（加 x_last 锚点）；此前输出的 candidate 处于
    # unanchored_v1 口径（≈persistence 量级），比较时必须区分
    from models.tfm import TFM_CANDIDATE_SCHEMA
    summary['candidate_schema'] = TFM_CANDIDATE_SCHEMA
    write_outputs(args_cli.out_dir, records, summary, roi_corr, n_experts)
    log.info('[tfm_experts] 输出目录: %s', args_cli.out_dir)
    log.info('[tfm_experts] top1_agreement=%.3f top2_recall=%.3f regret=%.5f '
             'spearman=%.3f oracle_gap=%.5f',
             summary['router_top1_agreement'], summary['router_top2_recall'],
             summary['router_regret_mean'],
             summary['router_competence_spearman_mean'],
             summary['oracle_best_expert_gap'])
    print(json.dumps({k: v for k, v in summary.items()
                      if k != 'competence_matrix'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
