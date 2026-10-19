# coding=utf-8
"""S2：旧权重统一重评——逐样本专家能力矩阵、可行 oracle 与策略对照
（RECC 执行计划 §6）。

对 moe checkpoint 在 val/test 评估协议（与 evaluate_variant 同一被试切分与
确定性 anchor）上以 **state_anchored_v2 契约** 重算：

- 逐 anchor：每专家 MAE/Huber（原始空间）、mixture（正式输出）、shared-only、
  Router full probs 与实际 routing weights、q 分布熵、pair candidate 误差；
- 策略对照（先被试内、再跨被试聚合）：shared-only、mixture、均匀混合、
  逐固定专家（含 LOSO cross-fit 版）、单专家 oracle、等权 pair oracle、
  实际 Top-2 pair（等权 / Router 配比 / 连续最优权重）；
- latent 修正量与 decoder 可见能量（activation 级，验证 E2 零空间解释）、
  相对 shared-only 的逐专家增量收益；
- 平局样本单列（spread < tie_tol），不强制精细排名。

oracle 使用未来真值，只能作事后诊断，不等于部署收益（§6.2）。主收益指标为
MAE；Huber 仅作训练监督口径分列，两者不做倍数/差值互比（§5.3）。

产物（out_dir）：reeval_summary.json / reeval_per_anchor.csv /
reeval_per_subject.csv / reeval_policy_table.csv / reeval_arrays.npz。
npz 含逐 anchor 输出数组，属受控存储：不得提交含被试标识的大数组（§6.1）。

用法::

    python -m analysis.reeval_tfm_experts --ckpt <finetuned_best.pt> \
        --out_dir results/reeval_B_best --split val [--limit_subjects 4]
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_variant import build_model_from_args, load_train_args
from train.tfm_coupling import compute_expert_competence_losses
from utils.common import set_seed


# ────────────────────────── 工具 ──────────────────────────

def _subject_mean(values: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """任务级数值 → 被试均值（先被试内、再跨被试，§5.3 统一口径）。"""
    uniq = np.unique(subj)
    return np.array([values[subj == s].mean() for s in uniq], dtype=np.float64)


def _spearman_rows(p: np.ndarray, neg_ell: np.ndarray, eps: float = 1e-8):
    ra = p.argsort(axis=1).argsort(axis=1).astype(np.float64)
    rb = neg_ell.argsort(axis=1).argsort(axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    return (ra * rb).sum(axis=1) / (
        np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1)) + eps)


def _optimal_pair_weight_mae(cand_i: np.ndarray, cand_j: np.ndarray,
                             target: np.ndarray) -> np.ndarray:
    """逐 anchor 的 MAE 最优凸组合权重 w∈[0,1]（一维分段线性凸 → 断点精确解）。

    g(w) = mean_f |w·(ci−cj)_f + (cj−y)_f|；候选集 = {0,1} ∪ clip(断点,0,1)。
    ci/cj/target 均为 [N,F]（可整批），返回 [N]。
    """
    d = cand_i - cand_j                                   # [N,F]
    r = target - cand_j                                   # [N,F]
    safe = np.abs(d) > 1e-12
    breakpoints = np.where(safe, r / np.where(safe, d, 1.0), 0.5)
    W = np.concatenate([np.zeros((len(d), 1)),
                        np.ones((len(d), 1)),
                        np.clip(breakpoints, 0.0, 1.0)], axis=1)  # [N,F+2]
    preds = W[:, :, None] * d[:, None, :] + cand_j[:, None, :]   # [N,K,F]
    losses = np.abs(preds - target[:, None, :]).mean(axis=-1)    # [N,K]
    return W[np.arange(len(d)), losses.argmin(axis=1)]


# ────────────────────────── 收集 ──────────────────────────

@torch.no_grad()
def collect(model, data_loader, device, huber_delta: float, tau: float,
            limit_subjects: int = 0):
    """遍历评估集，返回逐 anchor 字典数组与 latent 统计（activation 级）。"""
    model.eval()
    records = []
    seen_subjects = set()
    stop = False
    for batch in data_loader:
        if stop:
            break
        x = batch['x'].to(device, non_blocking=True)
        sc = batch['sc'].to(device, non_blocking=True)
        score = batch.get('pathology_score', None)
        if score is not None:
            score = score.to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)              # [B,F,H=1]
        target2d = y.reshape(y.shape[0], y.shape[1], -1)[:, :, 0]  # [B,F]
        pred, aux = model(x, sc, score, return_expert_pred=True,
                          return_pair_pred=True)
        cand = aux['tfm_expert_pred'].float()                     # [B,E,F] raw
        shared = aux['tfm_shared_pred'].float()                   # [B,F] raw
        mixture = pred[:, :, 0, 0].float()                        # [B,F] raw
        pair = aux['tfm_pair_pred'].float()                       # [B,P,F] raw
        pair_idx = aux['tfm_pair_indices'].cpu().numpy()          # [P,2]
        t = target2d
        ell_mae = (cand - t.unsqueeze(1)).abs().mean(dim=-1).cpu().numpy()
        ell_hub = compute_expert_competence_losses(
            cand, t, huber_delta=huber_delta).cpu().numpy()
        mix_mae = (mixture - t).abs().mean(dim=-1).cpu().numpy()
        mix_hub = compute_expert_competence_losses(
            mixture.unsqueeze(1), t, huber_delta=huber_delta
        ).squeeze(1).cpu().numpy()
        sh_mae = (shared - t).abs().mean(dim=-1).cpu().numpy()
        sh_hub = compute_expert_competence_losses(
            shared.unsqueeze(1), t, huber_delta=huber_delta
        ).squeeze(1).cpu().numpy()
        pair_mae = (pair - t.unsqueeze(1)).abs().mean(dim=-1).cpu().numpy()
        pair_hub = compute_expert_competence_losses(
            pair, t, huber_delta=huber_delta).cpu().numpy()
        probs = aux['tfm_router_probs'].float().cpu().numpy()
        weights = aux['tfm_router_weights'].float().cpu().numpy()
        topk = aux['tfm_topk_indices'].cpu().numpy()
        # q 分布（与训练 competence 同口径：z-score + 温度 softmax，§19）
        ell_t = torch.from_numpy(ell_hub)
        q = torch.softmax(-(ell_t - ell_t.mean(-1, keepdim=True))
                          / (ell_t.std(-1, keepdim=True) + 1e-8) / tau,
                          dim=-1).numpy()
        # latent 统计：||Δz_i||、decoder 可见能量比、decoded raw 修正幅度
        delta_z = aux.get('tfm_expert_delta_z_norm')
        latent_norm = vis_ratio = raw_corr = None
        if torch.is_tensor(delta_z):
            dz = delta_z.float()                                  # [B,E,F,D]
            latent_norm = dz.norm(dim=-1).mean(dim=-1).cpu().numpy()   # [B,E]
            w = model.one_step_head.decode.weight.detach().float()     # [1,D]
            vis = torch.einsum('d,befd->bef', w.view(-1), dz)          # [B,E,F]
            vis_ratio = ((vis ** 2).sum(dim=-1)
                         / ((dz ** 2).sum(dim=(-1, -2)) + 1e-12)
                        ).cpu().numpy()                           # [B,E]
            raw_corr = (cand - shared.unsqueeze(1)).abs().mean(
                dim=-1).cpu().numpy()                             # [B,E]
        subj = np.array([str(s) for s in batch['subj_id']])
        cutoffs = batch['cutoff'].numpy()
        ctx = batch['context_len'].numpy()
        hamd = (score.reshape(-1).cpu().numpy() if score is not None
                else np.full(len(subj), np.nan))
        b = len(subj)
        for i in range(b):
            records.append({
                'subject': subj[i], 'cutoff': int(cutoffs[i]),
                'context_len': int(ctx[i]), 'hamd': float(hamd[i]),
                'ell_mae': ell_mae[i], 'ell_hub': ell_hub[i],
                'mix_mae': float(mix_mae[i]), 'mix_hub': float(mix_hub[i]),
                'sh_mae': float(sh_mae[i]), 'sh_hub': float(sh_hub[i]),
                'pair_mae': pair_mae[i], 'pair_hub': pair_hub[i],
                'probs': probs[i], 'weights': weights[i],
                'topk': topk[i], 'q_entropy': float(
                    -(q[i] * np.log(q[i] + 1e-12)).sum()),
                'pair_idx': pair_idx,
                'cand': cand[i].cpu().numpy(), 'shared': shared[i].cpu().numpy(),
                'mixture': mixture[i].cpu().numpy(), 'target': t[i].cpu().numpy(),
                'pair': pair[i].cpu().numpy(),
                'latent_norm': (latent_norm[i] if latent_norm is not None
                                else None),
                'vis_ratio': (vis_ratio[i] if vis_ratio is not None else None),
                'raw_corr': (raw_corr[i] if raw_corr is not None else None),
            })
        seen_subjects.update(subj.tolist())
        if limit_subjects and len(seen_subjects) >= limit_subjects:
            stop = True                                   # 小样本首评即停
    return records


# ────────────────────────── 汇总 ──────────────────────────

def build_summaries(records, n_experts, huber_delta, tau, tie_tol):
    subj = np.array([r['subject'] for r in records])
    ell = np.stack([r['ell_mae'] for r in records])            # [N,E] MAE
    ellh = np.stack([r['ell_hub'] for r in records])           # [N,E] Huber
    cand = np.stack([r['cand'] for r in records])              # [N,E,F]
    target = np.stack([r['target'] for r in records])
    mix_mae = np.array([r['mix_mae'] for r in records])
    sh_mae = np.array([r['sh_mae'] for r in records])
    probs = np.stack([r['probs'] for r in records])
    weights = np.stack([r['weights'] for r in records])
    pair_mae = np.stack([r['pair_mae'] for r in records])      # [N,P]
    pair_idx = records[0]['pair_idx']
    n, p = pair_mae.shape

    def agg(v):                                                # 主口径
        return float(_subject_mean(v, subj).mean())

    spread = ell.max(axis=1) - ell.min(axis=1)
    spreadh = ellh.max(axis=1) - ellh.min(axis=1)
    mixture_smae = agg(mix_mae)
    benefit_threshold = 0.01 * mixture_smae                    # G2 建议门槛

    # ---- 策略对照（subject 聚合 MAE） ----
    uniform = agg(np.abs(cand.mean(axis=1) - target).mean(axis=1))
    fixed = {f'expert_{j}': agg(ell[:, j]) for j in range(n_experts)}
    fixed_insample = agg(ell[:, int(ell.mean(axis=0).argmin())])
    # LOSO cross-fit 固定专家：对每个被试用其余被试选专家（去训练集乐观）
    best_other = {}
    for s in np.unique(subj):
        mask = subj != s
        best_other[s] = int(ell[mask].mean(axis=0).argmin())
    cf = np.array([ell[i, best_other[subj[i]]] for i in range(n)])
    fixed_crossfit = agg(cf)
    single_oracle = agg(ell.min(axis=1))
    pair_eq_oracle = agg(pair_mae.min(axis=1))
    # 实际 Top-2 pair（等权 / Router 配比 / 连续最优）——批量计算
    pair_lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(pair_idx)}
    top2 = np.sort(np.argsort(-probs, axis=1)[:, :2], axis=1)     # [N,2]
    actual_k = np.array([pair_lookup[(int(a), int(b))] for a, b in top2])
    pair_actual_eq = agg(pair_mae[np.arange(n), actual_k])
    pair_actual_rw = mixture_smae        # Top-2 外权重=0 → mixture 即实际配比
    rows = np.arange(n)
    ci = cand[rows, top2[:, 0]]
    cj = cand[rows, top2[:, 1]]
    w_actual = _optimal_pair_weight_mae(ci, cj, target)            # [N]
    pred_actual = w_actual[:, None] * ci + (1 - w_actual[:, None]) * cj
    pair_actual_opt = agg(np.abs(pred_actual - target).mean(axis=1))
    # 全 pair 连续 oracle（事后上界参考，含 look-ahead）
    pair_opt_all = np.empty((n, p))
    for k in range(p):
        ci_k = cand[:, pair_idx[k, 0]]
        cj_k = cand[:, pair_idx[k, 1]]
        w_k = _optimal_pair_weight_mae(ci_k, cj_k, target)
        pred_k = w_k[:, None] * ci_k + (1 - w_k[:, None]) * cj_k
        pair_opt_all[:, k] = np.abs(pred_k - target).mean(axis=1)
    pair_any_opt = agg(pair_opt_all.min(axis=1))

    # ---- Router 机制指标 ----
    top1_router = probs.argmax(axis=1)
    top1_oracle = ell.argmin(axis=1)
    tie_mask = spread < tie_tol
    agreement = float((top1_router == top1_oracle)[~tie_mask].mean()
                      if (~tie_mask).any() else float('nan'))
    agreement_all = float((top1_router == top1_oracle).mean())
    o2 = np.argsort(ell, axis=1)[:, :2]
    top2_recall = float(np.mean([
        len(set(top2[i].tolist()) & set(o2[i].tolist())) / 2 for i in range(n)]))
    regret = ell[np.arange(n), top1_router] - ell.min(axis=1)
    spearman = float(_spearman_rows(probs, -ell).mean())

    # ---- 潜在收益空间（G2 输入） ----
    policies = {
        'shared_only': agg(sh_mae),
        'mixture_router': mixture_smae,
        'uniform_mix': uniform,
        'fixed_in_sample_best': fixed_insample,
        'fixed_crossfit': fixed_crossfit,
        'single_expert_oracle': single_oracle,
        'pair_actual_top2_eq': pair_actual_eq,
        'pair_actual_top2_router_w': pair_actual_rw,
        'pair_actual_top2_optimal_w': pair_actual_opt,
        'pair_equal_weight_oracle': pair_eq_oracle,
        'pair_any_optimal_w_oracle': pair_any_opt,
        **fixed,
    }
    deployable = {k: v for k, v in policies.items()
                  if 'oracle' not in k and 'optimal' not in k}
    # pair_actual_top2_optimal_w 的 w 由未来真值解出，属半事后上界，不算部署策略
    best_deployable = min(deployable, key=deployable.get)

    summary = {
        'n_anchors': n, 'n_subjects': int(len(np.unique(subj))),
        'n_experts': n_experts, 'huber_delta': huber_delta,
        'competence_tau': tau, 'tie_tol_mae': tie_tol,
        'aggregation': 'subject_mean_then_mean（先被试内、再跨被试）',
        'primary_metric': 'MAE(raw space, state_anchored_v2 candidates)',
        'mixture_subject_mae': mixture_smae,
        'shared_only_subject_mae': agg(sh_mae),
        'incremental_benefit_mixture_vs_shared_pct':
            100.0 * (agg(sh_mae) - mixture_smae) / agg(sh_mae),
        'per_expert_incremental_vs_shared_pct': [
            100.0 * (agg(sh_mae) - agg(ell[:, j])) / agg(sh_mae)
            for j in range(n_experts)],
        'per_expert_mean_mae': [float(ell[:, j].mean())
                                for j in range(n_experts)],
        'per_expert_mean_huber': [float(ellh[:, j].mean())
                                  for j in range(n_experts)],
        'per_sample_spread_mae': {
            'mean': float(spread.mean()), 'median': float(np.median(spread)),
            'q10': float(np.quantile(spread, .1)),
            'q25': float(np.quantile(spread, .25)),
            'q75': float(np.quantile(spread, .75)),
            'q90': float(np.quantile(spread, .9)),
            'share_above_benefit_threshold': float(
                (spread > benefit_threshold).mean()),
            'benefit_threshold_1pct_mixture': benefit_threshold},
        'per_sample_spread_huber_mean': float(spreadh.mean()),
        'tie_fraction_below_tol': float(tie_mask.mean()),
        'best_expert_switch_rate_between_adjacent_anchors':
            _switch_rate(records, top1_oracle),
        'policies_subject_mae': policies,
        'best_deployable_policy': best_deployable,
        'gap_best_deployable_to_single_oracle':
            deployable[best_deployable] - single_oracle,
        'gap_mixture_to_single_oracle': mixture_smae - single_oracle,
        'router_top1_agreement_excl_ties': agreement,
        'router_top1_agreement_all': agreement_all,
        'router_top2_recall': top2_recall,
        'router_regret_mean_mae': float(regret.mean()),
        'router_regret_mean_huber': float(
            (ellh[np.arange(n), top1_router] - ellh.min(axis=1)).mean()),
        'router_competence_spearman_mae': spearman,
        'q_entropy_mean': float(np.mean([r['q_entropy'] for r in records])),
        'router_top1_share': [float((top1_router == j).mean())
                              for j in range(n_experts)],
    }
    if records[0]['latent_norm'] is not None:
        ln = np.stack([r['latent_norm'] for r in records])
        vr = np.stack([r['vis_ratio'] for r in records])
        rc = np.stack([r['raw_corr'] for r in records])
        summary['latent_correction'] = {
            'delta_z_norm_mean_per_expert': ln.mean(axis=0).tolist(),
            'decoder_visible_energy_ratio_per_expert':
                vr.mean(axis=0).tolist(),
            'decoded_raw_correction_mae_per_expert': rc.mean(axis=0).tolist(),
            'note': 'activation 级可见能量（非参数级）；visible≪1 支持'
                    '零空间解释（问题核查 §3.3）',
        }
    return summary, {'spread': spread, 'top1_router': top1_router,
                     'top1_oracle': top1_oracle, 'top2': top2,
                     'opt_w': w_actual}


def _switch_rate(records, top1_oracle):
    """同被试相邻 cutoff 的 oracle 最佳专家切换率（§6.2-2）。"""
    by_subj = defaultdict(list)
    for r, b in zip(records, top1_oracle):
        by_subj[r['subject']].append((r['cutoff'], int(b)))
    switches = total = 0
    for rows in by_subj.values():
        rows.sort()
        for (_, a), (_, b) in zip(rows, rows[1:]):
            total += 1
            switches += int(a != b)
    return float(switches / total) if total else float('nan')


# ────────────────────────── 输出 ──────────────────────────

def write_outputs(out_dir, records, summary, extras):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    e = summary['n_experts']
    pair_idx = records[0]['pair_idx']
    p = len(pair_idx)
    top1_oracle = extras['top1_oracle']

    with open(out_dir / 'reeval_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / 'reeval_per_anchor.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        head = (['subject', 'cutoff', 'context_len', 'hamd',
                 'mix_mae', 'sh_mae', 'oracle_best_mae', 'router_top1',
                 'spread_mae', 'q_entropy', 'opt_w_actual_pair']
                + [f'mae_e{j}' for j in range(e)]
                + [f'huber_e{j}' for j in range(e)]
                + [f'prob_e{j}' for j in range(e)]
                + [f'weight_e{j}' for j in range(e)]
                + [f'pair_mae_k{k}' for k in range(p)])
        w.writerow(head)
        for i, r in enumerate(records):
            w.writerow([r['subject'], r['cutoff'], r['context_len'],
                        f"{r['hamd']:.6f}",
                        f"{r['mix_mae']:.6f}", f"{r['sh_mae']:.6f}",
                        f"{top1_oracle[i]}", int(extras['top1_router'][i]),
                        f"{extras['spread'][i]:.6f}",
                        f"{r['q_entropy']:.6f}",
                        f"{extras['opt_w'][i]:.6f}"]
                       + [f"{v:.6f}" for v in r['ell_mae']]
                       + [f"{v:.6f}" for v in r['ell_hub']]
                       + [f"{v:.6f}" for v in r['probs']]
                       + [f"{v:.6f}" for v in r['weights']]
                       + [f"{v:.6f}" for v in r['pair_mae']])

    # 逐被试聚合
    subj_ids = sorted({r['subject'] for r in records})
    with open(out_dir / 'reeval_per_subject.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'hamd_mean', 'n_anchors', 'mix_mae', 'sh_mae',
                    'oracle_best_mae', 'uniform_mix_mae', 'spread_mae_median']
                   + [f'mae_e{j}' for j in range(e)])
        for s in subj_ids:
            rows = [r for r in records if r['subject'] == s]
            idx = [i for i, r in enumerate(records) if r['subject'] == s]
            ell = np.stack([records[i]['ell_mae'] for i in idx])
            cand = np.stack([records[i]['cand'] for i in idx])
            tgt = np.stack([records[i]['target'] for i in idx])
            w.writerow([s, f"{np.mean([r['hamd'] for r in rows]):.4f}",
                        len(rows),
                        f"{np.mean([r['mix_mae'] for r in rows]):.6f}",
                        f"{np.mean([r['sh_mae'] for r in rows]):.6f}",
                        f"{ell.min(axis=1).mean():.6f}",
                        f"{np.abs(cand.mean(axis=1) - tgt).mean():.6f}",
                        f"{np.median(ell.max(1) - ell.min(1)):.6f}"]
                       + [f"{ell[:, j].mean():.6f}" for j in range(e)])

    with open(out_dir / 'reeval_policy_table.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['policy', 'subject_mae', 'rel_vs_mixture_pct', 'note'])
        mix = summary['policies_subject_mae']['mixture_router']
        notes = {
            'single_expert_oracle': '事后上界（使用未来真值），非部署收益',
            'pair_equal_weight_oracle': '事后上界（等权 pair，look-ahead）',
            'pair_any_optimal_w_oracle': '事后上界（全 pair 连续权重，look-ahead）',
            'pair_actual_top2_optimal_w': '半事后上界：实际 Top-2 pair 的 w 由'
                                          '未来真值解出，非部署策略',
            'pair_actual_top2_router_w': '等于 mixture（Top-2 外权重为 0）',
            'fixed_crossfit': 'LOSO 被试级 cross-fit 的固定最佳专家（可部署）',
            'fixed_in_sample_best': '全样本选择的固定专家（乐观，供参考）',
        }
        for name, v in summary['policies_subject_mae'].items():
            w.writerow([name, f"{v:.6f}", f"{100.0 * (mix - v) / mix:.4f}",
                        notes.get(name, '')])

    # 受控数组存储：npz 不入版本控制（含被试标识）
    np.savez_compressed(
        out_dir / 'reeval_arrays.npz',
        cand=np.stack([r['cand'] for r in records]),
        target=np.stack([r['target'] for r in records]),
        shared=np.stack([r['shared'] for r in records]),
        mixture=np.stack([r['mixture'] for r in records]),
        pair=np.stack([r['pair'] for r in records]),
        pair_idx=pair_idx,
        probs=np.stack([r['probs'] for r in records]),
        weights=np.stack([r['weights'] for r in records]),
        subj=np.array([r['subject'] for r in records]),
        cutoff=np.array([r['cutoff'] for r in records]),
        hamd=np.array([r['hamd'] for r in records]),
    )


def main():
    ap = argparse.ArgumentParser(description='S2 旧权重统一重评（只读推理）')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--limit_subjects', type=int, default=0,
                    help='>0 时只评前 N 个被试（小样本首评，§6.1）')
    ap.add_argument('--device', default='',
                    help='空 = 自动（有 GPU 用 cuda:0）')
    ap.add_argument('--tie_tol', type=float, default=1e-4,
                    help='逐样本 MAE 平局容差（绝对量纲）')
    ap.add_argument('--huber_delta', type=float, default=None,
                    help='默认取 checkpoint 快照 cpm_huber_delta')
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

    from analysis import next_point_eval as npe
    data_loader, subjects, _ = npe.build_next_point_loader(
        train_args, args_cli.split,
        refresh_split_manifest=args_cli.refresh_split_manifest)
    n_experts = int(getattr(train_args, 'tfm_num_experts', 4))
    tau = float(getattr(train_args, 'tfm_competence_temperature', 1.0))
    huber_delta = float(args_cli.huber_delta
                        or getattr(train_args, 'cpm_huber_delta', 1.0))
    set_seed(2024)

    print(f'[reeval] ckpt={args_cli.ckpt} split={args_cli.split} '
          f'subjects={len(subjects)} limit={args_cli.limit_subjects or "all"} '
          f'device={device}')
    records = collect(model, data_loader, device, huber_delta, tau,
                      args_cli.limit_subjects)
    if args_cli.limit_subjects:
        # 精确截断到先出现的 N 个被试（collect 停止条件按 batch 粒度，会超采）
        seen = []
        for r in records:
            if r['subject'] not in seen:
                seen.append(r['subject'])
        keep = set(seen[:args_cli.limit_subjects])
        records = [r for r in records if r['subject'] in keep]
    summary, extras = build_summaries(records, n_experts, huber_delta, tau,
                                      args_cli.tie_tol)
    summary['checkpoint'] = args_cli.ckpt
    summary['checkpoint_sha256_16'] = hashlib.sha256(
        Path(args_cli.ckpt).read_bytes()).hexdigest()[:16]
    summary['split'] = args_cli.split
    summary['candidate_schema'] = 'state_anchored_v2_20261007'
    summary['training_competence_schema'] = (
        'unanchored_v1_inferred（历史运行，≤2026-10-07；复评不等于修复训练，§6.3）')
    summary['partial'] = bool(args_cli.limit_subjects
                              and summary['n_subjects'] < len(subjects))
    write_outputs(args_cli.out_dir, records, summary, extras)
    pol = summary['policies_subject_mae']
    print(json.dumps({k: summary[k] for k in (
        'n_anchors', 'n_subjects', 'mixture_subject_mae',
        'per_sample_spread_mae', 'tie_fraction_below_tol',
        'best_deployable_policy',
        'gap_best_deployable_to_single_oracle',
        'gap_mixture_to_single_oracle')}, ensure_ascii=False, indent=2))
    print('[reeval] policies:', json.dumps(pol, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
