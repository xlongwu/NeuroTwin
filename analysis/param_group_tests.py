# coding=utf-8
"""MINDy 式参数组间分析（冻结权重，不新增训练）。

触发背景（group_tests.json T2_judgment）：模型偏离信号相对六描述量基线
无增量（显著 ROI 数 0 vs 0）。按 v2 §7 预登记动作，对模型**内置个体参数**
做匹配子集上的组间比较，复用 B7 的站点块置换 + FDR 框架。

分析对象：模型的个体 profile 码 ``p``（``personalized=True`` 的
``profile_encoder`` 对每被试最早 ``calibration_length=32`` TR 前缀一次编码，
``[profile_dim]``）。p 是冻结模型的有效个体参数摘要（动力学个性化基底），
不是观测描述量——与 MINDy 对个体化有效参数做组间比较的逻辑同构。

检验：匹配子集 1504 对，逐 profile 维配对差 + 站点块置换（10,000 次，
seed=2024）+ BH-FDR（q=0.05）+ bootstrap CI；另报 profile 范数与余弦相似度
的组间差异（MINDy 文献常用摘要量）。输出 ``param_group_tests.json`` /
``param_dim_table.csv``。解释边界：组间差不等于病理因果；p 的语义由模型
训练目标决定。

用法::

    python -m analysis.param_group_tests \
      --ckpt checkpoints/neurotwin_pretrain/base_best.pt
"""
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from analysis.pathology_tests import (bh_fdr, paired_bootstrap_ci,
                                      site_blocked_paired_permutation)

log = logging.getLogger(__name__)
DEFAULT_SUBSET = Path('data/splits/site_matched_subset.json')


@torch.no_grad()
def extract_profiles(model, loader, subjects: List[str], device) -> np.ndarray:
    """逐被试 profile：最早 calibration_length TR 前缀一次编码（[N, D]）。"""
    ds = loader.base_dataset
    rows = []
    for sid in subjects:
        series = ds.get_series(str(sid))                    # [F, T] 校准尺度
        L = int(getattr(ds, 'calibration_length', 32) or 32)
        cal = torch.from_numpy(
            np.ascontiguousarray(series[:, :L][:, None, :].astype(np.float32))
        ).unsqueeze(0).to(device)                           # [1, F, 1, L]
        rows.append(model.calibrate_profile(cal)[0].float().cpu().numpy())
    return np.stack(rows, axis=0)


def _build(args, group, data_root, condition_stats):
    from analysis.rolling_one_step import _build_loader_with_condition
    if group == 'hc':
        from main import condition_loader_kwargs
        kwargs = condition_loader_kwargs(args, condition_stats)
    else:
        from analysis.evaluate import condition_stats_from_bundle
        kwargs = {'condition_group': 'hc3', 'condition_file': None,
                  'condition_base_stats': condition_stats_from_bundle(
                      condition_stats, 'hc3'),
                  'condition_stats_mode': 'inherit', 'condition_control': 'none',
                  'split_file': str(Path(data_root)
                                    / 'subject_split_finetune.json'),
                  'stratify_key': 'Age'}
    return _build_loader_with_condition(args, group, data_root, 64, 8, 0, kwargs)


def main(argv: Optional[List[str]] = None) -> None:
    from analysis.checkpoint import load_checkpoint_compat
    from analysis.evaluate import (build_model_from_args,
                                              load_condition_bundle,
                                              load_train_args)
    from utils.common import set_seed

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', type=Path,
                    default=Path('checkpoints/'
                                 'neurotwin_pretrain/base_best.pt'))
    ap.add_argument('--data-root', type=Path, default=Path('data'))
    ap.add_argument('--subset', type=Path, default=DEFAULT_SUBSET)
    ap.add_argument('--out-dir', type=Path,
                    default=Path('results/pathology_analysis/run1'))
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = load_train_args(str(ns.ckpt))
    bundle = load_condition_bundle(str(ns.ckpt))
    set_seed(int(args.seed))
    model = build_model_from_args(args, device, condition_bundle=bundle)
    model, compat = load_checkpoint_compat(model, str(ns.ckpt), device, strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')
    if getattr(model, 'profile_encoder', None) is None:
        raise ValueError('checkpoint 未启用 personalized profile，无参数分析对象。')

    pairs = json.loads(ns.subset.read_text(encoding='utf-8'))['pairs']
    mdd_ids = [p['mdd'] for p in pairs]
    hc_ids = [p['hc'] for p in pairs]
    log.info('[T2] 匹配对 %d | 提取 profile（HC + MDD 各一次前缀编码）', len(pairs))

    hc_loader = _build(args, 'hc', ns.data_root, bundle)
    hc_p = extract_profiles(model, hc_loader, hc_ids, device)
    del hc_loader
    mdd_loader = _build(args, 'mdd', ns.data_root, bundle)
    mdd_p = extract_profiles(model, mdd_loader, mdd_ids, device)

    diff = mdd_p - hc_p                                     # [P, D]
    sites = [p['site'] for p in pairs]
    p_val = site_blocked_paired_permutation(diff, sites, n_perm=ns.n_perm,
                                            seed=ns.seed)
    q = bh_fdr(p_val)
    lo, hi = paired_bootstrap_ci(diff, n_boot=min(ns.n_boot, 5000), seed=ns.seed)
    n_sig = int((q < ns.fdr_q).sum())

    # MINDy 式摘要：范数与个体 profile 主方向（首奇异向量载荷投影）
    norm_m = np.linalg.norm(mdd_p, axis=1)
    norm_h = np.linalg.norm(hc_p, axis=1)
    u, s, vt = np.linalg.svd(0.5 * (mdd_p + hc_p), full_matrices=False)
    proj = (0.5 * (mdd_p + hc_p)) @ vt[0]
    summary_stats = {
        'norm_mean_mdd': float(norm_m.mean()), 'norm_mean_hc': float(norm_h.mean()),
        'norm_paired_delta': float((norm_m - norm_h).mean()),
        'norm_p': float(site_blocked_paired_permutation(
            (norm_m - norm_h)[:, None], sites, n_perm=ns.n_perm, seed=ns.seed)[0]),
        'svd_top_singular_value': float(s[0]),
        'svd_top_explained_ratio': float(s[0] ** 2 / (s ** 2).sum()),
    }

    rows = [{'dim': j, 'delta': float(diff[:, j].mean()),
             'p': float(p_val[j]), 'q': float(q[j]),
             'ci_lo': float(lo[j]), 'ci_hi': float(hi[j]),
             'significant': bool(q[j] < ns.fdr_q)} for j in range(diff.shape[1])]
    with (ns.out_dir / 'param_dim_table.csv').open('w', newline='',
                                                   encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    report = {
        'analysis': 'T2 指定扩展动作：MINDy 式参数组间分析（v2 §7）',
        'trigger': 'group_tests.json T2_judgment：模型信号相对描述量基线无增量',
        'object': '个体 profile p [D]（calibrate_profile，最早 32 TR 前缀，冻结权重）',
        'n_pairs': len(pairs), 'profile_dim': int(diff.shape[1]),
        'perm_settings': {'n_perm': ns.n_perm, 'n_boot': ns.n_boot,
                          'seed': ns.seed, 'fdr_q': ns.fdr_q},
        'n_significant_dim': n_sig,
        'max_abs_delta': float(np.abs(diff.mean(axis=0)).max()),
        'median_abs_delta': float(np.median(np.abs(diff.mean(axis=0)))),
        'summary_stats': summary_stats,
        'top5_dims': sorted(rows, key=lambda r: -abs(r['delta']))[:5],
        'interpretation': 'profile 是冻结模型的有效个体参数摘要；组间差（若有）'
                          '只说明模型参数化下两组可分，不等于病理因果；'
                          'p 值未考虑 profile 维度间的相关性。',
    }
    (ns.out_dir / 'param_group_tests.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('[T2] 完成 | D=%d | 显著维 %d @q<%.2f | max|Δ|=%.4f | 范数Δ=%.4f (p=%.4f)',
             diff.shape[1], n_sig, ns.fdr_q, report['max_abs_delta'],
             summary_stats['norm_paired_delta'], summary_stats['norm_p'])
    print(json.dumps({k: report[k] for k in (
        'n_pairs', 'profile_dim', 'n_significant_dim', 'max_abs_delta',
        'summary_stats', 'top5_dims')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
