# coding=utf-8
"""匹配子集组间检验、并排基线与临床关联。

输入（全部来自已完成的 v2 产物）：

- z 矩阵：``results/pathology_analysis/run1/z_maps/z_matrix_<group>_<split>.npz``
- 匹配子集：``data/splits/site_matched_subset.json``（1504 对，36 站）
- 人口学与临床：merged 表（hc3 / mdd8）
- 描述量基线：在线拟合 ``analysis/twin_reference`` 参照（HC train）并对匹配
  被试与 HC val/test 打分（只读数据，不重训动态模型）

检验框架（v2 §6，全部预登记）：

1. **逐 ROI / 网络级组间差异**（匹配子集，配对）：站点块置换（站内翻转配对
   符号，10,000 次，seed=2024）+ BH-FDR（q=0.05）+ 配对 bootstrap 95% CI；
2. **负控**：HC val vs HC test 同流程（站内置换组标签）；应无显著 ROI，
   出现显著先怀疑流程而非效应；
3. **离散度补充**（预登记）：被试级 mean|z| 的组间方差比，同站点块置换；
4. **并排基线**：六描述量 d_ref 走同一检验框架；静态 FC+岭回归（站内 5 折
   CV，OOF AUC）；
5. **临床关联**：网络级 z 与 HAMD/HAMA 偏相关（控制 Age/Sex/站点；
   HAMA 只报告完整观测集，不插补）；
6. **T2 判定**：模型信号相对描述量基线在「显著 ROI 数 × 效应量」上有无增量。

判读规则第 1 条：任何组间差异只说明队列可分，不等于病理机制定位。

用法::

    python -m analysis.pathology_tests --run-dir results/pathology_analysis/run1
"""
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.metrics import NETWORK_GROUPS, get_aal116_labels
from analysis.rolling_one_step import GROUP_SUBDIR
from utils.site_subset import parse_site

log = logging.getLogger(__name__)

N_PERM = 10000
SEED = 2024
N_BOOT = 10000
FDR_Q = 0.05
HC_MERGED_FILE = 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'
MDD_MERGED_FILE = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'
EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════
#  统计核心（纯函数，可独立测试）
# ══════════════════════════════════════════════════════════════════════════

def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """BH-FDR q 值（单调化；与 next_point_eval.bh_fdr 同实现）。"""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    n = p.size
    order = np.argsort(p, kind='mergesort')
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def site_blocked_paired_permutation(diff: np.ndarray, sites: Sequence[str],
                                    n_perm: int = N_PERM,
                                    seed: int = SEED) -> np.ndarray:
    """配对差异 [P, M] 的站点块置换 p（站内翻转符号；双侧）。

    统计量 = 逐列配对均值。p 用 (计数+1)/(n_perm+1) 平滑。
    """
    diff = np.asarray(diff, dtype=np.float64)
    p_total = diff.shape[0]
    obs = diff.mean(axis=0)
    rng = np.random.RandomState(int(seed))
    null_mat = np.zeros((int(n_perm), diff.shape[1]), dtype=np.float64)
    by_site: Dict[str, List[int]] = {}
    for i, s in enumerate(sites):
        by_site.setdefault(str(s), []).append(i)
    for idx in by_site.values():
        signs = rng.choice((-1.0, 1.0), size=(int(n_perm), len(idx)))
        null_mat += signs @ diff[idx]                    # [n_perm, M] 逐置换累加
    null_mat /= float(p_total)
    exceed = (np.abs(null_mat) >= np.abs(obs)[None, :] - 1e-12).sum(axis=0)
    return (exceed + 1) / (float(n_perm) + 1)


def site_blocked_unpaired_permutation(a: np.ndarray, b: np.ndarray,
                                      sites_a: Sequence[str],
                                      sites_b: Sequence[str],
                                      n_perm: int = N_PERM,
                                      seed: int = SEED) -> np.ndarray:
    """两组 [N, M] 的站点块置换 p（站内置换组标签；双侧）。

    统计量与 null 同构：``Σ_s (n_s/N)·(mean_a_s − mean_b_s)``（站点人数加权的
    组间均值差；置换在站内抽子集，向量化的无放回子集采样）。p 用
    (计数+1)/(n_perm+1) 平滑。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f'a/b 列数不一致：{a.shape[1]} vs {b.shape[1]}')
    by_site: Dict[str, Dict[str, List[int]]] = {}
    for i, s in enumerate(sites_a):
        by_site.setdefault(str(s), {'a': [], 'b': []})['a'].append(i)
    for i, s in enumerate(sites_b):
        by_site.setdefault(str(s), {'a': [], 'b': []})['b'].append(i)
    n_total = a.shape[0] + b.shape[0]
    obs = np.zeros(a.shape[1], dtype=np.float64)
    null_mat = np.zeros((int(n_perm), a.shape[1]), dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    for idx in by_site.values():
        ia, ib = np.asarray(idx['a'], dtype=int), np.asarray(idx['b'], dtype=int)
        n_s = len(ia) + len(ib)
        if n_s < 2 or len(ia) == 0 or len(ib) == 0:
            continue
        pooled = np.concatenate([a[ia], b[ib]], axis=0)          # [n_s, M]
        w = n_s / float(n_total)
        obs += w * (a[ia].mean(axis=0) - b[ib].mean(axis=0))
        order = np.argsort(rng.rand(int(n_perm), n_s), axis=1)    # 随机无放回子集
        mask_a = np.zeros((int(n_perm), n_s), dtype=np.float64)
        np.put_along_axis(mask_a, order[:, :len(ia)], 1.0, axis=1)
        sum_a = mask_a @ pooled                                   # [n_perm, M]
        sum_b = pooled.sum(axis=0)[None, :] - sum_a
        null_mat += w * (sum_a / len(ia) - sum_b / len(ib))
    exceed = (np.abs(null_mat) >= np.abs(obs)[None, :] - 1e-12).sum(axis=0)
    return (exceed + 1) / (float(n_perm) + 1)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOT,
                        seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """配对均值的 percentile bootstrap 95% CI。"""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, diff.shape[0], size=(int(n_boot), diff.shape[0]))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def network_columns(n_rois: int = 116) -> Dict[str, np.ndarray]:
    """网络名 → ROI 列（0 基；未命名 ROI 归 Other）。"""
    cols: Dict[str, List[int]] = {'Other': []}
    for net, regions in NETWORK_GROUPS.items():
        cols[net] = [r - 1 for r in regions if 1 <= r <= n_rois]
    cols['Other'] = [j for j in range(n_rois)
                     if all(j + 1 not in regs for regs in NETWORK_GROUPS.values())]
    return cols


def partial_corr(y: np.ndarray, x: np.ndarray, controls: np.ndarray
                 ) -> Dict[str, float]:
    """残差化偏相关（OLS 残差的 Pearson r；t 近似 p）。"""
    from scipy import stats
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    C = np.asarray(controls, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if C.shape[0] != len(y):
        C = C.T
    design = np.concatenate([np.ones((len(y), 1)), C], axis=1)
    beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    ry = y - design @ beta_y
    rx = x - design @ beta_x
    if rx.std() < EPS or ry.std() < EPS:
        return {'r': float('nan'), 'p': float('nan'), 'n': int(len(y))}
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = max(len(y) - design.shape[1] - 1, 1)
    t = r * np.sqrt(dof / max(1 - r ** 2, EPS))
    return {'r': r, 'p': float(2 * stats.t.sf(abs(t), dof)),
            'n': int(len(y)), 'dof': int(dof)}


# ══════════════════════════════════════════════════════════════════════════
#  数据装配
# ══════════════════════════════════════════════════════════════════════════

def load_z(run_dir: Path, group: str, split: str) -> Tuple[List[str], np.ndarray]:
    path = Path(run_dir) / 'z_maps' / f'z_matrix_{group}_{split}.npz'
    if not path.exists():
        raise FileNotFoundError(f'缺少 {path}')
    with np.load(path) as z:
        return [str(s) for s in z['subjects']], z['z'].astype(np.float64)


def load_matched_pairs(subset_path: Path) -> List[dict]:
    payload = json.loads(Path(subset_path).read_text(encoding='utf-8'))
    if not payload.get('pairs'):
        raise ValueError(f'{subset_path} 无匹配对；请先运行 utils.site_subset --match')
    return payload['pairs']


def _demo_rows(data_root: Path, merged_file: str,
               subjects: Sequence[str]) -> List[Dict[str, Optional[float]]]:
    from utils.dataloader import load_condition_table
    table = load_condition_table(str(data_root), merged_file, 'hc3')
    out = []
    for sid in subjects:
        row = table.row_of_id.get(str(sid), -1)
        demo = {'Age': None, 'education': None, 'Sex': None}
        if row >= 0:
            for j, spec in enumerate(table.specs):
                if spec.name in demo and int(table.mask[row, j]):
                    v = float(table.values[row, j])
                    if np.isfinite(v):
                        demo[spec.name] = v
        out.append(demo)
    return out


def collect_descriptors(data_root: Path, ckpt: Path,
                        hc_subjects: Sequence[str], mdd_subjects: Sequence[str]
                        ) -> Tuple[Dict[str, np.ndarray], dict]:
    """六描述量基线：fit（HC train）+ 逐被试 d_ref [F]。

    返回 ({subject_id: d_ref}, fit 元信息)。序列只读（raw 校准尺度，最早 64 TR）。
    """
    from analysis.rolling_one_step import (_build_loader_with_condition,
                                           get_split_subjects)
    from analysis.twin_reference import fit_reference, score_reference
    from analysis.evaluate import (load_condition_bundle,
                                              load_train_args)
    from main import condition_loader_kwargs

    args = load_train_args(str(ckpt))
    bundle = load_condition_bundle(str(ckpt))
    hc_loader = _build_loader_with_condition(
        args, 'hc', data_root, 64, 8, 0, condition_loader_kwargs(args, bundle))
    payload = fit_reference(hc_loader.base_dataset,
                            hc_loader.get_train_subjects())
    log.info('[descriptor] 参照拟合 | HC train=%d | σ 中位=%.4f',
             payload['n_subjects'], float(np.median(payload['s_ref'])))

    demo_cache: Dict[str, Dict] = {}
    out: Dict[str, np.ndarray] = {}
    hc_demo = _demo_rows(data_root, HC_MERGED_FILE, hc_subjects)
    for sid, demo in zip(hc_subjects, hc_demo):
        demo_cache[str(sid)] = demo
        hist = hc_loader.base_dataset.get_series(str(sid))[:, :64].T
        out[str(sid)] = score_reference(hist[None], demo, payload)[
            'normal_distance'][0]

    if mdd_subjects:
        from analysis.evaluate import condition_stats_from_bundle
        mdd_loader = _build_loader_with_condition(
            args, 'mdd', data_root, 64, 8, 0,
            {'condition_group': 'hc3', 'condition_file': None,
             'condition_base_stats': condition_stats_from_bundle(bundle, 'hc3'),
             'condition_stats_mode': 'inherit', 'condition_control': 'none',
             'split_file': str(Path(data_root) / 'subject_split_finetune.json'),
             'stratify_key': 'Age'})
        mdd_demo = _demo_rows(data_root, MDD_MERGED_FILE, mdd_subjects)
        for sid, demo in zip(mdd_subjects, mdd_demo):
            demo_cache[str(sid)] = demo
            hist = mdd_loader.base_dataset.get_series(str(sid))[:, :64].T
            out[str(sid)] = score_reference(hist[None], demo, payload)[
                'normal_distance'][0]
    meta = {'n_train': payload['n_subjects'],
            'sigma_median': float(np.median(payload['s_ref'])),
            'demo_cache': demo_cache}
    return out, meta


def static_fc_features(data_root: Path, ckpt: Path, group: str,
                       subjects: Sequence[str]) -> np.ndarray:
    """最早 64 TR 的静态 FC 上三角（与描述量同一窗口，只读）。"""
    from analysis.rolling_one_step import _build_loader_with_condition
    from analysis.evaluate import (load_condition_bundle,
                                              load_train_args)
    from main import condition_loader_kwargs
    args = load_train_args(str(ckpt))
    bundle = load_condition_bundle(str(ckpt))
    if group == 'hc':
        loader = _build_loader_with_condition(
            args, 'hc', data_root, 64, 8, 0, condition_loader_kwargs(args, bundle))
    else:
        from analysis.evaluate import condition_stats_from_bundle
        loader = _build_loader_with_condition(
            args, 'mdd', data_root, 64, 8, 0,
            {'condition_group': 'hc3', 'condition_file': None,
             'condition_base_stats': condition_stats_from_bundle(bundle, 'hc3'),
             'condition_stats_mode': 'inherit', 'condition_control': 'none',
             'split_file': str(Path(data_root) / 'subject_split_finetune.json'),
             'stratify_key': 'Age'})
    iu = np.triu_indices(116, k=1)
    feats = []
    for sid in subjects:
        hist = loader.base_dataset.get_series(str(sid))[:, :64]   # [F, 64]
        h = hist - hist.mean(axis=1, keepdims=True)
        h = h / (h.std(axis=1, keepdims=True) + 1e-6)
        fc = (h @ h.T) / hist.shape[1]
        feats.append(fc[iu])
    return np.stack(feats, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  检验流程
# ══════════════════════════════════════════════════════════════════════════

def roi_test_paired(z_by_subj: Dict[str, np.ndarray], pairs: List[dict],
                    n_perm: int = N_PERM, n_boot: int = N_BOOT,
                    seed: int = SEED) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """匹配子集配对检验 → (diff_mean, p, q, ci_lo, ci_hi 各 [M])。"""
    mdd_ids = [p['mdd'] for p in pairs]
    hc_ids = [p['hc'] for p in pairs]
    missing = [s for s in mdd_ids + hc_ids if s not in z_by_subj]
    if missing:
        raise ValueError(f'z 矩阵缺少匹配被试 {missing[:5]}（共 {len(missing)}）')
    diff = np.stack([np.atleast_1d(z_by_subj[m] - z_by_subj[h])
                     for m, h in zip(mdd_ids, hc_ids)], axis=0)   # [P, M>=1]
    if diff.ndim != 2:
        diff = diff.reshape(diff.shape[0], -1)
    sites = [p['site'] for p in pairs]
    p_val = site_blocked_paired_permutation(diff, sites, n_perm=n_perm, seed=seed)
    q = bh_fdr(p_val)
    lo, hi = paired_bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    return diff.mean(axis=0), p_val, q, lo, hi


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path,
                    default=Path('results/pathology_analysis/run1'))
    ap.add_argument('--data-root', type=Path, default=Path('data'))
    ap.add_argument('--ckpt', type=Path,
                    default=Path('checkpoints/'
                                 'h1_8_v2/neurotwin_pretrain/base_best.pt'))
    ap.add_argument('--subset', type=Path,
                    default=Path('data/splits/site_matched_subset.json'))
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--fdr-q', type=float, default=FDR_Q)
    ap.add_argument('--skip-descriptors', action='store_true',
                    help='跳过六描述量/FC 基线（只跑 z 检验；调试用）')
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # ---- z 装配（模型信号） ----
    z_maps: Dict[str, np.ndarray] = {}
    subj_lists: Dict[str, List[str]] = {}
    for group, splits in (('hc', ('train', 'val', 'test')),
                          ('mdd', ('train', 'val', 'test'))):
        for sp in splits:
            try:
                sids, z = load_z(ns.run_dir, group, sp)
            except FileNotFoundError:
                continue
            z_maps.update({sid: z[i] for i, sid in enumerate(sids)})
            subj_lists[f'{group}_{sp}'] = sids
    pairs = load_matched_pairs(ns.subset)
    log.info('[B7] z 被试 %d | 匹配对 %d | 站点 %d', len(z_maps), len(pairs),
             len({p['site'] for p in pairs}))

    labels = get_aal116_labels()

    # ---- 1) 逐 ROI 配对检验（模型 z） ----
    diff, p_val, q, lo, hi = roi_test_paired(
        z_maps, pairs, n_perm=ns.n_perm, n_boot=ns.n_boot, seed=ns.seed)
    n_sig = int((q < ns.fdr_q).sum())
    log.info('[B7] 模型 z：显著 ROI %d 个 @q<%.2f | max|Δz|=%.4f',
             n_sig, ns.fdr_q, float(np.abs(diff).max()))
    rows = [{'roi_index': j + 1, 'label': labels[j],
             'delta_z': float(diff[j]), 'p': float(p_val[j]),
             'q': float(q[j]), 'ci_lo': float(lo[j]), 'ci_hi': float(hi[j]),
             'significant': bool(q[j] < ns.fdr_q)} for j in range(len(diff))]
    with (ns.run_dir / 'roi_fdr_table.csv').open('w', newline='',
                                                 encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- 网络级 + 全脑 ----
    netcols = network_columns(diff.shape[0])
    net_tests = []
    for net, cols in netcols.items():
        z_net = {sid: v[cols].mean() for sid, v in z_maps.items()}
        d_n, p_n, q_n, lo_n, hi_n = roi_test_paired(
            z_net, pairs, n_perm=ns.n_perm, n_boot=min(ns.n_boot, 5000),
            seed=ns.seed)
        net_tests.append({'level': 'network', 'name': net, 'n_rois': len(cols),
                          'delta': float(d_n[0]), 'p': float(p_n[0]),
                          'q': float(q_n[0]), 'ci_lo': float(lo_n[0]),
                          'ci_hi': float(hi_n[0])})
    z_wb = {sid: v.mean() for sid, v in z_maps.items()}
    d_w, p_w, q_w, lo_w, hi_w = roi_test_paired(
        z_wb, pairs, n_perm=ns.n_perm, n_boot=min(ns.n_boot, 5000), seed=ns.seed)
    net_tests.append({'level': 'whole_brain', 'name': 'whole_brain', 'n_rois': 116,
                      'delta': float(d_w[0]), 'p': float(p_w[0]), 'q': float(q_w[0]),
                      'ci_lo': float(lo_w[0]), 'ci_hi': float(hi_w[0])})

    # ---- 2) 负控：HC val vs HC test（站内置换组标签） ----
    s_val, z_val = load_z(ns.run_dir, 'hc', 'val')
    s_test, z_test = load_z(ns.run_dir, 'hc', 'test')
    p_neg = site_blocked_unpaired_permutation(
        z_val, z_test, [parse_site(s) for s in s_val],
        [parse_site(s) for s in s_test], n_perm=min(ns.n_perm, 2000), seed=ns.seed)
    q_neg = bh_fdr(p_neg)
    neg_sig = int((q_neg < ns.fdr_q).sum())

    # ---- 3) 离散度补充：组间方差比（mean|z|） ----
    mdd_ids = [p['mdd'] for p in pairs]
    hc_ids = [p['hc'] for p in pairs]
    absm = np.asarray([[np.abs(z_maps[s]).mean()] for s in mdd_ids])
    absh = np.asarray([[np.abs(z_maps[s]).mean()] for s in hc_ids])
    var_ratio = float(absm.var() / (absh.var() + EPS))
    p_disp = site_blocked_paired_permutation(
        (absm - absh), [p['site'] for p in pairs],
        n_perm=min(ns.n_perm, 5000), seed=ns.seed)

    report: Dict = {
        'analysis': 'B7 匹配子集组间检验（v2 §6 预登记框架）',
        'n_pairs': len(pairs),
        'n_sites': len({p['site'] for p in pairs}),
        'perm_settings': {'n_perm': ns.n_perm, 'n_boot': ns.n_boot,
                          'seed': ns.seed, 'fdr_q': ns.fdr_q},
        'model_z': {
            'n_significant_roi': n_sig,
            'max_abs_delta_z': float(np.abs(diff).max()),
            'median_abs_delta_z': float(np.median(np.abs(diff))),
            'top5': sorted(rows, key=lambda r: -abs(r['delta_z']))[:5],
        },
        'network_and_whole_brain': net_tests,
        'negative_control': {
            'comparison': 'HC val vs HC test（站内置换组标签）',
            'n_significant_roi': neg_sig,
            'max_abs_delta': float(np.abs(
                z_val.mean(0) - z_test.mean(0)).max()),
            'verdict': ('负控出现显著 ROI → 先怀疑流程/划分混杂，不解读为效应'
                        if neg_sig > 0 else '负控无显著 ROI（通过）'),
        },
        'dispersion_test': {
            'statistic': 'var(mean|z|_MDD) / var(mean|z|_HC)（匹配子集）',
            'var_ratio': var_ratio,
            'p_paired_siteblocked': float(p_disp[0]),
        },
        'interpretation': '判读规则第 1 条：组间差异只说明队列可分，'
                          '不等于病理机制定位；显著性以站点块置换 + FDR 为准。',
    }

    # ---- 4) 并排基线 ----
    if not ns.skip_descriptors:
        hc_need = sorted({p['hc'] for p in pairs} | set(s_val) | set(s_test))
        mdd_need = sorted({p['mdd'] for p in pairs})
        d_ref, dmeta = collect_descriptors(ns.data_root, ns.ckpt, hc_need, mdd_need)
        diff_d, p_d, q_d, lo_d, hi_d = roi_test_paired(
            d_ref, pairs, n_perm=ns.n_perm, n_boot=ns.n_boot, seed=ns.seed)
        n_sig_d = int((q_d < ns.fdr_q).sum())
        rows_d = [{'roi_index': j + 1, 'label': labels[j],
                   'delta_dref': float(diff_d[j]), 'p': float(p_d[j]),
                   'q': float(q_d[j]), 'ci_lo': float(lo_d[j]),
                   'ci_hi': float(hi_d[j]),
                   'significant': bool(q_d[j] < ns.fdr_q)}
                  for j in range(len(diff_d))]
        with (ns.run_dir / 'roi_fdr_table_descriptor.csv').open(
                'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_d[0]))
            w.writeheader()
            w.writerows(rows_d)
        # 负控同流程（描述量）
        zv = np.stack([d_ref[s] for s in s_val])
        zt = np.stack([d_ref[s] for s in s_test])
        p_nd = site_blocked_unpaired_permutation(
            zv, zt, [parse_site(s) for s in s_val],
            [parse_site(s) for s in s_test], n_perm=min(ns.n_perm, 2000),
            seed=ns.seed)
        neg_sig_d = int((bh_fdr(p_nd) < ns.fdr_q).sum())

        # 静态 FC + 岭回归（站内 5 折 CV OOF AUC）
        import warnings
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import roc_auc_score
        all_ids = hc_ids + mdd_ids
        X = np.concatenate([static_fc_features(ns.data_root, ns.ckpt, 'hc', hc_ids),
                            static_fc_features(ns.data_root, ns.ckpt, 'mdd',
                                               mdd_ids)], axis=0)
        ybin = np.concatenate([np.zeros(len(hc_ids)), np.ones(len(mdd_ids))])
        site_of = np.asarray([parse_site(s) for s in all_ids])
        fold = np.zeros(len(all_ids), dtype=int)
        for site in np.unique(site_of):
            idx = np.where(site_of == site)[0]
            for k, i in enumerate(idx):
                fold[i] = k % 5                      # 站内轮转分配折
        oof = np.zeros(len(all_ids))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for k in range(5):
                tr, te = fold != k, fold == k
                clf = RidgeClassifier(alpha=1.0)
                clf.fit(X[tr], ybin[tr])
                oof[te] = clf.decision_function(X[te])
        auc = float(roc_auc_score(ybin, oof))
        report['descriptor_baseline'] = {
            'n_significant_roi': n_sig_d,
            'max_abs_delta': float(np.abs(diff_d).max()),
            'median_abs_delta': float(np.median(np.abs(diff_d))),
            'fit_meta': {k: v for k, v in dmeta.items() if k != 'demo_cache'},
            'negative_control_n_sig': neg_sig_d,
        }
        report['static_fc_ridge'] = {
            'features': '最早 64 TR 的 FC 上三角（6670 边）',
            'model': 'RidgeClassifier(alpha=1)，站内轮转 5 折 CV，OOF AUC',
            'auc': auc, 'n': int(len(all_ids)),
        }
        # T2 判定
        model_increment = bool(
            n_sig > n_sig_d
            and float(np.abs(diff).max()) > float(np.abs(diff_d).max()))
        report['T2_judgment'] = {
            'rule': '模型显著 ROI 数与 max|效应| 均大于描述量基线 → 有增量',
            'model_n_sig': n_sig, 'descriptor_n_sig': n_sig_d,
            'model_max_abs': float(np.abs(diff).max()),
            'descriptor_max_abs': float(np.abs(diff_d).max()),
            'model_has_increment': model_increment,
            'action': ('维持模型偏离信号为主线，MINDy 式参数组间分析不启动'
                       if model_increment else
                       '触发 T2：启动指定扩展动作——MINDy 式参数组间分析'
                       '（冻结权重参数/个体 profile 的组间比较，不新增训练）'),
        }

    # ---- 5) 临床关联（网络级偏相关；HAMA 完整观测集） ----
    try:
        from utils.dataloader import load_condition_table
        table = load_condition_table(str(ns.data_root), MDD_MERGED_FILE, 'mdd8')
        names = [s.name for s in table.specs]
        site_names = sorted({p['site'] for p in pairs})
        groups: Dict[Tuple[str, str], Dict[str, list]] = {}
        for p in pairs:
            sid = p['mdd']
            row = table.row_of_id.get(sid, -1)
            if row < 0:
                continue
            demo = p['mdd_demo']
            if demo['Age'] is None or demo['Sex'] is None:
                continue
            base_ctrl = [float(demo['Age']), float(demo['Sex'])] + \
                [1.0 if p['site'] == s else 0.0 for s in site_names]
            for score_name in ('HAMD', 'HAMA'):
                j = names.index(score_name)
                score = (float(table.values[row, j]) if int(table.mask[row, j])
                         else None)
                if score is None:
                    continue
                for net, cols in netcols.items():
                    if net == 'Other':
                        continue
                    g = groups.setdefault((net, score_name),
                                          {'y': [], 'x': [], 'C': []})
                    g['y'].append(float(z_maps[sid][cols].mean()))
                    g['x'].append(score)
                    g['C'].append(base_ctrl)
        clin_rows = []
        for (net, score_name), g in sorted(groups.items()):
            res = partial_corr(np.asarray(g['y']), np.asarray(g['x']),
                               np.asarray(g['C'], dtype=np.float64))
            clin_rows.append({'network': net, 'score': score_name,
                              'partial_r': res['r'], 'p': res['p'],
                              'n': res['n']})
        if clin_rows:
            with (ns.run_dir / 'clinical_partial_corr.csv').open(
                    'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(clin_rows[0]))
                w.writeheader()
                w.writerows(clin_rows)
        hama_n = next((r['n'] for r in clin_rows if r['score'] == 'HAMA'), None)
        report['clinical_association'] = {
            'method': '网络级 z 与 HAMD/HAMA 偏相关（控制 Age/Sex/站点哑变量；'
                      '残差化 + Pearson；t 近似 p）',
            'rows': clin_rows,
            'hama_complete_case_n': hama_n,
            'note': 'HAMA 只报告完整观测集，不插补；p 未做多重校正，'
                    '作为描述性关联报告。',
        }
    except Exception as exc:                    # noqa: BLE001 —— 临床段失败不阻断主检验
        log.warning('[B7] 临床关联段失败：%s', exc)
        report['clinical_association'] = {'error': str(exc)}

    (ns.run_dir / 'group_tests.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('[B7] 报告已写入 %s/group_tests.json', ns.run_dir)
    print(json.dumps({k: report[k] for k in (
        'model_z', 'negative_control', 'dispersion_test') if k in report},
        ensure_ascii=False, indent=2)[:3000])


if __name__ == '__main__':
    main()
