# coding=utf-8
"""T1 统计重算（论文图 F6/F7/F9 的数据源；只读，不触碰 run1）。

- D2 站点混杂方法学演示：全队列未匹配 MDD−HC 逐 ROI 检验 + 负控
  （HC val vs test）逐 ROI 检验，复用 run1 的统计函数与种子；
- D3 队列可分性：静态 FC + 岭回归（与 run1 完全同协议），额外落盘
  OOF 分数用于 ROC 与 bootstrap CI；
- D5 临床关联散点数据：匹配子集网络级 z 与 HAMD/HAMA 的残差化坐标。

产物目录 results/paper_stats/。本脚本不产生新的病理推断声明：所有统计量
均为 run1 已有结论的重组织与方法学演示。

运行：python -m analysis.paper_figs.t1_recompute [--parts d2,d3,d5]
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from analysis.paper_figs.common import ROOT, RUN1, STATS_DIR, matched_pairs
from analysis.pathology_tests import (MDD_MERGED_FILE, bh_fdr,
                                      network_columns,
                                      site_blocked_unpaired_permutation)
from analysis.pathology_tests import load_z, partial_corr
from utils.site_subset import parse_site

CKPT = ROOT / 'checkpoints' / 'neurotwin_pretrain' / 'base_best.pt'
SEED = 2024


def load_all_z():
    z_maps, subj_split = {}, {}
    for group in ('hc', 'mdd'):
        for sp in ('train', 'val', 'test'):
            sids, z = load_z(RUN1, group, sp)
            z_maps.update({sid: z[i] for i, sid in enumerate(sids)})
            subj_split[f'{group}_{sp}'] = sids
    return z_maps, subj_split


def part_d2(z_maps, subj_split):
    from analysis.metrics import get_aal116_labels
    labels = get_aal116_labels()
    hc_all = sum((subj_split[f'hc_{s}'] for s in ('train', 'val', 'test')), [])
    mdd_all = sum((subj_split[f'mdd_{s}'] for s in ('train', 'val', 'test')), [])
    H = np.stack([z_maps[s] for s in hc_all])
    M = np.stack([z_maps[s] for s in mdd_all])
    p_un = site_blocked_unpaired_permutation(
        M, H, [parse_site(s) for s in mdd_all], [parse_site(s) for s in hc_all],
        n_perm=10000, seed=SEED)
    q_un = bh_fdr(p_un)
    d_un = M.mean(0) - H.mean(0)

    s_val, z_val = load_z(RUN1, 'hc', 'val')
    s_test, z_test = load_z(RUN1, 'hc', 'test')
    p_neg = site_blocked_unpaired_permutation(
        z_val, z_test, [parse_site(s) for s in s_val],
        [parse_site(s) for s in s_test], n_perm=2000, seed=SEED)
    q_neg = bh_fdr(p_neg)
    d_neg = z_val.mean(0) - z_test.mean(0)

    import csv
    with (STATS_DIR / 'unmatched_negative.csv').open('w', newline='',
                                                      encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['roi_index', 'label', 'delta_unmatched', 'p_unmatched',
                    'q_unmatched', 'delta_negative', 'p_negative', 'q_negative'])
        for j in range(116):
            w.writerow([j + 1, labels[j], f'{d_un[j]:.6f}', f'{p_un[j]:.6g}',
                        f'{q_un[j]:.6g}', f'{d_neg[j]:.6f}', f'{p_neg[j]:.6g}',
                        f'{q_neg[j]:.6g}'])
    top_un = sorted(
        ({'label': labels[j], 'delta': float(d_un[j]), 'q': float(q_un[j])}
         for j in range(116)), key=lambda r: -abs(r['delta']))[:10]
    summary = {
        'analysis': 'D2 站点混杂方法学演示（未匹配 vs 匹配 vs 负控）',
        'unmatched': {
            'n_hc': len(hc_all), 'n_mdd': len(mdd_all),
            'n_significant_roi': int((q_un < 0.05).sum()),
            'whole_brain_delta': float(d_un.mean()),
            'max_abs_delta': float(np.abs(d_un).max()),
            'top10_by_abs_delta': top_un,
        },
        'negative_control': {
            'comparison': 'HC val vs HC test（站内置换组标签，n_perm=2000）',
            'n_significant_roi': int((q_neg < 0.05).sum()),
            'max_abs_delta': float(np.abs(d_neg).max()),
            'median_abs_delta': float(np.median(np.abs(d_neg))),
        },
        'matched_reference': 'run1 roi_fdr_table.csv（0/116 过 FDR，'
                             'max|Δz|=0.090）',
    }
    (STATS_DIR / 'd2_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary['unmatched'], ensure_ascii=False, indent=1)[:900])
    print('negative control:', summary['negative_control'])


def part_d3(z_maps, subj_split):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import roc_auc_score

    from analysis.pathology_tests import static_fc_features
    pairs = matched_pairs()
    hc_ids = [p['hc'] for p in pairs]
    mdd_ids = [p['mdd'] for p in pairs]
    X = np.concatenate([static_fc_features(ROOT / 'data', CKPT, 'hc', hc_ids),
                        static_fc_features(ROOT / 'data', CKPT, 'mdd',
                                           mdd_ids)], axis=0)
    y = np.concatenate([np.zeros(len(hc_ids)), np.ones(len(mdd_ids))])
    all_ids = hc_ids + mdd_ids
    site_of = np.asarray([parse_site(s) for s in all_ids])
    fold = np.zeros(len(all_ids), dtype=int)
    for site in np.unique(site_of):
        idx = np.where(site_of == site)[0]
        for k, i in enumerate(idx):
            fold[i] = k % 5
    oof = np.zeros(len(all_ids))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k in range(5):
            tr, te = fold != k, fold == k
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.decision_function(X[te])
    auc = float(roc_auc_score(y, oof))
    np.savez(STATS_DIR / 'fc_ridge_oof.npz', oof=oof, y=y,
             site=site_of, subject=np.asarray(all_ids))
    (STATS_DIR / 'd3_summary.json').write_text(json.dumps(
        {'analysis': 'D3 静态 FC + 岭回归（run1 同协议，落盘 OOF 分数）',
         'model': 'RidgeClassifier(alpha=1)，站内轮转 5 折 CV，OOF AUC',
         'n': int(len(all_ids)), 'auc': auc}, indent=1))
    print(f'D3 OOF AUC = {auc:.4f} (n={len(all_ids)})')


def part_d5(z_maps, subj_split):
    from utils.dataloader import load_condition_table
    pairs = matched_pairs()
    netcols = network_columns(116)
    table = load_condition_table(str(ROOT / 'data'), MDD_MERGED_FILE, 'mdd8')
    names = [s.name for s in table.specs]
    site_names = sorted({p['site'] for p in pairs})
    rows = []
    for p in pairs:
        sid = p['mdd']
        row = table.row_of_id.get(sid, -1)
        if row < 0:
            continue
        demo = p['mdd_demo']
        if demo['Age'] is None or demo['Sex'] is None:
            continue
        ctrl = [float(demo['Age']), float(demo['Sex'])] + \
            [1.0 if p['site'] == s else 0.0 for s in site_names]
        for score_name in ('HAMD', 'HAMA'):
            j = names.index(score_name)
            if not int(table.mask[row, j]):
                continue
            score = float(table.values[row, j])
            for net, cols in netcols.items():
                if net == 'Other':
                    continue
                rows.append({'subject': sid, 'network': net, 'score': score_name,
                             'net_z': float(z_maps[sid][cols].mean()),
                             'ctrl': ctrl})
    # 残差化（OLS on ctrl），落盘散点坐标 + 复算偏相关核对
    import pandas as pd
    df = pd.DataFrame([{'subject': r['subject'], 'network': r['network'],
                        'score': r['score'], 'net_z': r['net_z']} for r in rows])
    resid = []
    for (net, sc), g in df.groupby(['network', 'score']):
        sub = [r for r in rows if r['network'] == net and r['score'] == sc]
        C = np.asarray([r['ctrl'] for r in sub], dtype=float)
        Cd = np.column_stack([np.ones(len(C)), C])
        yv = np.asarray([r['net_z'] for r in sub], dtype=float)
        xv = np.asarray([r['score'] for r in sub], dtype=float)
        by = np.linalg.lstsq(Cd, yv, rcond=None)[0]
        bx = np.linalg.lstsq(Cd, xv, rcond=None)[0]
        ry, rx = yv - Cd @ by, xv - Cd @ bx
        r_stat, p_val = partial_corr(yv, xv, C)
        for i, r in enumerate(sub):
            resid.append({'subject': r['subject'], 'network': net,
                          'score': sc, 'resid_net_z': ry[i],
                          'resid_score': rx[i],
                          'partial_r': r_stat, 'p': p_val})
    out = pd.DataFrame(resid)
    out.to_csv(STATS_DIR / 'clinical_residuals.csv', index=False)
    n_hama_ln = int(((out['network'] == 'LN') & (out['score'] == 'HAMA')).sum())
    (STATS_DIR / 'd5_summary.json').write_text(json.dumps(
        {'analysis': 'D5 临床关联残差坐标（run1 同被试集与控制变量）',
         'n_hama_ln': n_hama_ln,
         'pairs_by_net_score': {f'{n}_{s}': int(((out.network == n) &
                                                 (out.score == s)).sum())
                                for n in out.network.unique()
                                for s in out.score.unique()}}, indent=1))
    print(f'D5 rows={len(out)}, LN×HAMA n={n_hama_ln}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parts', default='d2,d3,d5')
    ns = ap.parse_args()
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    z_maps, subj_split = load_all_z()
    if 'd2' in ns.parts:
        part_d2(z_maps, subj_split)
    if 'd3' in ns.parts:
        part_d3(z_maps, subj_split)
    if 'd5' in ns.parts:
        part_d5(z_maps, subj_split)


if __name__ == '__main__':
    main()
