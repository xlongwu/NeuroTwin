# coding=utf-8
"""v2 结果图（B7 计划内交付，复用 visualize_roi_importance 的视觉风格）。

四联图 ``v2_results_summary.png``：

1. Top-20 ROI 配对 Δz 森林图（95% CI，FDR 显著标记；模型 z，匹配子集）；
2. 网络 / 全脑 Δz 森林图（模型 z）；
3. 效应量对照：负控（HC val−test）逐 ROI |Δ| 分布 vs MDD−HC 匹配 |Δ| 分布；
4. 被试级全脑均值 z 分布（HC vs MDD，匹配子集）。

输入全部为已落盘的 v2 产物；不重算检验。

用法::

    python -m analysis.visualize_v2_results \
      --run-dir results/pathology_analysis/run1
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_roi_table(run_dir: Path):
    import csv
    with (run_dir / 'roi_fdr_table.csv').open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_z(run_dir: Path, tag: str):
    with np.load(run_dir / 'z_maps' / f'z_matrix_{tag}.npz') as z:
        return [str(s) for s in z['subjects']], z['z'].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path,
                    default=Path('results/pathology_analysis/run1'))
    ap.add_argument('--subset', type=Path,
                    default=Path('data/splits/site_matched_subset.json'))
    ns = ap.parse_args()

    rows = load_roi_table(ns.run_dir)
    for r in rows:
        for k in ('delta_z', 'ci_lo', 'ci_hi', 'q'):
            r[k] = float(r[k])
    top = sorted(rows, key=lambda r: -abs(r['delta_z']))[:20][::-1]
    report = json.loads((ns.run_dir / 'group_tests.json').read_text(encoding='utf-8'))
    nets = [r for r in report['network_and_whole_brain'] if r['level'] == 'network']
    wb = next(r for r in report['network_and_whole_brain']
              if r['level'] == 'whole_brain')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Top-20 ROI forest
    ax = axes[0][0]
    y = np.arange(len(top))
    ax.errorbar([r['delta_z'] for r in top], y,
                xerr=[[r['delta_z'] - r['ci_lo'] for r in top],
                      [r['ci_hi'] - r['delta_z'] for r in top]],
                fmt='o', ms=4, lw=1, color='#1f77b4', ecolor='#7fb3d5',
                capsize=2)
    for i, r in enumerate(top):
        if r['q'] < 0.05:
            ax.plot(r['delta_z'], i, 'r*', ms=10)
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']} ({r['roi_index']})" for r in top],
                       fontsize=8)
    ax.set_xlabel('paired Δz (MDD − HC), site-matched, 95% CI')
    ax.set_title(f"Top-20 |Δz| ROIs — model z deviation "
                 f"({report['model_z']['n_significant_roi']} significant @FDR q<0.05; "
                 f"red* = significant)")

    # 2) Network forest
    ax = axes[0][1]
    net_rows = nets + [wb]
    y = np.arange(len(net_rows))
    ax.errorbar([r['delta'] for r in net_rows], y,
                xerr=[[r['delta'] - r['ci_lo'] for r in net_rows],
                      [r['ci_hi'] - r['delta'] for r in net_rows]],
                fmt='s', ms=5, lw=1.2, color='#2ca02c', ecolor='#a5d6a7',
                capsize=3)
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([r['name'] for r in net_rows])
    ax.set_xlabel('paired Δz (MDD − HC), 95% CI')
    ax.set_title('Network-level & whole-brain deviations (model z)')

    # 3) |Δ| distribution: negative control vs MDD effect
    ax = axes[1][0]
    s_val, z_val = load_z(ns.run_dir, 'hc_val')
    s_test, z_test = load_z(ns.run_dir, 'hc_test')
    neg = np.abs(z_val.mean(0) - z_test.mean(0))
    pairs = json.loads(ns.subset.read_text(encoding='utf-8'))['pairs']
    z_maps = {}
    for tag in ('hc_train', 'hc_val', 'hc_test', 'mdd_train', 'mdd_val',
                'mdd_test'):
        sids, z = load_z(ns.run_dir, tag)
        z_maps.update(dict(zip(sids, z)))
    mdd_eff = np.abs(np.mean([z_maps[p['mdd']] - z_maps[p['hc']]
                              for p in pairs], axis=0))
    ax.boxplot([neg, mdd_eff], tick_labels=['HC val−test\n(negative control)',
                                            'MDD−HC\n(site-matched)']),
    ax.set_ylabel('|Δz| per ROI')
    ax.set_title(f'Effect size vs noise floor\n'
                 f'median: control {np.median(neg):.3f} vs MDD '
                 f'{np.median(mdd_eff):.3f} (max {mdd_eff.max():.3f})')

    # 4) subject-level whole-brain mean z
    ax = axes[1][1]
    hc_m = np.asarray([z_maps[p['hc']].mean() for p in pairs])
    mdd_m = np.asarray([z_maps[p['mdd']].mean() for p in pairs])
    ax.hist(hc_m, bins=40, alpha=0.65, label=f'HC (n={len(hc_m)})', density=True)
    ax.hist(mdd_m, bins=40, alpha=0.65, label=f'MDD (n={len(mdd_m)})',
            density=True)
    ax.axvline(hc_m.mean(), color='C0', ls='--', lw=1)
    ax.axvline(mdd_m.mean(), color='C1', ls='--', lw=1)
    ax.set_xlabel('subject whole-brain mean z')
    ax.set_ylabel('density')
    ax.legend()
    ax.set_title(f'Subject-level whole-brain z (matched pairs)\n'
                 f'Δ = {mdd_m.mean() - hc_m.mean():+.4f} '
                 f'(p={wb["p"]:.3f}, site-blocked)')

    fig.suptitle('NeuroTwin v2 — pathology deviation results '
                 '(site-matched 1504 pairs; guardrail passed, no FDR-significant ROI)',
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = ns.run_dir / 'v2_results_summary.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
