# coding=utf-8
"""F5 病理主结果：Manhattan + Top-20 ROI 森林 + 网络/全脑森林。

(a) 借鉴 oa-bio-plos-pbio-3000391-f02 卡片（共享位置轴、分块着色条带）；
(b)(c) 借鉴 oa-tech-pone-0258309-f05 点区间图（点估计 + 区间 + 显著性深浅）。
数据：run1/roi_fdr_table.csv、group_tests.json（B7 主结果，0/116 过 FDR）。
运行：python -m analysis.paper_figs.fig_f5_pathology_main
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper_figs.common import (DIV_CMAP, INK, NETWORK_COLORS,
                                        NETWORK_LABELS, NETWORK_ORDER, ROOT,
                                        RUN1, apply_style, roi_fdr, savefig)

DISPLAY_ORDER = ['VN', 'SMN', 'DAN', 'SN', 'FN', 'DMN', 'LN']
DISPLAY_ALIAS = {'SN': 'VAN', 'FN': 'FPN'}          # AAL 表缩写 → 常用名


def load_display_partition():
    """AAL116 展示分组（仅用于排布与着色；统计口径仍为 run1）。"""
    df = pd.read_excel(ROOT / 'data' / 'AAL116.xlsx')
    net = df['对应网络'].astype(str).tolist()
    order = {n: i for i, n in enumerate(DISPLAY_ORDER)}
    return net, order


def color_of(net):
    key = DISPLAY_ALIAS.get(net, net)
    return NETWORK_COLORS.get(key, NETWORK_COLORS['Other'])


def main():
    apply_style()
    rf = roi_fdr()
    gt = json.loads((RUN1 / 'group_tests.json').read_text())
    net, order = load_display_partition()
    rf['net'] = [net[i - 1] for i in rf['roi_index']]
    rf['ord'] = [order[n] for n in rf['net']]
    rf = rf.sort_values(['ord', 'roi_index']).reset_index(drop=True)
    rf['x'] = np.arange(1, len(rf) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4),
                             gridspec_kw={'width_ratios': [1.5, 1.1, 0.9]})
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.3,
                        wspace=0.34)

    # (a) Manhattan
    ax = axes[0]
    p = rf['p'].to_numpy()
    y = -np.log10(np.clip(p, 1e-5, 1))
    ax.scatter(rf['x'], y, s=16, c=[color_of(n) for n in rf['net']],
               edgecolor=INK, lw=0.35, zorder=3)
    ps = np.sort(p)
    cands = [ps[k] for k in range(len(ps)) if ps[k] <= 0.05 * (k + 1) / len(ps)]
    thr = max(cands) if cands else 0.05 / len(ps)   # 无通过时退化为 Bonferroni
    ax.axhline(-np.log10(thr), color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.985, -np.log10(thr) + 0.03,
            'BH-FDR q=0.05' + ('' if cands else ' (=Bonferroni, none pass)'),
            transform=ax.get_yaxis_transform(), fontsize=7.5, ha='right',
            color=INK)
    bonf = 0.05 / len(p)
    ax.axhline(-np.log10(bonf), color='#888888', lw=0.8, ls=(0, (1, 3)),
               zorder=2)
    ax.text(0.985, -np.log10(bonf) - 0.05, 'Bonferroni', fontsize=7.5,
            ha='right', color='#888888', transform=ax.get_yaxis_transform())
    # 网络分块底色条
    for name, grp in rf.groupby('net', sort=False):
        ax.axvspan(grp['x'].min() - 0.5, grp['x'].max() + 0.5,
                   color=color_of(name), alpha=0.08, lw=0, zorder=1)
    ax.set_xlabel('ROIs grouped by network (AAL116 display order)')
    ax.set_ylabel('−log₁₀(p), site-blocked paired permutation')
    ax.set_title('(a) ROI-level group difference: 0/116 significant',
                 fontsize=9.5)

    # (b) Top-20 |Δz| 森林
    ax = axes[1]
    top = rf.reindex(rf['delta_z'].abs().sort_values(ascending=False).index)
    top = top.head(20).iloc[::-1]
    yy = np.arange(len(top))
    ax.hlines(yy, top['ci_lo'], top['ci_hi'],
              colors=[color_of(n) for n in top['net']], lw=1.8, zorder=3)
    ax.scatter(top['delta_z'], yy, s=22, c=[color_of(n) for n in top['net']],
               edgecolor=INK, lw=0.5, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(yy, [l.replace('_', ' ') for l in top['label']], fontsize=7)
    ax.set_xlabel('Δz (MDD − HC, site-matched pairs)')
    ax.set_title(f'(b) Top-20 |Δz| ROIs (max |Δz| = '
                 f'{rf["delta_z"].abs().max():.3f})', fontsize=9.5)

    # (c) 网络 / 全脑森林
    ax = axes[2]
    rows = [r for r in gt['network_and_whole_brain']]
    rows = sorted(rows, key=lambda r: r['name'])
    yy = np.arange(len(rows))
    for i, r in enumerate(rows):
        col = NETWORK_COLORS.get(r['name'], '#6B6B6B' if r['name'] != 'whole'
                                 else '#444444')
        sig = r['q'] < 0.05
        ax.hlines(i, r['ci_lo'], r['ci_hi'], color=col, lw=2.2,
                  alpha=1.0 if sig else 0.75, zorder=3)
        ax.scatter(r['delta'], i, s=30, facecolor=col if sig else 'white',
                   edgecolor=col, lw=1.4, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(yy, [f'{r["name"]} ({r["n_rois"]})' for r in rows],
                  fontsize=8)
    ax.set_xlabel('Δz (MDD − HC)')
    ax.set_title('(c) Network / whole-brain: all n.s.', fontsize=9.5)

    fig.suptitle('B7 main result: no localizable MDD–HC deviation after '
                 'site matching', fontweight='bold', fontsize=11, y=0.99)
    handles = [plt.Line2D([], [], marker='o', ls='', color=color_of(n),
                          label=NETWORK_LABELS.get(DISPLAY_ALIAS.get(n, n), n))
               for n in DISPLAY_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=7.8,
               handletextpad=0.1, columnspacing=1.1,
               bbox_to_anchor=(0.5, 0.0))
    savefig(fig, 'fig_f5_pathology_main')


if __name__ == '__main__':
    main()
