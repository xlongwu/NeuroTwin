# coding=utf-8
"""F8 个体异质性：匹配子集 mean|z| 雨云 + 个体 z 剖面热图 + 组内离散度。

(a)(c) 借鉴 raincloud-median-badges 卡片（点云 + 半小提琴 + 中位圆标）；
(b) 借鉴 oa-bio-plos-pbio-3000391-f06（共享位置轴的彩色条带 + 侧栏摘要）。
数据：run1/z_maps/*.npz 与 site_matched_subset.json（1504 对）。
运行：python -m analysis.paper_figs.fig_f8_subject_heterogeneity
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper_figs.common import (DIV_CMAP, HC, HC_PALE, INK, MDD,
                                        MDD_PALE, apply_style, half_violin,
                                        matched_pairs, roi_labels, savefig,
                                        z_matrix)
from analysis.paper_figs.fig_f5_pathology_main import (color_of,
                                                       load_display_partition)


def matched_stats():
    pairs = matched_pairs()
    hc_ids = [p['hc'] for p in pairs]
    mdd_ids = [p['mdd'] for p in pairs]
    zh = {}
    for split in ('train', 'val', 'test'):
        s, z = z_matrix('hc', split)
        zh.update(dict(zip(s.tolist(), z)))
    zm = {}
    for split in ('train', 'val', 'test'):
        s, z = z_matrix('mdd', split)
        zm.update(dict(zip(s.tolist(), z)))
    H = np.stack([zh[i] for i in hc_ids])
    M = np.stack([zm[i] for i in mdd_ids])
    return H, M


def main():
    apply_style()
    H, M = matched_stats()
    labels = roi_labels()
    net, order = load_display_partition()
    roi_ord = np.argsort([order[n] for n in net], kind='stable')
    labels_o = [labels[i] for i in roi_ord]

    hm, mm = np.abs(H).mean(axis=1), np.abs(M).mean(axis=1)
    hs, ms = H.std(axis=1), M.std(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6),
                             gridspec_kw={'width_ratios': [0.8, 1.7, 0.8]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.26, wspace=0.42)

    def raincloud(ax, data, pos, col, pale):
        rng = np.random.RandomState(1)
        xj = rng.uniform(pos - 0.3, pos - 0.06, size=len(data))
        ax.scatter(xj, data, s=5, color=col, alpha=0.3, lw=0, zorder=3)
        half_violin(ax, data, pos, col, pale=pale, side='right', width=0.4)
        med = float(np.median(data))
        ax.scatter([pos + 0.05], [med], s=80, facecolor='white',
                   edgecolor=INK, lw=1.1, zorder=6)
        ax.text(pos + 0.1, med, f'{med:.3f}', va='center', fontsize=8,
                color=INK, fontweight='bold')

    ax = axes[0]
    raincloud(ax, hm, 0, HC, HC_PALE)
    raincloud(ax, mm, 1, MDD, MDD_PALE)
    ax.set_xlim(-0.55, 1.75)
    ax.set_xticks([0, 1], ['HC', 'MDD'])
    ax.set_ylabel('subject mean |z| (116 ROIs)')
    ax.set_title('(a) Deviation magnitude', fontsize=9.5)
    ax.text(0.5, 0.03, 'var ratio 0.98, p=0.27 (site-blocked)',
            transform=ax.transAxes, ha='center', fontsize=7.5, color='#555555')

    ax = axes[1]
    rng = np.random.RandomState(2)
    sel = {'MDD extreme': mm[np.argsort(mm)[::-1][:2]],
           'MDD median': mm[np.argsort(mm)[len(mm) // 2:len(mm) // 2 + 1]],
           'HC median': hm[np.argsort(hm)[len(hm) // 2:len(hm) // 2 + 1]],
           'HC extreme': hm[np.argsort(hm)[::-1][:2]]}
    rows, row_names = [], []
    idx_m = np.argsort(mm)
    idx_h = np.argsort(hm)
    rows += [M[idx_m[-1]], M[idx_m[-2]]]; row_names += ['MDD max-1', 'MDD max-2']
    rows += [M[idx_m[len(M) // 2]]]; row_names += ['MDD median']
    rows += [H[idx_h[len(H) // 2]]]; row_names += ['HC median']
    rows += [H[idx_h[-1]], H[idx_h[-2]]]; row_names += ['HC max-1', 'HC max-2']
    mat = np.stack([r[roi_ord] for r in rows])
    vmax = np.percentile(np.abs(mat), 99)
    im = ax.imshow(mat, aspect='auto', cmap=DIV_CMAP, vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    for k in range(1, 7):
        ax.axhline(k - 0.5, color='white', lw=0.8)
    ax.set_yticks(range(len(rows)), row_names, fontsize=7.5)
    ax.set_xticks([])
    ax.set_xlabel('116 ROIs (network order)')
    ax.set_title('(b) Individual z profiles (matched subset)', fontsize=9.5)
    # 网络条带
    nets_o = [net[i] for i in roi_ord]
    bounds = []
    start = 0
    for i in range(1, 117):
        if i == 116 or nets_o[i] != nets_o[start]:
            bounds.append((start, i, nets_o[start])); start = i
    for s0, s1, n in bounds:
        ax.add_patch(plt.Rectangle((s0, -2.6), s1 - s0, 1.1,
                                   facecolor=color_of(n), edgecolor='none',
                                   clip_on=False))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('z (demographic-adjusted)', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax = axes[2]
    raincloud(ax, hs, 0, HC, HC_PALE)
    raincloud(ax, ms, 1, MDD, MDD_PALE)
    ax.set_xlim(-0.55, 1.75)
    ax.set_xticks([0, 1], ['HC', 'MDD'])
    ax.set_ylabel('subject std(z) across ROIs')
    ax.set_title('(c) Within-subject dispersion', fontsize=9.5)

    fig.suptitle('Individual heterogeneity: overlapping deviation '
                 'distributions, no group shift', fontweight='bold',
                 fontsize=11, y=0.99)
    savefig(fig, 'fig_f8_subject_heterogeneity')


if __name__ == '__main__':
    main()
