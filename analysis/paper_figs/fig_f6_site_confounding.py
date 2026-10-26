# coding=utf-8
"""F6 站点混杂方法学图（本文核心贡献）：未匹配 vs 匹配 vs 负控。

上排三联脑图借鉴 oa-geo-cp-20-1919-2024-f07 卡片（多面板差值场共享同一
发散色阶与阈值）；下排为逐 ROI 效应量对照（分布 + Top-10 配对棒棒糖）。
数据：run1 roi_fdr_table.csv（匹配）与 results/paper_stats/
unmatched_negative.csv（T1 重算：未匹配 + 负控，种子与 run1 一致）。
运行：python -m analysis.paper_figs.fig_f6_site_confounding
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper_figs.common import (DIV_CMAP, INK, MDD, NEGC, RUN1,
                                        STATS_DIR, apply_style, roi_fdr,
                                        roi_labels, savefig)


def aal_stat_image(values):
    """116 个 ROI 数值 → nilearn AAL(SPM12) 体素统计图。"""
    from nilearn import datasets
    from nilearn.image import new_img_like
    atlas = datasets.fetch_atlas_aal(version='SPM12')
    import nibabel as nib
    img = nib.load(atlas.maps)
    data = np.zeros(img.shape, dtype=np.float32)
    idx = np.asarray([int(v) for v in atlas.indices
                      if int(v) != 0 and 'background' not in str(v).lower()])
    assert len(idx) == len(values), (len(idx), len(values))
    lab2val = {lab: values[i] for i, lab in enumerate(idx)}
    atlas_data = np.asarray(img.dataobj)
    for lab, val in lab2val.items():
        data[atlas_data == lab] = val
    return new_img_like(img, data)


def main():
    apply_style()
    m = roi_fdr()                      # 匹配（run1）
    un = pd.read_csv(STATS_DIR / 'unmatched_negative.csv')
    labels = roi_labels()

    d_un = un['delta_unmatched'].to_numpy()
    d_mt = m['delta_z'].to_numpy()
    d_ng = un['delta_negative'].to_numpy()
    vmax = float(np.percentile(np.abs(np.concatenate([d_un, d_mt, d_ng])), 99.5))

    fig = plt.figure(figsize=(11.0, 5.2))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.9, bottom=0.09,
                        wspace=0.12, hspace=0.28)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82])

    from nilearn.plotting import plot_glass_brain
    specs = [
        (d_un, f'(a) UNMATCHED  MDD−HC\nwhole-brain Δz = {d_un.mean():+.3f}',
         MDD),
        (d_mt, f'(b) SITE-MATCHED (B7)\n0/116 FDR-significant, max|Δz| = '
         f'{np.abs(d_mt).max():.3f}', '#4a7a6f'),
        (d_ng, f'(c) NEGATIVE CONTROL  HC val−test\nmax|Δz| = '
         f'{np.abs(d_ng).max():.3f} (site mix only)', NEGC),
    ]
    for k, (vals, title, ring) in enumerate(specs):
        ax = fig.add_subplot(gs[0, k])
        display = plot_glass_brain(
            aal_stat_image(vals), display_mode='lzry', axes=ax,
            cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, plot_abs=False,
            colorbar=(k == 2), cbar_tick_format='%+.2f')
        ax.set_title(title, fontsize=8.8, pad=2)

    # (d) 逐 ROI |Δ| 分布
    ax = fig.add_subplot(gs[1, 0])
    parts = [np.abs(d_un), np.abs(d_mt), np.abs(d_ng)]
    names = ['Unmatched\nMDD−HC', 'Matched\nMDD−HC', 'Negative\nHC val−test']
    cols = [MDD, '#4a7a6f', NEGC]
    vp = ax.violinplot(parts, showextrema=False, widths=0.8)
    for b, c in zip(vp['bodies'], cols):
        b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor(c)
    for i, (p, c) in enumerate(zip(parts, cols)):
        rng = np.random.RandomState(3)
        ax.scatter(rng.uniform(i - 0.13, i + 0.13, len(p)), p, s=4, color=c,
                   alpha=0.5, lw=0, zorder=3)
        ax.scatter([i], [np.median(p)], s=46, facecolor='white', edgecolor=INK,
                   lw=1.1, zorder=5)
    ax.set_xticks([1, 2, 3], names, fontsize=8)
    ax.set_ylabel('per-ROI |Δz|')
    ax.set_title('(d) Effect-size distributions', fontsize=9)

    # (e) Top-10 未匹配 ROI：未匹配 vs 匹配配对
    ax = fig.add_subplot(gs[1, 1:])
    top = np.argsort(-np.abs(d_un))[:10]
    order = top[np.argsort(np.abs(d_un)[top])]
    yy = np.arange(len(order))
    ax.hlines(yy, 0, d_un[order], color=MDD, lw=2.4, alpha=0.85, zorder=3)
    ax.scatter(d_un[order], yy, s=34, color=MDD, edgecolor=INK, lw=0.5,
               zorder=4, label='unmatched Δz')
    ax.scatter(d_mt[order], yy, s=30, facecolor='white', edgecolor='#4a7a6f',
               lw=1.5, zorder=4, label='same ROI, site-matched Δz')
    rng = np.random.RandomState(4)
    ax.scatter(d_ng[order] * rng.choice((-1, 1), len(order)), yy, s=12,
               color=NEGC, alpha=0.6, zorder=3, label='negative control |Δz| '
               '(sign randomised)')
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(yy, [labels[i].replace('_', ' ') for i in order], fontsize=7.5)
    ax.set_xlabel('Δz (MDD−HC, or HC val−test)')
    ax.set_title('(e) Top-10 unmatched ROIs: the “classic-region” effects '
                 'vanish after matching', fontsize=9)
    ax.legend(loc='lower right', fontsize=7.8, handletextpad=0.3)

    fig.suptitle('Site composition, not pathology: unmatched group effects '
                 'are reproduced by site mix alone', fontweight='bold',
                 fontsize=11, y=0.985)
    savefig(fig, 'fig_f6_site_confounding')


if __name__ == '__main__':
    main()
