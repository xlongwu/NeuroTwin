# coding=utf-8
"""F2 长程精度衰减：分段 MAE 小提琴 + 历史均值基线。

借鉴图库 oa-geo-hess-30-629-2026-f10 卡片（淡彩小提琴 + 同色数值标注）；
虚线为历史均值基线（实验结论与证据汇总.md：test H17–32 历史均值 0.922955）。
数据：results/paper_figures/complete_20261026/sources/h32/metrics_{val,test}.json 与 per_subject_{val,test}.csv。
运行：python -m analysis.paper_figs.fig_f2_horizon
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper_figs.common import (HC, HC_PALE, INK, MDD, MDD_PALE,
                                        apply_style, final_metrics,
                                        per_subject_segment, savefig)

SEGS = ['H1_8', 'H9_16', 'H17_32', 'H1_32']
SEG_LABELS = ['H1–8', 'H9–16', 'H17–32', 'H1–32']
BASELINE = 0.922955          # 历史均值基线（H17–32，test 口径）


def main():
    apply_style()
    mv, mt = final_metrics()

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.24)
    xs = np.arange(len(SEGS))
    for k, (split, col, pale, dx) in enumerate(
            [('val', HC, HC_PALE, -0.19), ('test', MDD, MDD_PALE, 0.19)]):
        df = per_subject_segment(split)
        data = [df.loc[df['segment'] == s, 'mae'].to_numpy() for s in SEGS]
        agg = mv if split == 'val' else mt
        means = [agg['ode_mae'][s] for s in SEGS]
        for i, (d, mn) in enumerate(zip(data, means)):
            vp = ax.violinplot([d], positions=[xs[i] + dx], widths=0.32,
                               showextrema=False, showmedians=False)
            for b in vp['bodies']:
                b.set_facecolor(pale)
                b.set_alpha(0.85)
                b.set_edgecolor(col)
                b.set_linewidth(0.9)
            ax.scatter([xs[i] + dx], [mn], s=26, color=col, edgecolor=INK,
                       lw=0.7, zorder=5)
            ax.text(xs[i] + dx + 0.03, mn + 0.006, f'{mn:.3f}', fontsize=7.8,
                    color=col, fontweight='bold', ha='left')
    ax.axhline(BASELINE, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=2)
    ax.text(len(SEGS) - 0.52, BASELINE - 0.012,
            'historical-mean baseline (H17–32): 0.923', fontsize=8,
            ha='right', va='top', color=INK)
    ax.set_xticks(xs, SEG_LABELS)
    ax.set_xlim(-0.55, len(SEGS) - 0.35)
    ax.set_ylabel('ODE MAE (H32 anchors, subject-equal-weight)')
    ax.set_xlabel('Prediction horizon segment')
    ax.set_title('Long-horizon accuracy: degradation toward the '
                 'historical-mean level', pad=8)
    ax.text(0.012, 0.955,
            'far-horizon gain vs baseline ≈ 2.9%;  temporal PCC (H17–32) ≈ 0.18',
            transform=ax.transAxes, fontsize=8, va='top', color='#555555')
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=HC,
                                  label='val (225)'),
                       plt.Line2D([], [], marker='o', ls='', color=MDD,
                                  label='test (225)')],
              loc='lower right', fontsize=8.5, handletextpad=0.2)
    savefig(fig, 'fig_f2_horizon')


if __name__ == '__main__':
    main()
