# coding=utf-8
"""F10 条件使用（B5 终裁）：Δpred 分布 + 零化条件配对 + FiLM 参数分布。

左板借鉴图库 oa-bio-plos-pcbi-1007396-f02 卡片（分布 + 参考线）；中板为
配对散点 + 对角线；右板借鉴 oa-tech-pone-0317619-f03（密度小多图）。
数据：results/pathology_analysis/condition_usage.csv（HC val 222 人）。
运行：python -m analysis.paper_figs.fig_f10_condition_usage
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper_figs.common import (HC, HC_PALE, INK, apply_style,
                                        condition_usage, savefig)

FIELDS = ['ode_film_gamma', 'ode_film_beta', 'encoder_adarms_gamma',
          'encoder_adarms_beta']
FIELD_LABELS = ['ODE FiLM\nγ', 'ODE FiLM\nβ', 'Encoder AdaRMS\nγ',
                'Encoder AdaRMS\nβ']


def main():
    apply_style()
    cu = condition_usage()
    rel = cu['delta_pred_relative'].to_numpy() * 100
    med = float(np.median(rel))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2),
                             gridspec_kw={'width_ratios': [1, 1, 1.35]})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2, wspace=0.3)

    # (a) Δpred / 信号尺度 分布
    ax = axes[0]
    ax.hist(rel, bins=36, color=HC_PALE, edgecolor=HC, lw=0.7, zorder=3)
    ax.axvline(med, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.text(med, ax.get_ylim()[1] * 0.96, f' median {med:.2f}%',
            fontsize=8.5, va='top', ha='left', color=INK)
    ax.set_xlabel('Δpred / signal scale (%)')
    ax.set_ylabel('Subjects')
    ax.set_title('(a) Condition-induced prediction change', fontsize=9.5)

    # (b) 配对散点：MAE(条件) vs MAE(零化条件)
    ax = axes[1]
    x = cu['mae_zero_cond'].to_numpy()
    y = cu['mae_cond'].to_numpy()
    lims = [min(x.min(), y.min()) * 0.97, max(x.max(), y.max()) * 1.03]
    ax.plot(lims, lims, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.scatter(x, y, s=9, color=HC, alpha=0.55, lw=0, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('MAE with conditions zeroed')
    ax.set_ylabel('MAE with conditions')
    d = y.mean() - x.mean()
    ax.text(0.05, 0.93, f'zeroing ΔMAE = {d:+.5f} (slightly better)',
            transform=ax.transAxes, fontsize=8.5, va='top')
    ax.set_title('(b) Paired MAE: conditions vs zeroed', fontsize=9.5)

    # (c) FiLM / AdaRMS 参数分布小多图
    ax = axes[2]
    for i, f in enumerate(FIELDS):
        v = cu[f].to_numpy()
        vp = ax.violinplot([v], positions=[i], widths=0.75, showextrema=False)
        for b in vp['bodies']:
            b.set_facecolor(HC_PALE)
            b.set_edgecolor(HC)
            b.set_alpha(0.9)
            b.set_linewidth(0.9)
        ax.scatter([i], [np.median(v)], s=24, color=HC, edgecolor=INK, lw=0.7,
                   zorder=5)
    ax.set_xticks(range(len(FIELDS)), FIELD_LABELS, fontsize=8)
    ax.set_ylabel('Parameter value')
    ax.set_title('(c) Conditioning parameter magnitudes', fontsize=9.5)

    fig.suptitle('Condition usage (B5): demographic FiLM conditioning is '
                 'near-zero', fontweight='bold', fontsize=11, y=0.99)
    savefig(fig, 'fig_f10_condition_usage')


if __name__ == '__main__':
    main()
