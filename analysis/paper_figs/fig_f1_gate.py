# coding=utf-8
"""F1 滚动单步护栏：模型 vs persistence + 被试级 MAE 分布。

左板借鉴图库 raincloud-median-badges 卡片（点云 + 半小提琴 + 中位圆标）；
右板为组级条形（数值直接标注）。数据：gate_report.json 与
condition_usage.csv（HC val 222 人，lag-1，K=64）。
运行：python -m analysis.paper_figs.fig_f1_gate
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper_figs.common import (ACCENT_RED, HC, HC_PALE, INK,
                                        apply_style, condition_usage, gate_report,
                                        half_violin, savefig)


def main():
    apply_style()
    g = gate_report()
    cu = condition_usage()
    mae = cu['mae_cond'].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.16, wspace=0.25)

    # (a) 组级条形：模型 vs persistence
    ax = axes[0]
    vals = [g['mae_model_subject_equal_weight'],
            g['mae_persistence_subject_equal_weight']]
    bars = ax.bar([0, 1], vals, width=0.55, color=[HC, '#B8B8B8'],
                  edgecolor='black', lw=0.9, zorder=3)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.012, f'{v:.4f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold')
    ax.annotate('', xy=(1, vals[1] * 0.98), xytext=(1, vals[1] * 0.52),
                arrowprops=dict(arrowstyle='-', color=ACCENT_RED, lw=1.2))
    ax.set_xticks([0, 1], ['HC-reference model', 'Persistence baseline'])
    ax.set_ylabel('Rolling lag-1 MAE (subject-equal-weight)')
    ax.set_ylim(0, max(vals) * 1.18)
    ax.text(0.5, vals[1] * 0.75,
            f'improvement\n{g["improvement"]*100:.1f}%\nperm p={g["permutation_p"]:.0e}',
            ha='center', va='center', fontsize=9, color=ACCENT_RED,
            fontweight='bold')
    ax.set_title('(a) Guardrail B4 (HC val, n=222)', fontsize=9.5)

    # (b) 被试级模型 MAE：点云 + 半小提琴（raincloud 卡片样式）
    ax = axes[1]
    rng = np.random.RandomState(0)
    xj = rng.uniform(-0.16, -0.04, size=mae.size)
    ax.scatter(xj, mae, s=7, color=HC, alpha=0.45, lw=0, zorder=3)
    half_violin(ax, mae, 0.0, HC, pale=HC_PALE, side='right', width=0.42)
    med = float(np.median(mae))
    ax.scatter([0.06], [med], s=90, facecolor='white', edgecolor=INK, lw=1.2,
               zorder=6)
    ax.text(0.10, med, f'{med:.3f}', va='center', fontsize=8.5, color=INK,
            fontweight='bold')
    ax.set_xlim(-0.42, 0.62)
    ax.set_xticks([0], ['per-subject\nmodel lag-1 MAE'])
    ax.set_yticklabels([])
    ax.tick_params(left=False)
    ax.set_title('(b) Subject-level model MAE', fontsize=9.5)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    fig.suptitle('Rolling one-step guardrail: model strongly beats persistence',
                 fontweight='bold', fontsize=11, y=0.99)
    savefig(fig, 'fig_f1_gate')


if __name__ == '__main__':
    main()
