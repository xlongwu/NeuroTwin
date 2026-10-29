"""Plot existing test-set subject-level H1–8 metrics without rerunning inference."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/H1-8当前结果'
OUTPUT = ROOT / 'results/H1-8分析_20261028'


def main():
    tables = OUTPUT / '01_核心指标'
    figures = OUTPUT / '04_图件'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics = [('mae', 'MAE'), ('spatial_pcc', 'Spatial PCC'),
               ('direction_accuracy_step', 'Direction accuracy')]
    rows = []
    for group, folder in [('HC', '健康对照'), ('MDD', '抑郁症')]:
        df = pd.read_csv(SOURCE / folder / '逐被试指标_测试集.csv')
        for key, _ in metrics:
            for h in range(1, 9):
                values = pd.to_numeric(df[f'ODE_H{h}_{key}'], errors='raise')
                good = np.isfinite(values)
                rows.append({'group': group, 'metric': key, 'horizon': h,
                             'mean': values[good].mean(), 'sd': values[good].std(ddof=0),
                             'n_subjects': int(good.sum()),
                             'n_tasks': int(df.loc[good, 'n_tasks'].sum()),
                             'n_invalid_subjects': int((~good).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(tables / '表01_预测指标_H1-8_被试等权.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    colors = {'HC': '#df5252', 'MDD': '#643c33'}
    for ax, (key, label) in zip(axes, metrics):
        for group in colors:
            d = result[(result.metric == key) & (result.group == group)]
            x, y, sd = d.horizon.to_numpy(), d['mean'].to_numpy(), d.sd.to_numpy()
            ax.plot(x, y, marker='o', markersize=3, color=colors[group], label=group)
            ax.fill_between(x, y - sd, y + sd, color=colors[group], alpha=.14)
        ax.set(xlabel='Forecast horizon (TR)', ylabel=label, xticks=range(1, 9))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, title='Test cohort')
    fig.suptitle('Fixed-anchor H1–8 prediction: subject mean ± SD', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'图01_预测指标_H1-8_被试等权.{suffix}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
