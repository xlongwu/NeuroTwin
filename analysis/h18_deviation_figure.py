"""Plot prespecified matched ROI and network differences from saved tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/H1-8分析_20261028'
NETWORKS = ['VN', 'SMN', 'DAN', 'SN', 'FN', 'DMN', 'LN']


def main():
    roi = pd.read_csv(OUT/'02_病理分析/表09_组间偏离_滚动单步_匹配ROI.csv')
    net = pd.read_csv(OUT/'02_病理分析/表10_组间偏离_滚动单步_匹配网络.csv').set_index('network').loc[NETWORKS]
    roi = pd.concat([roi[roi.network == n] for n in NETWORKS], ignore_index=True)
    fig, (ax, heat) = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios':[3,1]}, constrained_layout=True)
    x = np.arange(len(NETWORKS))
    ax.errorbar(x, net.delta_z, yerr=[net.delta_z-net.ci_low, net.ci_high-net.delta_z],
                fmt='o', capsize=3, color='#27408b', label='Matched mean and 95% bootstrap CI')
    ax.axhline(0, color='0.6', lw=.8)
    ax.set(xticks=x, xticklabels=NETWORKS, ylabel='MDD − HC mean z', title=f'Matched pairs across splits (n={int(roi.n_pairs.iloc[0]):,}); mean and 95% bootstrap CI')
    ax.spines[['top','right']].set_visible(False)
    limit = np.abs(roi.delta_z).max()
    im = heat.imshow(roi.delta_z.to_numpy()[None,:], aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
    mids, end = [], 0
    for n in NETWORKS:
        count = int((roi.network == n).sum())
        mids.append(end+(count-1)/2)
        if end: heat.axvline(end-.5,color='white',lw=.8)
        end += count
    heat.set(xticks=mids,xticklabels=NETWORKS,yticks=[],title=f'All 116 ROI mean differences; {int((roi.q < .05).sum())} pass FDR')
    fig.colorbar(im, ax=heat, label='MDD − HC mean z', fraction=.03, pad=.02)
    figure_dir = OUT/'04_图件'; figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(figure_dir/f'图05_健康偏离_滚动单步_匹配组.{ext}',dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
