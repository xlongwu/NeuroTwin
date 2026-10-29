"""Describe existing rolling test errors by ROI, network and site."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.site_subset import parse_site

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/H1-8当前结果'
OUT = ROOT / 'results/H1-8分析_20261028'
NETWORKS = ['VN', 'SMN', 'DAN', 'SN', 'FN', 'DMN', 'LN']


def main():
    table_dir, fig_dir = OUT / '03_可解释性', OUT / '04_图件'
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_excel(ROOT / 'data/AAL116.xlsx')
    if len(atlas) != 116 or set(atlas['对应网络']) != set(NETWORKS):
        raise ValueError('AAL116 网络表与预期不一致')
    # Match the project's established correction for duplicate atlas labels 25/26.
    atlas.loc[24, 'micro命名'] = 'Frontal_Med_Orb_L'
    atlas.loc[25, 'micro命名'] = 'Frontal_Med_Orb_R'
    roi_rows, network_rows, site_rows = [], [], []
    arrays = {}
    for group, folder, sub in [('HC', '滚动单步_健康对照', 'hc'),
                               ('MDD', '滚动单步_抑郁症_微调', 'mdd')]:
        with np.load(SOURCE / folder / sub / '逐被试_ROI误差_测试集.npz') as d:
            subjects = d['subjects'].astype(str)
            model, persistence = d['mae_model'].astype(float), d['mae_persistence'].astype(float)
        subject_table = pd.read_csv(SOURCE / folder / sub / '逐被试指标_测试集.csv').set_index('subject')
        anchors = subject_table.loc[subjects, 'n_anchors'].to_numpy(dtype=int)
        if model.shape != (len(subjects), 116) or persistence.shape != model.shape:
            raise ValueError('逐被试 ROI 误差维度不匹配')
        arrays[group] = np.nanmean(model, axis=0)
        for i in range(116):
            good = np.isfinite(model[:, i]) & np.isfinite(persistence[:, i]) & (persistence[:, i] > 0)
            improvement = 1 - model[good, i] / persistence[good, i]
            roi_rows.append(dict(group=group, roi=i + 1, label=atlas.iloc[i]['micro命名'],
                                 network=atlas.iloc[i]['对应网络'], n_subjects=int(good.sum()),
                                 n_samples=int(anchors[good].sum()), n_invalid=int((~good).sum()), mae_mean=model[good, i].mean(),
                                 mae_sd=model[good, i].std(), improvement_mean=improvement.mean()))
        for network in NETWORKS:
            idx = np.asarray(atlas['对应网络'] == network)
            m, p = np.nanmean(model[:, idx], axis=1), np.nanmean(persistence[:, idx], axis=1)
            good = np.isfinite(m) & np.isfinite(p) & (p > 0)
            network_rows.append(dict(group=group, network=network, n_rois=int(idx.sum()),
                                     n_subjects=int(good.sum()), n_invalid=int((~good).sum()),
                                     n_samples=int(anchors[good].sum()),
                                     mae_mean=m[good].mean(), mae_sd=m[good].std(),
                                     improvement_mean=(1-m[good]/p[good]).mean()))
        sites = np.array([parse_site(s) for s in subjects])
        m, p = np.nanmean(model, axis=1), np.nanmean(persistence, axis=1)
        for site in sorted(set(sites)):
            good = (sites == site) & np.isfinite(m) & np.isfinite(p) & (p > 0)
            site_rows.append(dict(group=group, site=site, n_subjects=int(good.sum()),
                                  n_invalid=int(((sites == site) & ~good).sum()),
                                  n_samples=int(anchors[good].sum()),
                                  mae_mean=m[good].mean() if good.any() else np.nan,
                                  mae_sd=m[good].std() if good.any() else np.nan,
                                  improvement_mean=(1-m[good]/p[good]).mean() if good.any() else np.nan))
    roi, net, sites = map(pd.DataFrame, (roi_rows, network_rows, site_rows))
    roi.to_csv(table_dir / '表03_误差结构_滚动单步_逐ROI.csv', index=False)
    net.to_csv(table_dir / '表04_误差结构_滚动单步_网络.csv', index=False)
    sites.to_csv(table_dir / '表05_误差结构_滚动单步_站点.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), constrained_layout=True)
    x = np.arange(len(NETWORKS))
    for group, color, offset in [('HC', '#df5252', -.09), ('MDD', '#643c33', .09)]:
        d = net[net.group == group].set_index('network').loc[NETWORKS]
        axes[0].errorbar(x + offset, d.mae_mean, yerr=d.mae_sd, color=color,
                         fmt='o', capsize=3, label=group)
    axes[0].set(xticks=x, xticklabels=NETWORKS, ylabel='Rolling MAE', title='Network: subject mean ± SD')
    axes[0].legend(frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1, 1.02))
    order = np.concatenate([np.flatnonzero(atlas['对应网络'].to_numpy() == n) for n in NETWORKS])
    im = axes[1].imshow(np.stack([arrays['HC'][order], arrays['MDD'][order]]),
                        aspect='auto', cmap='magma', interpolation='nearest')
    mids, end = [], 0
    for n in NETWORKS:
        count = int((atlas['对应网络'] == n).sum())
        mids.append(end + (count - 1)/2)
        if end: axes[1].axvline(end-.5, color='white', lw=.7)
        end += count
    axes[1].set(yticks=[0, 1], yticklabels=['HC', 'MDD'], xticks=mids,
                xticklabels=NETWORKS, title='ROI: subject mean rolling MAE')
    fig.colorbar(im, ax=axes[1], label='MAE', pad=.02)
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'图02_误差结构_滚动单步_被试等权.{ext}', dpi=300)
    plt.close(fig)

    names = sorted(sites.site.unique())
    fig, ax = plt.subplots(figsize=(8, 13), constrained_layout=True)
    for group, color, offset in [('HC', '#df5252', -.16), ('MDD', '#643c33', .16)]:
        d = sites[sites.group == group].set_index('site').reindex(names)
        y = np.arange(len(names)) + offset
        ax.scatter(d.mae_mean, y, color=color, s=12, label=group)
        for value, yy, count in zip(d.mae_mean, y, d.n_subjects):
            if np.isfinite(value):
                ax.annotate(str(int(count)), (value, yy), xytext=(4, 0),
                            textcoords='offset points', va='center', fontsize=6, color=color)
    ax.set(yticks=np.arange(len(names)), yticklabels=names, xlabel='Subject mean rolling MAE',
           title='Test sites: number beside each point is cohort size')
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=.1)
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'图03_误差结构_滚动单步_站点.{ext}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
