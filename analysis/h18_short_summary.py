"""Summarize exploratory within-scan H1–8 identity and ACF results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/H1-8分析_20261028'


def main():
    d=pd.read_csv(OUT/'03_可解释性/表26_短序列一致性_H1-8_逐被试.csv')
    metrics=['predicted_fc_identified','observed_fc_identified','profile_half_identified','acf_mae_lag1_5',
             'profile_half_own_distance','profile_half_other_distance']
    rows=[]
    for group,g in d.groupby('group'):
        for metric in metrics:
            x=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy()
            rows.append({'group':group,'metric':metric,'mean':float(x.mean()) if len(x) else None,
                         'sd':float(x.std()) if len(x) else None,'n_subjects':len(x),'n_invalid':len(g)-len(x)})
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'03_可解释性/表32_短序列一致性_H1-8_被试等权.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    cats=metrics[:3];labels=['Predicted → true\nH1–8 FC','History 8 → true\nH1–8 FC','Profile\nhalves'];x=np.arange(3)
    for group,color,offset in [('hc','#df5252',-.12),('mdd','#643c33',.12)]:
        v=summary[summary.group==group].set_index('metric').loc[cats]
        axes[0].scatter(x+offset,v['mean'],color=color,label=f'{group.upper()} (n={int(v.n_subjects.iloc[0])})')
    axes[0].axhline(1/222,color='.6',ls='--',lw=.8,label='Chance ≈ 1/n')
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1.05),ylabel='Same-subject top-1 rate',title='Within-scan identity (exploratory)')
    axes[0].legend(frameon=False,fontsize=7)
    rng=np.random.RandomState(2024)
    for i,(group,color) in enumerate([('hc','#df5252'),('mdd','#643c33')]):
        vals=d[d.group==group].acf_mae_lag1_5.dropna().to_numpy()
        axes[1].boxplot(vals,positions=[i],widths=.35,showfliers=False,patch_artist=True,
                        boxprops={'facecolor':color,'alpha':.2},medianprops={'color':color})
        axes[1].scatter(i+rng.uniform(-.15,.15,len(vals)),vals,s=5,alpha=.25,color=color)
    axes[1].set(xticks=[0,1],xticklabels=['HC','MDD'],ylabel='ACF MAE, lags 1–5',title='H1–8 short-window ACF')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('First test anchor; same-scan comparisons, not external identification')
    for ext in ('png','pdf'):fig.savefig(OUT/f'04_图件/图13_短序列一致性_H1-8_测试集.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
