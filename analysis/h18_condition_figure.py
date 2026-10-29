"""Condition-switch sensitivity; HC validation and MDD test shown descriptively."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/H1-8分析_20261028'


def main():
    p=OUT/'03_可解释性';datasets=[]
    for group,path in [('HC val / hc3','表02_条件使用_滚动单步_逐被试.csv'),('MDD test / mdd8','表28_条件使用MDD_滚动单步_逐被试.csv')]:
        d=pd.read_csv(p/path);d['group']=group;d['mae_change']=d.mae_cond-d.mae_zero_cond;datasets.append(d)
    data=pd.concat(datasets,ignore_index=True);fig,axes=plt.subplots(1,2,figsize=(8,3.6),constrained_layout=True)
    rng=np.random.RandomState(2024)
    for ax,metric,title in zip(axes,['delta_pred_relative','mae_change'],['Prediction change / signal scale','MAE: full − missing condition']):
        for i,(group,color) in enumerate([('HC val / hc3','#df5252'),('MDD test / mdd8','#643c33')]):
            vals=data[data.group==group][metric].dropna().to_numpy()
            ax.boxplot(vals,positions=[i],widths=.4,showfliers=False,patch_artist=True,
                       boxprops={'facecolor':color,'alpha':.2},medianprops={'color':color})
            ax.scatter(i+rng.uniform(-.18,.18,len(vals)),vals,s=5,color=color,alpha=.25)
        ax.axhline(0,color='.65',lw=.7)
        ax.set(xticks=[0,1],xticklabels=['HC val\nhc3','MDD test\nmdd8'],title=title,ylabel='Relative change' if metric=='delta_pred_relative' else 'MAE change')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Rolling lag-1; settings differ between cohorts')
    for ext in ('png','pdf'):fig.savefig(OUT/f'04_图件/图14_条件使用_滚动单步_逐被试.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
