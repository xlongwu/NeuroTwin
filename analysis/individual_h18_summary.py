"""Summarize first-anchor H1–8 permutation effects with subjects as units."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.h18_graph import NETWORKS,ROOT
from analysis.individual_perturbation_summary import bootstrap_mean_ci


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=('hc','mdd'),required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();root=a.cache_dir/a.group
    man=json.loads((root/'数据清单.json').read_text())
    missing=[sid for sid in man['subjects'] if not (root/f'{sid}.npz').exists()]
    if missing:raise ValueError(f'{len(missing)} subjects missing')
    atlas=pd.read_excel(ROOT/'data/AAL116.xlsx'); nets=atlas['对应网络'].to_numpy()
    if len(nets)!=116 or set(nets)!=set(NETWORKS):raise ValueError('Invalid atlas')
    values=[]; baseline=[]; diffs=[]
    for sid in man['subjects']:
        with np.load(root/f'{sid}.npz') as z:
            pred,true,base=z['perturbed'],z['true'],z['baseline']
            if pred.shape!=(116,3,116,8) or not all(np.isfinite(x).all() for x in (pred,true,base)):
                raise ValueError(f'{sid}: invalid H1–8 array')
            delta=np.abs(pred-true[None,None,:,:]).mean(axis=2)-np.abs(base-true).mean(axis=0)[None,None,:]
            values.append(delta.mean(axis=1));baseline.append(np.abs(base-true).mean(axis=0))
            diffs.append(float(z['first_h1_difference']))
    values=np.stack(values);baseline=np.stack(baseline)
    global_vals=values.mean(axis=1)
    network_vals=np.stack([values[:,nets==net,:].mean(axis=1) for net in NETWORKS],axis=1)
    ci=bootstrap_mean_ci(global_vals)
    dest=a.out_dir/a.group;dest.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame({'horizon':np.arange(1,9),'baseline_mean':baseline.mean(axis=0),
                         'delta_mean':global_vals.mean(axis=0),'delta_median':np.median(global_vals,axis=0),
                         'ci_low':ci[0],'ci_high':ci[1],'n_subjects':len(values)})
    result.to_csv(dest/'H1-8_整体汇总.csv',index=False)
    rows=[]
    for i,sid in enumerate(man['subjects']):
        for h in range(8):
            rows.append({'subject':sid,'horizon':h+1,'baseline_mae':baseline[i,h],
                         'delta_mae':global_vals[i,h],**{net:network_vals[i,j,h] for j,net in enumerate(NETWORKS)}})
    pd.DataFrame(rows).to_csv(dest/'H1-8_逐被试.csv',index=False)
    net_rows=[]
    for j,net in enumerate(NETWORKS):
        interval=bootstrap_mean_ci(network_vals[:,j,:])
        for h in range(8):net_rows.append({'network':net,'n_roi':int((nets==net).sum()),'horizon':h+1,
                                           'delta_mean':network_vals[:,j,h].mean(),'ci_low':interval[0,h],
                                           'ci_high':interval[1,h],'n_subjects':len(values)})
    pd.DataFrame(net_rows).to_csv(dest/'H1-8_网络汇总.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True); h=np.arange(1,9)
    axes[0].plot(h,global_vals.mean(axis=0),color='#2b438d',marker='o',ms=3)
    axes[0].fill_between(h,ci[0],ci[1],color='#87ceeb',alpha=.45)
    axes[0].axhline(0,color='#aaa',lw=.7);axes[0].set(xlabel='Prediction horizon',ylabel='ΔMAE',title='All source ROIs, 95% subject bootstrap CI')
    colors=['#2b438d','#87ceeb','#6592ac','#b36c66','#8b73ac','#629d86','#b38c4d']
    for j,net in enumerate(NETWORKS):axes[1].plot(h,network_vals[:,j,:].mean(axis=0),label=net,color=colors[j],lw=1.2)
    axes[1].axhline(0,color='#aaa',lw=.7);axes[1].set(xlabel='Prediction horizon',ylabel='ΔMAE',title='Source networks (subject means)')
    axes[1].legend(ncol=2,frameon=False,fontsize=7)
    fig.suptitle(f'{a.group.upper()}: first anchor H1–8, no truth feedback; n={len(values)}')
    for ext in ('png','pdf'):fig.savefig(dest/f'H1-8_误差变化.{ext}',dpi=300)
    plt.close(fig)
    (dest/'H1-8报告.md').write_text('\n'.join([f'# {a.group.upper()} 首锚点 H1–8 补充分析','',
      f'被试 {len(values)} 人；每人复用主分析首锚点的 116 ROI、每 ROI 3 次扰动，无真值反馈预测 H1–8。',
      '先对各被试的重复和源 ROI 求平均，再对被试等权汇总；区间为 10000 次被试 bootstrap（seed=2024）。这不是滚动单步评价。',
      f'H1 原始前向与主分析最大差 {max(diffs):.3g}。','']))


if __name__=='__main__':main()
