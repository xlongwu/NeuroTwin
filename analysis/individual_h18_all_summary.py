"""Aggregate full-future eligible H1–8 anchors within each subject first."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.h18_graph import ROOT,NETWORKS
from analysis.individual_perturbation_summary import bootstrap_mean_ci


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=('hc','mdd'),required=True)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.source/a.group;m=json.loads((root/'数据清单.json').read_text());ids=m['subjects']
 if any(not (root/f'{sid}.npz').is_file() for sid in ids):raise ValueError('被试缓存尚未完成')
 nets=pd.read_excel(ROOT/'data/AAL116.xlsx')['对应网络'].to_numpy();vals=[];bases=[];coverage=[]
 for sid in ids:
  with np.load(root/f'{sid}.npz') as z:
   pred,true,base=z['perturbed'],z['true'],z['baseline'];n=len(z['cutoff'])
   if pred.shape!=(n,116,3,116,8) or not all(np.isfinite(x).all() for x in (pred,true,base)):raise ValueError(f'{sid}: shape/finite')
   vals.append((np.abs(pred-true[:,None,None,:,:]).mean(axis=3)-np.abs(base-true).mean(axis=1)[:,None,None,:]).mean(axis=(0,2)))
   bases.append(np.abs(base-true).mean(axis=(0,1)))
   coverage.append({'subject':sid,'n_anchors':n,'cutoffs':','.join(map(str,z['cutoff'])),
                    'h1_baseline_max_abs':float(z['h1_baseline_max_abs'].max()),'h1_perturbed_max_abs':float(z['h1_perturbed_max_abs'].max())})
 vals=np.stack(vals);bases=np.stack(bases);overall=vals.mean(axis=1)
 net=np.stack([vals[:,nets==name,:].mean(axis=1) for name in NETWORKS],axis=1)
 interval=bootstrap_mean_ci(overall);out=a.output/a.group;out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(coverage).to_csv(out/'锚点覆盖.csv',index=False)
 pd.DataFrame({'horizon':np.arange(1,9),'n_subjects':len(ids),'baseline_mean':bases.mean(axis=0),'delta_mean':overall.mean(axis=0),
               'delta_median':np.median(overall,axis=0),'ci_low':interval[0],'ci_high':interval[1]}).to_csv(out/'整体汇总.csv',index=False)
 subject=[];network=[]
 for i,sid in enumerate(ids):
  for h in range(8):subject.append({'subject':sid,'horizon':h+1,'baseline_mae':bases[i,h],'delta_mae':overall[i,h],
                                   **{name:net[i,j,h] for j,name in enumerate(NETWORKS)}})
 pd.DataFrame(subject).to_csv(out/'逐被试.csv',index=False)
 for j,name in enumerate(NETWORKS):
  ci=bootstrap_mean_ci(net[:,j,:])
  for h in range(8):network.append({'network':name,'n_roi':int((nets==name).sum()),'horizon':h+1,'n_subjects':len(ids),
                                    'delta_mean':net[:,j,h].mean(),'ci_low':ci[0,h],'ci_high':ci[1,h]})
 pd.DataFrame(network).to_csv(out/'网络汇总.csv',index=False)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True);x=np.arange(1,9)
 axes[0].plot(x,overall.mean(axis=0),color='#2b438d',marker='o',ms=3);axes[0].fill_between(x,interval[0],interval[1],color='#87ceeb',alpha=.45)
 axes[0].axhline(0,color='#aaa',lw=.7);axes[0].set(xlabel='Prediction horizon',ylabel='ΔMAE',title='All source ROIs; 95% subject bootstrap CI')
 colors=['#2b438d','#87ceeb','#6592ac','#b36c66','#8b73ac','#629d86','#b38c4d']
 for j,name in enumerate(NETWORKS):axes[1].plot(x,net[:,j,:].mean(axis=0),label=name,color=colors[j],lw=1.2)
 axes[1].axhline(0,color='#aaa',lw=.7);axes[1].set(xlabel='Prediction horizon',ylabel='ΔMAE',title='Source network means');axes[1].legend(ncol=2,frameon=False,fontsize=7)
 fig.suptitle(f'{a.group.upper()}: all eligible sampled anchors, H1–8; n={len(ids)}')
 for ext in ('png','pdf'):fig.savefig(out/f'全有效锚点_H1-8.{ext}',dpi=300)
 plt.close(fig)
 c=pd.DataFrame(coverage)
 (out/'报告.md').write_text('\n'.join([f'# {a.group.upper()} 全有效锚点 H1–8','',f'被试 {len(ids)}，每人有效锚点范围 {c.n_anchors.min()}–{c.n_anchors.max()}。仅使用预测点后至少有 8 TR 的主分析抽样锚点；复用原始置换索引，无真值反馈。',
 '每人先对有效锚点、源 ROI 和置换平均，再对被试等权汇总；95% 区间为 10000 次被试 bootstrap，seed=2024。',
 f'H1 基线最大差 {c.h1_baseline_max_abs.max():.3g}，H1 扰动最大差 {c.h1_perturbed_max_abs.max():.3g}。','']))
 print(a.group,'subjects',len(ids),'anchor range',int(c.n_anchors.min()),int(c.n_anchors.max()),'H1 errors',c.h1_perturbed_max_abs.max())

if __name__=='__main__':main()
