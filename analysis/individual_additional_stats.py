"""Graph trajectories, input quality and subject-level sign-flip tests."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.h18_graph import ROOT,NETWORKS
from analysis.h18_probes import GROUPS
from analysis.h18_data import calibrate_raw
from inference_twin import load_raw_series


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def bh(p):
 order=np.argsort(p);q=np.empty_like(p);sorted_p=p[order];q[order]=np.minimum.accumulate((sorted_p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1);return q


def signflip(vals,seed=2024,n=10000):
 rng=np.random.default_rng(seed);obs=np.abs(vals.mean(axis=0));counts=np.zeros(vals.shape[1],dtype=np.int64)
 for start in range(0,n,250):
  k=min(250,n-start);signs=rng.choice(np.array([-1.,1.]),size=(k,len(vals)))
  counts+=(np.abs(signs@vals/len(vals))>=obs-1e-15).sum(axis=0)
 return (counts+1)/(n+1)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 atlas=pd.read_excel(ROOT/'data/AAL116.xlsx');nets=atlas['对应网络'].to_numpy();inventory=json.loads((a.root/'数据来源清单.json').read_text())
 rows=[];trend=[];network_trend=[];test_rows=[];summary=[]
 for group in ('hc','mdd'):
  ckpt,split,condition_file=GROUPS[group];ids=json.loads((ROOT/'data'/split).read_text())['test']
  for file,key in [(ROOT/ckpt,'checkpoint_sha256'),(ROOT/'data'/split,'split_sha256'),(ROOT/'data'/condition_file,'condition_table_sha256')]:
   if sha(file)!=inventory['groups'][group][key]:raise ValueError(f'{file}: 与来源清单不一致')
  for sid in ids:
   raw_path=ROOT/f'data/Raw/{group.upper()}/ROISignals_{sid}.mat'
   if sha(raw_path)!=inventory['groups'][group]['raw_subject_sha256'][sid]:raise ValueError(f'{sid}: 原始数据哈希变化')
   raw=load_raw_series(raw_path,116);series=calibrate_raw(raw,32)
   with np.load(a.root/'基线缓存'/group/f'{sid}.npz') as z:
    anchors=z['cutoff'];edges=np.triu_indices(116,1)
    for ai,t in enumerate(anchors):
     history=series[t-63:t+1];sd=history.std(axis=0)
     for roi in range(116):rows.append({'group':group,'subject':sid,'anchor':int(t),'roi':roi+1,'network':nets[roi],
       'finite':bool(np.isfinite(history[:,roi]).all()),'exact_constant':bool(np.ptp(history[:,roi])==0),
       'near_constant':bool(sd[roi]<=1e-6),'history_sd':float(sd[roi])})
     for key in ('history_fc_abs','A_func','A_eff'):
      mat=z[key][ai];v=mat[edges]
      trend.append({'group':group,'subject':sid,'anchor_rank':ai+1,'cutoff':int(t),'relative_position':float(t/(len(series)-1)),
                    'graph':key,'edge_mean':float(v.mean()),'edge_median':float(np.median(v))})
      for left in NETWORKS:
       for right in NETWORKS:
        ii,jj=np.meshgrid(np.flatnonzero(nets==left),np.flatnonzero(nets==right),indexing='ij');valid=ii!=jj
        network_trend.append({'group':group,'subject':sid,'anchor_rank':ai+1,'graph':key,'source_network':left,
                              'target_network':right,'mean':float(mat[ii[valid],jj[valid]].mean())})
   prefix=raw[:32];prefix_sd=prefix.std(axis=0)
   for roi in range(116):rows.append({'group':group,'subject':sid,'anchor':-1,'roi':roi+1,'network':nets[roi],
      'finite':bool(np.isfinite(prefix[:,roi]).all()),'exact_constant':bool(np.ptp(prefix[:,roi])==0),
      'near_constant':bool(prefix_sd[roi]<=1e-6*max(abs(float(prefix[:,roi].mean())),1.0)), 'history_sd':float(prefix_sd[roi])})
  table=pd.read_csv(a.root/'主分析图表'/group/'逐被试_ROI.csv');v=table.pivot(index='subject',columns='roi',values='delta_mae').loc[ids].to_numpy()
  nvals=np.stack([v[:,nets==n].mean(axis=1) for n in NETWORKS],axis=1)
  for family,values,names in [('ROI',v,[str(i) for i in range(1,117)]),('network',nvals,NETWORKS)]:
   pv=signflip(values);qv=bh(pv)
   for j,name in enumerate(names):test_rows.append({'group':group,'family':family,'item':name,'n_subjects':len(ids),
     'effect_mean':values[:,j].mean(),'p_two_sided':pv[j],'q_bh':qv[j],'permutations':10000,'seed':2024})
  summary.append({'group':group,'subjects':len(ids)})
 pd.DataFrame(rows).to_csv(a.out/'输入质量_逐锚点ROI.csv',index=False)
 pd.DataFrame(trend).to_csv(a.out/'图轨迹_逐被试.csv',index=False)
 pd.DataFrame(network_trend).to_csv(a.out/'网络块轨迹_逐被试.csv',index=False)
 tests=pd.DataFrame(test_rows);tests.to_csv(a.out/'符号翻转与FDR.csv',index=False)
 q=pd.DataFrame(rows);q.groupby(['group','anchor'],as_index=False)[['finite','exact_constant','near_constant']].sum().to_csv(a.out/'输入质量_覆盖.csv',index=False)
 df=pd.DataFrame(trend);fig,axes=plt.subplots(2,3,figsize=(10,5.8),sharey='col',constrained_layout=True)
 for i,group in enumerate(('hc','mdd')):
  for j,key in enumerate(('history_fc_abs','A_func','A_eff')):
   d=df[(df.group==group)&(df.graph==key)];g=d.groupby('anchor_rank').edge_mean
   mean=g.mean();se=g.std()/np.sqrt(g.count());x=mean.index.to_numpy()
   axes[i,j].plot(x,mean,color='#2b438d',marker='o',ms=3);axes[i,j].fill_between(x,mean-1.96*se,mean+1.96*se,color='#87ceeb',alpha=.4)
   axes[i,j].set_title(f'{group.upper()} {key}');axes[i,j].set_xlabel('Anchor rank');axes[i,j].set_ylabel('Mean non-self edge')
 fig.suptitle('Graph trajectories: subject mean and normal-approximation 95% interval')
 for ext in ('png','pdf'):fig.savefig(a.out/f'图轨迹.{ext}',dpi=300)
 plt.close(fig)
 for group in ('hc','mdd'):
  z=q[q.group==group];s=tests[tests.group==group]
  print(group,'history exact constant',int(z[z.anchor>=0].exact_constant.sum()),'near constant',int(z[z.anchor>=0].near_constant.sum()),'prefix near',int(z[z.anchor<0].near_constant.sum()),'ROI q<.05',int(((s.family=='ROI')&(s.q_bh<.05)).sum()),'network q<.05',int(((s.family=='network')&(s.q_bh<.05)).sum()),flush=True)

if __name__=='__main__':main()
