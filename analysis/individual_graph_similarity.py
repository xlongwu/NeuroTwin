"""Nonoverlapping within-scan correlations for all three model graphs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.site_subset import parse_site


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[];subject_rows=[];edge=np.triu_indices(116,1)
 for group in ('hc','mdd'):
  src=a.source/group;m=json.loads((src/'数据清单.json').read_text())
  for sid in m['subjects']:
   with np.load(src/f'{sid}.npz') as z:
    anchors=z['cutoff'];pairs=[(i,j) for i in range(len(anchors)) for j in range(i+1,len(anchors)) if anchors[j]-anchors[i]>=64]
    for key in ('history_fc_abs','A_func','A_eff'):
     matrices=z[key][:,edge[0],edge[1]].astype(np.float64);matrices-=matrices.mean(axis=1,keepdims=True);norm=np.linalg.norm(matrices,axis=1)
     vals=[];gaps=[]
     for i,j in pairs:
      den=norm[i]*norm[j];r=float(matrices[i]@matrices[j]/den) if den>0 else np.nan;gap=int(anchors[j]-anchors[i])
      rows.append({'group':group,'subject':sid,'site':parse_site(sid),'graph':key,'anchor_1':int(anchors[i]),'anchor_2':int(anchors[j]),'gap_tr':gap,'pearson_r':r})
      vals.append(r);gaps.append(gap)
     valid=np.array(vals,dtype=float);valid=valid[np.isfinite(valid)]
     subject_rows.append({'group':group,'subject':sid,'site':parse_site(sid),'graph':key,'n_anchor_pairs':len(pairs),
                          'n_valid_pairs':len(valid),'gap_min':min(gaps) if gaps else np.nan,'gap_max':max(gaps) if gaps else np.nan,
                          'mean_pearson_r':float(valid.mean()) if len(valid) else np.nan})
 pd.DataFrame(rows).to_csv(a.out/'三图_非重叠窗逐对相似性.csv',index=False)
 subjects=pd.DataFrame(subject_rows);subjects.to_csv(a.out/'三图_非重叠窗逐被试.csv',index=False)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(8,3.8),sharey=True,constrained_layout=True);rng=np.random.default_rng(2024)
 for ax,group in zip(axes,('hc','mdd')):
  arrays=[subjects[(subjects.group==group)&(subjects.graph==key)].mean_pearson_r.dropna().to_numpy() for key in ('history_fc_abs','A_func','A_eff')]
  ax.boxplot(arrays,patch_artist=True,showfliers=False,boxprops={'facecolor':'#87ceeb','edgecolor':'#2b438d'},medianprops={'color':'#2b438d'})
  for i,v in enumerate(arrays):ax.scatter(i+1+rng.uniform(-.14,.14,len(v)),v,s=5,c='#2b438d',alpha=.4)
  ax.set_xticks([1,2,3],['History |FC|','Learned','Effective']);ax.set_title(f'{group.upper()} (subject means)');ax.set_ylabel('Within-subject nonoverlap Pearson r')
 fig.suptitle('Model graph similarity within scan; windows do not overlap')
 for ext in ('png','pdf'):fig.savefig(a.out/f'三图_非重叠窗相似性.{ext}',dpi=300)
 plt.close(fig)
 print(subjects.groupby(['group','graph']).mean_pearson_r.agg(['count','median']).to_string())

if __name__=='__main__':main()
