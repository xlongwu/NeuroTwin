"""Summarize profile reference probe and descriptive segment similarities."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.site_subset import parse_site


def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);xc=x-x.mean();yc=y-y.mean();den=np.linalg.norm(xc)*np.linalg.norm(yc)
 return float(xc@yc/den) if den>1e-12 else np.nan

def cosine(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);den=np.linalg.norm(x)*np.linalg.norm(y)
 return float(x@y/den) if den>1e-12 else np.nan

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);reports=[];groups={}
 for group in ('hc','mdd'):
  root=a.source/group;m=json.loads((root/'数据清单.json').read_text());missing=[s for s in m['subjects'] if not (root/f'{s}.npz').exists()]
  if missing:raise ValueError(f'{group}: missing {len(missing)} subjects')
  subjects=[];segments=[];profiles=[];reference=None
  for sid in m['subjects']:
   with np.load(root/f'{sid}.npz') as z:
    base,ref,true=z['baseline'],z['reference_prediction'],z['true'];original=z['original_profile'];reference=z['reference_profile'].copy()
    denom=np.abs(base).mean();profile_diff=np.linalg.norm(original-reference)
    starts=z['segment_start'];vectors=z['segment_profile'];profiles.append(original.copy())
    pairs=[]
    for start,v in zip(starts[1:],vectors[1:]):
     c=cosine(original,v);r=corr(original,v);pairs.append((c,r))
     segments.append({'group':group,'subject':sid,'site':parse_site(sid),'start_tr':int(start),'gap_tr':int(start),
                      'cosine':c,'pearson':r,'l2_distance':float(np.linalg.norm(original-v))})
    subjects.append({'subject':sid,'site':parse_site(sid),'n_anchors':len(base),'n_segments':len(starts),
       'n_later_segments':len(pairs),'baseline_mae':np.abs(base-true).mean(),
       'reference_mae':np.abs(ref-true).mean(),'delta_mae':np.abs(ref-true).mean()-np.abs(base-true).mean(),
       'mean_abs_output_difference':np.abs(ref-base).mean(),'relative_output_difference':np.abs(ref-base).mean()/denom if denom>0 else np.nan,
       'profile_norm':np.linalg.norm(original),'reference_norm':np.linalg.norm(reference),'reference_l2_distance':profile_diff,
       'cached_profile_max_abs':np.max(np.abs(original-z['cached_profile'])),
       'first_segment_max_abs':np.max(np.abs(original-vectors[0])),
       'mean_later_cosine':np.nanmean([x[0] for x in pairs]) if pairs else np.nan,
       'mean_later_pearson':np.nanmean([x[1] for x in pairs]) if pairs else np.nan})
  subjects=pd.DataFrame(subjects);segments=pd.DataFrame(segments);profiles=np.stack(profiles)
  # Descriptive baseline: correlations of each prefix profile with other test subjects.
  across=np.corrcoef(profiles);np.fill_diagonal(across,np.nan)
  subjects['between_subject_mean_pearson']=np.nanmean(across,axis=1)
  same=[];n_same=[];sites=subjects['site'].to_numpy()
  for i in range(len(subjects)):
   index=np.flatnonzero((sites==sites[i])&(np.arange(len(subjects))!=i));n_same.append(len(index))
   same.append(float(np.nanmean(across[i,index])) if len(index) else np.nan)
  subjects['same_site_mean_pearson']=same;subjects['n_other_same_site']=n_same
  actual_min=profiles.min(axis=0);actual_max=profiles.max(axis=0)
  dimensions=pd.DataFrame({'dimension':np.arange(1,33),'reference':reference,'observed_min':actual_min,'observed_max':actual_max,
                            'outside_test_range':(reference<actual_min)|(reference>actual_max)})
  subjects.to_csv(a.out/f'{group}_profile探针_逐被试.csv',index=False);segments.to_csv(a.out/f'{group}_profile片段相似性.csv',index=False)
  dimensions.to_csv(a.out/f'{group}_profile参考尺度.csv',index=False);groups[group]=subjects
  reports.append({'group':group,'subjects':len(subjects),'reference_norm':float(np.linalg.norm(reference)),
                  'observed_profile_norm_min':float(subjects.profile_norm.min()),'observed_profile_norm_max':float(subjects.profile_norm.max()),
                  'reference_outside_test_range_dimensions':int(dimensions.outside_test_range.sum()),
                  'profile_l2_distance_median':float(subjects.reference_l2_distance.median()),
                  'delta_mae_mean':float(subjects.delta_mae.mean()),'relative_output_median':float(subjects.relative_output_difference.median()),
                  'later_pearson_median':float(subjects.mean_later_pearson.median()),
                  'between_subject_pearson_median':float(subjects.between_subject_mean_pearson.median()),
                  'same_site_pearson_median':float(subjects.same_site_mean_pearson.median()),
                  'n_with_same_site_peer':int((subjects.n_other_same_site>0).sum()),'later_cosine_median':float(subjects.mean_later_cosine.median()),
                  'n_with_later_segments':int((subjects.n_later_segments>0).sum()),'max_cached_profile_difference':float(subjects.cached_profile_max_abs.max())})
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(8,6),sharey='col',constrained_layout=True);rng=np.random.default_rng(2024)
 for i,group in enumerate(('hc','mdd')):
  d=groups[group]
  for j,(col,label) in enumerate([('delta_mae','Reference − original MAE'),('mean_later_pearson','Mean within-scan profile Pearson r')]):
   ax=axes[i,j];v=d[col].dropna().to_numpy();ax.boxplot([v],positions=[1],widths=.35,patch_artist=True,showfliers=False,
    boxprops={'facecolor':'#87ceeb','edgecolor':'#2b438d'},medianprops={'color':'#2b438d'})
   ax.scatter(1+rng.uniform(-.12,.12,len(v)),v,s=5,c='#2b438d',alpha=.4);ax.set_xticks([1],[group.upper()]);ax.set_ylabel(label)
 fig.suptitle('Exploratory profile reference and nonoverlapping segments')
 for ext in ('png','pdf'):fig.savefig(a.out/f'profile补充.{ext}',dpi=300)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(8,3.7),sharey=True,constrained_layout=True);rng=np.random.default_rng(2024)
 for ax,group in zip(axes,('hc','mdd')):
  d=groups[group];arrays=[d[c].dropna().to_numpy() for c in ('mean_later_pearson','between_subject_mean_pearson','same_site_mean_pearson')]
  ax.boxplot(arrays,patch_artist=True,showfliers=False,boxprops={'facecolor':'#87ceeb','edgecolor':'#2b438d'},medianprops={'color':'#2b438d'})
  for j,v in enumerate(arrays):ax.scatter(j+1+rng.uniform(-.12,.12,len(v)),v,s=5,c='#2b438d',alpha=.3)
  ax.set_xticks([1,2,3],['Within scan','Other subjects','Same site'],rotation=15);ax.set_ylabel('Mean profile Pearson r');ax.set_title(f'{group.upper()}')
 fig.suptitle('Profile similarity: within scan and test-subject baselines')
 for ext in ('png','pdf'):fig.savefig(a.out/f'profile相似性参照.{ext}',dpi=300)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(7,3.6),constrained_layout=True);rng=np.random.default_rng(2024)
 for ax,(column,label) in zip(axes,[('mean_abs_output_difference','Mean absolute output difference'),('relative_output_difference','Mean absolute difference / mean absolute baseline')]):
  arrays=[groups[g][column].dropna().to_numpy() for g in ('hc','mdd')]
  ax.boxplot(arrays,patch_artist=True,showfliers=False,boxprops={'facecolor':'#87ceeb','edgecolor':'#2b438d'},medianprops={'color':'#2b438d'})
  for j,v in enumerate(arrays):ax.scatter(j+1+rng.uniform(-.12,.12,len(v)),v,s=5,c='#2b438d',alpha=.3)
  ax.set_xticks([1,2],['HC','MDD']);ax.set_ylabel(label)
 fig.suptitle('Fixed profile reference: output difference (groups analyzed separately)')
 for ext in ('png','pdf'):fig.savefig(a.out/f'profile输出差.{ext}',dpi=300)
 plt.close(fig);(a.out/'profile补充汇总.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
