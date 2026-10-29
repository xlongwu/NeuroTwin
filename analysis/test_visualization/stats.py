"""Test-only descriptive tables. No archived mixed-cohort p/q values are reused."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from scipy.stats import pearsonr

from utils.site_subset import load_demo,build_matched_subset,parse_site
from utils.dataloader import load_condition_table
from analysis.pathology_tests import HC_MERGED_FILE,MDD_MERGED_FILE
from analysis.paper_figs.run_20261026.style import labels,partition,NETWORK_ORDER
from .common import ROOT,DATA,TABLES,test_ids,check_ids,samples,write_json,read_json


def write_table(directory: Path,name: str,rows) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(directory/f'{name}.csv',index=False)


def bootstrap_interval(values: np.ndarray,repetitions: int=1000) -> np.ndarray:
    x=np.asarray(values)
    rng=np.random.RandomState(2024)
    # Indexing in small blocks bounds memory for 116-dimensional arrays.
    means=[]
    for start in range(0,repetitions,50):
        ix=rng.randint(0,len(x),size=(min(50,repetitions-start),len(x)))
        means.append(x[ix].mean(1))
    return np.quantile(np.concatenate(means),[.025,.975],axis=0)


def base_tables(data_dir: Path,table_dir: Path) -> None:
    net,_=partition();names=labels()['micro命名'].tolist()
    metrics=[];subjects=[];horizon=[];roi_rows=[];network_rows=[];condition_rows=[];profile_rows=[]
    graph_mean={};all_z={};demo={}
    for group,merged in [('hc',HC_MERGED_FILE),('mdd',MDD_MERGED_FILE)]:
        ids=test_ids(group);ss=samples(group,data_dir)
        demo[group],missing=load_demo(ROOT/'data',merged,ids)
        for directory in ['滚动单步','自主生成32步']:
            table=pd.read_csv(ROOT/f'results/表格/健康参照模型/{directory}/per_subject.csv')
            check_ids(table.loc[table.group==group,'subject'],group)
        ztable=pd.read_csv(ROOT/f'results/表格/健康参照模型/参照偏离/{group}/roi_z.csv')
        check_ids(ztable.subject,group)
        all_z[group]={sid:row for sid,row in zip(ztable.subject,ztable.iloc[:,1:].to_numpy())}
        with np.load(ROOT/f'results/数据/健康参照模型/滚动单步/{group}/roi_mae.npz') as d:
            check_ids(d['subjects'],group)
            mm,pp=d['mae_model'].astype(float),d['mae_persistence'].astype(float)
            if np.any(pp<=0):raise ValueError(f'Undefined persistence improvement: {group}')
            improvement=100*(1-mm/pp)
            for j in range(116):
                roi_rows.append(dict(group=group,roi=j+1,label=names[j],network=net[j],
                    mae_model=mm[:,j].mean(),mae_persistence=pp[:,j].mean(),improvement=improvement[:,j].mean()))
            for n in NETWORK_ORDER:
                network_rows.append(dict(group=group,network=n,improvement=improvement[:,net==n].mean()))
        for sid,s in zip(ids,ss):
            pred,true=s['pred'],s['true']
            error=np.abs(pred-true).mean(1)
            mean_base=np.abs(s['history'].mean(0)[None]-true).mean(1)
            persistence=np.abs(s['history'][-1][None]-true).mean(1)
            cp=pred-pred.mean(0);ct=true-true.mean(0)
            den=np.sqrt((cp**2).sum(0)*(ct**2).sum(0))
            temporal=np.divide((cp*ct).sum(0),den,out=np.full(116,np.nan),where=den>0)
            metrics.append(dict(group=group,subject=sid,site=parse_site(sid),fc_r=float(s['fc_r']),
                psd_kl=float(s['psd_kl']),mae=float(s['mae']),temporal_pcc=np.nanmean(temporal),
                lambda_value=s['lambda_value'].mean(),model='HC reference',split='test'))
            for t in range(32):
                horizon.append(dict(group=group,subject=sid,horizon=t+1,mae=error[t],
                    mean_baseline_mae=mean_base[t],persistence_mae=persistence[t]))
            for seg,a,b in [('H1_8',0,8),('H9_16',8,16),('H17_32',16,32),('H1_32',0,32)]:
                subjects.append(dict(group=group,subject=sid,segment=seg,mae=error[a:b].mean(),
                    mean_baseline_mae=mean_base[a:b].mean(),persistence_mae=persistence[a:b].mean()))
            signal=np.abs(true[0]-s['history'][-1]).mean()
            condition_rows.append(dict(group=group,subject=sid,
                delta_pred_relative=np.abs(pred[0]-s['pred_zero'][0]).mean()/signal,
                mae_cond=np.abs(pred[0]-true[0]).mean(),mae_zero=np.abs(s['pred_zero'][0]-true[0]).mean(),
                delta_h32=np.abs(pred-s['pred_zero']).mean(),
                gamma=float(s['ode_film_gamma']),beta=float(s['ode_film_beta'])))
            profile_rows.extend(dict(group=group,subject=sid,dim=j,value=float(v)) for j,v in enumerate(s['profile']))
        graph_mean[group]={k:np.stack([s[k] for s in ss]).mean(0) for k in ['prior_norm','A_func','A_eff']}
        np.savez_compressed(data_dir/f'graph_mean_{group}.npz',**graph_mean[group])
    for name,rows in [('generation_metrics',metrics),('horizon_by_subject',subjects),('horizon_by_step',horizon),
        ('rolling_roi',roi_rows),('rolling_network',network_rows),('condition_usage',condition_rows),('profiles',profile_rows)]:
        write_table(table_dir,name,rows)
    # Matching uses only fixed test IDs and the existing prespecified demographic rules.
    matched=build_matched_subset(test_ids('hc'),test_ids('mdd'),demo['hc'],demo['mdd'],min_per_group=10,caliper=5.)
    pairs=matched['pairs']
    if not pairs:raise ValueError('No test-only same-site/sex/age-matched controls')
    write_json(data_dir/'test_matching.json',matched)
    write_table(table_dir,'test_pairs',[{k:p[k] for k in ['site','hc','mdd']} for p in pairs])
    # Negative control is a site-stratified random split within HC TEST only.
    negative={'A':[],'B':[]};rng=np.random.RandomState(2024)
    for site in sorted({parse_site(s) for s in test_ids('hc')}):
        ids=sorted(s for s in test_ids('hc') if parse_site(s)==site)
        if len(ids)<2:continue
        shuffled=[ids[i] for i in rng.permutation(len(ids))]
        half=len(ids)//2
        negative['A'].extend(shuffled[:half]);negative['B'].extend(shuffled[half:2*half])
    write_json(data_dir/'hc_test_negative_control.json',negative)
    md=np.stack([all_z['mdd'][p['mdd']] for p in pairs])
    hc=np.stack([all_z['hc'][p['hc']] for p in pairs]);diff=md-hc
    ci=bootstrap_interval(diff)
    unmatched=np.stack(list(all_z['mdd'].values())).mean(0)-np.stack(list(all_z['hc'].values())).mean(0)
    nc=np.stack([all_z['hc'][s] for s in negative['A']]).mean(0)-np.stack([all_z['hc'][s] for s in negative['B']]).mean(0)
    write_table(table_dir,'roi_descriptive_contrasts',[dict(roi=j+1,label=names[j],network=net[j],
        matched=diff[:,j].mean(),ci_low=ci[0,j],ci_high=ci[1,j],unmatched=unmatched[j],hc_negative=nc[j],
        n_pairs=len(pairs)) for j in range(116)])
    zrows=[]
    for group,zmap in all_z.items():
        wanted={p[group] for p in pairs}
        for sid,z in zmap.items():
            zrows.append(dict(group=group,subject=sid,site=parse_site(sid),mean_abs_z=np.abs(z).mean(),
                z_std=z.std(),matched=sid in wanted))
    write_table(table_dir,'deviation_subjects',zrows)
    # Prespecified scalar score, no classifier fitted or selected on test labels.
    zframe=pd.DataFrame(zrows);rocrows=[];aucs=[]
    for label,sub in [('All test',zframe),('Matched test',zframe[zframe.matched])]:
        y=(sub.group=='mdd').astype(int).to_numpy();score=sub.mean_abs_z.to_numpy()
        fpr,tpr,_=roc_curve(y,score)
        rocrows.extend(dict(comparison=label,fpr=x,tpr=y0) for x,y0 in zip(fpr,tpr))
        aucs.append(dict(comparison=label,n=len(sub),auc=roc_auc_score(y,score),
                         score='mean absolute normative deviation; higher=MDD; no fitted classifier'))
    write_table(table_dir,'roc',rocrows);write_table(table_dir,'roc_summary',aucs)
    clinical_tables(table_dir,all_z['mdd'],net)
    write_json(data_dir/'base_table_summary.json',dict(groups={g:len(test_ids(g)) for g in ['hc','mdd']},
        test_matched_pairs=len(pairs),matched_sites=len(set(p['site'] for p in pairs)),
        negative_n_per_half=len(negative['A']),scope='Descriptive test-only; no mixed-split inference'))


def clinical_tables(table_dir: Path,zmap: dict,net: np.ndarray) -> None:
    table=load_condition_table(str(ROOT/'data'),MDD_MERGED_FILE,'mdd8')
    names=[s.name for s in table.specs]
    rows=[];residuals=[]
    for score in ['HAMD','HAMA']:
        valid=[]
        for sid in test_ids('mdd'):
            ix=table.row_of_id.get(sid,-1)
            cols=[names.index(k) for k in [score,'Age','Sex']]
            if ix<0 or not np.all(table.mask[ix,cols]):continue
            value,age,sex=table.values[ix,cols]
            valid.append((sid,float(value),float(age),float(sex),parse_site(sid)))
        sites=sorted({r[4] for r in valid})
        design=np.array([[1.,r[2],r[3]]+[float(r[4]==s) for s in sites[1:]] for r in valid])
        rank=np.linalg.matrix_rank(design)
        x=np.array([r[1] for r in valid]);rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
        for network in NETWORK_ORDER:
            y=np.array([zmap[r[0]][net==network].mean() for r in valid])
            ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
            r=float(pearsonr(rx,ry).statistic) if len(valid)-rank>=3 and rx.std()>0 and ry.std()>0 else np.nan
            rows.append(dict(network=network,score=score,r=r,n=len(valid),control_rank=rank,
                             mdd_test_total=len(test_ids('mdd')),grouping='AAL116 display network'))
            residuals.extend(dict(subject=t[0],score=score,network=network,resid_score=a,resid_z=b)
                             for t,a,b in zip(valid,rx,ry))
    write_table(table_dir,'clinical_correlations',rows);write_table(table_dir,'clinical_residuals',residuals)


def importance_tables(data_dir: Path,table_dir: Path) -> None:
    importance_rows=[];subject_importance=[]
    net,_=partition();names=labels()['micro命名'].tolist()
    for group in ['hc','mdd']:
        ids=test_ids(group);delta=[]
        for roi in range(116):
            with np.load(data_dir/f'重要性/{group}/roi_{roi+1:03d}.npz') as d:
                check_ids(d['subjects'],group)
                if d['subjects'].tolist()!=ids:raise ValueError('Importance subject order changed')
                delta.append(d['delta_mae'])
        delta=np.stack(delta,axis=1);ci=bootstrap_interval(delta)
        for roi in range(116):
            importance_rows.append(dict(group=group,roi=roi+1,label=names[roi],network=net[roi],
                delta_mae=delta[:,roi].mean(),ci_low=ci[0,roi],ci_high=ci[1,roi]))
        subject_importance.extend(dict(group=group,subject=sid,roi=roi+1,delta_mae=delta[i,roi])
                                  for i,sid in enumerate(ids) for roi in range(116))
    write_table(table_dir,'roi_importance',importance_rows)
    write_table(table_dir,'importance_subjects',subject_importance)


def probe_tables(data_dir: Path,table_dir: Path) -> None:
    if not (table_dir/'roi_importance.csv').exists():importance_tables(data_dir,table_dir)
    history=[];sensitivity=[];gate=[]
    for group in ['hc','mdd']:
        ids=test_ids(group);early=[];late=[]
        for sid in ids:
            with np.load(data_dir/f'扰动/{group}/{sid}.npz') as d:
                if d['sham_max']>1e-6:raise ValueError(f'Sham failed {sid}')
                early.append(d['response_early']);late.append(d['response_late'])
                rel=100*(d['history_mae'].mean(1)/d['baseline_mae'][None]-1)
                history.extend(dict(group=group,subject=sid,block=b+1,segment=seg,relative_change=rel[b,j])
                    for b in range(8) for j,seg in enumerate(['H1_8','H9_16','H17_32']))
                gate.append(dict(group=group,subject=sid,sham_max=float(d['sham_max']),
                                 source_difference=float(d['saved_prediction_max_difference'])))
                sensitivity.extend(dict(group=group,subject=sid,horizon=j+1,response=float(v))
                                   for j,v in enumerate(d['response_time']))
        early,late=np.stack(early),np.stack(late)
        np.savez_compressed(data_dir/f'propagation_{group}.npz',subjects=np.array(ids),early=early,late=late)
    for name,rows in [('history_sensitivity',history),('perturbation_timecourse',sensitivity),('probe_gates',gate)]:
        write_table(table_dir,name,rows)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['base','probe','importance'],required=True)
    ap.add_argument('--data-dir',type=Path,default=DATA);ap.add_argument('--table-dir',type=Path,default=TABLES)
    ns=ap.parse_args()
    {'base':base_tables,'probe':probe_tables,'importance':importance_tables}[ns.stage](ns.data_dir,ns.table_dir)


if __name__=='__main__':main()
