"""Reuse completed D2/D3, compute clinical residuals and bootstrap ROC.
No existing result is replaced. New numbers are descriptive only.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from analysis.paper_figs.common import ROOT, RUN1, matched_pairs
from analysis.pathology_tests import load_z, network_columns, partial_corr, MDD_MERGED_FILE
from utils.dataloader import load_condition_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ns = ap.parse_args()
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    if any(ns.out_dir.iterdir()):
        raise FileExistsError(f'{ns.out_dir} must be empty')
    source = ROOT / 'results/paper_stats/run_20261026'
    zmap = {}
    for group in ('hc','mdd'):
        for split in ('train','val','test'):
            ids,z = load_z(RUN1,group,split)
            zmap.update(zip(ids,z))
    pairs = matched_pairs()
    reuse = pd.read_csv(source / 'unmatched_negative.csv')
    # Independently reconcile descriptive raw deltas against the existing matrices.
    hs = np.stack([zmap[s] for p in pairs for s in [p['hc']]])
    ms = np.stack([zmap[s] for p in pairs for s in [p['mdd']]])
    main_table = pd.read_csv(RUN1 / 'roi_fdr_table.csv')
    np.testing.assert_allclose(ms.mean(0)-hs.mean(0),main_table.delta_z,atol=1e-7)
    bygroup = {}
    for group in ('hc','mdd'):
        bygroup[group] = np.concatenate([load_z(RUN1,group,sp)[1] for sp in ('train','val','test')])
    np.testing.assert_allclose(bygroup['mdd'].mean(0)-bygroup['hc'].mean(0),reuse.delta_unmatched,atol=5.1e-7)
    neg = load_z(RUN1,'hc','val')[1].mean(0)-load_z(RUN1,'hc','test')[1].mean(0)
    np.testing.assert_allclose(neg,reuse.delta_negative,atol=5.1e-7)
    for name in ('unmatched_negative.csv','d2_summary.json','d3_summary.json','fc_ridge_oof.npz'):
        shutil.copy2(source/name,ns.out_dir/name)
    joined = main_table.merge(reuse,on=['roi_index','label'])
    joined.to_csv(ns.out_dir/'roi_three_comparisons.csv',index=False)
    top = []
    for comparison,col in [('Matched','delta_z'),('Unmatched','delta_unmatched'),('HC negative control','delta_negative')]:
        for rank,i in enumerate(np.argsort(-np.abs(joined[col]))[:20],1):
            r=joined.iloc[i]
            top.append(dict(comparison=comparison,rank=rank,roi_index=r.roi_index,label=r.label,delta=r[col]))
    pd.DataFrame(top).to_csv(ns.out_dir/'top_roi_changes.csv',index=False)
    # Reuse scored OOF predictions; bootstrap paired subjects within site.
    d=np.load(source/'fc_ridge_oof.npz')
    y,oof,sites,subject=[d[k] for k in ('y','oof','site','subject')]
    auc=roc_auc_score(y,oof)
    assert abs(auc-json.loads((RUN1/'group_tests.json').read_text())['static_fc_ridge']['auc'])<1e-12
    index={s:i for i,s in enumerate(subject)}
    pairidx=np.array([[index[p['hc']],index[p['mdd']]] for p in pairs])
    psite=np.array([p['site'] for p in pairs])
    rng=np.random.RandomState(2024)
    grid=np.linspace(0,1,101)
    curves,aucs=[],[]
    for _ in range(500):
        take=np.concatenate([rng.choice(np.flatnonzero(psite==s),size=(psite==s).sum(),replace=True) for s in np.unique(psite)])
        ix=pairidx[take].ravel()
        fpr,tpr,_=roc_curve(y[ix],oof[ix])
        curves.append(np.interp(grid,fpr,tpr)); aucs.append(roc_auc_score(y[ix],oof[ix]))
    lo,hi=np.percentile(curves,[2.5,97.5],axis=0)
    fpr,tpr,_=roc_curve(y,oof)
    pd.DataFrame({'fpr':fpr,'tpr':tpr}).to_csv(ns.out_dir/'roc_observed.csv',index=False)
    pd.DataFrame({'fpr':grid,'tpr_lo':lo,'tpr_hi':hi}).to_csv(ns.out_dir/'roc_bootstrap.csv',index=False)
    # Clinical residuals: same subjects, variables and partial_corr as run1.
    table=load_condition_table(str(ROOT/'data'),MDD_MERGED_FILE,'mdd8')
    names=[s.name for s in table.specs]
    site_names=sorted(set(p['site'] for p in pairs))
    raw=[]
    for p in pairs:
        sid=p['mdd']; row=table.row_of_id.get(sid,-1); demo=p['mdd_demo']
        if row<0 or demo['Age'] is None or demo['Sex'] is None:
            continue
        controls=[float(demo['Age']),float(demo['Sex'])]+[float(p['site']==s) for s in site_names]
        for score in ('HAMD','HAMA'):
            j=names.index(score)
            if not table.mask[row,j]:
                continue
            for net,cols in network_columns().items():
                if net!='Other' and len(cols):
                    raw.append(dict(subject=sid,network=net,score=score,score_value=float(table.values[row,j]),net_z=float(zmap[sid][cols].mean()),controls=controls))
    residuals=[]
    reference=pd.read_csv(RUN1/'clinical_partial_corr.csv')
    for (net,score),sub in pd.DataFrame(raw).groupby(['network','score']):
        C=np.array(sub.controls.tolist()); D=np.column_stack([np.ones(len(sub)),C])
        x,y0=sub.score_value.to_numpy(),sub.net_z.to_numpy()
        rx=x-D@np.linalg.lstsq(D,x,rcond=None)[0]; ry=y0-D@np.linalg.lstsq(D,y0,rcond=None)[0]
        stat=partial_corr(y0,x,C)
        ref=reference[(reference.network==net)&(reference.score==score)].iloc[0]
        assert len(sub)==ref.n and abs(stat['r']-ref.partial_r)<1e-10
        for i,(_,r) in enumerate(sub.iterrows()):
            residuals.append(dict(subject=r.subject,network=net,score=score,resid_score=rx[i],resid_net_z=ry[i],partial_r=stat['r'],p=stat['p']))
    pd.DataFrame(residuals).to_csv(ns.out_dir/'clinical_residuals.csv',index=False)
    # Missingness is descriptive only, not a mechanism analysis or new inference.
    mh=set(p['mdd'] for p in pairs)
    available=[]
    for split in ('train','val','test'):
        ids,_=load_z(RUN1,'mdd',split)
        for sid in ids:
            row=table.row_of_id.get(sid,-1)
            available.append(dict(subject=sid,matched=sid in mh,hama_available=bool(row>=0 and table.mask[row,names.index('HAMA')])))
    pd.DataFrame(available).to_csv(ns.out_dir/'hama_availability.csv',index=False)
    summary=dict(auc=float(auc),auc_ci95=np.percentile(aucs,[2.5,97.5]).tolist(),n_boot=500,
                 bootstrap='resample matched pairs within site; fixed OOF scores, no model refit; pointwise ROC interval',
                 clinical_rows=len(residuals),hama_all=sum(r['hama_available'] for r in available),n_mdd=len(available),
                 hama_matched=sum(r['hama_available'] and r['matched'] for r in available),n_matched=len(pairs),
                 verification='D2 deltas, D3 AUC, every clinical partial correlation and n reconciled to sources',
                 d2_note='Deltas are unadjusted raw means; reused unmatched/negative p-values test site-weighted differences, not these raw means. These p-values are not used for new inference.')
    (ns.out_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
