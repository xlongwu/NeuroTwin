"""Fit the fixed static-FC baseline on TRAIN only; score independent TEST only.

This does not train or alter NeuroTwin. No validation/test model selection or
threshold tuning: the original baseline's RidgeClassifier(alpha=1) is fixed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score,roc_curve

from analysis.pathology_tests import static_fc_features
from utils.site_subset import parse_site
from .common import ROOT,CHECKPOINT,DATA,TABLES,SPLITS,test_ids,read_json,write_json,digest


def write_test_roc(output: pd.DataFrame,table_dir: Path) -> None:
    pairs=pd.read_csv(table_dir/'test_pairs.csv');matched_ids=set(pairs.hc)|set(pairs.mdd)
    curves=[];summaries=[];intervals=[];grid=np.linspace(0,1,101)
    for name,sub in [('All test',output),('Matched test',output[output.subject.isin(matched_ids)])]:
        sub=sub.reset_index(drop=True)
        fpr,tpr,_=roc_curve(sub.label,sub.fc_ridge_score)
        curves.extend(dict(comparison=name,fpr=a,tpr=b) for a,b in zip(fpr,tpr))
        rng=np.random.RandomState(2024);boot_auc=[];boot_tpr=[]
        strata=[ix.to_numpy() for _,ix in sub.groupby(['group','site']).groups.items()]
        if name=='Matched test':
            lookup={sid:i for i,sid in enumerate(sub.subject)}
            pair_indices=np.array([[lookup[p.hc],lookup[p.mdd]] for p in pairs.itertuples()])
            pair_strata=[np.flatnonzero(pairs.site.to_numpy()==site) for site in sorted(pairs.site.unique())]
        y=sub.label.to_numpy();score=sub.fc_ridge_score.to_numpy()
        for _ in range(500):
            if name=='Matched test':
                chosen=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in pair_strata])
                take=pair_indices[chosen].ravel()
            else:take=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
            bf,bt,_=roc_curve(y[take],score[take]);boot_tpr.append(np.interp(grid,bf,bt));boot_auc.append(roc_auc_score(y[take],score[take]))
        lo,hi=np.quantile(boot_tpr,[.025,.975],axis=0);auc_ci=np.quantile(boot_auc,[.025,.975])
        intervals.extend(dict(comparison=name,fpr=f,tpr_low=a,tpr_high=b) for f,a,b in zip(grid,lo,hi))
        summaries.append(dict(comparison=name,n=len(sub),auc=roc_auc_score(y,score),ci_low=auc_ci[0],ci_high=auc_ci[1]))
    pd.DataFrame(curves).to_csv(table_dir/'fc_ridge_roc.csv',index=False)
    pd.DataFrame(summaries).to_csv(table_dir/'fc_ridge_roc_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(table_dir/'fc_ridge_roc_ci.csv',index=False)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=DATA);ap.add_argument('--table-dir',type=Path,default=TABLES)
    ns=ap.parse_args();train=[];test=[];xtrain=[];xtest=[]
    for group,y in [('hc',0),('mdd',1)]:
        split=read_json(ROOT/'data'/SPLITS[group]);tr=split['train'];te=test_ids(group)
        if set(tr)&set(te):raise ValueError('Classifier train/test overlap')
        print(f'FC baseline features {group}: train {len(tr)}, test {len(te)}',flush=True)
        xtrain.append(static_fc_features(ROOT/'data',ROOT/CHECKPOINT,group,tr))
        xtest.append(static_fc_features(ROOT/'data',ROOT/CHECKPOINT,group,te))
        train.extend((group,s,y) for s in tr);test.extend((group,s,y) for s in te)
    if {s for _,s,_ in train}&{s for _,s,_ in test}:raise ValueError('Cross-cohort train/test overlap')
    xtr=np.concatenate(xtrain);xte=np.concatenate(xtest)
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all():raise ValueError('Nonfinite static FC')
    classifier=RidgeClassifier(alpha=1.)
    classifier.fit(xtr,np.array([y for _,_,y in train]))
    score=classifier.decision_function(xte)
    output=pd.DataFrame([dict(group=g,subject=s,site=parse_site(s),label=y,fc_ridge_score=float(v))
                         for (g,s,y),v in zip(test,score)])
    output.to_csv(ns.table_dir/'fc_ridge_test_scores.csv',index=False)
    write_test_roc(output,ns.table_dir)
    np.savez_compressed(ns.data_dir/'fc_ridge_train_parameters.npz',coef=classifier.coef_,intercept=classifier.intercept_)
    write_json(ns.data_dir/'fc_ridge_protocol.json',dict(classifier='RidgeClassifier(alpha=1)',fit='train only',
        train_counts={g:sum(a==g for a,_,_ in train) for g in ['hc','mdd']},
        test_counts={g:sum(a==g for a,_,_ in test) for g in ['hc','mdd']},features='First observed 64 TR; 6670 FC upper-triangle edges',
        preprocessing='Same fixed static_fc_features as original baseline; no fitted test transform',
        hyperparameter_selection='None; alpha fixed before test scoring',
        split_hashes={g:digest(ROOT/'data'/SPLITS[g]) for g in ['hc','mdd']},
        checkpoint_sha256=digest(ROOT/CHECKPOINT),test_auc=float(roc_auc_score(output.label,output.fc_ridge_score))))
    print('Independent static FC ridge test AUC:',roc_auc_score(output.label,output.fc_ridge_score),flush=True)


if __name__=='__main__':main()
