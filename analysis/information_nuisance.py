"""Supplementary demographic-only baseline with the same nested site CV and kernel probes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.information_readout import ALPHAS, kernel_scores, kernels, load_features, site_auc
from utils.condition_schema import file_sha256 as digest
from train.pathology_readout import site_folds, site_weights


def demographic_features(meta: dict, train: np.ndarray) -> np.ndarray:
    """Fit imputation/scales on training rows only; never residualize the nuisance baseline itself."""
    values, mask = meta['values'],meta['mask'].astype(bool)
    mean=np.array([values[train,j][mask[train,j]].mean() for j in range(values.shape[1])])
    filled=np.where(mask,values,mean)
    scale=filled[train].std(0)
    x=(filled-mean)/np.maximum(scale,1e-6)
    return np.column_stack((x,mask.astype(float)))/np.sqrt(6.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir',type=Path,required=True)
    p.add_argument('--mechanism-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    _,meta,_,_=load_features(a.feature_dir,a.mechanism_dir)
    labels,sites=meta['labels'],meta['sites']
    predictions={name:np.empty(len(labels)) for name in ('linear','rbf')}
    selections=[]
    for fold,(train,valid) in enumerate(site_folds(sites)):
        validation_scores={name:[] for name in predictions}
        for inner_train,inner_valid in site_folds(sites[train]):
            it,iv=train[inner_train],train[inner_valid]
            ks=kernels(demographic_features(meta,it),it,iv)
            for name,(k,cross) in ks.items():
                pred=kernel_scores(k,cross,labels[it],site_weights(sites[it],labels[it]))
                validation_scores[name].append([site_auc(labels[iv],pred[:,j],sites[iv]).mean() for j in range(len(ALPHAS))])
        ks=kernels(demographic_features(meta,train),train,valid)
        for name,(k,cross) in ks.items():
            choice=int(np.argmax(np.mean(validation_scores[name],axis=0)))
            predictions[name][valid]=kernel_scores(k,cross,labels[train],site_weights(sites[train],labels[train]),
                                                  [ALPHAS[choice]])[:,0]
            selections.append(dict(fold=fold,reader=name,alpha=ALPHAS[choice]))
    rng=np.random.default_rng(2024)
    site_names=sorted(set(sites))
    indices=rng.integers(0,len(site_names),(10000,len(site_names)))
    records=[];differences=[]
    for name,pred in predictions.items():
        auc=site_auc(labels,pred,sites)
        ci=np.quantile(auc[indices].mean(1),[.025,.975])
        records.append(dict(reader=name,macro_site_auc=float(auc.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
        for layer in ('normalized_history','encoded_state','learned_graph','effective_graph','response7','response_full'):
            with np.load(a.out_dir/f'读出_{layer}.npz') as z:
                other=site_auc(labels,z[name],sites)
            diff=other-auc
            interval=np.quantile(diff[indices].mean(1),[.025,.975])
            differences.append(dict(layer=layer,reader=name,delta_vs_demographic=float(diff.mean()),
                                    ci_low=float(interval[0]),ci_high=float(interval[1])))
    pd.DataFrame(records).to_csv(a.out_dir/'人口学参照.csv',index=False)
    pd.DataFrame(differences).to_csv(a.out_dir/'人口学参照比较.csv',index=False)
    np.savez_compressed(a.out_dir/'人口学参照读出.npz',**predictions,subjects=np.asarray(meta['subjects'],dtype=str))
    (a.out_dir/'人口学参照协议.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        selections=selections,fields=['Sex','Age','education','three observed masks'],
        purpose='post-result nuisance diagnostic; not a selected brain feature or a causal adjustment test',
        limitation='separate baseline AUC difference is not a test of conditional incremental information'),ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':
    main()
