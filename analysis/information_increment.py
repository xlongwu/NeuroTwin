"""Test brain-feature readout increment over demographics with identical nested site folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.information_nuisance import demographic_features
from analysis.information_readout import (ALPHAS, LAYERS, adjusted_features, kernel_scores,
                                          kernels, load_features, site_auc)
from analysis.individual_pilot import digest
from train.pathology_readout import site_folds, site_weights


def combined_kernels(features: np.ndarray, meta: dict, train: np.ndarray, valid: np.ndarray) -> dict:
    brain=kernels(adjusted_features(features,meta,train),train,valid)
    demo=kernels(demographic_features(meta,train),train,valid)
    # Equal kernel weights are fixed, not selected using held-out scores.
    return {name:tuple((b+d)/2 for b,d in zip(brain[name],demo[name])) for name in brain}


def nested_increment(features: np.ndarray, meta: dict) -> tuple:
    labels,sites=meta['labels'],meta['sites']
    predictions={name:np.empty(len(labels)) for name in ('linear','rbf')}
    selections=[]
    for fold,(train,valid) in enumerate(site_folds(sites)):
        validation_scores={name:[] for name in predictions}
        for inner_train,inner_valid in site_folds(sites[train]):
            it,iv=train[inner_train],train[inner_valid]
            for name,(k,cross) in combined_kernels(features,meta,it,iv).items():
                pred=kernel_scores(k,cross,labels[it],site_weights(sites[it],labels[it]))
                validation_scores[name].append([site_auc(labels[iv],pred[:,j],sites[iv]).mean() for j in range(len(ALPHAS))])
        for name,(k,cross) in combined_kernels(features,meta,train,valid).items():
            choice=int(np.argmax(np.mean(validation_scores[name],axis=0)))
            predictions[name][valid]=kernel_scores(k,cross,labels[train],site_weights(sites[train],labels[train]),
                                                  [ALPHAS[choice]])[:,0]
            selections.append(dict(fold=fold,reader=name,alpha=ALPHAS[choice]))
    return predictions,selections


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir',type=Path,required=True)
    p.add_argument('--mechanism-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    features,meta,_,_=load_features(a.feature_dir,a.mechanism_dir)
    with np.load(a.out_dir/'人口学参照读出.npz') as z:
        if not np.array_equal(z['subjects'],meta['subjects']):
            raise ValueError('Demographic reference subject order differs')
        baseline={name:site_auc(meta['labels'],z[name],meta['sites']) for name in ('linear','rbf')}
    rng=np.random.default_rng(2024)
    indices=rng.integers(0,len(set(meta['sites'])),(10000,len(set(meta['sites']))))
    result=[]
    for layer in LAYERS:
        predictions,selections=nested_increment(features[layer],meta)
        np.savez_compressed(a.out_dir/f'条件增量_{layer}.npz',**predictions,subjects=np.asarray(meta['subjects'],dtype=str),
                            selections=np.array(json.dumps(selections)))
        for reader,pred in predictions.items():
            auc=site_auc(meta['labels'],pred,meta['sites'])
            diff=auc-baseline[reader]
            ci=np.quantile(diff[indices].mean(1),[.025,.975])
            result.append(dict(layer=layer,reader=reader,demographic_auc=float(baseline[reader].mean()),
                               combined_auc=float(auc.mean()),delta_auc=float(diff.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
        print(layer,[(r['reader'],round(r['delta_auc'],4)) for r in result[-2:]],flush=True)
    pd.DataFrame(result).to_csv(a.out_dir/'条件增量汇总.csv',index=False)
    (a.out_dir/'条件增量协议.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        layers=list(LAYERS),kernel='fixed equal average of brain and demographic kernels; same two readers and alpha grid',
        scope='post-result targeted check motivated by strong demographic baseline; original features unchanged',
        limitation='conditional descriptive intervals, no multiplicity correction; negative fixed-kernel increment does not prove no conditional information'),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
