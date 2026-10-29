"""HC-train-only split-conformal reference; separate from the all-train z model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.normative_deviation import load_roi_mae, load_demo_matrix, fit_normative, score_normative

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/H1-8分析_20261028/02_病理分析'


def main():
    rolling=OUT/'滚动缓存_健康对照'
    subjects,mae=load_roi_mae(rolling,'hc','train')
    expected=json.loads((ROOT/'data/subject_split_pretrain.json').read_text())['train']
    if len(subjects)!=len(expected) or set(subjects)!=set(expected):raise ValueError('HC train 不完整')
    values,mask,idx=load_demo_matrix(ROOT/'data','Rest-meta-MDD-V1V2-Merged-HC.xlsx',subjects)
    order=np.argsort(np.asarray(subjects));rng=np.random.RandomState(2024);order=rng.permutation(order)
    n_cal=int(np.ceil(.2*len(order)));cal=order[:n_cal];train=order[n_cal:]
    fit=fit_normative(mae[train],values[train],mask[train],idx)
    z=score_normative(mae[cal],values[cal],mask[cal],idx,fit)
    residual=np.abs(z)*fit['sigma_roi'][None,:]
    rank=int(np.ceil((n_cal+1)*.9))
    if rank>n_cal:raise ValueError('校准集太小，无法得到所需分位数')
    q=np.partition(residual,rank-1,axis=0)[rank-1]
    np.savez_compressed(OUT/'健康参照_训练分割校准_20261028.npz',W=fit['W'],sigma_roi=fit['sigma_roi'],
                        conformal_q=q,design_layout=json.dumps(fit['design_layout']),demo=json.dumps(fit['demo']),
                        fit_subjects=np.asarray(subjects)[train],calibration_subjects=np.asarray(subjects)[cal],
                        alpha=.1,seed=2024)
    pd.DataFrame({'roi':np.arange(1,117),'absolute_residual_quantile':q,'n_fit':len(train),
                  'n_calibration':n_cal,'rank':rank}).to_csv(OUT/'表33_健康参照_训练分割_校准分位.csv',index=False)
    report={'split':'HC train only; shuffled sorted IDs','seed':2024,'fit_n':len(train),'calibration_n':n_cal,
            'alpha':.1,'rank_1_based':rank,'n_rois':116,
            'interpretation':'absolute residual split conformal; conditional on exchangeability; no test coverage evaluated',
            'separate_from':'all-HC-train z reference used for existing group tests'}
    (OUT/'原始_分割校准_20261028.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
