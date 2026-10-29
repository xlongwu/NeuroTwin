"""Train-only, site-stratified 5-fold OOF static-FC ridge baseline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from inference_twin import load_raw_series
from analysis.h18_data import calibrate_raw
from utils.site_subset import parse_site

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/H1-8分析_20261028/02_病理分析'


def main():
    if not json.loads((OUT/'原始_健康负控_20261028.json').read_text())['passed_roi_negative_control']:
        raise RuntimeError('HC 负控未通过，停止疾病基线')
    subjects=[];features=[]
    upper=np.triu_indices(116,k=1)
    for group,split in [('hc','subject_split_pretrain.json'),('mdd','subject_split_finetune.json')]:
        ids=json.loads((ROOT/'data'/split).read_text())['train']
        for sid in ids:
            raw=load_raw_series(ROOT/f'data/Raw/{group.upper()}/ROISignals_{sid}.mat',116)
            h=calibrate_raw(raw)[:64].T.astype(np.float64)
            h-=h.mean(axis=1,keepdims=True)
            scale=np.sqrt((h*h).sum(axis=1,keepdims=True))
            if np.any(scale<=1e-12):raise ValueError(f'{sid}: 有常量 ROI，FC 未定义')
            h/=scale
            features.append((h@h.T)[upper].astype(np.float32))
            subjects.append({'subject':sid,'group':group,'site':parse_site(sid),'label':int(group=='mdd')})
    data=pd.DataFrame(subjects); X=np.stack(features); y=data.label.to_numpy()
    rng=np.random.RandomState(2024);fold=np.zeros(len(data),dtype=int)
    for (_, _),idx in data.groupby(['site','group']).groups.items():
        ix=np.array(list(idx));rng.shuffle(ix);fold[ix]=np.arange(len(ix))%5
    pred=np.empty(len(data),dtype=float)
    for k in range(5):
        train=fold!=k;test=fold==k
        model=RidgeClassifier(alpha=1.0,solver='lsqr')
        model.fit(X[train],y[train]);pred[test]=model.decision_function(X[test])
        print(f'fold {k+1}/5 n_train={train.sum()} n_oof={test.sum()}',flush=True)
    auc=float(roc_auc_score(y,pred))
    data['fold']=fold;data['oof_score']=pred
    data.to_csv(OUT/'表24_静态FC岭回归_训练集_逐被试OOF.csv',index=False)
    report={'analysis':'static FC ridge','split':'train only','n_hc':int((y==0).sum()),
            'n_mdd':int((y==1).sum()),'n_features':int(X.shape[1]),'folds':5,'seed':2024,
            'fold_assignment':'within site and group, seeded shuffled round-robin',
            'alpha':1.0,'solver':'lsqr','oof_auc':auc,
            'limitation':'train-only OOF; no independent test-set AUC; exploratory baseline, not model selection'}
    (OUT/'原始_静态FC岭回归_20261028.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
