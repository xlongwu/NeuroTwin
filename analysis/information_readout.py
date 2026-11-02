"""Matched site-held-out readout probes of inputs, frozen states and existing metrics.

All preprocessing and regularization selection are refitted within training folds.
This is exploratory nested CV inside the previously inspected model test cohort.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

from analysis.individual_pilot import digest
from analysis.normative_deviation import load_demo_matrix
from analysis.shared_mechanism import GROUP_FILES, NETWORKS, ROOT
from train.pathology_fingerprint import fit_reference, reference_features
from train.pathology_readout import site_folds, site_weights

ALPHAS = (.01, .1, 1., 10.)
LAYERS = ('input_history', 'input_fc', 'input_descriptors', 'normalized_history',
          'encoded_state', 'advanced_state', 'profile', 'learned_graph', 'effective_graph',
          'prediction', 'prediction_error', 'response7', 'response49', 'response_full')


def adjusted_features(features: np.ndarray, meta: dict, train: np.ndarray) -> np.ndarray:
    """Reuse the existing HC-only population reference; held-out rows never enter fitting."""
    hc = train[meta['labels'][train] == 0]
    group = dict(values=meta['values'], mask=meta['mask'], idx=meta['idx'], sites=meta['sites'],
                 features=features, residual_width=features.shape[1], protocol={})
    reference = fit_reference({**group, **{k:group[k][hc] for k in ('values','mask','sites','features')}})
    return reference_features(reference, group, 'residual')


def kernels(x: np.ndarray, train: np.ndarray, valid: np.ndarray) -> dict:
    a, b = x[train], x[valid]
    linear = a @ a.T
    cross = b @ a.T
    train_sq = np.sum(a*a, axis=1)
    test_sq = np.sum(b*b, axis=1)
    dist_train = np.maximum(train_sq[:,None]+train_sq[None,:]-2*linear, 0.)
    dist_test = np.maximum(test_sq[:,None]+train_sq[None,:]-2*cross, 0.)
    return {'linear':(linear,cross), 'rbf':(np.exp(-.5*dist_train),np.exp(-.5*dist_test))}


def kernel_scores(train_kernel: np.ndarray, cross_kernel: np.ndarray,
                  labels: np.ndarray, weights: np.ndarray, alphas=ALPHAS) -> np.ndarray:
    """Weighted kernel ridge, shared fixed alpha grid for linear and smooth nonlinear probes."""
    root = np.sqrt(weights)
    k = root[:,None]*train_kernel*root[None,:]
    eigenvalue, eigenvector = eigh(k)
    projection = eigenvector.T @ (root*(2*labels-1))
    dual = root[:,None]*(eigenvector @ (projection[:,None]/(np.maximum(eigenvalue,0)[:,None]+np.array(alphas))))
    return cross_kernel @ dual


def site_auc(labels: np.ndarray, scores: np.ndarray, sites: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(labels[sites==site], scores[sites==site])
                     for site in sorted(set(sites)) if len(set(labels[sites==site])) == 2])


def nested_readout(features: np.ndarray, meta: dict) -> tuple:
    """Identical outer/inner site folds at every layer; never orient scores on test labels."""
    labels, sites = meta['labels'], meta['sites']
    predictions = {name:np.empty(len(labels)) for name in ('linear','rbf')}
    selections = []
    fold_ids = np.empty(len(labels), dtype=int)
    for fold, (train, valid) in enumerate(site_folds(sites)):
        validation_scores = {name:[] for name in predictions}
        for inner_train, inner_valid in site_folds(sites[train]):
            it, iv = train[inner_train], train[inner_valid]
            transformed = adjusted_features(features, meta, it)
            ks = kernels(transformed, it, iv)
            weights = site_weights(sites[it], labels[it])
            for name,(k,cross) in ks.items():
                pred = kernel_scores(k,cross,labels[it],weights)
                validation_scores[name].append([site_auc(labels[iv], pred[:,j], sites[iv]).mean()
                                                for j in range(len(ALPHAS))])
        transformed = adjusted_features(features, meta, train)
        ks = kernels(transformed, train, valid)
        for name,(k,cross) in ks.items():
            scores = np.mean(validation_scores[name], axis=0)
            choice = int(np.argmax(scores))
            predictions[name][valid] = kernel_scores(k,cross,labels[train],site_weights(sites[train],labels[train]),
                                                    alphas=[ALPHAS[choice]])[:,0]
            selections.append(dict(fold=fold, reader=name, alpha=ALPHAS[choice],
                                   inner_macro_site_auc=float(scores[choice]),
                                   train_sites=sorted(set(sites[train])), valid_sites=sorted(set(sites[valid]))))
        fold_ids[valid] = fold
    return predictions, selections, fold_ids


def load_features(feature_dir: Path, mechanism_dir: Path) -> tuple:
    subjects = pd.read_csv(mechanism_dir/'被试与质控.csv')
    expected, demo, masks = [], [], []
    feature_rows = {layer:[] for layer in LAYERS}
    manifests = {}
    atlas = pd.read_excel(ROOT/'data/AAL116.xlsx')['对应网络'].to_numpy()
    response_table = pd.read_csv(mechanism_dir/'逐被试网络响应.csv')
    response_table = response_table[response_table.arm=='combined'].pivot(index='subject',columns='network',values='mean')
    upper_meta = None
    max_baseline = 0.
    for group,(split,table) in GROUP_FILES.items():
        ids = json.loads((ROOT/'data'/split).read_text())['test']
        manifests[group] = json.loads((feature_dir/group/'数据清单.json').read_text())
        if manifests[group]['subjects'] != ids or manifests[group]['split_sha256'] != digest(ROOT/'data'/split):
            raise ValueError('Feature split does not match original test manifest')
        if manifests[group]['condition_table_sha256'] != digest(ROOT/'data'/table):
            raise ValueError('Condition table changed since extraction')
        values, mask, idx = load_demo_matrix(ROOT/'data',table,ids)
        if upper_meta is not None and upper_meta != idx:
            raise ValueError('Demographic field layouts differ')
        upper_meta = idx
        demo.append(values); masks.append(mask);expected.extend(ids)
        for sid in ids:
            with np.load(feature_dir/group/f'{sid}.npz') as z:
                for layer in LAYERS[:11]:
                    feature_rows[layer].append(z[layer].copy())
                max_baseline = max(max_baseline,float(z['baseline_max_abs']))
            with np.load(mechanism_dir/'缓存'/group/f'{sid}.npz') as z:
                response = abs(z['combined']-z['baseline'][:,None,:]).mean(0)
                np.fill_diagonal(response,0)
                blocks = np.array([[response[(atlas[:,None]==s)&(atlas[None,:]==t)&~np.eye(116,dtype=bool)].mean()
                                    for t in NETWORKS] for s in NETWORKS])
                feature_rows['response49'].append(blocks.reshape(-1))
                feature_rows['response_full'].append(response[~np.eye(116,dtype=bool)])
                feature_rows['response7'].append(response_table.loc[sid,list(NETWORKS)].to_numpy())
    if manifests['hc']['checkpoint_sha256'] != manifests['mdd']['checkpoint_sha256']:
        raise ValueError('Feature checkpoints differ')
    if subjects.subject.tolist() != expected:
        raise ValueError('Mechanism metadata and extraction ordering differ')
    values, mask = np.concatenate(demo),np.concatenate(masks)
    complete = (mask[:,upper_meta['Age']] != 0) & (mask[:,upper_meta['Sex']] != 0)
    shared = set(subjects.loc[complete & (subjects.group=='hc'),'site']) & set(subjects.loc[complete & (subjects.group=='mdd'),'site'])
    keep = complete & subjects.site.isin(shared).to_numpy()
    meta = dict(values=values[keep],mask=mask[keep],idx=upper_meta,sites=subjects.site.to_numpy()[keep],
                labels=(subjects.group=='mdd').to_numpy(dtype=int)[keep], subjects=subjects.subject.to_numpy()[keep])
    features = {name:np.stack(rows)[keep].astype(np.float64) for name,rows in feature_rows.items()}
    if any(not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Nonfinite probe input')
    return features, meta, manifests, max_baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir',type=Path,required=True)
    p.add_argument('--mechanism-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    features,meta,manifests,max_baseline=load_features(a.feature_dir,a.mechanism_dir)
    protocol=dict(subjects=meta['subjects'].tolist(),checkpoint_sha256=manifests['hc']['checkpoint_sha256'],
                  layers=list(LAYERS),readers=['linear kernel ridge','RBF kernel ridge; gamma=0.5'],alphas=ALPHAS,
                  cv='3 outer site folds, 3 inner site folds; fixed common splits across layers',
                  preprocessing='existing HC-training-fold-only age/age-squared/sex/education/missing-mask/site ridge reference; HC residual scale; divide sqrt(width)',
                  metric='equal-site mean within-site AUC; scores always higher=MDD; all included sites have both labels',
                  weights='equal total weight for each site and diagnosis cell in probe training',
                  uncertainty='10000 paired site bootstrap draws of fixed outer-OOF AUCs; descriptive, does not refit CV models',
                  limitations=['previously explored test cohort; not new independent confirmation',
                               'frozen base model trained on other HC subjects from overlapping sites',
                               'two-window average can discard temporal information',
                               'negative probes do not prove absence of all disease information',
                               'linear and fixed RBF readers only; no exhaustive classifier search',
                               'held-out site nuisance coefficients are unavailable and encoded as zero; within-site AUC avoids pure site score offsets',
                               'bootstrap intervals are conditional and not corrected for all layer/reader comparisons'],
                  source_sha256=digest(Path(__file__)),max_baseline_abs=max_baseline)
    manifest=a.out_dir/'读出协议.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Readout protocol changed; use a new output directory')
    manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    site_names=sorted(set(meta['sites']))
    rng=np.random.default_rng(2024)
    bootstrap=rng.integers(0,len(site_names),(10000,len(site_names)))
    summary=[];site_records=[]
    all_site_aucs={}
    for layer in LAYERS:
        target=a.out_dir/f'读出_{layer}.npz'
        if target.exists():
            with np.load(target) as z:
                predictions={name:z[name].copy() for name in ('linear','rbf')}
                selections=json.loads(str(z['selections']));fold_ids=z['fold']
        else:
            predictions,selections,fold_ids=nested_readout(features[layer],meta)
            np.savez_compressed(target,**predictions,subjects=meta['subjects'],fold=fold_ids,
                                selections=np.array(json.dumps(selections)))
        pd.DataFrame(selections).to_json(a.out_dir/f'选参_{layer}.json',orient='records',force_ascii=False,indent=2)
        for reader,pred in predictions.items():
            auc=site_auc(meta['labels'],pred,meta['sites'])
            ci=np.quantile(auc[bootstrap].mean(axis=1),[.025,.975])
            all_site_aucs[layer,reader]=auc
            summary.append(dict(layer=layer,reader=reader,width=features[layer].shape[1],n=len(pred),n_sites=len(site_names),
                                macro_site_auc=float(auc.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                                pooled_auc=float(roc_auc_score(meta['labels'],pred))))
            for site,value in zip(site_names,auc):
                ix=meta['sites']==site
                site_records.append(dict(layer=layer,reader=reader,site=site,n_hc=int(sum(meta['labels'][ix]==0)),
                                         n_mdd=int(sum(meta['labels'][ix]==1)),auc=float(value)))
        print(layer,[(r['reader'],round(r['macro_site_auc'],4)) for r in summary[-2:]],flush=True)
    pd.DataFrame(summary).to_csv(a.out_dir/'信息读出汇总.csv',index=False)
    pd.DataFrame(site_records).to_csv(a.out_dir/'逐站点读出.csv',index=False)
    contrasts=[]
    pairs=[('input_fc','effective_graph'),('input_fc','encoded_state'),('normalized_history','encoded_state'),
           ('encoded_state','advanced_state'),('encoded_state','prediction'),('encoded_state','prediction_error'),
           ('response_full','response49'),('response49','response7')]
    for first,second in pairs:
        for reader in ('linear','rbf'):
            diff=all_site_aucs[second,reader]-all_site_aucs[first,reader]
            ci=np.quantile(diff[bootstrap].mean(axis=1),[.025,.975])
            contrasts.append(dict(reference=first,comparison=second,reader=reader,
                                  delta_auc=float(diff.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    pd.DataFrame(contrasts).to_csv(a.out_dir/'配对层间比较.csv',index=False)
    pd.DataFrame(dict(subject=meta['subjects'],site=meta['sites'],label=meta['labels'],fold=fold_ids)).to_csv(a.out_dir/'统一被试与划分.csv',index=False)
    lines=['# 冻结模型的信息保留定位','','本次只拟合辅助读出器，主模型结构、训练策略、条件字段和权重均未改变。',
           '',f'统一纳入 {len(meta["labels"])} 人，{len(site_names)} 个站点。现有测试集内部按站点做三折外层、三折内层交叉验证；整个站点不跨折，所有预处理和选参仅使用训练折。',
           '', '| 特征层 | 维度 | 线性读出 AUC（描述区间） | RBF 读出 AUC（描述区间） |','|---|---:|---|---|']
    for layer in LAYERS:
        rows=[r for r in summary if r['layer']==layer]
        lines.append(f'| {layer} | {rows[0]["width"]} | '+ ' | '.join(f'{r["macro_site_auc"]:.3f}（{r["ci_low"]:.3f}–{r["ci_high"]:.3f}）' for r in rows)+' |')
    lines += ['', 'AUC 为各站点内部 AUC 的等权平均，方向预先固定为高分=MDD；未翻转低于 0.5 的分数。区间来自固定 OOF 分数的站点 bootstrap，没有重拟合读出器，也未对全部特征层和读出器比较校正，仅用于描述不确定性。',
              '', '输入和模型表征使用同两个不重叠 K64 窗，逐窗提取后取均值。response_full 是去掉自身目标的完整绝对响应矩阵；response49 是网络块；response7 是原有七网络汇总。prediction_error 为这两个锚点的 ROI 绝对误差，不能等同于密集滚动的完整 z 参照。',
              '', '这些读出器测量可迁移的标签信息，不证明真实疾病机制。负结果不能证明数据没有信息；不同表征的可读性差异也不等价于已确定的结构或训练故障。HC 主模型已在原训练站点预训练，因此本次站点隔离只适用于辅助读出器，不能称整个系统的外部站点验证。',
              '', '来源、限制和预处理见 [读出协议](读出协议.json)，逐站点结果见 [逐站点读出](逐站点读出.csv)，同口径配对差值见 [层间比较](配对层间比较.csv)。']
    (a.out_dir/'定位报告.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
