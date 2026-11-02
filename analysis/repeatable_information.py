"""Frozen-model incremental information and cross-window transfer at held-out sites."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.h18_data import calibrate_raw
from analysis.individual_pilot import digest
from analysis.information_nuisance import demographic_features
from analysis.information_readout import ALPHAS, adjusted_features, kernel_scores, kernels, site_auc
from analysis.normative_deviation import load_demo_matrix
from analysis.shared_mechanism import GROUP_FILES, ROOT, outgoing_response
from inference_twin import TwinSession, load_raw_series
from train.pathology_readout import site_folds, site_weights

LAYERS=('input_fc','encoded_state','effective_graph','response7')
MIXES=(0.,.5)


def load_meta(mechanism_dir: Path) -> dict:
    records=pd.read_csv(mechanism_dir/'被试与质控.csv')
    values=[];masks=[];ids=[]
    for group,(split,table) in GROUP_FILES.items():
        group_ids=json.loads((ROOT/'data'/split).read_text())['test']
        v,m,idx=load_demo_matrix(ROOT/'data',table,group_ids)
        values.append(v);masks.append(m);ids.extend(group_ids)
    if records.subject.tolist()!=ids:
        raise ValueError('Subject ordering differs from original test cohorts')
    values=np.concatenate(values);mask=np.concatenate(masks)
    complete=(mask[:,idx['Age']]!=0)&(mask[:,idx['Sex']]!=0)
    shared=set(records.loc[complete&(records.group=='hc'),'site'])&set(records.loc[complete&(records.group=='mdd'),'site'])
    keep=complete&records.site.isin(shared).to_numpy()
    return dict(subjects=records.subject.to_numpy(dtype=str)[keep],groups=records.group.to_numpy()[keep],
                labels=(records.group=='mdd').to_numpy(dtype=int)[keep],sites=records.site.to_numpy()[keep],
                values=values[keep],mask=mask[keep],idx=idx)


def extract_windows(meta: dict, mechanism_dir: Path, dest: Path, device: str, checkpoint: Path) -> dict:
    target=dest/'逐窗特征.npz'
    protocol=dict(checkpoint_sha256=digest(checkpoint),subjects=meta['subjects'].tolist(),layers=LAYERS,
                  source_sha256=digest(Path(__file__)),device=device,
                  model_source_sha256={f:digest(ROOT/f) for f in ('models/tfm.py','models/tfm_dynamics.py','models/graph_prior.py')},
                  conditions={g:digest(ROOT/'data'/spec[1]) for g,spec in GROUP_FILES.items()})
    manifest=dest/'逐窗来源.json'
    if target.exists():
        if json.loads(manifest.read_text())!=json.loads(json.dumps(protocol)):
            raise ValueError('Cached per-window feature protocol differs')
        with np.load(target) as z:
            return {layer:z[layer].copy() for layer in LAYERS}
    atlas=pd.read_excel(ROOT/'data/AAL116.xlsx')['对应网络'].to_numpy()
    upper=np.triu_indices(116,1)
    rows={layer:[] for layer in LAYERS};max_difference=0.;raw_hashes={}
    checkpoint_hash=protocol['checkpoint_sha256']
    with torch.inference_mode():
        for group,(_,table) in GROUP_FILES.items():
            source=mechanism_dir/'缓存'/group
            m=json.loads((source/'数据清单.json').read_text())
            if m['checkpoint_sha256']!=checkpoint_hash:
                raise ValueError('Must use original shared HC checkpoint')
            session=TwinSession(str(checkpoint),device=device,condition_file=table)
            model=session.model
            if session.bundle['config']['group']!='hc3' or session.calibration_length!=32:
                raise ValueError('Requires current HC model and calibration contract')
            ids=meta['subjects'][meta['groups']==group]
            for number,sid in enumerate(ids,1):
                raw_path=ROOT/f'data/Raw/{group.upper()}/ROISignals_{sid}.mat'
                raw_hashes[sid]=digest(raw_path)
                with np.load(source/f'{sid}.npz') as z:
                    if raw_hashes[sid]!=str(z['raw_sha256']):
                        raise ValueError(f'{sid}: changed raw input')
                    cutoff=z['cutoff'].copy();expected=z['baseline'].copy()
                    response=abs(z['combined']-z['baseline'][:,None,:])
                if cutoff[1]-cutoff[0]<64:
                    raise ValueError('Overlapping history windows')
                series=calibrate_raw(load_raw_series(raw_path,116),32)
                x=torch.from_numpy(np.stack([series[t-63:t+1].T for t in cutoff]).copy())[:,:,None,:].to(device)
                cal=torch.from_numpy(series[:32].T.copy())[None,:,None,:].to(device)
                profile=model.calibrate_profile(cal);condition=session.condition_for(sid)
                captured={}
                def hook(_module,inputs):
                    captured['state']=inputs[0].detach()
                handle=model.one_step_head.register_forward_pre_hook(hook)
                try:
                    _,aux=model(x,sc_matrix=None,profile=profile.expand(2,-1),
                                condition={k:v.expand(2,*v.shape[1:]) for k,v in condition.items()},
                                dynamics_horizon=1,with_aux=False)
                finally:
                    handle.remove()
                discrepancy=float(np.max(abs(aux['dynamics_pred'][:,:,0].cpu().numpy()-expected)))
                if discrepancy>1e-5:
                    raise ValueError(f'{sid}: baseline changed')
                max_difference=max(max_difference,discrepancy)
                rows['input_fc'].append(model._history_fc_signed(x[:,:,0,:]).cpu().numpy()[:,upper[0],upper[1]])
                rows['encoded_state'].append(captured['state'].cpu().numpy().reshape(2,-1))
                rows['effective_graph'].append(aux['sc_prior']['A_eff'].cpu().numpy()[:,upper[0],upper[1]])
                rows['response7'].append(np.array([outgoing_response(r,atlas) for r in response]))
                if number%50==0 or number==len(ids):
                    print(f'extract {group}: {number}/{len(ids)}',flush=True)
    features={name:np.array(value) for name,value in rows.items()}
    if any(x.ndim!=3 or x.shape[:2]!=(len(meta['labels']),2) or not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Invalid per-window features')
    np.savez_compressed(target,**features,subjects=meta['subjects'])
    manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    (dest/'逐窗检查.json').write_text(json.dumps(dict(subjects=len(meta['labels']),max_baseline_abs=max_difference,
        raw_sha256=raw_hashes),ensure_ascii=False,indent=2)+'\n')
    return features


def transfer_kernels(windows: np.ndarray, meta: dict, train: np.ndarray, valid: np.ndarray,
                     source: int, include_target: bool) -> tuple:
    """Fit reference only to source-window training HC; apply unchanged to target windows."""
    n=len(meta['labels'])
    expanded={**meta,**{key:np.concatenate((meta[key],meta[key])) for key in ('labels','sites','values','mask')}}
    features=np.concatenate((windows[:,source],windows[:,1-source]))
    transformed=adjusted_features(features,expanded,train)
    test=np.concatenate((valid,n+valid)) if include_target else valid
    brain=kernels(transformed,train,test)
    demo=np.tile(demographic_features(meta,train),(2,1))
    nuisance=kernels(demo,train,test)
    return brain,nuisance


def nested_transfer(windows: np.ndarray, meta: dict, source: int) -> tuple:
    labels,sites=meta['labels'],meta['sites'];n=len(labels)
    predictions={reader:{kind:np.empty((n,2)) for kind in ('baseline','joint')} for reader in ('linear','rbf')}
    selections=[]
    for fold,(train,valid) in enumerate(site_folds(sites)):
        inner={reader:[] for reader in predictions}
        for it0,iv0 in site_folds(sites[train]):
            it,iv=train[it0],train[iv0]
            brain,demo=transfer_kernels(windows,meta,it,iv,source,False)
            for reader in predictions:
                scores=[]
                for mix in MIXES:
                    k,cross=[mix*b+(1-mix)*d for b,d in zip(brain[reader],demo[reader])]
                    p=kernel_scores(k,cross,labels[it],site_weights(sites[it],labels[it]))
                    scores.append([site_auc(labels[iv],p[:,j],sites[iv]).mean() for j in range(len(ALPHAS))])
                inner[reader].append(scores)
        brain,demo=transfer_kernels(windows,meta,train,valid,source,True)
        for reader in predictions:
            means=np.mean(inner[reader],axis=0)
            mix_index,alpha_index=np.unravel_index(np.argmax(means),means.shape)
            base_index=int(np.argmax(means[0]))
            mix=MIXES[mix_index]
            k,cross=[mix*b+(1-mix)*d for b,d in zip(brain[reader],demo[reader])]
            weights=site_weights(sites[train],labels[train])
            joint=kernel_scores(k,cross,labels[train],weights,[ALPHAS[alpha_index]])[:,0]
            baseline=kernel_scores(*demo[reader],labels[train],weights,[ALPHAS[base_index]])[:,0]
            # First cross block = source window; second = other window. Restore chronological columns.
            for kind,p in (('baseline',baseline),('joint',joint)):
                predictions[reader][kind][valid,source]=p[:len(valid)]
                predictions[reader][kind][valid,1-source]=p[len(valid):]
            selections.append(dict(fold=fold,reader=reader,source=source,mix=mix,alpha=ALPHAS[alpha_index],
                                   baseline_alpha=ALPHAS[base_index],train_sites=sorted(set(sites[train])),
                                   valid_sites=sorted(set(sites[valid]))))
    return predictions,selections


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mechanism-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/neurotwin_pretrain/base_best.pt')
    p.add_argument('--device',default='cpu')
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    meta=load_meta(a.mechanism_dir)
    protocol=dict(source_sha256=digest(Path(__file__)),layers=LAYERS,readers=['linear','rbf'],mixes=MIXES,
                  alphas=ALPHAS,outer_inner='three outer and three inner site folds, same across layers/windows',
                  selection='source-window inner folds only; mixture zero includes baseline; locked model transfers to other window',
                  uncertainty='10000 paired site bootstrap of fixed OOF predictions; descriptive only, no CV refits or multiplicity correction',
                  limits=['same previously explored test cohort; not independent confirmation',
                          'windows are disjoint but same scan, shared profile and calibration',
                          'base HC model saw other subjects from these sites during pretraining',
                          'added reader increment is conditional predictive evidence, not neural causality'])
    manifest=a.out_dir/'分析协议.json'
    if manifest.exists() and json.loads(manifest.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Protocol changed; choose a new result directory')
    manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    features=extract_windows(meta,a.mechanism_dir,a.out_dir,a.device,a.checkpoint)
    rng=np.random.default_rng(2024);site_names=sorted(set(meta['sites']))
    indices=rng.integers(0,len(site_names),(10000,len(site_names)))
    rows=[];site_rows=[];selection_rows=[];predictions_to_save={}
    for layer in LAYERS:
        for source in (0,1):
            predictions,selections=nested_transfer(features[layer],meta,source)
            selection_rows.extend([dict(layer=layer,**r) for r in selections])
            for reader,kind in predictions.items():
                for target in (0,1):
                    baseline=site_auc(meta['labels'],kind['baseline'][:,target],meta['sites'])
                    joint=site_auc(meta['labels'],kind['joint'][:,target],meta['sites'])
                    diff=joint-baseline
                    ci=np.quantile(diff[indices].mean(1),[.025,.975])
                    rows.append(dict(layer=layer,reader=reader,source=source,target=target,n=len(meta['labels']),
                                     n_sites=len(site_names),baseline_auc=float(baseline.mean()),joint_auc=float(joint.mean()),
                                     delta_auc=float(diff.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                                     positive_sites=int(sum(diff>1e-12)),zero_sites=int(sum(abs(diff)<=1e-12))))
                    for site,b,j in zip(site_names,baseline,joint):
                        site_rows.append(dict(layer=layer,reader=reader,source=source,target=target,site=site,
                                              baseline_auc=float(b),joint_auc=float(j),delta_auc=float(j-b)))
                for name,pred in kind.items():
                    predictions_to_save[f'{layer}_{source}_{reader}_{name}']=pred
            print(layer,source,[(r['reader'],round(r['delta_auc'],4)) for r in rows[-4:] if r['target']!=source],flush=True)
    pd.DataFrame(rows).to_csv(a.out_dir/'逐窗增量汇总.csv',index=False)
    pd.DataFrame(site_rows).to_csv(a.out_dir/'逐站点增量.csv',index=False)
    (a.out_dir/'选参与站点划分.json').write_text(json.dumps(selection_rows,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(a.out_dir/'逐窗OOF分数.npz',**predictions_to_save,subjects=meta['subjects'])


if __name__=='__main__':
    main()
