"""Read-only, full-test inference. No train/val subjects, optimization or fitting.

Stages write disjoint files and can run in CPU partitions. Probe settings retain
the registered previous analysis, now applied to every fixed test subject.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from inference_twin import TwinSession,load_raw_series
from analysis.condition_usage import zero_condition,film_norms
from analysis.pathology_tests import HC_MERGED_FILE,MDD_MERGED_FILE
from analysis.paper_figs.run_20261026.t2_infer import generation_metrics
from analysis.paper_figs.run_20261026.complete_infer import probe_subject
from models.common import prepare_sc_matrix
from .common import ROOT,CHECKPOINT,DATA,test_ids,sample_path,read_json,digest


@torch.no_grad()
def baseline(session: TwinSession,group: str,sid: str) -> dict:
    raw=load_raw_series(ROOT/f'data/Raw/{group.upper()}/ROISignals_{sid}.mat',116)
    session.calibrate_scan(raw)
    series=session.series.astype(np.float32)
    if len(series)<96:
        raise ValueError(f'{group}/{sid}: insufficient K64/H32 history')
    x=torch.from_numpy(series[:64].T.copy())[None,:,None,:]
    cal=x[...,:32].clone()
    profile=session.model.calibrate_profile(cal)
    condition=session.condition_for(sid)
    _,aux=session.model(x,sc_matrix=None,condition=condition,profile=profile,
                        calibration_bold=cal,dynamics_horizon=32)
    _,zero=session.model(x,sc_matrix=None,condition=zero_condition(condition),profile=profile,
                         calibration_bold=cal,dynamics_horizon=32)
    pred=aux['dynamics_pred'][0].cpu().numpy().T
    std=aux['dynamics_std'][0].cpu().numpy().T
    true=series[64:96]
    pzero=zero['dynamics_pred'][0].cpu().numpy().T
    prior=session.model._history_fc_adj(x[:,:,0,:])
    _,prior=prepare_sc_matrix(prior,x)
    graph=aux['sc_prior']
    condition_embedding=session.model._prepare_condition(None,condition,None)
    norms=film_norms(session.model,condition_embedding)
    data=dict(history=series[:64],true=true,pred=pred,std=std,
        calibration=cal[0].cpu().numpy(),profile=profile[0].cpu().numpy(),
        pred_zero=pzero,A_eff=graph['A_eff'][0].cpu().numpy(),
        A_func=graph['A_func'][0].cpu().numpy(),prior_norm=prior[0].cpu().numpy(),
        lambda_value=graph['lambda'].cpu().numpy().reshape(-1),
        **generation_metrics(pred,true),**{k:np.array(v) for k,v in norms.items()})
    data.update({k:v[0].cpu().numpy() for k,v in condition.items()})
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError(f'Nonfinite base inference: {sid}')
    return data


@torch.no_grad()
def prepare_importance(model,group: str,data_dir: Path,batch_size: int) -> tuple:
    """Cache the identical unperturbed batch result once for every source ROI."""
    ids=test_ids(group)
    all_data=[]
    for sid in ids:
        with np.load(sample_path(group,sid,data_dir)) as d:
            all_data.append({k:d[k].copy() for k in ['history','true','pred','calibration','profile','cond_z','cond_code','cond_mask']})
    batches=[]
    for lo in range(0,len(ids),batch_size):
        ss=all_data[lo:lo+batch_size];n=len(ss)
        original=torch.from_numpy(np.stack([s['history'].T for s in ss]))[:,:,None,:]
        cal=torch.from_numpy(np.stack([s['calibration'] for s in ss]))
        profile=torch.from_numpy(np.stack([s['profile'] for s in ss]))
        cond={k:torch.from_numpy(np.stack([s[k] for s in ss])) for k in ('cond_z','cond_code','cond_mask')}
        true=torch.from_numpy(np.stack([s['true'][0] for s in ss]))
        _,aux=model(original,sc_matrix=None,calibration_bold=cal,profile=profile,
                    condition=cond,dynamics_horizon=1)
        base=(aux['dynamics_pred'][:,:,0]-true).abs().mean(1).cpu().numpy()
        batches.append((lo,original,cal,profile,cond,true,base))
    return all_data,batches


@torch.no_grad()
def importance(model,group: str,roi: int,prepared: tuple) -> dict:
    ids=test_ids(group);all_data,batches=prepared
    donor=np.random.RandomState(2024+roi).permutation(len(ids));deltas=[]
    for lo,original,cal,profile,cond,true,base in batches:
        changed=original.clone()
        for j in range(len(original)):
            changed[j,roi,0]=torch.from_numpy(all_data[donor[lo+j]]['history'][:,roi])
        _,aux=model(changed,sc_matrix=None,calibration_bold=cal,profile=profile,
                    condition=cond,dynamics_horizon=1)
        error=(aux['dynamics_pred'][:,:,0]-true).abs().mean(1).cpu().numpy()
        deltas.extend(error-base)
    return dict(subjects=np.array(ids),delta_mae=np.array(deltas),donor_index=donor,roi=np.array(roi+1))


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['base','probe','importance'],required=True)
    ap.add_argument('--data-dir',type=Path,default=DATA)
    ap.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    ap.add_argument('--partition',type=int,default=0)
    ap.add_argument('--partitions',type=int,default=1)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=16)
    ns=ap.parse_args()
    if not 0<=ns.partition<ns.partitions:
        raise ValueError('Invalid partition')
    protocol=read_json(ns.data_dir/'protocol.json')
    if digest(ns.checkpoint)!=protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    torch.set_num_threads(ns.threads);torch.manual_seed(2024);np.random.seed(2024)
    # Exact historical registration permutations, shared across both cohorts.
    rng=np.random.RandomState(2024)
    perms=np.array([[rng.permutation(8) for _ in range(3)] for _ in range(8)])
    for group,merged in [('hc',HC_MERGED_FILE),('mdd',MDD_MERGED_FILE)]:
        session=TwinSession(str(ns.checkpoint),device='cpu',condition_file=merged)
        if ns.stage=='importance':
            prepared=prepare_importance(session.model,group,ns.data_dir,ns.batch_size)
            for roi in range(ns.partition,116,ns.partitions):
                target=ns.data_dir/f'重要性/{group}/roi_{roi+1:03d}.npz'
                if target.exists():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                result=importance(session.model,group,roi,prepared)
                np.savez_compressed(target,**result)
                print(f'{group} importance ROI {roi+1}/116',flush=True)
            continue
        ids=test_ids(group)[ns.partition::ns.partitions]
        for index,sid in enumerate(ids):
            target=(sample_path(group,sid,ns.data_dir) if ns.stage=='base'
                    else ns.data_dir/f'扰动/{group}/{sid}.npz')
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            start=time.monotonic()
            if ns.stage=='base':
                result=baseline(session,group,sid)
            else:
                with np.load(sample_path(group,sid,ns.data_dir)) as d:
                    sample={k:d[k].copy() for k in d.files}
                probe=probe_subject(session.model,sample,perms,.1,ns.batch_size)
                response=np.abs(probe.pop('response')).mean(0)
                result={**probe,'response_early':response[:,:,:8].mean(-1),
                    'response_late':response[:,:,16:].mean(-1),
                    'response_time':response[~np.eye(116,dtype=bool)].mean(0)}
            np.savez_compressed(target,**result)
            print(f'{ns.stage} {group} {index+1}/{len(ids)} {sid} {time.monotonic()-start:.1f}s',flush=True)
    print(f'Finished {ns.stage} partition {ns.partition}',flush=True)


if __name__=='__main__':main()
