"""Paired Temporal PCC experiment using the archived E2 recipe and shared evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from experiments.loss.compare_phase1 import load_arm, validate_alignment, _paired_test
from analysis.next_point_eval import amplitude_quantile_edges, _step_deltas, _step_valid

ROOT = Path(__file__).resolve().parents[2]


def trajectory_pcc(pred, target, mask):
    """Per-anchor ROI-mean temporal PCC with the training validity definition."""
    w = mask[:, None, :].astype(float)
    n = w.sum(-1).clip(1)
    pc = (pred - (pred*w).sum(-1, keepdims=True)/n[..., None])*w
    tc = (target - (target*w).sum(-1, keepdims=True)/n[..., None])*w
    ps, ts = (pc*pc).sum(-1), (tc*tc).sum(-1)
    valid = (w.sum(-1) >= 3) & (ts/n > 1e-8)
    corr = (pc*tc).sum(-1)/(np.sqrt(ps+1e-8)*np.sqrt(ts+1e-8)+1e-8)
    count = valid.sum(-1)
    return np.divide(np.where(valid,corr,0).sum(-1),count,
                     out=np.full(len(pred),np.nan),where=count>0)


def compare(out):
    arms={name:load_arm(name,out/name/'eval','test') for name in ('TP0','TP1')}
    validate_alignment(arms)
    arrays={}
    for name in arms:
        with np.load(out/name/'eval/predictions_test.npz',allow_pickle=False) as z:
            arrays[name]={k:z[k] for k in ('subj_ids','cutoff','context_len','x_last','future','future_mask','ode_pred','cpm_pred')}
        arrays[name]['future']=arrays[name]['future'][...,:8]
        arrays[name]['future_mask']=arrays[name]['future_mask'][:,:8]
    ref=arrays['TP0']
    for key in ('subj_ids','cutoff','context_len','x_last','future','future_mask'):
        if not np.array_equal(ref[key],arrays['TP1'][key]):
            raise ValueError(f'TP0/TP1 have different anchors/targets: {key}')
    edges=amplitude_quantile_edges(ref['future'],ref['future_mask'],ref['x_last'])
    frames={}
    metrics=['MAE','RMSE','PCC_spatial','ODE_MAE','CPM_MAE','ODE_MAE_H8','CPM_MAE_H8',
             'ODE_delta_MAE_step','CPM_delta_MAE_step','ODE_DPCC_H2_8','CPM_DPCC_H2_8']
    for name,arm in arms.items():
        a=arrays[name]; frame=arm['subject'].set_index('subject_id')
        task=pd.DataFrame({'subject_id':a['subj_ids'].astype(str)})
        for head,key in (('ODE','ode_pred'),('CPM','cpm_pred')):
            pred,target=a[key].astype(float),a['future'].astype(float)
            task[f'{head}_Temporal_PCC']=trajectory_pcc(pred,target,a['future_mask'])
            for label,hs in (('H1_2',(1,2)),('H4_8',(4,5,6,7,8))):
                cols=[f'{head}_H{h}_mae' for h in hs]
                frame[f'{head}_{label}_MAE']=frame[cols].mean(axis=1)
            _,dt=_step_deltas(pred,target,a['x_last'])
            valid=np.broadcast_to(_step_valid(a['future_mask'].astype(bool))[:,None,:],target.shape)
            for q in (1,5):
                selected=valid & (np.abs(dt)>=edges[q-1]) & (np.abs(dt)<edges[q])
                n=selected.sum((1,2))
                sums=np.where(selected,np.abs(pred-target),0).sum((1,2))
                task[f'{head}_Q{q}_point_MAE']=np.divide(sums,n,out=np.full(len(pred),np.nan),where=n>0)
        frames[name]=frame.join(task.groupby('subject_id').mean())
    for head in ('ODE','CPM'):
        metrics += [f'{head}_Temporal_PCC',f'{head}_H1_2_MAE',f'{head}_H4_8_MAE',
                    f'{head}_Q1_point_MAE',f'{head}_Q5_point_MAE']
    summary=pd.DataFrame({name:frame[metrics].mean() for name,frame in frames.items()}).T
    dest=out/'compare';dest.mkdir(exist_ok=True)
    summary.to_csv(dest/'summary.csv',index_label='experiment')
    base=frames['TP0']; candidate=frames['TP1'].reindex(base.index)
    rows=[]
    for metric in metrics:
        lower='PCC' not in metric and 'DPCC' not in metric
        result=_paired_test(candidate[metric],base[metric],lower)
        if result:
            rows.append(dict(metric=metric,**result,
                relative_improvement_percent=100*result['mean_improvement']/abs(base[metric].mean())))
    pd.DataFrame(rows).to_csv(dest/'paired_vs_control.csv',index=False)
    for name,frame in frames.items():frame[metrics].to_csv(dest/f'{name}_subject_metrics.csv')
    (dest/'protocol.json').write_text(json.dumps(dict(
        aggregation='per-subject means; trajectory PCC and amplitude strata first normalize each anchor',
        temporal_pcc='future H1-H8 only; >=3 valid points; exclude target variance <=1e-8; constant predictions retained',
        lambda_temporal_pcc=.05,amplitude_edges=edges),indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--out',type=Path,default=Path('results/temporal_pcc'))
    ap.add_argument('--devices',default='cuda:0,cuda:1')
    ap.add_argument('--compare-only',action='store_true')
    ns=ap.parse_args();os.chdir(ROOT)
    if ns.compare_only:
        compare(ns.out);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    devices=ns.devices.split(',')
    if len(devices)!=2:ap.error('--devices requires two device entries')
    source=json.loads(Path('results/loss/L0/training_config.json').read_text())
    ns.out.mkdir(parents=True,exist_ok=True)
    status_path=ns.out/'status.json'
    status_path.write_text(json.dumps(dict(status='running',control='TP0',candidate='TP1',weight=.05),indent=2)+'\n')
    jobs=[]
    for name,device,weight in zip(('TP0','TP1'),devices,(0.,.05)):
        directory=ns.out/name
        if directory.exists():raise FileExistsError(directory)
        directory.mkdir()
        cmd=source['command'].copy();cmd[0]=sys.executable
        overrides=dict(name=name,device=device,checkpoint_dir=str(directory/'checkpoints'),
                       loss_experiment_id=name,lambda_temporal_pcc=weight)
        for key,value in overrides.items():
            flag=f'--{key}'
            if flag in cmd:cmd[cmd.index(flag)+1]=str(value)
            else:cmd.extend((flag,str(value)))
        # Omit the duplicate post-training evaluation; shared final evaluation follows.
        cmd.append('--no_eval_after_train')
        config=dict(experiment=name,status='training',command=cmd,args={**source['args'],**overrides,'eval_after_train':False},
                    initialization_sha256=source['initialization_sha256'],
                    scheduler_sha256=hashlib.sha256(Path('train/optim.py').read_bytes()).hexdigest(),
                    source_configuration='results/loss/L0/training_config.json')
        config_path=directory/'training_config.json'
        config_path.write_text(json.dumps(config,indent=2)+'\n')
        jobs.append((name,device,cmd,config,config_path))
    def run(job):
        name,device,cmd,config,config_path=job
        directory=ns.out/name
        with (directory/'train.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        config['status']='evaluating';config_path.write_text(json.dumps(config,indent=2)+'\n')
        with (directory/'eval.log').open('w') as log:
            subprocess.run([sys.executable,'-m','experiments.evaluate_variant','--ckpt',
                str(directory/'checkpoints'/name/'finetuned_best.pt'),'--out_dir',str(directory/'eval'),
                '--splits','test','--save_arrays','--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
        config['status']='evaluated';config_path.write_text(json.dumps(config,indent=2)+'\n')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
        compare(ns.out)
    except Exception as exc:
        status_path.write_text(json.dumps(dict(status='failed',error=str(exc)),indent=2)+'\n')
        raise
    status_path.write_text(json.dumps(dict(status='evaluated',stage='awaiting_result_analysis'),indent=2)+'\n')


if __name__=='__main__':main()
