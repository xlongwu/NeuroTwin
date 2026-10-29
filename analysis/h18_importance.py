"""Fixed-donor ROI history replacement using completed H1–8 probe caches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from inference_twin import TwinSession
from analysis.h18_probes import GROUPS, ROOT, OUT


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--group',choices=GROUPS,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--cache-dir',type=Path,default=OUT)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/H1-8分析_20261028/03_可解释性/重要性缓存')
    ap.add_argument('--limit',type=int,default=0,help='仅限临时目录冒烟')
    ap.add_argument('--roi-limit',type=int,default=0,help='仅限临时目录冒烟')
    ns=ap.parse_args()
    checkpoint, split, condition_file=GROUPS[ns.group]
    ids=json.loads((ROOT/'data'/split).read_text())['test']
    manifest=json.loads((ns.cache_dir/ns.group/'数据清单.json').read_text())
    digest=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()
    if manifest['subjects']!=ids or manifest['checkpoint_sha256']!=digest or manifest['horizon']!=8:
        raise ValueError('H1–8 缓存清单与完整测试集或 checkpoint 不符')
    ids=ids[:ns.limit] if ns.limit else ids
    if len(ids)<2:raise ValueError('固定供体要求至少两名被试')
    samples=[]
    for sid in ids:
        with np.load(ns.cache_dir/ns.group/f'{sid}.npz') as d:
            if float(d['sham_max'])>1e-6:raise ValueError(f'{sid}: sham 未通过')
            samples.append({k:d[k].copy() for k in ('history','calibration','profile','true','cond_z','cond_code','cond_mask')})
    directory=ns.out_dir/ns.group; directory.mkdir(parents=True,exist_ok=True)
    protocol={'checkpoint_sha256':digest,'group':ns.group,'subjects':ids,'anchor':'first K64',
              'outcome':'ODE lag-1 MAE','donor_seed':2024,
              'donor_selection':'seeded random cycle; no self donor',
              'batch_size':ns.batch_size,'target_condition_calibration_profile':'fixed'}
    path=directory/'数据清单.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('重要性缓存协议不一致')
    else:path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    session=TwinSession(str(ROOT/checkpoint),device=ns.device,condition_file=condition_file)
    model=session.model
    batches=[]
    for start in range(0,len(ids),ns.batch_size):
        block=samples[start:start+ns.batch_size]
        x=torch.from_numpy(np.stack([s['history'] for s in block]))[:,:,None,:].to(session.device)
        cal=torch.from_numpy(np.stack([s['calibration'] for s in block])).to(session.device)
        profile=torch.from_numpy(np.stack([s['profile'] for s in block])).to(session.device)
        cond={k:torch.from_numpy(np.stack([s[k] for s in block])).to(session.device) for k in ('cond_z','cond_code','cond_mask')}
        target=torch.from_numpy(np.stack([s['true'][:,0] for s in block])).to(session.device)
        _,aux=model(x,sc_matrix=None,calibration_bold=cal,profile=profile,condition=cond,dynamics_horizon=1)
        base=(aux['dynamics_pred'][:,:,0]-target).abs().mean(1).cpu().numpy()
        batches.append((start,x,cal,profile,cond,target,base))
    order=np.random.RandomState(2024).permutation(len(ids))
    donor=np.empty(len(ids),dtype=int)
    donor[order]=np.roll(order,1)
    for roi in range(ns.roi_limit or 116):
        path=directory/f'roi_{roi+1:03d}.npz'
        if path.exists():continue
        delta=[];batch_means=[]
        for start,x,cal,profile,cond,target,base in batches:
            changed=x.clone()
            donor_history=np.stack([samples[donor[start+i]]['history'][roi] for i in range(len(x))])
            changed[:,roi,0,:]=torch.from_numpy(donor_history).to(session.device)
            _,aux=model(changed,sc_matrix=None,calibration_bold=cal,profile=profile,condition=cond,dynamics_horizon=1)
            change=(aux['dynamics_pred'][:,:,0]-target).abs().mean(1).cpu().numpy()-base
            delta.extend(change);batch_means.append(float(change.mean()))
        np.savez_compressed(path,subjects=np.array(ids),donor_index=donor,delta_mae=np.array(delta),batch_means=np.array(batch_means))
        print(f'{ns.group}: ROI {roi+1}/116',flush=True)


if __name__=='__main__':main()
