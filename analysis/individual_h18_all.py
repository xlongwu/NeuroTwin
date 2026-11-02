"""Reuse main permutations for H1-8 at every sampled anchor with 8 future TRs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from analysis.h18_data import calibrate_raw
from analysis.h18_probes import GROUPS,ROOT
from analysis.individual_pilot import digest
from inference_twin import TwinSession,load_raw_series


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--device',required=True)
 p.add_argument('--batch-size',type=int,default=32);p.add_argument('--limit',type=int,default=0)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();ckpt,split,condition_file=GROUPS[a.group]
 ids=json.loads((ROOT/'data'/split).read_text())['test'];src=a.source/a.group
 man=json.loads((src/'数据清单.json').read_text())
 if man['subjects']!=ids or man['checkpoint_sha256']!=digest(ROOT/ckpt) or man['n_roi']!=116:raise ValueError('主分析清单不匹配')
 selected=ids[:a.limit] if a.limit else ids;dest=a.output/a.group;dest.mkdir(parents=True,exist_ok=True)
 protocol={'group':a.group,'checkpoint_sha256':digest(ROOT/ckpt),'source_manifest_sha256':digest(src/'数据清单.json'),
           'subjects':selected,'anchors':'all sampled anchors with cutoff+8 < scan length',
           'horizon':8,'permutation':'reuse original saved indices','feedback':'none','device':a.device,
           'dtype':'float32','batch_size':a.batch_size,'torch':torch.__version__}
 path=dest/'数据清单.json'
 if path.exists():
  if json.loads(path.read_text())!=protocol:raise ValueError('清单不同，请使用新目录')
 else:path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 session=TwinSession(str(ROOT/ckpt),device=a.device,condition_file=condition_file);model=session.model;start_time=time.perf_counter()
 with torch.inference_mode():
  for si,sid in enumerate(selected):
   path=dest/f'{sid}.npz'
   if path.exists():continue
   with np.load(src/f'{sid}.npz') as z:
    anchors=z['cutoff'].copy();indices=z['permutation_indices'].copy();h1_base=z['baseline'].copy();h1_pert=z['perturbed'].copy()
   raw=load_raw_series(ROOT/f'data/Raw/{a.group.upper()}/ROISignals_{sid}.mat',116);series=calibrate_raw(raw,32)
   keep=np.flatnonzero(anchors+8<len(series));chosen=anchors[keep]
   if not len(keep):raise ValueError(f'{sid}: 无完整 H1–8 锚点')
   cal=torch.from_numpy(series[:32].T.copy())[None,:,None,:].to(a.device);profile=model.calibrate_profile(cal);cond=session.condition_for(sid)
   def forward(x):
    n=len(x);_,aux=model(x,sc_matrix=None,calibration_bold=cal.expand(n,-1,-1,-1),profile=profile.expand(n,-1),
       condition={k:v.expand(n,*v.shape[1:]) for k,v in cond.items()},dynamics_horizon=8)
    return aux['dynamics_pred'].cpu().numpy()
   baselines=[];outputs=[];targets=[];diff_b=[];diff_p=[]
   for anchor_index,cutoff in zip(keep,chosen):
    x=torch.from_numpy(series[cutoff-63:cutoff+1].T.copy())[None,:,None,:].to(a.device)
    base=forward(x)[0];pred=np.empty((116,3,116,8),np.float32);tasks=[(r,j) for r in range(116) for j in range(3)]
    for start in range(0,len(tasks),a.batch_size):
     subset=tasks[start:start+a.batch_size];changed=x.repeat(len(subset),1,1,1)
     for row,(roi,rep) in enumerate(subset):changed[row,roi,0,:]=x[0,roi,0,indices[anchor_index,roi,rep].astype(np.int64)]
     values=forward(changed)
     for row,(roi,rep) in enumerate(subset):pred[roi,rep]=values[row]
    if not np.isfinite(pred).all() or not np.isfinite(base).all():raise ValueError(f'{sid}: 非有限输出')
    baselines.append(base);outputs.append(pred);targets.append(series[cutoff+1:cutoff+9].T)
    diff_b.append(np.max(np.abs(base[:,0]-h1_base[anchor_index])));diff_p.append(np.max(np.abs(pred[:,:,:,0]-h1_pert[anchor_index])))
   temp=dest/f'.{sid}.tmp.npz';np.savez_compressed(temp,cutoff=chosen,anchor_indices=keep,baseline=np.stack(baselines),
       perturbed=np.stack(outputs),true=np.stack(targets),h1_baseline_max_abs=np.array(diff_b),h1_perturbed_max_abs=np.array(diff_p))
   temp.replace(path);print(f'{a.group} {si+1}/{len(selected)} anchors={len(keep)} elapsed={time.perf_counter()-start_time:.1f}s',flush=True)

if __name__=='__main__':main()
