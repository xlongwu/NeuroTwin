"""H1–8 history-order and input perturbation probes at the first test anchor."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from inference_twin import TwinSession, load_raw_series
from analysis.h18_data import calibrate_raw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/H1-8分析_20261028/03_可解释性/扰动缓存'
GROUPS = {
    'hc': ('checkpoints/neurotwin_pretrain/base_best.pt', 'subject_split_pretrain.json', 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'),
    'mdd': ('checkpoints/neurotwin_finetune/finetuned_best.pt', 'subject_split_finetune.json', 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=GROUPS, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ns = ap.parse_args()
    checkpoint, split, condition_file = GROUPS[ns.group]
    ids = json.loads((ROOT/'data'/split).read_text())['test']
    ids = ids[:ns.limit] if ns.limit else ids
    directory = ns.out_dir/ns.group
    directory.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT/checkpoint
    protocol = {'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                'split_sha256':hashlib.sha256((ROOT/'data'/split).read_bytes()).hexdigest(),
                'group':ns.group, 'subjects':ids, 'anchor':'first K64', 'horizon':8,
                'permutation_seed':2024, 'history_block':8, 'repeats':3,
                'perturbation_last_tr':8, 'amplitude':.1, 'batch_size':ns.batch_size,
                'sham_threshold':1e-6}
    manifest = directory/'数据清单.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('已有缓存协议不同，请选用新目录')
    else:
        manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    rng = np.random.RandomState(2024)
    permutations = np.array([[rng.permutation(8) for _ in range(3)] for _ in range(8)])
    session = TwinSession(str(ckpt),device=ns.device,condition_file=condition_file)
    bs = ns.batch_size
    for number, sid in enumerate(ids,1):
        target = directory/f'{sid}.npz'
        if target.exists():
            continue
        raw = load_raw_series(ROOT/f'data/Raw/{ns.group.upper()}/ROISignals_{sid}.mat',116)
        series = calibrate_raw(raw, session.calibration_length)
        if len(series) < 72:
            raise ValueError(f'{sid}: 不足 K64 与完整 H1–8')
        x = torch.from_numpy(series[:64].T.astype(np.float32))[None,:,None,:].to(session.device)
        cal = x[..., :32]
        profile = session.model.calibrate_profile(cal)
        condition = session.condition_for(sid)
        true = series[64:72].T.astype(np.float32)

        def forward(history):
            n = len(history)
            _, aux = session.model(history, sc_matrix=None, calibration_bold=cal.expand(n,-1,-1,-1),
                                   profile=profile.expand(n,-1),
                                   condition={key:value.expand(n,*value.shape[1:]) for key,value in condition.items()},
                                   dynamics_horizon=8)
            p = aux['dynamics_pred'].float().cpu().numpy()
            if p.shape != (n,116,8) or not np.isfinite(p).all():
                raise ValueError(f'{sid}: H1–8 前向输出不合法 {p.shape}')
            return p

        base_x = x.repeat(bs,1,1,1)
        base = forward(base_x)
        sham_max = float(np.max(np.abs(forward(base_x.clone()) - base)))
        if sham_max > 1e-6:
            raise ValueError(f'{sid}: sham 最大差 {sham_max} > 1e-6')
        baseline_mae = np.array([np.abs(base[0]-true).mean(), np.abs(base[0,:,7]-true[:,7]).mean()])
        history_mae = np.empty((8,3,2),dtype=np.float32)
        tasks = [(b,r) for b in range(8) for r in range(3)]
        for start in range(0,len(tasks),bs):
            history = base_x.clone()
            for row,(block,repeat) in enumerate(tasks[start:start+bs]):
                lo = block*8
                history[row,:,:,lo:lo+8] = x[0,:,:,lo+permutations[block,repeat]]
            preds = forward(history)
            for row,(block,repeat) in enumerate(tasks[start:start+bs]):
                history_mae[block,repeat] = [np.abs(preds[row]-true).mean(), np.abs(preds[row,:,7]-true[:,7]).mean()]
        response = np.empty((2,116,116,8),dtype=np.float32)
        tasks = [(sign,roi) for sign in range(2) for roi in range(116)]
        for start in range(0,len(tasks),bs):
            history = base_x.clone()
            for row,(sign,roi) in enumerate(tasks[start:start+bs]):
                history[row,roi,0,-8:] += .1 * (1 if sign == 0 else -1)
            preds = forward(history)
            for row,(sign,roi) in enumerate(tasks[start:start+bs]):
                response[sign,roi] = preds[row]-base[row]
        payload = {'history':x[0,:,0,:].cpu().numpy(), 'calibration':cal[0].cpu().numpy(),
                   'profile':profile[0].cpu().numpy(), 'true':true, 'baseline':base[0],
                   'baseline_mae':baseline_mae, 'history_mae':history_mae,
                   'response':response, 'sham_max':np.array(sham_max)}
        payload.update({key:value[0].cpu().numpy() for key,value in condition.items()})
        np.savez_compressed(target,**payload)
        print(f'{ns.group}: {number}/{len(ids)}, sham={sham_max:.2g}',flush=True)


if __name__ == '__main__':
    main()
