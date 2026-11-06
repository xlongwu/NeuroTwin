"""Reuse first-anchor within-subject permutations for no-feedback H1–8."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from analysis.h18_data import calibrate_raw
from analysis.h18_probes import GROUPS, ROOT
from utils.condition_schema import file_sha256 as digest
from inference_twin import TwinSession, load_raw_series

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--perturbation-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    ckpt, split, condition_file = GROUPS[a.group]
    ids = json.loads((ROOT / 'data' / split).read_text())['test']
    src = a.perturbation_dir / a.group
    man = json.loads((src / '数据清单.json').read_text())
    if man['subjects'] != ids or man['checkpoint_sha256'] != digest(ROOT / ckpt) or man['n_roi'] != 116:
        raise ValueError('Complete matching main permutation manifest required')
    selected = ids[:a.limit] if a.limit else ids
    dst = a.out_dir / a.group
    dst.mkdir(parents=True, exist_ok=True)
    protocol = {'group': a.group, 'checkpoint_sha256': digest(ROOT / ckpt), 'perturbation_manifest_sha256': digest(src / '数据清单.json'), 'subjects': selected, 'anchors': 'first K64; require full H1–8', 'perturbations': 'reuse actual first-anchor 116×3 indices from main cache', 'horizon': 8, 'feedback': 'none', 'device': a.device, 'dtype': 'float32', 'batch_size': a.batch_size, 'torch': torch.__version__}
    manifest = dst / '数据清单.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('Existing manifest differs')
    else:
        manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    session = TwinSession(str(ROOT / ckpt), device=a.device, condition_file=condition_file)
    model = session.model
    start_time = time.perf_counter()
    with torch.inference_mode():
        for i, sid in enumerate(selected):
            out = dst / f'{sid}.npz'
            if out.exists():
                continue
            with np.load(src / f'{sid}.npz') as z:
                cutoff = int(z['cutoff'][0])
                indices = z['permutation_indices'][0].copy()
                h1 = z['baseline'][0].copy()
            raw = load_raw_series(ROOT / f'data/Raw/{a.group.upper()}/ROISignals_{sid}.mat', 116)
            series = calibrate_raw(raw, 32)
            if cutoff != 63 or cutoff + 8 >= len(series):
                raise ValueError(f'{sid}: insufficient H1–8 or first anchor differs')
            x = torch.from_numpy(series[:64].T.copy())[None, :, None, :].to(a.device)
            condition = session.condition_for(sid)

            def forward(history):
                n = len(history)
                aux_prediction, aux = model(history, condition={k: v.expand(n, *v.shape[1:]) for k, v in condition.items()})
                return aux_prediction[:, :, :8].cpu().numpy()
            base = forward(x)[0]
            pred = np.empty((116, 3, 116, 8), np.float32)
            tasks = [(r, j) for r in range(116) for j in range(3)]
            for start in range(0, len(tasks), a.batch_size):
                subset = tasks[start:start + a.batch_size]
                changed = x.repeat(len(subset), 1, 1, 1)
                for row, (roi, rep) in enumerate(subset):
                    changed[row, roi, 0, :] = x[0, roi, 0, indices[roi, rep].astype(np.int64)]
                value = forward(changed)
                for row, (roi, rep) in enumerate(subset):
                    pred[roi, rep] = value[row]
            if not np.isfinite(pred).all() or not np.isfinite(base).all():
                raise ValueError(f'{sid}: nonfinite output')
            tmp = dst / f'.{sid}.tmp.npz'
            np.savez_compressed(tmp, baseline=base, perturbed=pred, true=series[64:72].T, first_h1_difference=np.max(np.abs(base[:, 0] - h1)))
            tmp.replace(out)
            print(f'{a.group} {i + 1}/{len(selected)} {sid} elapsed={time.perf_counter() - start_time:.1f}s', flush=True)
if __name__ == '__main__':
    main()
