"""Audit cached HC-reference H32 predictions and paired profile controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inference_twin import TwinSession
from analysis.pathology_tests import HC_MERGED_FILE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    session = TwinSession(str(args.checkpoint), device='cpu', condition_file=HC_MERGED_FILE)
    model = session.model
    model.eval()
    rows = []
    for group in ('hc', 'mdd'):
        paths = sorted(args.cache_dir.glob(f'{group}_*.npz'))
        if len(paths) < 2:
            raise ValueError(f'{group}: need at least two cached subjects')
        samples = []
        for path in paths:
            with np.load(path) as d:
                samples.append({k: d[k].copy() for k in d.files})
        x = torch.tensor(np.stack([s['history'].T for s in samples]), dtype=torch.float32)[:, :, None, :]
        profile = torch.tensor(np.stack([s['profile'] for s in samples]), dtype=torch.float32)
        cond = {k: torch.tensor(np.stack([s[k] for s in samples])) for k in ('cond_z', 'cond_code', 'cond_mask')}
        with torch.no_grad():
            _, own = model(x, sc_matrix=None, condition=cond, profile=profile, dynamics_horizon=32)
            _, swapped = model(x, sc_matrix=None, condition=cond, profile=profile.roll(1, 0), dynamics_horizon=32)
        pred = own['dynamics_pred'].cpu().numpy().transpose(0, 2, 1)
        std = own['dynamics_std'].cpu().numpy().transpose(0, 2, 1)
        alt = swapped['dynamics_pred'].cpu().numpy().transpose(0, 2, 1)
        for i, (path, s) in enumerate(zip(paths, samples)):
            y, p = s['true'], pred[i]
            anchor = s['history'][-1]
            dp = np.diff(np.vstack([anchor, p]), axis=0)
            dy = np.diff(np.vstack([anchor, y]), axis=0)
            eligible = np.abs(dy) > .05
            iu = np.triu_indices(y.shape[1], 1)
            fc_p, fc_y = np.corrcoef(p.T)[iu], np.corrcoef(y.T)[iu]
            mae = np.abs(p-y).mean()
            base = np.abs(anchor-y).mean()
            ratio = np.var(p, axis=0) / np.var(y, axis=0)
            rows.append(dict(group=group, subject=path.stem[len(group)+1:], mae=float(mae),
                persistence_mae=float(base), persistence_gain=float(1-mae/base),
                delta_corr=float(np.corrcoef(dp.ravel(),dy.ravel())[0,1]),
                direction_accuracy=float(np.mean(np.sign(dp[eligible])==np.sign(dy[eligible]))),
                temporal_variance_ratio=float(np.mean(ratio)),
                fc_corr=float(np.corrcoef(fc_p,fc_y)[0,1]), fc_mae=float(np.mean(np.abs(fc_p-fc_y))),
                profile_swap_mae=float(np.abs(alt[i]-y).mean()),
                profile_gain=float(np.abs(alt[i]-y).mean()-mae),
                coverage_50=float(np.mean(np.abs(p-y)<=0.67448975*std[i])),
                coverage_90=float(np.mean(np.abs(p-y)<=1.64485363*std[i])),
                coverage_95=float(np.mean(np.abs(p-y)<=1.95996398*std[i])),
                coverage_90_h1=float(np.mean(np.abs(p[0]-y[0])<=1.64485363*std[i,0])),
                width_90=float(np.mean(2*1.64485363*std[i])),
                replication_max_abs=float(np.max(np.abs(p-s['pred'])))))
    import csv
    with (args.out_dir/'per_subject.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for group in ('hc', 'mdd'):
        subset = [r for r in rows if r['group']==group]
        summary[group] = {'n':len(subset), **{k:float(np.mean([r[k] for r in subset])) for k in rows[0] if k not in ('group','subject')}}
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
