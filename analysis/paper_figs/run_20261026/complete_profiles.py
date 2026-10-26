"""Read-only profiles for the missing D6 descriptive control panels.

No new p-values or disease decision rules: matched inference is reused verbatim
from run1. Unmatched and HC val/test means are only visual controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from inference_twin import TwinSession, load_raw_series
from analysis.pathology_tests import HC_MERGED_FILE
from .style import ROOT, RUN1


@torch.no_grad()
def extraction_diagnostic(session: TwinSession) -> dict:
    """One fixed prefix checks the two observed extraction differences."""
    source=json.loads((ROOT/'results/paper_infer/run_20261026/manifest.json').read_text())
    sid=source['pairs'][0]['hc']
    raw=load_raw_series(ROOT/f'data/Raw/HC/ROISignals_{sid}.mat',116)
    session.calibrate_scan(raw)
    cal64=torch.from_numpy(session.series[:32].T.copy().astype(np.float32))[None,:,None,:]
    x=np.ascontiguousarray(raw.astype(np.float32).T)
    c=x[:,:32]
    mu=c.mean(1,keepdims=True);sd=np.sqrt(c.var(1,keepdims=True)+1e-6)
    cal32=torch.from_numpy(((x-mu)/sd)[:,:32].copy())[None,:,None,:]
    session.model.eval()
    e1=session.model.calibrate_profile(cal64);e2=session.model.calibrate_profile(cal64)
    ef=session.model.calibrate_profile(cal32)
    session.model.train()
    t1=session.model.calibrate_profile(cal64);t2=session.model.calibrate_profile(cal64)
    session.model.eval()
    return dict(eval_repeat_max=float((e1-e2).abs().max()),
        eval_float64_vs_loader_float32_max=float((e1-ef).abs().max()),
        training_mode_repeat_max=float((t1-t2).abs().max()),
        dropout_p=sorted(set(float(m.p) for m in session.model.modules() if isinstance(m,torch.nn.Dropout))),
        scope='One fixed HC prefix; diagnostic only. No training or weight save.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    ns = parser.parse_args()
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    target = ns.out_dir/'profiles.npz'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    torch.manual_seed(2024)
    session = TwinSession(str(ROOT/'checkpoints/neurotwin_pretrain/base_best.pt'),
                          device='cpu', condition_file=HC_MERGED_FILE)
    entries = [(group, split, str(sid)) for group in ('hc', 'mdd')
               for split in ('train', 'val', 'test')
               for sid in pd.read_csv(RUN1/f'z_maps/per_subject_z_{group}_{split}.csv').subject]
    result = []
    for start in range(0, len(entries), ns.batch_size):
        chunk = entries[start:start+ns.batch_size]
        calibrated = []
        for group, split, sid in chunk:
            raw = load_raw_series(ROOT/f'data/Raw/{group.upper()}/ROISignals_{sid}.mat', 116)
            session.calibrate_scan(raw)
            calibrated.append(session.series[:32].T.copy())
        with torch.no_grad():
            profile = session.model.calibrate_profile(
                torch.from_numpy(np.stack(calibrated).astype(np.float32))[:, :, None, :])
        result.append(profile.cpu().numpy())
        if start % (ns.batch_size*5) == 0:
            print(f'Profiles {min(start+len(chunk),len(entries))}/{len(entries)}', flush=True)
    np.savez_compressed(target, subjects=np.array([s for _, _, s in entries]),
                        groups=np.array([g for g, _, _ in entries]),
                        splits=np.array([sp for _, sp, _ in entries]), profiles=np.concatenate(result))
    diagnostic=extraction_diagnostic(session)
    (ns.out_dir/'extraction_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print('Profile control extraction complete; no new hypothesis tests', flush=True)


if __name__ == '__main__':
    main()
