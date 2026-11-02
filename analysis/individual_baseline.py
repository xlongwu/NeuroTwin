"""Eight-anchor baseline, graph and missing-condition probes for individual analysis."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from analysis.condition_usage import zero_condition
from analysis.h18_data import calibrate_raw
from analysis.h18_probes import GROUPS, ROOT
from analysis.individual_pilot import digest
from inference_twin import TwinSession, load_raw_series
from models.common import prepare_sc_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Only for separate pilot output')
    a = p.parse_args()
    ckpt, split, condition_file = GROUPS[a.group]
    split_path = ROOT / 'data' / split
    ids = json.loads(split_path.read_text())['test']
    if a.limit:
        ids = ids[:a.limit]
    dest = a.out_dir / a.group
    dest.mkdir(parents=True, exist_ok=True)
    protocol = {'group': a.group, 'checkpoint_sha256': digest(ROOT / ckpt),
                'split_sha256': digest(split_path), 'atlas_sha256': digest(ROOT / 'data/AAL116.xlsx'),
                'subjects': ids, 'history_length': 64, 'calibration_length': 32,
                'anchors': 'linspace round over legal cutoffs 63..T-2; unique; at most 8',
                'profile': 'first 32 calibrated TR; fixed across anchors',
                'condition_comparison': 'cond_z/code/mask all zero', 'graph_source': 'history_fc',
                'dtype': 'float32', 'device': a.device, 'torch': torch.__version__,
                'python': platform.python_version()}
    manifest = dest / '数据清单.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('Existing manifest differs; select a new output directory')
    else:
        manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    session = TwinSession(str(ROOT / ckpt), device=a.device, condition_file=condition_file)
    if session.calibration_length != 32:
        raise ValueError('Unexpected calibration length')
    model = session.model
    with torch.inference_mode():
        for i, sid in enumerate(ids, 1):
            path = dest / f'{sid}.npz'
            if path.exists():
                continue
            raw = load_raw_series(ROOT / f'data/Raw/{a.group.upper()}/ROISignals_{sid}.mat', 116)
            series = calibrate_raw(raw, 32)
            if len(series) < 65 or not np.isfinite(series).all():
                print(f'{a.group} {sid}: insufficient or nonfinite data', flush=True)
                continue
            anchors = np.arange(63, len(series) - 1)
            if len(anchors) > 8:
                anchors = anchors[np.unique(np.linspace(0, len(anchors)-1, 8).round().astype(int))]
            histories = np.stack([series[t-63:t+1].T for t in anchors])
            x = torch.from_numpy(histories.copy())[:, :, None, :].to(a.device)
            cal = torch.from_numpy(series[:32].T.copy())[None, :, None, :].to(a.device)
            profile = model.calibrate_profile(cal)
            cond = session.condition_for(sid)
            n = len(anchors)
            cal_batch = cal.expand(n, -1, -1, -1)
            prof_batch = profile.expand(n, -1)
            cond_batch = {k: v.expand(n, *v.shape[1:]) for k, v in cond.items()}
            _, aux = model(x, sc_matrix=None, calibration_bold=cal_batch, profile=prof_batch,
                           condition=cond_batch, dynamics_horizon=1)
            _, missing = model(x, sc_matrix=None, calibration_bold=cal_batch, profile=prof_batch,
                               condition=zero_condition(cond_batch), dynamics_horizon=1)
            graph_x = model.rev_norm(x, 'norm')[0] if model.rev_norm is not None else x
            fc = model._history_fc_adj(graph_x[:, :, 0, :])
            _, prior = prepare_sc_matrix(fc, graph_x)
            info = aux['sc_prior']
            def cpu(t):
                return t.detach().cpu().numpy()
            payload = {'cutoff': anchors, 'true': series[anchors+1],
                       'prediction': cpu(aux['dynamics_pred'][:, :, 0]),
                       'missing_prediction': cpu(missing['dynamics_pred'][:, :, 0]),
                       'profile': cpu(profile[0]), 'history_fc_abs': cpu(fc),
                       'prior': cpu(prior), 'A_func': cpu(info['A_func']),
                       'A_eff': cpu(info['A_eff']), 'lambda': cpu(info['lambda'])}
            if any(not np.isfinite(v).all() for v in payload.values()):
                print(f'{a.group} {sid}: nonfinite forward output', flush=True)
                continue
            np.savez_compressed(path, **payload)
            print(f'{a.group} {i}/{len(ids)} {sid}: {n} anchors', flush=True)


if __name__ == '__main__':
    main()
