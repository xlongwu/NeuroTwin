"""Within-subject blockwise ROI history permutation at eight rolling anchors."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from analysis.h18_data import calibrate_raw
from analysis.h18_probes import GROUPS, ROOT
from analysis.individual_pilot import digest
from inference_twin import TwinSession, load_raw_series


def permutations(subject_index: int, anchor_index: int, group: str, n_roi: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([2024, 0 if group == 'hc' else 1,
                                                        subject_index, anchor_index]))
    out = np.empty((n_roi, 3, 64), dtype=np.uint8)
    for roi in range(n_roi):
        for repeat in range(3):
            for block in range(8):
                start = 8 * block
                count = 8 if block < 7 else 7
                out[roi, repeat, start:start+count] = start + rng.permutation(count)
            out[roi, repeat, 63] = 63
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Use only in a separate pilot directory')
    parser.add_argument('--roi-limit', type=int, default=0, help='Use only in a separate pilot directory')
    a = parser.parse_args()
    if a.batch_size < 1 or a.roi_limit < 0 or a.roi_limit > 116:
        raise ValueError('Invalid batch size or ROI limit')
    ckpt, split, condition_file = GROUPS[a.group]
    ids = json.loads((ROOT / 'data' / split).read_text())['test']
    baseline = a.baseline_dir / a.group
    base_manifest = json.loads((baseline / '数据清单.json').read_text())
    if (base_manifest['subjects'] != ids or base_manifest['checkpoint_sha256'] != digest(ROOT / ckpt)
            or base_manifest['split_sha256'] != digest(ROOT / 'data' / split)):
        raise ValueError('Baseline source does not match complete test split and checkpoint')
    selected = ids[:a.limit] if a.limit else ids
    n_roi = a.roi_limit or 116
    dest = a.out_dir / a.group
    dest.mkdir(parents=True, exist_ok=True)
    protocol = {'group': a.group, 'checkpoint_sha256': digest(ROOT / ckpt),
                'split_sha256': digest(ROOT / 'data' / split), 'subjects': selected,
                'baseline_manifest_sha256': digest(baseline / '数据清单.json'),
                'seed': 2024, 'rng': 'numpy default_rng SeedSequence(seed, group index, test subject index, anchor index)',
                'n_roi': n_roi, 'repeats': 3, 'block_size': 8, 'final_value_fixed': True,
                'history_length': 64, 'profile_calibration': 'original first 32 TR',
                'graph': 'recomputed from perturbed history', 'device': a.device,
                'batch_size': a.batch_size, 'dtype': 'float32', 'torch': torch.__version__}
    manifest = dest / '数据清单.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('Existing perturbation manifest differs; select a new directory')
    else:
        manifest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    session = TwinSession(str(ROOT / ckpt), device=a.device, condition_file=condition_file)
    model = session.model
    start_run = time.perf_counter()
    with torch.inference_mode():
        for subject_index, sid in enumerate(selected):
            output = dest / f'{sid}.npz'
            if output.exists():
                continue
            with np.load(baseline / f'{sid}.npz') as saved:
                anchors = saved['cutoff'].copy()
                cpu_base = saved['prediction'].copy()
                expected_true = saved['true'].copy()
            raw = load_raw_series(ROOT / f'data/Raw/{a.group.upper()}/ROISignals_{sid}.mat', 116)
            series = calibrate_raw(raw, 32)
            if not np.isfinite(series).all() or not np.array_equal(series[anchors+1], expected_true):
                raise ValueError(f'{sid}: nonfinite raw data or baseline targets differ')
            calibration = torch.from_numpy(series[:32].T.copy())[None, :, None, :].to(a.device)
            profile = model.calibrate_profile(calibration)
            condition = session.condition_for(sid)
            def forward(histories):
                count = len(histories)
                _, aux = model(histories, sc_matrix=None,
                               calibration_bold=calibration.expand(count, -1, -1, -1),
                               profile=profile.expand(count, -1),
                               condition={k: v.expand(count, *v.shape[1:]) for k,v in condition.items()},
                               dynamics_horizon=1)
                return aux['dynamics_pred'][:, :, 0]
            all_pred = np.empty((len(anchors), n_roi, 3, 116), np.float32)
            base_pred = np.empty((len(anchors), 116), np.float32)
            all_indices = np.empty((len(anchors), n_roi, 3, 64), np.uint8)
            for ai, cutoff in enumerate(anchors):
                x = torch.from_numpy(series[cutoff-63:cutoff+1].T.copy())[None, :, None, :].to(a.device)
                base_pred[ai] = forward(x)[0].cpu().numpy()
                inds = permutations(subject_index, ai, a.group, n_roi)
                all_indices[ai] = inds
                tasks = [(r, j) for r in range(n_roi) for j in range(3)]
                for start in range(0, len(tasks), a.batch_size):
                    subset = tasks[start:start+a.batch_size]
                    changed = x.repeat(len(subset), 1, 1, 1)
                    for row, (roi, repeat) in enumerate(subset):
                        changed[row, roi, 0, :] = x[0, roi, 0, inds[roi, repeat].astype(np.int64)]
                    pred = forward(changed).cpu().numpy()
                    for row, (roi, repeat) in enumerate(subset):
                        all_pred[ai, roi, repeat] = pred[row]
            if not np.isfinite(all_pred).all() or not np.isfinite(base_pred).all():
                raise ValueError(f'{sid}: nonfinite predictions')
            tmp = dest / f'.{sid}.tmp.npz'
            np.savez_compressed(tmp, cutoff=anchors, true=expected_true, baseline=base_pred,
                                perturbed=all_pred, permutation_indices=all_indices,
                                cpu_baseline_max_abs=np.max(np.abs(base_pred-cpu_base)))
            tmp.replace(output)
            print(f'{a.group} {subject_index+1}/{len(selected)} {sid} anchors={len(anchors)} '
                  f'elapsed={time.perf_counter()-start_run:.1f}s', flush=True)


if __name__ == '__main__':
    main()
