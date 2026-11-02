"""Read-only one-subject feasibility check; this does not produce study results."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from analysis.h18_data import calibrate_raw
from analysis.h18_probes import GROUPS, OUT, ROOT
from inference_twin import TwinSession, load_raw_series


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(group: str, device: str, batch_size: int) -> dict:
    checkpoint, split, condition_file = GROUPS[group]
    split_path = ROOT / 'data' / split
    ids = json.loads(split_path.read_text())['test']
    sid = ids[0]
    with np.load(OUT / group / f'{sid}.npz') as data:
        history = data['history'].copy()
        calibration = data['calibration'].copy()
        profile_data = data['profile'].copy()
        condition_data = {k: data[k].copy() for k in ('cond_z', 'cond_code', 'cond_mask')}
        cached_h1 = data['baseline'][:, 0].copy()
    session = TwinSession(str(ROOT / checkpoint), device=device, condition_file=condition_file)
    raw = load_raw_series(ROOT / f'data/Raw/{group.upper()}/ROISignals_{sid}.mat', 116)
    series = calibrate_raw(raw, session.calibration_length)
    x = torch.from_numpy(history)[None, :, None, :].to(device)
    cal = torch.from_numpy(calibration)[None].to(device)
    profile = torch.from_numpy(profile_data)[None].to(device)
    condition = {k: torch.from_numpy(v)[None].to(device) for k, v in condition_data.items()}
    rng = np.random.RandomState(2024)
    indices = np.concatenate([8 * b + rng.permutation(8 if b < 7 else 7)
                              for b in range(8)] + [np.array([63])])
    perturbed = x.clone()
    perturbed[0, 0, 0] = x[0, 0, 0, indices]
    assert all(set(indices[8*b:8*b+8]) == set(range(8*b, 8*b+8)) for b in range(8))
    assert torch.equal(perturbed[:, 1:], x[:, 1:]) and torch.equal(perturbed[..., -1], x[..., -1])

    def forward(history_tensor):
        n = len(history_tensor)
        return session.model(history_tensor, sc_matrix=None,
                             calibration_bold=cal.expand(n, -1, -1, -1),
                             profile=profile.expand(n, -1),
                             condition={k: v.expand(n, *v.shape[1:]) for k, v in condition.items()},
                             dynamics_horizon=1)[1]

    with torch.inference_mode():
        a, b = forward(x), forward(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        c = forward(perturbed.repeat(batch_size, 1, 1, 1))
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {'group': group, 'n_test': len(ids), 'subject': sid,
            'checkpoint_sha256': digest(ROOT / checkpoint), 'split_sha256': digest(split_path),
            'device': device, 'torch_version': torch.__version__, 'dtype': str(x.dtype),
            'batch_size': batch_size, 'seconds': elapsed, 'samples_per_second': batch_size / elapsed,
            'raw_history_max_abs': float(np.max(np.abs(series[:64].T - history))),
            'repeat_max_abs': float((a['dynamics_pred'] - b['dynamics_pred']).abs().max()),
            'cached_h1_max_abs': float(np.max(np.abs(a['dynamics_pred'][0, :, 0].cpu().numpy() - cached_h1))),
            'effective_graph_change_max_abs': float((a['sc_prior']['A_eff'][0] - c['sc_prior']['A_eff'][0]).abs().max()),
            'permutation_indices': indices.tolist(), 'other_rois_and_last_frame_unchanged': True,
            'scope': 'single subject and one ROI perturbation; no statistical conclusions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'输出已存在，请指定新路径：{args.output}')
    result = [run(group, args.device, args.batch_size) for group in ('hc', 'mdd')]
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
