"""Frozen-model F13/F14 probes registered before computation; no training.

Run from the repository root. Inputs are the existing fixed 20 held-out pairs.
Each completed subject is cached; --resume requires an identical protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from inference_twin import TwinSession
from analysis.pathology_tests import HC_MERGED_FILE
from .style import ROOT


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def probe_subject(model, sample: dict, permutations: np.ndarray,
                  amplitude: float, batch_size: int) -> dict:
    """One subject: block shuffles and signed, single-ROI input perturbations."""
    x = torch.from_numpy(sample['history'].T.copy()).float()[None, :, None, :]
    calibration = torch.from_numpy(sample['calibration'].copy()).float()[None]
    profile = torch.from_numpy(sample['profile'].copy()).float()[None]
    condition = {key: torch.from_numpy(sample[key].copy())[None]
                 for key in ('cond_z', 'cond_code', 'cond_mask')}

    def forward(history: torch.Tensor) -> np.ndarray:
        count = len(history)
        _, aux = model(history, sc_matrix=None, calibration_bold=calibration.expand(count, -1, -1, -1),
                       profile=profile.expand(count, -1),
                       condition={key: value.expand(count, *value.shape[1:])
                                  for key, value in condition.items()}, dynamics_horizon=32)
        pred = aux['dynamics_pred'].float().cpu().numpy()
        if pred.shape != (count, 116, 32) or not np.isfinite(pred).all():
            raise ValueError(f'Invalid prediction: {pred.shape}')
        return pred

    # Keep batch size identical for baseline, sham, and all perturbations.
    base_history = x.repeat(batch_size, 1, 1, 1)
    base = forward(base_history)
    sham = forward(base_history.clone())
    sham_max = float(np.max(np.abs(sham - base)))
    if sham_max > 1e-6:
        raise ValueError(f'Sham gate failed: {sham_max}')
    baseline = base[0]
    saved_difference = float(np.max(np.abs(baseline.T - sample['pred'])))
    # Stored inference used B=1; batch arithmetic can differ in float32.
    if saved_difference > 1e-4:
        raise ValueError(f'Source prediction mismatch: {saved_difference}')
    true = sample['true'].T
    bounds = [(0, 8), (8, 16), (16, 32)]
    base_mae = np.array([np.abs(baseline[:, a:b] - true[:, a:b]).mean()
                         for a, b in bounds])
    history_mae = np.empty((8, len(permutations[0]), 3), dtype=np.float64)
    tasks = [(block, repeat) for block in range(8) for repeat in range(len(permutations[0]))]
    for offset in range(0, len(tasks), batch_size):
        chunk = tasks[offset:offset + batch_size]
        history = base_history.clone()
        for row, (block, repeat) in enumerate(chunk):
            start = block * 8
            history[row, :, :, start:start+8] = x[0, :, :, start+permutations[block, repeat]]
        predictions = forward(history)
        for row, (block, repeat) in enumerate(chunk):
            history_mae[block, repeat] = [np.abs(predictions[row, :, a:b] - true[:, a:b]).mean()
                                           for a, b in bounds]

    # Store signed source x target x horizon responses, keeping subjects distinct.
    response = np.empty((2, 116, 116, 32), dtype=np.float32)
    tasks = [(sign, source) for sign in range(2) for source in range(116)]
    for offset in range(0, len(tasks), batch_size):
        chunk = tasks[offset:offset + batch_size]
        history = base_history.clone()
        for row, (sign, source) in enumerate(chunk):
            history[row, source, 0, -8:] += amplitude * (1 if sign == 0 else -1)
        predictions = forward(history)
        for row, (sign, source) in enumerate(chunk):
            response[sign, source] = (predictions[row] - base[row]) / amplitude
    return dict(baseline=baseline.T, baseline_mae=base_mae, history_mae=history_mae,
                response=response, sham_max=np.array(sham_max),
                saved_prediction_max_difference=np.array(saved_difference))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'results/paper_infer/run_20261026')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--amplitude', type=float, default=0.1)
    parser.add_argument('--pair-start', type=int, default=0,
                        help='Inclusive pair index for disjoint CPU work partitions.')
    parser.add_argument('--pair-stop', type=int, default=20,
                        help='Exclusive pair index; selection protocol remains unchanged.')
    parser.add_argument('--resume', action='store_true')
    ns = parser.parse_args()
    if ns.amplitude != 0.1:
        raise ValueError('This registered analysis uses amplitude=0.1.')
    if ns.batch_size < 1:
        raise ValueError('batch-size must be positive')
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    if any(ns.out_dir.iterdir()) and not ns.resume:
        raise FileExistsError('Use a fresh output directory or --resume.')
    source = json.loads((ns.source_dir/'manifest.json').read_text())
    checkpoint = ROOT/'checkpoints/neurotwin_pretrain/base_best.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from fixed source.')
    torch.set_num_threads(ns.threads)
    torch.manual_seed(2024)
    np.random.seed(2024)
    rng = np.random.RandomState(2024)
    permutations = np.array([[rng.permutation(8) for _ in range(3)] for _ in range(8)])
    registration = ROOT/'docs/VISUALIZATION_COMPLETION_PREREG.md'
    manifest = dict(source_manifest=source, amplitude=ns.amplitude, batch_size=ns.batch_size,
                    seed=2024, history_block_size=8, history_repeats=3,
                    block_permutations=permutations.tolist(), input_perturbation_last_tr=8,
                    registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),
                    sham_max_threshold=1e-6, role='Descriptive model sensitivity; no disease inference')
    target = ns.out_dir/'manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError('Resume protocol differs.')
    else:
        write_json(target, manifest)
    session = TwinSession(str(checkpoint), device='cpu', condition_file=HC_MERGED_FILE)
    if not 0 <= ns.pair_start < ns.pair_stop <= len(source['pairs']):
        raise ValueError('Invalid pair index range')
    for index in range(ns.pair_start, ns.pair_stop):
        pair = source['pairs'][index]
        for group in ('hc', 'mdd'):
            name = f'{group}_{pair[group]}.npz'
            target = ns.out_dir/name
            if target.exists():
                continue
            start = time.monotonic()
            with np.load(ns.source_dir/name) as data:
                sample = {key: data[key].copy() for key in data.files}
            result = probe_subject(session.model, sample, permutations, ns.amplitude, ns.batch_size)
            np.savez_compressed(target, **result)
            print(f'Pair {index+1}/{len(source["pairs"])} {group}: {time.monotonic()-start:.1f}s; sham={float(result["sham_max"]):.2g}', flush=True)
    print('F13/F14 inference complete', flush=True)


if __name__ == '__main__':
    main()
