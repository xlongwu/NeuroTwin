"""Compare subject FC matrices from dense rolling-one-step prediction caches.

Example (run analysis.rolling_one_step on the same split first)::

    python -m analysis.fc_sequence_eval --cache-dir results/fc_test/rolling/hc \
        --split-file data/subject_split_pretrain.json --split test \
        --out-dir results/fc_test/hc

Each subject contributes one Pearson correlation between the strict upper
triangles of predicted and true FC. Subjects are never concatenated. This is
observed-history, rolling lag-1 evaluation, not autonomous free generation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.rolling_one_step import manifest_path
from utils.common import split_label


def fc_edges(series: np.ndarray) -> np.ndarray:
    """Return signed Pearson FC edges from a [time, ROI] sequence.

    Edges involving a constant ROI remain undefined (NaN).
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 32 or x.shape[1] < 3:
        raise ValueError(f'Expected [time >= 32, ROI >= 3], got {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError('Sequence contains non-finite values')
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=0))
    den = norm[:, None] * norm[None, :]
    fc = np.full(den.shape, np.nan)
    np.divide(x.T @ x, den, out=fc, where=den > 0)
    return np.clip(fc[np.triu_indices(x.shape[1], k=1)], -1.0, 1.0)


def compare_fc(pred: np.ndarray, true: np.ndarray) -> dict:
    """Within-subject FC pattern correlation and edge errors."""
    if np.shape(pred) != np.shape(true):
        raise ValueError(f'Prediction/target shape mismatch: {np.shape(pred)}/{np.shape(true)}')
    p, t = fc_edges(pred), fc_edges(true)
    valid = np.isfinite(p) & np.isfinite(t)
    p, t = p[valid], t[valid]
    pc = p - p.mean() if p.size else p
    tc = t - t.mean() if t.size else t
    den = np.linalg.norm(pc) * np.linalg.norm(tc)
    return {
        'fc_pcc': float(np.clip(pc @ tc / den, -1, 1))
                  if p.size >= 3 and den > 0 else float('nan'),
        'fc_mae': float(np.abs(p - t).mean()) if p.size else float('nan'),
        'fc_rmse': float(np.sqrt(((p - t) ** 2).mean())) if p.size else float('nan'),
        'n_valid_edges': int(valid.sum()),
        'n_edges': int(valid.size),
    }


def summarize(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {'n_valid': int(x.size),
            **{key: float(fn(x)) if x.size else None for key, fn in (
                ('mean', np.mean), ('std', np.std), ('median', np.median),
                ('min', np.min), ('max', np.max))}}


def evaluate_cache(cache_dir: Path, split_file: Path, split: str) -> tuple:
    manifest = json.loads(manifest_path(cache_dir).read_text())
    splits = json.loads(split_file.read_text())
    subjects = splits[split]
    others = {s for k in ('train', 'val', 'test') if k != split for s in splits[k]}
    if set(subjects) & others or len(set(subjects)) != len(subjects):
        raise ValueError('Subject split contains overlapping or duplicate IDs')
    missing = [s for s in subjects if s not in manifest['subjects_done']
               or not (cache_dir / f'{s}.npz').is_file()]
    if missing:
        raise ValueError(f'{len(missing)} subjects missing from cache; first: {missing[:3]}')
    rows = []
    for sid in subjects:
        with np.load(cache_dir / f'{sid}.npz', allow_pickle=False) as data:
            cutoff = data['cutoff']
            if cutoff.shape != (len(data['pred']),) or not np.all(np.diff(cutoff) == 1):
                raise ValueError(f'{sid}: expected consecutive, ordered rolling anchors')
            row = {'subject_id': sid, 'n_timepoints': len(cutoff),
                   'first_target_index': int(cutoff[0]) + 1,
                   'last_target_index': int(cutoff[-1]) + 1,
                   **compare_fc(data['pred'], data['true'])}
            rows.append(row)
    summary = {
        'split': split, 'group': manifest['group'], 'n_subjects': len(rows),
        'checkpoint': manifest['checkpoint'],
        'checkpoint_sha256': manifest['checkpoint_sha256'],
        'split_file': str(split_file),
        'split_file_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
        'protocol': {
            'prediction': 'ODE lag-1, true observed history refreshed at every anchor',
            'context_length': manifest['context_length'],
            'calibration_length': manifest['calibration_length'],
            'bold_source': manifest['bold_source'],
            'graph_source': manifest['graph_source'],
            'fc': 'signed temporal Pearson FC; strict upper triangle excluding diagonal',
            'correlation': 'Pearson across FC edges within each subject',
            'aggregation': 'equal subject weight; population SD; invalid values excluded and counted',
            'constant_roi': 'incident edges excluded, undefined FC correlation reported as NaN',
            'limitation': 'Rolling one-step FC fidelity does not measure autonomous generation fidelity',
        },
        'timepoints': summarize([r['n_timepoints'] for r in rows]),
    }
    for key in ('fc_pcc', 'fc_mae', 'fc_rmse'):
        summary[key] = summarize([r[key] for r in rows])
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', required=True, type=Path)
    ap.add_argument('--split-file', required=True, type=Path)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--out-dir', required=True, type=Path)
    ns = ap.parse_args()
    rows, summary = evaluate_cache(ns.cache_dir, ns.split_file, ns.split)
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    split_cn = split_label(ns.split)
    csv_path = ns.out_dir / f'逐被试功能连接_{split_cn}.csv'
    json_path = ns.out_dir / f'功能连接汇总_{split_cn}.json'
    if csv_path.exists() or json_path.exists():
        raise FileExistsError('结果已存在；请指定新的 --out-dir，避免覆盖既有结果')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
