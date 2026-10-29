"""Plot true vs predicted FC matrices for the H1-8 and rolling one-step protocols.

The H1-8 panels rebuild the fixed-anchor evaluation view from a checkpoint; the
rolling panels read the dense lag-1 caches written by ``analysis.rolling_one_step``.
Group panels average FC matrices with equal subject weight, subject panels show
the median subject by upper-triangle FC edge correlation.
"""
import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import next_point_eval as npe
from analysis.checkpoint import load_checkpoint_compat
from analysis.evaluate import (build_model_from_args, load_condition_bundle,
                               load_train_args)
from analysis.rolling_one_step import manifest_path
from main import condition_batch, dynamics_extras, next_point_batch
from train.losses import (complete_sequence_mask, correlation_matrix,
                          fc_mse_per_sample, fc_pattern_pcc)


def _edges(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _edge_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_edges(a), _edges(b))[0, 1])


def _edge_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(_edges(a) - _edges(b)).mean())


def _finish_subject(rows: dict, sid: str) -> dict:
    pred = rows['pred_sum'] / rows['count']
    true = rows['true_sum'] / rows['count']
    pcc = float(np.mean(rows['pcc']))
    return {'subject_id': str(sid), 'pred': pred, 'true': true,
            'edge_r': _edge_r(pred, true), 'edge_mae': _edge_mae(pred, true),
            'sample_pcc': pcc, 'n_samples': int(rows['count'])}


def _median_subject(subjects: dict, key: str) -> dict:
    ordered = sorted(subjects.values(), key=lambda item: item[key])
    return ordered[len(ordered) // 2]


def collect_h1_8(ckpt: str, split: str = 'test', device: str = 'cuda:0') -> dict:
    args = load_train_args(ckpt)
    args.device = device
    args.num_workers = 0
    args.pin_memory = False
    args.eval_batch_size = 128
    args.eval_anchors_per_subject = 8
    args.cpm_horizon = 8
    args.eval_rollout_tasks_per_subject = 0
    args.eval_rollout_steps = 0
    args.train_rollout_steps = 0
    args.eval_all_future_steps = 8
    args.eval_anchor_min_future_steps = 8
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    bundle = load_condition_bundle(ckpt)
    loader, _, _ = npe.build_next_point_loader(
        args, split, condition_bundle=bundle)
    model = build_model_from_args(args, dev, condition_bundle=bundle)
    model, _ = load_checkpoint_compat(model, ckpt, dev, strict=True)
    model.eval()

    horizon = 8
    n_rois = int(getattr(args, 'num_rois', 116))
    subjects = {}
    group_pred = np.zeros((n_rois, n_rois), np.float64)
    group_true = np.zeros_like(group_pred)
    group_pcc = []
    group_mse = []
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            bx, _, _, _, bs, _ = next_point_batch(batch, dev)
            cond = condition_batch(batch, dev)
            bcal, bctl = dynamics_extras(batch, dev)
            future = batch['future'].to(dev, non_blocking=True)
            fmask = batch['future_mask'].to(dev, non_blocking=True)
            controls = bctl[:, :horizon] if bctl is not None else None
            _, aux = model(bx, bs, None, pathology_epoch=None, condition=cond,
                           condition_epoch=None, controls=controls,
                           calibration_bold=bcal, dynamics_horizon=horizon,
                           with_aux=False)
            pred = aux['dynamics_pred'][:, :, :horizon]
            target = future[:, :, :horizon]
            complete = complete_sequence_mask(fmask[:, :horizon], horizon)
            if not bool(complete.any()):
                continue
            pred_c = pred[complete]
            target_c = target[complete]
            p_fc = correlation_matrix(pred_c).double().cpu().numpy()
            t_fc = correlation_matrix(target_c).double().cpu().numpy()
            pcc = fc_pattern_pcc(pred_c, target_c).cpu().numpy()
            mse = fc_mse_per_sample(pred_c, target_c).cpu().numpy()
            group_pred += p_fc.sum(axis=0)
            group_true += t_fc.sum(axis=0)
            group_pcc.extend(pcc.tolist())
            group_mse.extend(mse.tolist())
            n_samples += int(complete.sum())
            for local, index in enumerate(
                    complete.nonzero(as_tuple=False).squeeze(1).tolist()):
                sid = str(batch['subj_id'][index])
                row = subjects.setdefault(sid, {
                    'pred_sum': np.zeros((n_rois, n_rois), np.float64),
                    'true_sum': np.zeros((n_rois, n_rois), np.float64),
                    'count': 0, 'pcc': []})
                row['pred_sum'] += p_fc[local]
                row['true_sum'] += t_fc[local]
                row['count'] += 1
                row['pcc'].append(float(pcc[local]))

    model.cpu()
    del model, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finished = [_finish_subject(row, sid) for sid, row in subjects.items()]
    group = {
        'protocol': 'fixed_anchor_H1_8',
        'split': split,
        'checkpoint': str(ckpt),
        'n_subjects': len(finished),
        'n_samples': n_samples,
        'pred': group_pred / max(n_samples, 1),
        'true': group_true / max(n_samples, 1),
        'sample_pcc_mean': float(np.mean(group_pcc)),
        'sample_mse_mean': float(np.mean(group_mse)),
        'subjects': finished,
        'median_subject': _median_subject(
            {row['subject_id']: row for row in finished}, 'sample_pcc'),
    }
    group['edge_r'] = _edge_r(group['pred'], group['true'])
    group['edge_mae'] = _edge_mae(group['pred'], group['true'])
    group['subject_pcc_mean'] = float(np.mean([r['sample_pcc'] for r in finished]))
    return group


def collect_rolling(cache_dir: str) -> dict:
    cache = Path(cache_dir)
    manifest = json.loads(manifest_path(cache).read_text())
    finished = []
    for sid in manifest['subjects_done']:
        with np.load(cache / f'{sid}.npz', allow_pickle=False) as data:
            pred, true = data['pred'], data['true']
        if pred.shape[0] < 32:
            continue
        p_fc = np.corrcoef(pred, rowvar=False)
        t_fc = np.corrcoef(true, rowvar=False)
        valid = np.isfinite(p_fc) & np.isfinite(t_fc)
        if not np.all(valid[np.triu_indices(p_fc.shape[0], k=1)]):
            raise ValueError(f'{sid}: non-finite FC edges in rolling cache')
        finished.append({'subject_id': str(sid), 'pred': p_fc, 'true': t_fc,
                         'edge_r': _edge_r(p_fc, t_fc),
                         'edge_mae': _edge_mae(p_fc, t_fc),
                         'n_timepoints': int(pred.shape[0])})
    pred = np.mean([row['pred'] for row in finished], axis=0)
    true = np.mean([row['true'] for row in finished], axis=0)
    return {
        'protocol': 'rolling_one_step_full_sequence',
        'cache_dir': str(cache),
        'n_subjects': len(finished),
        'checkpoint': manifest.get('checkpoint'),
        'pred': pred, 'true': true,
        'edge_r': _edge_r(pred, true), 'edge_mae': _edge_mae(pred, true),
        'subject_r_mean': float(np.mean([r['edge_r'] for r in finished])),
        'subjects': finished,
        'median_subject': _median_subject(
            {row['subject_id']: row for row in finished}, 'edge_r'),
    }


def _panel(ax, matrix, title, vmin, vmax, cmap, label):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap,
                   interpolation='nearest', origin='upper')
    ax.set_title(title, fontsize=10)
    ticks = [0, 19, 39, 59, 79, 99, 115]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=7)
    ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=7)
    ax.set_xlabel('ROI (AAL116)', fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    return im


def _scatter(ax, record, title, subject_r=None):
    true = _edges(record['true'])
    pred = _edges(record['pred'])
    ax.scatter(true, pred, s=3, alpha=0.35, linewidths=0)
    ax.plot([-1, 1], [-1, 1], 'k--', linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('True FC edge r', fontsize=8)
    ax.set_ylabel('Predicted FC edge r', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.4)
    text = f"r = {record['edge_r']:.4f}\nMAE = {record['edge_mae']:.4f}"
    if subject_r is not None:
        text += f"\nmedian subject r = {subject_r:.4f}"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                      edgecolor='0.7'))


def _row(axs, record, mode, row_title, diff_vmax):
    subject = record['median_subject']
    matrix_pred = record['pred'] if mode == 'group' else subject['pred']
    matrix_true = record['true'] if mode == 'group' else subject['true']
    diff = matrix_pred - matrix_true
    _panel(axs[0], matrix_true, 'True FC', -1, 1, 'RdBu_r', 'ROI (AAL116)')
    _panel(axs[1], matrix_pred, 'Predicted FC', -1, 1, 'RdBu_r', 'ROI (AAL116)')
    _panel(axs[2], diff, 'Predicted - True', -diff_vmax, diff_vmax,
           'RdBu_r', 'ROI (AAL116)')
    _scatter(axs[3], {'true': matrix_true, 'pred': matrix_pred,
                      'edge_r': _edge_r(matrix_pred, matrix_true),
                      'edge_mae': _edge_mae(matrix_pred, matrix_true)},
             'Edge scatter', subject_r=subject['edge_r'])
    axs[0].text(-0.24, 0.5, row_title, transform=axs[0].transAxes,
                rotation=90, va='center', ha='center', fontsize=10,
                fontweight='bold')


def plot_figure(records, mode, out_stem, suptitle, footnote):
    if mode == 'group':
        diff_vmax = max(0.02, max(
            np.abs(r['pred'] - r['true']).max() for r in records))
        diff_vmax = float(np.ceil(diff_vmax * 20) / 20)
    else:
        diff_vmax = max(0.02, max(
            np.abs(r['median_subject']['pred'] - r['median_subject']['true']).max()
            for r in records))
        diff_vmax = float(np.ceil(diff_vmax * 20) / 20)

    fig, axes = plt.subplots(len(records), 4, figsize=(17, 8.6),
                             squeeze=False)
    for row, record in enumerate(records):
        title = (f"{record['group']} test - group mean (n={record['n_subjects']})"
                 if mode == 'group' else
                 f"{record['group']} test - median subject {record['median_subject']['subject_id']}"
                 f" (r={record['median_subject']['edge_r']:.3f})")
        _row(axes[row], record, mode, title, diff_vmax)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.text(0.5, 0.015, footnote, ha='center', va='bottom', fontsize=8,
             color='0.3', wrap=True)
    fig.tight_layout(rect=(0.015, 0.04, 1, 0.96))
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix('.png'), dpi=200)
    fig.savefig(out_stem.with_suffix('.pdf'))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hc-ckpt', required=True)
    ap.add_argument('--mdd-ckpt', required=True)
    ap.add_argument('--hc-rolling-cache', required=True)
    ap.add_argument('--mdd-rolling-cache', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('results/H1-8当前结果/图件'))
    ns = ap.parse_args()

    h1_hc = collect_h1_8(ns.hc_ckpt, ns.split, ns.device)
    h1_hc['group'] = 'HC'
    h1_mdd = collect_h1_8(ns.mdd_ckpt, ns.split, ns.device)
    h1_mdd['group'] = 'MDD'
    roll_hc = collect_rolling(ns.hc_rolling_cache)
    roll_hc['group'] = 'HC'
    roll_mdd = collect_rolling(ns.mdd_rolling_cache)
    roll_mdd['group'] = 'MDD'

    plot_figure(
        [h1_hc, h1_mdd], 'group', ns.out_dir / 'H1-8功能连接组均值',
        'H1-8 predicted vs true FC - fixed-anchor protocol (group mean)',
        'FC: Pearson correlation across the 8 future TRs for each ROI pair; '
        '6,670 upper-triangle edges, diagonal excluded. Each subject contributes '
        '8 anchors; group panel averages all anchor-level FC matrices.')
    plot_figure(
        [h1_hc, h1_mdd], 'subject', ns.out_dir / 'H1-8功能连接中位被试',
        'H1-8 predicted vs true FC - fixed-anchor protocol (median subject)',
        'Subject shown is the median by per-subject mean anchor-level FC edge r; '
        'its matrix is the average of that subject\'s 8 anchor-level FC matrices.')
    plot_figure(
        [roll_hc, roll_mdd], 'group', ns.out_dir / '滚动单步功能连接组均值',
        'Full-sequence predicted vs true FC - rolling one-step protocol (group mean)',
        'FC from each subject\'s full predicted/true sequence in the dense rolling '
        'cache; true history is refreshed at every TR. Group panel gives each '
        'subject equal weight; this is tracking fidelity, not autonomous generation.')

    stats = {}
    for record in (h1_hc, h1_mdd, roll_hc, roll_mdd):
        prefix = ('h1_8' if record['protocol'] == 'fixed_anchor_H1_8'
                  else 'rolling') + '_' + record['group'].lower()
        stats[prefix] = {
            'protocol': record['protocol'], 'n_subjects': record['n_subjects'],
            'group_edge_r': record['edge_r'],
            'group_edge_mae': record['edge_mae'],
            'median_subject': record['median_subject']['subject_id'],
            'median_subject_edge_r': record['median_subject']['edge_r'],
        }
        if record['protocol'] == 'fixed_anchor_H1_8':
            stats[prefix].update({
                'n_samples': record['n_samples'],
                'sample_fc_pcc_mean': record['sample_pcc_mean'],
                'sample_fc_mse_mean': record['sample_mse_mean'],
                'subject_fc_pcc_mean': record['subject_pcc_mean'],
            })
        else:
            stats[prefix]['subject_edge_r_mean'] = record['subject_r_mean']
    path = ns.out_dir / '功能连接图件统计.json'
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
