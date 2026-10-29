"""PSD and probabilistic forecast diagnostics for the H1-8 and rolling protocols.

H1-8: fixed-anchor predictions from a checkpoint, using the VarianceHead
marginal Gaussian N(mu, sigma). Reports CRPS, PIT/coverage, Gaussian NLL,
Welch PSD shape/MAE, and p=2 variogram scores. The model only defines diagonal
marginals; variogram scores therefore use three explicit plug-in assumptions:
independence, predicted 8-step FC correlation, and oracle true 8-step FC.

Rolling: full-sequence lag-1 caches contain only point predictions, so only PSD
metrics are computed there. CRPS/variogram are not defined for those caches
without re-running the model to obtain sigma.
"""
import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch
from scipy.special import ndtr
from scipy.stats import kstest, norm

from analysis import next_point_eval as npe
from analysis.checkpoint import load_checkpoint_compat
from analysis.evaluate import (build_model_from_args, load_condition_bundle,
                               load_train_args)
from analysis.metrics import gaussian_crps
from analysis.rolling_one_step import manifest_path
from main import condition_batch, dynamics_extras, next_point_batch
from train.losses import complete_sequence_mask

EPS = 1e-12
HORIZON = 8
TR = 2.0
FS_HZ = 1.0 / TR
LOW_BAND_HZ = (0.01, 0.1)
ROI_PANELS = [(5, 'DMN'), (8, 'FPN'), (12, 'VAN'), (10, 'LN')]
NETWORK_COLORS = {'DMN': '#3B6EA8', 'FPN': '#D05A50',
                  'VAN': '#4B9B82', 'LN': '#A2678A'}


def _agg(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {'mean': None, 'std': None, 'n': 0}
    return {'mean': float(x.mean()), 'std': float(x.std()), 'n': int(x.size)}


def _corr_last(a, b):
    """Pearson correlation along the last axis; NaN for zero variance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide(num, den, out=np.full(den.shape, np.nan), where=den > EPS)


def _corr_across_rois(a, b):
    """Per-frequency correlation across ROIs for [F, n_freq] arrays."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide(num, den, out=np.full(den.shape, np.nan), where=den > EPS)


def _batch_corr_last(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide(num, den, out=np.full(den.shape, np.nan), where=den > EPS)


def _psd_metrics_single(pred_psd, true_psd, freqs):
    """PSD shape and amplitude diagnostics for one subject's [F, freq] PSD."""
    keep = freqs > 0
    f = freqs[keep]
    p, t = pred_psd[:, keep], true_psd[:, keep]
    pn = p / (p.sum(axis=1, keepdims=True) + EPS)
    tn = t / (t.sum(axis=1, keepdims=True) + EPS)
    band = (f >= LOW_BAND_HZ[0]) & (f <= LOW_BAND_HZ[1])
    if band.any():
        rel_p = p[:, band].sum(axis=1) / (p.sum(axis=1) + EPS)
        rel_t = t[:, band].sum(axis=1) / (t.sum(axis=1) + EPS)
        rel_mae = float(np.abs(rel_p - rel_t).mean())
    else:
        rel_mae = float('nan')
    return {
        'psd_shape_r': _corr_last(pn, tn),
        'psd_roi_r': float(np.nanmean(_corr_across_rois(pn, tn))),
        'psd_flat_r': float(_corr_last(pn.reshape(1, -1),
                                       tn.reshape(1, -1))[0]),
        'psd_mae_norm': float(np.abs(pn - tn).mean()),
        'psd_mae_raw': float(np.abs(p - t).mean()),
        'psd_mae_log10': float(np.abs(np.log10(p + EPS)
                                      - np.log10(t + EPS)).mean()),
        'rel_band_power_mae': rel_mae,
        'freqs': f,
    }


def _psd_metrics_batch(pred_psd, true_psd, freqs):
    keep = freqs > 0
    f = freqs[keep]
    p, t = pred_psd[..., keep], true_psd[..., keep]
    pn = p / (p.sum(axis=-1, keepdims=True) + EPS)
    tn = t / (t.sum(axis=-1, keepdims=True) + EPS)
    shape_r = _batch_corr_last(pn, tn)                     # [n, F]
    roi_r = _corr_last(
        pn.transpose(0, 2, 1), tn.transpose(0, 2, 1))      # [n, freq]
    flat_r = _batch_corr_last(pn.reshape(pn.shape[0], -1),
                              tn.reshape(tn.shape[0], -1))  # [n]
    band = (f >= LOW_BAND_HZ[0]) & (f <= LOW_BAND_HZ[1])
    if band.any():
        rel_p = p[..., band].sum(axis=-1) / (p.sum(axis=-1) + EPS)
        rel_t = t[..., band].sum(axis=-1) / (t.sum(axis=-1) + EPS)
        rel_mae = np.abs(rel_p - rel_t).mean(axis=1)
    else:
        rel_mae = np.full(pn.shape[0], np.nan)
    return {
        'psd_shape_r': shape_r,
        'psd_roi_r': np.nanmean(roi_r, axis=1),
        'psd_flat_r': flat_r,
        'psd_mae_norm': np.abs(pn - tn).mean(axis=(1, 2)),
        'psd_mae_raw': np.abs(p - t).mean(axis=(1, 2)),
        'psd_mae_log10': np.abs(np.log10(p + EPS)
                                - np.log10(t + EPS)).mean(axis=(1, 2)),
        'rel_band_power_mae': rel_mae,
        'freqs': f,
    }


def _batch_corr_matrix(x):
    """Cross-ROI correlation from [n, F, T] trajectories."""
    n, f, t = x.shape
    xc = x - x.mean(axis=-1, keepdims=True)
    s = np.sqrt((xc * xc).mean(axis=-1))                    # [n, F]
    den = s[:, :, None] * s[:, None, :]
    corr = np.einsum('nft,ngt->nfg', xc, xc) / (t * den + EPS)
    corr = np.where(den > EPS, corr, 0.0)
    diag = np.arange(f)
    corr[:, diag, diag] = 1.0
    return np.clip(corr, -1.0, 1.0)


def _variogram_batch(mu, std, target, corr_pred, corr_true,
                     iu0, iu1):
    """p=2 variogram score under independence / predicted / oracle FC."""
    s_prod = std[:, :, None] * std[:, None, :]
    mean_diff2 = (mu[:, :, None] - mu[:, None, :]) ** 2
    var_diff = std[:, :, None] ** 2 + std[:, None, :] ** 2
    true_diff2 = (target[:, :, None] - target[:, None, :]) ** 2

    def score(e):
        return ((true_diff2 - e)[:, iu0, iu1] ** 2).mean(axis=1)

    e_ind = mean_diff2 + var_diff
    e_pred = mean_diff2 + np.maximum(var_diff - 2.0 * corr_pred * s_prod, 0.0)
    e_oracle = mean_diff2 + np.maximum(var_diff - 2.0 * corr_true * s_prod, 0.0)
    return {'vs_indep': score(e_ind), 'vs_pred_fc': score(e_pred),
            'vs_oracle_fc': score(e_oracle)}


def _coverage_curve(z, n_points=19):
    nominal = np.linspace(0.05, 0.95, n_points)
    empirical = []
    for q in nominal:
        cut = norm.ppf((1.0 + q) / 2.0)
        empirical.append(float(np.mean(np.abs(z) <= cut)))
    return nominal, np.asarray(empirical)


def _calibration_block(z, pit):
    z = np.asarray(z, dtype=np.float64).ravel()
    pit = np.asarray(pit, dtype=np.float64).ravel()
    return {
        'z_std': float(np.std(z)),
        'coverage_50': float(np.mean(np.abs(z) <= norm.ppf(0.75))),
        'coverage_90': float(np.mean(np.abs(z) <= norm.ppf(0.95))),
        'coverage_95': float(np.mean(np.abs(z) <= norm.ppf(0.975))),
        'pit_ks_uniform': float(kstest(pit, 'uniform').statistic),
        'pit_mean': float(pit.mean()),
        'pit_std': float(pit.std()),
        'pit_below_0.01': float(np.mean(pit < 0.01)),
        'pit_above_0.99': float(np.mean(pit > 0.99)),
    }


def collect_h1_8(ckpt, split='test', device='cuda:0'):
    args = load_train_args(ckpt)
    args.device = device
    args.num_workers = 0
    args.pin_memory = False
    args.eval_batch_size = 128
    args.eval_anchors_per_subject = 8
    args.cpm_horizon = HORIZON
    args.eval_rollout_tasks_per_subject = 0
    args.eval_rollout_steps = 0
    args.train_rollout_steps = 0
    args.eval_all_future_steps = HORIZON
    args.eval_anchor_min_future_steps = HORIZON
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    bundle = load_condition_bundle(ckpt)
    loader, _, _ = npe.build_next_point_loader(
        args, split, condition_bundle=bundle)
    model = build_model_from_args(args, dev, condition_bundle=bundle)
    model, _ = load_checkpoint_compat(model, ckpt, dev, strict=True)
    model.eval()

    subj = defaultdict(lambda: defaultdict(list))
    psd_pred_sum = defaultdict(lambda: np.zeros((116, 0)))
    psd_true_sum = defaultdict(lambda: np.zeros((116, 0)))
    psd_count = defaultdict(int)
    mu_h8, std_h8, y_h8 = [], [], []
    pit_h1, pit_h8, z_h1, z_h8 = [], [], [], []
    pit_all_list, z_all_list = [], []
    n_samples = 0
    iu = np.triu_indices(116, k=1)
    iu0, iu1 = iu
    with torch.no_grad():
        for batch in loader:
            bx, _, _, _, bs, _ = next_point_batch(batch, dev)
            cond = condition_batch(batch, dev)
            bcal, bctl = dynamics_extras(batch, dev)
            future = batch['future'].to(dev, non_blocking=True)
            fmask = batch['future_mask'].to(dev, non_blocking=True)
            x_last = bx[:, :, 0, -1]
            controls = bctl[:, :HORIZON] if bctl is not None else None
            _, aux = model(bx, bs, None, pathology_epoch=None, condition=cond,
                           condition_epoch=None, controls=controls,
                           calibration_bold=bcal, dynamics_horizon=HORIZON,
                           with_aux=True)
            mu = aux['dynamics_pred'][:, :, :HORIZON].cpu().numpy()
            std = aux['dynamics_std'][:, :, :HORIZON].cpu().numpy()
            target = future[:, :, :HORIZON].cpu().numpy()
            xl = x_last.cpu().numpy()
            complete = complete_sequence_mask(fmask[:, :HORIZON], HORIZON)
            complete = complete.cpu().numpy()
            if not complete.any():
                continue
            mu_c, std_c, target_c, xl_c = (mu[complete], std[complete],
                                           target[complete], xl[complete])
            if not np.isfinite(std_c).all() or not (std_c > 0).all():
                raise ValueError(
                    'Non-finite or non-positive dynamics_std in batch; '
                    'cannot compute CRPS/PIT')
            sids = [str(batch['subj_id'][i]) for i in np.flatnonzero(complete)]
            n = mu_c.shape[0]
            n_samples += n

            crps = gaussian_crps(mu_c, std_c, target_c)
            crps_clim = gaussian_crps(
                np.zeros_like(target_c), np.ones_like(target_c), target_c)
            crps_pers = gaussian_crps(
                np.broadcast_to(xl_c[:, :, None], target_c.shape),
                np.ones_like(target_c), target_c)
            z = (target_c - mu_c) / std_c
            pit = ndtr(z)
            pit_h1.append(pit[:, :, 0].ravel())
            pit_h8.append(pit[:, :, -1].ravel())
            z_h1.append(z[:, :, 0].ravel())
            z_h8.append(z[:, :, -1].ravel())
            pit_all_list.append(pit.ravel())
            z_all_list.append(z.ravel())

            freqs, psd_p = welch(mu_c, fs=FS_HZ,
                                  nperseg=HORIZON, noverlap=HORIZON // 2,
                                  axis=-1, detrend='constant',
                                  scaling='density')
            _, psd_t = welch(target_c, fs=FS_HZ,
                             nperseg=HORIZON, noverlap=HORIZON // 2,
                             axis=-1, detrend='constant',
                             scaling='density')
            psd_b = _psd_metrics_batch(psd_p, psd_t, freqs)

            corr_pred = _batch_corr_matrix(mu_c)
            corr_true = _batch_corr_matrix(target_c)
            vs_h = {key: [] for key in ('vs_indep', 'vs_pred_fc', 'vs_oracle_fc')}
            for h in range(HORIZON):
                block = _variogram_batch(
                    mu_c[:, :, h], std_c[:, :, h], target_c[:, :, h],
                    corr_pred, corr_true, iu0, iu1)
                for key in vs_h:
                    vs_h[key].append(block[key])
            vs_sample = {key: np.asarray(vs_h[key], dtype=np.float64).mean(axis=0)
                         for key in vs_h}

            mu_h8.append(mu_c[:, :, -1])
            std_h8.append(std_c[:, :, -1])
            y_h8.append(target_c[:, :, -1])

            for i, sid in enumerate(sids):
                s = subj[sid]
                s['crps'].append(float(crps[i].mean()))
                if 'crps_h_sum' not in s:
                    s['crps_h_sum'] = np.zeros(HORIZON)
                    s['crps_h_n'] = 0
                s['crps_h_sum'] += crps[i].mean(axis=0)
                s['crps_h_n'] += 1
                s['crps_h1'].append(float(crps[i][:, 0].mean()))
                s['crps_h8'].append(float(crps[i][:, -1].mean()))
                s['crps_clim'].append(float(crps_clim[i].mean()))
                s['crps_pers'].append(float(crps_pers[i].mean()))
                s['nll'].append(float(np.mean(
                    0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(std_c[i])
                           + z[i] ** 2))))
                s['width_90'].append(float(np.mean(
                    2.0 * norm.ppf(0.95) * std_c[i])))
                s['psd_shape_anchor'].append(
                    float(np.nanmean(psd_b['psd_shape_r'][i])))
                s['psd_roi_anchor'].append(float(psd_b['psd_roi_r'][i]))
                s['psd_flat_anchor'].append(float(psd_b['psd_flat_r'][i]))
                s['psd_mae_norm_anchor'].append(
                    float(psd_b['psd_mae_norm'][i]))
                s['psd_mae_log10_anchor'].append(
                    float(psd_b['psd_mae_log10'][i]))
                s['rel_band_anchor'].append(
                    float(psd_b['rel_band_power_mae'][i]))
                for key in vs_h:
                    if f'{key}_h_sum' not in s:
                        s[f'{key}_h_sum'] = np.zeros(HORIZON)
                        s[f'{key}_h_n'] = 0
                    s[f'{key}_h_sum'] += np.mean(
                        np.asarray(vs_h[key], dtype=np.float64), axis=1)
                    s[f'{key}_h_n'] += 1
                s['vs_indep'].append(float(vs_sample['vs_indep'][i]))
                s['vs_pred_fc'].append(float(vs_sample['vs_pred_fc'][i]))
                s['vs_oracle_fc'].append(float(vs_sample['vs_oracle_fc'][i]))

                if psd_pred_sum[sid].shape[1] == 0:
                    psd_pred_sum[sid] = np.zeros_like(psd_p[i])
                    psd_true_sum[sid] = np.zeros_like(psd_t[i])
                psd_pred_sum[sid] += psd_p[i]
                psd_true_sum[sid] += psd_t[i]
                psd_count[sid] += 1

    model.cpu()
    del model, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    subject_psd = {}
    psd_curves_pred, psd_curves_true, per_roi_shape = [], [], []
    for sid in sorted(subj):
        p = psd_pred_sum[sid] / max(psd_count[sid], 1)
        t = psd_true_sum[sid] / max(psd_count[sid], 1)
        m = _psd_metrics_single(p, t, freqs)
        subject_psd[sid] = m
        pn = p[:, freqs > 0] / (p[:, freqs > 0].sum(axis=1, keepdims=True) + EPS)
        tn = t[:, freqs > 0] / (t[:, freqs > 0].sum(axis=1, keepdims=True) + EPS)
        psd_curves_pred.append(pn)
        psd_curves_true.append(tn)
        per_roi_shape.append(m['psd_shape_r'])

    def pval(key):
        return [subject_psd[s][key] for s in subject_psd]

    crps_h = np.stack([subj[s]['crps_h_sum'] / subj[s]['crps_h_n']
                       for s in subj])
    vs_h = {key: np.stack([subj[s][f'{key}_h_sum'] / subj[s][f'{key}_h_n']
                           for s in subj])
            for key in ('vs_indep', 'vs_pred_fc', 'vs_oracle_fc')}
    pit_all = np.concatenate(pit_all_list)
    z_all = np.concatenate(z_all_list)
    pit_h1_all = np.concatenate(pit_h1)
    pit_h8_all = np.concatenate(pit_h8)
    calib = _calibration_block(z_all, pit_all)
    calib['width_90_mean'] = float(np.mean([
        np.mean(subj[s]['width_90']) for s in subj]))
    calib['pit_ks_h1'] = float(kstest(pit_h1_all, 'uniform').statistic)
    calib['pit_ks_h8'] = float(kstest(pit_h8_all, 'uniform').statistic)
    calib['by_horizon'] = {
        'h1': _calibration_block(np.concatenate(z_h1), pit_h1_all),
        'h8': _calibration_block(np.concatenate(z_h8), pit_h8_all),
    }
    nominal, empirical = _coverage_curve(z_all)
    nominal_h1, empirical_h1 = _coverage_curve(np.concatenate(z_h1))
    nominal_h8, empirical_h8 = _coverage_curve(np.concatenate(z_h8))
    metrics = {
        'protocol': 'fixed_anchor_H1_8',
        'split': split,
        'checkpoint': str(ckpt),
        'n_subjects': len(subj),
        'n_samples': int(n_samples),
        'horizon': HORIZON,
        'crps': {
            'h1_8': _agg([np.mean(subj[s]['crps']) for s in subj]),
            'h1': _agg([np.mean(subj[s]['crps_h1']) for s in subj]),
            'h8': _agg([np.mean(subj[s]['crps_h8']) for s in subj]),
            'per_horizon': [_agg(crps_h[:, h]) for h in range(HORIZON)],
            'climatology_N01': _agg([np.mean(subj[s]['crps_clim']) for s in subj]),
            'persistence_N01': _agg([np.mean(subj[s]['crps_pers']) for s in subj]),
            'nll': _agg([np.mean(subj[s]['nll']) for s in subj]),
        },
        'calibration': calib,
        'coverage_curve': {
            'nominal': nominal.tolist(),
            'empirical': empirical.tolist(),
            'h1': {'nominal': nominal_h1.tolist(),
                   'empirical': empirical_h1.tolist()},
            'h8': {'nominal': nominal_h8.tolist(),
                   'empirical': empirical_h8.tolist()},
        },
        'psd': {
            'estimator': 'Welch, Hann, nperseg=8, noverlap=4, fs=0.5 Hz, DC excluded',
            'freq_hz': freqs[freqs > 0].tolist(),
            'n_subjects': len(subject_psd),
            'psd_shape_r': _agg([float(np.nanmean(v))
                                 for v in pval('psd_shape_r')]),
            'psd_roi_r': _agg(pval('psd_roi_r')),
            'psd_flat_r': _agg(pval('psd_flat_r')),
            'psd_mae_norm': _agg(pval('psd_mae_norm')),
            'psd_mae_raw': _agg(pval('psd_mae_raw')),
            'psd_mae_log10': _agg(pval('psd_mae_log10')),
            'rel_band_power_mae': _agg(pval('rel_band_power_mae')),
            'per_anchor': {
                'psd_shape_r': _agg([np.nanmean(subj[s]['psd_shape_anchor'])
                                     for s in subj]),
                'psd_roi_r': _agg([np.nanmean(subj[s]['psd_roi_anchor'])
                                   for s in subj]),
                'psd_flat_r': _agg([np.nanmean(subj[s]['psd_flat_anchor'])
                                    for s in subj]),
                'psd_mae_norm': _agg([np.nanmean(subj[s]['psd_mae_norm_anchor'])
                                      for s in subj]),
                'psd_mae_log10': _agg([np.nanmean(subj[s]['psd_mae_log10_anchor'])
                                       for s in subj]),
                'rel_band_power_mae': _agg([np.nanmean(subj[s]['rel_band_anchor'])
                                            for s in subj]),
            },
        },
        'variogram': {
            'p': 2,
            'definition': 'VS = mean_{i<j} ((y_i-y_j)^2 - E[(X_i-X_j)^2])^2; '
                          'per horizon then subject-equal; 6,670 edges',
            'joint_assumption': 'diagonal Gaussian marginals from VarianceHead; '
                                'cross-ROI correlation from the predicted or true '
                                '8-step trajectory, or set to zero for independence',
            'indep': _agg([np.mean(subj[s]['vs_indep']) for s in subj]),
            'pred_fc': _agg([np.mean(subj[s]['vs_pred_fc']) for s in subj]),
            'oracle_fc': _agg([np.mean(subj[s]['vs_oracle_fc']) for s in subj]),
            'indep_h1': _agg(vs_h['vs_indep'][:, 0]),
            'pred_fc_h1': _agg(vs_h['vs_pred_fc'][:, 0]),
            'oracle_fc_h1': _agg(vs_h['vs_oracle_fc'][:, 0]),
            'indep_h8': _agg(vs_h['vs_indep'][:, -1]),
            'pred_fc_h8': _agg(vs_h['vs_pred_fc'][:, -1]),
            'oracle_fc_h8': _agg(vs_h['vs_oracle_fc'][:, -1]),
        },
    }
    rows = []
    for sid in sorted(subj):
        ps = subject_psd[sid]
        rows.append({
            'group': '', 'subject_id': sid,
            'crps_h1_8': float(np.mean(subj[sid]['crps'])),
            'crps_h1': float(np.mean(subj[sid]['crps_h1'])),
            'crps_h8': float(np.mean(subj[sid]['crps_h8'])),
            'psd_shape_r': float(np.nanmean(ps['psd_shape_r'])),
            'psd_roi_r': ps['psd_roi_r'],
            'psd_flat_r': ps['psd_flat_r'],
            'psd_mae_norm': ps['psd_mae_norm'],
            'psd_mae_log10': ps['psd_mae_log10'],
            'rel_band_power_mae': ps['rel_band_power_mae'],
            'vs_indep': float(np.mean(subj[sid]['vs_indep'])),
            'vs_pred_fc': float(np.mean(subj[sid]['vs_pred_fc'])),
            'vs_oracle_fc': float(np.mean(subj[sid]['vs_oracle_fc'])),
        })
    plots = {
        'freqs': freqs[freqs > 0],
        'psd_pred': np.mean(psd_curves_pred, axis=0),
        'psd_true': np.mean(psd_curves_true, axis=0),
        'per_roi_shape_r': np.nanmean(per_roi_shape, axis=0),
        'pit_h1': np.concatenate(pit_h1),
        'pit_h8': np.concatenate(pit_h8),
        'nominal': nominal, 'empirical': empirical,
        'nominal_h1': nominal_h1, 'empirical_h1': empirical_h1,
        'nominal_h8': nominal_h8, 'empirical_h8': empirical_h8,
        'crps_h': crps_h.mean(axis=0),
        'mu_h8': np.concatenate(mu_h8, axis=0),
        'std_h8': np.concatenate(std_h8, axis=0),
        'y_h8': np.concatenate(y_h8, axis=0),
    }
    return metrics, rows, plots


def collect_rolling(cache_dir):
    cache = Path(cache_dir)
    manifest = json.loads(manifest_path(cache).read_text())
    rows = []
    curves_pred, curves_true, per_roi_shape = [], [], []
    metrics_list = []
    freqs = None
    for sid in manifest['subjects_done']:
        with np.load(cache / f'{sid}.npz', allow_pickle=False) as data:
            pred, true = data['pred'], data['true']
        if pred.shape[0] < 32:
            continue
        freqs, p = welch(pred.T, fs=FS_HZ, nperseg=32, noverlap=16, axis=-1,
                         detrend='constant', scaling='density')
        _, t = welch(true.T, fs=FS_HZ, nperseg=32, noverlap=16, axis=-1,
                     detrend='constant', scaling='density')
        m = _psd_metrics_single(p, t, freqs)
        metrics_list.append(m)
        pn = p[:, freqs > 0] / (p[:, freqs > 0].sum(axis=1, keepdims=True) + EPS)
        tn = t[:, freqs > 0] / (t[:, freqs > 0].sum(axis=1, keepdims=True) + EPS)
        curves_pred.append(pn)
        curves_true.append(tn)
        per_roi_shape.append(m['psd_shape_r'])
        rows.append({
            'group': '', 'subject_id': sid,
            'psd_shape_r': float(np.nanmean(m['psd_shape_r'])),
            'psd_roi_r': m['psd_roi_r'],
            'psd_flat_r': m['psd_flat_r'],
            'psd_mae_norm': m['psd_mae_norm'],
            'psd_mae_raw': m['psd_mae_raw'],
            'psd_mae_log10': m['psd_mae_log10'],
            'rel_band_power_mae': m['rel_band_power_mae'],
        })

    def vals(key):
        return [m[key] for m in metrics_list]

    metrics = {
        'protocol': 'rolling_one_step_full_sequence',
        'cache_dir': str(cache),
        'checkpoint': manifest.get('checkpoint'),
        'n_subjects': len(metrics_list),
        'psd': {
            'estimator': 'Welch, Hann, nperseg=32, noverlap=16, fs=0.5 Hz, DC excluded',
            'freq_hz': freqs[freqs > 0].tolist(),
            'psd_shape_r': _agg([float(np.nanmean(v)) for v in vals('psd_shape_r')]),
            'psd_roi_r': _agg(vals('psd_roi_r')),
            'psd_flat_r': _agg(vals('psd_flat_r')),
            'psd_mae_norm': _agg(vals('psd_mae_norm')),
            'psd_mae_raw': _agg(vals('psd_mae_raw')),
            'psd_mae_log10': _agg(vals('psd_mae_log10')),
            'rel_band_power_mae': _agg(vals('rel_band_power_mae')),
        },
    }
    plots = {
        'freqs': freqs[freqs > 0],
        'psd_pred': np.mean(curves_pred, axis=0),
        'psd_true': np.mean(curves_true, axis=0),
        'per_roi_shape_r': np.nanmean(per_roi_shape, axis=0),
    }
    return metrics, rows, plots


def _psd_figure(records, out_stem, title, short):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), squeeze=False)
    for row, rec in enumerate(records):
        f = rec['freqs']
        for col, (roi, net) in enumerate(ROI_PANELS):
            ax = axes[row, col]
            true = rec['psd_true'][roi - 1]
            pred = rec['psd_pred'][roi - 1]
            ax.plot(f, true, color='0.25', lw=1.6, marker='o', ms=3,
                    label='True')
            ax.plot(f, pred, color=NETWORK_COLORS[net], lw=1.6, ls='--',
                    marker='s', ms=3, label='Predicted')
            ax.set_yscale('log')
            ax.set_xlabel('Frequency (Hz)', fontsize=8)
            ax.set_ylabel('Normalized PSD', fontsize=8)
            ax.set_title(f'ROI {roi} ({net})  r={rec["per_roi_shape_r"][roi - 1]:.3f}',
                         fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, linewidth=0.4)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
        axes[row, 0].text(-0.28, 0.5, f"{rec['group']} test (n={rec['n_subjects']})",
                          transform=axes[row, 0].transAxes, rotation=90,
                          va='center', ha='center', fontsize=10, fontweight='bold')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.text(0.5, 0.015,
             f'{short} PSD is normalized per ROI after removing DC; Welch estimator '
             'with 50% overlap. Shape r is Pearson correlation across frequency bins.',
             ha='center', va='bottom', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0.015, 0.04, 1, 0.95))
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix('.png'), dpi=200)
    fig.savefig(out_stem.with_suffix('.pdf'))
    plt.close(fig)


def _density_figure(records, out_stem):
    grid = np.linspace(-4, 4, 300)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), squeeze=False)
    for row, rec in enumerate(records):
        for col, (roi, net) in enumerate(ROI_PANELS):
            ax = axes[row, col]
            mu = rec['mu_h8'][:, roi - 1]
            sd = rec['std_h8'][:, roi - 1]
            y = rec['y_h8'][:, roi - 1]
            dens = np.exp(-0.5 * ((grid[:, None] - mu[None, :])
                                  / sd[None, :]) ** 2)
            dens = (dens / (sd[None, :] * np.sqrt(2.0 * np.pi))).mean(axis=1)
            ax.hist(y, bins=50, range=(-4, 4), density=True, color='0.55',
                    alpha=0.35, label='True H8')
            ax.plot(grid, dens, color=NETWORK_COLORS[net], lw=1.8,
                    label='Predictive mixture')
            ax.set_xlim(-4, 4)
            ax.set_xlabel('BOLD (z-scored)', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'ROI {roi} ({net}), H8', fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, linewidth=0.4)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
        axes[row, 0].text(-0.28, 0.5, f"{rec['group']} test (n={rec['n_subjects']})",
                          transform=axes[row, 0].transAxes, rotation=90,
                          va='center', ha='center', fontsize=10, fontweight='bold')
    fig.suptitle('Marginal predictive density at H8 - Gaussian VarianceHead',
                 fontsize=13, fontweight='bold')
    fig.text(0.5, 0.015,
             'Solid: equal-weight mixture of N(mu_anchor, sigma_anchor) over all '
             'complete test anchors. Histogram: true H8 values. Anchors overlap, '
             'so the histogram is descriptive, not an independent sample.',
             ha='center', va='bottom', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0.015, 0.04, 1, 0.95))
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix('.png'), dpi=200)
    fig.savefig(out_stem.with_suffix('.pdf'))
    plt.close(fig)


def _pit_figure(records, out_stem):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), squeeze=False)
    bins = np.linspace(0, 1, 21)
    for row, rec in enumerate(records):
        for col, (key, title) in enumerate([('pit_h1', 'PIT H1'),
                                            ('pit_h8', 'PIT H8')]):
            ax = axes[row, col]
            pit = rec[key]
            ax.hist(pit, bins=bins, density=True, color='#3B6EA8',
                    alpha=0.75, edgecolor='white', linewidth=0.3)
            ax.axhline(1.0, color='k', ls='--', lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_xlabel('PIT', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'{title}  KS={rec["pit_ks"][col]:.3f}', fontsize=9)
            ax.tick_params(labelsize=7)
        ax = axes[row, 2]
        ax.plot(rec['nominal_h1'], rec['empirical_h1'], marker='o', ms=2.5,
                color=NETWORK_COLORS['DMN'], lw=1.4, label='H1')
        ax.plot(rec['nominal_h8'], rec['empirical_h8'], marker='s', ms=2.5,
                color=NETWORK_COLORS['FPN'], lw=1.4, label='H8')
        ax.plot(rec['nominal'], rec['empirical'], marker='^', ms=2.5,
                color='0.35', lw=1.2, label='H1-8')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Nominal')
        ax.set_xlabel('Nominal coverage', fontsize=8)
        ax.set_ylabel('Empirical coverage', fontsize=8)
        ax.set_title('Coverage by horizon', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2, linewidth=0.4)
        ax.legend(fontsize=7)
        ax = axes[row, 3]
        ax.plot(np.arange(1, HORIZON + 1), rec['crps_h'], marker='o', ms=3,
                color=NETWORK_COLORS['FPN'], lw=1.5)
        ax.set_xlabel('Horizon', fontsize=8)
        ax.set_ylabel('CRPS', fontsize=8)
        ax.set_title('CRPS by horizon', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
        axes[row, 0].text(-0.28, 0.5, f"{rec['group']} test (n={rec['n_subjects']})",
                          transform=axes[row, 0].transAxes, rotation=90,
                          va='center', ha='center', fontsize=10, fontweight='bold')
    fig.suptitle('Probabilistic calibration - Gaussian marginal forecasts',
                 fontsize=13, fontweight='bold')
    fig.text(0.5, 0.015,
             'PIT should be Uniform(0,1) if calibrated. Coverage and KS are '
             'descriptive; anchors overlap and are not independent.',
             ha='center', va='bottom', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0.015, 0.04, 1, 0.95))
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix('.png'), dpi=200)
    fig.savefig(out_stem.with_suffix('.pdf'))
    plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hc-ckpt', required=True)
    ap.add_argument('--mdd-ckpt', required=True)
    ap.add_argument('--hc-rolling-cache', required=True)
    ap.add_argument('--mdd-rolling-cache', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('results/H1-8当前结果/频谱与概率'))
    ns = ap.parse_args()

    h1_hc, rows_hc, plots_hc = collect_h1_8(ns.hc_ckpt, ns.split, ns.device)
    h1_mdd, rows_mdd, plots_mdd = collect_h1_8(ns.mdd_ckpt, ns.split, ns.device)
    roll_hc = collect_rolling(ns.hc_rolling_cache)
    roll_mdd = collect_rolling(ns.mdd_rolling_cache)

    for group, rows, metrics in (('HC', rows_hc, h1_hc),
                                 ('MDD', rows_mdd, h1_mdd)):
        for row in rows:
            row['group'] = group
    for group, (metrics, rows, plots) in (('HC', roll_hc), ('MDD', roll_mdd)):
        for row in rows:
            row['group'] = group
        metrics['group'] = group
        plots['group'] = group
        plots['n_subjects'] = metrics['n_subjects']

    for metrics, plots, group in ((h1_hc, plots_hc, 'HC'),
                                  (h1_mdd, plots_mdd, 'MDD')):
        metrics['group'] = group
        plots['group'] = group
        plots['n_subjects'] = metrics['n_subjects']
        plots['pit_ks'] = [metrics['calibration']['pit_ks_h1'],
                           metrics['calibration']['pit_ks_h8']]

    out = ns.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {'h1_8_hc': h1_hc, 'h1_8_mdd': h1_mdd,
               'rolling_hc': roll_hc[0], 'rolling_mdd': roll_mdd[0]}
    (out / '频谱与概率指标.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    _write_csv(out / '逐被试_H1-8概率指标.csv', rows_hc + rows_mdd)
    _write_csv(out / '逐被试_滚动单步功率谱.csv',
               roll_hc[1] + roll_mdd[1])

    _psd_figure([plots_hc, plots_mdd], out / 'H1-8功率谱密度被试平均',
                'PSD - H1-8 fixed-anchor protocol',
                'H1-8: 8-TR future chunks; subject-average PSD.')
    _psd_figure([roll_hc[2], roll_mdd[2]], out / '滚动单步功率谱密度全序列',
                'PSD - full-sequence rolling one-step protocol',
                'Rolling: full predicted/true sequence per subject.')
    _density_figure([plots_hc, plots_mdd], out / 'H8预测概率密度')
    _pit_figure([plots_hc, plots_mdd], out / 'H1-8概率校准')

    print(json.dumps({
        'h1_8_hc': {k: h1_hc[k] for k in ('n_subjects', 'n_samples')},
        'h1_8_mdd': {k: h1_mdd[k] for k in ('n_subjects', 'n_samples')},
        'rolling_hc': roll_hc[0]['psd'],
        'rolling_mdd': roll_mdd[0]['psd'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
