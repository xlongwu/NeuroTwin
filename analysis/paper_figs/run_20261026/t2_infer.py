"""Read-only K64/H32 inference and bounded ROI importance; never train.

Run from repository root; existing output files are refused unless --resume.
Importance follows next_point_eval's across-sample permutation and batch-level
one-sided t approximation, preserving conditions and calibration for each subject.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.stats import t as student_t
from inference_twin import TwinSession, load_raw_series
from analysis.pathology_tests import HC_MERGED_FILE, MDD_MERGED_FILE, bh_fdr
from models.common import prepare_sc_matrix

ROOT = Path(__file__).resolve().parents[3]
CKPT = Path('checkpoints/neurotwin_pretrain/base_best.pt')


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def generation_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    fc_p, fc_t = np.corrcoef(pred.T), np.corrcoef(true.T)
    iu = np.triu_indices(pred.shape[1], 1)
    f, pp = welch(pred, fs=1.0, nperseg=32, noverlap=16, axis=0)
    _, pt = welch(true, fs=1.0, nperseg=32, noverlap=16, axis=0)
    # DC removed; probability vectors describe spectral shape, not amplitude.
    pp, pt = pp[1:] + 1e-12, pt[1:] + 1e-12
    pp, pt = pp / pp.sum(0), pt / pt.sum(0)
    kl = np.sum(pt * np.log(pt / pp), axis=0).mean()
    return dict(fc_pred=fc_p, fc_true=fc_t, frequency=f[1:], psd_pred=pp,
                psd_true=pt, fc_r=float(np.corrcoef(fc_p[iu], fc_t[iu])[0, 1]),
                psd_kl=float(kl), mae=float(np.abs(pred - true).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, default=CKPT)
    ap.add_argument('--per-group', type=int, default=20)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-importance', action='store_true')
    ns = ap.parse_args()
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    if any(ns.out_dir.iterdir()) and not ns.resume:
        raise FileExistsError('Output directory is nonempty; use --resume to reuse files.')
    torch.set_num_threads(ns.threads)
    torch.manual_seed(2024)
    np.random.seed(2024)
    ckpt_hash = hashlib.sha256(ns.ckpt.read_bytes()).hexdigest()
    pairs = json.loads((ROOT / 'data/splits/site_matched_subset.json').read_text())['pairs']
    # Same-site matched held-out pairs; seeded random selection, no metric selection.
    test_ids = {}
    for group in ('hc', 'mdd'):
        test_ids[group] = set().union(*(set(pd.read_csv(ROOT / f'results/pathology_analysis/run1/z_maps/per_subject_z_{group}_{sp}.csv').subject) for sp in ('val','test')))
    candidates = [p for p in pairs if p['hc'] in test_ids['hc'] and p['mdd'] in test_ids['mdd']]
    rng = np.random.RandomState(20261009)
    chosen = [candidates[i] for i in rng.permutation(len(candidates))[:ns.per_group]]
    if len(chosen) != ns.per_group:
        raise ValueError(f'Only {len(chosen)} matched held-out pairs available.')
    manifest = dict(checkpoint=str(ns.ckpt), checkpoint_sha256=ckpt_hash,
                    model_role='HC reference; forecast_offsets=1 / cpm_horizon=8; H32 descriptive extrapolation',
                    seed=2024, sampling_seed=20261009, K=64, H=32,
                    calibration_length=32, split='val/test', n_pairs=len(chosen),
                    selection='seeded random same-site pairs with both subjects held out (val/test)',
                    pairs=[{k:p[k] for k in ('hc','mdd','site')} for p in chosen],
                    spectral_units='cycles/TR; TR seconds not assumed',
                    spectral_note='Welch nperseg=32/noverlap=16; DC excluded; mean ROI KL(true||pred); only 32 future TR',
                    graph_note='history_fc is functional coupling, not independent anatomical SC')
    if (ns.out_dir / 'manifest.json').exists():
        if json.loads((ns.out_dir / 'manifest.json').read_text()) != manifest:
            raise ValueError('Resume manifest differs from requested run.')
    else:
        write_json(ns.out_dir / 'manifest.json', manifest)
    samples, rows = [], []
    for group, merged in [('hc', HC_MERGED_FILE), ('mdd', MDD_MERGED_FILE)]:
        session = TwinSession(str(ns.ckpt), device='cpu', condition_file=merged)
        for p in chosen:
            sid = p[group]
            target = ns.out_dir / f'{group}_{sid}.npz'
            start = time.monotonic()
            if not target.exists():
                raw = load_raw_series(ROOT / f'data/Raw/{group.upper()}/ROISignals_{sid}.mat', 116)
                session.calibrate_scan(raw)
                session.subject_id = sid
                # TwinSession.init_state currently transposes [K,F] twice. Use its
                # calibration/condition/profile helpers and the training forward directly.
                profile = session.compute_profile()
                cond = session.condition_for(sid)
                cal = torch.from_numpy(session.series[:32].T.astype(np.float32))[None,:,None,:]
                x = torch.from_numpy(session.series[:64].T.astype(np.float32))[None, :, None, :]
                with torch.no_grad():
                    _, aux = session.model(x, sc_matrix=None, condition=cond,
                                           calibration_bold=cal, profile=profile,
                                           dynamics_horizon=32)
                    pred = aux['dynamics_pred'][0].cpu().numpy().T
                    info = aux['sc_prior']
                    history_fc = session.model._history_fc_adj(x[:, :, 0, :])
                    _, prior_norm = prepare_sc_matrix(history_fc, x)
                true = session.series[64:96]
                metrics = generation_metrics(pred, true)
                data = dict(pred=pred, true=true, history=session.series[:64],
                            calibration=cal.cpu().numpy()[0],
                            profile=profile.cpu().numpy()[0],
                            A_eff=info['A_eff'][0].cpu().numpy(),
                            A_func=info['A_func'][0].cpu().numpy(),
                            prior_norm=prior_norm[0].cpu().numpy(),
                            history_fc=history_fc[0].cpu().numpy(),
                            lambda_value=info['lambda'].cpu().numpy().reshape(-1),
                            **metrics)
                data.update({k:v.cpu().numpy()[0] for k,v in cond.items()})
                np.savez_compressed(target, **data)
                print(f'{group} {sid}: {time.monotonic()-start:.2f}s', flush=True)
            with np.load(target) as d:
                samples.append({k:d[k].copy() for k in d.files})
                rows.append(dict(subject=sid, group=group, site=p['site'],
                                 fc_r=float(d['fc_r']), psd_kl=float(d['psd_kl']),
                                 mae=float(d['mae']), lambda_mean=float(d['lambda_value'].mean())))
        del session
    pd.DataFrame(rows).to_csv(ns.out_dir / 'generation_metrics.csv', index=False)
    if not ns.skip_importance and not (ns.out_dir / 'feature_importance_heldout.json').exists():
        # Four batches, ten held-out subjects each (within-group permutations).
        session = TwinSession(str(ns.ckpt), device='cpu', condition_file=HC_MERGED_FILE)
        model = session.model
        batches = []
        for start in range(0, len(samples), 10):
            ss = samples[start:start+10]
            x = torch.from_numpy(np.stack([s['history'].T for s in ss]).astype(np.float32))[:, :, None, :]
            cal = torch.from_numpy(np.stack([s['calibration'] for s in ss]))
            profile = torch.from_numpy(np.stack([s['profile'] for s in ss]))
            cond = {k:torch.from_numpy(np.stack([s[k] for s in ss])) for k in ('cond_z','cond_code','cond_mask')}
            y = torch.from_numpy(np.stack([s['true'][0] for s in ss]).astype(np.float32))
            batches.append((x, y, cal, profile, cond))
        @torch.no_grad()
        def error(x, y, cal, profile, cond):
            out, _ = model(x, sc_matrix=None, calibration_bold=cal, profile=profile,
                           condition=cond, dynamics_horizon=1)
            return float((out[:, :, 0, 0]-y).abs().mean())
        baseline = np.array([error(*b) for b in batches])
        deltas = np.empty((116, len(batches)))
        gen = torch.Generator().manual_seed(2024)
        for roi in range(116):
            for j, (x, y, cal, profile, cond) in enumerate(batches):
                xp = x.clone()
                xp[:,roi] = x[torch.randperm(len(x), generator=gen),roi]
                deltas[roi,j] = error(xp,y,cal,profile,cond)-baseline[j]
            if roi % 20 == 0:
                print(f'ROI importance {roi+1}/116', flush=True)
        mean = deltas.mean(1)
        se = deltas.std(1, ddof=1)/np.sqrt(len(batches))
        pvals = np.ones(116)
        ok = se > 0
        pvals[ok] = student_t.sf(mean[ok]/se[ok],len(batches)-1)
        q = bh_fdr(pvals)
        write_json(ns.out_dir / 'feature_importance_heldout.json',dict(
            roi_importance=mean.tolist(),roi_pvalues=pvals.tolist(),
            roi_pvalues_fdr=q.tolist(),roi_std_errors=se.tolist(),
            roi_importance_rank=np.argsort(mean)[::-1].tolist(),
            significant_roi_indices=(np.flatnonzero(q<0.05)+1).tolist(),
            fdr_alpha=0.05, n_tasks=len(samples), n_batches=len(batches),
            baseline_mae=float(baseline.mean()), seed=2024,
            method='fast: one within-group across-subject shuffle per ROI/batch; one-sided batch t approximation; BH-FDR',
            note='Adapted from analysis.next_point_eval.roi_permutation_importance to retain hc3 conditions, fixed calibration and profiles. Endpoint: primary lag-1 head, not H32. Small-sample structural sensitivity, not clinical localization.'))
        np.savez_compressed(ns.out_dir / 'importance_batch_deltas.npz',deltas=deltas,baseline=baseline)
    print('T2 complete', flush=True)

if __name__ == '__main__':
    main()
