"""Descriptive ACF, FC fingerprint and Gaussian CRPS on cached H32 samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.fc_sequence_eval import fc_edges
from analysis.metrics import autocorrelation, fc_similarity, gaussian_crps
from analysis.pathology_tests import HC_MERGED_FILE
from inference_twin import TwinSession


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = TwinSession(str(args.checkpoint), device='cpu', condition_file=HC_MERGED_FILE).model
    model.eval()
    summaries, rows = {}, []
    for group in ('hc', 'mdd'):
        paths = sorted(args.cache_dir.glob(f'{group}_*.npz'))
        if len(paths) < 2:
            raise ValueError(f'{group}: need at least two subjects')
        samples = []
        for path in paths:
            with np.load(path) as z:
                samples.append({k:z[k].copy() for k in z.files})
        x = torch.tensor(np.stack([s['history'].T for s in samples]), dtype=torch.float32)[:, :, None, :]
        profile = torch.tensor(np.stack([s['profile'] for s in samples]), dtype=torch.float32)
        cond = {k:torch.tensor(np.stack([s[k] for s in samples])) for k in ('cond_z','cond_code','cond_mask')}
        with torch.no_grad():
            _, out = model(x, sc_matrix=None, condition=cond, profile=profile, dynamics_horizon=32)
            first_profile = model.calibrate_profile(x[..., :32]).cpu().numpy()
            second_profile = model.calibrate_profile(x[..., 32:]).cpu().numpy()
        pred = out['dynamics_pred'].cpu().numpy().transpose(0, 2, 1)
        std = out['dynamics_std'].cpu().numpy().transpose(0, 2, 1)
        true = np.stack([s['true'] for s in samples])
        history = np.stack([s['history'] for s in samples])
        generated_scores = fc_similarity([fc_edges(p) for p in pred], [fc_edges(t) for t in true])
        empirical_scores = fc_similarity([fc_edges(h[32:]) for h in history], [fc_edges(t) for t in true])
        half_scores = fc_similarity([fc_edges(h[:32]) for h in history], [fc_edges(h[32:]) for h in history])
        profile_dist = np.linalg.norm(first_profile[:, None, :] - second_profile[None, :, :], axis=2)
        np.savez_compressed(args.out_dir / f'{group}_similarities.npz',
                            subjects=np.array([p.stem[len(group)+1:] for p in paths]),
                            generated_vs_future=generated_scores, observed_history_vs_future=empirical_scores,
                            observed_first_vs_second_half=half_scores, profile_first=first_profile,
                            profile_second=second_profile, profile_half_distance=profile_dist)
        for i, path in enumerate(paths):
            a, b = autocorrelation(pred[i]), autocorrelation(true[i])
            c = gaussian_crps(pred[i], std[i], true[i])
            rows.append({'group':group, 'subject':path.stem[len(group)+1:],
                         'acf_mae_lags_1_5':float(np.nanmean(np.abs(a-b))),
                         'acf_valid_fraction':float(np.isfinite(a-b).mean()),
                         'crps_h1':float(c[0].mean()), 'crps_h1_32':float(c.mean()),
                         'crps_h17_32':float(c[16:].mean()),
                         'generated_identified':int(np.argmax(generated_scores[i])==i),
                         'observed_identified':int(np.argmax(empirical_scores[i])==i),
                         'split_half_identified':int(np.argmax(half_scores[i])==i),
                         'profile_half_identified':int(np.argmin(profile_dist[i])==i),
                         'profile_half_own_distance':float(profile_dist[i,i]),
                         'profile_half_other_distance':float(np.delete(profile_dist[i],i).mean()),
                         'replication_max_abs':float(np.max(np.abs(pred[i]-samples[i]['pred'])))})
        subset = [r for r in rows if r['group']==group]
        summaries[group] = {'n':len(subset), 'chance_identification':1/len(subset),
                            **{k:float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ('group','subject')}}
    with (args.out_dir / 'per_subject.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    report={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'source_manifest':str(args.cache_dir/'manifest.json'), 'groups':summaries}
    (args.out_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
