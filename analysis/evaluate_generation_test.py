"""Evaluate priority dynamics and individual metrics on subject-level test splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.fc_sequence_eval import fc_edges
from analysis.metrics import autocorrelation, fc_similarity, gaussian_crps
from analysis.pathology_tests import HC_MERGED_FILE, MDD_MERGED_FILE
from inference_twin import TwinSession, load_raw_series


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--data-root', default=Path('data'), type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--batch-size', default=16, type=int)
    ap.add_argument('--threads', default=2, type=int)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--group', choices=('hc', 'mdd', 'both'), default='both')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    rows, summaries, manifests = [], {}, {}
    for group, split_name, condition_file in (
        ('hc', 'subject_split_pretrain.json', HC_MERGED_FILE),
        ('mdd', 'subject_split_finetune.json', MDD_MERGED_FILE),
    ):
        if args.group != 'both' and args.group != group:
            continue
        split_path = args.data_root / split_name
        split = json.loads(split_path.read_text())
        ids = split['test']
        if len(ids) != len(set(ids)) or set(ids) & (set(split['train']) | set(split['val'])):
            raise ValueError(f'{group}: invalid subject splits')
        session = TwinSession(str(args.checkpoint), device=args.device, condition_file=condition_file)
        histories, targets, conditions = [], [], []
        for sid in ids:
            raw = load_raw_series(args.data_root / 'Raw' / group.upper() / f'ROISignals_{sid}.mat', 116)
            series, _, _ = session.calibrate_scan(raw)
            if len(series) < 96:
                raise ValueError(f'{group} {sid}: fewer than 96 TR')
            histories.append(series[:64].astype(np.float32))
            targets.append(series[64:96].astype(np.float32))
            conditions.append(session.condition_for(sid))
        history, true = np.stack(histories), np.stack(targets)
        predictions, deviations, first_profiles, second_profiles = [], [], [], []
        for lo in range(0, len(ids), args.batch_size):
            hi = min(lo + args.batch_size, len(ids))
            x = torch.from_numpy(history[lo:hi].transpose(0, 2, 1)).to(args.device)[:, :, None, :]
            cond = {k:torch.cat([c[k] for c in conditions[lo:hi]], dim=0)
                    for k in ('cond_z', 'cond_code', 'cond_mask')}
            with torch.no_grad():
                p1 = session.model.calibrate_profile(x[..., :32])
                p2 = session.model.calibrate_profile(x[..., 32:])
                _, result = session.model(x, sc_matrix=None, condition=cond, profile=p1,
                                          dynamics_horizon=32)
            predictions.append(result['dynamics_pred'].cpu().numpy().transpose(0, 2, 1))
            deviations.append(result['dynamics_std'].cpu().numpy().transpose(0, 2, 1))
            first_profiles.append(p1.cpu().numpy())
            second_profiles.append(p2.cpu().numpy())
            print(f'{group}: {hi}/{len(ids)}', flush=True)
        pred, std = np.concatenate(predictions), np.concatenate(deviations)
        first, second = np.concatenate(first_profiles), np.concatenate(second_profiles)
        gen = fc_similarity([fc_edges(p) for p in pred], [fc_edges(t) for t in true])
        observed = fc_similarity([fc_edges(h[32:]) for h in history], [fc_edges(t) for t in true])
        halves = fc_similarity([fc_edges(h[:32]) for h in history], [fc_edges(h[32:]) for h in history])
        distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
        for i, sid in enumerate(ids):
            diff = autocorrelation(pred[i]) - autocorrelation(true[i])
            crps = gaussian_crps(pred[i], std[i], true[i])
            rows.append(dict(group=group, subject=sid, acf_mae_lags_1_5=float(np.nanmean(np.abs(diff))),
                             acf_valid_fraction=float(np.isfinite(diff).mean()),
                             crps_h1=float(crps[0].mean()), crps_h1_32=float(crps.mean()),
                             crps_h17_32=float(crps[16:].mean()),
                             generated_identified=int(np.argmax(gen[i]) == i),
                             observed_identified=int(np.argmax(observed[i]) == i),
                             split_half_identified=int(np.argmax(halves[i]) == i),
                             profile_half_identified=int(np.argmin(distances[i]) == i),
                             profile_half_own_distance=float(distances[i, i]),
                             profile_half_other_distance=float(np.delete(distances[i], i).mean())))
        subset = [r for r in rows if r['group'] == group]
        summaries[group] = {'n': len(ids), 'chance_identification': 1/len(ids),
            **{k:float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ('group', 'subject')}}
        manifests[group] = {'split':str(split_path), 'split_sha256':digest(split_path), 'subjects':len(ids)}
        np.savez_compressed(args.out_dir / f'{group}_results.npz', subjects=np.array(ids),
            pred=pred, true=true, std=std, history=history, profile_first=first, profile_second=second,
            generated_similarity=gen, observed_similarity=observed, half_similarity=halves,
            profile_half_distance=distances)
    with (args.out_dir / 'per_subject.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={'checkpoint':str(args.checkpoint), 'checkpoint_sha256':digest(args.checkpoint),
             'device':args.device, 'splits':manifests, 'groups':summaries}
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
