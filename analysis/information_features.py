"""Extract input and frozen HC-model states at the existing two disjoint anchors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.h18_data import calibrate_raw
from analysis.individual_pilot import digest
from analysis.shared_mechanism import GROUP_FILES, ROOT
from analysis.twin_reference import describe_history
from inference_twin import TwinSession, load_raw_series


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=GROUP_FILES, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/neurotwin_pretrain/base_best.pt')
    p.add_argument('--mechanism-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    split, condition_file = GROUP_FILES[a.group]
    ids = json.loads((ROOT/'data'/split).read_text())['test']
    source = a.mechanism_dir/a.group
    m = json.loads((source/'数据清单.json').read_text())
    checkpoint_hash = digest(a.checkpoint)
    if m['subjects'] != ids or not m['complete_test'] or m['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Requires the complete original test cohort and same HC checkpoint')
    session = TwinSession(str(a.checkpoint), device=a.device, condition_file=condition_file)
    model = session.model
    if session.bundle['config']['group'] != 'hc3' or session.calibration_length != 32:
        raise ValueError('Requires the current hc3 model with 32 TR calibration')
    dest = a.out_dir/a.group
    dest.mkdir(parents=True, exist_ok=True)
    protocol = dict(subjects=ids, checkpoint_sha256=checkpoint_hash,
                    mechanism_manifest_sha256=digest(source/'数据清单.json'),
                    split_sha256=digest(ROOT/'data'/split),
                    condition_table_sha256=digest(ROOT/'data'/condition_file),
                    source_sha256=digest(Path(__file__)),
                    torch=torch.__version__, device=a.device,
                    windows='same first/last K64 as mechanism cache; per-window features averaged',
                    model='eval; all parameters frozen; own fixed first-32TR profile; hc3 training statistics',
                    label_use='no diagnosis or symptoms used in model feature extraction')
    manifest = dest/'数据清单.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Existing feature protocol differs; choose a new directory')
    manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    upper = np.triu_indices(116, 1)
    with torch.inference_mode():
        for number, sid in enumerate(ids, 1):
            target = dest/f'{sid}.npz'
            if target.exists():
                continue
            raw_path = ROOT/f'data/Raw/{a.group.upper()}/ROISignals_{sid}.mat'
            with np.load(source/f'{sid}.npz') as z:
                if digest(raw_path) != str(z['raw_sha256']):
                    raise ValueError(f'{sid}: raw input has changed')
                cutoffs, expected = z['cutoff'].copy(), z['baseline'].copy()
            series = calibrate_raw(load_raw_series(raw_path, 116), 32)
            histories = np.stack([series[t-63:t+1].T for t in cutoffs])
            x = torch.from_numpy(histories.copy())[:, :, None, :].to(a.device)
            cal = torch.from_numpy(series[:32].T.copy())[None, :, None, :].to(a.device)
            profile = model.calibrate_profile(cal)
            condition = session.condition_for(sid)
            captured = {}

            def input_hook(_module, inputs):
                captured['normalized_history'] = inputs[0].detach()

            def state_hook(_module, inputs, output):
                captured['encoded_state'] = inputs[0].detach()
                captured['advanced_state'] = output[0].detach()

            handles = [model.patch_embed.register_forward_pre_hook(input_hook),
                       model.one_step_head.register_forward_hook(state_hook)]
            try:
                _, aux = model(x, sc_matrix=None, profile=profile.expand(2,-1),
                               condition={k:v.expand(2,*v.shape[1:]) for k,v in condition.items()},
                               dynamics_horizon=1, with_aux=False)
            finally:
                for handle in handles:
                    handle.remove()
            prediction = aux['dynamics_pred'][:,:,0].cpu().numpy()
            discrepancy = float(np.max(abs(prediction-expected)))
            if discrepancy > 1e-5:
                raise ValueError(f'{sid}: baseline changed by {discrepancy}')
            fc = model._history_fc_signed(x[:,:,0,:]).cpu().numpy()
            payload = {'input_history': histories.mean(0).reshape(-1),
                       'input_fc': fc[:,upper[0],upper[1]].mean(0),
                       'input_descriptors': describe_history(histories.transpose(0,2,1)).mean(0).reshape(-1),
                       **{k:v.cpu().numpy().mean(0).reshape(-1) for k,v in captured.items()},
                       'profile':profile[0].cpu().numpy(),
                       'learned_graph':aux['sc_prior']['A_func'].cpu().numpy()[:,upper[0],upper[1]].mean(0),
                       'effective_graph':aux['sc_prior']['A_eff'].cpu().numpy()[:,upper[0],upper[1]].mean(0),
                       'prediction':prediction.mean(0),
                       'prediction_error':abs(prediction-series[cutoffs+1]).mean(0)}
            if any(v.ndim != 1 or not np.isfinite(v).all() for v in payload.values()):
                raise ValueError(f'{sid}: invalid extracted features')
            payload.update(cutoff=cutoffs, raw_sha256=np.array(digest(raw_path)),
                           baseline_max_abs=np.array(discrepancy))
            temp = dest/f'.{sid}.tmp.npz'
            np.savez_compressed(temp, **payload)
            temp.replace(target)
            if number % 25 == 0 or number == len(ids):
                print(f'{a.group}: {number}/{len(ids)}', flush=True)


if __name__ == '__main__':
    main()
