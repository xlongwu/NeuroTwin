"""Frozen HC-model, self-excluded cross-ROI sensitivity in both test cohorts."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis.h18_data import calibrate_raw
from utils.condition_schema import file_sha256 as digest
from inference_twin import TwinSession, load_raw_series
from model import load_model_config

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ('SMN', 'DAN', 'LN', 'DMN', 'SN', 'FN', 'VN')
GROUP_FILES = {
    'hc': ('subject_split_pretrain.json', 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'),
    'mdd': ('subject_split_finetune.json', 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'),
}


def nonoverlap_anchors(length: int) -> np.ndarray:
    """First and last legal H1 windows; never silently use overlapping windows."""
    if length < 129:
        raise ValueError(f'T={length}: two disjoint K64 histories and H1 need T>=129')
    return np.array([63, length - 2], dtype=np.int64)


def permutation_indices(n_roi: int = 116) -> np.ndarray:
    """Common random numbers across subjects/groups/windows; preserve each block and last TR."""
    rng = np.random.default_rng(2024)
    indices = np.empty((n_roi, 64), dtype=np.int64)
    for roi in range(n_roi):
        for start in range(0, 64, 8):
            count = 7 if start == 56 else 8
            indices[roi, start:start+count] = start + rng.permutation(count)
        indices[roi, 63] = 63
    return indices


@contextmanager
def fixed_effective_graph(model, graph: dict):
    """Analysis-only hook freezes ALL effective-graph construction, including learned part."""
    def replace(_module, _inputs, _output):
        return graph
    handle = model.sc_prior.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


def network_responses(response: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[source ROI,target ROI] -> 7x7 blocks, omitting ROI self pairs even within network."""
    response = np.asarray(response)
    if response.shape != (len(labels), len(labels)):
        raise ValueError('Response must be square and match atlas labels')
    mask = ~np.eye(len(labels), dtype=bool)
    return np.array([[response[mask & (labels[:, None] == s) & (labels[None, :] == t)].mean()
                      for t in NETWORKS] for s in NETWORKS])


def outgoing_response(response: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Seven source-network means, with equal weight for each of the other 115 targets."""
    mask = ~np.eye(len(labels), dtype=bool)
    return np.array([response[mask & (labels[:, None] == s)].mean() for s in NETWORKS])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=GROUP_FILES, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/load_model_config()['weights']['pretrain']['path'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Pilot only; separate output directory')
    a = p.parse_args()
    if a.batch_size < 1 or a.limit < 0:
        p.error('batch-size must be positive; limit nonnegative')
    split, cond_file = GROUP_FILES[a.group]
    ids = json.loads((ROOT/'data'/split).read_text())['test']
    selected = ids[:a.limit] if a.limit else ids
    protocol = dict(version=1, group=a.group, subjects=selected, complete_test=not bool(a.limit),
                    checkpoint_sha256=digest(a.checkpoint), split_sha256=digest(ROOT/'data'/split),
                    condition_table_sha256=digest(ROOT/'data'/cond_file), condition_group='hc3',
                    atlas_sha256=digest(ROOT/'data/AAL116.xlsx'),
                    source_sha256=digest(Path(__file__)), seed=2024, repeats=1,
                    anchors='first and last legal H1; histories disjoint',
                    permutation='same per-ROI blockwise permutation for all subjects and windows; last TR fixed',
                    arms=['combined', 'fixed_graph'], calibration='first 32 TR fixed scale',
                    primary='seven source-network mean absolute output responses, ROI self excluded',
                    graph_contrast='absolute combined minus fixed-graph predictions; not causal mediation',
                    device=a.device, batch_size=a.batch_size, torch=torch.__version__)
    dest = a.out_dir/a.group
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest/'数据清单.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Existing manifest differs; choose a new output directory')
    manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    session = TwinSession(str(a.checkpoint), device=a.device, condition_file=cond_file)
    # The HC checkpoint determines hc3 encoding and its training statistics for BOTH groups.
    if session.bundle['config']['group'] != 'hc3':
        raise ValueError('Requires HC hc3 CPM checkpoint')
    model = session.model
    indices = permutation_indices()
    begin = time.perf_counter()
    with torch.inference_mode():
        for number, sid in enumerate(selected, 1):
            target = dest/f'{sid}.npz'
            if target.exists():
                continue
            raw_path = ROOT/f'data/Raw/{a.group.upper()}/ROISignals_{sid}.mat'
            series = calibrate_raw(load_raw_series(raw_path, 116), session.calibration_length)
            anchors = nonoverlap_anchors(len(series))
            condition = session.condition_for(sid)

            def forward(x):
                n=len(x)
                pred,aux=model(x,condition={k:v.expand(n,*v.shape[1:]) for k,v in condition.items()})
                return {**aux,'prediction':pred}

            baseline, combined, fixed, history_fc, pert_fc, graph_changes = [], [], [], [], [], []
            sham_max, graph_sham_max = 0., 0.
            for cutoff in anchors:
                x = torch.from_numpy(series[cutoff-63:cutoff+1].T.copy())[None, :, None, :].to(a.device)
                base = forward(x)
                original = base['prediction'][0, :, 0]
                graph = base['sc_prior']
                with fixed_effective_graph(model, graph):
                    sham = forward(x)
                sham_max = max(sham_max, float((sham['prediction'][0, :, 0]-original).abs().max()))
                baseline.append(original.cpu().numpy())
                original_fc = model._history_fc_signed(x[:, :, 0, :])[0]
                history_fc.append(original_fc.cpu().numpy())
                cp, fp, fc = [], [], []
                for start in range(0, 116, a.batch_size):
                    rois = np.arange(start, min(start+a.batch_size, 116))
                    changed = x.repeat(len(rois), 1, 1, 1)
                    for row, roi in enumerate(rois):
                        changed[row, roi, 0] = x[0, roi, 0, indices[roi]]
                    normal = forward(changed)
                    cp.append(normal['prediction'][:, :, 0].cpu().numpy())
                    frozen = {k: v.expand(len(rois), *v.shape[1:]) if torch.is_tensor(v) and v.ndim >= 3
                              else v for k, v in graph.items()}
                    with fixed_effective_graph(model, frozen):
                        control = forward(changed)
                    graph_sham_max = max(graph_sham_max, float((control['sc_prior']['A_eff']-frozen['A_eff']).abs().max()))
                    fp.append(control['prediction'][:, :, 0].cpu().numpy())
                    # FC baseline: only changed source-row correlations to other ROIs.
                    corr = model._history_fc_signed(changed[:, :, 0, :])
                    fc.append(corr[torch.arange(len(rois), device=a.device), torch.as_tensor(rois, device=a.device)].cpu().numpy())
                    graph_changes.append(float((normal['sc_prior']['A_eff']-frozen['A_eff']).abs().max()))
                combined.append(np.concatenate(cp)); fixed.append(np.concatenate(fp)); pert_fc.append(np.concatenate(fc))
            payload = dict(cutoff=anchors, baseline=np.stack(baseline), true=series[anchors+1],
                           combined=np.stack(combined), fixed_graph=np.stack(fixed),
                           history_fc=np.stack(history_fc), perturbed_fc_rows=np.stack(pert_fc),
                           permutation_indices=indices, sham_max=np.array(sham_max),
                           fixed_graph_max_diff=np.array(graph_sham_max),
                           recomputed_graph_max_diff=np.max(graph_changes), raw_sha256=np.array(digest(raw_path)))
            if sham_max > 1e-5 or graph_sham_max != 0 or any(
                    not np.isfinite(v).all() for k, v in payload.items() if k != 'raw_sha256'):
                raise ValueError(f'{sid}: nonfinite results or failed sham/frozen graph check')
            tmp = dest/f'.{sid}.tmp.npz'
            np.savez_compressed(tmp, **payload)
            tmp.replace(target)
            print(f'{a.group} {number}/{len(selected)} elapsed={time.perf_counter()-begin:.1f}s', flush=True)


if __name__ == '__main__':
    main()
