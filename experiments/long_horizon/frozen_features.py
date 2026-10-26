"""Frozen K64/H32 features for paired CPM adaptation; extraction is opt-in."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analysis.evaluate import (build_model_from_args, load_condition_bundle,
                                          load_train_args)

DEFAULT_CHECKPOINT = Path('archive/checkpoints/h32_best.pt')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_map(fn, value):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(fn, v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    return value


def to_device(value, device):
    return tree_map(lambda t: t.to(device), value)


def slice_features(features: dict, indices) -> dict:
    return tree_map(lambda t: t[indices], features)


def cat_features(chunks: list) -> dict:
    def cat(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: cat([v[key] for v in values]) for key in first}
        if isinstance(first, tuple):
            return tuple(cat([v[i] for v in values]) for i in range(len(first)))
        return torch.cat(values) if torch.is_tensor(first) else first
    return cat(chunks)


def load_model(checkpoint: Path, device):
    args = load_train_args(checkpoint)
    if (args.cpm_horizon != 32 or args.dynamics_mode != 'graph_ode'):
        raise ValueError('需要确定性 Graph-ODE 完整 H32 checkpoint')
    model = build_model_from_args(args, device, load_condition_bundle(checkpoint))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.requires_grad_(False).eval()
    return model, args, payload['config']


def cached_cpm(model, features: dict) -> torch.Tensor:
    pred = model.cpm_head(features['tokens'], model.patch_embed.roi_embed,
                          features['cond'], None)
    return model._denorm_3d(pred, features['norm_stats'])


@torch.no_grad()
def extract_batch(model, batch: dict, device) -> dict:
    """Capture the existing forward inputs, without a second history encoding."""
    from main import dynamics_extras
    captured = {}

    def cpm_hook(module, args):
        captured.update(tokens=args[0], cond=args[2])

    def ode_hook(module, args, kwargs):
        captured.update(z0=args[0], q0=args[1], adj=args[2],
                        profile=kwargs.get('profile'), memory=kwargs.get('memory'))

    x = batch['x'].to(device)
    if not (batch['future_mask'][:, :32] > 0).all():
        raise ValueError('缓存只接受完整 H32 锚点')
    if x.shape[-1] != 64:
        raise ValueError('缓存要求 K64 历史')
    if 'calibration_bold' in batch and (batch['cutoff'] + 1 <
                                       batch['calibration_bold'].shape[-1]).any():
        raise ValueError('个体校准前缀不能超过当前可见历史')
    cal, controls = dynamics_extras(batch, device)
    if controls is not None and controls[:, :32].count_nonzero():
        raise ValueError('本实验只支持无控制自由预测，不能缓存受控锚点')
    condition = ({k: batch[k].to(device) for k in
                  ('cond_z', 'cond_code', 'cond_mask', 'subject_index') if k in batch}
                 if 'cond_z' in batch else None)
    pathology = batch.get('pathology_score')
    hooks = [model.cpm_head.register_forward_pre_hook(cpm_hook),
             model.one_step_head.register_forward_pre_hook(ode_hook, with_kwargs=True)]
    try:
        _, aux = model(x, batch['sc'].to(device),
                       pathology.to(device) if pathology is not None else None,
                       condition=condition, calibration_bold=cal,
                       controls=controls, dynamics_horizon=32)
    finally:
        for hook in hooks:
            hook.remove()
    captured.update(norm_stats=aux.get('norm_stats'),
                    target=batch['future'][:, :, :32].to(device),
                    x_last=batch['x_last'].to(device),
                    ode_pred=aux['dynamics_pred'], cpm_pred=aux['cpm_pred'])
    torch.testing.assert_close(cached_cpm(model, captured), aux['cpm_pred'],
                               rtol=1e-5, atol=1e-6)
    return tree_map(lambda t: t.detach().float().cpu().contiguous(), captured)


def task_id(subject, cutoff: int, context: int = 64) -> str:
    return json.dumps([str(subject), int(cutoff), int(context)], separators=(',', ':'))


def cache_batches(root: Path, split: str):
    """Stream only the active task selection, including after cache expansion."""
    manifest = json.loads((Path(root) / 'manifest.json').read_text())
    spec = manifest['splits'][split]
    active = set(spec['active_ids'])
    for shard in spec['shards']:
        keep = [i for i, key in enumerate(shard['ids']) if key in active]
        if keep:
            features = torch.load(Path(root) / shard['file'], map_location='cpu',
                                  weights_only=False)
            yield slice_features(features, keep), [shard['ids'][i] for i in keep]


def extract(checkpoint: Path, out_dir: Path, splits: list, device: str,
            subjects: int = 64, anchors: int = 4, batch_size: int = 16,
            data_root=None, split_file=None, extend: bool = False) -> None:
    from analysis.next_point_eval import build_next_point_loader
    from main import split_hash
    from utils.common import set_seed

    model, args, meta = load_model(checkpoint, device)
    set_seed(args.seed)
    bundle = load_condition_bundle(checkpoint)
    if getattr(args, 'condition_arm', 'none') != 'none' and not (bundle or {}).get('stats'):
        raise ValueError('缓存必须继承 checkpoint 的 train-only 条件统计量')
    args.data_root = str(data_root or args.data_root)
    args.split_file = str(split_file or args.split_file)
    if not Path(args.split_file).is_file():
        raise ValueError('必须提供现存 split manifest，不允许重新划分')
    args.refresh_split_manifest = False
    args.eval_context_length = 64
    args.eval_anchors_per_subject = anchors
    args.eval_anchor_min_future_steps = 32
    args.eval_rollout_tasks_per_subject = 0
    args.dynamics_metric_horizon = 32
    args.eval_batch_size = batch_size
    args.num_workers = 0
    args.persistent_workers = False
    fingerprint = sha256(checkpoint)
    manifest_path = out_dir / 'manifest.json'
    if out_dir.exists() and not extend:
        raise FileExistsError(f'{out_dir} 已存在；追加缓存须显式 --extend')
    manifest = (json.loads(manifest_path.read_text()) if extend else
                {'version': 1, 'checkpoint_sha256': fingerprint,
                 'seed': args.seed, 'sampling_seed': args.sampling_seed,
                 'dtype': 'float32', 'context_length': 64, 'horizon': 32,
                 'splits': {}})
    if manifest['checkpoint_sha256'] != fingerprint:
        raise ValueError('追加缓存的 checkpoint SHA 不一致')
    for split in splits:
        if split not in ('train', 'val'):
            raise ValueError('初筛不允许读取 test')
        loader, _, owner = build_next_point_loader(args, split, condition_bundle=bundle)
        actual_hash = split_hash(owner, args)
        expected_hash = (meta.get('condition') or {}).get('split_hash')
        if expected_hash and actual_hash != expected_hash:
            raise ValueError('缓存被试划分与 checkpoint 不一致')
        if manifest.get('split_hash', actual_hash) != actual_hash:
            raise ValueError('追加缓存的被试划分不一致')
        manifest['split_hash'] = actual_hash
        view = loader.dataset
        positions = sorted({t[0] for t in view.tasks},
                           key=lambda pos: view.base.subject_ids[pos])
        rng = np.random.default_rng(args.sampling_seed)
        rng.shuffle(positions)
        chosen = set(positions[:subjects] if subjects else positions)
        indices = [i for i, t in enumerate(view.tasks) if t[0] in chosen]
        ids = [task_id(view.base.subject_ids[view.tasks[i][0]], view.tasks[i][1])
               for i in indices]
        spec = manifest['splits'].setdefault(split, {'shards': [], 'active_ids': []})
        previous = {key for shard in spec['shards'] for key in shard['ids']}
        pending = [i for i, key in zip(indices, ids) if key not in previous]
        tokens_bytes = (len(pending) * args.num_rois * (64 // args.tfm_patch_len)
                        * args.tfm_dim * 4)
        print(f'{split}: selected={len(ids)}, new={len(pending)}, '
              f'estimated new tokens={tokens_bytes / 2**20:.2f} MiB', flush=True)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        for batch in DataLoader(Subset(view, pending), batch_size=batch_size, shuffle=False):
            features = extract_batch(model, batch, device)
            batch_ids = [task_id(s, t) for s, t in zip(batch['subj_id'], batch['cutoff'])]
            filename = f'{split}_{len(spec["shards"]):05d}.pt'
            path = out_dir / filename
            if path.exists():
                raise FileExistsError(f'拒绝覆盖缓存分片 {path}')
            size = sum(t.numel() * t.element_size() for t in features.values()
                       if torch.is_tensor(t))
            print(f'{filename}: tokens={tuple(features["tokens"].shape)}, '
                  f'flat tensors={size / 2**20:.2f} MiB', flush=True)
            torch.save(features, path)
            spec['shards'].append({'file': filename, 'ids': batch_ids})
            # A partial run can be extended without encoding completed shards again.
            previous.update(batch_ids)
            spec['active_ids'] = [key for key in ids if key in previous]
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        spec.update(active_ids=ids, subjects_limit=subjects, anchors_per_subject=anchors)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    parser.add_argument('--subjects', type=int, default=64, help='每 split 人数；0=全部')
    parser.add_argument('--anchors', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--extend', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.subjects < 0 or args.anchors < 1 or args.batch_size < 1:
        parser.error('subjects 必须非负，anchors/batch-size 必须为正')
    if not args.execute:
        print(f'Dry run: extract {args.splits}, subjects={args.subjects}, '
              f'anchors={args.anchors}; add --execute to write cache')
        return
    extract(args.checkpoint, args.out_dir, args.splits, args.device, args.subjects,
            args.anchors, args.batch_size, args.data_root, args.split_file, args.extend)


if __name__ == '__main__':
    main()
