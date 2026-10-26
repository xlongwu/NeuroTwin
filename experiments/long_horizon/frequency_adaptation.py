"""Paired CPM adaptation on frozen features; no real-data work without --execute."""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from analysis.next_point_eval import DEFAULT_DIRECTION_DEADZONE, per_task_head_metrics
from experiments.long_horizon.frozen_features import (
    DEFAULT_CHECKPOINT, cache_batches, cached_cpm, cat_features, load_model,
    sha256, slice_features, to_device)
from train.losses import TFMDualLoss, cross_head_consistency
from utils.common import set_seed


def complex_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean complex residual magnitude, orthonormal H32 rFFT, excluding DC."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 32:
        raise ValueError('频域监督要求同形状 [B,F,32] 完整轨迹')
    residual = torch.fft.rfft(pred.float() - target.float(), dim=-1, norm='ortho')
    return residual[..., 1:].abs().mean()


def fixed_cpm_objective(args, metadata: dict):
    """Keep exactly the CPM-dependent terms of the original objective."""
    values = vars(args)
    aliases = {'huber_delta': 'cpm_huber_delta'}
    kwargs = {name: values[aliases.get(name, name)]
              for name in inspect.signature(TFMDualLoss).parameters
              if aliases.get(name, name) in values}
    for name, cast in (('temporal_lags', int), ('temporal_scale_weights', float)):
        if isinstance(kwargs.get(name), str):
            kwargs[name] = tuple(map(cast, kwargs[name].split(',')))
    weighting = metadata.get('loss_weighting') or {}
    weights = weighting.get('weights') or {}
    if 'cpm' in weights:
        weight = float(weights['cpm'])
        source = 'checkpoint.loss_weighting.weights (frozen)'
    else:
        weight = float(args.lambda_cpm)
        source = 'fixed substitute from checkpoint args; learned weights unavailable'
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('本实验需要已启用且权重为正的 CPM 主项')
    kwargs.update(weighting='fixed', lambda_one=0., lambda_cpm=weight,
                  lambda_dynamics=0., lambda_variance=0.)
    # These are fixed-weight terms in TFMDualLoss, not uncertainty tasks.
    for task in ('temporal_diff', 'cross_head', 'temporal_pcc'):
        if task in weights:
            kwargs[f'lambda_{task}'] = float(weights[task])
    return TFMDualLoss(**kwargs), {'source': source, 'kwargs': kwargs,
                                 'cross_head_epoch': None}


def original_loss(criterion, pred: torch.Tensor, features: dict) -> torch.Tensor:
    one = features['ode_pred'][:, :, :1].unsqueeze(-1)
    target = features['target']
    # Omit ODE-only constant terms, but keep its frozen trajectory as CPM teacher.
    loss, _ = criterion(one, target[:, :, :1].unsqueeze(-1), pred, target,
                        context_last=features['x_last'], epoch=None)
    if criterion.lambda_cross_head > 0:
        loss = loss + criterion.lambda_cross_head * cross_head_consistency(
            pred, features['ode_pred'], eps=criterion.eps)
    return loss


def freeze_for_cpm(model):
    model.requires_grad_(False).eval()
    model.cpm_head.requires_grad_(True).train()
    return list(model.cpm_head.parameters())


def effective_batches(n: int, batch_size: int, updates: int, seed: int):
    """Use every epoch's final partial batch, with no dropped samples."""
    rng = torch.Generator().manual_seed(seed)
    count = 0
    while count < updates:
        order = torch.randperm(n, generator=rng)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]
            count += 1
            if count == updates:
                return


def objective_grad_norms(model, criterion, features, indices, microbatch, device):
    """Accumulate gradients on the same effective calibration batch as training."""
    params = list(model.cpm_head.parameters())
    sums = [[torch.zeros_like(p) for p in params] for _ in range(2)]
    # Calibration must not consume the adaptation arms' dropout RNG stream.
    cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        for start in range(0, len(indices), microbatch):
            chosen = indices[start:start + microbatch]
            batch = to_device(slice_features(features, chosen), device)
            pred = cached_cpm(model, batch)
            losses = (original_loss(criterion, pred, batch),
                      complex_frequency_loss(pred, batch['target']))
            for i, loss in enumerate(losses):
                grads = torch.autograd.grad(loss * (len(chosen) / len(indices)), params,
                                            retain_graph=(i == 0), allow_unused=True)
                for total, grad in zip(sums[i], grads):
                    if grad is not None:
                        total.add_(grad)
    return [float(torch.sqrt(sum(g.double().square().sum() for g in grads)))
            for grads in sums]


def calibrate_coefficient(model, criterion, features, indices, microbatch, device):
    base_norm, freq_norm = objective_grad_norms(
        model, criterion, features, indices, microbatch, device)
    if not all(np.isfinite(v) and v > 0 for v in (base_norm, freq_norm)):
        raise ValueError(f'频域系数校准梯度必须有限且非零：{base_norm}, {freq_norm}')
    return .1 * base_norm / freq_norm, {'original_grad_norm': base_norm,
                                       'frequency_grad_norm': freq_norm,
                                       'target_ratio': .1}


def adapt_arm(model, criterion, features: dict, *, updates: int, batch_size: int,
              microbatch: int, seed: int, lr: float, weight_decay: float,
              grad_clip: float, coefficient: float, device):
    if not 1 <= updates <= 300 or min(batch_size, microbatch) < 1:
        raise ValueError('更新次数须为1–300，batch/microbatch须为正')
    params = freeze_for_cpm(model)
    set_seed(seed)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                  betas=(.9, .95))
    history = []
    n = len(features['tokens'])
    for step, indices in enumerate(effective_batches(n, batch_size, updates, seed), 1):
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(2)
        for start in range(0, len(indices), microbatch):
            chosen = indices[start:start + microbatch]
            batch = to_device(slice_features(features, chosen), device)
            pred = cached_cpm(model, batch)
            base = original_loss(criterion, pred, batch)
            freq = complex_frequency_loss(pred, batch['target'])
            fraction = len(chosen) / len(indices)
            loss = (base + coefficient * freq) * fraction
            if not torch.isfinite(loss):
                raise ValueError(f'update {step}: non-finite loss')
            loss.backward()
            totals += np.array([base.detach().item(), freq.detach().item()]) * fraction
        torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip > 0 else float('inf'),
                                       error_if_nonfinite=True)
        optimizer.step()
        history.append({'update': step, 'original_loss': float(totals[0]),
                        'frequency_loss': float(totals[1])})
        if step == 1 or step % 25 == 0 or step == updates:
            print(f'update={step}/{updates}, original={totals[0]:.6f}, '
                  f'frequency={totals[1]:.6f}', flush=True)
    model.eval()
    return history


def segment_metrics(pred, target, anchor, direction_deadzone):
    """Per-anchor metrics. Constant predictions retain zero PCC on valid targets."""
    mask = np.ones((len(pred), 32))
    dynamic = per_task_head_metrics(pred, target, mask, anchor, direction_deadzone)
    result = {}
    for label, lo, hi in (('H1', 0, 1), ('H1_8', 0, 8), ('H17_32', 16, 32),
                          ('H1_32', 0, 32)):
        p, t = pred[..., lo:hi].astype(np.float64), target[..., lo:hi].astype(np.float64)
        metrics = {'mae': np.abs(p - t).mean(axis=(1, 2)),
                   'delta_mae_step': dynamic['delta_mae_step'][:, lo:hi].mean(axis=1)}
        direction = dynamic['direction_accuracy_step'][:, lo:hi]
        count = np.isfinite(direction).sum(axis=1)
        metrics['direction_accuracy_step'] = np.divide(
            np.nansum(direction, axis=1), count,
            out=np.full(len(p), np.nan), where=count > 0)
        if hi - lo > 1:
            pc, tc = p - p.mean(axis=-1, keepdims=True), t - t.mean(axis=-1, keepdims=True)
            ps, ts = p.std(axis=-1), t.std(axis=-1)
            target_valid, pred_valid = ts > 1e-8, ps > 1e-8
            both = target_valid & pred_valid
            denominator = np.sqrt((pc * pc).sum(-1) * (tc * tc).sum(-1))
            corr = np.divide((pc * tc).sum(-1), denominator,
                             out=np.zeros_like(ps), where=both)
            valid_count = target_valid.sum(axis=1)
            for name, values in (
                    ('temporal_pcc', np.where(target_valid, corr, 0.)),
                    ('std_ratio', np.divide(ps, ts, out=np.zeros_like(ps), where=target_valid))):
                metrics[name] = np.divide(values.sum(axis=1), valid_count,
                                          out=np.full(len(p), np.nan), where=valid_count > 0)
            metrics['pcc_target_valid_fraction'] = target_valid.mean(axis=1)
            metrics['pcc_valid_fraction'] = both.mean(axis=1)
        result[label] = metrics
    return result


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def summarize_subjects(chunks: list, ids: list) -> dict:
    subjects = np.array([json.loads(key)[0] for key in ids])
    output = {}
    for segment in chunks[0]:
        output[segment] = {}
        for metric in chunks[0][segment]:
            values = np.concatenate([c[segment][metric] for c in chunks])
            by_subject = {sid: finite_mean(values[subjects == sid]) for sid in sorted(set(subjects))}
            output[segment][metric] = {'mean': finite_mean(list(by_subject.values())),
                                       'subjects': by_subject}
    return output


def paired_differences(left: dict, right: dict, seed: int) -> dict:
    """right - left, with paired subject bootstrap descriptive intervals."""
    rng = np.random.default_rng(seed)
    output = {}
    for segment in left:
        output[segment] = {}
        for metric in left[segment]:
            a, b = left[segment][metric]['subjects'], right[segment][metric]['subjects']
            differences = {sid: b[sid] - a[sid] for sid in sorted(a)
                           if a[sid] is not None and b[sid] is not None}
            values = np.array(list(differences.values()))
            interval = None
            if len(values):
                boot = rng.choice(values, (1000, len(values)), replace=True).mean(axis=1)
                interval = np.quantile(boot, [.025, .975]).tolist()
            output[segment][metric] = {
                'mean': finite_mean(values), 'interval_95': interval,
                'fraction_positive': float((values > 0).mean()) if len(values) else None,
                'fraction_negative': float((values < 0).mean()) if len(values) else None,
                'subjects': differences}
    return output


def progression_gate(arms: dict, full_val: bool) -> dict:
    def value(arm, segment, metric):
        return arms[arm][segment][metric]['mean']

    def compare(fn, *values):
        return all(v is not None and np.isfinite(v) for v in values) and bool(fn(*values))

    checks = {
        'late_pcc_vs_original_ode': compare(lambda e, o: e - o >= .03,
            value('frequency', 'H17_32', 'temporal_pcc'), value('original_ode', 'H17_32', 'temporal_pcc')),
        'late_pcc_vs_control': compare(lambda e, c: e > c,
            value('frequency', 'H17_32', 'temporal_pcc'), value('control', 'H17_32', 'temporal_pcc')),
        'late_direction_improves': compare(lambda e, o, c: e > o and e > c,
            value('frequency', 'H17_32', 'direction_accuracy_step'),
            value('original_ode', 'H17_32', 'direction_accuracy_step'),
            value('control', 'H17_32', 'direction_accuracy_step'))}
    for segment in ('H1', 'H1_8', 'H17_32', 'H1_32'):
        checks[f'{segment}_mae_guard'] = compare(lambda e, o: e <= 1.01 * o,
            value('frequency', segment, 'mae'), value('original_ode', segment, 'mae'))
    return {'status': ('numeric_gates_passed_review_subjects' if all(checks.values()) else 'stop')
                       if full_val else 'screening_only', 'checks': checks,
            'note': 'std_ratio is descriptive; subject benefit distribution needs inspection'}


@torch.no_grad()
def evaluate_pair(model, cache: Path, heads: dict, device, seed: int,
                  direction_deadzone=DEFAULT_DIRECTION_DEADZONE):
    model.eval()
    chunks = {name: [] for name in ('original_ode', 'original_cpm', 'control', 'frequency')}
    all_ids = []
    for cpu_features, ids in cache_batches(cache, 'val'):
        features = to_device(cpu_features, device)
        target, anchor = cpu_features['target'].numpy(), cpu_features['x_last'].numpy()
        predictions = {'original_ode': cpu_features['ode_pred'].numpy(),
                       'original_cpm': cpu_features['cpm_pred'].numpy()}
        for arm, head in heads.items():
            model.cpm_head.load_state_dict(head, strict=True)
            predictions[arm] = cached_cpm(model, features).cpu().numpy()
        for arm, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError(f'{arm}: non-finite validation predictions')
            chunks[arm].append(segment_metrics(pred, target, anchor, direction_deadzone))
        all_ids.extend(ids)
    if not all_ids:
        raise ValueError('val 缓存为空')
    arms = {arm: summarize_subjects(values, all_ids) for arm, values in chunks.items()}
    manifest = json.loads((cache / 'manifest.json').read_text())
    spec = manifest['splits']['val']
    full_val = (spec['subjects_limit'] == 0 and spec['anchors_per_subject'] == 16
                and len({json.loads(key)[0] for key in all_ids}) == 225)
    return {'arms': arms, 'n_subjects': len({json.loads(k)[0] for k in all_ids}),
            'n_anchors': len(all_ids), 'full_val': full_val,
            'direction_deadzone': direction_deadzone,
            'aggregation': 'ROI/time -> anchor -> within subject -> equal subjects',
            'pcc_rule': 'exclude constant targets; constant predictions score zero; report valid fractions',
            'paired': {name: paired_differences(arms[left], arms[right], seed)
                       for name, left, right in (
                           ('frequency_minus_control', 'control', 'frequency'),
                           ('frequency_minus_original_ode', 'original_ode', 'frequency'),
                           ('control_minus_original_cpm', 'original_cpm', 'control'))},
            'progression': progression_gate(arms, full_val),
            'interval_note': 'descriptive on repeatedly explored val; not independent confirmation'}


def validate_cache(root: Path, fingerprint: str) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['checkpoint_sha256'] != fingerprint:
        raise ValueError('cache/checkpoint SHA 不一致')
    for split, spec in manifest['splits'].items():
        keys = [key for shard in spec['shards'] for key in shard['ids']]
        if (len(keys) != len(set(keys)) or not spec['active_ids']
                or not set(spec['active_ids']).issubset(keys)):
            raise ValueError(f'{split}: 缓存锚点缺失或重复')
    return manifest


def run_pair(checkpoint: Path, cache: Path, out_dir: Path, device: str,
             updates: int = 300, batch_size: int = 64, microbatch: int = 8,
             val_cache=None, pair=None) -> None:
    fingerprint = sha256(checkpoint)
    train_manifest = validate_cache(cache, fingerprint)
    val_cache = Path(val_cache or cache)
    val_manifest = validate_cache(val_cache, fingerprint)
    if train_manifest['split_hash'] != val_manifest['split_hash']:
        raise ValueError('train/val 缓存划分不一致')
    if out_dir.exists():
        raise FileExistsError(f'拒绝覆盖实验结果 {out_dir}')
    model, args, metadata = load_model(checkpoint, device)
    seed = args.seed
    criterion, objective = fixed_cpm_objective(args, metadata)
    criterion.to(device)
    if pair is None:
        train_ids = train_manifest['splits']['train']['active_ids']
        val_ids = val_manifest['splits']['val']['active_ids']
        if {json.loads(k)[0] for k in train_ids} & {json.loads(k)[0] for k in val_ids}:
            raise ValueError('train/val 被试重叠')
        features = cat_features([f for f, _ in cache_batches(cache, 'train')])
        initial = deepcopy(model.cpm_head.state_dict())
        freeze_for_cpm(model)
        set_seed(seed)
        first = next(effective_batches(len(features['tokens']), batch_size, 1, seed))
        coefficient, calibration = calibrate_coefficient(
            model, criterion, features, first, microbatch, device)
        heads, histories = {}, {}
        optimizer_config = {'lr': args.lr_init, 'weight_decay': args.weight_decay,
                            'grad_clip': args.grad_clip}
        for arm, weight in (('control', 0.), ('frequency', coefficient)):
            model.cpm_head.load_state_dict(initial, strict=True)
            print(f'{arm}: coefficient={weight:.8g}', flush=True)
            histories[arm] = adapt_arm(model, criterion, features, updates=updates,
                batch_size=batch_size, microbatch=microbatch, seed=seed,
                coefficient=weight, device=device, **optimizer_config)
            heads[arm] = {k: v.detach().cpu().clone() for k, v in model.cpm_head.state_dict().items()}
        payload = {'heads': heads, 'checkpoint_sha256': fingerprint,
                   'split_hash': train_manifest['split_hash'], 'seed': seed,
                   'objective': objective, 'calibration': calibration, 'coefficient': coefficient,
                   'updates': updates, 'batch_size': batch_size, 'microbatch': microbatch,
                   'optimizer': {**optimizer_config, 'betas': [.9, .95], 'schedule': 'constant'},
                   'selection': 'last update; no val selection; no EMA', 'history': histories,
                   'train_ids': train_ids, 'initial_val_ids': val_ids}
    else:
        payload = torch.load(pair, map_location='cpu', weights_only=False)
        if payload['checkpoint_sha256'] != fingerprint or payload['split_hash'] != val_manifest['split_hash']:
            raise ValueError('配对权重与 checkpoint/val 缓存不一致')
        if {json.loads(k)[0] for k in payload['train_ids']} & {
                json.loads(k)[0] for k in val_manifest['splits']['val']['active_ids']}:
            raise ValueError('train/val 被试重叠')
        heads = payload['heads']
    report = evaluate_pair(model, val_cache, heads, device, seed,
                           getattr(args, 'direction_deadzone', DEFAULT_DIRECTION_DEADZONE))
    report['experiment'] = {k: v for k, v in payload.items() if k not in ('heads', 'history')}
    out_dir.mkdir(parents=True)
    if pair is None:
        torch.save(payload, out_dir / 'pair.pt')
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--val-cache', type=Path)
    parser.add_argument('--pair', type=Path, help='仅评价已有配对权重，不更新参数')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 300 or min(args.batch_size, args.microbatch) < 1:
        parser.error('updates 须为1–300，batch-size/microbatch须为正')
    if not args.execute:
        print(f'Dry run: {"evaluate" if args.pair else "paired adaptation"}, '
              f'updates per arm={args.updates}; add --execute to run')
        return
    run_pair(args.checkpoint, args.cache, args.out_dir, args.device, args.updates,
             args.batch_size, args.microbatch, args.val_cache, args.pair)


if __name__ == '__main__':
    main()
