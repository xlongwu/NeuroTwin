"""在完全相同的完整 H32 锚点上比较 H8/H32 监督对照。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoint import load_checkpoint_compat
from analysis.next_point_eval import build_next_point_loader, run_inference
from baseline import CONTRACT_FIELDS
from experiments.evaluate_variant import (build_model_from_args, load_condition_bundle,
                                          load_train_args)
from experiments.long_horizon.train_supervision import DEFAULT_CONFIG
from main import split_hash
from utils.common import set_seed

PRIMARY = 'metric_mae_dynamics_h1_32_subject'
SEGMENTS = ((1, 8), (9, 16), (17, 32))


def subject_segment_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                        subjects) -> tuple:
    """[N,F,H] → 被试等权 MAE；所有分段使用同一批完整 H32 锚点。"""
    pred, target, mask = map(np.asarray, (pred, target, mask))
    if (pred.ndim != 3 or target.ndim != 3 or pred.shape[:2] != target.shape[:2]
            or min(pred.shape[2], target.shape[2]) < 32
            or mask.shape != (target.shape[0], target.shape[2])):
        raise ValueError('H32 指标需要匹配的 [N,F,H>=32] 预测/真值与 [N,H] mask')
    if not (mask[:, :32] > 0).all():
        raise ValueError('H32 对照不允许不完整锚点或分段使用不同锚点')
    subjects = np.asarray(subjects).astype(str)
    if len(subjects) != len(pred) or not len(subjects):
        raise ValueError('被试 ID 必须与非空预测逐锚点对齐')
    error = np.abs(pred[:, :, :32].astype(np.float64) - target[:, :, :32])
    if not np.isfinite(error).all():
        raise ValueError('H32 预测或真值包含非有限数值')
    rows, summary = {}, {}
    for lo, hi in ((1, 32), *SEGMENTS):
        key = f'H{lo}_{hi}'
        anchor_mae = error[:, :, lo-1:hi].mean(axis=(1, 2))
        values = {sid: float(anchor_mae[subjects == sid].mean())
                  for sid in sorted(set(subjects))}
        rows[key] = values
        summary[key] = float(np.mean(list(values.values())))
    return summary, rows


def validate_pair(left_args, right_args, left_meta: dict, right_meta: dict) -> None:
    """阻止把历史冻结臂或不同划分/随机源混入正式 H8/H32 对照。"""
    shared_fields = ('seed', 'sampling_seed', 'cond_dropout_seed', 'pretrained_weight',
                     'split_file', 'train_anchor_max_offset', 'eval_anchor_min_future_steps',
                     'dynamics_metric_horizon', 'freeze_backbone_epochs',
                     'eval_context_length', 'eval_anchors_per_subject',
                     'eval_rollout_tasks_per_subject')
    shared_fields = set(shared_fields) | {key for _, key in CONTRACT_FIELDS
                                         if key != 'cpm_horizon'}
    missing = object()
    for name in sorted(shared_fields):
        left, right = getattr(left_args, name, missing), getattr(right_args, name, missing)
        if left is missing or right is missing or left != right:
            raise ValueError(f'对照配置不一致：{name}={left!r}/{right!r}')
    for args, meta, horizon in ((left_args, left_meta, 8), (right_args, right_meta, 32)):
        if (args.cpm_horizon != horizon or args.dynamics_mode != 'graph_ode'
                or not args.dynamics_history_memory or args.s4_variant != 'off'):
            raise ValueError('需要 S2 Graph-ODE 确定性 H8/H32 两臂')
        if (args.train_anchor_max_offset != 32 or args.eval_anchor_min_future_steps != 32
                or args.dynamics_metric_horizon != 32 or args.freeze_backbone_epochs != 0
                or args.eval_rollout_tasks_per_subject != 0):
            raise ValueError('对照需共用 H32 锚点、H1–32 选优并从首轮解冻')
        if meta.get('backbone_unfrozen') is not True or meta.get('ckpt_metric_key') != PRIMARY:
            raise ValueError('checkpoint 必须记录有效解冻且按 H1–32 被试 MAE 选优')
    hashes = [m.get('condition', {}).get('split_hash') for m in (left_meta, right_meta)]
    if not hashes[0] or hashes[0] != hashes[1]:
        raise ValueError(f'checkpoint 被试划分哈希不一致或缺失：{hashes}')


def evaluate_pair(h8, h32, out_dir, split: str = 'test', device: str = 'cpu',
                  config_path=DEFAULT_CONFIG, data_root=None, num_workers: int = 0) -> dict:
    config = json.loads(Path(config_path).read_text())
    paths = {'h8': Path(h8), 'h32': Path(h32)}
    args_by_arm = {arm: load_train_args(path) for arm, path in paths.items()}
    metadata = {arm: torch.load(path, map_location='cpu', weights_only=False)['config']
                for arm, path in paths.items()}
    validate_pair(args_by_arm['h8'], args_by_arm['h32'], metadata['h8'], metadata['h32'])
    report = {'primary_metric': PRIMARY, 'split': split,
              'aggregation': '全 ROI/步数 → 锚点 → 被试内平均 → 被试等权平均',
              'checkpoints': {arm: str(path) for arm, path in paths.items()}, 'arms': {}}
    reference = None
    per_subject = {}
    device = torch.device(device)
    for arm, path in paths.items():
        args = args_by_arm[arm]
        set_seed(args.seed)
        if data_root is not None:
            args.data_root = str(data_root)
        args.num_workers = num_workers
        if (args.eval_context_length != config['evaluation']['context_length']
                or args.eval_anchors_per_subject != config['evaluation']['anchors_per_subject']):
            raise ValueError('checkpoint 选优锚点协议与实验评估配置不一致')
        args.eval_rollout_steps = 32
        bundle = load_condition_bundle(path)
        model = build_model_from_args(args, device, condition_bundle=bundle)
        model, _ = load_checkpoint_compat(model, str(path), device, strict=True)
        if bundle and model.conditioner.layout()['layout_hash'] != bundle['meta']['layout_hash']:
            raise ValueError('评估重建的条件布局与 checkpoint 不一致')
        data_loader, _, loader = build_next_point_loader(args, split, condition_bundle=bundle)
        current_split_hash = split_hash(loader, args)
        if current_split_hash != metadata[arm]['condition']['split_hash']:
            raise ValueError('当前数据划分与训练 checkpoint 不一致')
        if bundle:
            sources = loader.base_dataset.condition_table.audit.alias_sources
            if sources != bundle['meta']['resolved_source_map']:
                raise ValueError('条件来源列与 checkpoint 不一致')
        data = run_inference(model, data_loader, device, offsets=[1], rollout_steps=0,
                             eval_rollout=False, collect_cpm=True, dynamics_horizon=32)
        anchors = list(zip(data['subj_ids'].astype(str).tolist(),
                           data['cutoff'].tolist(), data['context_len'].tolist()))
        if reference is None:
            reference = {key: data[key] for key in ('future', 'future_mask', 'x_last')}
            reference['anchors'] = anchors
            reference['subjects'] = data['subj_ids']
            report['anchor_sha256'] = hashlib.sha256(json.dumps(anchors).encode()).hexdigest()
            report['n_anchors'] = len(anchors)
            report['n_subjects'] = len(set(data['subj_ids']))
            report['split_hash'] = current_split_hash
            report['seeds'] = {k: getattr(args, k) for k in
                               ('seed', 'sampling_seed', 'cond_dropout_seed')}
        elif anchors != reference['anchors'] or any(
                not np.array_equal(data[key], reference[key])
                for key in ('future', 'future_mask', 'x_last')):
            raise ValueError('两臂的锚点、未来真值或可见历史末帧不一致')
        summary, rows = subject_segment_mae(data['ode_pred'], data['future'],
                                             data['future_mask'], data['subj_ids'])
        report['arms'][arm] = {'ODE_MAE': summary, PRIMARY: summary['H1_32'],
                              'CPM_horizon': args.cpm_horizon,
                              'checkpoint_epoch': metadata[arm]['epoch_1based']}
        per_subject[arm] = rows
        # 每次只保留一个完整模型和一臂轨迹，避免双臂同时占用 GPU。
        del data, model, loader, data_loader
    report['delta_h32_minus_h8'] = {
        key: report['arms']['h32']['ODE_MAE'][key] - report['arms']['h8']['ODE_MAE'][key]
        for key in per_subject['h8']}
    persistence = np.broadcast_to(reference['x_last'][:, :, None], reference['future'].shape)
    report['persistence_MAE'], _ = subject_segment_mae(
        persistence, reference['future'], reference['future_mask'], reference['subjects'])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'metrics_{split}.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with (out_dir / f'per_subject_{split}.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'segment', 'h8_mae', 'h32_mae', 'delta_h32_minus_h8'])
        for segment, values in per_subject['h8'].items():
            for sid, left in values.items():
                right = per_subject['h32'][segment][sid]
                writer.writerow([sid, segment, left, right, right-left])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h8', required=True)
    parser.add_argument('--h32', required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data_root', type=Path)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out_dir or Path(json.loads(args.config.read_text())['results_root'])
    report = evaluate_pair(args.h8, args.h32, out_dir, args.split, args.device, args.config,
                           args.data_root, args.num_workers)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
