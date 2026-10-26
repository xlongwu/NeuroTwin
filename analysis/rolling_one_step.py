# coding=utf-8
"""v2 密集滚动单步评估器（B1+B2）：护栏指标与偏离信号的共同数据源。

协议（v2 研究计划 §3，与 checkpoint 正式口径对齐）：

- 模型：HC 参照 ``checkpoints/neurotwin_pretrain/base_best.pt``
  （从 ``config.args`` 快照 strict 重建，条件统计量 inherit，不重拟合）；
- 数据：``NeuroTwinNextPointDataset`` raw 分支（``bold_source=raw``、
  ``calibration_length=32``、``graph_source=history_fc``，均取自快照）；
  HC 用预训练划分 manifest（split_hash 已核对），MDD 用
  ``data/subject_split_finetune.json``，条件仍为 hc3 三字段（MDD merged 表
  含同名列 Sex/Age/education），统计量继承 HC 快照；
- 锚点：K=64（``--context_length``），每被试枚举全部合法锚点
  t ∈ [max(K-1, calib-1), T-2]，每锚点只取 ODE 头 ``dynamics_pred`` 的 lag-1 帧
  （``dynamics_horizon=1``，不做 32 步积分）；
- persistence 基线内建于同一次遍历（``x_last`` 即 persistence 预测），零额外前向。

逐被试缓存（断点可续跑）：``<out_dir>/<group>/<subj>.npz``（err_abs / pred /
true / x_last / cutoff，float32）+ ``manifest.json``（checkpoint SHA、K、锚点规则、
方向死区、dtype）。汇总层（B2 护栏指标族，供 B3/B4 消费）：

- 逐被试逐 ROI：``mae_model / mae_persistence / mse_model / mse_persistence /
  improvement / delta_corr / dir_acc / n_valid_delta``；
- 产物：``per_subject_<split>.csv``（被试级标量）、
  ``per_subject_roi_mae_<split>.npz``（subjects × ROI 的 mae_model/mae_persistence，
  供 normative_deviation 拟合与门槛检查）、``summary_<split>.json``。

口径：improvement = 1 − mae_model/mae_persistence（判据 1）；delta_corr 为该被试
该 ROI 上 corr(pred−x_last, true−x_last)（判据 2，有效变化点 ≥ ``min_points``
才计算，记录有效率）；dir_acc 为 sign 一致率（真实变化 |Δ| ≤ deadzone 的点不计，
与 ``analysis/next_point_eval.py`` 的方向死区同口径）。恒定/近恒定序列的
delta_corr / dir_acc / improvement 记 NaN 并计入无效池，不填理想值。

用法::

    # 5 被试冒烟（HC val）
    python -m analysis.rolling_one_step \
      --ckpt checkpoints/neurotwin_pretrain/base_best.pt \
      --group hc --split val --limit 5 --device cuda:0
    # 全量：hc train/val/test 与 mdd train/val/test 依次执行
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

#: 方向指标死区（BOLD 已逐 ROI z-score；与 next_point_eval 同口径）
DEFAULT_DEADZONE = 0.05
#: delta_corr 的最少有效变化点数（v2 §3）
DEFAULT_MIN_POINTS = 8
#: 逐被试缓存目录（相对 out_dir）
GROUP_SUBDIR = {'hc': 'hc', 'mdd': 'mdd'}
EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════
#  B2 护栏指标（纯 numpy，可独立测试）
# ══════════════════════════════════════════════════════════════════════════

def guardrail_metrics(pred: np.ndarray, true: np.ndarray, x_last: np.ndarray,
                      min_points: int = DEFAULT_MIN_POINTS,
                      deadzone: float = DEFAULT_DEADZONE) -> Dict[str, np.ndarray]:
    """单被试的逐 ROI 护栏指标族。

    Args:
        pred/true/x_last: ``[n_anchor, F]`` 模型 lag-1 预测 / 真值 / 锚点末位
        min_points: delta_corr 需要的最少有效变化点（|true−x_last| > deadzone）
        deadzone: 方向 / 有效性判定的死区

    Returns:
        逐 ROI 数组 ``[F]``：mae_model, mae_persistence, mse_model,
        mse_persistence, improvement, delta_corr, dir_acc, n_valid_delta,
        dir_eligible_fraction。无效量（恒定序列、无有效变化点）为 NaN，
        不产生告警或理想值。
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    x_last = np.asarray(x_last, dtype=np.float64)
    if not (pred.shape == true.shape == x_last.shape) or pred.ndim != 2:
        raise ValueError(
            f'pred/true/x_last 期望同形 [n_anchor, F]，收到 '
            f'{pred.shape}/{true.shape}/{x_last.shape}')
    n, f = pred.shape
    err = pred - true
    mae_model = np.abs(err).mean(axis=0)
    mae_pers = np.abs(true - x_last).mean(axis=0)
    mse_model = (err ** 2).mean(axis=0)
    mse_pers = ((true - x_last) ** 2).mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        improvement = 1.0 - mae_model / mae_pers
    improvement = np.where(mae_pers > EPS, improvement, np.nan)

    d_pred = pred - x_last                       # [n, F]
    d_true = true - x_last
    valid = np.abs(d_true) > float(deadzone)     # 有效变化点
    n_valid = valid.sum(axis=0).astype(np.float64)
    eligible_frac = n_valid / float(n)

    # delta_corr：仅在有效变化点 >= min_points 且两序列非恒定时定义
    delta_corr = np.full(f, np.nan)
    dp_c = d_pred - d_pred.mean(axis=0, keepdims=True)
    dt_c = d_true - d_true.mean(axis=0, keepdims=True)
    denom = np.sqrt((dp_c ** 2).sum(axis=0) * (dt_c ** 2).sum(axis=0))
    corr_all = np.where(denom > EPS, (dp_c * dt_c).sum(axis=0) / np.maximum(denom, EPS),
                        np.nan)
    delta_corr = np.where(n_valid >= int(min_points), corr_all, np.nan)

    agree = (np.sign(d_pred) == np.sign(d_true)) & valid
    dir_acc = np.where(n_valid > 0, agree.sum(axis=0) / np.maximum(n_valid, 1), np.nan)

    return {
        'mae_model': mae_model, 'mae_persistence': mae_pers,
        'mse_model': mse_model, 'mse_persistence': mse_pers,
        'improvement': improvement, 'delta_corr': delta_corr,
        'dir_acc': dir_acc, 'n_valid_delta': n_valid,
        'dir_eligible_fraction': eligible_frac,
    }


def subject_scalar_summary(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """逐 ROI 指标 → 被试级标量（无效 ROI 自然排除，记录有效 ROI 数）。"""
    out: Dict[str, float] = {}
    with np.errstate(invalid='ignore'):
        for key in ('mae_model', 'mae_persistence', 'mse_model', 'mse_persistence',
                    'improvement', 'delta_corr', 'dir_acc'):
            vals = np.asarray(metrics[key], dtype=np.float64)
            ok = np.isfinite(vals)
            out[key] = float(vals[ok].mean()) if ok.any() else float('nan')
        out['improvement_subject'] = float(
            1.0 - metrics['mae_model'].mean() / metrics['mae_persistence'].mean())
        out['delta_corr_valid_fraction'] = float(
            np.isfinite(metrics['delta_corr']).mean())
        out['dir_acc_valid_fraction'] = float(
            np.isfinite(metrics['dir_acc']).mean())
    return out


# ══════════════════════════════════════════════════════════════════════════
#  checkpoint / 数据重建（复用 evaluate 的严格路径）
# ══════════════════════════════════════════════════════════════════════════

def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _mdd_condition_kwargs(bundle) -> dict:
    """MDD 侧条件参数：hc3 三字段 + HC 快照统计量 inherit（v2 §2）。"""
    from analysis.evaluate import condition_stats_from_bundle
    if bundle is None or not bundle.get('stats'):
        raise ValueError(
            'checkpoint 缺少 config.condition.stats，无法按 hc3 inherit 重建条件编码。')
    stats = condition_stats_from_bundle(bundle, 'hc3')
    return {
        'condition_group': 'hc3',
        # condition_file=None → finetune 模式默认 MDD merged 表（含 Sex/Age/education）
        'condition_file': None,
        'condition_base_stats': stats,
        'condition_stats_mode': 'inherit',
        'condition_control': 'none',
    }


def get_split_subjects(loader, split: str) -> List[str]:
    if split == 'train':
        return list(loader.get_train_subjects())
    if split == 'val':
        return list(loader.get_val_subjects())
    if split == 'test':
        return list(loader.get_test_subjects())
    raise ValueError(f"split 仅支持 train/val/test，收到 '{split}'")


def build_dense_view(loader, subjects: Sequence[str], context_length: int,
                     anchor_stride: int = 1):
    """全锚点确定性评估视图（NextPointEvalView anchors_per_subject=0）。"""
    from utils.dataloader import NextPointEvalView
    indices = loader._expand_subjects_to_sample_indices(list(subjects))
    view = NextPointEvalView(
        loader.base_dataset, indices,
        context_length=int(context_length),
        anchors_per_subject=0, rollout_tasks_per_subject=0, rollout_steps=0,
        all_future_steps=1, anchor_min_future_steps=0)
    if int(anchor_stride) > 1:
        stride = int(anchor_stride)
        # 同被试锚点连续存放（视图按被试逐个枚举），降级选项：隔 stride 取一个
        tasks, seen = [], {}
        for task in view.tasks:
            subj_pos = task[0]
            seen[subj_pos] = seen.get(subj_pos, 0) + 1
            if (seen[subj_pos] - 1) % stride == 0:
                tasks.append(task)
        view.tasks = tasks
        view.labels = [context_length] * len(tasks)
    return view


# ══════════════════════════════════════════════════════════════════════════
#  逐被试前向 + 缓存
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_subjects(model, view, subjects: Sequence[str], device,
                     batch_size: int, num_workers: int, out_dir: Path,
                     group: str, manifest: dict) -> List[Path]:
    """逐被试前向并落盘 npz 缓存（已存在且 manifest 兼容的被试跳过）。

    返回本次写入/复用的 npz 路径列表。每被试锚点按视图顺序成批前向；
    ODE lag-1 = ``aux['dynamics_pred'][:, :, 0]``。
    """
    from main import dynamics_extras
    model.eval()
    group_dir = out_dir / GROUP_SUBDIR[group]
    group_dir.mkdir(parents=True, exist_ok=True)

    by_subject: Dict[str, Dict[str, list]] = {}
    dl = DataLoader(view, batch_size=int(batch_size), shuffle=False,
                    num_workers=int(num_workers))
    for batch in dl:
        x = batch['x'].to(device, non_blocking=True)              # [B,F,1,K]
        sc = batch['sc'].to(device, non_blocking=True)
        cond = None
        if 'cond_z' in batch:
            cond = {'cond_z': batch['cond_z'].to(device, non_blocking=True),
                    'cond_code': batch['cond_code'].to(device, non_blocking=True),
                    'cond_mask': batch['cond_mask'].to(device, non_blocking=True)}
            if 'subject_index' in batch:
                cond['subject_index'] = batch['subject_index'].to(device,
                                                                  non_blocking=True)
        cal, ctl = dynamics_extras(batch, device)
        pred, aux = model(x, sc, None, condition=cond, controls=None,
                          calibration_bold=cal, dynamics_horizon=1)
        dyn = aux.get('dynamics_pred', None) if isinstance(aux, dict) else None
        if not torch.is_tensor(dyn) or dyn.shape[:2] != x.shape[:2] or dyn.shape[2] < 1:
            raise ValueError(
                f"checkpoint 前向未返回可用 dynamics_pred（形状 "
                f"{tuple(dyn.shape) if torch.is_tensor(dyn) else type(dyn)}）；"
                'rolling_one_step 依赖 graph_ode 路线的 ODE 头 lag-1 输出。')
        lag1 = dyn[:, :, 0].float().cpu().numpy()                  # [B,F]
        true = batch['y'][:, :, 0].float().numpy()                 # [B,F]（offset=1）
        x_last = batch['x_last'].float().numpy()                   # [B,F]
        err_abs = np.abs(lag1 - true)
        for i, sid in enumerate(batch['subj_id']):
            entry = by_subject.setdefault(str(sid), {'err': [], 'pred': [],
                                                     'true': [], 'last': [],
                                                     'cutoff': []})
            entry['err'].append(err_abs[i])
            entry['pred'].append(lag1[i])
            entry['true'].append(true[i])
            entry['last'].append(x_last[i])
            entry['cutoff'].append(int(batch['cutoff'][i].item()))

    written: List[Path] = []
    for sid in [str(s) for s in subjects]:
        entry = by_subject.get(sid)
        if not entry:
            raise ValueError(f'评估视图中没有被试 {sid} 的锚点（数据/划分不一致）')
        path = group_dir / f'{sid}.npz'
        if path.exists():
            log.info('[rolling] 跳过已缓存被试 %s', path)
            written.append(path)
            continue
        payload = {
            'err_abs': np.stack(entry['err']).astype(np.float32),
            'pred': np.stack(entry['pred']).astype(np.float32),
            'true': np.stack(entry['true']).astype(np.float32),
            'x_last': np.stack(entry['last']).astype(np.float32),
            'cutoff': np.asarray(entry['cutoff'], dtype=np.int32),
        }
        np.savez_compressed(path, **payload)
        written.append(path)
    _update_manifest(out_dir / GROUP_SUBDIR[group] / 'manifest.json', manifest,
                     [p.stem for p in written])
    return written


def _protocol_block(manifest: dict) -> dict:
    return {k: manifest[k] for k in ('checkpoint_sha256', 'context_length',
                                     'calibration_length', 'bold_source',
                                     'graph_source', 'anchor_rule', 'dtype',
                                     'deadzone', 'min_points')}


def _update_manifest(path: Path, manifest: dict, new_subjects: List[str]) -> None:
    """组目录 manifest：协议块不一致即报错（防止混用不同协议的缓存）。"""
    proto = _protocol_block(manifest)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if _protocol_block(old) != proto:
            raise RuntimeError(
                f'{path} 已存在且协议与本次运行不一致（{old["checkpoint_sha256"][:12]}… '
                f'vs {proto["checkpoint_sha256"][:12]}… 等）；请更换 --out_dir '
                '或删除旧缓存，不允许混用不同协议的逐被试缓存。')
        done = set(old.get('subjects_done', []))
    else:
        done = set()
    done.update(new_subjects)
    path.write_text(json.dumps(
        {**manifest, 'subjects_done': sorted(done)},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


# ══════════════════════════════════════════════════════════════════════════
#  汇总层
# ══════════════════════════════════════════════════════════════════════════

def summarize_group(out_dir: Path, group: str, split: str,
                    deadzone: float, min_points: int,
                    subjects_filter: Optional[Sequence[str]] = None) -> dict:
    """消费逐被试 npz → per_subject CSV / ROI-MAE 矩阵 / summary JSON。

    ``subjects_filter`` 传入该 split 的完整被试名单（组目录缓存可含多 split），
    只统计名单内的被试；None 时统计全部缓存（仅冒烟小目录使用）。
    """
    import csv
    group_dir = out_dir / GROUP_SUBDIR[group]
    manifest_path = group_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'缺少 {manifest_path}；请先运行 rolling_one_step。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    # 只把逐被试缓存当输入：排除本函数自己产出的 ROI-MAE 汇总 npz
    npzs = sorted(p for p in group_dir.glob('*.npz')
                  if not p.name.startswith('per_subject_roi_mae'))
    if subjects_filter is not None:
        members = {str(s) for s in subjects_filter}
        npzs = [p for p in npzs if p.stem in members]
    if not npzs:
        raise ValueError(f'{group_dir} 下没有逐被试缓存。')

    rows, mae_m, mae_p, subj_ids = [], [], [], []
    n_anchor_list = []
    for npz in npzs:
        with np.load(npz) as z:
            pred, true, x_last = z['pred'], z['true'], z['x_last']
        metrics = guardrail_metrics(pred, true, x_last,
                                    min_points=min_points, deadzone=deadzone)
        scalars = subject_scalar_summary(metrics)
        rows.append({'subject': npz.stem, 'n_anchors': int(pred.shape[0]),
                     **{k: round(v, 6) for k, v in scalars.items()}})
        mae_m.append(metrics['mae_model'])
        mae_p.append(metrics['mae_persistence'])
        subj_ids.append(npz.stem)
        n_anchor_list.append(int(pred.shape[0]))

    mae_m = np.stack(mae_m).astype(np.float32)     # [N_subj, F]
    mae_p = np.stack(mae_p).astype(np.float32)
    stem = f'per_subject_roi_mae_{split}'
    np.savez_compressed(group_dir / f'{stem}.npz', subjects=np.asarray(subj_ids),
                        mae_model=mae_m, mae_persistence=mae_p)

    csv_path = group_dir / f'per_subject_{split}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    imp_subj = 1.0 - mae_m.mean(axis=1) / mae_p.mean(axis=1)
    summary = {
        'group': group, 'split': split,
        'n_subjects_cached': len(subj_ids),
        'n_subjects_this_split': len(subj_ids),
        'anchor_total': int(sum(n_anchor_list)),
        'anchors_per_subject_min_median_max': [
            int(min(n_anchor_list)), int(np.median(n_anchor_list)),
            int(max(n_anchor_list))],
        'deadzone': deadzone, 'min_points': min_points,
        'mae_model_subject_equal_weight': float(mae_m.mean(axis=1).mean()),
        'mae_persistence_subject_equal_weight': float(mae_p.mean(axis=1).mean()),
        'improvement_subject_equal_weight': float(imp_subj.mean()),
        'improvement_median': float(np.median(imp_subj)),
        'dir_acc_mean': float(np.nanmean([r['dir_acc'] for r in rows])),
        'delta_corr_mean': float(np.nanmean([r['delta_corr'] for r in rows])),
        'delta_corr_valid_fraction_mean': float(
            np.mean([r['delta_corr_valid_fraction'] for r in rows])),
        'manifest_protocol': _protocol_block(manifest),
        'note': 'CSV/summary/ROI-MAE 矩阵均已按该 split 的划分名单过滤；'
                'B3/B4 直接消费 ROI-MAE 矩阵。',
    }
    (group_dir / f'summary_{split}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> None:
    from analysis.checkpoint import load_checkpoint_compat
    from analysis.evaluate import (build_model_from_args,
                                              load_condition_bundle,
                                              load_train_args)
    from utils.common import set_seed

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='HC 参照 checkpoint（严格重建）')
    ap.add_argument('--group', required=True, choices=['hc', 'mdd'])
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'],
                    help='HC 用预训练划分；MDD 用 subject_split_finetune.json')
    ap.add_argument('--context_length', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--device', default='',
                    help='如 cuda:0；留空自动选择 GPU/CPU')
    ap.add_argument('--out_dir', type=Path, default=Path('results/rolling_eval'))
    ap.add_argument('--data_root', type=Path, default=None,
                    help='覆盖 checkpoint 快照中的 data_root（迁移机器时使用）')
    ap.add_argument('--limit', type=int, default=0,
                    help='只处理前 N 个被试（冒烟）；0 = 全部')
    ap.add_argument('--anchor_stride', type=int, default=1,
                    help='降级选项：隔 N 取锚点（记录进 manifest）；默认 1 = 全锚点')
    ap.add_argument('--deadzone', type=float, default=DEFAULT_DEADZONE)
    ap.add_argument('--min_points', type=int, default=DEFAULT_MIN_POINTS)
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    device = torch.device(ns.device) if ns.device else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    args = load_train_args(ns.ckpt)
    bundle = load_condition_bundle(ns.ckpt)
    data_root = Path(ns.data_root) if ns.data_root else Path(args.data_root)
    set_seed(int(args.seed))

    model = build_model_from_args(args, device, condition_bundle=bundle)
    model, compat = load_checkpoint_compat(model, ns.ckpt, device, strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')
    if getattr(model, 'conditioner', None) is not None:
        saved_hash = (bundle or {}).get('meta', {}).get('layout_hash')
        now_hash = model.conditioner.layout()['layout_hash']
        if saved_hash and saved_hash != now_hash:
            raise RuntimeError(
                f'条件布局不一致：checkpoint={saved_hash}，重建后={now_hash}。')

    if ns.group == 'mdd':
        # MDD 条件统计量继承 HC 快照（hc3 三字段，不重拟合）
        loader_kwargs = _mdd_condition_kwargs(bundle)
    else:
        from main import condition_loader_kwargs
        loader_kwargs = condition_loader_kwargs(args, bundle)

    from utils.dataloader import NeuroTwinDataLoader  # noqa: F401（类型提示用）
    loader = _build_loader_with_condition(args, ns.group, data_root,
                                          ns.context_length, ns.batch_size,
                                          ns.num_workers, loader_kwargs)
    subjects = get_split_subjects(loader, ns.split)
    todo = subjects[:ns.limit] if ns.limit else subjects
    log.info('[rolling] group=%s split=%s | 划分被试 %d | 本次处理 %d | 设备 %s',
             ns.group, ns.split, len(subjects), len(todo), device)

    view = build_dense_view(loader, todo, ns.context_length, ns.anchor_stride)
    manifest = {
        'checkpoint': str(ns.ckpt),
        'checkpoint_sha256': sha256(ns.ckpt),
        'group': ns.group, 'mode': ('pretrain' if ns.group == 'hc' else 'finetune'),
        'context_length': int(ns.context_length),
        'calibration_length': int(getattr(args, 'calibration_length', 0) or 0),
        'bold_source': str(getattr(args, 'bold_source', 'auto')),
        'graph_source': str(getattr(args, 'graph_source', 'legacy_mask')),
        'anchor_rule': ('全部合法锚点 t∈[max(K-1,calib-1), T-2]，'
                        + (f'隔 {ns.anchor_stride} 取一' if ns.anchor_stride > 1
                           else '不抽稀')),
        'dtype': 'float32', 'deadzone': float(ns.deadzone),
        'min_points': int(ns.min_points),
        'condition_group': 'hc3',
        'lag_output': "aux['dynamics_pred'][:, :, 0]（ODE 头 lag-1）",
    }
    written = forward_subjects(model, view, todo, device, ns.batch_size,
                               ns.num_workers, ns.out_dir, ns.group, manifest)
    summary = summarize_group(ns.out_dir, ns.group, ns.split,
                              ns.deadzone, ns.min_points,
                              subjects_filter=subjects)
    log.info('[rolling] 完成 | 写入/复用 %d 被试 | 锚点总数 %d | 模型 MAE %.6f | '
             'persistence MAE %.6f | improvement %.4f',
             len(written), summary['anchor_total'],
             summary['mae_model_subject_equal_weight'],
             summary['mae_persistence_subject_equal_weight'],
             summary['improvement_subject_equal_weight'])
    print(json.dumps({k: summary[k] for k in (
        'group', 'split', 'n_subjects_cached', 'anchor_total',
        'mae_model_subject_equal_weight', 'mae_persistence_subject_equal_weight',
        'improvement_subject_equal_weight', 'dir_acc_mean', 'delta_corr_mean',
        'delta_corr_valid_fraction_mean')}, ensure_ascii=False, indent=2))


def _build_loader_with_condition(args, group, data_root, context_length,
                                 batch_size, num_workers, condition_kwargs):
    """build_loader 的包装：注入组特定的条件 kwargs（hc=快照推导，mdd=hc3 inherit）。"""
    from main import parse_pathology_fields
    from utils.common import parse_int_list, resolve_forecast_offsets
    from utils.dataloader import NeuroTwinDataLoader

    if group == 'hc':
        mode = str(args.mode)
    else:
        mode = 'finetune'
        condition_kwargs = {**condition_kwargs,
                            'split_file': str(Path(data_root)
                                              / 'subject_split_finetune.json'),
                            # finetune 模式的划分后诊断块要求分层键在条件 schema 内
                            # （默认 HAMD 不在 hc3 里）；划分以 split_file 为准，
                            # 该键只影响一行分布打印，取 hc3 内的 Age。
                            'stratify_key': 'Age'}
    condition_kwargs.setdefault('split_file',
                                (str(getattr(args, 'split_file', '') or '')
                                 or None))

    return NeuroTwinDataLoader(
        data_root=str(data_root), mode=mode, batch_size=batch_size,
        pathology_input_dim=int(getattr(args, 'pathology_input_dim', 1)),
        pathology_fields=parse_pathology_fields(getattr(args, 'pathology_fields', None)),
        pathology_missing=str(getattr(args, 'pathology_missing', 'drop')),
        clinical_file=str(getattr(args, 'clinical_file',
                                  'Rest-meta-MDD-V1V2-Merged-MDD.xlsx')),
        total_windows=int(getattr(args, 'total_windows', 9)),
        seq_len=int(getattr(args, 'seq_len', 30)),
        num_workers=num_workers, pin_memory=False,
        seed=int(args.seed), val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        stratify_bins=int(getattr(args, 'stratify_bins', 5)),
        cache_in_memory=False, persistent_workers=False,
        prefetch_factor=(2 if num_workers > 0 else None),
        refresh_split_manifest=False,
        task_mode='next_timepoint',
        bold_source=str(getattr(args, 'bold_source', 'auto')),
        train_samples_per_subject=0,
        subject_cache_size=int(getattr(args, 'subject_cache_size', 256)),
        sampling_seed=int(getattr(args, 'sampling_seed', 2024)),
        context_min=int(args.context_min), context_max=int(args.context_max),
        context_lengths=(parse_int_list(args.context_lengths, 'context_lengths')
                         if getattr(args, 'context_lengths', None) else None),
        forecast_offsets=resolve_forecast_offsets(args),
        eval_context_length=int(context_length),
        eval_anchors_per_subject=0,
        eval_rollout_tasks_per_subject=0,
        eval_rollout_steps=0,
        eval_all_future_steps=1,
        eval_anchor_min_future_steps=0,
        num_rois=int(getattr(args, 'num_rois', 116)),
        graph_source=str(getattr(args, 'graph_source', 'legacy_mask')),
        calibration_length=int(getattr(args, 'calibration_length', 0) or 0),
        **condition_kwargs,
    )


if __name__ == '__main__':
    main()
