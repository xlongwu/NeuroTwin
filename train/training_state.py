"""训练状态保存、随机状态恢复与验证指标选优。"""
from __future__ import annotations

import json
import math
import os
import random as _random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as _np
import torch

TRAINING_STATE_VERSION = 'ts_1'

#: 恢复等价性自检必须覆盖的随机来源；缺一项就可能出现"看着一样、采样不一样"。
#: 键名与 :func:`capture_rng_state` 的实际输出**必须**保持一致，否则这里的完整性
#: 校验会把一个正常保存的状态判成残缺（python 状态是 625 个整数而非字节串，没有
#: 单一 'python'/'numpy' 键）。
_RNG_KEYS = ('python_version', 'python_state', 'python_has_gauss',
             'numpy_impl', 'numpy_key', 'numpy_pos', 'numpy_has_gauss',
             'numpy_cached_gauss', 'torch_cpu', 'torch_cuda')


def capture_rng_state() -> dict[str, Any]:
    """抓取全部随机来源状态，序列化为纯 int/bool 列表（不依赖 pickle）。

    ``random.getstate()[1]`` 是 625 个 32 位整数而**不是**字节串，按 bytes 转换会直接
    ValueError；numpy 的 MT19937 状态同理要按位存回，否则恢复后采样不等价。
    """
    py_state = _random.getstate()
    np_state = _np.random.get_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else [])
    return {
        'python_version': int(py_state[0]),
        'python_state': [int(v) for v in py_state[1]],
        'python_has_gauss': bool(py_state[2]),
        'numpy_impl': str(np_state[0]),
        'numpy_key': [int(v) for v in np_state[1]],
        'numpy_pos': int(np_state[2]),
        'numpy_has_gauss': int(np_state[3]),
        'numpy_cached_gauss': int(np_state[4]),
        'torch_cpu': torch.get_rng_state().tolist(),
        'torch_cuda': [s.tolist() for s in cuda_states],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    _random.setstate((int(state['python_version']),
                      tuple(int(v) for v in state['python_state']),
                      bool(state['python_has_gauss'])))
    _np.random.set_state((str(state['numpy_impl']),
                          _np.array(state['numpy_key'], dtype='uint32'),
                          int(state['numpy_pos']),
                          int(state['numpy_has_gauss']),
                          int(state['numpy_cached_gauss'])))
    torch.set_rng_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))
    if state.get('torch_cuda'):
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.tensor(s, dtype=torch.uint8) for s in state['torch_cuda']])


@dataclass
class TrainingState:
    """一次续训所需的全部进度信息（与权重分开保存，便于审计）。"""

    version: str = TRAINING_STATE_VERSION
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_metric_name: str = ''
    patience_counter: int = 0
    backbone_unfrozen: bool = False
    study: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unwrap(model):
    return getattr(model, '_orig_mod', model)


def save_training_state(path: os.PathLike | str, *, model, optimizer, scheduler,
                        scaler, ema, progress: TrainingState) -> str:
    """把全部训练状态写进**同一个**文件。

    ``weights_only=False`` 是不必要的：所有字段都是张量 / 基本类型 / dict，因此这里
    刻意只用 ``{'state_dict': tensors, 'plain': json_str}`` 的两段式布局 —— 恢复时
    可以走 ``weights_only=True`` 白名单解码，避免反序列化任意 Python 对象。
    """
    payload = {
        'version': TRAINING_STATE_VERSION,
        # 进度等纯数据序列化成 JSON 字符串，杜绝 pickle 执行任意对象钩子
        'plain': json.dumps({
            'progress': progress.to_dict(),
            'rng': capture_rng_state(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'ema_present': ema is not None,
        }, ensure_ascii=False, sort_keys=True),
        'model': _unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'ema': None if ema is None else getattr(ema, 'ema', ema).state_dict()
               if hasattr(ema, 'ema') else None,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return str(p)


def load_training_state(path: os.PathLike | str, *, model, optimizer, scheduler,
                        scaler, ema, strict: bool = True) -> TrainingState:
    """恢复训练状态。缺任何一项都报错，不做"尽力而为"的静默兜底。"""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload.get('version') != TRAINING_STATE_VERSION:
        raise ValueError(
            f'训练状态版本 {payload.get("version")!r} ≠ {TRAINING_STATE_VERSION!r}；'
            f'跨代恢复不可信，请重新从头训练。')
    plain = json.loads(payload['plain'])
    if tuple(plain['rng'].keys()) and not all(k in plain['rng'] for k in _RNG_KEYS):
        raise ValueError(f'随机状态缺来源：{sorted(plain["rng"])} 应含 {list(_RNG_KEYS)}')

    model.load_state_dict(payload['model'], strict=strict)
    optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and payload.get('scheduler') is not None:
        scheduler.load_state_dict(payload['scheduler'])
    if scaler is not None and plain.get('scaler') is not None:
        scaler.load_state_dict(plain['scaler'])
    if ema is not None and payload.get('ema') is not None:
        inner = getattr(ema, 'ema', None)
        if inner is not None:
            inner.load_state_dict(payload['ema'], strict=strict)
    restore_rng_state(plain['rng'])

    prog = plain['progress']
    if prog.get('version') != TRAINING_STATE_VERSION:
        raise ValueError(f'进度结构版本异常：{prog.get("version")!r}')
    return TrainingState(**prog)


__all__ = ['TRAINING_STATE_VERSION', 'TrainingState', 'capture_rng_state',
           'restore_rng_state', 'save_training_state', 'load_training_state']


DYNAMICS_CKPT_HORIZON = 8


def dynamics_metric_key(horizon: int) -> str:
    """主判据键名，如 ``metric_mae_dynamics_h1_8_subject``（h1..hN 逐被试等权）。"""
    if int(horizon) < 1:
        raise ValueError(f"horizon 必须 >= 1，收到 {horizon}")
    return f"metric_mae_dynamics_h1_{int(horizon)}_subject"


def dynamics_metric_name(horizon: int) -> str:
    """主判据的日志显示名。"""
    if int(horizon) < 1:
        raise ValueError(f"horizon 必须 >= 1，收到 {horizon}")
    return f"val graph-ODE H1-{int(horizon)} subject MAE"


def resolve_dynamics_metric_horizon(args) -> int:
    """本实验的选优 horizon：CLI 覆盖 > 模型的 cpm_horizon（graph_ode 头的范围）> 8。"""
    configured = int(getattr(args, "dynamics_metric_horizon", 0) or 0)
    if configured > 0:
        return configured
    return int(getattr(args, "cpm_horizon", 0) or 0) or DYNAMICS_CKPT_HORIZON


def dynamics_anchor_mae(
    dynamics_pred, future, future_mask, horizon: int = DYNAMICS_CKPT_HORIZON
):
    """Graph-ODE 轨迹的逐锚点 H1..H``horizon`` 全 ROI 平均绝对误差。

    Args:
        dynamics_pred: ``[B,F,H]`` Graph-ODE 持续推进得到的预测（数据尺度，
            与动态损失 ``dynamics_target`` 同尺度）
        future: ``[B,F,R]`` 未来真值（数据尺度，序列末尾截断处补零）
        future_mask: ``[B,R]`` 逐步真值有效性（动态口径用 future_mask，
            不是 CPM 的 cpm_mask）
    Returns:
        ``(anchor_mae [B], full_mask [B] bool)``。``full_mask`` 标记具备完整
        ``horizon`` 步真值的锚点；``anchor_mae`` 只在被标记处有意义，其余位置
        置零，避免短未来锚点以不同预测范围参与主指标。
    """
    if int(horizon) < 1:
        raise ValueError(f"horizon 必须 >= 1，收到 {horizon}")
    if dynamics_pred.ndim != 3:
        raise ValueError(
            f"dynamics_pred 期望 [B,F,H]，收到 {tuple(dynamics_pred.shape)}"
        )
    b, f, h = dynamics_pred.shape
    if h < horizon:
        raise ValueError(
            f"dynamics_pred 只有 {h} 步，无法计算 H1..{horizon} 指标；请确认模型的 dynamics_horizon/cpm_horizon >= {horizon}。"
        )
    if future.ndim != 3 or tuple(future.shape[:2]) != (b, f):
        raise ValueError(
            f"future 期望 [B,F,R] 且 (B,F)==({b},{f})，收到 {tuple(future.shape)}"
        )
    if future.shape[2] < horizon:
        raise ValueError(
            f"future 只有 {future.shape[2]} 步，少于指标要求的 {horizon} 步；请确认验证视图的 future_steps 覆盖完整预测范围。"
        )
    if tuple(future_mask.shape) != (b, future.shape[2]):
        raise ValueError(
            f"future_mask 期望 [B,{future.shape[2]}]，收到 {tuple(future_mask.shape)}"
        )
    err = (
        dynamics_pred[:, :, :horizon] - future[:, :, :horizon].to(dynamics_pred.dtype)
    ).abs()
    anchor_mae = err.mean(dim=(1, 2))
    full_mask = (future_mask[:, :horizon] > 0).all(dim=1)
    return (anchor_mae * full_mask.to(anchor_mae.dtype), full_mask)


def aggregate_dynamics_subject_metric(
    subj_acc, n_total_anchors: int, horizon: int = DYNAMICS_CKPT_HORIZON
):
    """``{subj: [Σ锚点MAE, n_完整锚点]}`` → (主指标, 被试数, 锚点数)。

    聚合顺序为「先被试内均值、再跨被试等权均值」，因此结果与 batch 划分和
    每名被试的锚点数无关。没有任何完整 ``horizon`` 步目标时直接报错，
    不回退到单步 MAE 等其他口径。
    """
    subj_means = [total / n for total, n in subj_acc.values() if n > 0]
    if not subj_means:
        raise RuntimeError(
            f"验证集没有具备完整 {horizon} 步真实目标的锚点（总锚点={n_total_anchors}，涉及被试={len(subj_acc)}），无法计算 {dynamics_metric_key(horizon)}。请检查验证视图的 future_steps / future_mask 是否覆盖 H1..{horizon}；本指标作为 checkpoint 选优判据时不允许回退到其他口径。"
        )
    n_anchors = int(sum((n for _, n in subj_acc.values())))
    return (sum(subj_means) / len(subj_means), len(subj_means), n_anchors)


class CheckpointSelector:
    """按单一验证指标跟踪 best checkpoint 与 patience（指标越低越好）。

    指标键在构造时固定：单步与多步 MAE 量纲不同、数值不可比较，因此同一 run
    内不允许中途换判据，也不继承其他判据的历史 best 值。
    """

    def __init__(self, metric_key: str, metric_name: str = "", patience: int = 1):
        if not metric_key:
            raise ValueError("metric_key 不能为空")
        if int(patience) < 1:
            raise ValueError(f"patience 必须 >= 1，收到 {patience}")
        self.metric_key = metric_key
        self.metric_name = metric_name or metric_key
        self.patience = int(patience)
        self.best = float("inf")
        self.counter = 0

    def step(self, metrics) -> tuple:
        """消费一个 epoch 的验证指标，返回 ``(is_best, current)`` 并更新 patience。"""
        if self.metric_key not in metrics:
            raise KeyError(
                f"验证结果缺少选优指标 {self.metric_key}；可用键示例: {sorted((k for k in metrics if k.startswith('metric_')))[:8]}"
            )
        current = float(metrics[self.metric_key])
        if not math.isfinite(current):
            raise ValueError(
                f"{self.metric_key} 为非有限值 {current}，无法据此选优；请先排查该 epoch 的动态预测是否发散，而不是让 NaN 参与比较。"
            )
        is_best = current < self.best
        if is_best:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
        return (is_best, current)

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience
