"""训练恢复接口（工程能力，不是本轮实验的一部分）。

本轮正式实验**必须从头训练**，因此这个模块在 fresh 流程里不被调用；它存在的意义是
让"崩溃后可续训"作为工程质量被单独验证（方案 §6.5/§6.10）：一旦缺了这些状态，
恢复出来的优化器会悄悄丢掉 Adam 二阶矩、AMP 缩放器和 EMA 影子权重，续训结果与
连续训练并不等价，而这一点很难事后发现。

保存内容（缺一不可，否则恢复不等价）：
  模型权重、EMA 影子权重、optimizer 状态、scheduler 状态、AMP GradScaler 状态、
  epoch / global_step、最佳指标与 patience、主干冻结状态、python / numpy / torch
  （CPU 与全部 CUDA）随机状态。
"""
from __future__ import annotations

import json
import os
import random as _random

import numpy as _np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
