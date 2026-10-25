"""Complete atomic epoch-boundary checkpoints for training-state continuation."""
from __future__ import annotations

import json
import os
import random as _random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as _np
import torch

TRAINING_STATE_VERSION = 'ts_2'
_REQUIRED_RNG_KEYS = {
    'python_version', 'python_state', 'python_gauss_next', 'numpy_impl',
    'numpy_key', 'numpy_pos', 'numpy_has_gauss', 'numpy_cached_gauss',
    'torch_cpu', 'torch_cuda',
}
_REQUIRED_PAYLOAD_KEYS = {
    'version', 'plain', 'model', 'criterion', 'optimizer', 'optimizer_layout',
    'scheduler', 'ema', 'loader_generator',
}


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch CPU/all-CUDA RNG states."""
    py_state = _random.getstate()
    np_state = _np.random.get_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else [])
    return {
        'python_version': int(py_state[0]),
        'python_state': [int(v) for v in py_state[1]],
        'python_gauss_next': (None if py_state[2] is None else float(py_state[2])),
        'numpy_impl': str(np_state[0]),
        'numpy_key': [int(v) for v in np_state[1]],
        'numpy_pos': int(np_state[2]),
        'numpy_has_gauss': int(np_state[3]),
        'numpy_cached_gauss': float(np_state[4]),
        'torch_cpu': torch.get_rng_state().tolist(),
        'torch_cuda': [s.tolist() for s in cuda_states],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    missing = _REQUIRED_RNG_KEYS - set(state)
    if missing:
        raise ValueError(f'训练状态缺少随机状态来源：{sorted(missing)}')
    _random.setstate((int(state['python_version']),
                      tuple(int(v) for v in state['python_state']),
                      (None if state['python_gauss_next'] is None
                       else float(state['python_gauss_next']))))
    _np.random.set_state((str(state['numpy_impl']),
                          _np.array(state['numpy_key'], dtype='uint32'),
                          int(state['numpy_pos']),
                          int(state['numpy_has_gauss']),
                          float(state['numpy_cached_gauss'])))
    torch.set_rng_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))
    cuda_states = state['torch_cuda']
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError('checkpoint 含 CUDA RNG 状态，但当前 PyTorch 无可用 CUDA')
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                f'checkpoint 保存了 {len(cuda_states)} 个 CUDA RNG 状态，'
                f'当前设备数为 {torch.cuda.device_count()}')
        torch.cuda.set_rng_state_all(
            [torch.tensor(s, dtype=torch.uint8) for s in cuda_states])


@dataclass
class TrainingState:
    """Progress at the most recently completed full epoch."""

    version: str = TRAINING_STATE_VERSION
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_metric_name: str = ''
    patience_counter: int = 0
    backbone_unfrozen: bool = False
    early_stopped: bool = False
    study: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unwrap(model):
    return getattr(model, '_orig_mod', model)


def optimizer_layout(model, criterion, optimizer) -> list[dict[str, Any]]:
    """Record parameter names and group order so optimizer state cannot load by shape alone."""
    owners = {}
    for name, param in _unwrap(model).named_parameters():
        owners[id(param)] = f'model.{name}'
    for name, param in criterion.named_parameters():
        owners[id(param)] = f'criterion.{name}'
    layout = []
    for index, group in enumerate(optimizer.param_groups):
        names = []
        for param in group['params']:
            name = owners.get(id(param))
            if name is None:
                raise ValueError('optimizer 含有不属于当前 model/criterion 的参数')
            names.append(name)
        layout.append({'name': str(group.get('name', f'group_{index}')),
                       'params': names})
    return layout


def _read_payload(path: os.PathLike | str) -> tuple[dict, dict]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f'完整训练状态文件不存在：{p}')
    try:
        payload = torch.load(p, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(
            f'{p} 无法作为完整训练状态读取。旧 base_best.pt/finetuned_best.pt '
            '只有推理权重，请用 --pretrained_weight 初始化一个新训练；'
            f'原始读取错误：{type(exc).__name__}: {exc}') from exc
    if not isinstance(payload, dict) or payload.get('version') != TRAINING_STATE_VERSION:
        raise ValueError(
            f'{p} 不是当前格式的完整训练状态（需要 {TRAINING_STATE_VERSION}）。'
            '旧 base_best.pt/finetuned_best.pt 仅含推理权重，无法恢复 optimizer、'
            'scheduler、EMA 与随机状态；请将它通过 --pretrained_weight 用作新训练初始化。')
    missing = _REQUIRED_PAYLOAD_KEYS - set(payload)
    if missing:
        raise ValueError(f'完整训练状态缺少必需内容：{sorted(missing)}')
    try:
        plain = json.loads(payload['plain'])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{p} 的训练状态元数据无法读取：{exc}') from exc
    missing_plain = {'progress', 'rng', 'scaler', 'ema_present', 'ema_decay'} - set(plain)
    if missing_plain:
        raise ValueError(f'完整训练状态缺少运行元数据：{sorted(missing_plain)}')
    missing_rng = _REQUIRED_RNG_KEYS - set(plain.get('rng', {}))
    if missing_rng:
        raise ValueError(f'完整训练状态缺少随机状态来源：{sorted(missing_rng)}')
    progress = plain.get('progress')
    if not isinstance(progress, dict) or progress.get('version') != TRAINING_STATE_VERSION:
        raise ValueError('完整训练状态的进度结构缺失或版本错误')
    try:
        TrainingState(**progress)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'完整训练状态进度字段不完整：{exc}') from exc
    return payload, plain


def read_training_state_metadata(path: os.PathLike | str) -> dict[str, Any]:
    """Read small JSON metadata for setup decisions; tensor state is discarded on return."""
    _, plain = _read_payload(path)
    return plain


def save_training_state(path: os.PathLike | str, *, model, criterion, optimizer,
                        scheduler, scaler, ema, loader_generator,
                        progress: TrainingState) -> str:
    """Atomically replace one complete checkpoint after an epoch has finished."""
    if scheduler is None or scaler is None or loader_generator is None:
        raise ValueError('完整训练状态要求 scheduler、scaler 和独立 loader generator')
    payload = {
        'version': TRAINING_STATE_VERSION,
        'plain': json.dumps({
            'progress': progress.to_dict(),
            'rng': capture_rng_state(),
            'scaler': scaler.state_dict(),
            'ema_present': ema is not None,
            'ema_decay': None if ema is None else float(ema.decay),
        }, ensure_ascii=False, sort_keys=True, allow_nan=False),
        'model': _unwrap(model).state_dict(),
        'criterion': criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_layout': optimizer_layout(model, criterion, optimizer),
        'scheduler': scheduler.state_dict(),
        'ema': None if ema is None else ema.ema.state_dict(),
        'loader_generator': loader_generator.get_state(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{p.name}.', suffix='.tmp', dir=p.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, p)
        try:
            dir_fd = os.open(p.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return str(p)


def load_training_state(path: os.PathLike | str, *, model, criterion, optimizer,
                        scheduler, scaler, ema, loader_generator,
                        restore_rng: bool = True) -> TrainingState:
    """Strictly restore every training component; RNG restore may be deferred by caller."""
    payload, plain = _read_payload(path)
    expected_layout = optimizer_layout(model, criterion, optimizer)
    if payload['optimizer_layout'] != expected_layout:
        raise ValueError(
            'optimizer 参数组拓扑与训练状态不一致。'
            f'\ncheckpoint={payload["optimizer_layout"]}'
            f'\ncurrent={expected_layout}')

    ema_expected = ema is not None
    if bool(plain.get('ema_present')) != ema_expected:
        raise ValueError('训练状态中的 EMA 启用状态与当前 --use_ema 不一致')
    if ema is not None and float(plain.get('ema_decay')) != float(ema.decay):
        raise ValueError('训练状态中的 EMA decay 与当前 --ema_decay 不一致')

    if scheduler is None or scaler is None or loader_generator is None:
        raise ValueError('恢复完整训练状态需要 scheduler、scaler 和独立 loader generator')
    if payload['scheduler'] is None or not isinstance(payload['criterion'], dict):
        raise ValueError('训练状态缺少 criterion 或 scheduler 状态')
    if payload['ema'] is None and ema is not None:
        raise ValueError('当前启用 EMA，但训练状态未保存 EMA 参数')
    if payload['ema'] is not None and ema is None:
        raise ValueError('训练状态含 EMA 参数，但当前配置未启用 EMA')
    scheduler_keys = set(scheduler.state_dict())
    if scheduler_keys - set(payload['scheduler']):
        raise ValueError(
            f'训练状态 scheduler 缺少字段：'
            f'{sorted(scheduler_keys - set(payload["scheduler"]))}')
    scaler_keys = set(scaler.state_dict())
    saved_scaler = plain['scaler']
    if not isinstance(saved_scaler, dict) or scaler_keys - set(saved_scaler):
        raise ValueError('训练状态 GradScaler 字段不完整或类型错误')

    _unwrap(model).load_state_dict(payload['model'], strict=True)
    criterion.load_state_dict(payload['criterion'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(plain['scaler'])
    loader_generator.set_state(payload['loader_generator'])
    if ema is not None:
        ema.ema.load_state_dict(payload['ema'], strict=True)

    if restore_rng:
        restore_rng_state(plain['rng'])
    return TrainingState(**plain['progress'])


__all__ = ['TRAINING_STATE_VERSION', 'TrainingState', 'capture_rng_state',
           'restore_rng_state', 'optimizer_layout', 'read_training_state_metadata',
           'save_training_state', 'load_training_state']
