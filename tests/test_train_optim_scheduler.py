# coding=utf-8
"""学习率调度契约：每个参数组按自身峰值比例衰减，不能反向升高。"""
import math
from types import SimpleNamespace

import pytest
import torch

from train.optim import build_scheduler


def _optimizer():
    backbone = torch.nn.Parameter(torch.zeros(1))
    condition = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([
        {'params': [backbone], 'lr': 2e-5, 'name': 'backbone'},
        {'params': [condition], 'lr': 1e-4, 'name': 'condition'},
    ])


def _args(**overrides):
    values = dict(train_epochs=20, warmup_epochs=5, lr_init=5e-5,
                  lr_peak=1e-4, lr_final=5e-5)
    values.update(overrides)
    return SimpleNamespace(**values)


def _lrs(optimizer):
    return [group['lr'] for group in optimizer.param_groups]


def _step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()


def test_scheduler_decays_each_group_to_relative_floor():
    opt = _optimizer()
    sched = build_scheduler(opt, _args())

    # Warmup 起点：两个参数组都按 lr_init / lr_peak 同比例收缩。
    assert _lrs(opt) == pytest.approx([1e-5, 5e-5])

    # Warmup 结束回到各自峰值。
    for _ in range(5):
        _step(opt, sched)
    assert _lrs(opt) == pytest.approx([2e-5, 1e-4])

    # 余弦阶段两组都必须单调不增，且保持 5 倍相对关系。
    previous = _lrs(opt)
    for _ in range(15):
        _step(opt, sched)
        current = _lrs(opt)
        assert current[0] <= previous[0] + 1e-15
        assert current[1] <= previous[1] + 1e-15
        assert current[1] / current[0] == pytest.approx(5.0)
        previous = current

    # 每组衰减到自身峰值的 lr_final / lr_peak，而不是统一的绝对 eta_min。
    assert _lrs(opt) == pytest.approx([1e-5, 5e-5])


def test_scheduler_without_warmup_starts_at_group_peak():
    opt = _optimizer()
    sched = build_scheduler(opt, _args(warmup_epochs=0))

    assert _lrs(opt) == pytest.approx([2e-5, 1e-4])
    for _ in range(20):
        _step(opt, sched)
    assert _lrs(opt) == pytest.approx([1e-5, 5e-5])


def test_scheduler_rejects_invalid_lr_bounds():
    with pytest.raises(ValueError, match='lr_peak'):
        build_scheduler(_optimizer(), _args(lr_peak=0.0))
    with pytest.raises(ValueError, match='lr_final'):
        build_scheduler(_optimizer(), _args(lr_final=2e-4))
    with pytest.raises(ValueError, match='lr_final'):
        build_scheduler(_optimizer(), _args(lr_final=-1e-5))


def test_scheduler_covers_groups_added_after_construction():
    """解冻后追加的 backbone 组必须按当前进度衰减，而不是保持恒定。"""
    opt = _optimizer()
    sched = build_scheduler(opt, _args())
    for _ in range(10):
        _step(opt, sched)

    backbone = torch.nn.Parameter(torch.zeros(1))
    opt.add_param_group({'params': [backbone], 'lr': 2e-5, 'name': 'backbone_late'})
    sched.sync_param_groups()

    progress = (10 - 5) / (20 - 5)
    factor = 0.5 + 0.5 * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert opt.param_groups[-1]['lr'] == pytest.approx(2e-5 * factor)

    for _ in range(10):
        _step(opt, sched)
    assert opt.param_groups[-1]['lr'] == pytest.approx(1e-5)
