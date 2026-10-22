# coding=utf-8
"""S4-1/S4-2 模块与 CRPS 损失的单元测试。"""
import torch
import pytest

from models.tfm import NeuroTwinTFM
from train.optim import s4_module_prefixes
from train.prob_losses import (ensemble_crps, ensemble_energy_score,
                               temporal_variogram_score)
from utils.common import set_seed


def _build(variant):
    set_seed(123)
    return NeuroTwinTFM(
        features=8, context_max=16, patch_len=4, dim=16,
        num_layers=1, num_heads=2, ff_ratio=2, dropout=0.0,
        cpm_horizon=4, cpm_layers=1, pretrain_mode=True,
        dynamics_mode='graph_ode', graph_source='history_fc', ode_steps=1,
        dynamics_history_memory=True, s4_variant=variant,
        s4_residual_rank=2, s4_residual_hidden=8,
        s4_noise_rank=2, s4_noise_scale=1e-2, s4_train_samples=4)


def _inputs():
    return torch.randn(2, 8, 1, 16), torch.eye(8)


def test_default_off_is_bitwise_unchanged():
    m_off = _build('off')
    m_res = _build('s4_1')
    m_inn = _build('s4_2')
    sd_off = m_off.state_dict()
    for other in (m_res, m_inn):
        sd = other.state_dict()
        shared = [k for k in sd_off if k in sd]
        assert shared
        assert max((sd_off[k] - sd[k]).abs().max().item() for k in shared) == 0.0

    x, sc = _inputs()
    m_off.eval()
    with torch.no_grad():
        _, aux = m_off(x, sc)
    assert 'dynamics_samples' not in aux
    assert aux['dynamics_pred'].shape == (2, 8, 4)


@pytest.mark.parametrize('variant', ['s4_1', 's4_2'])
def test_s4_forward_shape_finite_backward(variant):
    model = _build(variant)
    if variant == 's4_2':
        # 新初始化时 decode_rate 为零（BOLD 读出一开始是 persistence），
        # 随机 latent 不会改变 q；这里显式给读出层非零权重，验证真实梯度路径。
        with torch.no_grad():
            model.one_step_head.decode_rate.weight.normal_(std=0.1)
    model.train()
    x, sc = _inputs()
    out, aux = model(x, sc)
    samples = aux['dynamics_samples']
    assert samples.shape == (2, 4, 8, 4)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(out).all()
    assert samples.std(dim=1).mean() > 0
    loss = samples.pow(2).mean() + out.pow(2).mean()
    loss.backward()
    pattern = 's4_residual_head' if variant == 's4_1' else 'innovation'
    grads = [(n, p.grad) for n, p in model.named_parameters() if pattern in n]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for _, g in grads)
    if variant == 's4_2':
        assert any(g.abs().max().item() > 0 for _, g in grads)
    assert s4_module_prefixes(model)


def test_s4_requires_graph_ode():
    with pytest.raises(ValueError, match='graph_ode'):
        NeuroTwinTFM(features=8, context_max=16, patch_len=4, dim=16,
                     num_layers=1, num_heads=2, cpm_horizon=4, cpm_layers=1,
                     pretrain_mode=True, dynamics_mode='discrete',
                     s4_variant='s4_1')


def test_ensemble_crps_hand_computed():
    samples = torch.tensor([[[[0.0]], [[2.0]]]], dtype=torch.float32)
    target = torch.zeros(1, 1, 1)
    assert ensemble_crps(samples, target).item() == pytest.approx(0.5)
    single = torch.zeros(1, 1, 1, 1)
    assert ensemble_crps(single, target).item() == pytest.approx(0.0)
    # 全 mask 无效时不应产生 NaN（分母 clamp）
    mask = torch.zeros(1, 1)
    assert torch.isfinite(ensemble_crps(samples, target, mask)).all()


def test_ensemble_energy_score_hand_computed():
    samples = torch.tensor([[[[0.0]], [[2.0]]]], dtype=torch.float32)
    target = torch.zeros(1, 1, 1)
    # E||X-y||=1，0.5·E||X-X'||=0.5 → ES=0.5
    assert ensemble_energy_score(samples, target).item() == pytest.approx(0.5)


def test_temporal_variogram_score():
    target = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    same = target.unsqueeze(1)
    assert temporal_variogram_score(same, target).item() == pytest.approx(0.0)
    shifted = torch.tensor([[[0.0, -1.0], [0.0, -1.0]]]).unsqueeze(1)
    assert temporal_variogram_score(shifted, target).item() > 0


def test_paired_val_rng_pins_and_restores_random_stream():
    """验证期共同随机数：同种子重复采样一致，且全局随机流逐位恢复（训练不受影响）。"""
    from main import paired_val_rng

    torch.manual_seed(123)
    head = torch.randn(4)
    state = torch.get_rng_state()
    with paired_val_rng(True, 2024):
        a = torch.randn(4)
    assert torch.equal(torch.get_rng_state(), state)
    with paired_val_rng(True, 2024):
        b = torch.randn(4)
    assert torch.equal(a, b), '同一 val 种子的采样轨迹应逐位一致'
    assert not torch.equal(a, head)
    tail = torch.randn(4)
    torch.manual_seed(123)
    _ = torch.randn(4)
    expected = torch.randn(4)
    assert torch.equal(tail, expected), '退出上下文后训练随机流应如同未进入过'


def test_paired_val_rng_inactive_is_noop():
    from main import paired_val_rng
    torch.manual_seed(7)
    state = torch.get_rng_state()
    with paired_val_rng(False, 2024):
        drawn = torch.randn(3)
    assert not torch.equal(torch.get_rng_state(), state)
    assert drawn.shape == (3,)


def test_paired_val_rng_restores_even_on_exception():
    from main import paired_val_rng
    torch.manual_seed(11)
    state = torch.get_rng_state()
    with pytest.raises(RuntimeError):
        with paired_val_rng(True, 2024):
            raise RuntimeError('boom')
    assert torch.equal(torch.get_rng_state(), state)
