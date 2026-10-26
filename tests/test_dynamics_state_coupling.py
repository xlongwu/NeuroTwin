# coding=utf-8
"""历史记忆状态耦合的最小验证。

关注两点：
1. 零初始化 ⇒ 启用开关的模型在起点与 baseline 逐位等价；
2. 默认关闭时不创建任何新参数，旧 checkpoint 的 strict 加载不受影响。
"""
import torch

from models.tfm import NeuroTwinTFM
from models.tfm_dynamics import GraphODEDynamicsHead


def _head(**kw):
    torch.manual_seed(0)
    return GraphODEDynamicsHead(dim=8, hidden_dim=16, cond_dim=3,
                                profile_dim=4, ode_steps=2, **kw)


def _field_inputs():
    torch.manual_seed(1)
    z = torch.randn(2, 5, 8)
    adj = torch.rand(2, 5, 5)
    adj = 0.5 * (adj + adj.transpose(-1, -2))
    return z, adj, torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 5, 1)


def test_history_memory_zero_init_matches_baseline():
    base = _head()
    mem = _head(history_memory=True)
    mem.load_state_dict(base.state_dict(), strict=False)
    z, adj, cond, profile, control = _field_inputs()
    memory = torch.randn(2, 5, 8)
    assert torch.allclose(base._field(z, adj, cond, profile, control),
                          mem._field(z, adj, cond, profile, control,
                                     memory=memory))
    # 全局记忆 [B,D] 与逐 ROI 记忆 [B,F,D] 都必须可用
    assert torch.allclose(
        mem._field(z, adj, cond, profile, control, memory=memory[:, 0, :]),
        mem._field(z, adj, cond, profile, control, memory=memory))


def test_integrate_interval_backward_compatible_without_coupling():
    base = _head()
    mem = _head(history_memory=True)
    mem.load_state_dict(base.state_dict(), strict=False)
    z, adj, cond, profile, control = _field_inputs()
    q = torch.randn(2, 5)
    dt = torch.ones(2)
    z0, q0 = base.integrate_interval(z, q, adj, dt, cond, profile, control)
    z1, q1 = mem.integrate_interval(z, q, adj, dt, cond, profile, control,
                                    memory=None)
    assert torch.allclose(z0, z1) and torch.allclose(q0, q1)


def _tfm(**kw):
    torch.manual_seed(0)
    return NeuroTwinTFM(features=6, context_max=8, patch_len=2, dim=16,
                        num_layers=1, num_heads=4, ff_ratio=2, cpm_horizon=4,
                        cpm_layers=1, pretrain_mode=True,
                        dynamics_mode='graph_ode', graph_source='history_fc',
                        ode_steps=2, **kw)


def test_history_memory_does_not_shift_downstream_rng():
    """同一 seed 下，历史记忆模块不得改变共享参数的初始化（消融无初始化混杂）。"""
    kw = dict(personalized=True, profile_dim=8)
    base = _tfm(**kw).state_dict()
    other = _tfm(**kw, dynamics_history_memory=True).state_dict()
    diff = [k for k in base if k in other and not torch.equal(base[k], other[k])]
    assert diff == [], f'dynamics_history_memory 改变了共享参数初始化：{diff[:5]}'


def test_history_memory_is_opt_in():
    plain = _tfm()
    assert plain.one_step_head.memory_film is None
    assert plain._history_memory(torch.randn(2, 6, 4, 16)) is None

    coupled = _tfm(dynamics_history_memory=True)
    keys = coupled.state_dict().keys()
    assert 'one_step_head.memory_film.weight' in keys
    assert not any('memory_film' in k for k in plain.state_dict())


def test_history_memory_runs_and_keeps_state_api():
    model = _tfm(dynamics_history_memory=True)
    x = torch.randn(3, 6, 1, 8)
    sc = torch.rand(6, 6)
    out, aux = model(x, sc)
    assert out.shape == (3, 6, 1, 1) and torch.isfinite(out).all()
    assert aux['dynamics_pred'].shape == (3, 6, 4)

    hist = x[:, :, 0, :].transpose(1, 2).contiguous()           # [B,K,F]
    state = model.initialize_state(hist, sc)
    assert state['memory'] is not None
    state = model.advance_state(state)
    assert state['memory'] is not None
    state = model.update_state(state, torch.randn(3, 6))
    assert state['memory'] is not None
