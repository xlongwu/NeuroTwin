# coding=utf-8
"""D1/D2 状态刷新诊断的指标与分支测试。"""
import numpy as np
import torch

from experiments.long_horizon.refresh_rollout import (
    paired_summary, per_anchor_metrics, rollout_mode, subject_table,
)
from models.tfm_dynamics import GraphODEDynamicsHead


def _trajectory(scale=1.0, steps=4, rois=6):
    """构造确定的 [N,R,F] 轨迹与锚点。"""
    n = 2
    base = np.linspace(-1.0, 1.0, rois, dtype=np.float64)
    target = np.stack([base * (h + 1) for h in range(steps)], axis=0)
    target = np.broadcast_to(target, (n, steps, rois)).copy()
    pred = target * scale
    x_last = np.zeros((n, rois), dtype=np.float64)
    mask = np.ones((n, steps), dtype=np.float64)
    return pred, target, mask, x_last


def test_perfect_rollout_metrics():
    pred, target, mask, x_last = _trajectory(scale=1.0)
    m = per_anchor_metrics(pred, target, mask, x_last)
    assert np.allclose(m['mae'], 0.0)
    assert np.allclose(m['delta_std_ratio'], 1.0)
    assert np.allclose(m['delta_pcc'], 1.0)
    assert np.allclose(m['direction_accuracy'], 1.0)


def test_half_amplitude_is_compressed():
    pred, target, mask, x_last = _trajectory(scale=0.5)
    m = per_anchor_metrics(pred, target, mask, x_last)
    assert np.allclose(m['delta_std_ratio'], 0.5)
    assert np.allclose(m['delta_pcc'], 1.0)
    assert np.all(m['mae'] > 0)


def test_mask_makes_horizon_nan():
    pred, target, mask, x_last = _trajectory()
    mask[0, 2] = 0.0
    m = per_anchor_metrics(pred, target, mask, x_last)
    assert np.isnan(m['mae'][0, 2])
    assert not np.isnan(m['mae'][1, 2])
    # 第 3 步的相对差分也依赖上一步有效，必须同步失效
    assert np.isnan(m['delta_pcc'][0, 3])


def test_subject_table_averages_within_subject():
    subj = ['a', 'a', 'b']
    vals = np.array([1.0, 3.0, 5.0])
    order, out = subject_table(subj, vals)
    assert order == ['a', 'b']
    assert np.allclose(out, [2.0, 5.0])


def test_paired_summary_identical_is_not_significant():
    free = np.array([[1.0, 2.0], [1.2, 2.2]])
    out = paired_summary(free, free.copy())
    assert out[0]['p'] == 1.0
    assert out[0]['delta_mean'] == 0.0


class _FakeHead:
    def field_terms(self, z, adj, cond=None, profile=None, control=None,
                    q=None, memory=None):
        n, f, d = z.shape
        return {
            'diffusion': torch.zeros_like(z),
            'residual': torch.zeros_like(z),
            'control': None,
            'q_feedback': None,
            'kappa': torch.ones(n, 1),
        }


class _FakeModel:
    """只实现 rollout_mode 依赖的状态 API，记录 update_state 的输入。"""

    def __init__(self):
        self.one_step_head = _FakeHead()
        self.updates = []

    def readout_state(self, state):
        return state['z'][:, :, 0]

    def advance_state(self, state, control=None):
        new = dict(state)
        new['z'] = state['z'] + 1.0
        return new

    def update_state(self, state, observation):
        self.updates.append(observation.clone())
        new = dict(state)
        new['z'] = torch.cat(
            [observation.unsqueeze(-1), state['z'][:, :, 1:]], dim=-1)
        return new


def _fake_state(n=2, f=4, d=3):
    return {'z': torch.zeros(n, f, d), 'adj': torch.eye(f).expand(n, f, f),
            'cond': None, 'profile': None}


def test_d1_uses_future_observation_and_d2_uses_prediction():
    model = _FakeModel()
    future = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    pred, _ = rollout_mode(model, _fake_state(), 3, None, 'observed', future)
    assert len(model.updates) == 3
    assert torch.allclose(model.updates[0], future[:, 0, :])
    assert torch.allclose(pred[:, 0, :], torch.ones(2, 4))

    model = _FakeModel()
    pred, _ = rollout_mode(model, _fake_state(), 3, None, 'self', None)
    assert len(model.updates) == 3
    assert torch.allclose(model.updates[0], pred[:, 0, :])

    model = _FakeModel()
    rollout_mode(model, _fake_state(), 3, None, None, None)
    assert model.updates == []


def test_field_terms_reconstruct_total_field():
    torch.manual_seed(0)
    head = GraphODEDynamicsHead(dim=8, hidden_dim=16, cond_dim=3, profile_dim=4)
    z = torch.randn(2, 5, 8)
    adj = torch.rand(2, 5, 5)
    adj = 0.5 * (adj + adj.transpose(-1, -2))
    cond, profile = torch.randn(2, 3), torch.randn(2, 4)
    control = torch.randn(2, 5, 1)
    total, terms = head._field(z, adj, cond, profile, control, return_terms=True)
    rebuilt = terms['diffusion'] + terms['residual'] + terms['control']
    assert torch.allclose(total, rebuilt)
    assert terms['kappa'].shape == (2, 1)
