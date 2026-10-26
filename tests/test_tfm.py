# coding=utf-8
"""NeuroTwinTFM 完整模型单元测试（方案 §8–§11 / §15 / §16）。

重点验证：
- 双预测头契约：one-step 输出 [B,F,1,1]、CPM 输出 [B,F,H]（原始空间）；
- 初始化不变量：one-step 头零初始化残差 + persistence 锚点 → 初始预测 = x_t；
- FiLM 零初始化 → 有/无病理条件输出一致（不破坏预训练行为）；
- 标准接口（predict_next_state / rollout_next_states / forecast_horizon）；
- A_eff 全层共享与双随机性；梯度流；变长 K；intervention 接口默认关闭。
"""
import numpy as np
import pytest
import torch

from models.conditioning import ConditioningConfig
from models.tfm import NeuroTwinTFM
from utils.condition_schema import MDD_FIELDS, fit_condition_stats


def _make_model(pretrain=True, **kwargs):
    torch.manual_seed(0)
    defaults = dict(features=8, context_max=16, patch_len=4, dim=16,
                    num_layers=2, num_heads=4, dropout=0.0, cpm_horizon=4)
    if not pretrain:
        values = np.asarray([[1, 20, 12, 0, 5, 0, 10, 5],
                             [2, 30, 14, 1, 8, 1, 20, 10],
                             [1, 40, 16, 2, 10, 2, 30, 15],
                             [2, 50, 18, 0, 15, 0, 40, 20]], dtype=np.float32)
        stats = fit_condition_stats(MDD_FIELDS, values, np.ones_like(values), np.arange(4))
        defaults.update(condition_config=ConditioningConfig(group='mdd8', arm='cat').to_dict(),
                        condition_stats_payload=stats.to_dict())
    defaults.update(kwargs)
    return NeuroTwinTFM(pretrain_mode=pretrain, **defaults)


def _make_batch(b=2, f=8, k=16, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(b, f, 1, k)                    # [B,F,1,K] 内部布局
    sc = torch.rand(b, f, f)
    return x, sc


def test_forward_shapes_and_cpm_output():
    model = _make_model().eval()
    x, sc = _make_batch()
    with torch.no_grad():
        pred, aux = model(x, sc)
    assert pred.shape == (2, 8, 1, 1)              # one-step 主输出
    assert aux['cpm_pred'].shape == (2, 8, 4)      # CPM 全 horizon
    assert torch.isfinite(pred).all() and torch.isfinite(aux['cpm_pred']).all()


def test_one_step_init_equals_persistence():
    """零初始化残差 + x_t 锚点：初始 one-step 预测必须等于 persistence。"""
    model = _make_model().eval()
    x, sc = _make_batch()
    x_last = x[:, :, 0, -1]                        # [B,F]
    with torch.no_grad():
        pred, _ = model(x, sc)
    assert torch.allclose(pred[:, :, 0, 0], x_last, atol=1e-4), \
        "one-step 头初始应退化为 persistence（x_t 锚点 + 零初始化 Δ̂）"


def test_rollout_interface():
    model = _make_model().eval()
    hist = torch.randn(2, 12, 8)                   # [B,K,F]
    with torch.no_grad():
        out = model.predict_next_state(hist, torch.rand(2, 8, 8))
        assert out['pred_next'].shape == (2, 8, 1)
        assert out['pred_delta'].shape == (2, 8, 1)
        assert out['aux_info']['norm_stats'][0].shape == (2, 8, 1, 1)
        roll = model.rollout_next_states(hist, torch.rand(2, 8, 8), steps=3)
        assert roll.shape == (2, 3, 8)
        cpm = model.forecast_horizon(hist, torch.rand(2, 8, 8))
        assert cpm.shape == (2, 4, 8)              # [B,H,F] 非自回归


def test_cpm_depends_on_context_not_persistence_copy():
    """CPM 输出应随 context 变化（不是常数复制）。"""
    model = _make_model().eval()
    x, sc = _make_batch()
    with torch.no_grad():
        _, aux1 = model(x, sc)
        x2 = x.clone()
        x2[:, :, 0, :8] = torch.randn_like(x2[:, :, 0, :8]) * 3
        _, aux2 = model(x2, sc)
    assert not torch.allclose(aux1['cpm_pred'], aux2['cpm_pred'])


def test_variable_context_len_padding():
    """K=10（不是 patch_len=4 的整数倍）→ 开头补零、P=3，前向不报错。"""
    model = _make_model().eval()
    torch.manual_seed(3)
    x = torch.randn(2, 8, 1, 10)
    with torch.no_grad():
        pred, aux = model(x, torch.rand(2, 8, 8))
    assert pred.shape == (2, 8, 1, 1)
    assert aux['cpm_pred'].shape == (2, 8, 4)
    assert model.patch_embed.num_patches(10) == 3


def test_a_eff_doubly_stochastic_and_shared():
    """soft_prior 模式下 A_eff 近似双随机（行和≈1），且写入 aux 供全层共享。"""
    model = _make_model().eval()
    x, sc = _make_batch()
    with torch.no_grad():
        _, aux = model(x, sc)
    a_eff = aux['sc_prior']['A_eff']
    row_sums = a_eff.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
    assert a_eff.shape == (2, 8, 8)


def test_pretrain_mode_rejects_pathology_score():
    model = _make_model(pretrain=True).eval()
    x, sc = _make_batch()
    with pytest.raises(ValueError):
        model(x, sc, torch.rand(2, 1))


def test_finetune_film_zero_init_identity():
    """FiLM 零初始化：提供/不提供字段条件，初始输出应逐位一致。"""
    model = _make_model(pretrain=False).eval()
    x, sc = _make_batch()
    condition = {'cond_z': torch.zeros(2, 8),
                 'cond_code': torch.zeros(2, 8, dtype=torch.long),
                 'cond_mask': torch.ones(2, 8)}
    with torch.no_grad():
        pred_c, aux_c = model(x, sc, condition=condition)
        pred_n, aux_n = model(x, sc, condition=None)
    assert torch.allclose(pred_c, pred_n, atol=1e-6)
    assert torch.allclose(aux_c['cpm_pred'], aux_n['cpm_pred'], atol=1e-6)
    assert 'pathology_cond' in aux_c
    # FiLM 生效性：把 one-step 解码层置为非零（模拟训练后的状态）再放大
    # scale，输出必须改变（初始 decode 零初始化会屏蔽下游变化，属冷启动特性）
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.fill_(0.1)
        model.one_step_head.film.proj.weight.fill_(0.1)
        model.one_step_head.film.proj.bias.fill_(0.05)
        pred_c2, _ = model(x, sc, condition=condition)
        pred_n2, _ = model(x, sc, condition=None)
    assert not torch.allclose(pred_c2, pred_n2, atol=1e-6)


def test_finetune_film_gradient_path():
    """病理条件通路的梯度接线：CPM 头 film 直接有非零梯度；one-step 头 film
    因 decode 零初始化在初始步梯度为 0（冷启动自举），需在 decode 非零后验证。"""
    model = _make_model(pretrain=False)
    x, sc = _make_batch()
    condition = {'cond_z': torch.zeros(2, 8),
                 'cond_code': torch.zeros(2, 8, dtype=torch.long),
                 'cond_mask': torch.ones(2, 8)}
    pred, aux = model(x, sc, condition=condition)
    loss = pred.sum() + aux['cpm_pred'].sum()
    loss.backward()
    assert model.cpm_head.film.proj.weight.grad is not None
    assert model.cpm_head.film.proj.weight.grad.abs().sum() > 0
    # 模拟训练后的状态：decode 非零 → film 出现在梯度路径上
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.fill_(0.1)
    pred, _ = model(x, sc, condition=condition)
    pred.sum().backward()
    assert model.one_step_head.film.proj.weight.grad.abs().sum() > 0


def test_full_gradient_flow():
    """Check all ODE/CPM/variance/control parameters with their actual objectives."""
    model = _make_model(pretrain=False)
    x, sc = _make_batch()
    condition = {'cond_z': torch.zeros(2, 8),
                 'cond_code': torch.zeros(2, 8, dtype=torch.long),
                 'cond_mask': torch.ones(2, 8)}
    controls = torch.randn(2, 4, 8)
    pred, aux = model(x, sc, condition=condition, controls=controls)
    loss = pred.sum() + aux['cpm_pred'].sum() + aux['dynamics_logvar_norm'].sum()
    loss.backward()
    detached = [n for n, p in model.named_parameters() if p.grad is None]
    assert not detached, f"Disconnected parameters: {detached}"
    for name in ('patch_embed.patch_proj.weight', 'blocks.0.attn_temporal.qkv.weight',
                 'blocks.0.attn_roi.beta', 'sc_prior.u_proj.weight',
                 'one_step_head.decode_rate.weight', 'one_step_head.decode_rate.bias',
                 'one_step_head.variance_head.out.bias',
                 'cpm_head.out_proj.weight', 'cpm_head.horizon_embed'):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name
    for name in ('one_step_head.w1.weight', 'one_step_head.film.proj.weight'):
        assert dict(model.named_parameters())[name].grad.abs().sum() == 0, name
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.fill_(0.1)
    pred, _ = model(x, sc, condition=condition, controls=controls)
    pred.sum().backward()
    for name in ('one_step_head.w1.weight', 'one_step_head.w2.weight',
                 'one_step_head.film.proj.weight', 'one_step_head.control_proj.weight'):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name


def test_control_interface_uses_ode_interval_controls():
    """Zero interval controls preserve predictions; nonzero controls affect ODE readout."""
    model = _make_model().eval()
    assert not hasattr(model.cpm_head, 'intervention_embed')
    x, sc = _make_batch()
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.fill_(0.1)
        base, _ = model(x, sc)
        zero, _ = model(x, sc, controls=torch.zeros(2, 4, 8))
        stimulated, _ = model(x, sc, controls=torch.ones(2, 4, 8))
    assert torch.equal(base, zero)
    assert not torch.allclose(zero, stimulated)
    with pytest.raises(ValueError, match='controls'):
        model(x, sc, future_control=torch.zeros(2, 8, 4))


def test_mixed_precision_cuda_smoke():
    """GPU + AMP 冒烟（无 CUDA 时跳过）。"""
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用')
    model = _make_model(pretrain=False).cuda()
    x, sc = _make_batch()
    x, sc = x.cuda(), sc.cuda()
    condition = {'cond_z': torch.zeros(2, 8),
                 'cond_code': torch.zeros(2, 8, dtype=torch.long),
                 'cond_mask': torch.ones(2, 8)}
    condition = {key: value.cuda() for key, value in condition.items()}
    with torch.amp.autocast('cuda'):
        pred, aux = model(x, sc, condition=condition)
        loss = pred.float().abs().mean() + aux['cpm_pred'].float().abs().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_finetune_rejects_retired_scalar_pathology_input():
    model = _make_model(pretrain=False)
    x, sc = _make_batch()
    with pytest.raises(ValueError, match='字段 condition'):
        model(x, sc, pathology_score=torch.ones(2, 1))


def test_current_structure_rejects_unused_cpm_intervention_embedding():
    with pytest.raises(ValueError, match='CPM 不包含干预嵌入'):
        _make_model(cpm_intervention=True)
