# coding=utf-8
"""NeuroTwinTFM 完整模型单元测试（方案 §8–§11 / §15 / §16）。

重点验证：
- 双预测头契约：one-step 输出 [B,F,1,1]、CPM 输出 [B,F,H]（原始空间）；
- 初始化不变量：one-step 头零初始化残差 + persistence 锚点 → 初始预测 = x_t；
- FiLM 零初始化 → 有/无病理条件输出一致（不破坏预训练行为）；
- 标准接口（predict_next_state / rollout_next_states / forecast_horizon）；
- A_eff 全层共享与双随机性；梯度流；变长 K；intervention 接口默认关闭。
"""
import pytest
import torch

from models.tfm import NeuroTwinTFM


def _make_model(pretrain=True, **kwargs):
    torch.manual_seed(0)
    defaults = dict(features=8, context_max=16, patch_len=4, dim=16,
                    num_layers=2, num_heads=4, dropout=0.0, cpm_horizon=4)
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
        assert out['aux_info']['cpm_pred'].shape == (2, 8, 4)
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
    """FiLM 零初始化：提供/不提供 HAMD 评分，初始输出应逐位一致。"""
    model = _make_model(pretrain=False, pathology_input_dim=1).eval()
    x, sc = _make_batch()
    score = torch.tensor([[10.0], [25.0]])
    with torch.no_grad():
        pred_c, aux_c = model(x, sc, score)
        pred_n, aux_n = model(x, sc, None)
    assert torch.allclose(pred_c, pred_n, atol=1e-6)
    assert torch.allclose(aux_c['cpm_pred'], aux_n['cpm_pred'], atol=1e-6)
    assert 'pathology_cond' in aux_c
    # FiLM 生效性：把 one-step 解码层置为非零（模拟训练后的状态）再放大
    # scale，输出必须改变（初始 decode 零初始化会屏蔽下游变化，属冷启动特性）
    with torch.no_grad():
        model.one_step_head.decode.weight.fill_(0.1)
        model.one_step_head.film.proj.weight.fill_(0.1)
        model.one_step_head.film.proj.bias.fill_(0.05)
        pred_c2, _ = model(x, sc, score)
        pred_n2, _ = model(x, sc, None)
    assert not torch.allclose(pred_c2, pred_n2, atol=1e-6)


def test_finetune_film_gradient_path():
    """病理条件通路的梯度接线：CPM 头 film 直接有非零梯度；one-step 头 film
    因 decode 零初始化在初始步梯度为 0（冷启动自举），需在 decode 非零后验证。"""
    model = _make_model(pretrain=False, pathology_input_dim=1)
    x, sc = _make_batch()
    score = torch.tensor([[10.0], [25.0]])
    pred, aux = model(x, sc, score)
    loss = pred.sum() + aux['cpm_pred'].sum()
    loss.backward()
    assert model.cpm_head.film.proj.weight.grad is not None
    assert model.cpm_head.film.proj.weight.grad.abs().sum() > 0
    # 模拟训练后的状态：decode 非零 → film 出现在梯度路径上
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.one_step_head.decode.weight.fill_(0.1)
    pred, _ = model(x, sc, score)
    pred.sum().backward()
    assert model.one_step_head.film.proj.weight.grad.abs().sum() > 0


def test_full_gradient_flow():
    """梯度接线审计：区分「初始即活参数」与「零初始化冷启动门控参数」。

    one-step 头为 persistence 锚点 + 零初始化残差解码（稳定性设计）：
    初始步只有 decode 获得梯度，transition / FiLM 从第 2 步起自举
    （与 DiT 的 AdaLN-zero / GPT-2 residual 零初始化门控同一模式）。
    """
    model = _make_model(pretrain=False, pathology_input_dim=1)
    x, sc = _make_batch()
    score = torch.tensor([[10.0], [25.0]])
    pred, aux = model(x, sc, score)
    (pred.sum() + aux['cpm_pred'].sum()).backward()
    # 1) 连通性：每个参数都必须有梯度（None = 未接入计算图，属 bug）
    detached = [n for n, p in model.named_parameters() if p.grad is None]
    assert not detached, f"以下参数未接入梯度图: {detached}"
    # 2) 初始即应非零梯度的关键参数（backbone / SC 先验 / CPM 头 / decode）
    for name in ('patch_embed.patch_proj.weight', 'blocks.0.attn_temporal.qkv.weight',
                 'blocks.0.attn_roi.beta', 'sc_prior.u_proj.weight',
                 'one_step_head.decode.weight', 'one_step_head.decode.bias',
                 'cpm_head.out_proj.weight', 'cpm_head.horizon_embed'):
        p = dict(model.named_parameters())[name]
        assert p.grad.abs().sum() > 0, f"关键参数 {name} 初始梯度为 0"
    # 3) 冷启动门控参数（初始为 0，第 2 步起自举）
    for name in ('one_step_head.transition.0.weight',
                 'one_step_head.film.proj.weight'):
        p = dict(model.named_parameters())[name]
        assert float(p.grad.abs().sum()) == 0.0, \
            f"{name} 初始梯度应为 0（零初始化冷启动），否则初始化策略被改动"
    # 4) 模拟训练后的状态（decode 非零）→ 冷启动参数进入梯度路径
    #    （transition[-1] 零初始化门控其前层，末层线性本身可先获得梯度）
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.one_step_head.decode.weight.fill_(0.1)
    pred, _ = model(x, sc)
    pred.sum().backward()
    assert dict(model.named_parameters())[
        'one_step_head.transition.3.weight'].grad.abs().sum() > 0


def test_intervention_interface_off_by_default_and_on():
    """V1 默认不构建 intervention 嵌入；显式开启时 future_known 控制量生效。"""
    x, sc = _make_batch()
    hist = torch.randn(2, 12, 8)
    model_off = _make_model(pretrain=False, pathology_input_dim=1).eval()
    assert model_off.cpm_head.intervention_embed is None
    with torch.no_grad():
        pred, aux = model_off(x, sc, torch.rand(2, 1))
    assert pred.shape == (2, 8, 1, 1)

    model_on = _make_model(pretrain=False, pathology_input_dim=1,
                           cpm_intervention=True).eval()
    u = torch.zeros(2, 8, 4)
    with torch.no_grad():
        base = model_on.forecast_horizon(hist, sc, torch.rand(2, 1))
        stim = model_on.forecast_horizon(hist, sc, torch.rand(2, 1),
                                         future_control=u)
        assert torch.allclose(base, stim, atol=1e-6)   # U=0 ⇒ 基线仿真
        u1 = torch.ones(2, 8, 4)
        stim1 = model_on.forecast_horizon(hist, sc, torch.rand(2, 1),
                                          future_control=u1)
    assert not torch.allclose(stim, stim1)             # 干预量改变未来预测


def test_mixed_precision_cuda_smoke():
    """GPU + AMP 冒烟（无 CUDA 时跳过）。"""
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用')
    model = _make_model(pretrain=False, pathology_input_dim=1).cuda()
    x, sc = _make_batch()
    x, sc = x.cuda(), sc.cuda()
    score = torch.tensor([[10.0], [25.0]]).cuda()
    with torch.amp.autocast('cuda'):
        pred, aux = model(x, sc, score)
        loss = pred.float().abs().mean() + aux['cpm_pred'].float().abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
