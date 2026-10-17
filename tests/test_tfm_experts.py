# coding=utf-8
"""TFM 病理残差专家 MoE 单元测试（RECC 方案 §32–§36/§51/§89）。

重点验证（§89 验收项）：
- 形状契约：expert_delta_z [B,E,F,D]、router_probs [B,E]；
- mixture 权重和为 1；非 Top-K 专家权重恰为 0；
- 初始化不变量：初始 |Δz_path| 很小（§12 末层小初始化），moe 模式初始预测
  退化为 persistence；各专家初始方向互不相同；
- dense warmup / sparse / 部署态（epoch=None）三种路由行为（§14/§33）；
- film 模式回归兼容：不构建 MoE、aux 无 tfm_* 键（§91）；
- 配置 fail-fast（§51）；candidate 反归一化与 pair candidate 线性一致性；
- evaluate_variant 按 config.args 快照重建 moe 结构并 strict 加载（§92/§93）。
"""
import pytest
import torch

from models.tfm import NeuroTwinTFM
from models.tfm_experts import (
    TFMPathologyExpert,
    TFMPathologyMoE,
    TFMPathologyRouter,
    build_pair_indices,
)


def _make_moe(dim=16, cond_dim=3, num_experts=4, top_k=2, **kwargs):
    torch.manual_seed(0)
    defaults = dict(expert_hidden_dim=32, router_hidden_dim=16,
                    dense_warmup_epochs=5, dropout=0.0)
    defaults.update(kwargs)
    return TFMPathologyMoE(dim=dim, cond_dim=cond_dim,
                           num_experts=num_experts, top_k=top_k, **defaults)


def _make_model(pretrain=False, patho_mode='moe', **kwargs):
    torch.manual_seed(0)
    defaults = dict(features=8, context_max=16, patch_len=4, dim=16,
                    num_layers=2, num_heads=4, dropout=0.0, cpm_horizon=4,
                    use_revin=True, pathology_input_dim=1)
    defaults.update(kwargs)
    return NeuroTwinTFM(pretrain_mode=pretrain, tfm_patho_mode=patho_mode,
                        **defaults)


def _make_batch(b=3, f=8, k=16, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(b, f, 1, k)
    sc = torch.rand(b, f, f)
    score = torch.randn(b, 1) * 5 + 18
    return x, sc, score


# ─────────────────────────── 组件级测试 ───────────────────────────

def test_expert_delta_z_and_router_shapes():
    """形状契约（§89）：expert_delta_z [B,E,F,D]、router_probs [B,E]。"""
    b, f, d, e = 3, 8, 16, 4
    moe = _make_moe(dim=d)
    z_t, z_g = torch.randn(b, f, d), torch.randn(b, f, d)
    cond = torch.randn(b, 3)
    out = moe(z_t, z_g, cond, epoch=9, need_expert=True)
    assert out['tfm_expert_delta_z_norm'].shape == (b, e, f, d)
    assert out['tfm_router_probs'].shape == (b, e)
    assert out['tfm_router_weights'].shape == (b, e)
    assert out['tfm_router_logits'].shape == (b, e)
    assert out['tfm_topk_indices'].shape == (b, 2)
    assert out['delta_path_z'].shape == (b, f, d)
    assert torch.isfinite(out['delta_path_z']).all()


def test_mixture_weights_sum_to_one_dense_and_sparse():
    """dense（warmup 期）与 sparse（Top-K）的 mixture 权重都必须归一（§13/§14）。"""
    b, d = 5, 16
    moe = _make_moe(dim=d)
    z_t, z_g = torch.randn(b, 8, d), torch.randn(b, 8, d)
    cond = torch.randn(b, 3)
    dense = moe(z_t, z_g, cond, epoch=0)          # epoch 0 < warmup 5 → dense
    assert torch.allclose(dense['tfm_router_weights'],
                          dense['tfm_router_probs'], atol=1e-6)
    assert torch.allclose(dense['tfm_router_weights'].sum(-1),
                          torch.ones(b), atol=1e-5)
    sparse = moe(z_t, z_g, cond, epoch=99)        # warmup 之后 → Top-2
    assert torch.allclose(sparse['tfm_router_weights'].sum(-1),
                          torch.ones(b), atol=1e-5)


def test_non_topk_weights_exactly_zero():
    """Top-K 外的专家权重必须恰为 0（不参与 mixture 与梯度，§13）。"""
    b, d, e, k = 6, 16, 4, 2
    moe = _make_moe(dim=d, num_experts=e, top_k=k)
    z_t, z_g = torch.randn(b, 8, d), torch.randn(b, 8, d)
    out = moe(z_t, z_g, torch.randn(b, 3), epoch=9)
    w = out['tfm_router_weights']
    zero_mask = w == 0
    assert int(zero_mask.sum()) == b * (e - k)
    # 0 权重位置必须与 topk_indices 互补
    for i in range(b):
        topk = set(out['tfm_topk_indices'][i].tolist())
        for j in range(e):
            assert (w[i, j].item() == 0.0) == (j not in topk)


def test_initial_delta_path_small_and_experts_differ():
    """初始化不变量（§12）：末层小初始化 → |Δz_path| 很小；各专家方向不同。"""
    b, f, d = 4, 8, 16
    moe = _make_moe(dim=d)
    z_t = torch.randn(b, f, d)
    z_g = torch.randn(b, f, d)
    out = moe(z_t, z_g, torch.randn(b, 3), epoch=9, need_expert=True)
    # z 幅度 ~O(1)，Δz_path 应远小于输入幅度（末层 std=1e-3）
    assert out['delta_path_z'].abs().max().item() < 0.05
    delta = out['tfm_expert_delta_z_norm']        # [B,E,F,D]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(delta[:, i], delta[:, j], atol=1e-8), \
                "各专家初始修正方向必须不同（§12）"


def test_expert_zero_init_when_film_zero():
    """Expert FiLM 零初始化：有/无条件时初始 Δz 逐位一致（不破坏 HC 动力学）。"""
    torch.manual_seed(3)
    expert = TFMPathologyExpert(dim=16, hidden_dim=32, cond_dim=3, dropout=0.0)
    u = torch.randn(2, 8, 32)
    d_with = expert(u, torch.randn(2, 3))
    d_without = expert(u, None)
    assert torch.allclose(d_with, d_without, atol=1e-7)


def test_router_coupling_probs_match_forward_soft_probs():
    """coupling_probs 与 forward 的 soft_probs 对同一输入应逐位一致（§33）。"""
    torch.manual_seed(5)
    router = TFMPathologyRouter(state_dim=64, cond_dim=3, num_experts=4,
                                hidden_dim=16, dropout=0.0)
    state, cond = torch.randn(5, 64), torch.randn(5, 3)
    out = router(state, cond, top_k=2, dense=False)
    p_cpl = router.coupling_probs(state.detach(), cond.detach())
    assert torch.allclose(out['soft_probs'], p_cpl, atol=1e-6)
    assert torch.allclose(p_cpl.sum(-1), torch.ones(5), atol=1e-5)


def test_build_pair_indices():
    """4 专家 → 6 对：01,02,03,12,13,23（§27）。"""
    idx = build_pair_indices(4)
    assert idx.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert build_pair_indices(2).tolist() == [[0, 1]]


# ─────────────────────────── 模型级测试 ───────────────────────────

def test_moe_model_forward_candidate_shapes():
    """moe 模式前向：candidate / shared-only / 耦合分布 / pair candidate 形状。"""
    b, f, e = 3, 8, 4
    model = _make_model().eval()
    x, sc, score = _make_batch(b=b, f=f)
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=9,
                          return_expert_pred=True, return_pair_pred=True)
    assert pred.shape == (b, f, 1, 1)
    assert aux['tfm_expert_pred'].shape == (b, e, f)          # 原始空间完整 candidate
    assert aux['tfm_expert_state_norm'].shape == (b, e, f)    # 归一化完整状态
    assert aux['tfm_expert_delta_norm'].shape == (b, e, f)    # 归一化 residual
    assert aux['tfm_shared_pred'].shape == (b, f)             # shared-only 原始空间
    assert aux['tfm_expert_delta_z_norm'].shape == (b, e, f, 16)
    assert aux['tfm_router_probs_coupling'].shape == (b, e)
    assert aux['tfm_pair_pred'].shape == (b, 6, f)
    assert aux['tfm_pair_indices'].shape == (6, 2)
    assert aux['tfm_candidate_schema'] == 'state_anchored_v2_20261007'
    for key in ('tfm_expert_pred', 'tfm_shared_pred', 'tfm_pair_pred',
                'tfm_router_probs_coupling'):
        assert torch.isfinite(aux[key]).all()


def test_moe_init_equals_persistence():
    """零初始化（专家末层 + decode）→ moe 模式初始预测退化为 persistence（§12）。"""
    model = _make_model().eval()
    x, sc, score = _make_batch()
    x_last = x[:, :, 0, -1]
    with torch.no_grad():
        pred, _ = model(x, sc, score)
    assert torch.allclose(pred[:, :, 0, 0], x_last, atol=1e-4)


def test_deploy_epoch_none_uses_sparse_routing():
    """epoch=None（部署/评估态）与 warmup 后 epoch 的路由行为一致（Top-K 稀疏）。"""
    model = _make_model().eval()
    x, sc, score = _make_batch()
    with torch.no_grad():
        _, a_deploy = model(x, sc, score)
        _, a_sparse = model(x, sc, score, pathology_epoch=99)
        _, a_dense = model(x, sc, score, pathology_epoch=0)
    assert torch.allclose(a_deploy['tfm_router_weights'],
                          a_sparse['tfm_router_weights'], atol=1e-6)
    assert torch.allclose(a_dense['tfm_router_weights'],
                          a_dense['tfm_router_probs'], atol=1e-6)


def test_pair_candidate_equals_mean_of_expert_candidates():
    """raw 空间线性性：x̂_ij = (x̂_i + x̂_j)/2（共享 decode 基线 + RevIN 仿射均线性）。"""
    model = _make_model().eval()
    x, sc, score = _make_batch()
    with torch.no_grad():
        model.one_step_head.decode.weight.fill_(0.1)   # 模拟训练后状态（bias 仍 0）
        _, aux = model(x, sc, score, return_expert_pred=True,
                       return_pair_pred=True)
    ep, pp, pidx = (aux['tfm_expert_pred'], aux['tfm_pair_pred'],
                    aux['tfm_pair_indices'])
    for k, (i, j) in enumerate(pidx.tolist()):
        assert torch.allclose(pp[:, k], (ep[:, i] + ep[:, j]) / 2, atol=1e-5)


def test_film_mode_regression_no_moe_keys():
    """film 模式回归兼容（§91）：不构建 MoE、aux 无 tfm_* 键、数值行为不变。"""
    model = _make_model(patho_mode='film').eval()
    assert model.one_step_head.pathology_moe is None
    assert model.one_step_head.film is not None
    x, sc, score = _make_batch()
    with torch.no_grad():
        pred, aux = model(x, sc, score, return_expert_pred=True)  # 请求被忽略
    assert not any(k.startswith('tfm_') for k in aux)
    assert 'tfm_expert_pred' not in aux


def test_pretrain_mode_forces_no_pathology():
    """HC 预训练强制 none（§51/§52）：即使误传 moe 也不构建任何病理模块。"""
    model = _make_model(pretrain=True, patho_mode='moe').eval()
    assert model.tfm_patho_mode == 'none'
    assert model.one_step_head.pathology_moe is None
    assert model.one_step_head.film is None
    assert model.pathology_normalizer is None


def test_config_fail_fast():
    """配置约束 fail-fast（§51）。"""
    with pytest.raises(ValueError):
        _make_moe(num_experts=1)                       # E >= 2
    with pytest.raises(ValueError):
        _make_moe(num_experts=4, top_k=5)              # top_k <= E
    with pytest.raises(ValueError):
        TFMPathologyRouter(state_dim=64, cond_dim=0, num_experts=4,
                           temperature=0.0)            # 温度 > 0
    with pytest.raises(ValueError):
        # moe 需要病理条件通路：直接以 cond_dim=None 构造 One-Step 头应报错
        from models.tfm import OneStepDynamicsHead

        OneStepDynamicsHead(dim=16, cond_dim=None, pathology_mode='moe')


def test_dense_warmup_gives_all_experts_gradient():
    """dense warmup 期所有专家都经 mixture 获得梯度；sparse 期未选中专家无梯度
    （这正是 §14/§25 引入 dense warmup + load-balance 的原因）。"""
    model = _make_model(dropout=0.1).train()
    x, sc, score = _make_batch(b=8)
    with torch.no_grad():
        model.one_step_head.decode.weight.fill_(0.1)   # 模拟训练后状态（自举）
    model.zero_grad(set_to_none=True)
    pred, _ = model(x, sc, score, pathology_epoch=0)   # dense
    pred.sum().backward()
    named = dict(model.named_parameters())
    for i in range(4):
        g = named[f'one_step_head.pathology_moe.experts.{i}.out_proj.weight'].grad
        assert g is not None and g.abs().sum() > 0, f"dense 期 expert{i} 应有梯度"
    model.zero_grad(set_to_none=True)
    pred, _ = model(x, sc, score, pathology_epoch=99)  # sparse Top-2
    pred.sum().backward()
    got = [i for i in range(4)
           if named[f'one_step_head.pathology_moe.experts.{i}.out_proj.weight']
           .grad.abs().sum() > 0]
    assert 0 < len(got) <= 2, f"sparse 期获得梯度的专家应 <= top_k，实际 {got}"


def test_checkpoint_snapshot_rebuild_moe():
    """evaluate_variant 按 config.args 快照自省重建 moe 结构并 strict 加载（§92/§93）。"""
    from argparse import Namespace

    from experiments.evaluate_variant import build_model_from_args

    model = _make_model().eval()
    snapshot = dict(
        mode='finetune', num_rois=8, context_max=16, tfm_patch_len=4,
        tfm_dim=16, tfm_layers=2, tfm_heads=4, tfm_ff_ratio=4, dropout=0.0,
        tfm_one_step_hidden_dim=256, cpm_horizon=4, cpm_layers=2,
        cpm_intervention=False, sc_prior_mode='soft_prior',
        sc_lambda_mode='global', sc_lambda_init=0.7, sc_prior_rank=12,
        sc_delta_a=True, sc_delta_rank=None, sc_sinkhorn_iters=64,
        pathology_input_dim=1, pathology_norm_mode='robust_z',
        pathology_norm_quantiles=64, pathology_norm_rbf_knots=8, norm=True,
        tfm_patho_mode='moe', tfm_num_experts=4, tfm_top_k=2,
        tfm_expert_hidden_dim=256, tfm_router_hidden_dim=128,
        tfm_router_temperature=1.0, tfm_moe_dense_warmup_epochs=5)
    rebuilt = build_model_from_args(Namespace(**snapshot), torch.device('cpu'))
    assert rebuilt.tfm_patho_mode == 'moe'
    rebuilt.load_state_dict(model.state_dict(), strict=True)


# ─────────────────────── 专家去相关正则（Phase B-E2） ───────────────────────

def test_expert_diversity_zero_for_identical_and_orthogonal():
    """cos² 正则的数值锚点：完全相同 → 1；两两正交 → 0；缺 Δz → 零兜底。"""
    from train.tfm_coupling import compute_tfm_expert_diversity

    b, e, f, d = 2, 4, 3, 5
    same = torch.ones(b, e, f, d).expand(b, e, f, d).contiguous()
    loss_same, stats_same = compute_tfm_expert_diversity(
        {'tfm_expert_delta_z_norm': same})
    assert torch.isclose(loss_same, torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(stats_same['expert_dz_cos'], torch.tensor(1.0), atol=1e-6)

    # 逐样本不同的正交基（E=4 → 每样本 4 个相互正交的方向）
    ortho = torch.zeros(b, e, f * d)
    ortho[:, 0, 0] = 1.0
    ortho[:, 1, 1] = 1.0
    ortho[:, 2, 2] = 1.0
    ortho[:, 3, 3] = 1.0
    loss_o, stats_o = compute_tfm_expert_diversity(
        {'tfm_expert_delta_z_norm': ortho.reshape(b, e, f, d)})
    assert torch.isclose(loss_o, torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(stats_o['expert_dz_cos'], torch.tensor(0.0), atol=1e-7)

    loss_none, stats_none = compute_tfm_expert_diversity({})
    assert float(loss_none) == 0.0 and float(stats_none['expert_diversity']) == 0.0


def test_expert_diversity_gradient_reaches_experts():
    """梯度通路：div 损失应对 Expert 参数产生非零梯度（与 RECC 只训 Router 相反）。"""
    from train.tfm_coupling import compute_tfm_expert_diversity

    moe = _make_moe(dim=8, dropout=0.0)
    z_t, z_g = torch.randn(2, 8, 8), torch.randn(2, 8, 8)
    out = moe(z_t, z_g, cond=torch.randn(2, 3), epoch=0, need_expert=True)
    loss, _ = compute_tfm_expert_diversity(out)
    loss.backward()
    grads = [e.out_proj.weight.grad for e in moe.experts]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
