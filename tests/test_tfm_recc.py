# coding=utf-8
"""TFM RECC（Router–Expert Competence Coupling）单元测试（RECC 方案 §41–§43/§90）。

重点验证（§90 验收项）：
- Perfect agreement：q == p → KL ≈ 0；
- Correct best expert：competence 分布的 argmax 必须指向误差最小的专家；
- Uniform competence：误差全相同 → q 均匀分布；
- Gradient isolation（§43 核心验收项）：RECC-only backward 只更新 Router，
  Experts / TFM backbone / SC prior / CPM / 病理归一化器梯度全部为 0；
- Pair count：4 专家 → 6 对 candidate，pair-RECC 可反传；
- RevIN candidate：批量 candidate 反归一化与逐专家单独反归一化逐位一致（§39）；
- RECC 权重 warmup 调度（§24）与 Router 正则各项（§25）。
"""
import pytest
import torch

from models.tfm import NeuroTwinTFM
from models.tfm_experts import build_pair_indices
from train.tfm_coupling import (
    _competence_distribution,
    compute_expert_competence_losses,
    compute_tfm_coupling_diagnostics,
    compute_tfm_pair_recc,
    compute_tfm_recc,
    compute_tfm_router_regularization,
    get_tfm_recc_weight,
)


def _make_model(**kwargs):
    torch.manual_seed(0)
    defaults = dict(features=8, context_max=16, patch_len=4, dim=16,
                    num_layers=2, num_heads=4, dropout=0.0, cpm_horizon=4,
                    use_revin=True, pathology_input_dim=1,
                    tfm_patho_mode='moe', tfm_num_experts=4, tfm_top_k=2,
                    tfm_expert_hidden_dim=32, tfm_router_hidden_dim=16)
    defaults.update(kwargs)
    return NeuroTwinTFM(pretrain_mode=False, **defaults)


def _forward_with_candidates(model, b=3, seed=7):
    torch.manual_seed(seed)
    x = torch.randn(b, 8, 1, 16)
    sc = torch.rand(b, 8, 8)
    score = torch.randn(b, 1) * 5 + 18
    return model(x, sc, score, pathology_epoch=9, return_expert_pred=True,
                 return_pair_pred=True)


def _expert_pred_from_ell(target, ell_target):
    """构造 expert_pred 使逐专家 Huber 误差恰为 ell_target（δ=1：e≤1 取 0.5e²）。

    返回 [B, E, F] 的 candidate 张量（每行误差相同）。
    """
    e = ell_target.numel()
    errs = []
    for val in ell_target.tolist():
        if val <= 0.5:
            errs.append((2 * val) ** 0.5)
        else:
            errs.append(val + 0.5)          # Huber(e) = e − 0.5（e > δ）
    b, f = target.shape
    delta = torch.tensor(errs).view(1, e, 1).expand(b, e, f)
    return target.unsqueeze(1) + delta


# ─────────────────────── competence / RECC 损失 ───────────────────────

def test_perfect_agreement_zero_kl():
    """q == p_cpl → KL ≈ 0（§90 Perfect agreement）。"""
    torch.manual_seed(0)
    b, e, f = 4, 4, 8
    target = torch.randn(b, f)
    expert_pred = torch.randn(b, e, f)
    ell = compute_expert_competence_losses(expert_pred, target)
    q, _ = _competence_distribution(ell, tau=1.0)
    aux = {'tfm_expert_pred': expert_pred,
           'tfm_router_probs_coupling': q.detach().clone()}
    loss, stats = compute_tfm_recc(aux, target)
    assert float(loss) < 1e-6, f"q==p 时 KL 应为 0，实际 {float(loss)}"
    assert torch.isfinite(stats['recc_competence_entropy'])


def test_correct_best_expert():
    """误差 [0.5, 0.1, 0.8, 0.4] → argmax q == 1（§90 Correct best expert）。"""
    b, e, f = 3, 4, 8
    target = torch.zeros(b, f)
    ell_target = torch.tensor([0.5, 0.1, 0.8, 0.4])
    expert_pred = _expert_pred_from_ell(target, ell_target)
    ell = compute_expert_competence_losses(expert_pred, target)
    assert torch.allclose(ell, ell_target.expand(b, e), atol=1e-5)
    q, _ = _competence_distribution(ell, tau=1.0)
    assert int(q[0].argmax()) == 1
    aux = {'tfm_expert_pred': expert_pred,
           'tfm_router_probs_coupling': torch.full((b, e), 0.25)}
    _, stats = compute_tfm_recc(aux, target)
    assert torch.isfinite(stats['recc_loss'])


def test_uniform_competence():
    """误差全相同 → z-score 为 0 → q 均匀分布（§90 Uniform competence）。"""
    b, e, f = 3, 4, 8
    target = torch.randn(b, f)
    expert_pred = target.unsqueeze(1).expand(b, e, f) + 0.3   # 各专家等误差
    ell = compute_expert_competence_losses(expert_pred, target)
    q, _ = _competence_distribution(ell, tau=1.0)
    assert torch.allclose(q, torch.full_like(q, 1.0 / e), atol=1e-6)


def test_recc_missing_keys_returns_zero():
    """非 moe / recc-off 路径：aux 缺 candidate 键时返回 0 损失（防御性）。"""
    target = torch.randn(2, 8, 1, 1)
    loss, stats = compute_tfm_recc({}, target)
    assert float(loss) == 0.0 and stats == {}
    loss_p, _ = compute_tfm_pair_recc({}, target)
    assert float(loss_p) == 0.0


def test_recc_shape_mismatch_raises():
    target = torch.randn(2, 8)
    aux = {'tfm_expert_pred': torch.randn(2, 4, 8),
           'tfm_router_probs_coupling': torch.randn(3, 4)}
    with pytest.raises(ValueError):
        compute_tfm_recc(aux, target)


# ─────────────────────── 梯度隔离（§43 核心验收项） ───────────────────────

def test_recc_gradient_isolation():
    """RECC-only backward：Router grad != 0；Experts/backbone/SC/CPM/归一化器 == 0。

    耦合分支在 detach 的 state/cond 上计算 p_cpl（方案 §21），KL(sg(q) ‖ p_cpl)
    的梯度只更新 Router 参数——这是 RECC 的核心验收项（§43）。
    """
    model = _make_model()
    model.train()
    _, aux = _forward_with_candidates(model)
    target = torch.randn(3, 8, 1, 1)
    loss, _ = compute_tfm_recc(aux, target)
    loss.backward()

    named = dict(model.named_parameters())
    router_params = {n: p for n, p in named.items()
                     if '.pathology_moe.router.' in n}
    assert router_params, "未找到 Router 参数"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in router_params.values()), "Router 应获得梯度"
    for name, p in named.items():
        if '.pathology_moe.router.' in name:
            continue
        assert p.grad is None or p.grad.abs().sum() == 0, \
            f"RECC 梯度泄漏到非 Router 参数: {name}"
    # 抽样显式断言（文档 §43 清单；PathologyNormalizer 仅含 buffer 无参数）
    for name in ('patch_embed.patch_proj.weight',
                 'blocks.0.attn_temporal.qkv.weight',
                 'sc_prior.u_proj.weight',
                 'cpm_head.out_proj.weight',
                 'cpm_head.film.proj.weight',
                 'one_step_head.transition.0.weight',
                 'one_step_head.decode.weight',
                 'one_step_head.pathology_moe.experts.0.in_proj.weight',
                 'one_step_head.pathology_moe.experts.3.out_proj.weight'):
        p = named[name]
        assert p.grad is None or p.grad.abs().sum() == 0, \
            f"RECC-only backward 时 {name} 梯度应为 0"


def test_pair_recc_and_pair_count():
    """4 专家 → 6 对 candidate；pair-RECC 可计算并可反传（§27/§90 Pair count）。"""
    model = _make_model()
    _, aux = _forward_with_candidates(model, b=3)
    assert aux['tfm_pair_pred'].shape == (3, 6, 8)
    assert aux['tfm_pair_indices'].tolist() == [[0, 1], [0, 2], [0, 3],
                                                [1, 2], [1, 3], [2, 3]]
    target = torch.randn(3, 8, 1, 1)
    pair_loss, stats = compute_tfm_pair_recc(aux, target)
    assert torch.isfinite(pair_loss)
    pair_loss.backward()          # pair 分支与 V1 共用耦合通路，梯度同样只进 Router
    named = dict(model.named_parameters())
    assert named['one_step_head.pathology_moe.router.mlp.3.weight'].grad \
        .abs().sum() > 0
    for name, p in named.items():
        if '.pathology_moe.router.' in name:
            continue
        assert p.grad is None or p.grad.abs().sum() == 0, \
            f"pair-RECC 梯度泄漏到 {name}"


# ─────────────────────── RevIN candidate 一致性（§39/§90） ───────────────────────

def test_revin_candidate_denorm_matches_per_expert_call():
    """批量 candidate 反归一化必须与逐专家单独调用 BrainRevIN 一致（§90）。

    candidate 口径为 state_anchored_v2：入参是已加 x_last 锚点的归一化完整
    状态（``tfm_expert_state_norm``），反归一化后与正式输出同空间。
    """
    model = _make_model().eval()
    b = 3
    torch.manual_seed(11)
    x = torch.randn(b, 8, 1, 16)
    sc = torch.rand(b, 8, 8)
    score = torch.randn(b, 1) * 5 + 18
    with torch.no_grad():
        _, aux = model(x, sc, score, return_expert_pred=True)
        # 评估态下对同一输入重估 RevIN 统计量与前向内部逐位一致
        _, stats = model.rev_norm(x, 'norm')
        expert_state_norm = aux['tfm_expert_state_norm']     # [B,E,F]
        for e_i in range(expert_state_norm.shape[1]):
            single = model.rev_norm(
                expert_state_norm[:, e_i].unsqueeze(-1).unsqueeze(-1),
                'denorm', stats=stats).squeeze(-1).squeeze(-1)
            assert torch.allclose(aux['tfm_expert_pred'][:, e_i], single,
                                  atol=1e-6), f"expert {e_i} 反归一化不一致"


# ─────────────────────── 权重调度 / Router 正则 / 诊断 ───────────────────────

def test_recc_weight_schedule():
    """λ_recc 调度（§24）：start 前为 0，ramp 期线性，之后固定。"""
    ws = [get_tfm_recc_weight(e, 0.05, start_epoch=5, ramp_epochs=10)
          for e in range(17)]
    assert all(w == 0.0 for w in ws[:5])
    assert ws[5] == pytest.approx(0.05 * 1 / 10)
    assert ws[9] == pytest.approx(0.05 * 5 / 10)
    assert ws[14] == pytest.approx(0.05)
    assert ws[15] == pytest.approx(0.05)
    assert get_tfm_recc_weight(100, 0.05, 5, 10) == pytest.approx(0.05)
    assert get_tfm_recc_weight(10, 0.0, 5, 10) == 0.0       # 权重为 0 恒关闭


def test_router_regularization_terms():
    """balance / z-loss / entropy 各项有限且可反传；权重全 0 → 零正则（§25）。"""
    torch.manual_seed(0)
    model = _make_model()
    _, aux = _forward_with_candidates(model, b=5)
    reg, stats = compute_tfm_router_regularization(
        aux, balance_weight=0.01, z_weight=0.001, entropy_weight=0.0005)
    assert torch.isfinite(reg)
    reg.backward()
    named = dict(model.named_parameters())
    assert named['one_step_head.pathology_moe.router.mlp.3.weight'].grad \
        .abs().sum() > 0
    for key in ('router_balance', 'router_zloss', 'router_entropy'):
        assert torch.isfinite(stats[key])

    zero_reg, _ = compute_tfm_router_regularization(
        aux, balance_weight=0.0, z_weight=0.0, entropy_weight=0.0)
    assert float(zero_reg) == 0.0
    zero_reg2, _ = compute_tfm_router_regularization({}, 0.01, 0.001, 0.0005)
    assert float(zero_reg2) == 0.0                          # 非 moe 路径防御


def test_coupling_diagnostics_sanity():
    """构造 Router 完美对齐场景：agreement=1、regret=0、spearman=1（§64–§67）。

    Router 概率取 softmax(−ℓ/0.05)：对 −ℓ 严格单调 → argmax p = argmin ℓ、
    Top-2 与 Spearman 秩完全一致。
    """
    b, e, f = 4, 4, 8
    target = torch.zeros(b, f)
    base_errs = torch.tensor([0.5, 0.1, 0.8, 0.4])
    perm = torch.tensor([[0, 1, 2, 3],        # ℓ = [0.5,0.1,0.8,0.4] best=1
                         [1, 0, 2, 3],        # ℓ = [0.1,0.5,0.8,0.4] best=0
                         [3, 2, 1, 0],        # ℓ = [0.4,0.8,0.1,0.5] best=2
                         [2, 0, 3, 1]])       # ℓ = [0.8,0.5,0.4,0.1] best=3
    expert_pred = target.unsqueeze(1) + base_errs.view(1, e, 1)
    expert_pred = torch.gather(expert_pred, 1,
                               perm.view(b, e, 1).expand(b, e, f))
    ell = compute_expert_competence_losses(expert_pred, target)
    # base_errs 均 ≤ δ=1 → Huber(e) = 0.5·e²（单调，保序）
    assert torch.allclose(ell, 0.5 * base_errs[perm] ** 2, atol=1e-5)
    probs = torch.softmax(-ell / 0.05, dim=-1)
    aux = {'tfm_expert_pred': expert_pred, 'tfm_router_probs': probs}
    diag = compute_tfm_coupling_diagnostics(aux, target,
                                            actual_pred=target + 0.2)
    assert float(diag['router_top1_agreement']) == pytest.approx(1.0)
    assert float(diag['router_regret']) == pytest.approx(0.0, abs=1e-6)
    assert float(diag['router_top2_recall']) == pytest.approx(1.0)
    assert float(diag['router_competence_spearman']) == pytest.approx(1.0,
                                                                      abs=1e-5)
    assert float(diag['expert_loss_mean']) > 0
    assert torch.isfinite(diag['competence_entropy'])
    assert torch.isfinite(diag['oracle_best_expert_gap'])


def test_diagnostics_missing_keys_returns_empty():
    assert compute_tfm_coupling_diagnostics({}, torch.randn(2, 8)) == {}


def test_oracle_best_pair_gap():
    """§70：Router 实际 Top-2 pair 的损失 − oracle 最优等权 pair 损失。

    构造 Router Top-2 恰为最优 pair → gap ≈ 0；人为选次优 → gap > 0。
    """
    b, e, f = 3, 4, 8
    target = torch.zeros(b, f)
    # 单专家误差 [0.5, 0.1, 0.8, 0.4]；等权 pair 的混合误差可解析给出
    base = torch.tensor([0.5, 0.1, 0.8, 0.4])
    expert_pred = target.unsqueeze(1) + base.view(1, e, 1)
    pair_idx = build_pair_indices(e)
    # pair 误差取两专家误差的均值（candidate 为等权混合，target=0）
    pair_err = 0.5 * (base[pair_idx[:, 0]] + base[pair_idx[:, 1]])  # [P]
    pair_pred = target.unsqueeze(1) + pair_err.view(1, -1, 1)
    # Router Top-2 = {0, 1}（误差 0.5/0.1 的两个专家）→ 该 pair 误差 0.3，
    # 而最优 pair 为 {1,3}（0.25）→ gap = 0.05
    probs = torch.full((b, e), 0.1)
    probs[:, 0] = 0.4
    probs[:, 1] = 0.3
    aux = {'tfm_expert_pred': expert_pred, 'tfm_router_probs': probs,
           'tfm_pair_pred': pair_pred}
    diag = compute_tfm_coupling_diagnostics(aux, target)
    # Huber(δ=1, target=0)：loss(e)=0.5·e²。Router pair (0,1) 误差 0.3 → 0.045；
    # 最优 pair (1,3) 误差 0.25 → 0.03125；gap = 0.01375
    gap = float(diag['oracle_best_pair_gap'])
    assert gap == pytest.approx(0.5 * 0.3 ** 2 - 0.5 * 0.25 ** 2, abs=1e-6), gap
    # 无 pair candidate 时不应产出该指标（recc off / r2e 模式路径）
    diag2 = compute_tfm_coupling_diagnostics(
        {'tfm_expert_pred': expert_pred, 'tfm_router_probs': probs}, target)
    assert 'oracle_best_pair_gap' not in diag2


def test_moe_mode_requires_recc_mode_consistency():
    """recc 损失在 moe 模型 aux 上正常工作；film 模型 aux 缺键 → 返回 0（防御）。"""
    film_model = _make_model(tfm_patho_mode='film').eval()
    with torch.no_grad():
        _, aux = film_model(torch.randn(2, 8, 1, 16), torch.rand(2, 8, 8),
                            torch.randn(2, 1))
    loss, stats = compute_tfm_recc(aux, torch.randn(2, 8, 1, 1))
    assert float(loss) == 0.0 and stats == {}
