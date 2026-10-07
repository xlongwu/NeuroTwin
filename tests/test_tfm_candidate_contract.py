# coding=utf-8
"""完整 candidate 预测契约测试（RECC 执行计划 §5.2 验收表 / G1 工程验收门）。

背景（执行计划 §2.2/§5.1，问题核查 §2.3/§3.1）：历史运行的 candidate 构造漏加
``x_last`` 锚点——``decode(z_shared+Δz_i)`` 是相对 x_last 的 **residual**，却被
直接当作完整状态反归一化，导致 candidate 误差处于 persistence 量级、与 target
的比较跨空间。共同偏移不会在“candidate 与固定 target 的损失”中自动抵消，漏锚
可以翻转 competence 排名（``test_anchor_omission_flips_expert_ranking``），因此
原排名指标与 RECC 监督不能默认有效。

本文件验证 state_anchored_v2 契约下的完整链路：

    expert_delta_norm = decode(z_shared + expert_delta_z)        # residual
    expert_state_norm = x_last_norm[:, None, :] + expert_delta_norm
    expert_state_raw  = denorm_with_context_stats(expert_state_norm)

以及梯度边界（Top-1/Top-2 Router task 梯度）、冻结阶段参数覆盖、diagnose
开关一致性（观察工具不得改变输出/损失/RNG/单步更新）与精度容差量化。
"""
from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from models.common import BrainRevIN
from models.tfm import NeuroTwinTFM, OneStepDynamicsHead, TFM_CANDIDATE_SCHEMA
from train.optim import get_param_groups, set_finetune_stage
from train.tfm_coupling import (
    compute_expert_competence_losses,
    compute_tfm_router_regularization,
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


def _make_batch(b=3, f=8, k=16, seed=7):
    torch.manual_seed(seed)
    x = torch.randn(b, f, 1, k)
    sc = torch.rand(b, f, f)
    score = torch.randn(b, 1) * 5 + 18
    return x, sc, score


def _patch_one_hot_routing(model):
    """把 Router 的实际 mixture 权重替换为 one-hot（其余输出不变）。

    用于验证“正式预测 == 被选专家的完整 candidate”：真实 forward 中
    routing_weights 来自 softmax/Top-K 重归一，one-hot 是其权重和为 1 的特例。
    """
    router = model.one_step_head.pathology_moe.router
    original = router.forward

    def one_hot(state, cond, top_k=None, dense=False):
        out = original(state, cond, top_k=top_k, dense=dense)
        idx = out['soft_probs'].argmax(dim=-1, keepdim=True)      # [B,1]
        weights = torch.zeros_like(out['soft_probs']).scatter_(1, idx, 1.0)
        out['routing_weights'] = weights
        out['topk_indices'] = idx
        return out

    router.forward = one_hot          # 实例属性替换（不入类，不影响其他实例）
    return router


# ─────────────────── 契约恒等式（§5.2 表 1–3） ───────────────────

def test_one_hot_routing_formal_equals_full_candidate():
    """one-hot 路由：正式 prediction 必须逐位等于对应完整 candidate（原始空间）。"""
    model = _make_model().eval()
    x, sc, score = _make_batch()
    router = _patch_one_hot_routing(model)
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=None,
                          return_expert_pred=True)
    chosen = aux['tfm_topk_indices'][:, 0]                        # [B]
    cand = aux['tfm_expert_pred']                                 # [B,E,F]
    formal = pred[:, :, 0, 0]                                     # [B,F]
    assert torch.allclose(
        formal, cand[torch.arange(formal.shape[0]), chosen], atol=1e-5), \
        "one-hot 路由下正式预测必须等于被选专家的完整 candidate（state_anchored_v2）"


def test_identical_experts_all_candidates_equal_mixture():
    """相同专家：全部 candidate 相等且等于正式 mixture（权重和为 1 的分解不唯一，
    但相同专家时分解唯一）。"""
    model = _make_model().eval()
    moe = model.one_step_head.pathology_moe
    sd = moe.experts[0].state_dict()
    for e in moe.experts[1:]:
        e.load_state_dict(sd)
    x, sc, score = _make_batch()
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=9,
                          return_expert_pred=True)
    cand = aux['tfm_expert_pred']                                 # [B,E,F]
    for i in range(1, cand.shape[1]):
        assert torch.allclose(cand[:, i], cand[:, 0], atol=1e-6)
    assert torch.allclose(pred[:, :, 0, 0], cand[:, 0], atol=1e-5)


def test_raw_mixture_equals_weighted_sum_of_raw_candidates():
    """共享线性 decode + 仿射 RevIN + 权重和为 1 ⇒ raw mixture = Σ w_i·raw candidate。

    dense（epoch=0，权重=soft_probs）与 sparse（epoch=None，Top-K 重归一）都必须
    满足；这是 candidate 与正式输出同一变换链路的充分条件（§5.2 表 3）。
    """
    model = _make_model().eval()
    # 非平凡 RevIN affine：排除恒等 RevIN 掩盖广播/affine 错误的可能
    with torch.no_grad():
        model.rev_norm.affine_weight.fill_(1.3)
        model.rev_norm.affine_bias.fill_(0.2)
    x, sc, score = _make_batch(b=4)
    for epoch in (0, None):
        with torch.no_grad():
            pred, aux = model(x, sc, score, pathology_epoch=epoch,
                              return_expert_pred=True)
        w = aux['tfm_router_weights']                             # [B,E]
        assert torch.allclose(w.sum(-1), torch.ones(4), atol=1e-5)
        mixture = torch.einsum('be,bef->bf', w, aux['tfm_expert_pred'])
        assert torch.allclose(pred[:, :, 0, 0], mixture, atol=1e-4), \
            f"epoch={epoch} 时 raw mixture 恒等不成立"


def test_pair_candidate_equals_mean_and_actual_top2_mixture():
    """raw pair candidate = (cand_i + cand_j)/2；实际 Top-2 配比（Router 重归一权重）
    的 mixture 亦等于对应 candidate 加权（部署权重包含在 pair 候选集合内）。"""
    model = _make_model().eval()
    x, sc, score = _make_batch()
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=None,
                          return_expert_pred=True, return_pair_pred=True)
    cand, pair, pidx = (aux['tfm_expert_pred'], aux['tfm_pair_pred'],
                        aux['tfm_pair_indices'])
    for k, (i, j) in enumerate(pidx.tolist()):
        assert torch.allclose(pair[:, k], (cand[:, i] + cand[:, j]) / 2, atol=1e-5)
    # 实际路由：Top-K 外权重恰为 0 → mixture == Σ_{top2} w·cand
    w = aux['tfm_router_weights']
    mixture = torch.einsum('be,bef->bf', w, cand)
    assert torch.allclose(pred[:, :, 0, 0], mixture, atol=1e-4)


# ─────────────────── 退化为 persistence / shared-only（§5.2 表 4–5） ───────────────────

def test_zero_decode_residual_equals_persistence():
    """decode residual 全零：candidate / 正式输出 / shared-only 全部等于 persistence。

    这同时是锚点回归守卫：漏加 x_last 的 v1 实现会得到 denorm(0)=context 均值
    而非 x_last，本测试在其上必然失败。
    """
    model = _make_model().eval()
    with torch.no_grad():
        model.one_step_head.decode.weight.zero_()
        model.one_step_head.decode.bias.zero_()
    x, sc, score = _make_batch()
    x_last = x[:, :, 0, -1]                                       # [B,F] persistence
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=9,
                          return_expert_pred=True)
    assert torch.allclose(pred[:, :, 0, 0], x_last, atol=1e-5)
    assert torch.allclose(aux['tfm_expert_pred'], x_last.unsqueeze(1), atol=1e-5)
    assert torch.allclose(aux['tfm_shared_pred'], x_last, atol=1e-5)


def test_zero_pathology_residual_equals_shared_only():
    """病理 residual 全零（Δz_i=0）：正式输出与全部 candidate 等于明确绕过病理
    分支的 shared-only 输出（§5.2 表 5）。"""
    model = _make_model().eval()
    moe = model.one_step_head.pathology_moe
    with torch.no_grad():
        for e in moe.experts:
            e.out_proj.weight.zero_()
            e.out_proj.bias.zero_()
    x, sc, score = _make_batch()
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=9,
                          return_expert_pred=True)
    shared = aux['tfm_shared_pred']
    formal = pred[:, :, 0, 0]
    assert torch.allclose(formal, shared, atol=1e-6)
    assert torch.allclose(aux['tfm_expert_pred'], shared.unsqueeze(1), atol=1e-6)


# ─────────────────── RevIN 仿射与 no-RevIN（§5.2 表 6） ───────────────────

def test_nontrivial_revin_affine_matches_manual_denorm():
    """非平凡 RevIN（非零 bias、非单位 weight）：candidate 反归一化与手写
    RevIN affine 逐位一致；不同样本的 mean/std 不同时广播正确。"""
    model = _make_model().eval()
    with torch.no_grad():
        model.rev_norm.affine_weight.copy_(
            torch.linspace(0.8, 1.5, model.rev_norm.num_features))
        model.rev_norm.affine_bias.copy_(
            torch.linspace(-0.3, 0.4, model.rev_norm.num_features))
    x, sc, score = _make_batch(b=3)
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=9,
                          return_expert_pred=True)
        _, stats = model.rev_norm(x, 'norm')
        mean, std = stats
        # 手写 affine：x_raw = (x_norm − bias)/(weight+eps)·std + mean（§3.1）
        state_norm = aux['tfm_expert_state_norm']                 # [B,E,F]
        w = model.rev_norm.affine_weight.view(1, 1, -1)
        b_ = model.rev_norm.affine_bias.view(1, 1, -1)
        manual = (state_norm - b_) / (w + model.rev_norm.eps) \
            * std.view(3, 1, -1) + mean.view(3, 1, -1)
        assert torch.allclose(aux['tfm_expert_pred'], manual, atol=1e-5)
        # 不同被试统计量确实不同（否则广播错误被掩盖）
        assert (std.view(3, -1)[0] - std.view(3, -1)[1]).abs().max() > 1e-3


def test_no_revin_candidates_still_contract_consistent():
    """no-RevIN（use_revin=False）：identity denorm 下各恒等式仍成立。"""
    model = _make_model(use_revin=False).eval()
    x, sc, score = _make_batch()
    router = _patch_one_hot_routing(model)
    with torch.no_grad():
        pred, aux = model(x, sc, score, pathology_epoch=None,
                          return_expert_pred=True)
    chosen = aux['tfm_topk_indices'][:, 0]
    formal = pred[:, :, 0, 0]
    assert torch.allclose(
        formal, aux['tfm_expert_pred'][torch.arange(formal.shape[0]), chosen],
        atol=1e-5)
    # x_last 锚点：zero decode 后 candidate == x_last（no-RevIN 时即原始锚点）
    with torch.no_grad():
        model.one_step_head.decode.weight.zero_()
        model.one_step_head.decode.bias.zero_()
        pred2, aux2 = model(x, sc, score, pathology_epoch=9,
                            return_expert_pred=True)
    x_last = x[:, :, 0, -1]
    assert torch.allclose(aux2['tfm_expert_pred'], x_last.unsqueeze(1), atol=1e-6)
    assert torch.allclose(pred2[:, :, 0, 0], x_last, atol=1e-6)


# ─────────────────── 漏锚翻转排名反例（§5.2 表 7） ───────────────────

def test_anchor_omission_flips_expert_ranking():
    """问题核查 §2.3 最小反例：共同漏加锚点可以翻转专家排名（target=1, anchor=1,
    residual candidate 0 / 0.2）。锚定后 Expert 0 胜；漏锚后 Expert 1 胜。

    该反例说明 v1 candidate 下的 agreement/regret/q 与 RECC 监督不能默认有效，
    也是本契约测试存在的核心动机。
    """
    rev = BrainRevIN(num_features=1)
    with torch.no_grad():                       # 恒等仿射：raw == norm
        rev.affine_weight.fill_(1.0)
        rev.affine_bias.fill_(0.0)
    x_last_norm = torch.ones(1, 1)              # anchor=1
    residual = torch.zeros(1, 2, 1)             # [B=1, E=2, F=1]
    residual[0, 1, 0] = 0.2                     # residual candidate 0 / 0.2
    target = torch.ones(1, 1)                   # target=1（raw==norm）
    # 恒等仿射下 denorm(x)=x：mean=0 / std=1 的 stats
    stats_zero = (torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))

    def _denorm(norm_states):
        cand4 = norm_states.permute(0, 2, 1).unsqueeze(-1)        # [1,1,E,1]
        return rev(cand4, 'denorm', stats=stats_zero).squeeze(-1).permute(0, 2, 1)

    anchored_raw = _denorm(x_last_norm.unsqueeze(1) + residual)   # v2 契约
    unanchored_raw = _denorm(residual)                            # v1（漏锚）

    ell_anchored = compute_expert_competence_losses(anchored_raw, target)
    ell_unanchored = compute_expert_competence_losses(unanchored_raw, target)
    assert int(ell_anchored[0].argmin()) == 0, \
        f"锚定后 Expert 0 应胜，实际 {ell_anchored.tolist()}"
    assert int(ell_unanchored[0].argmin()) == 1, \
        f"漏锚后 Expert 1 应胜，实际 {ell_unanchored.tolist()}"


# ─────────────────── 梯度边界（§5.2 表 8–10） ───────────────────

def test_task_gradient_selected_experts_sparse():
    """sparse Top-2：task loss 只训练被选中专家；Router 经 routing_weights 获得
    task 梯度（Top-2 时标准实现非零，审计参考值 ≈0.07 量级）。"""
    model = _make_model(dropout=0.1).train()
    with torch.no_grad():
        # decode 零初始化时 expert→输出梯度为 0，先模拟训练后状态（与既有
        # test_dense_warmup_gives_all_experts_gradient 同一口径）
        model.one_step_head.decode.weight.fill_(0.1)
    x, sc, score = _make_batch(b=8)
    pred, aux = model(x, sc, score, pathology_epoch=99)   # sparse
    pred.sum().backward()
    named = dict(model.named_parameters())
    w = aux['tfm_router_weights'].detach()
    for i in range(4):
        g = named[f'one_step_head.pathology_moe.experts.{i}.out_proj.weight'].grad
        selected = bool((w[:, i] > 0).any().item())
        if selected:
            assert g is not None and g.abs().sum() > 0, \
                f"被选中 expert{i} 应有 task 梯度"
        else:
            assert g is None or g.abs().sum() == 0, \
                f"未选中 expert{i} 不应有 task 梯度"
    router_g = named['one_step_head.pathology_moe.router.mlp.3.weight'].grad
    assert router_g is not None and router_g.abs().sum() > 0, \
        "Top-2 下 Router 应获得 task 梯度"


def test_top1_router_task_gradient_zero_but_regularizer_updates_router():
    """Top-1/Top-2 梯度记录（§5.2 表 10）：标准归一化 Top-1 下 task loss 对
    Router logits 的梯度为零（选择边界不可微、唯一选中权重恒为 1）——这是
    **实测记录**而非预设：它把 E1 标注为固定/间接分区实验，不作为工程 bug。
    Router 正则（balance/z/entropy）仍可更新 Router。"""
    model = _make_model(dropout=0.1, tfm_top_k=1).train()
    with torch.no_grad():
        model.one_step_head.decode.weight.fill_(0.1)   # 使 Router 梯度非退化
    x, sc, score = _make_batch(b=8)
    pred, _ = model(x, sc, score, pathology_epoch=99)
    pred.sum().backward()
    named = dict(model.named_parameters())
    router_keys = [k for k in named if '.pathology_moe.router.' in k]
    task_grad_max = max(
        (named[k].grad.abs().max().item() if named[k].grad is not None else 0.0)
        for k in router_keys)
    # 解析上为 0（唯一选中权重 p/p≡1，TopK 索引不可微）；浮点重归一化的除法
    # 残差 ≈1e-10（相对 p 的舍入），比 Top-2 的 ≈0.07 低 5 个数量级以上。
    assert task_grad_max < 1e-6, \
        f"标准 Top-1 下 Router task 梯度应仅为数值噪声（<1e-6），实测 {task_grad_max}"

    model.zero_grad(set_to_none=True)
    _, aux = model(x, sc, score, pathology_epoch=99)
    reg, _ = compute_tfm_router_regularization(aux)
    reg.backward()
    reg_grad_max = max(
        (named[k].grad.abs().max().item() if named[k].grad is not None else 0.0)
        for k in router_keys)
    assert reg_grad_max > 0, "正则-only 路径应能更新 Router"


def test_recc_only_backward_gradient_isolation():
    """RECC-only backward：Router 可获梯度，expert/backbone/SC/CPM 不获（表 8）。"""
    model = _make_model().train()
    x, sc, score = _make_batch()
    _, aux = model(x, sc, score, pathology_epoch=9, return_expert_pred=True)
    target = torch.randn(3, 8, 1, 1)
    from train.tfm_coupling import compute_tfm_recc
    loss, _ = compute_tfm_recc(aux, target)
    loss.backward()
    named = dict(model.named_parameters())
    router_params = {n: p for n, p in named.items() if '.pathology_moe.router.' in n}
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in router_params.values()), "Router 应获得 RECC 梯度"
    for name, p in named.items():
        if '.pathology_moe.router.' in name:
            continue
        assert p.grad is None or p.grad.abs().sum() == 0, \
            f"RECC 梯度泄漏到非 Router 参数: {name}"


# ─────────────────── 冻结阶段覆盖（§5.2 表 11） ───────────────────

def test_frozen_stage_param_groups_cover_moe():
    """冻结阶段：Router 与全部应训练专家在 optimizer 覆盖内，shared 分支冻结。

    这验证 Stage-1 冻结期 optimizer 确实包含全部 MoE 参数（§5.2 表 11 /
    问题核查 §3.6：参数组遗漏会让部分专家静默无梯度）。
    """
    model = _make_model()
    set_finetune_stage(model, backbone_unfrozen=False)
    named = dict(model.named_parameters())
    # 冻结期应训练：病理 MoE（Router+Experts）+ 条件适配器（CPM FiLM）；
    # 其余 shared/backbone 参数全部冻结
    adapter_ids = set()
    for _, module in model.named_modules():
        if getattr(module, 'is_conditioning_adapter', False):
            adapter_ids.update(id(p) for p in module.parameters())
    for name, p in named.items():
        in_moe = '.pathology_moe.' in name
        if in_moe or id(p) in adapter_ids:
            assert p.requires_grad, f"冻结期应参与训练: {name}"
        else:
            assert not p.requires_grad, f"冻结阶段 shared 参数不应训练: {name}"
    args = Namespace(mode='finetune', lr_peak=1e-4, backbone_lr_scale=0.2,
                     adapter_lr_scale=1.0, tfm_moe_lr_scale=1.0)
    groups = {g['name']: g['params'] for g in get_param_groups(model, args)}
    assert 'tfm_moe' in groups
    moe_total = sum(p.numel() for n, p in named.items() if '.pathology_moe.' in n)
    assert sum(p.numel() for p in groups['tfm_moe']) == moe_total
    # 全部 4 个专家都被覆盖
    for i in range(4):
        expert_params = {id(p) for n, p in named.items()
                         if f'.experts.{i}.' in n}
        covered = {id(p) for p in groups['tfm_moe']}
        assert expert_params <= covered, f"expert{i} 未被 optimizer 覆盖"


# ─────────────────── diagnose 开关一致性（§5.2 表 12） ───────────────────

def test_diagnose_switch_output_loss_and_update_consistency():
    """diagnose 作为观察工具不得改变训练：同输入、同状态、同 RNG 下，
    return_expert_pred on/off 的正式输出、task loss、梯度与单步 SGD 更新逐位一致。

    dropout>0 使 RNG 消耗差异可被检测（moe 分支无论开关都运行全部专家，
    因此 RNG 流应完全一致）。
    """
    model = _make_model(dropout=0.1).train()
    model_c = deepcopy(model)                       # diagnose 关闭的对照
    x, sc, score = _make_batch(b=4)
    lr = 1e-2

    def _step(m, need_expert):
        torch.manual_seed(123)                      # 同 RNG 起点
        pred, aux = m(x, sc, score, pathology_epoch=9,
                      return_expert_pred=need_expert)
        loss = pred.sum()
        loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
        return pred.detach(), float(loss), aux

    pred_a, loss_a, aux_a = _step(model, True)      # diagnose 打开（多 candidate）
    pred_b, loss_b, _ = _step(model_c, False)       # diagnose 关闭
    assert torch.allclose(pred_a, pred_b, atol=1e-6), "正式输出不应受 diagnose 影响"
    assert loss_a == pytest.approx(loss_b, abs=1e-6)
    assert 'tfm_expert_pred' in aux_a               # 打开时才有 candidate
    for (n_a, p_a), (n_b, p_b) in zip(model.named_parameters(),
                                      model_c.named_parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-7), \
            f"diagnose 开关改变了参数更新: {n_a}"


# ─────────────────── 精度容差量化（§5.2 表 13） ───────────────────

def test_fp64_vs_fp32_candidate_tolerance():
    """FP64 vs FP32 的 candidate 损失差量化：逻辑错误（漏锚 ≈0.44 量级）必须
    远大于数值误差（≲1e-4），数值容差不得掩盖口径错误。"""
    torch.manual_seed(0)
    model64 = _make_model().eval().double()
    torch.manual_seed(0)
    model32 = _make_model().eval()
    x, sc, score = _make_batch()
    with torch.no_grad():
        _, aux64 = model64(x.double(), sc.double(), score.double(),
                           pathology_epoch=9, return_expert_pred=True)
        _, aux32 = model32(x, sc, score, pathology_epoch=9,
                           return_expert_pred=True)
    target = torch.randn(3, 8)
    ell64 = compute_expert_competence_losses(
        aux64['tfm_expert_pred'].float(), target)
    ell32 = compute_expert_competence_losses(aux32['tfm_expert_pred'], target)
    diff = float((ell64 - ell32).abs().max())
    assert diff < 1e-3, f"FP64/FP32 candidate 损失差应远小于口径误差，实测 {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP 需要 CUDA")
def test_amp_vs_fp32_candidate_diff_quantified():
    """AMP vs FP32 的 candidate 损失差量化（§5.2 表 13）：只量化数量级，不以此
    放宽 FP32 断言。"""
    model = _make_model().eval().cuda()
    x, sc, score = _make_batch()
    x, sc, score = x.cuda(), sc.cuda(), score.cuda()
    with torch.no_grad():
        _, aux_fp32 = model(x, sc, score, pathology_epoch=9,
                            return_expert_pred=True)
        with torch.amp.autocast('cuda'):
            _, aux_amp = model(x, sc, score, pathology_epoch=9,
                               return_expert_pred=True)
    target = torch.randn(3, 8, device='cuda')
    ell32 = compute_expert_competence_losses(aux_fp32['tfm_expert_pred'], target)
    ell_amp = compute_expert_competence_losses(
        aux_amp['tfm_expert_pred'].float(), target)
    diff = float((ell32 - ell_amp).abs().max())
    assert diff < 0.1, f"AMP/FP32 损失差应有限且远小于口径误差，实测 {diff}"


# ─────────────────── 契约元信息 ───────────────────

def test_candidate_schema_metadata_present():
    """aux 必须携带 candidate 口径版本（§6.3：旧权重配新评估不能只标统一 v2）。"""
    model = _make_model().eval()
    x, sc, score = _make_batch()
    with torch.no_grad():
        _, aux = model(x, sc, score, pathology_epoch=9, return_expert_pred=True)
    assert aux['tfm_candidate_schema'] == TFM_CANDIDATE_SCHEMA


def test_one_step_head_returns_residual_contract():
    """OneStepDynamicsHead 输出是 residual（非完整状态）——正式链路
    x_last + delta 的锚定职责在 NeuroTwinTFM.forward，头部不得重复加锚。"""
    head = OneStepDynamicsHead(dim=8, hidden_dim=16, cond_dim=3,
                               pathology_mode='moe')
    z_t = torch.randn(2, 4, 8)
    adj = torch.eye(4).unsqueeze(0).expand(2, 4, 4)     # [B,F,F]
    with torch.no_grad():
        head.transition[-1].weight.zero_()
        head.transition[-1].bias.zero_()      # Δz_shared=0 → z_next=z_t+Δz_path
        head.decode.weight.zero_()
        head.decode.bias.zero_()              # decode=0 → delta 恒 0
    delta, aux = head(z_t, adj, torch.randn(2, 3), pathology_epoch=9)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7), \
        "头部输出必须是 residual：decode 零初始化时 delta 恒 0"
