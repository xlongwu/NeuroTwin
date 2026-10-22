# coding=utf-8
"""TFMDualLoss 单元测试（方案 §19 / §20）。

验证：Huber+PCC 的 one-step 项、γ 加权逐 horizon 的 CPM 项、mask 语义、
完美预测零损失、闭式参考值核对与配置校验。
"""
import math

import pytest
import torch

from train.losses import (TFMDualLoss, cross_head_consistency,
                          temporal_difference_huber)


def test_format_loss_weight_handles_fixed_terms():
    """固定权重项的 log_s=None 不能按浮点格式打印（曾导致训练首轮崩溃）。"""
    from main import _format_loss_weight
    fixed = _format_loss_weight('temporal_diff', {'weight': 0.05, 'log_s': None})
    assert 'fixed' in fixed and '0.0500' in fixed
    learned = _format_loss_weight('one', {'weight': 1.0, 'log_s': -2.5})
    assert '-2.5000' in learned


def test_perfect_prediction_zero_loss():
    torch.manual_seed(0)
    b, f, h = 3, 5, 4
    y = torch.randn(b, f)
    yt = torch.randn(b, f, h)
    crit = TFMDualLoss()
    total, stats = crit(y.view(b, f, 1, 1), y.view(b, f, 1, 1),
                        yt.clone(), yt.clone())
    # 完美预测：Huber=0，spatial PCC=1 → 总损失 0
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    assert float(stats['pcc_one']) == pytest.approx(1.0, abs=1e-6)


def test_one_step_huber_matches_reference():
    torch.manual_seed(1)
    b, f = 4, 6
    pred = torch.randn(b, f)
    y = torch.randn(b, f)
    crit = TFMDualLoss(lambda_one=1.0, lambda_cpm=0.0, lambda_pcc=0.0,
                       huber_delta=0.5)
    total, stats = crit(pred.view(b, f, 1, 1), y.view(b, f, 1, 1),
                        torch.zeros(b, f, 2), torch.zeros(b, f, 2))
    e = (pred - y).abs()
    ref = torch.where(e <= 0.5, 0.5 * e ** 2, 0.5 * (e - 0.25)).mean()
    assert float(stats['loss_one']) == pytest.approx(float(ref), rel=1e-5)
    assert float(total) == pytest.approx(float(ref), rel=1e-5)


def test_cpm_gamma_weighting():
    """γ<1 时远期 horizon 降权：γ^0 : γ^1 : γ^2 = 1 : 0.5 : 0.25。"""
    torch.manual_seed(2)
    b, f, h = 2, 3, 3
    cpm_pred = torch.zeros(b, f, h)
    # 每个 horizon 的误差设为可分辨的常数
    cpm_target = cpm_pred.clone()
    cpm_target[:, :, 0] = 1.0     # huber(1.0, δ=1) = 0.5
    cpm_target[:, :, 1] = 1.0
    cpm_target[:, :, 2] = 1.0
    # 用不同 horizon 不同误差更直接：err_h = h → huber = 0.5·h²（δ=1 内）
    cpm_target[:, :, 0] = 0.4
    cpm_target[:, :, 1] = 0.8
    cpm_target[:, :, 2] = 1.2     # 超出 δ=1 → 1·(1.2-0.5) = 0.7
    gamma = 0.5
    crit = TFMDualLoss(lambda_one=0.0, lambda_cpm=1.0, lambda_pcc=0.0,
                       cpm_gamma=gamma, huber_delta=1.0)
    total, stats = crit(torch.zeros(b, f, 1, 1), torch.zeros(b, f, 1, 1),
                        cpm_pred, cpm_target)
    hub = [0.5 * 0.4 ** 2, 0.5 * 0.8 ** 2, 1.2 - 0.5]
    ws = [1.0, gamma, gamma ** 2]
    ref = sum(w * e for w, e in zip(ws, hub)) / sum(ws)
    assert float(stats['loss_cpm']) == pytest.approx(ref, rel=1e-5)


def test_cpm_mask_excludes_horizon():
    """mask=0 的 horizon 不参与损失（含 NaN 值也不应污染总损失）。"""
    torch.manual_seed(3)
    b, f, h = 2, 4, 3
    cpm_pred = torch.zeros(b, f, h)
    cpm_target = torch.zeros(b, f, h)
    cpm_target[:, :, -1] = 3.0            # 最后一个 horizon 误差大
    mask = torch.ones(b, h)
    mask[:, -1] = 0.0                     # 屏蔽最后一个 horizon
    crit = TFMDualLoss(lambda_one=0.0, lambda_cpm=1.0, lambda_pcc=0.0)
    total, stats = crit(torch.zeros(b, f, 1, 1), torch.zeros(b, f, 1, 1),
                        cpm_pred, cpm_target, cpm_mask=mask)
    assert math.isfinite(float(total))
    assert f'loss_cpm_h{h}' not in stats  # 被屏蔽的 horizon 不产出明细
    assert float(stats['loss_cpm']) == pytest.approx(0.0, abs=1e-7)


def test_config_validation():
    y = torch.zeros(2, 4)
    yt = torch.zeros(2, 4, 2)
    with pytest.raises(ValueError):
        TFMDualLoss(lambda_one=0.0, lambda_cpm=0.0)
    with pytest.raises(ValueError):
        TFMDualLoss(cpm_gamma=1.5)
    with pytest.raises(ValueError):
        TFMDualLoss(huber_delta=0.0)
    crit = TFMDualLoss()
    with pytest.raises(ValueError):
        crit(y.view(2, 4, 1, 1), torch.zeros(2, 4, 2, 1),
             torch.zeros(2, 4, 2), torch.zeros(2, 4, 2))          # one-step 形状不一致
    with pytest.raises(ValueError):
        crit(torch.zeros(2, 4, 2, 1), torch.zeros(2, 4, 2, 1),
             torch.zeros(2, 4, 2), torch.zeros(2, 4, 2))          # one-step 须为 [B,F,1,1]
    with pytest.raises(ValueError):
        crit(y.view(2, 4, 1, 1), y.view(2, 4, 1, 1),
             torch.zeros(2, 4, 2), torch.zeros(2, 4, 3))          # CPM 形状不一致
    with pytest.raises(ValueError):
        crit(y.view(2, 4, 1, 1), y.view(2, 4, 1, 1),
             torch.zeros(2, 4, 2), torch.zeros(2, 4, 2),
             cpm_mask=torch.ones(2, 3))                            # mask 形状不一致


def test_gradient_flow():
    torch.manual_seed(4)
    b, f, h = 2, 4, 3
    cpm_pred = torch.randn(b, f, h, requires_grad=True)
    one_pred = torch.randn(b, f, 1, 1, requires_grad=True)
    crit = TFMDualLoss()
    total, _ = crit(one_pred, torch.randn(b, f, 1, 1),
                    cpm_pred, torch.randn(b, f, h))
    total.backward()
    assert one_pred.grad is not None and cpm_pred.grad is not None
    assert torch.isfinite(one_pred.grad).all() and torch.isfinite(cpm_pred.grad).all()


# ═══════════════ 同方差不确定性加权（weighting='uncertainty'） ═══════════════

def _dyn_inputs(b=2, f=3, h=4, seed=7):
    """构造 dynamics/variance 项的全部输入（mask 末步无效，覆盖部分有效路径）。"""
    torch.manual_seed(seed)
    dyn_pred = torch.randn(b, f, h)
    dyn_tgt = torch.randn(b, f, h)
    mask = torch.ones(b, h)
    mask[:, -1] = 0.0
    mean_norm = torch.randn(b, f, h)
    logvar_norm = torch.randn(b, f, h) * 0.1
    tgt_norm = torch.randn(b, f, h)
    return dyn_pred, dyn_tgt, mask, mean_norm, logvar_norm, tgt_norm


def test_uncertainty_init_weights_equal_lambdas():
    """warm start：初始有效权重 exp(-log_s) 必须精确等于对应 λ。"""
    crit = TFMDualLoss(lambda_dynamics=0.5, lambda_variance=0.05,
                       weighting='uncertainty')
    lw = crit.get_loss_weights()
    assert lw['mode'] == 'uncertainty'
    assert lw['one']['weight'] == pytest.approx(1.0)
    assert lw['dynamics']['weight'] == pytest.approx(0.5)
    assert lw['dynamics']['log_s'] == pytest.approx(-math.log(0.5))
    assert lw['variance']['log_s'] == pytest.approx(-math.log(0.05))
    # log 方差是可学习参数（进 named_parameters，可被优化器收集）
    assert any('log_s' in n for n in dict(crit.named_parameters()))
    assert all(p.requires_grad for p in crit.parameters())


def test_uncertainty_total_equals_fixed_plus_log_s_constants():
    """总损失 = Σ exp(-log_s_i)·L_i + log_s_i；初始权重=λ → 与 fixed 差=Σ s_i。"""
    torch.manual_seed(8)
    b, f, h = 2, 3, 4
    pred, target = torch.randn(b, f, 1, 1), torch.randn(b, f, 1, 1)
    cpm_pred, cpm_tgt = torch.randn(b, f, h), torch.randn(b, f, h)
    (dyn_pred, dyn_tgt, mask, mean_norm,
     logvar_norm, tgt_norm) = _dyn_inputs(b, f, h)
    kw = dict(lambda_one=1.0, lambda_cpm=1.0, lambda_pcc=0.0,
              lambda_dynamics=0.5, lambda_variance=0.05)
    dyn_kwargs = dict(dynamics_pred=dyn_pred, dynamics_target=dyn_tgt,
                      dynamics_mask=mask, dynamics_mean_norm=mean_norm,
                      dynamics_logvar_norm=logvar_norm,
                      dynamics_target_norm=tgt_norm)
    fixed_total, _ = TFMDualLoss(weighting='fixed', **kw)(
        pred, target, cpm_pred, cpm_tgt, **dyn_kwargs)
    unc_total, _ = TFMDualLoss(weighting='uncertainty', **kw)(
        pred, target, cpm_pred, cpm_tgt, **dyn_kwargs)
    s_dyn, s_var = -math.log(0.5), -math.log(0.05)
    assert float(unc_total) == pytest.approx(
        float(fixed_total) + s_dyn + s_var, rel=1e-5)


def test_uncertainty_log_s_receives_gradient_and_steps():
    """log_s 必须可训练：有有限梯度，且优化器步进后取值变化。"""
    torch.manual_seed(9)
    b, f = 2, 3
    # 只启用 one/cpm 两项，保证 forward 覆盖全部已建参数
    crit = TFMDualLoss(lambda_pcc=0.0, lambda_dynamics=0.0,
                       lambda_variance=0.0, weighting='uncertainty')
    opt = torch.optim.AdamW(crit.parameters(), lr=0.1, weight_decay=0.0)
    s_before = {k: float(v) for k, v in crit.log_s.items()}
    for _ in range(3):
        opt.zero_grad()
        total, _ = crit(torch.randn(b, f, 1, 1), torch.randn(b, f, 1, 1),
                        torch.randn(b, f, 2), torch.randn(b, f, 2))
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in crit.log_s.parameters())
        opt.step()
    assert any(float(crit.log_s[k]) != s_before[k] for k in s_before)


def test_uncertainty_disabled_task_excluded():
    """λ=0 的任务项不建参数、不产生常数项（与 fixed 的禁用语义一致）。"""
    crit = TFMDualLoss(lambda_one=0.0, lambda_cpm=1.0, lambda_pcc=0.0,
                       lambda_dynamics=0.0, lambda_variance=0.0,
                       weighting='uncertainty')
    assert 'one' not in crit.log_s and 'cpm' in crit.log_s
    assert 'dynamics' not in crit.log_s and 'variance' not in crit.log_s
    total, _ = crit(torch.zeros(2, 3, 1, 1), torch.zeros(2, 3, 1, 1),
                    torch.zeros(2, 3, 2), torch.zeros(2, 3, 2))
    # 完美预测且 s_cpm=-log(1)=0 → 总损失为 0（无常数偏移）
    assert float(total) == pytest.approx(0.0, abs=1e-6)


def test_uncertainty_all_masked_cpm_adds_no_constant():
    """CPM 整批被 mask 时整项为零（含 log_s 常数项），与「无常数项」约束一致。"""
    b, f, h = 2, 3, 3
    zero_pred = torch.zeros(b, f, 1, 1)
    y = torch.randn(b, f, 1, 1)
    cpm_pred, cpm_tgt = torch.zeros(b, f, h), torch.randn(b, f, h)
    kw = dict(lambda_one=0.0, lambda_cpm=1.0, lambda_pcc=0.0)
    fixed_total, _ = TFMDualLoss(weighting='fixed', **kw)(
        zero_pred, y, cpm_pred, cpm_tgt, cpm_mask=torch.zeros(b, h))
    unc_total, _ = TFMDualLoss(weighting='uncertainty', **kw)(
        zero_pred, y, cpm_pred, cpm_tgt, cpm_mask=torch.zeros(b, h))
    assert float(fixed_total) == pytest.approx(0.0, abs=1e-7)
    assert float(unc_total) == pytest.approx(float(fixed_total), abs=1e-7)


def test_uncertainty_invalid_weighting_raises():
    with pytest.raises(ValueError):
        TFMDualLoss(weighting='naive')


# ═══════════ P0：一阶时间差分与双头一致性（固定权重项） ═══════════

def test_temporal_diff_first_step_uses_context():
    """h=1 参考真值历史末帧 x_t；无 context_last 时该步被排除。"""
    pred = torch.zeros(1, 1, 3)
    target = torch.full((1, 1, 3), 5.0)
    ctx = torch.zeros(1, 1)
    # 有 context：[Δŷ_1=0, Δy_1=5−0=5] → huber(δ=1)=4.5，其余步差分为 0
    loss_ctx = temporal_difference_huber(pred, target, context_last=ctx,
                                         huber_delta=1.0)
    assert float(loss_ctx) == pytest.approx(4.5 / 3.0, rel=1e-5)
    # 无 context：h=1 不可得，仅 h=2..3（差分恰为 0）→ 0
    loss_noctx = temporal_difference_huber(pred, target, context_last=None)
    assert float(loss_noctx) == pytest.approx(0.0, abs=1e-7)


def test_temporal_diff_requires_prev_valid():
    """horizon h 与 h−1 的真值须同时存在；末步被 mask 时不影响前一步。"""
    pred = torch.zeros(1, 1, 3)
    target = torch.zeros(1, 1, 3)
    target[0, 0, 2] = 5.0                       # 仅末步有差
    mask = torch.ones(1, 3)
    mask[0, 1] = 0.0                            # h=2 无真值 → h=2 与其邻居均不计
    loss = temporal_difference_huber(pred, target, mask=mask,
                                     context_last=torch.zeros(1, 1),
                                     huber_delta=1.0)
    # h=1: Δ=0；h=2: 被 mask 排除；h=3: 需要 h=2 有效 → 排除 → 0
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_temporal_diff_shapes_and_validation():
    with pytest.raises(ValueError):
        temporal_difference_huber(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))
    with pytest.raises(ValueError):
        temporal_difference_huber(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4),
                                  mask=torch.ones(2, 3))
    with pytest.raises(ValueError):
        temporal_difference_huber(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4),
                                  context_last=torch.zeros(2, 2))


def test_cross_head_stop_gradient_only_updates_cpm():
    """ODE 作 Teacher：梯度只回 CPM 头，动力学头接到 stop-gradient（grad=None）。"""
    torch.manual_seed(11)
    cpm = torch.randn(2, 3, 4, requires_grad=True)
    dyn = torch.randn(2, 3, 4, requires_grad=True)
    loss = cross_head_consistency(cpm, dyn)
    loss.backward()
    assert cpm.grad is not None and torch.isfinite(cpm.grad).all()
    assert dyn.grad is None


def test_cross_head_zero_when_identical():
    x = torch.randn(2, 3, 4)
    assert float(cross_head_consistency(x, x.clone())) == pytest.approx(
        0.0, abs=1e-7)


def test_cross_head_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cross_head_consistency(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))


def test_cpm_horizon_weight_increasing():
    """increasing：w_h=1+α(h−1)/(H−1)，与逐 horizon Huber 加权闭式核对。"""
    b, f, h = 2, 3, 3
    cpm_pred = torch.zeros(b, f, h)
    cpm_target = torch.zeros(b, f, h)
    cpm_target[:, :, 0] = 0.2
    cpm_target[:, :, 1] = 0.4
    cpm_target[:, :, 2] = 0.6
    alpha = 0.5
    crit = TFMDualLoss(lambda_one=0.0, lambda_cpm=1.0, lambda_pcc=0.0,
                       cpm_horizon_weight='increasing', cpm_horizon_alpha=alpha)
    _, stats = crit(torch.zeros(b, f, 1, 1), torch.zeros(b, f, 1, 1),
                    cpm_pred, cpm_target)
    hub = [0.5 * 0.2 ** 2, 0.5 * 0.4 ** 2, 0.5 * 0.6 ** 2]
    ramp = [0.0, 0.5, 1.0]
    ws = [1.0 + alpha * r for r in ramp]
    ref = sum(w * e for w, e in zip(ws, hub)) / sum(ws)
    assert float(stats['loss_cpm']) == pytest.approx(ref, rel=1e-5)


def test_p0_terms_default_off_keep_behaviour():
    """默认 λ=0：新项不产生损失、不改变总损失，且不出现在固定权重报告里。"""
    torch.manual_seed(12)
    b, f, h = 2, 3, 4
    crit = TFMDualLoss()
    total, stats = crit(torch.randn(b, f, 1, 1), torch.randn(b, f, 1, 1),
                        torch.randn(b, f, h), torch.randn(b, f, h),
                        context_last=torch.randn(b, f))
    assert float(stats['loss_temporal_diff']) == pytest.approx(0.0, abs=1e-8)
    assert float(stats['loss_cross_head']) == pytest.approx(0.0, abs=1e-8)
    lw = crit.get_loss_weights()
    assert 'temporal_diff' not in lw and 'cross_head' not in lw
    assert math.isfinite(float(total))


def test_p0_terms_excluded_from_uncertainty_log_s():
    """新增项为固定权重：uncertainty 模式下不建 log_s，但以固定 λ 报告。"""
    crit = TFMDualLoss(lambda_temporal_diff=0.05, lambda_cross_head=0.02,
                       weighting='uncertainty')
    assert 'temporal_diff' not in crit.log_s and 'cross_head' not in crit.log_s
    lw = crit.get_loss_weights()
    assert lw['temporal_diff'] == {'weight': 0.05, 'log_s': None}
    assert lw['cross_head'] == {'weight': 0.02, 'log_s': None}


def test_cross_head_warmup_gate():
    """epoch < cross_head_start_epoch 时一致性项被跳过。"""
    torch.manual_seed(13)
    b, f, h = 2, 3, 4
    crit = TFMDualLoss(lambda_one=0.0, lambda_cpm=1.0, lambda_pcc=0.0,
                       lambda_cross_head=0.5, cross_head_start_epoch=5)
    cpm_pred = torch.zeros(b, f, h)
    dyn_pred = torch.full((b, f, h), 2.0)
    kw = dict(dynamics_pred=dyn_pred, dynamics_target=torch.zeros(b, f, h),
              dynamics_mask=torch.ones(b, h))
    z = torch.zeros(b, f, 1, 1)
    _, s_early = crit(z, z, cpm_pred, torch.zeros(b, f, h), epoch=0, **kw)
    assert float(s_early['loss_cross_head']) == pytest.approx(0.0, abs=1e-8)
    _, s_late = crit(z, z, cpm_pred, torch.zeros(b, f, h), epoch=5, **kw)
    assert float(s_late['loss_cross_head']) == pytest.approx(2.0, rel=1e-5)


def test_p0_config_validation():
    with pytest.raises(ValueError):
        TFMDualLoss(lambda_temporal_diff=-1.0)
    with pytest.raises(ValueError):
        TFMDualLoss(lambda_cross_head=-1.0)
    with pytest.raises(ValueError):
        TFMDualLoss(cpm_horizon_weight='ramp')
    with pytest.raises(ValueError):
        TFMDualLoss(cpm_horizon_alpha=-0.1)
    with pytest.raises(ValueError):
        TFMDualLoss(cross_head_start_epoch=-1)


def test_optimizer_collects_log_s_into_loss_weight_group():
    """build_optimizer 必须把 log_s 收进 loss_weight 组（wd=0、独立 lr 比例）。"""
    from types import SimpleNamespace

    from train.optim import build_optimizer
    tiny = torch.nn.Linear(2, 2)
    args = SimpleNamespace(mode='pretrain', lr_peak=1e-4, loss_lr_scale=0.5,
                           weight_decay=0.01)
    crit = TFMDualLoss(weighting='uncertainty')
    opt = build_optimizer(tiny, crit, args)
    groups = [g for g in opt.param_groups if g.get('name') == 'loss_weight']
    assert len(groups) == 1
    g = groups[0]
    assert g['weight_decay'] == 0.0
    assert g['lr'] == pytest.approx(1e-4 * 0.5)
    assert len(g['params']) == len(list(crit.log_s.parameters()))
    # fixed 模式无参数 → 不应产生空的 loss_weight 组
    crit_fixed = TFMDualLoss(weighting='fixed')
    opt_fixed = build_optimizer(tiny, crit_fixed, args)
    assert not [g for g in opt_fixed.param_groups
                if g.get('name') == 'loss_weight']
