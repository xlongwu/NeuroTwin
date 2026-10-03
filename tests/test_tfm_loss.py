# coding=utf-8
"""TFMDualLoss 单元测试（方案 §19 / §20）。

验证：Huber+PCC 的 one-step 项、γ 加权逐 horizon 的 CPM 项、mask 语义、
完美预测零损失、闭式参考值核对与配置校验。
"""
import math

import pytest
import torch

from train.losses import TFMDualLoss


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
