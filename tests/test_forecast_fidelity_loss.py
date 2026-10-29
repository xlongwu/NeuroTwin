# coding=utf-8
"""H1–8 三项损失（重建 / FC / 频域）与不确定性加权测试。"""
import math

import pytest
import torch

from train.losses import (ForecastFidelityLoss, fc_mse, fc_pattern_pcc,
                          frequency_l1, masked_l1, masked_l1_per_sample)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(3, 8, 8, generator=g)
    pred = torch.randn(3, 8, 8, generator=g)
    mask = torch.ones(3, 8)
    return pred, target, mask


def test_raw_terms_zero_for_perfect_prediction():
    _, target, mask = _batch()
    assert masked_l1(target, target, mask).item() == pytest.approx(0.0, abs=1e-6)
    assert fc_mse(target, target, mask).item() == pytest.approx(0.0, abs=1e-6)
    assert frequency_l1(target, target, mask).item() == pytest.approx(0.0, abs=1e-6)
    assert fc_pattern_pcc(target, target).min().item() == pytest.approx(1.0, abs=1e-5)


def test_reconstruction_mask_ignores_invalid_horizons():
    pred, target, _ = _batch()
    mask = torch.zeros(3, 8)
    mask[:, :2] = 1
    got = masked_l1(pred, target, mask)
    expected = (pred[:, :, :2] - target[:, :, :2]).abs().mean()
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)
    per = masked_l1_per_sample(pred, target, mask)
    assert per.shape == (3,)


def test_fc_terms_require_complete_sequences():
    pred, target, _ = _batch()
    mask = torch.zeros(3, 8)
    mask[0, :] = 1
    mask[1, :4] = 1
    # Only the first sample has a complete H1–8 future.
    assert fc_mse(pred, target, mask).item() == pytest.approx(
        fc_mse(pred[:1], target[:1], None).item(), rel=1e-6)
    assert frequency_l1(pred, target, mask).item() == pytest.approx(
        frequency_l1(pred[:1], target[:1], None).item(), rel=1e-6)


def test_fixed_weighting_matches_configured_sum():
    pred, target, mask = _batch()
    crit = ForecastFidelityLoss(
        lambda_rec=1.0, lambda_fc=10.0, lambda_freq=2.0, weighting='fixed')
    total, stats = crit(pred, target, mask)
    expected = (1.0 * stats['loss_rec'] + 10.0 * stats['loss_fc']
                + 2.0 * stats['loss_freq'])
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)


def test_uncertainty_warm_start_matches_lambdas():
    crit = ForecastFidelityLoss(
        lambda_rec=1.0, lambda_fc=10.0, lambda_freq=2.0,
        weighting='uncertainty')
    weights = crit.get_loss_weights()
    assert weights['mode'] == 'uncertainty'
    assert weights['rec']['weight'] == pytest.approx(1.0, rel=1e-5)
    assert weights['fc']['weight'] == pytest.approx(10.0, rel=1e-5)
    assert weights['freq']['weight'] == pytest.approx(2.0, rel=1e-5)
    for name in ('rec', 'fc', 'freq'):
        assert weights[name]['log_s'] == pytest.approx(
            -math.log({'rec': 1.0, 'fc': 10.0, 'freq': 2.0}[name]), rel=1e-5)


def test_uncertainty_loss_backpropagates_to_log_s():
    pred, target, mask = _batch()
    pred.requires_grad_(True)
    crit = ForecastFidelityLoss(weighting='uncertainty')
    total, _ = crit(pred, target, mask)
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    for name in ('rec', 'fc', 'freq'):
        grad = crit.log_s[name].grad
        assert grad is not None and torch.isfinite(grad).all()


def test_invalid_batch_skips_fc_and_freq_terms():
    pred, target, _ = _batch()
    mask = torch.zeros(3, 8)
    crit = ForecastFidelityLoss(weighting='fixed')
    total, stats = crit(pred, target, mask)
    assert stats['loss_fc'].item() == pytest.approx(0.0)
    assert stats['loss_freq'].item() == pytest.approx(0.0)
    assert total.item() == pytest.approx(stats['loss_rec'].item(), rel=1e-6)
