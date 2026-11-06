"""B 六项固定损失、有效掩码和 H1 差分语义。"""
import pytest
import torch
from model import load_model_config
from train.losses import (ForecastFidelityLoss,masked_l1,masked_l1_per_sample,fc_mse,
                          frequency_l1,temporal_difference_huber,temporal_pcc_loss)


def batch():
    torch.manual_seed(2024)
    return torch.randn(3,4,32,requires_grad=True),torch.randn(3,4,32),torch.ones(3,32),torch.randn(3,4)


def test_six_terms_equal_the_fixed_weighted_sum():
    p,t,m,a=batch();weights=load_model_config()['loss'];criterion=ForecastFidelityLoss(**weights)
    loss,stats=criterion(p,t,m,a)
    expected=sum(weights['lambda_'+k]*stats['loss_'+k] for k in ('rec','fc','freq','delta','corr','local_corr'))
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_masks_preserve_reconstruction_and_exclude_incomplete_fc_fft():
    p,t,m,a=batch();m.zero_();m[0]=1;m[1,:3]=1
    torch.testing.assert_close(fc_mse(p,t,m),fc_mse(p[:1],t[:1]))
    torch.testing.assert_close(frequency_l1(p,t,m),frequency_l1(p[:1],t[:1]))
    expected=((p-t).abs()*m[:,None,:]).sum()/(m.sum()*4)
    torch.testing.assert_close(masked_l1(p,t,m),expected)
    assert masked_l1_per_sample(p,t,m).shape==(3,)


def test_all_horizons_have_equal_reconstruction_gradient():
    p=torch.zeros(2,3,32,requires_grad=True);t=torch.ones_like(p)
    masked_l1(p,t).backward()
    torch.testing.assert_close(p.grad,torch.full_like(p,-1/p.numel()))


def test_delta_h1_starts_at_observed_anchor():
    p=torch.zeros(1,1,32,requires_grad=True);t=torch.zeros_like(p);t[...,0]=1
    loss=temporal_difference_huber(p,t,context_last=torch.zeros(1,1))
    assert loss.item()==pytest.approx(1/32)


def test_empty_mask_and_constant_targets_have_finite_gradients():
    p,t,m,a=batch();m.zero_()
    loss,_=ForecastFidelityLoss(**load_model_config()['loss'])(p,t,m,a)
    assert loss.item()==0
    (loss+temporal_pcc_loss(p,torch.zeros_like(t))).backward()
    assert torch.isfinite(p.grad).all()


def test_perfect_forecast_has_zero_six_term_loss():
    _,t,m,a=batch()
    loss,_=ForecastFidelityLoss(**load_model_config()['loss'])(t,t,m,a)
    assert abs(float(loss))<1e-5
