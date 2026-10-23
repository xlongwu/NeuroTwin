"""Focused Temporal PCC checks: mask, degenerate trajectories and E2 compatibility."""
import pytest
import torch
from train.losses import temporal_pcc_loss, TFMDualLoss


def test_temporal_pcc_affine_and_reversed():
    target=torch.tensor([[[1.,2.,4.,7.]]])
    assert temporal_pcc_loss(3*target+8,target).item()==pytest.approx(0,abs=1e-6)
    assert temporal_pcc_loss(-target,target).item()==pytest.approx(2,abs=1e-6)


def test_masked_temporal_pcc_matches_selected_points():
    pred=torch.tensor([[[1.,7.,3.,9.,2.]]],requires_grad=True)
    target=torch.tensor([[[2.,-1.,5.,4.,6.]]])
    mask=torch.tensor([[1.,0.,1.,0.,1.]])
    got=temporal_pcc_loss(pred,target,mask)
    expected=1-torch.corrcoef(torch.stack((pred[0,0,[0,2,4]],target[0,0,[0,2,4]])))[0,1]
    torch.testing.assert_close(got,expected)
    got.backward()
    assert pred.grad[0,0,[1,3]].abs().sum()==0
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


@pytest.mark.parametrize('mask', [torch.zeros(2,8), torch.tensor([[1.,1.,0.,0.,0.,0.,0.,0.]]*2)])
def test_insufficient_targets_zero(mask):
    pred=torch.randn(2,3,8,requires_grad=True)
    loss=temporal_pcc_loss(pred,torch.randn_like(pred),mask)
    assert loss.item()==0
    loss.backward();assert torch.equal(pred.grad,torch.zeros_like(pred))


def test_constant_target_excluded_constant_prediction_supervised():
    target=torch.tensor([[[0.,0.,0.,0.],[0.,1.,2.,3.]]])
    pred=torch.zeros_like(target,requires_grad=True)
    loss=temporal_pcc_loss(pred,target)
    assert loss.item()==pytest.approx(1)
    loss.backward()
    assert pred.grad[0,0].abs().sum()==0
    assert torch.isfinite(pred.grad).all() and pred.grad[0,1].abs().sum()>0


def test_both_branches_and_disabled_e2():
    torch.manual_seed(21)
    one=torch.randn(2,3,1,1)
    target=torch.randn(2,3,8)
    cpm=torch.randn_like(target,requires_grad=True)
    ode=torch.randn_like(target,requires_grad=True)
    opts=dict(lambda_temporal_diff=.05,lambda_cross_head=.02,weighting='uncertainty')
    old=TFMDualLoss(**opts)
    disabled=TFMDualLoss(**opts,lambda_temporal_pcc=0)
    active=TFMDualLoss(**opts,lambda_temporal_pcc=.05)
    kwargs=dict(cpm_mask=torch.ones(2,8),dynamics_pred=ode,
                dynamics_target=target,dynamics_mask=torch.zeros(2,8),context_last=one[...,0,0])
    a,_=old(one,one,cpm,target,**kwargs);b,_=disabled(one,one,cpm,target,**kwargs)
    assert torch.equal(a,b)
    active_loss,stats=active(one,one,cpm,target,**kwargs)
    torch.testing.assert_close(active_loss,b+.05*temporal_pcc_loss(cpm,target))
    assert stats['loss_temporal_pcc_dyn']==0
    kwargs['dynamics_mask']=torch.ones(2,8)
    _,stats=active(one,one,cpm,target,**kwargs)
    tp=temporal_pcc_loss(cpm,target)+temporal_pcc_loss(ode,target)
    torch.testing.assert_close(stats['loss_temporal_pcc'],tp.detach())
    tp.backward()
    for p in (cpm,ode): assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
