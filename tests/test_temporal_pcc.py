"""Focused Temporal PCC checks: mask, degenerate trajectories and E2 compatibility."""
import pytest
import torch
from train.losses import temporal_pcc_loss


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
