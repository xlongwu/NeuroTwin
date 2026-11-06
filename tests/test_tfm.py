"""正式 CPM 前向、图、条件和公开预测接口。"""
import numpy as np
import pytest
import torch
from models.tfm import NeuroTwinTFM
from models.conditioning import ConditioningConfig
from utils.condition_schema import HC_FIELDS,fit_condition_stats


def tiny_model(conditioned=False,dropout=0.):
    kwargs={}
    if conditioned:
        values=np.array([[1,20,12],[2,30,16],[1,40,14]],dtype=float)
        stats=fit_condition_stats(HC_FIELDS,values,np.ones_like(values),np.arange(3))
        kwargs=dict(condition_arm='demo',condition_config=ConditioningConfig(group='hc3',arm='demo').to_dict(),condition_stats_payload=stats.to_dict())
    return NeuroTwinTFM(features=4,context_max=64,patch_len=8,dim=8,num_layers=1,
                        num_heads=2,cpm_horizon=32,cpm_layers=1,dropout=dropout,**kwargs)


def test_single_model_interfaces_and_actual_analysis_outputs():
    torch.manual_seed(2024);model=tiny_model().eval()
    x=torch.randn(2,4,1,64)
    pred,aux=model(x,return_tokens=True)
    assert pred.shape==(2,4,32)
    assert aux['history_tokens'].shape==(2,4,8,8)
    assert aux['sc_prior']['A_eff'].shape==(2,4,4)
    history=x[:,:,0].transpose(1,2)
    torch.testing.assert_close(model.forecast_horizon(history),pred.transpose(1,2))
    single=model.predict_next_state(history)
    torch.testing.assert_close(single['pred_next'],pred[:,:,:1])
    torch.testing.assert_close(single['pred_delta'],pred[:,:,:1]-history[:,-1,:,None])
    assert not hasattr(model,'one_step_head') and not hasattr(model,'profile_encoder')


def test_prediction_uses_only_history_and_condition_gradients_are_finite():
    model=tiny_model(conditioned=True)
    x=torch.randn(2,4,1,64,requires_grad=True)
    cond=dict(cond_z=torch.randn(2,3),cond_code=torch.ones(2,3,dtype=torch.long),cond_mask=torch.ones(2,3))
    pred,aux=model(x,condition=cond)
    pred.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert aux['condition_embedding'].shape==(2,32)
    assert model.cpm_head.film.proj.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_illegal_history_and_condition_are_rejected():
    model=tiny_model()
    with pytest.raises(ValueError):model(torch.zeros(2,4,2,64))
    with pytest.raises(ValueError):model.forecast_horizon(torch.zeros(4,64))
    with pytest.raises(ValueError):model(torch.zeros(2,4,1,64),condition={})
