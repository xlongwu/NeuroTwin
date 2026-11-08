"""唯一正式配置与权重的严格身份检查。"""
from types import SimpleNamespace
import pytest
import torch
from model import (MODEL_CONFIG,build_model,contract_fields,config_value,
                   load_model_config,verify_checkpoint_contract,build_model_from_args)


def test_formal_config_only_contains_active_b_structure():
    cfg=load_model_config()
    assert cfg['architecture']['cpm_horizon']==32
    assert cfg['architecture']['context_min']==cfg['architecture']['context_max']==64
    assert cfg['training']['batch_size']==64
    assert cfg['loss']==dict(lambda_rec=1.,lambda_fc=.05,lambda_freq=.05,
        lambda_delta=.5,lambda_corr=.1,lambda_local_corr=.05,huber_delta=1.)
    assert not any(key in cfg['architecture'] for key in ('dynamics_mode','personalized','spectral_memory'))
    assert sorted(p.name for p in MODEL_CONFIG.parent.glob('*.json'))==[
        'neurotwin.json', 'neurotwin_multiscale_phase.json']


def test_contract_checks_loss_and_training_drift():
    cfg=load_model_config()
    args=SimpleNamespace(**{name:config_value(cfg,path) for path,name in contract_fields(cfg)},
                         mode='finetune',condition_group='mdd8',condition_arm='cat')
    verify_checkpoint_contract(args,cfg)
    args.lambda_delta=.25
    with pytest.raises(ValueError,match='lambda_delta'):verify_checkpoint_contract(args,cfg)


@pytest.mark.parametrize('role',['pretrain','finetune'])
def test_actual_best_weights_strict_load_with_default_cpm_prediction(role):
    registered=load_model_config(MODEL_CONFIG.parent/'neurotwin_multiscale_phase.json')
    path=MODEL_CONFIG.parent.parent/registered['evaluation_protocol']['baseline_weights'][role]['path']
    if not path.exists():pytest.skip('本机未提供已登记权重')
    torch.set_num_threads(1)
    model,_=build_model(mode=role,checkpoint_path=path)
    with torch.no_grad():
        prediction,info=model(torch.randn(1,116,1,64),return_tokens=True)
    assert prediction.shape==(1,116,32) and torch.isfinite(prediction).all()
    assert info['history_tokens'].shape==(1,116,8,256)
    assert not model.multiscale_phase


def test_hc_mdd_transfer_rejects_missing_shared_condition_weight(monkeypatch):
    from train import optim
    cfg=load_model_config()
    root=MODEL_CONFIG.parent.parent
    baseline=load_model_config(MODEL_CONFIG.parent/'neurotwin_multiscale_phase.json')['evaluation_protocol']['baseline_weights']
    hc=root/baseline['pretrain']['path']
    mdd=root/baseline['finetune']['path']
    if not hc.exists() or not mdd.exists():pytest.skip('本机未提供已登记权重')
    source=optim.read_checkpoint(hc)
    destination=optim.read_checkpoint(mdd)
    args=SimpleNamespace(**destination['config']['args'])
    model=build_model_from_args(args,'cpu',destination['config']['condition'])
    report=optim.load_initial_weights(model,hc,args)
    assert report['new_initialized'] and not any('.demo.' in k for k in report['new_initialized'])
    corrupted={**source,'state_dict':dict(source['state_dict'])}
    shared=next(k for k in corrupted['state_dict'] if k.startswith('conditioner.demo.'))
    del corrupted['state_dict'][shared]
    monkeypatch.setattr(optim,'read_checkpoint',lambda _:corrupted)
    with pytest.raises(ValueError,match='未登记参数'):
        optim.load_initial_weights(model,hc,args)


@pytest.mark.parametrize('role',['pretrain','finetune'])
def test_trained_multiscale_configuration_strict_loads_registered_best(role):
    config=MODEL_CONFIG.parent/'neurotwin_multiscale_phase.json'
    cfg=load_model_config(config)
    path=MODEL_CONFIG.parent.parent/cfg['weights'][role]['path']
    if not path.exists():pytest.skip('本机未提供新实验权重')
    torch.set_num_threads(1)
    model,_=build_model(mode=role,config_path=config)
    with torch.no_grad():
        prediction,info=model(torch.randn(1,116,1,64),return_tokens=True)
    assert model.multiscale_phase and prediction.shape==(1,116,32)
    assert info['phase_features'].shape==(1,116,32,256)
    assert info['multiscale_state'].shape==(1,116,256)
