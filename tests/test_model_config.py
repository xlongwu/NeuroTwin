"""唯一V2配置、实际权重身份及HC→MDD迁移。"""
from types import SimpleNamespace

import pytest
import torch

from model import (MODEL_CONFIG, build_model, build_model_from_args, config_value,
                   contract_fields, load_model_config, verify_checkpoint_contract)
from models.phase_forecast import LocalGlobalPhaseForecast
from train import optim


def test_only_v2_configuration_and_training_recipe_are_active():
    cfg = load_model_config()
    assert [p.name for p in MODEL_CONFIG.parent.glob('*.json')] == ['neurotwin.json']
    assert cfg['architecture']['context_min'] == cfg['architecture']['context_max'] == 64
    assert cfg['architecture']['cpm_horizon'] == 32
    assert not {'multiscale_phase', 'active_dynamic_init', 'local_global_phase', 'future_temporal'} & cfg['architecture'].keys()
    assert cfg['training']['ema_decay'] == .995 and cfg['training']['unfreeze_last_layers'] == 2
    assert cfg['training']['backbone_lr_scale'] == cfg['training']['cpm_lr_scale'] == .1
    assert cfg['weights']['initializer']['path'] == cfg['weights']['pretrain']['path']
    assert cfg['loss'] == dict(lambda_rec=1., lambda_fc=.02, lambda_freq=.02,
                              lambda_delta=.3, lambda_corr=.2, lambda_local_corr=.03,
                              huber_delta=1., horizon_weight_start=1., horizon_weight_end=1.5,
                              local_corr_window=16, local_corr_stride=8)


def test_contract_rejects_loss_and_training_drift():
    cfg = load_model_config()
    args = SimpleNamespace(**{name: config_value(cfg, path) for path, name in contract_fields(cfg)},
                           mode='finetune', condition_group='mdd8', condition_arm='cat',
                           augmentation=cfg['augmentation'])
    verify_checkpoint_contract(args, cfg)
    args.lambda_delta = .25
    with pytest.raises(ValueError, match='lambda_delta'):
        verify_checkpoint_contract(args, cfg)


@pytest.mark.parametrize('role', ['pretrain', 'finetune'])
def test_actual_v2_best_strict_load_and_forward(role):
    torch.set_num_threads(1)
    model, cfg = build_model(mode=role)
    with torch.no_grad():
        prediction, info = model(torch.randn(1, 116, 1, 64), return_tokens=True)
    assert prediction.shape == (1, 116, 32) and torch.isfinite(prediction).all()
    assert info['history_tokens'].shape == (1, 116, 8, 256)
    assert info['phase_features'].shape == (1, 116, 32, 256)
    assert isinstance(model.phase_forecast, LocalGlobalPhaseForecast)
    assert model.cpm_head.future_temporal is not None
    assert cfg['weights'][role]['epoch'] == 149


def test_hc_mdd_transfer_rejects_missing_shared_condition_weight(monkeypatch):
    cfg = load_model_config()
    root = MODEL_CONFIG.parent.parent
    hc, mdd = [root/cfg['weights'][role]['path'] for role in ('pretrain', 'finetune')]
    source, destination = optim.read_checkpoint(hc), optim.read_checkpoint(mdd)
    args = SimpleNamespace(**destination['config']['args'])
    model = build_model_from_args(args, 'cpu', destination['config']['condition'])
    report = optim.load_initial_weights(model, hc, args)
    assert report['new_initialized'] and all(k.startswith('conditioner.') for k in report['new_initialized'])
    corrupted = {**source, 'state_dict': dict(source['state_dict'])}
    shared = next(k for k in corrupted['state_dict'] if k.startswith('conditioner.demo.'))
    del corrupted['state_dict'][shared]
    monkeypatch.setattr(optim, 'read_checkpoint', lambda _: corrupted)
    with pytest.raises(ValueError, match='未登记参数'):
        optim.load_initial_weights(model, hc, args)


def test_incomplete_or_old_structure_checkpoint_is_rejected(tmp_path):
    path = tmp_path/'old.pt'
    torch.save(dict(arch_version=7, state_dict={}, config={}), path)
    with pytest.raises(ValueError, match='完整V2'):
        optim.read_checkpoint(path)
