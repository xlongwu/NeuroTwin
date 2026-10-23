# coding=utf-8
"""正式基线（S2 + E2）契约测试：配置 → 权重 → 模型三者可追溯。

依赖真实权重的用例在 checkpoint 缺失时自动 skip。
"""
import json
from types import SimpleNamespace

import pytest
import torch

from baseline import (BASELINE_CONFIG, CONTRACT_FIELDS, config_value,
                      load_baseline_config, verify_checkpoint_contract)
from utils.condition_schema import DROPOUT_SEED, MODEL_SEED, SAMPLING_SEED

CKPT = BASELINE_CONFIG.parent.parent / 'checkpoints' / 'phase2_S2' \
    / 'neurotwin_tfm_finetune' / 'finetuned_best.pt'

requires_ckpt = pytest.mark.skipif(
    not CKPT.exists(), reason='S2 基线权重不在本机（权重不入版本库）')


def test_baseline_config_declares_contract():
    cfg = load_baseline_config()
    assert cfg['architecture']['dynamics_history_memory'] is True
    assert cfg['architecture']['dynamics_mode'] == 'graph_ode'
    assert cfg['architecture']['s4_variant'] == 'off'
    assert cfg['loss']['lambda_temporal_diff'] == 0.05
    assert cfg['loss']['lambda_cross_head'] == 0.02
    assert cfg['training']['seed'] == MODEL_SEED
    assert cfg['training']['sampling_seed'] == SAMPLING_SEED
    assert cfg['training']['cond_dropout_seed'] == DROPOUT_SEED
    conditioning = cfg['data_and_split']['conditioning']
    assert conditioning['pretrain'] == {
        'condition_arm': 'demo', 'condition_group': 'hc3'}
    assert conditioning['finetune'] == {
        'condition_arm': 'cat', 'condition_group': 'mdd8'}
    main = cfg['weights']['main']
    assert main['path'].startswith('checkpoints/phase2_S2/')
    assert len(main['sha256']) == 64


def test_baseline_split_manifest_matches_config():
    """版本化划分清单是实际 split 的事实来源，必须与基线记录逐项一致。"""
    cfg = load_baseline_config()
    manifests = [BASELINE_CONFIG.parent.parent / 'data'
                 / f'subject_split_{mode}.json'
                 for mode in ('pretrain', 'finetune')]
    manifests = [path for path in manifests if path.exists()]
    if not manifests:
        pytest.skip('版本化划分清单不在本机（data/ 不入版本库）')
    for path in manifests:
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['seed'] == cfg['training']['split_seed'], path
        assert payload['val_ratio'] == cfg['data_and_split']['val_ratio'], path
        assert payload['test_ratio'] == cfg['data_and_split']['test_ratio'], path


def _args_from_config(cfg):
    return SimpleNamespace(**{
        arg_key: config_value(cfg, path) for path, arg_key in CONTRACT_FIELDS})


def test_checkpoint_contract_accepts_matching_snapshot():
    cfg = load_baseline_config()
    verify_checkpoint_contract(_args_from_config(cfg), cfg)


def test_checkpoint_contract_rejects_seed_drift():
    cfg = load_baseline_config()
    args = _args_from_config(cfg)
    args.sampling_seed = 2024
    with pytest.raises(RuntimeError, match='sampling_seed'):
        verify_checkpoint_contract(args, cfg)


def test_checkpoint_contract_requires_all_but_legacy_s4_variant():
    cfg = load_baseline_config()
    args = _args_from_config(cfg)
    del args.s4_variant
    verify_checkpoint_contract(args, cfg)

    del args.seed
    with pytest.raises(RuntimeError, match='seed'):
        verify_checkpoint_contract(args, cfg)


def test_baseline_config_rejects_legacy_final_model():
    """基线入口与历史冻结契约（final_model.json，V4 discrete）必须分离。"""
    legacy = BASELINE_CONFIG.parent / 'final_model.json'
    if not legacy.exists():
        pytest.skip('final_model.json 不存在')
    cfg = load_baseline_config()
    assert cfg['weights']['main']['path'] != 'checkpoints/neurotwin_tfm_finetune/finetuned_best.pt'


@requires_ckpt
def test_baseline_build_and_load():
    from baseline import build_baseline_model
    model, cfg = build_baseline_model('cpu')
    assert model.dynamics_mode == 'graph_ode'
    assert model.dynamics_history_memory is True
    assert model.s4_variant == 'off'
    assert model.one_step_head.memory_film is not None
    # E2 损失权重来自基线配置，与 checkpoint 快照在 build 内已交叉校验
    assert cfg['loss']['lambda_temporal_diff'] == 0.05
    assert cfg['loss']['lambda_cross_head'] == 0.02
    x = torch.randn(2, model.features, 1, 8)
    sc = torch.rand(model.features, model.features)
    profile = torch.randn(2, model.profile_dim)
    with torch.no_grad():
        out, aux = model(x, sc, profile=profile)
    assert out.shape == (2, model.features, 1, 1)
    assert torch.isfinite(out).all()
    assert aux['dynamics_pred'].shape[-1] == model.cpm_horizon
