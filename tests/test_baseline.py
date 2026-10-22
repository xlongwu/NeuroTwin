# coding=utf-8
"""正式基线（S2 + E2）契约测试：配置 → 权重 → 模型三者可追溯。

依赖真实权重的用例在 checkpoint 缺失时自动 skip。
"""
import pytest
import torch

from baseline import BASELINE_CONFIG, load_baseline_config

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
    assert cfg['training']['seed'] == 2024
    main = cfg['weights']['main']
    assert main['path'].startswith('checkpoints/phase2_S2/')
    assert len(main['sha256']) == 64


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
