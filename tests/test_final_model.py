# coding=utf-8
"""最终定型主模型（configs/final_model.json + inference.py）契约测试。

覆盖（定型验收 §四）：
- 配置工件：最终值固化、无机器专属路径、权重哈希与文件一致；
- 加载契约：strict 加载无 missing/unexpected、patch=4 进入模型构造（不受
  CLI 默认 patch=8 影响）、参数量 6,029,728 / FiLM 2,048、冻结期可训练
  参数恰为 FiLM 的 2,048、HAMD normalizer buffers 随权重加载且已拟合；
- 推理契约：最小前向形状/有限性、入口与直接调用原模型输出一致、
  steps>8 明确报错（不静默切换 B 权重/rollout 协议）；
- 错误路径：MoE/HC 权重被 FiLM 闸门拒绝、哈希不一致拒绝、
  arch_version 不一致拒绝、缺失 pathology_score / 非法形状清晰报错。

依赖真实权重的用例在 checkpoints/ 缺失时自动 skip；其余用例确定性可重复。
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import inference
from inference import (DEFAULT_CONFIG_PATH, ROOT, forecast_horizon,
                       load_final_config, load_final_model,
                       normalize_raw_sc, predict_next_state)
from train.optim import count_trainable_params, set_finetune_stage

EXPECTED_MODEL = {
    'task_mode': 'next_timepoint',
    'model_arch': 'tfm',
    'num_rois': 116,
    'context_min': 16,
    'context_max': 64,
    'eval_context_length': 64,
    'tfm_patch_len': 4,
    'tfm_dim': 256,
    'tfm_layers': 4,
    'tfm_heads': 4,
    'tfm_ff_ratio': 4,
    'tfm_one_step_hidden_dim': 256,
    'dropout': 0.1,
    'norm': True,
    'sc_prior_mode': 'soft_prior',
    'sc_lambda_mode': 'global',
    'sc_lambda_init': 0.7,
    'sc_prior_rank': 12,
    'sc_delta_a': True,
    'sc_delta_rank': None,
    'sc_delta_rank_effective': 6,
    'sc_sinkhorn_iters': 64,
    'cpm_horizon': 8,
    'cpm_layers': 2,
    'cpm_intervention': False,
    'tfm_patho_mode': 'film',
    'tfm_num_experts': None,
    'tfm_top_k': None,
    'pathology_input_dim': 1,
    'pathology_norm_mode': 'robust_z',
    'pathology_fields': None,
    'pathology_fields_effective': ['HAMD'],
    'pathology_missing': 'drop',
    'forecast_offsets': [1],
    'moe_router_recc': 'not_used',
}
EXPECTED_LOSS = {
    'lambda_one': 1.0, 'lambda_cpm': 1.0, 'lambda_pcc': 0.1,
    'cpm_gamma': 1.0, 'cpm_huber_delta': 1.0,
    'sc_sparsity_weight': 0.001, 'sc_entropy_weight': 0.001,
    'sc_temporal_weight': 0.0,
}
EXPECTED_RECIPE = {
    'seed': 2024, 'batch_size': 32, 'train_epochs': 150, 'patience': 20,
    'adam_betas': [0.9, 0.95], 'weight_decay': 0.01, 'grad_clip': 1.0,
    'warmup_epochs': 15, 'freeze_backbone_epochs': 10,
    'backbone_lr_scale': 0.2, 'use_ema': True, 'ema_decay': 0.999,
}
EXPECTED_PARAM_COUNTS = {
    'total': 6029728, 'film_total': 2048, 'frozen_stage_trainable': 2048,
    'unfrozen_trainable': 6029728,
}


def _ckpt_path() -> Path:
    cfg = load_final_config()
    return ROOT / cfg['weights']['main']['path']


MAIN_CKPT_EXISTS = _ckpt_path().exists()

pytestmark_ckpt = pytest.mark.skipif(
    not MAIN_CKPT_EXISTS, reason='checkpoints/ 主权重不在本机（权重不入版本库）')


# ---------------------------------------------------------------------------
# 配置工件（不依赖权重）
# ---------------------------------------------------------------------------

def test_final_config_values_frozen():
    cfg = load_final_config()
    assert cfg['model'] == EXPECTED_MODEL
    loss = dict(cfg['loss'])
    pcc_terms = loss.pop('pcc_terms')
    assert loss == EXPECTED_LOSS
    # 任务3 口径：One-Step 与 CPM 均含 PCC 项；future mask 排除不可用目标
    assert '存在' in pcc_terms['one_step'] and '存在' in pcc_terms['cpm']
    assert 'target_mask' in pcc_terms['future_mask']
    for k, v in EXPECTED_RECIPE.items():
        assert cfg['training_recipe'][k] == v, f'training_recipe.{k}'
    assert 'known_issue_lr_schedule' in cfg['training_recipe']
    for k, v in EXPECTED_PARAM_COUNTS.items():
        assert cfg['param_counts'][k] == v, f'param_counts.{k}'
    sig = cfg['structure_signature']
    assert sig == {'task_mode': 'next_timepoint', 'model_arch': 'tfm',
                   'in_window': 1, 'pred_window': 1, 'seq_len': 64,
                   'patch_len': 4, 'cpm_horizon': 8, 'arch_version': 4}
    # 结构签名的 patch_len 必须与 model 段一致（防 CLI 默认 8 回渗）
    assert sig['patch_len'] == cfg['model']['tfm_patch_len'] == 4


def _iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def test_final_config_has_no_machine_paths():
    cfg = load_final_config()
    for s in _iter_strings(cfg):
        assert not s.startswith('/') and not s.startswith('~'), \
            f'配置疑似机器专属路径: {s}'


def test_final_config_weight_hash_matches_file():
    cfg = load_final_config()
    ckpt = ROOT / cfg['weights']['main']['path']
    if not ckpt.exists():
        pytest.skip('checkpoints/ 主权重不在本机（权重不入版本库）')
    import hashlib
    h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert h == cfg['weights']['main']['sha256']


def test_final_config_matches_checkpoint_snapshot():
    """配置 model 段与权重 config.args 快照一致（防手改配置漂移）。"""
    if not MAIN_CKPT_EXISTS:
        pytest.skip('checkpoints/ 主权重不在本机')
    from experiments.evaluate_variant import load_train_args
    args = load_train_args(str(_ckpt_path()))
    snapshot_map = {
        'num_rois': 'num_rois', 'context_min': 'context_min',
        'context_max': 'context_max', 'eval_context_length': 'eval_context_length',
        'tfm_patch_len': 'tfm_patch_len', 'tfm_dim': 'tfm_dim',
        'tfm_layers': 'tfm_layers', 'tfm_heads': 'tfm_heads',
        'tfm_ff_ratio': 'tfm_ff_ratio',
        'tfm_one_step_hidden_dim': 'tfm_one_step_hidden_dim',
        'dropout': 'dropout', 'norm': 'norm', 'sc_prior_mode': 'sc_prior_mode',
        'sc_lambda_mode': 'sc_lambda_mode', 'sc_lambda_init': 'sc_lambda_init',
        'sc_prior_rank': 'sc_prior_rank', 'sc_delta_a': 'sc_delta_a',
        'sc_delta_rank': 'sc_delta_rank',
        'sc_sinkhorn_iters': 'sc_sinkhorn_iters', 'cpm_horizon': 'cpm_horizon',
        'cpm_layers': 'cpm_layers', 'cpm_intervention': 'cpm_intervention',
        'pathology_input_dim': 'pathology_input_dim',
        'pathology_norm_mode': 'pathology_norm_mode',
        'pathology_fields': 'pathology_fields',
        'pathology_missing': 'pathology_missing',
    }
    cfg = load_final_config()['model']
    for key, attr in snapshot_map.items():
        assert cfg[key] == getattr(args, attr), f'config.{key} 与快照不一致'
    # 快照尚未引入 tfm_patho_mode 键（FiLM baseline 训练时无该开关），
    # 重建时走构造默认 'film'；MoE 权重的快照显式为 'moe'，由 FiLM 闸门拒绝
    assert getattr(args, 'tfm_patho_mode', 'film') == 'film'


# ---------------------------------------------------------------------------
# 加载与参数量契约
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def final_model():
    if not MAIN_CKPT_EXISTS:
        pytest.skip('checkpoints/ 主权重不在本机（权重不入版本库）')
    np.random.seed(0)
    return load_final_model(device='cpu')


@pytestmark_ckpt
def test_strict_load_param_counts_and_freeze_stage(final_model):
    assert sum(p.numel() for p in final_model.parameters()) == 6029728
    n_film = sum(p.numel() for m in final_model.modules()
                 if getattr(m, 'is_conditioning_adapter', False)
                 for p in m.parameters())
    assert n_film == 2048
    set_finetune_stage(final_model, backbone_unfrozen=False)
    assert count_trainable_params(final_model) == 2048
    set_finetune_stage(final_model, backbone_unfrozen=True)
    assert count_trainable_params(final_model) == 6029728
    # 恢复 eval（load_final_model 的交付状态）
    final_model.eval()


@pytestmark_ckpt
def test_patch4_from_snapshot_not_cli_default(final_model):
    # CLI --tfm_patch_len 默认已改为 8；最终入口必须按 checkpoint 快照的 4 重建
    assert final_model.patch_len == 4
    assert final_model.cpm_horizon == 8
    assert final_model.features == 116
    assert final_model.context_max == 64
    assert final_model.tfm_patho_mode == 'film'
    assert final_model.one_step_head.pathology_mode == 'film'
    assert final_model.one_step_head.pathology_moe is None


@pytestmark_ckpt
def test_normalizer_buffers_loaded_not_refit(final_model):
    norm = final_model.pathology_normalizer
    assert norm.is_fitted()
    # 定型说明/检查点：robust_z 统计量来自 MDD train 被试 HAMD 拟合并随权重保存
    assert float(norm.center[0]) == pytest.approx(22.0)
    assert float(norm.scale[0]) == pytest.approx(5.930318832397461, rel=1e-6)


@pytestmark_ckpt
def test_min_forward_shapes_and_finite(final_model):
    torch.manual_seed(0)
    b, k, f = 1, 64, 116
    hist = torch.randn(b, k, f)
    sc = normalize_raw_sc(np.random.rand(f, f))
    sc = torch.from_numpy(sc)
    hamd = torch.tensor([[18.0]])
    with torch.no_grad():
        out = predict_next_state(final_model, hist, sc, hamd)
        traj = forecast_horizon(final_model, hist, sc, hamd, steps=8)
    assert tuple(out['pred'].shape) == (b, 116, 1, 1)
    assert tuple(out['pred_next'].shape) == (b, 116, 1)
    assert tuple(traj.shape) == (b, 8, 116)
    assert torch.isfinite(out['pred_next']).all()
    assert torch.isfinite(traj).all()


@pytestmark_ckpt
def test_entry_equals_direct_model_calls(final_model):
    torch.manual_seed(1)
    b, k, f = 2, 64, 116
    hist = torch.randn(b, k, f)
    sc = torch.from_numpy(normalize_raw_sc(np.random.rand(f, f)))
    hamd = torch.tensor([[18.0], [24.0]])
    with torch.no_grad():
        via_entry = predict_next_state(final_model, hist, sc, hamd)
        direct = final_model.predict_next_state(hist, sc, hamd)
        traj_entry = forecast_horizon(final_model, hist, sc, hamd, steps=8)
        traj_direct = final_model.forecast_horizon(hist, sc, hamd)
    assert torch.allclose(via_entry['pred_next'], direct['pred_next'])
    assert torch.allclose(traj_entry, traj_direct)   # [B,8,F] 与完整 CPM 一致
    # steps < 8 是完整 CPM 轨迹的前缀（非自回归，非重算）
    with torch.no_grad():
        traj2 = forecast_horizon(final_model, hist, sc, hamd, steps=2)
    assert torch.allclose(traj2, traj_direct[:, :2, :])


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------

@pytestmark_ckpt
def test_horizon_above_limit_rejected(final_model):
    torch.manual_seed(2)
    hist = torch.randn(1, 64, 116)
    sc = torch.from_numpy(normalize_raw_sc(np.random.rand(116, 116)))
    hamd = torch.tensor([[18.0]])
    with pytest.raises(RuntimeError, match=r'H≤8'):
        forecast_horizon(final_model, hist, sc, hamd, steps=9)
    with pytest.raises(RuntimeError, match=r'H≤8'):
        forecast_horizon(final_model, hist, sc, hamd, steps=16)
    with pytest.raises(ValueError, match='steps 必须 >= 1'):
        forecast_horizon(final_model, hist, sc, hamd, steps=0)


@pytestmark_ckpt
def test_missing_pathology_score_rejected(final_model):
    hist = torch.randn(1, 32, 116)
    sc = torch.from_numpy(normalize_raw_sc(np.random.rand(116, 116)))
    with pytest.raises(ValueError, match='pathology_score'):
        predict_next_state(final_model, hist, sc, None)
    with pytest.raises(ValueError, match='pathology_score'):
        forecast_horizon(final_model, hist, sc, None)


@pytestmark_ckpt
def test_invalid_shapes_rejected(final_model):
    hist_bad_k = torch.randn(1, 65, 116)
    hist_bad_f = torch.randn(1, 64, 100)
    sc = torch.from_numpy(normalize_raw_sc(np.random.rand(116, 116)))
    hamd = torch.tensor([[18.0]])
    with pytest.raises(ValueError, match='context_max'):
        predict_next_state(final_model, hist_bad_k, sc, hamd)
    with pytest.raises(ValueError, match='ROI'):
        predict_next_state(final_model, hist_bad_f, sc, hamd)
    with pytest.raises(ValueError, match='116'):
        predict_next_state(final_model, torch.randn(1, 64, 116),
                           torch.rand(100, 100), hamd)
    with pytest.raises(ValueError, match='pathology_score'):
        predict_next_state(final_model, torch.randn(2, 64, 116), sc, hamd)


def _write_fake_ckpt(path: Path, arch_version=4, mode='finetune',
                     extra_args=None, state_dict=None):
    """构造最小 checkpoint（真实参数快照 + 指定 arch_version/state_dict）。"""
    args = {'mode': mode, 'task_mode': 'next_timepoint', 'num_rois': 116,
            'context_max': 64, 'context_min': 16, 'eval_context_length': 64,
            'tfm_patch_len': 4, 'tfm_dim': 256, 'tfm_layers': 4,
            'tfm_heads': 4, 'tfm_ff_ratio': 4, 'dropout': 0.1, 'norm': True,
            'tfm_one_step_hidden_dim': 256, 'cpm_horizon': 8, 'cpm_layers': 2,
            'cpm_intervention': False, 'pathology_input_dim': 1,
            'pathology_norm_mode': 'robust_z', 'pathology_missing': 'drop'}
    if extra_args:
        args.update(extra_args)
    torch.save({'state_dict': state_dict or {}, 'arch_version': arch_version,
                'config': {'args': args}}, path)
    return path


def test_film_gate_rejects_moe_checkpoint(tmp_path):
    """MoE checkpoint（快照 tfm_patho_mode='moe'）被最终入口明确拒绝。"""
    ckpt = _write_fake_ckpt(tmp_path / 'moe.pt',
                            extra_args={'tfm_patho_mode': 'moe',
                                        'tfm_num_experts': 4, 'tfm_top_k': 2})
    with pytest.raises(RuntimeError, match='tfm_patho_mode'):
        load_final_model(ckpt_path=ckpt, device='cpu', verify_hash=False)


def test_film_gate_rejects_hc_pretrain_checkpoint(tmp_path):
    """HC 预训练权重（pretrain_mode=True → 无条件模块）同样被拒绝。"""
    ckpt = _write_fake_ckpt(tmp_path / 'hc.pt', mode='pretrain')
    with pytest.raises(RuntimeError, match='tfm_patho_mode'):
        load_final_model(ckpt_path=ckpt, device='cpu', verify_hash=False)


def test_arch_version_mismatch_rejected(tmp_path):
    ckpt = _write_fake_ckpt(tmp_path / 'old.pt', arch_version=999)
    with pytest.raises(RuntimeError, match='arch_version'):
        load_final_model(ckpt_path=ckpt, device='cpu', verify_hash=False)


def test_hash_mismatch_rejected(tmp_path):
    """默认主权重路径指向校验不符的文件时报错（sha256 闸门）。"""
    fake = _write_fake_ckpt(tmp_path / 'main.pt')
    cfg = load_final_config()
    cfg['weights']['main']['path'] = str(fake)
    cfg['weights']['main']['sha256'] = '0' * 64
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(RuntimeError, match='sha256'):
        load_final_model(config_path=cfg_path, device='cpu')


@pytestmark_ckpt
def test_strict_load_rejects_wrong_structure(tmp_path):
    """结构不匹配（state_dict 与快照重建的模型不一致）明确报错。"""
    ckpt = _write_fake_ckpt(tmp_path / 'broken.pt',
                            state_dict={'blocks.0.attn.wrong': torch.zeros(1)})
    with pytest.raises(RuntimeError, match='strict'):
        load_final_model(ckpt_path=ckpt, device='cpu', verify_hash=False)


# ---------------------------------------------------------------------------
# SC 预处理口径（沿用数据侧实现）
# ---------------------------------------------------------------------------

def test_normalize_raw_sc_pipeline():
    np.random.seed(0)
    sc = np.full((4, 4), np.nan)
    sc[0, 1] = 3.0
    sc[2, 3] = 30.0
    out = normalize_raw_sc(sc)
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.allclose(out, out.T)                    # 对称化
    assert out[2, 3] > out[0, 1]                      # 单调保持
    assert out[2, 3] / out[0, 1] < 30.0 / 3.0         # log1p 压缩动态范围
