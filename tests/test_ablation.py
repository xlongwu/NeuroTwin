# coding=utf-8
"""消融实验代码单元测试（方案 §25.4 条件消融 + G30_TFM 变体注册表）。

覆盖：
- TFM 条件消融开关（--tfm_use_patho_cond False）：评分到达但被忽略、无 FiLM 参数；
- shuffled-HAMD 负对照的评分重映射辅助函数；
- G30_TFM 变体注册表的参数合法性（override 键必须存在于 main.py）。
"""
import numpy as np
import pytest
import torch

from experiments.variants import filter_variants
from models.tfm import NeuroTwinTFM
from utils.common import parse_int_list, resolve_forecast_offsets


def _make_finetune_model(use_cond=True):
    torch.manual_seed(0)
    return NeuroTwinTFM(features=8, context_max=16, patch_len=4, dim=16,
                        num_layers=2, num_heads=4, dropout=0.0, cpm_horizon=4,
                        pretrain_mode=False, pathology_input_dim=1,
                        use_patho_cond=use_cond).eval()


def test_cond_off_ignores_pathology_score():
    """条件关闭：评分到达但输出与 None 逐位一致（评分被忽略而非报错）。"""
    model = _make_finetune_model(use_cond=False)
    torch.manual_seed(1)
    x = torch.randn(2, 8, 1, 16)
    sc = torch.rand(2, 8, 8)
    score = torch.tensor([[10.0], [25.0]])
    assert model.pathology_normalizer is None
    assert model.one_step_head.film is None
    assert model.cpm_head.film is None
    with torch.no_grad():
        pred_with, aux_with = model(x, sc, score)
        pred_without, aux_without = model(x, sc, None)
    assert torch.allclose(pred_with, pred_without)
    assert torch.allclose(aux_with['cpm_pred'], aux_without['cpm_pred'])
    assert 'pathology_cond' not in aux_with


def test_cond_off_no_film_params_and_pretrain_error_kept():
    """条件关闭不产生 FiLM 参数；预训练模式收到评分仍然报错。"""
    model = _make_finetune_model(use_cond=False)
    film_params = [n for n, _ in model.named_parameters() if 'film' in n]
    assert not film_params
    torch.manual_seed(0)
    pretrain = NeuroTwinTFM(features=8, context_max=16, patch_len=4, dim=16,
                            num_layers=2, num_heads=4, cpm_horizon=4,
                            pretrain_mode=True).eval()
    with pytest.raises(ValueError):
        pretrain(torch.randn(2, 8, 1, 16), torch.rand(2, 8, 8), torch.rand(2, 1))


def test_remap_pathology():
    """评分重映射：按 subj_id 替换、缺被试报错、输出形状/ dtype 保持。"""
    from analysis.next_point_eval import remap_pathology
    patho = torch.tensor([[10.0], [20.0], [30.0]])
    subj = ['A', 'B', 'C']
    override = {'A': np.array([99.0], dtype=np.float32),
                'B': np.array([98.0], dtype=np.float32),
                'C': np.array([97.0], dtype=np.float32)}
    out = remap_pathology(patho, subj, override)
    assert out.shape == patho.shape and out.dtype == patho.dtype
    assert torch.allclose(out, torch.tensor([[99.0], [98.0], [97.0]]))
    with pytest.raises(KeyError):
        remap_pathology(patho, subj, {'A': np.array([1.0])})


def test_tfm_checkpoint_rebuild_roundtrip():
    """快照重建 strict 加载往返（条件开/关两态）。

    防止 _KWARG_ALIAS_TFM 缺键导致重建模型与检查点结构不一致
    （strict 加载报 Missing keys，评估入口直接失败）。
    """
    from argparse import Namespace
    from experiments.evaluate_variant import _build_tfm_from_args
    kwargs = dict(
        num_rois=8, context_max=16, tfm_patch_len=4, tfm_dim=16, tfm_layers=2,
        tfm_heads=4, tfm_ff_ratio=4, tfm_one_step_hidden_dim=32,
        tfm_use_patho_cond=True, cpm_horizon=4, cpm_layers=2,
        cpm_intervention=False, dropout=0.0, sc_prior_mode='soft_prior',
        sc_lambda_mode='global', sc_lambda_init=0.7, sc_prior_rank=8,
        sc_delta_a=True, sc_delta_rank=None, sc_sinkhorn_iters=16,
        pathology_input_dim=1, pathology_norm_mode='robust_z',
        pathology_norm_quantiles=64, pathology_norm_rbf_knots=8,
        mode='finetune', norm=True)
    for use_cond in (True, False):
        args = Namespace(**{**kwargs, 'tfm_use_patho_cond': use_cond})
        model = _build_tfm_from_args(args, torch.device('cpu'))
        state = model.state_dict()
        rebuilt = _build_tfm_from_args(args, torch.device('cpu'))
        rebuilt.load_state_dict(state, strict=True)   # 不抛错 = 结构一致
        assert bool(rebuilt.pathology_normalizer is not None) is use_cond


def test_g30_tfm_variants_valid():
    """G30_TFM 全部变体的 override 键都存在于 main.py（防拼写静默失效）。"""
    from experiments.base_config import validate_overrides
    variants = filter_variants(group='G30_TFM')
    assert len(variants) >= 8
    ids = {v['id'] for v in variants}
    assert {'tfm_baseline', 'tfm_patch1', 'tfm_patch8', 'tfm_sc_adaptive',
            'tfm_no_cpm', 'tfm_no_cond'} <= ids
    for v in variants:
        validate_overrides(v['overrides'])          # 非法键会抛 ValueError
        # 条件消融开关键名核对
        if v['id'] == 'tfm_no_cond':
            assert v['overrides']['tfm_use_patho_cond'] is False
        if v['id'] == 'tfm_no_cpm':
            assert v['overrides']['lambda_cpm'] == 0.0
        # 微调类变体必须显式声明预训练来源
        if not v['requires_pretrain']:
            assert v['pretrained_from'] == 'neurotwin_tfm_pretrain'


def test_g30_tfm_offsets_resolve_single():
    """所有 TFM 变体的 forecast_offsets 必须解析为 [1]（TFM one-step 口径约束）。"""
    from types import SimpleNamespace
    for v in filter_variants(group='G30_TFM'):
        args = SimpleNamespace(forecast_offsets=v['overrides']['forecast_offsets'],
                               enable_mtp=False)
        assert resolve_forecast_offsets(args) == [1], v['id']
