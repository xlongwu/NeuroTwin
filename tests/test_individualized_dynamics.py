"""个体化方案的采样边界、算子梯度与冻结契约。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.io import savemat

from models.tfm import NeuroTwinTFM
from train.individualized import subject_representation_loss, temporal_difference_l1
from train.optim import (get_param_groups, migrate_pretrained_checkpoint,
                         save_backbone_weights, set_finetune_trainability)
from utils.dataloader import NeuroTwinNextPointDataset


def tiny_model(**kwargs):
    torch.manual_seed(2024)
    return NeuroTwinTFM(features=4, context_max=8, patch_len=2, dim=8,
                        num_layers=1, num_heads=2, cpm_layers=1, dropout=0,
                        personalized=True, profile_dim=4, **kwargs)


def test_zero_init_preserves_baseline_and_learns_new_coefficients():
    base = tiny_model().eval()
    model = tiny_model(personal_dynamics=True, pathology_dynamics=True).eval()
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    # 非零读出模拟已有 checkpoint，排除 persistence 起点对新增分支的屏蔽。
    with torch.no_grad():
        base.one_step_head.decode_rate.weight.fill_(0.1)
        model.one_step_head.decode_rate.weight.copy_(base.one_step_head.decode_rate.weight)
    x = torch.randn(3, 4, 1, 8)
    _, expected = base(x, torch.eye(4), calibration_bold=x, with_aux=False)
    _, aux = model(x, torch.eye(4), calibration_bold=x, with_aux=False)
    torch.testing.assert_close(expected['dynamics_pred'], aux['dynamics_pred'])
    aux['dynamics_pred'].square().mean().backward()
    head = model.one_step_head
    assert head.personal_dynamics.generator[-1].weight.grad.abs().sum() > 0
    assert head.pathology_dynamics.coefficients.weight.grad.abs().sum() > 0


def test_personal_graph_and_state_api_agree():
    model = tiny_model(personal_dynamics=True, pathology_dynamics=True).eval()
    with torch.no_grad():
        model.one_step_head.personal_dynamics.generator[-1].weight.normal_(std=0.01)
        model.one_step_head.pathology_dynamics.coefficients.weight.normal_(std=0.01)
        model.one_step_head.decode_rate.weight.normal_(std=0.1)
    x = torch.randn(2, 4, 1, 8)
    _, aux = model(x, torch.eye(4), calibration_bold=x, with_aux=False)
    graph = aux['sc_prior']['A_eff']
    assert graph.min() >= 0
    torch.testing.assert_close(graph, graph.transpose(-1, -2))
    torch.testing.assert_close(graph.sum(-1), torch.ones(2, 4), atol=1e-5, rtol=1e-5)
    state = model.initialize_state(x[:, :, 0].transpose(1, 2), calibration_bold=x)
    frozen_personal, frozen_path = state['personal_parameters'], state['pathology_parameters']
    for h in range(8):
        state = model.advance_state(state)
        torch.testing.assert_close(model.readout_state(state), aux['dynamics_pred'][:, :, h])
        assert state['personal_parameters'] is frozen_personal
        assert state['pathology_parameters'] is frozen_path
    updated = model.update_state(state, torch.randn(2, 4))
    assert updated['personal_parameters'] is frozen_personal


def test_pdd_only_trains_pathology_even_when_unfreeze_requested():
    model = tiny_model(personal_dynamics=True, pathology_dynamics=True)
    set_finetune_trainability(model, backbone_unfrozen=True)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable and all(name.startswith('one_step_head.pathology_dynamics.') for name in trainable)
    groups = get_param_groups(model, SimpleNamespace(mode='finetune', lr_peak=0.01))
    assert len(groups) == 1 and groups[0]['name'] == 'pathology'
    assert {id(p) for p in groups[0]['params']} == {
        id(p) for p in model.one_step_head.pathology_dynamics.parameters()}


def test_fresh_forecast_reaches_backbone_and_personal_generator():
    model = tiny_model(personal_dynamics=True, dynamics_readout_init_std=0.01)
    x = torch.randn(2, 4, 1, 8)
    _, aux = model(x, torch.eye(4), calibration_bold=x, with_aux=False)
    (aux['dynamics_pred'] - torch.randn(2, 4, 8)).abs().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.patch_embed.parameters())
    assert model.one_step_head.personal_dynamics.generator[-1].weight.grad.abs().sum() > 0


def test_pdd_readout_has_separate_low_learning_rate():
    model = tiny_model(personal_dynamics=True, pathology_dynamics=True)
    set_finetune_trainability(model, pathology_train_readout=True)
    args = SimpleNamespace(mode='finetune', lr_peak=0.01, pathology_readout_lr_scale=0.2)
    groups = get_param_groups(model, args)
    assert [g['name'] for g in groups] == ['pathology', 'pathology_readout']
    assert groups[1]['lr'] == pytest.approx(0.002)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for g in groups for p in g['params']}
    assert not any(p.requires_grad for p in model.one_step_head.personal_dynamics.parameters())


def test_stable_schedule_skips_auxiliary_encoder_during_warmup():
    from train.individualized import auxiliary_dynamics_loss, stable_loss_factor
    assert [stable_loss_factor(e, 5, 10) for e in (0, 4, 5, 9, 14, 20)] == [0, 0, .1, .5, 1, 1]
    model = tiny_model(stable_representation=True)
    args = SimpleNamespace(lambda_delta=0, lambda_personal_reg=0,
        lambda_subject=.005, stable_warmup_epochs=5, stable_ramp_epochs=10)
    # batch 无 SDRL 字段：warmup 必须跳过第二次编码及其数据读取。
    loss, stats = auxiliary_dynamics_loss(model, {'dynamics_pred': torch.zeros(2, 4, 8)},
        {}, torch.zeros(2, 4, 8), torch.ones(2, 8), args, epoch=0)
    assert loss == 0 and stats['stable_loss_factor'] == 0


def test_migration_allows_only_explicit_new_modules(tmp_path):
    source = tiny_model()
    dest = tiny_model(personal_dynamics=True)
    path = tmp_path / 'base.pt'
    save_backbone_weights(path, source)
    with pytest.raises(RuntimeError, match='非条件键'):
        migrate_pretrained_checkpoint(dest, path)
    report = migrate_pretrained_checkpoint(
        dest, path, new_module_prefixes=('one_step_head.personal_dynamics.',))
    assert report['new_initialized'] and not report['missing']
    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor, dest.state_dict()[key])


def test_two_windows_never_use_future(tmp_path):
    raw = tmp_path / 'Raw' / 'HC'
    raw.mkdir(parents=True)
    series = np.random.default_rng(2024).normal(size=(80, 4)).astype(np.float32)
    savemat(raw / 'ROISignals_s01.mat', {'ROISignals': series})
    ds = NeuroTwinNextPointDataset(tmp_path, context_min=8, context_max=8,
        bold_source='raw', graph_source='history_fc', num_rois=4,
        calibration_length=32, stable_window_length=32, cache_in_memory=True)
    early = ds.make_sample(0, 8, 31, future_steps=8)
    assert not early['stable_pair_valid'] and early['stable_bold_b'].count_nonzero() == 0
    late = ds.make_sample(0, 8, 63, future_steps=8)
    assert late['stable_pair_valid']
    saved = late['stable_bold_b'].clone()
    ds._get_subject('s01')['series'][:, 64:] += 100
    same = ds.make_sample(0, 8, 63, future_steps=8)
    assert torch.equal(saved, same['stable_bold_b'])
    assert torch.equal(late['calibration_bold'], same['calibration_bold'])


def test_temporal_difference_and_duplicate_subjects():
    pred = torch.tensor([[[2., 4., 99.]]])
    true = torch.tensor([[[1., 3., -99.]]])
    loss = temporal_difference_l1(pred, true, torch.zeros(1, 1), torch.tensor([[1., 1., 0.]]))
    assert loss == 0.5  # 第一步误差 1，第二步误差 0，缺失尾部不参与。
    p = torch.eye(2)
    cons, contrast = subject_representation_loss(p, p, ['a', 'b'])
    cons2, contrast2 = subject_representation_loss(p[[0, 0, 1]], p[[0, 0, 1]], ['a', 'a', 'b'])
    assert cons == cons2 == 0
    torch.testing.assert_close(contrast, contrast2)


def test_checkpoint_rebuild_with_new_options():
    from analysis.evaluate import build_model_from_args
    model = tiny_model(personal_dynamics=True, pathology_dynamics=True)
    args = SimpleNamespace(mode='finetune', num_rois=4, context_max=8,
        tfm_patch_len=2, tfm_dim=8, tfm_layers=1, tfm_heads=2, cpm_layers=1,
        dropout=0, personalized=True, profile_dim=4, personal_dynamics=True,
        pathology_dynamics=True)
    rebuilt = build_model_from_args(args, 'cpu')
    rebuilt.load_state_dict(model.state_dict(), strict=True)


def test_auxiliary_losses_and_evaluation_export(tmp_path):
    from torch.utils.data import DataLoader
    from analysis.individualized_dynamics import evaluate_profiles
    from train.individualized import auxiliary_dynamics_loss
    from utils.dataloader import NextPointEvalView

    raw = tmp_path / 'Raw' / 'HC'
    raw.mkdir(parents=True)
    for i in range(2):
        savemat(raw / f'ROISignals_s{i}.mat', {'ROISignals':
            np.random.default_rng(i).normal(size=(30, 4)).astype(np.float32)})
    ds = NeuroTwinNextPointDataset(tmp_path, context_min=8, context_max=8,
        bold_source='raw', graph_source='history_fc', num_rois=4,
        calibration_length=4, stable_window_length=4, cache_in_memory=True)
    view = NextPointEvalView(ds, [0, 1], 8, anchors_per_subject=1,
                            all_future_steps=8, anchor_min_future_steps=8)
    loader = DataLoader(view, batch_size=2)
    model = tiny_model(personal_dynamics=True, pathology_dynamics=True,
                       stable_representation=True).eval()
    batch = next(iter(loader))
    _, aux = model(batch['x'], batch['sc'], calibration_bold=batch['calibration_bold'], with_aux=False)
    args = SimpleNamespace(lambda_delta=0.1, lambda_personal_reg=0.001,
        lambda_subject=0.05, contrastive_temperature=0.1, lambda_disease=0.1,
        lambda_path_reg=0.001, lambda_hc_reference=1.0, lambda_clinical=0.0)
    loss, stats = auxiliary_dynamics_loss(model, aux, batch, batch['future'],
        batch['future_mask'], args, disease_label=0.0)
    assert torch.isfinite(loss)
    assert all(value.is_floating_point() for value in stats.values())
    loss.backward()
    predictions, stability, parameters = evaluate_profiles(model, loader, ds, ['s0', 's1'], 'cpu')
    assert len(predictions) == len(stability) == len(parameters) == 2
    assert '交换MAE增量' in predictions and '病理增量MAE改善' in predictions
    assert set(predictions['Profile供体']) == {'s0', 's1'}
