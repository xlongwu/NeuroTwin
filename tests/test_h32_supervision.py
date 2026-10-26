"""H32 采样、迁移、监督与选优的 CPU 小模型验证，不依赖真实数据。"""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from analysis.next_point_eval import run_inference
from model import CONTRACT_FIELDS, config_value, load_model_config
from experiments.long_horizon.evaluate_supervision import subject_segment_mae, validate_pair
from experiments.long_horizon import evaluate_supervision as paired_eval
from main import evaluate_tfm, split_hash
from models.tfm import NeuroTwinTFM
from train.losses import TFMDualLoss
from train.optim import (check_pretrain_config_compat, migrate_pretrained_checkpoint,
                         save_backbone_weights)
from utils.dataloader import NextPointEvalView, NextPointTrainView


def tiny_model(horizon):
    return NeuroTwinTFM(
        features=4, context_max=8, patch_len=2, dim=8, num_layers=1,
        num_heads=2, ff_ratio=2, dropout=0., one_step_hidden_dim=8,
        cpm_horizon=horizon, cpm_layers=1, dynamics_mode='graph_ode',
        graph_source='history_fc', personalized=True, profile_dim=4,
        dynamics_history_memory=True, ode_steps=1, sc_sinkhorn_iters=2)


def test_horizon_migration_is_explicit_and_preserves_every_other_weight(tmp_path):
    source, target = tiny_model(8), tiny_model(32)
    path = tmp_path / 'source.pt'
    saved_args = dict(task_mode='next_timepoint', context_max=8, tfm_patch_len=2,
                      cpm_horizon=8)
    save_backbone_weights(path, source, {'args': saved_args})
    dims = dict(task_mode='next_timepoint', in_window=1, pred_window=1, seq_len=8,
                patch_len=2, cpm_horizon=32)
    with pytest.raises(RuntimeError, match='任务口径'):
        check_pretrain_config_compat(path, dims)
    check_pretrain_config_compat(path, dims, horizon_policy='repeat_last')
    with pytest.raises(RuntimeError, match='形状冲突'):
        migrate_pretrained_checkpoint(target, path)
    report = migrate_pretrained_checkpoint(target, path, horizon_policy='repeat_last')
    assert not report['skipped'] and not report['missing']
    assert report['horizon_expanded'][0]['target_steps'] == 32
    for key, value in source.state_dict().items():
        if key == 'cpm_head.horizon_embed':
            torch.testing.assert_close(target.state_dict()[key][:8], value, rtol=0, atol=0)
            torch.testing.assert_close(target.state_dict()[key][8:], value[-1:].expand(24, -1))
        else:
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    bad = torch.load(path, weights_only=False)
    bad['state_dict']['patch_embed.roi_embed'] = torch.zeros(3, 8)
    torch.save(bad, path)
    with pytest.raises(RuntimeError, match='形状冲突'):
        migrate_pretrained_checkpoint(target, path, horizon_policy='repeat_last')


class SyntheticSeries:
    subject_ids = ['short', 'long']
    max_offset = 1
    context_max = 16

    def legal_context_lengths(self, sid):
        return list(range(8, 17))

    def legal_anchor_range(self, sid, k, max_offset=None):
        n = 40 if sid == 'short' else 80
        return k-1, n-1-(max_offset or 1)

    def legal_task_count(self, sid):
        return 100

    def make_sample(self, pos, k, t, future_steps=0):
        n = 40 if pos == 0 else 80
        return {'future': torch.zeros(2, future_steps),
                'future_mask': torch.tensor([float(t+h < n)
                                            for h in range(1, future_steps+1)])}


def test_h8_h32_share_full_anchor_grid_and_training_boundary():
    base = SyntheticSeries()
    views = [NextPointEvalView(base, [0, 1], context_length=8,
                              all_future_steps=32, anchor_min_future_steps=32,
                              anchors_per_subject=4) for _ in range(2)]
    assert views[0].tasks == views[1].tasks
    assert all(views[0][i]['future_mask'].all() for i in range(len(views[0])))
    train = NextPointTrainView(base, [0, 1], seed=20261009,
                               anchor_max_offset=32, future_steps=32)
    # 40 TR 的短被试只允许 K=8，不能采样到无完整未来的 K>8。
    assert all(k == 8 for pos, k in train.table if pos == 0)
    assert all(train[i]['future_mask'].all() for i in range(len(train)))
    with pytest.raises(ValueError, match='future'):
        NextPointEvalView(base, [1], 8, all_future_steps=8, anchor_min_future_steps=32)


@pytest.mark.parametrize('horizon', [8, 32])
def test_training_loss_and_validation_selection_have_independent_horizons(horizon):
    torch.manual_seed(2024)
    model = tiny_model(horizon)
    # 读出非零，确保远期监督确实传到 dynamics 参数，而非仅得到有限损失。
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.normal_(std=.1)
    x = torch.randn(2, 4, 1, 8)
    future = torch.randn(2, 4, 32)
    batch = dict(x=x, y=future[:, :, :1], x_last=x[:, :, 0, -1],
                 sc=torch.eye(4).expand(2, -1, -1), calibration_bold=x,
                 future=future, future_mask=torch.ones(2, 32),
                 target_mask=torch.ones(2, 1), subj_id=['a', 'b'],
                 cutoff=torch.tensor([7, 7]), context_len=torch.tensor([8, 8]))
    criterion = TFMDualLoss(lambda_dynamics=.5, lambda_variance=.05,
                            lambda_temporal_diff=.05, lambda_cross_head=.02,
                            cross_head_start_epoch=5)
    out, aux = model(x, batch['sc'], calibration_bold=x)
    assert aux['dynamics_pred'].shape[-1] == horizon
    loss, _ = criterion(out, batch['y'].unsqueeze(-1), aux['cpm_pred'],
                        future[:, :, :horizon], torch.ones(2, horizon),
                        context_last=batch['x_last'], epoch=5,
                        dynamics_pred=aux['dynamics_pred'],
                        dynamics_target=future[:, :, :horizon],
                        dynamics_mask=torch.ones(2, horizon),
                        dynamics_mean_norm=aux['dynamics_mean_norm'],
                        dynamics_logvar_norm=aux['dynamics_logvar_norm'],
                        dynamics_target_norm=future[:, :, :horizon])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.one_step_head.w1.weight.grad.count_nonzero() > 0
    assert model.cpm_head.horizon_embed.grad[-1].count_nonzero() > 0
    args = SimpleNamespace(cpm_horizon=horizon, dynamics_metric_horizon=32,
                            sc_sparsity_weight=0., sc_entropy_weight=0., sc_temporal_weight=0.)
    metrics = evaluate_tfm(model, [batch], criterion, 'cpu', args, loss_epoch=5)
    model.eval()
    with torch.no_grad():
        _, long = model(x, batch['sc'], calibration_bold=x, dynamics_horizon=32)
    expected, _ = subject_segment_mae(long['dynamics_pred'].numpy(), future.numpy(),
                                      np.ones((2, 32)), ['a', 'b'])
    assert metrics['metric_mae_dynamics_h1_32_subject'] == pytest.approx(expected['H1_32'])
    assert metrics['metric_mae_dynamics_h1_32_subject_n_anchors'] == 2
    data = run_inference(model, [batch], 'cpu', [1], 0, eval_rollout=False,
                          collect_cpm=True, dynamics_horizon=32)
    assert data['cpm_pred'].shape[-1] == horizon
    np.testing.assert_allclose(data['ode_pred'], long['dynamics_pred'].numpy())


def test_segments_are_subject_equal_and_reject_partial_future():
    target = np.zeros((3, 2, 32))
    pred = np.ones_like(target)
    pred[2] = 5
    pred[:, :, 8:16] *= 2
    pred[:, :, 16:] *= 3
    summary, rows = subject_segment_mae(pred, target, np.ones((3, 32)), ['a', 'a', 'b'])
    assert summary['H1_8'] == 3
    assert summary['H9_16'] == 6
    assert summary['H17_32'] == 9
    assert summary['H1_32'] == 6.75
    assert rows['H1_8']['a'] == 1
    mask = np.ones((3, 32))
    mask[0, -1] = 0
    with pytest.raises(ValueError, match='不完整'):
        subject_segment_mae(pred, target, mask, ['a', 'a', 'b'])


def test_pair_rejects_frozen_or_differently_seeded_checkpoints():
    shared = dict.fromkeys(key for _, key in CONTRACT_FIELDS)
    shared.update(seed=2024, sampling_seed=20261009, cond_dropout_seed=20261008,
                  pretrained_weight='shared.pt', split_file='split.json',
                  train_anchor_max_offset=32, eval_anchor_min_future_steps=32,
                  eval_context_length=64, eval_anchors_per_subject=16,
                  eval_rollout_tasks_per_subject=0,
                  dynamics_metric_horizon=32, freeze_backbone_epochs=0,
                  dynamics_mode='graph_ode', dynamics_history_memory=True)
    left, right = SimpleNamespace(**shared), SimpleNamespace(**shared)
    left.cpm_horizon, right.cpm_horizon = 8, 32
    meta = dict(backbone_unfrozen=True, ckpt_metric_key='metric_mae_dynamics_h1_32_subject',
                condition={'split_hash': 'same'})
    validate_pair(left, right, meta, meta)
    with pytest.raises(ValueError, match='有效解冻'):
        validate_pair(left, right, meta, {**meta, 'backbone_unfrozen': False})
    right.sampling_seed += 1
    with pytest.raises(ValueError, match='sampling_seed'):
        validate_pair(left, right, meta, meta)


def test_paired_evaluation_writes_report_on_shared_synthetic_batch(tmp_path, monkeypatch):
    baseline = load_model_config()
    common = {key: config_value(baseline, path) for path, key in CONTRACT_FIELDS}
    common.update(mode='finetune', pretrained_weight='same.pt', split_file='same.json',
                  freeze_backbone_epochs=0, dynamics_metric_horizon=32,
                  eval_context_length=64, eval_anchors_per_subject=16,
                  eval_rollout_tasks_per_subject=0,
                  train_anchor_max_offset=32, eval_anchor_min_future_steps=32)
    loader = SimpleNamespace(train_subjects=['train'], val_subjects=['val'],
                             test_subjects=['a', 'b'])
    digest = split_hash(loader, SimpleNamespace(**common))
    x, future = torch.randn(2, 4, 1, 8), torch.randn(2, 4, 32)
    batch = dict(x=x, y=future[:, :, :1], x_last=x[:, :, 0, -1],
                 sc=torch.eye(4).expand(2, -1, -1), calibration_bold=x,
                 future=future, future_mask=torch.ones(2, 32),
                 target_mask=torch.ones(2, 1), subj_id=['a', 'b'],
                 cutoff=torch.tensor([7, 7]), context_len=torch.tensor([8, 8]))
    monkeypatch.setattr(paired_eval, 'build_next_point_loader',
                        lambda *a, **k: ([batch], ['a', 'b'], loader))
    monkeypatch.setattr(paired_eval, 'build_model_from_args',
                        lambda args, *a, **k: tiny_model(args.cpm_horizon))
    for arm, horizon in [('h8', 8), ('h32', 32)]:
        save_backbone_weights(tmp_path / f'{arm}.pt', tiny_model(horizon), {
            'args': {**common, 'cpm_horizon': horizon}, 'backbone_unfrozen': True,
            'ckpt_metric_key': paired_eval.PRIMARY, 'epoch_1based': 1,
            'condition': {'split_hash': digest}})
    out_dir = tmp_path / 'report'
    report = paired_eval.evaluate_pair(tmp_path / 'h8.pt', tmp_path / 'h32.pt', out_dir)
    assert report['n_anchors'] == report['n_subjects'] == 2
    assert set(report['arms']['h8']['ODE_MAE']) == {'H1_32', 'H1_8', 'H9_16', 'H17_32'}
    assert json.loads((out_dir / 'metrics_test.json').read_text()) == report
    assert len((out_dir / 'per_subject_test.csv').read_text().splitlines()) == 9
