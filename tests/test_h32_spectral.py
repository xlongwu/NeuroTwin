"""H32 spectral state, training gradients, integration and history-only contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.io import savemat

from analysis.evaluate import build_model_from_args
from models.tfm import NeuroTwinTFM
from models.tfm_backbone import SpectralMemoryEncoder
from models.tfm_dynamics import SpectralOscillator
from train.forecast_metrics import STAGE_LABELS, h32_stage_metrics
from train.losses import ForecastFidelityLoss
from train.optim import get_param_groups, set_finetune_trainability
from utils.dataloader import NeuroTwinNextPointDataset, NextPointTrainView, NextPointEvalView


ROOT = Path(__file__).resolve().parents[1]


def tiny_model(**kwargs):
    torch.manual_seed(2024)
    defaults = dict(features=4, context_max=64, patch_len=8, dim=8,
                    num_layers=1, num_heads=2, one_step_hidden_dim=16,
                    cpm_layers=1, cpm_horizon=32, dropout=0,
                    dynamics_history_memory=True, personalized=True, profile_dim=4)
    defaults.update(kwargs)
    return NeuroTwinTFM(**defaults)


def test_fft_phase_is_aligned_to_anchor_and_rotation_uses_tr_once():
    enc = SpectralMemoryEncoder(64, 8)
    assert enc.bins.tolist() == list(range(2, 13))
    m, phase = 5, 0.37
    omega = torch.tensor(2 * torch.pi * m / 64)
    x = torch.cos(omega * torch.arange(64) + phase).view(1, 1, 64)
    _, u, v = enc(x)
    idx = enc.bins.tolist().index(m)
    expected_phase = omega * 63 + phase
    torch.testing.assert_close(u[0, 0, idx], torch.cos(expected_phase), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(v[0, 0, idx], torch.sin(expected_phase), atol=2e-6, rtol=1e-5)
    osc = SpectralOscillator(8, enc.omega)
    du, dv = osc.rotate(u, v, torch.ones(1))
    torch.testing.assert_close(du[0, 0, idx], torch.cos(omega * 64 + phase) * np.exp(-0.01), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close((du.square() + dv.square()).sqrt(),
                               (u.square() + v.square()).sqrt() * np.exp(-0.01))
    # Opposite phases have equal magnitudes and different complex features.
    a, _, _ = enc(x)
    b, _, _ = enc(-x)
    assert not torch.allclose(a, b)


@pytest.mark.parametrize('memory,oscillator', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('dt_kind', ['default', 'shared', 'batched'])
def test_h32_forward_matches_persistent_state_and_backward(memory, oscillator, dt_kind):
    model = tiny_model(spectral_memory=memory, spectral_oscillator=oscillator,
                       dynamics_dt=0.75, state_feedback=True,
                       spectral_latent_drive=oscillator).eval()
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.normal_(std=0.02)
        if oscillator:
            model.one_step_head.spectral_oscillator.readout.weight.normal_(std=0.02)
            model.one_step_head.spectral_oscillator.latent_proj.weight.normal_(std=0.01)
    x = torch.randn(2, 4, 1, 64, requires_grad=True)
    cal = torch.randn(2, 4, 1, 32, requires_grad=True)
    steps = torch.linspace(0.4, 1.3, 32)
    dt = None if dt_kind == 'default' else steps if dt_kind == 'shared' else steps.expand(2, -1)
    controls = torch.randn(2, 32, 4) * 0.1
    _, aux = model(x, None, calibration_bold=cal, with_aux=False,
                   controls=controls, time_deltas=dt)
    roll = model.rollout_next_states(x[:, :, 0].transpose(1, 2), None,
        calibration_bold=cal, steps=32, controls=controls, time_deltas=dt)
    torch.testing.assert_close(aux['dynamics_pred'], roll.transpose(1, 2), atol=2e-6, rtol=1e-5)
    state = model.initialize_state(x[:, :, 0].transpose(1, 2), calibration_bold=cal)
    next_state = model.advance_state(state)
    for key in ('adj', 'profile', 'cond', 'memory', 'norm_stats'):
        assert next_state[key] is state[key]
    if oscillator:
        assert not torch.equal(next_state['osc_u'], state['osc_u'])
    loss, _ = ForecastFidelityLoss()(aux['dynamics_pred'], torch.randn(2, 4, 32))
    loss.backward()
    for tensor in (x, cal):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if oscillator:
        for name in ('damping_raw', 'drive.weight', 'gate.weight', 'readout.weight'):
            param = dict(model.one_step_head.spectral_oscillator.named_parameters())[name]
            assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_zero_init_opens_all_spectral_gradient_paths_after_optimizer_steps():
    model = tiny_model(spectral_memory=True, spectral_oscillator=True)
    x, cal = torch.randn(2, 4, 1, 64), torch.randn(2, 4, 1, 32)
    target = torch.randn(2, 4, 32)
    criterion = ForecastFidelityLoss(rec_stage_weights=[0.3, 0.3, 0.4])
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)
    with torch.autograd.detect_anomaly():
        for _ in range(3):
            optimizer.zero_grad()
            _, aux = model(x, None, calibration_bold=cal, with_aux=False)
            loss, _ = criterion(aux['dynamics_pred'], target)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
    for name, p in model.named_parameters():
        if name.startswith(('spectral_', 'one_step_head.spectral_')):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize('context_length', [16, 63, 64])
def test_spectral_variable_history_and_cpu_autocast(context_length):
    model = tiny_model(spectral_memory=True, spectral_oscillator=True)
    x = torch.randn(2, 4, 1, context_length)
    cal = torch.randn(2, 4, 1, 32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        _, aux = model(x, None, calibration_bold=cal, with_aux=False)
        loss, _ = ForecastFidelityLoss()(aux['dynamics_pred'], torch.randn(2, 4, 32))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_checkpoint_frequency_units_must_match(tmp_path):
    from train.optim import check_pretrain_config_compat
    from utils.common import resolve_task_dims
    args = SimpleNamespace(task_mode='next_timepoint', model_arch='tfm',
        context_max=64, forecast_offsets='1', tfm_patch_len=8, cpm_horizon=32,
        spectral_memory=True, spectral_oscillator=True, spectral_tr=2.,
        spectral_min_hz=.01, spectral_max_hz=.1, spectral_latent_drive=False)
    checkpoint = tmp_path / 'hc.pt'
    torch.save({'state_dict': {}, 'config': {'args': vars(args)}}, checkpoint)
    dims = resolve_task_dims(args)
    dims.update({k: v for k, v in vars(args).items() if k.startswith('spectral_')})
    check_pretrain_config_compat(checkpoint, dims)
    dims['spectral_tr'] = 2.01
    with pytest.raises(RuntimeError, match='spectral_tr'):
        check_pretrain_config_compat(checkpoint, dims)


def test_staged_loss_has_stage_means_and_masks_and_preserves_fc_fft():
    pred = torch.zeros(2, 3, 32, requires_grad=True)
    target = torch.cat((torch.ones(2, 3, 8), torch.full((2, 3, 8), 2.),
                        torch.full((2, 3, 16), 3.)), -1)
    crit = ForecastFidelityLoss(weighting='fixed', rec_stage_weights=[.3, .3, .4])
    loss, stats = crit(pred, target)
    assert stats['loss_rec'] == pytest.approx(.3 + .6 + 1.2)
    _, base = ForecastFidelityLoss(weighting='fixed')(pred, target)
    torch.testing.assert_close(stats['loss_fc'], base['loss_fc'])
    torch.testing.assert_close(stats['loss_freq'], base['loss_freq'])
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    mask = torch.zeros(2, 32)
    mask[:, :8] = 1
    assert crit(pred, target, mask)[1]['loss_rec'] == pytest.approx(1)
    assert crit(pred, target, mask.zero_())[0] == 0
    with pytest.raises(ValueError, match='H32'):
        crit(pred[..., :8], target[..., :8])
    with pytest.raises(ValueError, match='weights'):
        ForecastFidelityLoss(rec_stage_weights=[.3, .3, .3])


def test_metrics_perfect_scaled_and_incomplete_trajectories():
    torch.manual_seed(2024)
    true, anchor = torch.randn(3, 4, 32), torch.randn(3, 4)
    mask = torch.ones(3, 32)
    mask[2, -1] = 0
    got = h32_stage_metrics(true, true, mask, anchor)
    assert torch.isnan(got['mae'][2]).all()
    torch.testing.assert_close(got['mae'][:2], torch.zeros(2, len(STAGE_LABELS)))
    torch.testing.assert_close(got['temporal_pcc'][:2, 1:], torch.ones(2, 4))
    torch.testing.assert_close(got['std_ratio'][:2, 1:], torch.ones(2, 4))
    torch.testing.assert_close(got['acf1_error'][:2, 1:], torch.zeros(2, 4))
    scaled = h32_stage_metrics(.2 * true, true, mask, anchor)
    torch.testing.assert_close(scaled['std_ratio'][:2, 1:], torch.full((2, 4), .2))
    assert torch.isfinite(h32_stage_metrics(torch.zeros_like(true), true, mask, anchor)['temporal_pcc'][:2, 1:]).all()


def test_data_h32_anchors_and_history_do_not_read_future(tmp_path):
    raw = tmp_path / 'Raw' / 'HC'
    raw.mkdir(parents=True)
    series = np.random.default_rng(2024).normal(size=(112, 4)).astype('float32')
    savemat(raw / 'ROISignals_s01.mat', {'ROISignals': series})
    ds = NeuroTwinNextPointDataset(tmp_path, mode='pretrain', context_min=64,
        context_max=64, bold_source='raw', graph_source='history_fc', num_rois=4,
        calibration_length=32, cache_in_memory=True)
    train = NextPointTrainView(ds, [0], samples_per_subject=2, random_context=False,
        future_steps=32, anchor_max_offset=32)
    val = NextPointEvalView(ds, [0], 64, all_future_steps=32, anchor_min_future_steps=32)
    for view in (train, val):
        for sample in view:
            assert sample['future'].shape == (4, 32)
            assert sample['future_mask'].all()
            assert sample['cutoff'] >= 63 and sample['cutoff'] + 32 < len(series)
    before = ds.make_sample(0, 64, 63, future_steps=32)
    model = tiny_model(spectral_memory=True, spectral_oscillator=True).eval()
    state = model.initialize_state(before['x'][:, 0].T.unsqueeze(0),
        calibration_bold=before['calibration_bold'].unsqueeze(0))
    ds._get_subject('s01')['series'][:, 64:] += 100
    after = ds.make_sample(0, 64, 63, future_steps=32)
    changed = model.initialize_state(after['x'][:, 0].T.unsqueeze(0),
        calibration_bold=after['calibration_bold'].unsqueeze(0))
    for key in ('z', 'q', 'adj', 'profile', 'memory', 'osc_u', 'osc_v'):
        torch.testing.assert_close(state[key], changed[key])
    assert not torch.equal(before['future'], after['future'])


def test_spectral_freeze_unfreeze_and_strict_rebuild():
    model = tiny_model(spectral_memory=True, spectral_oscillator=True)
    set_finetune_trainability(model, False)
    assert not any(p.requires_grad for p in model.spectral_encoder.parameters())
    assert not any(p.requires_grad for p in model.one_step_head.spectral_oscillator.parameters())
    set_finetune_trainability(model, True)
    assert all(p.requires_grad for p in model.spectral_encoder.parameters())
    args = SimpleNamespace(mode='finetune', lr_peak=.001, backbone_lr_scale=.2)
    groups = get_param_groups(model, args)
    trained = {id(p) for g in groups for p in g['params']}
    assert all(id(p) in trained for p in model.spectral_encoder.parameters())
    snapshot = SimpleNamespace(mode='pretrain', num_rois=4, context_max=64,
        tfm_patch_len=8, tfm_dim=8, tfm_layers=1, tfm_heads=2,
        tfm_one_step_hidden_dim=16, cpm_layers=1, cpm_horizon=32, dropout=0,
        personalized=True, profile_dim=4, dynamics_history_memory=True,
        spectral_memory=True, spectral_oscillator=True)
    rebuilt = build_model_from_args(snapshot, 'cpu')
    rebuilt.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize('suffix', ['baseline', 'memory', 'spectral', 'staged'])
def test_h32_config_cli_defaults_and_isolated_checkpoints(monkeypatch, suffix):
    from main import parse_args
    path = ROOT / 'configs' / f'neurotwin_h32_{suffix}.json'
    cfg = json.loads(path.read_text())
    monkeypatch.setattr('sys.argv', ['main.py', '--mode', 'finetune', '--model_config', str(path)])
    args = parse_args()
    assert args.cpm_horizon == 32 and args.context_min == args.context_max == 64
    assert args.pretrained_weight == cfg['weights']['pretrain']['path']
    assert args.spectral_memory == cfg['architecture']['spectral_memory']
    assert args.spectral_oscillator == cfg['architecture']['spectral_oscillator']
    assert args.checkpoint_dir == cfg['training']['checkpoint_dir']
    assert args.eval_rollout and args.eval_fc and args.eval_spectral


@pytest.mark.parametrize('kwargs', [dict(spectral_min_hz=.2, spectral_max_hz=.1),
                                   dict(spectral_tr=0), dict(spectral_min_hz=.3),
                                   dict(spectral_latent_drive=True, spectral_oscillator=False)])
def test_invalid_spectral_settings_fail(kwargs):
    options = dict(spectral_memory=True, spectral_oscillator=True)
    options.update(kwargs)
    with pytest.raises(ValueError):
        tiny_model(**options)
