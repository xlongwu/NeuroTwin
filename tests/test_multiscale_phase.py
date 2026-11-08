"""多尺度相位预测的因果性、相位原点、梯度、归一化和训练分组。"""
import math
from types import SimpleNamespace

import pytest
import torch

from main import parse_args
from model import load_model_config
from models.phase_forecast import PhaseAwareForecast
from models.temporal_multiscale import MultiScaleTemporalEncoder, CausalTemporalBlock
from models.tfm import NeuroTwinTFM
from train.losses import ForecastFidelityLoss, temporal_pcc_loss
from train.optim import (build_optimizer, build_scheduler, prediction_adapter_param_names,
                         set_trainability, set_training_mode, unfreeze_while_preserving_state)


def model():
    return NeuroTwinTFM(features=4, dim=16, num_layers=1, num_heads=2,
                        cpm_layers=1, dropout=0., multiscale_phase=True, multiscale_channels=8)


def test_multiscale_temporal_blocks_are_causal_and_latest_state_is_used():
    torch.manual_seed(2024)
    encoder = MultiScaleTemporalEncoder(8, 16)
    x = torch.randn(2, 4, 64)
    altered = x.clone(); altered[..., 41:] += 20
    for before, after in zip(encoder.temporal_features(x), encoder.temporal_features(altered)):
        torch.testing.assert_close(before[..., :41], after[..., :41], rtol=0, atol=0)
    assert not torch.allclose(encoder(x), encoder(altered))


@pytest.mark.parametrize('dilation', [1, 2, 4, 8])
def test_causal_window_linear_matches_conv_values_and_gradients(dilation):
    from torch.nn import functional as F
    torch.manual_seed(2024)
    block = CausalTemporalBlock(8, dilation).double()
    x = torch.randn(2, 8, 64, dtype=torch.float64, requires_grad=True)
    actual = block(x)
    normed = block.norm(x.transpose(1, 2)).transpose(1, 2)
    reference = x + block.mix(F.silu(block.conv(F.pad(normed, (2*dilation, 0)))))
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    inputs = (x, *block.parameters())
    ga = torch.autograd.grad(actual.square().mean(), inputs, retain_graph=True)
    gr = torch.autograd.grad(reference.square().mean(), inputs)
    for left, right in zip(ga, gr):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_fft_phase_reference_predicts_h1_after_last_history_frame():
    phase = PhaseAwareForecast(dim=16)
    t = torch.arange(64)
    history = torch.cos(2*math.pi*3*t/64 + .7).view(1, 1, 64)
    coeff = phase.history_coefficients(history)
    extrapolated = phase.extrapolate_modes(coeff, torch.zeros_like(coeff.real)).real.sum(-1)
    expected = torch.cos(2*math.pi*3*torch.arange(64, 96)/64 + .7).view(1, 1, 32)
    torch.testing.assert_close(extrapolated, expected, atol=5e-6, rtol=5e-6)
    # 同幅值而不同相位必须产生不同的未来，不能仅保留magnitude。
    other = phase.history_coefficients(torch.cos(2*math.pi*3*t/64 - .7).view(1, 1, 64))
    torch.testing.assert_close(coeff.abs(), other.abs(), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(extrapolated, phase.extrapolate_modes(other, torch.zeros_like(other.real)).real.sum(-1))


def test_oscillation_is_bounded_and_zero_history_is_finite():
    phase = PhaseAwareForecast(dim=16)
    x, state = torch.randn(2, 4, 64), torch.randn(2, 4, 16)
    with torch.no_grad():
        phase.mode_parameters.weight.zero_()
        phase.mode_parameters.bias.fill_(100.)
    features, osc = phase(x, state)
    bound = 1.5*2*phase.history_coefficients(x).abs().sum(-1)
    assert bool((osc.abs() <= bound[..., None]+1e-5).all())
    z = torch.zeros_like(x, requires_grad=True)
    features, osc = phase(z, state)
    (features.square().mean()+osc.square().mean()).backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(features).all()


@pytest.mark.parametrize('mode', ['pretrain', 'finetune'])
def test_new_modules_train_during_freeze_and_join_scheduler(mode):
    torch.manual_seed(2024)
    m = model()
    args = SimpleNamespace(mode=mode, lr_peak=5e-5, lr_init=2e-5, lr_final=1e-5,
                           train_epochs=20, warmup_epochs=3, backbone_lr_scale=.2, weight_decay=.01)
    set_trainability(m, mode, False)
    set_training_mode(m, False)
    dynamics = set(prediction_adapter_param_names(m))
    assert all(p.requires_grad for n, p in m.named_parameters() if n in dynamics)
    assert not any(p.requires_grad for n, p in m.named_parameters() if n.startswith('blocks.'))
    if mode == 'finetune':
        assert not m.cpm_head.out_proj.weight.requires_grad
    opt = build_optimizer(m, None, args)
    scheduler = build_scheduler(opt, args)
    pred, _ = m(torch.randn(2, 4, 1, 64))
    pred.square().mean().backward()
    for n, p in m.named_parameters():
        if n in dynamics:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
    opt.step(); scheduler.step()
    dynamic_param = next(p for n, p in m.named_parameters() if n in dynamics)
    original_state = opt.state[dynamic_param]
    set_trainability(m, mode, True)
    added, _ = unfreeze_while_preserving_state(m, opt, args)
    scheduler.sync_param_groups()
    assert added > 0 and opt.state[dynamic_param] is original_state
    params = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(params) == len(set(params)) == len(list(m.parameters()))
    lrs = {g['name']: g['lr'] for g in opt.param_groups}
    assert lrs['encoder'] == pytest.approx(lrs['prediction']*.2)


def test_fusion_is_in_one_revin_space_and_interfaces_remain_compatible():
    m = model().eval()
    x = torch.randn(2, 4, 1, 64)*2 + 5
    calls, head_outputs = [], []
    norm_hook = m.rev_norm.register_forward_hook(lambda _, args, out: calls.append(args[1]))
    head_hook = m.cpm_head.register_forward_hook(lambda _, args, out: head_outputs.append(out))
    pred, aux = m(x, return_tokens=True)
    norm_hook.remove(); head_hook.remove()
    assert calls == ['norm', 'denorm']
    normalized = head_outputs[0] + aux['oscillation_gate']*aux['oscillation_pred']
    expected = m.rev_norm(normalized[..., None], 'denorm', stats=aux['norm_stats']).squeeze(-1)
    torch.testing.assert_close(pred, expected)
    assert pred.shape == (2, 4, 32) and aux['phase_features'].shape == (2, 4, 32, 16)
    assert aux['history_tokens'].shape == (2, 4, 8, 16)
    torch.testing.assert_close(m.forecast_horizon(x[:, :, 0].transpose(1, 2)), pred.transpose(1, 2))
    assert float(aux['oscillation_gate'].max()) < .025
    original = NeuroTwinTFM(features=4, dim=16, num_layers=1, num_heads=2, cpm_layers=1)
    assert not prediction_adapter_param_names(original)
    assert not any('phase' in n or 'multiscale' in n for n in original.state_dict())


def test_weighted_all_horizon_loss_and_partial_mask_normalization():
    cfg = load_model_config('configs/neurotwin_multiscale_phase.json')['loss']
    cfg = {**cfg, **{k: 0. for k in cfg if k.startswith('lambda_') and k != 'lambda_rec'}}
    criterion = ForecastFidelityLoss(**cfg)
    p = torch.zeros(2, 3, 32, requires_grad=True)
    target = torch.ones_like(p)
    mask = torch.ones(2, 32); mask[1, 7:] = 0
    loss, _ = criterion(p, target, mask, torch.zeros(2, 3))
    assert loss.item() == pytest.approx(1.)
    loss.backward()
    assert p.grad[0, 0, -1]/p.grad[0, 0, 0] == pytest.approx(1.5)
    assert bool((p.grad[0] < 0).all()) and bool((p.grad[1, :, 7:] == 0).all())
    zero, _ = criterion(p, target, torch.zeros_like(mask), torch.zeros(2, 3))
    assert zero.item() == 0


def test_local_pcc_uses_three_overlapping_sixteen_tr_windows():
    cfg = load_model_config('configs/neurotwin_multiscale_phase.json')['loss']
    p, target, mask = torch.randn(2, 3, 32), torch.randn(2, 3, 32), torch.ones(2, 32)
    _, terms = ForecastFidelityLoss(**cfg)(p, target, mask, torch.zeros(2, 3))
    expected = torch.stack([temporal_pcc_loss(p[..., s:s+16], target[..., s:s+16], mask[:, s:s+16])
                            for s in (0, 8, 16)]).mean()
    torch.testing.assert_close(terms['loss_local_corr'], expected)
    args = parse_args(['--model_config', 'configs/neurotwin_multiscale_phase.json', '--mode', 'finetune'])
    assert args.selection_corr_weight == .15 and args.augmentation['scale_std'] == 0
