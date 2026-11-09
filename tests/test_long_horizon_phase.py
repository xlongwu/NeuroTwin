"""V2的频率原点、未来因果性、梯度和局部解冻契约。"""
import math
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main import parse_args
from model import load_model_config
from models.phase_forecast import LocalGlobalPhaseForecast
from models.tfm import FutureTemporalAdapter, NeuroTwinTFM
from train.optim import (build_optimizer, build_scheduler, prediction_adapter_param_names,
                         set_trainability, set_training_mode, unfreeze_while_preserving_state,
                         save_checkpoint, load_initial_weights)


def small_model(**kwargs):
    return NeuroTwinTFM(features=3, dim=16, num_layers=4, num_heads=2, cpm_layers=1,
                        dropout=0., multiscale_channels=8, **kwargs)


def test_window_coefficients_share_last_frame_phase_and_slow_mode_uses_global():
    torch.manual_seed(2024)
    phase = LocalGlobalPhaseForecast(dim=16)
    t = torch.arange(64)
    x = torch.cos(2*math.pi*4*t/64 + .7).view(1, 1, 64)
    coeff = phase.window_coefficients(x)
    expected = torch.polar(torch.tensor(.5), torch.tensor(2*math.pi*4*63/64 + .7))
    torch.testing.assert_close(coeff[0, 0, :, 3], expected.expand(3), atol=2e-6, rtol=2e-6)
    full = torch.fft.rfft(x, dim=-1, norm='forward')[..., 1:phase.modes+1]
    rotation = 2*math.pi*phase.bins*63/64
    torch.testing.assert_close(coeff[..., -1, :], full*torch.polar(torch.ones_like(rotation), rotation),
                               atol=2e-6, rtol=2e-6)
    # 只保留最慢一个模式时，仅K64能覆盖一个周期；任意scale logits都不能改变它。
    slow = LocalGlobalPhaseForecast(dim=16, modes=1)
    x = torch.cos(2*math.pi*t/64 + .3).view(1, 1, 64)
    state = torch.randn(1, 1, 16)
    before = slow(x, state)
    with torch.no_grad():
        slow.scale_weight.weight.fill_(10.)
        slow.window_embed[:2].fill_(20.)
    after = slow(x, state)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)


def test_future_temporal_adapter_exchanges_only_previous_horizon_states():
    torch.manual_seed(2024)
    adapter = FutureTemporalAdapter(16).eval()
    x = torch.randn(2, 3, 32, 16)
    changed = x.clone(); changed[:, :, 17:] += 10
    torch.testing.assert_close(adapter(x)[:, :, :17], adapter(changed)[:, :, :17], atol=0, rtol=0)
    changed = x.clone(); changed[:, :, 0] += 1
    assert not torch.allclose(adapter(x)[:, :, 1], adapter(changed)[:, :, 1])


def test_v2_forward_backward_zero_input_and_bounded_oscillation():
    torch.manual_seed(2024)
    m = small_model()
    x = torch.randn(2, 3, 1, 64)
    y, aux = m(x, return_tokens=True)
    assert y.shape == (2, 3, 32) and torch.isfinite(y).all()
    y.square().mean().backward()
    names = set(prediction_adapter_param_names(m))
    for name, p in m.named_parameters():
        if name in names:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
    phase = m.phase_forecast
    normalized = torch.randn(2, 3, 64)
    _, osc = phase(normalized, torch.randn(2, 3, 16))
    bound = 3*phase.window_coefficients(normalized).abs().amax(-2).sum(-1)
    assert (osc.abs() <= bound[..., None]+1e-5).all()
    z = torch.zeros(2, 3, 64, requires_grad=True)
    features, osc = phase(z, torch.zeros(2, 3, 16))
    (features.square().mean()+osc.square().mean()).backward()
    assert torch.isfinite(features).all() and torch.isfinite(z.grad).all()


@pytest.mark.parametrize('mode', ['pretrain', 'finetune'])
def test_partial_unfreeze_retains_optimizer_state_and_frozen_dropout(mode):
    args = parse_args(['--mode', mode])
    m = small_model()
    set_trainability(m, mode, False, 2)
    opt = build_optimizer(m, None, args)
    scheduler = build_scheduler(opt, args)
    y, _ = m(torch.randn(1, 3, 1, 64))
    y.square().mean().backward(); opt.step(); scheduler.step()
    dynamic = m.cpm_head.dynamic_adapter.query_scales
    old_state = opt.state[dynamic]
    set_trainability(m, mode, True, 2)
    unfreeze_while_preserving_state(m, opt, args); scheduler.sync_param_groups()
    set_training_mode(m, True, 2)
    assert opt.state[dynamic] is old_state
    assert not m.patch_embed.training and not m.blocks[0].training and not m.blocks[1].training
    assert m.blocks[2].training and m.blocks[3].training and m.cpm_head.training
    for name, p in m.named_parameters():
        if name.startswith(('blocks.0.', 'blocks.1.', 'patch_embed.', 'sc_prior.')):
            assert not p.requires_grad, name
    params = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(params) == len(set(params)) == sum(p.requires_grad for p in m.parameters())
    lrs = {group['name']: group['lr'] for group in opt.param_groups}
    assert lrs['encoder'] == pytest.approx(.1*lrs['prediction'])
    assert lrs['cpm'] == pytest.approx(.1*lrs['prediction'])
    with pytest.raises(ValueError, match='unfreeze_last_layers'):
        set_trainability(m, mode, True, 5)


def test_default_recipe_disables_test_and_rejects_retired_switches():
    args = parse_args([])
    assert args.ema_decay == .995 and not args.eval_after_train
    for flag in ('--model_config', '--multiscale_phase', '--local_global_phase', '--future_temporal'):
        with pytest.raises(SystemExit):
            parse_args([flag, 'False'])


def test_temporary_training_logs_raw_and_ema_and_strictly_rebuilds_v2(tmp_path, monkeypatch):
    import main as training
    from model import build_model_from_args, load_model_checkpoint
    from models.conditioning import ConditioningConfig
    from utils.condition_schema import HC_FIELDS, fit_condition_stats

    torch.manual_seed(2024)
    args = parse_args(['--device', 'cpu', '--num_rois', '3', '--tfm_dim', '16',
                       '--multiscale_channels', '8', '--tfm_heads', '2', '--cpm_layers', '1',
                       '--train_epochs', '3', '--freeze_backbone_epochs', '1',
                       '--num_threads', '1', '--checkpoint_dir', str(tmp_path/'runs'),
                       '--init_weight', str(tmp_path/'source.pt'), '--save_training_state'])
    values = np.array([[1, 20, 12], [2, 30, 16], [1, 40, 14]], dtype=float)
    stats = fit_condition_stats(HC_FIELDS, values, np.ones_like(values), np.arange(3))
    bundle = dict(config=ConditioningConfig(group='hc3', arm='demo').to_dict(), stats=stats.to_dict(), meta={})
    original = SimpleNamespace(**vars(args))
    baseline = build_model_from_args(original, 'cpu', bundle)
    bundle['meta'] = baseline.conditioner.layout()
    save_checkpoint(args.init_weight, baseline, dict(args=vars(original), condition=bundle))

    x = torch.randn(2, 3, 1, 64)
    batch = dict(x=x, future=torch.randn(2, 3, 32), future_mask=torch.ones(2, 32),
                 x_last=x[:, :, 0, -1], subj_id=['s0', 's1'])
    class Batches(list):
        batch_sampler = SimpleNamespace(set_epoch=lambda epoch: None)
    batches = Batches([batch])
    loader = SimpleNamespace(get_train=lambda: batches, get_val=lambda: batches,
                             get_condition_stats=lambda: stats, get_condition_meta=lambda: bundle['meta'],
                             train_subjects=['train'], val_subjects=['val'], test_subjects=['test'],
                             loader_generator=torch.Generator().manual_seed(2024))
    monkeypatch.setattr(training, 'build_data_loader', lambda *a, **k: loader)
    run_dir = training.main(args)
    records = [json.loads(line) for line in (run_dir/'验证训练记录.jsonl').read_text().splitlines()]
    assert len(records) == 3
    assert all(r['val'] == r['val_ema'] and 'metric_selection_h32' in r['val_raw'] for r in records)
    assert (run_dir/'training_state.pt').exists()
    rebuilt, saved_args, _ = load_model_checkpoint(run_dir/'base_best.pt')
    assert isinstance(rebuilt.phase_forecast, LocalGlobalPhaseForecast)
    assert rebuilt.cpm_head.future_temporal is not None
    with torch.no_grad():
        assert torch.isfinite(rebuilt(x)[0]).all()
    # 相同动态结构再次初始化时，不允许静默遗漏新增模块参数。
    clone = build_model_from_args(saved_args, 'cpu', bundle)
    report = load_initial_weights(clone, run_dir/'base_best.pt', saved_args)
    assert report['new_initialized'] == []
    args.train_epochs = 4
    args.resume_training_state = str(run_dir/'training_state.pt')
    assert training.main(args) == run_dir
    records = [json.loads(line) for line in (run_dir/'验证训练记录.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in records] == [1, 2, 3, 4]
