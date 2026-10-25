"""Small contract checks for training-only horizon supervision masks."""
import torch

from main import apply_train_supervision_horizon
from train.losses import TFMDualLoss


def _inputs(seed=20261024):
    torch.manual_seed(seed)
    batch, rois, horizon = 2, 5, 32
    one_pred = torch.randn(batch, rois, 1, 1)
    one_target = torch.randn(batch, rois, 1, 1)
    cpm_pred = torch.randn(batch, rois, horizon)
    cpm_target = torch.randn(batch, rois, horizon)
    dynamics_pred = torch.randn(batch, rois, horizon)
    dynamics_target = torch.randn(batch, rois, horizon)
    mean_norm = torch.randn(batch, rois, horizon)
    logvar_norm = torch.randn(batch, rois, horizon)
    target_norm = torch.randn(batch, rois, horizon)
    context_last = torch.randn(batch, rois)
    cpm_mask = torch.ones(batch, horizon)
    dynamics_mask = torch.ones(batch, horizon)
    cpm_mask[0, 2] = 0.0
    dynamics_mask[1, 4] = 0.0
    return locals()


def _criterion():
    return TFMDualLoss(
        lambda_one=1.0, lambda_cpm=1.0, lambda_pcc=0.1,
        lambda_dynamics=0.5, lambda_variance=0.05,
        weighting='uncertainty', lambda_temporal_diff=0.05,
        lambda_cross_head=0.02, cross_head_start_epoch=0)


def _run(values, cpm_mask, dynamics_mask, *, grad=False):
    tensors = {name: values[name] for name in (
        'one_pred', 'cpm_pred', 'dynamics_pred', 'logvar_norm')}
    if grad:
        tensors = {name: value.detach().clone().requires_grad_(True)
                   for name, value in tensors.items()}
    criterion = _criterion()
    loss, stats = criterion(
        tensors['one_pred'], values['one_target'],
        tensors['cpm_pred'], values['cpm_target'], cpm_mask,
        context_last=values['context_last'], epoch=6,
        dynamics_pred=tensors['dynamics_pred'],
        dynamics_target=values['dynamics_target'],
        dynamics_mask=dynamics_mask,
        dynamics_mean_norm=values['mean_norm'],
        dynamics_logvar_norm=tensors['logvar_norm'],
        dynamics_target_norm=values['target_norm'])
    return loss, stats, tensors


def test_default_and_full_horizon_preserve_full_criterion():
    values = _inputs()
    cpm_default = apply_train_supervision_horizon(values['cpm_mask'], 0)
    dyn_default = apply_train_supervision_horizon(values['dynamics_mask'], 0)
    cpm_full = apply_train_supervision_horizon(values['cpm_mask'], 32)
    dyn_full = apply_train_supervision_horizon(values['dynamics_mask'], 32)
    assert cpm_default is values['cpm_mask']
    assert dyn_default is values['dynamics_mask']
    assert cpm_full is values['cpm_mask']
    assert dyn_full is values['dynamics_mask']

    default_loss, default_stats, _ = _run(values, cpm_default, dyn_default)
    full_loss, full_stats, _ = _run(values, cpm_full, dyn_full)
    torch.testing.assert_close(default_loss, full_loss, rtol=0, atol=0)
    for key in ('loss_one', 'loss_cpm', 'loss_dynamics', 'loss_variance',
                'loss_temporal_diff', 'loss_cross_head'):
        torch.testing.assert_close(default_stats[key], full_stats[key],
                                   rtol=0, atol=0)


def test_h8_masks_targets_gradients_and_h8_h9_difference_boundary():
    values = _inputs()
    cpm_h8 = apply_train_supervision_horizon(values['cpm_mask'], 8)
    dyn_h8 = apply_train_supervision_horizon(values['dynamics_mask'], 8)
    expected_cpm = torch.zeros(2, 32)
    expected_dynamics = torch.zeros(2, 32)
    expected_cpm[:, :8] = 1
    expected_dynamics[:, :8] = 1
    expected_cpm[0, 2] = 0
    expected_dynamics[1, 4] = 0
    torch.testing.assert_close(cpm_h8, expected_cpm)
    torch.testing.assert_close(dyn_h8, expected_dynamics)
    assert torch.all(values['cpm_mask'][:, 8:] == 1)
    assert torch.all(values['dynamics_mask'][:, 8:] == 1)

    changed = {key: value.clone() if torch.is_tensor(value) else value
               for key, value in values.items()}
    changed['cpm_target'][:, :, 8:] += 7.0
    changed['dynamics_target'][:, :, 8:] -= 5.0
    changed['target_norm'][:, :, 8:] += 3.0

    loss_a, stats_a, tensors_a = _run(values, cpm_h8, dyn_h8, grad=True)
    loss_b, stats_b, tensors_b = _run(changed, cpm_h8, dyn_h8, grad=True)
    names = tuple(tensors_a)
    grads_a = torch.autograd.grad(loss_a, tuple(tensors_a.values()))
    grads_b = torch.autograd.grad(loss_b, tuple(tensors_b.values()))
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=1e-7)
    torch.testing.assert_close(stats_a['loss_temporal_diff'],
                               stats_b['loss_temporal_diff'], rtol=0, atol=1e-7)
    for grad_a, grad_b in zip(grads_a, grads_b):
        torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=1e-7)
    for name, grad in zip(names, grads_a):
        if name in ('cpm_pred', 'dynamics_pred', 'logvar_norm'):
            assert torch.count_nonzero(grad[:, :, 8:]) == 0


def test_cutoff_validation():
    mask = torch.ones(2, 32)
    assert apply_train_supervision_horizon(mask, 8).shape == (2, 32)
    try:
        apply_train_supervision_horizon(mask, 33)
    except ValueError as exc:
        assert '超过当前 future' in str(exc)
    else:
        raise AssertionError('cutoff beyond future horizon should fail')
