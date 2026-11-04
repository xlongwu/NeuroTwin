"""Read-only validation diagnostics of the trained E2; no fitting or source edits.

Run from repository root with PYTHONPATH=. and an available CUDA device.
Inference branch removal is a dependency check, not retrained E0/E1 evidence.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from analysis.evaluate import load_train_args, load_condition_bundle, build_model_from_args
from analysis.next_point_eval import build_next_point_loader
from main import condition_batch, dynamics_extras
from train.forecast_metrics import STAGE_LABELS, h32_stage_metrics
from train.losses import masked_l1, fc_mse, frequency_l1


ROOT = Path.cwd()
RUN = Path('checkpoints/h32_spectral/E2')
OUT = Path(__file__).resolve().parents[1]
CKPT = RUN / 'neurotwin_finetune/finetuned_best.pt'
device = torch.device('cuda:0')
torch.set_num_threads(4)
args = load_train_args(CKPT)
bundle = load_condition_bundle(CKPT)
model = build_model_from_args(args, device, condition_bundle=bundle).eval()
payload = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(payload['state_dict'], strict=True)
assert not model.state_feedback and model.one_step_head.spectral_oscillator.latent_proj is None
args.num_workers = 0
args.persistent_workers = False
args.eval_batch_size = 32
args.eval_rollout = False  # Keep the same fixed H32 anchors; no extra rollout tasks.
original_loader, val_subjects, source = build_next_point_loader(args, 'val', condition_bundle=bundle)
ordered = sorted(val_subjects)
selected = [ordered[i] for i in np.linspace(0, len(ordered) - 1, 64).round().astype(int)]
assert len(set(selected)) == 64
view = original_loader.dataset
indices = [i for i, (sid_pos, _, _) in enumerate(view.tasks)
           if source.base_dataset.subject_ids[sid_pos] in selected]
loader = DataLoader(Subset(view, indices), batch_size=32, shuffle=False, num_workers=0)
assert len(indices) == 64 * 8
head = model.one_step_head
osc = head.spectral_oscillator

weights = None
current_epoch = None
best_epoch = payload['config']['epoch_1based']
for line in (RUN / '微调.log').read_text(errors='replace').replace('\r', '\n').splitlines():
    m = re.match(r'Epoch (\d+) \|', line)
    if m:
        current_epoch = int(m[1])
    if line.startswith('[LossW]') and current_epoch == best_epoch:
        weights = {key: float(re.search(key + r': w=([-+\deE.]+)', line)[1])
                   for key in ('rec', 'fc', 'freq')}
assert weights is not None

metric_rows, state_rows, component_rows, gradient_rows, spectrum_rows, decomposition_rows = [], [], [], [], [], []
numeric_check = None
normalization_check = None


def append_metrics(arm, pred, target, mask, anchor, ids):
    values = h32_stage_metrics(pred, target, mask, anchor)
    fc = fc_mse(pred, target, mask).item()
    for i, sid in enumerate(ids):
        for j, label in enumerate(STAGE_LABELS):
            row = {'arm': arm, 'subject': str(sid), 'stage': label}
            row.update({key: float(value[i, j]) for key, value in values.items()})
            metric_rows.append(row)
    return fc


@torch.no_grad()
def rollout(state, record=False, disable_osc=False, ids=None, force_gate=False):
    current = state
    predicted, main_delta, osc_delta = [], [], []
    main_rates, osc_rates, gates = [], [], []
    hooks = []
    if record:
        hooks.append(head.decode_rate.register_forward_hook(lambda m, a, out: main_rates.append(out.squeeze(-1))))
        hooks.append(osc.gate.register_forward_hook(lambda m, a, out: gates.append(out.sigmoid())))
        def osc_readout(m, a, out):
            osc_rates.append((gates[-1] * out).squeeze(-1))
        hooks.append(osc.readout.register_forward_hook(osc_readout))
    if disable_osc:
        hooks.append(osc.readout.register_forward_hook(lambda m, a, out: torch.zeros_like(out)))
    if force_gate:
        hooks.append(osc.gate.register_forward_hook(lambda m, a, out: torch.full_like(out, 20)))
    try:
        for h in range(32):
            current = model.advance_state(current)
            predicted.append(model.readout_state(current))
            if record:
                stage_factor = .5 / head.ode_steps
                main_delta.append(stage_factor * torch.stack(main_rates[-2 * head.ode_steps:]).sum(0))
                osc_delta.append(stage_factor * torch.stack(osc_rates[-2 * head.ode_steps:]).sum(0))
                if h in (7, 15, 31):
                    z = current['z']
                    initial = state['z']
                    def diversity(tensor):
                        return (tensor - tensor.mean(1, keepdim=True)).square().mean((1, 2)).sqrt()
                    zn = torch.nn.functional.normalize(z, dim=-1)
                    n = z.shape[1]
                    similarity = (zn.sum(1).square().sum(-1) - n) / (n * (n - 1))
                    ratio = diversity(z) / diversity(initial).clamp_min(1e-8)
                    force_u, force_v = osc.forcing(z, current['cond'], current['profile'])
                    force_rms = (force_u.square() + force_v.square()).mean((1, 2)).sqrt()
                    gamma = torch.nn.functional.softplus(osc.damping_raw)
                    rotation_rms = ((gamma.square() + osc.omega.square())
                                    * (current['osc_u'].square() + current['osc_v'].square())).mean((1, 2)).sqrt()
                    for i, sid in enumerate(ids):
                        state_rows.append({'subject': sid, 'horizon': h + 1,
                                           'latent_roi_diversity_ratio': ratio[i].item(),
                                           'latent_roi_mean_cosine': similarity[i].item(),
                                           'forcing_to_rotation_rms_ratio': (force_rms[i] / rotation_rms[i].clamp_min(1e-8)).item(),
                                           'oscillator_state_rms': current['osc_u'][i].square().add(current['osc_v'][i].square()).mean().sqrt().item()})
    finally:
        for hook in hooks:
            hook.remove()
    prediction = torch.stack(predicted, -1)
    if record:
        initial_q = state['q'].unsqueeze(-1)
        main_q = torch.stack(main_delta, -1).cumsum(-1)
        osc_q = torch.stack(osc_delta, -1).cumsum(-1)
        reconstructed = model._denorm_3d(initial_q + main_q + osc_q, state['norm_stats'])
        torch.testing.assert_close(reconstructed, prediction, atol=2e-5, rtol=2e-5)
        anchor_data = model.readout_state(state).unsqueeze(-1)
        main_data = model._denorm_3d(initial_q + main_q, state['norm_stats']) - anchor_data
        osc_data = model._denorm_3d(initial_q + osc_q, state['norm_stats']) - anchor_data
        for h in (0, 7, 15, 23, 31):
            for i, sid in enumerate(ids):
                component_rows.append({'subject': sid, 'horizon': h + 1,
                                       'main_cumulative_change_rms': main_data[i, :, h].square().mean().sqrt().item(),
                                       'osc_cumulative_change_rms': osc_data[i, :, h].square().mean().sqrt().item(),
                                       'gate_mean': torch.stack(gates[h * 4:(h + 1) * 4])[:, i].mean().item(),
                                       'main_instantaneous_rate_rms': torch.stack(main_rates[h * 4:(h + 1) * 4])[:, i].square().mean().sqrt().item(),
                                       'osc_instantaneous_rate_rms': torch.stack(osc_rates[h * 4:(h + 1) * 4])[:, i].square().mean().sqrt().item()})
    return prediction


for batch_index, batch in enumerate(loader):
    x = batch['x'].to(device)
    target, mask = batch['future'].to(device), batch['future_mask'].to(device)
    ids = list(batch['subj_id'])
    assert mask.bool().all()
    cal, control = dynamics_extras(batch, device)
    assert control is None
    temporal = []
    hook = model.spectral_memory_gate.register_forward_pre_hook(
        lambda m, a: temporal.append(a[0][..., :model.dim]))
    with torch.no_grad():
        state = model.initialize_state(x[:, :, 0].transpose(1, 2),
            condition=condition_batch(batch, device), calibration_bold=cal)
    hook.remove()
    assert len(temporal) == 1
    memory_delta = state['memory'] - temporal[0]
    ratio = memory_delta.square().mean((1, 2)).sqrt() / temporal[0].square().mean((1, 2)).sqrt().clamp_min(1e-8)
    for i, sid in enumerate(ids):
        zn = torch.nn.functional.normalize(state['z'][i], dim=-1)
        n = zn.shape[0]
        state_rows.append({'subject': sid, 'horizon': 0, 'frequency_memory_relative_rms': ratio[i].item(),
                           'latent_roi_diversity_ratio': 1.0,
                           'latent_roi_mean_cosine': ((zn.sum(0).square().sum() - n) / (n * (n - 1))).item()})
    with torch.no_grad():
        anchor = x[:, :, 0, -1]
        full = rollout(state, record=True, ids=ids)
        append_metrics('full_E2', full, target, mask, anchor, ids)
        gate_one = rollout(state, force_gate=True)
        append_metrics('osc_gate_forced_one', gate_one, target, mask, anchor, ids)
        steps = head.ode_steps
        try:
            head.ode_steps = 4
            refined_all = rollout(state)
        finally:
            head.ode_steps = steps
        append_metrics('heun_4_substeps', refined_all, target, mask, anchor, ids)
        for label, start, end in [('H1_8', 0, 8), ('H9_16', 8, 16), ('H17_32', 16, 32)]:
            p_stage, t_stage = full[..., start:end], target[..., start:end]
            pc = p_stage - p_stage.mean(-1, keepdim=True)
            tc = t_stage - t_stage.mean(-1, keepdim=True)
            bias = (p_stage.mean(-1) - t_stage.mean(-1)).square().mean(-1)
            pv, tv = pc.square().mean((1, 2)), tc.square().mean((1, 2))
            cov = (pc * tc).mean((1, 2))
            mse = (p_stage - t_stage).square().mean((1, 2))
            torch.testing.assert_close(mse, bias + pv + tv - 2 * cov)
            for i, sid in enumerate(ids):
                decomposition_rows.append({'subject': sid, 'stage': label, 'mse': mse[i].item(),
                                           'mean_bias_squared': bias[i].item(), 'pred_temporal_variance': pv[i].item(),
                                           'true_temporal_variance': tv[i].item(), 'twice_covariance': 2 * cov[i].item(),
                                           'mse_contribution_of_predicted_variation': (pv[i] - 2 * cov[i]).item()})
        no_osc = rollout(state, disable_osc=True)
        append_metrics('no_osc_readout', no_osc, target, mask, anchor, ids)
        no_memory_state = {**state, 'memory': temporal[0]}
        no_memory = rollout(no_memory_state)
        append_metrics('no_frequency_memory', no_memory, target, mask, anchor, ids)
        neither = rollout(no_memory_state, disable_osc=True)
        append_metrics('neither_spectral_branch', neither, target, mask, anchor, ids)
        hist_mean = x[:, :, 0].mean(-1, keepdim=True).expand_as(target)
        append_metrics('history_mean', hist_mean, target, mask, anchor, ids)

        # Direct Fourier continuation, not a learned-model prediction.
        history = x[:, :, 0]
        coeff = torch.fft.rfft(history, norm='forward')
        bins = model.spectral_encoder.bins
        frequencies = 2 * torch.pi * bins / history.shape[-1]
        angles = frequencies[:, None] * torch.arange(64, 96, device=device)[None, :]
        selected_coeff = coeff.index_select(-1, bins)
        future_fft = (2 * (selected_coeff.real[..., None] * angles.cos()
                          - selected_coeff.imag[..., None] * angles.sin())).sum(-2) + history.mean(-1, keepdim=True)
        append_metrics('unlearned_fourier_continuation', future_fft, target, mask, anchor, ids)
        powers = coeff.abs().square()
        fraction = (powers.index_select(-1, bins).sum(-1) * 2) / (powers[..., 1:-1].sum(-1) * 2 + powers[..., -1]).clamp_min(1e-8)
        for i, sid in enumerate(ids):
            spectrum_rows.append({'subject': sid, 'selected_history_nonDC_power_fraction': fraction[i].mean().item()})

        if batch_index == 0:
            shifted = target.roll(8, dims=-1)
            normalization_check = {'construction': 'validation truth circularly shifted by 8 TR; artificial loss counterexample, not model prediction',
                                   'rec': masked_l1(shifted, target).item(), 'fc': fc_mse(shifted, target).item(),
                                   'freq': frequency_l1(shifted, target).item()}
            steps = head.ode_steps
            try:
                head.ode_steps = 4
                refined = rollout(state)
            finally:
                head.ode_steps = steps
            numeric_check = {'anchors': len(ids), 'max_output_difference': (full - refined).abs().max().item(),
                             'mean_output_difference': (full - refined).abs().mean().item(),
                             'far_mae_2_steps': (full[..., 16:] - target[..., 16:]).abs().mean().item(),
                             'far_mae_4_steps': (refined[..., 16:] - target[..., 16:]).abs().mean().item()}

    # Output-space gradient diagnostics, not a claim about parameter-gradient dominance.
    p = full.detach().clone().requires_grad_(True)
    grads, values = {}, {}
    for key, fn in [('rec', masked_l1), ('fc', fc_mse), ('freq', frequency_l1)]:
        loss = fn(p, target)
        grads[key] = torch.autograd.grad(loss, p)[0].detach() * weights[key]
        values[key] = loss.item()
    row = {'batch': batch_index, **{f'loss_{k}': v for k, v in values.items()}}
    for key, value in grads.items():
        row[f'weighted_output_gradient_norm_{key}'] = value.norm().item()
        row[f'far_gradient_share_{key}'] = (value[..., 16:].square().sum() / value.square().sum().clamp_min(1e-20)).item()
    for key in ('fc', 'freq'):
        row[f'cosine_rec_{key}'] = torch.nn.functional.cosine_similarity(grads['rec'].flatten(), grads[key].flatten(), dim=0).item()
    gradient_rows.append(row)
    print(f'batch {batch_index + 1}/{len(loader)} done', flush=True)


metrics = pd.DataFrame(metric_rows)
per_subject = metrics.groupby(['arm', 'subject', 'stage']).mean(numeric_only=True).reset_index()
summary = per_subject.groupby(['arm', 'stage']).mean(numeric_only=True).reset_index()
per_subject.to_csv(OUT / '表格/验证子集逐被试推理消融.csv', index=False)
summary.to_csv(OUT / '表格/验证子集推理消融汇总.csv', index=False)
pd.DataFrame(state_rows).to_csv(OUT / '表格/隐状态与记忆诊断.csv', index=False)
pd.DataFrame(component_rows).to_csv(OUT / '表格/BOLD积分分量诊断.csv', index=False)
pd.DataFrame(gradient_rows).to_csv(OUT / '表格/损失输出梯度诊断.csv', index=False)
pd.DataFrame(spectrum_rows).to_csv(OUT / '表格/历史频带能量诊断.csv', index=False)
pd.DataFrame(decomposition_rows).to_csv(OUT / '表格/误差方差分解.csv', index=False)
gamma = torch.nn.functional.softplus(osc.damping_raw).detach().cpu().numpy()
mode_table = pd.DataFrame({'bin': model.spectral_encoder.bins.cpu().numpy(),
                           'frequency_hz': osc.omega.cpu().numpy() / (2 * np.pi * args.spectral_tr),
                           'damping_per_TR': gamma, 'retained_amplitude_H32_without_forcing': np.exp(-32 * gamma),
                           'rate_readout_U_weight': osc.readout.weight.detach().cpu().numpy()[0]})
mode_table.to_csv(OUT / '表格/振荡模式参数.csv', index=False)
record = {'checkpoint': str(CKPT), 'checkpoint_epoch': best_epoch,
          'validation_subjects': selected, 'n_validation_subjects': len(selected), 'n_anchors': len(indices),
          'selection': '64 subjects spaced uniformly in sorted validation IDs, all original 8 anchors each; exploratory matched subset',
          'not_retrained_ablation': True, 'loss_weights_from_best_epoch_log': weights,
          'loss_invariance_counterexample': normalization_check, 'heun_refinement_check': numeric_check,
          'saved_checkpoints_modified': False, 'model_source_modified': False}
(OUT / '诊断记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(summary[['arm', 'stage', 'mae', 'temporal_pcc', 'std_ratio']].to_string(index=False), flush=True)
