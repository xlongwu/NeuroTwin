"""个体化模型退化诊断：只读消融、损失梯度和训练状态检查，不执行参数更新。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from analysis.evaluate import build_model_from_args, load_condition_bundle, load_train_args
from analysis.next_point_eval import build_next_point_loader
from main import condition_batch
from train.individualized import subject_representation_loss, temporal_difference_l1
from train.losses import ForecastFidelityLoss, fc_mse, frequency_l1, masked_l1
from utils.common import set_seed


ROOT = Path(__file__).resolve().parents[1]


def load_model(path: Path, device, state: dict | None = None, initialize: bool = False):
    args = load_train_args(path)
    set_seed(args.seed)
    model = build_model_from_args(args, device, load_condition_bundle(path))
    if not initialize:
        saved = state if state is not None else torch.load(path, map_location='cpu', weights_only=True)['state_dict']
        model.load_state_dict(saved, strict=True)
    return model, args


def sample_batch(dataset, subjects: list[str], count: int, device) -> dict:
    chosen = np.linspace(0, len(subjects) - 1, min(count, len(subjects))).round().astype(int)
    samples = [dataset.make_sample(dataset._subject_pos[subjects[i]], 64, 63, future_steps=8) for i in chosen]
    batch = default_collate(samples)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def predict(model, batch, **kwargs):
    return model(batch['x'], batch['sc'], condition=condition_batch(batch, batch['x'].device),
                 calibration_bold=batch['calibration_bold'], with_aux=False, **kwargs)[1]


@torch.no_grad()
def forecast_probe(model, batch, name: str, **kwargs) -> dict:
    model.eval()
    aux = predict(model, batch, **kwargs)
    pred, target = aux['dynamics_pred'], batch['future']
    rates = (pred[:, :, 7] - pred[:, :, 6]).abs().mean(dim=1)
    observed = (target[:, :, 7] - target[:, :, 6]).abs().mean(dim=1)
    state = model.initialize_state(batch['x'][:, :, 0].transpose(1, 2), batch['sc'],
        condition=condition_batch(batch, pred.device), calibration_bold=batch['calibration_bold'])
    terms = model.one_step_head.field_terms(state['z'], state['adj'],
        cond=state['cond'], profile=state['profile'], memory=state['memory'],
        personal_parameters=state['personal_parameters'],
        pathology_parameters=state['pathology_parameters'] if kwargs.get('pathology_active', True) else None)
    result = {'配置': name, '被试数': len(pred),
              'H1_MAE': float((pred[:, :, 0] - target[:, :, 0]).abs().mean()),
              'H1至8_MAE': float((pred - target).abs().mean()),
              'H8_MAE': float((pred[:, :, 7] - target[:, :, 7]).abs().mean()),
              'H8差分幅度比': float((rates / observed.clamp_min(1e-8)).mean()),
              '读出权重范数': float(model.one_step_head.decode_rate.weight.norm()),
              '初始BOLD速率RMS': float(model.one_step_head._rate(state['z']).square().mean().sqrt()),
              '初始latent_RMS': float(state['z'].square().mean().sqrt()),
              'kappa均值': float(terms['kappa'].mean()), 'kappa最大值': float(terms['kappa'].max())}
    for key in ('diffusion', 'residual', 'individual', 'pathology'):
        value = terms[key]
        result[key + '_RMS'] = float(value.square().mean().sqrt()) if value is not None else 0.0
    return result


def gradient_probe(model, args, batch, name: str, criterion_state: dict | None = None) -> tuple:
    model.train()
    set_seed(args.seed)
    criterion = ForecastFidelityLoss().to(batch['x'].device)
    if criterion_state is not None:
        criterion.load_state_dict(criterion_state, strict=True)
    aux = predict(model, batch)
    pred, target, mask = aux['dynamics_pred'], batch['future'], batch['future_mask']
    second = model.calibrate_profile(batch['stable_bold_b'])
    consistency, contrast = subject_representation_loss(
        aux['personal_profile'], second, batch['subj_id'], args.contrastive_temperature)
    parts = {
        '重建': masked_l1(pred, target, mask) * (-criterion.log_s['rec']).exp(),
        'FC': fc_mse(pred, target, mask) * (-criterion.log_s['fc']).exp(),
        '频域': frequency_l1(pred, target, mask) * (-criterion.log_s['freq']).exp(),
        '时间差分': args.lambda_delta * temporal_difference_l1(pred, target, batch['x_last'], mask),
        'Profile一致性': args.lambda_subject * consistency,
        '主体对比': args.lambda_subject * contrast,
        '个体参数正则': args.lambda_personal_reg * model.one_step_head.personal_dynamics.regularization(
            aux['personal_parameters'], model.one_step_head.kappa_raw)}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    shared = [i for i, (n, _) in enumerate(named) if n.startswith(('patch_embed.', 'blocks.', 'final_norm.'))]
    readout = [i for i, (n, _) in enumerate(named) if n.startswith('one_step_head.decode_rate.')]
    rows, gradients = [], {}
    for label, loss in parts.items():
        grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        def norm(indices):
            return float(sum((grad[i].float().square().sum() for i in indices if grad[i] is not None), loss.new_zeros(())).sqrt())
        rows.append({'模型': name, '损失项': label, '加权损失值': float(loss.detach()),
                     '模型梯度范数': norm(range(len(parameters))), '共享编码器梯度范数': norm(shared),
                     '读出梯度范数': norm(readout)})
        gradients[label] = [grad[i].detach() if grad[i] is not None else torch.zeros_like(parameters[i]) for i in shared]
    base = [sum(items) for items in zip(*(gradients[k] for k in ('重建', 'FC', '频域', '时间差分', '个体参数正则')))]
    stable = [a + b for a, b in zip(gradients['主体对比'], gradients['Profile一致性'])]
    dot = sum((a * b).sum() for a, b in zip(base, stable))
    nb = sum(a.square().sum() for a in base).sqrt()
    ns = sum(a.square().sum() for a in stable).sqrt()
    summary = {'模型': name, '预测及正则_共享梯度范数': float(nb), 'SDRL_共享梯度范数': float(ns),
               'SDRL除预测梯度范数': float(ns / nb) if nb > 0 else None,
               '预测与SDRL梯度余弦': float(dot / (nb * ns)) if nb > 0 and ns > 0 else None,
               '说明': '合法训练被试固定首锚点，无数据增强；FP32单批次局部梯度；未执行optimizer.step'}
    return rows, summary


def training_state_summary(path: Path) -> tuple:
    state = torch.load(path, map_location='cpu', weights_only=True)
    plain = json.loads(state['plain'])
    steps = [float(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    progress = plain['progress']
    summary = {'epoch': progress['epoch'], '批次计数': progress['global_step'],
        'Adam步数最小': min(steps), 'Adam步数最大': max(steps),
        '跳过更新数': progress['global_step'] - max(steps), 'scaler': plain['scaler']}
    return state, summary


def write_training_history(run: Path, out: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    records, selection = [], {}
    for path in run.rglob('events.out.tfevents*'):
        cohort = 'HC' if 'B2_pretrain' in str(path) else 'MDD'
        events = EventAccumulator(str(path), size_guidance={'scalars': 0}).Reload()
        by_step = {}
        for tag in events.Tags()['scalars']:
            for event in events.Scalars(tag):
                by_step.setdefault(event.step, {'阶段': cohort, '轮次': event.step + 1})[tag] = event.value
        ordered = [by_step[k] for k in sorted(by_step)]
        records.extend(ordered)
        best = min(ordered, key=lambda row: row['Val/Selection_H1_8'])
        last = ordered[-1]
        selection[cohort] = {
            '选中轮次': best['轮次'], '选中MAE': best['Val/H1_8_MAE'], '选中FC_MSE': best['Val/FC_MSE'],
            '末轮': last['轮次'], '末轮MAE': last['Val/H1_8_MAE'], '末轮FC_MSE': last['Val/FC_MSE']}
    pd.DataFrame(records).to_csv(out / '训练目标与选模轨迹.csv', index=False)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    ns = parser.parse_args()
    torch.set_num_threads(4)
    run, out, device = ns.run.resolve(), ns.out_dir.resolve(), torch.device(ns.device)
    out.mkdir(parents=True, exist_ok=True)
    hc = run / 'neurotwin_B2_pretrain/base_best.pt'
    mdd = run / 'neurotwin_B3_finetune/finetuned_best.pt'
    args = load_train_args(hc)
    args.num_workers, args.eval_batch_size, args.data_root = 0, 32, str(ROOT / 'data')
    _, _, loader = build_next_point_loader(args, 'val', condition_bundle=load_condition_bundle(hc))
    loader.base_dataset.stable_window_length = 32
    validation = sample_batch(loader.base_dataset, loader.val_subjects, 32, device)
    training = sample_batch(loader.base_dataset, loader.train_subjects, 16, device)
    summaries, states = {}, {}
    for label, folder in (('HC', 'neurotwin_B2_pretrain'), ('MDD', 'neurotwin_B3_finetune')):
        states[label], summaries[label] = training_state_summary(run / folder / 'training_state.pt')
    rows, amplitude_rows = [], []
    model, _ = load_model(ROOT / 'checkpoints/1031/neurotwin_pretrain/base_best.pt', device)
    rows.append(forecast_probe(model, validation, '旧HC最佳'))
    old_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred = predict(model.eval(), validation)['dynamics_pred']
        anchor = validation['x_last'].unsqueeze(-1)
        for factor in (0.01, 0.1, 1.0):
            scaled = anchor + factor * (pred - anchor)
            amplitude_rows.append({'配置': '旧HC预测残差乘系数', '系数': factor,
                '重建MAE': float(masked_l1(scaled, validation['future'])),
                'FC_MSE': float(fc_mse(scaled, validation['future']))})
    del model
    warm, _ = load_model(hc, device, initialize=True)
    warm_state = warm.state_dict()
    for key, value in old_state.items():
        warm_state[key] = value
    warm.load_state_dict(warm_state, strict=True)
    rows.append(forecast_probe(warm, validation, '旧HC权重加零初始化IDG'))
    del warm, old_state
    model, _ = load_model(hc, device)
    rows.append(forecast_probe(model, validation, '新HC最佳41轮'))
    for gain in (5.0, 20.0):
        weight, bias = model.one_step_head.decode_rate.weight, model.one_step_head.decode_rate.bias
        with torch.no_grad():
            original_w, original_b = weight.clone(), bias.clone()
            weight.mul_(gain)
            bias.mul_(gain)
        rows.append(forecast_probe(model, validation, f'新HC读出乘{gain:g}'))
        with torch.no_grad():
            weight.copy_(original_w)
            bias.copy_(original_b)
    generator = model.one_step_head.personal_dynamics
    model.one_step_head.personal_dynamics = None
    rows.append(forecast_probe(model, validation, '新HC关闭全部IDG'))
    model.one_step_head.personal_dynamics = generator
    for attribute, value, label in [('personalized_graph', False, '新HC关闭个体图'),
                                    ('regional_diffusion', False, '新HC关闭区域扩散'),
                                    ('scale', 0.0, '新HC关闭低秩残差')]:
        original = getattr(generator, attribute)
        setattr(generator, attribute, value)
        rows.append(forecast_probe(model, validation, label))
        setattr(generator, attribute, original)
    del model
    for key, label in (('model', '新HC末轮53原始参数'), ('ema', '新HC末轮53EMA')):
        model, _ = load_model(hc, device, states['HC'][key])
        rows.append(forecast_probe(model, validation, label))
        del model
    args_mdd = load_train_args(mdd)
    args_mdd.num_workers, args_mdd.eval_batch_size = 0, 32
    _, _, mdd_loader = build_next_point_loader(args_mdd, 'val', condition_bundle=load_condition_bundle(mdd))
    mdd_validation = sample_batch(mdd_loader.base_dataset, mdd_loader.val_subjects, 32, device)
    model, _ = load_model(mdd, device)
    rows.append(forecast_probe(model, mdd_validation, 'MDD启用PDD'))
    rows.append(forecast_probe(model, mdd_validation, 'MDD关闭PDD', pathology_active=False))
    del model
    gradients, conflicts = [], []
    for label, initialize in [('随机初始化B2', True), ('新HC最佳41轮', False)]:
        model, model_args = load_model(hc, device, initialize=initialize)
        criterion_state = None
        if not initialize:
            meta = torch.load(hc, map_location='cpu', weights_only=True)['config']['loss_weighting']['log_s']
            criterion_state = {f'log_s.{k}': torch.tensor(v) for k, v in meta.items()}
        per_loss, conflict = gradient_probe(model, model_args, training, label, criterion_state)
        gradients.extend(per_loss)
        conflicts.append(conflict)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    # 同一已训模型在额外三个合法训练批次上检查局部梯度，避免只看首锚点。
    rng = np.random.default_rng(2024)
    model, model_args = load_model(hc, device)
    meta = torch.load(hc, map_location='cpu', weights_only=True)['config']['loss_weighting']['log_s']
    criterion_state = {f'log_s.{k}': torch.tensor(v) for k, v in meta.items()}
    for number, fraction in enumerate((0.33, 0.66, 1.0), start=2):
        ids = rng.choice(loader.train_subjects, 16, replace=False)
        samples = []
        ds = loader.base_dataset
        for sid in ids:
            lo, hi = ds.legal_anchor_range(sid, 64, max_offset=8)
            cutoff = lo + round((hi-lo) * fraction)
            samples.append(ds.make_sample(ds._subject_pos[sid], 64, cutoff, future_steps=8))
        batch = default_collate(samples)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        per_loss, conflict = gradient_probe(model, model_args, batch,
            f'新HC最佳41轮_诊断批次{number}', criterion_state)
        gradients.extend(per_loss)
        conflicts.append(conflict)
    pd.DataFrame(rows).to_csv(out / '只读前向消融.csv', index=False)
    pd.DataFrame(gradients).to_csv(out / '加权损失梯度.csv', index=False)
    pd.DataFrame(conflicts).to_csv(out / '共享编码器梯度竞争.csv', index=False)
    pd.DataFrame(amplitude_rows).to_csv(out / 'FC与振幅尺度诊断.csv', index=False)
    summary = {'训练状态': summaries, '梯度对比': conflicts,
               '初始化': {'init_weight': getattr(args, 'init_weight', ''),
                           'pretrained_weight': getattr(args, 'pretrained_weight', '')},
               '选模': write_training_history(run, out),
               '范围': 'HC和MDD各32名验证被试的首K64锚点；4批各16名HC训练被试、合法K64锚点的梯度；局部诊断而非全测试集新成绩',
               '执行': '仅forward和autograd.grad，无优化器更新，未写回checkpoint'}
    (out / '诊断摘要.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(gradients).to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
