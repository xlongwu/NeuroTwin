# coding=utf-8
"""D1/D2 状态刷新 rollout 诊断（只读评估，不训练、不改模型行为）。

三条轨迹在同一锚点、同一控制输入上逐区间推进：

- ``free``：``initialize_state`` 一次后连续 ``advance_state``（现有 free rollout）；
- ``d1``（Observed Refresh）：每步预测后用**真实未来观测**调用 ``update_state``
  重编码，再推进下一步；这是使用未来真值的诊断上界，不是可部署预测；
- ``d2``（Self Refresh）：每步预测后用**模型自身预测**调用 ``update_state``
  重编码，用于判断固定 latent/图/历史编码是否限制长期推进。

同时记录每个区间的图扩散项范数、神经残差项范数、``kappa``、latent 步长与
ROI latent 同质化程度（平均两两 cosine），用于检查 H5–H8 是否存在动态衰减。

用法::

    python -m experiments.long_horizon.refresh_rollout \
        --ckpt checkpoints/phase1_E2/neurotwin_tfm_finetune/finetuned_best.pt \
        --out_dir results/phase1/refresh_E2 --split test --steps 8 --device cuda:0
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from analysis import next_point_eval as npe
from analysis.checkpoint import load_checkpoint_compat
from experiments.evaluate_variant import (
    build_model_from_args, load_condition_bundle, load_train_args,
)
from main import _index_condition, dynamics_extras
from utils.common import set_seed

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

MODES = ('free', 'd1', 'd2')
REFRESH = {'free': None, 'd1': 'observed', 'd2': 'self'}
METRICS = ('mae', 'delta_mae', 'delta_pcc', 'delta_std_ratio', 'direction_accuracy')
STATE_KEYS = ('kappa', 'diffusion_norm', 'residual_norm', 'control_norm',
              'q_feedback_norm', 'field_norm', 'residual_to_diffusion',
              'latent_step_norm', 'state_step_abs', 'z_roi_variance',
              'z_pairwise_cos')


def _batch_condition(batch, device):
    cond = None
    if 'cond_z' in batch:
        cond = {
            'cond_z': batch['cond_z'].to(device, non_blocking=True),
            'cond_code': batch['cond_code'].to(device, non_blocking=True),
            'cond_mask': batch['cond_mask'].to(device, non_blocking=True),
        }
        if 'subject_index' in batch:
            cond['subject_index'] = batch['subject_index'].to(device, non_blocking=True)
    return cond


def _pairwise_cos_mean(z):
    """[n,F,D] → 每个样本 ROI 两两 cosine 的均值（排除对角）。"""
    zn = F.normalize(z.float(), dim=-1)
    cos = torch.bmm(zn, zn.transpose(1, 2))
    f = z.shape[1]
    return (cos.sum(dim=(1, 2)) - f) / (f * (f - 1))


def _step_state_stats(model, state, control):
    """推进前向量场分量 + 当前 latent 结构统计。"""
    head = model.one_step_head
    terms = head.field_terms(state['z'], state['adj'], state['cond'],
                             state['profile'], control, q=state.get('q'),
                             memory=state.get('memory'))
    diff = terms['diffusion']
    res = terms['residual']
    ctrl = terms['control']
    qfb = terms.get('q_feedback')
    diff_norm = diff.norm(dim=-1).mean(dim=1)
    res_norm = res.norm(dim=-1).mean(dim=1)
    field_norm = (diff + res + (ctrl if ctrl is not None else 0.0)
                  + (qfb if qfb is not None else 0.0)).norm(dim=-1).mean(dim=1)
    z = state['z']
    return {
        'kappa': terms['kappa'].squeeze(-1),
        'diffusion_norm': diff_norm,
        'residual_norm': res_norm,
        'control_norm': (ctrl.norm(dim=-1).mean(dim=1) if ctrl is not None
                         else torch.zeros_like(diff_norm)),
        'q_feedback_norm': (qfb.norm(dim=-1).mean(dim=1) if qfb is not None
                            else torch.zeros_like(diff_norm)),
        'field_norm': field_norm,
        'residual_to_diffusion': res_norm / (diff_norm + 1e-8),
        'z_roi_variance': z.float().var(dim=1, unbiased=False).mean(dim=1),
        'z_pairwise_cos': _pairwise_cos_mean(z),
    }


@torch.no_grad()
def rollout_mode(model, state0, steps, controls, refresh, future):
    """按 refresh 策略推进 ``steps`` 个区间。

    Args:
        state0: ``initialize_state`` 返回的状态 dict（不被修改）。
        controls: [n,steps,F] 或 None。
        future: [n,F,R] 真实未来（仅 D1 使用）。
    Returns:
        pred [n,steps,F]；stats {key: [n,steps]}
    """
    state = state0
    preds, stats = [], {k: [] for k in STATE_KEYS}
    for k in range(steps):
        u = controls[:, k, :].unsqueeze(-1) if controls is not None else None
        step_stats = _step_state_stats(model, state, u)
        z_old = state['z']
        x_old = model.readout_state(state)
        state = model.advance_state(state, control=u)
        x_new = model.readout_state(state)
        preds.append(x_new)
        step_stats['latent_step_norm'] = (state['z'] - z_old).norm(
            dim=-1).mean(dim=1)
        step_stats['state_step_abs'] = (x_new - x_old).abs().mean(dim=1)
        for key in STATE_KEYS:
            stats[key].append(step_stats[key])
        if refresh == 'observed':
            if future is None:
                raise ValueError('D1 需要真实 future 观测')
            state = model.update_state(state, future[:, k, :])
        elif refresh == 'self':
            state = model.update_state(state, x_new)
        elif refresh is not None:
            raise ValueError(f'未知 refresh={refresh}')
    return (torch.stack(preds, dim=1),
            {k: torch.stack(v, dim=1) for k, v in stats.items()})


def per_anchor_metrics(pred, target, mask, x_last, deadzone=0.05):
    """H1..H 的 MAE/Delta MAE/Delta PCC/Delta Std Ratio/方向准确率。

    pred/target: [N,R,F]；mask: [N,R]；x_last: [N,F]。
    一阶差分与训练 ``temporal_difference_huber`` 同口径：h=0 相对 x_t，
    之后相对上一帧预测/真值。返回 {metric: [N,H]}。
    """
    n, r, _ = pred.shape
    out = {k: np.full((n, r), np.nan, dtype=np.float64) for k in METRICS}
    for h in range(r):
        cur_ok = mask[:, h] > 0.5
        prev_ok = np.ones(n, dtype=bool) if h == 0 else mask[:, h - 1] > 0.5
        ok = cur_ok & prev_ok
        if not ok.any():
            continue
        p = pred[:, h, :].astype(np.float64)
        t = target[:, h, :].astype(np.float64)
        pp = x_last.astype(np.float64) if h == 0 else pred[:, h - 1, :].astype(np.float64)
        pt = x_last.astype(np.float64) if h == 0 else target[:, h - 1, :].astype(np.float64)
        dp, dt = p - pp, t - pt

        mae = np.abs(p - t).mean(axis=1)
        dmae = np.abs(dp - dt).mean(axis=1)
        out['mae'][ok, h] = mae[ok]
        out['delta_mae'][ok, h] = dmae[ok]

        sd = dt.std(axis=1)
        ratio = dp.std(axis=1) / (sd + 1e-8)
        ratio[sd <= 1e-8] = np.nan
        out['delta_std_ratio'][ok, h] = ratio[ok]

        dp_c = dp - dp.mean(axis=1, keepdims=True)
        dt_c = dt - dt.mean(axis=1, keepdims=True)
        den = np.sqrt((dp_c ** 2).sum(axis=1) * (dt_c ** 2).sum(axis=1))
        pcc = (dp_c * dt_c).sum(axis=1) / (den + 1e-8)
        pcc[den <= 1e-8] = np.nan
        out['delta_pcc'][ok, h] = pcc[ok]

        active = np.abs(dt) > deadzone
        agree = (np.sign(dp) == np.sign(dt)) & active
        cnt = active.sum(axis=1)
        acc = agree.sum(axis=1) / np.maximum(cnt, 1)
        acc[cnt == 0] = np.nan
        out['direction_accuracy'][ok, h] = acc[ok]
    return out


def temporal_variance_ratio(pred, target, mask, horizon):
    """未来轨迹逐 ROI 时间方差比再对 ROI 平均（仅完整有效轨迹）。"""
    n = pred.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    ok = mask[:, :horizon].all(axis=1)
    if not ok.any():
        return out
    p = pred[ok, :horizon, :].astype(np.float64)
    t = target[ok, :horizon, :].astype(np.float64)
    vp = p.var(axis=1)
    vt = t.var(axis=1)
    ratio = vp / (vt + 1e-8)
    ratio[vt <= 1e-8] = np.nan
    out[ok] = np.nanmean(ratio, axis=1)
    return out


def subject_table(subj_ids, values):
    """[N] / [N,H] → 被试内均值；返回 (subject_order, [S] / [S,H])。"""
    vals = np.asarray(values, dtype=np.float64)
    order = sorted(set(str(s) for s in subj_ids))
    index = {s: i for i, s in enumerate(order)}
    idx = np.array([index[str(s)] for s in subj_ids])
    if vals.ndim == 1:
        out = np.full(len(order), np.nan)
        for i in range(len(order)):
            sub = vals[idx == i]
            ok = np.isfinite(sub)
            if ok.any():
                out[i] = sub[ok].mean()
        return order, out
    out = np.full((len(order), vals.shape[1]), np.nan)
    for i in range(len(order)):
        sub = vals[idx == i]
        ok = np.isfinite(sub)
        cnt = ok.sum(axis=0)
        total = np.where(ok, sub, 0.0).sum(axis=0)
        out[i] = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    return order, out


def summary_stats(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'mean': None, 'std': None, 'sem': None, 'median': None, 'n': 0}
    return {
        'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if v.size > 1 else 0.0,
        'sem': float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
        'median': float(np.median(v)), 'n': int(v.size),
    }


def paired_summary(free, other, lower_is_better=True):
    """逐 horizon 配对差（other - free）与 t 检验。"""
    out = []
    for h in range(free.shape[1]):
        a, b = free[:, h], other[:, h]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 2:
            out.append({'horizon': h + 1, 'n': int(ok.sum())})
            continue
        a, b = a[ok], b[ok]
        diff = b - a
        improve = diff < 0 if lower_is_better else diff > 0
        if np.allclose(diff, 0.0):
            t, p = 0.0, 1.0
        else:
            t, p = stats.ttest_rel(b, a)
        sem = diff.std(ddof=1) / np.sqrt(len(diff))
        out.append({
            'horizon': h + 1, 'n': int(ok.sum()),
            'free_mean': float(a.mean()), 'other_mean': float(b.mean()),
            'delta_mean': float(diff.mean()),
            'delta_ci95': [float(diff.mean() - 1.96 * sem),
                           float(diff.mean() + 1.96 * sem)],
            't': float(t) if np.isfinite(t) else None,
            'p': float(p) if np.isfinite(p) else None,
            'improved_fraction': float(improve.mean()),
        })
    return out


def collect(model, loader, device, steps, deadzone, max_batches=0):
    rows = {m: {'pred': [], 'target': [], 'mask': [], 'x_last': [],
                'subj': [], 'cutoff': [], 'stats': {k: [] for k in STATE_KEYS}}
            for m in MODES}
    for bi, batch in enumerate(tqdm(loader, desc='[Refresh rollout]', leave=False)):
        if max_batches and bi >= max_batches:
            break
        rf = batch['rollout_flag'].bool()
        if not rf.any():
            continue
        rf_gpu = rf.to(device)
        x = batch['x'].to(device, non_blocking=True)
        sc = batch['sc'].to(device, non_blocking=True)
        pathology = batch.get('pathology_score', None)
        if pathology is not None:
            pathology = pathology.to(device, non_blocking=True)
        cond = _batch_condition(batch, device)
        bcal, bctl = dynamics_extras(batch, device)

        hist = x[rf_gpu][:, :, 0, :].transpose(1, 2).contiguous()
        future = batch['future'][rf].transpose(1, 2).contiguous().to(
            device, non_blocking=True)                              # [n,R,F]
        fmask = batch['future_mask'][rf].to(device, non_blocking=True)
        steps_i = int(min(steps, future.shape[2]))
        controls = (bctl[rf_gpu][:, :steps_i] if bctl is not None else None)

        state0 = model.initialize_state(
            hist, sc[rf_gpu],
            pathology_score=(pathology[rf_gpu] if pathology is not None else None),
            condition=(_index_condition(cond, rf_gpu) if cond else None),
            calibration_bold=(bcal[rf_gpu] if bcal is not None else None))

        target = future[:, :steps_i, :]
        mask = fmask[:, :steps_i]
        x_last = batch['x_last'][rf].to(device, non_blocking=True)
        subj = [str(s) for s, keep in zip(batch['subj_id'], rf.tolist()) if keep]
        cutoff = batch['cutoff'][rf].numpy()
        for mode in MODES:
            pred, st = rollout_mode(model, state0, steps_i, controls,
                                    REFRESH[mode], future)
            rows[mode]['pred'].append(pred.float().cpu().numpy())
            rows[mode]['target'].append(target.float().cpu().numpy())
            rows[mode]['mask'].append(mask.float().cpu().numpy())
            rows[mode]['x_last'].append(x_last.float().cpu().numpy())
            rows[mode]['subj'].extend(subj)
            rows[mode]['cutoff'].append(cutoff)
            for key in STATE_KEYS:
                rows[mode]['stats'][key].append(st[key].float().cpu().numpy())

    out = {}
    for mode in MODES:
        r = rows[mode]
        out[mode] = {
            'pred': np.concatenate(r['pred'], axis=0),
            'target': np.concatenate(r['target'], axis=0),
            'mask': np.concatenate(r['mask'], axis=0),
            'x_last': np.concatenate(r['x_last'], axis=0),
            'subj_ids': np.array(r['subj'], dtype=object),
            'cutoff': np.concatenate(r['cutoff'], axis=0),
            'stats': {k: np.concatenate(v, axis=0) for k, v in r['stats'].items()},
        }
    return out


def analyse(data, steps, deadzone, state_stats=True):
    report = {'protocol': {'steps': int(steps),
                           'direction_deadzone': float(deadzone)},
              'horizons': list(range(1, steps + 1)),
              'modes': {}, 'paired': {}, 'state': {}}
    subject_series = {}
    subject_orders = {}
    for mode in MODES:
        d = data[mode]
        m = per_anchor_metrics(d['pred'], d['target'], d['mask'], d['x_last'],
                               deadzone=deadzone)
        order, subj_metrics = {}, {}
        for key, arr in m.items():
            order, subj_metrics[key] = subject_table(d['subj_ids'], arr)
        tvr = temporal_variance_ratio(d['pred'], d['target'], d['mask'], steps)
        _, subj_tvr = subject_table(d['subj_ids'], tvr)
        subject_orders[mode] = order
        report['modes'][mode] = {
            **{key: {'per_horizon': [summary_stats(subj_metrics[key][:, h])
                                     for h in range(steps)]}
               for key in METRICS},
            'temporal_variance_ratio': summary_stats(subj_tvr),
            'n_subjects': len(order),
            'n_anchors': int(d['pred'].shape[0]),
        }
        subject_series[mode] = subj_metrics
        if state_stats:
            st = d['stats']
            report['state'][mode] = {}
            for key in STATE_KEYS:
                _, subj_st = subject_table(d['subj_ids'], st[key])
                report['state'][mode][key] = [
                    summary_stats(subj_st[:, h]) for h in range(steps)]
    for mode in ('d1', 'd2'):
        report['paired'][f'{mode}_vs_free'] = {}
        for key, lower in (('mae', True), ('delta_mae', True),
                           ('delta_pcc', False), ('direction_accuracy', False)):
            report['paired'][f'{mode}_vs_free'][key] = paired_summary(
                subject_series['free'][key], subject_series[mode][key],
                lower_is_better=lower)
        free_dev = np.abs(subject_series['free']['delta_std_ratio'] - 1.0)
        other_dev = np.abs(subject_series[mode]['delta_std_ratio'] - 1.0)
        report['paired'][f'{mode}_vs_free']['delta_std_dev'] = paired_summary(
            free_dev, other_dev, lower_is_better=True)
    return report, subject_series, subject_orders


def write_csv(path, subject_series, subject_orders):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'subject_id', 'horizon', 'mae', 'delta_mae',
                    'delta_pcc', 'delta_std_ratio', 'direction_accuracy'])
        for mode in MODES:
            series = subject_series[mode]
            order = subject_orders[mode]
            n_subj, n_h = series['mae'].shape
            for s in range(n_subj):
                for h in range(n_h):
                    w.writerow([
                        mode, order[s], h + 1,
                        *[('' if not np.isfinite(series[k][s, h])
                           else f'{series[k][s, h]:.8f}') for k in METRICS],
                    ])


def plot_diagnostics(report, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h = np.arange(1, len(report['horizons']) + 1)
    colors = {'free': '#4C72B0', 'd1': '#DD8452', 'd2': '#55A868'}
    labels = {'free': 'Free rollout', 'd1': 'D1 observed refresh',
              'd2': 'D2 self refresh'}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    specs = [
        ('mae', 'H MAE (lower better)', axes[0, 0], None),
        ('delta_pcc', 'Delta spatial PCC (higher better)', axes[0, 1], None),
        ('delta_std_ratio', 'Delta std ratio (target 1)', axes[0, 2], 1.0),
    ]
    for key, title, ax, ref in specs:
        for mode in MODES:
            vals = [np.nan if x['mean'] is None else x['mean']
                    for x in report['modes'][mode][key]['per_horizon']]
            ax.plot(h, vals, marker='o', ms=3, color=colors[mode],
                    label=labels[mode])
        if ref is not None:
            ax.axhline(ref, color='gray', ls='--', lw=1)
        ax.set_title(title)
        ax.set_xlabel('horizon')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    state_specs = [('diffusion_norm', '||diffusion|| per ROI', axes[1, 0]),
                   ('residual_norm', '||neural residual|| per ROI', axes[1, 1]),
                   ('kappa', 'kappa', axes[1, 2])]
    for key, title, ax in state_specs:
        for mode in MODES:
            vals = [np.nan if x['mean'] is None else x['mean']
                    for x in report['state'][mode][key]]
            ax.plot(h, vals, marker='o', ms=3, color=colors[mode],
                    label=labels[mode])
        ax.set_title(title)
        ax.set_xlabel('horizon')
        ax.grid(alpha=0.3)
    axes[1, 0].set_yscale('log')
    axes[1, 1].set_yscale('log')
    fig.suptitle('Graph-ODE rollout state refresh diagnostics')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_latent_state(report, out_path):
    """latent 结构随 horizon 的变化（方差塌缩 / ROI 同质化 / 步长）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h = np.arange(1, len(report['horizons']) + 1)
    colors = {'free': '#4C72B0', 'd1': '#DD8452', 'd2': '#55A868'}
    labels = {'free': 'Free rollout', 'd1': 'D1 observed refresh',
              'd2': 'D2 self refresh'}
    specs = [('z_roi_variance', 'ROI latent variance'),
             ('z_pairwise_cos', 'mean pairwise ROI cosine'),
             ('latent_step_norm', '||z step|| per ROI')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for (key, title), ax in zip(specs, axes):
        for mode in MODES:
            vals = [np.nan if x['mean'] is None else x['mean']
                    for x in report['state'][mode][key]]
            ax.plot(h, vals, marker='o', ms=3, color=colors[mode],
                    label=labels[mode])
        ax.set_title(title)
        ax.set_xlabel('horizon')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Graph-ODE latent state under free / observed / self refresh')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(ckpt, out_dir, split='test', steps=8, device=None, seed=2024,
        deadzone=0.05, max_batches=0):
    set_seed(seed)
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = load_train_args(ckpt)
    bundle = load_condition_bundle(ckpt)
    model = build_model_from_args(args, device, condition_bundle=bundle)
    model, compat = load_checkpoint_compat(model, ckpt, device, strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')
    model.eval()
    if getattr(model, 'dynamics_mode', None) != 'graph_ode':
        raise ValueError('refresh_rollout 仅支持 dynamics_mode=graph_ode')
    if not hasattr(model, 'update_state'):
        raise ValueError('模型缺少 update_state 接口')

    loader, subjects, _ = npe.build_next_point_loader(
        args, split, condition_bundle=bundle)
    log.info('[refresh] split=%s | 被试数=%d | 任务数=%d | steps=%d',
             split, len(subjects), len(loader.dataset), steps)
    data = collect(model, loader, device, steps, deadzone, max_batches=max_batches)
    report, subject_series, subject_orders = analyse(data, steps, deadzone)
    report['protocol'].update({
        'checkpoint': str(ckpt), 'split': split, 'seed': int(seed),
        'device': str(device), 'n_subjects': len(set(map(str, data['free']['subj_ids']))),
        'n_anchors': int(data['free']['pred'].shape[0]),
    })
    (out_dir / 'refresh_rollout_metrics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(out_dir / 'refresh_rollout_subject.csv', subject_series,
              subject_orders)
    np.savez_compressed(
        out_dir / 'refresh_rollout_arrays.npz',
        **{f'{m}_{k}': data[m][k]
           for m in MODES for k in ('pred', 'target', 'mask', 'x_last',
                                    'subj_ids', 'cutoff')},
        **{f'{m}_stat_{k}': data[m]['stats'][k]
           for m in MODES for k in STATE_KEYS})
    plot_diagnostics(report, out_dir / 'fig5_refresh_rollout.png')
    plot_latent_state(report, out_dir / 'fig6_latent_state.png')
    log.info('[refresh] 输出完成：%s', out_dir)
    return report


def main():
    ap = argparse.ArgumentParser(description='D1/D2 状态刷新 rollout 诊断')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='test', choices=('test', 'val'))
    ap.add_argument('--steps', type=int, default=8, help='rollout 步数（H1..H）')
    ap.add_argument('--device', default='')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--direction_deadzone', type=float, default=0.05)
    ap.add_argument('--max_batches', type=int, default=0,
                    help='仅处理前 N 个 batch（0=全部；调试用）')
    ns = ap.parse_args()
    run(ns.ckpt, ns.out_dir, split=ns.split, steps=ns.steps,
        device=(torch.device(ns.device) if ns.device else None),
        seed=ns.seed, deadzone=ns.direction_deadzone,
        max_batches=ns.max_batches)


if __name__ == '__main__':
    main()
