# coding=utf-8
"""第一阶段（Temporal Difference / Cross-Head Consistency）三臂对比与作图。

输入是 ``experiments.evaluate_variant`` 对 E0/E1/E2 的评估目录（每个目录含
``metrics_<split>.json``、``per_subject_metrics_<split>.csv``、
``per_task_metrics_<split>.csv``，建议同时带 ``--save_arrays`` 的
``predictions_<split>.npz``）。脚本不加载模型、不重新推理。

统计口径：
  - 主口径是**被试级**：先把每个被试的 anchor/horizon 指标做被试内平均，
    再跨被试报告 mean/std；同一被试在三臂中使用完全相同的 test 锚点。
  - 配对比较按 subject_id 对齐，报告 E1−E0、E2−E1、E2−E0 的配对均值差、
    95% CI、配对 t 检验 p 值与“改善被试比例”。
  - 误差类指标（MAE/RMSE/Delta MAE/Gap/both-fail）越低越好；PCC/DA 越高越好。

用法::

    python -m experiments.compare_phase1 \
        --arms E0=results/phase1/eval_E0,E1=results/phase1/eval_E1,E2=results/phase1/eval_E2 \
        --split test --out_dir results/phase1/compare
"""
import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

ARM_COLORS = {'E0': '#4C72B0', 'E1': '#DD8452', 'E2': '#55A868',
              'E3': '#8172B3', 'E4': '#937860'}
HORIZONS = tuple(range(1, 9))

# (内部名, 展示名, 越低越好)
METRICS = (
    ('MAE', 't+1 MAE', True),
    ('RMSE', 't+1 RMSE', True),
    ('PCC_spatial', 't+1 spatial PCC', False),
    ('ODE_MAE', 'ODE MAE (H1-8)', True),
    ('ODE_delta_MAE_step', 'ODE Delta MAE (H1-8)', True),
    ('ODE_delta_MAE_step_H2_8', 'ODE Delta MAE (H2-8)', True),
    ('ODE_DA_step', 'ODE Direction Acc. (H1-8)', False),
    ('ODE_DA_step_H2_8', 'ODE Direction Acc. (H2-8)', False),
    ('ODE_MAE_H8', 'ODE MAE (H8)', True),
    ('CPM_MAE', 'CPM MAE (H1-8)', True),
    ('CPM_delta_MAE_step', 'CPM Delta MAE (H1-8)', True),
    ('CPM_delta_MAE_step_H2_8', 'CPM Delta MAE (H2-8)', True),
    ('CPM_DA_step', 'CPM Direction Acc. (H1-8)', False),
    ('CPM_DA_step_H2_8', 'CPM Direction Acc. (H2-8)', False),
    ('CPM_MAE_H8', 'CPM MAE (H8)', True),
    ('GAP', 'CPM-ODE Gap (H1-8)', True),
    ('GAP_H8', 'CPM-ODE Gap (H8)', True),
    ('BOTH_FAIL_H8', 'Both fail vs persistence (H8)', True),
    ('GAP_norm', 'Normalized Gap (H1-8)', True),
    ('BOTH_FAIL', 'Both fail vs persistence', True),
    ('ERR_CORR', 'Head error correlation', True),
    ('ORACLE_MAE', 'Oracle MAE (per anchor)', True),
    ('CPM_WIN', 'CPM wins per anchor', False),
    # 幅度诊断：ratio 目标为 1；DEV 越接近 0 越好
    ('ODE_DSTD_H1', 'ODE Delta std ratio H1 (target 1)', False),
    ('ODE_DSTD_H2_8', 'ODE Delta std ratio H2-8 (target 1)', False),
    ('ODE_DSTD_DEV', 'ODE |Delta std ratio-1| H2-8', True),
    ('ODE_DPCC_H1', 'ODE Delta PCC H1', False),
    ('ODE_DPCC_H2_8', 'ODE Delta PCC H2-8', False),
    ('ODE_TVAR', 'ODE temporal variance ratio (target 1)', False),
    ('ODE_TVAR_DEV', 'ODE |temporal variance ratio-1|', True),
    ('ODE_AMP_Q1_MAE', 'ODE Delta MAE Q1 (small changes)', True),
    ('ODE_AMP_Q5_MAE', 'ODE Delta MAE Q5 (large changes)', True),
    ('CPM_DSTD_H1', 'CPM Delta std ratio H1 (target 1)', False),
    ('CPM_DSTD_H2_8', 'CPM Delta std ratio H2-8 (target 1)', False),
    ('CPM_DSTD_DEV', 'CPM |Delta std ratio-1| H2-8', True),
    ('CPM_DPCC_H1', 'CPM Delta PCC H1', False),
    ('CPM_DPCC_H2_8', 'CPM Delta PCC H2-8', False),
    ('CPM_TVAR', 'CPM temporal variance ratio (target 1)', False),
    ('CPM_AMP_Q5_MAE', 'CPM Delta MAE Q5 (large changes)', True),
)
LOWER_IS_BETTER = {name: lower for name, _, lower in METRICS}


def _require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'缺少评估产物：{path}')
    return path


def _horizon_columns(df, head, metric, horizons=HORIZONS):
    return [f'{head}_H{h}_{metric}' for h in horizons
            if f'{head}_H{h}_{metric}' in df.columns]


def _horizon_mean(df, head, metric, horizons=HORIZONS):
    cols = _horizon_columns(df, head, metric, horizons=horizons)
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[cols].mean(axis=1, skipna=True)


def load_arm(name, eval_dir, split):
    """读取单臂评估产物，构造被试级指标表。"""
    eval_dir = Path(eval_dir)
    metrics = json.loads(_require(eval_dir / f'metrics_{split}.json').read_text(
        encoding='utf-8'))
    subj = pd.read_csv(_require(eval_dir / f'per_subject_metrics_{split}.csv'))
    task = pd.read_csv(_require(eval_dir / f'per_task_metrics_{split}.csv'))
    if 'subject_id' not in subj.columns or 't+1_MAE' not in subj.columns:
        raise ValueError(f'{name}: per_subject CSV 缺少 subject_id/t+1_MAE')

    out = pd.DataFrame({'subject_id': subj['subject_id'].astype(str)})
    out['MAE'] = subj['t+1_MAE'].to_numpy()
    out['RMSE'] = subj['t+1_RMSE'].to_numpy()
    out['PCC_spatial'] = subj['t+1_PCC_spatial'].to_numpy()

    # 逐 horizon 先在被试内平均（CSV 已是逐 horizon 的被试均值），再跨 H 平均
    for metric, key in (('mae', 'MAE'), ('delta_mae_step', 'delta_MAE_step'),
                        ('direction_accuracy_step', 'DA_step'),
                        ('direction_accuracy_anchor', 'DA_anchor')):
        out[f'ODE_{key}'] = _horizon_mean(subj, 'ODE', metric).to_numpy()
        out[f'CPM_{key}'] = _horizon_mean(subj, 'CPM', metric).to_numpy()
    # H2-8：去掉与单步 MAE 定义重合的 H1，单独衡量局部动力学
    for head in ('ODE', 'CPM'):
        out[f'{head}_delta_MAE_step_H2_8'] = _horizon_mean(
            subj, head, 'delta_mae_step', horizons=range(2, 9)).to_numpy()
        out[f'{head}_DA_step_H2_8'] = _horizon_mean(
            subj, head, 'direction_accuracy_step', horizons=range(2, 9)).to_numpy()
        h8 = f'{head}_H8_mae'
        out[f'{head}_MAE_H8'] = (subj[h8].to_numpy() if h8 in subj.columns
                                 else np.full(len(subj), np.nan))
    h8_gap = 'DUAL_H8_gap'
    h8_fail = 'DUAL_H8_both_fail_vs_persistence'
    out['GAP_H8'] = (subj[h8_gap].to_numpy() if h8_gap in subj.columns
                     else np.full(len(subj), np.nan))
    out['BOTH_FAIL_H8'] = (subj[h8_fail].to_numpy() if h8_fail in subj.columns
                           else np.full(len(subj), np.nan))
    for metric, key in (('gap', 'GAP'), ('gap_norm', 'GAP_norm'),
                        ('both_fail_vs_persistence', 'BOTH_FAIL'),
                        ('head_error_corr', 'ERR_CORR'),
                        ('oracle_mae', 'ORACLE_MAE'),
                        ('cpm_win_rate', 'CPM_WIN'),
                        ('direction_agreement', 'DIR_AGREE')):
        out[key] = _horizon_mean(subj, 'DUAL', metric).to_numpy()

    # 保留逐 horizon 原始列，供 Figure 3 的 H1-H8 曲线使用
    horizon_cols = [c for c in subj.columns
                    if c.startswith(('ODE_H', 'CPM_H', 'DUAL_H'))]
    if horizon_cols:
        out = pd.concat([out, subj[horizon_cols].reset_index(drop=True)], axis=1)

    # 幅度诊断（由 evaluate_variant 额外落盘；不存在时跳过，保持向后兼容）
    diag_path = eval_dir / f'phase1_diagnostics_subject_{split}.csv'
    if diag_path.is_file():
        diag = pd.read_csv(diag_path)
        if 'subject_id' not in diag.columns:
            raise ValueError(f'{name}: 诊断 CSV 缺少 subject_id')
        out = out.set_index('subject_id')
        diag = diag.drop(columns=['n_tasks'], errors='ignore').set_index('subject_id')
        missing = out.index.difference(diag.index)
        if len(missing):
            raise ValueError(f'{name}: 诊断 CSV 缺少被试 {list(missing)[:5]}')
        out = out.join(diag, how='left').reset_index()

    # “接近 1”类指标：同时保留原始比值和 |ratio−1|
    for head in ('ODE', 'CPM'):
        aliases = {
            f'{head}_DSTD_H1': f'{head}_delta_std_ratio_h1',
            f'{head}_DPCC_H1': f'{head}_delta_pcc_h1',
            f'{head}_DPCC_H2_8': f'{head}_delta_pcc_h2_8',
            f'{head}_AMP_Q1_MAE': f'{head}_Q1_MAE',
            f'{head}_AMP_Q5_MAE': f'{head}_Q5_MAE',
        }
        for dst, src in aliases.items():
            if src in out.columns:
                out[dst] = out[src]
        for raw, ratio_key, dev_key in (
                (f'{head}_delta_std_ratio_h2_8', f'{head}_DSTD_H2_8', f'{head}_DSTD_DEV'),
                (f'{head}_temporal_variance_ratio', f'{head}_TVAR', f'{head}_TVAR_DEV')):
            if raw in out.columns:
                out[ratio_key] = out[raw]
                out[dev_key] = np.abs(out[raw] - 1.0)

    return {
        'name': name, 'dir': eval_dir, 'metrics': metrics, 'subject': out,
        'task': task, 'subjects_raw': subj,
    }


def _summary(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'sem': float('nan'),
                'median': float('nan'), 'n': 0}
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {'mean': float(v.mean()), 'std': std,
            'sem': std / np.sqrt(v.size) if v.size > 0 else float('nan'),
            'median': float(np.median(v)), 'n': int(v.size)}


def _paired_test(a, b, lower_is_better):
    """a/b 为同被试的两个指标；返回配对差值（按“改善”方向）与检验结果。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return None
    raw_diff = a - b
    improvement = -raw_diff if lower_is_better else raw_diff
    mean_imp = float(improvement.mean())
    std = float(improvement.std(ddof=1))
    sem = std / np.sqrt(improvement.size)
    ci = (mean_imp - 1.96 * sem, mean_imp + 1.96 * sem)
    try:
        from scipy import stats
        tstat, pvalue = stats.ttest_rel(a, b)
    except Exception:  # pragma: no cover
        tstat, pvalue = float('nan'), float('nan')
    return {
        'n_subjects': int(improvement.size),
        'mean_improvement': mean_imp,
        'ci95_low': float(ci[0]), 'ci95_high': float(ci[1]),
        'paired_t': float(tstat), 'p_value': float(pvalue),
        'win_rate': float((improvement > 0).mean()),
        'tie_rate': float((np.abs(improvement) <= 1e-12).mean()),
        'raw_mean_a_minus_b': float(raw_diff.mean()),
    }


def _task_keys(task):
    return (task['subj_id'].astype(str) + '|'
            + task['cutoff'].astype(int).astype(str)).to_numpy()


def validate_alignment(arms, strict=True):
    """核对三臂使用同一批 test 被试与锚点。"""
    names = list(arms)
    ref = arms[names[0]]
    ref_subj = set(ref['subject']['subject_id'])
    ref_tasks = set(_task_keys(ref['task']))
    problems = []
    for name in names[1:]:
        arm = arms[name]
        subj = set(arm['subject']['subject_id'])
        tasks = set(_task_keys(arm['task']))
        if subj != ref_subj:
            problems.append(f'{name}: 被试集合与 {names[0]} 不一致 '
                            f'(only_{name}={len(subj - ref_subj)}, '
                            f'only_{names[0]}={len(ref_subj - subj)})')
        if tasks != ref_tasks:
            problems.append(f'{name}: 锚点集合与 {names[0]} 不一致 '
                            f'(only_{name}={len(tasks - ref_tasks)}, '
                            f'only_{names[0]}={len(ref_tasks - tasks)})')
    if problems:
        msg = '评估口径不一致：' + '；'.join(problems)
        if strict:
            raise ValueError(msg + '。三臂必须使用相同 test 被试与预测锚点。')
        log.warning(msg)
    return problems


def build_comparison(arms, strict=True):
    validate_alignment(arms, strict=strict)
    names = list(arms)
    # 按 subject_id 对齐
    table = None
    for name in names:
        df = arms[name]['subject'].set_index('subject_id')
        df = df.add_prefix(f'{name}_')
        table = df if table is None else table.join(df, how='inner')
    table = table.reset_index()

    summaries = {}
    for name in names:
        summaries[name] = {}
        for metric, _, _ in METRICS:
            col = f'{name}_{metric}'
            if col in table.columns:
                summaries[name][metric] = _summary(table[col])
    pairs = {}
    for a, b in combinations(names, 2):
        pairs[f'{b}-{a}'] = {}
        for metric, _, lower in METRICS:
            ca, cb = f'{a}_{metric}', f'{b}_{metric}'
            if ca not in table.columns or cb not in table.columns:
                continue
            res = _paired_test(table[cb].to_numpy(), table[ca].to_numpy(),
                               lower_is_better=lower)
            if res is not None:
                pairs[f'{b}-{a}'][metric] = res
    return table, summaries, pairs


def _bar_panel(ax, summaries, names, metric, title, ylabel):
    vals = [summaries[n].get(metric, {}).get('mean', np.nan) for n in names]
    errs = [summaries[n].get(metric, {}).get('sem', np.nan) for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, yerr=errs, capsize=4, width=0.62,
                  color=[ARM_COLORS.get(n, '#777777') for n in names],
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)


def _hcurve_panel(ax, table, names, head, metric, title, ylabel):
    for name in names:
        df = table
        means, sems, hs = [], [], []
        for h in HORIZONS:
            col = f'{name}_{head}_H{h}_{metric}'
            if col not in df.columns:
                continue
            v = df[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            hs.append(h)
            means.append(float(v.mean()))
            sems.append(float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)
        if not hs:
            continue
        ax.errorbar(hs, means, yerr=sems, marker='o', ms=4, capsize=3,
                    color=ARM_COLORS.get(name, '#777777'), label=name)
    ax.set_xticks(list(HORIZONS))
    ax.set_xlabel('Forecast horizon (TR)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=8)


def _pick_representative(npz_path, horizon):
    """在 E0 预测数组上选一个确定性的中等难度 anchor 与 ROI。"""
    try:
        arr = np.load(npz_path, allow_pickle=False)
        ode = np.asarray(arr['ode_pred'], dtype=float)
        future = np.asarray(arr['future'], dtype=float)
        fmask = np.asarray(arr['future_mask'], dtype=float)
        if ode.size == 0 or future.size == 0:
            return None
    except Exception as exc:
        log.warning('代表曲线跳过：无法读取 %s（%s）', npz_path, exc)
        return None
    h = min(int(horizon), ode.shape[2], future.shape[2])
    valid = fmask[:, :h].astype(bool).all(axis=1)
    if not valid.any():
        return None
    err = np.abs(ode[:, :, :h] - future[:, :, :h]).mean(axis=(1, 2))
    cand = np.where(valid)[0]
    cand = cand[np.isfinite(err[cand])]
    if cand.size == 0:
        return None
    idx = int(cand[np.argsort(err[cand])[len(cand) // 2]])
    roi_err = np.abs(ode[idx, :, :h] - future[idx, :, :h]).mean(axis=1)
    roi = int(np.argsort(roi_err)[len(roi_err) // 2])
    return {'idx': idx, 'roi': roi, 'horizon': h,
            'subject_id': str(arr['subj_ids'][idx]),
            'cutoff': int(arr['cutoff'][idx]),
            'anchor_error_h8': float(err[idx])}


def _load_curve(npz_path, subject_id, cutoff, roi, horizon):
    try:
        arr = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None
    subj = np.asarray(arr['subj_ids']).astype(str)
    cut = np.asarray(arr['cutoff']).astype(int)
    hits = np.where((subj == str(subject_id)) & (cut == int(cutoff)))[0]
    if hits.size == 0:
        return None
    i = int(hits[0])
    h = min(int(horizon), np.asarray(arr['future']).shape[2])
    out = {
        'x_last': float(np.asarray(arr['x_last'])[i, roi]),
        'future': np.asarray(arr['future'])[i, roi, :h].astype(float).tolist(),
    }
    for key in ('ode_pred', 'cpm_pred'):
        v = np.asarray(arr[key]) if key in arr.files else np.array([])
        out[key] = (v[i, roi, :min(h, v.shape[2])].astype(float).tolist()
                    if v.size else None)
    return out


def plot_figures(arms, tables, summaries, out_dir, split):
    if plt is None:
        log.warning('matplotlib 不可用：跳过作图')
        return
    names = list(arms)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 点值预测精度
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _bar_panel(axes[0], summaries, names, 'MAE', 't+1 MAE', 'MAE')
    _bar_panel(axes[1], summaries, names, 'RMSE', 't+1 RMSE', 'RMSE')
    _bar_panel(axes[2], summaries, names, 'PCC_spatial',
               't+1 spatial PCC', 'Pearson r (ROI)')
    fig.suptitle(f'Figure 1 | Point prediction accuracy ({split}, subject-level mean ± SEM)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / 'fig1_accuracy.png', dpi=180)
    plt.close(fig)

    # Figure 2: 动力学指标 + 代表 ROI 曲线
    fig = plt.figure(figsize=(13, 4.2))
    ax0 = fig.add_subplot(1, 3, 1)
    _bar_panel(ax0, summaries, names, 'ODE_delta_MAE_step',
               'ODE Delta MAE (adjacent-frame)', 'MAE')
    ax1 = fig.add_subplot(1, 3, 2)
    _bar_panel(ax1, summaries, names, 'ODE_DA_step',
               'ODE Direction Accuracy', 'accuracy')
    ax1.set_ylim(0, 1.02)
    ax2 = fig.add_subplot(1, 3, 3)
    rep = _pick_representative(Path(arms[names[0]]['dir'])
                               / f'predictions_{split}.npz', horizon=8)
    if rep is None:
        ax2.text(0.5, 0.5, 'predictions NPZ unavailable', ha='center', va='center')
        ax2.set_axis_off()
    else:
        hs = np.arange(1, rep['horizon'] + 1)
        curve0 = _load_curve(Path(arms[names[0]]['dir'])
                             / f'predictions_{split}.npz',
                             rep['subject_id'], rep['cutoff'], rep['roi'],
                             rep['horizon'])
        if curve0 is not None:
            ax2.plot(np.r_[0, hs], np.r_[curve0['x_last'], curve0['future']],
                     color='black', marker='s', ms=3, lw=1.4, label='truth')
        for name in names:
            c = _load_curve(Path(arms[name]['dir']) / f'predictions_{split}.npz',
                            rep['subject_id'], rep['cutoff'], rep['roi'],
                            rep['horizon'])
            if c is None or c.get('ode_pred') is None:
                continue
            ax2.plot(hs, c['ode_pred'], marker='o', ms=3, lw=1.2,
                     color=ARM_COLORS.get(name, '#777777'), label=f'{name} ODE')
        ax2.set_xticks(np.r_[0, hs])
        ax2.set_xlabel('Step (0 = anchor x_t)', fontsize=9)
        ax2.set_ylabel('BOLD (z)', fontsize=9)
        ax2.set_title(f"Representative subject {rep['subject_id']}, ROI {rep['roi'] + 1}",
                      fontsize=9)
        ax2.grid(alpha=0.25, linestyle='--')
        ax2.legend(fontsize=7)
    fig.suptitle(f'Figure 2 | Temporal dynamics ({split}, subject-level mean ± SEM)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / 'fig2_dynamics.png', dpi=180)
    plt.close(fig)

    # Figure 3: 多步误差与双头一致性
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    _hcurve_panel(axes[0], tables, names, 'ODE', 'mae',
                  'Graph-ODE multi-step MAE', 'MAE')
    _hcurve_panel(axes[1], tables, names, 'CPM', 'mae',
                  'CPM multi-step MAE', 'MAE')
    _hcurve_panel(axes[2], tables, names, 'DUAL', 'gap',
                  'CPM-ODE prediction gap', 'L1 gap')
    # both-fail 作为虚线叠加在 gap 面板（同一 horizon 轴）
    for name in names:
        df = tables
        means, hs = [], []
        for h in HORIZONS:
            col = f'{name}_DUAL_H{h}_both_fail_vs_persistence'
            if col not in df.columns:
                continue
            v = df[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                hs.append(h); means.append(float(v.mean()))
        if hs:
            ax3 = axes[2]
            ax3.plot(hs, means, linestyle='--', marker='x', ms=3,
                     color=ARM_COLORS.get(name, '#777777'),
                     label=f'{name} both-fail')
    axes[2].legend(fontsize=7)
    fig.suptitle(f'Figure 3 | Multi-step error and dual-head consistency ({split})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / 'fig3_multistep_dual.png', dpi=180)
    plt.close(fig)

    plot_amplitude_diagnostics(arms, tables, summaries, out_dir, split)


def _col_mean(table, arm, col):
    key = f'{arm}_{col}'
    if key not in table.columns:
        return float('nan')
    v = table[key].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


def _baseline_diag(arms, name, key):
    """取第一条臂 metrics JSON 中的基线诊断均值（三臂基线完全相同）。"""
    arm = next(iter(arms.values()))
    blk = (arm['metrics'].get('phase1', {}).get('diagnostics', {})
           .get('blocks', {}).get(name, {}))
    return blk.get('overall', {}).get(key, {}).get('mean', float('nan'))


def plot_amplitude_diagnostics(arms, table, summaries, out_dir, split):
    """Figure 4：幅度比、Delta PCC、轨迹方差比、幅度分层 Delta MAE。"""
    if plt is None:
        return
    names = list(arms)
    del summaries  # 统一从逐被试宽表取均值，便于加入基线
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    # (a) ODE Delta std ratio：H1 vs H2-8
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.8 / max(1, len(names))
    for i, name in enumerate(names):
        vals = [_col_mean(table, name, 'ODE_delta_std_ratio_h1'),
                _col_mean(table, name, 'ODE_delta_std_ratio_h2_8')]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               color=ARM_COLORS.get(name, '#777777'), label=name)
    ax.axhline(1.0, color='black', ls='--', lw=1, label='target = 1')
    for name, ls in (('ar1', ':'), ('trend', '-.')):
        vals = [_baseline_diag(arms, name, 'delta_std_ratio_h1'),
                _baseline_diag(arms, name, 'delta_std_ratio_h2_8')]
        if np.isfinite(vals).any():
            ax.plot(x, vals, ls=ls, marker='x', color='black',
                    label=f'{name} reference')
    ax.set_xticks(x); ax.set_xticklabels(['H1', 'H2-8'])
    ax.set_ylabel('std(Δŷ) / std(Δy)')
    ax.set_title('(a) ODE amplitude ratio', fontsize=10)
    ax.grid(axis='y', alpha=0.25, ls='--')
    ax.legend(fontsize=7)

    # (b) ODE Delta PCC：H1 vs H2-8
    ax = axes[0, 1]
    for i, name in enumerate(names):
        vals = [_col_mean(table, name, 'ODE_delta_pcc_h1'),
                _col_mean(table, name, 'ODE_delta_pcc_h2_8')]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               color=ARM_COLORS.get(name, '#777777'), label=name)
    for name, ls in (('ar1', ':'), ('trend', '-.')):
        vals = [_baseline_diag(arms, name, 'delta_pcc_h1'),
                _baseline_diag(arms, name, 'delta_pcc_h2_8')]
        if np.isfinite(vals).any():
            ax.plot(x, vals, ls=ls, marker='x', color='black',
                    label=f'{name} reference')
    ax.set_xticks(x); ax.set_xticklabels(['H1', 'H2-8'])
    ax.set_ylabel('Corr(Δŷ, Δy)')
    ax.set_title('(b) ODE Delta PCC', fontsize=10)
    ax.grid(axis='y', alpha=0.25, ls='--')
    ax.legend(fontsize=7)

    # (c) 轨迹方差比（ODE/CPM），目标为 1
    ax = axes[1, 0]
    xg = np.arange(2)
    for i, name in enumerate(names):
        vals = [_col_mean(table, name, 'ODE_temporal_variance_ratio'),
                _col_mean(table, name, 'CPM_temporal_variance_ratio')]
        ax.bar(xg + i * width - 0.4 + width / 2, vals, width,
               color=ARM_COLORS.get(name, '#777777'), label=name)
    ax.axhline(1.0, color='black', ls='--', lw=1, label='target = 1')
    ax.set_xticks(xg); ax.set_xticklabels(['ODE', 'CPM'])
    ax.set_ylabel('Var(ŷ) / Var(y)')
    ax.set_title('(c) Temporal variance ratio', fontsize=10)
    ax.grid(axis='y', alpha=0.25, ls='--')
    ax.legend(fontsize=7)

    # (d) 幅度分层 Delta MAE：Q1 小变化 → Q5 大变化
    ax = axes[1, 1]
    qs = np.arange(1, 6)
    for name in names:
        vals = [_col_mean(table, name, f'ODE_Q{q}_MAE') for q in qs]
        ax.plot(qs, vals, marker='o', color=ARM_COLORS.get(name, '#777777'),
                label=name)
    base = (next(iter(arms.values()))['metrics'].get('phase1', {})
            .get('diagnostics', {}).get('blocks', {}).get('ar1', {}))
    ar1_vals = [base.get('amplitude_strata', {}).get('bins', {})
                .get(f'Q{q}', {}).get('MAE', float('nan')) for q in qs]
    if np.isfinite(ar1_vals).any():
        ax.plot(qs, ar1_vals, ls=':', marker='x', color='black',
                label='AR(1) reference')
    ax.set_xticks(qs); ax.set_xticklabels([f'Q{q}' for q in qs])
    ax.set_xlabel('True |Δy| quintile (Q5 = largest changes)', fontsize=9)
    ax.set_ylabel('Delta MAE')
    ax.set_title('(d) Amplitude-stratified Delta MAE (ODE)', fontsize=10)
    ax.grid(alpha=0.25, ls='--')
    ax.legend(fontsize=7)

    fig.suptitle(f'Figure 4 | Amplitude diagnostics ({split}, subject-level mean)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(Path(out_dir) / 'fig4_amplitude_diagnostics.png', dpi=180)
    plt.close(fig)


def _print_summary(arms, summaries, pairs):
    names = list(arms)
    print('\n=== Subject-level summary (mean ± std) ===')
    header = f"{'metric':34s}" + ''.join(f'{n:>18s}' for n in names)
    print(header)
    for metric, label, _ in METRICS:
        row = f'{label:34s}'
        for n in names:
            s = summaries[n].get(metric, {})
            if s.get('n', 0):
                row += f"{s['mean']:>10.4f}±{s['std']:<7.4f}"
            else:
                row += f"{'n/a':>18s}"
        print(row)
    for pair, block in pairs.items():
        print(f'\n=== Paired improvement: {pair} (positive = better) ===')
        for metric, label, _ in METRICS:
            r = block.get(metric)
            if not r:
                continue
            print(f"{label:34s} Δ={r['mean_improvement']:+.4f} "
                  f"CI95=[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}] "
                  f"p={r['p_value']:.3g} win={r['win_rate']:.1%} "
                  f"(n={r['n_subjects']})")


def main():
    ap = argparse.ArgumentParser(description='第一阶段 E0/E1/E2 对比与作图')
    ap.add_argument('--arms', required=True,
                    help='逗号分隔的 NAME=EVAL_DIR，例如 '
                         'E0=results/phase1/eval_E0,E1=...,E2=...')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--allow_misaligned', action='store_true',
                    help='锚点/被试不一致时只告警（默认直接报错）')
    ap.add_argument('--no_figures', action='store_true')
    ns = ap.parse_args()

    arms = {}
    for item in ns.arms.split(','):
        if '=' not in item:
            raise SystemExit(f'--arms 项格式应为 NAME=EVAL_DIR，收到 {item!r}')
        name, path = item.split('=', 1)
        arms[name.strip()] = load_arm(name.strip(), path.strip(), ns.split)

    table, summaries, pairs = build_comparison(
        arms, strict=not ns.allow_misaligned)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(out_dir / f'phase1_subject_metrics_{ns.split}.csv', index=False)
    payload = {
        'split': ns.split,
        'arms': {n: {'eval_dir': str(arms[n]['dir']),
                     'subjects': int(len(arms[n]['subject']))} for n in arms},
        'metrics': {m: {'label': label, 'lower_is_better': lower}
                    for m, label, lower in METRICS},
        'summaries': summaries,
        'paired': pairs,
        'diagnostics': {
            n: {k: v.get('overall', {}) for k, v in
                arms[n]['metrics'].get('phase1', {}).get('diagnostics', {})
                .get('blocks', {}).items()}
            for n in arms
        },
        'alignment': validate_alignment(arms, strict=False),
    }
    with (out_dir / 'phase1_comparison.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows = []
    for name in arms:
        for metric, label, lower in METRICS:
            s = summaries[name].get(metric, {})
            rows.append({'arm': name, 'metric': metric, 'label': label,
                         'lower_is_better': lower, **s})
    pd.DataFrame(rows).to_csv(out_dir / 'phase1_summary.csv', index=False)
    pair_rows = []
    for pair, block in pairs.items():
        for metric, r in block.items():
            pair_rows.append({'pair': pair, 'metric': metric, **r})
    pd.DataFrame(pair_rows).to_csv(out_dir / 'phase1_paired_tests.csv', index=False)

    if not ns.no_figures:
        plot_figures(arms, table, summaries, out_dir, ns.split)
    _print_summary(arms, summaries, pairs)
    log.info('对比结果已写入 %s', out_dir)


if __name__ == '__main__':
    main()
