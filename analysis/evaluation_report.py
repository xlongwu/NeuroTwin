# coding=utf-8
"""测试集评估产物的只读分析与可视化。

输入是 ``main.py`` 微调结束时自动落盘的三份产物（默认在 ``<save_dir>/eval/``）：

* ``指标_<split>.json``                   —— 协议、模型/基线/rollout/CPM/FC 汇总
* ``逐被试指标_<split>.csv``              —— 被试级指标 + 条件字段取值与有效性标记
* ``逐任务指标_<split>.csv``              —— 逐任务（被试 × anchor）指标

为读取历史 eval 目录，若中文文件名不存在则回退到
``metrics_<split>.json`` / ``per_subject_metrics_<split>.csv`` /
``per_task_metrics_<split>.csv``。

脚本不加载权重、不重新推理，因此**不会重复消费留出集**；所有图只做统计与对照展示。

用法::

    python -m analysis.evaluation_report \
        --eval_dir checkpoints/h1_8_v2/neurotwin_finetune/eval --out_dir results/evaluation_report \
        --meta_xlsx data/Rest-meta-MDD-V1V2-Merged-MDD.xlsx

产出 8 张图 + ``summary.md`` + ``pathology_strata.csv``：t+1 与平凡基线对照、被试级行为
与表型梯度、多步退化、功能连接、条件字段可见性、任务级误差结构、ID 前缀分组效应、
病理维度分层（HAMD 五等分 / episode / illness / medication / Sex / HAMA 共病）。
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.common import split_label

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 环境缺 matplotlib 时仍出表
    matplotlib = None
    plt = None

C_MODEL = '#1f77b4'
C_PERSIST = '#7f7f7f'
C_TREND = '#2ca02c'
C_AR1 = '#d62728'
C_CPM = '#ff7f0e'


# ---------------------------------------------------------------- 数据装载

def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f'缺少评估产物 {path}；请先确认 --eval_dir/--split 指向 main.py 落盘的 eval 目录')
    return path


def _artifact(eval_dir: Path, preferred: str, fallback: str) -> Path:
    """优先读取中文产物；仅在历史英文产物存在时回退。"""
    preferred_path = eval_dir / preferred
    fallback_path = eval_dir / fallback
    if preferred_path.is_file() or not fallback_path.is_file():
        return preferred_path
    return fallback_path


def load_eval_artifacts(eval_dir: Path, split: str) -> Dict[str, object]:
    """读取三份评估产物，返回 {metrics, subject, task}。"""
    label = split_label(split)
    metrics_path = _artifact(eval_dir, f'指标_{label}.json', f'metrics_{split}.json')
    subj_path = _artifact(eval_dir, f'逐被试指标_{label}.csv',
                          f'per_subject_metrics_{split}.csv')
    task_path = _artifact(eval_dir, f'逐任务指标_{label}.csv',
                          f'per_task_metrics_{split}.csv')
    metrics = json.loads(_require(metrics_path).read_text(encoding='utf-8'))
    subj = pd.read_csv(_require(subj_path))
    task = pd.read_csv(_require(task_path))
    if not {'subject_id', 't+1_MAE', 't+1_PCC_spatial'} <= set(subj.columns):
        raise ValueError(f'逐被试 CSV 缺少必需列，实际列：{list(subj.columns)}')
    return {'metrics': metrics, 'subject': subj, 'task': task}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float('nan')
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    return _pearson(_rank(a[ok]), _rank(b[ok]))


# ---------------------------------------------------------------- 图 1｜t+1 与平凡基线

def plot_vs_baselines(subj: pd.DataFrame, task: pd.DataFrame, out_dir: Path) -> None:
    base = [('persistence', C_PERSIST), ('trend', C_TREND), ('ar1', C_AR1)]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    ax = axes[0, 0]
    pm = subj['persistence_t+1_MAE'].to_numpy()
    mm = subj['t+1_MAE'].to_numpy()
    ax.plot([pm.min() * .95, pm.max() * 1.05] * 2, [pm.min() * .95, pm.max() * 1.05] * 2,
            ls='--', c='k', lw=1, label='y = x (no gain)')
    ax.scatter(pm, mm, s=14, alpha=.7, c=C_MODEL, edgecolors='none')
    ax.set_xlabel('per-subject MAE: persistence')
    ax.set_ylabel('per-subject MAE: model')
    ax.set_title(f't+1 model vs persistence (n={len(subj)}, '
                 f'win rate={(mm < pm).mean():.1%})')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    series = [('model', subj['t+1_MAE'], C_MODEL)] + \
             [(n, subj[f'{n}_t+1_MAE'], c) for n, c in base]
    parts = ax.boxplot([s[1].dropna().to_numpy() for s in series],
                       positions=np.arange(len(series)), widths=.2, showfliers=False,
                       patch_artist=True)
    for pc, (_, _, color) in zip(parts['boxes'], series):
        pc.set(facecolor=color, alpha=.75)
    ax.set_xticks(np.arange(len(series)))
    ax.set_xticklabels([s[0] for s in series], fontsize=9)
    ax.set_ylabel('t+1 MAE (subject level)')
    ax.set_title('t+1 error distribution: model vs trivial baselines')

    ax = axes[1, 0]
    gain = subj['persistence_t+1_MAE'] - subj['t+1_MAE']
    ax.hist(gain, bins=30, color=C_MODEL, alpha=.8)
    ax.axvline(gain.median(), c='k', ls='--', lw=1,
               label=f'median={gain.median():.3f}')
    ax.set_xlabel('persistence MAE - model MAE (larger = better)')
    ax.set_ylabel('#subjects')
    ax.set_title(f'per-subject absolute gain vs persistence\n'
                 f'all positive: {bool((gain > 0).all())}')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(task['t+1_PCC_spatial'].dropna(), bins=60, color=C_MODEL, alpha=.8)
    ax.axvline(task['t+1_PCC_spatial'].mean(), c='k', ls='--', lw=1,
               label=f'task mean={task["t+1_PCC_spatial"].mean():.4f}')
    ax.set_xlabel('t+1 spatial PCC (task level)')
    ax.set_ylabel('#tasks')
    ax.set_title(f'task-level spatial PCC (n={len(task)})')
    ax.legend(fontsize=8)

    fig.suptitle('Test split: one-step next-state prediction', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out_dir / 'fig1_next_state_vs_baselines.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 2｜被试与表型梯度

def plot_subject_profile(subj: pd.DataFrame, metrics: Dict, out_dir: Path) -> None:
    fields = [c[len('obs_'):] for c in subj.columns if c.startswith('obs_')]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    ax = axes[0, 0]
    pcc = np.sort(subj['t+1_PCC_spatial'].to_numpy())
    ax.plot(np.arange(1, len(pcc) + 1), pcc, marker='.', ms=4, lw=.8, c=C_MODEL)
    ax.axhline(pcc.mean(), ls='--', c='k', lw=1, label=f'mean={pcc.mean():.4f}')
    ax.fill_between([1, len(pcc)], pcc.mean() - pcc.std(), pcc.mean() + pcc.std(),
                    color=C_MODEL, alpha=.15, label='±1 sd')
    ax.set_xlabel('subject (sorted)')
    ax.set_ylabel('t+1 spatial PCC')
    ax.set_title(f'subject-level spread (min={pcc.min():.4f}, max={pcc.max():.4f})')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ok = subj['hamd'].notna()
    r_p = _pearson(subj.loc[ok, 'hamd'], subj.loc[ok, 't+1_PCC_spatial'])
    r_s = _spearman(subj.loc[ok, 'hamd'], subj.loc[ok, 't+1_PCC_spatial'])
    ax.scatter(subj.loc[ok, 'hamd'], subj.loc[ok, 't+1_PCC_spatial'], s=16, alpha=.7,
               c=C_AR1, edgecolors='none')
    if ok.sum() > 2:
        coef = np.polyfit(subj.loc[ok, 'hamd'], subj.loc[ok, 't+1_PCC_spatial'], 1)
        xs = np.linspace(subj.loc[ok, 'hamd'].min(), subj.loc[ok, 'hamd'].max(), 20)
        ax.plot(xs, np.polyval(coef, xs), c='k', lw=1, ls='--')
    ax.set_xlabel('HAMD total')
    ax.set_ylabel('t+1 spatial PCC')
    ax.set_title(f'symptom severity vs accuracy (pearson={r_p:.3f}, spearman={r_s:.3f})')
    ref = metrics.get('hamd', {}).get('corr_hamd_PCC_t+1')
    if ref is not None:
        ax.text(.02, .04, f'eval json corr = {ref:.4f}', transform=ax.transAxes, fontsize=8)

    ax = axes[1, 0]
    if fields:
        n_obs = subj[['obs_' + f for f in fields]].sum(axis=1)
        groups = sorted(n_obs.unique())
        ax.boxplot([subj.loc[n_obs == g, 't+1_MAE'].to_numpy() for g in groups],
                   widths=.6, showfliers=False)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([f'{int(g)}/{len(fields)}\n(n={int((n_obs == g).sum())})'
                            for g in groups], fontsize=8)
        ax.set_xlabel('#observed condition fields per subject')
        ax.set_title('prediction error vs condition completeness')
    else:
        ax.text(.5, .5, 'no obs_* columns in this CSV\n(condition injection not enabled)',
                transform=ax.transAxes, ha='center', fontsize=10)
        ax.set_title('prediction error vs condition completeness')
    ax.set_ylabel('t+1 MAE')

    ax = axes[1, 1]
    if 'obs_HAMA' in subj:
        has = subj['obs_HAMA'] > 0
        parts = ax.boxplot([subj.loc[~has, 't+1_MAE'].to_numpy(),
                            subj.loc[has, 't+1_MAE'].to_numpy()],
                           widths=.5, showfliers=False, patch_artist=True)
        for pc, c in zip(parts['boxes'], (C_PERSIST, C_CPM)):
            pc.set(facecolor=c, alpha=.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'HAMA missing\n(n={int((~has).sum())})',
                            f'HAMA present\n(n={int(has.sum())})'], fontsize=9)
        ax.set_ylabel('t+1 MAE')
        ax.set_title('scarce field as de-facto subgroup marker')
    else:
        ax.axis('off')

    fig.suptitle('Test split: subject-level behaviour', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out_dir / 'fig2_subject_profile.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 3｜horizon 退化

def plot_horizon(metrics: Dict, out_dir: Path) -> None:
    roll = metrics.get('rollout', {})
    if not roll.get('enabled'):
        log.warning('rollout 未启用，跳过 horizon 图')
        return
    model = roll['model']
    hs = sorted(model, key=lambda h: int(h[1:]))
    idx = [int(h[1:]) for h in hs]
    base = roll.get('baselines', {})

    def curve(dct, key):
        return [dct.get(h, {}).get(key, np.nan) for h in hs]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    ax = axes[0, 0]
    ax.plot(idx, curve(model, 'PCC_spatial'), 'o-', c=C_MODEL, label='rollout (model)')
    cpm = metrics.get('cpm_horizon', {}).get('model', {})
    if cpm:
        ci = sorted(cpm, key=lambda h: int(h[1:]))
        ax.plot([int(h[1:]) for h in ci], [cpm[h]['PCC_spatial'] for h in ci], 's--',
                c=C_CPM, label='CPM (one-shot)')
    for name, color in (('persistence', C_PERSIST), ('trend', C_TREND), ('ar1', C_AR1)):
        if name in base:
            ax.plot(idx, curve(base[name], 'PCC_spatial'), 'x:', c=color, alpha=.85,
                    label=name)
    ax.set_xticks(idx)
    ax.set_xlabel('rollout horizon (TR)')
    ax.set_ylabel('spatial PCC')
    ax.set_title('multi-step spatial pattern fidelity')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(idx, curve(model, 'MAE'), 'o-', c=C_MODEL, label='model')
    for name, color in (('persistence', C_PERSIST), ('trend', C_TREND), ('ar1', C_AR1)):
        if name in base:
            ax.plot(idx, curve(base[name], 'MAE'), 'x:', c=color, alpha=.85, label=name)
    ax.set_xticks(idx)
    ax.set_yscale('log')
    ax.set_xlabel('rollout horizon (TR)')
    ax.set_ylabel('MAE (log)')
    ax.set_title('amplitude error growth')

    ax = axes[1, 0]
    vr = curve(model, 'variance_ratio')
    ax.plot(idx, vr, 'o-', c=C_AR1)
    ax.axhline(1.0, ls='--', c='k', lw=1, label='variance ratio = 1')
    ax.fill_between(idx, vr, 1.0, color=C_AR1, alpha=.12)
    ax.set_xticks(idx)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('rollout horizon (TR)')
    ax.set_ylabel('Var(pred) / Var(true)')
    ax.set_title('variance collapse under free rollout')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    imps = roll.get('improvement_vs_persistence', {})
    di = [imps.get(h, {}).get('delta_PCC', np.nan) for h in hs]
    dm = [imps.get(h, {}).get('delta_MAE', np.nan) for h in hs]
    x = np.arange(len(hs))
    ax.bar(x - .18, di, width=.36, color=C_MODEL, label='delta PCC (model - persistence)')
    ax.bar(x + .18, dm, width=.36, color=C_TREND, label='delta MAE (model - persistence)')
    ax.axhline(0, c='k', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(hs, fontsize=9)
    ax.set_ylabel('difference')
    ax.set_title('margin over persistence per horizon')
    ax.legend(fontsize=8)

    fig.suptitle('Test split: autoregressive degradation', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out_dir / 'fig3_horizon_degradation.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 4｜功能连接

def plot_fc(metrics: Dict, out_dir: Path) -> None:
    fc = metrics.get('fc', {})
    if not fc.get('model', {}).get('available'):
        log.warning('FC 不可用（长度 < fc_min_length 或未 rollout），跳过 FC 图')
        return
    names = [n for n in ('model', 'persistence', 'trend', 'ar1') if n in fc]
    colors = {'model': C_MODEL, 'persistence': C_PERSIST, 'trend': C_TREND, 'ar1': C_AR1}
    scopes = ['all', 'within', 'between']

    def value(node, scope, key):
        if node.get('available') is False:
            return np.nan
        d = node if scope == 'all' else node.get('network', {}).get(scope)
        return d.get(key, np.nan) if d else np.nan

    net = fc['model'].get('network', {})
    labels = ['all edges',
              f"within-network\n(n={net.get('within', {}).get('n_edges', '?')})",
              f"between-network\n(n={net.get('between', {}).get('n_edges', '?')})"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, key, ylab in ((axes[0], 'fc_mae', 'FC MAE (lower better)'),
                          (axes[1], 'edge_pcc_mean', 'edge-wise PCC (higher better)')):
        x = np.arange(len(scopes))
        for i, name in enumerate(names):
            vals = [value(fc[name], s, key) for s in scopes]
            off = x + (i - (len(names) - 1) / 2) * .2
            ax.bar(off, vals, width=.2, color=colors[name], label=name, alpha=.9)
            for xx, v in zip(off, vals):
                if np.isfinite(v):
                    ax.text(xx, v, f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, c='k', lw=.8)
        ax.set_ylabel(ylab)
        ax.set_title(f"{fc['model']['n_edges']} edges, rollout length={fc['model']['length']}")
        ax.margins(y=.18)
        ax.legend(fontsize=8, ncol=len(names), loc='upper center', framealpha=.9)

    fig.suptitle('Test split: functional connectivity of the rollout trajectory', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out_dir / 'fig4_functional_connectivity.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 5｜条件覆盖与分布

def plot_condition_coverage(subj: pd.DataFrame, out_dir: Path) -> None:
    fields = [c[len('obs_'):] for c in subj.columns if c.startswith('obs_')]
    if not fields:
        log.warning('逐被试 CSV 无 obs_* 列（该次评估未启用条件），跳过条件覆盖图')
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    ax = axes[0, 0]
    rate = [subj['obs_' + f].mean() for f in fields]
    order = np.argsort(rate)
    ax.barh(np.arange(len(fields)), np.array(rate)[order], color=C_MODEL, alpha=.85)
    ax.set_yticks(np.arange(len(fields)))
    ax.set_yticklabels([fields[i] for i in order], fontsize=9)
    ax.set_xlim(0, 1.05)
    for y, v in zip(np.arange(len(fields)), np.array(rate)[order]):
        ax.text(v + .01, y, f'{v:.0%}', va='center', fontsize=8)
    ax.set_xlabel('share of test subjects with a usable value')
    ax.set_title('condition field observability (test subjects)')

    ax = axes[0, 1]
    n_obs = subj[['obs_' + f for f in fields]].sum(axis=1)
    counts = n_obs.value_counts().sort_index()
    xpos = np.arange(len(counts))
    ax.bar(xpos, counts.to_numpy(), color=C_TREND, alpha=.85)
    for xx, yy in zip(xpos, counts.to_numpy()):
        ax.text(xx, yy, str(yy), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(int(k)) for k in counts.index], fontsize=9)
    ax.set_xlabel('#fields observed per subject')
    ax.set_ylabel('#subjects')
    ax.set_title('completeness profile')

    ax = axes[1, 0]
    for name, color in (('cond_HAMD', C_AR1), ('cond_HAMA', C_CPM)):
        obs = 'obs_' + name[len('cond_'):]
        if name in subj and obs in subj:
            v = subj.loc[subj[obs] > 0, name].dropna()
            if len(v):
                ax.hist(v, bins=20, alpha=.6, color=color,
                        label=f'{name[len("cond_"):]} (n={len(v)}, mean={v.mean():.1f})')
    ax.set_xlabel('score value')
    ax.set_ylabel('#subjects')
    ax.set_title('rating-scale dynamic range actually fed to the encoder')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if 'cond_illness' in subj:
        v = subj.loc[subj['obs_illness'] > 0, 'cond_illness'].dropna().to_numpy()
        ax.hist(np.log10(np.clip(v, 1, None)), bins=20, color=C_MODEL, alpha=.8)
        ax.set_xlabel('log10 illness (value unit as recorded in merged table)')
        ax.set_ylabel('#subjects')
        ax.set_title(f'skewed continuous field (median={np.median(v):.0f}, '
                     f'max={np.max(v):.0f})')
        ax.axvline(np.log10(max(np.median(v), 1)), ls='--', c='k', lw=1)
    else:
        ax.axis('off')

    fig.suptitle('Test split: what the condition encoder can actually see', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out_dir / 'fig5_condition_coverage.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 6｜任务级误差结构

def plot_task_structure(task: pd.DataFrame, out_dir: Path) -> None:
    lengths = sorted(task['context_len'].unique())
    if len(lengths) == 1:
        log.info('context length 全为 %d，无长度混杂可检验', int(lengths[0]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    cut = task['cutoff'].to_numpy()
    pcc = task['t+1_PCC_spatial'].to_numpy()
    ax.scatter(cut, pcc, s=5, alpha=.25, c=C_MODEL, edgecolors='none')
    bins = pd.qcut(pd.Series(cut), 10, duplicates='drop')
    gm = pd.Series(pcc).groupby(bins, observed=True).mean()
    gx = pd.Series(cut).groupby(bins, observed=True).mean()
    ax.plot(gx, gm, 'o-', c=C_AR1, lw=1.5, label='decile mean')
    ax.set_xlabel('anchor cutoff (timepoint index in session)')
    ax.set_ylabel('t+1 spatial PCC')
    ax.set_title('within-session position effect')
    ax.legend(fontsize=8)

    ax = axes[1]
    subj_mean = task.groupby('subj_id')['t+1_PCC_spatial'].transform('mean')
    total = np.var(pcc)
    ax.scatter(pcc - subj_mean, subj_mean - pcc.mean(), s=6, alpha=.35, c=C_MODEL,
               edgecolors='none')
    between = np.var(subj_mean)
    ax.set_xlabel('task deviation from its subject mean')
    ax.set_ylabel('subject mean deviation from group mean')
    ax.set_title('variance decomposition\n'
                 f'between-subject share = {between / total:.1%} of total task variance')

    fig.suptitle('Test split: where the residual error lives', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out_dir / 'fig6_task_structure.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 7｜ID 前缀分组效应

def _group_table(subj: pd.DataFrame) -> pd.DataFrame:
    """按 ``subject_id`` 前缀汇总被试级指标（前缀是数据里的站点/入组编码）。"""
    df = subj.copy()
    df['group'] = df['subject_id'].astype(str).str.extract(r'^(.*?)-\d')[0]
    out = df.groupby('group').agg(n=('t+1_MAE', 'size'), mae=('t+1_MAE', 'mean'),
                                  pcc=('t+1_PCC_spatial', 'mean'),
                                  base=('persistence_t+1_MAE', 'mean'))
    return out.sort_values('mae')


def plot_group_effect(subj: pd.DataFrame, out_dir: Path,
                      dataset: pd.Series = None) -> None:
    """看误差是否成簇，并与「队列（Dataset）层面」对照。

    ``dataset`` 为可选的 V1/V2 归属（来自 merged 元数据表），用于区分「队列层面差异」
    与「个别编组差异」。
    """
    df = subj.copy()
    g = _group_table(subj)
    hot = g['mae'] > 1.5 * g['mae'].median()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.barh(np.arange(len(g)), g['mae'], color=[C_AR1 if h else C_MODEL for h in hot])
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels([f'{i} (n={int(n)})' for i, n in zip(g.index, g['n'])], fontsize=7)
    ax.axvline(g['mae'].median(), ls='--', c='k', lw=1,
               label=f'median of groups={g["mae"].median():.4f}')
    ax.set_xlabel('mean t+1 MAE per ID-prefix group')
    ax.set_title(f'group-level error (red = >1.5x median; {int(hot.sum())} groups)')
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[1]
    ax.scatter(g['base'], g['mae'], s=[12 * n for n in g['n']], c=C_MODEL, alpha=.7,
               edgecolors='none')
    for idx, row in g[hot].iterrows():
        ax.annotate(idx, (row['base'], row['mae']), fontsize=8,
                    xytext=(4, 4), textcoords='offset points', color=C_AR1)
    lim = [min(g['base'].min(), g['mae'].min()) * .9, g['base'].max() * 1.1]
    ax.plot(lim, lim, ls='--', c='k', lw=1)
    ax.set_xlabel('group mean persistence MAE')
    ax.set_ylabel('group mean model MAE')
    ax.set_title('hard groups: model-only or data-only?\n'
                 f'groups below y=x: {int((g["mae"] < g["base"]).sum())}/{len(g)}')

    ax = axes[2]
    if dataset is not None and dataset.notna().any():
        df['dataset'] = dataset.reindex(df.index).to_numpy()
        names = sorted(df['dataset'].dropna().unique())
        parts = ax.boxplot([df.loc[df['dataset'] == d, 't+1_MAE'].to_numpy() for d in names],
                           widths=.5, showfliers=False, patch_artist=True)
        for pc in parts['boxes']:
            pc.set(facecolor=C_TREND, alpha=.7)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([f'{d}\n(n={int((df["dataset"] == d).sum())})' for d in names],
                           fontsize=9)
        ax.set_ylabel('t+1 MAE')
        means = ' '.join(f'{d}={df.loc[df["dataset"] == d, "t+1_MAE"].mean():.4f}'
                         for d in names)
        ax.set_title(f'cohort level (Dataset): {means}')
    else:
        ax.scatter(g['pcc'], g['n'], s=28, c=C_CPM, alpha=.8, edgecolors='none')
        ax.set_xlabel('group mean t+1 spatial PCC')
        ax.set_ylabel('#subjects in group')
        ax.set_title('group size vs accuracy (no --meta_xlsx given)')

    fig.suptitle('Test split: is the residual error subject-specific or grouped?', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out_dir / 'fig7_group_effect.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 图 8｜病理分层

def _spearman_test(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Spearman 秩相关与双侧 p（t 近似）；scipy 缺失时 p 记为 nan。"""
    rho = _spearman(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    n = int(np.isfinite(np.asarray(a, dtype=np.float64)).sum())
    if not np.isfinite(rho) or n < 3:
        return rho, float('nan')
    try:
        from scipy import stats
        return rho, float(stats.spearmanr(a, b).pvalue)
    except Exception:  # noqa: BLE001 - scipy 缺失只损失 p 值，不损失结论
        log.warning('scipy 不可用，Spearman p 值未计算')
        return rho, float('nan')


def _kruskal(groups: List[np.ndarray]) -> float:
    try:
        from scipy import stats
        keep = [g for g in groups if len(g) >= 2]
        if len(keep) < 2:
            return float('nan')
        return float(stats.kruskal(*keep).pvalue)
    except Exception:  # noqa: BLE001
        return float('nan')


def _strata_table(subj: pd.DataFrame, name: str,
                  labels: pd.Series) -> pd.DataFrame:
    """按给定分层标签汇总被试级指标（独立单位是被试，不做任务级混算）。"""
    df = subj.assign(strata=labels,
                     gain=subj['persistence_t+1_MAE'] - subj['t+1_MAE'])
    out = df.groupby('strata', observed=True).agg(
        n=('t+1_MAE', 'size'), mae_mean=('t+1_MAE', 'mean'), mae_sd=('t+1_MAE', 'std'),
        pcc_mean=('t+1_PCC_spatial', 'mean'), persistence_mean=('persistence_t+1_MAE', 'mean'),
        gain_mean=('gain', 'mean'), hamd_mean=('hamd', 'mean'))
    out.insert(0, 'stratifier', name)
    return out.reset_index()


def plot_pathology(subj: pd.DataFrame, out_dir: Path,
                   hot_groups: Tuple[str, ...] = ()) -> List[Tuple[str, str]]:
    """病理维度分层：严重度分位、episode、illness、medication、Sex、HAMA 共病。

    分层口径沿用项目自身的 HAMD 五等分（``utils/dataloader.py`` 的
    ``stratify_key='HAMD'`` / ``stratify_bins=5``），不自拟临床严重度切点：元数据里
    HAMD1-17 有值、HAMD18-24 全空，无法确认量表版本，切点阈值不可靠。

    独立统计单位是被试（与主口径一致）；组间检验用 Kruskal-Wallis（非参数，不假设正态）。
    """
    df = subj.copy()
    df['gain'] = df['persistence_t+1_MAE'] - df['t+1_MAE']

    strata: Dict[str, pd.Series] = {}
    hamd = df['cond_HAMD'].where(df['obs_HAMD'] > 0) if 'obs_HAMD' in df else df['hamd']
    strata['HAMD_quintile'] = pd.qcut(hamd, 5, duplicates='drop')
    for col in ('episode', 'medication', 'Sex'):
        obs = df.get('obs_' + col)
        if obs is None:
            continue
        vals = df['cond_' + col].where(obs > 0)
        if vals.notna().sum() >= 20 and vals.nunique(dropna=True) > 1:
            strata[col] = vals.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    if {'cond_illness', 'obs_illness'} <= set(df.columns):
        illness = df['cond_illness'].where(df['obs_illness'] > 0)
        if illness.notna().sum() >= 30:
            strata['illness_tertile'] = pd.qcut(illness, 3, duplicates='drop')

    tables, kw_p = [], {}
    for name, lab in strata.items():
        t = _strata_table(df, name, lab)
        tables.append(t)
        kw_p[name] = _kruskal([df.loc[lab == k, 't+1_MAE'].to_numpy() for k in t['strata']])
    tab = pd.concat(tables, ignore_index=True)
    tab['kruskal_p'] = tab['stratifier'].map(kw_p)
    tab.to_csv(out_dir / 'pathology_strata.csv', index=False)

    rho, p_rho = _spearman_test(hamd, df['t+1_MAE'])
    rho_g, p_g = _spearman_test(hamd, df['gain'])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    qt = tab[tab['stratifier'] == 'HAMD_quintile']
    qtick = [f'{s}\nn={int(n)}' for s, n in zip(qt['strata'], qt['n'])]

    ax = axes[0, 0]
    ax.bar(range(len(qt)), qt['mae_mean'], yerr=qt['mae_sd'], capsize=4,
           color=C_MODEL, alpha=.85)
    ax.set_xticks(range(len(qt)))
    ax.set_xticklabels(qtick, fontsize=7)
    ax.set_ylabel('t+1 MAE')
    ax.set_xlabel('HAMD quintile (project stratifier)')
    ax.set_title(f'error vs severity: rho={rho:.3f} (p={p_rho:.3g})')

    ax = axes[0, 1]
    ax.bar(range(len(qt)), qt['gain_mean'], color=C_TREND, alpha=.85)
    ax.axhline(qt['gain_mean'].mean(), ls='--', c='k', lw=1)
    ax.set_xticks(range(len(qt)))
    ax.set_xticklabels(qtick, fontsize=7)
    ax.set_ylabel('persistence MAE - model MAE')
    ax.set_xlabel('HAMD quintile')
    ax.set_title('does the advantage survive at high severity?\n'
                 f'rho(gain, HAMD)={rho_g:.3f} (p={p_g:.3g})')

    ax = axes[0, 2]
    grp = df['subject_id'].astype(str).str.extract(r'^(.*?)-\d')[0]
    hot = grp.isin(hot_groups) if hot_groups else pd.Series(False, index=df.index)
    if hot.sum() >= 5:
        ax.hist(df.loc[~hot, 'hamd'], bins=15, alpha=.65, color=C_MODEL,
                label=f'other subjects (n={int((~hot).sum())})')
        ax.hist(df.loc[hot, 'hamd'], bins=15, alpha=.65, color=C_AR1,
                label=f'error-clustered (n={int(hot.sum())})')
        r_c, p_c = _spearman_test(df.loc[hot, 'hamd'], df.loc[hot, 't+1_MAE'])
        ax.set_title(f'is the error cluster a severity cluster?\n'
                     f'within-cluster rho={r_c:.3f} (p={p_c:.3g}, n={int(hot.sum())})')
    else:
        ax.hist(df['hamd'], bins=15, alpha=.8, color=C_MODEL)
        ax.set_title('HAMD distribution (test subjects)')
    ax.set_xlabel('HAMD total')
    ax.set_ylabel('#subjects')
    if hot.sum() >= 5:
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    cat_names = [n for n in strata if n not in ('HAMD_quintile', 'illness_tertile')]
    pos, seen = 0, []
    for j, name in enumerate(cat_names):
        tt = tab[tab['stratifier'] == name]
        ax.bar(np.arange(len(tt)) + pos, tt['mae_mean'], yerr=tt['mae_sd'], capsize=3,
               width=.8, color=[C_MODEL, C_AR1, C_CPM][j % 3], alpha=.85,
               label=f'{name} (KW p={kw_p[name]:.3g})')
        seen += [f'{name}\n{s}' for s in tt['strata']]
        pos += len(tt) + 1
    if seen:
        ax.set_xticks(range(len(seen)))
        ax.set_xticklabels(seen, fontsize=7, rotation=35, ha='right')
        ax.axhline(df['t+1_MAE'].mean(), ls=':', c='k', lw=1, label='cohort mean')
        ax.set_ylabel('t+1 MAE')
        ax.set_title('categorical clinical strata')
        ax.legend(fontsize=8)
    else:
        ax.axis('off')

    ax = axes[1, 1]
    if 'illness_tertile' in strata:
        it = tab[tab['stratifier'] == 'illness_tertile']
        ax.bar(range(len(it)), it['mae_mean'], yerr=it['mae_sd'], capsize=4,
               color=C_CPM, alpha=.85)
        ax.set_xticks(range(len(it)))
        ax.set_xticklabels([f'{s}\nn={int(n)}' for s, n in zip(it['strata'], it['n'])],
                           fontsize=7)
        ax.set_ylabel('t+1 MAE')
        ax.set_xlabel('illness tertile')
        ax.set_title(f'illness-history tertiles (KW p={kw_p["illness_tertile"]:.3g})')
    else:
        n_hama = int((df.get('obs_HAMA', pd.Series(0, index=df.index)) > 0).sum())
        ax.text(.5, .5, f'illness usable for <30 subjects (HAMA rated for {n_hama})',
                transform=ax.transAxes, ha='center')
        ax.set_title('illness-history tertiles')

    ax = axes[1, 2]
    if {'obs_HAMA', 'cond_HAMA'} <= set(df.columns):
        both = (df['obs_HAMD'] > 0) & (df['obs_HAMA'] > 0)
        ax.set_xlabel('HAMD total')
        ax.set_ylabel('HAMA total')
        if both.sum() >= 10:
            r_h, p_h = _spearman_test(df.loc[both, 'cond_HAMD'], df.loc[both, 'cond_HAMA'])
            ax.scatter(df.loc[both, 'cond_HAMD'], df.loc[both, 'cond_HAMA'], s=22,
                       alpha=.8, c=C_AR1, edgecolors='none')
            coef = np.polyfit(df.loc[both, 'cond_HAMD'], df.loc[both, 'cond_HAMA'], 1)
            xs = np.linspace(df.loc[both, 'cond_HAMD'].min(),
                             df.loc[both, 'cond_HAMD'].max(), 10)
            ax.plot(xs, np.polyval(coef, xs), c='k', ls='--', lw=1)
            ax.set_title(f'comorbidity: only n={int(both.sum())} rated on both\n'
                         f'rho={r_h:.3f} (p={p_h:.3g})')
        else:
            ax.text(.5, .5, f'HAMA rated for only {int(both.sum())} subjects',
                    transform=ax.transAxes, ha='center')
            ax.set_title('comorbidity structure')
    else:
        ax.axis('off')

    fig.suptitle('Test split: pathology-stratified behaviour', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out_dir / 'fig8_pathology_strata.png', dpi=150)
    plt.close(fig)

    rows: List[Tuple[str, str]] = [
        ('rho(MAE, HAMD) / p', f'{rho:.4f} / {p_rho:.3g}'),
        ('rho(gain, HAMD) / p', f'{rho_g:.4f} / {p_g:.3g}')]
    for name in strata:
        tt = tab[tab['stratifier'] == name]
        rows.append((f'{name}: MAE over strata',
                     f'{tt["mae_mean"].min():.4f} - {tt["mae_mean"].max():.4f}, '
                     f'n={[int(v) for v in tt["n"]]}, KW p={kw_p[name]:.3g}'))
    return rows


# ---------------------------------------------------------------- 文字摘要

def build_summary(metrics: Dict, subj: pd.DataFrame, task: pd.DataFrame,
                  group_table: pd.DataFrame = None) -> List[Tuple[str, str]]:
    m = metrics['next_state']['model']['t+1']
    cmp_ = metrics.get('baselines', {}).get('comparison', {})
    rows = [('subjects / tasks', f"{metrics['protocol']['n_subjects']} / {metrics['protocol']['n_tasks']}"),
            ('target', metrics['protocol'].get('prediction_target', 'n/a')),
            ('t+1 spatial PCC', f"{m['PCC_spatial']:.4f} ± {m.get('PCC_spatial_std', float('nan')):.4f}"),
            ('t+1 MAE / RMSE', f"{m['MAE']:.4f} / {m['RMSE']:.4f}"),
            ('t+1 R2', f"{m['R2']:.4f}"),
            ('t+1 delta direction', f"{m['delta_direction']:.4f}"),
            ('worst subject PCC', f"{subj['t+1_PCC_spatial'].min():.4f}"),
            ('HAMD vs PCC (pearson)', f"{_pearson(subj['hamd'], subj['t+1_PCC_spatial']):.4f}")]
    for k in ('delta_MAE_model_minus_persistence', 'delta_MAE_model_minus_trend',
              'delta_MAE_model_minus_ar1'):
        v = cmp_.get(k, {}).get('t+1')
        if v is not None:
            rows.append((k.replace('delta_MAE_model_minus_', 'MAE gain vs '), f'{v:.4f}'))
    roll = metrics.get('rollout', {}).get('model', {})
    if roll:
        rows.append(('rollout PCC', ' '.join(
            f"{h}:{roll[h]['PCC_spatial']:.3f}" for h in sorted(roll, key=lambda x: int(x[1:])))))
        rows.append(('rollout variance ratio', ' '.join(
            f"{h}:{roll[h].get('variance_ratio', float('nan')):.3f}"
            for h in sorted(roll, key=lambda x: int(x[1:])))))
    cpm = metrics.get('cpm_horizon', {}).get('model', {})
    if cpm:
        rows.append(('CPM PCC', ' '.join(
            f"{h}:{cpm[h]['PCC_spatial']:.3f}" for h in sorted(cpm, key=lambda x: int(x[1:])))))
    fc = metrics.get('fc', {})
    if fc.get('model', {}).get('available'):
        rows.append(('FC MAE (model/persistence)',
                     f"{fc['model']['fc_mae']:.4f} / {fc['persistence']['fc_mae']:.4f}"))
        rows.append(('FC edge PCC (model/persistence)',
                     f"{fc['model']['edge_pcc_mean']:.4f} / {fc['persistence']['edge_pcc_mean']:.4f}"))
    rows.append(('task MAE (mean/sd/max)',
                 f"{task['t+1_MAE'].mean():.4f} / {task['t+1_MAE'].std():.4f} / "
                 f"{task['t+1_MAE'].max():.4f}"))
    if group_table is not None and len(group_table):
        hot = group_table[group_table['mae'] > 1.5 * group_table['mae'].median()]
        rest = group_table.drop(hot.index)
        rows.append((f'ID-prefix groups ({len(group_table)})',
                     f'median group MAE={group_table["mae"].median():.4f}, '
                     f'range=[{group_table["mae"].min():.4f}, {group_table["mae"].max():.4f}]'))
        if len(hot):
            ratio = hot['mae'].mean() / rest['mae'].mean() if len(rest) else float('nan')
            rows.append(('>1.5x-median groups',
                         ', '.join(f'{i}(n={int(r["n"])}, MAE={r["mae"]:.4f}, '
                                   f'persistence={r["base"]:.4f})'
                                   for i, r in hot.iterrows()) + f' | vs rest x{ratio:.2f}'))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='测试集评估产物的只读分析与可视化')
    ap.add_argument('--eval_dir', required=True,
                    help='含 指标_<split>.json 与两份逐被试/逐任务 CSV 的目录')
    ap.add_argument('--out_dir', required=True, help='图片与摘要输出目录（建议放在 gitignore 内）')
    ap.add_argument('--split', default='test', choices=['test', 'val'], help='评估产物所属划分')
    ap.add_argument('--meta_xlsx', default='',
                    help='可选：merged 元数据表（含 ID/Dataset 两列），用于队列层面分组对照')
    args = ap.parse_args()

    eval_dir, out_dir = Path(args.eval_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_eval_artifacts(eval_dir, args.split)
    metrics, subj, task = data['metrics'], data['subject'], data['task']

    dataset = None
    if args.meta_xlsx:
        meta = pd.read_excel(_require(Path(args.meta_xlsx)), usecols=['ID', 'Dataset'])
        dataset = subj['subject_id'].map(meta.drop_duplicates('ID').set_index('ID')['Dataset'])
        matched = int(dataset.notna().sum())
        log.info('元数据匹配 %d/%d 被试的 Dataset 归属', matched, len(subj))
        if matched == 0:
            raise ValueError(f'{args.meta_xlsx} 的 ID 列与 {args.split} CSV 的 subject_id 无交集')

    gtab = _group_table(subj)
    hot_groups = tuple(gtab[gtab['mae'] > 1.5 * gtab['mae'].median()].index)
    rows = build_summary(metrics, subj, task, gtab)
    if plt is None:
        log.warning('matplotlib 不可用，仅输出摘要')
    else:
        rows += plot_pathology(subj, out_dir, hot_groups)
    width = max(len(k) for k, _ in rows)
    print(f'== {args.split} summary ({eval_dir}) ==')
    for k, v in rows:
        print(f'  {k:<{width}} : {v}')

    if plt is None:
        return
    plot_vs_baselines(subj, task, out_dir)
    plot_subject_profile(subj, metrics, out_dir)
    plot_horizon(metrics, out_dir)
    plot_fc(metrics, out_dir)
    plot_condition_coverage(subj, out_dir)
    plot_task_structure(task, out_dir)
    plot_group_effect(subj, out_dir, dataset)
    (out_dir / 'summary.md').write_text(
        '# ' + str(eval_dir) + f' ({args.split})\n\n' +
        '\n'.join(f'- **{k}**: {v}' for k, v in rows) + '\n', encoding='utf-8')
    log.info('输出目录：%s', out_dir)


if __name__ == '__main__':
    main()
