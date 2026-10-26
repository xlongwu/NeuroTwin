# coding=utf-8
"""Next-Timepoint（Next Brain-State Prediction）评估协议。

协议要点（与训练口径一致，prompt §十九–§二十三、§三十、§三十一）：

1. **确定性**：每个被试固定 ``K_eval = --eval_context_length``，预测位置 anchor 在
   合法区间内等间隔确定性选取（NextPointEvalView），每次评估的 (subject, K, t) 一致；
2. **被试级聚合**：被试内先平均、跨被试再统计 mean/std/median（论文统计的独立
   单位是 subject，任务级明细只用于调试）；
3. **trivial baselines**：persistence / linear trend / AR(1)（AR(1) 只用 train
   subjects 拟合，禁止 test leakage），rollout 阶段三者均**递归**生成；
4. **free rollout**：H ∈ --eval_rollout_horizons（默认 1,2,4,8,16），中间不使用
   任何真值；逐 horizon 报告 MAE/RMSE/spatial PCC/temporal PCC/R2/variance ratio；
5. **FC**：仅当 rollout 长度 >= --fc_min_length（默认 32）时在 rollout 轨迹上估计
   FC 并比较上三角边，并按 AAL116 网络标签分解为 within-network / between-network
   （prompt §十九、§二十六）；6. **频谱**：可选（--eval_spectral），Welch PSD 低频段。

指标口径（重要）：
  - spatial_pcc：单个时间点上**沿 ROI 维**的空间 pattern 相关（next-state 主口径）；
  - temporal_pcc：rollout 多时间点上**沿时间维**逐 ROI 相关后再对 ROI 平均；
  - r2：相对该任务真值方差的解释比例；variance_ratio = Var(pred)/Var(true)，
    用于发现自回归模型的方差塌缩（越来越平）。
"""
import csv
import json
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.common import (parse_int_list, resolve_forecast_offsets,
                          resolve_task_dims)
from utils.dataloader import NeuroTwinDataLoader

log = logging.getLogger(__name__)

EPS = 1e-8
DEFAULT_ROLLOUT_HORIZONS = (1, 2, 4, 8, 16)
DEFAULT_SPECTRAL_BAND = (0.01, 0.1)   # Hz（低频段）
# 方向准确率死区：真实一阶差分绝对值不超过该阈值的 (anchor, ROI) 视为噪声，
# 不计入 sign 一致率。BOLD 已逐 ROI z-score，默认 0.05 约等于 5% ROI 标准差/TR。
DEFAULT_DIRECTION_DEADZONE = 0.05


# ══════════════════════════════════════════════════════════════════════════
#  §0 公共 helper
# ══════════════════════════════════════════════════════════════════════════

def _subject_mean(values: np.ndarray, subj_idx: np.ndarray) -> np.ndarray:
    """把任务级数值按被试聚合为被试均值 [n_unique]。"""
    uniq = np.unique(subj_idx)
    return np.array([values[subj_idx == s].mean() for s in uniq], dtype=np.float64)


def spectral_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                     tr: float = 2.0, band=DEFAULT_SPECTRAL_BAND) -> dict:
    """Welch PSD 低频段一致性（可选指标，不做训练损失）。

    对每个任务把有效未来 chunk 拼接成 [F, T]，逐 ROI 计算 PSD 后比较：
      - 低频段相对功率（band power / total power）的 MAE 与跨 (task, ROI) 相关；
      - log-PSD 曲线的逐 (task, ROI) 相关（整体频谱形状一致性）。
    """
    try:
        from scipy.signal import welch
    except ImportError:  # pragma: no cover
        return {'available': False, 'reason': 'scipy.signal 不可用'}
    n, f, h, s = pred.shape
    valid = np.asarray(mask, dtype=bool) if mask is not None else np.ones((n, h), bool)
    rel_p, rel_t, corr_psd = [], [], []
    fs = 1.0 / float(tr)
    for i in range(n):
        h_i = int(valid[i].sum())
        if h_i < 2:
            continue
        p = pred[i, :, :h_i, :].reshape(f, -1)
        t = target[i, :, :h_i, :].reshape(f, -1)
        nperseg = min(64, p.shape[-1])
        fp, psd_p = welch(p, fs=fs, nperseg=nperseg, axis=-1)
        _, psd_t = welch(t, fs=fs, nperseg=nperseg, axis=-1)
        m = (fp >= band[0]) & (fp <= band[1])
        if not m.any():
            continue
        rel_p.append(psd_p[:, m].sum(-1) / (psd_p.sum(-1) + EPS))
        rel_t.append(psd_t[:, m].sum(-1) / (psd_t.sum(-1) + EPS))
        lp, lt = np.log10(psd_p + EPS), np.log10(psd_t + EPS)
        lp = lp - lp.mean(-1, keepdims=True)
        lt = lt - lt.mean(-1, keepdims=True)
        corr_psd.append(((lp * lt).sum(-1)
                         / (np.sqrt((lp ** 2).sum(-1) * (lt ** 2).sum(-1)) + EPS)))
    if not rel_p:
        return {'available': False, 'reason': '有效任务不足（需 >= 2 个未来 chunk）'}
    rel_p = np.concatenate(rel_p)
    rel_t = np.concatenate(rel_t)
    corr = np.concatenate(corr_psd)
    r = np.corrcoef(rel_p, rel_t)[0, 1] if rel_p.std() > 0 and rel_t.std() > 0 else float('nan')
    return {'available': True, 'tr': float(tr), 'band_hz': [float(band[0]), float(band[1])],
            'n_points': int(rel_p.size),
            'relative_band_power_mae': float(np.abs(rel_p - rel_t).mean()),
            'relative_band_power_pcc': float(r),
            'log_psd_pcc_mean': float(np.nanmean(corr))}


# ══════════════════════════════════════════════════════════════════════════
#  §0 配置解析（与训练侧 main.py 保持同一口径）
# ══════════════════════════════════════════════════════════════════════════

def eval_horizons(args):
    """free rollout 的 horizon 列表（默认 [1,2,4,8,16]）。"""
    vals = parse_int_list(getattr(args, 'eval_rollout_horizons', None),
                          'eval_rollout_horizons')
    return list(vals) if vals else list(DEFAULT_ROLLOUT_HORIZONS)


def eval_rollout_steps(args, horizons):
    """单次 rollout 的滚动步数 R：0=auto（max(horizons)，FC/频谱启用时抬到 fc_min_length）。"""
    explicit = int(getattr(args, 'eval_rollout_steps', 0) or 0)
    if explicit > 0:
        return explicit
    steps = max(horizons)
    if bool(getattr(args, 'eval_fc', True)) or bool(getattr(args, 'eval_spectral', False)):
        steps = max(steps, int(getattr(args, 'fc_min_length', 32)))
    return steps


# ══════════════════════════════════════════════════════════════════════════
#  §1 逐任务指标与聚合
# ══════════════════════════════════════════════════════════════════════════

def per_task_next_state(pred, target, mask, anchor=None):
    """next-state 逐 (task, offset) 指标数组，形状均为 [N, H]（无效位置 NaN）。

    pred/target: [N, F, H]（单个时间点预测，原始空间）；mask: [N, H]。
    指标：mae / rmse / spatial_pcc（沿 ROI 维相关）/ r2（相对 ROI 方差）。
    若给定 ``anchor`` [N, F]（context 末位 = 预测锚点 x_t），额外统计
    ``delta_direction``：逐 ROI 变化方向 sign(x̂−x_t) 与 sign(x_true−x_t) 的一致比例
    （prompt §二十；delta MAE 与 MAE 在 delta 口径下恒等，故不重复报告）。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致：{pred.shape} vs {target.shape}")
    n, f, h = pred.shape
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.ones((n, h), bool))
    if valid.shape != (n, h):
        raise ValueError(f"mask 形状 {valid.shape} 应为 {(n, h)}")

    err = pred - target
    mae = np.abs(err).mean(axis=1)                       # [N,H]
    rmse = np.sqrt((err ** 2).mean(axis=1))
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1)) + EPS
    pcc = (pc * tc).sum(axis=1) / denom
    r2 = 1.0 - (err ** 2).sum(axis=1) / ((tc ** 2).sum(axis=1) + EPS)

    arrays = {'mae': mae, 'rmse': rmse, 'spatial_pcc': pcc, 'r2': r2}
    if anchor is not None:
        a = np.asarray(anchor, dtype=pred.dtype)[:, :, None]     # [N,F,1]
        agree = np.sign(pred - a) == np.sign(target - a)         # [N,F,H]
        arrays['delta_direction'] = agree.mean(axis=1)           # [N,H]
    for k in arrays:
        arrays[k] = np.where(valid, arrays[k], np.nan)
    return arrays


def per_task_head_metrics(pred, target, mask, anchor,
                          direction_deadzone=DEFAULT_DIRECTION_DEADZONE):
    """单个预测头（Graph-ODE 或 CPM）的逐 horizon 指标 [N, H]。

    与 ``per_task_next_state`` 的区别是显式区分两种差分口径：

    - ``delta_mae_anchor``：``|(ŷ_h − x_t) − (y_h − x_t)|``，在数学上恒等于
      ``MAE``（共同锚点抵消），保留该列只为证明/核对这一点；
    - ``delta_mae_step``：相邻帧一阶差分误差。``Δŷ_1=ŷ_1−x_t``、
      ``Δŷ_h=ŷ_h−ŷ_(h−1)``（真值同理），与训练侧
      ``temporal_difference_huber`` 完全同口径。只有该指标能反映模型是否学到
      局部时间动力学，而不是单纯复制锚点。

    ``direction_accuracy_step`` / ``direction_accuracy_anchor`` 分别对上述两种
    差分统计 sign 一致率；真实差分绝对值 <= ``direction_deadzone`` 的样本不计入，
    避免近零噪声左右方向指标。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致：{pred.shape} vs {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"pred/target 期望 [N,F,H]，收到 {pred.shape}")
    n, f, h = pred.shape
    if anchor is None:
        raise ValueError("方向/差分指标需要 anchor [N,F]（context 末位 x_t）")
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (n, f):
        raise ValueError(f"anchor 形状应为 {(n, f)}，收到 {anchor.shape}")
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.ones((n, h), bool))
    if valid.shape != (n, h):
        raise ValueError(f"mask 形状 {valid.shape} 应为 {(n, h)}")

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    mae = np.abs(err).mean(axis=1)
    rmse = np.sqrt((err ** 2).mean(axis=1))
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1)) + EPS
    spatial_pcc = (pc * tc).sum(axis=1) / denom
    r2 = 1.0 - (err ** 2).sum(axis=1) / ((tc ** 2).sum(axis=1) + EPS)
    var_ratio = np.where(
        target.var(axis=1) > EPS, pred.var(axis=1) / (target.var(axis=1) + EPS),
        np.nan)

    # 相邻帧差分：h=1 用真实锚点，h>=2 用各自轨迹的前一帧
    prev_p = np.concatenate([anchor[:, :, None], pred[:, :, :-1]], axis=2)
    prev_t = np.concatenate([anchor[:, :, None], target[:, :, :-1]], axis=2)
    d_pred = pred - prev_p
    d_true = target - prev_t
    step_valid = valid.copy()
    if h > 1:
        step_valid[:, 1:] &= valid[:, :-1]
    d_err = np.abs(d_pred - d_true)
    delta_mae_step = np.where(
        step_valid, d_err.mean(axis=1), np.nan)
    # 数学恒等列：Δ 相对锚点，与点值 MAE 相同
    delta_mae_anchor = mae.copy()

    eligible_step = (step_valid[:, None, :]
                     & (np.abs(d_true) > float(direction_deadzone)))
    agree_step = (np.sign(d_pred) == np.sign(d_true)) & eligible_step
    n_eligible_step = eligible_step.sum(axis=1)
    da_step = np.where(
        n_eligible_step > 0,
        agree_step.sum(axis=1) / np.maximum(n_eligible_step, 1), np.nan)

    h_delta = pred - anchor[:, :, None]
    h_delta_true = target - anchor[:, :, None]
    eligible_anchor = (valid[:, None, :]
                       & (np.abs(h_delta_true) > float(direction_deadzone)))
    agree_anchor = (np.sign(h_delta) == np.sign(h_delta_true)) & eligible_anchor
    n_eligible_anchor = eligible_anchor.sum(axis=1)
    da_anchor = np.where(
        n_eligible_anchor > 0,
        agree_anchor.sum(axis=1) / np.maximum(n_eligible_anchor, 1), np.nan)

    # 变化幅度比：预测一阶差分 / 真实一阶差分（<1 表示趋平，>1 表示过冲）
    mag_p = np.abs(d_pred).mean(axis=1)
    mag_t = np.abs(d_true).mean(axis=1)
    mag_ratio = np.where(step_valid & (mag_t > EPS), mag_p / (mag_t + EPS), np.nan)

    arrays = {
        'mae': mae, 'rmse': rmse, 'spatial_pcc': spatial_pcc, 'r2': r2,
        'variance_ratio': var_ratio,
        'delta_mae_anchor': delta_mae_anchor,
        'delta_mae_step': delta_mae_step,
        'delta_direction': da_step,
        'direction_accuracy_step': da_step,
        'direction_accuracy_anchor': da_anchor,
        'delta_magnitude_ratio': mag_ratio,
    }
    for key in ('mae', 'rmse', 'spatial_pcc', 'r2', 'delta_mae_anchor'):
        arrays[key] = np.where(valid, arrays[key], np.nan)
    return arrays


def _dual_direction_agreement(cpm, ode, anchor, target, direction_deadzone):
    """逐 (N,H) 统计两头相对锚点方向一致的 ROI 比例（排除近零真实变化）。"""
    anchor3 = anchor[:, :, None]
    eligible = np.abs(target - anchor3) > float(direction_deadzone)   # [N,F,H]
    agree = (np.sign(cpm - anchor3) == np.sign(ode - anchor3)) & eligible
    n_eligible = eligible.sum(axis=1)
    return np.where(n_eligible > 0,
                    agree.sum(axis=1) / np.maximum(n_eligible, 1), np.nan)


def per_task_dual_head_metrics(cpm_pred, ode_pred, target, mask, anchor,
                               direction_deadzone=DEFAULT_DIRECTION_DEADZONE):
    """CPM 与 Graph-ODE 双头的逐 horizon 一致性/质量指标 [N, H]。

    仅看 Gap 降低不足以说明一致性约束有效：两个头可能一起收敛到同一个错误。
    因此同时报告：

    - ``gap``：``mean_ROI |ŷ_CPM − ŷ_ODE|``（L1，与训练侧一致性损失同口径）；
    - ``gap_norm``：Gap / 两头平均 MAE，用于跨臂比较（避免误差尺度变化造成假下降）；
    - ``cpm_mae`` / ``ode_mae``：各自预测质量；
    - ``both_fail_vs_persistence``：同一 (task, horizon) 两头 MAE 都大于
      persistence 基线的比例（“一起预测错误”的直接证据）；
    - ``head_error_corr``：同一 anchor 上逐 ROI 绝对误差的 Pearson 相关
      （误差同向移动的程度，1 表示错误结构几乎相同）；
    - ``oracle_mae``：逐 anchor 取两头较小 MAE，相对单头最优值的下降量
      ``oracle_gain_vs_best_head`` 衡量两头互补性。
    """
    if cpm_pred.shape != ode_pred.shape or cpm_pred.shape != target.shape:
        raise ValueError(
            "双头指标要求 CPM/ODE/target 同形，收到 "
            f"{cpm_pred.shape}/{ode_pred.shape}/{target.shape}")
    if cpm_pred.ndim != 3:
        raise ValueError(f"双头指标期望 [N,F,H]，收到 {cpm_pred.shape}")
    n, f, h = cpm_pred.shape
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (n, f):
        raise ValueError(f"anchor 形状应为 {(n, f)}，收到 {anchor.shape}")
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.ones((n, h), bool))
    if valid.shape != (n, h):
        raise ValueError(f"mask 形状 {valid.shape} 应为 {(n, h)}")

    cpm = np.asarray(cpm_pred, dtype=np.float64)
    ode = np.asarray(ode_pred, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    cpm_abs = np.abs(cpm - tgt)
    ode_abs = np.abs(ode - tgt)
    cpm_mae = cpm_abs.mean(axis=1)
    ode_mae = ode_abs.mean(axis=1)
    pers_mae = np.abs(anchor[:, :, None] - tgt).mean(axis=1)
    gap = np.abs(cpm - ode).mean(axis=1)
    gap_norm = gap / (0.5 * (cpm_mae + ode_mae) + EPS)
    oracle = np.minimum(cpm_mae, ode_mae)
    both_fail = ((cpm_mae > pers_mae) & (ode_mae > pers_mae)).astype(np.float64)
    cpm_win = (cpm_mae < ode_mae).astype(np.float64)

    # 逐 (anchor, horizon) 沿 ROI 的绝对误差相关
    cc = cpm_abs - cpm_abs.mean(axis=1, keepdims=True)
    oc = ode_abs - ode_abs.mean(axis=1, keepdims=True)
    corr_den = np.sqrt((cc ** 2).sum(axis=1) * (oc ** 2).sum(axis=1))
    head_error_corr = np.where(
        corr_den > EPS, (cc * oc).sum(axis=1) / (corr_den + EPS), np.nan)

    arrays = {
        'gap': gap, 'gap_norm': gap_norm,
        'cpm_mae': cpm_mae, 'ode_mae': ode_mae,
        'persistence_mae': pers_mae,
        'both_fail_vs_persistence': both_fail,
        'head_error_corr': head_error_corr,
        'oracle_mae': oracle,
        'oracle_gain_vs_cpm': cpm_mae - oracle,
        'oracle_gain_vs_ode': ode_mae - oracle,
        'cpm_win_rate': cpm_win,
        # 两头相对锚点的变化方向是否一致（不涉及真值），用于判断是否“同错”；
        # 同样排除真实变化接近零的 ROI，避免噪声抬高一致性
        'direction_agreement': _dual_direction_agreement(
            cpm, ode, anchor, target, direction_deadzone),
    }
    for key in ('gap', 'gap_norm', 'cpm_mae', 'ode_mae', 'persistence_mae',
                'both_fail_vs_persistence', 'head_error_corr', 'oracle_mae',
                'oracle_gain_vs_cpm', 'oracle_gain_vs_ode', 'cpm_win_rate',
                'direction_agreement'):
        arrays[key] = np.where(valid, arrays[key], np.nan)
    return arrays


def _step_deltas(pred, target, anchor):
    """相邻帧一阶差分（与 temporal_difference_huber 同口径）。"""
    prev_p = np.concatenate([anchor[:, :, None], pred[:, :, :-1]], axis=2)
    prev_t = np.concatenate([anchor[:, :, None], target[:, :, :-1]], axis=2)
    return pred - prev_p, target - prev_t


def _step_valid(mask):
    """相邻帧差分有效性：h 与 h−1 的真值都有效；h=1 只需自身有效。"""
    mask = np.asarray(mask, dtype=bool)
    valid = mask.copy()
    if mask.shape[1] > 1:
        valid[:, 1:] &= mask[:, :-1]
    return valid


def amplitude_quantile_edges(target, mask, anchor, n_bins=5):
    """用真实相邻帧变化幅度的分位数确定幅度分层边界。

    三臂使用完全相同的 test 锚点与真值，因此边界也相同；返回长度
    ``n_bins+1``（首边界 0、末边界 +inf），供 ``per_task_amplitude_diagnostics``
    对模型与基线做同一套分层。
    """
    if int(n_bins) < 1:
        raise ValueError(f'n_bins 必须 >= 1，收到 {n_bins}')
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    mask = (np.asarray(mask, dtype=bool) if mask is not None
            else np.ones(target.shape[:2], bool))
    _, dt = _step_deltas(target, target, anchor)
    valid = _step_valid(mask)
    vals = np.abs(dt)[np.broadcast_to(valid[:, None, :], dt.shape)]
    if vals.size == 0:
        raise ValueError('真实差分没有有效元素，无法确定幅度分层边界')
    edges = np.quantile(vals, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[0] = 0.0
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges.tolist()


def per_task_amplitude_diagnostics(pred, target, mask, anchor,
                                   amp_edges=None, n_bins=5):
    """逐 anchor 的幅度/动态诊断，回答“方向对但幅度是否被压缩”。

    输出的标量数组均为 ``[N]``，逐 horizon 数组为 ``[N,H]``，分层数组为
    ``[N,B]``：

    - ``delta_std_ratio``：``std(Δŷ)/std(Δy)``（H1–H8 × 全部 ROI 的有效元素），
      <1 表示变化幅度整体偏小；
    - ``delta_std_ratio_h1`` / ``_h2_8``：分别看单步与多步推进的幅度比。
      H1≈1 而 H2–H8<1 说明压缩来自状态推进/条件均值，而非单步 readout；
    - ``delta_pcc`` / ``delta_pcc_h2_8``：``Corr(Δŷ, Δy)``，衡量是否跟随变化过程；
    - ``temporal_variance_ratio``：逐 ROI 在有效未来轨迹上的
      ``Var(ŷ_t..)/Var(y_t..)`` 再对 ROI 平均；<1 表示轨迹方差塌缩；
    - ``trajectory_variance_ratio_flat``：把 H×ROI 展平后的方差比（对照口径）；
    - ``amp_bin_mae``：按真实 ``|Δy|`` 分位分层的 Delta MAE（默认 5 层），
      可直接检查“大幅变化是否尤其欠拟合”。

    ``amp_edges`` 由 :func:`amplitude_quantile_edges` 在真值上预先算好，保证
    模型与基线、不同实验臂使用完全相同的分层边界。
    """
    if pred.shape != target.shape:
        raise ValueError(f'pred/target 形状不一致：{pred.shape} vs {target.shape}')
    if pred.ndim != 3:
        raise ValueError(f'pred/target 期望 [N,F,H]，收到 {pred.shape}')
    n, f, h = pred.shape
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (n, f):
        raise ValueError(f'anchor 形状应为 {(n, f)}，收到 {anchor.shape}')
    mask = (np.asarray(mask, dtype=bool) if mask is not None
            else np.ones((n, h), bool))
    if mask.shape != (n, h):
        raise ValueError(f'mask 形状 {mask.shape} 应为 {(n, h)}')
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dp, dt = _step_deltas(pred, target, anchor)
    step_valid = _step_valid(mask)
    if amp_edges is None:
        amp_edges = amplitude_quantile_edges(target, mask, anchor, n_bins=n_bins)
    amp_edges = np.asarray(amp_edges, dtype=np.float64)
    nb = len(amp_edges) - 1

    keys = ('delta_std_ratio', 'delta_pcc', 'temporal_variance_ratio',
            'trajectory_variance_ratio_flat',
            'delta_std_ratio_h1', 'delta_std_ratio_h2_8',
            'delta_pcc_h1', 'delta_pcc_h2_8',
            'mean_abs_pred_delta', 'mean_abs_true_delta')
    out = {k: np.full(n, np.nan, dtype=np.float64) for k in keys}
    out['delta_std_ratio_h'] = np.full((n, h), np.nan)
    out['delta_pcc_h'] = np.full((n, h), np.nan)
    out['amp_bin_mae'] = np.full((n, nb), np.nan)
    out['amp_bin_count'] = np.zeros((n, nb), dtype=np.float64)
    out['amp_bin_true_abs_mean'] = np.full((n, nb), np.nan)

    for i in range(n):
        valid = step_valid[i]
        if not valid.any():
            continue
        m3 = np.broadcast_to(valid[None, :], (f, h))
        x = dp[i][m3]
        y = dt[i][m3]
        sx, sy = float(np.std(x)), float(np.std(y))
        if sy > EPS:
            out['delta_std_ratio'][i] = sx / sy
        if sx > EPS and sy > EPS:
            out['delta_pcc'][i] = float(np.corrcoef(x, y)[0, 1])
        out['mean_abs_pred_delta'][i] = float(np.abs(x).mean())
        out['mean_abs_true_delta'][i] = float(np.abs(y).mean())

        # 轨迹方差比：逐 ROI 在有效 horizon 上的时间方差
        vh = mask[i]
        if vh.sum() >= 3:
            pv = pred[i][:, vh]
            tv = target[i][:, vh]
            vp, vt = pv.var(axis=1), tv.var(axis=1)
            ok = vt > EPS
            if ok.any():
                out['temporal_variance_ratio'][i] = float(
                    np.mean(vp[ok] / (vt[ok] + EPS)))
            fp = pred[i][:, vh].reshape(-1)
            ft = target[i][:, vh].reshape(-1)
            if ft.var() > EPS:
                out['trajectory_variance_ratio_flat'][i] = float(
                    fp.var() / (ft.var() + EPS))

        # H1 与 H2-8 分开看压缩发生在哪一段
        for key, sel in (('h1', np.array([0])),
                         ('h2_8', np.arange(1, h))):
            if h == 1 and key == 'h2_8':
                continue
            sel = sel[valid[sel]]
            if sel.size == 0:
                continue
            xx = dp[i][:, sel].reshape(-1)
            yy = dt[i][:, sel].reshape(-1)
            syy = float(np.std(yy))
            if syy > EPS:
                out[f'delta_std_ratio_{key}'][i] = float(np.std(xx)) / syy
            if xx.std() > EPS and syy > EPS:
                out[f'delta_pcc_{key}'][i] = float(np.corrcoef(xx, yy)[0, 1])

        # 逐 horizon 的幅度比与 PCC（跨 ROI）
        for hi in range(h):
            if not valid[hi]:
                continue
            xx = dp[i][:, hi]
            yy = dt[i][:, hi]
            if yy.std() > EPS:
                out['delta_std_ratio_h'][i, hi] = xx.std() / yy.std()
            if xx.std() > EPS and yy.std() > EPS:
                out['delta_pcc_h'][i, hi] = float(np.corrcoef(xx, yy)[0, 1])

        # 按真实变化幅度分位分层
        mag = np.abs(y)
        err = np.abs(x - y)
        which = np.digitize(mag, amp_edges[1:-1], right=False)
        for b in range(nb):
            sel = which == b
            cnt = int(sel.sum())
            out['amp_bin_count'][i, b] = cnt
            if cnt:
                out['amp_bin_mae'][i, b] = float(err[sel].mean())
                out['amp_bin_true_abs_mean'][i, b] = float(mag[sel].mean())
    return out


def aggregate_columns(arrays, subj_idx, labels):
    """把 [N, C] 指标数组聚合为 {metric: {label: {...}}}（被试级为主口径）。

    每个 label 报告：被试级 mean/std/median（先被试内平均、再跨被试统计）+
    任务级 pooled mean + 任务数/被试数。
    """
    out = {}
    for metric, arr in arrays.items():
        block = {}
        for c, label in enumerate(labels):
            col = arr[:, c]
            ok = ~np.isnan(col)
            if not ok.any():
                continue
            vals = col[ok]
            subj_vals = _subject_mean(vals, subj_idx[ok])
            block[str(label)] = {
                'mean': float(subj_vals.mean()),
                'std': float(subj_vals.std()),
                'median': float(np.median(subj_vals)),
                'task_mean': float(vals.mean()),
                'n_tasks': int(ok.sum()),
                'n_subjects': int(len(subj_vals)),
            }
        if block:
            out[metric] = block
    return out


def _block_summary(agg, label):
    """把某个 label 的聚合结果整理成报告块（PCC/MAE/RMSE/R2）。"""
    block = {}
    if 'spatial_pcc' in agg and label in agg['spatial_pcc']:
        d = agg['spatial_pcc'][label]
        block.update({'PCC_spatial': d['mean'], 'PCC_spatial_std': d['std'],
                      'PCC_spatial_median': d['median'],
                      'n_tasks': d['n_tasks'], 'n_subjects': d['n_subjects']})
        block['PCC'] = d['mean']          # 兼容 registry/summary 的通用键
        block['PCC_std'] = d['std']
    for metric in ('mae', 'rmse', 'r2'):
        if metric in agg and label in agg[metric]:
            d = agg[metric][label]
            block[metric.upper()] = d['mean']
            block[f'{metric.upper()}_std'] = d['std']
    if 'delta_direction' in agg and label in agg['delta_direction']:
        block['delta_direction'] = agg['delta_direction'][label]['mean']
        block['delta_direction_std'] = agg['delta_direction'][label]['std']
    # 第一阶段新增指标：差分幅度/方向、双头一致性与质量
    for metric, key in (
            ('delta_mae_step', 'delta_MAE_step'),
            ('delta_mae_anchor', 'delta_MAE_anchor'),
            ('direction_accuracy_step', 'direction_accuracy_step'),
            ('direction_accuracy_anchor', 'direction_accuracy_anchor'),
            ('delta_magnitude_ratio', 'delta_magnitude_ratio'),
            ('variance_ratio', 'variance_ratio'),
            ('gap', 'gap'), ('gap_norm', 'gap_norm'),
            ('cpm_mae', 'CPM_MAE'), ('ode_mae', 'ODE_MAE'),
            ('persistence_mae', 'persistence_MAE'),
            ('both_fail_vs_persistence', 'both_fail_vs_persistence'),
            ('head_error_corr', 'head_error_corr'),
            ('oracle_mae', 'oracle_MAE'),
            ('oracle_gain_vs_cpm', 'oracle_gain_vs_cpm'),
            ('oracle_gain_vs_ode', 'oracle_gain_vs_ode'),
            ('cpm_win_rate', 'cpm_win_rate'),
            ('direction_agreement', 'direction_agreement')):
        if metric in agg and label in agg[metric]:
            d = agg[metric][label]
            block[key] = d['mean']
            block[f'{key}_std'] = d['std']
            block[f'{key}_median'] = d['median']
    return block


# ══════════════════════════════════════════════════════════════════════════
#  §2 baselines（persistence / trend / AR(1)）
# ══════════════════════════════════════════════════════════════════════════

def fit_ar1_from_train(base_dataset, train_subjects):
    """用 **train subjects** 的连续序列拟合逐 ROI AR(1)：x_(t+1) = a_i·x_t + b_i。

    最小二乘闭式解（逐 ROI 汇总 Σx/Σy/Σx²/Σxy 后一次求解），不使用任何
    val/test 被试；返回 {'a': [F], 'b': [F], 'n_pairs': int}。
    """
    sx = sy = sxx = sxy = None
    n_pairs = 0
    for sid in train_subjects:
        series = np.asarray(base_dataset.get_series(sid), dtype=np.float64)
        if series.ndim != 2 or series.shape[1] < 2:
            continue
        x, y = series[:, :-1], series[:, 1:]
        if sx is None:
            f = series.shape[0]
            sx = np.zeros(f); sy = np.zeros(f)
            sxx = np.zeros(f); sxy = np.zeros(f)
        sx += x.sum(axis=1); sy += y.sum(axis=1)
        sxx += (x * x).sum(axis=1); sxy += (x * y).sum(axis=1)
        n_pairs += x.shape[1]
    if sx is None or n_pairs == 0:
        raise ValueError("AR(1) 拟合失败：训练集没有可用的连续序列。")
    denom = n_pairs * sxx - sx * sx
    a = np.where(np.abs(denom) > EPS, (n_pairs * sxy - sx * sy) / denom, 0.0)
    b = (sy - a * sx) / float(n_pairs)
    return {'a': a.astype(np.float32), 'b': b.astype(np.float32),
            'n_pairs': int(n_pairs)}


def baseline_next_state(x_last, x_prev, offsets, ar1=None):
    """next-state 的三条 baseline 预测，形状均为 [N, F, H]。

    - persistence：x̂_(t+δ) = x_t；
    - trend：x̂_(t+δ) = x_t + δ·(x_t − x_(t−1))（线性趋势外推）；
    - ar1：逐 ROI 递推 ``x ← a·x + b`` δ 次（仅用 train subjects 拟合的参数）。
    """
    offs = torch.as_tensor(list(offsets), device=x_last.device,
                           dtype=x_last.dtype).view(1, 1, -1)
    slope = x_last - x_prev
    out = {
        'persistence': x_last.unsqueeze(-1).expand(-1, -1, len(offsets)),
        'trend': x_last.unsqueeze(-1) + offs * slope.unsqueeze(-1),
    }
    if ar1 is not None:
        a = torch.as_tensor(ar1['a'], device=x_last.device, dtype=x_last.dtype).view(1, -1)
        b = torch.as_tensor(ar1['b'], device=x_last.device, dtype=x_last.dtype).view(1, -1)
        cur = x_last
        steps = []
        for _ in range(max(int(o) for o in offsets)):
            cur = a * cur + b
            steps.append(cur)
        stacked = torch.stack(steps, dim=1)                     # [N, maxδ, F]
        idx = torch.as_tensor([int(o) - 1 for o in offsets], device=x_last.device)
        out['ar1'] = stacked[:, idx, :].transpose(1, 2)         # [N, F, H]
    return out


def ar1_rollout(ar1, x0, steps):
    """AR(1) 递归 rollout：x̂_(t+k) = a·x̂_(t+k−1) + b，返回 [N, steps, F]。"""
    a = torch.as_tensor(ar1['a'], device=x0.device, dtype=x0.dtype).view(1, -1)
    b = torch.as_tensor(ar1['b'], device=x0.device, dtype=x0.dtype).view(1, -1)
    cur = x0
    outs = []
    for _ in range(int(steps)):
        cur = a * cur + b
        outs.append(cur)
    return torch.stack(outs, dim=1)


def baseline_rollout(x_last, slope, steps, ar1=None):
    """rollout 阶段的 baseline（全部递归生成，与模型 rollout 同一协议）。

    - persistence：未来所有时刻保持最后真实状态；
    - trend：用自身预测继续外推（等价于固定斜率 δ·(x_t − x_(t−1))）；
    - ar1：逐 ROI 递推 AR(1)。
    返回 {name: [N, steps, F]}。
    """
    steps_i = int(steps)
    k = torch.arange(1, steps_i + 1, device=x_last.device, dtype=x_last.dtype)
    out = {
        'persistence': x_last.unsqueeze(1).expand(-1, steps_i, -1).contiguous(),
        'trend': (x_last.unsqueeze(1) + k.view(1, -1, 1) * slope.unsqueeze(1)).contiguous(),
    }
    if ar1 is not None:
        out['ar1'] = ar1_rollout(ar1, x_last, steps_i)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  §3 rollout / FC 指标
# ══════════════════════════════════════════════════════════════════════════

def per_task_rollout(pred, target, mask, horizons):
    """rollout 逐 (task, horizon) 指标数组，形状 [N, len(horizons)]。

    pred/target: [N, R, F]（自回归轨迹，原始空间）；mask: [N, R]。
    每个 horizon h 使用**前 h 步**的轨迹：
      - mae/rmse：第 h 步的逐元素误差（单时间点全脑状态）；
      - spatial_pcc：第 h 步沿 ROI 维的空间 pattern 相关；
      - temporal_pcc：前 h 步逐 ROI 时间轴相关后对 ROI 平均（h < 2 时不计算）；
      - r2：前 h 步展平后的解释比例；
      - variance_ratio：Var(pred)/Var(true)（前 h 步展平；用于发现方差塌缩）。
    """
    if pred.shape != target.shape:
        raise ValueError(f"rollout pred/target 形状不一致：{pred.shape} vs {target.shape}")
    n, r, f = pred.shape
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.ones((n, r), bool))
    hs = [int(h) for h in horizons if 1 <= int(h) <= r]
    if not hs:
        raise ValueError(f"rollout horizons {list(horizons)} 超出可用长度 R={r}")
    # np.cumprod 会把 bool 提升为 int64，直接用作布尔掩码会退化成整数索引
    #（全 1 时等价于取"第 1 号任务"），必须显式转回 bool
    cum_valid = np.cumprod(valid, axis=1).astype(bool)          # [N,R] 前 h 步均有效

    arrays = {k: np.full((n, len(hs)), np.nan) for k in
              ('mae', 'rmse', 'spatial_pcc', 'temporal_pcc', 'r2', 'variance_ratio')}
    for j, h in enumerate(hs):
        ok = cum_valid[:, h - 1]
        if not ok.any():
            continue
        p, t = pred[ok], target[ok]
        step_err = p[:, h - 1, :] - t[:, h - 1, :]
        arrays['mae'][ok, j] = np.abs(step_err).mean(axis=1)
        arrays['rmse'][ok, j] = np.sqrt((step_err ** 2).mean(axis=1))
        pc = p[:, h - 1, :] - p[:, h - 1, :].mean(axis=1, keepdims=True)
        tc = t[:, h - 1, :] - t[:, h - 1, :].mean(axis=1, keepdims=True)
        denom = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1)) + EPS
        arrays['spatial_pcc'][ok, j] = (pc * tc).sum(axis=1) / denom
        pf, tf = p[:, :h, :].reshape(len(p), -1), t[:, :h, :].reshape(len(t), -1)
        err = pf - tf
        tcen = tf - tf.mean(axis=1, keepdims=True)
        arrays['r2'][ok, j] = 1.0 - (err ** 2).sum(axis=1) / ((tcen ** 2).sum(axis=1) + EPS)
        arrays['variance_ratio'][ok, j] = (pf.var(axis=1) + EPS) / (tf.var(axis=1) + EPS)
        if h >= 2:
            pr = p[:, :h, :].transpose(0, 2, 1)                  # [n,F,h]
            tr = t[:, :h, :].transpose(0, 2, 1)
            pcr = pr - pr.mean(axis=2, keepdims=True)
            tcr = tr - tr.mean(axis=2, keepdims=True)
            dr = np.sqrt((pcr ** 2).sum(axis=2) * (tcr ** 2).sum(axis=2)) + EPS
            arrays['temporal_pcc'][ok, j] = ((pcr * tcr).sum(axis=2) / dr).mean(axis=1)
    return arrays


def _fc_batch(x):
    """x [N, F, T] → Pearson FC [N, F, F]（沿时间轴标准化后求相关）。"""
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-6)
    return np.einsum('nft,ngt->nfg', x, x) / x.shape[-1]


# AAL116 脑区 → 功能网络标签（data/AAL116.xlsx「对应网络」列），用于把 FC
# 分解为 within-network / between-network（prompt §十九、§二十六）。
DEFAULT_AAL_FILE = Path(__file__).resolve().parents[1] / 'data' / 'AAL116.xlsx'
_aal_networks_cache = {}


def load_aal_networks(n_rois=116, aal_file=None):
    """读取 AAL116 的「脑区 → 功能网络」映射，返回长度 n_rois 的网络名列表。

    文件缺失、缺少「对应网络」列或行数与 ``n_rois`` 不一致时返回 ``None``（调用方
    据此跳过网络级 FC），绝不对缺失标签做任何猜测。结果按 (路径, n_rois) 缓存。
    """
    path = Path(aal_file) if aal_file else DEFAULT_AAL_FILE
    key = (str(path), int(n_rois))
    if key in _aal_networks_cache:
        return _aal_networks_cache[key]
    nets = None
    try:
        import pandas as pd
        df = pd.read_excel(path)
        if '对应网络' in df.columns and len(df) == int(n_rois):
            nets = [str(v).strip() for v in df['对应网络'].tolist()]
        else:
            log.warning('网络级 FC 跳过：%s 缺少「对应网络」列或行数(%d) != n_rois(%d)',
                        path, len(df), int(n_rois))
    except Exception as exc:            # 文件缺失/pandas 不可用：仅告警，不中断评估
        log.warning('网络级 FC 跳过：读取 AAL116 网络标签失败（%s）：%s', path, exc)
    _aal_networks_cache[key] = nets
    return nets


def _fc_edge_metrics(fc_p, fc_t, subj_idx=None):
    """给定边上 FC（``fc_p``/``fc_t`` 均 [N, E]）计算 FC 误差与逐边跨任务 PCC。"""
    if fc_p.shape[1] == 0:
        return {'n_edges': 0}
    diff = fc_p - fc_t
    pc = fc_p - fc_p.mean(axis=0, keepdims=True)
    tc = fc_t - fc_t.mean(axis=0, keepdims=True)
    edge_pcc = ((pc * tc).sum(axis=0)
                / (np.sqrt((pc ** 2).sum(axis=0) * (tc ** 2).sum(axis=0)) + EPS))
    out = {'n_edges': int(fc_p.shape[1]),
           'fc_mae': float(np.abs(diff).mean()),
           'fc_rmse': float(np.sqrt((diff ** 2).mean())),
           'edge_pcc_mean': float(edge_pcc.mean()),
           'edge_pcc_std': float(edge_pcc.std())}
    if subj_idx is not None:
        per_task = np.abs(diff).mean(axis=1)
        subj_vals = _subject_mean(per_task, subj_idx)
        out['fc_mae_subject_mean'] = float(subj_vals.mean())
        out['fc_mae_subject_std'] = float(subj_vals.std())
    return out


def fc_rollout_metrics(pred, target, length, subj_idx=None, networks=None):
    """在 rollout 前 length 步上估计 FC 并比较上三角非对角边。

    pred/target: [N, R, F]；要求 R >= length（不足时返回 available=False，prompt §二十六：
    H 太短不计算 FC）。整体返回 fc_mae / fc_rmse / edge_pcc_mean（逐边跨任务相关再对边
    平均）与 subject 级 fc_mae；若 AAL116 网络标签可用，另返回 ``network`` 块，按
    within-network / between-network 两条边集合分别给出同口径指标（prompt §十九）。
    """
    n, r, f = pred.shape
    if length > r:
        return {'available': False,
                'reason': f'rollout 长度 R={r} < fc_min_length={length}'}
    if n < 2:
        return {'available': False, 'reason': '有效任务数 < 2'}
    iu = np.triu_indices(f, k=1)
    p = pred[:, :length, :].transpose(0, 2, 1).reshape(n, f, length)
    t = target[:, :length, :].transpose(0, 2, 1).reshape(n, f, length)
    fc_p = _fc_batch(p)[:, iu[0], iu[1]]
    fc_t = _fc_batch(t)[:, iu[0], iu[1]]
    out = {'available': True, 'n_tasks': int(n), 'length': int(length)}
    out.update(_fc_edge_metrics(fc_p, fc_t, subj_idx=subj_idx))

    net = networks if networks is not None else load_aal_networks(f)
    if net is not None and len(net) == f:
        net_arr = np.asarray(net)
        within = net_arr[iu[0]] == net_arr[iu[1]]      # 同网络边（含同半球/跨半球）
        out['network'] = {
            'atlas': 'AAL116',
            'within': _fc_edge_metrics(fc_p[:, within], fc_t[:, within],
                                       subj_idx=subj_idx),
            'between': _fc_edge_metrics(fc_p[:, ~within], fc_t[:, ~within],
                                        subj_idx=subj_idx),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
#  §3.5 ROI 置换重要性（--feature_importance）
# ══════════════════════════════════════════════════════════════════════════

def bh_fdr(pvalues):
    """Benjamini–Hochberg FDR 校正，返回与输入同序的 q 值（含单调化）。"""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind='mergesort')
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]          # 保证 q 单调不减
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _masked_mae(pred, target, mask=None):
    """next-state MAE（pred/target [B,F,H]；给定 mask [B,H] 时只统计有效位置）。"""
    err = (pred - target).abs()
    if mask is None:
        return float(err.mean())
    m = mask.unsqueeze(1)                              # [B,1,H]
    return float((err * m).sum() / (float(m.sum()) * pred.shape[1] + EPS))


def _onesided_t_pvalue(t, df):
    """单侧（重要性 > 0）t 检验 p 值。"""
    from scipy import stats
    return float(1.0 - stats.t.cdf(t, df))


@torch.no_grad()
def roi_permutation_importance(model, data_loader, device, max_tasks=256,
                               n_permutations=0, seed=2024, fdr_alpha=0.05):
    """逐 ROI 置换重要性（next_timepoint 口径）。

    与旧 ``ModelAnalyzer.compute_feature_importance`` 的置换重要性同义（ΔMAE 增量），
    但改为本任务的 ``x [B,F,1,K] → ŷ [B,F,H]`` 形状：

      - 先冻结一个**确定性任务子集**（前 ``max_tasks`` 个任务），只在其上统计以控制
        代价；MAE 在全部 forecast offset（H 个）上取平均，即重要性针对 next-state
        整体而非单个 offset；
      - 对每个 ROI i，逐 batch 把 ``x[:, i]`` 沿**样本维** randperm 置换（保持该 ROI
        的边缘分布、破坏其与其余 ROI 的协变关系），重跑前向得到 ΔMAE = MAE_perm − MAE_base；
      - ``n_permutations=0``（默认）：单次置换 + 逐 batch 单样本 t 检验近似（等价旧
        'fast' 模式），自由度为 batch 数 − 1；
      - ``n_permutations>0``：额外做 n_permutations 次置换检验，p 值为经验零分布中
        不小于观测增量的比例（等价旧 'permutation' 模式）；
      - p 值再做 BH-FDR 校正，q < ``fdr_alpha`` 的 ROI 记入 ``significant_roi_indices``
        （**1 基** AAL 索引，便于直接对接 AAL116 图谱）。

    置换跨样本进行，因此要求 batch 内样本数 >= 2（``eval_batch_size`` 需调大）且同一
    batch 内 context 长度一致（评估口径下均为 ``eval_context_length``）；不满足时显式
    报错，不做静默兜底。返回 numpy/标量组成的 dict。
    """
    model.eval()
    xs, ys, scs, paths, masks = [], [], [], [], []
    n_collected = 0
    for batch in data_loader:
        x = batch['x'].to(device, non_blocking=True)
        if x.shape[0] < 2:
            raise ValueError(
                f'置换重要性要求 batch 内样本数 >= 2（收到 {x.shape[0]}），'
                f'请调大 eval_batch_size 后重试。')
        y = batch['y'].to(device, non_blocking=True)
        sc = batch['sc'].to(device, non_blocking=True)
        patho = batch.get('pathology_score', None)
        patho = patho.to(device, non_blocking=True) if patho is not None else None
        mask = batch.get('target_mask', None)
        mask = None if mask is None else mask.to(device, non_blocking=True)
        take = x.shape[0]
        if max_tasks > 0:
            take = min(take, max_tasks - n_collected)
        xs.append(x[:take]); ys.append(y[:take]); scs.append(sc[:take])
        paths.append(None if patho is None else patho[:take])
        masks.append(None if mask is None else mask[:take])
        n_collected += take
        if max_tasks > 0 and n_collected >= max_tasks:
            break
    if n_collected < 2:
        raise ValueError(f'置换重要性可用任务数不足（{n_collected} < 2），无法统计。')

    n_rois = xs[0].shape[1]
    baseline_maes = np.array(
        [_masked_mae(model(x, sc, p)[0][:, :, :, 0], y, m)
         for x, y, sc, p, m in zip(xs, ys, scs, paths, masks)], dtype=np.float64)
    baseline_mae = float(baseline_maes.mean())

    gen = torch.Generator().manual_seed(int(seed))
    roi_importance = np.zeros(n_rois, dtype=np.float64)
    roi_pvalues = np.ones(n_rois, dtype=np.float64)
    roi_std_errors = np.zeros(n_rois, dtype=np.float64)
    for roi in tqdm(range(n_rois), desc='[ROI importance]', leave=False):
        deltas = np.empty(len(xs), dtype=np.float64)
        for j, (x, y, sc, p, m) in enumerate(zip(xs, ys, scs, paths, masks)):
            perm = torch.randperm(x.shape[0], generator=gen).to(x.device)
            x_perm = x.clone()
            x_perm[:, roi] = x_perm[perm, roi]         # 沿样本维置换该 ROI 的 context
            deltas[j] = _masked_mae(model(x_perm, sc, p)[0][:, :, :, 0], y, m) \
                - baseline_maes[j]
        obs = float(deltas.mean())
        roi_importance[roi] = obs
        se = (float(deltas.std(ddof=1) / np.sqrt(deltas.size))
              if deltas.size > 1 else 0.0)
        roi_std_errors[roi] = se
        if n_permutations <= 0:
            if deltas.size > 1 and se > 0:
                roi_pvalues[roi] = _onesided_t_pvalue(obs / se, deltas.size - 1)
        else:
            null = np.empty(n_permutations, dtype=np.float64)
            for k in range(n_permutations):
                acc = np.empty(len(xs), dtype=np.float64)
                for j, (x, y, sc, p, m) in enumerate(zip(xs, ys, scs, paths, masks)):
                    perm = torch.randperm(x.shape[0], generator=gen).to(x.device)
                    x_perm = x.clone()
                    x_perm[:, roi] = x_perm[perm, roi]
                    acc[j] = _masked_mae(model(x_perm, sc, p)[0][:, :, :, 0], y, m) \
                        - baseline_maes[j]
                null[k] = acc.mean()
            roi_pvalues[roi] = max(float((null >= obs).mean()),
                                   1.0 / (n_permutations + 1))

    q = bh_fdr(roi_pvalues)
    significant = (np.flatnonzero(q < float(fdr_alpha)) + 1).tolist()
    return {
        'roi_importance': roi_importance,
        'roi_pvalues': roi_pvalues,
        'roi_pvalues_fdr': q,
        'roi_std_errors': roi_std_errors,
        'roi_importance_rank': np.argsort(roi_importance)[::-1],
        'baseline_mae': baseline_mae,
        'method': 'permutation' if n_permutations > 0 else 'fast',
        'n_permutations': int(n_permutations),
        'n_tasks': int(n_collected),
        'n_rois': int(n_rois),
        'fdr_alpha': float(fdr_alpha),
        'significant_roi_indices': significant,
        'seed': int(seed),
    }


def write_feature_importance(out_dir, split, result, offsets):
    """落盘 ``feature_importance_<split>.json``（键与旧 evaluate 产物兼容）。

    ``roi_importance`` / ``roi_pvalues`` / ``roi_std_errors`` / ``roi_importance_rank`` /
    ``baseline_mae`` / ``method`` 沿用旧文件名与键名，供
    analysis/visualize_roi_importance.py 与 analysis/visualize_sig_region.py 直接消费；
    另加 FDR 校正 q 值与显著 ROI（1 基 AAL 索引）供显著脑区可视化使用。
    """
    payload = OrderedDict(
        (k, v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in result.items())
    payload['task_mode'] = 'next_timepoint'
    payload['split'] = split
    payload['offsets'] = [int(o) for o in offsets]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f'feature_importance_{split}.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    log.info('[next_timepoint] ROI 置换重要性已写入: %s（method=%s, n_tasks=%d, '
             '显著 ROI %d 个 @FDR<%.2f）', path, payload['method'], payload['n_tasks'],
             len(payload['significant_roi_indices']), payload['fdr_alpha'])
    return payload


# ══════════════════════════════════════════════════════════════════════════
#  §4 评估主流程
# ══════════════════════════════════════════════════════════════════════════

def build_next_point_loader(args, split, refresh_split_manifest=False,
                            condition_bundle=None):
    """按 checkpoint 的训练配置重建 next_timepoint DataLoader（被试级切分 manifest 复用）。

    ``condition_bundle`` 为 checkpoint 的 ``config.condition``；提供时条件统计量走
    **inherit**（逐位复用训练期 train-only 快照），不在评估侧重拟合。

    ``split`` 额外支持 ``train``：用与 val/test 相同的确定性 anchor / context
    协议构造**训练被试**评估视图（no rollout），仅供概率校准的常数方差参照等
    只读分析使用，不参与模型选择。
    """
    from main import condition_loader_kwargs, parse_pathology_fields
    dims = resolve_task_dims(args)
    if dims['task_mode'] != 'next_timepoint':
        raise ValueError("build_next_point_loader 仅支持 task_mode=next_timepoint 的配置快照")
    horizons = eval_horizons(args)
    loader = NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode, batch_size=args.batch_size,
        **condition_loader_kwargs(args, condition_bundle),
        pathology_input_dim=args.pathology_input_dim,
        pathology_fields=parse_pathology_fields(args.pathology_fields),
        pathology_missing=args.pathology_missing,
        clinical_file=args.clinical_file,
        total_windows=args.total_windows, seq_len=args.seq_len,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        eval_batch_size=args.eval_batch_size,
        refresh_split_manifest=bool(getattr(args, 'refresh_split_manifest', False)
                                    or refresh_split_manifest),
        task_mode='next_timepoint',
        bold_source=getattr(args, 'bold_source', 'auto'),
        random_context=getattr(args, 'random_context', True),
        random_cutoff=getattr(args, 'random_cutoff', True),
        train_samples_per_subject=int(getattr(args, 'train_samples_per_subject', 0)),
        subject_cache_size=int(getattr(args, 'subject_cache_size', 256)),
        sampling_seed=int(getattr(args, 'sampling_seed', 2024)),
        context_min=args.context_min, context_max=args.context_max,
        context_lengths=(parse_int_list(args.context_lengths, 'context_lengths')
                         if getattr(args, 'context_lengths', None) else None),
        forecast_offsets=resolve_forecast_offsets(args),
        eval_context_length=(int(getattr(args, 'eval_context_length', 0) or 0)
                             or int(args.context_max)),
        eval_anchors_per_subject=int(getattr(args, 'eval_anchors_per_subject', 16)),
        eval_rollout_tasks_per_subject=int(
            getattr(args, 'eval_rollout_tasks_per_subject', 2)),
        eval_rollout_steps=eval_rollout_steps(args, horizons),
        train_rollout_steps=0,
        eval_all_future_steps=max(int(getattr(args, 'cpm_horizon', 0) or 0),
                                  int(getattr(args, 'dynamics_metric_horizon', 0) or 0)),
        eval_anchor_min_future_steps=int(getattr(args, 'eval_anchor_min_future_steps', 0)),
        # graph_ode 动态路线的数据参数（旧快照缺省时与 v5 行为一致）
        num_rois=int(getattr(args, 'num_rois', 116)),
        graph_source=str(getattr(args, 'graph_source', 'legacy_mask')),
        calibration_length=int(getattr(args, 'calibration_length', 0) or 0),
        control_manifest=(getattr(args, 'control_manifest', '') or None),
    )
    if split == 'test':
        return loader.get_test(), loader.get_test_subjects(), loader
    if split == 'val':
        return loader.get_val(), loader.get_val_subjects(), loader
    if split == 'train':
        from torch.utils.data import DataLoader
        from utils.dataloader import NextPointEvalView
        subjects = loader.get_train_subjects()
        indices = loader._expand_subjects_to_sample_indices(subjects)
        view = NextPointEvalView(
            loader.base_dataset, indices,
            context_length=loader.eval_context_length,
            anchors_per_subject=loader.eval_anchors_per_subject,
            rollout_tasks_per_subject=0, rollout_steps=0,
            all_future_steps=loader.eval_all_future_steps,
            anchor_min_future_steps=loader.eval_anchor_min_future_steps)
        kwargs = dict(batch_size=loader.eval_batch_size, shuffle=False,
                      num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory)
        if loader.num_workers > 0:
            kwargs.update(persistent_workers=loader.persistent_workers,
                          prefetch_factor=loader.prefetch_factor)
        return DataLoader(view, **kwargs), subjects, loader
    raise ValueError(f"split 仅支持 train/val/test，收到 '{split}'")


@torch.no_grad()
def run_inference(model, data_loader, device, offsets, rollout_steps,
                  ar1=None, eval_rollout=True, collect_cpm=False,
                  dynamics_horizon=None):
    """跑一遍 next_state 评估集，收集模型/基线预测与 rollout 子集。

    返回 dict（numpy）：
        pred/target   [N, F, H]        模型与真值的 next-state 预测（原始空间）
        mask          [N, H]           目标有效性
        baselines     {name: [N, F, H]} persistence / trend / ar1
        x_last/x_prev [N, F]           context 末位/次末位（供核对）
        subj_ids/cutoff/context_len/hamd
        roll_pred/roll_target/roll_baselines [n, R, F]（仅 rollout 锚点）
        roll_mask [n, R] / roll_subj [n] / roll_cutoff [n]
        cpm_pred [N, F, Hc] / cpm_target [N, F, R] / cpm_mask [N, R]
                                       （仅 collect_cpm=True：TFM 的 CPM 头输出，
                                         要求评估视图为所有 anchor 提供 future）
        ode_pred [N, F, Hc]            Graph-ODE 动态头轨迹（原始空间；
                                       仅 graph_ode 且 forward 返回 dynamics_pred 时）
        future [N, F, R] / future_mask [N, R]
                                       评估视图为所有 anchor 提供的未来真值
    """
    model.eval()
    from main import _index_condition, dynamics_extras
    dyn_h = int(dynamics_horizon or getattr(model, 'cpm_horizon', 0) or 8)
    chunk = defaultdict(list)
    for batch in tqdm(data_loader, desc='[NextPoint eval]', leave=False):
        x = batch['x'].to(device, non_blocking=True)              # [B,F,1,K]
        y = batch['y'].to(device, non_blocking=True)              # [B,F,H]
        x_last = batch['x_last'].to(device, non_blocking=True)    # [B,F]
        x_prev = x[:, :, 0, -2].contiguous()                      # [B,F]
        mask = batch.get('target_mask', None)
        mask = (torch.ones(y.shape[:2], device=device) if mask is None
                else mask.to(device, non_blocking=True))
        sc = batch['sc'].to(device, non_blocking=True)
        pathology = batch.get('pathology_score', None)
        if pathology is not None:
            pathology = pathology.to(device, non_blocking=True)
        # 字段级条件注入（与 legacy 数值通路互斥；由 Dataset 是否产出 cond_z 决定）
        cond = None
        if 'cond_z' in batch:
            cond = {
                'cond_z': batch['cond_z'].to(device, non_blocking=True),
                'cond_code': batch['cond_code'].to(device, non_blocking=True),
                'cond_mask': batch['cond_mask'].to(device, non_blocking=True),
            }
            if 'subject_index' in batch:
                cond['subject_index'] = batch['subject_index'].to(device, non_blocking=True)
        bcal, bctl = dynamics_extras(batch, device)

        pred, aux_info = model(x, sc, pathology, condition=cond,
                               controls=(bctl[:, :dyn_h] if bctl is not None else None),
                               calibration_bold=bcal,
                               **({'dynamics_horizon': dyn_h}
                                  if dynamics_horizon is not None else {}))
        pred = pred[:, :, :, 0]                                   # [B,F,H]
        bases = baseline_next_state(x_last, x_prev, offsets, ar1=ar1)

        chunk['pred'].append(pred.float().cpu().numpy())
        chunk['target'].append(y.float().cpu().numpy())
        chunk['mask'].append(mask.float().cpu().numpy())
        chunk['x_last'].append(x_last.float().cpu().numpy())
        chunk['x_prev'].append(x_prev.float().cpu().numpy())
        for name, v in bases.items():
            chunk[f'base_{name}'].append(v.float().cpu().numpy())
        chunk['subj'].extend([str(s) for s in batch['subj_id']])
        chunk['cutoff'].append(batch['cutoff'].numpy())
        chunk['ctx_len'].append(batch['context_len'].numpy())
        if pathology is not None:
            p_np = pathology.detach().cpu().numpy()
            chunk['hamd'].append(np.asarray(p_np).reshape(p_np.shape[0], -1)[:, 0])
        if cond is not None:
            # 逐 anchor 的**矩阵**：绝不 reshape(-1)，否则行数会变成 anchor 数的 K 倍
            # （把 [N, K] 当成 [N] 读是经典口径错误，方案 §14 明确要求避免）
            for key in ('cond_z', 'cond_code', 'cond_mask'):
                chunk[key].append(cond[key].detach().cpu().numpy())
            if 'cond_raw' in batch:
                chunk['cond_raw'].append(batch['cond_raw'].numpy())
            if 'subject_index' in cond:
                chunk['subject_index'].append(
                    cond['subject_index'].detach().cpu().numpy())

        # ---- 未来真值与 Graph-ODE 轨迹（每次前向已计算，不额外推理）----
        if 'future' in batch:
            chunk['future'].append(batch['future'].float().numpy())       # [B,F,R]
            chunk['future_mask'].append(batch['future_mask'].float().numpy())  # [B,R]
        ode = (aux_info.get('dynamics_pred', None)
               if isinstance(aux_info, dict) else None)
        if torch.is_tensor(ode):
            chunk['ode_pred'].append(ode.float().cpu().numpy())           # [B,F,H]
        # Graph-ODE VarianceHead 的逐 ROI/horizon 输出。std 已由模型换算到
        # 数据尺度；logvar 保留归一化空间，供校准诊断复核。旧模型/其他路线没有
        # 这两个键时不写入，保存端以空数组占位保持 npz 字段稳定。
        for key, aux_key in (('dynamics_std', 'dynamics_std'),
                             ('dynamics_logvar_norm', 'dynamics_logvar_norm')):
            val = (aux_info.get(aux_key, None)
                   if isinstance(aux_info, dict) else None)
            if torch.is_tensor(val):
                chunk[key].append(val.float().cpu().numpy())

        # ---- CPM 全 horizon 预测（TFM；评估视图须为所有 anchor 提供 future）----
        if collect_cpm:
            if 'future' not in batch:
                raise ValueError(
                    "collect_cpm=True 要求评估视图为所有 anchor 提供 future 真值"
                    "（NeuroTwinDataLoader 的 eval_all_future_steps > 0）")
            cpm = (aux_info.get('cpm_pred', None)
                   if isinstance(aux_info, dict) else None)
            if not torch.is_tensor(cpm):
                raise ValueError("模型未输出 cpm_pred（collect_cpm 仅适用于 TFM）")
            chunk['cpm_pred'].append(cpm.float().cpu().numpy())   # [B,F,Hc]
            chunk['cpm_target'].append(batch['future'].float().numpy())   # [B,F,R]
            chunk['cpm_mask'].append(batch['future_mask'].float().numpy())  # [B,R]

        # ---- free rollout（仅 rollout 锚点；模型自由滚动，中间不使用真值）----
        if eval_rollout and rollout_steps > 0 and 'rollout_flag' in batch:
            # rf 留在 CPU（batch['future']/'cutoff' 等仍是 CPU 张量）；GPU 张量用 rf_gpu
            rf = batch['rollout_flag'].bool()
            rf_gpu = rf.to(device)
            if rf_gpu.any():
                hist = x[rf_gpu][:, :, 0, :].transpose(1, 2).contiguous()   # [n,K,F]
                roll = model.rollout_next_states(
                    hist, sc[rf_gpu], (pathology[rf_gpu] if pathology is not None else None),
                    steps=rollout_steps,
                    condition=(_index_condition(cond, rf_gpu) if cond else None),
                    controls=(bctl[rf_gpu][:, :rollout_steps]
                              if bctl is not None else None),
                    calibration_bold=(bcal[rf_gpu] if bcal is not None else None),
                )                                                   # [n,R,F]
                base_roll = baseline_rollout(x_last[rf_gpu], (x_last - x_prev)[rf_gpu],
                                             rollout_steps, ar1=ar1)
                chunk['roll_pred'].append(roll.float().detach().cpu().numpy())
                chunk['roll_target'].append(
                    batch['future'][rf].transpose(1, 2).contiguous().float().numpy())
                chunk['roll_mask'].append(batch['future_mask'][rf].float().numpy())
                for name, v in base_roll.items():
                    chunk[f'roll_base_{name}'].append(v.float().cpu().numpy())
                chunk['roll_subj'].extend(
                    [str(s) for s, keep in zip(batch['subj_id'], rf.tolist()) if keep])
                chunk['roll_cutoff'].append(batch['cutoff'][rf].numpy())

    data = {
        'pred': np.concatenate(chunk['pred'], axis=0),
        'target': np.concatenate(chunk['target'], axis=0),
        'mask': np.concatenate(chunk['mask'], axis=0),
        'x_last': np.concatenate(chunk['x_last'], axis=0),
        'x_prev': np.concatenate(chunk['x_prev'], axis=0),
        'subj_ids': np.array(chunk['subj'], dtype=object),
        'cutoff': np.concatenate(chunk['cutoff'], axis=0),
        'context_len': np.concatenate(chunk['ctx_len'], axis=0),
        'hamd': (np.concatenate(chunk['hamd'], axis=0) if chunk['hamd']
                 else np.full(len(chunk['subj']), np.nan, dtype=np.float32)),
        'baselines': {name: np.concatenate(chunk[f'base_{name}'], axis=0)
                      for name in ('persistence', 'trend', 'ar1')
                      if chunk[f'base_{name}']},
        'ode_pred': (np.concatenate(chunk['ode_pred'], axis=0)
                     if chunk['ode_pred'] else None),
        'dynamics_std': (np.concatenate(chunk['dynamics_std'], axis=0)
                         if chunk['dynamics_std'] else None),
        'dynamics_logvar_norm': (np.concatenate(chunk['dynamics_logvar_norm'], axis=0)
                                 if chunk['dynamics_logvar_norm'] else None),
        'future': (np.concatenate(chunk['future'], axis=0)
                   if chunk['future'] else None),
        'future_mask': (np.concatenate(chunk['future_mask'], axis=0)
                        if chunk['future_mask'] else None),
        'roll_pred': (np.concatenate(chunk['roll_pred'], axis=0)
                      if chunk['roll_pred'] else None),
        'roll_target': (np.concatenate(chunk['roll_target'], axis=0)
                        if chunk['roll_target'] else None),
        'roll_mask': (np.concatenate(chunk['roll_mask'], axis=0)
                      if chunk['roll_mask'] else None),
        'roll_baselines': {name: np.concatenate(chunk[f'roll_base_{name}'], axis=0)
                           for name in ('persistence', 'trend', 'ar1')
                           if chunk[f'roll_base_{name}']},
        'roll_subj': (np.array(chunk['roll_subj'], dtype=object)
                      if chunk['roll_subj'] else None),
        'roll_cutoff': (np.concatenate(chunk['roll_cutoff'], axis=0)
                        if chunk['roll_cutoff'] else None),
    }
    n_anchor = len(data['subj_ids'])
    for key in ('future', 'future_mask', 'ode_pred', 'cpm_pred',
                'dynamics_std', 'dynamics_logvar_norm'):
        arr = data.get(key)
        if arr is not None and arr.shape[0] != n_anchor:
            raise AssertionError(
                f'{key} 行数 {arr.shape[0]} != anchor 数 {n_anchor}；'
                '评估视图必须为每个 anchor 提供同形未来真值/轨迹。')

    # ---- 字段级条件矩阵（按**字段名**读取，绝不假设第 0 列是 HAMD）----
    conditioner = getattr(model, 'conditioner', None)
    fields = list(conditioner.names) if conditioner is not None else []
    data['condition_fields'] = fields
    if chunk['cond_z']:
        n_anchor = len(chunk['subj'])
        for key in ('cond_z', 'cond_code', 'cond_mask', 'cond_raw', 'subject_index'):
            if not chunk[key]:
                continue
            arr = np.concatenate(chunk[key], axis=0)
            if arr.shape[0] != n_anchor:
                raise AssertionError(
                    f'条件数组 {key} 行数 {arr.shape[0]} != anchor 数 {n_anchor}；'
                    '多字段评估必须保持 [N, K]，把 [N, K] 展平成 N*K 会让每个被试'
                    '被计成 K 次（方案 §14 明确禁止 reshape(-1)）')
            data[key] = arr
        if chunk['cond_raw'] and 'HAMD' in fields:
            data['hamd'] = data['cond_raw'][:, fields.index('HAMD')].astype(np.float64)
    if collect_cpm:
        data['cpm_pred'] = (np.concatenate(chunk['cpm_pred'], axis=0)
                            if chunk['cpm_pred'] else None)
        data['cpm_target'] = (np.concatenate(chunk['cpm_target'], axis=0)
                              if chunk['cpm_target'] else None)
        data['cpm_mask'] = (np.concatenate(chunk['cpm_mask'], axis=0)
                            if chunk['cpm_mask'] else None)
    return data


def _subject_metric(array, subj_idx, col):
    """取某列的逐任务指标 → 被试级均值（无效任务由 NaN 自然排除）。"""
    col_vals = array[:, col]
    ok = ~np.isnan(col_vals)
    if not ok.any():
        return np.array([])
    return _subject_mean(col_vals[ok], subj_idx[ok])


def _hamd_association(data, arrays, subj_idx, labels):
    """被试级 spatial PCC 与 HAMD 的关联（仅 finetune 评估可用）。"""
    hamd = np.asarray(data['hamd'], dtype=np.float64).reshape(-1)
    if not np.isfinite(hamd).any():
        return {'available': False, 'reason': '无 HAMD（pretrain 或缺失）'}
    out = {'available': True}
    for i, label in enumerate(labels):
        col = arrays['spatial_pcc'][:, i]
        ok = (~np.isnan(col)) & np.isfinite(hamd)
        if ok.sum() < 3:
            continue
        x = _subject_mean(col[ok], subj_idx[ok])
        y = _subject_mean(hamd[ok], subj_idx[ok])
        if x.std() > 0 and y.std() > 0:
            out[f'corr_hamd_PCC_{label}'] = float(np.corrcoef(x, y)[0, 1])
        out[f'n_subjects_{label}'] = int(len(x))
    return out


def save_predictions_npz(path, **arrays):
    """保存评估预测为 ``np.savez_compressed``，并保证 ``allow_pickle=False`` 可读。

    ``condition_fields`` 等字段历史上以 ``dtype=object`` 落盘，读取时必须
    ``allow_pickle=True``。这里统一把 object/字符串字段转成固定宽度字符串数组，
    数值字段原样保存；``None`` 跳过（调用方对可选字段传 ``[]`` 以保持键存在）。
    """
    payload = {}
    for key, value in arrays.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.dtype == object:
            arr = arr.astype(str)
        payload[key] = arr
    np.savez_compressed(path, **payload)
    return path


def evaluate_split(model, data_loader, args, device, out_dir, split,
                   eval_rollout=True, eval_fc=True, eval_spectral=False,
                   spectral_tr=2.0, save_arrays=False, n_params=None,
                   extra_report=None, ar1=None, compute_importance=False,
                   importance_max_tasks=256, importance_permutations=0,
                   importance_fdr_alpha=0.05, direction_deadzone=None):
    """单个 split 的完整 next_timepoint 评估，落盘 metrics_<split>.json 与两个 csv。

    ``compute_importance=True`` 时额外做 ROI 置换重要性并落盘
    ``feature_importance_<split>.json``（见 ``roi_permutation_importance``）。
    """
    out_dir = Path(out_dir)
    offsets = resolve_forecast_offsets(args)
    horizons = eval_horizons(args)
    rollout_steps = eval_rollout_steps(args, horizons)
    fc_min_len = int(getattr(args, 'fc_min_length', 32))
    cpm_horizon = int(getattr(args, 'cpm_horizon', 0) or 0)
    if direction_deadzone is None:
        direction_deadzone = float(getattr(args, 'direction_deadzone',
                                           DEFAULT_DIRECTION_DEADZONE))
    direction_deadzone = float(direction_deadzone)
    if direction_deadzone < 0:
        raise ValueError(f'direction_deadzone 必须 >= 0，收到 {direction_deadzone}')
    data = run_inference(model, data_loader, device, offsets, rollout_steps,
                         ar1=ar1, eval_rollout=eval_rollout,
                         collect_cpm=cpm_horizon > 0)

    pred, target, mask = data['pred'], data['target'], data['mask']
    subj_idx = np.array([str(s) for s in data['subj_ids']], dtype=object)
    labels = [f't+{o}' for o in offsets]
    anchor = data['x_last']                                   # 预测锚点 x_t（context 末位）
    arrays = per_task_next_state(pred, target, mask, anchor=anchor)
    agg = aggregate_columns(arrays, subj_idx, labels)
    base_arrays = {name: per_task_next_state(v, target, mask, anchor=anchor)
                   for name, v in data['baselines'].items()}
    base_agg = {name: aggregate_columns(a, subj_idx, labels)
                for name, a in base_arrays.items()}

    report = OrderedDict()
    report['protocol'] = {
        'task_mode': 'next_timepoint', 'split': split,
        'aggregation': 'per-timepoint prediction → per-subject mean → group-level statistics',
        'context_min': int(args.context_min), 'context_max': int(args.context_max),
        'eval_context_length': (int(getattr(args, 'eval_context_length', 0) or 0)
                                or int(args.context_max)),
        'forecast_offsets': list(offsets),
        'prediction_target': 'delta_residual',
        'eval_anchors_per_subject': int(getattr(args, 'eval_anchors_per_subject', 16)),
        'rollout_steps': int(rollout_steps), 'rollout_horizons': list(horizons),
        'fc_min_length': fc_min_len,
        'direction_deadzone': direction_deadzone,
        'direction_rule': ('sign(Δŷ_h)==sign(Δy_h)，Δ 为相邻帧一阶差分；'
                           '|Δy_h| <= direction_deadzone 的 ROI 不计入'),
        'n_subjects': int(len(np.unique(subj_idx))), 'n_tasks': int(len(subj_idx)),
        'n_rollout_tasks': (int(len(data['roll_subj']))
                            if data['roll_subj'] is not None else 0),
        'ar1_fit': ({'n_pairs': int(ar1['n_pairs']),
                     'a_mean': float(np.mean(ar1['a'])),
                     'a_min': float(np.min(ar1['a'])),
                     'a_max': float(np.max(ar1['a']))} if ar1 else
                    {'available': False, 'reason': '未拟合（ar1=None）'}),
        'model_dims': {k: v for k, v in resolve_task_dims(args).items()
                       if k != 'task_mode'},
        'normalization': 'BrainRevIN 统计量仅由历史 context 估计；目标不参与统计',
        'caveat': 'BOLD 预处理（data/Normlize）已按逐 ROI 全序列 z-score，属于数据层'
                  '固定仿射变换；模型层归一化无未来泄漏，但该预处理无法在训练侧撤销。',
    }

    # ---- next-state 主指标与 baselines ----
    model_block = OrderedDict()
    for label in labels:
        blk = _block_summary(agg, label)
        if blk:
            model_block[label] = blk
    model_block['overall'] = (model_block.get(labels[0], {})
                              if len(labels) == 1 else _overall_block(agg, labels))
    report['next_state'] = OrderedDict([('model', model_block)])
    base_report = OrderedDict()
    for name, a in base_agg.items():
        block = OrderedDict()
        for label in labels:
            blk = _block_summary(a, label)
            if blk:
                block[label] = blk
        base_report[name] = block
    base_report['comparison'] = OrderedDict([
        (f'delta_MAE_model_minus_{name}',
         {label: float(_block_summary(agg, label).get('MAE', float('nan'))
                       - _block_summary(base_agg[name], label).get('MAE', float('nan')))
          for label in labels})
        for name in base_agg])
    base_report['note'] = ('delta_MAE_model_minus_* 用于确认模型是否真的优于'
                           '「复制最近 BOLD」「线性趋势外推」「AR(1) 递推」；'
                           'AR(1) 参数只用 train subjects 拟合。')
    report['baselines'] = base_report

    # ---- free rollout ----
    if eval_rollout and data['roll_pred'] is not None:
        roll_labels = [f'H{h}' for h in horizons if h <= rollout_steps]
        roll_arrays = per_task_rollout(data['roll_pred'], data['roll_target'],
                                       data['roll_mask'], horizons)
        roll_subj = np.array([str(s) for s in data['roll_subj']], dtype=object)
        roll_agg = aggregate_columns(roll_arrays, roll_subj, roll_labels)
        roll_block = OrderedDict()
        for label in roll_labels:
            blk = _block_summary(roll_agg, label)
            if blk:
                blk['variance_ratio'] = roll_agg.get('variance_ratio', {}).get(
                    label, {}).get('mean')
                blk['temporal_pcc'] = roll_agg.get('temporal_pcc', {}).get(
                    label, {}).get('mean')
                roll_block[label] = blk
        base_roll_report = OrderedDict()
        for name, arr in data['roll_baselines'].items():
            a = per_task_rollout(arr, data['roll_target'], data['roll_mask'], horizons)
            base_roll_report[name] = OrderedDict(
                (label, _block_summary(aggregate_columns(a, roll_subj, roll_labels),
                                       label)) for label in roll_labels)
        pers = base_roll_report.get('persistence', {})
        report['rollout'] = OrderedDict([
            ('enabled', True),
            ('model', roll_block),
            ('baselines', base_roll_report),
            ('improvement_vs_persistence', {
                label: {
                    'delta_MAE': float(roll_block.get(label, {}).get('MAE', float('nan'))
                                       - pers.get(label, {}).get('MAE', float('nan'))),
                    'delta_PCC': float(roll_block.get(label, {}).get('PCC_spatial',
                                                                     float('nan'))
                                       - pers.get(label, {}).get('PCC_spatial',
                                                                 float('nan'))),
                } for label in roll_labels}),
            ('degradation', {
                f'MAE_{roll_labels[0]}_{roll_labels[-1]}': float(
                    roll_block.get(roll_labels[-1], {}).get('MAE', float('nan'))
                    - roll_block.get(roll_labels[0], {}).get('MAE', float('nan'))),
                f'PCC_{roll_labels[0]}_{roll_labels[-1]}': float(
                    roll_block.get(roll_labels[-1], {}).get('PCC_spatial', float('nan'))
                    - roll_block.get(roll_labels[0], {}).get('PCC_spatial', float('nan'))),
                'note': 'MAE 上升 / PCC 下降表示随 rollout 步数性能退化；'
                        'variance_ratio 明显 < 1 表示方差塌缩（预测趋平）',
            }),
            ('note', 'rollout = 自由自回归滚动（把预测的下一 TR 接回 context 末尾），'
                     '中间不使用真值；baselines 同样递归生成'),
        ])
    else:
        report['rollout'] = {'enabled': False,
                             'reason': '--eval_rollout_tasks_per_subject <= 0 或 eval_rollout=False'}

    # ---- CPM 全 horizon（非自回归一次输出 t+1..t+H，方案 §8）----
    if data.get('cpm_pred') is not None:
        cpm_labels = [f'H{h}' for h in range(1, cpm_horizon + 1)]
        cpm_pred = data['cpm_pred'][:, :, :cpm_horizon]
        cpm_target = data['cpm_target'][:, :, :cpm_horizon]
        cpm_mask = data['cpm_mask'][:, :cpm_horizon]
        cpm_arrays = per_task_next_state(cpm_pred, cpm_target, cpm_mask,
                                         anchor=anchor)
        cpm_agg = aggregate_columns(cpm_arrays, subj_idx, cpm_labels)
        # persistence 基线同口径（x̂ = x_t），供 §25.3 的预测模式对比
        pers_pred = np.repeat(data['x_last'][:, :, None], cpm_horizon, axis=2)
        pers_arrays = per_task_next_state(pers_pred, cpm_target, cpm_mask,
                                          anchor=anchor)
        pers_agg = aggregate_columns(pers_arrays, subj_idx, cpm_labels)
        cpm_block = OrderedDict((label, _block_summary(cpm_agg, label))
                                for label in cpm_labels
                                if _block_summary(cpm_agg, label))
        pers_block = OrderedDict((label, _block_summary(pers_agg, label))
                                 for label in cpm_labels
                                 if _block_summary(pers_agg, label))
        report['cpm_horizon'] = OrderedDict([
            ('enabled', True), ('horizon', int(cpm_horizon)),
            ('model', cpm_block),
            ('persistence', pers_block),
            ('improvement_vs_persistence', {
                label: {'delta_MAE': float(
                            cpm_block.get(label, {}).get('MAE', float('nan'))
                            - pers_block.get(label, {}).get('MAE', float('nan'))),
                        'delta_PCC': float(
                            cpm_block.get(label, {}).get('PCC_spatial', float('nan'))
                            - pers_block.get(label, {}).get('PCC_spatial', float('nan')))}
                for label in cpm_labels}),
            ('note', 'CPM = 非自回归全 horizon 预测（一次 forward 输出 t+1..t+H）；'
                     '与 rollout（自回归滚动）对比可量化 error accumulation'),
        ])
    else:
        report['cpm_horizon'] = {'enabled': False,
                                 'reason': '评估视图提供 all_future_steps 时计算'}

    # ---- 第一阶段核心：Graph-ODE / CPM 逐 horizon 动力学 + 双头一致性 ----
    head_metrics_arrays = OrderedDict()
    head_metrics_labels = OrderedDict()
    head_diag_arrays = OrderedDict()
    head_diag_labels = []
    if data.get('future') is not None:
        future = data['future']
        future_mask = data['future_mask']
        max_h = int(future.shape[2])
        head_h = min(int(cpm_horizon or max_h), max_h)
        fut = future[:, :, :head_h]
        fmask = future_mask[:, :head_h]
        head_labels = [f'H{k}' for k in range(1, head_h + 1)]

        ode_pred = data.get('ode_pred')
        cpm_pred = data.get('cpm_pred')
        if ode_pred is not None:
            ho = min(int(ode_pred.shape[2]), head_h)
            labels_o = head_labels[:ho]
            a = per_task_head_metrics(
                ode_pred[:, :, :ho], fut[:, :, :ho], fmask[:, :ho],
                anchor=anchor, direction_deadzone=direction_deadzone)
            head_metrics_arrays['ODE'] = a
            head_metrics_labels['ODE'] = labels_o
        if cpm_pred is not None:
            hc = min(int(cpm_pred.shape[2]), head_h)
            labels_c = head_labels[:hc]
            a = per_task_head_metrics(
                cpm_pred[:, :, :hc], fut[:, :, :hc], fmask[:, :hc],
                anchor=anchor, direction_deadzone=direction_deadzone)
            head_metrics_arrays['CPM'] = a
            head_metrics_labels['CPM'] = labels_c
        if 'ODE' in head_metrics_arrays and 'CPM' in head_metrics_arrays:
            hd = min(int(head_metrics_arrays['ODE']['mae'].shape[1]),
                     int(head_metrics_arrays['CPM']['mae'].shape[1]))
            labels_d = head_labels[:hd]
            a = per_task_dual_head_metrics(
                cpm_pred[:, :, :hd], ode_pred[:, :, :hd],
                fut[:, :, :hd], fmask[:, :hd], anchor=anchor,
                direction_deadzone=direction_deadzone)
            head_metrics_arrays['DUAL'] = a
            head_metrics_labels['DUAL'] = labels_d

        head_blocks = OrderedDict()
        for name, a in head_metrics_arrays.items():
            agg_h = aggregate_columns(a, subj_idx, head_metrics_labels[name])
            block = OrderedDict()
            for label in head_metrics_labels[name]:
                blk = _block_summary(agg_h, label)
                if blk:
                    block[label] = blk
            block['overall'] = _overall_all(agg_h, head_metrics_labels[name])
            head_blocks[name] = block

        head_subject = OrderedDict()
        if 'ODE' in head_metrics_arrays:
            head_subject['ODE'] = _subj_block(
                head_metrics_arrays['ODE'], subj_idx,
                ('mae', 'rmse', 'spatial_pcc',
                 'delta_mae_step', 'delta_mae_anchor',
                 'direction_accuracy_step', 'direction_accuracy_anchor',
                 'delta_magnitude_ratio', 'variance_ratio'))
        if 'CPM' in head_metrics_arrays:
            head_subject['CPM'] = _subj_block(
                head_metrics_arrays['CPM'], subj_idx,
                ('mae', 'rmse', 'spatial_pcc',
                 'delta_mae_step', 'delta_mae_anchor',
                 'direction_accuracy_step', 'direction_accuracy_anchor',
                 'delta_magnitude_ratio', 'variance_ratio'))
        if 'DUAL' in head_metrics_arrays:
            head_subject['DUAL'] = _subj_block(
                head_metrics_arrays['DUAL'], subj_idx,
                ('gap', 'gap_norm', 'cpm_mae', 'ode_mae', 'persistence_mae',
                 'both_fail_vs_persistence', 'head_error_corr', 'oracle_mae',
                 'oracle_gain_vs_cpm', 'oracle_gain_vs_ode', 'cpm_win_rate',
                 'direction_agreement'))
        # ---- 幅度诊断：ODE/CPM + 三条轨迹基线，共用同一套分位分层 ----
        diag_edges = amplitude_quantile_edges(fut, fmask, anchor)
        if ode_pred is not None:
            head_diag_arrays['ODE'] = per_task_amplitude_diagnostics(
                ode_pred[:, :, :head_h], fut, fmask, anchor, amp_edges=diag_edges)
        if cpm_pred is not None:
            head_diag_arrays['CPM'] = per_task_amplitude_diagnostics(
                cpm_pred[:, :, :head_h], fut, fmask, anchor, amp_edges=diag_edges)
        for bname, btraj in _trajectory_baselines(
                data['x_last'], data['x_prev'], ar1, head_h).items():
            head_diag_arrays[bname] = per_task_amplitude_diagnostics(
                btraj, fut, fmask, anchor, amp_edges=diag_edges)
        head_diag_labels = head_labels
        diag_report = OrderedDict(
            (name, _diag_block(d, subj_idx, head_labels, diag_edges))
            for name, d in head_diag_arrays.items())

        report['head_metrics'] = OrderedDict([
            ('enabled', True),
            ('horizon', int(head_h)),
            ('labels', head_labels),
            ('direction_deadzone', direction_deadzone),
            ('definition', {
                'delta_MAE_step': '相邻帧一阶差分误差：Δŷ_1=ŷ_1−x_t，'
                                  'Δŷ_h=ŷ_h−ŷ_(h−1)（与训练 temporal_difference_huber 同口径）',
                'delta_MAE_anchor': '相对锚点差分误差 |(ŷ_h−x_t)−(y_h−x_t)|，'
                                    '数学上恒等于点值 MAE（保留用于核对）',
                'direction_accuracy_step': 'sign(Δŷ_h)==sign(Δy_h) 的 ROI 比例；'
                                           '排除 |Δy_h|<=deadzone 的近零变化',
                'direction_accuracy_anchor': 'sign(ŷ_h−x_t)==sign(y_h−x_t) 的 ROI 比例；'
                                             '排除 |y_h−x_t|<=deadzone 的近零变化',
                'gap': 'mean_ROI |ŷ_CPM−ŷ_ODE|（与训练 cross_head_consistency 同口径）',
                'gap_norm': 'gap / (0.5·(CPM_MAE+ODE_MAE))，避免误差尺度变化造成假下降',
                'both_fail_vs_persistence': '同一 anchor/horizon 两头 MAE 均大于 '
                                            'persistence 的比例（“一起预测错误”）',
                'head_error_corr': '同一 anchor 上逐 ROI 绝对误差的 Pearson 相关',
                'oracle_MAE': '逐 anchor 取两头较小 MAE（用于衡量两头互补性）',
            }),
            ('ODE', head_blocks.get('ODE', {'enabled': False,
                                           'reason': '模型未返回 dynamics_pred'})),
            ('CPM', head_blocks.get('CPM', {'enabled': False,
                                            'reason': '未收集 CPM 输出'})),
            ('dual_head', head_blocks.get('DUAL', {
                'enabled': False, 'reason': '需要 ODE 与 CPM 同时可用'})),
            ('subject_level', head_subject),
            ('diagnostics', OrderedDict([
                ('amplitude_bin_edges',
                 [float(e) if np.isfinite(e) else 1e9 for e in diag_edges]),
                ('definition', {
                    'delta_std_ratio': 'std(Δŷ)/std(Δy)；<1 表示变化幅度压缩',
                    'delta_std_ratio_h1': 'H1 单步幅度比（检验 readout 是否压缩）',
                    'delta_std_ratio_h2_8': 'H2–H8 多步幅度比（检验状态推进/条件均值）',
                    'delta_pcc': 'Corr(Δŷ, Δy)，H1–H8 × 全部 ROI 的有效元素',
                    'temporal_variance_ratio': '逐 ROI 在有效未来轨迹上的 '
                                               'Var(ŷ)/Var(y) 再对 ROI 平均',
                    'trajectory_variance_ratio_flat': 'H×ROI 展平后的方差比',
                    'amp_bin_mae': '按真实 |Δy| 五分位的 Delta MAE；Q5 为最大幅变化',
                    'baselines': 'persistence / trend / AR(1) 用同一套真值分层计算；'
                                 'AR(1) 参数只用 train subjects 拟合',
                }),
                ('blocks', diag_report),
            ])),
        ])
    else:
        report['head_metrics'] = {'enabled': False,
                            'reason': '评估视图未提供 future/future_mask'}

    # ---- FC（rollout 轨迹；长度不足则不计算） ----
    if eval_fc and data['roll_pred'] is not None:
        roll_subj = np.array([str(s) for s in data['roll_subj']], dtype=object)
        fc_block = OrderedDict()
        fc_block['model'] = fc_rollout_metrics(data['roll_pred'], data['roll_target'],
                                               fc_min_len, subj_idx=roll_subj)
        for name, arr in data['roll_baselines'].items():
            fc_block[name] = fc_rollout_metrics(arr, data['roll_target'], fc_min_len,
                                                subj_idx=roll_subj)
        fc_block['note'] = (f'FC 在 rollout 前 {fc_min_len} 步上估计（prompt §二十六：'
                            '长度 < fc_min_length 不计算 FC），只比较上三角非对角边；'
                            'edge_pcc_mean 为逐边跨任务相关再对边平均；network 块按 '
                            'AAL116「对应网络」把边分为 within-network / between-network '
                            '两条集合，同口径给出 fc_mae/edge_pcc（标签不可用时省略）')
        report['fc'] = fc_block
    else:
        report['fc'] = {'enabled': False, 'reason': '--eval_fc False 或未做 rollout'}

    # ---- 频谱（可选；Welch PSD 低频段口径，见本模块 spectral_metrics） ----
    if eval_spectral and data['roll_pred'] is not None:
        n_roll = len(data['roll_pred'])
        one_col_mask = np.ones((n_roll, 1), np.float32)
        spec = spectral_metrics(data['roll_pred'][:, None, :, :],
                                data['roll_target'][:, None, :, :],
                                one_col_mask, tr=spectral_tr)
        spec['persistence'] = spectral_metrics(
            data['roll_baselines']['persistence'][:, None, :, :],
            data['roll_target'][:, None, :, :], one_col_mask, tr=spectral_tr)
        report['spectral'] = spec
    else:
        report['spectral'] = {'enabled': False, 'reason': '--eval_spectral False 或未做 rollout'}

    # ---- 被试级汇总与 HAMD 关联 ----
    subj_pcc = {label: _subject_metric(arrays['spatial_pcc'], subj_idx, i)
                for i, label in enumerate(labels)}
    subj_mae = {label: _subject_metric(arrays['mae'], subj_idx, i)
                for i, label in enumerate(labels)}
    report['subject_level'] = {
        'n_subjects': int(len(np.unique(subj_idx))),
        'PCC_spatial_mean': float(np.mean([v.mean() for v in subj_pcc.values() if len(v)])),
        'MAE_mean': float(np.mean([v.mean() for v in subj_mae.values() if len(v)])),
        'per_offset': {label: {'PCC_spatial_mean': float(subj_pcc[label].mean()),
                               'MAE_mean': float(subj_mae[label].mean())}
                       for label in labels if len(subj_pcc[label])},
    }
    report['hamd'] = _hamd_association(data, arrays, subj_idx, labels)
    report['n_params'] = int(n_params) if n_params is not None else None
    if extra_report:
        report.update(extra_report)

    # ---- ROI 置换重要性（可选；逐 ROI 重跑前向，代价随 n_rois × max_tasks 增长）----
    if compute_importance:
        imp = roi_permutation_importance(
            model, data_loader, device,
            max_tasks=int(importance_max_tasks),
            n_permutations=int(importance_permutations),
            seed=int(getattr(args, 'seed', 2024)),
            fdr_alpha=float(importance_fdr_alpha))
        payload = write_feature_importance(out_dir, split, imp, offsets)
        report['feature_importance'] = {
            'available': True,
            'file': f'feature_importance_{split}.json',
            'method': payload['method'],
            'n_tasks': payload['n_tasks'],
            'n_permutations': payload['n_permutations'],
            'baseline_mae': payload['baseline_mae'],
            'fdr_alpha': payload['fdr_alpha'],
            'n_significant': len(payload['significant_roi_indices']),
            'significant_roi_indices': payload['significant_roi_indices'],
            'top10_roi_indices': [int(r) + 1
                                  for r in payload['roi_importance_rank'][:10]],
            'note': 'roi_importance 为置换该 ROI 的 context 后 next-state MAE 的增量'
                    '（1 基 AAL 索引见 significant_roi_indices，BH-FDR 校正）',
        }
    else:
        report['feature_importance'] = {'available': False,
                                        'reason': '未请求（--feature_importance）'}

    # ---- 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f'metrics_{split}.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    head_extra = OrderedDict(
        (head, {'labels': head_metrics_labels[head], 'arrays': arrs})
        for head, arrs in head_metrics_arrays.items())
    _write_subject_csv(out_dir / f'per_subject_metrics_{split}.csv', data, arrays,
                       base_arrays, subj_idx, labels, extra_arrays=head_extra)
    _write_task_csv(out_dir / f'per_task_metrics_{split}.csv', data, arrays, labels,
                    extra_arrays=head_extra)
    if head_diag_arrays:
        _write_head_diagnostics_csv(
            out_dir / f'head_diagnostics_subject_{split}.csv',
            head_diag_arrays, head_diag_labels, subj_idx)
    if save_arrays:
        extra = {}
        # 条件矩阵按 [N, K] 原样落盘（**不展平**），字段名一并保存，供置换诊断、
        # 缺失分层与逐被试复核复用（HAMD/HAMA 按字段名读取，不假设列序）
        for key in ('cond_z', 'cond_code', 'cond_mask', 'cond_raw', 'subject_index'):
            if data.get(key) is not None:
                extra[key] = data[key]
        extra['condition_fields'] = np.asarray(
            data.get('condition_fields', []), dtype=str)
        save_predictions_npz(
            out_dir / f'predictions_{split}.npz',
            pred=pred, target=target, mask=mask, x_last=data['x_last'],
            x_prev=data['x_prev'],
            subj_ids=np.asarray(data['subj_ids']).astype(str), cutoff=data['cutoff'],
            context_len=data['context_len'], hamd=data['hamd'],
            offsets=np.array(offsets),
            pred_persistence=data['baselines'].get('persistence'),
            pred_trend=data['baselines'].get('trend'),
            pred_ar1=data['baselines'].get('ar1'),
            ode_pred=(data['ode_pred'] if data.get('ode_pred') is not None
                      else []),
            dynamics_std=(data['dynamics_std']
                          if data.get('dynamics_std') is not None else []),
            dynamics_logvar_norm=(data['dynamics_logvar_norm']
                                  if data.get('dynamics_logvar_norm') is not None
                                  else []),
            cpm_pred=(data['cpm_pred'] if data.get('cpm_pred') is not None
                      else []),
            future=(data['future'] if data.get('future') is not None else []),
            future_mask=(data['future_mask']
                         if data.get('future_mask') is not None else []),
            rollout=(data['roll_pred'] if data['roll_pred'] is not None else []),
            rollout_target=(data['roll_target']
                            if data['roll_target'] is not None else []),
            rollout_preds_ar1=data['roll_baselines'].get('ar1', []),
            **extra)
    log.info('[next_timepoint] %s 评估完成 | next MAE=%.4f | spatial PCC=%.4f',
             split, model_block.get(labels[0], {}).get('MAE', float('nan')),
             model_block.get(labels[0], {}).get('PCC_spatial', float('nan')))
    return report


def _overall_block(agg, labels):
    """多偏移时的 overall（按有效任务数加权的跨偏移平均）。"""
    block = {}
    for metric, key in (('spatial_pcc', 'PCC_spatial'), ('mae', 'MAE'),
                        ('rmse', 'RMSE'), ('r2', 'R2')):
        vals, ws = [], []
        for label in labels:
            d = agg.get(metric, {}).get(label)
            if d:
                vals.append(d['mean']); ws.append(d['n_tasks'])
        if vals:
            block[key] = float(sum(v * w for v, w in zip(vals, ws)) / max(1, sum(ws)))
    return block


def _overall_all(agg, labels):
    """把 ``_block_summary`` 的全部标量指标按任务数加权跨 horizon 平均。

    与 ``_overall_block`` 的区别是不限定 MAE/RMSE/PCC/R2，因此 Delta/DA、
    双头 Gap 等新增指标也能得到 overall 行；``_std``/``_median``/``n_*`` 不重复
    聚合（跨 horizon 的标准差没有直接解释）。
    """
    blocks = {label: _block_summary(agg, label) for label in labels}
    keys = set()
    for blk in blocks.values():
        keys.update(blk)
    out = {}
    for key in sorted(keys):
        if key.endswith(('_std', '_median')) or key.startswith('n_'):
            continue
        vals, ws = [], []
        for label, blk in blocks.items():
            v = blk.get(key)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
            ws.append(float(blk.get('n_tasks', 1)))
        if vals:
            out[key] = float(sum(v * w for v, w in zip(vals, ws)) / max(1e-12, sum(ws)))
    return out


def _subject_overall(array, subj_idx):
    """逐被试把 [N,H] 指标压成标量（被试内跨 anchor/horizon 等权平均）。"""
    vals = []
    for s in np.unique(subj_idx):
        col = np.asarray(array, dtype=np.float64)[subj_idx == s]
        ok = np.isfinite(col)
        if ok.any():
            vals.append(float(col[ok].mean()))
    return np.asarray(vals, dtype=np.float64)


def _subject_summary(values):
    """被试级均值序列 → mean/std/median/n。"""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'mean': float('nan'), 'std': float('nan'),
                'median': float('nan'), 'n_subjects': 0}
    return {'mean': float(v.mean()), 'std': float(v.std()),
            'median': float(np.median(v)), 'n_subjects': int(v.size)}


def _subj_block(arrays, subj_idx, metrics):
    """对一组指标做被试级汇总（论文统计主口径）。"""
    return {metric: _subject_summary(_subject_overall(arrays[metric], subj_idx))
            for metric in metrics if metric in arrays}


_DIAG_SCALAR_KEYS = (
    'delta_std_ratio', 'delta_pcc', 'temporal_variance_ratio',
    'trajectory_variance_ratio_flat',
    'delta_std_ratio_h1', 'delta_std_ratio_h2_8',
    'delta_pcc_h1', 'delta_pcc_h2_8',
    'mean_abs_pred_delta', 'mean_abs_true_delta',
)


def _diag_block(diag, subj_idx, labels, edges):
    """把逐 anchor 诊断数组整理成 JSON 报告块（被试级 mean/std/median）。"""
    overall = {key: _subject_summary(_subject_overall(diag[key], subj_idx))
               for key in _DIAG_SCALAR_KEYS if key in diag}
    per_horizon = {}
    for metric in ('delta_std_ratio_h', 'delta_pcc_h'):
        agg = aggregate_columns({metric: diag[metric]}, subj_idx, labels)
        per_horizon[metric] = {
            label: agg[metric][label] for label in labels
            if metric in agg and label in agg[metric]}
    bin_labels = [f'Q{i + 1}' for i in range(diag['amp_bin_mae'].shape[1])]
    mae_agg = aggregate_columns({'amp_bin_mae': diag['amp_bin_mae']},
                                subj_idx, bin_labels).get('amp_bin_mae', {})
    mean_agg = aggregate_columns({'amp_bin_true_abs_mean': diag['amp_bin_true_abs_mean']},
                                 subj_idx, bin_labels).get('amp_bin_true_abs_mean', {})
    bins = {}
    for label in bin_labels:
        if label not in mae_agg:
            continue
        bins[label] = {
            'MAE': mae_agg[label]['mean'],
            'MAE_std': mae_agg[label]['std'],
            'true_abs_delta_mean': mean_agg.get(label, {}).get('mean'),
            'n_tasks': mae_agg[label]['n_tasks'],
            'n_subjects': mae_agg[label]['n_subjects'],
        }
    return {
        'overall': overall,
        'per_horizon': per_horizon,
        'amplitude_strata': {
            'labels': bin_labels,
            'edges': [float(e) if np.isfinite(e) else 1e9 for e in edges],
            'bins': bins,
            'note': 'Q1..Q5 按真实 |Δy| 的五分位分层；Q5 为最大幅变化',
        },
    }


def _trajectory_baselines(x_last, x_prev, ar1, horizon):
    """构造 persistence / trend / AR(1) 的全 horizon 轨迹 [N,F,H]。"""
    x_last = np.asarray(x_last, dtype=np.float64)
    x_prev = np.asarray(x_prev, dtype=np.float64)
    h = int(horizon)
    n, f = x_last.shape
    k = np.arange(1, h + 1, dtype=np.float64)
    out = {
        'persistence': np.repeat(x_last[:, :, None], h, axis=2),
        'trend': (x_last[:, :, None]
                  + k[None, None, :] * (x_last - x_prev)[:, :, None]),
    }
    if ar1 is not None:
        a = np.asarray(ar1['a'], dtype=np.float64).reshape(1, f)
        b = np.asarray(ar1['b'], dtype=np.float64).reshape(1, f)
        cur = x_last.copy()
        steps = []
        for _ in range(h):
            cur = a * cur + b
            steps.append(cur.copy())
        out['ar1'] = np.stack(steps, axis=2)
    return out


def _write_head_diagnostics_csv(path, diag_arrays, labels, subj_idx):
    """逐被试写出幅度诊断（供 配对评价 配对使用）。"""
    subjects = sorted(set(str(s) for s in subj_idx))
    rows = []
    for s in subjects:
        sel = subj_idx == s
        row = {'subject_id': s, 'n_tasks': int(sel.sum())}
        for name, diag in diag_arrays.items():
            for key in _DIAG_SCALAR_KEYS:
                if key in diag:
                    row[f'{name}_{key}'] = float(
                        np.nanmean(diag[key][sel])) if np.isfinite(
                            diag[key][sel]).any() else float('nan')
            for hi, label in enumerate(labels):
                for metric in ('delta_std_ratio_h', 'delta_pcc_h'):
                    v = diag[metric][sel, hi]
                    row[f'{name}_{label}_{metric}'] = (
                        float(np.nanmean(v)) if np.isfinite(v).any()
                        else float('nan'))
            for b in range(diag['amp_bin_mae'].shape[1]):
                v = diag['amp_bin_mae'][sel, b]
                row[f'{name}_Q{b + 1}_MAE'] = (
                    float(np.nanmean(v)) if np.isfinite(v).any()
                    else float('nan'))
        rows.append(row)
    cols = list(rows[0].keys()) if rows else ['subject_id', 'n_tasks']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log.info('[next_timepoint] 幅度诊断已写入: %s（%d 个被试）', path, len(rows))


def _write_subject_csv(path, data, arrays, base_arrays, subj_idx, labels,
                       extra_arrays=None):
    """被试级 CSV（论文统计的主口径：先被试内平均，再跨被试统计）。

    条件注入时逐字段值与有效性按**字段名**成列写出（不假设第 0 列是 HAMD），
    并附真实缺失模式，供缺失分层与来源分层报告直接使用。
    """
    subjects = sorted(set(str(s) for s in subj_idx))
    hamd_map, n_tasks = {}, {}
    for i, s in enumerate(subj_idx):
        s = str(s)
        n_tasks[s] = n_tasks.get(s, 0) + 1
        if np.isfinite(data['hamd'][i]):
            hamd_map[s] = float(data['hamd'][i])
    fields = list(data.get('condition_fields') or [])
    raw = data.get('cond_raw')
    msk = data.get('cond_mask')
    cond_map, obs_map = {}, {}
    if raw is not None and fields:
        for i, s in enumerate(subj_idx):
            s = str(s)
            cond_map[s] = {name: (float(raw[i, j]) if np.isfinite(raw[i, j]) else float('nan'))
                           for j, name in enumerate(fields)}
            obs_map[s] = {name: int(msk[i, j] > 0) for j, name in enumerate(fields)} \
                if msk is not None else {}
    rows = []
    for s in subjects:
        sel = subj_idx == s
        row = {'subject_id': s, 'hamd': hamd_map.get(s, float('nan')),
               'n_tasks': n_tasks[s]}
        for j, name in enumerate(fields):
            row[f'cond_{name}'] = cond_map.get(s, {}).get(name, float('nan'))
            row[f'obs_{name}'] = obs_map.get(s, {}).get(name, -1)
        for i, label in enumerate(labels):
            for metric, arr in (('MAE', arrays['mae']), ('RMSE', arrays['rmse']),
                                ('PCC_spatial', arrays['spatial_pcc']),
                                ('R2', arrays['r2'])):
                col = arr[sel, i]
                col = col[~np.isnan(col)]
                row[f'{label}_{metric}'] = float(col.mean()) if len(col) else float('nan')
        for name, arrs in base_arrays.items():
            for i, label in enumerate(labels):
                col = arrs['mae'][sel, i]
                col = col[~np.isnan(col)]
                row[f'{name}_{label}_MAE'] = float(col.mean()) if len(col) else float('nan')
        for head, spec in (extra_arrays or {}).items():
            for metric, arr in spec['arrays'].items():
                for i, label in enumerate(spec['labels']):
                    col = arr[sel, i]
                    col = col[~np.isnan(col)]
                    row[f'{head}_{label}_{metric}'] = (
                        float(col.mean()) if len(col) else float('nan'))
        rows.append(row)
    cols = (['subject_id', 'hamd', 'n_tasks']
            + sum(([f'cond_{name}', f'obs_{name}'] for name in fields), [])
            + [f'{label}_{m}' for label in labels
               for m in ('MAE', 'RMSE', 'PCC_spatial', 'R2')]
            + [f'{name}_{label}_MAE' for name in base_arrays for label in labels]
            + [f'{head}_{label}_{metric}'
               for head, spec in (extra_arrays or {}).items()
               for metric in spec['arrays'] for label in spec['labels']])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log.info('[next_timepoint] 被试级指标已写入: %s（%d 个被试）', path, len(rows))


def _write_task_csv(path, data, arrays, labels, extra_arrays=None):
    """任务级 CSV（cutoff/context 长度明细，便于逐任务核对/配对比较）。"""
    rows = []
    for i, s in enumerate(data['subj_ids']):
        row = {'subj_id': str(s), 'context_len': int(data['context_len'][i]),
               'cutoff': int(data['cutoff'][i])}
        for j, label in enumerate(labels):
            row[f'{label}_MAE'] = float(arrays['mae'][i, j])
            row[f'{label}_RMSE'] = float(arrays['rmse'][i, j])
            row[f'{label}_PCC_spatial'] = float(arrays['spatial_pcc'][i, j])
            row[f'{label}_R2'] = float(arrays['r2'][i, j])
        for head, spec in (extra_arrays or {}).items():
            for metric, arr in spec['arrays'].items():
                for j, label in enumerate(spec['labels']):
                    row[f'{head}_{label}_{metric}'] = float(arr[i, j])
        rows.append(row)
    cols = (['subj_id', 'context_len', 'cutoff']
            + [f'{label}_{m}' for label in labels
               for m in ('MAE', 'RMSE', 'PCC_spatial', 'R2')]
            + [f'{head}_{label}_{metric}'
               for head, spec in (extra_arrays or {}).items()
               for metric in spec['arrays'] for label in spec['labels']])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize(report):
    """把评估报告压缩为 registry 兼容的扁平键（含 next-state / rollout 主指标）。"""
    split = report['protocol']['split']
    labels = [f't+{o}' for o in report['protocol']['forecast_offsets']]
    first = labels[0]
    model = report.get('next_state', {}).get('model', {})
    block = model.get(first, {})
    summary = {
        f'{split}_pcc': block.get('PCC_spatial'),
        f'{split}_mae': block.get('MAE'),
        f'{split}_rmse': block.get('RMSE'),
        f'{split}_r2': block.get('R2'),
        f'{split}_subj_pcc': report.get('subject_level', {}).get('PCC_spatial_mean'),
        f'{split}_next_mae': block.get('MAE'),
        f'{split}_next_pcc_spatial': block.get('PCC_spatial'),
        'n_params': report.get('n_params'),
    }
    baselines = report.get('baselines', {})
    for name in ('persistence', 'trend', 'ar1'):
        blk = baselines.get(name, {}).get(first, {})
        if blk:
            summary[f'{split}_{name}_mae'] = blk.get('MAE')
    roll = report.get('rollout', {}).get('model', {})
    for label, blk in roll.items():
        summary[f'{split}_rollout_mae_{label}'] = blk.get('MAE')
        summary[f'{split}_rollout_pcc_{label}'] = blk.get('PCC_spatial')
    cpm = report.get('cpm_horizon', {})
    if cpm.get('enabled'):
        for label, blk in cpm.get('model', {}).items():
            summary[f'{split}_cpm_mae_{label}'] = blk.get('MAE')
            summary[f'{split}_cpm_pcc_{label}'] = blk.get('PCC_spatial')
    head_metrics = report.get('head_metrics', {})
    if head_metrics.get('enabled'):
        for head, key in (('ODE', 'ode'), ('CPM', 'cpm_head')):
            head_block = head_metrics.get(head, {})
            for label, blk in head_block.items():
                if label == 'overall' or not isinstance(blk, dict):
                    continue
                summary[f'{split}_{key}_mae_{label}'] = blk.get('MAE')
                summary[f'{split}_{key}_delta_mae_step_{label}'] = blk.get(
                    'delta_MAE_step')
                summary[f'{split}_{key}_da_step_{label}'] = blk.get(
                    'direction_accuracy_step')
            overall = head_block.get('overall', {})
            if overall:
                summary[f'{split}_{key}_mae_overall'] = overall.get('MAE')
                summary[f'{split}_{key}_delta_mae_step_overall'] = overall.get(
                    'delta_MAE_step')
                summary[f'{split}_{key}_da_step_overall'] = overall.get(
                    'direction_accuracy_step')
        dual = head_metrics.get('dual_head', {})
        for label, blk in dual.items():
            if label == 'overall' or not isinstance(blk, dict):
                continue
            summary[f'{split}_gap_{label}'] = blk.get('gap')
            summary[f'{split}_dual_both_fail_{label}'] = blk.get(
                'both_fail_vs_persistence')
        dual_overall = dual.get('overall', {})
        if dual_overall:
            summary[f'{split}_gap_overall'] = dual_overall.get('gap')
            summary[f'{split}_gap_norm_overall'] = dual_overall.get('gap_norm')
    fc = report.get('fc', {}).get('model', {}) if isinstance(report.get('fc'), dict) else {}
    if isinstance(fc, dict) and fc.get('available'):
        summary[f'{split}_fc_upper_mae'] = fc.get('fc_mae')
        summary[f'{split}_edge_pcc'] = fc.get('edge_pcc_mean')
    return summary
