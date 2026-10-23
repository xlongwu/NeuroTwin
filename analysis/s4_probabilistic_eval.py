# coding=utf-8
"""S4 概率轨迹评估：对角高斯（S4-0）与样本集合（S4-1/S4-2）两种模式。

解析高斯模式（npz 只有 ``ode_pred`` + ``dynamics_std``）评估逐 ROI / horizon 对角高斯：

    p(y_{t+h,f} | x_{≤t}) = N(mean_{f,h}, std_{f,h}^2)

其中 mean = ``aux_info['dynamics_pred']``（npz 的 ``ode_pred``），std =
``aux_info['dynamics_std']``（数据尺度）。

集合模式（npz 另含 ``dynamics_samples [N,K,F,H]``）改用经验分布：CRPS 取排序闭式，
区间取样本分位数，并额外报告 Energy Score（跨 ROI 联合结构）与时间变差函数得分
（跨 horizon 依赖）。该模式会并行给出一行 ``model_iid`` 对照——逐 ``(ROI,horizon)``
独立打乱样本维，边际分布完全不变、依赖结构被摧毁，因此 "model 优于 model_iid"
只能来自相关结构本身，而不是放大方差。集合模式下逐元素高斯 NLL 无定义，报告为 NaN。

两种模式的默认参照都是：

- ``constant_roi_h``：用 **train 被试**的模型残差按 ROI/horizon 估计常数方差；
- ``constant_global``：用 train 残差估计单个全局常数方差。

两者都是“输入无关、跨 ROI/horizon 独立”的对角高斯基线。本脚本只做**事后**
评估：测试集真实变化幅度仅用于 Q1–Q5 分层与校准曲线，不进入任何条件输入。

口径：逐任务先对有效 ROI/horizon 求均值，再按被试求均值；跨被试统计与配对
检验是主报告口径。运行示例：

    python -m analysis.s4_probabilistic_eval \
        --pred_npz results/phase4/S4_0/eval/predictions_test.npz \
        --ckpt checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt \
        --out_dir results/phase4/S4_0

集合模式先以测试期 K 采样并落盘样本，再评估：

    python -m experiments.evaluate_variant --ckpt <S4 ckpt> --out_dir <dir> \
        --splits test --save_arrays --s4_samples 64
    python -m analysis.s4_probabilistic_eval \
        --pred_npz <dir>/predictions_test.npz --out_dir <dir> \
        --variant S4-1 --prefix s4_1

``--variant`` 缺省时从 ``--ckpt`` 快照的 ``s4_variant`` 推导；两者不一致时以
``--variant`` 为准（train 参照 npz 常来自 S2 权重，此时无法从 ckpt 推出标签）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8
SIGMA_FLOOR = 1e-6
DEFAULT_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
REPORT_LEVELS = (0.8, 0.9)
METHODS = ('model', 'constant_roi_h', 'constant_global')


# ══════════════════════════════════════════════════════════════════════════
#  §1 概率指标（纯 numpy，可直接单元测试）
# ══════════════════════════════════════════════════════════════════════════

def _check_shapes(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> None:
    if mean.shape != std.shape or mean.shape != target.shape:
        raise ValueError(
            f'mean/std/target 形状必须一致，收到 {mean.shape}/{std.shape}/{target.shape}')
    if mean.ndim != 3:
        raise ValueError(f'期望 [N,F,H]，收到 {mean.shape}')


def _valid_mask(mask, n: int, f: int, h: int) -> np.ndarray:
    if mask is None:
        return np.ones((n, f, h), dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape == (n, h):
        m = np.broadcast_to(m[:, None, :], (n, f, h))
    elif m.shape != (n, f, h):
        raise ValueError(f'mask 形状应为 {(n, h)} 或 {(n, f, h)}，收到 {m.shape}')
    return m.copy()


def _with_invalid_nan(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, values, np.nan)


def gaussian_nll(mean, std, target, mask=None,
                 sigma_floor: float = SIGMA_FLOOR) -> np.ndarray:
    """逐元素高斯 NLL：``0.5·[log(2πσ²) + (y−μ)²/σ²]``，无效位置 NaN。"""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _check_shapes(mean, std, target)
    n, f, h = mean.shape
    valid = _valid_mask(mask, n, f, h) & np.isfinite(mean) & np.isfinite(target)
    sigma = np.maximum(np.abs(std), float(sigma_floor))
    z = (target - mean) / sigma
    nll = 0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(sigma) + z * z)
    return _with_invalid_nan(nll, valid)


def gaussian_crps(mean, std, target, mask=None,
                  sigma_floor: float = SIGMA_FLOOR) -> np.ndarray:
    """解析高斯 CRPS：``σ·(z(2Φ(z)−1)+2φ(z)−1/√π)``，z=(y−μ)/σ。"""
    from scipy.stats import norm

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _check_shapes(mean, std, target)
    n, f, h = mean.shape
    valid = _valid_mask(mask, n, f, h) & np.isfinite(mean) & np.isfinite(target)
    sigma = np.maximum(np.abs(std), float(sigma_floor))
    z = (target - mean) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0)
                    + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    return _with_invalid_nan(crps, valid)


def gaussian_interval(mean, std, target, level: float,
                      mask=None, sigma_floor: float = SIGMA_FLOOR):
    """中心预测区间：返回 ``(covered [N,F,H], width [N,F,H])``。

    区间为 ``μ ± z_{(1+level)/2}·σ``，宽度 ``2·z·σ``；无效位置 NaN。
    """
    from scipy.stats import norm

    if not 0.0 < float(level) < 1.0:
        raise ValueError(f'level 必须在 (0,1)，收到 {level}')
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _check_shapes(mean, std, target)
    n, f, h = mean.shape
    valid = _valid_mask(mask, n, f, h) & np.isfinite(mean) & np.isfinite(target)
    sigma = np.maximum(np.abs(std), float(sigma_floor))
    z = float(norm.ppf(0.5 * (1.0 + float(level))))
    covered = np.abs(target - mean) <= z * sigma
    width = 2.0 * z * sigma
    return _with_invalid_nan(covered.astype(np.float64), valid), \
        _with_invalid_nan(width, valid)


MIN_ENSEMBLE_SAMPLES = 2
ENERGY_CHUNK = 128
CEILING_SIM_COUNT = 200_000


def _as_samples(samples) -> np.ndarray:
    s = np.asarray(samples)
    if s.ndim != 4:
        raise ValueError(f'samples 期望 [N,K,F,H]，收到 {s.shape}')
    if s.shape[1] < MIN_ENSEMBLE_SAMPLES:
        raise ValueError(f'集合指标需要 K>={MIN_ENSEMBLE_SAMPLES}，收到 K={s.shape[1]}')
    return s


def _element_valid(mask, n: int, f: int, h: int) -> np.ndarray:
    """mask [N,H] 或 [N,F,H] → bool [N,F,H]（与 _valid_mask 同语义，但不复制全量）。"""
    if mask is None:
        return np.ones((n, f, h), dtype=bool)
    m = np.asarray(mask)
    if m.shape == (n, h):
        return np.broadcast_to(m[:, None, :].astype(bool), (n, f, h))
    if m.shape != (n, f, h):
        raise ValueError(f'mask 形状应为 {(n, h)} 或 {(n, f, h)}，收到 {m.shape}')
    return m.astype(bool)


def ensemble_crps(samples, target, mask=None, fair: bool = False) -> np.ndarray:
    """[N,K,F,H] → 逐元素经验 CRPS（排序闭式，与 train.prob_losses 同式）。"""
    return _crps_sorted(np.sort(_as_samples(samples), axis=1), target, mask,
                        fair=fair)


def _crps_sorted(s: np.ndarray, target, mask=None, fair: bool = False) -> np.ndarray:
    """在已按样本维排序的 ``s [N,K,F,H]`` 上算经验 CRPS。

    ``CRPS ≈ 1/K Σ|x_(i)−y| − 1/K² Σ(2i−K−1)·x_(i)``，复杂度 O(K log K)，
    不需要 K×K 两两差分。

    ``fair=True`` 时离散度项改用 ``1/(2K(K−1))Σ_{i≠j}|x_i−x_j|``（即把上面的第二项乘
    以 ``K/(K−1)``）。``1/K²`` 形式对有限集合系统性高估 CRPS，偏差约
    ``½E|X−X'|/K``；fair 形式在成员独立同分布假设下无此偏差。训练损失用前者，
    跨方法比较应与所选口径一致。
    """
    y = np.asarray(target, dtype=np.float32)
    n, k, f, h = s.shape
    if tuple(y.shape) != (n, f, h):
        raise ValueError(f'target 期望 {(n, f, h)}，收到 {tuple(y.shape)}')
    term1 = np.abs(s - y[:, None, :, :]).mean(axis=1, dtype=np.float64)
    coeff = 2.0 * np.arange(k, dtype=np.float64) - float(k) + 1.0
    term2 = (s.astype(np.float64)
             * coeff.reshape(1, k, 1, 1)).sum(axis=1) / float(k * k)
    if fair:
        term2 = term2 * float(k) / float(k - 1)
    valid = _element_valid(mask, n, f, h) & np.isfinite(y)
    return np.where(valid, term1 - term2, np.nan)


def ensemble_interval(samples, target, level: float, mask=None):
    """样本分位数中心区间 → ``(covered [N,F,H], width [N,F,H])``。"""
    return _interval_sorted(np.sort(_as_samples(samples), axis=1), target,
                            level, mask)


def _interval_sorted(s: np.ndarray, target, level: float, mask=None):
    """在已排序的 ``s`` 上取分位数区间，使多个 level 复用同一次排序。

    分位数用线性插值，与 numpy ``method='linear'`` 一致。
    """
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f'level 必须在 (0,1)，收到 {level}')
    y = np.asarray(target, dtype=np.float32)
    n, k, f, h = s.shape
    if tuple(y.shape) != (n, f, h):
        raise ValueError(f'target 期望 {(n, f, h)}，收到 {tuple(y.shape)}')

    def _q(prob):
        pos = prob * (k - 1)
        i0 = int(math.floor(pos))
        w = pos - i0
        i1 = min(i0 + 1, k - 1)
        return (1.0 - w) * s[:, i0] + w * s[:, i1]

    lo, hi = _q((1.0 - float(level)) / 2.0), _q((1.0 + float(level)) / 2.0)
    valid = _element_valid(mask, n, f, h) & np.isfinite(y)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    return (np.where(valid, covered.astype(np.float64), np.nan),
            np.where(valid, width.astype(np.float64), np.nan))


def iid_shuffle_samples(samples, seed: int = 0, chunk_tasks: int = 256) -> np.ndarray:
    """逐 ``(n,f,h)`` 独立打乱样本维：保留每个元素的边际分布，摧毁跨 ROI 与跨
    horizon 的依赖结构。用于回答"相关结构是否真的带来概率收益"。
    """
    s = _as_samples(samples)
    n, k, f, h = s.shape
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(s)
    for start in range(0, n, chunk_tasks):
        stop = min(start + int(chunk_tasks), n)
        block = np.moveaxis(s[start:stop], 1, -1)          # [nc,F,H,K]
        order = np.argsort(rng.random(block.shape, dtype=np.float64), axis=-1)
        out[start:stop] = np.moveaxis(
            np.take_along_axis(block, order, axis=-1), -1, 1)
    return out


def gaussian_samples(mean, std, seed: int = 0, n_samples: int = 32) -> np.ndarray:
    """从对角高斯 ``N(mean, std²)`` 采样 [N,K,F,H]（逐元素独立，即"独立噪声基线"）。"""
    rng = np.random.default_rng(int(seed))
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.abs(np.asarray(std, dtype=np.float32)), SIGMA_FLOOR)
    noise = rng.standard_normal(
        (mean.shape[0], int(n_samples)) + mean.shape[1:], dtype=np.float32)
    return mean[:, None, :, :] + std[:, None, :, :] * noise


def ensemble_rank(samples, target, mask=None):
    """真值在 K 条样本中的名次 ``rank = Σ1(x_i ≤ y) + 1`` ∈ [1, K+1]。

    返回 ``(rank [N,F,H] float64, valid [N,F,H] bool)``。校准模型的 rank 应在
    1..K+1 上均匀分布（离散 PIT 的“秩直方图”口径）。
    """
    s = _as_samples(samples).astype(np.float32)
    y = np.asarray(target, dtype=np.float32)
    n, k, f, h = s.shape
    if tuple(y.shape) != (n, f, h):
        raise ValueError(f'target 期望 {(n, f, h)}，收到 {tuple(y.shape)}')
    valid = _element_valid(mask, n, f, h) & np.isfinite(y)
    cmp = (s <= y[:, None, :, :]).sum(axis=1, dtype=np.float64)
    return np.where(valid, cmp + 1.0, np.nan), valid


def pit_diagnostics(samples, target, mask=None, seed: int = 0,
                   n_hist_bins: int = 10) -> Dict[str, object]:
    """概率积分变换（PIT）与秩直方图。

    秩 ``rank = Σ1(x_i ≤ y) + 1`` 在 1..K+1 上；校准模型应使其均匀（秩直方图为平
    坦），均值 ``(K+2)/2``。随机化 PIT 取 ``u = ((rank−1) + U)/K, U~N(0,1)``，用来
    消除有限 K 的离散性；其在有限 K 下的**精确**理想矩为

        E[u] = 0.5 + 1/(2K)，Var[u] = (K+2)/(12K) + 1/(12K²)

    （K→∞ 时回到均匀分布的 0.5 与 1/12）。报告同时给出该精确值与 K→∞ 参照值，
    判读以 ``pit_mean_ideal`` / ``pit_std_ideal`` 为准，避免把有限采样偏差误读成
    校准问题。典型失配形态：分布外预测过多 → U 形（欠散/过度自信）；真值总落在
    集合中部 → 中间隆起（过散/信心不足）。种子固定保证可复现。
    """
    rank, valid = ensemble_rank(samples, target, mask)
    k = int(np.asarray(samples).shape[1])
    rng = np.random.default_rng(int(seed))
    u = (rank - 1.0 + rng.random(rank.shape)) / float(k)
    edges = np.linspace(0.0, 1.0, int(n_hist_bins) + 1)
    hist = np.histogram(u[valid], bins=edges)[0].astype(np.float64)
    total = float(hist.sum())
    per_horizon_mean, per_horizon_std = [], []
    h = u.shape[2]
    for hi in range(h):
        col = u[:, :, hi][valid[:, :, hi]]
        per_horizon_mean.append(float(col.mean()) if col.size else float('nan'))
        per_horizon_std.append(float(col.std(ddof=1)) if col.size > 1 else float('nan'))
    task_pit = np.full((u.shape[0],), np.nan)
    cnt = valid.sum(axis=(1, 2))
    task_pit[cnt > 0] = np.where(valid, u, 0.0).sum(axis=(1, 2))[cnt > 0] \
        / cnt[cnt > 0]
    ideal_mean = 0.5 + 1.0 / (2.0 * k)
    ideal_var = (float(k) + 2.0) / (12.0 * k) + 1.0 / (12.0 * k * k)
    return {'n_elements': int(valid.sum()),
            'n_bins': int(n_hist_bins),
            'bin_edges': [float(e) for e in edges],
            'rank_histogram_density': (hist / total).tolist() if total else [],
            'uniform_expected_density': [1.0 / float(n_hist_bins)] * int(n_hist_bins),
            'rank_mean': float(np.nanmean(rank)),
            'rank_ideal_mean': (float(k) + 2.0) / 2.0,
            'pit_mean': float(np.nanmean(u[valid])) if valid.any() else float('nan'),
            'pit_std': float(np.nanstd(u[valid], ddof=1)) if valid.sum() > 1 else float('nan'),
            'pit_mean_ideal': ideal_mean,
            'pit_std_ideal': float(math.sqrt(ideal_var)),
            'pit_mean_by_horizon': per_horizon_mean,
            'pit_std_by_horizon': per_horizon_std,
            'pit_task_mean': task_pit,
            'ideal_infinite_K': {'pit_mean': 0.5,
                                 'pit_std': float(1.0 / math.sqrt(12.0))},
            'definition': '随机化 PIT=((rank−1)+U)/K，rank∈[1,K+1]；校准时应匹配'
                          '有限 K 的精确理想矩（见 pit_mean_ideal/pit_std_ideal），'
                          '秩直方图应接近均匀'}


def finite_k_coverage_ceiling(k: int, levels: Sequence[float],
                              n_sim: int = CEILING_SIM_COUNT,
                              seed: int = 0) -> Dict[str, Dict[str, float]]:
    """完全校准模型在有限 K 条采样下的经验覆盖率/区间宽度（蒙特卡洛上界）。

    样本分位数是区间估计，有限 K 会系统性低估中心区间覆盖率——即使预测分布与真值
    分布完全一致。因此 ``coverage`` 不能直接与标称值比较：判据应是
    ``coverage / ceiling``。这里以标准正态边际（与分位数插值方式无关的线性插值实现）
    模拟，返回每个标称水平的 ``coverage`` 与 ``width``。K 越大越接近标称值。
    """
    k = int(k)
    if k < 1:
        raise ValueError(f'k 必须 >= 1，收到 {k}')
    n_sim = int(n_sim)
    if n_sim < 1000:
        raise ValueError(f'n_sim 太小，无法稳定估计上界：{n_sim}')
    rng = np.random.default_rng(int(seed))
    # 分块模拟，避免一次性生成超大样本矩阵
    out = {}
    blocks = max(1, n_sim // 200_000)
    chunk = max(1, n_sim // blocks)
    acc = {float(level): [0.0, 0.0, 0] for level in levels}
    for _ in range(blocks):
        drawn = min(chunk, n_sim)
        s = np.sort(rng.standard_normal((drawn, k, 1, 1)).astype(np.float32), axis=1)
        y = rng.standard_normal((drawn, 1, 1)).astype(np.float32)
        for level in levels:
            cov, wid = _interval_sorted(s, y, float(level), None)
            acc[float(level)][0] += float(np.nansum(cov))
            acc[float(level)][1] += float(np.nansum(wid))
            acc[float(level)][2] += int(np.isfinite(cov).sum())
    for level, (cov_sum, wid_sum, cnt) in acc.items():
        if cnt == 0:
            raise RuntimeError('上界模拟没有得到有限元素')
        out[str(level)] = {'nominal': level,
                           'empirical_coverage': cov_sum / cnt,
                           'empirical_width': wid_sum / cnt,
                           'n_sim': int(cnt),
                           'infinite_K_coverage': level,
                           'infinite_K_width': float(2.0 * _normal_ppf(level))}
    return out


def _normal_ppf(prob: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 * (1.0 + float(prob))))


def temporal_variogram(samples, target, mask=None) -> Dict[str, np.ndarray]:
    """时间变差函数得分（逐任务标量 + 分 lag 明细）。

        γ(τ) = mean_{t,f}(x_{t+τ,f} − x_{t,f})²，score = mean_τ (γ_pred − γ_true)²

    与 :func:`train.prob_losses.temporal_variogram_score` 同式，但保留逐任务值以便
    做被试级配对；只在 ``t`` 与 ``t+τ`` 都有效的元素上统计。
    """
    s = _as_samples(samples).astype(np.float64)
    y = np.asarray(target, dtype=np.float64)
    n, k, f, h = s.shape
    if tuple(y.shape) != (n, f, h):
        raise ValueError(f'target 期望 {(n, f, h)}，收到 {tuple(y.shape)}')
    m = _element_valid(mask, n, f, h).astype(np.float64)
    pred_lag, true_lag, valid_lag = [], [], []
    for tau in range(1, h):
        w = m[:, :, tau:] * m[:, :, :-tau]                  # [N,F,H-τ]
        dp = s[:, :, :, tau:] - s[:, :, :, :-tau]           # [N,K,F,H-τ]
        dt = y[:, :, tau:] - y[:, :, :-tau]                 # [N,F,H-τ]
        denom = w.sum(axis=(1, 2))
        ok = denom > 0
        pg = ((dp * dp).mean(axis=1) * w).sum(axis=(1, 2)) / np.where(ok, denom, 1.0)
        tg = ((dt * dt) * w).sum(axis=(1, 2)) / np.where(ok, denom, 1.0)
        pred_lag.append(np.where(ok, pg, np.nan))
        true_lag.append(np.where(ok, tg, np.nan))
        valid_lag.append(ok.astype(np.float64))
    pred = np.stack(pred_lag, axis=1)                       # [N,H-1]
    true = np.stack(true_lag, axis=1)
    weight = np.stack(valid_lag, axis=1)
    # 无效 lag 先置 0 再加权：NaN×0 仍是 NaN，会让整任务被误判为缺失
    sq = np.where(weight > 0, np.square(pred - true), 0.0)
    per_task = sq.sum(axis=1) / np.maximum(weight.sum(axis=1), 1.0)
    per_task = np.where(weight.sum(axis=1) > 0, per_task, np.nan)
    return {'per_task_score': per_task,
            'pred_gamma': pred, 'true_gamma': true,
            'lag_valid': weight.astype(bool)}


def energy_score(samples, target, mask=None, chunk_tasks: int = ENERGY_CHUNK
                 ) -> np.ndarray:
    """集合 Energy Score（逐任务标量，距离用 masked RMS over (F,H)）。

        ES = mean_k rms(x_k − y) − 0.5 · mean_{i,j} rms(x_i − x_j)

    两两项用 Gram 恒等式按 ``(F,H)`` 向量化，避免显式 ``K×K`` 差分值；分块处理以
    限制峰值内存。与 :func:`train.prob_losses.ensemble_energy_score` 同式。
    """
    s = _as_samples(samples).astype(np.float32)
    y = np.asarray(target, dtype=np.float32)
    n, k, f, h = s.shape
    if tuple(y.shape) != (n, f, h):
        raise ValueError(f'target 期望 {(n, f, h)}，收到 {tuple(y.shape)}')
    out = np.full(n, np.nan, dtype=np.float64)
    for start in range(0, n, int(chunk_tasks)):
        stop = min(start + int(chunk_tasks), n)
        w = _element_valid(mask, n, f, h)[start:stop].astype(np.float32)
        cnt = w.sum(axis=(1, 2)).astype(np.float64)           # 有效 (ROI,horizon) 元素数
        ok = cnt > 0
        sw = np.sqrt(w)[:, None, :, :]                        # [nc,1,F,H]
        x = (s[start:stop] * sw).reshape(stop - start, k, -1) \
            .astype(np.float64)                               # [nc,K,F·H]
        yy = (y[start:stop] * np.sqrt(w)).reshape(stop - start, -1) \
            .astype(np.float64)                               # [nc,F·H]
        denom = np.where(ok, cnt, 1.0)
        x2 = (x * x).sum(axis=2)                              # [nc,K]
        y2 = (yy * yy).sum(axis=1)                            # [nc]
        d1 = np.sqrt(np.maximum(
            x2 + y2[:, None] - 2.0 * (x * yy[:, None, :]).sum(axis=2),
            0.0) / denom[:, None])
        gram = np.matmul(x, x.transpose(0, 2, 1))              # [nc,K,K] 批量 BLAS
        d2 = np.sqrt(np.maximum(
            x2[:, :, None] + x2[:, None, :] - 2.0 * gram, 0.0)
            / denom[:, None, None])
        es = d1.mean(axis=1) - 0.5 * d2.sum(axis=(1, 2)) / float(k * k)
        out[start:stop] = np.where(ok, es, np.nan)
    return out


def reduce_tasks(values: np.ndarray, per_horizon: bool = False) -> np.ndarray:
    """[N,F,H] 元素指标 → 逐任务均值（先 ROI，再 horizon）。

    ``per_horizon=True`` 时只沿 ROI 平均，保留 [N,H] 供分 horizon 报告。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError(f'期望 [N,F,H]，收到 {v.shape}')
    n, f, h = v.shape
    finite = np.isfinite(v)
    if per_horizon:
        cnt = finite.sum(axis=1)
        total = np.where(finite, v, 0.0).sum(axis=1)
        return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    cnt = finite.sum(axis=(1, 2))
    total = np.where(finite, v, 0.0).sum(axis=(1, 2))
    return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)


def subject_means(task_values: np.ndarray, subj_idx: Sequence[str]) -> np.ndarray:
    """逐任务值 [N] 或 [N,C] → 被试均值；返回 [S] 或 [S,C]。"""
    v = np.asarray(task_values, dtype=np.float64)
    subj = np.asarray([str(s) for s in subj_idx])
    if v.shape[0] != subj.shape[0]:
        raise ValueError(f'任务行数 {v.shape[0]} 与被试标签数 {subj.shape[0]} 不一致')
    uniq = np.unique(subj)
    out = []
    for s in uniq:
        sel = subj == s
        if v.ndim == 1:
            col = v[sel]
            col = col[np.isfinite(col)]
            out.append(float(col.mean()) if col.size else np.nan)
        else:
            block = v[sel]
            finite = np.isfinite(block)
            cnt = finite.sum(axis=0)
            total = np.where(finite, block, 0.0).sum(axis=0)
            out.append(np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan))
    return np.asarray(out, dtype=np.float64)


def summary(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'sem': float('nan'),
                'median': float('nan'), 'n': 0}
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if v.size > 1 else 0.0,
            'sem': float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0,
            'median': float(np.median(v)), 'n': int(v.size)}


def paired_stats(a_values: Sequence[float], b_values: Sequence[float]) -> Dict[str, float]:
    """被试级配对统计（a−b）：配对 t、Wilcoxon、Cohen's dz 与 95% CI。"""
    from scipy import stats as st

    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f'配对数组形状不一致：{a.shape} vs {b.shape}')
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(a.size)
    diff = a - b
    out = {'n': n, 'mean_a': float(a.mean()) if n else float('nan'),
           'mean_b': float(b.mean()) if n else float('nan'),
           'mean_diff': float(diff.mean()) if n else float('nan'),
           'std_diff': float(diff.std(ddof=1)) if n > 1 else float('nan')}
    if n < 2 or not np.isfinite(diff.std(ddof=1)) or diff.std(ddof=1) <= 0:
        out.update({'t': float('nan'), 'p': float('nan'),
                    'wilcoxon_p': float('nan'), 'cohens_dz': float('nan'),
                    'ci95_low': float('nan'), 'ci95_high': float('nan')})
        return out
    t, p = st.ttest_rel(a, b)
    try:
        wil = float(st.wilcoxon(a, b).pvalue)
    except ValueError:
        wil = float('nan')
    dz = float(diff.mean() / diff.std(ddof=1))
    half = float(st.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n))
    out.update({'t': float(t), 'p': float(p), 'wilcoxon_p': wil,
                'cohens_dz': dz,
                'ci95_low': float(diff.mean() - half),
                'ci95_high': float(diff.mean() + half)})
    return out


# ══════════════════════════════════════════════════════════════════════════
#  §2 npz 读取、常数方差参照、分位数分层
# ══════════════════════════════════════════════════════════════════════════

def load_npz(path) -> Dict[str, np.ndarray]:
    """读取评估 npz；旧文件含 object 字段时显式回退，但新输出应可无 pickle 读取。"""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except ValueError as exc:
        warnings.warn(f'{path} 含 object 字段，回退 allow_pickle=True：{exc}')
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}


def build_constant_variance(train_mean: np.ndarray, train_target: np.ndarray,
                            train_mask: np.ndarray, horizon: int,
                            sigma_floor: float = SIGMA_FLOOR):
    """用 train 残差估计 ``(var_roi_h [F,H], var_global, counts [F,H])``。"""
    h = int(horizon)
    mean = np.asarray(train_mean, dtype=np.float64)[:, :, :h]
    target = np.asarray(train_target, dtype=np.float64)[:, :, :h]
    if mean.shape != target.shape:
        raise ValueError(f'train mean/target 形状不一致：{mean.shape}/{target.shape}')
    mask = np.asarray(train_mask, dtype=bool)[:, :h]
    if mask.shape != (mean.shape[0], h):
        raise ValueError(f'train mask 形状应为 {(mean.shape[0], h)}，收到 {mask.shape}')
    resid2 = (target - mean) ** 2
    valid = np.broadcast_to(mask[:, None, :], resid2.shape)
    counts = valid.sum(axis=0).astype(np.float64)                 # [F,H]
    sums = np.where(valid, resid2, 0.0).sum(axis=0)
    var_roi_h = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    total_count = float(counts.sum())
    var_global = float(np.where(valid, resid2, 0.0).sum() / max(total_count, 1.0))
    floor = float(sigma_floor) ** 2
    var_global = max(var_global, floor)
    var_roi_h = np.where(np.isfinite(var_roi_h), var_roi_h, var_global)
    var_roi_h = np.maximum(var_roi_h, floor)
    return var_roi_h, var_global, counts


def quantile_edges(magnitude: np.ndarray, mask: np.ndarray,
                   n_bins: int) -> np.ndarray:
    """按真实 |Δy| 的全局分位确定 Q1–Q5 边界（末边界 +inf）。"""
    if int(n_bins) < 1:
        raise ValueError(f'n_bins 必须 >= 1，收到 {n_bins}')
    mag = np.asarray(magnitude, dtype=np.float64)
    valid = np.broadcast_to(np.asarray(mask, dtype=bool)[:, None, :], mag.shape)
    vals = mag[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError('没有有效的 |Δy| 元素，无法确定分位边界')
    edges = np.quantile(vals, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[0] = 0.0
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def bin_task_means(values: np.ndarray, magnitude: np.ndarray, mask: np.ndarray,
                   edges: np.ndarray) -> np.ndarray:
    """逐任务、逐 Q 的有效元素均值 → [N,n_bins]（无元素为 NaN）。"""
    v = np.asarray(values, dtype=np.float64)
    mag = np.asarray(magnitude, dtype=np.float64)
    valid = np.broadcast_to(np.asarray(mask, dtype=bool)[:, None, :], v.shape)
    which = np.digitize(mag, np.asarray(edges, dtype=np.float64)[1:-1], right=False)
    n, nb = v.shape[0], len(edges) - 1
    out = np.full((n, nb), np.nan, dtype=np.float64)
    for b in range(nb):
        sel = valid & (which == b) & np.isfinite(v)
        cnt = sel.sum(axis=(1, 2))
        total = np.where(sel, v, 0.0).sum(axis=(1, 2))
        out[:, b] = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  §3 评估主流程
# ══════════════════════════════════════════════════════════════════════════

def _metric_keys(metrics) -> Tuple[str, ...]:
    """逐元素标量指标键（解析模式为 nll/crps；集合模式另含 crps_fair）。"""
    return tuple(k for k in ('nll', 'crps', 'crps_fair') if k in metrics)


def _element_metrics(mean, std, target, mask, levels):
    nll = gaussian_nll(mean, std, target, mask)
    crps = gaussian_crps(mean, std, target, mask)
    coverage, width = {}, {}
    for level in levels:
        cov, wid = gaussian_interval(mean, std, target, level, mask)
        coverage[level] = cov
        width[level] = wid
    return {'nll': nll, 'crps': crps,
            'coverage': coverage, 'width': width}


def _ensemble_element_metrics(samples, target, mask, levels):
    """[N,K,F,H] 样本集合 → 与 :func:`_element_metrics` 同结构的指标字典。

    经验 CRPS 与分位数区间共用一次排序；``nll`` 对任意集合无定义，置 NaN 以保持
    下游聚合接口一致（``summary``/``paired_stats`` 会跳过非有限值）。``crps_fair``
    额外给出无有限集合偏差的口径（见 :func:`_crps_sorted`）。
    """
    s = np.sort(_as_samples(samples), axis=1)
    crps = _crps_sorted(s, target, mask)
    coverage, width = {}, {}
    for level in levels:
        cov, wid = _interval_sorted(s, target, level, mask)
        coverage[level] = cov
        width[level] = wid
    return {'nll': np.full_like(crps, np.nan), 'crps': crps,
            'crps_fair': _crps_sorted(s, target, mask, fair=True),
            'coverage': coverage, 'width': width}


def _method_summary(metrics, mask, subj_idx):
    out = {}
    for key in _metric_keys(metrics):
        task_vals = reduce_tasks(metrics[key])
        out[key] = summary(subject_means(task_vals, subj_idx))
    for level in metrics['coverage']:
        tag = f'{int(round(level * 100))}'
        cov_task = reduce_tasks(metrics['coverage'][level])
        wid_task = reduce_tasks(metrics['width'][level])
        out[f'coverage{tag}'] = summary(subject_means(cov_task, subj_idx))
        out[f'width{tag}'] = summary(subject_means(wid_task, subj_idx))
    # pooled 元素级覆盖率与平均宽度（校准曲线用；报告中显式区分 pooled）
    for level in metrics['coverage']:
        tag = f'{int(round(level * 100))}'
        cov = metrics['coverage'][level]
        wid = metrics['width'][level]
        out[f'pooled_coverage{tag}'] = float(np.nanmean(cov))
        out[f'pooled_width{tag}'] = float(np.nanmean(wid))
    out['pooled_nll'] = _nanmean_or_nan(metrics['nll'])
    out['pooled_crps'] = _nanmean_or_nan(metrics['crps'])
    if 'crps_fair' in metrics:
        out['pooled_crps_fair'] = _nanmean_or_nan(metrics['crps_fair'])
    return out


def _nanmean_or_nan(values) -> float:
    """全 NaN 时返回 NaN 而不触发 numpy 告警（集合模式的 nll 恒为 NaN）。"""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    return float(v[finite].mean()) if finite.any() else float('nan')


def evaluate_arrays(test: Dict[str, np.ndarray], var_roi_h: np.ndarray,
                    var_global: float, n_bins: int = 5,
                    levels: Sequence[float] = DEFAULT_LEVELS,
                    sigma_floor: float = SIGMA_FLOOR,
                    samples: Optional[np.ndarray] = None,
                    iid_seed: int = 0) -> Dict:
    """在已加载的 test 数组上计算概率指标。

    ``samples=None`` 为 S4-0 路径：模型行使用解析对角高斯（``dynamics_std``）。
    传入 ``[N,K,F,H]`` 样本集合时进入 S4-1/S4-2 路径，模型行改用经验 CRPS 与样本
    分位数区间，并追加两行对照：

    - ``model_iid``：逐元素打乱样本维，边际分布不变、依赖结构被摧毁，用于判断概率
      收益是否来自相关结构而非单纯放大方差；
    - ``constant_gaussian_K``：用与模型**相同 K** 独立采样的常数方差高斯。解析常数行
      （``constant_roi_h``/``constant_global``）是 K→∞ 口径，其 CRPS 偏低、覆盖率偏
      高，不能与有限集合行直接比较；跨方法结论应以同 K 行为准。

    集合模式还输出 ``ensemble`` 块：Energy Score、时间变差函数、随机化 PIT 与秩直方图，
    以及有限 K 覆盖率上界 ``coverage_ceiling`` 与 ``coverage_over_ceiling``。
    """
    if 'dynamics_std' not in test or np.asarray(test['dynamics_std']).size == 0:
        raise KeyError('test npz 缺少 dynamics_std；请先用新版 run_inference 重跑评估并 --save_arrays')
    mean_all = np.asarray(test['ode_pred'], dtype=np.float64)
    std_all = np.asarray(test['dynamics_std'], dtype=np.float64)
    future = np.asarray(test['future'], dtype=np.float64)
    fmask = np.asarray(test['future_mask'], dtype=bool)
    anchor = np.asarray(test['x_last'], dtype=np.float64)
    subj_idx = np.asarray([str(s) for s in test['subj_ids']])
    if mean_all.size == 0 or std_all.size == 0:
        raise ValueError('ode_pred/dynamics_std 为空')
    h = min(mean_all.shape[2], std_all.shape[2], future.shape[2])
    mean = mean_all[:, :, :h]
    std = std_all[:, :, :h]
    target = future[:, :, :h]
    mask = fmask[:, :h]
    if anchor.shape != mean.shape[:2]:
        raise ValueError(f'x_last 形状 {anchor.shape} 与 mean [N,F,H] 不匹配')
    if not np.isfinite(std).all():
        raise ValueError('dynamics_std 含非有限值，无法评估高斯预测')
    if (std <= 0).any():
        warnings.warn('dynamics_std 含非正值，已按 sigma_floor 截断')

    ref = np.sqrt(np.asarray(var_roi_h, dtype=np.float64))[None, :, :h]
    ref_global = math.sqrt(float(var_global))
    std_map = {
        'model': std,
        'constant_roi_h': np.broadcast_to(ref, std.shape),
        'constant_global': np.full_like(std, ref_global),
    }

    method_names = list(METHODS)
    metrics = {}
    shuffled = None
    const_samples = None
    if samples is not None:
        k_ens = int(samples.shape[1])
        metrics['model'] = _ensemble_element_metrics(samples, target, mask, levels)
        shuffled = iid_shuffle_samples(samples, seed=iid_seed)
        method_names.insert(1, 'model_iid')
        metrics['model_iid'] = _ensemble_element_metrics(
            shuffled, target, mask, levels)
        # 与模型同一 K、同一锚点的 iid 高斯基线：解析常数行的 CRPS/覆盖率是
        # K→∞ 口径，与有限集合不可直接比较；本行才是可比的同 K 参照。
        const_samples = gaussian_samples(mean, np.broadcast_to(ref, mean.shape),
                                         seed=iid_seed + 1, n_samples=k_ens)
        method_names.append('constant_gaussian_K')
        metrics['constant_gaussian_K'] = _ensemble_element_metrics(
            const_samples, target, mask, levels)
        # S4-0 的学到的对角高斯按同一 K iid 采样：与 S4-1 只差在“σ 是否条件化”与
        # “是否有相关结构”，聚合层级与模型行完全一致
        s40_samples = gaussian_samples(mean, std, seed=iid_seed + 2,
                                       n_samples=k_ens)
        method_names.append('s4_0_gaussian_K')
        metrics['s4_0_gaussian_K'] = _ensemble_element_metrics(
            s40_samples, target, mask, levels)
    for name, std_row in std_map.items():
        if name in metrics:
            continue
        metrics[name] = _element_metrics(mean, std_row, target, mask, levels)
    subject_block = {name: _method_summary(metrics[name], mask, subj_idx)
                     for name in method_names}

    # ---- 被试级配对：模型 vs 独立性对照/同 K 高斯（仅集合模式）与常数参照 ----
    ref_names = list(METHODS[1:])
    if samples is not None:
        ref_names = ['model_iid', 'constant_gaussian_K', 's4_0_gaussian_K'] + ref_names
    paired = {}
    for ref_name in ref_names:
        block = {}
        # 只比较两侧都有的口径：解析参照行没有有限集合概念，不含 crps_fair
        for key in [k for k in _metric_keys(metrics['model'])
                    if k in metrics[ref_name]]:
            task_a = reduce_tasks(metrics['model'][key])
            task_b = reduce_tasks(metrics[ref_name][key])
            block[key] = paired_stats(subject_means(task_a, subj_idx),
                                      subject_means(task_b, subj_idx))
        for level in levels:
            tag = f'{int(round(level * 100))}'
            cov_a = subject_means(reduce_tasks(metrics['model']['coverage'][level]),
                                  subj_idx)
            cov_b = subject_means(
                reduce_tasks(metrics[ref_name]['coverage'][level]), subj_idx)
            block[f'coverage{tag}'] = paired_stats(cov_a, cov_b)
            wid_a = subject_means(reduce_tasks(metrics['model']['width'][level]),
                                  subj_idx)
            wid_b = subject_means(
                reduce_tasks(metrics[ref_name]['width'][level]), subj_idx)
            block[f'width{tag}'] = paired_stats(wid_a, wid_b)
        paired[ref_name] = block

    # ---- 分 horizon ----
    horizon_block = {}
    horizon_labels = [f'H{k}' for k in range(1, h + 1)]
    for name in method_names:
        block = {}
        for hi, label in enumerate(horizon_labels):
            row = {}
            for key in _metric_keys(metrics[name]):
                task_vals = reduce_tasks(metrics[name][key], per_horizon=True)[:, hi]
                row[key] = summary(subject_means(task_vals, subj_idx))
            for level in REPORT_LEVELS:
                tag = f'{int(round(level * 100))}'
                cov = reduce_tasks(metrics[name]['coverage'][level],
                                   per_horizon=True)[:, hi]
                wid = reduce_tasks(metrics[name]['width'][level],
                                   per_horizon=True)[:, hi]
                row[f'coverage{tag}'] = summary(subject_means(cov, subj_idx))
                row[f'width{tag}'] = summary(subject_means(wid, subj_idx))
            block[label] = row
        horizon_block[name] = block

    # ---- 校准曲线（pooled + 被试级） ----
    calibration = {}
    for level in levels:
        calibration[str(level)] = {}
        for name in method_names:
            cov = metrics[name]['coverage'][level]
            subj_cov = subject_means(reduce_tasks(cov), subj_idx)
            calibration[str(level)][name] = {
                'nominal': float(level),
                'pooled_coverage': float(np.nanmean(cov)),
                'subject_mean_coverage': float(np.nanmean(subj_cov)),
                'subject_std_coverage': float(np.nanstd(subj_cov, ddof=1)),
                'mean_width': float(np.nanmean(metrics[name]['width'][level])),
            }

    # ---- Q1–Q5 事后分层（真实 |y−x_t|，仅评估） ----
    magnitude = np.abs(target - anchor[:, :, None])
    edges = quantile_edges(magnitude, mask, n_bins)
    bins = {}
    bin_labels = [f'Q{i + 1}' for i in range(int(n_bins))]
    for name in method_names:
        bins[name] = {}
        for key in _metric_keys(metrics[name]):
            task_bins = bin_task_means(metrics[name][key], magnitude, mask, edges)
            bins[name][key] = {}
            for bi, label in enumerate(bin_labels):
                subj_vals = subject_means(task_bins[:, bi], subj_idx)
                bins[name][key][label] = summary(subj_vals)
        for level in REPORT_LEVELS:
            tag = f'{int(round(level * 100))}'
            for key, element in (('coverage', metrics[name]['coverage'][level]),
                                 ('width', metrics[name]['width'][level])):
                task_bins = bin_task_means(element, magnitude, mask, edges)
                bins[name][f'{key}{tag}'] = {}
                for bi, label in enumerate(bin_labels):
                    subj_vals = subject_means(task_bins[:, bi], subj_idx)
                    bins[name][f'{key}{tag}'][label] = summary(subj_vals)
    # 每个 Q 的真实 |Δy| 均值与元素数（pooled）
    bin_ref = {}
    for bi, label in enumerate(bin_labels):
        sel = np.broadcast_to(mask[:, None, :], magnitude.shape) & np.isfinite(magnitude)
        which = np.digitize(magnitude, edges[1:-1], right=False)
        sel_b = sel & (which == bi)
        bin_ref[label] = {
            'mean_true_abs_delta': (float(magnitude[sel_b].mean())
                                    if sel_b.any() else float('nan')),
            'n_elements': int(sel_b.sum()),
        }

    # ---- 联合分布与时间依赖（仅集合模式）----
    # Energy Score 检查跨 ROI 联合结构，变差函数检查跨 horizon 时间结构；两者都
    # 与"同边际但独立打乱"的对照比较，才能区分收益来自相关结构还是单纯放大方差。
    ensemble_block = None
    if samples is not None:
        k_ens = int(samples.shape[1])
        rows = {'model': samples, 'model_iid': shuffled,
                'constant_gaussian_K': const_samples,
                's4_0_gaussian_K': s40_samples}
        per_task, gamma_by_lag = {}, {}
        for label, arr in rows.items():
            vg = temporal_variogram(arr, target, mask)
            per_task[label] = {
                'energy_score': energy_score(arr, target, mask),
                'variogram': vg['per_task_score']}
            # 逐 lag 的 γ_pred 与 γ_true：定位时间依赖失配在哪个尺度
            lag_w = vg['lag_valid']
            denom = np.maximum(lag_w.sum(axis=0), 1)
            gamma_by_lag[label] = {
                'gamma_pred': (np.nan_to_num(vg['pred_gamma']) * lag_w).sum(axis=0)
                               / denom,
                'gamma_true': (np.nan_to_num(vg['true_gamma']) * lag_w).sum(axis=0)
                              / denom,
                'lag_valid_counts': lag_w.sum(axis=0).tolist()}
        # 有限 K 上界：完全校准的模型在 K 条采样下也达不到标称覆盖率，因此判据用
        # coverage/ceiling，而不是 coverage vs nominal。键名与 coverage80/90 对齐。
        ceiling = {f'{int(round(float(level) * 100))}': stats
                   for level, stats in
                   finite_k_coverage_ceiling(k_ens, REPORT_LEVELS).items()}
        ensemble_block = {'n_samples': k_ens, 'coverage_ceiling': ceiling}
        for label, arr in rows.items():
            pit = pit_diagnostics(arr, target, mask, seed=iid_seed)
            pit_task = pit.pop('pit_task_mean')
            row = {metric: summary(subject_means(per_task[label][metric], subj_idx))
                   for metric in ('energy_score', 'variogram')}
            row['pit'] = pit
            row['pit_subject'] = summary(subject_means(pit_task, subj_idx))
            row['variogram_by_lag'] = gamma_by_lag[label]
            row['coverage_over_ceiling'] = {
                f'{int(round(level * 100))}': float(
                    subject_block[label][
                        f'pooled_coverage{int(round(level * 100))}']
                    / ceiling[f'{int(round(level * 100))}']['empirical_coverage'])
                for level in REPORT_LEVELS}
            ensemble_block[label] = row
        ensemble_block['paired_model_minus_iid'] = {
            metric: paired_stats(subject_means(per_task['model'][metric], subj_idx),
                                 subject_means(per_task['model_iid'][metric], subj_idx))
            for metric in ('energy_score', 'variogram')}
        ensemble_block['definitions'] = {
            'energy_score': '逐任务 masked-RMS Energy Score（越低越好）',
            'variogram': 'mean_τ (γ_pred(τ) − γ_true(τ))²，越低越好',
            'model_iid': '逐 (ROI,horizon) 独立打乱样本维：边际不变、依赖结构摧毁',
            'constant_gaussian_K': 'N(μ_S2, σ_constant²) 逐元素独立采样，与模型同 K',
            'coverage_over_ceiling': '实测覆盖率 / 同 K 完全校准模型的蒙特卡洛上界，'
                                     '≈1 表示边际校准良好',
        }

    return {
        'horizon': int(h),
        'n_tasks': int(mean.shape[0]),
        'n_subjects': int(len(np.unique(subj_idx))),
        'sigma': {
            'model_mean': float(std.mean()),
            'constant_roi_h_mean': float(ref.mean()),
            'constant_global': ref_global,
            'model_over_constant_roi_h_mean': float(
                std.mean() / max(ref.mean(), sigma_floor)),
            'model_min': float(std.min()), 'model_max': float(std.max()),
        },
        'subject_level': subject_block,
        'paired_model_minus_reference': paired,
        'horizon_level': horizon_block,
        'calibration': calibration,
        'amplitude_strata': {
            'labels': bin_labels,
            'edges': [float(e) if np.isfinite(e) else None for e in edges],
            'reference': bin_ref,
            'methods': bins,
            'definition': 'Q1..Q5 按真实 |y_{t+h,f}−x_{t,f}| 的全局五分位分层；'
                          '仅用于事后评估，不进入任何条件输入',
        },
        'mask_count': int(np.asarray(mask, dtype=bool).sum()),
        'ensemble': ensemble_block,
        # 逐元素指标供 CSV 复用（集合模式无法由 mean+std 重算）；不进 JSON
        '_metrics': metrics,
        '_arrays': {'levels': levels, 'bin_labels': bin_labels},
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_csvs(report: Dict, test: Dict[str, np.ndarray], subj_idx,
               out_dir: Path, levels: Sequence[float],
               metrics: Optional[Dict[str, Dict]] = None,
               prefix: str = 's4_0') -> Dict[str, str]:
    """逐被试、逐 horizon、逐被试×horizon 三个 CSV。

    ``metrics`` 为 ``evaluate_arrays`` 已算好的逐元素指标（集合模式必须传入，因为
    经验 CRPS/分位数区间无法由 mean+std 重算）；缺省时按解析高斯重新计算。
    CSV 与 JSON 均先逐锚点归一化，再被试内平均，最后跨被试等权平均。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(test['ode_pred'], dtype=np.float64)
    std = np.asarray(test['dynamics_std'], dtype=np.float64)
    future = np.asarray(test['future'], dtype=np.float64)
    fmask = np.asarray(test['future_mask'], dtype=bool)
    h = report['horizon']
    mean, std = mean[:, :, :h], std[:, :, :h]
    target = future[:, :, :h]
    mask = fmask[:, :h]
    var_roi_h = np.asarray(report['_var_roi_h'], dtype=np.float64)
    ref = np.broadcast_to(np.sqrt(var_roi_h)[None, :, :h], std.shape)
    ref_global = math.sqrt(float(report['_var_global']))
    std_maps = {'model': std, 'constant_roi_h': ref,
                'constant_global': np.full_like(std, ref_global)}
    if metrics is None:
        metrics = {name: _element_metrics(mean, std_maps[name], target, mask, levels)
                   for name in METHODS}
    method_names = list(metrics)
    subj = np.asarray([str(s) for s in subj_idx])
    uniq = np.unique(subj)

    # 逐被试总体
    rows = []
    for s in uniq:
        sel = subj == s
        row = {'subject_id': s, 'n_tasks': int(sel.sum())}
        for name in method_names:
            m = metrics[name]
            for key in _metric_keys(m):
                row[f'{name}_{key}'] = _nanmean_or_nan(reduce_tasks(m[key][sel]))
            for level in REPORT_LEVELS:
                tag = f'{int(round(level * 100))}'
                row[f'{name}_coverage{tag}'] = _nanmean_or_nan(
                    reduce_tasks(m['coverage'][level][sel]))
                row[f'{name}_width{tag}'] = _nanmean_or_nan(
                    reduce_tasks(m['width'][level][sel]))
        if 'constant_roi_h' in method_names:
            row['delta_nll_model_minus_constant_roi_h'] = (
                row['model_nll'] - row['constant_roi_h_nll'])
            row['delta_crps_model_minus_constant_roi_h'] = (
                row['model_crps'] - row['constant_roi_h_crps'])
        rows.append(row)
    subject_csv = out_dir / f'{prefix}_subject_metrics_test.csv'
    with subject_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 逐 horizon（被试级均值）
    horizon_rows = []
    for hi, label in enumerate(report['_horizon_labels']):
        for name in method_names:
            m = metrics[name]
            row = {'horizon': label, 'method': name}
            for key in _metric_keys(m):
                row[f'{key}_subject_mean'] = _nanmean_or_nan(
                    subject_means(reduce_tasks(m[key][:, :, hi:hi + 1]), subj))
            for level in REPORT_LEVELS:
                tag = f'{int(round(level * 100))}'
                row[f'coverage{tag}_subject_mean'] = _nanmean_or_nan(
                    subject_means(reduce_tasks(
                        m['coverage'][level][:, :, hi:hi + 1]), subj))
                row[f'width{tag}_subject_mean'] = _nanmean_or_nan(
                    subject_means(reduce_tasks(
                        m['width'][level][:, :, hi:hi + 1]), subj))
            horizon_rows.append(row)
    horizon_csv = out_dir / f'{prefix}_horizon_metrics_test.csv'
    with horizon_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(horizon_rows[0].keys()))
        w.writeheader(); w.writerows(horizon_rows)

    # 逐被试 × horizon（长表，供配对/分层复核）
    long_rows = []
    for s in uniq:
        sel = subj == s
        for hi, label in enumerate(report['_horizon_labels']):
            for name in method_names:
                m = metrics[name]
                row = {'subject_id': s, 'horizon': label, 'method': name,
                       'n_tasks': int(sel.sum())}
                for key in _metric_keys(m):
                    row[key] = _nanmean_or_nan(
                        reduce_tasks(m[key][sel][:, :, hi:hi + 1]))
                for level in REPORT_LEVELS:
                    tag = f'{int(round(level * 100))}'
                    row[f'coverage{tag}'] = _nanmean_or_nan(
                        reduce_tasks(m['coverage'][level][sel][:, :, hi:hi + 1]))
                    row[f'width{tag}'] = _nanmean_or_nan(
                        reduce_tasks(m['width'][level][sel][:, :, hi:hi + 1]))
                long_rows.append(row)
    long_csv = out_dir / f'{prefix}_subject_horizon_metrics_test.csv'
    with long_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader(); w.writerows(long_rows)
    return {'subject_csv': str(subject_csv), 'horizon_csv': str(horizon_csv),
            'subject_horizon_csv': str(long_csv)}


def make_plots(report: Dict, out_dir: Path, variant: str = 'S4-0') -> Dict[str, str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = ['#1f77b4', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd']
    labels = {'model': 'model', 'model_iid': 'model (iid-shuffled)',
              'constant_roi_h': 'constant (ROI×H)',
              'constant_global': 'constant (global)'}
    method_names = list(report['subject_level'])
    colors = {name: palette[i % len(palette)] for i, name in enumerate(method_names)}
    labels = {name: labels.get(name, name) for name in method_names}

    # 1) 校准曲线
    levels = [float(k) for k in report['calibration']]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
    for name in method_names:
        vals = [report['calibration'][str(level)][name]['pooled_coverage']
                for level in levels]
        ax.plot(levels, vals, 'o-', color=colors[name], label=labels[name])
    ax.set_xlabel('nominal coverage')
    ax.set_ylabel('empirical coverage (pooled)')
    ax.set_title(f'{variant} calibration (test)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    cal_path = out_dir / 'calibration_curve.png'
    fig.savefig(cal_path, dpi=160); plt.close(fig)

    # 2) Q1–Q5
    bins = report['amplitude_strata']
    qlabels = bins['labels']
    panels = [('nll', 'NLL'), ('crps', 'CRPS'),
              ('coverage80', 'coverage @80%'), ('width80', 'width @80%')]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name in method_names:
            vals = [bins['methods'][name][key][q]['mean'] for q in qlabels]
            ax.plot(qlabels, vals, 'o-', color=colors[name], label=labels[name])
        ax.set_title(title); ax.grid(alpha=0.25)
        if key.startswith('coverage'):
            ax.axhline(0.8, color='k', ls=':', lw=1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{variant} by true |y−x_t| quintile', y=0.99)
    fig.tight_layout()
    q_path = out_dir / 'q1_q5.png'
    fig.savefig(q_path, dpi=160); plt.close(fig)

    # 3) horizon
    hz = report['horizon_level']
    hlabels = list(hz['model'].keys())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, key, title in ((axes[0], 'nll', 'NLL'), (axes[1], 'crps', 'CRPS')):
        for name in method_names:
            vals = [hz[name][h][key]['mean'] for h in hlabels]
            ax.plot(hlabels, vals, 'o-', color=colors[name], label=labels[name])
        ax.set_title(title); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{variant} by horizon', y=0.99)
    fig.tight_layout()
    hz_path = out_dir / 'horizon_probabilistic.png'
    fig.savefig(hz_path, dpi=160); plt.close(fig)

    return {'calibration_png': str(cal_path), 'q1_q5_png': str(q_path),
            'horizon_png': str(hz_path)}


# ══════════════════════════════════════════════════════════════════════════
#  §4 train 参照推断与 CLI
# ══════════════════════════════════════════════════════════════════════════

def run_train_reference(ckpt: str, out_path: Path, device: str = '',
                        anchors_per_subject: int = 16, seed: int = 2024) -> Path:
    """用 checkpoint 在 **train 被试** 的确定性评估视图上前向，保存参照 npz。"""
    import torch
    from analysis import next_point_eval as npe
    from analysis.checkpoint import load_checkpoint_compat
    from experiments.evaluate_variant import (build_model_from_args,
                                              load_condition_bundle,
                                              load_train_args)
    from utils.common import resolve_forecast_offsets, set_seed

    set_seed(int(seed))
    dev = torch.device(device) if device else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    args = load_train_args(ckpt)
    bundle = load_condition_bundle(ckpt)
    model = build_model_from_args(args, dev, condition_bundle=bundle)
    model, compat = load_checkpoint_compat(model, ckpt, dev, strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')
    if anchors_per_subject > 0:
        args.eval_anchors_per_subject = int(anchors_per_subject)
    loader, subjects, _ = npe.build_next_point_loader(
        args, 'train', condition_bundle=bundle)
    offsets = resolve_forecast_offsets(args)
    data = npe.run_inference(model, loader, dev, offsets, rollout_steps=0,
                             eval_rollout=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    npe.save_predictions_npz(
        out_path,
        ode_pred=data['ode_pred'], dynamics_std=data['dynamics_std'],
        dynamics_logvar_norm=data['dynamics_logvar_norm'],
        future=data['future'], future_mask=data['future_mask'],
        x_last=data['x_last'], subj_ids=np.asarray(data['subj_ids']).astype(str),
        pred=data['pred'], target=data['target'], mask=data['mask'])
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description='S4 概率轨迹评估（对角高斯 S4-0 / 样本集合 S4-1·S4-2）')
    ap.add_argument('--pred_npz', required=True,
                    help='test 评估 npz（含 ode_pred / dynamics_std / future / future_mask）')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--train_npz', default='',
                    help='可选的 train 参照 npz；为空时用 --ckpt 现场推断并落盘')
    ap.add_argument('--ckpt', default='',
                    help='checkpoint 路径（train 参照推断必需）')
    ap.add_argument('--device', default='')
    ap.add_argument('--n_bins', type=int, default=5)
    ap.add_argument('--sigma_floor', type=float, default=SIGMA_FLOOR)
    ap.add_argument('--train_anchors_per_subject', type=int, default=16)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--variant', default='',
                    help="报告中的实验标签；缺省时解析式高斯记为 S4-0，"
                         '集合模式从 --ckpt 快照的 s4_variant 推导')
    ap.add_argument('--prefix', default='',
                    help='输出 CSV 文件名前缀（缺省 S4-0→s4_0，集合→s4_prob）')
    ap.add_argument('--iid_seed', type=int, default=0,
                    help='独立性对照（逐元素打乱样本维）的随机种子')
    ns = ap.parse_args()

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_npz(ns.pred_npz)
    if 'dynamics_std' not in test or np.asarray(test['dynamics_std']).size == 0:
        raise SystemExit('pred npz 缺少 dynamics_std；请用新版 run_inference 重跑评估。')
    samples = (np.asarray(test['dynamics_samples'])
               if 'dynamics_samples' in test else None)
    if samples is not None and samples.size == 0:
        samples = None
    if samples is not None and samples.ndim != 4:
        raise SystemExit(f'dynamics_samples 期望 [N,K,F,H]，收到 {samples.shape}')
    ensemble = samples is not None

    # 标签：解析式高斯=S4-0；集合模式优先取 --variant，其次取 ckpt 快照的 s4_variant
    variant = ns.variant
    if not variant:
        if not ensemble:
            variant = 'S4-0'
        else:
            variant = 'S4-prob'
            if ns.ckpt:
                from experiments.evaluate_variant import load_train_args
                key = str(getattr(load_train_args(ns.ckpt), 's4_variant', '') or '')
                variant = {'s4_1': 'S4-1', 's4_2': 'S4-2'}.get(key, 'S4-prob')
    prefix = ns.prefix or (variant.lower().replace('-', '_')
                           if ensemble else 's4_0')

    train_npz = Path(ns.train_npz) if ns.train_npz else out_dir / 'train_reference_predictions.npz'
    if not train_npz.exists():
        if not ns.ckpt:
            raise SystemExit(
                f'train 参照不存在且未提供 --ckpt：{train_npz}。请提供 --train_npz '
                '或用 --ckpt 现场推断。')
        print(f'[{variant}] 生成 train 参照：{train_npz}')
        run_train_reference(ns.ckpt, train_npz, device=ns.device,
                            anchors_per_subject=ns.train_anchors_per_subject,
                            seed=ns.seed)
    train = load_npz(train_npz)
    lengths = [np.asarray(test[k]).shape[2]
               for k in ('ode_pred', 'dynamics_std', 'future')]
    if ensemble:
        lengths.append(samples.shape[3])
    h = min(lengths)
    var_roi_h, var_global, counts = build_constant_variance(
        train['ode_pred'], train['future'], train['future_mask'], h,
        sigma_floor=ns.sigma_floor)

    report = evaluate_arrays(
        test, var_roi_h, var_global, n_bins=ns.n_bins, sigma_floor=ns.sigma_floor,
        samples=(samples[:, :, :, :h] if ensemble else None), iid_seed=ns.iid_seed)
    report['protocol'] = {
        'variant': variant,
        'description': ('S2 Graph-ODE mean + trained diagonal VarianceHead (no training)'
                        if not ensemble else
                        f'{variant}：{int(samples.shape[1])} 条采样轨迹的经验分布指标'),
        'pred_npz': str(ns.pred_npz),
        'train_reference_npz': str(train_npz),
        'mean_key': 'ode_pred (aux_info.dynamics_pred, data scale)',
        'std_key': ('dynamics_std (data scale)' if not ensemble
                    else 'dynamics_samples [N,K,F,H] 样本集合（data scale）；'
                         'nll 对集合无定义，报告为 NaN'),
        'horizon': int(h),
        'aggregation': 'per-task ROI/horizon mean → per-subject mean → group stats',
        'quantile_strata': 'true |y_{t+h,f} − x_{t,f}| quintiles (post-hoc only)',
        'constant_reference': 'train-subject model residual variance; '
                              'diagonal, input-independent Gaussian',
        'train_reference_shape': list(np.asarray(train['ode_pred']).shape),
        'train_valid_elements_per_roi_h_mean': float(counts.mean()),
    }
    if ensemble:
        report['protocol']['model_iid_control'] = (
            '逐 (ROI,horizon) 独立打乱样本维：边际分布不变、跨 ROI/跨 horizon '
            '依赖被摧毁，用于区分"相关结构收益"与"单纯放大方差"')
        report['protocol']['iid_seed'] = int(ns.iid_seed)
    report['_var_roi_h'] = var_roi_h
    report['_var_global'] = var_global
    report['_horizon_labels'] = [f'H{k}' for k in range(1, h + 1)]
    files = write_csvs(report, test, np.asarray([str(s) for s in test['subj_ids']]),
                       out_dir, DEFAULT_LEVELS, metrics=report['_metrics'],
                       prefix=prefix)
    try:
        files.update(make_plots(report, out_dir, variant=variant))
    except Exception as exc:                       # 图失败不阻断指标落盘
        files['plot_error'] = f'{type(exc).__name__}: {exc}'
    report['files'] = files
    metrics_path = out_dir / f'metrics_{prefix}.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(_jsonable(report), f, ensure_ascii=False, indent=2)
    print(f'[{variant}] 指标已写入 {metrics_path}')
    s = report['subject_level']
    d = report['paired_model_minus_reference']['constant_roi_h']
    if not ensemble:
        print(f"[{variant}] subjects={report['n_subjects']} "
              f"tasks={report['n_tasks']} H={h} | "
              f"CRPS model={s['model']['crps']['mean']:.4f} "
              f"const={s['constant_roi_h']['crps']['mean']:.4f} "
              f"(Δ={d['crps']['mean_diff']:+.4f}, p={d['crps']['p']:.3g}) | "
              f"cov80 model={s['model']['coverage80']['mean']:.4f} "
              f"const={s['constant_roi_h']['coverage80']['mean']:.4f}")
    if not ensemble:
        print(f"[S4-0] NLL model={s['model']['nll']['mean']:.4f} "
              f"const={s['constant_roi_h']['nll']['mean']:.4f} "
              f"(Δ={d['nll']['mean_diff']:+.4f}, p={d['nll']['p']:.3g})")
    else:
        ens = report['ensemble']
        dk = report['paired_model_minus_reference']['constant_gaussian_K']
        di = report['paired_model_minus_reference']['model_iid']
        ceil = ens['coverage_ceiling']
        print(f"[{variant}] 同 K={ens['n_samples']} 可比口径（解析常数行是 K→∞ 口径，"
              f"不与集合行直接比较）")
        print(f"[{variant}] CRPS model={s['model']['crps']['mean']:.4f} "
              f"(fair={s['model']['crps_fair']['mean']:.4f}) "
              f"const_at_same_K={s['constant_gaussian_K']['crps']['mean']:.4f} "
              f"(Δ={dk['crps']['mean_diff']:+.4f}, p={dk['crps']['p']:.3g}) | "
              f"vs iid Δ={di['crps']['mean_diff']:+.4f}")
        print(f"[{variant}] cov80={s['model']['coverage80']['mean']:.4f} "
              f"(ceiling={ceil['80']['empirical_coverage']:.4f}, "
              f"ratio={ens['model']['coverage_over_ceiling']['80']:.3f}) "
              f"cov90={s['model']['coverage90']['mean']:.4f} "
              f"(ceiling={ceil['90']['empirical_coverage']:.4f}, "
              f"ratio={ens['model']['coverage_over_ceiling']['90']:.3f}) | "
              f"PIT std ratio="
              f"{ens['model']['pit']['pit_std'] / ens['model']['pit']['pit_std_ideal']:.3f}")
        print(f"[{variant}] EnergyScore model={ens['model']['energy_score']['mean']:.4f} "
              f"iid={ens['model_iid']['energy_score']['mean']:.4f} "
              f"const_gauss={ens['constant_gaussian_K']['energy_score']['mean']:.4f} | "
              f"Variogram model={ens['model']['variogram']['mean']:.4f} "
              f"iid={ens['model_iid']['variogram']['mean']:.4f} "
              f"const_gauss={ens['constant_gaussian_K']['variogram']['mean']:.4f} | "
              f"vs iid ES Δ="
              f"{ens['paired_model_minus_iid']['energy_score']['mean_diff']:+.4f} "
              f"(p={ens['paired_model_minus_iid']['energy_score']['p']:.3g})")


if __name__ == '__main__':
    main()
