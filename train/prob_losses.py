# coding=utf-8
"""S4 概率轨迹的集合型损失/指标。

- :func:`ensemble_crps`：训练主损失。使用排序后的闭式经验 CRPS，避免
  ``K×K`` 两两差分矩阵的显存开销；
- :func:`ensemble_energy_score` 与 :func:`temporal_variogram_score`：验证指标，
  分别检查联合分布距离与时间依赖；指定 lags 时提供 P1 的可微训练目标。

样本形状统一为 ``[B,K,F,H]``（K 个轨迹样本），真值 ``[B,F,H]``，有效 mask
``[B,H]``；ROI/horizon 维度按有效元素数归一，batch 内不因缺测改变权重。
"""
from __future__ import annotations

from typing import Optional

import torch


def _check_inputs(samples: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor]):
    if samples.ndim != 4:
        raise ValueError(f'samples 期望 [B,K,F,H]，收到 {tuple(samples.shape)}')
    b, k, f, h = samples.shape
    if target.shape != (b, f, h):
        raise ValueError(
            f'target 期望 {(b, f, h)}，收到 {tuple(target.shape)}')
    if k < 1:
        raise ValueError(f'K 必须 >=1，收到 {k}')
    if mask is None:
        mask = torch.ones(b, h, device=samples.device, dtype=samples.dtype)
    else:
        mask = mask.to(device=samples.device, dtype=samples.dtype)
        if mask.shape != (b, h):
            raise ValueError(f'mask 期望 {(b, h)}，收到 {tuple(mask.shape)}')
    return mask


def ensemble_crps(samples: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  eps: float = 1e-8) -> torch.Tensor:
    """排序闭式经验 CRPS（标量，masked 元素均值）。

    对每个 ``(b,f,h)``，把 K 个样本排序为 ``x_(1)≤…≤x_(K)``：

        CRPS = 1/K Σ|x_(i)−y| − 1/K² Σ (2i−K−1)·x_(i)   （i 从 1 开始）

    展开的离散度项等价于 ``1/(2K²)Σ_iΣ_j|x_i−x_j|``，但只需 O(K log K)。
    """
    mask = _check_inputs(samples, target, mask)
    s, _ = torch.sort(samples, dim=1)                          # [B,K,F,H]
    term1 = (s - target.unsqueeze(1)).abs().mean(dim=1)        # [B,F,H]
    k = s.shape[1]
    coeff = (2.0 * torch.arange(k, device=s.device, dtype=s.dtype)
             - float(k) + 1.0)                                 # [K]
    term2 = (s * coeff.view(1, k, 1, 1)).sum(dim=1) / float(k * k)
    crps = term1 - term2
    w = mask.unsqueeze(1)                                      # [B,1,H]
    denom = (w.sum() * samples.shape[2]).clamp_min(eps)
    return (crps * w).sum() / denom


def _masked_rms_norm(diff: torch.Tensor, mask4: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """[B,K,F,H] 差值 → 逐样本 masked RMS（沿 F,H 的 L2/√N，跨 K 广播）。"""
    n = mask4.sum(dim=(2, 3)).clamp_min(eps)                   # [B,K] 有效元素数
    sumsq = (diff * diff * mask4).sum(dim=(2, 3))
    return torch.sqrt(sumsq / n)


def ensemble_energy_score(samples: torch.Tensor, target: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          eps: float = 1e-8) -> torch.Tensor:
    """集合 Energy Score：``E||X−y|| − 0.5·E||X−X'||``（masked RMS 距离）。"""
    mask = _check_inputs(samples, target, mask)
    b, k, f, h = samples.shape
    mask4 = mask.view(b, 1, 1, h).expand(b, k, f, h)
    tgt = target.unsqueeze(1).expand_as(samples)
    first = _masked_rms_norm(samples - tgt, mask4, eps=eps).mean(dim=1)  # [B]
    pair = samples.new_zeros(b)
    for i in range(k):
        diff = samples[:, i:i + 1, :, :] - samples                   # [B,K,F,H]
        pair = pair + _masked_rms_norm(diff, mask4, eps=eps).sum(dim=1)
    pair = pair / float(k * k)
    score = first - 0.5 * pair
    return score.mean()


def temporal_variogram_score(samples: torch.Tensor, target: torch.Tensor,
                             mask: Optional[torch.Tensor] = None,
                             eps: float = 1e-8, *, lags=None, lag_weights=None,
                             power: float = 0.5) -> torch.Tensor:
    """时间变差函数得分（H−1 个 lag 的平均平方差，越小越好）。

        γ_x(τ) = mean_t (x_{t+τ} − x_t)²
        score  = mean_{τ,f} (E_K[γ_sample(τ)] − γ_target(τ))²

    lags=None 保留历史指标；指定 lags=(1,2,4) 时使用逐时间点/ROI 的
    E|delta|**power 与真实 |delta|**power 的平方差，用于 P1 训练。

    只在 ``t`` 与 ``t+τ`` 都有效的元素上统计；样本维先取均值。该指标对
    “独立噪声放大方差但时间相关结构错误”敏感，作为 S4 的验证指标而非训练项。
    """
    mask = _check_inputs(samples, target, mask)
    b, k, f, h = samples.shape
    if h < 2:
        raise ValueError(f'变差函数需要 H>=2，收到 H={h}')
    if lags is not None:
        # P1: square each time/ROI discrepancy before averaging; the historical
        # score below instead squares an already pooled second moment.
        lags = tuple(lags)
        if not lags:
            raise ValueError('temporal variogram requires at least one lag')
        weights = tuple(lag_weights) if lag_weights is not None else (1 / len(lags),) * len(lags)
        import math
        if (not lags or len(lags) != len(weights) or len(set(lags)) != len(lags)
                or any(t not in (1, 2, 4) for t in lags)
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-7)
                or not math.isfinite(power) or power <= 0):
            raise ValueError('invalid temporal variogram lags/weights/power')
        def moment(delta):
            # Fractional powers have an infinite derivative at zero. Evaluate
            # a safe inactive branch and use the zero subgradient there.
            a = delta.float().abs()
            safe = torch.where(a > 0, a, torch.ones_like(a))
            return torch.where(a > 0, safe.pow(power), torch.zeros_like(a))
        total = samples[..., :0].sum()
        for tau, weight in zip(lags, weights):
            if tau >= h:
                continue
            valid = (mask[:, tau:] * mask[:, :-tau])[:, None, :]
            pm = moment(samples[..., tau:] - samples[..., :-tau]).mean(1)
            tm = moment(target[..., tau:] - target[..., :-tau])
            total = total + weight * ((pm - tm).square() * valid).sum() / (valid.sum() * f).clamp_min(eps)
        return total
    pred_gamma = []
    true_gamma = []
    valid_lag = []
    for tau in range(1, h):
        v = (mask[:, tau:] * mask[:, :-tau])                        # [B,H-tau]
        dp = samples[:, :, :, tau:] - samples[:, :, :, :-tau]       # [B,K,F,H-tau]
        dt = target[:, :, tau:] - target[:, :, :-tau]               # [B,F,H-tau]
        # 样本维先取均值，再在有效 (t,ROI) 元素上平均
        pg_elem = (dp * dp).mean(dim=1)                             # [B,F,L]
        n_valid = (v.sum(dim=1) * f).clamp_min(eps)                 # [B]
        pg = (pg_elem * v[:, None, :]).sum(dim=(1, 2)) / n_valid    # [B]
        tg = ((dt * dt) * v[:, None, :]).sum(dim=(1, 2)) / n_valid
        pred_gamma.append(pg)
        true_gamma.append(tg)
        valid_lag.append(v.sum(dim=1) > 0)
    weights = torch.stack(valid_lag, dim=1).to(samples.dtype)       # [B,H-1]
    pg = torch.stack(pred_gamma, dim=1)                             # [B,H-1]
    tg = torch.stack(true_gamma, dim=1)
    denom = weights.sum(dim=1).clamp_min(eps)
    return ((((pg - tg) ** 2) * weights).sum(dim=1) / denom).mean()
