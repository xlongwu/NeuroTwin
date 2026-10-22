# coding=utf-8
"""S4 概率轨迹的集合型损失/指标。

- :func:`ensemble_crps`：训练主损失。使用排序后的闭式经验 CRPS，避免
  ``K×K`` 两两差分矩阵的显存开销；
- :func:`ensemble_energy_score` 与 :func:`temporal_variogram_score`：验证指标，
  分别检查联合分布距离与时间依赖（第一轮不加入训练）。

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
                             eps: float = 1e-8) -> torch.Tensor:
    """时间变差函数得分（H−1 个 lag 的平均平方差，越小越好）。

        γ_x(τ) = mean_t (x_{t+τ} − x_t)²
        score  = mean_{τ,f} (E_K[γ_sample(τ)] − γ_target(τ))²

    只在 ``t`` 与 ``t+τ`` 都有效的元素上统计；样本维先取均值。该指标对
    “独立噪声放大方差但时间相关结构错误”敏感，作为 S4 的验证指标而非训练项。
    """
    mask = _check_inputs(samples, target, mask)
    b, k, f, h = samples.shape
    if h < 2:
        raise ValueError(f'变差函数需要 H>=2，收到 H={h}')
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
