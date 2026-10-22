# coding=utf-8
"""S4 概率轨迹模块：输出端低秩残差头（S4-1）与潜状态低秩创新（S4-2）。

设计约束（S4 方案）：

- 不使用逐 ROI / 逐 horizon 独立噪声；随机性只在 rank ``r`` 维创新空间中生成，
  再通过共享空间因子映射到 ROI；
- S4-1 在**输出端**给确定性 Graph-ODE 均值加相关残差，时间维用 AR(1) 递推，
  不改变 S2 的均值预测路径；
- S4-2 在 latent 状态转移上加入 ``sqrt(dt)·Gφ(Z,M)ε``，保留原 BOLD 积分读出；
- 两者默认关闭；构造时用 fork_rng 隔离随机流，关闭时 S2 参数逐位不变。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_common(rank: int, samples: int) -> None:
    if int(rank) < 1:
        raise ValueError(f'rank 必须 >= 1，收到 {rank}')
    if int(samples) < 1:
        raise ValueError(f'n_samples 必须 >= 1，收到 {samples}')


class LowRankSpatioTemporalResidualHead(nn.Module):
    """S4-1：输出端低秩时空残差头 ``Rφ(X, ε)`` → [B,K,F,H]。

    对每个样本 k 与 horizon h，在 rank 空间递推

        a_{k,h} = ρ·a_{k,h−1} + (1−ρ)·b(X) + s(X)⊙ε_{k,h}
        r_{k,f,h} = Σ_j S_{f,j}·a_{k,h,j}

    其中 ``S [F,r]`` 是共享空间因子，``ε`` 只在 rank 维独立采样。因此跨 ROI
    相关、跨 horizon 通过 AR(1) 相关，且秩不超过 ``r``。``b``/``s`` 由历史
    latent 汇总向量 ``X`` 经小 MLP 得到；``s`` 末层零初始化（小起步）。
    """

    is_s4_module = True

    def __init__(self, dim: int, n_rois: int, horizon: int,
                 rank: int = 8, hidden: int = 128, n_samples: int = 32,
                 scale_init: float = 0.1, rho_init: float = 0.9):
        super().__init__()
        _validate_common(rank, n_samples)
        if int(dim) < 1 or int(n_rois) < 1 or int(horizon) < 1:
            raise ValueError(
                f'dim/n_rois/horizon 必须 >=1：{dim}/{n_rois}/{horizon}')
        if not 0.0 < float(rho_init) < 1.0:
            raise ValueError(f'rho_init 必须在 (0,1)，收到 {rho_init}')
        self.dim = int(dim)
        self.n_rois = int(n_rois)
        self.horizon = int(horizon)
        self.rank = int(rank)
        self.n_samples = int(n_samples)
        self.scale_init = float(scale_init)

        self.norm = nn.LayerNorm(self.dim)
        self.hidden = nn.Sequential(
            nn.Linear(self.dim, int(hidden)), nn.GELU())
        nn.init.xavier_uniform_(self.hidden[0].weight)
        nn.init.zeros_(self.hidden[0].bias)
        self.to_base = nn.Linear(int(hidden), self.rank)
        self.to_scale = nn.Linear(int(hidden), self.rank)
        nn.init.zeros_(self.to_base.weight)
        nn.init.zeros_(self.to_base.bias)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        # 单位 RMS 空间因子 + 小尺度 (0.1)：初始残差 std 约 0.1–0.2，
        # 既不是零方差（会让 CRPS 尺度梯度消失），也不破坏 S2 均值。
        self.spatial = nn.Parameter(torch.empty(self.n_rois, self.rank))
        nn.init.normal_(self.spatial, std=1.0 / float(self.rank) ** 0.5)
        self.rho_raw = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(float(rho_init))))))

    def forward(self, context: torch.Tensor,
                n_samples: Optional[int] = None) -> torch.Tensor:
        """context ``[B,D]``（或 ``[B,F,D]``，按 ROI 平均）→ 残差 ``[B,K,F,H]``。"""
        if context.ndim == 3:
            context = context.mean(dim=1)
        if context.ndim != 2 or context.shape[-1] != self.dim:
            raise ValueError(
                f'context 期望 [B,{self.dim}] 或 [B,F,{self.dim}]，收到 {tuple(context.shape)}')
        k = int(n_samples) if n_samples is not None else self.n_samples
        _validate_common(self.rank, k)
        b = context.shape[0]
        h = self.horizon
        hidden = self.hidden(self.norm(context))
        base = self.to_base(hidden)                                  # [B,r]
        scale = F.softplus(self.to_scale(hidden)) * self.scale_init  # [B,r]
        rho = torch.sigmoid(self.rho_raw)
        eps = torch.randn(b, k, h, self.rank,
                          device=context.device, dtype=hidden.dtype)
        coeff = context.new_zeros(b, k, self.rank, dtype=hidden.dtype)
        coeffs = []
        for hi in range(h):
            coeff = (rho * coeff
                     + (1.0 - rho) * base[:, None, :]
                     + scale[:, None, :] * eps[:, :, hi, :])
            coeffs.append(coeff)
        coeffs = torch.stack(coeffs, dim=2)                           # [B,K,H,r]
        residual = torch.einsum('fr,bkhr->bkfh', self.spatial, coeffs)
        if not torch.isfinite(residual).all():
            raise FloatingPointError('S4-1 残差出现非有限值')
        return residual


class LowRankLatentInnovation(nn.Module):
    """S4-2：潜状态低秩创新 ``Gφ(Z,M)ε`` → [B,F]。

    每个 batch 样本仅在 rank 维采样 ``ε [B,r]``（同一 ε 作用于所有 ROI），
    ROI 相关增益由状态 ``Z`` 与历史记忆 ``M`` 经 LayerNorm+Linear 得到：

        G_{b,f,j} = softplus(W_g [LN(Z_b,f)+M_b,f]) · scale
        ξ_{b,f}   = Σ_j G_{b,f,j}·ε_{b,j}

    末层零初始化（初始增益约 ``0.693·scale``，默认 scale=1e-2 的小创新），
    不推进全局 RNG。
    """

    is_s4_module = True

    def __init__(self, dim: int, rank: int = 8, scale_init: float = 1e-2):
        super().__init__()
        _validate_common(rank, 1)
        if int(dim) < 1:
            raise ValueError(f'dim 必须 >=1，收到 {dim}')
        if float(scale_init) <= 0:
            raise ValueError(f'scale_init 必须 >0，收到 {scale_init}')
        self.dim = int(dim)
        self.rank = int(rank)
        self.scale_init = float(scale_init)
        self.norm = nn.LayerNorm(self.dim)
        self.to_gain = nn.Linear(self.dim, self.rank)
        nn.init.zeros_(self.to_gain.weight)
        nn.init.zeros_(self.to_gain.bias)

    def forward(self, z: torch.Tensor, memory: Optional[torch.Tensor] = None,
                eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z ``[B,F,D]``；memory ``[B,D]`` / ``[B,F,D]`` / None → noise ``[B,F]``。"""
        if z.ndim != 3 or z.shape[-1] != self.dim:
            raise ValueError(f'z 期望 [B,F,{self.dim}]，收到 {tuple(z.shape)}')
        hidden = self.norm(z)
        if memory is not None:
            mem = memory if memory.ndim == 3 else memory.unsqueeze(1)
            if mem.shape[-1] != self.dim or mem.shape[0] != z.shape[0]:
                raise ValueError(
                    f'memory 期望 [B,{self.dim}] 或 [B,F,{self.dim}]，'
                    f'收到 {tuple(memory.shape)}')
            hidden = hidden + mem.to(hidden.dtype)
        gain = F.softplus(self.to_gain(hidden)) * self.scale_init  # [B,F,r]
        if eps is None:
            eps = torch.randn(z.shape[0], self.rank,
                              device=z.device, dtype=hidden.dtype)
        if eps.shape != (z.shape[0], self.rank):
            raise ValueError(
                f'eps 期望 {(z.shape[0], self.rank)}，收到 {tuple(eps.shape)}')
        noise = torch.einsum('bfr,br->bf', gain, eps.to(hidden.dtype))
        if not torch.isfinite(noise).all():
            raise FloatingPointError('S4-2 创新项出现非有限值')
        return noise
