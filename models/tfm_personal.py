# coding=utf-8
"""个人基线 profile 与 PersonalFiLM（方案 §5 / §6.1.2）。

ProfileEncoder 从同一扫描最早 ``calibration_length``（默认 32）TR 的
``calibration_bold`` 出发，经**共享** TFM 编码器得到基线 tokens 后，沿 ROI 和
patch 双均值池化，经两层 MLP + tanh 得到固定维（默认 32）的个体校准码 p。

- 基线编码不注入当前临床 FiLM / AdaRMSNorm（cond_mods=None 路径），临床影响
  留给向量场的 c，避免把同一份当前评分再编码进 profile；
- 不创建 ID embedding 或逐人训练参数；
- 训练时前向保留梯度，不跨 optimizer step 缓存 detached tokens；
- ``personalized=False`` 时不创建本模块（由 ``models/tfm.py`` 控制）。

跨访视稳定性是待验证性质，p 是当前基线的编码表示，不是终生稳定特征。

PersonalFiLM 把 profile p 映射为逐通道 (scale, shift)，零初始化，风格与
``models/tfm.py::ConditionFiLM`` 一致；输出沿 ROI 维广播进入向量场隐层。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProfileEncoder(nn.Module):
    """基线 tokens ``[B, F, P0, D]`` → 个体校准码 ``[B, profile_dim]``。

    池化：:math:`b = \\frac{1}{F P_0} \\sum_{i,p} T^{\\mathrm{cal}}_{ip}`（ROI 与
    patch 双均值）。MLP：``LayerNorm → Linear(D, hidden) → GELU →
    Linear(hidden, profile_dim) → tanh``。两层正常随机初始化（``nn.Linear``
    默认），不清零——profile 不是条件注入 adapter，不需要恒等起步。

    Args:
        dim: 共享 TFM 编码器的 latent 宽度 D。
        profile_dim: 输出个体码宽度（默认 32）。
        hidden: MLP 隐藏宽度（默认 64）。
    """

    def __init__(self, dim: int, profile_dim: int = 32, hidden: int = 64):
        super().__init__()
        if dim < 1 or profile_dim < 1 or hidden < 1:
            raise ValueError(
                f'ProfileEncoder 宽度必须为正：dim={dim} profile_dim={profile_dim} '
                f'hidden={hidden}')
        self.dim = int(dim)
        self.profile_dim = int(profile_dim)
        self.hidden = int(hidden)
        self.norm = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, self.hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden, self.profile_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens ``[B, F, P0, D]`` → p ``[B, profile_dim]``。

        基线 tokens 由调用方经共享 TFM 编码器得到（本模块不持有编码器）；
        梯度路径由调用方控制，本模块不做 detach。
        """
        if tokens.ndim != 4:
            raise ValueError(
                f'ProfileEncoder 期望基线 tokens [B,F,P0,D]，收到 {tuple(tokens.shape)}')
        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f'基线 tokens 通道宽 {tokens.shape[-1]} 与登记的 D={self.dim} 不一致')
        b = tokens.mean(dim=(1, 2))                            # [B,D]
        return torch.tanh(self.fc2(self.act(self.fc1(self.norm(b)))))


class PersonalFiLM(nn.Module):
    """个体 profile 的 FiLM 调制：``h' = h·(1+γ_p(p)) + β_p(p)``。

    与 :class:`models.tfm.ConditionFiLM` 同风格：零初始化 → 初始不改基准动力学
    （方案 §6.1.4：FiLM 零初始化使个体码初始不改变基准向量场）。标记
    ``is_conditioning_adapter=True``，冻结主干阶段随条件模块一起训练。
    """

    is_conditioning_adapter = True

    def __init__(self, profile_dim: int, dim: int):
        super().__init__()
        if profile_dim <= 0:
            raise ValueError(f'profile_dim 必须 > 0，收到 {profile_dim}')
        self.profile_dim = int(profile_dim)
        self.dim = int(dim)
        self.proj = nn.Linear(self.profile_dim, 2 * self.dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, profile: torch.Tensor):
        """p ``[B, profile_dim]`` → (scale [B,D], shift [B,D])。"""
        if profile.ndim != 2 or profile.shape[1] != self.profile_dim:
            raise ValueError(
                f'profile 期望 [B, {self.profile_dim}]，收到 {tuple(profile.shape)}')
        scale, shift = self.proj(profile).chunk(2, dim=-1)
        return scale, shift
