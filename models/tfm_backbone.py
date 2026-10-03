# coding=utf-8
"""NeuroTwin-TFM 主干模块（借鉴 TimesFM-3 的多变量时序设计，方案 §4–§7 / §30 V1）。

核心思想（方案 §3）：TimesFM 的 Variate ⟺ 脑 ROI，Temporal Attention ⟺ ROI 内时间
动力学，Variate Attention ⟺ ROI 间脑网络交互。本文件实现四个构件：

- :class:`TemporalPatchEmbed`      [B,F,K] → [B,F,P,D]：时间轴 patch 化 + ROI / 时间
  位置嵌入（§4.2，patch 长度 p∈{1,4,8} 消融，不使用 TimesFM 的 p=32——K≤64 下
  temporal token 数会退化到 2，见方案 §4.3 / §26.2）；
- :class:`CausalTemporalAttention` 逐 ROI 沿 patch 轴的 causal multi-head attention
  （§5：当前 patch 只能访问 ≤ 当前 patch，与 forecasting 语义一致）；
- :class:`SCGuidedROIAttention`    沿 ROI 轴的 attention，加性结构先验偏置
  ``β·log(ε + A_eff_ij)``（§6.2：用 SC 偏置 attention，而不是用 SC 禁止 attention）；
- :class:`TFMEncoderBlock`         RMSNorm → Causal Temporal Attention → Residual →
  RMSNorm → SC-guided ROI Attention → Residual → RMSNorm → FFN → Residual（§6.4），
  堆叠 N=4~6 层。

SC 注入约定（§7）：``A_eff`` 由模型级 SoftAnatomicalPrior **一次计算、全层共享**，
层内只有轻量可学习偏置标量 β，不再每层重建完整 SC pipeline。

变长 context 约定：同一 batch 内 context 长度 K 一致（按 K 分桶）；K 不是 p 的整数
倍时在**序列开头**补零（归一化空间中 0 ≈ 均值，属中性填充），``P = ceil(K/p)``；
位置嵌入按前缀切片（第 i 个 patch 恒对应嵌入表的第 i 个位置）。
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-Mean-Square 归一化（方案 §6.4 的 Block 使用 RMSNorm 而非 LayerNorm）。

    ``y = x / sqrt(mean(x²) + eps) * weight``（无 bias）。输入最后一维宽度可小于
    ``weight`` 的建模宽度（变长场景不需要——RMSNorm 按最后一维统计，与宽度无关，
    但 affine 参数按宽度建模时仍走前缀切片以保持一致性）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = int(x.shape[-1])
        if d > self.dim:
            raise ValueError(
                f"输入宽度 {d} 超过 RMSNorm 建模宽度 {self.dim}")
        w = self.weight[:d] if d != self.dim else self.weight
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * w


class TemporalPatchEmbed(nn.Module):
    """时间 patch 化嵌入（§4.2）：[B,F,K] → [B,F,P,D]。

    每个连续 p 个 TR 的 patch 经共享线性层映射为一个 token（token 表示局部时间
    模式而非单点观测，缩短 temporal 序列、降低 attention 计算量）；随后加上
    逐 ROI 嵌入与逐 patch 时间位置嵌入。

    变长 K：``P = ceil(K/p)``，开头补零对齐到 ``P*p``（归一化空间 0 ≈ 均值）。
    """

    def __init__(self, features: int, context_max: int, patch_len: int, dim: int,
                 dropout: float = 0.1):
        super().__init__()
        if patch_len < 1:
            raise ValueError(f"patch_len 必须 >= 1，收到 {patch_len}")
        if context_max < patch_len:
            raise ValueError(
                f"context_max({context_max}) 必须 >= patch_len({patch_len})")
        self.features = int(features)
        self.context_max = int(context_max)
        self.patch_len = int(patch_len)
        self.dim = int(dim)
        self.patch_proj = nn.Linear(self.patch_len, self.dim)
        self.roi_embed = nn.Parameter(torch.zeros(self.features, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(self._max_patches(), self.dim))
        self.norm = RMSNorm(self.dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.roi_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _max_patches(self) -> int:
        return math.ceil(self.context_max / self.patch_len)

    def num_patches(self, context_len: int) -> int:
        """给定 context 长度 K 的 patch 数（含开头补零对齐）。"""
        if not 1 <= context_len <= self.context_max:
            raise ValueError(
                f"context_len={context_len} 超出 [1, {self.context_max}]")
        return math.ceil(context_len / self.patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F, K] 归一化后的 context（K 同 batch 内一致）
        Returns:
            [B, F, P, D] patch token 网格（§2.2 的二维 token grid）
        """
        if x.ndim != 3:
            raise ValueError(f"TemporalPatchEmbed 期望 [B,F,K]，收到 {tuple(x.shape)}")
        b, f, k = x.shape
        if f != self.features:
            raise ValueError(f"ROI 数 {f} 与建模的 {self.features} 不一致")
        n_patch = self.num_patches(k)
        if k % self.patch_len != 0:
            pad = n_patch * self.patch_len - k
            x = F.pad(x, (pad, 0))                    # 开头补零（归一化空间 ≈ 均值）
        # [B,F,P,p] → 共享线性 → RMSNorm → 加 ROI / 时间位置嵌入 → [B,F,P,D]
        # （先归一化 patch 投影、嵌入作加性残差：第 i 个 patch 的输出恒含
        #   pos_embed[i]，变长 K 的位置语义因此逐位可对齐）
        tokens = x.reshape(b, f, n_patch, self.patch_len)
        h = self.norm(self.patch_proj(tokens))
        h = h + self.roi_embed.view(1, f, 1, self.dim)
        h = h + self.pos_embed[:n_patch].view(1, 1, n_patch, self.dim)
        return self.dropout(h)


class CausalTemporalAttention(nn.Module):
    """逐 ROI 的 causal temporal attention（§5）。

    对每个 ROI 独立执行沿 patch 轴的多头注意力，严格 causal mask
    （``p_k ← p_{≤k}``），建模“单个 ROI 自身如何随时间演化”。实现为
    [B,F,P,D] → [B*F,P,D] 的 batch 合并 + ``scaled_dot_product_attention``。
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim({dim}) 必须能被 num_heads({num_heads}) 整除")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = float(dropout)
        self._causal_mask: Optional[torch.Tensor] = None
        self._mask_len = 0

    def _causal(self, n: int, device, dtype) -> torch.Tensor:
        """缓存 [n,n] 布尔 causal mask（True = 允许注意）。"""
        if self._mask_len < n or self._causal_mask is None \
                or self._causal_mask.device != device:
            m = torch.ones(n, n, dtype=torch.bool, device=device).tril(0)
            self._causal_mask = m
            self._mask_len = n
        return self._causal_mask[:n, :n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F, P, D]
        Returns:
            [B, F, P, D]（token 已聚合其因果历史上下文）
        """
        b, f, p, d = x.shape
        qkv = self.qkv(x).reshape(b * f, p, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # 各 [B*F, h, P, d]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self._causal(p, x.device, x.dtype),
            dropout_p=(self.dropout if self.training else 0.0))
        out = out.transpose(1, 2).reshape(b * f, p, self.dim)
        out = self.out_proj(out)
        return out.view(b, f, p, d)


class SCGuidedROIAttention(nn.Module):
    """SC-guided ROI（variate）attention（§6）。

    对每个 patch 位置独立执行沿 ROI 轴的多头注意力，并在 attention 打分上加性
    注入结构先验（§6.2）::

        S^brain_ij = q_i·k_j / √d_head + β · log(ε + A_eff_ij)

    ``A_eff`` 由模型级先验一次计算、全层共享（§7）；β 为逐头可学习标量
    （初始 1.0）。A_eff 取值 [0,1] 量级、双随机（行和 1），log 偏置把高结构
    连接的 ROI 对的打分系统性抬高，但不禁止低连接边（DTI 假阴性可在数据
    支持下被 attention“复活”）。
    """

    LOG_EPS = 1e-6

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim({dim}) 必须能被 num_heads({num_heads}) 整除")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.out_proj = nn.Linear(self.dim, self.dim)
        # 逐头结构偏置标量 β（初始 1.0，方案 §6.2 的显式系数）
        self.beta = nn.Parameter(torch.ones(self.num_heads))
        self.attn_drop = float(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, F, P, D]
            adj: [B, F, F] 结构先验 A_eff（模型级共享；要求非负）
        Returns:
            [B, F, P, D]
        """
        if adj is None:
            raise ValueError(
                "SCGuidedROIAttention 需要结构先验 A_eff；如需关闭 SC 先验，"
                "请通过先验模块的 adaptive_only 模式提供纯功能图，而非传 None")
        b, f, p, d = x.shape
        if tuple(adj.shape) != (b, f, f):
            raise ValueError(
                f"A_eff 形状 {tuple(adj.shape)} 与输入 {(b, f, f)} 不一致")
        # [B,F,P,D] → [B*P,F,D]：先把 patch 维换到 batch 位再折叠，
        # 直接 reshape 会把 (B,F) 与 P 交错，令 ROI 轴混合串扰不同 patch
        qkv = self.qkv(x).permute(0, 2, 1, 3).reshape(
            b * p, f, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # 各 [B*P, h, F, d]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # [B*P,h,F,F] ← [B,1,1,F,F] 广播加性结构偏置（逐头独立 β）
        bias = torch.log((adj + self.LOG_EPS).clamp_min(0.0))
        scores = scores.view(b, p, self.num_heads, f, f) \
            + bias.view(b, 1, 1, f, f) \
            * self.beta.view(1, 1, self.num_heads, 1, 1)
        attn = F.softmax(scores.flatten(0, 1), dim=-1)   # [(B*P), h, F, F]
        if self.attn_drop > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop)
        out = torch.matmul(attn, v)                      # [B*P, h, F, d]
        out = out.transpose(1, 2).reshape(b * p, f, self.dim)
        out = self.out_proj(out)                         # [B*P, F, D]
        return out.view(b, p, f, self.dim).permute(0, 2, 1, 3)


class TFMEncoderBlock(nn.Module):
    """SC-guided Temporal–Variate Block（方案 §6.4）::

        h ← h + Dropout(CausalTemporalAttention(RMSNorm(h)))
        h ← h + Dropout(SCGuidedROIAttention(RMSNorm(h), A_eff))
        h ← h + Dropout(FFN(RMSNorm(h)))

    FFN 为 ``dim → ff_dim → dim`` 的 GELU 两层 MLP（ff_dim 默认 4×dim）。
    """

    def __init__(self, dim: int, num_heads: int = 4, ff_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_temporal = RMSNorm(dim)
        self.attn_temporal = CausalTemporalAttention(dim, num_heads, dropout)
        self.norm_roi = RMSNorm(dim)
        self.attn_roi = SCGuidedROIAttention(dim, num_heads, dropout)
        self.norm_ffn = RMSNorm(dim)
        ff_dim = int(ff_ratio) * dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [B,F,P,D]，adj: [B,F,F] A_eff → [B,F,P,D]。"""
        h = x + self.dropout(self.attn_temporal(self.norm_temporal(x)))
        h = h + self.dropout(self.attn_roi(self.norm_roi(h), adj))
        h = h + self.dropout(self.ffn(self.norm_ffn(h)))
        return h


def build_context_grid(x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 [B,F,W,S] 的内部布局转成 TFM 的 [B,F,K] 网格（W=1 时）。

    与 NeuroTwin 旧主干的 ``[B,F,W,S]`` 兼容：next_timepoint 协议下整段 context
    是单个窗口（W=1），此处仅做去窗口维；W≠1 时显式报错（TFM 不做窗口轴混
    合，方案 §4.1 删除 window-axis 结构）。
    """
    if x_norm.ndim != 4:
        raise ValueError(f"期望 [B,F,W,S]，收到 {tuple(x_norm.shape)}")
    if x_norm.shape[2] != 1:
        raise ValueError(
            f"TFM 仅支持单窗口输入（W=1），收到 W={x_norm.shape[2]}；"
            "window-axis 结构已按方案 §4.1 删除。")
    return x_norm[:, :, 0, :], x_norm
