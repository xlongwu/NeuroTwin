# coding=utf-8
"""预测头解码组件：未来查询解码器 + 幅值一致性损失。

背景（见 docs/Version1_docs_0921/NeuroTwin_当前问题与修复.md 1.5 / 1.7a）：
现有 `NeuroTwinForecastHead` 把融合特征直接 `Linear → (pred_window × pred_seq_len)`
展开成扁平向量，**未来位置（哪个窗口、哪个时间步）的语义在输出层之前就丢失了**，
只能靠最后一层线性权重隐式区分。`FutureQueryDecoder` 改为显式的
“未来槽位查询 × 融合记忆” 交叉注意力：

    memory  = 现有 6 路融合分支输出（不再新增编码器）
    query   = ROI 嵌入(+ 未来窗口嵌入)(+ 未来时间步嵌入)
    out     = cross_attn(query, memory) → 残差投影 → [B, F, pred_window, pred_seq_len]

输出投影**零初始化**，因此初始时刻 query 分支贡献为 0，与原有 `shape_head`
的“零初始化基线”对齐，便于把该结构改为可消融的加法项。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureQueryDecoder(nn.Module):
    """未来查询解码器：显式建模“哪个窗口 / 哪个时间步”的位置语义。

    Args:
        features:    ROI 数 F
        pred_window: 预测窗口数 W'
        pred_seq_len: 每个预测窗口的时间点数 S'
        memory_dim:  记忆（融合特征）维度
        query_dim:   查询/注意力内部维度
        num_layers:  交叉注意力层数
        num_heads:   注意力头数
        mode:        roi（每 ROI 一个查询）/
                     roi_window（每 ROI×窗口 一个查询）/
                     roi_time（每 ROI×窗口×时间步 一个查询，3480 查询，opt-in）
        dropout:     dropout 率

    每个查询发射的标量数按模式自动推导，保证总输出恒为
    `F × pred_window × pred_seq_len`。
    """

    MODES = ('roi', 'roi_window', 'roi_time')

    def __init__(
        self,
        features: int,
        pred_window: int,
        pred_seq_len: int,
        memory_dim: int,
        query_dim: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        mode: str = 'roi',
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"FutureQueryDecoder mode 仅支持 {self.MODES}，收到 '{mode}'")
        if query_dim % num_heads != 0:
            raise ValueError(
                f"future_query_dim ({query_dim}) 必须能被 num_heads ({num_heads}) 整除")

        self.features = int(features)
        self.pred_window = int(pred_window)
        self.pred_seq_len = int(pred_seq_len)
        self.mode = mode
        self.query_dim = int(query_dim)
        self.n_emit = self._emit_size(mode, self.pred_window, self.pred_seq_len)

        self.mem_proj = nn.Linear(memory_dim, self.query_dim)
        self.roi_emb = nn.Parameter(torch.randn(1, self.features, self.query_dim) * 0.02)
        self.win_emb = None
        self.time_emb = None
        if mode in ('roi_window', 'roi_time'):
            self.win_emb = nn.Parameter(
                torch.randn(1, 1, self.pred_window, self.query_dim) * 0.02)
        if mode == 'roi_time':
            self.time_emb = nn.Parameter(
                torch.randn(1, 1, 1, self.pred_seq_len, self.query_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.query_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True)
            for _ in range(max(1, int(num_layers)))
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.query_dim) for _ in range(max(1, int(num_layers)))
        ])
        self.drop = nn.Dropout(dropout)
        # 输出投影零初始化 → 初始时刻该分支贡献为 0（相对既有 shape_head 的基线）
        self.out_proj = nn.Linear(self.query_dim, self.n_emit)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _emit_size(mode: str, pred_window: int, pred_seq_len: int) -> int:
        if mode == 'roi':
            return pred_window * pred_seq_len
        if mode == 'roi_window':
            return pred_seq_len
        return 1

    def _build_queries(self, b: int, memory: torch.Tensor) -> torch.Tensor:
        """构造查询张量 [B, Q, query_dim] 与通道索引。

        返回 (queries, shape_info)：
          - 'roi':        [B, F, d]
          - 'roi_window': [B, F*W', d]
          - 'roi_time':   [B, F*W'*S', d]
        """
        roi = self.roi_emb.expand(b, -1, -1)
        if self.mode == 'roi':
            return roi
        q = roi.unsqueeze(2) + self.win_emb            # [B, F, W', d]
        if self.mode == 'roi_window':
            return q.reshape(b, self.features * self.pred_window, self.query_dim)
        q = q.unsqueeze(3) + self.time_emb             # [B, F, W', S', d]
        return q.reshape(
            b, self.features * self.pred_window * self.pred_seq_len, self.query_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: [B, F, memory_dim] 融合特征（每个 ROI 一个记忆 token）
        Returns:
            [B, F, pred_window, pred_seq_len] 的加法形状贡献（初始为 0）
        """
        if memory.ndim != 3:
            raise ValueError(f"FutureQueryDecoder 期望 memory [B,F,D]，收到 {tuple(memory.shape)}")
        b, f, _ = memory.shape
        if f != self.features:
            raise ValueError(f"memory 的 ROI 数 {f} 与 features {self.features} 不一致")

        mem = self.mem_proj(memory)                    # [B, F, d]
        q = self._build_queries(b, mem)                # [B, Q, d]
        for attn, norm in zip(self.layers, self.norms):
            attended, _ = attn(q, mem, mem, need_weights=False)
            q = norm(q + self.drop(attended))

        out = self.out_proj(q)                         # [B, Q, n_emit]
        # 还原成 [B, F, W', S']
        if self.mode == 'roi':
            return out.view(b, self.features, self.pred_window, self.pred_seq_len)
        if self.mode == 'roi_window':
            return out.view(b, self.features, self.pred_window, self.pred_seq_len)
        return out.view(b, self.features, self.pred_window, self.pred_seq_len)


def standardize_future(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """按 (窗口, 时间步) 维做零均值单位方差标准化（与预测头共享同一口径）。"""
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


class AmplitudeConsistencyLoss(nn.Module):
    """幅值一致性损失：约束“显式幅值参数”与“预测实际幅值”一致。

    动机（1.7a）：`pred = anchor + trend + scale×shape` 中，`trend` 与
    `scale×shape` 同时携带幅值自由度，二者可以互相抵消/放大，导致
    `scale` 头学不到真实幅值而只做数值补偿。重参数化为
    `pred = anchor + softplus(scale)·(z(trend) + z(shape))` 后，幅值只由
    `softplus(scale)` 决定，因此可以直接监督它与预测的**实际幅值**
    （去掉 anchor 后沿时间轴的逐 (ROI, 窗口) 标准差）一致。

    未启用 `scale_mod_trend` 或 aux 中缺少幅值统计时返回 0，不影响其余训练。
    """

    def __init__(self, granularity: str = 'window', mode: str = 'abs', eps: float = 1e-5):
        super().__init__()
        if mode not in ('abs', 'log'):
            raise ValueError(f"AmplitudeConsistencyLoss mode 仅支持 abs/log，收到 '{mode}'")
        if granularity not in ('window', 'timestep'):
            raise ValueError(
                f"AmplitudeConsistencyLoss granularity 仅支持 window/timestep，收到 '{granularity}'")
        self.granularity = granularity
        self.mode = mode
        self.eps = float(eps)

    def _diff(self, a: torch.Tensor, b: torch.Tensor,
              weight: torch.Tensor = None) -> torch.Tensor:
        if self.mode == 'log':
            d = (torch.log(a.clamp_min(self.eps))
                 - torch.log(b.clamp_min(self.eps))).abs()
        else:
            d = (a - b).abs()
        if weight is None:
            return d.mean()
        w = weight.expand_as(d)
        return (d * w).sum() / w.sum().clamp_min(self.eps)

    def forward(self, target: torch.Tensor, aux_info,
                mask: torch.Tensor = None) -> torch.Tensor:
        """target 与 aux 中的 anchor 必须处于同一数值空间（由 NeuroTwin 保证）。

        - ``window``：逐 (ROI, 窗口) 比较 ``softplus(scale)`` 与目标残差的
          时间轴标准差 ``std_S(y - anchor)``；
        - ``timestep``：逐元素比较 ``softplus(scale)`` 与 ``|y - anchor|``
          （元素级幅值代理量）。

        mask: 可选的窗口级掩码 [B, W]（1 有效 / 0 填充）。`--variable_cutoff`
              下填充窗的目标是占位零，其幅值无意义，需按掩码加权排除。
        """
        stats = aux_info.get('head_amp', None) if isinstance(aux_info, dict) else None
        if not isinstance(stats, dict):
            return None
        scale = stats.get('scale', None)      # [B, F, W', 1] 或 [B, F, W', S']
        anchor = stats.get('anchor', None)    # [B, F, W', S']
        if not torch.is_tensor(scale) or not torch.is_tensor(anchor):
            return None
        if anchor.shape != target.shape:
            return None
        resid = (target - anchor).detach()
        weight = None
        if mask is not None:
            weight = mask.to(resid.dtype).reshape(mask.shape[0], 1, mask.shape[1])
        if self.granularity == 'window':
            tgt_amp = resid.std(dim=-1, unbiased=False)                 # [B,F,W']
            s = scale.squeeze(-1) if scale.shape[-1] == 1 else scale.mean(-1)
            if s.shape != tgt_amp.shape:
                return None
            return self._diff(s, tgt_amp, weight)
        tgt_amp = resid.abs()
        s = scale if scale.shape == tgt_amp.shape else scale.expand_as(tgt_amp)
        return self._diff(s, tgt_amp, weight.unsqueeze(-1) if weight is not None else None)
