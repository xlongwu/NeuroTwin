from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    BrainRevIN, DFCAdapter, BrainMDM, GraphODEDDI,
    IterativePredictionRefiner, prefix_layernorm, prefix_linear, prefix_proj,
)
from models.decoder import FutureQueryDecoder
from models.graph_prior import SoftAnatomicalPrior
from models.pathology import PathologyNormalizer, AuxInversionHead
from models.neurotwin_moe import NeuroTwinMoE


# -------------------------------------------------
# Shared helpers for the base forecasting head
# -------------------------------------------------

def standardize_future(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std  = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def make_zero_last_linear(linear: nn.Linear, bias: float = 0.0) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.constant_(linear.bias, bias)


# ════════════════════════════════════════════════════════════════════════
#  WindowTemporalAttention
# ════════════════════════════════════════════════════════════════════════

class WindowTemporalAttention(nn.Module):
    """
    Causal cross-window attention: explicitly model temporal evolution
    dependencies across W windows before flattening history in NeuroTwinForecastHead.

    Design motivation
    ────────
    PCC measures the morphological correlation between predicted and true
    time series, which is extremely sensitive to phase and rhythm.
    The W windows in history are sliding samples of dFC along the time axis,
    whose evolution trajectory (which ROI functional connections are strengthening
    and which are weakening) contains the most critical prior for the next window.
    Direct flattening can only implicitly learn this structure through linear
    projection; causal attention allows each window's features to explicitly
    aggregate its historical context.

    Input/Output shape: [B, F, W, D] (shape-preserving)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_windows: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"WindowTemporalAttention: hidden_dim ({hidden_dim}) "
                f"must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim  = hidden_dim
        self.num_windows = num_windows
        self.num_heads   = num_heads
        self.dropout_rate = dropout

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_windows, hidden_dim) * 0.02)

        self.gate = nn.Parameter(torch.zeros(1))

        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(num_windows, num_windows), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F, W, D]
        Returns:
            [B, F, W, D], each window has aggregated causal historical context
        """
        b, f, w, d = x.shape

        x_seq = x.reshape(b * f, w, d)
        x_seq = x_seq + self.pos_embed[:, :w, :]

        mask = self.causal_mask[:w, :w] if w <= self.num_windows else \
               torch.triu(torch.ones(w, w, device=x.device), diagonal=1).bool()
        attended, _ = self.mha(x_seq, x_seq, x_seq, attn_mask=mask)

        gate = torch.sigmoid(self.gate)
        x_seq = self.norm(x_seq + gate * self.dropout(attended))

        return x_seq.reshape(b, f, w, d)


# ════════════════════════════════════════════════════════════════════════
#  NeuroTwinForecastHead
# ════════════════════════════════════════════════════════════════════════

class NeuroTwinForecastHead(nn.Module):
    """
    Correlation-aware base forecaster for NeuroTwin.

    分支构成（均可用 --head_use_* 单独消融，fusion 输入宽度随活跃分支数自适应）：
      1. history_proj          扁平化历史特征
      2. latent_proj           扁平化潜在特征
      3. temporal              时间轴 depthwise 卷积
      4. cross_roi_hist/latent ROI 轴混合（--head_cross_roi 控制形态）
      5. win_attn              因果跨窗注意力
      6. revin_stats           实例归一化的 mean/std（已 detach，来自 RevIN）

    形状分支有两种形态（--head_shape_mode）：
      - ``flatten``：``Linear → [pred_window × pred_seq_len]`` 展平（旧行为）；
      - ``query``（默认）：``FutureQueryDecoder`` 显式以 ROI/窗口/时间步查询
        对融合记忆做交叉注意力，保留未来位置语义；输出投影零初始化。

    幅值有两种重参数化（--head_amp_mode）：
      - ``legacy``：``pred = anchor + trend_raw + softplus(scale)·z(shape)``；
      - ``scale_mod_trend``（默认）：
        ``pred = anchor + softplus(scale)·(z(trend_raw) + z(shape_raw))``，
        幅值只由 ``softplus(scale)`` 承担，配合 ``AmplitudeConsistencyLoss``
        监督其等于目标残差实际幅值，消除 trend 与 scale×shape 的双重自由。

    最后接 3 轮迭代 SC 约束细化（IterativePredictionRefiner）。
    """

    def __init__(
        self,
        features: int,
        in_window: int,
        in_seq_len: int,
        pred_window: int,
        pred_seq_len: int,
        dropout: float = 0.2,
        refiner_rounds: int = 3,
        refiner_adaptive: bool = False,
        head_cross_roi: str = 'conv1',
        head_shape_mode: str = 'query',
        future_query_mode: str = 'roi',
        future_query_dim: int = 32,
        future_query_layers: int = 1,
        future_query_heads: int = 4,
        head_amp_mode: str = 'scale_mod_trend',
        head_scale_granularity: str = 'window',
        head_anchor_mode: str = 'history_window',
        head_use_history_proj: bool = True,
        head_use_latent_proj: bool = True,
        head_use_temporal: bool = True,
        head_use_cross_roi: bool = True,
        head_use_win_attn: bool = True,
        head_use_revin_stats: bool = True,
        pred_head: str = 'gaussian',
        pred_quantiles=(0.1, 0.5, 0.9),
        pred_logvar_init: float = 0.0,
    ):
        super().__init__()
        if head_cross_roi not in ('conv1', 'skip', 'sc_prior'):
            raise ValueError(
                f"head_cross_roi 仅支持 conv1/skip/sc_prior，收到 '{head_cross_roi}'")
        if pred_head not in ('point', 'gaussian', 'quantile'):
            raise ValueError(
                f"pred_head 仅支持 point/gaussian/quantile，收到 '{pred_head}'")
        self.pred_head = pred_head
        self.pred_quantiles = tuple(float(q) for q in pred_quantiles)
        if pred_head == 'quantile':
            if len(self.pred_quantiles) < 2:
                raise ValueError("quantile 模式至少需要 2 个分位点")
            if any(not 0.0 < q < 1.0 for q in self.pred_quantiles) or \
                    list(self.pred_quantiles) != sorted(self.pred_quantiles):
                raise ValueError(
                    f"pred_quantiles 必须为 (0,1) 内严格递增序列，收到 {self.pred_quantiles}")
        # 供 NeuroTwin._decode 读取的概率输出（归一化空间，随后统一反归一化）
        self.last_pred_logvar = None
        self.last_pred_quantiles = None
        if head_shape_mode not in ('flatten', 'query'):
            raise ValueError(f"head_shape_mode 仅支持 flatten/query，收到 '{head_shape_mode}'")
        if head_amp_mode not in ('legacy', 'scale_mod_trend'):
            raise ValueError(
                f"head_amp_mode 仅支持 legacy/scale_mod_trend，收到 '{head_amp_mode}'")
        if head_scale_granularity not in ('window', 'timestep'):
            raise ValueError(
                "head_scale_granularity 仅支持 window/timestep，"
                f"收到 '{head_scale_granularity}'")
        if head_anchor_mode not in ('history_window', 'last_timestep', 'zero'):
            raise ValueError(
                "head_anchor_mode 仅支持 history_window/last_timestep/zero，"
                f"收到 '{head_anchor_mode}'")
        if head_anchor_mode == 'last_timestep' and pred_seq_len != 1:
            raise ValueError(
                "head_anchor_mode='last_timestep'（next_timepoint 的 x_t 锚点）要求 "
                f"pred_seq_len == 1，收到 pred_seq_len={pred_seq_len}")

        self.head_shape_mode = head_shape_mode
        self.head_amp_mode = head_amp_mode
        self.head_scale_granularity = head_scale_granularity
        self.head_anchor_mode = head_anchor_mode
        # 供 NeuroTwin 判断是否走「next-state / delta」语义（next_timepoint 任务）
        self.is_next_state_head = (head_anchor_mode != 'history_window')
        # head_use_cross_roi=False 等价于 head_cross_roi='skip'
        self.head_cross_roi = 'skip' if not head_use_cross_roi else head_cross_roi
        self.use_history_proj = bool(head_use_history_proj)
        self.use_latent_proj  = bool(head_use_latent_proj)
        self.use_temporal     = bool(head_use_temporal)
        self.use_win_attn     = bool(head_use_win_attn)
        self.use_cross_roi    = bool(head_use_cross_roi)
        self.use_revin_stats  = bool(head_use_revin_stats)
        # 全部消融会导致 fusion 输入为空：强制保留历史投影分支作为兜底
        if not any([self.use_history_proj, self.use_latent_proj, self.use_temporal,
                    self.use_cross_roi, self.use_win_attn, self.use_revin_stats]):
            self.use_history_proj = True
        # 供 NeuroTwin._decode 读取的最近一次幅值统计（AmplitudeConsistencyLoss 使用）
        self.last_amp_stats = None
        # 供 NeuroTwin 计算 next-state delta 的最近一次锚点（detach，归一化空间）
        self.last_anchor = None

        self.features     = features
        self.in_window    = in_window
        self.pred_window  = pred_window
        self.pred_seq_len = pred_seq_len
        self.in_dim  = in_window * in_seq_len
        self.out_dim = pred_window * pred_seq_len
        hidden_dim   = min(768, max(256, self.in_dim * 2))
        self.hidden_dim = hidden_dim

        self.history_norm   = nn.LayerNorm(self.in_dim)
        self.latent_norm    = nn.LayerNorm(self.in_dim)
        # cross-ROI 分支消费的是投影到 hidden_dim 的历史/潜在特征，因此即使
        # 对应的 *_proj 分支被消融，也仍需要保留投影（只是不再作为独立 fusion 分支）
        cross_active = self.head_cross_roi != 'skip'
        need_hist = self.use_history_proj or cross_active
        need_lat  = self.use_latent_proj or cross_active
        self.history_proj = (nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            if need_hist else None)
        self.latent_proj = (nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            if need_lat else None)
        if self.use_temporal:
            self.temporal_branch = nn.Sequential(
                nn.Conv1d(features, features, kernel_size=5, padding=2, groups=features),
                nn.GELU(), nn.Dropout(dropout))
            self.temporal_proj  = nn.Linear(self.in_dim, hidden_dim)
        else:
            self.temporal_branch = None
            self.temporal_proj = None

        if self.head_cross_roi == 'skip':
            self.cross_roi_hist = None
            self.cross_roi_latent = None
        elif self.head_cross_roi == 'sc_prior':
            # 逐 ROI（depthwise）增益，不含自由 ROI×ROI 混合矩阵
            self.cross_roi_hist = nn.Sequential(
                nn.Conv1d(features, features, kernel_size=1, groups=features),
                nn.GELU(), nn.Dropout(dropout))
            self.cross_roi_latent = nn.Sequential(
                nn.Conv1d(features, features, kernel_size=1, groups=features),
                nn.GELU(), nn.Dropout(dropout))
        else:
            self.cross_roi_hist = nn.Sequential(
                nn.Conv1d(features, features, kernel_size=1), nn.GELU(), nn.Dropout(dropout))
            self.cross_roi_latent = nn.Sequential(
                nn.Conv1d(features, features, kernel_size=1), nn.GELU(), nn.Dropout(dropout))

        if self.use_win_attn:
            win_hidden_dim = self._choose_win_hidden(in_seq_len, hidden_dim, num_heads=4)
            self.win_proj = nn.Sequential(
                nn.LayerNorm(in_seq_len),
                nn.Linear(in_seq_len, win_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.window_temporal_attn = WindowTemporalAttention(
                hidden_dim=win_hidden_dim,
                num_heads=4,
                num_windows=in_window,
                dropout=dropout,
            )
            self.win_flatten_proj = nn.Sequential(
                nn.Linear(in_window * win_hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.win_proj = None
            self.window_temporal_attn = None
            self.win_flatten_proj = None

        # RevIN 统计分支：mean/std 逐 ROI，已 detach；线性投影到 hidden_dim
        self.revin_proj = (nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            if self.use_revin_stats else None)

        n_fuse = self._count_fuse_branches()
        self.n_fuse = n_fuse
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * n_fuse, hidden_dim), nn.GELU(), nn.Dropout(dropout))

        self.trend_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim))

        if self.head_shape_mode == 'flatten':
            self.shape_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.out_dim))
            self.future_query = None
        else:
            self.shape_head = None
            self.future_query = FutureQueryDecoder(
                features=features, pred_window=pred_window, pred_seq_len=pred_seq_len,
                memory_dim=hidden_dim, query_dim=future_query_dim,
                num_layers=future_query_layers, num_heads=future_query_heads,
                mode=future_query_mode, dropout=dropout)

        scale_out = (pred_window if head_scale_granularity == 'window' else self.out_dim)
        self.scale_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, scale_out))

        # refiner_rounds=0 支持 no-refiner 消融对照：不构建 refiner，forward 直接返回 pred
        self.refiner = (
            IterativePredictionRefiner(
                features=features, n_rounds=refiner_rounds, dropout=dropout,
                adaptive=refiner_adaptive)
            if refiner_rounds >= 1 else None)

        # 概率输出头（Phase 7.1）：point 不构建任何额外模块，参数量与旧实现一致
        if pred_head == 'gaussian':
            self.logvar_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.out_dim))
            # 初始方差取 σ_norm=1（归一化空间残差的典型量级，实测 log(err²)≈1.9）。
            # 若取 σ=0.1（logvar=-4.6）会让初始 NLL 放大约两个数量级并主导早期损失。
            nn.init.zeros_(self.logvar_head[-1].weight)
            nn.init.constant_(self.logvar_head[-1].bias, float(pred_logvar_init))
            self.quantile_head = None
        elif pred_head == 'quantile':
            self.logvar_head = None
            self.quantile_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.pred_quantiles) * self.out_dim))
            # 零初始化：初始时所有分位点都等于点预测，随后被 pinball 损失拉开
            make_zero_last_linear(self.quantile_head[-1], bias=0.0)
        else:
            self.logvar_head = None
            self.quantile_head = None

        if self.shape_head is not None:
            make_zero_last_linear(self.shape_head[-1], bias=0.0)
        make_zero_last_linear(self.scale_head[-1], bias=-2.0)

    def _count_fuse_branches(self) -> int:
        n = int(self.use_history_proj) + int(self.use_latent_proj) + int(self.use_temporal)
        n += 2 if self.cross_roi_hist is not None else 0
        n += int(self.use_win_attn) + int(self.use_revin_stats)
        return max(1, n)

    @staticmethod
    def _choose_win_hidden(in_seq_len: int, hidden_dim: int, num_heads: int = 4) -> int:
        upper = max(num_heads * 4, hidden_dim // 2)
        candidate = max(num_heads * 4, (in_seq_len * 2 // num_heads) * num_heads)
        return min(candidate, upper)

    @staticmethod
    def _sc_mix(feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """用 SC 软先验做 ROI 混合：feat [B,F,D]，adj [B,F,F] → A_eff @ feat。

        adj 为 None（未启用软先验）时退化为恒等，保证 --sc_prior_mode scaled 数值不变。
        """
        if adj is None:
            return feat
        return torch.bmm(adj.to(dtype=feat.dtype), feat)

    def _build_anchor(self, history: torch.Tensor) -> torch.Tensor:
        """构造预测头的加法锚点（归一化空间），形状 [B, F, pred_window, pred_seq_len]。

        - ``history_window``（旧任务）：最后一窗 + 0.5×一阶趋势外推（`pred_window` 个窗）；
        - ``last_timestep``（next_timepoint / delta 预测）：context 最后一个 TR x_t
          广播到全部预测偏移——模型只需学习 Δx̂ = x̂ − x_t，避免直接回归 x_{t+1}
          时退化成「复制 x_t」；
        - ``zero``（next_timepoint / absolute 预测）：零锚点，模型直接预测
          （归一化空间中的）下一状态，用于 delta vs absolute 的公平消融。
        """
        b, f, w, s = history.shape
        if self.head_anchor_mode == 'zero':
            return torch.zeros(b, f, self.pred_window, self.pred_seq_len,
                               device=history.device, dtype=history.dtype)
        if self.head_anchor_mode == 'last_timestep':
            # context 的最后一个 TR（最近时间点）作为 x_t；pred_seq_len 已校验为 1
            last_tr = history[:, :, -1:, -1:]                    # [B, F, 1, 1]
            return last_tr.expand(b, f, self.pred_window, self.pred_seq_len)
        last  = history[:, :, -1:, :]
        slope = last - history[:, :, -2:-1, :] if w >= 2 else torch.zeros_like(last)
        anchors, current = [], last
        for _ in range(self.pred_window):
            current = current + 0.5 * slope
            anchors.append(current)
        return (torch.cat(anchors, dim=2) if anchors
                else torch.zeros(b, f, 0, s, device=history.device, dtype=history.dtype))

    def _win_proj_forward(self, history: torch.Tensor) -> torch.Tensor:
        """win_proj 的变长 S 轴版本（LayerNorm/Linear 按权重前缀切片）。

        ``win_proj = Sequential(LayerNorm(in_seq_len), Linear(in_seq_len, win_hidden), ...)``
        作用在 ``[B,F,W,S]`` 的最后一维（S）上；next_timepoint 的 context 长度
        K <= context_max 时走前缀切片，S == in_seq_len 时与直接调用逐位等价。
        """
        x = prefix_layernorm(self.win_proj[0], history)
        x = prefix_linear(self.win_proj[1], x)
        for layer in list(self.win_proj)[2:]:
            x = layer(x)
        return x

    def _maybe_standardize(self, x: torch.Tensor) -> torch.Tensor:
        """幅值重参数化的标准化；``out_dim == 1`` 时退化为恒等。

        ``standardize_future`` 对 (pred_window, pred_seq_len) 求统计量；单元素
        （next_timepoint 单步预测）时方差恒为 0，标准化会把输出整体置零，
        使 ``pred = anchor``（模型失去全部学习信号），因此显式跳过。
        """
        if self.out_dim <= 1:
            return x
        return standardize_future(x)

    def _revin_feature(self, revin_stats, b, f, device, dtype):
        if self.revin_proj is None or revin_stats is None:
            return None
        mean, stdev = revin_stats
        mean = mean.to(device=device, dtype=dtype).reshape(b, f, -1)[..., :1]
        stdev = stdev.to(device=device, dtype=dtype).reshape(b, f, -1)[..., :1]
        stats = torch.cat([mean, stdev], dim=-1)          # [B, F, 2]，已 detach
        return self.revin_proj(stats)

    def forward(
        self,
        latent: torch.Tensor,
        history: torch.Tensor,
        sc_matrix: torch.Tensor,
        return_rounds: bool = False,
        adj: torch.Tensor = None,
        revin_stats=None,
    ):
        """
        Args:
            return_rounds: True 时额外返回轮间中间预测（不含最终预测），
                           供轮间监督（deep supervision）使用。
            adj: 可选的 [B, F, F] SC 软先验 A_eff；head_cross_roi='sc_prior'
                 时用于 ROI 混合，并在 sc_refiner_inject 命中时传给 refiner。
            revin_stats: 可选的 (mean, stdev)，供 --head_use_revin_stats 分支使用。
        Returns:
            pred，或 (pred, round_preds)
        """
        if latent.ndim != 4 or history.ndim != 4:
            raise ValueError(
                f"Expected [B,F,W,S], got {tuple(latent.shape)} and {tuple(history.shape)}")
        b, f, w, _ = latent.shape
        dev, dt = latent.device, latent.dtype
        if w > self.in_window:
            raise ValueError(
                f"历史窗口数 {w} 超过预测头建模的 {self.in_window}："
                "请确认 history_max 与数据 chunk 数一致。")

        history_flat = history.reshape(b, f, -1)      # [B, F, L*S]
        latent_flat  = latent.reshape(b, f, -1)

        # 变长历史（L < in_window）走“权重前缀”路径：第 w 个窗口恒对应权重切片
        # [w*S:(w+1)*S]；L == in_window 时与旧实现逐位等价（见 models/common.py）。
        hist_norm = prefix_layernorm(self.history_norm, history_flat)
        lat_norm  = prefix_layernorm(self.latent_norm, latent_flat)
        hist_feat = (prefix_proj(self.history_proj, hist_norm)
                     if self.history_proj is not None else hist_norm)
        lat_feat = (prefix_proj(self.latent_proj, lat_norm)
                    if self.latent_proj is not None else lat_norm)

        fuse_feats = []
        if self.use_history_proj:
            fuse_feats.append(hist_feat)
        if self.use_latent_proj:
            fuse_feats.append(lat_feat)
        if self.temporal_proj is not None:
            fuse_feats.append(prefix_linear(
                self.temporal_proj, self.temporal_branch(history_flat)))
        if self.cross_roi_hist is not None:
            cross_hist = self.cross_roi_hist(hist_feat)
            cross_latent = self.cross_roi_latent(lat_feat)
            if self.head_cross_roi == 'sc_prior':
                cross_hist   = self._sc_mix(cross_hist, adj)
                cross_latent = self._sc_mix(cross_latent, adj)
            fuse_feats.append(cross_hist)
            fuse_feats.append(cross_latent)
        if self.win_flatten_proj is not None:
            win_x    = self._win_proj_forward(history)
            win_ctx  = self.window_temporal_attn(win_x)
            # win_flatten_proj 以 in_window×win_hidden 建模，变长历史走前缀路径
            fuse_feats.append(prefix_proj(
                self.win_flatten_proj, win_ctx.reshape(b, f, -1)))
        revin_feat = self._revin_feature(revin_stats, b, f, dev, dt)
        if revin_feat is not None:
            fuse_feats.append(revin_feat)

        fused = self.fusion(torch.cat(fuse_feats, dim=-1))

        anchor    = self._build_anchor(history)
        trend_raw = self.trend_head(fused).view(b, f, self.pred_window, self.pred_seq_len)
        if self.future_query is not None:
            shape_raw = self.future_query(fused)
        else:
            shape_raw = self.shape_head(fused).view(b, f, self.pred_window, self.pred_seq_len)

        scale_raw = F.softplus(self.scale_head(fused))
        if self.head_scale_granularity == 'window':
            scale = scale_raw.view(b, f, self.pred_window, 1)
        else:
            scale = scale_raw.view(b, f, self.pred_window, self.pred_seq_len)

        if self.head_amp_mode == 'legacy':
            pred = anchor + trend_raw + scale * self._maybe_standardize(shape_raw)
            amp_scale = None
        else:
            # 幅值只由 softplus(scale) 承担；trend 与 shape 都先标准化
            pred = anchor + scale * (self._maybe_standardize(trend_raw)
                                     + self._maybe_standardize(shape_raw))
            amp_scale = scale

        # 供 NeuroTwin 计算 next-state delta（原始空间）与 refiner 张量流使用
        self.last_anchor = anchor.detach()

        if amp_scale is not None:
            self.last_amp_stats = {'scale': amp_scale.detach(), 'anchor': anchor.detach()}
        else:
            self.last_amp_stats = None

        if self.refiner is None:
            out, round_preds = pred, None
        elif return_rounds:
            delta, round_preds = self.refiner.forward_with_rounds(pred, sc_matrix, adj=adj)
            out = pred + delta
        else:
            out = pred + self.refiner(pred, sc_matrix, adj=adj)
            round_preds = None

        # 概率输出（归一化空间；均值项与最终预测对齐，条件化在融合特征上）
        if self.logvar_head is not None:
            self.last_pred_logvar = self.logvar_head(fused).view(
                b, f, self.pred_window, self.pred_seq_len)
        elif self.quantile_head is not None:
            q_off = self.quantile_head(fused).view(
                b, f, len(self.pred_quantiles), self.pred_window, self.pred_seq_len)
            self.last_pred_quantiles = out.unsqueeze(2) + q_off
        if return_rounds:
            return out, round_preds
        return out


class NeuroTwin(nn.Module):
    """
    NeuroTwin: A pathology-conditioned mixture-of-denoising-experts digital twin
    framework for brain functional dynamics (PCC-oriented v2).

    Architecture overview
    ─────────────────────
    Stage 1 (pretrain): Physics backbone for healthy control brain dynamics
        BrainRevIN → DFCAdapter → BrainMDM → GraphODEDDI × N → NeuroTwinForecastHead

    Stage 2 (finetune): Pathology-specific NeuroTwinMoE for MDD residual modeling
        NeuroTwinMoE (RouterCond + pathology-conditioned experts + shared expert)
        with iterative SC-constrained delta refinement

    Loss: Uncertainty-weighted hybrid loss (PCC + MAE + temporal_diff + std)
    """

    def __init__(
        self,
        features: int,
        in_window: int,
        in_seq_len: int,
        pred_window: int,
        pred_seq_len: int,
        n_block: int = 2,
        dropout: float = 0.1,
        pathology_input_dim: int = 1,
        pathology_dim: int = 16,
        adapter_alpha: float = 0.5,
        norm: bool = True,
        pretrain_mode: bool = False,
        ode_steps: int = 5,
        ode_hidden_dim: int = 128,
        num_scales: int = 3,
        num_experts: int = 4,
        top_k: int = 2,
        stochastic_depth_rate: float = 0.1,
        moe_gate_temperature: float = 2.0,
        moe_expert_hidden_dim: int = 256,
        moe_use_shared_expert: bool = True,
        moe_router_cond_only: bool = False,
        moe_use_argmax: bool = False,
        moe_inference_temperature: float = 0.3,
        moe_eval_mode: str = 'dense_soft',
        moe_gate_features: str = 'state_revin',
        moe_gate_input_dim: int = None,
        moe_experts_mode: str = 'routed_shared',
        moe_expert_kind: str = 'homogeneous',
        moe_route_level: str = 'sample',
        moe_eval_mc_samples: int = 0,
        moe_expert_stats_interval: int = 0,
        pathology_norm_mode: str = 'robust_z',
        pathology_norm_quantiles: int = 64,
        pathology_norm_rbf_knots: int = 8,
        pathology_poly_expansion: bool = False,
        refiner_rounds: int = 3,
        delta_refiner_rounds: int = 2,
        refiner_adaptive: bool = False,
        refiner_return_rounds: bool = False,
        patho_cond_layer: str = 'joint',
        patho_adaln_targets: str = 'both',
        lora_enable: bool = True,
        lora_rank: int = 8,
        lora_n_blocks: int = 2,
        sc_prior_mode: str = 'soft_prior',
        sc_lambda_mode: str = 'global',
        sc_lambda_init: float = 0.7,
        sc_prior_rank: int = 12,
        sc_delta_a: bool = True,
        sc_delta_rank: int = None,
        sc_prob_mask: bool = False,
        sc_sinkhorn_iters: int = 64,
        sc_mask_mode: str = 'soft',
        sc_mask_tau: float = 1e-3,
        sc_refiner_inject: str = 'both',
        head_cross_roi: str = 'conv1',
        sc_temporal_weight: float = 0.0,
        mdm_scale_scheme: str = 'divisor',
        mdm_scale_gate: str = 'sample',
        ode_window_attn: str = 'on',
        ode_solver: str = 'rk2',
        ode_step_mode: str = 'learnable',
        ode_step_scale: float = 0.1,
        sde_noise_scale: float = 1e-3,
        ode_adaptive_rtol: float = 1e-3,
        ode_adaptive_atol: float = 1e-4,
        head_shape_mode: str = 'query',
        future_query_mode: str = 'roi',
        future_query_dim: int = 32,
        future_query_layers: int = 1,
        future_query_heads: int = 4,
        head_amp_mode: str = 'scale_mod_trend',
        head_scale_granularity: str = 'window',
        head_anchor_mode: str = 'history_window',
        head_use_history_proj: bool = True,
        head_use_latent_proj: bool = True,
        head_use_temporal: bool = True,
        head_use_cross_roi: bool = True,
        head_use_win_attn: bool = True,
        head_use_revin_stats: bool = True,
        pred_head: str = 'gaussian',
        pred_quantiles=(0.1, 0.5, 0.9),
        inversion_weight: float = 0.0,
        inversion_hidden_dim: int = 128,
    ):
        super().__init__()
        self.features         = features
        self.norm             = norm
        self.pretrain_mode    = pretrain_mode
        self.pred_window      = pred_window
        self.pred_seq_len     = pred_seq_len
        self.refiner_return_rounds = bool(refiner_return_rounds)

        # 病理条件化双层开关：特征级 AdaLN（主干） + 残差级 MoE
        if patho_cond_layer not in ('residual_only', 'feature_only', 'joint'):
            raise ValueError(
                "patho_cond_layer 仅支持 residual_only/feature_only/joint，"
                f"收到 '{patho_cond_layer}'")
        if patho_adaln_targets not in ('none', 'mdm', 'ode', 'both'):
            raise ValueError(
                "patho_adaln_targets 仅支持 none/mdm/ode/both，"
                f"收到 '{patho_adaln_targets}'")
        self.patho_cond_layer = patho_cond_layer
        # residual_only：条件只走 MoE 残差分支，主干不做特征级 AdaLN
        self.patho_adaln_targets = ('none' if patho_cond_layer == 'residual_only'
                                    else patho_adaln_targets)

        # SC 软先验相关开关
        if sc_prior_mode not in ('scaled', 'soft_prior', 'adaptive_only', 'functional_only'):
            raise ValueError(
                "sc_prior_mode 仅支持 scaled/soft_prior/adaptive_only/functional_only，"
                f"收到 '{sc_prior_mode}'")
        if sc_mask_mode not in ('hard', 'soft', 'none'):
            raise ValueError(f"sc_mask_mode 仅支持 hard/soft/none，收到 '{sc_mask_mode}'")
        if sc_refiner_inject not in ('base', 'delta', 'both', 'none'):
            raise ValueError(
                "sc_refiner_inject 仅支持 base/delta/both/none，"
                f"收到 '{sc_refiner_inject}'")
        self.sc_prior_mode = sc_prior_mode
        self.sc_mask_mode = sc_mask_mode
        self.sc_refiner_inject = sc_refiner_inject

        # 病理归一化器（仅微调存在；预训练是纯 HC 物理主干，无条件模块）
        self.pathology_normalizer = None
        cond_dim = 0
        if not pretrain_mode:
            self.pathology_normalizer = PathologyNormalizer(
                input_dim=pathology_input_dim,
                mode=pathology_norm_mode,
                n_quantiles=pathology_norm_quantiles,
                rbf_knots=pathology_norm_rbf_knots,
            )
            cond_dim = self.pathology_normalizer.out_dim

        ada_targets = self.patho_adaln_targets
        ada_cond_dim = cond_dim if (cond_dim > 0 and ada_targets != 'none') else None
        ada_mdm = ada_cond_dim if ada_targets in ('mdm', 'both') else None
        ada_ode = ada_cond_dim if ada_targets in ('ode', 'both') else None

        # LoRA 仅挂在最后 lora_n_blocks 个 ODE block（预训练不创建，微调新建）
        lora_ranks = [0] * int(n_block)
        if (not pretrain_mode) and lora_enable and lora_rank > 0:
            n_lora = min(int(lora_n_blocks), int(n_block))
            for i in range(int(n_block) - n_lora, int(n_block)):
                lora_ranks[i] = int(lora_rank)

        if self.norm:
            self.rev_norm = BrainRevIN(num_features=features)

        # SC 软解剖先验：'scaled' 时不构建（下游全部走旧 SC 路径，保证数值可复现）
        self.sc_prior = None
        if sc_prior_mode != 'scaled':
            self.sc_prior = SoftAnatomicalPrior(
                features=features, seq_len=in_seq_len,
                mode=sc_prior_mode, rank=sc_prior_rank,
                lambda_mode=sc_lambda_mode, lambda_init=sc_lambda_init,
                delta_a=sc_delta_a, delta_rank=sc_delta_rank,
                prob_mask=sc_prob_mask, sinkhorn_iters=sc_sinkhorn_iters,
                return_sequence=(sc_temporal_weight > 0),
            )

        self.input_dropout = nn.Dropout2d(p=min(0.2, dropout * 0.5))
        self.dfc_adapter   = DFCAdapter(num_nodes=features, alpha=adapter_alpha,
                                        sc_prior_mode=sc_prior_mode)
        self.pastmixing    = BrainMDM(
            features=features, num_window=in_window,
            seq_len=in_seq_len, num_scales=num_scales, dropout=dropout,
            ada_cond_dim=ada_mdm, scale_scheme=mdm_scale_scheme,
            scale_gate=mdm_scale_gate)

        self.ode_blocks = nn.ModuleList([
            GraphODEDDI(
                features=features, seq_len=in_seq_len,
                hidden_dim=ode_hidden_dim, ode_steps=ode_steps,
                dropout=dropout, stochastic_depth_rate=stochastic_depth_rate,
                num_heads=4, window_heads=5,
                ada_cond_dim=ada_ode, lora_rank=lora_ranks[i],
                sc_mask_mode=sc_mask_mode, sc_mask_tau=sc_mask_tau,
                use_window_attn=(ode_window_attn == 'on'),
                # Phase 4：积分器/步长可切换；默认 rk2+learnable 与旧实现数值一致
                solver=ode_solver, step_mode=ode_step_mode,
                step_scale=ode_step_scale, sde_noise_scale=sde_noise_scale,
                adaptive_rtol=ode_adaptive_rtol, adaptive_atol=ode_adaptive_atol,
            )
            for i in range(n_block)
        ])
        self.ode_block_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            for _ in range(n_block)
        ])

        self.post_fusion = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=1),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(features * 2, features, kernel_size=1))
        self.feature_norm = nn.GroupNorm(num_groups=1, num_channels=features)

        self.pretrain_head = NeuroTwinForecastHead(
            features=features, in_window=in_window, in_seq_len=in_seq_len,
            pred_window=pred_window, pred_seq_len=pred_seq_len, dropout=dropout,
            refiner_rounds=refiner_rounds, refiner_adaptive=refiner_adaptive,
            head_cross_roi=head_cross_roi,
            head_shape_mode=head_shape_mode,
            future_query_mode=future_query_mode,
            future_query_dim=future_query_dim,
            future_query_layers=future_query_layers,
            future_query_heads=future_query_heads,
            head_amp_mode=head_amp_mode,
            head_scale_granularity=head_scale_granularity,
            head_anchor_mode=head_anchor_mode,
            head_use_history_proj=head_use_history_proj,
            head_use_latent_proj=head_use_latent_proj,
            head_use_temporal=head_use_temporal,
            head_use_cross_roi=head_use_cross_roi,
            head_use_win_attn=head_use_win_attn,
            # norm=False 时没有 RevIN 统计量可喂给预测头，必须同步关闭该分支，
            # 否则 fusion 的拼接宽度与 n_fuse 不一致
            head_use_revin_stats=(head_use_revin_stats and norm),
            # Phase 7.1：概率输出头（point/gaussian/quantile）
            pred_head=pred_head,
            pred_quantiles=pred_quantiles)

        # next-state（next_timepoint 任务）：预测头以 x_t 或 0 为锚点，
        # 前向时额外导出归一化空间的 Δx̂ 与「归一化→原始空间」的逐 ROI 斜率，
        # 供 NextTimepointLoss 的 delta 项与诊断使用；旧任务下恒为 False，
        # 不产生任何额外计算与 aux 字段。
        self.next_state_mode = bool(getattr(self.pretrain_head, 'is_next_state_head', False))

        if self.pretrain_mode:
            self.moe = None
        else:
            # 门控中的 RevIN 统计项依赖 --norm；关闭归一化时这些通道恒为零，
            # 属于纯浪费容量，因此显式降级为 'state' 并提示（而非静默生效）
            gate_features = moe_gate_features
            if not norm and gate_features != 'state':
                print(f"提示: --norm False 时门控特征 '{gate_features}' 无 RevIN 统计可用，"
                      f"已自动降级为 'state'。")
                gate_features = 'state'
            self.moe = NeuroTwinMoE(
                features=features,
                in_w=in_window, in_s=in_seq_len,
                pred_w=pred_window, pred_s=pred_seq_len,
                pathology_input_dim=cond_dim,
                pathology_dim=pathology_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                gate_temperature=moe_gate_temperature,
                expert_hidden_dim=moe_expert_hidden_dim,
                use_shared_expert=moe_use_shared_expert,
                router_context_cond_only=moe_router_cond_only,
                use_argmax=moe_use_argmax,
                inference_temperature=moe_inference_temperature,
                eval_mode=moe_eval_mode,
                pathology_poly_expansion=pathology_poly_expansion,
                delta_refiner_rounds=delta_refiner_rounds,
                refiner_adaptive=refiner_adaptive,
                gate_features=gate_features,
                gate_input_dim=moe_gate_input_dim,
                experts_mode=moe_experts_mode,
                expert_kind=moe_expert_kind,
                route_level=moe_route_level,
                eval_mc_samples=moe_eval_mc_samples,
                expert_stats_interval=moe_expert_stats_interval,
            )

        # Phase 7.2：辅助反演头（pooled latent → 归一化病理条件）。
        # 默认 inversion_weight=0.0 不构建，避免给纯预测任务强加第二目标；
        # 预训练阶段没有病理条件模块，同样不构建。
        self.inversion_weight = float(inversion_weight)
        self.inversion_head = None
        if (not pretrain_mode) and self.inversion_weight > 0.0:
            if cond_dim <= 0:
                raise RuntimeError(
                    "inversion_weight>0 需要病理条件维度 >0：请确认使用微调模式")
            self.inversion_head = AuxInversionHead(
                latent_dim=features, out_dim=cond_dim,
                hidden_dim=inversion_hidden_dim, dropout=dropout)

    def set_moe_router_temperature(self, temperature: float) -> None:
        if self.moe is not None:
            self.moe.set_router_temperature(temperature)

    def set_moe_router_inference_temperature(self, temperature: float) -> None:
        if self.moe is not None:
            self.moe.set_router_inference_temperature(temperature)

    def _compute_sc_prior(self, h: torch.Tensor, sc_matrix: torch.Tensor):
        """计算 SC 软先验（A_eff）。未启用软先验（--sc_prior_mode scaled）时返回 None。

        先验由**归一化后的历史特征**导出（不受 input_dropout 影响，保证同一输入
        得到同一张图），一次计算后同时供 DFCAdapter / GraphODE / refiner 复用。
        """
        if self.sc_prior is None:
            return None
        return self.sc_prior(h, sc_matrix)

    def _encode(self, x_norm: torch.Tensor, sc_matrix: torch.Tensor,
                cond: torch.Tensor = None):
        """返回 (latent, sc_info, ode_diag)；sc_info 为 None 表示未启用 SC 软先验。

        ``ode_diag`` 汇总各 ODE block 的审计量（``||k1||/||k2||/||Δh||`` 与导数
        调用次数），键名形如 ``ode0_calls``，供训练侧写入 TensorBoard。
        """
        if x_norm.ndim != 4:
            raise ValueError(f"Expected [B,F,W,S], got {tuple(x_norm.shape)}")
        x = x_norm
        sc_info = self._compute_sc_prior(x_norm, sc_matrix)
        adj_eff = sc_info['A_eff'] if sc_info is not None else None
        if self.training:
            x = self.input_dropout(x)
        shallow = self.dfc_adapter(x, sc_matrix, adj=adj_eff)
        x = self.pastmixing(shallow, cond)
        ode_diag = {}
        for i, (block_scale, ode_block) in enumerate(
                zip(self.ode_block_scales, self.ode_blocks)):
            evolved = ode_block(x, sc_matrix, cond, adj_eff=adj_eff)
            diag = getattr(ode_block, 'last_ode_diag', None)
            if diag:
                for k, v in diag.items():
                    # 保持原始类型（0 维张量/float）：在此 float() 会在 torch.compile
                    # 下触发图断裂与设备同步，统一由消费端（main.py 记录时）转换。
                    ode_diag[f'ode{i}_{k}'] = v
            scale   = torch.sigmoid(block_scale)
            x = x + scale * (evolved - x)
        x = x + self.post_fusion(torch.cat([shallow, x], dim=1))
        x = self.feature_norm(x)
        return x, sc_info, ode_diag

    def _prepare_pathology_condition(self, pathology_score):
        """把原始临床评分映射为模型内部条件向量。

        预训练（无归一化器）或未提供评分时返回 None，使特征级 AdaLN
        与 MoE 都走无条件路径，与加入条件模块前的数值逐位一致。
        """
        if self.pathology_normalizer is None or pathology_score is None:
            return None
        return self.pathology_normalizer(pathology_score)

    def _decode(
        self,
        latent: torch.Tensor,
        history_norm: torch.Tensor,
        sc_matrix: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        sc_info: dict = None,
        revin_stats=None,
    ):
        adj_eff = sc_info['A_eff'] if sc_info is not None else None
        # 预测端 SC 注入点消融：base=主干 refiner，delta=MoE 残差 refiner
        base_adj    = adj_eff if self.sc_refiner_inject in ('base', 'both') else None
        delta_adj   = adj_eff if self.sc_refiner_inject in ('delta', 'both') else None

        if self.refiner_return_rounds:
            base_pred, round_preds = self.pretrain_head(
                latent, history_norm, sc_matrix, return_rounds=True, adj=base_adj,
                revin_stats=revin_stats)
        else:
            base_pred, round_preds = self.pretrain_head(
                latent, history_norm, sc_matrix, adj=base_adj,
                revin_stats=revin_stats), None

        # aux_info 恒为 dict：pretrain 返回空 dict，下游 _get() 可安全容忍缺失字段
        aux_info = {}
        if sc_info is not None:
            # 供图正则（稀疏/熵/时间一致性）读取；'scaled' 模式下不写入 → 正则为零
            aux_info['sc_prior'] = sc_info
        if round_preds:
            aux_info['refiner_round_preds'] = list(round_preds)
        amp = getattr(self.pretrain_head, 'last_amp_stats', None)
        if amp:
            # 复制一份，避免反归一化就地修改预测头内部状态
            aux_info['head_amp'] = {'scale': amp['scale'], 'anchor': amp['anchor']}

        # Phase 7.1：概率输出（当前均为归一化空间，随后由 _denorm_aux_prob 换算）
        head = self.pretrain_head
        if head.logvar_head is not None and head.last_pred_logvar is not None:
            aux_info['pred_logvar'] = head.last_pred_logvar
        elif head.quantile_head is not None and head.last_pred_quantiles is not None:
            aux_info['pred_quantiles'] = head.last_pred_quantiles
            aux_info['pred_quantile_levels'] = list(head.pred_quantiles)
        # Phase 7.2：反演预测需与病理条件同空间比较，条件向量一并写入 aux
        if self.inversion_head is not None:
            if cond is None:
                raise ValueError("反演头需要病理条件（cond）作为回归目标。")
            aux_info['inversion_pred'] = self.inversion_head(latent)
            aux_info['pathology_cond'] = cond

        if self.pretrain_mode:
            return base_pred, aux_info
        if cond is None:
            raise ValueError("病理条件（cond）必须提供：请检查 pathology_score 或归一化器。")
        if self.patho_cond_layer == 'feature_only':
            # 消融：关闭残差级（MoE）条件，仅保留主干特征级 AdaLN
            return base_pred, aux_info
        if self.moe is None:
            raise RuntimeError("self.moe is None while pretrain_mode=False")
        delta_pred, moe_aux = self.moe(
            latent, history_norm, base_pred, cond, sc_matrix, sc_adj=delta_adj,
            revin_stats=revin_stats)
        if moe_aux:
            aux_info.update(moe_aux)
        return base_pred + delta_pred, aux_info

    def _fill_next_state_delta(self, aux_info, pred_norm, history_norm, norm_stats) -> None:
        """把 next-state delta 写入 aux_info（next_timepoint 任务专用）。

        - ``pred_delta_norm = x̂ − x_t``：归一化空间中的状态变化量（模型的原生输出）；
        - ``revin_factor``：RevIN 反归一化的逐 ROI 斜率 ``stdev / (affine_weight + eps)``，
          用于把原始空间的 Δx（损失端由 target − x_t 得到）换算到归一化空间比较。

        两项都只依赖历史（x_t 取自 context 末位），未来 target 不参与；
        旧任务（next_state_mode=False）下本方法不会被调用。
        """
        last_tr = history_norm[:, :, -1:, -1:]                   # [B,F,1,1] = x_t
        aux_info['pred_delta_norm'] = pred_norm - last_tr        # [B,F,H,1]
        if self.norm and norm_stats is not None:
            _, stdev = norm_stats
            if self.rev_norm.affine:
                weight = self.rev_norm.affine_weight.view(1, -1, 1, 1)
                factor = stdev / (weight + self.rev_norm.eps)
            else:
                factor = stdev
            aux_info['revin_factor'] = factor.detach()

    @torch.no_grad()
    def rollout_forecast(self, history: torch.Tensor, sc_matrix: torch.Tensor,
                         pathology_score: Optional[torch.Tensor] = None,
                         horizon: int = None) -> torch.Tensor:
        """free rollout（自回归滚动预测，仅用于评估，不改变训练主任务）。

        每一步只取预测头的**第 0 个预测窗**作为“下一 chunk”，把它接到历史末位
        （滑出最早 1 个 chunk，历史长度 L 保持不变）后再前向；滚动 ``horizon`` 步
        得到自回归轨迹 ``[B, F, horizon, S]``。与 direct multi-horizon 预测对比可
        量化 error accumulation。

        注意：模型以 ``pred_window = H_max`` 建模，这里每次前向只用其第一个窗口；
        历史长度为 1 时无法滑动，直接抛出明确错误。
        """
        if history.ndim != 4:
            raise ValueError(f"rollout_forecast 期望 history [B,F,L,S]，收到 {tuple(history.shape)}")
        if history.shape[2] < 2:
            raise ValueError("rollout_forecast 至少需要 2 个历史 chunk 才能滑窗。")
        n_steps = int(horizon) if horizon else int(self.pred_window)
        preds: List[torch.Tensor] = []
        cur = history
        for k in range(n_steps):
            pred, _ = self.forward(cur, sc_matrix, pathology_score)
            nxt = pred[:, :, :1, :]
            preds.append(nxt)
            if k + 1 >= n_steps:
                break
            cur = torch.cat([cur[:, :, 1:, :], nxt], dim=2)
        return torch.cat(preds, dim=2)

    # ------------------------------------------------------------------
    # next_timepoint（Next Brain-State Prediction）对外标准接口
    # ------------------------------------------------------------------
    @staticmethod
    def _to_internal_history(bold_history: torch.Tensor) -> torch.Tensor:
        """标准输入 [B, K, F] → 主干内部布局 [B, F, W=1, S=K]。

        整段 context 作为**单个窗口**（W=1）、K 个 TR 作为该窗口的时间轴（S=K）：
        这样 BrainMDM 的时间卷积 / GraphODE 的时间轴算子原样生效，
        且不需要在多个模块间反复交换维度（变长 K 由 S 轴前缀切片支持）。
        """
        if bold_history.ndim != 3:
            raise ValueError(
                f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}")
        return bold_history.transpose(1, 2).unsqueeze(2).contiguous()

    def predict_next_state(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                           pathology_score: Optional[torch.Tensor] = None) -> dict:
        """下一步全脑状态预测（next_timepoint 任务的标准接口）。

        Args:
            bold_history: [B, K, F] 连续 BOLD 历史（K = context 长度，含最近 TR x_t）
            sc_matrix:    [F, F] 或 [B, F, F] 结构连接
            pathology_score: [B, 1] 或 None（HC 预训练无需）
        Returns:
            dict：
              ``pred_next``  [B, F, H] 下一状态预测（原始空间；H = forecast_offsets 数）
              ``pred_delta`` [B, F, H] Δx̂ = x̂ − x_t（原始空间）
              ``pred``       [B, F, H, 1] 内部 4D 张量（与既有 forward 输出同布局）
              ``aux_info``   模型附加输出（概率头 / MoE 统计 / delta 等）
        """
        if not self.next_state_mode:
            raise RuntimeError(
                "predict_next_state 仅适用于 --task_mode next_timepoint"
                "（预测头 head_anchor_mode != 'history_window'）。")
        x = self._to_internal_history(bold_history)
        pred, aux_info = self.forward(x, sc_matrix, pathology_score)
        pred_next = pred[:, :, :, 0]                              # [B,F,H]
        x_last = bold_history[:, -1, :].unsqueeze(-1)             # [B,F,1] = x_t
        return {'pred_next': pred_next, 'pred_delta': pred_next - x_last,
                'pred': pred, 'aux_info': aux_info}

    def rollout_next_states(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                            pathology_score: Optional[torch.Tensor] = None,
                            steps: int = 1) -> torch.Tensor:
        """free rollout：自回归滚动预测未来 ``steps`` 个 TR 的全脑状态。

        每一步用模型自己的预测（第 0 个 offset）接到 context 末尾、滑出最早 1 个 TR，
        **中间不重新使用任何 ground truth**，返回 ``[B, steps, F]`` 的自回归轨迹。
        既供评估（H=1,2,4,8,16 的 horizon 指标）复用，也可在
        ``--enable_rollout_loss`` 下带梯度调用（训练侧短程 rollout 损失）。
        """
        if not self.next_state_mode:
            raise RuntimeError("rollout_next_states 仅适用于 --task_mode next_timepoint。")
        if bold_history.ndim != 3:
            raise ValueError(f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}")
        n_steps = int(steps)
        if n_steps < 1:
            raise ValueError(f"steps 必须 >= 1，收到 {n_steps}")
        if bold_history.shape[1] < 2:
            raise ValueError("rollout_next_states 至少需要 K >= 2 的 context 才能滑窗。")
        cur = bold_history
        preds: List[torch.Tensor] = []
        for k in range(n_steps):
            out = self.predict_next_state(cur, sc_matrix, pathology_score)
            nxt = out['pred_next'][:, :, 0]                        # [B,F]（offset=+1）
            preds.append(nxt)
            if k + 1 >= n_steps:
                break
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)                           # [B, steps, F]

    def _denorm_aux_rounds(self, aux_info, norm_stats) -> None:
        """把轮间中间预测一并反归一化，使 aux 与最终预测处于同一数值空间。"""
        rounds = aux_info.get('refiner_round_preds', None) if aux_info else None
        if rounds:
            aux_info['refiner_round_preds'] = [
                self.rev_norm(r, 'denorm', stats=norm_stats) for r in rounds]
        # 幅值一致性损失需要 anchor/scale 与 target 同空间：anchor 走标准反归一化，
        # scale 是归一化空间中的乘性幅值，需乘上 (stdev / affine_weight) 因子
        amp = aux_info.get('head_amp', None) if aux_info else None
        if amp:
            _, stdev = norm_stats
            amp['anchor'] = self.rev_norm(amp['anchor'], 'denorm', stats=norm_stats)
            if self.rev_norm.affine:
                factor = stdev / (self.rev_norm.affine_weight.view(1, -1, 1, 1) + self.rev_norm.eps)
            else:
                factor = stdev
            amp['scale'] = amp['scale'] * factor

    def _denorm_aux_prob(self, aux_info, norm_stats) -> None:
        """把概率输出（logvar / 分位点）换算回与最终预测相同的数值空间。

        RevIN 反归一化是逐 ROI 的仿射变换 ``x_raw = (x - bias)/(w + eps) * stdev + mean``，
        斜率 ``factor = stdev / (w + eps)``：
        - 方差按斜率平方缩放 → ``logvar_raw = logvar_norm + 2*log(factor)``；
        - 分位点是对未来值的绝对预测，直接施加同一仿射变换。

        ``factor`` 影响均值项与（后续精修后的）最终预测的一致性，因此这里在
        ``forward`` / ``virtual_intervention`` 的 ``self.norm``
        分支统一调用；``--norm False`` 时无需换算。
        """
        if not aux_info or not self.norm:
            return
        mean, stdev = norm_stats
        if self.rev_norm.affine:
            weight = self.rev_norm.affine_weight.view(1, -1, 1, 1)
            bias   = self.rev_norm.affine_bias.view(1, -1, 1, 1)
            factor = stdev / (weight + self.rev_norm.eps)
        else:
            bias   = None
            factor = stdev

        lv = aux_info.get('pred_logvar', None)
        if lv is not None:
            aux_info['pred_logvar'] = lv + 2.0 * torch.log(
                factor.abs().clamp_min(1e-12))

        q = aux_info.get('pred_quantiles', None)
        if q is not None:
            b, f = q.shape[0], q.shape[1]
            f5 = factor.reshape(b, f, 1, 1, 1)
            m5 = mean.reshape(mean.shape[0], mean.shape[1], 1, 1, 1)
            if bias is None:
                aux_info['pred_quantiles'] = q * f5 + m5
            else:
                b5 = bias.reshape(1, -1, 1, 1, 1)
                aux_info['pred_quantiles'] = (q - b5) * f5 + m5

    def forward(
        self,
        dfc_data: torch.Tensor,
        sc_matrix: torch.Tensor,
        pathology_score: Optional[torch.Tensor] = None,
    ):
        """单次前向：整段 context 作为单窗口，并行预测各 offset 的下一时间点。"""
        history = dfc_data
        norm_stats = None
        if self.norm:
            history, norm_stats = self.rev_norm(history, 'norm')
        cond = self._prepare_pathology_condition(pathology_score)
        latent, sc_info, ode_diag = self._encode(history, sc_matrix, cond)
        pred_dfc, aux_info = self._decode(latent, history, sc_matrix, cond, sc_info,
                                          revin_stats=norm_stats)
        if ode_diag:
            aux_info['ode_diag'] = ode_diag
        if self.next_state_mode:
            # 必须在反归一化之前：pred_delta_norm / revin_factor 都在归一化空间
            self._fill_next_state_delta(aux_info, pred_dfc, history, norm_stats)
        if self.norm:
            pred_dfc = self.rev_norm(pred_dfc, 'denorm', stats=norm_stats)
            self._denorm_aux_rounds(aux_info, norm_stats)
            self._denorm_aux_prob(aux_info, norm_stats)
        return pred_dfc, aux_info

    @torch.no_grad()
    def virtual_intervention(
        self,
        dfc_data: torch.Tensor,
        sc_matrix: torch.Tensor,
        pathology_score: Optional[torch.Tensor] = None,
        target_roi_idx: int = 0,
        intervention_type: str = 'excitatory',
        intensity: float = 1.0,
        intervention_mode: str = 'roi',
        target_pathology_score=None,
    ):
        """虚拟干预入口。

        ``intervention_mode``：
          - ``roi``（默认，向后兼容）：在 pastmixing 输出上对 ``target_roi_idx``
            做兴奋/抑制/方差扰动；
          - ``latent`` / ``parametric``：病理条件反事实干预，需给
            ``target_pathology_score``，内部委托给
            :meth:`counterfactual_pathology_sweep`（只做单个目标条件），
            此时 ``target_roi_idx``/``intervention_type`` 被忽略，``intensity``
            作为条件与潜在状态的混合系数。
        """
        if intervention_mode in ('latent', 'parametric'):
            if target_pathology_score is None:
                raise ValueError(
                    "intervention_mode='latent'/'parametric' 需要 target_pathology_score")
            sweep = self.counterfactual_pathology_sweep(
                dfc_data, sc_matrix, pathology_score=pathology_score,
                target_scores=[target_pathology_score],
                intervention_mode=intervention_mode, intensity=intensity)
            return sweep['preds'][0], sweep['aux'][0]
        if intervention_mode != 'roi':
            raise ValueError(
                "intervention_mode 仅支持 roi/latent/parametric，"
                f"收到 '{intervention_mode}'")
        if not (0 <= target_roi_idx < self.features):
            raise IndexError(
                f"target_roi_idx={target_roi_idx} out of range [0, {self.features - 1}]")
        history = dfc_data
        norm_stats = None
        if self.norm:
            history, norm_stats = self.rev_norm(history, 'norm')
        cond = self._prepare_pathology_condition(pathology_score)
        sc_info = self._compute_sc_prior(history, sc_matrix)
        adj_eff = sc_info['A_eff'] if sc_info is not None else None
        shallow = self.dfc_adapter(history, sc_matrix, adj=adj_eff)
        x = self.pastmixing(shallow, cond).clone()

        if intervention_type == 'excitatory':
            x[:, target_roi_idx] = x[:, target_roi_idx] + intensity
        elif intervention_type == 'inhibitory':
            x[:, target_roi_idx] = x[:, target_roi_idx] - intensity
        elif intervention_type == 'variance_boost':
            x[:, target_roi_idx] = x[:, target_roi_idx] * (1.0 + intensity)
        elif intervention_type == 'variance_suppress':
            x[:, target_roi_idx] = x[:, target_roi_idx] * max(0.0, 1.0 - intensity)
        else:
            raise ValueError(f"Unsupported intervention_type: {intervention_type}")

        for block_scale, ode_block in zip(self.ode_block_scales, self.ode_blocks):
            evolved = ode_block(x, sc_matrix, cond, adj_eff=adj_eff)
            scale   = torch.sigmoid(block_scale)
            x = x + scale * (evolved - x)
        x = x + self.post_fusion(torch.cat([shallow, x], dim=1))
        latent = self.feature_norm(x)
        pred_dfc, aux_info = self._decode(latent, history, sc_matrix, cond, sc_info,
                                          revin_stats=norm_stats)
        if self.norm:
            pred_dfc = self.rev_norm(pred_dfc, 'denorm', stats=norm_stats)
            self._denorm_aux_rounds(aux_info, norm_stats)
            self._denorm_aux_prob(aux_info, norm_stats)
        return pred_dfc, aux_info

    # ------------------------------------------------------------------
    # Phase 7.3 反事实扫掠
    # ------------------------------------------------------------------
    @staticmethod
    def _as_target_score_row(value, batch: int, device) -> torch.Tensor:
        """把单个目标评分规整为 [B] 张量（标量广播到整个 batch）。"""
        t = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        t = t.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            return t.expand(batch)
        if t.numel() != batch:
            raise ValueError(f"目标评分长度需为 1 或 batch={batch}，收到 {t.numel()}")
        return t

    @torch.no_grad()
    def counterfactual_pathology_sweep(
        self,
        dfc_data: torch.Tensor,
        sc_matrix: torch.Tensor,
        pathology_score: Optional[torch.Tensor] = None,
        target_scores=None,
        intervention_mode: str = 'latent',
        intensity: float = 1.0,
    ):
        """病理条件反事实扫掠：仅改变条件向量，观察预测 dFC 如何随之变化。

        ⚠️ **该输出不构成任何临床因果结论**：数据是观察性的、条件是相关量表评分，
        这里得到的只是**模型对条件输入的内部响应**，用于检查条件通路是否真的被
        主干与残差使用（条件敏感性诊断），不能解释为干预疗效或因果效应。

        Args:
            pathology_score: 基线原始评分 ``[B, 1]`` 或 ``[B]``；必需。
            target_scores: 目标原始评分序列，元素可为标量（广播到整个 batch）或
                ``[B]`` 张量。``None`` 时默认以基线评分为中心、按归一化器尺度
                （IQR/1.349，即 robust_z 的 1 个单位）取 ``-2σ…+2σ`` 五个点
                （截断到非负，HAMD 等量表评分非负）。
            intervention_mode:
                - ``'latent'``（默认）：分别编码基线与目标条件得到两个潜在状态，
                  再按 ``h_base + intensity·(h_target - h_base)`` 混合后解码；
                  条件向量按同一系数混合，避免 AdaLN/MoE 与潜在状态不一致；
                - ``'parametric'``：直接用目标条件完整重跑一次前向（``intensity``
                  在此模式下不生效）。
            intensity: ``'latent'`` 模式下的混合系数（0 = 基线，1 = 目标）。

        Returns:
            dict: ``{'mode', 'target_scores'[T,B], 'baseline'[B,F,W,S],
            'preds'[T,B,F,W,S], 'baseline_aux', 'aux'}``。

        Note:
            该方法按**单次前向**计算（与 forward 相同的并行预测口径）；
            建议在 ``model.eval()`` 下调用（内部已 ``torch.no_grad``）。
        """
        if self.pathology_normalizer is None:
            raise ValueError("反事实扫掠需要病理条件模块（微调模式）。")
        if intervention_mode not in ('latent', 'parametric'):
            raise ValueError(
                "intervention_mode 仅支持 latent/parametric，"
                f"收到 '{intervention_mode}'")
        if self.pathology_normalizer.input_dim != 1:
            raise NotImplementedError(
                "反事实扫掠目前仅支持单通道病理评分（pathology_input_dim=1），"
                f"收到 input_dim={self.pathology_normalizer.input_dim}。")
        if pathology_score is None:
            raise ValueError("反事实扫掠需要基线病理评分 pathology_score。")
        if self.pretrain_mode:
            raise ValueError("预训练模式没有病理条件模块，无法做反事实扫掠。")

        dev = dfc_data.device
        b = int(dfc_data.shape[0])
        norm = self.pathology_normalizer
        raw_base = pathology_score.detach().to(
            device=dev, dtype=torch.float32).reshape(b, -1)[:, 0]

        if target_scores is None:
            step = norm.scale.reshape(-1)[0].to(
                device=dev, dtype=torch.float32).clamp_min(1e-6)
            rows = [(raw_base + k * step).clamp_min(0.0)
                    for k in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        else:
            rows = [self._as_target_score_row(v, b, dev) for v in target_scores]

        history = dfc_data
        norm_stats = None
        if self.norm:
            history, norm_stats = self.rev_norm(history, 'norm')
        # A_eff 只由归一化历史与 SC 导出，与条件无关，可全程复用
        sc_info = self._compute_sc_prior(history, sc_matrix)
        cond_base = self._prepare_pathology_condition(pathology_score)
        lat_base = self._encode(history, sc_matrix, cond_base)[0]

        def _denorm(pred_norm, aux_info):
            if not self.norm:
                return pred_norm
            pred = self.rev_norm(pred_norm, 'denorm', stats=norm_stats)
            self._denorm_aux_rounds(aux_info, norm_stats)
            self._denorm_aux_prob(aux_info, norm_stats)
            return pred

        preds, aux_list = [], []
        for row in rows:
            cond_t = norm(row.reshape(b, 1))
            if intervention_mode == 'parametric':
                latent, cond_use = self._encode(history, sc_matrix, cond_t)[0], cond_t
            else:
                lat_t = self._encode(history, sc_matrix, cond_t)[0]
                latent = lat_base + float(intensity) * (lat_t - lat_base)
                cond_use = (cond_base + float(intensity) * (cond_t - cond_base)
                            if cond_base is not None else cond_t)
            pred_norm, aux_info = self._decode(
                latent, history, sc_matrix, cond_use, sc_info, revin_stats=norm_stats)
            preds.append(_denorm(pred_norm, aux_info))
            aux_list.append(aux_info)

        base_norm, base_aux = self._decode(
            lat_base, history, sc_matrix, cond_base, sc_info, revin_stats=norm_stats)
        return {
            'mode': intervention_mode,
            'target_scores': torch.stack(rows, dim=0),        # [T, B]
            'baseline': _denorm(base_norm, base_aux),
            'baseline_aux': base_aux,
            'preds': torch.stack(preds, dim=0),               # [T, B, F, W, S]
            'aux': aux_list,
        }