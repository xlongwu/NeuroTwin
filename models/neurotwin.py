from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    BrainRevIN, DFCAdapter, BrainMDM, GraphODEDDI,
    IterativePredictionRefiner,
)
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

    Features 6 fusion branches:
      1. history_proj (flattened history features)
      2. latent_proj (flattened latent features)
      3. temporal_branch (depthwise conv on time axis)
      4. cross_roi_hist (1x1 conv on ROI axis of history)
      5. cross_roi_latent (1x1 conv on ROI axis of latent)
      6. window_temporal_attn (causal cross-window attention)

    With 3-round iterative SC-constrained refinement (IterativePredictionRefiner).
    """

    def __init__(
        self,
        features: int,
        in_window: int,
        in_seq_len: int,
        pred_window: int,
        pred_seq_len: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features     = features
        self.in_window    = in_window
        self.pred_window  = pred_window
        self.pred_seq_len = pred_seq_len
        self.in_dim  = in_window * in_seq_len
        self.out_dim = pred_window * pred_seq_len
        hidden_dim   = min(768, max(256, self.in_dim * 2))

        self.history_norm   = nn.LayerNorm(self.in_dim)
        self.latent_norm    = nn.LayerNorm(self.in_dim)
        self.history_proj   = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.latent_proj    = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=5, padding=2, groups=features),
            nn.GELU(), nn.Dropout(dropout))
        self.temporal_proj  = nn.Linear(self.in_dim, hidden_dim)
        self.cross_roi_hist = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=1), nn.GELU(), nn.Dropout(dropout))
        self.cross_roi_latent = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=1), nn.GELU(), nn.Dropout(dropout))

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

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.Dropout(dropout))

        self.trend_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim))
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim))
        self.scale_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, pred_window))

        self.refiner = IterativePredictionRefiner(
            features=features, n_rounds=3, dropout=dropout)

        make_zero_last_linear(self.shape_head[-1], bias=0.0)
        make_zero_last_linear(self.scale_head[-1], bias=-2.0)

    @staticmethod
    def _choose_win_hidden(in_seq_len: int, hidden_dim: int, num_heads: int = 4) -> int:
        upper = max(num_heads * 4, hidden_dim // 2)
        candidate = max(num_heads * 4, (in_seq_len * 2 // num_heads) * num_heads)
        return min(candidate, upper)

    def _build_anchor(self, history: torch.Tensor) -> torch.Tensor:
        b, f, w, s = history.shape
        last  = history[:, :, -1:, :]
        slope = last - history[:, :, -2:-1, :] if w >= 2 else torch.zeros_like(last)
        anchors, current = [], last
        for _ in range(self.pred_window):
            current = current + 0.5 * slope
            anchors.append(current)
        return (torch.cat(anchors, dim=2) if anchors
                else torch.zeros(b, f, 0, s, device=history.device, dtype=history.dtype))

    def forward(
        self,
        latent: torch.Tensor,
        history: torch.Tensor,
        sc_matrix: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 4 or history.ndim != 4:
            raise ValueError(
                f"Expected [B,F,W,S], got {tuple(latent.shape)} and {tuple(history.shape)}")
        b, f, _, _ = latent.shape

        history_flat = history.reshape(b, f, -1)
        latent_flat  = latent.reshape(b, f, -1)

        history_feat  = self.history_proj(self.history_norm(history_flat))
        latent_feat   = self.latent_proj(self.latent_norm(latent_flat))
        temporal_feat = self.temporal_proj(self.temporal_branch(history_flat))
        cross_hist    = self.cross_roi_hist(history_feat)
        cross_latent  = self.cross_roi_latent(latent_feat)

        win_x    = self.win_proj(history)
        win_ctx  = self.window_temporal_attn(win_x)
        win_feat = self.win_flatten_proj(win_ctx.reshape(b, f, -1))

        fused = self.fusion(torch.cat(
            [history_feat, latent_feat, temporal_feat,
             cross_hist, cross_latent, win_feat], dim=-1))

        anchor    = self._build_anchor(history)
        trend     = self.trend_head(fused).view(b, f, self.pred_window, self.pred_seq_len)
        shape_raw = self.shape_head(fused).view(b, f, self.pred_window, self.pred_seq_len)
        shape     = standardize_future(shape_raw)
        scale     = F.softplus(self.scale_head(fused)).view(b, f, self.pred_window, 1)
        pred = anchor + trend + scale * shape

        pred = pred + self.refiner(pred, sc_matrix)
        return pred


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
        moe_router_cond_only: bool = True,
        moe_use_argmax: bool = False,
        moe_inference_temperature: float = 0.3,
    ):
        super().__init__()
        self.features         = features
        self.norm             = norm
        self.pretrain_mode    = pretrain_mode
        self.pred_window      = pred_window
        self.pred_seq_len     = pred_seq_len

        if self.norm:
            self.rev_norm = BrainRevIN(num_features=features)

        self.input_dropout = nn.Dropout2d(p=min(0.2, dropout * 0.5))
        self.dfc_adapter   = DFCAdapter(num_nodes=features, alpha=adapter_alpha)
        self.pastmixing    = BrainMDM(
            features=features, num_window=in_window,
            seq_len=in_seq_len, num_scales=num_scales, dropout=dropout)

        self.ode_blocks = nn.ModuleList([
            GraphODEDDI(
                features=features, seq_len=in_seq_len,
                hidden_dim=ode_hidden_dim, ode_steps=ode_steps,
                dropout=dropout, stochastic_depth_rate=stochastic_depth_rate,
                num_heads=4, window_heads=5,
            )
            for _ in range(n_block)
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
            pred_window=pred_window, pred_seq_len=pred_seq_len, dropout=dropout)

        if self.pretrain_mode:
            self.moe = None
        else:
            self.moe = NeuroTwinMoE(
                features=features,
                in_w=in_window, in_s=in_seq_len,
                pred_w=pred_window, pred_s=pred_seq_len,
                pathology_input_dim=pathology_input_dim,
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
            )

    def set_moe_router_temperature(self, temperature: float) -> None:
        if self.moe is not None:
            self.moe.set_router_temperature(temperature)

    def set_moe_router_inference_temperature(self, temperature: float) -> None:
        if self.moe is not None:
            self.moe.set_router_inference_temperature(temperature)

    def _encode(self, x_norm: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 4:
            raise ValueError(f"Expected [B,F,W,S], got {tuple(x_norm.shape)}")
        x = x_norm
        if self.training:
            x = self.input_dropout(x)
        shallow = self.dfc_adapter(x, sc_matrix)
        x = self.pastmixing(shallow)
        for block_scale, ode_block in zip(self.ode_block_scales, self.ode_blocks):
            evolved = ode_block(x, sc_matrix)
            scale   = torch.sigmoid(block_scale)
            x = x + scale * (evolved - x)
        x = x + self.post_fusion(torch.cat([shallow, x], dim=1))
        x = self.feature_norm(x)
        return x

    def _decode(
        self,
        latent: torch.Tensor,
        history_norm: torch.Tensor,
        sc_matrix: torch.Tensor,
        pathology_score: Optional[torch.Tensor] = None,
    ):
        base_pred = self.pretrain_head(latent, history_norm, sc_matrix)
        if self.pretrain_mode:
            return base_pred, None
        if pathology_score is None:
            raise ValueError("pathology_score must be provided when pretrain_mode=False")
        if self.moe is None:
            raise RuntimeError("self.moe is None while pretrain_mode=False")
        delta_pred, aux_info = self.moe(
            latent, history_norm, base_pred, pathology_score, sc_matrix)
        return base_pred + delta_pred, aux_info

    def forward(
        self,
        dfc_data: torch.Tensor,
        sc_matrix: torch.Tensor,
        pathology_score: Optional[torch.Tensor] = None,
    ):
        history = dfc_data
        norm_stats = None
        if self.norm:
            history, norm_stats = self.rev_norm(history, 'norm')
        latent = self._encode(history, sc_matrix)
        pred_dfc, aux_info = self._decode(latent, history, sc_matrix, pathology_score)
        if self.norm:
            pred_dfc = self.rev_norm(pred_dfc, 'denorm', stats=norm_stats)
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
    ):
        if not (0 <= target_roi_idx < self.features):
            raise IndexError(
                f"target_roi_idx={target_roi_idx} out of range [0, {self.features - 1}]")
        history = dfc_data
        norm_stats = None
        if self.norm:
            history, norm_stats = self.rev_norm(history, 'norm')
        shallow = self.dfc_adapter(history, sc_matrix)
        x = self.pastmixing(shallow).clone()

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
            evolved = ode_block(x, sc_matrix)
            scale   = torch.sigmoid(block_scale)
            x = x + scale * (evolved - x)
        x = x + self.post_fusion(torch.cat([shallow, x], dim=1))
        latent = self.feature_norm(x)
        pred_dfc, aux_info = self._decode(latent, history, sc_matrix, pathology_score)
        if self.norm:
            pred_dfc = self.rev_norm(pred_dfc, 'denorm', stats=norm_stats)
        return pred_dfc, aux_info