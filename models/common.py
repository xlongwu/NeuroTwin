import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Shared utilities
# -------------------------------------------------


def prepare_sc_matrix(sc_matrix: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare subject-level structural connectivity.

    Args:
        sc_matrix: [F, F] or [B, F, F]
        x:         [B, F, W, S]

    Returns:
        sc_sym:    symmetrized non-negative SC
        adj_norm:  D^{-1/2} A D^{-1/2} normalized adjacency with self-loop
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x [B, F, W, S], got {tuple(x.shape)}")

    b, n, _, _ = x.shape
    sc_matrix = sc_matrix.to(device=x.device, dtype=x.dtype)

    if sc_matrix.ndim == 2:
        if sc_matrix.shape != (n, n):
            raise ValueError(f"Expected sc_matrix shape {(n, n)}, got {tuple(sc_matrix.shape)}")
        sc_matrix = sc_matrix.unsqueeze(0).expand(b, -1, -1)
    elif sc_matrix.ndim == 3:
        if sc_matrix.shape[0] != b or sc_matrix.shape[1:] != (n, n):
            raise ValueError(f"Expected batch sc_matrix shape {(b, n, n)}, got {tuple(sc_matrix.shape)}")
    else:
        raise ValueError(f"sc_matrix must be [F, F] or [B, F, F], got {tuple(sc_matrix.shape)}")

    sc_sym = 0.5 * (sc_matrix + sc_matrix.transpose(-1, -2))
    sc_sym = sc_sym.clamp_min(0.0)

    eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0)
    adj = sc_sym + eye
    deg = adj.sum(dim=-1)
    deg_inv_sqrt = deg.clamp_min(1e-6).pow(-0.5)
    adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
    return sc_sym, adj_norm


# -------------------------------------------------
# Normalization
# -------------------------------------------------


class BrainRevIN(nn.Module):
    """
    ROI-wise reversible instance normalization for [B, F, W, S].
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(2, 3), keepdim=True).detach()
        stdev = torch.sqrt(x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).detach()
        return mean, stdev

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        x = (x - mean) / stdev.clamp_min(self.eps)
        if self.affine:
            weight = self.affine_weight.view(1, -1, 1, 1)
            bias = self.affine_bias.view(1, -1, 1, 1)
            x = x * weight + bias
        return x

    def _denormalize(self, x: torch.Tensor, stats, target_slice=slice(None)) -> torch.Tensor:
        mean, stdev = stats
        if self.affine:
            weight = self.affine_weight[target_slice].view(1, -1, 1, 1)
            bias = self.affine_bias[target_slice].view(1, -1, 1, 1)
            x = (x - bias) / (weight + self.eps)
        x = x * stdev[:, target_slice, :, :]
        x = x + mean[:, target_slice, :, :]
        return x

    def forward(self, x: torch.Tensor, mode: str, stats=None, target_slice=slice(None)):
        if mode == 'norm':
            mean, stdev = self._get_statistics(x)
            return self._normalize(x, mean, stdev), (mean, stdev)
        if mode == 'denorm':
            if stats is None:
                raise ValueError("stats must be provided when mode='denorm'")
            return self._denormalize(x, stats, target_slice)
        raise NotImplementedError(f"Unsupported mode: {mode}")


class SEChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        pooled = x.mean(dim=(2, 3))
        gate = self.net(pooled).view(b, c, 1, 1)
        return x * gate


# -------------------------------------------------
# Backbone modules
# -------------------------------------------------


class DFCAdapter(nn.Module):
    """
    Multi-order SC-guided graph adapter.

    Compared with the previous single-pass diffusion, this block mixes
    identity / 1-hop / 2-hop graph propagation, which is more suitable
    for next-window correlation prediction because many ROI relations are
    not purely first-order on the structural graph.
    """

    def __init__(self, num_nodes: int, alpha: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        alpha = float(min(max(alpha, 1e-4), 1 - 1e-4))
        init_logit = math.log(alpha / (1 - alpha))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.self_loop = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.order_logits = nn.Parameter(torch.tensor([1.5, 1.0, 0.5], dtype=torch.float32))
        self.channel_gate = SEChannelGate(num_nodes, reduction=4)
        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=num_nodes)

    def forward(self, x: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B, F, W, S], got {tuple(x.shape)}")

        b, n, w, s = x.shape
        sc_sym, _ = prepare_sc_matrix(sc_matrix, x)

        learned = F.softplus(self.edge_logits)
        learned = 0.5 * (learned + learned.transpose(0, 1))
        adj = sc_sym * learned.unsqueeze(0)

        eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0)
        adj = adj + eye * F.softplus(self.self_loop)
        deg = adj.sum(dim=-1)
        deg_inv_sqrt = deg.clamp_min(1e-6).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        x_flat = x.reshape(b, n, -1)
        x1 = torch.bmm(adj, x_flat)
        x2 = torch.bmm(adj, x1)

        order_weights = F.softmax(self.order_logits, dim=0)
        mixed = (
            order_weights[0] * x_flat +
            order_weights[1] * x1 +
            order_weights[2] * x2
        ).reshape(b, n, w, s)

        mixed = self.channel_gate(mixed)
        alpha = torch.sigmoid(self.alpha_logit)
        return self.out_norm(x + alpha * (mixed - x))


class BrainMDM(nn.Module):
    """
    Dual-axis multi-scale mixer.

    The old version is good at local temporal texture, but the validation
    curves show a long plateau, which usually indicates that the model is
    not fully exploiting cross-window progression. Here we explicitly give
    separate branches to the sequence axis and the window axis.
    """

    def __init__(self, features: int, num_window: int, seq_len: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_window = num_window

        self.out_sizes_s = sorted(set(max(1, seq_len // (2 ** (num_scales - i))) for i in range(num_scales)))
        self.out_sizes_w = sorted(set(max(1, num_window // (2 ** (num_scales - i))) for i in range(num_scales)))

        self.pool_s = nn.ModuleList([nn.AdaptiveAvgPool1d(out_s) for out_s in self.out_sizes_s])
        self.linear_s = nn.ModuleList([nn.Linear(out_s, seq_len) for out_s in self.out_sizes_s])

        self.pool_w = nn.ModuleList([nn.AdaptiveAvgPool1d(out_w) for out_w in self.out_sizes_w])
        self.linear_w = nn.ModuleList([nn.Linear(out_w, num_window) for out_w in self.out_sizes_w])

        seq_kernels = [3, 5]
        win_kernels = [3, 5]
        self.seq_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(1, k), padding=(0, k // 2), groups=features),
                nn.GELU(),
                nn.Conv2d(features, features, kernel_size=1),
            )
            for k in seq_kernels
        ])
        self.win_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(k, 1), padding=(k // 2, 0), groups=features),
                nn.GELU(),
                nn.Conv2d(features, features, kernel_size=1),
            )
            for k in win_kernels
        ])
        self.cross_branch = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
        )

        total_branches = 1 + len(self.seq_branches) + len(self.win_branches) + 2 + 1
        hidden = max(features, features * 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(features * total_branches, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, features, kernel_size=1),
        )
        self.channel_gate = SEChannelGate(features, reduction=4)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=features)

    def _pool_along_s(self, x: torch.Tensor) -> torch.Tensor:
        b, f, w, s = x.shape
        x_s = x.reshape(b * f * w, s)
        agg = torch.zeros_like(x)
        for pool, linear in zip(self.pool_s, self.linear_s):
            pooled = pool(x_s.unsqueeze(1)).squeeze(1)
            projected = linear(pooled)
            agg = agg + projected.reshape(b, f, w, s)
        return agg / max(1, len(self.pool_s))

    def _pool_along_w(self, x: torch.Tensor) -> torch.Tensor:
        b, f, w, s = x.shape
        x_w = x.permute(0, 1, 3, 2).reshape(b * f * s, w)
        agg = torch.zeros_like(x)
        for pool, linear in zip(self.pool_w, self.linear_w):
            pooled = pool(x_w.unsqueeze(1)).squeeze(1)
            projected = linear(pooled)
            agg = agg + projected.reshape(b, f, s, w).permute(0, 1, 3, 2)
        return agg / max(1, len(self.pool_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B, F, W, S], got {tuple(x.shape)}")

        branches = [x]
        branches.extend(branch(x) for branch in self.seq_branches)
        branches.extend(branch(x) for branch in self.win_branches)
        branches.append(self._pool_along_s(x))
        branches.append(self._pool_along_w(x))
        branches.append(self.cross_branch(x))

        fused = self.fusion(torch.cat(branches, dim=1))
        fused = self.channel_gate(fused)
        return self.norm(fused + x)


class GraphODE(nn.Module):
    """
    SC-guided neural dynamics core.

    Key improvements over the previous core:
    1. ROI attention still uses SC as hard mask + soft bias.
    2. A new window-attention branch explicitly models history progression.
    3. Temporal local branch and FFN are retained for stability.
    """

    def __init__(
        self,
        features: int,
        seq_len: int,
        hidden_dim: int,
        dropout: float = 0.2,
        num_heads: int = 4,
        window_heads: int = 5,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        if seq_len % window_heads != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by window_heads ({window_heads})")

        self.features = features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(seq_len)
        self.temporal_branch = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=(1, 3), padding=(0, 1), groups=features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(features, features, kernel_size=1),
        )

        self.q_proj = nn.Linear(seq_len, hidden_dim)
        self.k_proj = nn.Linear(seq_len, hidden_dim)
        self.v_proj = nn.Linear(seq_len, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, seq_len)

        self.window_norm = nn.LayerNorm(seq_len)
        self.window_attn = nn.MultiheadAttention(
            embed_dim=seq_len,
            num_heads=window_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
        )

        self.attn_gate = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.temporal_gate = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.window_gate = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.ffn_gate = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.sc_bias_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _prepare_sc(self, sc_matrix: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, w, _ = h.shape
        sc_sym, _ = prepare_sc_matrix(sc_matrix, h)
        eye = torch.eye(n, device=h.device, dtype=h.dtype).unsqueeze(0)
        sc_with_self = (sc_sym + eye).clamp_min(0.0)
        mask = sc_with_self > 0
        bias = torch.log1p(sc_with_self)

        mask = mask.unsqueeze(1).expand(b, w, n, n).reshape(b * w, n, n)
        bias = bias.unsqueeze(1).expand(b, w, n, n).reshape(b * w, n, n)
        return mask, bias

    def _graph_attention(self, h: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        b, n, w, s = h.shape
        tokens = h.transpose(1, 2).reshape(b * w, n, s)

        q = self.q_proj(tokens).reshape(b * w, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).reshape(b * w, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).reshape(b * w, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask, bias = self._prepare_sc(sc_matrix, h)
        attn_scores = attn_scores + self.sc_bias_scale * bias.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), -1e4)

        attn = F.softmax(attn_scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(b * w, n, self.hidden_dim)
        out = self.out_proj(out)
        return out.reshape(b, w, n, s).transpose(1, 2)

    def _window_attention(self, h: torch.Tensor) -> torch.Tensor:
        b, n, w, s = h.shape
        tokens = h.reshape(b * n, w, s)
        tokens_norm = self.window_norm(tokens)
        out, _ = self.window_attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        return out.reshape(b, n, w, s)

    def forward(self, t, h: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        del t
        if h.ndim != 4:
            raise ValueError(f"Expected h [B, F, W, S], got {tuple(h.shape)}")

        h_norm = self.input_norm(h)
        dh_space = self._graph_attention(h_norm, sc_matrix)
        dh_time = self.temporal_branch(h_norm)
        dh_window = self._window_attention(h_norm)
        dh_ffn = self.ffn(h_norm)

        dh_dt = (
            torch.sigmoid(self.attn_gate) * dh_space +
            torch.sigmoid(self.temporal_gate) * dh_time +
            torch.sigmoid(self.window_gate) * dh_window +
            torch.sigmoid(self.ffn_gate) * dh_ffn
        )
        return dh_dt


class GraphODEDDI(nn.Module):
    """
    Heun / RK2 integration wrapper.
    """

    def __init__(
        self,
        features: int,
        seq_len: int,
        hidden_dim: int = 64,
        step_scale: float = 0.1,
        ode_steps: int = 5,
        dropout: float = 0.2,
        stochastic_depth_rate: float = 0.1,
        num_heads: int = 4,
        window_heads: int = 5,
    ):
        super().__init__()
        self.ode_func = GraphODE(
            features=features,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            window_heads=window_heads,
        )
        step_scale = float(max(step_scale, 1e-4))
        self.step_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(step_scale)), dtype=torch.float32))
        self.ode_steps = ode_steps
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / max(1, self.ode_steps)
        h = x
        step_scale = F.softplus(self.step_scale_raw)
        survival = max(1e-6, 1.0 - self.stochastic_depth_rate)

        for _ in range(self.ode_steps):
            k1 = self.ode_func(0, h, sc_matrix)
            k2 = self.ode_func(0, h + dt * k1, sc_matrix)
            delta = step_scale * dt * 0.5 * (k1 + k2)

            if self.training and self.stochastic_depth_rate > 0:
                if torch.rand(1, device=h.device).item() < self.stochastic_depth_rate:
                    continue
                delta = delta / survival

            h = h + delta
        return h


class PredictionRefiner(nn.Module):
    """
    Lightweight graph-temporal refinement on the predicted future signal.
    """

    def __init__(self, features: int, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=(1, 3), padding=(0, 1), groups=features),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(features * 3, features * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(features * 2, features, kernel_size=1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4:
            raise ValueError(f"Expected pred [B, F, W, S], got {tuple(pred.shape)}")

        b, n, w, s = pred.shape
        _, adj_norm = prepare_sc_matrix(sc_matrix, pred)
        pred_flat = pred.reshape(b, n, -1)
        g1 = torch.bmm(adj_norm, pred_flat).reshape(b, n, w, s)
        g2 = torch.bmm(adj_norm, torch.bmm(adj_norm, pred_flat)).reshape(b, n, w, s)
        local = self.local(pred)
        fused = self.mix(torch.cat([g1, g2, local], dim=1))
        return torch.tanh(self.res_scale) * fused


class IterativePredictionRefiner(nn.Module):
    """
    多轮级联细化：每轮用 SC 结构约束消除上一轮的残差误差。

    设计动机
    ────────
    PredictionRefiner 单次执行一步图卷积细化。预测残差往往包含多层级
    误差（全局趋势偏差、局部节奏偏差、ROI 间相关偏差），单次图卷积
    无法在单次前向中同时消除所有层级的误差。多轮迭代让模型逐级修正，
    类似数值求解器的迭代改进——每轮在修正前一轮的系统误差基础上
    进一步精化，通常 2~3 轮可收敛。

    接口一致性
    ──────────
    与 PredictionRefiner 保持相同接口：接收 pred + sc_matrix，
    返回累计残差 delta（调用方执行 pred = pred + delta），
    因此可作为 PredictionRefiner 的直接替换，无需修改调用代码。

    每轮权重设计
    ────────────
    round_scales 初始化为递减序列 [sigmoid(0), sigmoid(-0.69), sigmoid(-1.39)]
    ≈ [0.5, 0.33, 0.20]，第一轮贡献最大，后续轮贡献递减，
    符合"首轮消除主要误差，后续轮精细修正"的直觉。
    scales 是可学习参数，允许模型自适应调整各轮权重。

    Args:
        features:  ROI 通道数（与 PredictionRefiner 相同）
        n_rounds:  细化轮数，默认 3（通常 2~3 轮即可收敛）
        dropout:   各轮 refiner 的 dropout 率
    """

    def __init__(self, features: int, n_rounds: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_rounds = n_rounds
        self.refiners = nn.ModuleList([
            PredictionRefiner(features=features, dropout=dropout)
            for _ in range(n_rounds)
        ])
        # 递减初始化：第 i 轮的 logit = log(0.5^i / (1 - 0.5^i))
        # sigmoid(logit_i) ≈ 0.5^i，权重随轮次递减
        self.round_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(
                math.log(max(1e-4, (0.5 ** i)) / max(1e-4, 1.0 - (0.5 ** i))),
                dtype=torch.float32))
            for i in range(n_rounds)
        ])

    def forward(self, pred: torch.Tensor, sc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:      [B, F, W, S] 待细化的预测张量
            sc_matrix: [F, F] 或 [B, F, F] 结构连接矩阵

        Returns:
            accumulated delta [B, F, W, S]，与 PredictionRefiner 接口一致。
            调用方执行 pred = pred + refiner(pred, sc_matrix)。
        """
        accumulated = torch.zeros_like(pred)
        current = pred
        for refiner, scale_param in zip(self.refiners, self.round_scales):
            # 每轮 refiner 基于当前累计修正后的 pred 计算新的残差
            delta = refiner(current, sc_matrix)
            weighted_delta = torch.sigmoid(scale_param) * delta
            accumulated = accumulated + weighted_delta
            # 更新 current：下一轮基于修正后的预测，实现迭代精化
            current = current + weighted_delta
        return accumulated