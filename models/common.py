import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pathology import AdaLNConditioner, LowRankDelta


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
# Variable history length helpers
# -------------------------------------------------

# 变长前向统一约定（轨迹任务与 next_timepoint 任务共用）：
#
# - 窗口轴 W（历史 chunk / 窗口序号）：轨迹任务 history_min..history_max 可变；预测头 /
#   MoE 专家的扁平化分支按历史长度上限 L_max 建模，第 w 个窗口恒对应权重切片
#   [w*S : (w+1)*S]（长度不同时权重前缀共享、语义一致；L == L_max 时逐位等价）。
# - 时间轴 S（一个窗口内的 TR 序号；next_timepoint 任务下整段 context 就是单个
#   窗口，故 S == context 长度 K，K <= context_max）：所有按 S_max = context_max
#   建模的线性/LayerNorm 走**前缀切片**——context 的第 i 个时间点恒对应权重切片
#   中的第 i 个位置（输入侧取权重列前缀，输出侧取权重行前缀）；K == S_max 时与
#   定长实现逐位等价。卷积核（kernel 3/5）本身与位置无关，无需切片。
#
# 注意：这里刻意不把 nn.Linear 包成新 Module，保持 state_dict 键名与旧检查点一致。


def prefix_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """输入最后一维 d <= linear.in_features 时用权重前缀做线性投影。"""
    d = int(x.shape[-1])
    if d == linear.in_features:
        return linear(x)
    if d > linear.in_features:
        raise ValueError(
            f"输入宽度 {d} 超过该分支建模的最大宽度 {linear.in_features}："
            "history_max 配置与数据不一致。")
    return F.linear(x, linear.weight[:, :d], linear.bias)


def prefix_linear_out(linear: nn.Linear, x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """输出侧前缀截断：只取 linear 输出的前 ``out_dim`` 个通道（权重行前缀）。

    用于把“按最大宽度建模的回投影”（如 GraphODE 的 out_proj / FFN 输出、
    BrainMDM 的 S 轴池化回投影）对齐回实际宽度；``out_dim`` 等于权重行数时与
    ``linear(x)`` 逐位等价，保持定长路径数值不变。
    """
    n_out = int(linear.out_features)
    if out_dim == n_out:
        return linear(x)
    if out_dim > n_out:
        raise ValueError(
            f"目标宽度 {out_dim} 超过该分支建模的最大宽度 {n_out}。")
    bias = None if linear.bias is None else linear.bias[:out_dim]
    return F.linear(x, linear.weight[:out_dim], bias)


def prefix_layernorm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """按实际宽度做 LayerNorm，仿射参数取前缀（语义同 :func:`prefix_linear`）。"""
    d = int(x.shape[-1])
    if d == norm.normalized_shape[0]:
        return norm(x)
    if d > norm.normalized_shape[0]:
        raise ValueError(
            f"输入宽度 {d} 超过 LayerNorm 建模宽度 {norm.normalized_shape[0]}。")
    return F.layer_norm(x, (d,), norm.weight[:d], norm.bias[:d], norm.eps)


def prefix_proj(seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Sequential[Linear, ...] 的前缀版：首层按前缀投影，其余层原样执行。"""
    head = seq[0]
    if not isinstance(head, nn.Linear):
        raise TypeError(f"prefix_proj 期望首层为 nn.Linear，收到 {type(head).__name__}")
    out = prefix_linear(head, x)
    for layer in list(seq)[1:]:
        out = layer(out)
    return out


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

    def normalize_with(self, x: torch.Tensor, stats) -> torch.Tensor:
        """用**外部给定的**统计量归一化 x（不重新估计）。

        用于 latent 动力学监督：把未来真值 chunk 用“历史统计量”归一化后再过编码器，
        使 z_(t+k) 与历史 latent 处于同一仿射空间——同时保证归一化统计量只来自历史，
        未来 target 不参与任何统计量估计（无 normalization leakage）。
        """
        mean, stdev = stats
        return self._normalize(x, mean, stdev)

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

    def __init__(self, num_nodes: int, alpha: float = 0.5, sc_prior_mode: str = 'scaled'):
        super().__init__()
        self.num_nodes = num_nodes
        # 'scaled' 时逐字保留旧的 SC 度归一化路径；其余模式复用上游 A_eff（SC 软先验）
        self.sc_prior_mode = sc_prior_mode
        self.edge_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        alpha = float(min(max(alpha, 1e-4), 1 - 1e-4))
        init_logit = math.log(alpha / (1 - alpha))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.self_loop = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.order_logits = nn.Parameter(torch.tensor([1.5, 1.0, 0.5], dtype=torch.float32))
        self.channel_gate = SEChannelGate(num_nodes, reduction=4)
        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=num_nodes)

    def forward(self, x: torch.Tensor, sc_matrix: torch.Tensor,
                adj: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            adj: 可选的 [B, F, F] 软邻接矩阵（A_eff，已归一化）。仅在
                 sc_prior_mode != 'scaled' 时生效；为 None 时退回旧 SC 路径，
                 保证 --sc_prior_mode scaled 与引入软先验前的数值逐位一致。
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x [B, F, W, S], got {tuple(x.shape)}")

        b, n, w, s = x.shape
        learned = F.softplus(self.edge_logits)
        learned = 0.5 * (learned + learned.transpose(0, 1))

        if self.sc_prior_mode == 'scaled' or adj is None:
            sc_sym, _ = prepare_sc_matrix(sc_matrix, x)
            adj = sc_sym * learned.unsqueeze(0)
        else:
            # SC 软先验路径：A_eff 已含 SC 与功能先验；仍保留逐边可学习增益 learned
            adj = adj.to(dtype=x.dtype)
            if adj.ndim == 2:
                adj = adj.unsqueeze(0)
            if adj.shape[0] == 1 and b > 1:
                adj = adj.expand(b, -1, -1)
            adj = adj * learned.unsqueeze(0)

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

    def __init__(self, features: int, num_window: int, seq_len: int, num_scales: int = 3,
                 dropout: float = 0.1, ada_cond_dim: int = None,
                 scale_scheme: str = 'divisor', scale_gate: str = 'sample',
                 verbose: bool = True):
        super().__init__()
        if scale_scheme not in ('pow2', 'divisor'):
            raise ValueError(f"scale_scheme 仅支持 pow2/divisor，收到 '{scale_scheme}'")
        if scale_gate not in ('none', 'sample'):
            raise ValueError(f"scale_gate 仅支持 none/sample，收到 '{scale_gate}'")
        self.seq_len = seq_len
        self.num_window = num_window
        self.scale_scheme = scale_scheme
        self.scale_gate_mode = scale_gate

        self.out_sizes_s = self._scale_sizes(seq_len, num_scales, scale_scheme)
        self.out_sizes_w = self._scale_sizes(num_window, num_scales, scale_scheme)
        if verbose:
            # 运行期打印有效尺度数：pow2 方案在小 L 上会退化（W=6 → {1,3}），
            # divisor 方案给出 {2,3,6}，避免多尺度分支名存实亡。
            print(f'[BrainMDM] scale_scheme={scale_scheme} | '
                  f'seq 尺度={self.out_sizes_s} ({len(self.out_sizes_s)} 个) | '
                  f'win 尺度={self.out_sizes_w} ({len(self.out_sizes_w)} 个)')

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
        # 病理条件特征级调制（AdaLN）：None 时不生效，与无条件路径逐位一致
        self.cond_mod = (AdaLNConditioner(cond_dim=ada_cond_dim, num_channels=features)
                         if ada_cond_dim else None)

        # 多尺度合并：'none' 为旧的均匀平均；'sample' 为样本级 softmax 门控
        # （零初始化 → softmax(0) 均匀，与旧均匀平均在初始时刻数值一致）
        self.scale_gate_s = None
        self.scale_gate_w = None
        if scale_gate == 'sample':
            self.scale_gate_s = nn.Linear(seq_len, len(self.out_sizes_s))
            self.scale_gate_w = nn.Linear(seq_len, len(self.out_sizes_w))
            for lin in (self.scale_gate_s, self.scale_gate_w):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        # 门控温度：训练期可通过 set_scale_gate_temperature 退火（熵 warmup）
        self.scale_gate_temperature = 1.0
        self.last_scale_gate_entropy = 0.0

    @staticmethod
    def _scale_sizes(length: int, num_scales: int, scheme: str):
        """尺度池化目标长度集合。

        - ``pow2``：旧的 2 的幂次下采样 ``L // 2^(num_scales-i)``（小 L 时退化）；
        - ``divisor``：``L // i``（i=1..num_scales），W=6 时给出 {2,3,6}。
        """
        if scheme == 'pow2':
            sizes = [max(1, length // (2 ** (num_scales - i))) for i in range(num_scales)]
        else:
            sizes = [max(1, length // i) for i in range(1, max(1, num_scales) + 1)]
        return sorted(set(sizes))

    def set_scale_gate_temperature(self, temperature: float) -> None:
        self.scale_gate_temperature = float(max(1e-2, temperature))

    def _combine_scales(self, outs, gate_net, x):
        """多尺度分支合并：无门控时均匀平均，有门控时按样本级 softmax 加权。

        门控输入 ``pooled = x.mean(dim=(1,2))`` 宽度等于 S 轴长度：定长任务下
        恒等于 ``seq_len``；next_timepoint 任务的变长 context 下走前缀切片
        （``prefix_linear``），宽度相等时为恒等路径、数值不变。
        """
        if gate_net is None or len(outs) == 1:
            return sum(outs) / max(1, len(outs))
        pooled = x.mean(dim=(1, 2))                       # [B, S]
        logits = prefix_linear(gate_net, pooled) / self.scale_gate_temperature
        gate = F.softmax(logits, dim=-1)                  # [B, K]
        b = x.shape[0]
        agg = sum(gate[:, i].view(b, 1, 1, 1) * o for i, o in enumerate(outs))
        with torch.no_grad():
            ent = -(gate * gate.clamp_min(1e-8).log()).sum(-1).mean()
            # 保留 0 维张量而非 float(ent.item())：.item() 会在 torch.compile 下
            # 触发图断裂并引入一次设备同步；消费端（main.py 写 TensorBoard）再转 float。
            self.last_scale_gate_entropy = ent.detach()
        return agg

    def _pool_along_s(self, x: torch.Tensor) -> torch.Tensor:
        """时间轴（S）多尺度池化 + 回投影。

        池化目标长度由 ``seq_len``（= 建模的最大 S）在构造期确定，实际 S 轴长度
        ``s == seq_len`` 时逐位等价于旧实现；``s < seq_len``（next_timepoint 任务
        的变长 context）时对回投影结果做**前缀截断**（与输入侧前缀切片同一约定：
        第 i 个时间点恒对应权重第 i 行），池化本身对任意 s 都是良定义的。
        """
        b, f, w, s = x.shape
        x_s = x.reshape(b * f * w, s)
        outs = []
        for pool, linear in zip(self.pool_s, self.linear_s):
            pooled = pool(x_s.unsqueeze(1)).squeeze(1)
            out = prefix_linear_out(linear, pooled, s)   # [b*f*w, s]
            outs.append(out.reshape(b, f, w, s))
        return self._combine_scales(outs, self.scale_gate_s, x)

    def _pool_along_w(self, x: torch.Tensor) -> torch.Tensor:
        """窗口轴多尺度池化 + 回投影。

        池化目标长度由 ``num_window``（= 历史长度上限 L_max）在构造期确定，因此
        实际窗口数 ``w == num_window`` 时逐位等价于旧实现；``w < num_window``
        （轨迹任务的变长历史）时对回投影结果做**右对齐截断**（取末 w 个 token，
        保证最近窗口仍在序列末尾），池化本身对任意 w 都是良定义的。
        """
        b, f, w, s = x.shape
        x_w = x.permute(0, 1, 3, 2).reshape(b * f * s, w)
        outs = []
        for pool, linear in zip(self.pool_w, self.linear_w):
            pooled = pool(x_w.unsqueeze(1)).squeeze(1)
            out = linear(pooled)                       # [b*f*s, num_window]
            if w != self.num_window:
                if w > self.num_window:
                    raise ValueError(
                        f"窗口数 {w} 超过 BrainMDM 建模的 {self.num_window}："
                        "history_max 配置与数据不一致。")
                out = out[:, self.num_window - w:]
            outs.append(out.reshape(b, f, s, w).permute(0, 1, 3, 2))
        return self._combine_scales(outs, self.scale_gate_w, x)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
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
        if self.cond_mod is not None:
            fused = self.cond_mod(fused, cond)
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
        ada_cond_dim: int = None,
        lora_rank: int = 0,
        lora_alpha: float = None,
        sc_mask_mode: str = 'soft',
        sc_mask_tau: float = 1e-3,
        sc_bias_eps: float = 1e-6,
        use_window_attn: bool = True,
    ):
        super().__init__()
        self.use_window_attn = bool(use_window_attn)
        if sc_mask_mode not in ('hard', 'soft', 'none'):
            raise ValueError(
                f"sc_mask_mode 仅支持 hard/soft/none，收到 '{sc_mask_mode}'")
        self.sc_mask_mode = sc_mask_mode
        self.sc_mask_tau = float(sc_mask_tau)
        self.sc_bias_eps = float(sc_bias_eps)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        if self.use_window_attn and seq_len % window_heads != 0:
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

        # 窗口注意力分支：--ode_window_attn off 时不构建（与 head_use_win_attn
        # 构成去重 2×2 消融，避免同一信息在 ODE 与预测头中重复建模）
        self.window_norm = nn.LayerNorm(seq_len) if self.use_window_attn else None
        self.window_attn = (nn.MultiheadAttention(
            embed_dim=seq_len,
            num_heads=window_heads,
            dropout=dropout,
            batch_first=True,
        ) if self.use_window_attn else None)

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

        # 病理条件特征级调制（AdaLN）：None 时不生效
        self.cond_mod = (AdaLNConditioner(cond_dim=ada_cond_dim, num_channels=features)
                         if ada_cond_dim else None)

        # LoRA 低秩增量：仅作用于 q/k/v/out/FFN 投影，基础权重键名与形状不变。
        # lora_B 零初始化 → 初始增量恒为 0（identity-at-init）。
        self._lora = None
        if lora_rank and lora_rank > 0:
            self._lora = nn.ModuleDict({
                'q':    LowRankDelta(seq_len, hidden_dim, rank=lora_rank, alpha=lora_alpha),
                'k':    LowRankDelta(seq_len, hidden_dim, rank=lora_rank, alpha=lora_alpha),
                'v':    LowRankDelta(seq_len, hidden_dim, rank=lora_rank, alpha=lora_alpha),
                'out':  LowRankDelta(hidden_dim, seq_len, rank=lora_rank, alpha=lora_alpha),
                'ffn_in':  LowRankDelta(seq_len, hidden_dim, rank=lora_rank, alpha=lora_alpha),
                'ffn_out': LowRankDelta(hidden_dim, seq_len, rank=lora_rank, alpha=lora_alpha),
            })

    def _proj(self, name: str, base: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """基础线性投影 + （可选的）LoRA 低秩增量。

        输入最后一维小于 ``base.in_features`` 时（变长 context）走前缀切片
        （``prefix_linear`` / ``LowRankDelta`` 内部同理），宽度相等时数值不变。
        """
        out = prefix_linear(base, x)
        if self._lora is not None and name in self._lora:
            out = out + self._lora[name](x)
        return out

    def _prepare_sc(self, sc_matrix: torch.Tensor, h: torch.Tensor,
                    adj_eff: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (mask, bias)，形状 [B*W, F, F]。

        - ``sc_mask_mode='hard'`` 或未提供 ``adj_eff``：逐字保留旧路径
          （SC>0 硬掩码 + log1p 加性先验），保证 ``--sc_prior_mode scaled
          --sc_mask_mode hard`` 与引入软先验前数值一致；
        - ``'soft'``：用 ``A_eff > tau`` 取代硬掩码（软图几乎处处为正，
          因此不再永久切断 DTI 假阴性边），加性先验改为 ``log(A_eff + eps)``；
        - ``'none'``：不施加任何图先验，退化为自由 ROI 注意力。
        """
        b, n, w, _ = h.shape

        if self.sc_mask_mode == 'hard' or adj_eff is None:
            sc_sym, _ = prepare_sc_matrix(sc_matrix, h)
            eye = torch.eye(n, device=h.device, dtype=h.dtype).unsqueeze(0)
            sc_with_self = (sc_sym + eye).clamp_min(0.0)
            mask = sc_with_self > 0
            bias = torch.log1p(sc_with_self)
        elif self.sc_mask_mode == 'none':
            mask = torch.ones(b, n, n, dtype=torch.bool, device=h.device)
            bias = torch.zeros(b, n, n, dtype=h.dtype, device=h.device)
        else:
            a = adj_eff.to(dtype=h.dtype)
            if a.ndim == 2:
                a = a.unsqueeze(0)
            mask = a > self.sc_mask_tau
            bias = torch.log(a.clamp_min(self.sc_bias_eps))

        mask = mask.unsqueeze(1).expand(b, w, n, n).reshape(b * w, n, n)
        bias = bias.unsqueeze(1).expand(b, w, n, n).reshape(b * w, n, n)
        return mask, bias

    def _graph_attention(self, h: torch.Tensor, sc_matrix: torch.Tensor,
                         adj_eff: torch.Tensor = None) -> torch.Tensor:
        b, n, w, s = h.shape
        tokens = h.transpose(1, 2).reshape(b * w, n, s)

        q = self._proj('q', self.q_proj, tokens).reshape(
            b * w, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self._proj('k', self.k_proj, tokens).reshape(
            b * w, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self._proj('v', self.v_proj, tokens).reshape(
            b * w, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask, bias = self._prepare_sc(sc_matrix, h, adj_eff=adj_eff)
        attn_scores = attn_scores + self.sc_bias_scale * bias.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), -1e4)

        attn = F.softmax(attn_scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(b * w, n, self.hidden_dim)
        # out_proj / 对应 LoRA 增量均按 seq_len 建模：变长 context 下取输出前缀
        # （与输入侧前缀切片同一约定），宽度相等时数值不变。
        projected = prefix_linear_out(self.out_proj, out, s)
        if self._lora is not None and 'out' in self._lora:
            projected = projected + self._lora['out'](out)[..., :s]
        return projected.reshape(b, w, n, s).transpose(1, 2)

    def _window_attention(self, h: torch.Tensor) -> torch.Tensor:
        if self.window_attn is None:
            return torch.zeros_like(h)
        b, n, w, s = h.shape
        if s != self.seq_len:
            # nn.MultiheadAttention(embed_dim=seq_len) 的 token 特征维固定，
            # 变长 context 无法直接进入该分支；next_timepoint 任务的 W=1，
            # 该分支本身退化为逐 token 线性映射，调用方应关闭 --ode_window_attn。
            raise ValueError(
                f"GraphODE 窗口注意力要求 S 轴长度 == seq_len({self.seq_len})，"
                f"收到 s={s}；变长 context（next_timepoint）请设置 --ode_window_attn off。")
        tokens = h.reshape(b * n, w, s)
        tokens_norm = self.window_norm(tokens)
        out, _ = self.window_attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        return out.reshape(b, n, w, s)

    def forward(self, t, h: torch.Tensor, sc_matrix: torch.Tensor,
                cond: torch.Tensor = None,
                adj_eff: torch.Tensor = None) -> torch.Tensor:
        del t
        if h.ndim != 4:
            raise ValueError(f"Expected h [B, F, W, S], got {tuple(h.shape)}")

        s = int(h.shape[-1])
        # 变长 context（s < seq_len）时全部按 S 轴建模的 LayerNorm/Linear 走前缀切片；
        # s == seq_len（旧任务 / 轨迹任务）时逐位等价于原实现。
        h_norm = prefix_layernorm(self.input_norm, h)
        dh_space = self._graph_attention(h_norm, sc_matrix, adj_eff=adj_eff)
        dh_time = self.temporal_branch(h_norm)
        dh_window = self._window_attention(h_norm)
        # FFN 手动展开，以便在两层 Linear 上挂载 LoRA 增量（键名 ffn.0~ffn.4 不变）
        ffn_in = prefix_layernorm(self.ffn[0], h_norm)
        ffn_in = self._proj('ffn_in', self.ffn[1], ffn_in)
        ffn_in = self.ffn[2](ffn_in)
        ffn_in = self.ffn[3](ffn_in)
        dh_ffn = self._proj('ffn_out', self.ffn[4], ffn_in)[..., :s]

        dh_dt = (
            torch.sigmoid(self.attn_gate) * dh_space +
            torch.sigmoid(self.temporal_gate) * dh_time +
            torch.sigmoid(self.window_gate) * dh_window +
            torch.sigmoid(self.ffn_gate) * dh_ffn
        )
        if self.cond_mod is not None:
            dh_dt = self.cond_mod(dh_dt, cond)
        return dh_dt


class GraphODEDDI(nn.Module):
    """连续时间积分器：Euler / RK2（Heun）/ RK4 / adaptive / SDE 包装。

    相对旧实现的三点修正（见 docs/Version1_docs_0921 1.3）：

    1. **stochastic depth 修正**：旧代码先算完 ``k1/k2`` 再 ``continue``，跳过
       的步仍然付出了导数调用代价，且用 ``delta / survival`` 放大剩余步以补偿
       期望（等价于把噪声注入动力学，破坏确定性）。现在把判定**移到导数计算之前**，
       跳过即真的省下 1~2 次前向，并去掉 ``/survival``。
    2. **solver / 步长可选**：``--ode_solver {euler,rk2,rk4,adaptive,sde}``，
       ``--ode_step_mode {fixed,learnable}``。
    3. **审计导出**：``self.last_ode_diag`` 记录 ``||k1||,||k2||,||Δh||`` 与
       导数调用次数，由 NeuroTwin 汇总进 ``aux_info['ode_diag']``。范数保留为
       detached 的 0 维张量（不在此处 ``.item()``），由消费端（main.py 写
       TensorBoard 时）统一转 float：``.item()`` 会在 ``torch.compile`` 下触发
       图断裂，并对每个 ODE 步引入一次设备同步。

    ``adaptive`` 需要 ``torchdiffeq``（守卫式导入 + 可操作报错，故不作默认）；
    ``sde`` 在 RK2 基础上叠加 Langevin 噪声项，**默认关闭**（随机动力学破坏
    确定性早停，且缺少 Phase 7 的校准目标）。
    """

    SOLVERS = ('euler', 'rk2', 'rk4', 'adaptive', 'sde')
    STEP_MODES = ('fixed', 'learnable')

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
        ada_cond_dim: int = None,
        lora_rank: int = 0,
        lora_alpha: float = None,
        sc_mask_mode: str = 'soft',
        sc_mask_tau: float = 1e-3,
        use_window_attn: bool = True,
        solver: str = 'rk2',
        step_mode: str = 'learnable',
        sde_noise_scale: float = 1e-3,
        adaptive_rtol: float = 1e-3,
        adaptive_atol: float = 1e-4,
    ):
        super().__init__()
        if solver not in self.SOLVERS:
            raise ValueError(f"ode_solver 仅支持 {self.SOLVERS}，收到 '{solver}'")
        if step_mode not in self.STEP_MODES:
            raise ValueError(f"ode_step_mode 仅支持 {self.STEP_MODES}，收到 '{step_mode}'")
        if solver == 'adaptive':
            # 守卫式导入：环境未安装 torchdiffeq 时给出可操作的报错
            try:
                import torchdiffeq  # noqa: F401
            except ImportError as exc:  # pragma: no cover - 依赖缺失分支
                raise ImportError(
                    "--ode_solver adaptive 需要 torchdiffeq，请先安装（pip install torchdiffeq），"
                    "或改用 rk2/rk4/euler。") from exc

        self.ode_func = GraphODE(
            features=features,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            window_heads=window_heads,
            ada_cond_dim=ada_cond_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sc_mask_mode=sc_mask_mode,
            sc_mask_tau=sc_mask_tau,
            use_window_attn=use_window_attn,
        )
        step_scale = float(max(step_scale, 1e-4))
        # step_scale_raw 保留（learnable 模式使用）；fixed 模式直接使用常数，
        # 由于 softplus(log(expm1(s))) == s，两种模式在初始时刻数值完全一致
        self.step_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(step_scale)), dtype=torch.float32))
        self.step_scale_fixed = step_scale
        self.step_mode = step_mode
        self.ode_steps = ode_steps
        self.stochastic_depth_rate = stochastic_depth_rate
        self.solver = solver
        self.sde_noise_scale = float(sde_noise_scale)
        self.adaptive_rtol = float(adaptive_rtol)
        self.adaptive_atol = float(adaptive_atol)
        self.last_ode_diag = None

    @property
    def step_scale(self) -> torch.Tensor:
        if self.step_mode == 'learnable':
            return F.softplus(self.step_scale_raw)
        return torch.as_tensor(self.step_scale_fixed, dtype=self.step_scale_raw.dtype,
                               device=self.step_scale_raw.device)

    def _derivative(self, h, sc_matrix, cond, adj_eff):
        return self.ode_func(0, h, sc_matrix, cond, adj_eff=adj_eff)

    def _integrate_adaptive(self, h, sc_matrix, cond, adj_eff):
        """dopri5 自适应步长积分（t: 0 → step_scale，等价于总步长 step_scale）。"""
        from torchdiffeq import odeint

        counter = {'n': 0}

        def f(t, y):
            counter['n'] += 1
            return self._derivative(y, sc_matrix, cond, adj_eff)

        t = torch.tensor([0.0, float(self.step_scale.detach())], device=h.device, dtype=h.dtype)
        out = odeint(f, h, t, rtol=self.adaptive_rtol, atol=self.adaptive_atol, method='dopri5')
        return out[-1], counter['n']

    def forward(self, x: torch.Tensor, sc_matrix: torch.Tensor,
                cond: torch.Tensor = None,
                adj_eff: torch.Tensor = None) -> torch.Tensor:
        h = x
        calls = 0
        k1_norms, k2_norms, d_norms = [], [], []

        if self.solver == 'adaptive':
            h, calls = self._integrate_adaptive(h, sc_matrix, cond, adj_eff)
            with torch.no_grad():
                self.last_ode_diag = {'calls': float(calls)}
            return h

        dt = 1.0 / max(1, self.ode_steps)
        step_scale = self.step_scale
        for _ in range(self.ode_steps):
            # stochastic depth：必须在导数计算之前判定，否则跳过毫无收益
            if self.training and self.stochastic_depth_rate > 0:
                if torch.rand(1, device=h.device).item() < self.stochastic_depth_rate:
                    continue

            if self.solver == 'euler':
                k1 = self._derivative(h, sc_matrix, cond, adj_eff)
                delta = step_scale * dt * k1
                calls += 1
                with torch.no_grad():
                    k1_norms.append(k1.detach().norm())
                    k2_norms.append(0.0)
            else:
                k1 = self._derivative(h, sc_matrix, cond, adj_eff)
                if self.solver == 'rk4':
                    k2 = self._derivative(h + 0.5 * dt * k1, sc_matrix, cond, adj_eff)
                    k3 = self._derivative(h + 0.5 * dt * k2, sc_matrix, cond, adj_eff)
                    k4 = self._derivative(h + dt * k3, sc_matrix, cond, adj_eff)
                    delta = step_scale * dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                    calls += 4
                else:
                    # rk2 / sde 共用 Heun 的两级结构
                    k2 = self._derivative(h + dt * k1, sc_matrix, cond, adj_eff)
                    delta = step_scale * dt * 0.5 * (k1 + k2)
                    calls += 2
                if self.solver == 'sde':
                    delta = delta + self.sde_noise_scale * math.sqrt(dt) * torch.randn_like(h)
                with torch.no_grad():
                    k1_norms.append(k1.detach().norm())
                    k2_norms.append(k2.detach().norm())

            h = h + delta
            with torch.no_grad():
                d_norms.append(delta.detach().norm())

        with torch.no_grad():
            self.last_ode_diag = {
                'k1_norm': sum(k1_norms) / max(1, len(k1_norms)),
                'k2_norm': sum(k2_norms) / max(1, len(k2_norms)),
                'delta_norm': sum(d_norms) / max(1, len(d_norms)),
                'calls': float(calls),
            }
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

    def forward(
        self,
        pred: torch.Tensor,
        sc_matrix: torch.Tensor,
        adj: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:      [B, F, W, S] 待细化的预测张量
            sc_matrix: [F, F] 或 [B, F, F] 结构连接矩阵
            adj:       可选的 [B, F, F] 归一化邻接矩阵；显式传入时直接复用，
                       避免多处重复计算 SC 归一化（供 SC 先验共享路径使用）。
        """
        if pred.ndim != 4:
            raise ValueError(f"Expected pred [B, F, W, S], got {tuple(pred.shape)}")

        b, n, w, s = pred.shape
        if adj is None:
            _, adj_norm = prepare_sc_matrix(sc_matrix, pred)
        else:
            adj_norm = adj
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
    历史上权重由 `sigmoid(log(0.5**i / (1 - 0.5**i)))` 生成，但 i=0 时
    `log(1 / 1e-4) ≈ 9.21`、`sigmoid(9.21) ≈ 0.9999`，与注释声称的
    “首轮 0.5” 严重不符（首轮权重实际≈1.0）。现改为显式初始化表
    `init_scales`，默认 [0.50, 0.33, 0.20, 0.12]，>4 轮时按 0.6 比例
    继续几何衰减。scales 仍是可学习参数，允许模型自适应调整各轮权重。

    Args:
        features:    ROI 通道数（与 PredictionRefiner 相同）
        n_rounds:    细化轮数，默认 3（通常 2~3 轮即可收敛）
        dropout:     各轮 refiner 的 dropout 率
        init_scales: 各轮权重初值；None 时使用默认表
        adaptive:    是否启用 per-round 自适应门控（输入相关的 |δ| 统计调制，
                     zero-init 时门控恒为 1，与 adaptive=False 数值一致）
    """

    #: 默认每轮权重初值（显式表，取代原 sigmoid(log(...)) 的隐式生成）
    ROUND_INIT_SCALES = {
        1: (0.50,),
        2: (0.50, 0.33),
        3: (0.50, 0.33, 0.20),
        4: (0.50, 0.33, 0.20, 0.12),
    }

    def __init__(
        self,
        features: int,
        n_rounds: int = 3,
        dropout: float = 0.1,
        init_scales=None,
        adaptive: bool = False,
    ):
        super().__init__()
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        self.n_rounds = int(n_rounds)
        self.adaptive = bool(adaptive)
        self.eps = 1e-8

        self.refiners = nn.ModuleList([
            PredictionRefiner(features=features, dropout=dropout)
            for _ in range(self.n_rounds)
        ])

        scales = tuple(init_scales) if init_scales is not None else self._default_init_scales(self.n_rounds)
        if len(scales) != self.n_rounds:
            raise ValueError(
                f"init_scales 长度 {len(scales)} 与 n_rounds {self.n_rounds} 不一致")
        self.round_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(self._scale_to_logit(s), dtype=torch.float32))
            for s in scales
        ])

        if self.adaptive:
            # 每轮一个灵敏度：0 初始化 → tanh(0)=0 → 门控恒为 1（identity-at-init）
            self.adaptive_sensitivity = nn.Parameter(torch.zeros(self.n_rounds))

    @classmethod
    def _default_init_scales(cls, n_rounds: int):
        table = cls.ROUND_INIT_SCALES
        if n_rounds in table:
            return table[n_rounds]
        base = list(table[4])
        while len(base) < n_rounds:
            base.append(max(1e-3, base[-1] * 0.6))
        return tuple(base)

    @staticmethod
    def _scale_to_logit(scale: float) -> float:
        s = float(min(max(scale, 1e-3), 1.0 - 1e-3))
        return math.log(s / (1.0 - s))

    def initial_round_scales(self):
        """当前各轮权重的实际取值（sigmoid 后），供日志/断言使用。"""
        with torch.no_grad():
            return [float(torch.sigmoid(p).item()) for p in self.round_scales]

    def _run(self, pred: torch.Tensor, sc_matrix: torch.Tensor, adj: torch.Tensor = None):
        """返回 (累计残差, 各轮结束时的中间预测列表，最后一个即最终预测)。"""
        accumulated = torch.zeros_like(pred)
        current = pred
        intermediates = []
        ref_scale = None
        for r, (refiner, scale_param) in enumerate(zip(self.refiners, self.round_scales)):
            # 每轮 refiner 基于当前累计修正后的 pred 计算新的残差
            delta = refiner(current, sc_matrix, adj=adj)
            weight = torch.sigmoid(scale_param)
            if self.adaptive:
                mag = delta.detach().abs().mean().clamp_min(self.eps)
                if ref_scale is None:
                    ref_scale = mag
                ratio = (mag / ref_scale).clamp(0.0, 1e6)
                weight = weight * (1.0 + torch.tanh(self.adaptive_sensitivity[r] * ratio))
            weighted_delta = weight * delta
            accumulated = accumulated + weighted_delta
            # 更新 current：下一轮基于修正后的预测，实现迭代精化
            current = current + weighted_delta
            intermediates.append(current)
        return accumulated, intermediates

    def forward(
        self,
        pred: torch.Tensor,
        sc_matrix: torch.Tensor,
        adj: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:      [B, F, W, S] 待细化的预测张量
            sc_matrix: [F, F] 或 [B, F, F] 结构连接矩阵
            adj:       可选的 [B, F, F] 归一化邻接矩阵

        Returns:
            accumulated delta [B, F, W, S]，与 PredictionRefiner 接口一致。
            调用方执行 pred = pred + refiner(pred, sc_matrix)。
        """
        accumulated, _ = self._run(pred, sc_matrix, adj=adj)
        return accumulated

    def forward_with_rounds(
        self,
        pred: torch.Tensor,
        sc_matrix: torch.Tensor,
        adj: torch.Tensor = None,
    ):
        """额外返回轮间中间预测（不含最终预测），供轮间监督使用。

        Returns:
            (accumulated delta, [round_0_pred, ..., round_{R-2}_pred])
            最终预测（第 R-1 轮）已由主损失覆盖，故不在此列表中返回。
        """
        accumulated, intermediates = self._run(pred, sc_matrix, adj=adj)
        return accumulated, intermediates[:-1]