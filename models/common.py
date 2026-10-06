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
# Variable history length helpers
# -------------------------------------------------

# 变长前向统一约定：
#
# 时间轴 S（一个窗口内的 TR 序号；next_timepoint 任务下整段 context 就是单个
# 窗口，故 S == context 长度 K，K <= context_max）：所有按 S_max = context_max
# 建模的线性/LayerNorm 走**前缀切片**——context 的第 i 个时间点恒对应权重切片
# 中的第 i 个位置（输入侧取权重列前缀，输出侧取权重行前缀）；K == S_max 时与
# 定长实现逐位等价。
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


def prefix_layernorm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """按实际宽度做 LayerNorm，仿射参数取前缀（语义同 :func:`prefix_linear`）。"""
    d = int(x.shape[-1])
    if d == norm.normalized_shape[0]:
        return norm(x)
    if d > norm.normalized_shape[0]:
        raise ValueError(
            f"输入宽度 {d} 超过 LayerNorm 建模宽度 {norm.normalized_shape[0]}。")
    return F.layer_norm(x, (d,), norm.weight[:d], norm.bias[:d], norm.eps)


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

        用于把未来真值 chunk 用“历史统计量”归一化后再进模型，
        保证归一化统计量只来自历史，未来 target 不参与任何统计量估计
        （无 normalization leakage）。
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
