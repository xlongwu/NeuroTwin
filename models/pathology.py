# coding=utf-8
"""病理条件化组件。

`PathologyNormalizer` 把原始临床评分（默认 HAMD 总分）映射为模型内部使用的
条件向量。统计量以 buffer 形式持有并随 checkpoint 一起保存/加载，保证训练、
评估与分析脚本使用完全一致的变换。

buffer 中 `fitted` 使用 int8：`ModelEMA` 对非浮点 buffer 走 copy 而非平均，
从而不会把“是否已拟合”的标志平均成无意义的小数。
"""
import math
from typing import Union

import torch
import torch.nn as nn


class PathologyNormalizer(nn.Module):
    """临床评分归一化模块（模型内实现，统计量由训练集一次性拟合）。

    模式与输出宽度（D = `input_dim`，K = `rbf_knots`）：

    ==================  =============  ==================================================
    mode                out_dim        说明
    ==================  =============  ==================================================
    identity            D              恒等，不做任何归一化
    robust_z            D              (x - median) / (IQR / 1.349)，IQR 退化时回退 std
    zscore_quadratic    2D             稳健 z 后拼接平方项 [z, z^2]
    empirical_cdf       D              经验 CDF：P(X <= x) ∈ [0, 1]
    quantile            D              分位数变换：对经验 CDF 做 probit，近似标准正态
    zscore_rbf          D * (1 + K)    稳健 z 后拼接 K 个固定网格上的 RBF 特征
    ==================  =============  ==================================================

    系数含义：`empirical_cdf` 与 `quantile` 共用同一张经验分位表（`quantiles`），
    区别在于 `quantile` 额外做 probit 变换把取值域拉回近似标准正态，更适合
    作为神经网络的连续条件输入。
    """

    SUPPORTED_MODES = (
        'identity', 'robust_z', 'zscore_quadratic',
        'empirical_cdf', 'quantile', 'zscore_rbf',
    )
    #: 仅这些模式可无损反演回原始评分空间
    INVERTIBLE_MODES = ('identity', 'robust_z', 'zscore_quadratic')

    def __init__(
        self,
        input_dim: int = 1,
        mode: str = 'robust_z',
        n_quantiles: int = 64,
        rbf_knots: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}', expected one of {self.SUPPORTED_MODES}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
        if rbf_knots < 1:
            raise ValueError(f"rbf_knots must be >= 1, got {rbf_knots}")

        self.input_dim = int(input_dim)
        self.mode = mode
        self.n_quantiles = int(n_quantiles)
        self.rbf_knots = int(rbf_knots)
        self.eps = float(eps)

        # 统计量（未拟合时的初值保证 forward 可跑且为恒等/零均值单位方差）
        self.register_buffer('center', torch.zeros(self.input_dim))
        self.register_buffer('scale', torch.ones(self.input_dim))
        # 经验分位表：沿最后一维严格递增，供 searchsorted 使用
        self.register_buffer(
            'quantiles',
            torch.linspace(0.0, 1.0, self.n_quantiles, dtype=torch.float32)
            .unsqueeze(0).repeat(self.input_dim, 1).contiguous(),
        )
        # 固定 z 网格上的 RBF 中心与宽度
        if self.rbf_knots > 1:
            knots = torch.linspace(-2.0, 2.0, self.rbf_knots, dtype=torch.float32)
            width = float((4.0 / (self.rbf_knots - 1)) * 0.8)
        else:
            knots = torch.zeros(1, dtype=torch.float32)
            width = 1.0
        self.register_buffer(
            'knot_centers', knots.unsqueeze(0).repeat(self.input_dim, 1).contiguous())
        self.register_buffer('knot_width', torch.full((self.input_dim, 1), width))
        self.register_buffer('fitted', torch.zeros(1, dtype=torch.int8))

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def out_dim(self) -> int:
        """归一化后条件向量的宽度，供下游条件模块使用。"""
        if self.mode == 'zscore_quadratic':
            return self.input_dim * 2
        if self.mode == 'zscore_rbf':
            return self.input_dim * (1 + self.rbf_knots)
        return self.input_dim

    def is_fitted(self) -> bool:
        return bool(int(self.fitted.item()) > 0)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, mode={self.mode}, "
                f"out_dim={self.out_dim}, fitted={self.is_fitted()}")

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, raw: Union[torch.Tensor, 'object']) -> 'PathologyNormalizer':
        """用训练集（仅 train subjects）的原始评分拟合统计量。

        Args:
            raw: 原始评分，任意前导形状，最后一维须等于 `input_dim`；
                 可为 numpy 数组或 list。
        """
        x = raw if torch.is_tensor(raw) else torch.as_tensor(raw)
        x = x.detach().to(dtype=torch.float32).reshape(-1, self.input_dim)
        if x.numel() == 0:
            raise ValueError("PathologyNormalizer.fit 收到空输入，无法拟合统计量。")

        center = x.median(dim=0).values
        iqr = torch.quantile(x, 0.75, dim=0) - torch.quantile(x, 0.25, dim=0)
        scale = iqr / 1.349
        std = x.std(dim=0, unbiased=False)
        # IQR 退化（大量重复值）时回退到标准差，再退化则回退到 1
        scale = torch.where(scale > self.eps, scale, std)
        scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))

        self.center.copy_(center)
        self.scale.copy_(scale)

        if self.mode in ('empirical_cdf', 'quantile'):
            probs = torch.linspace(0.0, 1.0, self.n_quantiles, dtype=torch.float32)
            qs = torch.quantile(x, probs, dim=0)          # [Q, D]
            self.quantiles.copy_(qs.transpose(0, 1).contiguous())

        self.fitted.fill_(1)
        return self

    # ------------------------------------------------------------------
    # 前向 / 反演
    # ------------------------------------------------------------------
    def _interp_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """线性插值估计经验 CDF，返回 [B, D]，取值 [0, 1]。"""
        b, d = x.shape
        q = self.quantiles.unsqueeze(0).expand(b, d, self.n_quantiles)   # [B, D, Q]
        idx = torch.searchsorted(q, x.unsqueeze(-1), right=False)        # [B, D, 1]
        idx = idx.clamp(1, self.n_quantiles - 1)
        lo = q.gather(-1, idx - 1)
        hi = q.gather(-1, idx)
        p_lo = (idx - 1).to(x.dtype) / float(self.n_quantiles - 1)
        p_hi = idx.to(x.dtype) / float(self.n_quantiles - 1)
        denom = (hi - lo).clamp_min(self.eps)
        p = p_lo + (x.unsqueeze(-1) - lo) / denom * (p_hi - p_lo)
        return p.squeeze(-1).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] -> [B, out_dim]"""
        if self.mode == 'identity':
            return x
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        x = x.reshape(-1, self.input_dim)

        z = (x - self.center) / self.scale
        if self.mode == 'robust_z':
            return z
        if self.mode == 'zscore_quadratic':
            return torch.cat([z, z * z], dim=-1)
        if self.mode == 'empirical_cdf':
            return self._interp_cdf(x)
        if self.mode == 'quantile':
            p = self._interp_cdf(x).clamp(self.eps, 1.0 - self.eps)
            return torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)
        if self.mode == 'zscore_rbf':
            diff = (z.unsqueeze(-1) - self.knot_centers.unsqueeze(0)) / self.knot_width.unsqueeze(0)
            rbf = torch.exp(-0.5 * diff * diff)                # [B, D, K]
            return torch.cat([z, rbf.reshape(x.shape[0], -1)], dim=-1)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """把归一化后的条件向量映射回原始评分空间（仅 INVERTIBLE_MODES 支持）。"""
        if self.mode not in self.INVERTIBLE_MODES:
            raise NotImplementedError(
                f"mode='{self.mode}' 不可逆，无法从条件空间恢复原始评分。")
        if self.mode == 'identity':
            return z
        zz = z[..., :self.input_dim]
        return zz * self.scale + self.center
