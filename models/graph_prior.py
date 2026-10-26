# coding=utf-8
"""历史功能图与学习功能图的软先验。

保留当前训练结构：全局可学习混合系数、低秩邻接残差、自环和对称 Sinkhorn。
输入图在模型中由可见历史 FC 构造，不代表独立实测的解剖连接。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import prefix_layernorm, prefix_linear, prepare_sc_matrix


def prob_to_logit(p: float, eps: float = 1e-4) -> float:
    """概率 → logit（带边界裁剪，避免 ±inf）。"""
    q = float(min(max(p, eps), 1.0 - eps))
    return math.log(q / (1.0 - q))


@torch.compiler.disable
def symmetric_sinkhorn(a: torch.Tensor, iters: int = 64,
                       eps: float = 1e-6) -> torch.Tensor:
    """对称 Sinkhorn 归一化：迭代缩放到双随机矩阵，且每一步都严格保持对称。

    对对称非负矩阵 A，取 r = 1 / √(A·1) 并做 A ← diag(r) A diag(r)，
    不动点即行和（也是列和）为 1 的双随机矩阵，同时保持对称性。

    固定迭代 ``iters`` 次，不做收敛早停：早停需要 ``(row - 1.0).abs().max().item()``
    取 host 标量，会在 ``torch.compile`` 下产生图断裂，并对每个迭代步引入一次设备同步。
    去掉早停后结果更接近投影不动点（实测与早停版差异 < 1e-6，行和偏差反而更小）。

    函数整体用 ``torch.compiler.disable`` 排除在编译之外：F×F 上的逐元素缩放本就
    是微秒级，融合收益接近零，而把 64 次迭代展开进计算图会让首次编译耗时从数十秒
    膨胀到十分钟量级。排除后 dynamo 直接跳过（不产生 graph break 警告），
    代价是这里成为一次显式的编译边界。
    """
    a = a.clamp_min(0.0)
    for _ in range(max(1, int(iters))):
        row = a.sum(dim=-1, keepdim=True)
        r = row.clamp_min(eps).pow(-0.5)
        a = a * r * r.transpose(-1, -2)
    return a


class SoftAnatomicalPrior(nn.Module):
    """SC 软解剖先验：输出样本级软邻接矩阵 ``A_eff`` [B, F, F]。

    Args:
        features:    ROI 数 F
        seq_len:     每个窗口的时间长度 S（节点特征维度）
        mode:        固定为 soft_prior
        rank:        功能图低秩分解维度
        lambda_mode: 固定为 global
        lambda_init: λ 初值（默认 0.7，偏向信任 SC）
        delta_a:     是否启用低秩 subject-specific 邻接残差 ΔA
        delta_rank:  ΔA 的秩，None 时取 max(4, rank // 2)
        sinkhorn_iters: 对称 Sinkhorn 迭代次数
        self_loop_eps:  Sinkhorn 前附加的对角项，保证无零行（数值安全）
    """

    PRIOR_MODES = ('soft_prior',)
    LAMBDA_MODES = ('global',)

    def __init__(
        self,
        features: int,
        seq_len: int,
        mode: str = 'soft_prior',
        rank: int = 12,
        lambda_mode: str = 'global',
        lambda_init: float = 0.7,
        delta_a: bool = True,
        delta_rank: int = None,
        sinkhorn_iters: int = 64,
        self_loop_eps: float = 1e-4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in self.PRIOR_MODES:
            raise ValueError(
                f"SoftAnatomicalPrior mode 仅支持 {self.PRIOR_MODES}，收到 '{mode}'")
        if lambda_mode not in self.LAMBDA_MODES:
            raise ValueError(
                f"lambda_mode 仅支持 {self.LAMBDA_MODES}，收到 '{lambda_mode}'")

        self.features = int(features)
        self.seq_len = int(seq_len)
        self.mode = mode
        self.rank = max(2, int(rank))
        self.lambda_mode = lambda_mode
        self.lambda_init = float(min(max(lambda_init, 1e-3), 1.0 - 1e-3))
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.self_loop_eps = float(self_loop_eps)
        self.eps = float(eps)

        # ── 功能图低秩分解 U Vᵀ ─────────────────────────────────────
        self.node_norm = nn.LayerNorm(self.seq_len)
        self.u_proj = nn.Linear(self.seq_len, self.rank)
        self.v_proj = nn.Linear(self.seq_len, self.rank)
        for lin in (self.u_proj, self.v_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        # ── λ 混合系数（仅 soft_prior 需要；其余模式 SC 不参与）────────
        self.lambda_logit = nn.Parameter(
            torch.tensor(prob_to_logit(self.lambda_init), dtype=torch.float32))

        # ── 低秩邻接残差 ΔA（dp Xavier、dq 零 → 初始输出恒为 0）────────
        # 方案 §4.2：dp/dq 同时零初始化时乘积残差 ΔA = dp(dq(z))·scale 无法启动
        # 学习（乘积梯度恒零）；dp Xavier 非零让 dq 首步获得梯度，而 dq 零保证
        # 初始输出仍为零（残差起点不变）。state_dict 键不变，只影响新实例初始化，
        # 不重置从 checkpoint 加载的实际权重。
        self.delta_scale = None
        self.delta_rank = 0
        if delta_a:
            self.delta_rank = int(delta_rank) if delta_rank else max(4, self.rank // 2)
            self.dp = nn.Linear(self.seq_len, self.delta_rank)
            self.dq = nn.Linear(self.seq_len, self.delta_rank)
            nn.init.xavier_uniform_(self.dp.weight)
            nn.init.zeros_(self.dp.bias)
            nn.init.zeros_(self.dq.weight)
            nn.init.zeros_(self.dq.bias)
            self.delta_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 内部构件
    # ------------------------------------------------------------------
    def _functional_graph(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes [*, F, S] → 功能图 A_func [*, F, F]。

        打分先对称化 ``sym(U Vᵀ)``，再取行 softmax；由于 row softmax 本身
        不保对称（A_ij 与 A_ji 独立归一化），最后再做一次
        ``0.5 (S + Sᵀ)`` 对称化，使 A_func 严格对称 -> Sinkhorn 之后
        A_eff 才能同时满足“行归一”与“对称”两个要求。

        S 轴长度可变（next_timepoint 的 context 长度 K <= context_max）：节点特征
        投影全部走前缀切片（``prefix_layernorm`` / ``prefix_linear``），
        S == seq_len 时逐位等价于原实现。
        """
        z = prefix_layernorm(self.node_norm, nodes)
        u = prefix_linear(self.u_proj, z)
        v = prefix_linear(self.v_proj, z)
        scores = torch.matmul(u, v.transpose(-1, -2)) / math.sqrt(self.rank)
        scores = 0.5 * (scores + scores.transpose(-1, -2))
        p = F.softmax(scores, dim=-1)
        return 0.5 * (p + p.transpose(-1, -2))

    def _mix_lambda(self, nodes: torch.Tensor) -> torch.Tensor:
        """返回可广播到 [B,F,F] 的全局混合系数。"""
        return torch.sigmoid(self.lambda_logit).view(1, 1, 1)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor, sc_matrix: torch.Tensor,
                adj_norm: torch.Tensor = None) -> dict:
        """由输入特征 h [B, F, W, S] 生成软邻接矩阵。

        Args:
            h:          [B, F, W, S] 当前输入特征（已做实例归一化）
            sc_matrix:  [F, F] 或 [B, F, F] 结构连接矩阵
            adj_norm:   可选的 SC 度归一化邻接（由 prepare_sc_matrix 产出）；
                        为 None 时在本模块内部复用 prepare_sc_matrix 计算。

        Returns:
            dict：A_eff [B,F,F]、A_func [B,F,F]、mode 与全局 lambda。
        """
        if h.ndim != 4:
            raise ValueError(f"SoftAnatomicalPrior 期望 h [B,F,W,S]，收到 {tuple(h.shape)}")
        b, n, w, s = h.shape
        nodes = h.mean(dim=2)                       # [B, F, S] 窗口池化

        a_func = self._functional_graph(nodes)
        lam = self._mix_lambda(nodes)
        if adj_norm is None:
            _, adj_norm = prepare_sc_matrix(sc_matrix, h)
        a = lam * adj_norm.to(dtype=h.dtype) + (1.0 - lam) * a_func

        if self.delta_scale is not None:
            z = prefix_layernorm(self.node_norm, nodes)
            da = torch.matmul(prefix_linear(self.dp, z),
                              prefix_linear(self.dq, z).transpose(-1, -2)) \
                / math.sqrt(self.delta_rank)
            da = 0.5 * (da + da.transpose(-1, -2))
            # ΔA 可正可负，裁剪保证非负（Sinkhorn 要求非负）
            a = (a + F.softplus(self.delta_scale) * da).clamp_min(0.0)

        eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
        a = symmetric_sinkhorn(a + self.self_loop_eps * eye,
                               iters=self.sinkhorn_iters, eps=self.eps)
        return {'A_eff': a, 'A_func': a_func, 'mode': self.mode, 'lambda': lam}
