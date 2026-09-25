# coding=utf-8
"""SC 软解剖先验（Soft Anatomical Prior）。

把“静态 SC 硬掩码 + 永久切断非 SC 边”替换为随输入自适应的软图先验：

    A_eff = λ · A_SC + (1 - λ) · A_func(h)

- ``A_SC``：由 :func:`models.common.prepare_sc_matrix` 的度归一化给出（直接复用，
  不重复实现），是静态结构先验；
- ``A_func(h) = row_softmax(sym(U Vᵀ) / √rank)``：由当前输入特征导出的功能图，
  使 DTI 假阴性边在数据支持下可以“复活”；
- ``λ``：全局 / 样本级 / ROI 级可学习混合系数（``--sc_lambda_mode``）；
- 可选低秩 subject-specific 邻接残差 ΔA（``--sc_delta_a``，零初始化 → 初始无影响）；
- 可选可学习概率掩码（``--sc_prob_mask``，straight-through；初始化保留约 95% 边）；
- 最后用**对称 Sinkhorn** 迭代把矩阵投影为双随机矩阵，使 ``A_eff`` 同时满足
  “行归一”与“对称”（单纯的度归一化只能满足其一）。

``functional_only`` 模式跳过 Sinkhorn，返回 row_softmax 语义的纯功能注意力图。

窗口维说明：``A_func`` 由节点特征在窗口维池化后导出，因此 ``A_eff`` 为
``[B, F, F]`` 的样本级矩阵，可在 ODE 与 refiner 之间“一次计算、多处共享”。
当需要时间一致性正则（``--sc_temporal_weight > 0``）时，额外导出逐窗功能图
``A_seq``（``[B, W, F, F]``），此时池化后的 ``A_func`` 取逐窗结果的均值以保持一致。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import prepare_sc_matrix


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
        mode:        soft_prior / adaptive_only / functional_only
        rank:        功能图低秩分解维度
        lambda_mode: global / sample / roi（仅 soft_prior 生效）
        lambda_init: λ 初值（默认 0.7，偏向信任 SC）
        delta_a:     是否启用低秩 subject-specific 邻接残差 ΔA
        delta_rank:  ΔA 的秩，None 时取 max(4, rank // 2)
        prob_mask:   是否启用可学习概率掩码（straight-through）
        sinkhorn_iters: 对称 Sinkhorn 迭代次数
        self_loop_eps:  Sinkhorn 前附加的对角项，保证无零行（数值安全）
        return_sequence: 是否额外导出逐窗功能图 A_seq（供时间一致性正则）
    """

    PRIOR_MODES = ('soft_prior', 'adaptive_only', 'functional_only')
    LAMBDA_MODES = ('global', 'sample', 'roi')

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
        prob_mask: bool = False,
        sinkhorn_iters: int = 64,
        self_loop_eps: float = 1e-4,
        return_sequence: bool = False,
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
        self.return_sequence = bool(return_sequence)
        self.eps = float(eps)

        # ── 功能图低秩分解 U Vᵀ ─────────────────────────────────────
        self.node_norm = nn.LayerNorm(self.seq_len)
        self.u_proj = nn.Linear(self.seq_len, self.rank)
        self.v_proj = nn.Linear(self.seq_len, self.rank)
        for lin in (self.u_proj, self.v_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        # ── λ 混合系数（仅 soft_prior 需要；其余模式 SC 不参与）────────
        self.lambda_logit = None
        self.lambda_proj = None
        if mode == 'soft_prior':
            if lambda_mode == 'global':
                self.lambda_logit = nn.Parameter(
                    torch.tensor(prob_to_logit(self.lambda_init), dtype=torch.float32))
            else:
                # sample: 作用于池化特征 [B, S] → [B, 1]
                # roi:    作用于逐 ROI 特征 [B, F, S] → [B, F, 1]
                self.lambda_proj = nn.Linear(self.seq_len, 1)
                nn.init.zeros_(self.lambda_proj.weight)
                nn.init.constant_(self.lambda_proj.bias, prob_to_logit(self.lambda_init))

        # ── 低秩邻接残差 ΔA（零初始化 → 初始恒为 0）────────────────────
        self.delta_scale = None
        self.delta_rank = 0
        if delta_a:
            self.delta_rank = int(delta_rank) if delta_rank else max(4, self.rank // 2)
            self.dp = nn.Linear(self.seq_len, self.delta_rank)
            self.dq = nn.Linear(self.seq_len, self.delta_rank)
            for lin in (self.dp, self.dq):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.delta_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # ── 可学习概率掩码（straight-through，初始化保留约 95% 边）──────
        self.prob_mask_logits = None
        if prob_mask:
            self.prob_mask_logits = nn.Parameter(
                torch.full((self.features, self.features),
                           prob_to_logit(0.95), dtype=torch.float32))

    # ------------------------------------------------------------------
    # 内部构件
    # ------------------------------------------------------------------
    def _functional_graph(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes [*, F, S] → 功能图 A_func [*, F, F]。

        打分先对称化 ``sym(U Vᵀ)``，再取行 softmax；由于 row softmax 本身
        不保对称（A_ij 与 A_ji 独立归一化），最后再做一次
        ``0.5 (S + Sᵀ)`` 对称化，使 A_func 严格对称 -> Sinkhorn 之后
        A_eff 才能同时满足“行归一”与“对称”两个要求。
        """
        z = self.node_norm(nodes)
        u = self.u_proj(z)
        v = self.v_proj(z)
        scores = torch.matmul(u, v.transpose(-1, -2)) / math.sqrt(self.rank)
        scores = 0.5 * (scores + scores.transpose(-1, -2))
        p = F.softmax(scores, dim=-1)
        return 0.5 * (p + p.transpose(-1, -2))

    def _mix_lambda(self, nodes: torch.Tensor) -> torch.Tensor:
        """返回 λ（形状可广播到 [B, F, F]），非 soft_prior 模式返回 None。"""
        if self.mode != 'soft_prior':
            return None
        if self.lambda_mode == 'global':
            return torch.sigmoid(self.lambda_logit).view(1, 1, 1)
        if self.lambda_mode == 'sample':
            pooled = nodes.mean(dim=1)                       # [B, S]
            return torch.sigmoid(self.lambda_proj(pooled)).view(-1, 1, 1)
        return torch.sigmoid(self.lambda_proj(nodes))        # [B, F, 1]

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
            dict：A_eff [B,F,F]、A_func [B,F,F]、mode；soft_prior 下附 lambda；
                  return_sequence=True 时附 A_seq [B,W,F,F]。
        """
        if h.ndim != 4:
            raise ValueError(f"SoftAnatomicalPrior 期望 h [B,F,W,S]，收到 {tuple(h.shape)}")
        b, n, w, s = h.shape
        nodes = h.mean(dim=2)                       # [B, F, S] 窗口池化

        if self.return_sequence:
            nodes_seq = h.transpose(1, 2).reshape(b * w, n, s)   # [B*W, F, S]
            a_seq = self._functional_graph(nodes_seq).reshape(b, w, n, n)
            a_func = a_seq.mean(dim=1)
        else:
            a_seq = None
            a_func = self._functional_graph(nodes)

        lam = self._mix_lambda(nodes)

        if self.mode == 'soft_prior':
            if adj_norm is None:
                _, adj_norm = prepare_sc_matrix(sc_matrix, h)
            a = lam * adj_norm.to(dtype=h.dtype) + (1.0 - lam) * a_func
        else:
            # adaptive_only / functional_only：SC 不参与，纯数据驱动图
            a = a_func

        if self.delta_scale is not None:
            z = self.node_norm(nodes)
            da = torch.matmul(self.dp(z), self.dq(z).transpose(-1, -2)) / math.sqrt(self.delta_rank)
            da = 0.5 * (da + da.transpose(-1, -2))
            # ΔA 可正可负，裁剪保证非负（Sinkhorn 要求非负）
            a = (a + F.softplus(self.delta_scale) * da).clamp_min(0.0)

        if self.prob_mask_logits is not None:
            p = torch.sigmoid(self.prob_mask_logits)
            p = 0.5 * (p + p.transpose(-1, -2))
            hard = (p > 0.5).to(a.dtype)
            mask_st = hard + p - p.detach()          # straight-through
            a = a * mask_st.unsqueeze(0)

        # functional_only 是“no-SC 参考图”：保持 row-softmax 的原始尺度不做
        # 双随机归一化，用于对照“归一化本身”带来的差异；其余模式统一投影为
        # 对称双随机矩阵（行归一且对称）。
        if self.mode != 'functional_only':
            eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
            a = symmetric_sinkhorn(a + self.self_loop_eps * eye,
                                   iters=self.sinkhorn_iters, eps=self.eps)

        info = {'A_eff': a, 'A_func': a_func, 'mode': self.mode}
        if lam is not None:
            info['lambda'] = lam
        if a_seq is not None:
            info['A_seq'] = a_seq
        return info
