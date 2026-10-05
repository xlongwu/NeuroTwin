# coding=utf-8
"""NeuroTwinTFM：TimesFM-3 风格的 SC-guided 多变量脑动力学模型（方案 §16 / §30 V1）。

整体架构（方案 §16）::

    BOLD context [B,F,K]
      → Context-only RevIN（统计量只由历史 context 估计，§15）
      → A_eff = SoftAnatomicalPrior（一次计算、全层共享，§7）
      → Temporal Patching（p ∈ {1,4,8}，§4）
      → SC-guided Temporal–Variate Block × N（§6.4）
      ├── One-Step Head（§10）：z_t → z_(t+1) → x̂_(t+1)
      │     （承担状态转移 / assimilation / intervention 接口，§9）
      └── CPM Horizon Head（§8/§11）：ROI × Horizon 查询交叉注意力，
            一次非自回归输出 x̂_(t+1:t+H)

V1 明确不做（方案 §30）：quantile 头、intervention tokens 的正式训练、
assimilation、新 MoE、SDE、更复杂的 graph learning。仅预留
future-known covariate（intervention lookahead，§12.4/§13）的**接口**
（``cpm_intervention=True`` 时构建嵌入，默认关闭、不参与训练）。

对外接口与 :class:`models.neurotwin.NeuroTwin` 的 next_timepoint 协议对齐
（``forward`` / ``predict_next_state`` / ``rollout_next_states`` 同形），
使训练入口与评估链路可以按 ``--model_arch`` 直接分派。
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import BrainRevIN
from models.graph_prior import SoftAnatomicalPrior
from models.pathology import PathologyNormalizer
from models.tfm_backbone import (
    RMSNorm,
    TFMEncoderBlock,
    TemporalPatchEmbed,
    build_context_grid,
)


class ConditionFiLM(nn.Module):
    """病理条件 FiLM 调制（§12.2 / §16：Patient Conditions → FiLM）。

    条件向量 ``c ∈ R^C`` 经线性层映射为逐通道 (scale, shift)，以
    ``h ← h·(1+scale) + shift`` 的形式调制特征（广播到 ROI / patch 轴）。
    权重零初始化 → 初始为恒等调制，保证「无条件 → 有条件」切换不破坏
    预训练主干的数值行为（与 LoRA / 零初始化 ΔA 的稳定起步策略一致）。

    ``is_conditioning_adapter`` 标记供 train/optim.py 的微调冻结逻辑识别：
    冻结主干阶段 FiLM 仍参与训练（它们是“让病理条件真正生效”的唯一路径）。
    """

    is_conditioning_adapter = True

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        if cond_dim <= 0:
            raise ValueError(f"cond_dim 必须 > 0，收到 {cond_dim}")
        self.cond_dim = int(cond_dim)
        self.dim = int(dim)
        self.proj = nn.Linear(self.cond_dim, 2 * self.dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor):
        """cond [B, C] → (scale [B, D], shift [B, D])。"""
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"条件向量期望 [B, {self.cond_dim}]，收到 {tuple(cond.shape)}")
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return scale, shift


class OneStepDynamicsHead(nn.Module):
    """One-Step 状态转移头（§10）：``z_(t+1) = F_θ(z_t, A_eff, c)``。

    - ``z_t``：backbone 最后一个 causal token（逐 ROI）``[B, F, D]``；
    - SC 引导的图混合：``z_g = A_eff @ z_t``（跨 ROI 交互被限制在结构先验
      邻域内，与 ROI attention 的软偏置不同，这里显式做一次图卷积式传播）；
    - 转移 MLP 输入 ``[z_t, z_g]``，输出零初始化的残差 Δz（恒等转移起步，
      数值稳定，与预测端 ``x̂ = x_t + Δ̂`` 的 persistence 锚点一致）；
    - ``c`` 非空时在转移隐层上做 FiLM 调制（Task E 的最小病理条件通路）；
    - 解码 ``x̂_(t+1) = x_t + Δ̂``（归一化空间），Δ̂ 由零初始化线性层给出。
    """

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1,
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        in_dim = 2 * self.dim
        self.transition = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.dim))
        # 零初始化残差：初始转移为恒等（z_{t+1} = z_t）
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)
        self.decode = nn.Linear(self.dim, 1)
        # 零初始化解码：初始 x̂ = x_t（persistence 锚点），模型学习偏离量
        nn.init.zeros_(self.decode.weight)
        nn.init.zeros_(self.decode.bias)
        self.film = ConditionFiLM(cond_dim, self.dim) if cond_dim else None

    def forward(self, z_t: torch.Tensor, adj: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_t:  [B, F, D] 最后 causal token
            adj:  [B, F, F] A_eff（结构先验）
            cond: [B, C] 或 None（HC 预训练 / 未提供评分）
        Returns:
            Δx̂ [B, F]（归一化空间，相对 x_t 的变化量）
        """
        z_g = torch.bmm(adj.to(dtype=z_t.dtype), z_t)          # [B,F,D]
        h = self.transition(torch.cat([z_t, z_g], dim=-1))     # [B,F,D]
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        z_next = z_t + h
        return self.decode(z_next).squeeze(-1)                 # [B,F]


class CPMHorizonHead(nn.Module):
    """CPM 全 horizon 预测头（§8 / §11）：非自回归一次输出 ``x̂_(t+1:t+H)``。

    未来查询（§11）::

        q_{i,h} = e^ROI_i + e^horizon_h [+ E_u(U_{i,h})]

    对 backbone 上下文 token ``H_context ∈ R^{B×F×P×D}``（全部为历史信息，
    无需 causal mask）做多层多头交叉注意力后，逐 (ROI, horizon) 线性解码。
    ROI 嵌入与 backbone patch embedding 共享（同一套 e^ROI），horizon 嵌入
    为本头私有参数。``cpm_intervention=True`` 时构建 future-known covariate
    嵌入 E_u（§12.4/§13 接口预留，V1 默认关闭、不参与训练）。
    """

    def __init__(self, dim: int, horizon: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1,
                 cond_dim: Optional[int] = None,
                 build_intervention: bool = False):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"cpm_horizon 必须 >= 1，收到 {horizon}")
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.horizon_embed = nn.Parameter(torch.zeros(self.horizon, self.dim))
        nn.init.trunc_normal_(self.horizon_embed, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.layers.append(nn.ModuleDict({
                'norm_q': RMSNorm(self.dim),
                'norm_m': RMSNorm(self.dim),
                'attn': _CrossAttention(self.dim, num_heads, dropout),
                'norm_ffn': RMSNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, 4 * self.dim), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(4 * self.dim, self.dim)),
                'drop': nn.Dropout(dropout),
            }))
        self.out_proj = nn.Linear(self.dim, 1)
        self.film = ConditionFiLM(cond_dim, self.dim) if cond_dim else None
        # future-known covariate（intervention lookahead）接口：V1 仅预留，
        # 默认不构建；U 逐 (ROI, horizon) 标量 → 向量嵌入加到查询上
        self.intervention_embed = nn.Linear(1, self.dim) if build_intervention else None

    def forward(self, context: torch.Tensor, roi_embed: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                future_control: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context: [B, F, P, D] backbone 上下文 token
            roi_embed: [F, D] ROI 嵌入（与 patch embedding 共享）
            cond: [B, C] 或 None
            future_control: [B, F, H] 未来已知控制量（仅接口预留时提供）
        Returns:
            x̂ [B, F, H]（归一化空间的未来轨迹预测）
        """
        b, f, p, d = context.shape
        h_out = self.horizon
        # 查询 [B, F, H, D] = e^ROI + e^horizon (+ E_u)
        q = roi_embed.view(1, f, 1, d) + self.horizon_embed.view(1, 1, h_out, d)
        queries = q.expand(b, f, h_out, d)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)
            queries = queries * (1.0 + scale.view(b, 1, 1, d)) \
                + shift.view(b, 1, 1, d)
        if self.intervention_embed is not None:
            if future_control is None:
                future_control = torch.zeros(
                    b, f, h_out, device=context.device, dtype=context.dtype)
            if tuple(future_control.shape) != (b, f, h_out):
                raise ValueError(
                    f"future_control 期望 [B,{f},{h_out}]，"
                    f"收到 {tuple(future_control.shape)}")
            queries = queries + self.intervention_embed(
                future_control.unsqueeze(-1))

        # 交叉注意力：查询 [B, F*H, D] ← 记忆 [B, F*P, D]（SDPA，无需 mask）
        q_flat = queries.reshape(b, f * h_out, d)
        m_flat = context.reshape(b, f * p, d)
        for layer in self.layers:
            attn_out = layer['attn'](layer['norm_q'](q_flat),
                                     layer['norm_m'](m_flat))
            q_flat = q_flat + layer['drop'](attn_out)
            q_flat = q_flat + layer['drop'](layer['ffn'](layer['norm_ffn'](q_flat)))
        out = self.out_proj(q_flat)                            # [B, F*H, 1]
        return out.view(b, f, h_out)


class _CrossAttention(nn.Module):
    """多头交叉注意力（q 来自查询集，k/v 来自记忆集；SDPA 实现，无 mask）。"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) 必须能被 num_heads({num_heads}) 整除")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.kv_proj = nn.Linear(self.dim, self.dim * 2)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = float(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """q [B, Nq, D]，memory [B, Nm, D] → [B, Nq, D]。"""
        b, nq, _ = q.shape
        qh = self.q_proj(q).view(b, nq, self.num_heads, self.head_dim
                                 ).transpose(1, 2)             # [B,h,Nq,d]
        kh, vh = self.kv_proj(memory).chunk(2, dim=-1)
        kh = kh.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vh = vh.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            qh, kh, vh, dropout_p=(self.dropout if self.training else 0.0))
        out = out.transpose(1, 2).reshape(b, nq, self.dim)
        return self.out_proj(out)


class NeuroTwinTFM(nn.Module):
    """NeuroTwin-TFM V1 主模型（方案 §16）。

    与 :class:`models.neurotwin.NeuroTwin` 的差异（方案 §27 映射）：
    - 删除 window 轴结构（BrainMDM window pathway / ODE window attention）；
    - BrainMDM temporal pathway / GraphODE → Causal Temporal Attention；
    - DFCAdapter / GraphODE graph branch / 多处 SC 注入 → 统一 SoftAnatomicalPrior
      的 A_eff，仅在 ROI attention 做结构偏置 + one-step 头做一次图混合；
    - FutureQueryDecoder 思想 → ROI × Horizon 查询的 CPM 头；
    - 新增 CPM 全 horizon 预测目标（One-Step + CPM 双预测头，§9）。

    任务口径约束：one-step 头只建模 +1 偏移（§10），因此要求
    ``forecast_offsets == (1,)``；稀疏偏移 {1,2,4,8} 的监督/评估由 CPM 头的
    horizon 覆盖（§18 Task C 降级为 CPM 的子集）。
    """

    def __init__(
        self,
        features: int = 116,
        context_max: int = 64,
        patch_len: int = 4,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_ratio: int = 4,
        dropout: float = 0.1,
        # ---- SC 软先验（§6.3 三部分图信息，复用 SoftAnatomicalPrior）----
        sc_prior_mode: str = 'soft_prior',
        sc_lambda_mode: str = 'global',
        sc_lambda_init: float = 0.7,
        sc_prior_rank: int = 12,
        sc_delta_a: bool = True,
        sc_delta_rank: int = None,
        sc_sinkhorn_iters: int = 64,
        # ---- One-Step 头（§10）----
        one_step_hidden_dim: int = 256,
        # ---- CPM 头（§8/§11）----
        cpm_horizon: int = 8,
        cpm_layers: int = 2,
        cpm_intervention: bool = False,
        # ---- 条件化（§12.2；pretrain 模式无条件模块）----
        pretrain_mode: bool = True,
        pathology_input_dim: int = 1,
        pathology_norm_mode: str = 'robust_z',
        pathology_norm_quantiles: int = 64,
        pathology_norm_rbf_knots: int = 8,
        # ---- 条件消融（§25.4）----
        use_patho_cond: bool = True,
        # ---- 归一化（§15）----
        use_revin: bool = True,
    ):
        super().__init__()
        if patch_len < 1 or context_max < patch_len:
            raise ValueError(
                f"patch_len({patch_len}) 与 context_max({context_max}) 不兼容")
        if cpm_horizon < 1:
            raise ValueError(f"cpm_horizon 必须 >= 1，收到 {cpm_horizon}")
        self.features = int(features)
        self.context_max = int(context_max)
        self.patch_len = int(patch_len)
        self.dim = int(dim)
        self.cpm_horizon = int(cpm_horizon)
        self.pretrain_mode = bool(pretrain_mode)
        self.use_revin = bool(use_revin)
        self.sc_prior_mode = str(sc_prior_mode)

        # ---- Context-only RevIN（§15：μ/σ 只由历史 context 估计）----
        self.rev_norm = BrainRevIN(num_features=features) if self.use_revin else None

        # ---- SC 软解剖先验：A_eff 一次计算、全层共享（§7）----
        self.sc_prior = SoftAnatomicalPrior(
            features=features, seq_len=context_max,
            mode=sc_prior_mode, rank=sc_prior_rank,
            lambda_mode=sc_lambda_mode, lambda_init=sc_lambda_init,
            delta_a=sc_delta_a, delta_rank=sc_delta_rank,
            prob_mask=False, sinkhorn_iters=sc_sinkhorn_iters,
            return_sequence=False,
        )

        # ---- 病理条件（仅 finetune；HC 预训练无条件模块）----
        # use_patho_cond=False 为方案 §25.4 条件消融开关：不构建 PathologyNormalizer
        # 与 FiLM，前向收到的 pathology_score 一律忽略（与条件开启的对照实验
        # 共用同一 HC 预训练权重，隔离「条件通路」本身的贡献）
        self.pathology_normalizer = None
        cond_dim = 0
        if not self.pretrain_mode and use_patho_cond:
            self.pathology_normalizer = PathologyNormalizer(
                input_dim=pathology_input_dim,
                mode=pathology_norm_mode,
                n_quantiles=pathology_norm_quantiles,
                rbf_knots=pathology_norm_rbf_knots,
            )
            cond_dim = self.pathology_normalizer.out_dim
        self.cond_dim = cond_dim

        # ---- Backbone：temporal patching + SC-guided Temporal–Variate blocks ----
        self.patch_embed = TemporalPatchEmbed(
            features=features, context_max=context_max, patch_len=patch_len,
            dim=dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            TFMEncoderBlock(dim=dim, num_heads=num_heads, ff_ratio=ff_ratio,
                            dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = RMSNorm(dim)

        # ---- 双预测头（§9 Dual Dynamics Objective）----
        self.one_step_head = OneStepDynamicsHead(
            dim=dim, hidden_dim=one_step_hidden_dim, dropout=dropout,
            cond_dim=cond_dim or None)
        self.cpm_head = CPMHorizonHead(
            dim=dim, horizon=cpm_horizon, num_heads=num_heads,
            num_layers=cpm_layers, dropout=dropout,
            cond_dim=cond_dim or None,
            build_intervention=cpm_intervention)

    # ------------------------------------------------------------------
    # 内部构件
    # ------------------------------------------------------------------
    def _prepare_condition(self, pathology_score):
        """原始临床评分 → 模型内条件向量；预训练 / 未提供时返回 None。"""
        if self.pathology_normalizer is None or pathology_score is None:
            return None
        return self.pathology_normalizer(pathology_score)

    def _encode(self, x_grid: torch.Tensor, sc_matrix: torch.Tensor):
        """[B,F,K]（已归一化）→ (context tokens [B,F,P,D], sc_info)。

        A_eff 由 SoftAnatomicalPrior 从归一化 context 与 SC 导出，backbone
        全层与两个预测头共享（§7：SC → Soft Anatomical Prior → A_eff →
        所有 ROI Attention）。
        """
        x4 = x_grid.unsqueeze(2)                               # [B,F,1,K]
        sc_info = self.sc_prior(x4, sc_matrix)
        tokens = self.patch_embed(x_grid)
        for block in self.blocks:
            tokens = block(tokens, sc_info['A_eff'])
        return self.final_norm(tokens), sc_info

    def _denorm_3d(self, pred_norm: torch.Tensor, stats) -> torch.Tensor:
        """[B,F,H]（归一化）→ 原始空间（复用 BrainRevIN 的仿射反变换）。"""
        if self.rev_norm is None:
            return pred_norm
        out = self.rev_norm(pred_norm.unsqueeze(-1), 'denorm', stats=stats)
        return out.squeeze(-1)

    # ------------------------------------------------------------------
    # 前向（与 NeuroTwin 的 next_timepoint 协议同形：[B,F,1,K] → [B,F,1,1]）
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, sc_matrix: torch.Tensor,
                pathology_score: Optional[torch.Tensor] = None,
                future_control: Optional[torch.Tensor] = None):
        """单次前向：one-step 主输出 + CPM 全 horizon 辅助输出。

        Args:
            x: [B, F, 1, K] context（内部布局与 NeuroTwin 一致，K ≤ context_max）
            sc_matrix: [F, F] 或 [B, F, F]
            pathology_score: [B, D] 或 None（HC 预训练 / 未提供评分）
            future_control: [B, F, cpm_horizon] 未来已知控制量（仅
                ``cpm_intervention=True`` 时生效；V1 默认 None）
        Returns:
            pred: [B, F, 1, 1] one-step 下一状态预测（原始空间，与 NeuroTwin
                  next_timepoint 输出同形，供既有评估/损失直接消费）
            aux_info: dict：
                ``cpm_pred``  [B, F, H] CPM 全 horizon 预测（原始空间）
                ``sc_prior``  SoftAnatomicalPrior 输出（图正则消费）
                ``pathology_cond`` 条件向量（finetune；诊断用）
        """
        if x.ndim != 4 or x.shape[2] != 1:
            raise ValueError(
                f"NeuroTwinTFM.forward 期望 [B,F,1,K]，收到 {tuple(x.shape)}")
        x_grid, _ = build_context_grid(x)                      # [B,F,K]
        norm_stats = None
        if self.rev_norm is not None:
            x_norm4, norm_stats = self.rev_norm(x, 'norm')     # context-only 统计（§15）
            x_grid = x_norm4[:, :, 0, :]
        cond = self._prepare_condition(pathology_score)
        if pathology_score is not None and self.pretrain_mode:
            raise ValueError(
                "收到 pathology_score 但当前为预训练模式（无条件模块）；"
                "HC 预训练不应提供病理评分。")
        # use_patho_cond=False 的条件消融：评分到达但被忽略（cond=None）

        tokens, sc_info = self._encode(x_grid, sc_matrix)
        # One-Step 头（§10）：最后 causal token 为 z_t
        z_t = tokens[:, :, -1, :]                              # [B,F,D]
        x_last_norm = x_grid[:, :, -1]                         # [B,F] = x_t（归一化）
        delta_norm = self.one_step_head(z_t, sc_info['A_eff'], cond)
        one_pred_norm = (x_last_norm + delta_norm).unsqueeze(-1).unsqueeze(-1)

        # CPM 头（§11）：ROI × Horizon 查询，一次输出整段未来
        cpm_pred_norm = self.cpm_head(
            tokens, self.patch_embed.roi_embed, cond, future_control)

        aux_info = {'sc_prior': sc_info}
        if cond is not None:
            aux_info['pathology_cond'] = cond
        if self.rev_norm is not None:
            one_pred = self.rev_norm(one_pred_norm, 'denorm', stats=norm_stats)
            cpm_pred = self._denorm_3d(cpm_pred_norm, norm_stats)
        else:
            one_pred, cpm_pred = one_pred_norm, cpm_pred_norm
        aux_info['cpm_pred'] = cpm_pred                        # [B,F,H] 原始空间
        return one_pred, aux_info

    # ------------------------------------------------------------------
    # next_timepoint 对外标准接口（与 NeuroTwin 同形）
    # ------------------------------------------------------------------
    @staticmethod
    def _to_internal_history(bold_history: torch.Tensor) -> torch.Tensor:
        """标准输入 [B, K, F] → 内部布局 [B, F, 1, K]。"""
        if bold_history.ndim != 3:
            raise ValueError(
                f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}")
        return bold_history.transpose(1, 2).unsqueeze(2).contiguous()

    def predict_next_state(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                           pathology_score: Optional[torch.Tensor] = None) -> dict:
        """下一步全脑状态预测（one-step 头）。

        Returns:
            dict：``pred_next`` [B,F,1]（原始空间）、``pred_delta`` [B,F,1]、
            ``pred`` [B,F,1,1]、``aux_info``（含 ``cpm_pred`` [B,F,H]）。
        """
        x = self._to_internal_history(bold_history)
        pred, aux_info = self.forward(x, sc_matrix, pathology_score)
        pred_next = pred[:, :, :, 0]                           # [B,F,1]
        x_last = bold_history[:, -1, :].unsqueeze(-1)          # [B,F,1] = x_t
        return {'pred_next': pred_next, 'pred_delta': pred_next - x_last,
                'pred': pred, 'aux_info': aux_info}

    def forecast_horizon(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                         pathology_score: Optional[torch.Tensor] = None,
                         future_control: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CPM 非自回归全 horizon 预测（§8：一次 forward 输出 t+1..t+H）。

        与 :meth:`rollout_next_states` 的自回归轨迹对比可量化 error
        accumulation（§25.3）。返回 [B, H, F]（原始空间）。
        """
        x = self._to_internal_history(bold_history)
        _, aux_info = self.forward(x, sc_matrix, pathology_score,
                                   future_control=future_control)
        return aux_info['cpm_pred'].transpose(1, 2)            # [B,H,F]

    def rollout_next_states(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                            pathology_score: Optional[torch.Tensor] = None,
                            steps: int = 1) -> torch.Tensor:
        """free rollout：one-step 头自回归滚动 ``steps`` 步（中间不用真值）。

        与 NeuroTwin.rollout_next_states 同语义：每步取 +1 偏移预测接到
        context 末尾、滑出最早 1 个 TR，返回 [B, steps, F]。
        """
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
            nxt = out['pred_next'][:, :, 0]                    # [B,F]（+1 偏移）
            preds.append(nxt)
            if k + 1 >= n_steps:
                break
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)                       # [B, steps, F]
