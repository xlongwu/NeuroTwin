# coding=utf-8
"""NeuroTwin 的历史编码、个体化 Graph-ODE 与 CPM 模型。

历史 BOLD 经 Context-only RevIN、时间 patching 和图引导编码器，
由 Graph-ODE 推进下一状态与连续轨迹，CPM 提供直接多步辅助预测。
个体 profile 来自扫描前缀，历史记忆在状态推进期间保持固定；方差头
读取 detached latent 轨迹。预训练与微调分别使用健康人口学和临床字段条件。
"""
from contextlib import contextmanager, nullcontext
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import BrainRevIN
from models.conditioning import (ARMS, ConditioningConfig,
                                 build_condition_encoder)
from models.graph_prior import SoftAnatomicalPrior
from models.tfm_backbone import (
    RMSNorm,
    TFMEncoderBlock,
    TemporalPatchEmbed,
    build_context_grid,
)
from models.tfm_dynamics import GraphODEDynamicsHead
from models.tfm_personal import ProfileEncoder


def _pool_history_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """编码 token 在 patch 维取均值并转 fp32（历史记忆）。

    用 ``torch._dynamo.disable`` 强制图外执行：该归约会产生基于 view 的符号，
    触发 ``torch.compile(dynamic=True)`` 的形状守卫断言；在编译区外算好再作为
    普通输入张量传入，既稳定又保留正确数值。
    """
    return tokens.mean(dim=2).float()


if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'disable'):
    _pool_history_tokens = torch._dynamo.disable(_pool_history_tokens)


@contextmanager
def isolated_global_rng():
    """块内构建的模块**不消耗**外部全局 RNG（H0/H1 非条件参数逐位可比）。

    条件模块的默认初始化（``nn.Linear`` / ``nn.Embedding`` / ``trunc_normal_``）会
    推进全局 RNG 流；不隔离时，同一 seed 下 ``arm=none`` 与 ``arm=cat`` 的
    backbone / 双头非条件参数会逐位不同，验收「相同 seed 下 HC H0 无条件与 H1
    条件的非条件参数逐位相同」就不成立（臂间差异被初始噪声污染）。

    只包裹条件构件的构造调用点，不改动任何非条件模块的初始化语义（宽度/层数等
    按项目策略已冻结）；条件模块自身仍按进入块时的 RNG 状态正常随机初始化。
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _rng_scope(isolate: bool):
    """条件构件的 RNG 作用域开关。

    ``isolate=False`` 用于未注入条件时，保持
    其历史初始化顺序逐位不变；条件注入实验一律 ``True``，使条件模块的构造对
    后续非条件模块「不可见」。
    """
    return isolated_global_rng() if isolate else nullcontext()


class ConditionFiLM(nn.Module):
    """病理条件 FiLM 调制（§12.2 / §16：Patient Conditions → FiLM）。

    条件向量 ``c ∈ R^C`` 经线性层映射为逐通道 (scale, shift)，以
    ``h ← h·(1+scale) + shift`` 的形式调制特征（广播到 ROI / patch 轴）。
    权重零初始化 → 初始为恒等调制，保证「无条件 → 有条件」切换不破坏
    预训练主干的数值行为（与 LoRA / 零初始化 ΔA 的稳定起步策略一致）。

    ``lam`` 是**固定的**注入强度 :math:`\\lambda_\\ell`（默认 1）：
    :math:`h'=(1+\\lambda\\gamma(c))\\odot h+\\lambda\\beta(c)`。因为
    :math:`\\lambda` 可被可学习的 :math:`W_\\ell` 吸收，它**不是**「临床特征贡献
    比例」，也不能解释为稳定的物理配比（方案 §11 末）；位置比较使用开关而非
    连续强度，非零强度默认 1。

    ``is_conditioning_adapter`` 标记供 train/optim.py 的微调冻结逻辑识别：
    冻结主干阶段 FiLM 仍参与训练（它们是“让病理条件真正生效”的唯一路径）。
    """

    is_conditioning_adapter = True

    def __init__(self, cond_dim: int, dim: int, lam: float = 1.0):
        super().__init__()
        if cond_dim <= 0:
            raise ValueError(f"cond_dim 必须 > 0，收到 {cond_dim}")
        self.cond_dim = int(cond_dim)
        self.dim = int(dim)
        self.lam = float(lam)
        self.proj = nn.Linear(self.cond_dim, 2 * self.dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor):
        """cond [B, C] → (scale [B, D], shift [B, D])，已按 ``lam`` 缩放。"""
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"条件向量期望 [B, {self.cond_dim}]，收到 {tuple(cond.shape)}")
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        if self.lam != 1.0:
            scale, shift = self.lam * scale, self.lam * shift
        return scale, shift




class CPMHorizonHead(nn.Module):
    """CPM 全 horizon 预测头（§8 / §11）：非自回归一次输出 ``x̂_(t+1:t+H)``。

    未来查询（§11）::

        q_{i,h} = e^ROI_i + e^horizon_h [+ E_u(U_{i,h})]

    对 backbone 上下文 token ``H_context ∈ R^{B×F×P×D}``（全部为历史信息，
    无需 causal mask）做多层多头交叉注意力后，逐 (ROI, horizon) 线性解码。
    ROI 嵌入与 backbone patch embedding 共享（同一套 e^ROI），horizon 嵌入
    为本头私有参数；控制输入由 Graph-ODE 处理。

    ``cond_rng_isolated=True``（条件注入实验）时，本头 FiLM 的构造不消耗全局
    RNG，保证同 seed 下 ``arm=none`` 与本臂的非条件参数逐位相同。
    """

    def __init__(self, dim: int, horizon: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1,
                 cond_dim: Optional[int] = None,
                 cond_lambda: float = 1.0,
                 cond_rng_isolated: bool = False):
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
        # 条件 FiLM 的构造不推进全局 RNG（条件注入实验下 H0/H1 非条件参数逐位一致）
        with _rng_scope(cond_rng_isolated):
            self.film = (ConditionFiLM(cond_dim, self.dim, lam=cond_lambda)
                         if cond_dim else None)
    def forward(self, context: torch.Tensor, roi_embed: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context: [B, F, P, D] backbone 上下文 token
            roi_embed: [F, D] ROI 嵌入（与 patch embedding 共享）
            cond: [B, C] 或 None
        Returns:
            x̂ [B, F, H]（归一化空间的未来轨迹预测）
        """
        b, f, p, d = context.shape
        h_out = self.horizon
        # 查询 [B, F, H, D] = e^ROI + e^horizon
        q = roi_embed.view(1, f, 1, d) + self.horizon_embed.view(1, 1, h_out, d)
        queries = q.expand(b, f, h_out, d)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)
            queries = queries * (1.0 + scale.view(b, 1, 1, d)) \
                + shift.view(b, 1, 1, d)
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
    """历史编码器、功能图先验、Graph-ODE、CPM 与个体 profile。

    下一状态输出使用 ODE 轨迹首帧；CPM 输出辅助预测序列。
    condition_config 与训练期统计量定义字段编码，实际数据通路由权重快照重建。
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
        tfm_patho_mode: str = 'film',
        # ---- CPM 头（§8/§11）----
        cpm_horizon: int = 8,
        cpm_layers: int = 2,
        cpm_intervention: bool = False,
        # ---- 条件化（§12.2；pretrain 模式无条件模块）----
        pretrain_mode: bool = True,
        # ---- 条件注入（v1 revised 20261008；None/arm='none' = 不使用）----
        condition_config: Optional[dict] = None,
        condition_stats_payload: Optional[dict] = None,
        condition_arm: Optional[str] = None,
        cond_inject: str = 'heads',
        cond_lambda_cpm: float = 1.0,
        # ---- 归一化（§15）----
        use_revin: bool = True,
        dynamics_mode: str = 'graph_ode',
        ode_steps: int = 2,
        dynamics_dt: float = 1.0,
        graph_source: str = 'history_fc',
        personalized: bool = False,
        profile_dim: int = 32,
        dynamics_history_memory: bool = False,
        state_feedback: bool = False,
        spatial_profile: bool = False,
        condition_initial_state: bool = False,
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
        if cpm_intervention:
            raise ValueError("当前 CPM 不包含干预嵌入；控制输入请使用 ODE controls。")
        self.cpm_horizon = int(cpm_horizon)
        self.pretrain_mode = bool(pretrain_mode)
        self.use_revin = bool(use_revin)
        self.sc_prior_mode = str(sc_prior_mode)
        self.dynamics_mode = str(dynamics_mode)
        if self.dynamics_mode != 'graph_ode':
            raise ValueError(
                f"仅支持 dynamics_mode=graph_ode，收到 '{dynamics_mode}'")
        self.ode_steps = int(ode_steps)
        if self.ode_steps < 1:
            raise ValueError(f'ode_steps 必须 >= 1，收到 {ode_steps}')
        self.dynamics_dt = float(dynamics_dt)
        if self.dynamics_dt <= 0:
            raise ValueError(f'dynamics_dt 必须为正，收到 {dynamics_dt}')
        self.graph_source = str(graph_source)
        if self.graph_source != 'history_fc':
            raise ValueError(
                f"当前结构只支持 graph_source=history_fc，收到 '{graph_source}'")
        self.personalized = bool(personalized)
        if spatial_profile and not self.personalized:
            raise ValueError('spatial_profile 需要 personalized=True')
        self.profile_dim = int(profile_dim) if self.personalized else 0
        if self.personalized and self.profile_dim < 1:
            raise ValueError(f'profile_dim 必须 >= 1，收到 {profile_dim}')
        self.dynamics_history_memory = bool(dynamics_history_memory)
        self.state_feedback = bool(state_feedback)
        # 显式无条件声明用于对照分析，不隐式构建数值调制。
        self.condition_arm = (str(condition_arm)
                              if condition_arm is not None and str(condition_arm) != ''
                              else None)
        if self.condition_arm is not None and self.condition_arm not in ARMS:
            raise ValueError(f"condition_arm 必须是 {ARMS}，收到 '{self.condition_arm}'")
        # 字段条件同时用于健康预训练与临床微调。
        cfg_arm = (str(dict(condition_config).get('arm', 'none'))
                   if condition_config is not None else None)
        if self.condition_arm is not None and cfg_arm is not None \
                and self.condition_arm != cfg_arm:
            raise ValueError(
                f'condition_arm 声明（{self.condition_arm}）与 condition_config.arm'
                f'（{cfg_arm}）不一致；两处必须同源，否则模型的「无条件」语义与'
                '编码器配置互相矛盾。')
        if self.condition_arm not in (None, 'none') and condition_config is None:
            raise ValueError(
                f"condition_arm='{self.condition_arm}' 需要同时提供 condition_config "
                '与 condition_stats_payload；臂不能只靠声明而没有编码器配置。')
        # 生效臂：声明值与 condition_config 一致（已由上面的检查约束），取其一即可。
        active_arm = self.condition_arm if self.condition_arm is not None else cfg_arm
        self.condition_disabled = active_arm == 'none'
        self.tfm_patho_mode = 'film' if active_arm not in (None, 'none') else 'none'

        # ---- Context-only RevIN（§15：μ/σ 只由历史 context 估计）----
        self.rev_norm = BrainRevIN(num_features=features) if self.use_revin else None

        # ---- SC 软解剖先验：A_eff 一次计算、全层共享（§7）----
        self.sc_prior = SoftAnatomicalPrior(
            features=features, seq_len=context_max,
            mode=sc_prior_mode, rank=sc_prior_rank,
            lambda_mode=sc_lambda_mode, lambda_init=sc_lambda_init,
            delta_a=sc_delta_a, delta_rank=sc_delta_rank,
            sinkhorn_iters=sc_sinkhorn_iters,
        )

        # ---- 字段条件编码：健康共享字段与临床增量字段 ----
        from utils.condition_schema import ConditionStats, schema_for

        self.conditioner = None
        self.condition_cfg_dict = dict(condition_config) if condition_config else None
        self.cond_inject = str(cond_inject)
        if self.cond_inject != 'heads':
            raise ValueError('当前结构只支持 cond_inject=heads（预测头 FiLM）。')

        condition_cfg = None
        if condition_config is not None:
            condition_cfg = ConditioningConfig.from_dict(condition_config)
        self.conditioning_on = bool(condition_cfg is not None
                                     and condition_cfg.arm != 'none')
        if condition_initial_state and not self.conditioning_on:
            raise ValueError('condition_initial_state 需要启用字段条件编码')
        if self.condition_disabled:
            print('[condition] arm=none：绕过条件通路（不构建编码器，也不构建 '
                  'FiLM）。')

        cond_dim = 0
        if self.conditioning_on:
            if not condition_stats_payload:
                raise ValueError(
                    '条件注入需要 train-only 统计量快照（condition_stats_payload）；'
                    '不允许靠运行时默认值猜测 mask/词表语义。')
            specs = schema_for(condition_cfg.group)
            stats = ConditionStats.from_dict(specs, condition_stats_payload)
            # 条件编码器在 RNG 隔离作用域内构造：其初始化不消耗后续非条件模块的
            # 随机流（同 seed 下 H0/H1 非条件参数逐位相同）。
            with isolated_global_rng():
                self.conditioner = build_condition_encoder(
                    specs, stats, condition_cfg)
            cond_dim = int(condition_cfg.cond_dim)
            print(f'[condition] 编码器 arm={condition_cfg.arm} group={condition_cfg.group} '
                  f'| d_demo={condition_cfg.d_demo} d_new={condition_cfg.d_new} '
                  f'C={condition_cfg.cond_dim} H={condition_cfg.h_delta} '
                  f'style={condition_cfg.delta_style} dropout_p={condition_cfg.dropout_p} '
                  f'| 条件参数={self.conditioner.param_report()["total"]}')
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

        # ---- Graph-ODE 与 CPM 共享字段条件宽度，各自建立 FiLM ----
        head_cond_dim = cond_dim if self.conditioning_on else 0
        cond_rng_isolated = self.conditioning_on
        self.one_step_head = GraphODEDynamicsHead(
            dim=dim, hidden_dim=one_step_hidden_dim,
            cond_dim=head_cond_dim or None,
            profile_dim=self.profile_dim or None,
            ode_steps=self.ode_steps,
            history_memory=dynamics_history_memory,
            state_feedback=self.state_feedback)
        self.cpm_head = CPMHorizonHead(
            dim=dim, horizon=cpm_horizon, num_heads=num_heads,
            num_layers=cpm_layers, dropout=dropout,
            cond_dim=head_cond_dim or None,
            cond_lambda=cond_lambda_cpm,
            cond_rng_isolated=cond_rng_isolated)

        # ---- 个体基线 profile（方案 §5；personalized=False 不创建）----
        self.profile_encoder = None
        if self.personalized:
            self.profile_encoder = ProfileEncoder(
                dim=dim, profile_dim=self.profile_dim, hidden=64,
                spatial_features=features if spatial_profile else 0)
        self.initial_condition_film = None
        if condition_initial_state and head_cond_dim:
            self.initial_condition_film = ConditionFiLM(head_cond_dim, dim)

    # ------------------------------------------------------------------
    # 内部构件
    # ------------------------------------------------------------------
    def _prepare_condition(self, pathology_score, condition=None,
                           condition_epoch=None, subject_index=None):
        """字段条件编码；None 表示绕过，全缺失张量仍保留缺失表示。"""
        if pathology_score is not None:
            raise ValueError('已移除数值 pathology_score 通路，请使用字段 condition。')
        if condition is None:
            return None
        if self.conditioner is None:
            raise ValueError('收到 condition 但当前模型未构建条件编码器。')
        if not isinstance(condition, dict):
            raise TypeError(f'condition 需为字段张量 dict，收到 {type(condition)}')
        missing = [key for key in ('cond_z', 'cond_code', 'cond_mask')
                   if condition.get(key) is None]
        if missing:
            raise KeyError(f'condition 缺少必需键 {missing}')
        return self.conditioner(
            condition['cond_z'], condition['cond_code'], condition['cond_mask'],
            subject_index=condition.get('subject_index'), epoch=condition_epoch)

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

    def _history_fc_adj(self, x_grid: torch.Tensor) -> torch.Tensor:
        """当前可见历史的 signed Pearson 功能耦合图（方案 §4.2）。

        x_grid ``[B,F,K]``（归一化或数据尺度均可：Pearson 相关对逐 ROI 仿射不变）
        → ``[B,F,F]``：取绝对值、清零对角。该图作为 SoftAnatomicalPrior 的输入，
        复用其混合、self-loop 与 Sinkhorn；它是**功能耦合图**，不默认称独立解剖 SC。
        """
        return self._history_fc_signed(x_grid).abs()

    def _history_fc_signed(self, x_grid: torch.Tensor) -> torch.Tensor:
        r = x_grid - x_grid.mean(dim=-1, keepdim=True)
        num = torch.bmm(r, r.transpose(1, 2))
        denom = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-6)   # [B,F,1]
        c_hist = num / (denom * denom.transpose(1, 2))
        f = x_grid.shape[1]
        eye = torch.eye(f, dtype=torch.bool, device=x_grid.device)
        return c_hist.masked_fill(eye, 0.0)

    def _feedback_context(self, x_grid: torch.Tensor, signed_fc: Optional[torch.Tensor] = None):
        if not self.state_feedback:
            return None, None
        corr = signed_fc if signed_fc is not None else self._history_fc_signed(x_grid)
        corr = corr / corr.abs().sum(-1, keepdim=True).clamp_min(1e-6)
        return x_grid.float().mean(-1), corr

    def _initial_latent(self, tokens: torch.Tensor, cond: Optional[torch.Tensor]):
        z = tokens[:, :, -1, :]
        if self.initial_condition_film is not None and cond is not None:
            scale, shift = self.initial_condition_film(cond)
            z = z * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return z

    def _graph_input(self, x_grid: torch.Tensor, sc_matrix: torch.Tensor):
        """由可见历史构建功能耦合图。"""
        return self._history_fc_adj(x_grid)

    def _denorm_3d(self, pred_norm: torch.Tensor, stats) -> torch.Tensor:
        """[B,F,H]（归一化）→ 原始空间（复用 BrainRevIN 的仿射反变换）。"""
        if self.rev_norm is None:
            return pred_norm
        out = self.rev_norm(pred_norm.unsqueeze(-1), 'denorm', stats=stats)
        return out.squeeze(-1)



    # ------------------------------------------------------------------
    # 前向（next_timepoint 协议同形：[B,F,1,K] → [B,F,1,1]）
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, sc_matrix: torch.Tensor,
                pathology_score: Optional[torch.Tensor] = None,
                future_control: Optional[torch.Tensor] = None,
                pathology_epoch: Optional[int] = None,
                condition: Optional[dict] = None,
                condition_epoch: Optional[int] = None,
                controls: Optional[torch.Tensor] = None,
                time_deltas: Optional[torch.Tensor] = None,
                dynamics_horizon: Optional[int] = None,
                calibration_bold: Optional[torch.Tensor] = None,
                profile: Optional[torch.Tensor] = None,
                with_aux: bool = True):
        """单次前向：Graph-ODE H1..H 轨迹 + 可选 CPM/方差辅助输出。

        Args:
            x: [B, F, 1, K] context（K ≤ context_max）
            sc_matrix: [F, F] 或 [B, F, F]
            pathology_score: 保留的位置参数，仅接受 None；条件使用 condition。
            future_control: 仅接受 None；控制输入使用 controls。
            pathology_epoch: 当前训练 epoch（0 基）；保留的兼容参数。
            with_aux: True 时同时计算 CPM 全 horizon 与方差头（旧接口）；
                False 时跳过这两条辅助通路，只保留主 ODE 轨迹，训练路径使用。
        Returns:
            pred: [B, F, 1, 1] one-step 下一状态预测（原始空间，
                  供既有评估/损失直接消费）
            aux_info: dict：
                ``dynamics_pred`` [B, F, H] Graph-ODE 轨迹（原始空间）
                ``cpm_pred``  [B, F, H] CPM 全 horizon 预测（with_aux=True）
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
        cond = self._prepare_condition(pathology_score, condition, condition_epoch)

        signed_fc = self._history_fc_signed(x_grid) if self.state_feedback else None
        graph_input = signed_fc.abs() if signed_fc is not None else self._graph_input(x_grid, sc_matrix)
        tokens, sc_info = self._encode(x_grid, graph_input)
        # One-Step / 图 ODE 头：最后 causal token 为初始 latent 状态
        z_t = self._initial_latent(tokens, cond)                # [B,F,D]
        x_last_norm = x_grid[:, :, -1]                         # [B,F] = x_t（归一化）

        # ---- 受控图 ODE 分支（方案 §6.3.1）：同一次编码创建初始状态 ----
        if future_control is not None:
            raise ValueError(
                'future_control 已移除；控制输入请传 '
                'controls [B,H,F]，两者不自动转置混用。')
        if self.personalized:
            if profile is None and calibration_bold is not None:
                # 基线另走一次共享编码器；训练时保持梯度，不缓存 detached tokens
                profile = self.calibrate_profile(calibration_bold)
            elif profile is None:
                raise ValueError(
                    'personalized=True 的 forward 需要 profile 或 '
                    'calibration_bold 之一（不偷偷从包含未来的全序列计算）。')
        else:
            profile = None
        # dynamics_horizon 是可选覆盖，缺省复用 CPM H（不新增 horizon CLI）
        dyn_h = (int(dynamics_horizon)
                 if dynamics_horizon is not None else self.cpm_horizon)
        memory = self._history_memory(tokens)
        q_anchor, signed_adj = self._feedback_context(x_grid, signed_fc)
        z_final, q_traj_norm, z_traj = self.one_step_head(
            z_t, x_last_norm, sc_info['A_eff'], cond=cond, profile=profile,
            controls=controls, time_deltas=time_deltas, horizon=dyn_h,
            memory=memory, q_anchor=q_anchor, signed_adj=signed_adj)
        del z_final   # forward 只消费 q 轨迹；latent 末状态经 state API 使用
        # 主输出 = 动态轨迹第一帧（反归一化后 [B,F,1,1]），不再额外加 x_last
        one_pred_norm = q_traj_norm[:, :, 0:1].unsqueeze(-1)
        dyn_aux = {'dynamics_mean_norm': q_traj_norm,          # [B,F,H]
                   'norm_stats': norm_stats}
        if with_aux:
            logvar_hf = self.one_step_head.variance_head(z_traj.detach())
            dyn_aux['dynamics_logvar_norm'] = logvar_hf.permute(0, 2, 1)
        if profile is not None:
            dyn_aux['personal_profile'] = profile              # [B,profile_dim]

        aux_info = {'sc_prior': sc_info}
        if cond is not None:
            aux_info['pathology_cond'] = cond
            aux_info['condition_injected'] = torch.ones(1, device=cond.device)
        if self.rev_norm is not None:
            dyn_aux['dynamics_pred'] = self._denorm_3d(
                dyn_aux['dynamics_mean_norm'], norm_stats)   # [B,F,H] 数据尺度
            if with_aux:
                # 数据尺度 std：σ_norm · s / |a + eps|（复用实际 RevIN 分母，方案 §6.4；
                # 是 abs(affine_weight + eps)，不是 abs(affine_weight) + eps）
                sigma_norm = torch.exp(0.5 * dyn_aux['dynamics_logvar_norm'])
                _, std_s = norm_stats                            # [B,F,1,1]
                a_w = self.rev_norm.affine_weight                # [F]
                denom = (a_w + self.rev_norm.eps).abs().view(1, -1, 1)
                dyn_aux['dynamics_std'] = (sigma_norm
                                           * std_s.to(sigma_norm.dtype)[:, :, 0, 0].unsqueeze(-1)
                                           / denom)
        else:
            dyn_aux['dynamics_pred'] = dyn_aux['dynamics_mean_norm']
            if with_aux:
                dyn_aux['dynamics_std'] = torch.exp(
                    0.5 * dyn_aux['dynamics_logvar_norm'])
        aux_info.update(dyn_aux)
        if self.rev_norm is not None:
            one_pred = self.rev_norm(one_pred_norm, 'denorm', stats=norm_stats)
        else:
            one_pred = one_pred_norm
        if with_aux:
            # CPM 头（§11）：ROI × Horizon 查询，一次输出整段未来
            cpm_pred_norm = self.cpm_head(
                tokens, self.patch_embed.roi_embed, cond)
            if self.rev_norm is not None:
                aux_info['cpm_pred'] = self._denorm_3d(cpm_pred_norm, norm_stats)
            else:
                aux_info['cpm_pred'] = cpm_pred_norm
        return one_pred, aux_info

    # ------------------------------------------------------------------
    # next_timepoint 对外标准接口
    # ------------------------------------------------------------------
    @staticmethod
    def _to_internal_history(bold_history: torch.Tensor) -> torch.Tensor:
        """标准输入 [B, K, F] → 内部布局 [B, F, 1, K]。"""
        if bold_history.ndim != 3:
            raise ValueError(
                f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}")
        return bold_history.transpose(1, 2).unsqueeze(2).contiguous()

    def predict_next_state(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                           pathology_score: Optional[torch.Tensor] = None,
                           pathology_epoch: Optional[int] = None,
                           condition: Optional[dict] = None,
                           condition_epoch: Optional[int] = None,
                           control: Optional[torch.Tensor] = None,
                           dt: Optional[torch.Tensor] = None,
                           calibration_bold: Optional[torch.Tensor] = None,
                           profile: Optional[torch.Tensor] = None) -> dict:
        """下一步全脑状态预测（one-step / 图 ODE 单区间）。

        graph_ode 分支显式设动态 H=1（只为首步积分，避免无用的 H=8 动态运算），
        经 state API 初始化一次后 ``advance_state`` 推进一个区间。

        Args:
            control: [B,F] 或 [B,F,1] 区间控制（仅 graph_ode）；dt: [B] 或 None。
        Returns:
            dict：``pred_next`` [B,F,1]（原始空间）、``pred_delta`` [B,F,1]、
            ``pred`` [B,F,1,1]、``aux_info``。
        """
        x = self._to_internal_history(bold_history)
        state = self.initialize_state(
            bold_history, sc_matrix=sc_matrix,
            pathology_score=pathology_score, condition=condition,
            condition_epoch=condition_epoch, profile=profile,
            calibration_bold=calibration_bold)
        if control is not None and control.ndim == 2:
            control = control.unsqueeze(-1)                # [B,F,1]
        state = self.advance_state(state, control=control, dt=dt)
        pred_next = self.readout_state(state).unsqueeze(-1)  # [B,F,1]
        x_last = bold_history[:, -1, :].unsqueeze(-1)      # [B,F,1] = x_t
        aux_info = {'norm_stats': state['norm_stats']}
        if state['profile'] is not None:
            aux_info['personal_profile'] = state['profile']
        return {'pred_next': pred_next, 'pred_delta': pred_next - x_last,
                'pred': pred_next.unsqueeze(-1), 'aux_info': aux_info}

    def forecast_horizon(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                         pathology_score: Optional[torch.Tensor] = None,
                         future_control: Optional[torch.Tensor] = None,
                         pathology_epoch: Optional[int] = None,
                         condition: Optional[dict] = None,
                         condition_epoch: Optional[int] = None) -> torch.Tensor:
        """CPM 非自回归全 horizon 预测（§8：一次 forward 输出 t+1..t+H）。

        与 :meth:`rollout_next_states` 的自回归轨迹对比可量化 error
        accumulation（§25.3）。返回 [B, H, F]（原始空间）。
        """
        x = self._to_internal_history(bold_history)
        _, aux_info = self.forward(x, sc_matrix, pathology_score,
                                   future_control=future_control,
                                   pathology_epoch=pathology_epoch,
                                   condition=condition,
                                   condition_epoch=condition_epoch)
        return aux_info['cpm_pred'].transpose(1, 2)            # [B,H,F]

    def rollout_next_states(self, bold_history: torch.Tensor, sc_matrix: torch.Tensor,
                            pathology_score: Optional[torch.Tensor] = None,
                            steps: int = 1,
                            pathology_epoch: Optional[int] = None,
                            condition: Optional[dict] = None,
                            condition_epoch: Optional[int] = None,
                            controls: Optional[torch.Tensor] = None,
                            time_deltas: Optional[torch.Tensor] = None,
                            calibration_bold: Optional[torch.Tensor] = None,
                            profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """free rollout：连续 ``steps`` 步状态推进，返回 [B, steps, F]。

        graph_ode 分支只初始化**一次**当前历史（固定图、c/p、RevIN stats），
        随后按 ``controls [B,steps,F]`` 与 ``time_deltas [steps]`` 或 ``[B,steps]``
        持续推进同一 latent/BOLD 状态，不回填预测后再次编码（方案 §6.3.1）。
        """
        if bold_history.ndim != 3:
            raise ValueError(f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}")
        n_steps = int(steps)
        if n_steps < 1:
            raise ValueError(f"steps 必须 >= 1，收到 {n_steps}")
        b, _, f_dim = bold_history.shape
        if controls is not None and tuple(controls.shape) != (b, n_steps, f_dim):
            raise ValueError(
                f'controls 期望 [B,{n_steps},{f_dim}]，收到 {tuple(controls.shape)}')
        if time_deltas is not None and time_deltas.shape[-1] != n_steps:
            raise ValueError(
                f'time_deltas 末维 {time_deltas.shape[-1]} 与 steps={n_steps} 不一致')
        state = self.initialize_state(
            bold_history, sc_matrix=sc_matrix,
            pathology_score=pathology_score, condition=condition,
            condition_epoch=condition_epoch, profile=profile,
            calibration_bold=calibration_bold)
        outs: List[torch.Tensor] = []
        for k in range(n_steps):
            u = controls[:, k, :] if controls is not None else None
            dt_sel = None
            if time_deltas is not None:
                dt_sel = (time_deltas[:, k] if time_deltas.ndim == 2
                          else time_deltas[k])
            state = self.advance_state(
                state, control=(u.unsqueeze(-1) if u is not None else None),
                dt=dt_sel)
            outs.append(self.readout_state(state))
        return torch.stack(outs, dim=1)                    # [B, steps, F]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _denorm_q(self, q: torch.Tensor, norm_stats) -> torch.Tensor:
        """RevIN 空间 q [B,F] → 数据尺度（复用 BrainRevIN 统计量与 affine）。"""
        if self.rev_norm is None:
            return q
        out = self.rev_norm(q.unsqueeze(-1).unsqueeze(-1), 'denorm', stats=norm_stats)
        return out.squeeze(-1).squeeze(-1)

    def _encode_history_internal(self, history: torch.Tensor, sc_matrix,
                                 cond: Optional[torch.Tensor] = None):
        """数据尺度 history [B,K,F] → (tokens, x_grid_norm, norm_stats, sc_info)。

        图输入按 graph_source 路由（history_fc 从当前可见历史计算 FC）；
        cond 已编码时按需生成主干 AdaRMSNorm 调制（与 forward 口径一致）。
        """
        x = history.transpose(1, 2).unsqueeze(2)               # [B,F,1,K]
        norm_stats = None
        if self.rev_norm is not None:
            x_norm4, norm_stats = self.rev_norm(x, 'norm')
            x_grid = x_norm4[:, :, 0, :]
        else:
            x_grid = x[:, :, 0, :]
        signed_fc = self._history_fc_signed(x_grid) if self.state_feedback else None
        graph_input = signed_fc.abs() if signed_fc is not None else self._graph_input(x_grid, sc_matrix)
        tokens, sc_info = self._encode(x_grid, graph_input)
        if signed_fc is not None:
            sc_info['signed_fc'] = signed_fc
        return tokens, x_grid, norm_stats, sc_info

    def _symmetrize_adj(self, a_eff: torch.Tensor) -> torch.Tensor:
        """A_eff → 非负对称图 A（方案 §6.1.1：动态头只做对称化，不另学图）。"""
        return 0.5 * (a_eff + a_eff.transpose(-1, -2))

    def _history_memory(self, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """历史记忆：编码 token 在 patch 维取均值 → [B,F,D]；未启用时返回 None。

        记忆只由**已观测历史**编码得到，rollout 期间保持固定，使自由推进始终可以
        访问真实历史，而不必反复把预测值反馈回编码器（D2 已证明后者会放大漂移）。
        显式转 float32：ODE 积分在 autocast 关闭下运行，记忆在进入图 ODE 头之前就
        统一到 fp32，避免 torch.compile 图内出现 Half/float 混合的 matmul。
        """
        if not self.dynamics_history_memory:
            return None
        return _pool_history_tokens(tokens)

    def calibrate_profile(self, calibration_bold: torch.Tensor) -> torch.Tensor:
        """基线 ``[B,F,1,L]`` → p ``[B,profile_dim]``（共享编码器一次，方案 §5/§6.3）。

        基线编码不注入当前临床 FiLM/AdaRMSNorm；图输入取基线前缀自身的可见
        历史 FC（与 history_fc 的历史图同一定义，不需要外部 SC）。训练 forward
        内调用时保持梯度，不跨 optimizer step 缓存 detached tokens；推理每人
        计算一次后供 rollout / observation update 复用。
        """
        if self.profile_encoder is None:
            raise ValueError(
                'personalized=False 未创建 profile 模块，无基线校准可言。')
        if calibration_bold.ndim != 4 or calibration_bold.shape[2] != 1:
            raise ValueError(
                f'calibration_bold 期望 [B,F,1,L]，收到 {tuple(calibration_bold.shape)}')
        x_grid, _ = build_context_grid(calibration_bold)       # [B,F,L]
        if self.rev_norm is not None:
            cal_norm4, _ = self.rev_norm(calibration_bold, 'norm')
            x_grid = cal_norm4[:, :, 0, :]
        graph_input = self._history_fc_adj(x_grid)
        tokens, _ = self._encode(x_grid, graph_input)
        return self.profile_encoder(tokens)

    def initialize_state(self, history: torch.Tensor, sc_matrix=None,
                         pathology_score: Optional[torch.Tensor] = None,
                         condition: Optional[dict] = None,
                         condition_epoch: Optional[int] = None,
                         profile: Optional[torch.Tensor] = None,
                         calibration_bold: Optional[torch.Tensor] = None) -> dict:
        """当前历史 ``[B,K,F]``（数据尺度）→ 状态 dict（方案 §6.3）。

        返回普通 dict：``z [B,F,D]``（最后 patch token，无额外状态投影）、
        ``q [B,F]``（RevIN 空间）、``adj [B,F,F]``（A_eff 对称化）、
        ``cond [B,C]``（字段编码一次）、``profile [B,P]``、``norm_stats``、
        ``history [B,K,F]``（数据尺度 buffer）；图来自可见历史 FC。

        条件只接受字段张量 dict，数值病理评分由输入检查显式拒绝。
        """
        if history.ndim != 3:
            raise ValueError(f'history 期望 [B,K,F]，收到 {tuple(history.shape)}')
        cond = self._prepare_condition(pathology_score, condition, condition_epoch)
        if self.personalized:
            if profile is None:
                if calibration_bold is None:
                    raise ValueError(
                        'personalized=True 的 initialize_state 需要 profile 或 '
                        'calibration_bold 之一；不偷偷从包含未来的全序列计算。')
                profile = self.calibrate_profile(calibration_bold)
        else:
            profile = None
        tokens, x_grid, norm_stats, sc_info = self._encode_history_internal(
            history, sc_matrix, cond)
        q_anchor, signed_adj = self._feedback_context(x_grid, sc_info.get('signed_fc'))
        state = {'z': self._initial_latent(tokens, cond), 'q': x_grid[:, :, -1],
                 'adj': self._symmetrize_adj(sc_info['A_eff']),
                 'cond': cond, 'profile': profile,
                 'memory': self._history_memory(tokens),
                 'q_anchor': q_anchor, 'signed_adj': signed_adj,
                 'norm_stats': norm_stats, 'history': history}
        return state

    def advance_state(self, state: dict, control: Optional[torch.Tensor] = None,
                      dt: Optional[torch.Tensor] = None) -> dict:
        """推进一个预测区间：更新 z/q/history；图、p、已编码 c、RevIN stats 固定。

        Args:
            control: [B,F] 或 [B,F,1] 区间控制；None 与全零严格等价。
            dt: [B] 区间长度（TR）；None 用构造参数 ``dynamics_dt``。

        返回**新** dict，不原位覆盖、不对 z/q detach（保留反向图）；
        history 以数据尺度预测帧滑窗更新（追加末帧、丢弃最早帧，K 不变）。
        """
        if control is not None and control.ndim == 2:
            control = control.unsqueeze(-1)                    # [B,F,1]
        if dt is None:
            dt = torch.full((state['z'].shape[0],), float(self.dynamics_dt),
                            device=state['z'].device, dtype=torch.float32)
        z_new, q_new = self.one_step_head.integrate_interval(
            state['z'], state['q'], state['adj'], dt,
            cond=state['cond'], profile=state['profile'], control=control,
            memory=state.get('memory'), q_anchor=state.get('q_anchor'),
            signed_adj=state.get('signed_adj'))
        x_new = self._denorm_q(q_new, state['norm_stats'])     # [B,F] 数据尺度
        history_new = torch.cat(
            [state['history'][:, 1:, :], x_new.unsqueeze(1)], dim=1)
        new_state = dict(state)
        new_state.update({'z': z_new, 'q': q_new, 'history': history_new})
        return new_state

    def readout_state(self, state: dict) -> torch.Tensor:
        """q 反归一化 → 数据尺度 [B,F]；不再次编码。"""
        return self._denorm_q(state['q'], state['norm_stats'])

    def update_state(self, state: dict, observation: torch.Tensor) -> dict:
        """真实观测（数据尺度 [B,F]，对齐当前预测时点）替换 history 末帧并重编码。

        重新估计当前 RevIN / 图 / Z / q；**保留** profile 与已编码 c（不重复
        条件编码或 profile 计算）。这是观测重编码更新，不是 Bayesian 同化；
        首版只支持完整 ROI 观测。
        """
        if observation.ndim != 2:
            raise ValueError(
                f'observation 期望 [B,F]，收到 {tuple(observation.shape)}')
        history_new = state['history'].clone()
        expected = (history_new.shape[0], history_new.shape[2])
        if tuple(observation.shape) != expected:
            raise ValueError(
                f'observation 期望 [B={expected[0]},F={expected[1]}]，'
                f'收到 {tuple(observation.shape)}')
        history_new[:, -1, :] = observation
        sc_matrix = state.get('sc_matrix')
        tokens, x_grid, norm_stats, sc_info = self._encode_history_internal(
            history_new, sc_matrix, state['cond'])
        new_state = dict(state)
        q_anchor, signed_adj = self._feedback_context(x_grid, sc_info.get('signed_fc'))
        new_state.update({'z': self._initial_latent(tokens, state['cond']), 'q': x_grid[:, :, -1],
                          'adj': self._symmetrize_adj(sc_info['A_eff']),
                          'memory': self._history_memory(tokens),
                          'q_anchor': q_anchor, 'signed_adj': signed_adj,
                          'norm_stats': norm_stats, 'history': history_new})
        return new_state

    def simulate_intervention(self, history: torch.Tensor, controls: torch.Tensor,
                              sc_matrix=None,
                              pathology_score: Optional[torch.Tensor] = None,
                              condition: Optional[dict] = None,
                              condition_epoch: Optional[int] = None,
                              profile: Optional[torch.Tensor] = None,
                              calibration_bold: Optional[torch.Tensor] = None,
                              time_deltas: Optional[torch.Tensor] = None):
        """同初始状态分别用 U 与零 U 推进 H 步（方案 §6.3/§6.3.1）。

        Args:
            history: [B,K,F] 数据尺度；controls: [B,H,F] 新布局逐区间控制。
        Returns:
            (x_ctrl [B,H,F], x_zero [B,H,F], diff [B,H,F])；差值由两条反归一化
            轨迹相减，不对增量单独加均值。
        """
        if controls is None or controls.ndim != 3:
            raise ValueError(
                f'controls 期望 [B,H,F]，收到 '
                f'{tuple(controls.shape) if controls is not None else None}')
        b, h_out, f_dim = controls.shape
        if history.shape[0] != b or history.shape[2] != f_dim:
            raise ValueError(
                f'history [B,K,F]={tuple(history.shape)} 与 controls '
                f'[B,H,F]={tuple(controls.shape)} 的 B/F 不一致')
        state0 = self.initialize_state(
            history, sc_matrix=sc_matrix, pathology_score=pathology_score,
            condition=condition, condition_epoch=condition_epoch,
            profile=profile, calibration_bold=calibration_bold)
        xs_ctrl, xs_zero = [], []
        ctrl_state, zero_state = state0, state0
        for k in range(h_out):
            u = controls[:, k, :].unsqueeze(-1)                # [B,F,1]
            dt_sel = None
            if time_deltas is not None:
                dt_sel = (time_deltas[:, k] if time_deltas.ndim == 2
                          else time_deltas[k])
            ctrl_state = self.advance_state(ctrl_state, control=u, dt=dt_sel)
            zero_state = self.advance_state(zero_state, control=None, dt=dt_sel)
            xs_ctrl.append(self.readout_state(ctrl_state))
            xs_zero.append(self.readout_state(zero_state))
        x_ctrl = torch.stack(xs_ctrl, dim=1)                   # [B,H,F]
        x_zero = torch.stack(xs_zero, dim=1)                   # [B,H,F]
        return x_ctrl, x_zero, x_ctrl - x_zero
