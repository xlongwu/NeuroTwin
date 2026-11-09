# coding=utf-8
"""唯一V2：历史功能图引导TFM、局部—全局相位状态及未来时距交互CPM。"""
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import BrainRevIN
from models.conditioning import ARMS, ConditioningConfig, build_condition_encoder
from models.graph_prior import SoftAnatomicalPrior
from models.temporal_multiscale import MultiScaleTemporalEncoder, CausalTemporalBlock
from models.phase_forecast import LocalGlobalPhaseForecast
from models.tfm_backbone import RMSNorm, TFMEncoderBlock, TemporalPatchEmbed, build_context_grid

@contextmanager
def isolated_global_rng():
    """条件模块初始化不推进外部 RNG，沿用已训练 B 的随机流隔离。"""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _rng_scope(isolate: bool):
    """条件构件使用独立初始化作用域。"""
    return isolated_global_rng() if isolate else nullcontext()


class ConditionFiLM(nn.Module):
    """字段条件 FiLM 调制。

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


class DynamicForecastAdapter(nn.Module):
    """小幅动态查询与额外记忆；残差读取不扰动原patch记忆的softmax。"""
    is_prediction_adapter = True

    def __init__(self, dim: int):
        super().__init__()
        self.multiscale_query = nn.Linear(dim, dim)
        self.phase_query = nn.Linear(dim, dim)
        self.memory_proj = nn.Linear(dim, dim)
        self.query_scales = nn.Parameter(torch.full((2,), .1))
        self.memory_logit = nn.Parameter(torch.tensor(-3.))
        for layer in (self.multiscale_query, self.phase_query, self.memory_proj):
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, context: torch.Tensor, roi_embed: torch.Tensor,
                state: torch.Tensor, phase: torch.Tensor) -> tuple:
        memory = self.memory_proj(state) + roi_embed[None]
        delta = (self.query_scales[0]*self.multiscale_query(state).unsqueeze(2)
                 + self.query_scales[1]*self.phase_query(phase))
        return memory, delta, self.memory_logit.sigmoid()


class FutureTemporalAdapter(nn.Module):
    """逐 ROI 沿未来时间轴交换潜在状态；不读取真实未来。"""
    is_prediction_adapter = True

    def __init__(self, dim: int):
        super().__init__()
        self.blocks = nn.Sequential(*(CausalTemporalBlock(dim, d) for d in (1, 2, 4)))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.normal_(self.out_proj.weight, std=.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f, h, d = x.shape
        z = self.blocks(x.reshape(b*f, h, d).transpose(1, 2)).transpose(1, 2)
        return x + self.out_proj(z).reshape(b, f, h, d)


class CPMHorizonHead(nn.Module):
    """CPM 全 horizon 预测头：非自回归一次输出 ``x̂_(t+1:t+H)``。

    未来查询::

        q_{i,h} = e^ROI_i + e^horizon_h

    对 backbone 上下文 token ``H_context ∈ R^{B×F×P×D}``（全部为历史信息，
    无需 causal mask）做多层多头交叉注意力后，逐 (ROI, horizon) 线性解码。
    ROI 嵌入与 backbone patch embedding 共享（同一套 e^ROI），horizon 嵌入
    为本头私有参数。

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
        self.dynamic_adapter = DynamicForecastAdapter(dim)
        self.future_temporal = FutureTemporalAdapter(dim)
    def forward(self, context: torch.Tensor, roi_embed: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                multiscale_state: Optional[torch.Tensor] = None,
                phase_features: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        extra_memory, delta, memory_scale = self.dynamic_adapter(
            context, roi_embed, multiscale_state, phase_features)
        queries = queries + delta
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)
            queries = queries * (1.0 + scale.view(b, 1, 1, d)) \
                + shift.view(b, 1, 1, d)
        # 额外记忆以小门控残差读取；两组记忆都只来自可见历史。
        q_flat = queries.reshape(b, f * h_out, d)
        m_flat = context.reshape(b, -1, d)
        m_flat = torch.cat((m_flat, extra_memory), 1)
        for layer in self.layers:
            attn_out = layer['attn'](layer['norm_q'](q_flat),
                                     layer['norm_m'](m_flat), memory_scale,
                                     f)
            q_flat = q_flat + layer['drop'](attn_out)
            q_flat = q_flat + layer['drop'](layer['ffn'](layer['norm_ffn'](q_flat)))
        q_flat = self.future_temporal(q_flat.reshape(b, f, h_out, d)).reshape(b, f*h_out, d)
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

    def forward(self, q: torch.Tensor, memory: torch.Tensor,
                memory_scale: torch.Tensor,
                extra_memory_count: int) -> torch.Tensor:
        """q [B, Nq, D]，memory [B, Nm, D] → [B, Nq, D]。"""
        b, nq, _ = q.shape
        qh = self.q_proj(q).view(b, nq, self.num_heads, self.head_dim
                                 ).transpose(1, 2)             # [B,h,Nq,d]
        kh, vh = self.kv_proj(memory).chunk(2, dim=-1)
        kh = kh.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vh = vh.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        dropout = self.dropout if self.training else 0.0
        # 两组历史记忆共享投影，各自归一化后通过门控残差融合。
        split = kh.shape[2] - extra_memory_count
        out = F.scaled_dot_product_attention(qh, kh[:, :, :split], vh[:, :, :split], dropout_p=dropout)
        extra = F.scaled_dot_product_attention(qh, kh[:, :, split:], vh[:, :, split:], dropout_p=dropout)
        out = out + memory_scale * extra
        out = out.transpose(1, 2).reshape(b, nq, self.dim)
        return self.out_proj(out)


class NeuroTwinTFM(nn.Module):
    """历史 [B,F,1,K] → 未来 [B,F,H]；条件统计量来自训练集快照。"""

    def __init__(self, features: int = 116, context_max: int = 64,
                 patch_len: int = 8, dim: int = 256, num_layers: int = 4,
                 num_heads: int = 4, ff_ratio: int = 4, dropout: float = 0.1,
                 sc_prior_mode: str = 'soft_prior', sc_lambda_mode: str = 'global',
                 sc_lambda_init: float = 0.7, sc_prior_rank: int = 12,
                 sc_delta_a: bool = True, sc_delta_rank: Optional[int] = None,
                 sc_sinkhorn_iters: int = 64, cpm_horizon: int = 32,
                 cpm_layers: int = 2, condition_config: Optional[dict] = None,
                 condition_stats_payload: Optional[dict] = None,
                 condition_arm: Optional[str] = None, use_revin: bool = True,
                 multiscale_channels: int = 64,
                 phase_modes: int = 13, tr_seconds: float = 2.):
        super().__init__()
        if patch_len < 1 or context_max < patch_len:
            raise ValueError('patch_len 必须位于 [1, context_max]')
        self.features, self.context_max = int(features), int(context_max)
        self.patch_len, self.dim = int(patch_len), int(dim)
        self.cpm_horizon = int(cpm_horizon)
        self.use_revin = bool(use_revin)
        self.condition_cfg_dict = dict(condition_config) if condition_config else None
        cfg_arm = (condition_config or {}).get('arm')
        if condition_arm is not None and condition_arm not in ARMS:
            raise ValueError(f'未知条件臂: {condition_arm}')
        if condition_arm is not None and cfg_arm is not None and condition_arm != cfg_arm:
            raise ValueError('condition_arm 与 condition_config.arm 不一致')
        if condition_arm not in (None, 'none') and condition_config is None:
            raise ValueError('字段条件需要 condition_config 和训练统计量')
        self.rev_norm = BrainRevIN(features) if use_revin else None
        self.sc_prior = SoftAnatomicalPrior(
            features=features, seq_len=context_max, mode=sc_prior_mode,
            rank=sc_prior_rank, lambda_mode=sc_lambda_mode,
            lambda_init=sc_lambda_init, delta_a=sc_delta_a,
            delta_rank=sc_delta_rank, sinkhorn_iters=sc_sinkhorn_iters)
        self.conditioner = None
        condition_cfg = ConditioningConfig.from_dict(condition_config) if condition_config else None
        self.conditioning_on = condition_cfg is not None and condition_cfg.arm != 'none'
        cond_dim = 0
        if self.conditioning_on:
            from utils.condition_schema import ConditionStats, schema_for
            if not condition_stats_payload:
                raise ValueError('字段条件需要 train-only 统计量快照')
            specs = schema_for(condition_cfg.group)
            stats = ConditionStats.from_dict(specs, condition_stats_payload)
            with isolated_global_rng():
                self.conditioner = build_condition_encoder(specs, stats, condition_cfg)
            cond_dim = condition_cfg.cond_dim
        self.cond_dim = cond_dim
        self.patch_embed = TemporalPatchEmbed(
            features=features, context_max=context_max, patch_len=patch_len,
            dim=dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            TFMEncoderBlock(dim=dim, num_heads=num_heads, ff_ratio=ff_ratio, dropout=dropout)
            for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim)
        self.cpm_head = CPMHorizonHead(
            dim=dim, horizon=cpm_horizon, num_heads=num_heads,
            num_layers=cpm_layers, dropout=dropout, cond_dim=cond_dim or None,
            cond_rng_isolated=self.conditioning_on)
        self.multiscale_encoder = MultiScaleTemporalEncoder(multiscale_channels, dim)
        self.phase_forecast = LocalGlobalPhaseForecast(context_max, cpm_horizon, dim, phase_modes, tr_seconds)

    def _prepare_condition(self, condition, condition_epoch=None):
        if condition is None:
            return None
        if self.conditioner is None:
            raise ValueError('当前模型没有条件编码器，却收到 condition')
        return self.conditioner(condition['cond_z'], condition['cond_code'],
                                condition['cond_mask'],
                                subject_index=condition.get('subject_index'),
                                epoch=condition_epoch)

    def _history_fc_signed(self, x_grid: torch.Tensor) -> torch.Tensor:
        r = x_grid - x_grid.mean(dim=-1, keepdim=True)
        num = torch.bmm(r, r.transpose(1, 2))
        denom = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-6)
        corr = num / (denom * denom.transpose(1, 2))
        eye = torch.eye(x_grid.shape[1], dtype=torch.bool, device=x_grid.device)
        return corr.masked_fill(eye, 0.0)

    def _history_fc_adj(self, x_grid: torch.Tensor) -> torch.Tensor:
        return self._history_fc_signed(x_grid).abs()

    def _encode(self, x_grid: torch.Tensor):
        graph = self.sc_prior(x_grid.unsqueeze(2), self._history_fc_adj(x_grid))
        tokens = self.patch_embed(x_grid)
        for block in self.blocks:
            tokens = block(tokens, graph['A_eff'])
        return self.final_norm(tokens), graph

    def forward(self, x: torch.Tensor, condition: Optional[dict] = None,
                condition_epoch: Optional[int] = None,
                return_tokens: bool = False) -> tuple:
        """输出完整 [B,F,H] 预测及图、条件表示等辅助信息；不读取未来。"""
        x_grid, _ = build_context_grid(x)
        norm_stats = None
        if self.rev_norm is not None:
            normalized, norm_stats = self.rev_norm(x, 'norm')
            x_grid = normalized[:, :, 0, :]
        cond = self._prepare_condition(condition, condition_epoch)
        tokens, graph = self._encode(x_grid)
        state = self.multiscale_encoder(x_grid)
        phase_features, oscillation = self.phase_forecast(x_grid, state)
        prediction = self.cpm_head(tokens, self.patch_embed.roi_embed, cond, state, phase_features)
        gate = self.phase_forecast.correction_gate(state, phase_features)
        prediction = prediction + gate * oscillation
        dynamic = dict(multiscale_state=state, phase_features=phase_features,
                       oscillation_pred=oscillation, oscillation_gate=gate)
        if self.rev_norm is not None:
            prediction = self.rev_norm(
                prediction.unsqueeze(-1), 'denorm', stats=norm_stats).squeeze(-1)
        aux = {'sc_prior': graph, 'norm_stats': norm_stats}
        if cond is not None:
            aux['condition_embedding'] = cond
        if return_tokens:
            aux['history_tokens'] = tokens
            aux.update(dynamic)
        return prediction, aux

    def forecast_horizon(self, history: torch.Tensor,
                         condition: Optional[dict] = None) -> torch.Tensor:
        """公开预测接口：[B,K,F] → [B,H,F]。"""
        if history.ndim != 3:
            raise ValueError('history 期望 [B,K,F]')
        prediction, _ = self(history.transpose(1, 2).unsqueeze(2), condition=condition)
        return prediction.transpose(1, 2)

    def predict_next_state(self, history: torch.Tensor,
                           condition: Optional[dict] = None) -> dict:
        prediction, aux = self(history.transpose(1, 2).unsqueeze(2), condition=condition)
        first = prediction[:, :, :1]
        return {'pred_next': first, 'pred_delta': first - history[:, -1, :, None],
                'pred': first.unsqueeze(-1), 'aux_info': aux}
