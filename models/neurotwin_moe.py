# coding=utf-8
"""
NeuroTwinMoE — MoDE (Mixture of Denoising Experts) style module for NeuroTwin.

将 MoDE 的时间步条件 t 替换为病理分数条件 pathology_embedding，
用于 MDD 个体化病理残差建模。

Routing strategy:
- Training: multinomial 随机采样，高熵探索防止专家坍塌
- Inference: 低温幂次缩放（inference_temperature=0.3）后采样，
  缓解"训练均匀 → 推理坍塌"的不一致
- use_argmax=True（部署）: 硬 top-k 确定性路由
"""
import logging
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import IterativePredictionRefiner

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
#  第一部分：MoDE 通用组件
# ════════════════════════════════════════════════════════════════════════

class SwishGLU(nn.Module):
    """Swish-Gated Linear Unit，用于 CondRouterMLP。"""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class CondRouterMLP(nn.Module):
    """
    MoE 路由决策 MLP，根据输入（和条件信息）计算专家选择 logits。

    make_it_big=True 时隐层翻倍、层数加倍；否则为单隐层基础结构。
    """
    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        use_swish: bool = True,
        use_relus: bool = False,
        dropout: float = 0,
        make_it_big: bool = False,
    ):
        super().__init__()
        factor = 2 if make_it_big else 1
        repeat = 2 if make_it_big else 1
        layers = []
        for i in range(repeat):
            curr_embed = n_embd if i == 0 else factor * 2 * n_embd
            if use_swish:
                layers.append(SwishGLU(curr_embed, factor * 2 * n_embd))
            else:
                layers.append(nn.Linear(curr_embed, factor * 2 * n_embd))
                layers.append(nn.ReLU() if use_relus else nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(factor * 2 * n_embd, num_experts))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class RouterCond(nn.Module):
    """
    MoDE 条件路由模块（带专家容量限制）。

    原始 MoDE 中条件为时间步嵌入 t_emb [B, D]；
    本文件中条件改为病理嵌入 pathology_embedding [B, pathology_dim]。

    路由策略：
    - 训练（use_argmax=False）：multinomial 随机采样，高熵探索防止专家坍塌
    - 推理（use_argmax=False，默认）：低温幂次缩放（inference_temperature=0.3）
      后采样，缓解"训练均匀 → 推理坍塌"的不一致
    - use_argmax=True（部署）：硬 top-k 确定性路由

    capacity_factor 限制每个专家在一个 batch 中最多处理的样本数，
    超出容量的专家会被软惩罚，防止过载。
    """

    def __init__(
        self,
        hidden_states: int,
        cond_dim: int,
        num_experts: int,
        top_k: int,
        use_argmax: bool = False,
        normalize: bool = True,
        cond_router: bool = True,
        router_context_cond_only: bool = True,
        temperature: float = 2.0,
        inference_temperature: float = 0.3,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        self.temperature = max(1e-4, float(temperature))
        self.use_argmax = use_argmax
        self.cond_router = cond_router
        self.router_context_cond_only = router_context_cond_only
        self.inference_temperature = max(1e-4, float(inference_temperature))
        self.capacity_factor = capacity_factor

        self.router = self._create_router(hidden_states, cond_dim)
        self.logits: Optional[torch.Tensor] = None

    def _create_router(self, hidden_states: int, cond_dim: int) -> nn.Module:
        if self.cond_router:
            input_dim = (cond_dim if self.router_context_cond_only
                         else hidden_states + cond_dim)
        else:
            input_dim = hidden_states
        return CondRouterMLP(
            input_dim, self.num_experts,
            use_swish=False, dropout=0, make_it_big=False)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(1e-4, float(temperature))

    def set_inference_temperature(self, temperature: float) -> None:
        self.inference_temperature = max(1e-4, float(temperature))

    def forward(
        self,
        inputs: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = inputs.size()
        logits = self._compute_logits(inputs, cond)
        probs  = self._compute_probabilities(logits)
        router_mask, top_k_indices, router_probs = self._select_experts(probs, input_shape)
        return router_mask, top_k_indices, router_probs, probs.view(*input_shape[:-1], -1)

    def _compute_logits(self, inputs, cond):
        if self.cond_router:
            return self._compute_cond_logits(inputs, cond)
        return self._compute_uncond_logits(inputs)

    def _compute_cond_logits(self, inputs, cond):
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if cond.shape[1] != inputs.shape[1]:
            cond = cond.expand(-1, inputs.shape[1], -1)
        if self.router_context_cond_only:
            router_inputs = cond.reshape(-1, cond.size(-1))
        else:
            router_inputs = torch.cat([inputs, cond], dim=-1).reshape(
                -1, inputs.size(-1) + cond.size(-1))
        return self.router(router_inputs)

    def _compute_uncond_logits(self, inputs):
        return self.router(inputs.reshape(-1, inputs.size(-1)))

    def _compute_probabilities(self, logits):
        logits = (logits - logits.max(dim=-1, keepdim=True).values) / self.temperature
        self.logits = logits
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-9, max=1 - 1e-9)
        if not torch.isfinite(probs).all():
            logger.warning("Router probabilities contain inf or NaN")
        return probs

    def _select_experts(self, probs, input_shape):
        return self._select_without_shared(probs, input_shape)

    def _select_without_shared(self, probs, input_shape):
        flat_probs    = probs.view(-1, probs.size(-1))
        batch_size = flat_probs.size(0)

        # 容量限制：训练时增加 20% 软容量余量，避免过早饱和
        base_capacity = int(batch_size * self.top_k / self.num_experts * self.capacity_factor)
        if self.training:
            capacity = max(1, int(base_capacity * 1.2))
        else:
            capacity = max(1, base_capacity)

        # 初始化专家计数器和负载记录
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=flat_probs.device)
        overflow_penalty = torch.zeros(self.num_experts, device=flat_probs.device)

        # 为每个样本选择专家，考虑容量限制
        top_k_indices_list = []

        for i in range(batch_size):
            sample_probs = flat_probs[i]
            selected_experts = []
            temp_probs = sample_probs.clone()

            for _ in range(self.top_k):
                if self.use_argmax:
                    # 硬选择：考虑容量惩罚后的 argmax
                    penalized_probs = temp_probs - overflow_penalty
                    expert_idx = penalized_probs.argmax()
                else:
                    # 软选择：考虑容量的概率采样
                    # 计算每个专家的可用性（基于容量使用情况）
                    usage_ratio = expert_counts.float() / float(capacity)
                    # 软惩罚：当使用率达到容量时，概率逐渐降低
                    soft_mask = torch.clamp(1.0 - usage_ratio, min=0.1)  # 至少保留10%概率

                    if self.training:
                        # 训练时：结合原始概率和容量约束
                        # 使用 Gumbel-softmax 风格的探索
                        available_probs = temp_probs * soft_mask

                        # 如果所有专家都接近满负荷，允许轻微超载
                        if available_probs.sum() < 1e-8:
                            # 选择使用率最低的专家
                            expert_idx = usage_ratio.argmin()
                        else:
                            available_probs = available_probs / available_probs.sum()
                            # 添加 Gumbel 噪声鼓励探索
                            gumbel_noise = -torch.log(-torch.log(torch.rand_like(available_probs) + 1e-10) + 1e-10)
                            # 温度随使用率调整：使用率低时更随机，高时更确定
                            adaptive_temp = self.temperature * (1.0 + usage_ratio.mean())
                            noisy_probs = (available_probs.log() + gumbel_noise) / adaptive_temp
                            expert_idx = noisy_probs.argmax()
                    else:
                        # 推理时：低温缩放 + 容量感知
                        T = self.inference_temperature
                        scaled = (temp_probs * soft_mask).pow(1.0 / T)
                        if scaled.sum() < 1e-8:
                            expert_idx = usage_ratio.argmin()
                        else:
                            scaled = scaled / scaled.sum()
                            expert_idx = torch.multinomial(scaled, 1).item()

                expert_idx = int(expert_idx)
                selected_experts.append(expert_idx)
                expert_counts[expert_idx] += 1

                # 更新溢出惩罚（超过容量后指数增长）
                if expert_counts[expert_idx] > capacity:
                    overflow = (expert_counts[expert_idx] - capacity) / float(capacity)
                    overflow_penalty[expert_idx] = overflow * 10.0  # 强惩罚

                # 将已选专家的概率置零，避免重复选择
                temp_probs[expert_idx] = 0.0

            top_k_indices_list.append(selected_experts)
        
        top_k_indices = torch.tensor(top_k_indices_list, dtype=torch.long, device=flat_probs.device)
        
        router_mask  = torch.zeros_like(flat_probs).scatter_(1, top_k_indices, 1)
        router_probs = torch.zeros_like(flat_probs).scatter_(
            1, top_k_indices, flat_probs.gather(1, top_k_indices))
        router_mask   = router_mask.view(probs.shape)
        router_probs  = router_probs.view(probs.shape)
        top_k_indices = top_k_indices.view(probs.shape[:-1] + (self.top_k,))
        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

    def _format_output(self, router_mask, top_k_indices, router_probs, input_shape):
        router_mask   = router_mask.view(*input_shape[:-1], -1)
        top_k_indices = top_k_indices.view(*input_shape[:-1], -1)
        router_probs  = router_probs.view(*input_shape[:-1], -1)
        if self.normalize:
            s = router_probs.sum(dim=-1, keepdim=True)
            router_probs = router_probs / s.clamp_min(1e-8)
        return router_mask, top_k_indices, router_probs


# ════════════════════════════════════════════════════════════════════════
#  第二部分：脑信号专用适配
# ════════════════════════════════════════════════════════════════════════

def standardize_future(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std  = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def make_small_last_linear(
    linear: nn.Linear,
    scale: float = 0.01,
    bias: float = 0.0,
) -> None:
    """极小随机初始化，破除专家间对称性，保证初始残差接近零。"""
    nn.init.normal_(linear.weight, std=scale)
    nn.init.constant_(linear.bias, bias)


class RicherPathologyProjection(nn.Module):
    """
    病理分数 → pathology_embedding 投影。

    单标量输入（pathology_input_dim=1）时先做多项式特征扩展
    poly = [x, x², sin(πx)] 再投影，保留 HAMD 分数的非线性效应，
    避免路由器无法区分病理亚型；多维输入时退化为两层线性网络。
    输出维度恒为 pathology_dim，与下游模块接口兼容。
    """

    def __init__(self, pathology_input_dim: int = 1, pathology_dim: int = 32):
        super().__init__()
        self.pathology_input_dim = pathology_input_dim

        if pathology_input_dim == 1:
            poly_dim  = 3  # [x, x², sin(πx)]
            inner_dim = max(pathology_dim * 2, 64)
            self.proj = nn.Sequential(
                nn.Linear(poly_dim, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, pathology_dim),
                nn.GELU(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(pathology_input_dim, pathology_dim),
                nn.GELU(),
                nn.Linear(pathology_dim, pathology_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, pathology_input_dim] → [B, pathology_dim]"""
        if self.pathology_input_dim == 1:
            scalar = x[:, 0]
            poly = torch.stack(
                [scalar,
                 scalar ** 2,
                 torch.sin(math.pi * scalar)],
                dim=-1,
            )
            return self.proj(poly)
        # 多维病理输入：直接投影
        return self.proj(x)


class SharedPathologyExpert(nn.Module):
    """
    共享专家（对应 MoDE shared_mlp）：对所有样本始终激活，捕获通用 MDD 病理模式。

    结构与 MoDE shared_mlp 一致（SiLU 门控 MLP）：
        x → norm → gate_proj * SiLU(up_proj) → down_proj → delta

    增加轻量 per-ROI 病理调制（path_gate），初始化为全 1（恒等，不干扰预训练）。
    down_proj 使用 std=0.005，比路由专家更保守，防止共享专家主导初始输出。
    """

    def __init__(
        self,
        features: int,
        in_w: int,
        in_s: int,
        pred_w: int,
        pred_s: int,
        pathology_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features = features
        self.pred_w   = pred_w
        self.pred_s   = pred_s
        self.in_dim   = in_w * in_s
        self.out_dim  = pred_w * pred_s

        self.norm      = nn.LayerNorm(self.in_dim)
        self.gate_proj = nn.Linear(self.in_dim, hidden_dim)
        self.up_proj   = nn.Linear(self.in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, self.out_dim)
        self.act_fn    = nn.SiLU()
        self.dropout   = nn.Dropout(dropout)

        # 轻量 per-ROI 病理调制：初始化 bias=1 → 恒等变换
        self.path_gate = nn.Linear(pathology_dim, features)
        nn.init.zeros_(self.path_gate.weight)
        nn.init.ones_(self.path_gate.bias)

        # 保守初始化：避免共享专家一开始就主导输出
        nn.init.normal_(self.down_proj.weight, std=0.005)
        nn.init.zeros_(self.down_proj.bias)

    def forward(
        self,
        latent: torch.Tensor,
        pathology_embedding: torch.Tensor,
    ) -> torch.Tensor:
        b, f, _, _ = latent.shape
        x     = self.norm(latent.reshape(b, f, -1))
        x     = self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        delta = self.down_proj(x)

        path_scale = self.path_gate(pathology_embedding)  # [B, F]
        delta = delta * path_scale.unsqueeze(-1)

        return delta.view(b, f, self.pred_w, self.pred_s)


class PathologyResidualExpert(nn.Module):
    """
    路由专家（对应 MoDE experts[i]）：预测个体化病理残差。

    FiLM 条件化：以 pathology_embedding 调制融合特征（gamma/beta）。
    输出层 std=0.01：破除专家间对称性，同时保持初始残差接近零。
    expert_id 用于差异化初始化（不同种子/偏移），帮助路由器区分专家。
    """

    def __init__(
        self,
        features: int,
        in_w: int,
        in_s: int,
        pred_w: int,
        pred_s: int,
        pathology_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        expert_id: int = 0,
    ):
        super().__init__()
        self.features = features
        self.pred_w   = pred_w
        self.pred_s   = pred_s
        self.in_dim   = in_w * in_s
        self.out_dim  = pred_w * pred_s
        self.expert_id = expert_id

        self.history_norm = nn.LayerNorm(self.in_dim)
        self.latent_norm  = nn.LayerNorm(self.in_dim)
        self.base_norm    = nn.LayerNorm(self.out_dim)

        self.history_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.latent_proj  = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.base_proj    = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.cross_roi    = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=1),
            nn.GELU(), nn.Dropout(dropout))

        # FiLM 条件化
        self.film = nn.Linear(pathology_dim, hidden_dim * 2)

        # 专家差异化初始化：临时切换 RNG 种子，各专家初始行为不同但接近中性
        original_rng_state = torch.get_rng_state()
        torch.manual_seed(42 + expert_id * 17)

        film_std = 0.01 * (1 + expert_id * 0.05)
        nn.init.normal_(self.film.weight, mean=0.0, std=film_std)
        nn.init.normal_(self.film.bias, mean=0.0, std=film_std)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout))

        self.trend_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pred_w),
        )

        # 不同专家的输出层偏移不同，学习不同的初始趋势模式
        trend_bias_offset = (expert_id - 1.5) * 0.005
        shape_bias_offset = (expert_id - 1.5) * 0.003
        scale_bias_offset = -4.0 + (expert_id - 1.5) * 0.1

        make_small_last_linear(self.trend_head[-1], scale=0.01, bias=trend_bias_offset)
        make_small_last_linear(self.shape_head[-1], scale=0.01, bias=shape_bias_offset)
        make_small_last_linear(self.scale_head[-1], scale=0.01, bias=scale_bias_offset)

        torch.set_rng_state(original_rng_state)

    def forward(
        self,
        latent: torch.Tensor,
        history: torch.Tensor,
        base_pred: torch.Tensor,
        pathology_embedding: torch.Tensor,
    ) -> torch.Tensor:
        b, f, _, _ = latent.shape
        history_flat = history.reshape(b, f, -1)
        latent_flat  = latent.reshape(b, f, -1)
        base_flat    = base_pred.reshape(b, f, -1)

        history_feat = self.history_proj(self.history_norm(history_flat))
        latent_feat  = self.latent_proj(self.latent_norm(latent_flat))
        base_feat    = self.base_proj(self.base_norm(base_flat))
        cross_feat   = self.cross_roi(latent_feat)

        fused = self.fusion(
            torch.cat([history_feat, latent_feat, base_feat, cross_feat], dim=-1))

        # FiLM 条件化
        gamma, beta = self.film(pathology_embedding).chunk(2, dim=-1)
        fused = (fused * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1))
                 + 0.1 * beta.unsqueeze(1))

        trend     = self.trend_head(fused).view(b, f, self.pred_w, self.pred_s)
        shape_raw = self.shape_head(fused).view(b, f, self.pred_w, self.pred_s)
        shape     = standardize_future(shape_raw)
        scale     = F.softplus(self.scale_head(fused)).view(b, f, self.pred_w, 1)
        return trend + scale * shape


# ════════════════════════════════════════════════════════════════════════
#  第三部分：NeuroTwinMoE (MoDE-style main class)
# ════════════════════════════════════════════════════════════════════════

class NeuroTwinMoE(nn.Module):
    """
    Pathology-aware residual MoE (MoDE-style) for NeuroTwin.

    Forward flow (ref. MoDEBlock.forward):
    ─────────────────────────────────────────────────────────────────
    1. Build gate input [B, gate_dim] (5 × F dimensional statistics)
    2. RicherPathologyProjection: polynomial feature expansion
       pathology_embedding [B, pathology_dim]
    3. MoDE routing (RouterCond, pathology as condition)
    4. Routed expert loop
    5. Shared expert (always-on)
    6. IterativePredictionRefiner: multi-round graph convolution refinement
    7. Load balancing term (training only)
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        features: int,
        in_w: int,
        in_s: int,
        pred_w: int,
        pred_s: int,
        pathology_input_dim: int = 1,
        pathology_dim: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.2,
        gate_temperature: float = 2.0,
        expert_hidden_dim: int = 256,
        use_shared_expert: bool = True,
        router_normalize: bool = True,
        router_context_cond_only: bool = True,
        use_argmax: bool = False,
        inference_temperature: float = 0.3,
    ):
        super().__init__()
        self.pred_w              = pred_w
        self.pred_s              = pred_s
        self.features            = features
        self.num_experts         = num_experts
        self.top_k               = top_k
        self.pathology_input_dim = pathology_input_dim
        self.use_shared_expert   = use_shared_expert

        self.gate_input_dim  = 5 * features
        self.gate_input_norm = nn.LayerNorm(self.gate_input_dim)

        # 病理分数投影（输出维度 pathology_dim，与后续模块接口兼容）
        self.pathology_proj = RicherPathologyProjection(
            pathology_input_dim=pathology_input_dim,
            pathology_dim=pathology_dim,
        )

        # MoDE 路由器（RouterCond，以 pathology_embedding 为条件）
        self.router = RouterCond(
            hidden_states=self.gate_input_dim,
            cond_dim=pathology_dim,
            num_experts=num_experts,
            top_k=top_k,
            use_argmax=use_argmax,
            normalize=router_normalize,
            cond_router=True,
            router_context_cond_only=router_context_cond_only,
            temperature=gate_temperature,
            inference_temperature=inference_temperature,
            capacity_factor=1.0,
        )

        # 路由专家字典（MoDEBlock 风格）
        self.experts = nn.ModuleDict({
            f"expert_{i}": PathologyResidualExpert(
                features=features,
                in_w=in_w, in_s=in_s,
                pred_w=pred_w, pred_s=pred_s,
                pathology_dim=pathology_dim,
                hidden_dim=expert_hidden_dim,
                dropout=dropout,
                expert_id=i,
            )
            for i in range(num_experts)
        })

        # 共享专家（始终激活）
        if use_shared_expert:
            self.shared_expert = SharedPathologyExpert(
                features=features,
                in_w=in_w, in_s=in_s,
                pred_w=pred_w, pred_s=pred_s,
                pathology_dim=pathology_dim,
                hidden_dim=expert_hidden_dim,
                dropout=dropout,
            )
        else:
            self.shared_expert = None

        # 2 轮迭代 SC 约束细化（delta 已是残差，比主干的 3 轮更保守）
        self.delta_refiner = IterativePredictionRefiner(
            features=features, n_rounds=2, dropout=dropout)

        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('inference_expert_usage', torch.zeros(num_experts))
        self.total_tokens_processed = 0
        self.routing_probs: Optional[Dict[str, Any]] = None

    def set_router_temperature(self, temperature: float) -> None:
        self.router.set_temperature(temperature)

    def set_router_inference_temperature(self, temperature: float) -> None:
        self.router.set_inference_temperature(temperature)

    def get_expert_usage(self) -> torch.Tensor:
        return self.inference_expert_usage.clone()

    def reset_expert_usage(self) -> None:
        self.expert_usage.zero_()
        self.inference_expert_usage.zero_()
        self.total_tokens_processed = 0

    def _build_gate_input(
        self,
        latent: torch.Tensor,
        history: torch.Tensor,
        base_pred: torch.Tensor,
    ) -> torch.Tensor:
        """构建路由门控输入：5 × F 维统计特征向量 [B, 5*F]"""
        hist_mean   = history.mean(dim=(2, 3))
        hist_std    = history.std(dim=(2, 3), unbiased=False)
        latent_mean = latent.mean(dim=(2, 3))
        latent_std  = latent.std(dim=(2, 3), unbiased=False)
        base_mean   = base_pred.mean(dim=(2, 3))
        return torch.cat(
            [hist_mean, hist_std, latent_mean, latent_std, base_mean], dim=-1)

    def forward(
        self,
        latent: torch.Tensor,
        history: torch.Tensor,
        base_pred: torch.Tensor,
        pathology_score: torch.Tensor,
        sc_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # ── 输入校验 ──────────────────────────────────────────────
        if pathology_score is None:
            raise ValueError("pathology_score cannot be None in NeuroTwinMoE")
        if pathology_score.ndim == 1:
            pathology_score = pathology_score.unsqueeze(-1)
        if pathology_score.shape[-1] != self.pathology_input_dim:
            raise ValueError(
                f"pathology_score last dim {pathology_score.shape[-1]} "
                f"!= pathology_input_dim {self.pathology_input_dim}")

        b = latent.shape[0]

        # ── 1. 特征构建 ────────────────────────────────────────────
        gate_input = self.gate_input_norm(
            self._build_gate_input(latent, history, base_pred))   # [B, gate_dim]
        pathology_embedding = self.pathology_proj(pathology_score)  # [B, pathology_dim]

        # ── 2. MoDE 路由（L=1，样本级路由）──────────────────────────
        gate_input_3d = gate_input.unsqueeze(1)   # [B, 1, gate_dim]
        router_mask, top_k_indices, router_probs, true_probs = self.router(
            gate_input_3d, pathology_embedding)

        mask_2d   = router_mask.squeeze(1)
        probs_2d  = router_probs.squeeze(1)
        topk_2d   = top_k_indices.squeeze(1)
        tprobs_2d = true_probs.squeeze(1)

        # ── 3. 路由专家计算（MoDEBlock 风格循环）──────────────────────
        delta = torch.zeros(
            b, self.features, self.pred_w, self.pred_s,
            device=latent.device, dtype=latent.dtype)

        for idx in range(self.num_experts):
            expert_key    = f"expert_{idx}"
            token_indices = mask_2d[:, idx].bool()
            if not token_indices.any():
                continue
            expert = self.experts[expert_key]
            prob   = probs_2d[token_indices, idx].view(-1, 1, 1, 1)
            expert_out = expert(
                latent[token_indices],
                history[token_indices],
                base_pred[token_indices],
                pathology_embedding[token_indices],
            )
            delta[token_indices] = delta[token_indices] + prob * expert_out

            if self.training:
                self.expert_usage[idx] += token_indices.sum().item()
            else:
                self.inference_expert_usage[idx] += token_indices.sum().item()

        # ── 4. 共享专家（always-on）──────────────────────────────────
        if self.use_shared_expert and self.shared_expert is not None:
            delta = delta + self.shared_expert(latent, pathology_embedding)

        # ── 5. 迭代图结构后处理（2 轮）─────────────────────────────
        delta = delta + self.delta_refiner(delta, sc_matrix)

        # ── 6. 负载均衡项（Switch Transformer 风格）──────────────────
        load_balancing_term: Optional[torch.Tensor] = None
        if self.training:
            # importance：路由器分配给各专家的平均概率（soft gate）
            importance = tprobs_2d.mean(0)

            # load：实际被路由到各专家的样本比例（hard top-k）
            topk_oh = F.one_hot(topk_2d, num_classes=self.num_experts)
            load = topk_oh.float().sum(dim=(0, 1)) / float(max(1, b * self.top_k))

            # importance 与 load 一致（均匀路由）时损失最小
            load_balancing_term = self.num_experts * (importance * load).sum()

            # 熵奖励：惩罚低熵（确定性）路由，促进探索
            router_entropy = -(tprobs_2d * (tprobs_2d + 1e-10).log()).sum(dim=-1).mean()
            load_balancing_term = load_balancing_term - 0.01 * router_entropy

            self.routing_probs = {
                'probs':               tprobs_2d,
                'top_k_hot':           mask_2d,
                'load_balancing_term': load_balancing_term,
                'importance_per_expert': importance.detach(),
                'load_per_expert': load.detach(),
                'router_entropy': router_entropy.detach(),
            }

        self.total_tokens_processed += b

        # ── 7. 构建 aux_info ──────────────────────────────────────
        with torch.no_grad():
            topk_oh = F.one_hot(
                topk_2d, num_classes=self.num_experts).float()
            load      = topk_oh.sum(dim=(0, 1)) / float(max(1, b * self.top_k))
            importance = tprobs_2d.mean(0)
            selection_frequency = mask_2d.float().mean(0)
            hard_gates = torch.zeros(
                b, self.num_experts,
                device=latent.device, dtype=latent.dtype,
            ).scatter_(1, topk_2d, probs_2d.gather(1, topk_2d))

        aux_info = {
            'gates':               hard_gates,
            'soft_gates':          tprobs_2d,
            'top_k_indices':       topk_2d,
            'gate_logits':         (self.router.logits.view(b, -1)
                                    if self.router.logits is not None
                                    else None),
            'importance':          importance,
            'load':                load,
            'selection_frequency': selection_frequency,
            'load_balancing_term': (load_balancing_term.detach()
                                    if load_balancing_term is not None
                                    else torch.zeros(1, device=latent.device)),
        }
        return delta, aux_info
