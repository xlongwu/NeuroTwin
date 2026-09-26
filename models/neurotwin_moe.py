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


def _activation(name: str) -> nn.Module:
    """按名称返回无参数激活模块。

    异构专家（expert_kind='heterogeneous'）用不同激活拉开专家间差异；
    默认 'gelu' 与历史实现完全一致。
    """
    key = str(name).lower()
    if key == 'gelu':
        return nn.GELU()
    if key == 'silu':
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}', expected 'gelu' or 'silu'")


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
    - 推理（`eval_mode='dense_soft'`，默认）：对**全部专家**做 temperature=1.0
      的 softmax 并加权求和，不做容量循环、不做 multinomial 采样，因此评估
      指标完全确定；同权重两次 evaluate() 逐位一致，消除“验证指标随机 →
      最佳权重选择带噪声”的问题
    - 推理（`eval_mode='topk'`）：确定性的 hard top-k 路由
    - 推理（`eval_mode='legacy'`）：保留历史低温幂次缩放 + multinomial 采样，
      仅用于复现旧结果
    - use_argmax=True（部署）：硬 top-k 确定性路由（容量感知）

    capacity_factor 限制每个专家在一个 batch 中最多处理的样本数，
    超出容量的专家会被软惩罚，防止过载。
    """

    EVAL_MODES = ('dense_soft', 'topk', 'legacy')

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
        eval_mode: str = 'dense_soft',
    ):
        super().__init__()
        if eval_mode not in self.EVAL_MODES:
            raise ValueError(f"Unsupported eval_mode '{eval_mode}', expected one of {self.EVAL_MODES}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        self.temperature = max(1e-4, float(temperature))
        self.use_argmax = use_argmax
        self.cond_router = cond_router
        self.router_context_cond_only = router_context_cond_only
        self.inference_temperature = max(1e-4, float(inference_temperature))
        self.capacity_factor = capacity_factor
        self.eval_mode = eval_mode

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

    def set_eval_mode(self, eval_mode: str) -> None:
        if eval_mode not in self.EVAL_MODES:
            raise ValueError(f"Unsupported eval_mode '{eval_mode}', expected one of {self.EVAL_MODES}")
        self.eval_mode = eval_mode

    def forward(
        self,
        inputs: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        eval_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = inputs.size()
        mode = eval_mode if eval_mode is not None else self.eval_mode

        if (not self.training) and mode == 'dense_soft':
            # 确定性评估：全部专家参与，softmax(logits / 1.0) 加权
            logits = self._compute_logits(inputs, cond)
            probs = self._compute_probabilities(logits, temperature=1.0)
            router_mask, top_k_indices, router_probs = self._select_dense_soft(probs, input_shape)
            return router_mask, top_k_indices, router_probs, probs.view(*input_shape[:-1], -1)

        if (not self.training) and mode == 'topk':
            logits = self._compute_logits(inputs, cond)
            probs = self._compute_probabilities(logits, temperature=1.0)
            router_mask, top_k_indices, router_probs = self._select_topk(probs, input_shape)
            return router_mask, top_k_indices, router_probs, probs.view(*input_shape[:-1], -1)

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

    def _compute_probabilities(self, logits, temperature=None):
        temp = self.temperature if temperature is None else max(1e-4, float(temperature))
        logits = (logits - logits.max(dim=-1, keepdim=True).values) / temp
        self.logits = logits
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-9, max=1 - 1e-9)
        if not torch.isfinite(probs).all():
            logger.warning("Router probabilities contain inf or NaN")
        return probs

    def _select_dense_soft(self, probs, input_shape):
        """确定性全专家路由：所有专家均被激活，权重为完整 softmax 概率。"""
        flat = probs.view(-1, probs.size(-1))
        k = min(self.top_k, flat.size(-1))
        top_k_indices = flat.topk(k, dim=-1).indices
        router_mask   = torch.ones_like(flat)
        router_probs  = flat
        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

    def _select_topk(self, probs, input_shape):
        """确定性 hard top-k 路由（无容量循环、无采样）。"""
        flat = probs.view(-1, probs.size(-1))
        k = min(self.top_k, flat.size(-1))
        top_k_indices = flat.topk(k, dim=-1).indices
        router_mask   = torch.zeros_like(flat).scatter_(1, top_k_indices, 1)
        router_probs  = torch.zeros_like(flat).scatter_(
            1, top_k_indices, flat.gather(1, top_k_indices))
        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

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
                        # 按可用概率多项式采样（MoDE 设计行为）。此前的
                        # Gumbel-argmax + 自适应高温会把选择近似为与条件无关的
                        # 均匀随机，导致专家的 FiLM 条件调制与路由器的条件→专家
                        # 映射同时失去梯度信号（HAMD 条件注入失效的根因）；
                        # multinomial 保留「概率→选择」的保真通路，
                        # 温度调度仍通过 probs 本身控制探索程度。
                        available_probs = temp_probs * soft_mask

                        # 如果所有专家都接近满负荷，允许轻微超载
                        if available_probs.sum() < 1e-8:
                            # 选择使用率最低的专家
                            expert_idx = usage_ratio.argmin()
                        else:
                            available_probs = available_probs / available_probs.sum()
                            expert_idx = torch.multinomial(available_probs, 1).item()
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
    病理条件向量 → pathology_embedding 投影。

    输入默认来自 `PathologyNormalizer`（已完成归一化，含稳健 z-score / 经验 CDF /
    分位数变换等模式），本模块只负责**非线性升维**，因此默认使用两层 MLP，
    不再默认做多项式特征扩展。

    历史行为 `poly = [x, x², sin(πx)]` 仅在显式 `use_poly=True` 且输入为单标量时
    保留，用于复现旧结果；由于 raw 分数直接进入原始多项式会与归一化模块的职责
    重叠（且 sin(πx) 对 HAMD 取值范围没有物理依据），默认关闭。
    输出维度恒为 pathology_dim，与下游模块接口兼容。
    """

    def __init__(self, pathology_input_dim: int = 1, pathology_dim: int = 32,
                 use_poly: bool = False):
        super().__init__()
        self.pathology_input_dim = pathology_input_dim
        self.use_poly = bool(use_poly) and pathology_input_dim == 1

        in_dim = 3 if self.use_poly else pathology_input_dim  # [x, x², sin(πx)]
        inner_dim = max(pathology_dim * 2, 64)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, pathology_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, pathology_input_dim] → [B, pathology_dim]"""
        if self.use_poly:
            scalar = x[:, 0]
            x = torch.stack(
                [scalar,
                 scalar ** 2,
                 torch.sin(math.pi * scalar)],
                dim=-1,
            )
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
    act 指定隐层激活（默认 gelu），供异构专家（不同宽度 + 不同激活）使用。
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
        act: str = 'gelu',
    ):
        super().__init__()
        self.features = features
        self.pred_w   = pred_w
        self.pred_s   = pred_s
        self.in_dim   = in_w * in_s
        self.out_dim  = pred_w * pred_s
        self.expert_id = expert_id
        self.act_name  = str(act).lower()

        self.history_norm = nn.LayerNorm(self.in_dim)
        self.latent_norm  = nn.LayerNorm(self.in_dim)
        self.base_norm    = nn.LayerNorm(self.out_dim)

        self.history_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), _activation(self.act_name),
            nn.Dropout(dropout))
        self.latent_proj  = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), _activation(self.act_name),
            nn.Dropout(dropout))
        self.base_proj    = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim), _activation(self.act_name),
            nn.Dropout(dropout))
        self.cross_roi    = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=1),
            _activation(self.act_name), nn.Dropout(dropout))

        # FiLM 条件化
        self.film = nn.Linear(pathology_dim, hidden_dim * 2)

        # 专家差异化初始化：临时切换 RNG 种子，各专家初始行为不同但接近中性
        original_rng_state = torch.get_rng_state()
        torch.manual_seed(42 + expert_id * 17)

        film_std = 0.01 * (1 + expert_id * 0.05)
        nn.init.normal_(self.film.weight, mean=0.0, std=film_std)
        nn.init.normal_(self.film.bias, mean=0.0, std=film_std)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), _activation(self.act_name),
            nn.Dropout(dropout))

        self.trend_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), _activation(self.act_name),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), _activation(self.act_name),
            nn.Dropout(dropout),
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
    1. Build gate input（默认 5 项状态统计 + RevIN mean/std）
    2. RicherPathologyProjection: 病理条件投影
    3. MoDE routing (RouterCond, pathology as condition)
    4. Routed expert loop（sample 级或 token 级）
    5. Shared expert（可选，always-on）
    6. IterativePredictionRefiner: multi-round graph convolution refinement
    7. Load balancing term (training only)
    ─────────────────────────────────────────────────────────────────

    消融/扩展开关（默认值保证与旧行为数值一致）：
      - ``gate_features``：门控输入特征集合。
        ``'state'`` 为旧的 5 项统计；``'state_revin'``（默认）追加 RevIN
        mean/std（detach）；``'state_revin_difficulty'`` 再追加"难度"项
        （base 预测均值与最后一个历史窗均值的绝对差）。
      - ``experts_mode``：``'routed_shared'``（默认）/``'shared_only'``/
        ``'routed_only'``/``'none'``。``'none'`` 时不建路由与专家，残差恒零
        且跳过 refiner，得到"关闭 MoE 残差分支"的干净消融。
      - ``route_level``：``'sample'``（默认，每样本一个路由决策）/
        ``'token'``（每个 (样本, ROI) 一个决策）。token 级必须对整批样本整批
        计算每个专家再用逐 token 概率加权（``cross_roi`` 以 ROI 为通道，无法按
        token 切片），因此**计算量不随稀疏度下降**。
      - ``expert_kind``：``'homogeneous'``（默认，所有专家同宽同激活）/
        ``'heterogeneous'``（隐层宽度随编号 0.5×~1.0×，激活 SiLU/GELU 交替）。
      - ``eval_mc_samples``：评估期额外做 N 次 legacy 随机路由，报告路由概率
        标准差 ``moe_router_prob_std``；0（默认）不产生额外开销。
      - ``expert_stats_interval``：每 N 次前向采集一次专家输出范数与跨窗路由
        一致性，写入 ``aux_info['expert_diag']``；0（默认）关闭。开启时该次前向
        会对整批样本额外跑每个专家一次，仅用于诊断。
    """

    GATE_FEATURES = ('state', 'state_revin', 'state_revin_difficulty')
    EXPERTS_MODES = ('routed_shared', 'shared_only', 'routed_only', 'none')
    EXPERT_KINDS  = ('homogeneous', 'heterogeneous')
    ROUTE_LEVELS  = ('sample', 'token')

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
        eval_mode: str = 'dense_soft',
        pathology_poly_expansion: bool = False,
        delta_refiner_rounds: int = 2,
        refiner_adaptive: bool = False,
        # ── Phase 5：门控特征 / 专家臂 / 路由粒度 消融 ──
        gate_features: str = 'state_revin',
        gate_input_dim: int = None,
        experts_mode: str = None,
        expert_kind: str = 'homogeneous',
        route_level: str = 'sample',
        eval_mc_samples: int = 0,
        expert_stats_interval: int = 0,
    ):
        super().__init__()
        if gate_features not in self.GATE_FEATURES:
            raise ValueError(
                f"Unsupported gate_features '{gate_features}', "
                f"expected one of {self.GATE_FEATURES}")
        if expert_kind not in self.EXPERT_KINDS:
            raise ValueError(
                f"Unsupported expert_kind '{expert_kind}', "
                f"expected one of {self.EXPERT_KINDS}")
        if route_level not in self.ROUTE_LEVELS:
            raise ValueError(
                f"Unsupported route_level '{route_level}', "
                f"expected one of {self.ROUTE_LEVELS}")
        if experts_mode is None:
            # 未显式指定时按旧的 use_shared_expert 派生，保持调用方行为不变
            experts_mode = 'routed_shared' if use_shared_expert else 'routed_only'
        if experts_mode not in self.EXPERTS_MODES:
            raise ValueError(
                f"Unsupported experts_mode '{experts_mode}', "
                f"expected one of {self.EXPERTS_MODES}")
        if int(eval_mc_samples) < 0 or int(expert_stats_interval) < 0:
            raise ValueError(
                "eval_mc_samples / expert_stats_interval 必须 >= 0，"
                f"收到 {eval_mc_samples} / {expert_stats_interval}")

        self.pred_w              = pred_w
        self.pred_s              = pred_s
        self.features            = features
        self.num_experts         = num_experts
        self.top_k               = top_k
        self.pathology_input_dim = pathology_input_dim
        self.gate_features       = gate_features
        self.experts_mode        = experts_mode
        self.expert_kind         = expert_kind
        self.route_level         = route_level
        self.eval_mc_samples     = int(eval_mc_samples)
        self.expert_stats_interval = int(expert_stats_interval)

        # 专家臂开关（由 experts_mode 派生）
        self.has_routed = experts_mode in ('routed_shared', 'routed_only')
        self.has_shared = experts_mode in ('routed_shared', 'shared_only')
        self.use_shared_expert = self.has_shared       # 兼容旧属性名

        # 门控宽度：5 项状态统计 (+2 RevIN) (+1 难度)，逐 ROI 各一份
        self.parts_per_roi = (5
                              + (2 if gate_features != 'state' else 0)
                              + (1 if gate_features == 'state_revin_difficulty' else 0))
        derived_dim = self.parts_per_roi * features
        if gate_input_dim is None:
            self.gate_input_dim = derived_dim
        else:
            self.gate_input_dim = int(gate_input_dim)
            if self.gate_input_dim != derived_dim:
                raise ValueError(
                    f"gate_input_dim={self.gate_input_dim} 与 gate_features="
                    f"'{gate_features}' 推导出的 {derived_dim}"
                    f"（{self.parts_per_roi}×{features}）不一致")
        # experts_mode='none' 时门控与条件投影都不会被调用，
        # 不构建这些模块以保证消融参数表干净（参数量为 0）
        if experts_mode == 'none':
            self.gate_input_norm = None
            self.gate_token_norm = None
            self.pathology_proj = None
        else:
            self.gate_input_norm = nn.LayerNorm(self.gate_input_dim)
            # token 级路由逐 ROI 决策，需对 P 项特征单独归一化
            self.gate_token_norm = nn.LayerNorm(self.parts_per_roi)
            # 病理条件投影（输入为 PathologyNormalizer 的输出向量）
            self.pathology_proj = RicherPathologyProjection(
                pathology_input_dim=pathology_input_dim,
                pathology_dim=pathology_dim,
                use_poly=pathology_poly_expansion,
            )

        if self.has_routed:
            if router_context_cond_only and route_level == 'token':
                logger.warning(
                    "route_level='token' 与 router_context_cond_only=True 同时启用时，"
                    "所有 token 共享同一病理条件，token 级路由会退化为样本级；"
                    "建议使用 --moe_router_cond_only False。")
            # 路由器输入宽度：sample 级为展平后的 F×P；token 级为逐 ROI 的 P
            router_hidden = (self.gate_input_dim if route_level == 'sample'
                             else self.parts_per_roi)
            self.router = RouterCond(
                hidden_states=router_hidden,
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
                eval_mode=eval_mode,
            )
            self.experts = nn.ModuleDict({
                f"expert_{i}": PathologyResidualExpert(
                    features=features,
                    in_w=in_w, in_s=in_s,
                    pred_w=pred_w, pred_s=pred_s,
                    pathology_dim=pathology_dim,
                    hidden_dim=self._expert_hidden_dim(i, expert_hidden_dim),
                    dropout=dropout,
                    expert_id=i,
                    act=self._expert_act(i),
                )
                for i in range(num_experts)
            })
        else:
            self.router = None
            self.experts = None

        # 共享专家（experts_mode 允许时始终激活）
        if self.has_shared:
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

        # 迭代 SC 约束细化（delta 已是残差，轮数默认比主干更保守）
        # experts_mode='none' 时残差恒零，细化器一并跳过，避免空转
        if experts_mode == 'none':
            self.delta_refiner = None
        else:
            self.delta_refiner = IterativePredictionRefiner(
                features=features, n_rounds=delta_refiner_rounds, dropout=dropout,
                adaptive=refiner_adaptive)

        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('inference_expert_usage', torch.zeros(num_experts))
        self.total_tokens_processed = 0
        self.routing_probs: Optional[Dict[str, Any]] = None
        self.last_expert_diag: Optional[Dict[str, float]] = None
        self._fwd_calls = 0

    # ── 专家结构（异构时按编号变化）───────────────────────────────────
    def _expert_hidden_dim(self, idx: int, base: int) -> int:
        if self.expert_kind == 'homogeneous':
            return base
        ratio = 0.5 + 0.5 * (idx / max(1, self.num_experts - 1))
        return max(16, int(base * ratio))

    def _expert_act(self, idx: int) -> str:
        if self.expert_kind == 'homogeneous':
            return 'gelu'
        return 'silu' if idx % 2 == 0 else 'gelu'

    def set_router_temperature(self, temperature: float) -> None:
        if self.router is not None:
            self.router.set_temperature(temperature)

    def set_router_inference_temperature(self, temperature: float) -> None:
        if self.router is not None:
            self.router.set_inference_temperature(temperature)

    def get_expert_usage(self) -> torch.Tensor:
        return self.inference_expert_usage.clone()

    def reset_expert_usage(self) -> None:
        self.expert_usage.zero_()
        self.inference_expert_usage.zero_()
        self.total_tokens_processed = 0

    # ── 门控输入构造 ─────────────────────────────────────────────────
    @staticmethod
    def _flatten_parts(parts) -> torch.Tensor:
        """P 个 [B,F] 统计量按 p-major 顺序展平为 [B,P*F]（与旧 cat 顺序一致）。"""
        stacked = torch.stack(list(parts), dim=1)          # [B, P, F]
        return stacked.reshape(stacked.shape[0], -1)

    def _extra_gate_parts(self, base_mean: torch.Tensor, history: torch.Tensor,
                          revin_stats=None):
        """附加门控项：RevIN mean/std 与"难度"（均为 detach 的监控性特征）。"""
        extra = []
        if self.gate_features != 'state':
            if revin_stats is not None:
                mean, stdev = revin_stats
                mean  = mean.detach().reshape(mean.shape[0], -1)
                stdev = stdev.detach().reshape(stdev.shape[0], -1)
            else:
                # --norm False 等无 RevIN 统计的场景：补零保持门控宽度稳定
                mean  = torch.zeros_like(base_mean)
                stdev = torch.zeros_like(base_mean)
            extra.extend([mean, stdev])
        if self.gate_features == 'state_revin_difficulty':
            # 难度：base 预测均值与该样本最后历史窗均值的绝对偏差，越小越"容易"
            last_hist_mean = history[:, :, -1, :].mean(dim=-1)
            extra.append((base_mean - last_hist_mean).abs().detach())
        return extra

    def _build_gate_parts(self, latent: torch.Tensor, history: torch.Tensor,
                          base_pred: torch.Tensor, revin_stats=None):
        """返回 P 个 [B,F] 门控统计量（顺序固定，展平后与旧 5F 顺序一致）。"""
        base_mean = base_pred.mean(dim=(2, 3))
        parts = [
            history.mean(dim=(2, 3)),
            history.std(dim=(2, 3), unbiased=False),
            latent.mean(dim=(2, 3)),
            latent.std(dim=(2, 3), unbiased=False),
            base_mean,
        ]
        parts.extend(self._extra_gate_parts(base_mean, history, revin_stats))
        return parts

    def _build_gate_input(self, latent: torch.Tensor, history: torch.Tensor,
                          base_pred: torch.Tensor, revin_stats=None) -> torch.Tensor:
        """构建展平门控输入 [B, gate_input_dim]（sample 级路由使用）。"""
        return self._flatten_parts(
            self._build_gate_parts(latent, history, base_pred, revin_stats))

    def _router_input_from_parts(self, parts) -> torch.Tensor:
        """按 route_level 构造路由器输入：sample → [B,1,F×P]，token → [B,F,P]。"""
        if self.route_level == 'token':
            return self.gate_token_norm(torch.stack(list(parts), dim=-1))
        return self.gate_input_norm(self._flatten_parts(parts)).unsqueeze(1)

    def forward(
        self,
        latent: torch.Tensor,
        history: torch.Tensor,
        base_pred: torch.Tensor,
        pathology_score: torch.Tensor,
        sc_matrix: torch.Tensor,
        sc_adj: torch.Tensor = None,
        revin_stats=None,
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
        self._fwd_calls += 1

        # ── 0. 干净消融：关闭整个 MoE 残差分支 ──────────────────────
        if self.experts_mode == 'none':
            delta = torch.zeros(
                b, self.features, self.pred_w, self.pred_s,
                device=latent.device, dtype=latent.dtype)
            return delta, {}

        # ── 1. 特征构建 ────────────────────────────────────────────
        pathology_embedding = self.pathology_proj(pathology_score)  # [B, pathology_dim]
        parts = self._build_gate_parts(latent, history, base_pred, revin_stats)

        router_input = self._router_input_from_parts(parts)

        # ── 2. MoDE 路由 ───────────────────────────────────────────
        gate_logits = None
        mask_view = probs_view = topk_view = tprobs_view = None
        if self.has_routed:
            router_mask, top_k_indices, router_probs, true_probs = self.router(
                router_input, pathology_embedding)
            if self.route_level == 'token':
                mask_view, probs_view   = router_mask, router_probs
                topk_view, tprobs_view  = top_k_indices, true_probs
            else:
                mask_view, probs_view   = router_mask.squeeze(1), router_probs.squeeze(1)
                topk_view, tprobs_view  = top_k_indices.squeeze(1), true_probs.squeeze(1)
            # 先固化路由 logits：后续每步诊断/采样都会覆写 router.logits
            raw_logits = self.router.logits
            if raw_logits is not None:
                gate_logits = raw_logits.reshape(tprobs_view.shape)

        # ── 3. 路由专家计算 ───────────────────────────────────────
        delta = torch.zeros(
            b, self.features, self.pred_w, self.pred_s,
            device=latent.device, dtype=latent.dtype)

        if self.has_routed and self.route_level == 'token':
            # token 级：cross_roi 以 ROI 为通道，无法按 token 切片，
            # 因此对整批样本整批算一次每个专家，再按逐 token 概率加权聚合
            for idx in range(self.num_experts):
                expert = self.experts[f"expert_{idx}"]
                expert_out = expert(latent, history, base_pred, pathology_embedding)
                weight = probs_view[:, :, idx].unsqueeze(-1).unsqueeze(-1)
                delta = delta + weight * expert_out
                n_active = int(mask_view[:, :, idx].sum().item())
                if self.training:
                    self.expert_usage[idx] += n_active
                else:
                    self.inference_expert_usage[idx] += n_active
        elif self.has_routed:
            for idx in range(self.num_experts):
                token_indices = mask_view[:, idx].bool()
                if not token_indices.any():
                    continue
                expert = self.experts[f"expert_{idx}"]
                prob   = probs_view[token_indices, idx].view(-1, 1, 1, 1)
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
        if self.shared_expert is not None:
            delta = delta + self.shared_expert(latent, pathology_embedding)

        # ── 5. 迭代图结构后处理 ─────────────────────────────────────
        # sc_adj 为上游共享的 SC 软先验 A_eff；为 None 时退回旧 SC 归一化路径
        if self.delta_refiner is not None:
            delta = delta + self.delta_refiner(delta, sc_matrix, adj=sc_adj)

        # ── 6. 负载均衡项（Switch Transformer 风格）──────────────────
        load_balancing_term: Optional[torch.Tensor] = None
        if self.has_routed and self.training:
            flat_probs = tprobs_view.reshape(-1, self.num_experts)   # [N, E]
            n_tokens = float(max(1, flat_probs.shape[0]))
            # importance：路由器分配给各专家的平均概率（soft gate，按 token 求平均）
            importance = flat_probs.mean(0)
            # load：实际被路由到各专家的 token 比例（hard top-k）
            topk_oh = F.one_hot(topk_view, num_classes=self.num_experts)
            load = (topk_oh.float().sum(dim=tuple(range(topk_oh.ndim - 1)))
                    / (n_tokens * self.top_k))

            # importance 与 load 一致（均匀路由）时损失最小
            load_balancing_term = self.num_experts * (importance * load).sum()

            # 熵奖励：惩罚低熵（确定性）路由，促进探索
            router_entropy = -(tprobs_view * (tprobs_view + 1e-10).log()).sum(dim=-1).mean()
            load_balancing_term = load_balancing_term - 0.01 * router_entropy

            self.routing_probs = {
                'probs':               tprobs_view,
                'top_k_hot':           mask_view,
                'load_balancing_term': load_balancing_term,
                'importance_per_expert': importance.detach(),
                'load_per_expert': load.detach(),
                'router_entropy': router_entropy.detach(),
            }

        self.total_tokens_processed += b

        # ── 7. 构建 aux_info ──────────────────────────────────────
        aux_info: Dict[str, Any] = {}
        if self.has_routed:
            with torch.no_grad():
                flat_probs = tprobs_view.reshape(-1, self.num_experts)   # [N, E]
                n_tokens = float(max(1, flat_probs.shape[0]))
                topk_oh = F.one_hot(topk_view, num_classes=self.num_experts).float()
                load = (topk_oh.sum(dim=tuple(range(topk_oh.ndim - 1)))
                        / (n_tokens * self.top_k))
                importance = flat_probs.mean(0)
                selection_frequency = mask_view.reshape(
                    -1, self.num_experts).float().mean(0)
                hard_gates = torch.zeros_like(probs_view).scatter_(
                    -1, topk_view, probs_view.gather(-1, topk_view))

            aux_info.update({
                'gates':               hard_gates,
                'soft_gates':          tprobs_view,
                'top_k_indices':       topk_view,
                'gate_logits':         gate_logits,
                'importance':          importance,
                'load':                load,
                'selection_frequency': selection_frequency,
                'load_balancing_term': (load_balancing_term.detach()
                                        if load_balancing_term is not None
                                        else torch.zeros(1, device=latent.device)),
            })

        # ── 8. 评估期路由不确定性（可选，纯监控）──────────────────────
        if self.has_routed and self.eval_mc_samples > 0 and not self.training:
            with torch.no_grad():
                samples = []
                for _ in range(self.eval_mc_samples):
                    _, _, _, mc_probs = self.router(
                        router_input, pathology_embedding, eval_mode='legacy')
                    samples.append(mc_probs.reshape(tprobs_view.shape))
                aux_info['moe_router_prob_std'] = torch.stack(
                    samples, dim=0).std(dim=0, unbiased=False).mean()

        # ── 9. 专家诊断（可选，按间隔采样）──────────────────────────
        if (self.expert_stats_interval > 0
                and self._fwd_calls % self.expert_stats_interval == 0):
            diag = self._collect_expert_diag(
                latent, history, base_pred, pathology_embedding, revin_stats)
            if diag:
                self.last_expert_diag = diag
                aux_info['expert_diag'] = diag

        return delta, aux_info

    @torch.no_grad()
    def _collect_expert_diag(self, latent, history, base_pred, pathology_embedding,
                             revin_stats=None) -> Dict[str, float]:
        """专家输出范数与跨窗路由一致性（纯监控量，不参与反传）。

        为覆盖全部专家，这里对**整批样本**各跑一次专家前向，仅在
        ``--moe_expert_stats_interval > 0`` 时按间隔触发。
        """
        diag: Dict[str, float] = {}
        if self.experts is not None:
            for idx in range(self.num_experts):
                out = self.experts[f"expert_{idx}"](
                    latent, history, base_pred, pathology_embedding)
                per_sample = out.flatten(1).norm(dim=1)
                diag[f'expert_out_norm_e{idx}'] = float(per_sample.mean().item())
        if self.router is not None:
            consistency = self._window_route_consistency(
                latent, history, base_pred, pathology_embedding, revin_stats)
            if consistency is not None:
                diag['router_window_consistency'] = consistency
        return diag

    @torch.no_grad()
    def _window_route_consistency(self, latent, history, base_pred,
                                  pathology_embedding, revin_stats=None):
        """跨窗路由一致性：逐窗构造门控向量 → 复用同一路由器 → 窗间平均余弦。

        逐窗统计只取窗口相关项（该窗的 history/latent 均值方差 + 全局 base 均值），
        门控宽度与主路径一致，因此不需要额外参数。训练态下路由本身是随机采样，
        该指标只在评估期严格可比。窗数 < 2 时返回 None。
        """
        n_w = history.shape[2]
        if n_w < 2:
            return None
        base_mean = base_pred.mean(dim=(2, 3))
        extra = self._extra_gate_parts(base_mean, history, revin_stats)
        probs_per_window = []
        for w in range(n_w):
            hist_w   = history[:, :, w, :]
            latent_w = latent[:, :, w, :]
            parts = [hist_w.mean(dim=-1), hist_w.std(dim=-1, unbiased=False),
                     latent_w.mean(dim=-1), latent_w.std(dim=-1, unbiased=False),
                     base_mean]
            parts.extend(extra)
            vec = self._router_input_from_parts(parts)
            _, _, _, w_probs = self.router(vec, pathology_embedding)
            probs_per_window.append(w_probs)                     # [B,1,E] 或 [B,F,E]
        probs = torch.stack(probs_per_window, dim=1)             # [B, W, T, E]
        normed = F.normalize(probs, dim=-1).transpose(1, 2)      # [B, T, W, E]
        sim = normed @ normed.transpose(-1, -2)                  # [B, T, W, W]
        i, j = torch.triu_indices(n_w, n_w, offset=1, device=probs.device)
        return float(sim[:, :, i, j].mean().item())
