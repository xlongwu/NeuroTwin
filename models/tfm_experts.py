# coding=utf-8
"""TFM 病理残差专家模块（RECC 方案 §6 / §10–§14 / §32–§35）。

NeuroTwin-TFM 的病理条件化新路径：One-Step 头拆分为::

    z_(t+1) = z_t + Δz_shared（共享 HC 动力学转移，不使用 HAMD）
            + Δz_path  （病理残差专家 MoE：HAMD + 脑状态 → Router 加权）

设计要点（方案章节号对应）：
- Router 同时输入病理条件与全脑状态描述子（§8：[μ_t, σ_t, μ_g, σ_g] ∈ R^4D），
  避免退化成 HAMD 严重程度分箱（§7）；
- Expert 只处理逐 ROI latent ``[z_t, z_g]``，输出 latent 修正 Δz_i（§10/§34），
  不直接访问 raw BOLD（temporal 表示由 backbone 负责、结构信息由 A_eff 负责）；
- Expert 末层小初始化（std=1e-3，bias=0）：初始 Δz_path ≈ 0，MDD finetune 从
  HC 动力学附近平稳起步（§12）；各 Expert 的首层在全局 RNG 下独立初始化，
  保证初始修正方向互不相同；
- 训练前 ``dense_warmup_epochs`` 个 epoch 使用 dense soft mixture，之后切换
  Top-K 稀疏混合，避免早期 Router 随机选择 → 专家饿死 → collapse（§14）；
- RECC 耦合分支（§21/§33）：``coupling_probs`` 在 **detach 后**的 state/cond 上
  用同一 Router 计算完整 soft 分布，使 KL 梯度只更新 Router 参数。

本文件不依赖 models/tfm.py（由 tfm.py 单向导入，避免循环导入）。
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ZeroInitFiLM(nn.Module):
    """条件 → 逐通道 (scale, shift) 的 FiLM 调制，投影层零初始化（初始恒等）。

    与 models/tfm.py 的 ConditionFiLM 同构，但作用宽度与归属模块不同
    （Expert 隐层宽度，方案 §11），且不携带 ``is_conditioning_adapter`` 标记——
    冻结逻辑由外层 TFMPathologyMoE 统一声明，避免重复发现子模块。
    """

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
        """cond [B, C] → (scale [B, dim], shift [B, dim])。"""
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return scale, shift


class TFMPathologyRouter(nn.Module):
    """病理专家 Router（方案 §9）::

        LayerNorm(4D + C) → Linear → router_hidden → GELU → Dropout → Linear → E

    输入为全脑状态描述子 ``s ∈ R^4D`` 拼接病理条件 ``c``；输出 E 个专家上的
    logits / soft_probs（温度缩放 softmax）/ routing_weights / topk_indices。

    routing_weights 语义（§14/§33）：
      - dense（warmup 期或 top_k ≥ E）：等于 soft_probs；
      - sparse：Top-K 掩码 × soft_probs 后在选中集合内重新归一化（和为 1，
        未选中专家权重恰为 0，不参与 mixture 梯度）。
    """

    def __init__(self, state_dim: int, cond_dim: int, num_experts: int = 4,
                 hidden_dim: int = 128, temperature: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"num_experts 必须 >= 2，收到 {num_experts}")
        if temperature <= 0:
            raise ValueError(f"router_temperature 必须 > 0，收到 {temperature}")
        self.state_dim = int(state_dim)
        self.cond_dim = int(max(0, cond_dim))
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        in_dim = self.state_dim + self.cond_dim
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), self.num_experts))

    def _logits(self, state: torch.Tensor,
                cond: Optional[torch.Tensor]) -> torch.Tensor:
        """[B, 4D] (+ [B, C]) → logits [B, E]。"""
        if cond is not None:
            state = torch.cat([state, cond], dim=-1)
        return self.mlp(self.input_norm(state))

    def forward(self, state: torch.Tensor, cond: Optional[torch.Tensor],
                top_k: Optional[int] = None,
                dense: bool = False) -> Dict[str, torch.Tensor]:
        """主路由（mixture 权重；state/cond 携带梯度，任务损失可回传 Router）。"""
        logits = self._logits(state, cond)
        soft_probs = F.softmax(logits / self.temperature, dim=-1)
        k_eff = int(min(self.num_experts, top_k)) if top_k else self.num_experts
        topk_indices = torch.topk(soft_probs, k_eff, dim=-1).indices  # [B, K]
        if dense or k_eff >= self.num_experts:
            routing_weights = soft_probs
        else:
            mask = torch.zeros_like(soft_probs).scatter_(1, topk_indices, 1.0)
            routing_weights = soft_probs * mask
            routing_weights = routing_weights / \
                routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {'logits': logits, 'soft_probs': soft_probs,
                'routing_weights': routing_weights,
                'topk_indices': topk_indices}

    def coupling_probs(self, state_detached: torch.Tensor,
                       cond_detached: Optional[torch.Tensor]) -> torch.Tensor:
        """RECC 耦合分支（§21/§33）：入参必须已 detach，返回完整 soft 分布。

        梯度边界：state/cond 的 detach 切断到 backbone / 病理归一化器的通路，
        KL 损失只更新本 Router 的参数（核心验收项，方案 §43）。
        """
        return F.softmax(self._logits(state_detached, cond_detached)
                         / self.temperature, dim=-1)


class TFMPathologyExpert(nn.Module):
    """单个病理残差专家（方案 §11）::

        [z_t, z_g] → Linear(2D→H_e) → GELU → FiLM(HAMD) → Dropout → Linear(H_e→D) → Δz_i

    输出为 latent transition 层的逐 ROI 修正 Δz_i（§34），不做 raw BOLD 解码。
    末层小初始化（std=1e-3）：初始 Δz_i ≈ 0，保证 moe 模式 finetune 初始
    ``z_{t+1} ≈ z_shared``（§12）。
    """

    def __init__(self, dim: int, hidden_dim: int = 256,
                 cond_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.in_proj = nn.Linear(2 * self.dim, int(hidden_dim))
        self.act = nn.GELU()
        self.film = _ZeroInitFiLM(cond_dim, int(hidden_dim)) if cond_dim else None
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(int(hidden_dim), self.dim)
        # 小初始化末层：Δz_i 初始 ≈ 0 但各专家方向不同（首层由全局 RNG 独立采样）
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, u: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            u: [B, F, 2D]（[z_t, z_g] 拼接）
            cond: [B, C] 或 None
        Returns:
            Δz_i [B, F, D]
        """
        v = self.act(self.in_proj(u))                          # [B,F,H_e]
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)
            v = v * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out_proj(self.drop(v))


def build_pair_indices(num_experts: int) -> torch.Tensor:
    """E 个专家的全部 C(E,2) 组合，返回 [P, 2] 索引对（i<j，行序按字典序）。

    E=4 → 6 对：01, 02, 03, 12, 13, 23（方案 §27）。模型侧构造 pair candidate
    与损失侧计算 Router implied pair score 共用本顺序。
    """
    if num_experts < 2:
        raise ValueError(f"num_experts 必须 >= 2，收到 {num_experts}")
    pairs = [(i, j) for i in range(num_experts) for j in range(i + 1, num_experts)]
    return torch.tensor(pairs, dtype=torch.long)


class TFMPathologyMoE(nn.Module):
    """病理残差专家 MoE（方案 §6/§13/§33/§35）。

    ``is_conditioning_adapter = True`` 供 train/optim.py 的微调冻结逻辑识别：
    冻结 backbone 阶段 Router 与全部 Expert 仍参与训练（方案 §44/§46 Stage 1）。

    forward 返回字典（键名与 NeuroTwinTFM aux_info 约定一致，方案 §35/§40）：
      - ``delta_path_z``  [B,F,D]   mixture 后的病理修正（主路径消费）
      - ``tfm_router_probs``    [B,E]   soft 概率（含梯度）
      - ``tfm_router_weights``  [B,E]   实际 mixture 权重（sparse 期 Top-K 外为 0）
      - ``tfm_router_logits``   [B,E]
      - ``tfm_topk_indices``    [B,K]
      - ``tfm_expert_delta_z_norm`` [B,E,F,D]（仅 need_expert）
      - ``tfm_router_probs_coupling`` [B,E]（仅 need_coupling；detach 分支）
    """

    is_conditioning_adapter = True

    def __init__(self, dim: int, cond_dim: int, num_experts: int = 4,
                 top_k: int = 2, expert_hidden_dim: int = 256,
                 router_hidden_dim: int = 128, router_temperature: float = 1.0,
                 dropout: float = 0.1, dense_warmup_epochs: int = 5):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"tfm_num_experts 必须 >= 2，收到 {num_experts}")
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError(
                f"tfm_top_k 必须在 [1, num_experts] 内，收到 top_k={top_k}, "
                f"num_experts={num_experts}")
        self.dim = int(dim)
        self.cond_dim = int(max(0, cond_dim))
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.dense_warmup_epochs = int(dense_warmup_epochs)
        self.router = TFMPathologyRouter(
            state_dim=4 * self.dim, cond_dim=self.cond_dim,
            num_experts=self.num_experts, hidden_dim=router_hidden_dim,
            temperature=router_temperature, dropout=dropout)
        self.experts = nn.ModuleList([
            TFMPathologyExpert(dim=self.dim, hidden_dim=expert_hidden_dim,
                               cond_dim=self.cond_dim or None, dropout=dropout)
            for _ in range(self.num_experts)
        ])

    def state_descriptor(self, z_t: torch.Tensor,
                         z_g: torch.Tensor) -> torch.Tensor:
        """全脑状态描述子（方案 §8）：ROI 维的 [μ_t, σ_t, μ_g, σ_g] → [B, 4D]。"""
        return torch.cat([
            z_t.mean(dim=1), z_t.std(dim=1, unbiased=False),
            z_g.mean(dim=1), z_g.std(dim=1, unbiased=False),
        ], dim=-1)

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None,
                need_expert: bool = False,
                need_coupling: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t: [B,F,D] 最后 causal token
            z_g: [B,F,D] A_eff @ z_t（图混合）
            cond: [B,C] 病理条件；None 时回退零条件（router / expert FiLM 均
                  零初始化，零条件等价于关闭条件输入）
            epoch: 当前训练 epoch（0 基）；None 视为部署/评估态 → 直接 Top-K 稀疏
            need_expert: 是否返回逐专家 Δz（RECC / diagnose 需要）
            need_coupling: 是否计算耦合分支 soft 分布（RECC 需要）
        """
        b = z_t.shape[0]
        if cond is None and self.cond_dim > 0:
            cond = z_t.new_zeros(b, self.cond_dim)
        state = self.state_descriptor(z_t, z_g)                # [B,4D]
        dense = epoch is not None and epoch < self.dense_warmup_epochs
        route = self.router(state, cond, top_k=self.top_k, dense=dense)

        u = torch.cat([z_t, z_g], dim=-1)                      # [B,F,2D]
        expert_delta_z = torch.stack(
            [expert(u, cond) for expert in self.experts], dim=1)  # [B,E,F,D]
        delta_path_z = torch.einsum(
            'be,befd->bfd', route['routing_weights'], expert_delta_z)

        out = {
            'delta_path_z': delta_path_z,
            'tfm_router_probs': route['soft_probs'],
            'tfm_router_weights': route['routing_weights'],
            'tfm_router_logits': route['logits'],
            'tfm_topk_indices': route['topk_indices'],
        }
        if need_expert:
            out['tfm_expert_delta_z_norm'] = expert_delta_z
        if need_coupling:
            out['tfm_router_probs_coupling'] = self.router.coupling_probs(
                state.detach(), cond.detach() if cond is not None else None)
        return out
