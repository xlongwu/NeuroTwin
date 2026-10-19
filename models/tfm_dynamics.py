# coding=utf-8
"""受控条件图 ODE 动力学头与逐 ROI 方差读出。

GraphODEDynamicsHead 是当前动力学头：

- 组合 Laplacian 图扩散 ``L = diag(deg) - A``（不显式构造 L，按
  ``degree.unsqueeze(-1) * z - bmm(adj, z)`` 计算）；
- 条件化神经残差：ConditionFiLM（临床 c）+ PersonalFiLM（个体 p）沿 ROI 广播；
- 逐样本标量扩散率 ``κ = softplus(κ0 + w_cᵀc + w_pᵀp)``；
- 状态相关控制门 ``s = sigmoid(W_s[V, c, p])`` 与跨 ROI 共享 ``G u``；
- 连续 BOLD 速率读出 ``dq/dt = W_y Z + b_y``；
- 固定 Heun RK2 同步积分 Z 与 q（integrate_interval）。

向量场不加 Dropout / DropPath、不随机跳子步；积分在 autocast 关闭的 float32
中执行（编码器与 CPM 继续使用原 AMP）。非正 dt、ode_steps<1 与形状错误直接
报错，不做静默轨迹裁剪、自适应步长或假预测 fallback。

VarianceHead 从 **detached** latent 轨迹读出逐 ROI/horizon 对角 logvar，
仅训练该小头，不影响均值预测。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tfm_personal import PersonalFiLM

#: logvar 的数值边界
LOGVAR_MIN = -6.0
LOGVAR_MAX = 4.0


class VarianceHead(nn.Module):
    """逐 ROI / horizon 对角方差读出：``LayerNorm(D) → Linear(D,1)``。

    末层 weight/bias 零初始化（初始 logvar=0，即归一化空间 std=1 的中性起点）。
    输入必须是 **detach 后的** latent 轨迹（由调用方保证），因此方差损失只更新
    本头，不改变均值预测目标。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.out = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_trajectory: torch.Tensor) -> torch.Tensor:
        """z_trajectory ``[B, H, F, D]``（应已 detach）→ logvar ``[B, H, F]``。

        注意返回顺序是 (B, H, F)：latent 轨迹的时间轴在前，消费方按需 permute。
        """
        if z_trajectory.ndim != 4:
            raise ValueError(
                f'VarianceHead 期望 [B,H,F,D]，收到 {tuple(z_trajectory.shape)}')
        logvar = self.out(self.norm(z_trajectory)).squeeze(-1)   # [B,H,F]
        return logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)


class GraphODEDynamicsHead(nn.Module):
    """受控条件图 ODE 头：向量场 + 固定 Heun RK2。

    Args:
        dim: latent 宽度 D。
        hidden_dim: 神经残差隐层宽度（复用 ``tfm_one_step_hidden_dim``，默认 256）。
        cond_dim: 临床条件宽度 C；None = 不构建条件 FiLM / kappa_cond。
        profile_dim: 个体码宽度；None = 不构建 PersonalFiLM / kappa_personal。
        ode_steps: 每个预测区间的 RK2 子步数 N（默认 2）。
        residual_strength: 神经残差系数（默认 0.1，固定不可学习）。
        diffusion_init: 扩散率初值（默认 0.05；``kappa_raw = log(expm1(κ_init))``）。

    初始化：W1 Xavier、W2 正态 std=0.01、各 bias 零；两类 FiLM
    与 w_c/w_p 零初始化；G 正态 std=0.01 无 bias；控制门正常随机、b_s=0；
    W_y/b_y 零初始化（初始 BOLD 为 persistence）。W2、G 与控制门**不**一起清零。
    """

    def __init__(self, dim: int, hidden_dim: int = 256,
                 cond_dim: Optional[int] = None,
                 profile_dim: Optional[int] = None,
                 ode_steps: int = 2,
                 residual_strength: float = 0.1,
                 diffusion_init: float = 0.05):
        super().__init__()
        if dim < 1 or hidden_dim < 1:
            raise ValueError(f'dim/hidden_dim 必须为正：{dim}/{hidden_dim}')
        if int(ode_steps) < 1:
            raise ValueError(f'ode_steps 必须 >= 1，收到 {ode_steps}')
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.ode_steps = int(ode_steps)
        self.residual_strength = float(residual_strength)
        self.cond_dim = int(cond_dim) if cond_dim else 0
        self.profile_dim = int(profile_dim) if profile_dim else 0

        # ---- 神经残差：W1 Xavier / W2 std=0.01 / bias 零 ----
        self.norm_z = nn.LayerNorm(self.dim)
        self.w1 = nn.Linear(2 * self.dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        self.w2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.normal_(self.w2.weight, std=0.01)
        nn.init.zeros_(self.w2.bias)

        # ---- 条件化（零初始化，起点不改基准动力学）----
        # ConditionFiLM 在 models/tfm.py 中定义；局部导入避免模块级循环依赖
        # （tfm.py 在运行期构造本头时已完全加载）。
        from models.tfm import ConditionFiLM
        self.film = None
        self.kappa_cond = None
        if self.cond_dim:
            self.film = ConditionFiLM(self.cond_dim, self.hidden_dim)
            self.kappa_cond = nn.Linear(self.cond_dim, 1, bias=False)
            nn.init.zeros_(self.kappa_cond.weight)
        self.personal_film = None
        self.kappa_personal = None
        if self.profile_dim:
            self.personal_film = PersonalFiLM(self.profile_dim, self.hidden_dim)
            self.kappa_personal = nn.Linear(self.profile_dim, 1, bias=False)
            nn.init.zeros_(self.kappa_personal.weight)
        # 逐模型共享标量扩散基准：κ0 = log(expm1(diffusion_init))
        self.kappa_raw = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(float(diffusion_init)))),
                         dtype=torch.float32))

        # ---- 控制门 W_s 与跨 ROI 共享投影 G ----
        gate_in = self.dim + self.cond_dim + self.profile_dim
        self.control_gate = nn.Linear(gate_in, 1)
        nn.init.zeros_(self.control_gate.bias)
        self.control_proj = nn.Linear(1, self.dim, bias=False)
        nn.init.normal_(self.control_proj.weight, std=0.01)

        # ---- 连续 BOLD 速率读出（零初始化 → 初始 persistence）----
        self.decode_rate = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.decode_rate.weight)
        nn.init.zeros_(self.decode_rate.bias)

        # ---- 逐 ROI 方差读出 ----
        self.variance_head = VarianceHead(self.dim)

    # ------------------------------------------------------------------
    # 向量场
    # ------------------------------------------------------------------
    def _field(self, z: torch.Tensor, adj: torch.Tensor,
               cond: Optional[torch.Tensor] = None,
               profile: Optional[torch.Tensor] = None,
               control: Optional[torch.Tensor] = None) -> torch.Tensor:
        """dz/dt：``-κ·LZ + 0.1·R + s⊙G·u``。

        Args:
            z: [B,F,D] float32
            adj: [B,F,F] 非负对称图 A（调用方已对称化）
            cond: [B,C] 或 None；profile: [B,P] 或 None
            control: [B,F,1] 或 None（None 与全零严格等价）
        """
        b, f, d = z.shape
        v = self.norm_z(z)                                     # [B,F,D]
        av = torch.bmm(adj, v)                                 # [B,F,D]
        a = self.w1(torch.cat([v, av], dim=-1))                # [B,F,hidden]
        # FiLM 沿 ROI 维广播：h' = (1+γc+γp)⊙a + βc + βp
        if self.film is not None and cond is not None:
            gamma_c, beta_c = self.film(cond)
            a = a * (1.0 + gamma_c.unsqueeze(1)) + beta_c.unsqueeze(1)
        if self.personal_film is not None and profile is not None:
            gamma_p, beta_p = self.personal_film(profile)
            a = a * (1.0 + gamma_p.unsqueeze(1)) + beta_p.unsqueeze(1)
        r = self.w2(F.gelu(a))                                 # [B,F,D]

        # 组合 Laplacian：LZ = degree·z − A z（不显式构造 L）
        degree = adj.sum(dim=-1)                               # [B,F]
        lz = degree.unsqueeze(-1) * z - torch.bmm(adj, z)      # [B,F,D]

        # 逐样本标量扩散率 κ = softplus(κ0 + w_cᵀc + w_pᵀp)
        kappa_logit = self.kappa_raw.view(1, 1).expand(b, 1).to(z.dtype)
        if self.kappa_cond is not None and cond is not None:
            kappa_logit = kappa_logit + self.kappa_cond(cond)
        if self.kappa_personal is not None and profile is not None:
            kappa_logit = kappa_logit + self.kappa_personal(profile)
        kappa = F.softplus(kappa_logit)                        # [B,1]

        out = -kappa.unsqueeze(-1) * lz + self.residual_strength * r

        # 状态相关控制门：s = sigmoid(W_s[V_i, c, p])；u=None 与全零严格等价
        if control is not None:
            if control.shape != (b, f, 1):
                raise ValueError(
                    f'control 期望 [B,{f},1]，收到 {tuple(control.shape)}')
            gate_parts = [v]
            if self.cond_dim and cond is not None:
                gate_parts.append(cond.to(z.dtype).unsqueeze(1).expand(b, f, self.cond_dim))
            if self.profile_dim and profile is not None:
                gate_parts.append(
                    profile.to(z.dtype).unsqueeze(1).expand(b, f, self.profile_dim))
            gate_in = torch.cat(gate_parts, dim=-1)            # [B,F,D+C+P]
            s = torch.sigmoid(self.control_gate(gate_in))      # [B,F,1]
            out = out + s * self.control_proj(control.to(z.dtype))  # s⊙G·u
        return out

    def _rate(self, z: torch.Tensor) -> torch.Tensor:
        """dq/dt = W_y Z + b_y → [B,F]。"""
        return self.decode_rate(z).squeeze(-1)

    # ------------------------------------------------------------------
    # 固定 Heun RK2
    # ------------------------------------------------------------------
    def integrate_interval(self, z: torch.Tensor, q: torch.Tensor,
                           adj: torch.Tensor, dt: torch.Tensor,
                           cond: Optional[torch.Tensor] = None,
                           profile: Optional[torch.Tensor] = None,
                           control: Optional[torch.Tensor] = None):
        """在**一个**预测区间内以固定 Heun RK2 同步推进 latent Z 与归一化 BOLD q。

        Args:
            z: [B,F,D]；q: [B,F]（RevIN 空间）；adj: [B,F,F] 对称图
            dt: [B] 正的区间长度（单位 TR）；control: [B,F,1] 区间内固定
        Returns:
            (z_new [B,F,D], q_new [B,F]) float32，保留反向图（不 detach）。

        第二 stage 在预测 latent 上重新计算向量场，状态相关控制门随之重算；
        A/c/p/u 在区间内固定。积分关闭 autocast、以 float32 运算。
        """
        if dt.ndim != 1 or dt.shape[0] != z.shape[0]:
            raise ValueError(
                f'dt 期望 [B={z.shape[0]}]，收到 {tuple(dt.shape)}')
        if bool((dt <= 0).any()):
            raise ValueError(f'时间间隔必须全为正，收到 {dt.detach().cpu().tolist()}')
        device_type = z.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            z32 = z.float()
            q32 = q.float()
            adj32 = adj.float()
            c32 = cond.float() if cond is not None else None
            p32 = profile.float() if profile is not None else None
            u32 = control.float() if control is not None else None
            h = (dt.float() / self.ode_steps)                  # [B]
            hz = h[:, None, None]                              # z 步长 [B,1,1]
            hq = h[:, None]                                    # q 步长 [B,1]
            for _ in range(self.ode_steps):
                k1 = self._field(z32, adj32, c32, p32, u32)
                v1 = self._rate(z32)
                z_stage = z32 + hz * k1
                k2 = self._field(z_stage, adj32, c32, p32, u32)
                v2 = self._rate(z_stage)
                z32 = z32 + 0.5 * hz * (k1 + k2)
                q32 = q32 + 0.5 * hq * (v1 + v2)
        return z32, q32

    # ------------------------------------------------------------------
    # 多区间推进
    # ------------------------------------------------------------------
    def forward(self, z0: torch.Tensor, q0: torch.Tensor, adj: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                profile: Optional[torch.Tensor] = None,
                controls: Optional[torch.Tensor] = None,
                time_deltas: Optional[torch.Tensor] = None,
                horizon: int = 8):
        """从初始状态连续推进 ``horizon`` 个预测区间。

        Args:
            z0: [B,F,D] 最后 patch token；q0: [B,F] 最后归一化 BOLD
            adj: [B,F,F] A_eff（本头内部已对称化输入可原样传入）
            controls: [B,H,F] 逐区间控制（u[t] 对应到达观测 t 的区间）；None 无控制
            time_deltas: [H] 或 [B,H] 或 None（None = 全部 dt=1）
        Returns:
            z_final [B,F,D]；q_trajectory [B,F,H]；z_trajectory [B,H,F,D]
        """
        if z0.ndim != 3 or q0.ndim != 2:
            raise ValueError(
                f'z0 期望 [B,F,D]，q0 期望 [B,F]；收到 {tuple(z0.shape)}/{tuple(q0.shape)}')
        if adj.ndim != 3 or adj.shape[-2:] != (z0.shape[1], z0.shape[1]):
            raise ValueError(
                f'adj 期望 [B,{z0.shape[1]},{z0.shape[1]}]，收到 {tuple(adj.shape)}')
        b = z0.shape[0]
        h_out = int(horizon)
        if h_out < 1:
            raise ValueError(f'horizon 必须 >= 1，收到 {h_out}')
        adj = 0.5 * (adj + adj.transpose(-1, -2))              # 对称化为 A
        if controls is not None and tuple(controls.shape) != (b, h_out, z0.shape[1]):
            raise ValueError(
                f'controls 期望 [B,{h_out},{z0.shape[1]}]，收到 {tuple(controls.shape)}')
        if time_deltas is not None and time_deltas.ndim not in (1, 2):
            raise ValueError(
                f'time_deltas 期望 [H] 或 [B,H]，收到 {tuple(time_deltas.shape)}')
        if time_deltas is not None and time_deltas.shape[-1] != h_out:
            raise ValueError(
                f'time_deltas 末维 {time_deltas.shape[-1]} 与 horizon={h_out} 不一致')

        device = z0.device
        ones = torch.ones(b, device=device, dtype=torch.float32)
        z, q = z0, q0
        z_traj, q_traj = [], []
        for hidx in range(h_out):
            u = controls[:, hidx, :].unsqueeze(-1) if controls is not None else None
            if time_deltas is None:
                dt = ones
            elif time_deltas.ndim == 1:
                dt = time_deltas.to(device=device, dtype=torch.float32).expand(b)
            else:
                dt = time_deltas[:, hidx].to(device=device, dtype=torch.float32)
            z, q = self.integrate_interval(z, q, adj, dt, cond=cond,
                                           profile=profile, control=u)
            z_traj.append(z)
            q_traj.append(q)
        z_final = z
        q_trajectory = torch.stack(q_traj, dim=-1)             # [B,F,H]
        z_trajectory = torch.stack(z_traj, dim=1)              # [B,H,F,D]
        return z_final, q_trajectory, z_trajectory
