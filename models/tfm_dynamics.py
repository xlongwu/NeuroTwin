# coding=utf-8
"""受控条件图 ODE 动力学头与逐 ROI 方差读出（方案 §6）。

GraphODEDynamicsHead 在 ``dynamics_mode='graph_ode'`` 时**替换**
``OneStepDynamicsHead``（不包装旧头 residual、不构建第二套闲置参数）：

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
from models.tfm_probabilistic import LowRankLatentInnovation

#: logvar 的数值边界（方案 §6.4 / §8.1）
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
    """受控条件图 ODE 头（方案 §6.1）：向量场 + 固定 Heun RK2。

    Args:
        dim: latent 宽度 D。
        hidden_dim: 神经残差隐层宽度（复用 ``tfm_one_step_hidden_dim``，默认 256）。
        cond_dim: 临床条件宽度 C；None = 不构建条件 FiLM / kappa_cond。
        profile_dim: 个体码宽度；None = 不构建 PersonalFiLM / kappa_personal。
        ode_steps: 每个预测区间的 RK2 子步数 N（默认 2）。
        residual_strength: 神经残差系数（默认 0.1，固定不可学习）。
        diffusion_init: 扩散率初值（默认 0.05；``kappa_raw = log(expm1(κ_init))``）。

    初始化（方案 §6.1.4）：W1 Xavier、W2 正态 std=0.01、各 bias 零；两类 FiLM
    与 w_c/w_p 零初始化；G 正态 std=0.01 无 bias；控制门正常随机、b_s=0；
    W_y/b_y 零初始化（初始 BOLD 为 persistence）。W2、G 与控制门**不**一起清零。
    """

    def __init__(self, dim: int, hidden_dim: int = 256,
                 cond_dim: Optional[int] = None,
                 profile_dim: Optional[int] = None,
                 ode_steps: int = 2,
                 residual_strength: float = 0.1,
                 diffusion_init: float = 0.05,
                 history_memory: bool = False,
                 stochastic: bool = False,
                 noise_rank: int = 8,
                 noise_scale: float = 1e-2):
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
            self.film = ConditionFiLM(self.cond_dim, self.hidden_dim, lam=1.0)
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

        # ---- S2：历史记忆 FiLM（零初始化 → 起点等价于 baseline）----
        # 记忆 M_history 为观测历史编码 token 在 patch 维的均值，rollout 期间固定，
        # 使自由推进始终能访问真实历史，而不必反复重编码预测值。
        # fork_rng 内构建：不推进全局 RNG，保证同一 seed 下后续模块的初始化
        # 与 baseline 逐位一致（消融无初始化混杂）。
        self.memory_film = None
        if history_memory:
            with torch.random.fork_rng(devices=[]):
                self.memory_film = nn.Linear(self.dim, 2 * self.hidden_dim)
                nn.init.zeros_(self.memory_film.weight)
                nn.init.zeros_(self.memory_film.bias)

        # ---- S4-2：低秩潜状态创新（默认关闭；fork_rng 不推进全局随机流）----
        self.innovation = None
        if stochastic:
            with torch.random.fork_rng(devices=[]):
                self.innovation = LowRankLatentInnovation(
                    self.dim, rank=int(noise_rank), scale_init=float(noise_scale))

    # ------------------------------------------------------------------
    # 向量场（方案 §6.1.1–§6.1.3）
    # ------------------------------------------------------------------
    def _field(self, z: torch.Tensor, adj: torch.Tensor,
               cond: Optional[torch.Tensor] = None,
               profile: Optional[torch.Tensor] = None,
               control: Optional[torch.Tensor] = None,
               memory: Optional[torch.Tensor] = None,
               return_terms: bool = False):
        """dz/dt：``-κ·LZ + 0.1·R + s⊙G·u``（含 S2 历史记忆 FiLM）。

        Args:
            z: [B,F,D] float32
            adj: [B,F,F] 非负对称图 A（调用方已对称化）
            cond: [B,C] 或 None；profile: [B,P] 或 None
            control: [B,F,1] 或 None（None 与全零严格等价）
            memory: [B,D] 或 [B,F,D] 历史记忆（S2；None 时记忆 FiLM 关闭）
            return_terms: True 时额外返回各分量 dict（诊断用，不改变数值路径）
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
        if self.memory_film is not None and memory is not None:
            mem = memory if memory.ndim == 3 else memory.unsqueeze(1)
            gamma_m, beta_m = self.memory_film(mem.to(v.dtype)).chunk(2, dim=-1)
            a = a * (1.0 + gamma_m) + beta_m
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

        diffusion = -kappa.unsqueeze(-1) * lz
        residual = self.residual_strength * r
        out = diffusion + residual
        control_term = None

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
            control_term = s * self.control_proj(control.to(z.dtype))  # s⊙G·u
            out = out + control_term
        if return_terms:
            return out, {'diffusion': diffusion, 'residual': residual,
                         'control': control_term, 'kappa': kappa}
        return out

    def field_terms(self, z: torch.Tensor, adj: torch.Tensor,
                    cond: Optional[torch.Tensor] = None,
                    profile: Optional[torch.Tensor] = None,
                    control: Optional[torch.Tensor] = None,
                    memory: Optional[torch.Tensor] = None) -> dict:
        """诊断接口：返回向量场各分量（diffusion/residual/control/kappa）。"""
        _, terms = self._field(z, adj, cond, profile, control, memory=memory,
                               return_terms=True)
        return terms

    def _rate(self, z: torch.Tensor) -> torch.Tensor:
        """dq/dt = W_y Z + b_y → [B,F]。"""
        return self.decode_rate(z).squeeze(-1)

    # ------------------------------------------------------------------
    # 固定 Heun RK2（方案 §6.2）
    # ------------------------------------------------------------------
    def integrate_interval(self, z: torch.Tensor, q: torch.Tensor,
                           adj: torch.Tensor, dt: torch.Tensor,
                           cond: Optional[torch.Tensor] = None,
                           profile: Optional[torch.Tensor] = None,
                           control: Optional[torch.Tensor] = None,
                           memory: Optional[torch.Tensor] = None):
        """在**一个**预测区间内以固定 Heun RK2 同步推进 latent Z 与归一化 BOLD q。

        Args:
            z: [B,F,D]；q: [B,F]（RevIN 空间）；adj: [B,F,F] 对称图
            dt: [B] 正的区间长度（单位 TR）；control: [B,F,1] 区间内固定
            memory: [B,D] 或 [B,F,D] 历史记忆（S2；None 时关闭）
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
            m32 = memory.float() if memory is not None else None
            h = (dt.float() / self.ode_steps)                  # [B]
            hz = h[:, None, None]                              # z 步长 [B,1,1]
            hq = h[:, None]                                    # q 步长 [B,1]
            for _ in range(self.ode_steps):
                k1 = self._field(z32, adj32, c32, p32, u32, memory=m32)
                v1 = self._rate(z32)
                z_stage = z32 + hz * k1
                q_stage = q32 + hq * v1
                k2 = self._field(z_stage, adj32, c32, p32, u32,
                                 memory=m32)
                v2 = self._rate(z_stage)
                z32 = z32 + 0.5 * hz * (k1 + k2)
                q32 = q32 + 0.5 * hq * (v1 + v2)
        return z32, q32

    # ------------------------------------------------------------------
    # 多区间推进（方案 §6.2 末）
    # ------------------------------------------------------------------
    def forward(self, z0: torch.Tensor, q0: torch.Tensor, adj: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                profile: Optional[torch.Tensor] = None,
                controls: Optional[torch.Tensor] = None,
                time_deltas: Optional[torch.Tensor] = None,
                horizon: int = 8,
                memory: Optional[torch.Tensor] = None):
        """从初始状态连续推进 ``horizon`` 个预测区间。

        Args:
            z0: [B,F,D] 最后 patch token；q0: [B,F] 最后归一化 BOLD
            adj: [B,F,F] A_eff（本头内部已对称化输入可原样传入）
            controls: [B,H,F] 逐区间控制（u[t] 对应到达观测 t 的区间）；None 无控制
            time_deltas: [H] 或 [B,H] 或 None（None = 全部 dt=1）
            memory: [B,D] 或 [B,F,D] 历史记忆（S2；rollout 期间固定不变）
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
                                           profile=profile, control=u,
                                           memory=memory)
            z_traj.append(z)
            q_traj.append(q)
        z_final = z
        q_trajectory = torch.stack(q_traj, dim=-1)             # [B,F,H]
        z_trajectory = torch.stack(z_traj, dim=1)              # [B,H,F,D]
        return z_final, q_trajectory, z_trajectory

    def forward_samples(self, z0: torch.Tensor, q0: torch.Tensor,
                        adj: torch.Tensor,
                        cond: Optional[torch.Tensor] = None,
                        profile: Optional[torch.Tensor] = None,
                        controls: Optional[torch.Tensor] = None,
                        time_deltas: Optional[torch.Tensor] = None,
                        horizon: int = 8,
                        memory: Optional[torch.Tensor] = None,
                        n_samples: int = 32,
                        generator: Optional[torch.Generator] = None):
        """S4-2：低秩随机潜状态推进，返回 ``(q_samples [B,K,F,H], z_final [B,K,F,D])``。

        每个样本的区间推进为

            Z ← Z + sqrt(dt)·Gφ(Z,M)ε,   (Z,q) ← HeunRK2(Z,q)

        即创新在区间入口注入，再走原 Heun 积分与 BOLD 读出；``ε [B,K,r]`` 只在
        rank 维独立，ROI 相关由状态相关的增益 ``Gφ`` 提供。噪声与积分均在 fp32
        执行，与确定性路径的数值口径一致。
        """
        if self.innovation is None:
            raise RuntimeError('forward_samples 需要 stochastic=True 构建的创新模块')
        if z0.ndim != 3 or q0.ndim != 2:
            raise ValueError(
                f'z0 期望 [B,F,D]，q0 期望 [B,F]；收到 {tuple(z0.shape)}/{tuple(q0.shape)}')
        if adj.ndim != 3 or adj.shape[-2:] != (z0.shape[1], z0.shape[1]):
            raise ValueError(
                f'adj 期望 [B,{z0.shape[1]},{z0.shape[1]}]，收到 {tuple(adj.shape)}')
        b, f, d = z0.shape
        k = int(n_samples)
        if k < 1:
            raise ValueError(f'n_samples 必须 >= 1，收到 {k}')
        h_out = int(horizon)
        if h_out < 1:
            raise ValueError(f'horizon 必须 >= 1，收到 {h_out}')
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        if controls is not None and tuple(controls.shape) != (b, h_out, f):
            raise ValueError(
                f'controls 期望 [B,{h_out},{f}]，收到 {tuple(controls.shape)}')
        if time_deltas is not None and time_deltas.ndim not in (1, 2):
            raise ValueError(
                f'time_deltas 期望 [H] 或 [B,H]，收到 {tuple(time_deltas.shape)}')
        if time_deltas is not None and time_deltas.shape[-1] != h_out:
            raise ValueError(
                f'time_deltas 末维 {time_deltas.shape[-1]} 与 horizon={h_out} 不一致')

        def expand(x):
            if x is None:
                return None
            return x.repeat_interleave(k, dim=0)

        z = expand(z0)
        q = expand(q0)
        adj = expand(adj)
        cond = expand(cond)
        profile = expand(profile)
        memory = expand(memory)
        controls = expand(controls)
        if time_deltas is not None and time_deltas.ndim == 2:
            time_deltas = expand(time_deltas)
        ones = torch.ones(b * k, device=z0.device, dtype=torch.float32)
        q_traj = []
        for hidx in range(h_out):
            u = controls[:, hidx, :].unsqueeze(-1) if controls is not None else None
            if time_deltas is None:
                dt = ones
            elif time_deltas.ndim == 1:
                dt = time_deltas.to(device=z0.device, dtype=torch.float32).expand(b * k)
            else:
                dt = time_deltas[:, hidx].to(device=z0.device, dtype=torch.float32)
            with torch.autocast(device_type=z.device.type, enabled=False):
                eps = torch.randn(b * k, self.innovation.rank,
                                  device=z.device, dtype=torch.float32,
                                  generator=generator)
                noise = self.innovation(z.float(), memory=memory, eps=eps)
                z = z.float() + (dt.float().sqrt().view(-1, 1, 1)
                                 * noise.unsqueeze(-1))
            z, q = self.integrate_interval(z, q, adj, dt, cond=cond,
                                           profile=profile, control=u,
                                           memory=memory)
            q_traj.append(q)
        q_samples = torch.stack(q_traj, dim=-1).reshape(b, k, f, h_out)
        return q_samples, z.reshape(b, k, f, d)
