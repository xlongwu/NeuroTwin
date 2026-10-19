"""NeuroTwinTFM: history-FC guided personalized Graph-ODE with clinical field encoding."""

from contextlib import contextmanager
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import BrainRevIN
from models.conditioning import ARMS, ConditioningConfig, build_condition_encoder
from models.graph_prior import SoftAnatomicalPrior
from models.tfm_backbone import (
    RMSNorm,
    TFMEncoderBlock,
    TemporalPatchEmbed,
    build_context_grid,
)
from models.tfm_dynamics import GraphODEDynamicsHead
from models.tfm_personal import ProfileEncoder


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


class ConditionFiLM(nn.Module):
    """Zero-initialized channel scale/shift from clinical conditions."""

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
        """Conditions [B,C] to scale and shift [B,D]."""
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"条件向量期望 [B, {self.cond_dim}]，收到 {tuple(cond.shape)}"
            )
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return (scale, shift)


class CPMHorizonHead(nn.Module):
    """Non-autoregressive horizon queries over historical ROI tokens."""

    def __init__(
        self,
        dim: int,
        horizon: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"cpm_horizon 必须 >= 1，收到 {horizon}")
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.horizon_embed = nn.Parameter(torch.zeros(self.horizon, self.dim))
        nn.init.trunc_normal_(self.horizon_embed, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_q": RMSNorm(self.dim),
                        "norm_m": RMSNorm(self.dim),
                        "attn": _CrossAttention(self.dim, num_heads, dropout),
                        "norm_ffn": RMSNorm(self.dim),
                        "ffn": nn.Sequential(
                            nn.Linear(self.dim, 4 * self.dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * self.dim, self.dim),
                        ),
                        "drop": nn.Dropout(dropout),
                    }
                )
            )
        self.out_proj = nn.Linear(self.dim, 1)
        with isolated_global_rng():
            self.film = ConditionFiLM(cond_dim, self.dim) if cond_dim else None

    def forward(
        self,
        context: torch.Tensor,
        roi_embed: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """History tokens [B,F,P,D] to direct forecasts [B,F,H]."""
        b, f, p, d = context.shape
        h_out = self.horizon
        q = roi_embed.view(1, f, 1, d) + self.horizon_embed.view(1, 1, h_out, d)
        queries = q.expand(b, f, h_out, d)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)
            queries = queries * (1.0 + scale.view(b, 1, 1, d)) + shift.view(b, 1, 1, d)
        q_flat = queries.reshape(b, f * h_out, d)
        m_flat = context.reshape(b, f * p, d)
        for layer in self.layers:
            attn_out = layer["attn"](layer["norm_q"](q_flat), layer["norm_m"](m_flat))
            q_flat = q_flat + layer["drop"](attn_out)
            q_flat = q_flat + layer["drop"](layer["ffn"](layer["norm_ffn"](q_flat)))
        out = self.out_proj(q_flat)
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
        """Multi-head cross-attention from horizon queries to history."""
        b, nq, _ = q.shape
        qh = self.q_proj(q).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        kh, vh = self.kv_proj(memory).chunk(2, dim=-1)
        kh = kh.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vh = vh.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            qh, kh, vh, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, nq, self.dim)
        return self.out_proj(out)


class NeuroTwinTFM(nn.Module):
    """TFM encoder, field conditions, personalized Graph-ODE dynamics and auxiliary CPM."""

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
        sc_prior_mode: str = "soft_prior",
        sc_lambda_mode: str = "global",
        sc_lambda_init: float = 0.7,
        sc_prior_rank: int = 12,
        sc_delta_a: bool = True,
        sc_delta_rank: int = None,
        sc_sinkhorn_iters: int = 64,
        one_step_hidden_dim: int = 256,
        cpm_horizon: int = 8,
        cpm_layers: int = 2,
        pretrain_mode: bool = True,
        condition_config: Optional[dict] = None,
        condition_stats_payload: Optional[dict] = None,
        condition_arm: Optional[str] = None,
        use_revin: bool = True,
        dynamics_mode: str = "graph_ode",
        ode_steps: int = 2,
        dynamics_dt: float = 1.0,
        graph_source: str = "history_fc",
        personalized: bool = True,
        profile_dim: int = 32,
    ):
        """Graph-ODE model with history-derived graph and field conditions."""
        super().__init__()
        if patch_len < 1 or context_max < patch_len:
            raise ValueError("patch_len must be positive and <= context_max")
        if cpm_horizon < 1:
            raise ValueError("cpm_horizon must be positive")
        if dynamics_mode != "graph_ode" or graph_source != "history_fc":
            raise ValueError("Only graph_ode dynamics with history_fc are supported")
        self.features = int(features)
        self.context_max = int(context_max)
        self.patch_len = int(patch_len)
        self.dim = int(dim)
        self.cpm_horizon = int(cpm_horizon)
        self.pretrain_mode = bool(pretrain_mode)
        self.use_revin = bool(use_revin)
        self.sc_prior_mode = str(sc_prior_mode)
        self.dynamics_mode = "graph_ode"
        self.graph_source = "history_fc"
        self.ode_steps = int(ode_steps)
        self.dynamics_dt = float(dynamics_dt)
        if self.ode_steps < 1 or self.dynamics_dt <= 0:
            raise ValueError("ode_steps and dynamics_dt must be positive")
        self.personalized = bool(personalized)
        self.profile_dim = int(profile_dim) if self.personalized else 0
        if self.personalized and self.profile_dim < 1:
            raise ValueError("profile_dim must be positive")
        cfg = (
            ConditioningConfig.from_dict(condition_config) if condition_config else None
        )
        self.condition_arm = (
            str(condition_arm) if condition_arm else cfg.arm if cfg else "none"
        )
        if self.condition_arm not in ARMS:
            raise ValueError(f"condition_arm must be one of {ARMS}")
        if cfg is not None and self.condition_arm != cfg.arm:
            raise ValueError("condition_arm and condition_config.arm must match")
        if self.condition_arm != "none" and cfg is None:
            raise ValueError(
                "condition_arm requires condition_config and condition_stats_payload"
            )
        self.conditioning_on = self.condition_arm != "none"
        self.condition_cfg_dict = dict(condition_config) if condition_config else None
        self.rev_norm = BrainRevIN(num_features=features) if self.use_revin else None
        self.sc_prior = SoftAnatomicalPrior(
            features=features,
            seq_len=context_max,
            mode=sc_prior_mode,
            rank=sc_prior_rank,
            lambda_mode=sc_lambda_mode,
            lambda_init=sc_lambda_init,
            delta_a=sc_delta_a,
            delta_rank=sc_delta_rank,
            prob_mask=False,
            sinkhorn_iters=sc_sinkhorn_iters,
            return_sequence=False,
        )
        self.conditioner = None
        cond_dim = 0
        if self.conditioning_on:
            from utils.condition_schema import ConditionStats, schema_for

            if not condition_stats_payload:
                raise ValueError("Condition encoding requires training statistics")
            specs = schema_for(cfg.group)
            stats = ConditionStats.from_dict(specs, condition_stats_payload)
            with isolated_global_rng():
                self.conditioner = build_condition_encoder(specs, stats, cfg)
            cond_dim = int(cfg.cond_dim)
        self.cond_dim = cond_dim
        self.patch_embed = TemporalPatchEmbed(
            features=features,
            context_max=context_max,
            patch_len=patch_len,
            dim=dim,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TFMEncoderBlock(
                    dim=dim, num_heads=num_heads, ff_ratio=ff_ratio, dropout=dropout
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(dim)
        self.one_step_head = GraphODEDynamicsHead(
            dim=dim,
            hidden_dim=one_step_hidden_dim,
            cond_dim=cond_dim or None,
            profile_dim=self.profile_dim or None,
            ode_steps=self.ode_steps,
        )
        self.cpm_head = CPMHorizonHead(
            dim=dim,
            horizon=cpm_horizon,
            num_heads=num_heads,
            num_layers=cpm_layers,
            dropout=dropout,
            cond_dim=cond_dim or None,
        )
        self.profile_encoder = (
            ProfileEncoder(dim=dim, profile_dim=self.profile_dim, hidden=64)
            if self.personalized
            else None
        )

    def _prepare_condition(self, condition=None, condition_epoch=None):
        """Encode cond_z, cond_code and cond_mask; None bypasses conditions."""
        if condition is not None:
            if self.conditioner is None:
                raise ValueError(
                    "收到 condition 但当前模型未构建条件编码器（condition_config 缺失或 arm=none）。"
                )
            if not isinstance(condition, dict):
                raise TypeError(
                    f"condition 需为 dict（cond_z/cond_code/cond_mask），收到 {type(condition)}"
                )
            missing = [
                k
                for k in ("cond_z", "cond_code", "cond_mask")
                if condition.get(k) is None
            ]
            if missing:
                raise KeyError(f"condition 缺少必需键 {missing}")
            return self.conditioner(
                condition["cond_z"],
                condition["cond_code"],
                condition["cond_mask"],
                subject_index=condition.get("subject_index", None),
                epoch=condition_epoch,
            )
        return None

    def _encode(self, x_grid: torch.Tensor, sc_matrix: torch.Tensor):
        """Normalized BOLD [B,F,K] to history tokens and the shared graph."""
        x4 = x_grid.unsqueeze(2)
        sc_info = self.sc_prior(x4, sc_matrix)
        tokens = self.patch_embed(x_grid)
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, sc_info["A_eff"])
        return (self.final_norm(tokens), sc_info)

    def _history_fc_adj(self, x_grid: torch.Tensor) -> torch.Tensor:
        """当前可见历史的 signed Pearson 功能耦合图。

        x_grid ``[B,F,K]``（归一化或数据尺度均可：Pearson 相关对逐 ROI 仿射不变）
        → ``[B,F,F]``：取绝对值、清零对角。该图作为 SoftAnatomicalPrior 的输入，
        复用其混合、self-loop 与 Sinkhorn；它是**功能耦合图**，不默认称独立解剖 SC。
        """
        r = x_grid - x_grid.mean(dim=-1, keepdim=True)
        num = torch.bmm(r, r.transpose(1, 2))
        denom = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-06)
        c_hist = num / (denom * denom.transpose(1, 2))
        a_hist = c_hist.abs()
        f = x_grid.shape[1]
        eye = torch.eye(f, dtype=torch.bool, device=x_grid.device)
        return a_hist.masked_fill(eye, 0.0)

    def _graph_input(self, x_grid: torch.Tensor, sc_matrix=None):
        """Build the coupling graph from the visible history."""
        return self._history_fc_adj(x_grid)

    def _denorm_3d(self, pred_norm: torch.Tensor, stats) -> torch.Tensor:
        """[B,F,H]（归一化）→ 原始空间（复用 BrainRevIN 的仿射反变换）。"""
        if self.rev_norm is None:
            return pred_norm
        out = self.rev_norm(pred_norm.unsqueeze(-1), "denorm", stats=stats)
        return out.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        sc_matrix: torch.Tensor,
        condition: Optional[dict] = None,
        condition_epoch: Optional[int] = None,
        controls: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
        dynamics_horizon: Optional[int] = None,
        calibration_bold: Optional[torch.Tensor] = None,
        profile: Optional[torch.Tensor] = None,
    ):
        """BOLD [B,F,1,K] to next state and CPM/dynamics auxiliaries."""
        if x.ndim != 4 or x.shape[2] != 1:
            raise ValueError(
                f"NeuroTwinTFM.forward 期望 [B,F,1,K]，收到 {tuple(x.shape)}"
            )
        x_grid, _ = build_context_grid(x)
        norm_stats = None
        if self.rev_norm is not None:
            x_norm4, norm_stats = self.rev_norm(x, "norm")
            x_grid = x_norm4[:, :, 0, :]
        cond = self._prepare_condition(condition, condition_epoch)
        graph_input = self._graph_input(x_grid, sc_matrix)
        tokens, sc_info = self._encode(x_grid, graph_input)
        z_t = tokens[:, :, -1, :]
        x_last_norm = x_grid[:, :, -1]
        dyn_aux = None
        if self.personalized:
            if profile is None and calibration_bold is not None:
                profile = self.calibrate_profile(calibration_bold)
            elif profile is None:
                raise ValueError(
                    "personalized=True 的 forward 需要 profile 或 calibration_bold 之一（不偷偷从包含未来的全序列计算）。"
                )
        else:
            profile = None
        dyn_h = (
            int(dynamics_horizon) if dynamics_horizon is not None else self.cpm_horizon
        )
        z_final, q_traj_norm, z_traj = self.one_step_head(
            z_t,
            x_last_norm,
            sc_info["A_eff"],
            cond=cond,
            profile=profile,
            controls=controls,
            time_deltas=time_deltas,
            horizon=dyn_h,
        )
        del z_final
        one_pred_norm = q_traj_norm[:, :, 0:1].unsqueeze(-1)
        logvar_hf = self.one_step_head.variance_head(z_traj.detach())
        dyn_aux = {
            "dynamics_mean_norm": q_traj_norm,
            "dynamics_logvar_norm": logvar_hf.permute(0, 2, 1),
            "norm_stats": norm_stats,
        }
        if profile is not None:
            dyn_aux["personal_profile"] = profile
        cpm_pred_norm = self.cpm_head(tokens, self.patch_embed.roi_embed, cond)
        aux_info = {"sc_prior": sc_info}
        if cond is not None:
            aux_info["pathology_cond"] = cond
            aux_info["condition_injected"] = torch.ones(1, device=cond.device)
        if dyn_aux is not None:
            if self.rev_norm is not None:
                dyn_aux["dynamics_pred"] = self._denorm_3d(
                    dyn_aux["dynamics_mean_norm"], norm_stats
                )
                sigma_norm = torch.exp(0.5 * dyn_aux["dynamics_logvar_norm"])
                _, std_s = norm_stats
                a_w = self.rev_norm.affine_weight
                denom = (a_w + self.rev_norm.eps).abs().view(1, -1, 1)
                dyn_aux["dynamics_std"] = (
                    sigma_norm
                    * std_s.to(sigma_norm.dtype)[:, :, 0, 0].unsqueeze(-1)
                    / denom
                )
            else:
                dyn_aux["dynamics_pred"] = dyn_aux["dynamics_mean_norm"]
                dyn_aux["dynamics_std"] = torch.exp(
                    0.5 * dyn_aux["dynamics_logvar_norm"]
                )
            aux_info.update(dyn_aux)
        if self.rev_norm is not None:
            one_pred = self.rev_norm(one_pred_norm, "denorm", stats=norm_stats)
            cpm_pred = self._denorm_3d(cpm_pred_norm, norm_stats)
        else:
            one_pred, cpm_pred = (one_pred_norm, cpm_pred_norm)
        aux_info["cpm_pred"] = cpm_pred
        return (one_pred, aux_info)

    @staticmethod
    def _to_internal_history(bold_history: torch.Tensor) -> torch.Tensor:
        """标准输入 [B, K, F] → 内部布局 [B, F, 1, K]。"""
        if bold_history.ndim != 3:
            raise ValueError(
                f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}"
            )
        return bold_history.transpose(1, 2).unsqueeze(2).contiguous()

    def predict_next_state(
        self,
        bold_history: torch.Tensor,
        sc_matrix: torch.Tensor,
        condition: Optional[dict] = None,
        condition_epoch: Optional[int] = None,
        control: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        calibration_bold: Optional[torch.Tensor] = None,
        profile: Optional[torch.Tensor] = None,
    ) -> dict:
        """Initialize once and advance one interval; return next state and change."""
        state = self.initialize_state(
            bold_history,
            sc_matrix=sc_matrix,
            condition=condition,
            condition_epoch=condition_epoch,
            profile=profile,
            calibration_bold=calibration_bold,
        )
        if control is not None and control.ndim == 2:
            control = control.unsqueeze(-1)
        state = self.advance_state(state, control=control, dt=dt)
        pred_next = self.readout_state(state).unsqueeze(-1)
        x_last = bold_history[:, -1, :].unsqueeze(-1)
        aux_info = {"norm_stats": state["norm_stats"]}
        if state["profile"] is not None:
            aux_info["personal_profile"] = state["profile"]
        return {
            "pred_next": pred_next,
            "pred_delta": pred_next - x_last,
            "pred": pred_next.unsqueeze(-1),
            "aux_info": aux_info,
        }

    def forecast_horizon(
        self,
        bold_history: torch.Tensor,
        sc_matrix: torch.Tensor,
        condition: Optional[dict] = None,
        condition_epoch: Optional[int] = None,
        calibration_bold=None,
        profile=None,
    ) -> torch.Tensor:
        """Return direct auxiliary CPM predictions [B,H,F]."""
        x = self._to_internal_history(bold_history)
        _, aux_info = self.forward(
            x,
            sc_matrix,
            condition=condition,
            condition_epoch=condition_epoch,
            calibration_bold=calibration_bold,
            profile=profile,
        )
        return aux_info["cpm_pred"].transpose(1, 2)

    def rollout_next_states(
        self,
        bold_history: torch.Tensor,
        sc_matrix: torch.Tensor,
        steps: int = 1,
        condition: Optional[dict] = None,
        condition_epoch: Optional[int] = None,
        controls: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
        calibration_bold: Optional[torch.Tensor] = None,
        profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialize once and continuously advance steps intervals; return [B,steps,F]."""
        if bold_history.ndim != 3:
            raise ValueError(
                f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}"
            )
        n_steps = int(steps)
        if n_steps < 1:
            raise ValueError(f"steps 必须 >= 1，收到 {n_steps}")
        b, _, f_dim = bold_history.shape
        if controls is not None and tuple(controls.shape) != (b, n_steps, f_dim):
            raise ValueError(
                f"controls 期望 [B,{n_steps},{f_dim}]，收到 {tuple(controls.shape)}"
            )
        if time_deltas is not None and time_deltas.shape[-1] != n_steps:
            raise ValueError(
                f"time_deltas 末维 {time_deltas.shape[-1]} 与 steps={n_steps} 不一致"
            )
        state = self.initialize_state(
            bold_history,
            sc_matrix=sc_matrix,
            condition=condition,
            condition_epoch=condition_epoch,
            profile=profile,
            calibration_bold=calibration_bold,
        )
        outs: List[torch.Tensor] = []
        for k in range(n_steps):
            u = controls[:, k, :] if controls is not None else None
            dt_sel = None
            if time_deltas is not None:
                dt_sel = time_deltas[:, k] if time_deltas.ndim == 2 else time_deltas[k]
            state = self.advance_state(
                state, control=u.unsqueeze(-1) if u is not None else None, dt=dt_sel
            )
            outs.append(self.readout_state(state))
        return torch.stack(outs, dim=1)

    def _denorm_q(self, q: torch.Tensor, norm_stats) -> torch.Tensor:
        """RevIN 空间 q [B,F] → 数据尺度（复用 BrainRevIN 统计量与 affine）。"""
        if self.rev_norm is None:
            return q
        out = self.rev_norm(q.unsqueeze(-1).unsqueeze(-1), "denorm", stats=norm_stats)
        return out.squeeze(-1).squeeze(-1)

    def _encode_history_internal(
        self, history: torch.Tensor, sc_matrix, cond: Optional[torch.Tensor] = None
    ):
        """Encode history using the visible-history coupling graph."""
        x = history.transpose(1, 2).unsqueeze(2)
        norm_stats = None
        if self.rev_norm is not None:
            x_norm4, norm_stats = self.rev_norm(x, "norm")
            x_grid = x_norm4[:, :, 0, :]
        else:
            x_grid = x[:, :, 0, :]
        graph_input = self._graph_input(x_grid, sc_matrix)
        tokens, sc_info = self._encode(x_grid, graph_input)
        return (tokens, x_grid, norm_stats, sc_info)

    def _symmetrize_adj(self, a_eff: torch.Tensor) -> torch.Tensor:
        """A_eff → 非负对称图 A。"""
        return 0.5 * (a_eff + a_eff.transpose(-1, -2))

    def calibrate_profile(self, calibration_bold: torch.Tensor) -> torch.Tensor:
        """Encode baseline BOLD [B,F,1,L] with the shared unconditional encoder."""
        if self.profile_encoder is None:
            raise ValueError("personalized=False 未创建 profile 模块，无基线校准可言。")
        if calibration_bold.ndim != 4 or calibration_bold.shape[2] != 1:
            raise ValueError(
                f"calibration_bold 期望 [B,F,1,L]，收到 {tuple(calibration_bold.shape)}"
            )
        x_grid, _ = build_context_grid(calibration_bold)
        if self.rev_norm is not None:
            cal_norm4, _ = self.rev_norm(calibration_bold, "norm")
            x_grid = cal_norm4[:, :, 0, :]
        graph_input = self._history_fc_adj(x_grid)
        tokens, _ = self._encode(x_grid, graph_input)
        return self.profile_encoder(tokens)

    def initialize_state(
        self,
        history: torch.Tensor,
        sc_matrix=None,
        condition: Optional[dict] = None,
        condition_epoch: Optional[int] = None,
        profile: Optional[torch.Tensor] = None,
        calibration_bold: Optional[torch.Tensor] = None,
    ) -> dict:
        """Initialize latent/BOLD state, history graph, conditions, profile and normalization."""
        if history.ndim != 3:
            raise ValueError(f"history 期望 [B,K,F]，收到 {tuple(history.shape)}")
        cond = self._prepare_condition(condition, condition_epoch)
        if self.personalized:
            if profile is None:
                if calibration_bold is None:
                    raise ValueError(
                        "personalized=True 的 initialize_state 需要 profile 或 calibration_bold 之一；不偷偷从包含未来的全序列计算。"
                    )
                profile = self.calibrate_profile(calibration_bold)
        else:
            profile = None
        tokens, x_grid, norm_stats, sc_info = self._encode_history_internal(
            history, sc_matrix, cond
        )
        state = {
            "z": tokens[:, :, -1, :],
            "q": x_grid[:, :, -1],
            "adj": self._symmetrize_adj(sc_info["A_eff"]),
            "cond": cond,
            "profile": profile,
            "norm_stats": norm_stats,
            "history": history,
        }
        return state

    def advance_state(
        self,
        state: dict,
        control: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> dict:
        """推进一个预测区间：更新 z/q/history；图、p、已编码 c、RevIN stats 固定。

        Args:
            control: [B,F] 或 [B,F,1] 区间控制；None 与全零严格等价。
            dt: [B] 区间长度（TR）；None 用构造参数 ``dynamics_dt``。

        返回**新** dict，不原位覆盖、不对 z/q detach（保留反向图）；
        history 以数据尺度预测帧滑窗更新（追加末帧、丢弃最早帧，K 不变）。
        """
        if control is not None and control.ndim == 2:
            control = control.unsqueeze(-1)
        if dt is None:
            dt = torch.full(
                (state["z"].shape[0],),
                float(self.dynamics_dt),
                device=state["z"].device,
                dtype=torch.float32,
            )
        z_new, q_new = self.one_step_head.integrate_interval(
            state["z"],
            state["q"],
            state["adj"],
            dt,
            cond=state["cond"],
            profile=state["profile"],
            control=control,
        )
        x_new = self._denorm_q(q_new, state["norm_stats"])
        history_new = torch.cat([state["history"][:, 1:, :], x_new.unsqueeze(1)], dim=1)
        new_state = dict(state)
        new_state.update({"z": z_new, "q": q_new, "history": history_new})
        return new_state

    def readout_state(self, state: dict) -> torch.Tensor:
        """q 反归一化 → 数据尺度 [B,F]；不再次编码。"""
        return self._denorm_q(state["q"], state["norm_stats"])

    def update_state(self, state: dict, observation: torch.Tensor) -> dict:
        """真实观测（数据尺度 [B,F]，对齐当前预测时点）替换 history 末帧并重编码。

        重新估计当前 RevIN / 图 / Z / q；**保留** profile 与已编码 c（不重复
        条件编码或 profile 计算）。这是观测重编码更新，不是 Bayesian 同化；
        首版只支持完整 ROI 观测。
        """
        if observation.ndim != 2:
            raise ValueError(f"observation 期望 [B,F]，收到 {tuple(observation.shape)}")
        history_new = state["history"].clone()
        expected = (history_new.shape[0], history_new.shape[2])
        if tuple(observation.shape) != expected:
            raise ValueError(
                f"observation 期望 [B={expected[0]},F={expected[1]}]，收到 {tuple(observation.shape)}"
            )
        history_new[:, -1, :] = observation
        sc_matrix = state.get("sc_matrix")
        tokens, x_grid, norm_stats, sc_info = self._encode_history_internal(
            history_new, sc_matrix, state["cond"]
        )
        new_state = dict(state)
        new_state.update(
            {
                "z": tokens[:, :, -1, :],
                "q": x_grid[:, :, -1],
                "adj": self._symmetrize_adj(sc_info["A_eff"]),
                "norm_stats": norm_stats,
                "history": history_new,
            }
        )
        return new_state

    def simulate_intervention(
        self,
        history: torch.Tensor,
        controls: torch.Tensor,
        sc_matrix=None,
        condition: Optional[dict] = None,
        condition_epoch: Optional[int] = None,
        profile: Optional[torch.Tensor] = None,
        calibration_bold: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
    ):
        """同初始状态分别用 U 与零 U 推进 H 步。

        Args:
            history: [B,K,F] 数据尺度；controls: [B,H,F] 新布局逐区间控制。
        Returns:
            (x_ctrl [B,H,F], x_zero [B,H,F], diff [B,H,F])；差值由两条反归一化
            轨迹相减，不对增量单独加均值。
        """
        if controls is None or controls.ndim != 3:
            raise ValueError(
                f"controls 期望 [B,H,F]，收到 {(tuple(controls.shape) if controls is not None else None)}"
            )
        b, h_out, f_dim = controls.shape
        if history.shape[0] != b or history.shape[2] != f_dim:
            raise ValueError(
                f"history [B,K,F]={tuple(history.shape)} 与 controls [B,H,F]={tuple(controls.shape)} 的 B/F 不一致"
            )
        state0 = self.initialize_state(
            history,
            sc_matrix=sc_matrix,
            condition=condition,
            condition_epoch=condition_epoch,
            profile=profile,
            calibration_bold=calibration_bold,
        )
        xs_ctrl, xs_zero = ([], [])
        ctrl_state, zero_state = (state0, state0)
        for k in range(h_out):
            u = controls[:, k, :].unsqueeze(-1)
            dt_sel = None
            if time_deltas is not None:
                dt_sel = time_deltas[:, k] if time_deltas.ndim == 2 else time_deltas[k]
            ctrl_state = self.advance_state(ctrl_state, control=u, dt=dt_sel)
            zero_state = self.advance_state(zero_state, control=None, dt=dt_sel)
            xs_ctrl.append(self.readout_state(ctrl_state))
            xs_zero.append(self.readout_state(zero_state))
        x_ctrl = torch.stack(xs_ctrl, dim=1)
        x_zero = torch.stack(xs_zero, dim=1)
        return (x_ctrl, x_zero, x_ctrl - x_zero)
