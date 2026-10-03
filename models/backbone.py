"""SC-guided encoder and HAMD residual for next brain-state prediction."""

import torch
import torch.nn as nn

from models.common import DFCAdapter, BrainMDM, GraphODEDDI
from models.graph_prior import SoftAnatomicalPrior
from models.neurotwin_moe import NeuroTwinMoE
from models.pathology import PathologyNormalizer


class BrainDynamicsBackbone(nn.Module):
    """Encode one bounded history context; the prediction head lives in next_state.py."""

    def __init__(self, args):
        super().__init__()
        features = args.num_rois
        context = args.context_max
        self.features = features
        self.pretrain_mode = args.mode == 'pretrain'
        if args.patho_cond_layer not in ('residual_only', 'feature_only', 'joint'):
            raise ValueError('Invalid patho_cond_layer')
        if args.patho_adaln_targets not in ('none', 'mdm', 'ode', 'both'):
            raise ValueError('Invalid patho_adaln_targets')
        self.patho_cond_layer = args.patho_cond_layer
        ada_targets = ('none' if args.patho_cond_layer == 'residual_only'
                       else args.patho_adaln_targets)
        self.pathology_normalizer = None
        cond_dim = 0
        if not self.pretrain_mode:
            self.pathology_normalizer = PathologyNormalizer(
                input_dim=args.pathology_input_dim, mode=args.pathology_norm_mode,
                n_quantiles=args.pathology_norm_quantiles,
                rbf_knots=args.pathology_norm_rbf_knots)
            cond_dim = self.pathology_normalizer.out_dim
        ada_mdm = cond_dim if ada_targets in ('mdm', 'both') else None
        ada_ode = cond_dim if ada_targets in ('ode', 'both') else None

        lora_ranks = [0] * args.n_block
        if not self.pretrain_mode and args.lora_enable and args.lora_rank > 0:
            for index in range(max(0, args.n_block - args.lora_n_blocks), args.n_block):
                lora_ranks[index] = args.lora_rank

        self.sc_prior = None
        if args.sc_prior_mode != 'scaled':
            self.sc_prior = SoftAnatomicalPrior(
                features=features, seq_len=context, mode=args.sc_prior_mode,
                rank=args.sc_prior_rank, lambda_mode=args.sc_lambda_mode,
                lambda_init=args.sc_lambda_init, delta_a=args.sc_delta_a,
                delta_rank=args.sc_delta_rank, prob_mask=args.sc_prob_mask,
                sinkhorn_iters=args.sc_sinkhorn_iters,
                return_sequence=(args.sc_temporal_weight > 0))
        self.input_dropout = nn.Dropout2d(p=min(0.2, args.dropout * 0.5))
        self.dfc_adapter = DFCAdapter(num_nodes=features, alpha=args.alpha,
                                      sc_prior_mode=args.sc_prior_mode)
        self.pastmixing = BrainMDM(
            features=features, num_window=1, seq_len=context,
            num_scales=args.num_scales, dropout=args.dropout,
            ada_cond_dim=ada_mdm, scale_scheme=args.mdm_scale_scheme,
            scale_gate=args.mdm_scale_gate)
        window_heads = next((heads for heads in (4, 2, 1) if context % heads == 0), 1)
        self.ode_blocks = nn.ModuleList([
            GraphODEDDI(
                features=features, seq_len=context, hidden_dim=args.ode_hidden_dim,
                ode_steps=args.ode_steps, dropout=args.dropout,
                stochastic_depth_rate=args.stochastic_depth_rate,
                num_heads=4, window_heads=window_heads, ada_cond_dim=ada_ode,
                lora_rank=lora_ranks[index], sc_mask_mode=args.sc_mask_mode,
                sc_mask_tau=args.sc_mask_tau,
                use_window_attn=(args.ode_window_attn == 'on'),
                solver=args.ode_solver, step_mode=args.ode_step_mode,
                step_scale=args.ode_step_scale, sde_noise_scale=args.sde_noise_scale,
                adaptive_rtol=args.ode_adaptive_rtol,
                adaptive_atol=args.ode_adaptive_atol)
            for index in range(args.n_block)])
        self.ode_block_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            for _ in range(args.n_block)])
        self.post_fusion = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=1),
            nn.GELU(), nn.Dropout(args.dropout),
            nn.Conv2d(features * 2, features, kernel_size=1))
        self.feature_norm = nn.GroupNorm(num_groups=1, num_channels=features)

        self.moe = None
        if not self.pretrain_mode:
            # Context normalization is performed before this encoder. No RevIN
            # statistics are passed to MoE, so its gate uses state features.
            self.moe = NeuroTwinMoE(
                features=features, in_w=1, in_s=context, pred_w=1, pred_s=1,
                pathology_input_dim=cond_dim, pathology_dim=args.pathology_dim,
                num_experts=args.num_experts, top_k=args.top_k,
                dropout=args.dropout, gate_temperature=args.moe_gate_temp_start,
                expert_hidden_dim=args.moe_expert_hidden_dim,
                use_shared_expert=args.moe_use_shared_expert,
                router_context_cond_only=args.moe_router_cond_only,
                use_argmax=args.moe_use_argmax,
                inference_temperature=args.moe_inference_temperature,
                eval_mode=args.moe_eval_mode,
                pathology_poly_expansion=args.pathology_poly_expansion,
                delta_refiner_rounds=args.delta_refiner_rounds,
                refiner_adaptive=args.refiner_adaptive,
                gate_features='state', gate_input_dim=args.moe_gate_input_dim,
                experts_mode=args.moe_experts_mode,
                expert_kind=args.moe_expert_kind,
                route_level=args.moe_route_level,
                eval_mc_samples=args.moe_eval_mc_samples,
                expert_stats_interval=args.moe_expert_stats_interval)

    def _prepare_pathology_condition(self, score):
        if self.pathology_normalizer is None or score is None:
            return None
        return self.pathology_normalizer(score)

    def _encode(self, history, sc, condition=None):
        if history.ndim != 4:
            raise ValueError(f'Expected [B,F,1,K], got {tuple(history.shape)}')
        sc_info = self.sc_prior(history, sc) if self.sc_prior is not None else None
        adjacency = sc_info['A_eff'] if sc_info is not None else None
        x = self.input_dropout(history) if self.training else history
        shallow = self.dfc_adapter(x, sc, adj=adjacency)
        x = self.pastmixing(shallow, condition)
        ode_diag = {}
        for index, (scale, block) in enumerate(zip(self.ode_block_scales, self.ode_blocks)):
            evolved = block(x, sc, condition, adj_eff=adjacency)
            diag = getattr(block, 'last_ode_diag', None)
            if diag:
                ode_diag.update({f'ode{index}_{key}': value for key, value in diag.items()})
            x = x + torch.sigmoid(scale) * (evolved - x)
        x = x + self.post_fusion(torch.cat([shallow, x], dim=1))
        return self.feature_norm(x), sc_info, ode_diag
