"""Standalone runner for the retained trajectory research experiment.

The production main.py entry only trains next-timepoint states.
"""

import argparse

from main import parse_args as next_state_defaults
from train.trajectory import build_trajectory_model, run_trajectory
from utils.common import str2bool


TRAJECTORY_DEFAULTS = {
    'pretrained_weight': './checkpoints/hc_trajectory/base_best.pt',
    'chunk_size': 20, 'chunk_stride': None, 'history_min': 4,
    'history_max': 8, 'max_pred_horizon': 3,
    'horizon_weights': [1.0, 0.7, 0.5],
    'random_context': True, 'random_cutoff': True,
    'eval_rollout': True, 'amp': True, 'use_ema': True,
    'ema_decay': 0.999, 'prefetch_factor': 2,
    'pathology_fields': None, 'pathology_missing': 'drop',
    'recur_mode': 'none', 'pred_head': 'gaussian',
    'pred_quantiles': '0.1,0.5,0.9',
    'head_shape_mode': 'query', 'head_use_win_attn': True,
    'head_use_history_proj': True, 'head_use_latent_proj': True,
    'head_use_temporal': True, 'head_use_cross_roi': True,
    'head_use_revin_stats': True, 'head_cross_roi': 'conv1',
    'future_query_mode': 'roi', 'future_query_dim': 32,
    'future_query_layers': 1, 'future_query_heads': 4,
    'head_amp_mode': 'scale_mod_trend',
    'head_scale_granularity': 'window',
    'head_amp_consistency_weight': 0.01,
    'refiner_rounds': 3,
    'refiner_inter_sup_weight': 0.05,
    'refiner_inter_sup_decay': 0.5,
    'init_log_var_pcc': 0.0, 'init_log_var_mae': -1.5,
    'init_log_var_diff': -2.0, 'init_log_var_std': -2.0,
    'init_log_var_nll': 2.0, 'clamp_log_vars': True,
    'log_var_min': -6.0, 'log_var_max': 6.0,
    'loss_diff_mode': 'per_window',
    'moe_load_balance_weight': 0.05,
    'moe_entropy_weight': 0.001,
    'moe_z_loss_weight': 0.001,
    'moe_diversity_weight': 0.001,
    'sc_sparsity_weight': 0.001, 'sc_entropy_weight': 0.001,
    'enable_latent_loss': False, 'lambda_latent': 0.1,
    'enable_fc_loss': False, 'lambda_fc': 0.05,
    'inversion_weight': 0.0,
}


def parse_args(argv=None):
    defaults = vars(next_state_defaults([]))
    defaults.update(TRAJECTORY_DEFAULTS)
    parser = argparse.ArgumentParser(description='Separate NeuroTwin trajectory experiment')
    for name, default in defaults.items():
        option = '--' + name
        if isinstance(default, bool):
            parser.add_argument(option, type=str2bool, default=default)
        elif isinstance(default, list):
            parser.add_argument(option, nargs='+', type=type(default[0]), default=default)
        elif default is None:
            parser.add_argument(option,
                                type=int if name in ('chunk_stride', 'sc_delta_rank',
                                                     'moe_gate_input_dim') else str,
                                default=None)
        else:
            parser.add_argument(option, type=type(default), default=default)
    args = parser.parse_args(argv)
    args.task_mode = 'trajectory'
    return args


if __name__ == '__main__':
    run_trajectory(parse_args(), build_trajectory_model)
