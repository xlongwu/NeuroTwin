# coding=utf-8
"""消融实验基线配置：镜像 scripts/Finetune_MDD_next_point.sh 的 ARGS 数组。

⚠️ 与 scripts/Finetune_MDD_next_point.sh 需同步维护：该脚本调整训练超参时，本文件
必须同步修改，否则消融结果与既有基线（neurotwin_nextpoint_finetune）不可比。

约定：
- 键为 main.py argparse 参数名（不带 --）；未列出的参数回落到 main.py 默认值。
- 每次运行的 mode / seed / name / pretrained_weight 由 run_experiments.py 注入，
  不在本配置中。
- 值为 Python 原生类型，由 run_experiments.py 统一转为命令行字符串。
"""
from pathlib import Path
import re

# 项目根目录：experiments/ 的上一级
ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / 'data'
CHECKPOINT_DIR = ROOT / 'checkpoints'
CLINICAL_FILE = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'

BASE_ARGS = {
    # ---- 基础与路径 ----
    'data_root': str(DATA_ROOT),
    'checkpoint_dir': str(CHECKPOINT_DIR),
    'clinical_file': CLINICAL_FILE,
    'num_rois': 116,
    'seq_len': 30,
    'total_windows': 9,
    # ---- 模型规模 ----
    'n_block': 2,
    'alpha': 0.5,
    # 2026-09-27：Normlize 数据已逐 ROI 全序列 z-score，关闭 context 内 BrainRevIN 二次归一化
    'norm': False,
    'dropout': 0.2,
    'ode_steps': 3,  # 2026-09-25 消融结论：1/3/6/12 步差 ≤0.002，与 Finetune_MDD_next_point.sh 同步取 3
    'ode_hidden_dim': 256,
    'num_scales': 3,
    'stochastic_depth_rate': 0.10,
    # ---- 优化器与训练规模 ----
    'train_epochs': 150,
    'batch_size': 32,
    'lr_init': 5e-5,
    'lr_peak': 1e-4,
    'lr_final': 5e-5,
    'warmup_epochs': 15,
    'weight_decay': 1e-2,
    'grad_clip': 1.0,
    'patience': 20,
    'amp': True,
    'compile': True,
    'use_ema': True,
    'ema_decay': 0.999,
    'freeze_backbone_epochs': 10,
    'backbone_lr_scale': 0.2,
    'num_workers': 4,
    'num_threads': 4,
    'pin_memory': True,
    'cache_in_memory': False,
    'tf32': True,
    # ---- 损失学习率缩放 ----
    'loss_lr_scale': 0.5,
    # ---- Phase 0：协议基础（被试级 8:1:1 划分） ----
    'pretrained_arch_policy': 'require_match',
    'val_ratio': 0.10,
    'test_ratio': 0.10,
    'stratify_bins': 5,
    'refiner_rounds': 3,
    'delta_refiner_rounds': 1,  # 2026-09-25 消融结论：1 轮与 2 轮等价，与 Finetune_MDD_next_point.sh 同步
    'refiner_adaptive': False,
    'refiner_inter_sup_weight': 0.05,
    'refiner_inter_sup_decay': 0.5,
    # ---- Phase 1：病理条件化（HAMD 归一化在模型内完成） ----
    'pathology_input_dim': 1,
    'pathology_dim': 32,
    'pathology_norm_mode': 'robust_z',
    'reuse_pretrained_norm_stats': True,
    'pathology_poly_expansion': False,
    'pathology_missing': 'drop',
    'patho_cond_layer': 'joint',
    'patho_adaln_targets': 'both',
    'lora_enable': True,
    'lora_rank': 8,
    'lora_n_blocks': 2,
    'adapter_lr_scale': 1.0,
    # ---- Phase 2：SC 软先验 ----
    'sc_prior_mode': 'soft_prior',
    'sc_lambda_mode': 'global',
    'sc_lambda_init': 0.7,
    'sc_mask_mode': 'soft',
    'sc_delta_a': True,
    'sc_prob_mask': False,
    'sc_refiner_inject': 'both',
    'sc_sparsity_weight': 1e-3,
    'sc_entropy_weight': 1e-3,
    'sc_temporal_weight': 0.0,
    # ---- Phase 3：预测头 future query / 幅值重参数化 / 多尺度 ----
    'head_shape_mode': 'query',
    'future_query_mode': 'roi',
    'future_query_dim': 32,
    'future_query_layers': 1,
    'future_query_heads': 4,
    'head_amp_mode': 'scale_mod_trend',
    'head_scale_granularity': 'window',
    'head_amp_consistency_weight': 0.01,
    'head_cross_roi': 'conv1',
    'head_use_history_proj': True,
    'head_use_latent_proj': True,
    'head_use_temporal': True,
    'head_use_cross_roi': True,
    'head_use_win_attn': True,
    'head_use_revin_stats': True,
    'ode_window_attn': 'on',
    'mdm_scale_scheme': 'divisor',
    'mdm_scale_gate': 'sample',
    # ---- Phase 4：GraphODE 连续时间 ----
    'ode_solver': 'rk2',
    'ode_step_mode': 'learnable',
    'ode_step_scale': 0.1,
    # ---- Phase 5：MoE ----
    'num_experts': 4,
    'top_k': 2,
    'moe_load_balance_weight': 0.05,
    'moe_entropy_weight': 1e-3,
    'moe_z_loss_weight': 1e-3,
    'moe_diversity_weight': 0.001,
    'moe_gate_temp_start': 1.5,
    'moe_gate_temp_end': 1.0,
    'moe_expert_hidden_dim': 256,
    'moe_use_shared_expert': True,
    'moe_router_cond_only': False,
    'moe_use_argmax': False,
    'moe_inference_temperature': 0.3,
    'moe_eval_mode': 'dense_soft',
    'moe_gate_features': 'state_revin',
    'moe_route_level': 'sample',
    'moe_experts_mode': 'routed_only',  # 2026-09-25 消融结论：shared 增量≈0（ΔPCC −0.0016），移除省算力
    'moe_expert_kind': 'homogeneous',
    'moe_eval_mc_samples': 0,
    'moe_expert_stats_interval': 0,
    # 受控消融：finetune 加载预训练权重时强制随机初始化的分支（fnmatch 模式）
    'pretrained_skip_pattern': '',
    # ---- Phase 7：不确定性 / 反事实 ----
    'pred_head': 'gaussian',
    'pred_quantiles': '0.1,0.5,0.9',
    'init_log_var_nll': 2.0,
    'inversion_weight': 0.0,
    'inversion_hidden_dim': 128,
    'intervention_mode': 'latent',
    # ---- Phase 8：任务口径与数据采样（唯一口径 next_timepoint）----
    'task_mode': 'next_timepoint',
    'random_context': True,
    'random_cutoff': True,
    'bold_source': 'auto',
    'train_samples_per_subject': 0,      # 0 = auto（min(合法组合数, 12)）
    'sampling_seed': 2024,
    'subject_cache_size': 256,
    'eval_fc': True,
    'eval_spectral': False,
    'eval_spectral_tr': 2.0,
    # ---- Phase 9：连续 BOLD → 下一 TR 全脑状态（next_timepoint）----
    'context_min': 16,
    'context_max': 64,
    'context_lengths': '',               # 空 = 在 [context_min, context_max] 内随机采样
    'prediction_target': 'delta',
    'causal_training': 'random_context',
    'forecast_offsets': '',              # 空 = 按 enable_mtp 取 [1] 或 [1,2,4,8]
    'enable_mtp': False,
    'mtp_weights': '',
    'lambda_abs': 1.0,
    'lambda_delta': 1.0,
    'lambda_pcc': 0.1,
    'lambda_nll': 0.0,
    'enable_rollout_loss': False,
    'rollout_train_steps': 2,
    'lambda_rollout': 0.2,
    'eval_context_length': 0,            # 0 = auto（=context_max）
    'eval_anchors_per_subject': 16,
    'eval_rollout_tasks_per_subject': 2,
    'eval_rollout_horizons': '1,2,4,8,16',
    'eval_rollout_steps': 0,             # 0 = auto（max(horizons)，FC 启用时抬到 fc_min_length）
    'fc_min_length': 32,
    'load_backbone_only': False,
}


def _cli_value(value):
    """把配置值转为命令行字符串：bool -> True/False，其余 str()。"""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


def build_cli_args(args_dict):
    """把 {参数名: 值} 展开为 ['--k', 'v', ...] 列表。"""
    out = []
    for k, v in args_dict.items():
        out += [f'--{k}', _cli_value(v)]
    return out


def validate_overrides(overrides, main_py_path=None):
    """校验 override 键存在于 main.py 的 argparse 定义中，防止拼写错误静默失效。

    通过正则扫描 main.py 源码中的 add_argument('--xxx') 提取合法参数名，
    避免手工维护参数清单导致的不同步。
    """
    if not overrides:
        return
    main_py = Path(main_py_path) if main_py_path else ROOT / 'main.py'
    if not main_py.exists():
        raise FileNotFoundError(f'未找到 main.py：{main_py}')
    source = main_py.read_text(encoding='utf-8')
    valid = set(re.findall(r"add_argument\(\s*'--([A-Za-z0-9_]+)'", source))
    unknown = [k for k in overrides if k not in valid]
    if unknown:
        raise ValueError(
            f'以下 override 键不是 main.py 的合法参数：{unknown}\n'
            f'请核对 experiments/variants.py 中的参数拼写。')
