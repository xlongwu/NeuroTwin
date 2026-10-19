# coding=utf-8
"""训练基础设施：TFM 损失、SC 图正则、优化器/调度器与 EMA。"""
from train.losses import (
    TFMDualLoss,
    compute_graph_regularization,
    pearson,
    spatial_pcc,
)
from train.optim import (
    ModelEMA,
    build_optimizer,
    build_scheduler,
    count_trainable_params,
    get_param_groups,
    load_backbone_weights,
    save_backbone_weights,
    set_finetune_stage,
    unwrap_state_dict,
)

__all__ = [
    'TFMDualLoss',
    'compute_graph_regularization',
    'pearson',
    'spatial_pcc',
    'ModelEMA',
    'load_backbone_weights',
    'save_backbone_weights',
    'unwrap_state_dict',
    'set_finetune_stage',
    'get_param_groups',
    'build_optimizer',
    'build_scheduler',
    'count_trainable_params',
]
