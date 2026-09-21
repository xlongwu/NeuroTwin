# coding=utf-8
"""训练基础设施：损失函数、优化器/调度器、EMA 与 MoE 正则。"""
from train.losses import UncertaintyWeightedHybridLoss
from train.moe import _get, compute_moe_regularization, update_router_temperature
from train.optim import (
    ModelEMA,
    build_optimizer,
    build_scheduler,
    count_trainable_params,
    get_param_groups,
    load_backbone_weights,
    set_finetune_stage,
)

__all__ = [
    'UncertaintyWeightedHybridLoss',
    'ModelEMA',
    'load_backbone_weights',
    'set_finetune_stage',
    'get_param_groups',
    'build_optimizer',
    'build_scheduler',
    'count_trainable_params',
    '_get',
    'update_router_temperature',
    'compute_moe_regularization',
]
