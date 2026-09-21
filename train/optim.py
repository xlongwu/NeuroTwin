# coding=utf-8
"""优化器、调度器、EMA 与预训练权重加载等训练基础设施。"""
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models.neurotwin import NeuroTwin


class ModelEMA:
    """参数滑动平均（EMA），用于验证与最终权重保存。"""

    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for k, v in self.ema.state_dict().items():
            src = model.state_dict()[k].detach()
            v.mul_(self.decay).add_(src, alpha=1.0 - self.decay) \
                if v.dtype.is_floating_point else v.copy_(src)


def load_backbone_weights(model, pretrained_path, device):
    """按形状兼容过滤加载预训练权重（MoE 等新增模块自动跳过）。"""
    pretrained = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    filtered   = {k: v for k, v in pretrained.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"成功加载预训练参数: {len(filtered)} 个匹配键")
    missing = [k for k in model_dict if k not in filtered]
    if missing:
        print(f"提示: {len(missing)} 个键未从预训练文件加载（如 moe）")


def set_finetune_stage(model: NeuroTwin, backbone_unfrozen: bool = False):
    """微调分阶段训练：默认仅训练 MoE，backbone_unfrozen 后解冻主干。"""
    for p in model.parameters():
        p.requires_grad = False
    if model.moe is not None:
        for p in model.moe.parameters():
            p.requires_grad = True

    if backbone_unfrozen:
        for module in [model.dfc_adapter, model.pastmixing, model.ode_blocks,
                       model.post_fusion, model.feature_norm, model.pretrain_head]:
            for p in module.parameters():
                p.requires_grad = True
        for p in model.ode_block_scales.parameters():
            p.requires_grad = True
        if model.norm:
            for p in model.rev_norm.parameters():
                p.requires_grad = True


def get_param_groups(model: NeuroTwin, args):
    """按预训练/微调模式划分参数组（微调区分 backbone 与 moe 学习率）。"""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.mode == 'pretrain':
        return [{'params': [p for _, p in named], 'lr': args.lr_peak, 'name': 'all'}]
    moe_p      = [p for n, p in named if 'moe' in n]
    backbone_p = [p for n, p in named if 'moe' not in n]
    groups = []
    if backbone_p:
        groups.append({'params': backbone_p,
                       'lr': args.lr_peak * args.backbone_lr_scale, 'name': 'backbone'})
    if moe_p:
        groups.append({'params': moe_p, 'lr': args.lr_peak, 'name': 'moe'})
    return groups


def build_optimizer(model, criterion, args):
    """AdamW；loss 的 log-variance 参数单独成组（可配置学习率比例、无 weight decay）。"""
    groups = get_param_groups(model, args)
    loss_p = [p for p in criterion.parameters() if p.requires_grad]
    if loss_p:
        groups.append({'params': loss_p, 'lr': args.lr_peak * args.loss_lr_scale,
                       'name': 'loss_weight', 'weight_decay': 0.0})
    return AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def build_scheduler(optimizer, args):
    """Linear Warmup + CosineAnnealing；无 warmup 时直接余弦。"""
    if args.warmup_epochs > 0:
        start_factor = max(1e-4, args.lr_init / args.lr_peak)
        return SequentialLR(optimizer, schedulers=[
            LinearLR(optimizer, start_factor=start_factor, end_factor=1.0,
                     total_iters=args.warmup_epochs),
            CosineAnnealingLR(optimizer,
                              T_max=max(1, args.train_epochs - args.warmup_epochs),
                              eta_min=args.lr_final),
        ], milestones=[args.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=max(1, args.train_epochs), eta_min=args.lr_final)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
