# coding=utf-8
"""优化器、调度器、EMA 与预训练权重加载等训练基础设施。"""
import os
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import ARCH_VERSION
from models.neurotwin import NeuroTwin

#: 无 arch_version 元数据的旧检查点按此版本处理
LEGACY_ARCH_VERSION = 1
ARCH_POLICIES = ('require_match', 'warn', 'ignore')


class ModelEMA:
    """参数滑动平均（EMA），用于验证与最终权重保存。"""

    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # torch.compile 包装后 state_dict 键带 '_orig_mod.' 前缀，取内层原模型对齐 EMA 键
        inner = getattr(model, '_orig_mod', model)
        msd = inner.state_dict()
        ema_sd = self.ema.state_dict()
        # _foreach_ 批量融合：避免逐键 mul_/add_ 产生数百次小 kernel launch
        float_dst, float_src, other_dst, other_src = [], [], [], []
        for k, v in ema_sd.items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                float_dst.append(v)
                float_src.append(src)
            else:
                other_dst.append(v)
                other_src.append(src)
        if float_dst:
            torch._foreach_mul_(float_dst, self.decay)
            torch._foreach_add_(float_dst, float_src, alpha=1.0 - self.decay)
        if other_dst:
            torch._foreach_copy_(other_dst, other_src)


def unwrap_state_dict(obj):
    """兼容新旧两种检查点格式，返回 (state_dict, arch_version, config)。

    新格式：{'state_dict': ..., 'arch_version': int, 'config': dict}
    旧格式：裸 state_dict（无元数据，arch_version 记为 None）
    """
    if isinstance(obj, dict) and isinstance(obj.get('state_dict', None), dict):
        return obj['state_dict'], obj.get('arch_version', None), obj.get('config', None)
    if isinstance(obj, dict):
        return obj, None, None
    raise TypeError(f"无法识别的检查点类型: {type(obj)}")


def save_backbone_weights(path, model, meta=None):
    """保存带架构版本元数据的检查点。

    结构改动后 reload 时可按 arch_version 拦截“同 shape 不同语义”的权重，
    这是裸 state_dict 下唯一可靠的兼容性闸门。
    """
    inner = getattr(model, '_orig_mod', model)   # 解包 torch.compile
    payload = {
        'state_dict': deepcopy(inner.state_dict()),
        'arch_version': ARCH_VERSION,
        'config': dict(meta) if meta else {},
    }
    torch.save(payload, path)
    return path


def load_backbone_weights(model, pretrained_path, device, arch_policy='require_match',
                          skip_pattern=''):
    """按形状兼容过滤加载预训练权重（MoE 等新增模块自动跳过）。

    Args:
        arch_policy: 架构版本不匹配时的策略
            - 'require_match'（默认）：直接报错并提示重新预训练
            - 'warn'：打印警告后按 shape 过滤加载（逃生舱）
            - 'ignore'：跳过权重加载
        skip_pattern: fnmatch 模式（如 ``'future_query.*'``），命中的预训练键
            不加载、保留随机初始化。用于受控消融：量化「某分支未经预训练」
            对 finetune 结果的独立影响（如 head_flatten 的对照实验）。
    Returns:
        dict，含 loaded / skipped / missing / arch_version / skipped_load
    """
    if arch_policy not in ARCH_POLICIES:
        raise ValueError(f"Unsupported arch_policy '{arch_policy}', expected one of {ARCH_POLICIES}")
    if skip_pattern:
        import fnmatch

    ckpt = torch.load(pretrained_path, map_location=device)
    state, arch_version, _ = unwrap_state_dict(ckpt)
    fname = os.path.basename(str(pretrained_path))

    if arch_version is None:
        print(f"提示: {fname} 未包含 arch_version 元数据，按 arch_version={LEGACY_ARCH_VERSION} 处理。")
        arch_version = LEGACY_ARCH_VERSION

    if arch_version != ARCH_VERSION:
        msg = (f"预训练权重架构版本不匹配: 权重 arch_version={arch_version}, "
               f"当前代码 ARCH_VERSION={ARCH_VERSION}。"
               f"结构改动会使部分同名参数语义发生变化，按 shape 加载可能得到错误结果。")
        if arch_policy == 'require_match':
            raise RuntimeError(
                msg + f"\n请用当前代码重新预训练（scripts/Pretrain_HC.sh），"
                      f"或显式指定 --pretrained_arch_policy warn 强制按 shape 加载。")
        if arch_policy == 'ignore':
            print(msg + " arch_policy=ignore: 跳过预训练权重加载。")
            return {'loaded': 0, 'skipped': sorted(state.keys()), 'missing': [],
                    'arch_version': arch_version, 'skipped_load': True}
        print(msg + " arch_policy=warn: 继续按 shape 过滤加载。")

    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_dict and model_dict[k].shape == v.shape}
    if skip_pattern:
        # 受控跳过：命中模式的键不加载（保留随机初始化），计入 skipped 以便核对
        force_skipped = sorted(k for k in filtered if fnmatch.fnmatch(k, skip_pattern))
        for k in force_skipped:
            filtered.pop(k)
        if force_skipped:
            print(f"受控跳过预训练加载（--pretrained_skip_pattern '{skip_pattern}'）: "
                  f"{len(force_skipped)} 个键，例如 {force_skipped[:5]}")
    skipped  = sorted(k for k in state if k not in filtered)
    model_dict.update(filtered)
    model.load_state_dict(model_dict)

    missing = sorted(k for k in model_dict if k not in filtered)
    print(f"成功加载预训练参数: {len(filtered)} 个匹配键")
    if skipped:
        print(f"提示: {len(skipped)} 个预训练键因名称/形状不匹配被跳过，例如 {skipped[:5]}")
    if missing:
        print(f"提示: {len(missing)} 个键未从预训练文件加载（如 moe / 新增条件模块）")

    return {'loaded': len(filtered), 'skipped': skipped, 'missing': missing,
            'arch_version': arch_version, 'skipped_load': False}


def _iter_conditioning_modules(model):
    """发现带 `is_conditioning_adapter` 标记的条件模块（AdaLN/LoRA 等）。

    冻结/参数分组基于显式标记而非名字子串匹配，避免重命名导致行为漂移。
    """
    seen = set()
    for _, module in model.named_modules():
        if getattr(module, 'is_conditioning_adapter', False) and id(module) not in seen:
            seen.add(id(module))
            yield module


def set_finetune_stage(model: NeuroTwin, backbone_unfrozen: bool = False):
    """微调分阶段训练：默认训练 MoE + 条件模块，backbone_unfrozen 后解冻主干。"""
    for p in model.parameters():
        p.requires_grad = False
    if model.moe is not None:
        for p in model.moe.parameters():
            p.requires_grad = True
    # 条件模块在冻结阶段即参与训练：它们是“让病理条件真正生效”的唯一路径
    for module in _iter_conditioning_modules(model):
        for p in module.parameters():
            p.requires_grad = True

    if backbone_unfrozen:
        backbone_modules = [model.dfc_adapter, model.pastmixing, model.ode_blocks,
                            model.post_fusion, model.feature_norm, model.pretrain_head]
        # SC 软先验（λ / 功能图低秩分解 / ΔA / 概率掩码）属于主干结构组件，
        # 随主干一起解冻，避免在冻结阶段被误当作条件适配器训练
        if getattr(model, 'sc_prior', None) is not None:
            backbone_modules.append(model.sc_prior)
        for module in backbone_modules:
            for p in module.parameters():
                p.requires_grad = True
        for p in model.ode_block_scales.parameters():
            p.requires_grad = True
        if model.norm:
            for p in model.rev_norm.parameters():
                p.requires_grad = True


def get_param_groups(model: NeuroTwin, args):
    """按模式划分参数组。

    - pretrain: 单组 'all'
    - finetune: 'backbone' / 'adapter'（条件模块）/ 'moe'，各自独立学习率
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.mode == 'pretrain':
        return [{'params': [p for _, p in named], 'lr': args.lr_peak, 'name': 'all'}]

    adapter_ids = set()
    for module in _iter_conditioning_modules(model):
        for p in module.parameters():
            adapter_ids.add(id(p))

    adapter_p  = [p for _, p in named if id(p) in adapter_ids]
    moe_p      = [p for n, p in named if 'moe' in n and id(p) not in adapter_ids]
    backbone_p = [p for n, p in named if 'moe' not in n and id(p) not in adapter_ids]

    groups = []
    if backbone_p:
        groups.append({'params': backbone_p,
                       'lr': args.lr_peak * args.backbone_lr_scale, 'name': 'backbone'})
    if adapter_p:
        groups.append({'params': adapter_p,
                       'lr': args.lr_peak * getattr(args, 'adapter_lr_scale', 1.0),
                       'name': 'adapter'})
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
