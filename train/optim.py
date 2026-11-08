"""方案 B 的冻结策略、参数组、AdamW、相对余弦调度与 EMA。"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from models import ARCH_VERSION

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


def conditioning_module_prefixes(model) -> set:
    """按完整模块路径识别 conditioner 和 CPM FiLM，供冻结及参数分组使用。"""
    out = set()
    for mod_name, module in model.named_modules():
        if getattr(module, 'is_conditioning_adapter', False) and mod_name:
            out.add(mod_name + '.')
    return out


def conditioning_param_names(model) -> List[str]:
    """列出所有条件适配器参数名（顶层模块携带 ``is_conditioning_adapter``）。

    逐名校验（验收 L）依赖这份清单：冻结期可训练集合、optimizer 参数组、去重和
    numel 都必须与之对应；新增顶层条件模块若忘记打标，会在此处暴露为「未登记」。
    """
    prefixes = conditioning_module_prefixes(model)
    names = []
    for pname, _ in model.named_parameters():
        parts = pname.split('.')
        for i in range(len(parts) - 1, 0, -1):
            if '.'.join(parts[:i]) + '.' in prefixes:
                names.append(pname)
                break
    return sorted(names)


def prediction_adapter_prefixes(model) -> tuple:
    return tuple(name + '.' for name, module in model.named_modules()
                 if name and getattr(module, 'is_prediction_adapter', False))


def prediction_adapter_param_names(model) -> List[str]:
    prefixes = prediction_adapter_prefixes(model)
    return sorted(name for name, _ in model.named_parameters() if name.startswith(prefixes))


class _RelativeCosineLR(LambdaLR):
    """相对余弦调度；同时覆盖训练中途追加的参数组。

    finetune 的 backbone 组在冻结期结束后才加入 optimizer，构造 scheduler 时
    还不存在。只按 ``base_lrs`` 调度的实现会漏掉它，使其学习率保持恒定。
    """

    def sync_param_groups(self) -> None:
        """把新增 optimizer 组纳入当前进度，并立即按当前因子设置学习率。"""
        while len(self.base_lrs) < len(self.optimizer.param_groups):
            group = self.optimizer.param_groups[len(self.base_lrs)]
            base_lr = float(group['lr'])
            self.base_lrs.append(base_lr)
            self.lr_lambdas.append(self.lr_lambdas[0])
            group['lr'] = base_lr * self.lr_lambdas[0](self.last_epoch)

    def step(self, epoch=None):
        self.sync_param_groups()
        return super().step(epoch)


def build_scheduler(optimizer, args):
    """Linear warmup + cosine decay with a per-group relative final LR.

    ``CosineAnnealingLR`` applies one absolute ``eta_min`` to every group.  A
    group whose base LR is below ``eta_min`` (e.g. the backbone at
    ``lr_peak * backbone_lr_scale``) would then be pushed *up* during decay.
    Using a relative factor keeps every group on its own peak-to-final path.
    """
    if args.lr_peak <= 0:
        raise ValueError(f'lr_peak 必须 > 0，收到 {args.lr_peak}')
    if not 0 <= args.lr_final <= args.lr_peak:
        raise ValueError(
            f'lr_final 必须位于 [0, lr_peak]（lr_peak={args.lr_peak}，'
            f'lr_final={args.lr_final}）')

    total_epochs = max(1, int(args.train_epochs))
    warmup_epochs = max(0, int(args.warmup_epochs))
    start_factor = max(1e-4, args.lr_init / args.lr_peak)
    final_factor = args.lr_final / args.lr_peak

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return start_factor + (1.0 - start_factor) * (epoch / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_factor + (1.0 - final_factor) * cosine

    return _RelativeCosineLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(path, model, meta: dict) -> str:
    inner = getattr(model, '_orig_mod', model)
    torch.save({'state_dict': deepcopy(inner.state_dict()),
                'arch_version': ARCH_VERSION, 'config': meta}, path)
    return str(path)


def read_checkpoint(path) -> dict:
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(raw, dict) or raw.get('arch_version') != ARCH_VERSION:
        raise ValueError('需要方案 B 精简权重；旧权重请先通过 scripts/migrate_b_checkpoint.py 显式迁移')
    if not isinstance(raw.get('state_dict'), dict) or not isinstance(raw.get('config'), dict):
        raise ValueError(f'权重缺少 state_dict/config: {path}')
    return raw


def load_initial_weights(model, path, args) -> dict:
    """显式复制相同键；只允许登记的动态模块和HC→MDD临床字段为新增键。"""
    raw = read_checkpoint(path)
    saved, source = raw['config']['args'], raw['state_dict']
    if saved['mode'] != 'pretrain':
        raise ValueError('HC 初始化及 MDD 微调起点必须为 HC 权重')
    keys = ('num_rois', 'context_max', 'tfm_dim', 'tfm_layers', 'tfm_heads',
            'tfm_ff_ratio', 'tfm_patch_len', 'cpm_horizon', 'cpm_layers',
            'norm', 'dropout', 'sc_prior_rank', 'sc_delta_a', 'sc_delta_rank',
            'sc_lambda_mode', 'sc_lambda_init', 'sc_sinkhorn_iters')
    mismatch = {k: (saved[k], getattr(args, k)) for k in keys if saved[k] != getattr(args, k)}
    if mismatch:
        raise ValueError(f'初始化结构不匹配: {mismatch}')
    if saved.get('multiscale_phase', False) and getattr(args, 'multiscale_phase', False):
        dynamic_keys = ('multiscale_channels', 'phase_modes', 'tr_seconds')
        mismatch = {k: (saved[k], getattr(args, k)) for k in dynamic_keys if saved[k] != getattr(args, k)}
        if mismatch:
            raise ValueError(f'动态模块初始化结构不匹配: {mismatch}')
    condition = raw['config']['condition']
    layout = model.conditioner.layout()
    source_layout = condition['meta']['layout_hash']
    if args.mode == 'pretrain':
        if source_layout != layout['layout_hash'] or condition['stats'] != model.conditioner.stats.to_dict():
            raise ValueError('HC 热启动条件布局或训练统计量未逐位继承')
    elif source_layout != layout['demo_layout_hash']:
        raise ValueError('HC→MDD 共享三字段布局不匹配')
    target = model.state_dict()
    extra = sorted(set(source) - set(target))
    new = sorted(set(target) - set(source))
    new_prefixes = (() if args.mode == 'pretrain' else ('conditioner.delta.',) + tuple(
        f'conditioner.features.{kind}.{name}.'
        for name in model.conditioner.new_group for kind in ('continuous', 'categorical')))
    if not saved.get('multiscale_phase', False) and getattr(args, 'multiscale_phase', False):
        new_prefixes += prediction_adapter_prefixes(model)
    expected_new = {key for key in target if key.startswith(new_prefixes)}
    if extra or set(new) != expected_new:
        raise ValueError(f'初始化存在未登记参数: extra={extra}, new={new}')
    conflicts = [key for key in source if source[key].shape != target[key].shape]
    if conflicts:
        raise ValueError(f'HC→MDD 参数形状冲突: {conflicts}')
    target.update(source)
    model.load_state_dict(target, strict=True)
    return {'source': str(path), 'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'key_mapping': {key: key for key in sorted(source)},
            'copied': sorted(source), 'new_initialized': new}


def set_trainability(model, mode: str, encoder_unfrozen: bool) -> None:
    """保留原B冻结策略，新增预测适配器在HC/MDD冻结期始终可训练。"""
    inner = getattr(model, '_orig_mod', model)
    for p in inner.parameters():
        p.requires_grad_(encoder_unfrozen)
    cond_names = set(conditioning_param_names(inner))
    dynamics = set(prediction_adapter_param_names(inner))
    for name, p in inner.named_parameters():
        if name in cond_names or name in dynamics or (mode == 'pretrain' and name.startswith('cpm_head.')):
            p.requires_grad_(True)
    if encoder_unfrozen and getattr(model, '_orig_mod', None) is not None:
        torch.compiler.reset()


def set_training_mode(model, encoder_unfrozen: bool) -> None:
    model.train()
    if not encoder_unfrozen:
        inner = getattr(model, '_orig_mod', model)
        inner.eval()
        inner.cpm_head.train()
        if inner.conditioner is not None:
            inner.conditioner.train()
        for _, module in inner.named_modules():
            if getattr(module, 'is_prediction_adapter', False):
                module.train()


def get_param_groups(model, args) -> list:
    inner = getattr(model, '_orig_mod', model)
    adapters = set(conditioning_param_names(inner))
    dynamics = set(prediction_adapter_param_names(inner))
    buckets = {name: [] for name in ('encoder', 'cpm', 'condition', 'prediction')}
    for name, p in inner.named_parameters():
        if p.requires_grad:
            group = ('condition' if name in adapters else
                     'prediction' if name in dynamics else
                     'cpm' if name.startswith('cpm_head.') else 'encoder')
            buckets[group].append(p)
    scales = dict(encoder=args.backbone_lr_scale, cpm=1.0, condition=1.0, prediction=1.0)
    return [{'params': params, 'lr': args.lr_peak * scales[name], 'name': name}
            for name, params in buckets.items() if params]


def build_optimizer(model, criterion, args):
    return AdamW(get_param_groups(model, args), weight_decay=args.weight_decay, betas=(0.9, 0.95))


def unfreeze_while_preserving_state(model, optimizer, args) -> tuple:
    existing = {id(p) for g in optimizer.param_groups for p in g['params']}
    added = 0
    for group in get_param_groups(model, args):
        fresh = [p for p in group['params'] if id(p) not in existing]
        if not fresh:
            continue
        target = next((g for g in optimizer.param_groups if g['name'] == group['name']), None)
        if target is None:
            optimizer.add_param_group({**group, 'params': fresh})
        else:
            target['params'].extend(fresh)
        added += sum(p.numel() for p in fresh)
    return added, [g['name'] for g in optimizer.param_groups]
