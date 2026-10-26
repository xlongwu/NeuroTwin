# coding=utf-8
"""优化器、调度器、EMA 与预训练权重加载等训练基础设施。"""
import math
import os
from copy import deepcopy
from typing import List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models import ARCH_VERSION
from models.tfm import NeuroTwinTFM

#: 无 arch_version 元数据的旧检查点按此版本处理
LEGACY_ARCH_VERSION = 1
ARCH_POLICIES = ('require_match', 'warn', 'ignore')

#: 主干参数键前缀（--load_backbone_only 使用）：不含 one-step / CPM 预测头与条件模块
BACKBONE_KEY_PREFIXES = (
    'rev_norm.', 'sc_prior.', 'patch_embed.', 'blocks.', 'final_norm.',
)


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


def pretrain_config_signature(saved_args: dict) -> dict:
    """从检查点保存的参数快照提取「任务口径签名」：任务模式 + 模型架构 + 结构维度。

    next_timepoint 公共维度：in_window=1、pred_window=预测偏移数、
    seq_len=context_max。预测偏移数由 forecast_offsets 推导
    （与 utils.common.resolve_forecast_offsets 同一口径）。
    ``model_arch``（缺省按 tfm 处理）区分主干代际；TFM 额外纳入
    patch_len / cpm_horizon（决定 patch 网格与 CPM 头的输出 shape）。
    """
    saved = dict(saved_args or {})
    if 'task_mode' not in saved:
        raise RuntimeError(
            "检查点缺少 task_mode 字段（旧任务检查点），无法校验任务口径；"
            "请用当前 next_timepoint 口径重新预训练。")
    sig = {'task_mode': str(saved['task_mode']),
           'model_arch': str(saved.get('model_arch', 'tfm')),
           'in_window': 1,
           'pred_window': len(saved_forecast_offsets(saved)),
           'seq_len': int(saved.get('context_max'))}
    if sig['model_arch'] == 'tfm':
        sig['patch_len'] = int(saved.get('tfm_patch_len'))
        sig['cpm_horizon'] = int(saved.get('cpm_horizon'))
    return sig


def saved_forecast_offsets(saved_args: dict) -> list:
    """从检查点参数快照解析 next_timepoint 的预测偏移列表。"""
    saved = dict(saved_args or {})
    raw = saved.get('forecast_offsets', None)
    vals = []
    if raw:
        vals = [int(v) for v in str(raw).replace(',', ' ').split() if v.strip()]
    if not vals:
        vals = [1]
    return sorted(vals)


def check_pretrain_config_compat(pretrained_path, expected_dims: dict,
                                 arch_policy: str = 'require_match',
                                 horizon_policy: str = 'require_match') -> None:
    """校验预训练权重与当前运行的任务口径是否一致。

    不同 context_max / 预测偏移数对应的输入/输出 shape 不同；若直接交给
    :func:`load_backbone_weights` 的 shape 过滤，只会打印“跳过 N 个键”而不报错，
    存在静默错误加载风险。这里显式拦截（``--pretrained_arch_policy warn`` 可强制跳过）。
    """
    if arch_policy not in ARCH_POLICIES:
        raise ValueError(f"Unsupported arch_policy '{arch_policy}', expected one of {ARCH_POLICIES}")
    if horizon_policy not in ('require_match', 'repeat_last'):
        raise ValueError(f'Unsupported horizon_policy {horizon_policy!r}')
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    _, _, config = unwrap_state_dict(ckpt)
    saved_args = (config or {}).get('args', None) if isinstance(config, dict) else None
    if not isinstance(saved_args, dict):
        if horizon_policy != 'require_match':
            raise RuntimeError('CPM horizon 迁移需要 checkpoint config.args 快照')
        print(f"提示: {os.path.basename(str(pretrained_path))} 无 config.args 快照，"
              "跳过任务口径校验（仅按 shape 过滤加载权重）。")
        return
    saved = pretrain_config_signature(saved_args)
    expected = {'task_mode': expected_dims['task_mode'],
                'model_arch': str(expected_dims.get('model_arch', 'tfm')),
                'in_window': int(expected_dims['in_window']),
                'pred_window': int(expected_dims['pred_window']),
                'seq_len': int(expected_dims['seq_len'])}
    if expected['model_arch'] == 'tfm':
        expected['patch_len'] = int(expected_dims['patch_len'])
        expected['cpm_horizon'] = int(expected_dims['cpm_horizon'])
    mismatch = {k: (saved.get(k), expected[k]) for k in expected
                if saved.get(k) != expected[k]}
    if horizon_policy == 'repeat_last' and 'cpm_horizon' in mismatch:
        src_h, dst_h = mismatch['cpm_horizon']
        if not 0 < src_h < dst_h:
            raise RuntimeError(f'repeat_last 只允许扩展 CPM horizon：{src_h}→{dst_h}')
        del mismatch['cpm_horizon']
        print(f'[pretrain 兼容性] 显式 CPM horizon 迁移：{src_h}→{dst_h} (repeat_last)')
    if not mismatch:
        print(f"[pretrain 兼容性] 任务口径一致: {expected}")
        return
    msg = (f"预训练权重任务口径与当前运行不一致：{mismatch}\n"
           f"    权重: {saved}\n    当前: {expected}\n"
           "    next_timepoint 任务（连续 BOLD context → 下一 TR 全脑状态）在不同 "
           "context_max / 预测偏移数下输入输出 shape 语义不同，权重不可混用，"
           "否则大量键会因 shape 不匹配被静默跳过。")
    if arch_policy == 'require_match':
        raise RuntimeError(
            msg + "\n请用当前任务口径重新预训练（scripts/pretrain.sh），"
                  "或显式指定 --pretrained_arch_policy warn 强制按 shape 加载。")
    print("警告: " + msg + f"（arch_policy={arch_policy}：继续按 shape 过滤加载）")


def load_backbone_weights(model, pretrained_path, device, arch_policy='require_match',
                          skip_pattern=''):
    """按形状兼容过滤加载预训练权重（新增模块自动跳过）。

    Args:
        arch_policy: 架构版本不匹配时的策略
            - 'require_match'（默认）：直接报错并提示重新预训练
            - 'warn'：打印警告后按 shape 过滤加载（逃生舱）
            - 'ignore'：跳过权重加载
        skip_pattern: fnmatch 模式（如 ``'one_step_head.*'``），命中的预训练键
            不加载、保留随机初始化。用于受控消融：量化「某分支未经预训练」
            对 finetune 结果的独立影响。
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
                msg + f"\n请用当前代码重新预训练（scripts/pretrain.sh），"
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
        print(f"提示: {len(missing)} 个键未从预训练文件加载（如新增的条件模块）")

    return {'loaded': len(filtered), 'skipped': skipped, 'missing': missing,
            'arch_version': arch_version, 'skipped_load': False}


def load_backbone_only(model, pretrained_path, device):
    """只加载主干权重，预测头 / 条件模块保持随机初始化（prompt §三十六）。

    用途：跨任务口径复用主干（例如已经用同一 context_max 口径预训练过主干、
    仅想换预测头）做受控的「主干预训练 + 头部重训」实验。

    与 :func:`load_backbone_weights`（按 shape 过滤、静默跳过不匹配键）不同，
    这里**显式打印四类清单**：loaded（成功加载）、missing（主干中不存在该键）、
    incompatible（键存在但形状不匹配）、reinitialized
    （被有意重新初始化的头部/条件模块），避免 strict=False 的静默忽略。
    """
    prefixes = BACKBONE_KEY_PREFIXES
    ckpt = torch.load(pretrained_path, map_location=device)
    state, arch_version, _ = unwrap_state_dict(ckpt)
    fname = os.path.basename(str(pretrained_path))
    model_sd = model.state_dict()

    loaded, missing, incompatible, reinitialized = [], [], [], []
    for k, v in model_sd.items():
        if k.startswith(prefixes):
            src = state.get(k, None)
            if src is None:
                missing.append(k)
            elif tuple(src.shape) != tuple(v.shape):
                incompatible.append((k, tuple(src.shape), tuple(v.shape)))
            else:
                model_sd[k] = src
                loaded.append(k)
        else:
            reinitialized.append(k)
    model.load_state_dict(model_sd)

    print(f"[load_backbone_only] {fname}（arch_version={arch_version}）| "
          f"loaded={len(loaded)} | missing={len(missing)} | "
          f"incompatible={len(incompatible)} | "
          f"reinitialized(头部/条件模块)={len(reinitialized)}")
    print(f"[load_backbone_only] reinitialized 模块前缀: "
          f"{sorted({k.split('.')[0] for k in reinitialized})}")
    if missing:
        print(f"[load_backbone_only] 主干中缺失的键（{len(missing)} 个，例如 "
              f"{missing[:5]}）：checkpoint 中不存在，已保留随机初始化。")
    if incompatible:
        detail = ', '.join(f"{k} {src}→{dst}"
                           for k, src, dst in incompatible[:5])
        print(f"[load_backbone_only] 形状不兼容的键（{len(incompatible)} 个，例如 "
              f"{detail}）：已显式重新初始化（不静默忽略）。")
    return {'loaded': len(loaded), 'missing': missing,
            'incompatible': incompatible, 'reinitialized': reinitialized,
            'arch_version': arch_version}


def _iter_conditioning_modules(model):
    """发现带 `is_conditioning_adapter` 标记的条件模块（AdaLN/LoRA 等）。

    冻结/参数分组基于显式标记而非名字子串匹配，避免重命名导致行为漂移。
    """
    seen = set()
    for _, module in model.named_modules():
        if getattr(module, 'is_conditioning_adapter', False) and id(module) not in seen:
            seen.add(id(module))
            yield module


# ══════════════════════════════════════════════════════════════════════════
#  条件参数登记、冻结校验与显式迁移（v1 revised §8 / §11）
# ══════════════════════════════════════════════════════════════════════════

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


def conditioning_module_prefixes(model) -> set:
    """条件适配器模块的键前缀集合（用**模块路径**，不是参数首段）。

    必须按完整模块路径取前缀：``one_step_head.film`` 才是适配器，而同级的
    ``one_step_head.transition`` / ``.decode`` 是无条件主干部件。若退化成
    ``one_step_head.`` 会把主干键也当成「预期新增」，让迁移器失去拦截结构漂移
    的能力。
    """
    out = set()
    for mod_name, module in model.named_modules():
        if getattr(module, 'is_conditioning_adapter', False) and mod_name:
            out.add(mod_name + '.')
    return out






def param_report(model, optimizer=None) -> dict:
    """登记实际参数量（方案 §11：不硬套旧方案的 35040/36388 等数字）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cond_names = set(conditioning_param_names(model))
    cond_total = sum(p.numel() for n, p in model.named_parameters() if n in cond_names)
    cond_trainable = sum(p.numel() for n, p in model.named_parameters()
                         if n in cond_names and p.requires_grad)
    opt_total = (sum(p.numel() for g in optimizer.param_groups for p in g['params'])
                 if optimizer is not None else None)
    opt_distinct = (len({id(p) for g in optimizer.param_groups for p in g['params']})
                    if optimizer is not None else None)
    return {
        'total': int(total),
        'trainable': int(trainable),
        'conditioning': int(cond_total),
        'conditioning_trainable': int(cond_trainable),
        'rho_conditioning': float(cond_total / total) if total else 0.0,
        'optimizer_numel': opt_total,
        'optimizer_distinct_tensors': opt_distinct,
    }


def verify_conditioning_param_groups(model, optimizer, expect_frozen_backbone=False,
                                     allow_frozen_conditioning=False):
    """校验条件参数：全部进入优化器、无重复、numel 一致（验收 L）。

    同时检查「解冻后不会永久静默冻结」：条件模块的参数在冻结与解冻两态都必须
    出现在优化器参数组里，否则说明分阶段解冻漏掉了新增顶层模块。
    """
    cond = conditioning_param_names(model)
    named = dict(model.named_parameters())
    in_opt = {}
    for gi, g in enumerate(optimizer.param_groups):
        for p in g['params']:
            in_opt.setdefault(id(p), []).append(gi)
    dup = {k: v for k, v in in_opt.items() if len(v) > 1}
    problems = []
    missing, untrainable = [], []
    for n in cond:
        p = named[n]
        if not p.requires_grad:
            untrainable.append(n)
        elif id(p) not in in_opt:
            missing.append(n)
    cond_numel = sum(named[n].numel() for n in cond if named[n].requires_grad)
    opt_cond_numel = sum(named[n].numel() for n in cond
                         if named[n].requires_grad and id(named[n]) in in_opt)
    if missing:
        problems.append(f'条件参数未进入优化器（{len(missing)} 个，例如 {missing[:5]}）')
    if untrainable and not allow_frozen_conditioning:
        problems.append(f'条件参数 requires_grad=False（{len(untrainable)} 个，例如 '
                        f'{untrainable[:5]}）；冻结期条件模块必须可训练')
    elif untrainable:
        print(f'[params] {len(untrainable)} 个条件参数已冻结')
    if dup:
        problems.append(f'优化器存在重复参数（{len(dup)} 个张量出现在多个参数组）')
    if cond_numel != opt_cond_numel:
        problems.append(f'条件参数 numel 不一致：应训练 {cond_numel}，实入优化器 '
                        f'{opt_cond_numel}')
    if expect_frozen_backbone:
        backbone_trainable = sum(1 for n, p in named.items()
                                 if p.requires_grad and n not in set(cond))
        if backbone_trainable:
            problems.append(f'冻结期主干仍可训练（{backbone_trainable} 个参数）')
    if problems:
        raise RuntimeError('[条件参数校验] 失败：\n  - ' + '\n  - '.join(problems))
    return {'conditioning_params': len(cond), 'conditioning_numel': int(cond_numel),
            'optimizer_groups': [g.get('name', f'g{i}')
                                 for i, g in enumerate(optimizer.param_groups)]}


def classify_checkpoint_keys(model_state, ckpt_state, conditioning_prefixes):
    """把模型键分为 loaded / new_initialized / zero_expanded / skipped / missing。

    ``conditioning_prefixes`` 为条件模块的键前缀集合：它们**预期**不出现在旧
    checkpoint 中（新增结构），其余键缺失即视为语义问题（结构漂移），必须报错
    而不是静默随机初始化。
    """
    loaded, new_initialized, missing, skipped = [], [], [], []
    for k, v in model_state.items():
        src = ckpt_state.get(k, None)
        if src is None:
            if any(k.startswith(p) for p in conditioning_prefixes):
                new_initialized.append(k)
            else:
                missing.append(k)
        elif tuple(src.shape) != tuple(v.shape):
            # 同宽度但语义变化也会被 layout_hash 拦下；这里按形状冲突归类
            missing.append(f'{k} (shape {tuple(src.shape)}→{tuple(v.shape)})')
        else:
            loaded.append(k)
    skipped = sorted(k for k in ckpt_state if k not in model_state)
    return {'loaded': loaded, 'new_initialized': new_initialized,
            'missing': missing, 'skipped': skipped}


def migrate_pretrained_checkpoint(model, pretrained_path, device='cpu',
                                  arch_policy='require_match',
                                  zero_expanded_keys=(), layout_expect=None,
                                  layout_saved=None, horizon_policy='require_match'):
    r"""显式迁移器：旧 checkpoint → 新结构，逐键归类并汇报（方案 §8）。

    与「``strict=False`` + shape 过滤」不同，本函数把每个键归入
    ``loaded / copied / new_initialized / zero_expanded / skipped / missing``
    六类之一，任何**未分类**的键或语义冲突都直接报错：

    - ``loaded/copied``：同名同形状，逐位复制；
    - ``new_initialized``：新增条件模块的参数（旧文件里没有，属预期）；
    - ``zero_expanded``：调用方显式声明「按零初始化扩展」的键（例如给已有 FiLM
      追加调制维度），必须列在白名单里，否则报错；
    - ``skipped``：checkpoint 有、模型没有的键（旧模块被移除）；
    - ``missing``：模型有、checkpoint 没有、且**不属于**条件模块的键 → 结构漂移，
      报错。

    ``layout_expect`` / ``layout_saved``：条件向量的 ``layout_hash``。**字段顺序
    参与哈希**，因此「同宽度但字段顺序变化」也会被拒绝（验收 K）。

    ``horizon_policy='repeat_last'`` 仅显式扩展 CPM horizon embedding：保留
    原有行并重复最后一行填充新步数；报告写入 ``horizon_expanded``，其余冲突
    仍然报错。
    """
    if arch_policy not in ARCH_POLICIES:
        raise ValueError(f"Unsupported arch_policy '{arch_policy}'")
    if layout_expect and layout_saved and str(layout_expect) != str(layout_saved):
        raise RuntimeError(
            f'条件布局冲突：checkpoint layout_hash={layout_saved} 与当前运行 '
            f'{layout_expect} 不一致。字段名称/顺序/来源列/类别词表任一变化都会改变'
            f'哈希；即使宽度相同也**拒绝加载**（否则同 shape 的列会被静默错位复用）。')

    ckpt = torch.load(pretrained_path, map_location=device)
    state, arch_version, config = unwrap_state_dict(ckpt)
    if (getattr(model, 'conditioner', None) is not None
            and not any(key.startswith('conditioner.') for key in state)):
        raise ValueError('条件微调需要匹配条件编码与统计量的预训练权重；不再支持无条件实验起点。')
    fname = os.path.basename(str(pretrained_path))
    if arch_version is None:
        print(f'提示: {fname} 无 arch_version，按 {LEGACY_ARCH_VERSION} 处理。')
        arch_version = LEGACY_ARCH_VERSION
    if arch_version != ARCH_VERSION:
        msg = (f'架构版本不匹配: checkpoint arch_version={arch_version}, '
               f'当前 ARCH_VERSION={ARCH_VERSION}')
        if arch_policy == 'require_match':
            raise RuntimeError(
                msg + '。请使用显式迁移入口或重新预训练；'
                      '如需按 shape 强制加载（不推荐），传 --pretrained_arch_policy warn。')
        if arch_policy == 'ignore':
            print(msg + '：arch_policy=ignore，跳过加载。')
            return {'loaded': 0, 'copied': 0, 'new_initialized': [],
                    'zero_expanded': [], 'skipped': sorted(state), 'missing': [],
                    'arch_version': arch_version}
        print('警告: ' + msg + '（arch_policy=warn：继续迁移）')

    prefixes = set(conditioning_module_prefixes(model))
    cond_prefixes = tuple(sorted(prefixes))
    model_state = model.state_dict()
    horizon_expanded = []
    if horizon_policy not in ('require_match', 'repeat_last'):
        raise ValueError(f'Unsupported horizon_policy {horizon_policy!r}')
    if horizon_policy == 'repeat_last':
        key = 'cpm_head.horizon_embed'
        src, dst = state.get(key), model_state.get(key)
        if src is None or dst is None or src.ndim != 2 or dst.ndim != 2:
            raise RuntimeError(f'CPM horizon 迁移需要二维 {key}')
        if src.shape != dst.shape:
            saved_h = ((config or {}).get('args') or {}).get('cpm_horizon')
            if (src.shape[1] != dst.shape[1] or not 0 < src.shape[0] < dst.shape[0]
                    or saved_h != src.shape[0]):
                raise RuntimeError(f'CPM horizon 无法扩展：{tuple(src.shape)}→{tuple(dst.shape)}')
            expanded = src[-1:].expand_as(dst).clone()
            expanded[:src.shape[0]].copy_(src)
            state = dict(state)
            state[key] = expanded
            horizon_expanded.append({'key': key, 'source_steps': src.shape[0],
                                     'target_steps': dst.shape[0], 'policy': horizon_policy})
            print(f'[migrate] {key}: 复制 H1–{src.shape[0]}，'
                  f'H{src.shape[0]+1}–{dst.shape[0]} 重复最后一行（无随机丢弃）')
    cls = classify_checkpoint_keys(model_state, state, cond_prefixes)
    if horizon_policy != 'require_match' and cls['skipped']:
        raise RuntimeError(f'CPM horizon 迁移存在无法加载的源参数：{cls["skipped"]}')
    zero_expanded = [k for k in cls['new_initialized'] if k in set(zero_expanded_keys)]
    plain_new = [k for k in cls['new_initialized'] if k not in set(zero_expanded_keys)]
    # zero_expanded 键允许在 checkpoint 中存在但形状不同（按零填充扩展）
    shape_conflicts = [k for k in cls['missing'] if ' (shape ' in k]
    hard_missing = [k for k in cls['missing'] if ' (shape ' not in k]
    unknown_keys = sorted(set(zero_expanded_keys) - set(cls['new_initialized']))
    if hard_missing:
        raise RuntimeError(
            f'迁移失败：{len(hard_missing)} 个非条件键在 checkpoint 中缺失'
            f'（例如 {hard_missing[:5]}）。这表示主干结构发生了变化，不应静默随机初始化。')
    if shape_conflicts:
        raise RuntimeError(
            f'迁移失败：{len(shape_conflicts)} 个键形状冲突（例如 {shape_conflicts[:3]}）；'
            f'同宽度不同语义的权重不可复用，需要显式 --zero_expanded_keys 声明的'
            f'零扩展迁移。')
    if unknown_keys:
        raise RuntimeError(
            f'--zero_expanded_keys 含未在预期新增集合中的键（{unknown_keys[:5]}）；'
            f'未分类键必须报错，不能静默处理。')

    filtered = {k: state[k] for k in cls['loaded']}
    model_state.update(filtered)
    model.load_state_dict(model_state)
    report = {
        'loaded': len(cls['loaded']), 'copied': len(cls['loaded']),
        'new_initialized': plain_new, 'zero_expanded': zero_expanded,
        'skipped': cls['skipped'], 'missing': cls['missing'],
        'arch_version': arch_version, 'file': fname,
        'horizon_expanded': horizon_expanded,
    }
    print(f"[migrate] {fname}（arch_version={arch_version}）| loaded/copied="
          f"{report['loaded']} | new_initialized={len(plain_new)} | "
          f"zero_expanded={len(zero_expanded)} | skipped={len(cls['skipped'])} | "
          f"missing={len(cls['missing'])}")
    if plain_new:
        print(f"[migrate] 新增条件模块键示例: {plain_new[:5]}")
    return report


def unfreeze_while_preserving_state(model, optimizer, args):
    """训练中途解冻主干，但**保留已训练条件参数的 Adam 状态与 LR 进度**。

    旧写法是解冻后重建 optimizer + scheduler：AdamW 重建会让条件参数已积累的
    ``exp_avg`` / ``exp_avg_sq`` 全部丢失，且 scheduler 从 warmup 重新开始，等于
    把解冻前的条件训练进度抹掉（默认 freeze_backbone_epochs=10 < warmup_epochs=15
    时每次运行都会发生）。这里改为只把新可训练的参数**追加**进现有 param group。

    返回 ``(added_numel, group_names)``，供调用方打印可核对的证据。
    """
    existing = {id(p) for g in optimizer.param_groups for p in g['params']}
    added_groups, added_numel = [], 0
    for group in get_param_groups(model, args):
        fresh = [p for p in group['params'] if id(p) not in existing]
        if fresh:
            # 沿用组名与 LR：已存在的组不重建，避免任何 Adam 状态被丢弃
            target = next((g for g in optimizer.param_groups
                           if g.get('name') == group.get('name')), None)
            if target is not None:
                target['params'].extend(fresh)
            else:
                added_groups.append(group)
            added_numel += sum(p.numel() for p in fresh)
    for group in added_groups:
        optimizer.add_param_group(group)
    return added_numel, [g.get('name') for g in optimizer.param_groups]


def set_finetune_trainability(model: NeuroTwinTFM, backbone_unfrozen: bool = False):
    """微调分阶段训练：默认训练条件模块（病理 FiLM 等），backbone_unfrozen 后解冻主干。"""
    for p in model.parameters():
        p.requires_grad = False
    # 条件模块在冻结阶段即参与训练：它们是“让病理条件真正生效”的唯一路径
    for module in _iter_conditioning_modules(model):
        for p in module.parameters():
            p.requires_grad = True

    if backbone_unfrozen:
        backbone_modules = [model.patch_embed, model.blocks, model.final_norm,
                            model.one_step_head, model.cpm_head,
                            model.sc_prior]
        if model.rev_norm is not None:
            backbone_modules.append(model.rev_norm)
        # graph_ode 路线（方案 §10）：个体 profile 编码器属于共享主干解冻列表；
        # VarianceHead 在 GraphODEDynamicsHead 内部，随 one_step_head 一起解冻。
        # 条件 FiLM 等适配器已由上方条件模块循环解冻，不在此重复。
        if getattr(model, 'profile_encoder', None) is not None:
            backbone_modules.append(model.profile_encoder)
        for module in backbone_modules:
            for p in module.parameters():
                p.requires_grad = True
        if getattr(model, '_orig_mod', None) is not None:
            # 冻结期编译的反向图可能继续省略主干梯度，即使 requires_grad 已改为 True。
            # 清理缓存后由同一包装器重新编译；参数对象及 optimizer/EMA 状态保持不变。
            torch.compiler.reset()


def get_param_groups(model: NeuroTwinTFM, args):
    """按模式划分参数组。

    - pretrain: 单组 'all'（HC 预训练允许条件模块，此时条件参数随主干一起训练）
    - finetune: 'backbone' / 'condition'，分别设置学习率。
      条件组 LR 倍率取 ``condition_lr_scale``（方案 §11.2 的搜索维度），
      未提供时回退历史 ``adapter_lr_scale`` 以保持旧配置可复现。
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.mode == 'pretrain':
        return [{'params': [p for _, p in named], 'lr': args.lr_peak, 'name': 'all'}]

    adapter_ids = set()
    for module in _iter_conditioning_modules(model):
        for p in module.parameters():
            adapter_ids.add(id(p))

    adapter_p = [p for _, p in named
                 if id(p) in adapter_ids]
    backbone_p = [p for _, p in named
                  if id(p) not in adapter_ids]

    cond_scale = getattr(args, 'condition_lr_scale', None)
    if cond_scale is None:
        cond_scale = getattr(args, 'adapter_lr_scale', 1.0)
    groups = []
    if backbone_p:
        groups.append({'params': backbone_p,
                       'lr': args.lr_peak * args.backbone_lr_scale, 'name': 'backbone'})
    if adapter_p:
        groups.append({'params': adapter_p,
                       'lr': args.lr_peak * float(cond_scale), 'name': 'condition'})
    return groups


def build_optimizer(model, criterion, args):
    """AdamW；loss 的 log-variance 参数单独成组（可配置学习率比例、无 weight decay）。"""
    groups = get_param_groups(model, args)
    loss_p = [p for p in criterion.parameters() if p.requires_grad]
    if loss_p:
        groups.append({'params': loss_p, 'lr': args.lr_peak * args.loss_lr_scale,
                       'name': 'loss_weight', 'weight_decay': 0.0})
    return AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))


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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
