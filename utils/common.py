# coding=utf-8
"""通用工具：随机种子、CLI 布尔/数值列表参数解析与任务维度推导。"""
import argparse
import os
import random

import numpy as np
import torch

#: 当前唯一任务口径：连续 BOLD context → 下一 TR 全脑状态
TASK_MODES = ('next_timepoint',)

#: 主干架构：legacy=原 NeuroTwin（BrainMDM/GraphODE）；tfm=TimesFM-3 风格 TFM 主干
MODEL_ARCHES = ('legacy', 'tfm')

#: next_timepoint 任务下 --enable_mtp 未显式给出 --forecast_offsets 时的默认预测偏移
DEFAULT_MTP_OFFSETS = (1, 2, 4, 8)


def parse_float_list(text, name: str):
    """把逗号/空格分隔的字符串解析为 float 列表；空值返回 None。"""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        vals = [float(v) for v in text]
    else:
        vals = [float(v) for v in str(text).replace(',', ' ').split() if v.strip()]
    if not vals:
        return None
    if any(v < 0 for v in vals):
        raise ValueError(f"--{name} 不允许负数：{vals}")
    return vals


def parse_int_list(text, name: str):
    """把逗号/空格分隔的字符串解析为 int 列表；空值返回 None。

    用于 --context_lengths / --forecast_offsets / --eval_rollout_horizons 等
    正整数列表参数（非正整数直接报错，避免静默截断成 0 造成无意义配置）。
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        vals = [int(v) for v in text]
    else:
        vals = [int(v) for v in str(text).replace(',', ' ').split() if v.strip()]
    if not vals:
        return None
    if any(v < 1 for v in vals):
        raise ValueError(f"--{name} 必须全部为正整数：{vals}")
    return vals


def resolve_forecast_offsets(args):
    """解析 next_timepoint 任务的预测偏移列表（相对当前 TR 的 +Δ）。

    - ``--forecast_offsets`` 显式给出时以它为准（必须严格递增且互不相同）；
    - 否则 ``--enable_mtp True`` 时取 :data:`DEFAULT_MTP_OFFSETS`（[1,2,4,8]），
      关闭时为单步 ``[1]``。
    返回值同时用于数据侧目标构造、模型 pred_window 维度与损失权重对齐。
    """
    explicit = parse_int_list(getattr(args, 'forecast_offsets', None), 'forecast_offsets')
    if explicit is not None:
        offsets = explicit
    elif bool(getattr(args, 'enable_mtp', False)):
        offsets = list(DEFAULT_MTP_OFFSETS)
    else:
        offsets = [1]
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"--forecast_offsets 存在重复值：{offsets}")
    if sorted(offsets) != offsets:
        raise ValueError(f"--forecast_offsets 必须严格递增：{offsets}")
    return offsets


def resolve_mtp_weights(args, offsets):
    """解析逐偏移损失权重（长度必须等于 offsets 数；空值回退到均匀权重）。"""
    vals = parse_float_list(getattr(args, 'mtp_weights', None), 'mtp_weights')
    if vals is None:
        return [1.0] * len(offsets)
    if len(vals) != len(offsets):
        raise ValueError(
            f"--mtp_weights 长度 {len(vals)} 与预测偏移数 {len(offsets)}（{offsets}）不一致")
    if sum(vals) <= 0:
        raise ValueError(f"--mtp_weights 不能全为 0：{vals}")
    return vals


def resolve_task_dims(args):
    """把 CLI 参数解析为「模型实际使用的窗口/序列/预测维度」。

    Next-Timepoint 任务（next_timepoint，连续 BOLD → 下一 TR 全脑状态）：
        in_window = 1（整段 context 作为“单个窗口”，窗口轴恒为 1）
        pred_window = len(forecast_offsets)（默认 [1] → 1；MTP 时 = 4）
        seq_len = args.context_max（context TR 数上界，= 模型 S 轴建模宽度）

    说明：实际 context 长度 K 可以是任意 K <= context_max（S 轴前缀切片支持），
    参数集合不随 K 变化。``model_arch`` 区分 legacy / tfm 主干；tfm 的
    one-step 头只建模 +1 偏移（方案 §10），要求 forecast_offsets == [1]，
    稀疏偏移 {1,2,4,8} 由 CPM 头的 horizon 覆盖（§18 Task C）。
    """
    task_mode = str(getattr(args, 'task_mode', 'next_timepoint'))
    if task_mode not in TASK_MODES:
        raise ValueError(f"task_mode 仅支持 {TASK_MODES}，收到 '{task_mode}'")
    model_arch = str(getattr(args, 'model_arch', 'legacy'))
    if model_arch not in MODEL_ARCHES:
        raise ValueError(f"model_arch 仅支持 {MODEL_ARCHES}，收到 '{model_arch}'")
    dims = _resolve_next_timepoint_dims(args)
    dims['model_arch'] = model_arch
    if model_arch == 'tfm':
        if dims['pred_window'] != 1:
            raise ValueError(
                f"--model_arch tfm 的 one-step 头只建模 +1 偏移（方案 §10），"
                f"要求 forecast_offsets == [1]，收到 {dims['pred_window']} 个偏移；"
                "多步预测请用 --cpm_horizon 控制 CPM 头的 horizon。")
        patch_len = int(getattr(args, 'tfm_patch_len', 4))
        cpm_horizon = int(getattr(args, 'cpm_horizon', 8))
        if patch_len < 1 or patch_len > dims['seq_len']:
            raise ValueError(
                f"--tfm_patch_len 须在 [1, context_max={dims['seq_len']}] 内，"
                f"收到 {patch_len}")
        if cpm_horizon < 1:
            raise ValueError(
                f"--cpm_horizon 必须 >= 1，收到 {cpm_horizon}")
        dims['patch_len'] = patch_len
        dims['cpm_horizon'] = cpm_horizon
    return dims


def _resolve_next_timepoint_dims(args):
    """next_timepoint 任务的维度与合法性校验（context 长度 / 偏移 / 训练范式）。"""
    context_min = int(getattr(args, 'context_min', 16))
    context_max = int(getattr(args, 'context_max', 64))
    if context_min < 1:
        raise ValueError(f"--context_min 必须 >= 1，收到 {context_min}")
    if context_max < context_min:
        raise ValueError(
            f"--context_max({context_max}) 必须 >= --context_min({context_min})")

    lengths = parse_int_list(getattr(args, 'context_lengths', None), 'context_lengths')
    if lengths is not None:
        bad = [k for k in lengths if k < context_min or k > context_max]
        if bad:
            raise ValueError(
                f"--context_lengths 中的 {bad} 超出 [context_min, context_max] = "
                f"[{context_min}, {context_max}]")

    eval_k = int(getattr(args, 'eval_context_length', 0) or 0)
    if eval_k == 0:
        eval_k = context_max          # 0 = auto：与模型建模宽度一致（默认 64）
    if eval_k < 1 or eval_k > context_max:
        raise ValueError(
            f"--eval_context_length 必须在 [1, context_max={context_max}] 内，收到 {eval_k}")

    offsets = resolve_forecast_offsets(args)

    causal = str(getattr(args, 'causal_training', 'random_context'))
    if causal == 'full_sequence':
        raise NotImplementedError(
            "--causal_training full_sequence（GPT 式全序列 teacher-forcing、一次前向"
            "监督多个时间位置）未实现：现有主干的时间轴算子是双向卷积/池化 + 窗口维"
            "注意力（BrainMDM conv kernel=3/5、GraphODE temporal_branch、MDM 池化），"
            "要支持位置级 causal masking 需要重写这些算子（prompt §九 方案 A 被显式"
            "允许降级）。当前实现为方案 B：随机 context → 下一时间点，未来信息"
            "从设计上不进入前向（causality 由数据构造保证）。")
    if causal != 'random_context':
        raise ValueError(
            f"--causal_training 仅支持 random_context（方案 B），收到 '{causal}'")

    pred_target = str(getattr(args, 'prediction_target', 'delta'))
    if pred_target not in ('delta', 'absolute'):
        raise ValueError(
            f"--prediction_target 仅支持 delta/absolute，收到 '{pred_target}'")

    return {'task_mode': 'next_timepoint', 'in_window': 1,
            'pred_window': len(offsets), 'seq_len': context_max}


def parse_pred_quantiles(text):
    """把逗号分隔的 --pred_quantiles 解析为浮点元组；空值回退到默认分位点。"""
    vals = [v.strip() for v in str(text).split(',') if v.strip()]
    if not vals:
        return (0.1, 0.5, 0.9)
    if len(vals) < 2:
        raise ValueError(f"--pred_quantiles 至少需要 2 个分位点，收到 {text!r}")
    return tuple(float(v) for v in vals)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
