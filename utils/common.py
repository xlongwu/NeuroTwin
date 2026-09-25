# coding=utf-8
"""通用工具：随机种子与 CLI 布尔/数值列表参数解析。"""
import argparse
import os
import random

import numpy as np
import torch


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
