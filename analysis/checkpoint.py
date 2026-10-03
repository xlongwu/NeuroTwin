# coding=utf-8
"""checkpoint 加载（仅支持当前 next_timepoint 任务产出的检查点格式）。"""
import logging
from typing import Dict, List, Tuple

import torch

log = logging.getLogger(__name__)

def load_checkpoint_compat(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    """加载 train.optim.save_backbone_weights 产出的检查点（或裸 state_dict）。

    取错层级时（例如把整份元数据 dict 当 state_dict）load_state_dict 会把全部
    权重记为 missing，strict=False 下静默跳过、模型停在随机初始化——必须显式
    报错，不允许用兜底掩盖权重未加载。旧任务（滑窗/chunk 轨迹）检查点一律
    不再兼容。
    """
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and isinstance(raw.get('state_dict'), dict):
        state_dict = raw['state_dict']
    else:
        state_dict = raw
    if not isinstance(state_dict, dict) or not any('.' in str(k) for k in state_dict):
        raise ValueError(
            f'无法从 {ckpt_path} 解析出参数 state_dict；'
            f'顶层键={list(raw) if isinstance(raw, dict) else type(raw)}')

    result = model.load_state_dict(state_dict, strict=strict)
    compat_info = {
        'missing': list(result.missing_keys) if not strict else [],
        'unexpected': list(result.unexpected_keys) if not strict else [],
    }
    return model, compat_info

