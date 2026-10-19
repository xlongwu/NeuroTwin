# coding=utf-8
"""Checkpoint 配置读取、模型重建与权重加载。"""

import inspect
from argparse import Namespace
from typing import Dict, List, Tuple

import torch

from models.tfm import NeuroTwinTFM
from utils.condition_schema import (
    condition_stats_from_bundle as condition_stats_from_bundle,
)


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
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        state_dict = raw["state_dict"]
    else:
        state_dict = raw
    if not isinstance(state_dict, dict) or not any("." in str(k) for k in state_dict):
        raise ValueError(
            f"无法从 {ckpt_path} 解析出参数 state_dict；"
            f"顶层键={list(raw) if isinstance(raw, dict) else type(raw)}"
        )

    result = model.load_state_dict(state_dict, strict=strict)
    compat_info = {
        "missing": list(result.missing_keys) if not strict else [],
        "unexpected": list(result.unexpected_keys) if not strict else [],
    }
    return model, compat_info


_KWARG_ALIAS_TFM = {
    "features": "num_rois",
    "patch_len": "tfm_patch_len",
    "dim": "tfm_dim",
    "num_layers": "tfm_layers",
    "num_heads": "tfm_heads",
    "ff_ratio": "tfm_ff_ratio",
    "one_step_hidden_dim": "tfm_one_step_hidden_dim",
    "use_revin": "norm",
}


def load_train_args(ckpt_path):
    """从 checkpoint 读取训练时保存的完整参数快照。"""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config") if isinstance(raw, dict) else None
    saved = cfg.get("args") if isinstance(cfg, dict) else None
    if not isinstance(saved, dict):
        raise ValueError(
            f"{ckpt_path} 缺少 config.args 配置快照，无法重建模型；请确认该权重由当前版本 main.py 保存。"
        )
    return Namespace(**saved)


def load_condition_bundle(ckpt_path):
    """读取 checkpoint 的条件协议块（config / train-only 统计量 / 元数据）。

    ``condition_config`` 与 ``condition_stats_payload`` 都**不是** argparse 参数，
    而是训练期由数据拟合得到的结构输入；重建模型必须从这里取回，否则条件模块不会
    存在、strict 加载会把条件键判为 unexpected。
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config") if isinstance(raw, dict) else None
    cond = (cfg or {}).get("condition") if isinstance(cfg, dict) else None
    if not isinstance(cond, dict) or not cond.get("config"):
        return None
    return {
        "config": cond["config"],
        "stats": cond.get("stats"),
        "meta": cond.get("meta") or {},
        "split_hash": cond.get("split_hash"),
    }


def build_model_from_args(args, device, condition_bundle=None):
    """按 inspect.signature 自省从参数快照重建与训练一致的 NeuroTwinTFM。

    条件注入的结构参数（``condition_config`` / ``condition_stats_payload``）不在
    ``config.args`` 快照里，必须由 ``condition_bundle``（checkpoint 的
    ``config.condition``）提供；缺一都会让 strict 加载立即失败而不是静默降级。
    """
    model_arch = str(getattr(args, "model_arch", "tfm"))
    if model_arch != "tfm":
        raise ValueError(
            f"checkpoint 的 model_arch='{model_arch}'：legacy NeuroTwin 架构已移除，无法重建该模型。请改用 NeuroTwinTFM（model_arch=tfm）的 checkpoint。"
        )
    sig = inspect.signature(NeuroTwinTFM.__init__).parameters
    kwargs = {}
    for name, param in sig.items():
        if name == "self":
            continue
        if name == "features":
            kwargs[name] = int(args.num_rois)
        elif name == "context_max":
            kwargs[name] = int(getattr(args, "context_max"))
        else:
            alias = _KWARG_ALIAS_TFM.get(name, name)
            if hasattr(args, alias):
                kwargs[name] = getattr(args, alias)
            elif param.default is inspect.Parameter.empty:
                raise ValueError(f"参数快照缺少必填构造参数：{name}")
    kwargs["pretrain_mode"] = str(getattr(args, "mode", "finetune")) == "pretrain"
    arm = str(getattr(args, "condition_arm", "none") or "none")
    if arm != "none":
        if condition_bundle is None:
            raise ValueError(
                f"训练快照 condition_arm={arm}，但 checkpoint 无 config.condition：无法重建条件编码器。请用当前代码重新训练，或显式提供带条件元数据的权重。"
            )
        kwargs["condition_config"] = condition_bundle["config"]
        kwargs["condition_stats_payload"] = condition_bundle["stats"]
    if getattr(args, "dynamics_mode", None) != "graph_ode":
        raise ValueError("Only Graph-ODE checkpoints are supported")
    if getattr(args, "graph_source", None) != "history_fc":
        raise ValueError("Only history_fc checkpoints are supported")
    if getattr(args, "cond_inject", "heads") != "heads":
        raise ValueError("Checkpoint uses a removed backbone condition adapter")
    return NeuroTwinTFM(**kwargs).to(device)
