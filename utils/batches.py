"""训练与评估共用的张量批次搬运及条件切片。"""

import torch


def prepare_next_point_batch(batch, device):
    """搬运预测批次，将目标从 [B,F,H] 扩展为模型输出形状 [B,F,H,1]。"""
    history = batch["x"].to(device, non_blocking=True)
    targets = batch["y"].to(device, non_blocking=True).unsqueeze(-1)
    last_state = batch["x_last"].to(device, non_blocking=True)
    target_mask = batch.get("target_mask", None)
    if target_mask is not None:
        target_mask = target_mask.to(device, non_blocking=True)
    graph = batch["sc"].to(device, non_blocking=True)
    return (history, targets, last_state, target_mask, graph)


def prepare_condition_batch(batch, device):
    """把字段级条件张量搬到 device，返回模型 ``condition=`` 所需的 dict 或 None。

    ``cond_raw`` 只服务诊断/置换对照，不进模型，因此这里刻意不透传。
    三个必需键缺一即报错：静默降级成「无条件」会让对照臂悄悄变成不同的模型。
    """
    if "cond_z" not in batch:
        return None
    missing = [k for k in ("cond_z", "cond_code", "cond_mask") if k not in batch]
    if missing:
        raise KeyError(f"条件批次缺少必需键 {missing}")
    condition = {
        "cond_z": batch["cond_z"].to(device, non_blocking=True),
        "cond_code": batch["cond_code"].to(device, non_blocking=True),
        "cond_mask": batch["cond_mask"].to(device, non_blocking=True),
    }
    if "subject_index" in batch:
        condition["subject_index"] = batch["subject_index"].to(
            device, non_blocking=True
        )
    return condition


def slice_condition_batch(cond, index):
    """按布尔/整数索引切片条件批次（rollout 只取被选中的样本）。"""
    if cond is None:
        return None
    return {k: v[index] if torch.is_tensor(v) else v for k, v in cond.items()}


def prepare_dynamics_inputs(batch, device):
    """搬运 graph_ode 动态输入：calibration_bold [B,F,1,L] 与 controls [B,H,F]。

    缺少控制序列时返回 controls=None，表示零外部控制。
    """
    calibration_bold = batch.get("calibration_bold", None)
    if calibration_bold is not None:
        calibration_bold = calibration_bold.to(device, non_blocking=True)
    controls = batch.get("controls", None)
    if controls is not None:
        controls = controls.to(device, non_blocking=True)
    return (calibration_bold, controls)
