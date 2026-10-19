"""NeuroTwin-TFM 训练入口。

训练基础设施（损失/图正则/优化器/EMA）位于 train/ 包，
通用工具（set_seed/str2bool）位于 utils/common.py；
CLI 定义位于 train/cli.py，验证选优与恢复状态位于 train/training_state.py；
本文件负责训练与验证循环。
"""

import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            self.log_dir = kwargs.get("log_dir", None)

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None


from models.conditioning import ConditioningConfig
from models.tfm import NeuroTwinTFM
from train.cli import parse_args as parse_args
from train.losses import (
    TFMDualLoss,
    compute_cpm_mask,
    compute_graph_regularization,
    spatial_pcc,
)
from train.optim import (
    ModelEMA,
    build_optimizer,
    build_scheduler,
    check_pretrain_config_compat,
    count_trainable_params,
    load_backbone_weights,
    migrate_pretrained_checkpoint,
    param_report,
    save_backbone_weights,
    set_finetune_stage,
    unfreeze_while_preserving_state,
)
from train.training_state import (
    DYNAMICS_CKPT_HORIZON as DYNAMICS_CKPT_HORIZON,
)
from train.training_state import (
    CheckpointSelector as CheckpointSelector,
)
from train.training_state import (
    aggregate_dynamics_subject_metric as aggregate_dynamics_subject_metric,
)
from train.training_state import (
    dynamics_anchor_mae as dynamics_anchor_mae,
)
from train.training_state import (
    dynamics_metric_key as dynamics_metric_key,
)
from train.training_state import (
    dynamics_metric_name as dynamics_metric_name,
)
from train.training_state import (
    resolve_dynamics_metric_horizon as resolve_dynamics_metric_horizon,
)
from utils.batches import (
    prepare_condition_batch,
    prepare_dynamics_inputs,
    prepare_next_point_batch,
    slice_condition_batch,
)
from utils.common import (
    parse_int_list,
    resolve_forecast_offsets,
    resolve_task_dims,
    set_seed,
)
from utils.common import (
    parse_pathology_fields as parse_pathology_fields,
)
from utils.condition_schema import parse_extra_codes
from utils.dataloader import (
    CONDITION_FILE_BY_MODE,
    CONDITION_GROUP_BY_MODE,
    NeuroTwinDataLoader,
)
from utils.dataloader import (
    condition_loader_kwargs as condition_loader_kwargs,
)

PBAR_MIN_BAR = 12
PBAR_OVERHEAD = 30


def pbar_ncols() -> int:
    """进度条整行的目标列宽。

    重定向到文件/管道时 `shutil.get_terminal_size()` 走 `COLUMNS` → 80 的回退，
    因此日志行宽与终端里一致，不会因为宽度未知而每次刷新另起一行。
    """
    return max(
        PBAR_MIN_BAR + PBAR_OVERHEAD, min(int(shutil.get_terminal_size().columns), 120)
    )


def make_pbar(iterable, total: int, desc: str) -> tqdm:
    """宽度自适应的单行进度条。

    `tqdm` 的 `dynamic_ncols` 只收缩 `{bar}`：desc 与 postfix 一旦超过终端宽度，
    整行仍然折行，表现为每个 iteration 打印一行。这里固定 `bar_format`（去掉变长的
    remaining/rate 字段）、把 desc 截进宽度预算，并配合 :func:`pbar_metrics` 裁剪
    指标，使整行始终占一行。
    """
    cols = pbar_ncols()
    return tqdm(
        iterable,
        total=total,
        leave=False,
        ncols=cols,
        dynamic_ncols=sys.stderr.isatty(),
        mininterval=0.5,
        desc=desc[: max(8, cols - PBAR_OVERHEAD)],
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
    )


def pbar_metrics(pbar: tqdm, **metrics) -> None:
    """把指标拼成 `k=v` 串，并按剩余列宽从右往左丢弃放不下的项。

    `tqdm` 会在字符串 postfix 前自动加 `', '`，因此预算里扣掉这 2 列。
    """
    cols = pbar.ncols if isinstance(pbar.ncols, int) else pbar_ncols()
    count_w = max(len(str(pbar.total)), len(str(pbar.n)))
    budget = (
        cols - PBAR_MIN_BAR - PBAR_OVERHEAD - count_w - 2 - len(str(pbar.desc or ""))
    )
    shown, used = ([], 0)
    for name, value in metrics.items():
        part = f"{name}={value}"
        need = len(part) + (1 if shown else 0)
        if used + need > budget:
            break
        shown.append(part)
        used += need
    pbar.set_postfix_str(" ".join(shown), refresh=False)


def _normalized_future(model, fut: torch.Tensor, norm_stats):
    """用当前历史的 RevIN 统计量归一化未来真值（方差损失目标）。

    fut: [B,F,H]（数据尺度）→ [B,F,H]（归一化空间）。norm_stats 为 None
    （--norm 关闭 RevIN）时直接返回原值，不再另行估计统计量。
    """
    if norm_stats is None:
        return fut
    revin = getattr(model, "rev_norm", None)
    if revin is None:
        return fut
    return revin.normalize_with(fut.unsqueeze(-1), norm_stats).squeeze(-1)


def split_hash(dl, args) -> str:
    """被试划分的规范哈希：所有对照臂必须共享同一 cohort/split。

    只依赖 train/val/test 的被试名单（排序后），因此不同 seed 的重复运行、以及
    同一划分下的不同条件臂得到相同哈希；名单变化立即暴露。
    """
    import hashlib
    import json

    payload = {
        "mode": str(args.mode),
        "train": sorted(dl.train_subjects),
        "val": sorted(dl.val_subjects),
        "test": sorted(dl.test_subjects),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _report_conditioning(model, optimizer, args, dl):
    """登记实际参数量并校验条件参数全部进入优化器。"""
    from train.optim import param_report, verify_conditioning_param_groups

    rep = param_report(model, optimizer)
    print(
        f"[params] total={rep['total']} | conditioning={rep['conditioning']} | trainable={rep['trainable']} | rho_cond={rep['rho_conditioning']:.4f} | optimizer_numel={rep['optimizer_numel']} | distinct_tensors={rep['optimizer_distinct_tensors']}"
    )
    if getattr(model, "conditioner", None) is not None:
        print(
            f"[params] 条件编码器内部: {model.conditioner.param_report()} | layout={model.conditioner.layout()['layout_hash']}"
        )
        verify_conditioning_param_groups(model, optimizer)
        print("[params] 条件参数组校验通过（全部入优化器、无重复、numel 一致）")
    return rep


def load_condition_stats_payload(path):
    """从 checkpoint 读取 train-only 条件统计量快照（HC→MDD 继承用）。"""
    import json

    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--condition_inherit_stats 指定的文件不存在: {path}")
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "fields" not in payload:
            payload = ((payload.get("config") or {}).get("condition") or {}).get(
                "stats"
            )
    else:
        raw = torch.load(p, map_location="cpu", weights_only=False)
        cfg = raw.get("config") if isinstance(raw, dict) else None
        payload = (
            (cfg or {}).get("condition", {}).get("stats")
            if isinstance(cfg, dict)
            else None
        )
    if not isinstance(payload, dict) or "fields" not in payload:
        raise KeyError(
            f"{path} 中没有可用的条件统计量快照（需要 config.condition.stats 或裸 stats JSON）：无法继承 HC 的 Age/education 统计量与类别协议。"
        )
    return payload


def _masked_l1_weighted(pred, target, weight, eps=1e-08):
    """按 [B,H] 权重（广播到 [B,F,H,1]）加权的逐元素 L1。"""
    w = weight.to(pred.dtype).view(weight.shape[0], 1, weight.shape[1], 1)
    w = w.expand_as(pred)
    return (torch.abs(pred - target) * w).sum() / w.sum().clamp_min(eps)


@torch.no_grad()
def evaluate_tfm(
    model, data_loader, criterion, device, args, rollout_steps=0, rollout_horizons=None
):
    """Evaluate next state, CPM and continuous dynamics with subject-equal aggregation."""
    model.eval()
    meters = defaultdict(list)
    subj_mae_acc = defaultdict(lambda: [0.0, 0])
    dyn_metric_horizon = resolve_dynamics_metric_horizon(args)
    dyn_subj_acc = defaultdict(lambda: [0.0, 0])
    dyn_subjects_seen = set()
    dyn_anchor_total = 0
    rollout_acc = defaultdict(lambda: [0.0, 0, 0])
    cpm_acc = defaultdict(lambda: [0.0, 0, 0])
    rollout_hs = [
        int(h) for h in rollout_horizons or [] if int(h) <= int(rollout_steps)
    ]
    horizon = int(getattr(model, "cpm_horizon", 0) or args.cpm_horizon)
    pbar = make_pbar(data_loader, len(data_loader), "[Val-TFM]")
    for batch in pbar:
        history, targets, last_state, target_mask, graph = prepare_next_point_batch(batch, device)
        cond = prepare_condition_batch(batch, device)
        calibration_bold, controls = prepare_dynamics_inputs(batch, device)
        fut = batch["future"].to(device, non_blocking=True)
        fmask = batch["future_mask"].to(device, non_blocking=True)
        outputs, aux_info = model(
            history,
            graph,
            condition=cond,
            condition_epoch=None,
            controls=controls[:, :horizon] if controls is not None else None,
            calibration_bold=calibration_bold,
        )
        cpm_pred = aux_info["cpm_pred"]
        cpm_target = fut[:, :, :horizon]
        if controls is not None:
            cpm_mask = compute_cpm_mask(controls[:, :horizon], fmask[:, :horizon])
        else:
            cpm_mask = fmask[:, :horizon]
        dyn_kwargs = {}
        if aux_info.get("dynamics_pred", None) is not None:
            norm_stats = aux_info.get("norm_stats", None)
            dyn_kwargs = dict(
                dynamics_pred=aux_info["dynamics_pred"],
                dynamics_target=cpm_target,
                dynamics_mask=fmask[:, :horizon],
                dynamics_mean_norm=aux_info.get("dynamics_mean_norm", None),
                dynamics_logvar_norm=aux_info.get("dynamics_logvar_norm", None),
                dynamics_target_norm=_normalized_future(model, cpm_target, norm_stats),
            )
        total_loss, loss_stats = criterion(
            outputs, targets, cpm_pred, cpm_target, cpm_mask, **dyn_kwargs
        )
        graph_reg, graph_stats = compute_graph_regularization(
            aux_info,
            device,
            sparsity_weight=args.sc_sparsity_weight,
            entropy_weight=args.sc_entropy_weight,
            temporal_weight=args.sc_temporal_weight,
        )
        total_loss = total_loss + graph_reg
        b, f, h, _ = outputs.shape
        if target_mask is None:
            target_mask = torch.ones(b, h, device=outputs.device, dtype=outputs.dtype)
        w4 = target_mask.view(b, 1, h, 1)
        mae = _masked_l1_weighted(outputs, targets, target_mask)
        per_sample_mae = ((outputs - targets).abs() * w4.expand_as(outputs)).sum(
            dim=(1, 2, 3)
        ) / w4.expand_as(outputs).sum(dim=(1, 2, 3)).clamp_min(1e-08)
        subj_ids = batch.get("subj_id", None)
        if subj_ids is not None:
            for i, sid in enumerate(subj_ids):
                acc = subj_mae_acc[str(sid)]
                acc[0] += float(per_sample_mae[i].item())
                acc[1] += 1
        if aux_info.get("dynamics_pred", None) is not None:
            if subj_ids is None:
                raise KeyError(
                    "batch 缺少 subj_id：graph_ode 多步判据必须先按被试平均再跨被试等权平均，不能用 batch 均值替代。"
                )
            dyn_pred = aux_info["dynamics_pred"]
            dyn_anchor_mae, dyn_full = dynamics_anchor_mae(
                dyn_pred, fut, fmask, horizon=dyn_metric_horizon
            )
            dyn_anchor_total += int(dyn_pred.shape[0])
            for i, sid in enumerate(subj_ids):
                sid = str(sid)
                dyn_subjects_seen.add(sid)
                if bool(dyn_full[i]):
                    acc = dyn_subj_acc[sid]
                    acc[0] += float(dyn_anchor_mae[i].item())
                    acc[1] += 1
        rmse = ((outputs - targets) ** 2 * w4.expand_as(outputs)).sum() / w4.expand_as(
            outputs
        ).sum().clamp_min(1e-08)
        rmse = rmse.clamp_min(0).sqrt()
        pcc_spatial = (spatial_pcc(outputs, targets) * target_mask).sum() / target_mask.sum().clamp_min(1e-08)
        x_t = history[:, :, 0, -1:].unsqueeze(-1)
        x_prev = history[:, :, 0, -2:-1].unsqueeze(-1)
        pers = x_t.expand(b, f, h, 1)
        trend = (x_t + (x_t - x_prev)).expand(b, f, h, 1)
        pers_mae = _masked_l1_weighted(pers, targets, target_mask)
        trend_mae = _masked_l1_weighted(trend, targets, target_mask)
        cpm_mae_sum = 0.0
        cpm_mae_weight = 0.0
        for hi in range(horizon):
            mh = cpm_mask[:, hi].unsqueeze(1)
            n = float(mh.sum().item())
            if n <= 0:
                continue
            err = (cpm_pred[:, :, hi] - cpm_target[:, :, hi]).abs()
            wsum = float((err * mh).sum().item())
            cpm_acc[hi + 1][0] += wsum
            cpm_acc[hi + 1][1] += n
            cpm_acc[hi + 1][2] += n * f
            cpm_mae_sum += wsum
            cpm_mae_weight += n * f
            pers_err = (x_t[:, :, 0, 0] - cpm_target[:, :, hi]).abs()
            cpm_acc.setdefault(f"pers_{hi + 1}", [0.0, 0, 0])
            cpm_acc[f"pers_{hi + 1}"][0] += float((pers_err * mh).sum().item())
            cpm_acc[f"pers_{hi + 1}"][1] += n
            cpm_acc[f"pers_{hi + 1}"][2] += n * f
        metric_cpm_mae = cpm_mae_sum / cpm_mae_weight if cpm_mae_weight > 0 else 0.0
        for k, v in {
            **loss_stats,
            **graph_stats,
            "metric_total": total_loss.detach(),
            "metric_mae_next": mae.detach(),
            "metric_rmse_next": rmse.detach(),
            "metric_pcc_next": pcc_spatial.detach(),
            "metric_persistence_mae": pers_mae.detach(),
            "metric_trend_mae": trend_mae.detach(),
            "metric_cpm_mae": torch.tensor(metric_cpm_mae),
        }.items():
            meters[k].append(v.detach())
        pbar_metrics(
            pbar,
            MAE=f"{float(mae):.4f}",
            PCC=f"{float(pcc_spatial):.4f}",
            CPM=f"{metric_cpm_mae:.4f}",
        )
        if rollout_hs and "rollout_flag" in batch:
            rf = batch["rollout_flag"].bool()
            rf_gpu = rf.to(device)
            if rf_gpu.any():
                hist = history[rf_gpu][:, :, 0, :].transpose(1, 2).contiguous()
                roll = model.rollout_next_states(
                    hist,
                    graph[rf_gpu],
                    steps=int(rollout_steps),
                    condition=slice_condition_batch(cond, rf_gpu),
                    controls=controls[rf_gpu][:, : int(rollout_steps)]
                    if controls is not None
                    else None,
                    calibration_bold=calibration_bold[rf_gpu] if calibration_bold is not None else None,
                )
                fut_r = (
                    batch["future"][rf].transpose(1, 2).contiguous().to(device).float()
                )
                fmask_r = batch["future_mask"][rf].to(device).float()
                for rh in rollout_hs:
                    ok = fmask_r[:, rh - 1] > 0
                    k = int(ok.sum())
                    if k:
                        rollout_acc[rh][0] += float(
                            (roll[ok, rh - 1, :] - fut_r[ok, rh - 1, :]).abs().sum()
                        )
                        rollout_acc[rh][1] += k
                        rollout_acc[rh][2] += k * int(roll.shape[-1])
    for hi, (err_sum, _n_tasks, n_elem) in cpm_acc.items():
        if isinstance(hi, int) and n_elem:
            meters[f"cpm_mae_H{hi}"].append(torch.tensor(err_sum / n_elem))
    for key, (err_sum, _n_tasks, n_elem) in cpm_acc.items():
        if isinstance(key, str) and key.startswith("pers_") and n_elem:
            meters[f"cpm_persistence_mae_H{key.split('_')[1]}"].append(
                torch.tensor(err_sum / n_elem)
            )
    for rh, (err_sum, _n_tasks, n_elem) in rollout_acc.items():
        if n_elem:
            meters[f"rollout_mae_H{rh}"].append(torch.tensor(err_sum / n_elem))
    result = {k: torch.stack(vs).mean().item() for k, vs in meters.items()}
    if subj_mae_acc:
        subj_means = [s / n for s, n in subj_mae_acc.values() if n > 0]
        result["metric_mae_next_batchmean"] = result["metric_mae_next"]
        result["metric_mae_next"] = sum(subj_means) / len(subj_means)
        result["metric_mae_next_n_subjects"] = float(len(subj_means))
    if dyn_anchor_total:
        dyn_key = dynamics_metric_key(dyn_metric_horizon)
        metric_value, n_subj, n_anchors = aggregate_dynamics_subject_metric(
            dyn_subj_acc, dyn_anchor_total, horizon=dyn_metric_horizon
        )
        result[dyn_key] = metric_value
        result[f"{dyn_key}_n_subjects"] = float(n_subj)
        result[f"{dyn_key}_n_anchors"] = float(n_anchors)
        result[f"{dyn_key}_n_subjects_without_full_anchor"] = float(
            len(dyn_subjects_seen) - n_subj
        )
        result[f"{dyn_key}_n_anchors_total"] = float(dyn_anchor_total)
    return result


def check_seed_separation(args):
    """模型 seed / 采样 seed / Condition Dropout seed 必须是三条独立随机流。

    任意两者同值等于把两条流并为一条，「分离」名存实亡（``--seed`` 与 ``--sampling_seed``
    的历史默认值都是 2024，正好撞车）。这里拒绝，而不是静默改默认值；登记值见
    ``utils/condition_schema.py`` 的 MODEL_SEED / SAMPLING_SEED / DROPOUT_SEED。
    """
    flags = ("seed", "sampling_seed", "cond_dropout_seed")
    absent = [f for f in flags if not hasattr(args, f)]
    if absent:
        raise ValueError(
            f"缺少随机来源参数 {absent}：请经 main.py 的 argparse 入口调用，不要构造缺参的 Namespace。"
        )
    seen = {}
    for flag in flags:
        value = int(getattr(args, flag))
        if value in seen:
            raise ValueError(
                f"随机来源必须两两分离：--{flag}={value} 与 --{seen[value]} 同值。"
            )
        seen[value] = flag


def main(args):
    args.model_arch = "tfm"
    check_seed_separation(args)
    if getattr(args, "device", ""):
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and args.amp
    torch.set_num_threads(args.num_threads)
    if device.type == "cuda" and args.tf32:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)
    dims = resolve_task_dims(args)
    forecast_offsets = resolve_forecast_offsets(args)
    eval_rollout_horizons = parse_int_list(
        args.eval_rollout_horizons, "eval_rollout_horizons"
    ) or [1, 2, 4, 8, 16]
    needs_long_rollout = bool(args.eval_fc or args.eval_spectral)
    auto_steps = max(eval_rollout_horizons)
    if needs_long_rollout:
        auto_steps = max(auto_steps, int(args.fc_min_length))
    eval_rollout_steps = int(args.eval_rollout_steps) or auto_steps
    print("=" * 50)
    if args.mode == "pretrain":
        print(
            f"[Pretrain] 正在启动预训练 | task_mode={dims['task_mode']} | model_arch=tfm | CPM horizon: {dims['cpm_horizon']}"
        )
        print("=" * 50 + "\n" + "=" * 50)
        print("启动模式: [Stage 1] Healthy Control (HC) physics backbone pretraining")
    else:
        print(
            f"[Finetune] Starting finetuning | task_mode={dims['task_mode']} | model_arch=tfm | CPM horizon: {dims['cpm_horizon']}"
        )
        print("=" * 50 + "\n" + "=" * 50)
        print(
            "启动模式: [Stage 2] MDD individualized pathology finetuning [NeuroTwinTFM FiLM]"
        )
    print("=" * 50)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.checkpoint_dir, "runs", f"{args.name}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志已开启，保存路径: {log_dir}")
    print(
        f"[next_timepoint] 模型维度映射: in_window←1（整段 context 作为单窗口） | pred_window←len(forecast_offsets)={dims['pred_window']} | seq_len←context_max={dims['seq_len']} | forecast_offsets={forecast_offsets} | causal_training={args.causal_training}"
    )
    print(
        f"[next_timepoint] 训练 context: K∈[--context_min, --context_max]=[{args.context_min}, {args.context_max}]"
        + (
            f"（离散 {parse_int_list(args.context_lengths, 'context_lengths')}）"
            if args.context_lengths
            else "（连续区间随机采样）"
        )
        + f" | 评估固定 K_eval={int(args.eval_context_length) or args.context_max} | anchors/被试={args.eval_anchors_per_subject} | rollout: R={eval_rollout_steps}, horizons={eval_rollout_horizons}, 锚点/被试={args.eval_rollout_tasks_per_subject}"
    )
    cond_enabled = bool(getattr(args, "condition_arm", None) not in (None, "", "none"))
    condition_group = (
        (
            str(args.condition_group)
            if getattr(args, "condition_group", None)
            else CONDITION_GROUP_BY_MODE.get(args.mode)
        )
        if cond_enabled
        else None
    )
    if cond_enabled:
        print(
            f"[condition] 阶段={args.mode} 分组={condition_group} 臂={args.condition_arm} | merged 文件={args.condition_file or CONDITION_FILE_BY_MODE.get(args.mode)} （条件数据**只**来自该 merged 表）"
        )
    cond_base_stats = None
    if cond_enabled and getattr(args, "condition_inherit_stats", ""):
        from utils.condition_schema import DEMO_GROUP, load_inherited_stats, schema_for

        payload = load_condition_stats_payload(args.condition_inherit_stats)
        target_specs = schema_for(condition_group or "mdd8")
        src_names = [f["name"] for f in payload.get("fields", [])]
        share = [n for n in DEMO_GROUP if n in src_names]
        cond_base_stats = load_inherited_stats(payload, target_specs, share)
        print(
            f"[condition] 已载入继承统计量快照：{args.condition_inherit_stats} | 来源字段={src_names} → 继承共享字段={share}；其余字段将在 MDD-train 上拟合（不会覆盖继承值）"
        )
    dl = NeuroTwinDataLoader(
        data_root=args.data_root,
        mode=args.mode,
        batch_size=args.batch_size,
        condition_group=condition_group,
        condition_file=args.condition_file or None,
        split_file=args.split_file or None,
        stratify_key=args.stratify_key or None,
        condition_base_stats=cond_base_stats,
        condition_extra_codes=parse_extra_codes(args.condition_extra_codes)
        if args.condition_extra_codes
        else None,
        pathology_input_dim=args.pathology_input_dim,
        pathology_fields=parse_pathology_fields(args.pathology_fields),
        pathology_missing=args.pathology_missing,
        clinical_file=args.clinical_file,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        eval_batch_size=args.eval_batch_size,
        refresh_split_manifest=getattr(args, "refresh_split_manifest", False),
        task_mode=dims["task_mode"],
        bold_source=args.bold_source,
        random_context=args.random_context,
        random_cutoff=args.random_cutoff,
        train_samples_per_subject=args.train_samples_per_subject,
        subject_cache_size=args.subject_cache_size,
        sampling_seed=args.sampling_seed,
        context_min=args.context_min,
        context_max=args.context_max,
        context_lengths=parse_int_list(args.context_lengths, "context_lengths"),
        forecast_offsets=forecast_offsets,
        eval_context_length=int(args.eval_context_length) or args.context_max,
        eval_anchors_per_subject=args.eval_anchors_per_subject,
        eval_rollout_tasks_per_subject=args.eval_rollout_tasks_per_subject,
        eval_rollout_steps=eval_rollout_steps,
        train_rollout_steps=dims["cpm_horizon"],
        train_anchor_max_offset=dims["cpm_horizon"],
        eval_all_future_steps=dims["cpm_horizon"],
        num_rois=args.num_rois,
        graph_source=args.graph_source,
        calibration_length=args.calibration_length,
        control_manifest=args.control_manifest or None,
    )
    train_data, val_data = (dl.get_train(), dl.get_val())
    condition_cfg_dict = None
    condition_stats_payload = None
    if cond_enabled:
        condition_cfg_dict = ConditioningConfig(
            group=condition_group,
            arm=args.condition_arm,
            d_demo=args.cond_d_demo,
            d_new=args.cond_d_new,
            cond_dim=args.cond_dim,
            h_delta=args.cond_h_delta,
            delta_style=args.cond_delta_style,
            dropout_p=args.cond_dropout_p,
            dropout_seed=args.cond_dropout_seed,
        ).to_dict()
        condition_stats_payload = dl.get_condition_stats().to_dict()
        print(f"[condition] 编码器配置 {condition_cfg_dict}")
    model = NeuroTwinTFM(
        features=args.num_rois,
        context_max=dims["seq_len"],
        patch_len=args.tfm_patch_len,
        dim=args.tfm_dim,
        num_layers=args.tfm_layers,
        num_heads=args.tfm_heads,
        ff_ratio=args.tfm_ff_ratio,
        dropout=args.dropout,
        sc_prior_mode=args.sc_prior_mode,
        sc_lambda_mode=args.sc_lambda_mode,
        sc_lambda_init=args.sc_lambda_init,
        sc_prior_rank=args.sc_prior_rank,
        sc_delta_a=args.sc_delta_a,
        sc_delta_rank=args.sc_delta_rank,
        sc_sinkhorn_iters=args.sc_sinkhorn_iters,
        one_step_hidden_dim=args.tfm_one_step_hidden_dim,
        cpm_horizon=args.cpm_horizon,
        cpm_layers=args.cpm_layers,
        pretrain_mode=args.mode == "pretrain",
        condition_arm=args.condition_arm,
        condition_config=condition_cfg_dict,
        condition_stats_payload=condition_stats_payload,
        use_revin=args.norm,
        dynamics_mode=args.dynamics_mode,
        ode_steps=args.ode_steps,
        dynamics_dt=args.dynamics_dt,
        graph_source=args.graph_source,
        personalized=args.personalized,
        profile_dim=args.profile_dim,
    ).to(device)
    if args.mode == "finetune":
        if not os.path.exists(args.pretrained_weight):
            raise FileNotFoundError(f"未找到预训练权重: {args.pretrained_weight}")
        saved_layout = None
        same_group_arm = False
        if cond_enabled:
            from analysis.checkpoint import load_condition_bundle

            bundle = load_condition_bundle(args.pretrained_weight)
            if bundle is not None:
                saved_layout = (bundle.get("meta") or {}).get("layout_hash")
            if saved_layout and getattr(model, "conditioner", None) is not None:
                now = model.conditioner.layout()
                same_group_arm = (
                    bundle["config"].get("group") == condition_group
                    and bundle["config"].get("arm") == now["arm"]
                )
                if same_group_arm and saved_layout != now["layout_hash"]:
                    raise RuntimeError(
                        f"条件布局不一致：权重={saved_layout}，当前={now['layout_hash']}。字段顺序/来源列/类别词表任一变化都禁止复用该权重。"
                    )
                if not same_group_arm and saved_layout != now.get("demo_layout_hash"):
                    raise RuntimeError(
                        f"HC→MDD 共享三字段布局不一致：HC 权重 layout_hash={saved_layout}，当前 mdd8 的 demo 子集哈希={now.get('demo_layout_hash')}。Sex/Age/education 的顺序、类型或词表必须逐位一致才能继承。"
                    )
        if cond_enabled:
            check_pretrain_config_compat(
                args.pretrained_weight, dims, arch_policy=args.pretrained_arch_policy
            )
            migrate_pretrained_checkpoint(
                model,
                args.pretrained_weight,
                device,
                arch_policy=args.pretrained_arch_policy,
                layout_expect=model.conditioner.layout()["layout_hash"]
                if getattr(model, "conditioner", None) and same_group_arm
                else None,
                layout_saved=saved_layout if same_group_arm else None,
            )
        else:
            check_pretrain_config_compat(
                args.pretrained_weight, dims, arch_policy=args.pretrained_arch_policy
            )
            load_backbone_weights(
                model,
                args.pretrained_weight,
                device,
                arch_policy=args.pretrained_arch_policy,
            )
        set_finetune_stage(model, backbone_unfrozen=False)
        stage1_trainables = []
        if getattr(model.one_step_head, "film", None) is not None:
            stage1_trainables.append("One-Step FiLM")
        if getattr(model.cpm_head, "film", None) is not None:
            stage1_trainables.append("CPM FiLM")
        print(
            f"已进入分阶段微调: 前期仅训练条件适配器（{'+ '.join(stage1_trainables)}），后期逐步解冻主干。"
        )
    print(f"Total Model Parameters: {sum((p.numel() for p in model.parameters()))}")
    print(f"Trainable Parameters: {count_trainable_params(model)}")
    print(
        f"[TFM] dim={args.tfm_dim} layers={args.tfm_layers} heads={args.tfm_heads} patch_len={args.tfm_patch_len} ff_ratio={args.tfm_ff_ratio} | one_step_hidden={args.tfm_one_step_hidden_dim} | cpm_horizon={args.cpm_horizon} cpm_layers={args.cpm_layers} | revin={args.norm}"
    )
    print(
        f"[SC prior] mode={args.sc_prior_mode} | lambda_mode={args.sc_lambda_mode}(init={args.sc_lambda_init}) | rank={args.sc_prior_rank} | delta_a={args.sc_delta_a} | sinkhorn_iters={args.sc_sinkhorn_iters}"
    )
    criterion = TFMDualLoss(
        lambda_one=args.lambda_one,
        lambda_cpm=args.lambda_cpm,
        lambda_pcc=args.lambda_pcc,
        cpm_gamma=args.cpm_gamma,
        huber_delta=args.cpm_huber_delta,
        lambda_dynamics=args.lambda_dynamics,
        lambda_variance=args.lambda_variance,
        weighting=getattr(args, "loss_weighting", "fixed"),
    ).to(device)
    print(
        f"[TFM] 损失: lambda_one={args.lambda_one} lambda_cpm={args.lambda_cpm} lambda_pcc={args.lambda_pcc} cpm_gamma={args.cpm_gamma} huber_delta={args.cpm_huber_delta} | lambda_dynamics={args.lambda_dynamics} lambda_variance={args.lambda_variance} | cpm_horizon={args.cpm_horizon}（One-Step + CPM 双头，方案 §9/§20）"
    )
    if criterion.weighting == "uncertainty":
        n_log_s = len(list(criterion.log_s.parameters()))
        print(
            f"[TFM] 损失权重: 同方差不确定性加权（Kendall CVPR 2018，Σ exp(-log_s_i)·L_i + log_s_i）| 可学习 log 方差参数 {n_log_s} 个，初始有效权重=对应 --lambda_*（warm start）| 优化器组 loss_weight: lr=lr_peak×{args.loss_lr_scale}, weight_decay=0 | 启用任务项: "
            + ", ".join(
                (
                    k
                    for k, v in criterion.get_loss_weights().items()
                    if isinstance(v, dict)
                )
            )
        )
    optimizer = build_optimizer(model, criterion, args)
    scheduler = build_scheduler(optimizer, args)
    _report_conditioning(model, optimizer, args, dl)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if args.compile and hasattr(torch, "compile"):
        print(
            "[next_timepoint] 变长 context（K 随 batch 变化）会触发 torch.compile 反复重编译（每个 K 一份计算图），已自动跳过编译。"
        )
    dyn_metric_horizon = resolve_dynamics_metric_horizon(args)
    dyn_metric_key = dynamics_metric_key(dyn_metric_horizon)
    if dyn_metric_horizon > int(dims["cpm_horizon"]):
        raise ValueError(
            f"--dynamics_metric_horizon={dyn_metric_horizon} 超出模型的 --cpm_horizon={dims['cpm_horizon']}：多步判据无法覆盖到第 {dyn_metric_horizon} 步，请调小 horizon 或增大预测范围。"
        )
    ckpt_metric_key = dyn_metric_key
    ckpt_metric_name = dynamics_metric_name(dyn_metric_horizon)
    selector = CheckpointSelector(
        ckpt_metric_key, ckpt_metric_name, patience=args.patience
    )
    backbone_unfrozen = args.mode == "pretrain"
    print(
        f"[next_timepoint] checkpoint 选择判据: {ckpt_metric_name}（dynamics_mode={args.dynamics_mode}, metric_key={ckpt_metric_key}）| 单步 MAE 继续记录为观察指标（metric_mae_next）| 最终说明见训练日志/评估报告"
    )
    save_dir = os.path.join(args.checkpoint_dir, args.name)
    if os.path.exists(save_dir):
        import glob
        import re

        p = Path(save_dir)
        dirs = glob.glob(f"{p}*")
        ms = [re.search("%s(\\d+)" % p.stem, d) for d in dirs]
        idx = [int(m.groups()[0]) for m in ms if m]
        save_dir = f"{p}{(max(idx) + 1 if idx else 2)}"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(args.train_epochs):
        if (
            args.mode == "finetune"
            and (not backbone_unfrozen)
            and (epoch >= args.freeze_backbone_epochs)
        ):
            set_finetune_stage(model, backbone_unfrozen=True)
            backbone_unfrozen = True
            added, names = unfreeze_while_preserving_state(model, optimizer, args)
            print(
                f"Epoch {epoch + 1}: 已解冻主干，新增可训练参数 numel={added} groups={names}，可训练参数量 = {count_trainable_params(model)}（沿用现有 optimizer 状态与 scheduler 进度）"
            )
        model.train()
        batch_sampler = getattr(train_data, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        train_meters = defaultdict(list)
        live = defaultdict(float)
        pbar = make_pbar(
            train_data,
            len(train_data),
            f"Epoch {epoch + 1:03d}/{args.train_epochs:03d} [Train]",
        )
        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            history, targets, last_state, target_mask, graph = prepare_next_point_batch(batch, device)
            cond = prepare_condition_batch(batch, device)
            calibration_bold, controls = prepare_dynamics_inputs(batch, device)
            by_roll = batch.get("future", None)
            bm_roll = batch.get("future_mask", None)
            if by_roll is not None:
                by_roll = by_roll.to(device, non_blocking=True)
            if bm_roll is not None:
                bm_roll = bm_roll.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs, aux_info = model(
                    history,
                    graph,
                    condition=cond,
                    condition_epoch=epoch,
                    controls=controls,
                    calibration_bold=calibration_bold,
                )
                cpm_mask_train = bm_roll
                if controls is not None and bm_roll is not None:
                    cpm_mask_train = compute_cpm_mask(controls, bm_roll)
                dyn_kwargs = {}
                if (
                    aux_info.get("dynamics_pred", None) is not None
                    and by_roll is not None
                ):
                    dyn_kwargs = dict(
                        dynamics_pred=aux_info["dynamics_pred"],
                        dynamics_target=by_roll,
                        dynamics_mask=bm_roll,
                        dynamics_mean_norm=aux_info.get("dynamics_mean_norm", None),
                        dynamics_logvar_norm=aux_info.get("dynamics_logvar_norm", None),
                        dynamics_target_norm=_normalized_future(
                            model, by_roll, aux_info.get("norm_stats", None)
                        ),
                    )
                total_loss, loss_stats = criterion(
                    outputs,
                    targets,
                    aux_info["cpm_pred"],
                    by_roll,
                    cpm_mask_train,
                    **dyn_kwargs,
                )
                graph_reg, graph_stats = compute_graph_regularization(
                    aux_info,
                    device,
                    sparsity_weight=args.sc_sparsity_weight,
                    entropy_weight=args.sc_entropy_weight,
                    temporal_weight=args.sc_temporal_weight,
                )
                total_loss = total_loss + graph_reg
            scaler.scale(total_loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad]
                + [p for p in criterion.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            for k, v in {
                **loss_stats,
                **graph_stats,
                "metric_total": total_loss.detach(),
            }.items():
                train_meters[k].append(v.detach())
            live["metric_total"] += float(total_loss.item())
            live["loss_one"] += float(loss_stats["loss_one"].item())
            live["loss_cpm"] += float(loss_stats["loss_cpm"].item())
            postfix = {
                "Loss": f"{live['metric_total'] / step:.4f}",
                "One": f"{live['loss_one'] / step:.4f}",
                "CPM": f"{live['loss_cpm'] / step:.4f}",
            }
            if "loss_dynamics" in loss_stats:
                live["_dyn"] = live.get("_dyn", 0.0) + float(
                    loss_stats["loss_dynamics"].item()
                )
                postfix["Dyn"] = f"{live['_dyn'] / step:.4f}"
            if "loss_variance" in loss_stats:
                live["_var"] = live.get("_var", 0.0) + float(
                    loss_stats["loss_variance"].item()
                )
                postfix["Var"] = f"{live['_var'] / step:.4f}"
            pbar_metrics(pbar, **postfix)
        eval_model = ema.ema if ema is not None else model
        val_metrics = evaluate_tfm(
            eval_model,
            val_data,
            criterion,
            device,
            args,
            rollout_steps=eval_rollout_steps,
            rollout_horizons=eval_rollout_horizons,
        )
        scheduler.step()
        train_metrics = {
            k: torch.stack(vs).mean().item() for k, vs in train_meters.items()
        }
        group_lrs = {
            g.get("name", f"g{i}"): g["lr"]
            for i, g in enumerate(optimizer.param_groups)
        }
        lr_text = " | ".join((f"{k}: {v:.2e}" for k, v in group_lrs.items()))
        tqdm.write(
            f"Epoch {epoch + 1:03d} | {lr_text} | backbone={('unfrozen' if backbone_unfrozen else 'frozen')} | Train Total: {train_metrics['metric_total']:.4f} | one: {train_metrics['loss_one']:.4f} | cpm: {train_metrics['loss_cpm']:.4f} | Val Total: {val_metrics['metric_total']:.4f} | Val next MAE: {val_metrics['metric_mae_next']:.4f} | Val next PCC(sp): {val_metrics['metric_pcc_next']:.4f} | Val CPM MAE: {val_metrics['metric_cpm_mae']:.4f} | Val persistence MAE: {val_metrics['metric_persistence_mae']:.4f} | Val trend MAE: {val_metrics['metric_trend_mae']:.4f}"
        )
        cpm_h_log = [
            f"H{h}: {val_metrics[f'cpm_mae_H{h}']:.4f}"
            for h in range(1, int(args.cpm_horizon) + 1)
            if f"cpm_mae_H{h}" in val_metrics
        ]
        if cpm_h_log:
            tqdm.write("[TFM/CPM] val MAE per horizon | " + " | ".join(cpm_h_log))
        roll_log = [
            f"H{h}: {val_metrics[f'rollout_mae_H{h}']:.4f}"
            for h in eval_rollout_horizons or []
            if f"rollout_mae_H{h}" in val_metrics
        ]
        if roll_log:
            tqdm.write(
                f"[NextPoint] val rollout MAE (free, R={eval_rollout_steps}) | "
                + " | ".join(roll_log)
            )
        is_best, cur_metric = selector.step(val_metrics)
        if dyn_metric_key in val_metrics:
            dyn_text = f"{val_metrics[dyn_metric_key]:.4f}(被试={val_metrics[f'{dyn_metric_key}_n_subjects']:.0f}, 完整{dyn_metric_horizon}步锚点={val_metrics[f'{dyn_metric_key}_n_anchors']:.0f}, 无完整锚点被试={val_metrics[f'{dyn_metric_key}_n_subjects_without_full_anchor']:.0f})"
        else:
            dyn_text = "N/A（本配置无 Graph-ODE dynamics 轨迹）"
        tqdm.write(
            f"[NextPoint] 单步 MAE={val_metrics['metric_mae_next']:.4f} | 多步 MAE={dyn_text} | 选优指标 {ckpt_metric_name}={cur_metric:.4f} | model vs persistence ΔMAE={val_metrics['metric_persistence_mae'] - val_metrics['metric_mae_next']:+.4f}"
        )
        if criterion.weighting == "uncertainty":
            tqdm.write(
                "[LossW] "
                + " | ".join(
                    (
                        f"{k}: w={v['weight']:.4f} (log_s={v['log_s']:+.4f})"
                        for k, v in criterion.get_loss_weights().items()
                        if isinstance(v, dict)
                    )
                )
            )

        def _ckpt_meta():
            meta = {
                "args": vars(args),
                "epoch": int(epoch),
                "epoch_1based": int(epoch + 1),
                "val_metric": float(cur_metric),
                "ckpt_metric_key": selector.metric_key,
                "ckpt_metric_name": selector.metric_name,
                "backbone_unfrozen": bool(backbone_unfrozen),
            }
            meta["condition"] = {
                "config": condition_cfg_dict,
                "stats": dl.get_condition_stats().to_dict()
                if dl.base_dataset.condition_enabled
                else None,
                "meta": dl.get_condition_meta(),
                "split_hash": split_hash(dl, args),
            }
            meta["params"] = param_report(model, optimizer)
            lw = criterion.get_loss_weights()
            meta["loss_weighting"] = {
                "mode": criterion.weighting,
                "log_s": {k: v["log_s"] for k, v in lw.items() if isinstance(v, dict)},
                "weights": {
                    k: v["weight"] for k, v in lw.items() if isinstance(v, dict)
                },
            }
            return meta

        if is_best:
            best_name = (
                "base_best.pt" if args.mode == "pretrain" else "finetuned_best.pt"
            )
            save_backbone_weights(
                os.path.join(save_dir, best_name), eval_model, meta=_ckpt_meta()
            )
            tqdm.write(
                f"New Best Model Saved! (Best {ckpt_metric_name}: {selector.best:.4f})"
            )
        elif selector.should_stop:
            tqdm.write(
                f"\nEarly Stopping! 连续 {selector.patience} epoch 未改善。\n最佳 {ckpt_metric_name}: {selector.best:.4f}"
            )
            last_name = (
                "base_last.pt" if args.mode == "pretrain" else "finetuned_last.pt"
            )
            save_backbone_weights(
                os.path.join(save_dir, last_name),
                ema.ema if ema else model,
                meta=_ckpt_meta(),
            )
            break
        if epoch == args.train_epochs - 1:
            last_name = (
                "base_last.pt" if args.mode == "pretrain" else "finetuned_last.pt"
            )
            save_backbone_weights(
                os.path.join(save_dir, last_name),
                ema.ema if ema else model,
                meta=_ckpt_meta(),
            )
        tb_scalars = [
            ("Train/Total", train_metrics["metric_total"]),
            ("Train/One", train_metrics["loss_one"]),
            ("Train/CPM", train_metrics["loss_cpm"]),
            ("Val/Total", val_metrics["metric_total"]),
            ("Val/Next_MAE", val_metrics["metric_mae_next"]),
            ("Val/Next_RMSE", val_metrics["metric_rmse_next"]),
            ("Val/Next_PCC", val_metrics["metric_pcc_next"]),
            ("Val/CPM_MAE", val_metrics["metric_cpm_mae"]),
            ("Val/Persistence_MAE", val_metrics["metric_persistence_mae"]),
            ("Val/Trend_MAE", val_metrics["metric_trend_mae"]),
        ]
        if dyn_metric_key in val_metrics:
            tb_scalars.append(
                (
                    f"Val/Dynamics_MAE_H1_{dyn_metric_horizon}_Subject",
                    val_metrics[dyn_metric_key],
                )
            )
        for h, key in sorted(
            ((h, f"rollout_mae_H{h}") for h in eval_rollout_horizons or [])
        ):
            if key in val_metrics:
                tb_scalars.append((f"Val/Rollout_MAE_H{h}", val_metrics[key]))
        for tag, val in tb_scalars:
            writer.add_scalar(tag, val, epoch)
        for name, lr in group_lrs.items():
            writer.add_scalar(f"LR/{name}", lr, epoch)
        for tag, key in [
            ("Train/Graph_Sparsity", "graph_sparsity"),
            ("Train/Graph_Entropy", "graph_entropy"),
            ("Train/Graph_Temporal", "graph_temporal"),
        ]:
            if key in train_metrics:
                writer.add_scalar(tag, train_metrics[key], epoch)
        if criterion.weighting == "uncertainty":
            for name, v in criterion.get_loss_weights().items():
                if isinstance(v, dict):
                    writer.add_scalar(f"LossW/{name}", v["weight"], epoch)
    writer.close()
    best_n = "base_best.pt" if args.mode == "pretrain" else "finetuned_best.pt"
    best_path = os.path.join(save_dir, best_n)
    print(f"\n训练完成！最佳权重已保存至: {best_path}")
    if (
        args.mode == "finetune"
        and getattr(args, "eval_after_train", True)
        and os.path.exists(best_path)
    ):
        from experiments.evaluate_variant import evaluate as _eval_variant

        eval_dir = os.path.join(save_dir, "eval")
        try:
            summary = _eval_variant(
                best_path, eval_dir, eval_splits=("test",), seed=args.seed
            )
            print(
                f"[post-train eval] test_PCC={summary.get('test_pcc')} test_MAE={summary.get('test_mae')}"
            )
            print(f"[post-train eval] 详细指标见: {eval_dir}/metrics_test.json")
        except Exception as e:
            print(
                f"[post-train eval] 自动评估失败（训练产物不受影响）: {type(e).__name__}: {e}\n可手动复评: python -m experiments.evaluate_variant --ckpt {best_path} --out_dir {eval_dir}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
