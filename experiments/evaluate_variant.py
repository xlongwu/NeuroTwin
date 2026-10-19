"""统一评估入口：从 checkpoint 重建完整模型 → 评估 → 落盘指标。

唯一任务口径 next_timepoint（连续 BOLD context → 下一 TR 全脑状态），评估协议见
analysis/next_point_eval.py：next-state 主指标 + persistence/trend/AR(1) baselines +
free rollout + FC/频谱。

关键点（模型重建口径）：
  构造 NeuroTwinTFM 若只传部分参数，结构性变体（tfm_patch_len/cpm_horizon/
  sc_prior_mode 等）会按默认结构建模导致权重错配。训练结束时
  save_backbone_weights 已把完整 vars(args) 存进 ckpt['config']['args']
  （train/optim.py），因此这里从该快照 + inspect.signature 自省重建与训练
  完全一致的模型，再 strict 加载。

对外入口 ``evaluate`` 从快照读取 task_mode（仅 next_timepoint）并转发到
``evaluate_next_point``。
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoint import (
    build_model_from_args as build_model_from_args,
)
from analysis.checkpoint import (
    condition_stats_from_bundle as condition_stats_from_bundle,
)
from analysis.checkpoint import (
    load_condition_bundle as load_condition_bundle,
)
from analysis.checkpoint import (
    load_train_args as load_train_args,
)
from analysis.metrics import NETWORK_GROUPS
from utils.common import set_seed
from utils.dataloader import NeuroTwinDataLoader

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _next_point_model_dims(args) -> dict:
    """next_timepoint 的模型维度（与 main.py 的生效值一致）。

    in_window=1（整段 context 作为单窗口）、pred_window=1（one-step 头只建模
    +1 偏移）、in_seq_len=context_max（S 轴建模宽度）、pred_seq_len=1（每个偏移
    1 个时间点）；另附 patch_len / cpm_horizon（结构签名分量）。
    """
    dims = {
        "in_window": 1,
        "pred_window": 1,
        "in_seq_len": int(getattr(args, "context_max")),
        "pred_seq_len": 1,
        "model_arch": "tfm",
        "patch_len": int(args.tfm_patch_len),
        "cpm_horizon": int(args.cpm_horizon),
    }
    return dims


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


def build_eval_loader(args, eval_split, refresh_split_manifest=False):
    """构建 next_timepoint 评估 DataLoader（与训练同一被试级 8:1:1 划分，版本化 manifest 复用）。"""
    from analysis.next_point_eval import eval_horizons, eval_rollout_steps
    from utils.common import (
        parse_int_list,
        parse_pathology_fields,
        resolve_forecast_offsets,
    )

    horizons = eval_horizons(args)
    loader = NeuroTwinDataLoader(
        data_root=args.data_root,
        mode=args.mode,
        batch_size=args.batch_size,
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
        refresh_split_manifest=bool(
            getattr(args, "refresh_split_manifest", False) or refresh_split_manifest
        ),
        task_mode="next_timepoint",
        bold_source=getattr(args, "bold_source", "raw"),
        random_context=getattr(args, "random_context", True),
        random_cutoff=getattr(args, "random_cutoff", True),
        train_samples_per_subject=int(getattr(args, "train_samples_per_subject", 0)),
        subject_cache_size=int(getattr(args, "subject_cache_size", 256)),
        sampling_seed=int(getattr(args, "sampling_seed", 2024)),
        context_min=args.context_min,
        context_max=args.context_max,
        context_lengths=parse_int_list(args.context_lengths, "context_lengths")
        if getattr(args, "context_lengths", None)
        else None,
        forecast_offsets=resolve_forecast_offsets(args),
        eval_context_length=int(getattr(args, "eval_context_length", 0) or 0)
        or int(args.context_max),
        eval_anchors_per_subject=int(getattr(args, "eval_anchors_per_subject", 16)),
        eval_rollout_tasks_per_subject=int(
            getattr(args, "eval_rollout_tasks_per_subject", 2)
        ),
        eval_rollout_steps=eval_rollout_steps(args, horizons),
        train_rollout_steps=0,
        eval_all_future_steps=int(getattr(args, "cpm_horizon", 0) or 0),
        num_rois=int(getattr(args, "num_rois", 116)),
        graph_source=str(getattr(args, "graph_source", "history_fc")),
        calibration_length=int(getattr(args, "calibration_length", 32) or 0),
        control_manifest=getattr(args, "control_manifest", "") or None,
    )
    if eval_split == "test":
        return (loader.get_test(), loader.get_test_subjects())
    return (loader.get_val(), loader.get_val_subjects())


def _roi_network_labels(n_rois):
    """ROI 索引(0 基)→网络标签；未命名脑区归入 Other（NETWORK_GROUPS 为 1 基 AAL 索引）。"""
    labels = ["Other"] * n_rois
    for net, regions in NETWORK_GROUPS.items():
        for r in regions:
            if 1 <= r <= n_rois:
                labels[r - 1] = net
    return labels


def compute_fc_metrics(pred, target, eps=1e-06):
    """FC 指标（1.2/2.2 验收）：样本级 Pearson FC 的上三角 MAE/RMSE、edge-PCC，
    以及命名网络内/网络间边 MAE（含 Other 归组的边不参与网络级统计）。

    pred/target: [N, F, W, S]（或可展平为 [N, F, T] 的形状）。
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n, f = (p.shape[0], p.shape[1])
    p = p.reshape(n, f, -1)
    t = t.reshape(n, f, -1)

    def _fc(x):
        x = x - x.mean(axis=-1, keepdims=True)
        x = x / (x.std(axis=-1, keepdims=True) + eps)
        return np.einsum("nft,ngt->nfg", x, x) / x.shape[-1]

    fc_p, fc_t = (_fc(p), _fc(t))
    iu = np.triu_indices(f, k=1)
    a = fc_p[:, iu[0], iu[1]]
    b = fc_t[:, iu[0], iu[1]]
    diff = a - b
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0)) + eps
    edge_pcc = (a * b).sum(axis=0) / denom
    result = {
        "fc_upper_mae": float(np.abs(diff).mean()),
        "fc_upper_rmse": float(np.sqrt((diff**2).mean())),
        "edge_pcc_mean": float(edge_pcc.mean()),
        "edge_pcc_std": float(edge_pcc.std()),
    }
    labels = _roi_network_labels(f)
    named = [i for i, lb in enumerate(labels) if lb != "Other"]
    if len(named) >= 2:
        edge_idx = {(i, j): k for k, (i, j) in enumerate(zip(iu[0], iu[1]))}
        within, between = ([], [])
        for x in range(len(named)):
            for y in range(x + 1, len(named)):
                k = edge_idx.get((named[x], named[y]))
                if k is None:
                    continue
                (within if labels[named[x]] == labels[named[y]] else between).append(k)
        if within:
            result["fc_within_mae"] = float(np.abs(diff[:, within]).mean())
            result["n_edges_within"] = len(within)
        if between:
            result["fc_between_mae"] = float(np.abs(diff[:, between]).mean())
            result["n_edges_between"] = len(between)
    return result


def evaluate_next_point(
    ckpt_path,
    out_dir,
    args,
    eval_splits=("test",),
    seed=2024,
    device=None,
    save_arrays=False,
    eval_rollout=None,
    eval_fc=None,
    eval_spectral=None,
    spectral_tr=None,
    refresh_split_manifest=False,
    compute_importance=False,
    importance_max_tasks=256,
    importance_permutations=0,
    importance_fdr_alpha=0.05,
):
    """Next-Timepoint（Next Brain-State Prediction）评估入口。

    主指标是**单时间点全脑状态**的 MAE/RMSE/spatial PCC，baselines 为
    persistence/trend/AR(1)（AR(1) 只用 train subjects 拟合），free rollout 的
    horizon 列表、FC（fc_min_length）与频谱见 analysis/next_point_eval.py 的模块
    docstring。返回 registry 兼容的 summary 键（test_pcc/test_mae/...）。
    """
    from analysis import next_point_eval as npe
    from analysis.checkpoint import load_checkpoint_compat

    set_seed(seed)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rollout = (
        getattr(args, "eval_rollout", True) if eval_rollout is None else eval_rollout
    )
    eval_fc = getattr(args, "eval_fc", True) if eval_fc is None else eval_fc
    eval_spectral = (
        getattr(args, "eval_spectral", False)
        if eval_spectral is None
        else eval_spectral
    )
    spectral_tr = (
        getattr(args, "eval_spectral_tr", 2.0) if spectral_tr is None else spectral_tr
    )
    condition_bundle = load_condition_bundle(ckpt_path)
    model = build_model_from_args(args, device, condition_bundle=condition_bundle)
    model, compat = load_checkpoint_compat(model, ckpt_path, device, strict=True)
    if compat.get("missing") or compat.get("unexpected"):
        raise RuntimeError(f"权重加载不完整：{compat}")
    n_params = int(sum((p.numel() for p in model.parameters())))
    if getattr(model, "conditioner", None) is not None:
        saved_hash = (condition_bundle or {}).get("meta", {}).get("layout_hash")
        now_hash = model.conditioner.layout()["layout_hash"]
        if saved_hash and saved_hash != now_hash:
            raise RuntimeError(
                f"条件布局不一致：checkpoint={saved_hash}，重建后={now_hash}。字段顺序/来源列/类别词表任一变化都禁止复用旧权重。"
            )
        log.info(
            "[condition] strict 重建完成 | arm=%s layout_hash=%s",
            model.conditioner.cfg.arm,
            now_hash,
        )
    extra = {
        "conformal_calibration": {
            "available": False,
            "reason": "next_timepoint 协议未接入 conformal 校准（P3）",
        }
    }
    summary = {"n_params": n_params}
    for split in eval_splits:
        data_loader, subjects, loader = npe.build_next_point_loader(
            args,
            split,
            refresh_split_manifest=refresh_split_manifest,
            condition_bundle=condition_bundle,
        )
        if condition_bundle is not None and loader.base_dataset.condition_enabled:
            saved_src = (condition_bundle.get("meta") or {}).get("resolved_source_map")
            now_src = loader.base_dataset.condition_table.audit.alias_sources
            if saved_src and saved_src != now_src:
                raise RuntimeError(
                    f"条件来源列不一致：checkpoint={saved_src}，当前解析={now_src}。"
                )
            saved_sha = (condition_bundle.get("meta") or {}).get("merged_sha256")
            now_sha = loader.base_dataset.condition_table.merged_sha256
            if saved_sha and saved_sha != now_sha:
                log.warning(
                    "[condition] merged 文件内容已变化（训练 %s… → 当前 %s…）；条件值可能与训练期不同，结果需按当前表重新解读",
                    str(saved_sha)[:12],
                    str(now_sha)[:12],
                )
        log.info(
            "[next_timepoint] split=%s | 被试数=%d | 任务数=%d",
            split,
            len(subjects),
            len(data_loader.dataset),
        )
        ar1 = npe.fit_ar1_from_train(loader.base_dataset, loader.get_train_subjects())
        log.info(
            "[next_timepoint] AR(1) 拟合完成 | train pairs=%d | a∈[%.3f, %.3f]",
            ar1["n_pairs"],
            float(ar1["a"].min()),
            float(ar1["a"].max()),
        )
        report = npe.evaluate_split(
            model,
            data_loader,
            args,
            device,
            out_dir,
            split,
            eval_rollout=eval_rollout,
            eval_fc=eval_fc,
            eval_spectral=eval_spectral,
            spectral_tr=spectral_tr,
            save_arrays=save_arrays,
            n_params=n_params,
            extra_report=extra,
            ar1=ar1,
            compute_importance=compute_importance,
            importance_max_tasks=importance_max_tasks,
            importance_permutations=importance_permutations,
            importance_fdr_alpha=importance_fdr_alpha,
        )
        summary.update(npe.summarize(report))
        if split == "test":
            first = f"t+{report['protocol']['forecast_offsets'][0]}"
            summary["test_subj_pcc"] = report.get("subject_level", {}).get(
                "PCC_spatial_mean"
            )
            summary["test_n_params"] = n_params
            summary["test_pcc_H1"] = (
                report.get("next_state", {})
                .get("model", {})
                .get(first, {})
                .get("PCC_spatial")
            )
    return summary


def evaluate(
    ckpt_path,
    out_dir,
    eval_splits=("test",),
    seed=2024,
    device=None,
    visualize=False,
    compute_feature_importance=False,
    save_arrays=False,
    refresh_split_manifest=False,
    eval_rollout=None,
    eval_fc=None,
    eval_spectral=None,
    spectral_tr=None,
    importance_max_tasks=256,
    importance_permutations=0,
    importance_fdr_alpha=0.05,
):
    """评估单个 checkpoint 的对外入口（唯一口径 next_timepoint）。

    从 checkpoint 的 config.args 快照自省重建模型，转发到 evaluate_next_point。
    ``compute_feature_importance=True`` 时额外产 ``feature_importance_<split>.json``
    （ROI 置换重要性 + BH-FDR 显著脑区）；``visualize`` 已随旧版可视化组件移除，
    传入 True 时显式报错。
    """
    _args = load_train_args(ckpt_path)
    task_mode = str(getattr(_args, "task_mode", "next_timepoint"))
    if task_mode != "next_timepoint":
        raise ValueError(
            f"仅支持 task_mode=next_timepoint 的 checkpoint，收到 '{task_mode}'"
        )
    if visualize:
        raise ValueError(
            "--visualize 已随旧可视化组件移除；请用 analysis/visualize_roi_importance.py 与 analysis/visualize_sig_region.py。"
        )
    return evaluate_next_point(
        ckpt_path,
        out_dir,
        _args,
        eval_splits=eval_splits,
        seed=seed,
        device=device,
        save_arrays=save_arrays,
        eval_rollout=eval_rollout,
        eval_fc=eval_fc,
        eval_spectral=eval_spectral,
        spectral_tr=spectral_tr,
        refresh_split_manifest=refresh_split_manifest,
        compute_importance=compute_feature_importance,
        importance_max_tasks=importance_max_tasks,
        importance_permutations=importance_permutations,
        importance_fdr_alpha=importance_fdr_alpha,
    )


def main():
    ap = argparse.ArgumentParser(description="单 checkpoint 评估（next_timepoint）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--splits",
        default="test",
        help="评估划分（逗号分隔）；默认只评 test，需要 val 时显式传入",
    )
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument(
        "--device",
        default="",
        help="评估设备（如 cuda:1）。留空则自动选择可用 GPU/CPU。多卡并行搜索时由运行器显式传入，不依赖默认值。",
    )
    ap.add_argument(
        "--save_arrays",
        action="store_true",
        help="保存预测数组 predictions_<split>.npz",
    )
    ap.add_argument(
        "--refresh_split_manifest",
        action="store_true",
        help="重新生成 subject_split_finetune.json 切分 manifest（默认复用）",
    )
    ap.add_argument(
        "--rollout",
        dest="rollout",
        action="store_true",
        default=None,
        help="强制开启 free rollout",
    )
    ap.add_argument(
        "--no_rollout", dest="rollout", action="store_false", help="跳过 free rollout"
    )
    ap.add_argument(
        "--no_fc",
        dest="fc",
        action="store_false",
        default=None,
        help="跳过 FC 层面指标",
    )
    ap.add_argument("--fc", dest="fc", action="store_true", help="强制开启 FC 层面指标")
    ap.add_argument(
        "--spectral",
        dest="spectral",
        action="store_true",
        default=None,
        help="额外输出 Welch PSD 低频段一致性指标",
    )
    ap.add_argument(
        "--spectral_tr",
        type=float,
        default=None,
        help="频谱评估使用的 TR（秒）；默认沿用 checkpoint 配置",
    )
    ap.add_argument(
        "--feature_importance",
        action="store_true",
        help="额外计算 ROI 置换重要性并落盘 feature_importance_<split>.json",
    )
    ap.add_argument(
        "--feature_importance_max_tasks",
        type=int,
        default=256,
        help="重要性统计使用的任务数上限（0 = 全部；默认 256 以控制代价）",
    )
    ap.add_argument(
        "--feature_importance_permutations",
        type=int,
        default=0,
        help="每个 ROI 的置换检验次数；0 = 单次置换 + t 检验近似（快），>0 时用经验零分布（代价随该值线性增长）",
    )
    ap.add_argument(
        "--feature_importance_fdr_alpha",
        type=float,
        default=0.05,
        help="BH-FDR 显著脑区阈值（默认 0.05）",
    )
    ns = ap.parse_args()
    device = torch.device(ns.device) if ns.device else None
    summary = evaluate(
        ns.ckpt,
        ns.out_dir,
        tuple(ns.splits.split(",")),
        seed=ns.seed,
        device=device,
        save_arrays=ns.save_arrays,
        refresh_split_manifest=ns.refresh_split_manifest,
        eval_rollout=ns.rollout,
        eval_fc=ns.fc,
        eval_spectral=ns.spectral,
        spectral_tr=ns.spectral_tr,
        compute_feature_importance=ns.feature_importance,
        importance_max_tasks=ns.feature_importance_max_tasks,
        importance_permutations=ns.feature_importance_permutations,
        importance_fdr_alpha=ns.feature_importance_fdr_alpha,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
