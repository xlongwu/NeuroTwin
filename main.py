# coding=utf-8
"""NeuroTwin 训练入口。

训练基础设施（损失/优化器/EMA/MoE 正则）位于 train/ 包，
通用工具（set_seed/str2bool）位于 utils/common.py；
本文件保留 evaluate、main 训练循环与 CLI 定义。
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.log_dir = kwargs.get('log_dir', None)
        def add_scalar(self, *args, **kwargs): return None
        def close(self): return None

from utils.dataloader import NeuroTwinDataLoader
from utils.common import (parse_int_list, parse_pred_quantiles,
                          resolve_forecast_offsets, resolve_mtp_weights,
                          resolve_task_dims, set_seed, str2bool)
from models.neurotwin import NeuroTwin
from train.losses import (NextTimepointLoss, compute_inversion_loss,
                          compute_rollout_loss, spatial_pcc)
from train.optim import (ModelEMA, build_optimizer, build_scheduler,
                         check_pretrain_config_compat, count_trainable_params,
                         load_backbone_only, load_backbone_weights,
                         save_backbone_weights, set_finetune_stage)
from train.moe import (compute_expert_diversity, compute_graph_regularization,
                       compute_moe_regularization, update_router_temperature)
from models.decoder import AmplitudeConsistencyLoss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# ──────────────────────────────────────────────────────────────────────────
#  next_timepoint（Next Brain-State Prediction）验证循环
# ──────────────────────────────────────────────────────────────────────────

def next_point_batch(batch, device):
    """取出 next_timepoint batch 的字段。

    y 由 [B,F,H] 升维为 [B,F,H,1]（与模型输出 [B,F,H,1] 对齐）；x_last 为
    context 末位 x_t，target_mask [B,H] 标记各预测偏移是否有真值（MTP 用）。
    """
    bx = batch['x'].to(device, non_blocking=True)                # [B,F,1,K]
    by = batch['y'].to(device, non_blocking=True).unsqueeze(-1)  # [B,F,H,1]
    bl = batch['x_last'].to(device, non_blocking=True)           # [B,F]
    bm = batch.get('target_mask', None)
    if bm is not None:
        bm = bm.to(device, non_blocking=True)                    # [B,H]
    bs = batch['sc'].to(device, non_blocking=True)
    bp = batch.get('pathology_score', None)
    if bp is not None:
        bp = bp.to(device, non_blocking=True)
    return bx, by, bl, bm, bs, bp


def _masked_l1_weighted(pred, target, weight, eps=1e-8):
    """按 [B,H] 权重（广播到 [B,F,H,1]）加权的逐元素 L1。"""
    w = weight.to(pred.dtype).view(weight.shape[0], 1, weight.shape[1], 1)
    w = w.expand_as(pred)
    return (torch.abs(pred - target) * w).sum() / w.sum().clamp_min(eps)


@torch.no_grad()
def evaluate_next_point(model, data_loader, criterion, device, args,
                        offsets, mtp_weights, rollout_steps=0, rollout_horizons=None):
    """next_timepoint 的验证/测试循环。

    返回与训练损失同口径的 ``loss_*`` 指标，外加 next-state 主指标
    （MAE/RMSE/spatial PCC）与两条 trivial baseline（persistence / trend），
    供 checkpoint 选择（主判据 = ``metric_mae_next``）与日志/TensorBoard 使用。

    ``rollout_steps>0`` 时额外做 free autoregressive rollout（自回归滚动、中间不使用
    真值），按 ``rollout_horizons`` 输出 ``rollout_mae_H{h}``（prompt §四十）。
    """
    model.eval()
    meters = defaultdict(list)
    rollout_acc = defaultdict(lambda: [0.0, 0])        # horizon -> [Σ|err|, n_tasks]
    rollout_hs = [int(h) for h in (rollout_horizons or []) if int(h) <= int(rollout_steps)]
    pbar = tqdm(data_loader, total=len(data_loader), dynamic_ncols=True, leave=False)
    pbar.set_description('[Val-NextPoint]')


    for batch in pbar:
        bx, by, bl, bm, bs, bp = next_point_batch(batch, device)
        outputs, aux_info = model(bx, bs, bp)
        total_loss, loss_stats = criterion(
            outputs, by, bl, aux_info=aux_info, mask=bm,
            mtp_weights=mtp_weights, offsets=offsets)
        moe_reg, moe_stats = compute_moe_regularization(
            aux_info, device,
            load_balance_weight=args.moe_load_balance_weight,
            entropy_weight=args.moe_entropy_weight,
            z_loss_weight=args.moe_z_loss_weight,
            diversity_weight=args.moe_diversity_weight,
        )
        graph_reg, graph_stats = compute_graph_regularization(
            aux_info, device,
            sparsity_weight=args.sc_sparsity_weight,
            entropy_weight=args.sc_entropy_weight,
            temporal_weight=args.sc_temporal_weight,
        )
        total_loss = total_loss + moe_reg + graph_reg

        b, f, h, _ = outputs.shape
        if bm is None:
            bm = torch.ones(b, h, device=outputs.device, dtype=outputs.dtype)
        # next-state 主指标（与损失同一权重口径）
        w4 = bm.view(b, 1, h, 1)
        mae = _masked_l1_weighted(outputs, by, bm)
        rmse = (((outputs - by) ** 2) * w4.expand_as(outputs)).sum() / \
            (w4.expand_as(outputs).sum().clamp_min(1e-8))
        rmse = rmse.clamp_min(0).sqrt()
        pcc_spatial = ((spatial_pcc(outputs, by) * bm).sum()
                       / bm.sum().clamp_min(1e-8))
        # trivial baseline：persistence（x̂=x_t）与 linear trend（x̂=x_t+(x_t−x_{t−1})）
        x_t = bx[:, :, 0, -1:].unsqueeze(-1)                     # [B,F,1,1]
        x_prev = bx[:, :, 0, -2:-1].unsqueeze(-1)
        pers = x_t.expand(b, f, h, 1)
        trend = (x_t + (x_t - x_prev)).expand(b, f, h, 1)
        pers_mae = _masked_l1_weighted(pers, by, bm)
        trend_mae = _masked_l1_weighted(trend, by, bm)

        for k, v in {**loss_stats, **moe_stats, **graph_stats,
                     'metric_total': total_loss.detach(),
                     'metric_mae_next': mae.detach(),
                     'metric_rmse_next': rmse.detach(),
                     'metric_pcc_next': pcc_spatial.detach(),
                     'metric_persistence_mae': pers_mae.detach(),
                     'metric_trend_mae': trend_mae.detach()}.items():
            meters[k].append(v.detach())
        pbar.set_postfix({'MAE': f"{float(mae):.4f}",
                          'PCC_sp': f"{float(pcc_spatial):.4f}",
                          'pers': f"{float(pers_mae):.4f}"})

        # ---- free rollout（仅评估视图的 rollout 锚点；中间不使用真值）----
        if rollout_hs and 'rollout_flag' in batch and 'future' in batch:
            # rf 留在 CPU（batch['future'] 等仍是 CPU 张量）；GPU 张量用 rf_gpu 索引
            rf = batch['rollout_flag'].bool()
            rf_gpu = rf.to(device)
            if rf_gpu.any():
                hist = bx[rf_gpu][:, :, 0, :].transpose(1, 2).contiguous()   # [n,K,F]
                roll = model.rollout_next_states(
                    hist, bs[rf_gpu], (bp[rf_gpu] if bp is not None else None),
                    steps=int(rollout_steps))                            # [n,R,F]
                fut = batch['future'][rf].transpose(1, 2).contiguous().to(
                    device).float()                                      # [n,R,F]
                fmask = batch['future_mask'][rf].to(device).float()      # [n,R]
                for h in rollout_hs:
                    ok = fmask[:, h - 1] > 0
                    k = int(ok.sum())
                    if k:
                        rollout_acc[h][0] += float(
                            (roll[ok, h - 1, :] - fut[ok, h - 1, :]).abs().sum())
                        rollout_acc[h][1] += k

    for h, (err_sum, n_tasks) in rollout_acc.items():
        if n_tasks:
            meters[f'rollout_mae_H{h}'].append(torch.tensor(err_sum / n_tasks))
    return {k: torch.stack(vs).mean().item() for k, vs in meters.items()}


def parse_pathology_fields(fields):
    """把逗号分隔的字段字符串解析为列表；空/None 返回 None（由 Dataset 使用默认）。"""
    if fields is None:
        return None
    if isinstance(fields, (list, tuple)):
        parsed = [str(f).strip() for f in fields if str(f).strip()]
        return parsed or None
    parsed = [f.strip() for f in str(fields).split(',') if f.strip()]
    return parsed or None


def fit_pathology_normalizer(model, dl, args):
    """用训练集病理评分拟合模型内归一化模块的统计量。

    仅在 finetune 模式且模型持有 pathology_normalizer 时生效。
    当 --reuse_pretrained_norm_stats 为真且模块已带拟合标记时，跳过重新拟合。
    """
    norm = getattr(model, 'pathology_normalizer', None)
    if norm is None:
        return
    if getattr(args, 'reuse_pretrained_norm_stats', False) and norm.is_fitted():
        print(f'[HAMD norm] 复用已加载的归一化统计量 (mode={norm.mode})')
        return
    scores = dl.get_train_pathology_scores()
    if scores is None or len(scores) == 0:
        raise RuntimeError('无法获取训练集病理（HAMD）评分，pathology normalizer 拟合失败。')
    norm.fit(scores)
    print(f'[HAMD norm] mode={norm.mode} | 拟合样本数={len(scores)} | '
          f'center={norm.center.tolist()} | scale={norm.scale.tolist()} | '
          f'out_dim={norm.out_dim}')


def main(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    amp_enabled = (device.type == 'cuda' and args.amp)
    torch.set_num_threads(args.num_threads)
    if device.type == 'cuda' and args.tf32:
        # 输入形状固定，启用 cudnn 自动调优；TF32 加速 autocast 覆盖不到的 fp32 matmul
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    # 任务维度解析：next_timepoint（in_window=1, pred_window=len(forecast_offsets),
    # seq_len=context_max）。模型/数据/损失共用同一口径。
    dims = resolve_task_dims(args)
    forecast_offsets = resolve_forecast_offsets(args)
    mtp_weights = resolve_mtp_weights(args, forecast_offsets)
    eval_rollout_horizons = (parse_int_list(args.eval_rollout_horizons,
                                            'eval_rollout_horizons')
                             or [1, 2, 4, 8, 16])
    needs_long_rollout = bool(args.eval_fc or args.eval_spectral)
    auto_steps = max(eval_rollout_horizons)
    if needs_long_rollout:
        auto_steps = max(auto_steps, int(args.fc_min_length))
    eval_rollout_steps = int(args.eval_rollout_steps) or auto_steps

    print('=' * 50)
    if args.mode == 'pretrain':
        print(f'[Pretrain] 正在启动预训练 | task_mode={dims["task_mode"]} | '
              f'预测偏移数: {dims["pred_window"]}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 1] Healthy Control (HC) physics backbone pretraining')
    else:
        print(f'[Finetune] Starting finetuning | task_mode={dims["task_mode"]} | '
              f'预测偏移数: {dims["pred_window"]}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 2] MDD individualized pathology expert finetuning [NeuroTwinMoE]')
    print('=' * 50)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.checkpoint_dir, 'runs', f"{args.name}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志已开启，保存路径: {log_dir}")

    print(f"[next_timepoint] 模型维度映射: in_window←1（整段 context 作为单窗口） | "
          f"pred_window←len(forecast_offsets)={dims['pred_window']} | "
          f"seq_len←context_max={dims['seq_len']} | "
          f"forecast_offsets={forecast_offsets} | mtp_weights={mtp_weights} | "
          f"prediction_target={args.prediction_target} | "
          f"causal_training={args.causal_training}")
    print(f"[next_timepoint] 训练 context: K∈[--context_min, --context_max]="
          f"[{args.context_min}, {args.context_max}]"
          + (f"（离散 {parse_int_list(args.context_lengths, 'context_lengths')}）"
             if args.context_lengths else "（连续区间随机采样）")
          + f" | 评估固定 K_eval={int(args.eval_context_length) or args.context_max} | "
          f"anchors/被试={args.eval_anchors_per_subject} | "
          f"rollout: R={eval_rollout_steps}, horizons={eval_rollout_horizons}, "
          f"锚点/被试={args.eval_rollout_tasks_per_subject}")
    if args.enable_rollout_loss:
        print(f"[next_timepoint] rollout 训练损失已启用: steps={args.rollout_train_steps} "
              f"lambda={args.lambda_rollout}（每步额外前向，注意显存与稳定性）")

    dl = NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode, batch_size=args.batch_size,
        pathology_input_dim=args.pathology_input_dim,
        pathology_fields=parse_pathology_fields(args.pathology_fields),
        pathology_missing=args.pathology_missing,
        clinical_file=args.clinical_file,
        total_windows=args.total_windows, seq_len=args.seq_len,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        eval_batch_size=args.eval_batch_size,
        refresh_split_manifest=getattr(args, 'refresh_split_manifest', False),
        task_mode=dims['task_mode'],
        bold_source=args.bold_source,
        random_context=args.random_context, random_cutoff=args.random_cutoff,
        train_samples_per_subject=args.train_samples_per_subject,
        subject_cache_size=args.subject_cache_size,
        sampling_seed=args.sampling_seed,
        # next_timepoint 的数据协议
        context_min=args.context_min, context_max=args.context_max,
        context_lengths=parse_int_list(args.context_lengths, 'context_lengths'),
        forecast_offsets=forecast_offsets,
        eval_context_length=(int(args.eval_context_length) or args.context_max),
        eval_anchors_per_subject=args.eval_anchors_per_subject,
        eval_rollout_tasks_per_subject=args.eval_rollout_tasks_per_subject,
        eval_rollout_steps=eval_rollout_steps,
        train_rollout_steps=(int(args.rollout_train_steps)
                             if args.enable_rollout_loss else 0),
    )
    train_data, val_data = dl.get_train(), dl.get_val()

    # next_timepoint：预测头以 x_t 为锚点（prediction_target=delta，默认）或零锚点
    # （absolute）；因整段 context 作为单窗口（W=1），GraphODE 的跨窗注意力退化为
    # 逐 token 线性映射（且要求 S 可被 window_heads 整除），这里统一关闭并提示。
    head_anchor_mode = 'history_window'
    ode_window_attn = args.ode_window_attn
    head_scale_granularity = args.head_scale_granularity
    head_anchor_mode = ('last_timestep' if args.prediction_target == 'delta'
                        else 'zero')
    if ode_window_attn != 'off':
        print("[next_timepoint] W=1 → GraphODE 跨窗注意力退化为逐 token 线性映射，"
              "自动关闭 --ode_window_attn（等价消融，同时解除 S %% window_heads 约束）")
        ode_window_attn = 'off'
    if head_scale_granularity != 'timestep':
        print("[next_timepoint] 单时间点预测下 window 粒度幅值统计恒为 0，"
              "自动改用 --head_scale_granularity timestep（逐元素幅值代理量）")
        head_scale_granularity = 'timestep'
    # 把「生效值」写回 args：checkpoint 快照（meta={'args': vars(args)}）与
    # experiments/evaluate_variant 的模型重建都依赖它，否则会按 CLI 原始值
    # 重建出结构不一致的模型（ode_window_attn / 头部锚点 / 幅值粒度）
    args.head_anchor_mode = head_anchor_mode
    args.ode_window_attn = ode_window_attn
    args.head_scale_granularity = head_scale_granularity

    model = NeuroTwin(
        features=args.num_rois, in_window=dims['in_window'], in_seq_len=dims['seq_len'],
        # next_timepoint 每个预测偏移只输出 1 个时间点（pred_seq_len=1）
        pred_window=dims['pred_window'],
        pred_seq_len=1,
        n_block=args.n_block, dropout=args.dropout,
        pathology_input_dim=args.pathology_input_dim, pathology_dim=args.pathology_dim,
        adapter_alpha=args.alpha, norm=args.norm,
        pretrain_mode=(args.mode == 'pretrain'),
        ode_steps=args.ode_steps, ode_hidden_dim=args.ode_hidden_dim,
        num_scales=args.num_scales, num_experts=args.num_experts, top_k=args.top_k,
        stochastic_depth_rate=args.stochastic_depth_rate,
        moe_gate_temperature=args.moe_gate_temp_start,
        moe_expert_hidden_dim=args.moe_expert_hidden_dim,
        moe_use_shared_expert=args.moe_use_shared_expert,
        moe_router_cond_only=args.moe_router_cond_only,
        moe_use_argmax=args.moe_use_argmax,
        moe_inference_temperature=args.moe_inference_temperature,
        moe_eval_mode=args.moe_eval_mode,
        moe_gate_features=args.moe_gate_features,
        moe_gate_input_dim=args.moe_gate_input_dim,
        moe_experts_mode=args.moe_experts_mode,
        moe_expert_kind=args.moe_expert_kind,
        moe_route_level=args.moe_route_level,
        moe_eval_mc_samples=args.moe_eval_mc_samples,
        moe_expert_stats_interval=args.moe_expert_stats_interval,
        pathology_norm_mode=args.pathology_norm_mode,
        pathology_norm_quantiles=args.pathology_norm_quantiles,
        pathology_norm_rbf_knots=args.pathology_norm_rbf_knots,
        pathology_poly_expansion=args.pathology_poly_expansion,
        refiner_rounds=args.refiner_rounds,
        delta_refiner_rounds=args.delta_refiner_rounds,
        refiner_adaptive=args.refiner_adaptive,
        refiner_return_rounds=(args.refiner_inter_sup_weight > 0),
        patho_cond_layer=args.patho_cond_layer,
        patho_adaln_targets=args.patho_adaln_targets,
        lora_enable=args.lora_enable,
        lora_rank=args.lora_rank,
        lora_n_blocks=args.lora_n_blocks,
        sc_prior_mode=args.sc_prior_mode,
        sc_lambda_mode=args.sc_lambda_mode,
        sc_lambda_init=args.sc_lambda_init,
        sc_prior_rank=args.sc_prior_rank,
        sc_delta_a=args.sc_delta_a,
        sc_delta_rank=args.sc_delta_rank,
        sc_prob_mask=args.sc_prob_mask,
        sc_sinkhorn_iters=args.sc_sinkhorn_iters,
        sc_mask_mode=args.sc_mask_mode,
        sc_mask_tau=args.sc_mask_tau,
        sc_refiner_inject=args.sc_refiner_inject,
        head_cross_roi=args.head_cross_roi,
        sc_temporal_weight=args.sc_temporal_weight,
        mdm_scale_scheme=args.mdm_scale_scheme,
        mdm_scale_gate=args.mdm_scale_gate,
        ode_window_attn=ode_window_attn,
        ode_solver=args.ode_solver,
        ode_step_mode=args.ode_step_mode,
        ode_step_scale=args.ode_step_scale,
        sde_noise_scale=args.sde_noise_scale,
        ode_adaptive_rtol=args.ode_adaptive_rtol,
        ode_adaptive_atol=args.ode_adaptive_atol,
        head_shape_mode=args.head_shape_mode,
        future_query_mode=args.future_query_mode,
        future_query_dim=args.future_query_dim,
        future_query_layers=args.future_query_layers,
        future_query_heads=args.future_query_heads,
        head_amp_mode=args.head_amp_mode,
        head_scale_granularity=head_scale_granularity,
        head_anchor_mode=head_anchor_mode,
        head_use_history_proj=args.head_use_history_proj,
        head_use_latent_proj=args.head_use_latent_proj,
        head_use_temporal=args.head_use_temporal,
        head_use_cross_roi=args.head_use_cross_roi,
        head_use_win_attn=args.head_use_win_attn,
        head_use_revin_stats=args.head_use_revin_stats,
        # Phase 7：概率输出头 + 辅助反演头（inversion_weight=0 时不构建）
        pred_head=args.pred_head,
        pred_quantiles=parse_pred_quantiles(args.pred_quantiles),
        inversion_weight=args.inversion_weight,
        inversion_hidden_dim=args.inversion_hidden_dim,
    ).to(device)

    if args.mode == 'finetune':
        if not os.path.exists(args.pretrained_weight):
            raise FileNotFoundError(f"未找到预训练权重: {args.pretrained_weight}")
        if args.load_backbone_only:
            # 显式模式：只加载主干、头部/条件模块重新初始化（prompt §三十六）。
            # 任务口径闸门降级为 warn：口径差异只提示，主干按 shape 逐键加载并汇报。
            check_pretrain_config_compat(args.pretrained_weight, dims,
                                         arch_policy='warn')
            load_backbone_only(model, args.pretrained_weight, device)
        else:
            # 任务口径闸门：不同任务的目标/输入 shape 语义不同，
            # 不允许静默按 shape 过滤加载（形状对不上时会静默跳过大量键）
            check_pretrain_config_compat(args.pretrained_weight, dims,
                                         arch_policy=args.pretrained_arch_policy)
            load_backbone_weights(model, args.pretrained_weight, device,
                                  arch_policy=args.pretrained_arch_policy,
                                  skip_pattern=getattr(args, 'pretrained_skip_pattern', ''))
        set_finetune_stage(model, backbone_unfrozen=False)
        print('已进入分阶段微调: 前期仅训练 MoE 与条件模块（MoDE），后期逐步解冻主干。')

    if args.mode == 'finetune':
        fit_pathology_normalizer(model, dl, args)

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Parameters: {count_trainable_params(model)}")
    print(f"[ForecastHead] shape_mode={args.head_shape_mode} "
          f"(query_mode={args.future_query_mode}, dim={args.future_query_dim}, "
          f"layers={args.future_query_layers}, heads={args.future_query_heads}) | "
          f"amp_mode={args.head_amp_mode}({args.head_scale_granularity}, "
          f"w={args.head_amp_consistency_weight}) | "
          f"fuse branches: hist={args.head_use_history_proj} lat={args.head_use_latent_proj} "
          f"temp={args.head_use_temporal} xroi={args.head_use_cross_roi}({args.head_cross_roi}) "
          f"win={args.head_use_win_attn} revin={args.head_use_revin_stats} | "
          f"ode_win_attn={args.ode_window_attn} | "
          f"mdm={args.mdm_scale_scheme}/{args.mdm_scale_gate}")
    print(f"[ODE] solver={args.ode_solver} | step_mode={args.ode_step_mode} "
          f"| step_scale={args.ode_step_scale} | sde_noise={args.sde_noise_scale} "
          f"| adaptive_rtol={args.ode_adaptive_rtol} "
          f"| adaptive_atol={args.ode_adaptive_atol}")
    print(f"[SC prior] mode={args.sc_prior_mode} | mask_mode={args.sc_mask_mode} "
          f"| lambda_mode={args.sc_lambda_mode}(init={args.sc_lambda_init}) "
          f"| rank={args.sc_prior_rank} | delta_a={args.sc_delta_a} "
          f"| prob_mask={args.sc_prob_mask} | refiner_inject={args.sc_refiner_inject} "
          f"| head_cross_roi={args.head_cross_roi}")
    print(f"[Phase7] pred_head={args.pred_head}"
          + (f"(quantiles={parse_pred_quantiles(args.pred_quantiles)})"
             if args.pred_head == 'quantile' else "")
          + (f" | init_log_var_nll={args.init_log_var_nll}"
             if args.pred_head != 'point' else "")
          + f" | inversion_weight={args.inversion_weight}"
          + (f"(hidden={args.inversion_hidden_dim})" if args.inversion_weight > 0 else "")
          + f" | intervention_mode={args.intervention_mode}")

    # Next-Timepoint 主损失：absolute + delta + spatial PCC（prompt §十四/十五/三十二）
    criterion = NextTimepointLoss(
        lambda_abs=args.lambda_abs, lambda_delta=args.lambda_delta,
        lambda_pcc=args.lambda_pcc, lambda_nll=args.lambda_nll,
    ).to(device)
    print(f"[next_timepoint] 损失: lambda_abs={args.lambda_abs} "
          f"lambda_delta={args.lambda_delta} lambda_pcc={args.lambda_pcc} "
          f"lambda_nll={args.lambda_nll}（NLL 默认关闭：单点预测的 logvar "
          f"换算在近常数 context 上不稳定）")
    # 轮间（deep supervision）监督在 next_timepoint 下单时间点目标的 PCC 项无定义，
    # 该辅助损失不参与训练（其余损失不变）。
    if args.refiner_inter_sup_weight > 0:
        print("[next_timepoint] 轮间监督的 PCC 项在单时间点目标上无定义，"
              "已跳过 refiner_inter_sup（其余损失不变）。")
    # 幅值一致性损失：仅在 scale_mod_trend 且权重>0 时构建（否则该头无显式幅值参数）
    amp_loss_fn = (AmplitudeConsistencyLoss(
        granularity=head_scale_granularity).to(device)
        if (args.head_amp_consistency_weight > 0
            and args.head_amp_mode == 'scale_mod_trend') else None)

    optimizer = build_optimizer(model, criterion, args)
    scheduler = build_scheduler(optimizer, args)
    scaler    = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    ema       = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if args.compile and hasattr(torch, 'compile'):
        # 变长 context（K 逐 batch 变化）会让 dynamo 按 shape 反复重编译，
        # 因此该任务跳过编译；固定 --context_lengths 时可手动评估收益。
        print("[next_timepoint] 变长 context（K 随 batch 变化）会触发 torch.compile "
              "反复重编译（每个 K 一份计算图），已自动跳过编译。")

    # 主 checkpoint 判据 = 验证集 next-state MAE（prompt §三十八：不使用单纯 PCC 选模型）
    ckpt_metric_key  = 'metric_mae_next'
    ckpt_metric_name = 'val next-state MAE'
    best_ckpt_value  = float('inf')
    patience_counter = 0
    backbone_unfrozen = (args.mode == 'pretrain')
    print(f"[next_timepoint] checkpoint 选择判据: {ckpt_metric_name}"
          f"（同时记录其他指标，最终说明见训练日志/评估报告）")

    save_dir = os.path.join(args.checkpoint_dir, args.name)
    if os.path.exists(save_dir):
        import glob, re
        p = Path(save_dir)
        dirs = glob.glob(f"{p}*")
        ms   = [re.search(rf"%s(\d+)" % p.stem, d) for d in dirs]
        idx  = [int(m.groups()[0]) for m in ms if m]
        save_dir = f"{p}{max(idx)+1 if idx else 2}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.train_epochs):
        # 当前专家使用率统计（供负载均衡感知的温度调度使用）
        expert_usage_stats = None
        if hasattr(model, 'moe') and model.moe is not None:
            usage = model.moe.get_expert_usage()
            if usage is not None:
                expert_usage_stats = {f'E{i}': v.item() for i, v in enumerate(usage)}
        cur_temp = update_router_temperature(model, args, epoch, expert_usage_stats)

        # 多尺度门控的温度退火（熵 warmup）：前期高温 → 门控接近均匀，后期逐步锐化
        mdm = getattr(model, 'pastmixing', None)
        mdm_temp = None
        if mdm is not None and getattr(mdm, 'scale_gate_mode', None) == 'sample':
            warm = max(1, int(args.warmup_epochs))
            ratio = min(1.0, float(epoch) / warm)
            mdm_temp = (args.mdm_scale_gate_temp_start
                        + (args.mdm_scale_gate_temp_end - args.mdm_scale_gate_temp_start) * ratio)
            mdm.set_scale_gate_temperature(mdm_temp)

        if (args.mode == 'finetune'
                and not backbone_unfrozen
                and epoch >= args.freeze_backbone_epochs):
            set_finetune_stage(model, backbone_unfrozen=True)
            backbone_unfrozen = True
            optimizer = build_optimizer(model, criterion, args)
            scheduler = build_scheduler(optimizer, args)
            print(f"Epoch {epoch+1}: 已解冻主干，"
                  f"可训练参数量 = {count_trainable_params(model)}")

        model.train()
        # 轨迹任务：每 epoch 重新打乱分桶 sampler（同 batch 内 L 一致的约束保持不变）
        batch_sampler = getattr(train_data, 'batch_sampler', None)
        if hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)
        train_meters = defaultdict(list)    # tensor 累积，epoch 末一次性同步
        ode_meters = defaultdict(list)      # ODE 审计量（纯 float，不参与反传）
        live = defaultdict(float)           # 进度条所需的少数关键指标
        pbar = tqdm(train_data, total=len(train_data), dynamic_ncols=True, leave=False)
        pbar.set_description(f"Epoch {epoch+1:03d}/{args.train_epochs:03d} [Train]")

        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            # next_timepoint：x [B,F,1,K] / y [B,F,H,1] / x_last [B,F] / mask [B,H]
            bx, by, bl, bm, bs, bp = next_point_batch(batch, device)
            by_roll = batch.get('future', None)
            bm_roll = batch.get('future_mask', None)
            if by_roll is not None: by_roll = by_roll.to(device, non_blocking=True)
            if bm_roll is not None: bm_roll = bm_roll.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs, aux_info = model(bx, bs, bp)
                total_loss, loss_stats = criterion(
                    outputs, by, bl, aux_info=aux_info, mask=bm,
                    mtp_weights=mtp_weights, offsets=forecast_offsets)
                moe_reg, moe_stats = compute_moe_regularization(
                    aux_info, device,
                    load_balance_weight=args.moe_load_balance_weight,
                    entropy_weight=args.moe_entropy_weight,
                    z_loss_weight=args.moe_z_loss_weight,
                    diversity_weight=args.moe_diversity_weight,
                )
                graph_reg, graph_stats = compute_graph_regularization(
                    aux_info, device,
                    sparsity_weight=args.sc_sparsity_weight,
                    entropy_weight=args.sc_entropy_weight,
                    temporal_weight=args.sc_temporal_weight,
                )
                total_loss = total_loss + moe_reg + graph_reg
                # 可选短程 rollout 损失（--enable_rollout_loss）：模型自由滚动 R 步，
                # 中间不使用任何真值（未来真值只用于打分）
                roll_reg, roll_stats = 0.0, {}
                if args.enable_rollout_loss and args.lambda_rollout > 0 \
                        and by_roll is not None:
                    roll_steps = int(args.rollout_train_steps)
                    hist = bx[:, :, 0, :].transpose(1, 2).contiguous()   # [B,K,F]
                    roll_pred = model.rollout_next_states(hist, bs, bp, steps=roll_steps)
                    tgt_roll = by_roll[:, :, :roll_steps].transpose(1, 2).contiguous()
                    msk_roll = (bm_roll[:, :roll_steps] if bm_roll is not None else
                                torch.ones(roll_pred.shape[0], roll_steps,
                                           device=roll_pred.device))
                    roll_reg, roll_stats = compute_rollout_loss(
                        roll_pred, tgt_roll, msk_roll, args.lambda_rollout, device)
                    total_loss = total_loss + roll_reg
                inv_reg, inv_stats = compute_inversion_loss(
                    aux_info, args.inversion_weight, device)
                total_loss = total_loss + inv_reg
                amp_stats = {}
                if amp_loss_fn is not None:
                    amp_val = amp_loss_fn(by, aux_info, mask=bm)
                    if amp_val is not None:
                        total_loss = total_loss + args.head_amp_consistency_weight * amp_val
                        amp_stats['head_amp_consistency'] = amp_val.detach()

            scaler.scale(total_loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad]
                + [p for p in criterion.parameters() if p.requires_grad],
                max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if ema is not None: ema.update(model)

            if (args.moe_expert_stats_interval > 0
                    and getattr(model, 'moe', None) is not None
                    and aux_info.get('expert_diag')):
                for k, v in compute_expert_diversity(
                        model.moe, aux_info, device).items():
                    train_meters[k].append(v.detach())
            for k, v in {**loss_stats, **moe_stats, **graph_stats,
                         **inv_stats, **amp_stats, **roll_stats,
                         'metric_total': total_loss.detach()}.items():
                train_meters[k].append(v.detach())
            for k, v in (aux_info.get('ode_diag') or {}).items():
                ode_meters[k].append(float(v))
            live['metric_total'] += float(total_loss.item())
            live['loss_pcc']     += float(loss_stats['loss_pcc'].item())
            live['pcc']          += float(loss_stats['pcc'].item())
            postfix = {
                'Loss': f"{live['metric_total']/step:.4f}",
                'PCC_Loss': f"{live['loss_pcc']/step:.4f}",
                'PCC': f"{live['pcc']/step:.4f}",
            }
            if ode_meters:
                calls = sum(vs[-1] for vs in (ode_meters[k] for k in ode_meters
                                              if k.endswith('_calls')))
                postfix['ODE_calls'] = f'{calls:g}'
            pbar.set_postfix(postfix)

        eval_model  = ema.ema if ema is not None else model
        val_metrics = evaluate_next_point(eval_model, val_data, criterion, device,
                                          args, forecast_offsets, mtp_weights,
                                          rollout_steps=eval_rollout_steps,
                                          rollout_horizons=eval_rollout_horizons)
        scheduler.step()

        train_metrics = {k: torch.stack(vs).mean().item() for k, vs in train_meters.items()}
        ode_metrics = {k: sum(vs) / max(1, len(vs)) for k, vs in ode_meters.items()}
        group_lrs    = {g.get('name', f'g{i}'): g['lr']
                        for i, g in enumerate(optimizer.param_groups)}
        lr_text = ' | '.join(f"{k}: {v:.2e}" for k, v in group_lrs.items())

        # next-state 口径：报告绝对/delta/spatial-PCC 三项与 val 主指标 + trivial baseline
        tqdm.write(
            f"Epoch {epoch+1:03d} | {lr_text} | "
            f"Train Total: {train_metrics['metric_total']:.4f} | "
            f"abs: {train_metrics['loss_abs']:.4f} | "
            f"delta: {train_metrics['loss_delta']:.4f} | "
            f"PCC(sp): {train_metrics['pcc']:.4f} | "
            f"Val Total: {val_metrics['metric_total']:.4f} | "
            f"Val next MAE: {val_metrics['metric_mae_next']:.4f} | "
            f"Val next RMSE: {val_metrics['metric_rmse_next']:.4f} | "
            f"Val next PCC(sp): {val_metrics['metric_pcc_next']:.4f} | "
            f"Val persistence MAE: {val_metrics['metric_persistence_mae']:.4f} | "
            f"Val trend MAE: {val_metrics['metric_trend_mae']:.4f} | "
            f"NLL: {train_metrics.get('loss_nll', float('nan')):.4f}"
        )
        off_keys = [f'loss_off{o}' for o in forecast_offsets
                    if f'loss_off{o}' in train_metrics]
        if len(off_keys) > 1:
            tqdm.write('[NextPoint/MTP] ' + ' | '.join(
                f"t+{o}: loss={train_metrics.get(f'loss_off{o}', float('nan')):.4f} "
                f"pcc={train_metrics.get(f'pcc_off{o}', float('nan')):.4f} "
                f"n={train_metrics.get(f'n_off{o}', 0):.0f}"
                for o in forecast_offsets if f'loss_off{o}' in train_metrics))
        if 'rollout' in train_metrics:
            tqdm.write(f"[NextPoint] rollout_train_loss={train_metrics['rollout']:.4f} "
                       f"(steps={args.rollout_train_steps}, lambda={args.lambda_rollout})")
        roll_log = [f"H{h}: {val_metrics[f'rollout_mae_H{h}']:.4f}"
                    for h in (eval_rollout_horizons or [])
                    if f'rollout_mae_H{h}' in val_metrics]
        if roll_log:
            tqdm.write(f"[NextPoint] val rollout MAE (free, R={eval_rollout_steps}) | "
                       + ' | '.join(roll_log))
        tqdm.write(
            f"[NextPoint] checkpoint criterion = {ckpt_metric_name}"
            f"={val_metrics[ckpt_metric_key]:.4f} | "
            f"model vs persistence ΔMAE="
            f"{val_metrics['metric_persistence_mae'] - val_metrics['metric_mae_next']:+.4f}")
        if cur_temp is not None:
            usage = ' '.join([
                f"E{i}:I{train_metrics.get(f'moe_importance_e{i}',0):.3f}"
                f"/L{train_metrics.get(f'moe_load_e{i}',0):.3f}"
                for i in range(args.num_experts)])
            tqdm.write(
                f"MoE Temp: {cur_temp:.3f} | "
                f"LB: {train_metrics.get('moe_load_balance',0):.6f} | "
                f"LBT_raw: {train_metrics.get('moe_lbt_raw',0):.4f} | "
                f"Z-Loss: {train_metrics.get('moe_z_loss',0):.6f} | "
                f"Entropy: {train_metrics.get('moe_entropy',0):.4f} | "
                f"{usage}"
            )

        is_best = val_metrics[ckpt_metric_key] < best_ckpt_value
        cur_metric = val_metrics[ckpt_metric_key]
        if is_best:
            best_ckpt_value = cur_metric
            best_name = 'base_best.pt' if args.mode == 'pretrain' else 'finetuned_best.pt'
            save_backbone_weights(os.path.join(save_dir, best_name), eval_model,
                                  meta={'args': vars(args)})
            tqdm.write(f"New Best Model Saved! (Best {ckpt_metric_name}: "
                       f"{best_ckpt_value:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly Stopping! 连续 {args.patience} epoch 未改善。"
                           f"\n最佳 {ckpt_metric_name}: {best_ckpt_value:.4f}")
                last_name = 'base_last.pt' if args.mode == 'pretrain' else 'finetuned_last.pt'
                save_backbone_weights(os.path.join(save_dir, last_name),
                                      ema.ema if ema else model,
                                      meta={'args': vars(args)})
                break

        if epoch == args.train_epochs - 1:
            last_name = 'base_last.pt' if args.mode == 'pretrain' else 'finetuned_last.pt'
            save_backbone_weights(os.path.join(save_dir, last_name),
                                  ema.ema if ema else model,
                                  meta={'args': vars(args)})

        # TensorBoard
        tb_scalars = [
            ('Train/Total', train_metrics['metric_total']),
            ('Train/PCC_Loss', train_metrics['loss_pcc']),
            ('Train/PCC', train_metrics['pcc']),
            ('Val/Total', val_metrics['metric_total']),
            ('Val/PCC_Loss', val_metrics['loss_pcc']),
            ('Val/PCC', val_metrics['pcc']),
            ('Train/NLL', train_metrics['loss_nll']),
            ('Val/NLL', val_metrics['loss_nll']),
        ]
        # next-state 口径的日志键（prompt §三十七）：train/loss_abs|delta|pcc 与
        # val/next_MAE|next_RMSE|next_PCC + persistence/trend baseline
        tb_scalars += [
            ('Train/Abs', train_metrics['loss_abs']),
            ('Train/Delta', train_metrics['loss_delta']),
            ('Val/Next_MAE', val_metrics['metric_mae_next']),
            ('Val/Next_RMSE', val_metrics['metric_rmse_next']),
            ('Val/Next_PCC', val_metrics['metric_pcc_next']),
            ('Val/Persistence_MAE', val_metrics['metric_persistence_mae']),
            ('Val/Trend_MAE', val_metrics['metric_trend_mae']),
        ]
        for o in forecast_offsets:
            if f'loss_off{o}' in train_metrics:
                tb_scalars.append((f'Train/Loss_t+{o}', train_metrics[f'loss_off{o}']))
            if f'pcc_off{o}' in train_metrics:
                tb_scalars.append((f'Train/PCC_t+{o}', train_metrics[f'pcc_off{o}']))
        if 'rollout' in train_metrics:
            tb_scalars.append(('Train/Rollout_Loss', train_metrics['rollout']))
        for h, key in sorted((h, f'rollout_mae_H{h}')
                             for h in (eval_rollout_horizons or [])):
            if key in val_metrics:
                tb_scalars.append((f'Val/Rollout_MAE_H{h}', val_metrics[key]))
        for tag, val in tb_scalars:
            writer.add_scalar(tag, val, epoch)
        if 'inversion_mse' in train_metrics:
            writer.add_scalar('Train/Inversion_MSE', train_metrics['inversion_mse'], epoch)
            if 'inversion_mse' in val_metrics:
                writer.add_scalar('Val/Inversion_MSE', val_metrics['inversion_mse'], epoch)
        for lv in ('nll',):
            if f'log_var_{lv}' in train_metrics:
                writer.add_scalar(f'LossWeight/LogVar_{lv.upper()}',
                                  train_metrics[f'log_var_{lv}'], epoch)
        for name, lr in group_lrs.items():
            writer.add_scalar(f'LR/{name}', lr, epoch)
        for tag, key in [('Train/Graph_Sparsity', 'graph_sparsity'),
                          ('Train/Graph_Entropy', 'graph_entropy'),
                          ('Train/Graph_Temporal', 'graph_temporal')]:
            if key in train_metrics:
                writer.add_scalar(tag, train_metrics[key], epoch)
        for k, v in ode_metrics.items():
            writer.add_scalar(f'Train/ODE_{k}', v, epoch)
        if mdm_temp is not None:
            writer.add_scalar('Train/MDM_ScaleGateTemp', mdm_temp, epoch)
            writer.add_scalar('Train/MDM_ScaleGateEntropy',
                              float(getattr(mdm, 'last_scale_gate_entropy', 0.0)), epoch)
        if cur_temp is not None:
            writer.add_scalar('Train/MoE_Temp', cur_temp, epoch)
            for tag, key in [('Train/MoE_LoadBalance', 'moe_load_balance'),
                              ('Train/MoE_LBT_Raw', 'moe_lbt_raw'),
                              ('Train/MoE_ZLoss', 'moe_z_loss'),
                              ('Train/MoE_Entropy', 'moe_entropy')]:
                if key in train_metrics:
                    writer.add_scalar(tag, train_metrics[key], epoch)
            for k, v in train_metrics.items():
                if k.startswith('moe_expert_') or k == 'moe_router_prob_std':
                    writer.add_scalar(f'Train/MoE_{k[len("moe_"):]}', v, epoch)
            for i in range(args.num_experts):
                for suffix in ('importance', 'load', 'gate_mean', 'select_freq'):
                    k = f'moe_{suffix}_e{i}'
                    if k in train_metrics:
                        writer.add_scalar(f'Train/MoE_{suffix.title().replace("_","")}E{i}',
                                          train_metrics[k], epoch)
                    if k in val_metrics:
                        writer.add_scalar(f'Val/MoE_{suffix.title().replace("_","")}E{i}',
                                          val_metrics[k], epoch)

    writer.close()
    best_n = 'base_best.pt' if args.mode == 'pretrain' else 'finetuned_best.pt'
    best_path = os.path.join(save_dir, best_n)
    print(f"\n训练完成！最佳权重已保存至: {best_path}")

    if args.mode == 'finetune' and getattr(args, 'eval_after_train', True) \
            and os.path.exists(best_path):
        # 训练结束自动评 test（闭环）：复用版本化切分 manifest，含 conformal
        # 校准（自动补评 val 拟合）与 shuffled-HAMD 负对照。评估失败不影响训练产物。
        from experiments.evaluate_variant import evaluate as _eval_variant
        eval_dir = os.path.join(save_dir, 'eval')
        try:
            summary = _eval_variant(best_path, eval_dir, eval_splits=('test',),
                                    seed=args.seed, run_shuffled=True)
            print(f"[post-train eval] test_PCC={summary.get('test_pcc')} "
                  f"test_MAE={summary.get('test_mae')} "
                  f"PICP={summary.get('test_picp')} PICP_cal={summary.get('test_picp_cal')}")
            print(f"[post-train eval] 详细指标见: {eval_dir}/metrics_test.json")
        except Exception as e:
            print(f"[post-train eval] 自动评估失败（训练产物不受影响）: {type(e).__name__}: {e}\n"
                  f"可手动复评: python -m experiments.evaluate_variant --ckpt {best_path} --out_dir {eval_dir}")


def parse_args():
    p = argparse.ArgumentParser(description='NeuroTwin: Pathology-Conditioned Mixture-of-Denoising-Experts Digital Twin for Brain Dynamics')
    p.add_argument('--mode', default='pretrain', choices=['pretrain', 'finetune'])
    p.add_argument('--pretrained_weight', default='./checkpoints/neurotwin_pretrain/base_best.pt')
    p.add_argument('--pathology_input_dim', type=int, default=1)
    p.add_argument('--pathology_dim', type=int, default=32)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--name', default='neurotwin')
    p.add_argument('--clinical_file', default='Rest-meta-MDD-V1V2-Merged-MDD.xlsx')
    p.add_argument('--num_rois', type=int, default=116)
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--total_windows', type=int, default=9)
    p.add_argument('--n_block', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--norm', type=str2bool, default=False)  # 数据已 z-score，默认关闭 BrainRevIN（与 base_config 同步）
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--num_scales', type=int, default=3)
    p.add_argument('--ode_steps', type=int, default=3)  # 2026-09-25 消融：1–12 步差 ≤0.002，取 3 省算力（与 base_config 同步）
    p.add_argument('--ode_hidden_dim', type=int, default=256)
    p.add_argument('--stochastic_depth_rate', type=float, default=0.10)
    p.add_argument('--num_experts', type=int, default=4)
    p.add_argument('--top_k', type=int, default=2)
    p.add_argument('--train_epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_init', type=float, default=5e-5)
    p.add_argument('--lr_peak', type=float, default=1e-4)
    p.add_argument('--lr_final', type=float, default=5e-5)
    p.add_argument('--warmup_epochs', type=int, default=15)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--loss_lr_scale', type=float, default=1.0)

    # MoDE MoE 超参
    p.add_argument('--moe_load_balance_weight', type=float, default=0.05,
                   help='负载均衡损失权重。乘以 load_balancing_term（均匀时≈1.0）。')
    p.add_argument('--moe_entropy_weight', type=float, default=1e-3,
                   help='门控熵正则化权重，鼓励路由分布不过于集中。')
    p.add_argument('--moe_z_loss_weight', type=float, default=1e-3,
                   help='路由 Z-Loss 权重（ST-MoE），惩罚 logit 幅度过大。')
    p.add_argument('--moe_diversity_weight', type=float, default=0.001,
                   help='专家多样化损失权重，鼓励 batch 内路由决策的多样性。典型值 0.001~0.01。')
    p.add_argument('--moe_gate_temp_start', type=float, default=1.5,
                   help='路由温度初始值。高温使 softmax 更均匀，有利于 multinomial 探索。')
    p.add_argument('--moe_gate_temp_end', type=float, default=1.0,
                   help='路由温度终止值（backbone 解冻后线性衰减）。')
    p.add_argument('--moe_expert_hidden_dim', type=int, default=256,
                   help='所有路由专家统一隐层维度（均等容量）。')
    p.add_argument('--moe_use_shared_expert', type=str2bool, default=True,
                   help='是否启用共享专家（SharedPathologyExpert，始终激活）。')
    p.add_argument('--moe_router_cond_only', type=str2bool, default=False,
                   help='路由仅基于病理条件（不含脑信号特征）。默认 False：'
                        '仅用病理条件时门控与脑状态无关，路由近似退化为常量。')
    p.add_argument('--moe_use_argmax', type=str2bool, default=False,
                   help='推理时用 topk（False=默认），训练时用 multinomial 随机探索。')
    p.add_argument('--moe_inference_temperature', type=float, default=0.3,
                   help='推理时路由器的幂次缩放温度（0.2~0.5，默认 0.3），'
                        '解决"训练 multinomial 均匀 → 推理 topk 坍塌"的不一致问题。')

    # ---------- Phase 0：协议基础（检查点版本化 / 病理归一化 / 路由 / Refiner） ----------
    p.add_argument('--pretrained_arch_policy', default='require_match',
                   choices=['require_match', 'warn', 'ignore'],
                   help='预训练权重架构版本不匹配时的策略：require_match 直接报错；'
                        'warn 打印并跳过不兼容键；ignore 静默跳过。')
    p.add_argument('--pretrained_skip_pattern', default='',
                   help='fnmatch 模式，命中的预训练键不加载（保留随机初始化），'
                         '用于受控消融对照，如 head_shape_scratch 用 future_query.*')
    p.add_argument('--refresh_split_manifest', action='store_true',
                   help='重新生成 subject_split_<mode>.json 切分 manifest'
                        '（默认存在且配置一致时复用，配置不一致时报错）')
    p.add_argument('--no_eval_after_train', dest='eval_after_train',
                   action='store_false',
                   help='finetune 训练结束后不自动评 test（默认自动评估并落盘到'
                        ' <save_dir>/eval/）')
    p.add_argument('--pathology_norm_mode', default='robust_z',
                   choices=['identity', 'robust_z', 'zscore_quadratic',
                            'empirical_cdf', 'quantile', 'zscore_rbf'],
                   help='模型内病理评分归一化模式，决定条件输入宽度。')
    p.add_argument('--pathology_norm_quantiles', type=int, default=64,
                   help='empirical_cdf/quantile 模式的分位数网格点数。')
    p.add_argument('--pathology_norm_rbf_knots', type=int, default=8,
                   help='zscore_rbf 模式的 RBF 结点数。')
    p.add_argument('--pathology_poly_expansion', type=str2bool, default=False,
                   help='是否在病理投影中使用 [x, x², sin(πx)] 多项式展开（仅 input_dim==1）。')
    p.add_argument('--pathology_fields', default=None,
                   help='逗号分隔的病理字段列表，默认 None 时使用 HAMD 标量。')
    p.add_argument('--pathology_missing', default='drop',
                   choices=['drop', 'zero', 'mean'],
                   help='病理字段缺失处理策略。')
    p.add_argument('--reuse_pretrained_norm_stats', type=str2bool, default=True,
                   help='加载的检查点已带归一化统计量时是否复用（否则用 train subjects 重新拟合）。')
    p.add_argument('--moe_eval_mode', default='dense_soft',
                   choices=['dense_soft', 'topk', 'legacy'],
                   help='评估期路由模式：dense_soft 全专家软融合（确定性）；topk 确定性 top-k；legacy 旧行为。')
    p.add_argument('--moe_gate_features', default='state_revin',
                   choices=['state', 'state_revin', 'state_revin_difficulty'],
                   help='路由门控输入特征：state=旧 5 项状态统计；'
                        'state_revin=追加 RevIN mean/std（默认）；'
                        'state_revin_difficulty=再追加"难度"项（base 预测与最后历史窗的偏差）。')
    p.add_argument('--moe_gate_input_dim', type=int, default=None,
                   help='门控输入宽度，默认按 gate_features 与 ROI 数自动推导并校验。')
    p.add_argument('--moe_experts_mode', default='routed_only',
                   choices=['routed_shared', 'shared_only', 'routed_only', 'none'],
                   help='专家臂组合：routed_only=仅路由（默认，2026-09-25 消融：shared 增量≈0）；'
                        'routed_shared=路由+共享；shared_only=仅共享；'
                        'none=关闭 MoE 残差分支（干净消融，前向等于主干）。')
    p.add_argument('--moe_expert_kind', default='homogeneous',
                   choices=['homogeneous', 'heterogeneous'],
                   help='专家结构：homogeneous=同宽同激活（默认）；'
                        'heterogeneous=宽度 0.5×~1.0× 且激活 SiLU/GELU 交替。')
    p.add_argument('--moe_route_level', default='sample',
                   choices=['sample', 'token'],
                   help='路由粒度：sample=每样本一个决策（默认）；token=每个 (样本,ROI) 一个决策'
                        '（需配合 --moe_router_cond_only False；计算量不随稀疏度下降）。')
    p.add_argument('--moe_eval_mc_samples', type=int, default=0,
                   help='评估期额外做 N 次 legacy 随机路由以报告 moe_router_prob_std；0 关闭。')
    p.add_argument('--moe_expert_stats_interval', type=int, default=0,
                   help='每 N 次前向采集专家输出范数与跨窗路由一致性（诊断）；0 关闭。')
    p.add_argument('--refiner_rounds', type=int, default=3,
                   help='主干预测头 IterativePredictionRefiner 的迭代轮数。')
    p.add_argument('--delta_refiner_rounds', type=int, default=1,
                   help='MoE 病理残差细化器 IterativePredictionRefiner 的迭代轮数。'
                        '默认 1：2026-09-25 消融显示 1 轮与 2 轮等价。')
    p.add_argument('--refiner_adaptive', type=str2bool, default=False,
                   help='是否启用轮间自适应强度门控（依修正幅度调节每轮权重）。')
    p.add_argument('--refiner_inter_sup_weight', type=float, default=0.05,
                   help='轮间（deep supervision）损失权重，<=0 关闭。')
    p.add_argument('--refiner_inter_sup_decay', type=float, default=0.5,
                   help='轮间监督的几何衰减系数（越靠后轮权重越大）。')
    p.add_argument('--adapter_lr_scale', type=float, default=1.0,
                   help='条件适配器（adapter）参数组的学习率缩放。')

    # ---------- Phase 1：病理条件化（AdaLN 特征级调制 / LoRA） ----------
    p.add_argument('--patho_cond_layer', default='joint',
                   choices=['residual_only', 'feature_only', 'joint'],
                   help='条件注入层级：residual_only 仅 MoE 残差；'
                        'feature_only 仅主干 AdaLN；joint 两者同时启用。')
    p.add_argument('--patho_adaln_targets', default='both',
                   choices=['none', 'mdm', 'ode', 'both'],
                   help='特征级 AdaLN 的注入位置（residual_only 时强制为 none）。')
    p.add_argument('--lora_enable', type=str2bool, default=True,
                   help='微调时在最后 lora_n_blocks 个 ODE block 上启用 LoRA（B 零初始化）。')
    p.add_argument('--lora_rank', type=int, default=8,
                   help='LoRA 低秩维度。')
    p.add_argument('--lora_n_blocks', type=int, default=2,
                   help='挂载 LoRA 的 ODE block 数量（从最后一层往前数）。')

    # ---------- Phase 2：SC 软先验（soft anatomical prior / 图正则） ----------
    p.add_argument('--sc_prior_mode', default='soft_prior',
                   choices=['scaled', 'soft_prior', 'adaptive_only', 'functional_only'],
                   help='SC 注入形态：scaled=旧的静态 SC 度归一化（复现 Phase 1）；'
                        'soft_prior=λ·A_SC+(1-λ)·A_func（默认）；'
                        'adaptive_only=纯数据驱动功能图（同归一化口径）；'
                        'functional_only=功能图原始 row_softmax（不做 Sinkhorn）。')
    p.add_argument('--sc_lambda_mode', default='global',
                   choices=['global', 'sample', 'roi'],
                   help='λ 混合系数的粒度：global 全局标量 / sample 样本级 / roi 逐 ROI。')
    p.add_argument('--sc_lambda_init', type=float, default=0.7,
                   help='λ 初值（越大越信任 SC 结构先验）。')
    p.add_argument('--sc_prior_rank', type=int, default=12,
                   help='功能图 A_func 的低秩分解维度。')
    p.add_argument('--sc_delta_a', type=str2bool, default=True,
                   help='是否启用低秩 subject-specific 邻接残差 ΔA（零初始化）。')
    p.add_argument('--sc_delta_rank', type=int, default=None,
                   help='ΔA 的秩，默认 max(4, sc_prior_rank//2)。')
    p.add_argument('--sc_prob_mask', type=str2bool, default=False,
                   help='是否启用可学习概率掩码（straight-through，初始化保留约 95%% 边）。')
    p.add_argument('--sc_sinkhorn_iters', type=int, default=64,
                   help='对称 Sinkhorn 迭代次数（使 A_eff 同时行归一且对称；靠近均匀的图收敛慢）。')
    p.add_argument('--sc_mask_mode', default='soft',
                   choices=['hard', 'soft', 'none'],
                   help='GraphODE 图注意力掩码：hard=旧硬掩码（仅 scaled 模式等效）；'
                        'soft=A_eff>tau 软掩码（默认，不永久切断非 SC 边）；none=无图先验。')
    p.add_argument('--sc_mask_tau', type=float, default=1e-3,
                   help='soft 掩码阈值。')
    p.add_argument('--sc_refiner_inject', default='both',
                   choices=['base', 'delta', 'both', 'none'],
                   help='预测端 SC 注入点：base=主干 refiner / delta=MoE 残差 refiner / '
                        'both=两者（默认）/ none=不注入。')
    p.add_argument('--head_cross_roi', default='conv1',
                   choices=['conv1', 'skip', 'sc_prior'],
                   help='预测头 cross-ROI 分支形态：conv1=自由 ROI×ROI 线性混合（默认）；'
                        'sc_prior=用 A_eff 做图混合；skip=去掉该分支。')
    p.add_argument('--sc_sparsity_weight', type=float, default=1e-3,
                   help='SC 软先验稀疏正则权重（A_eff 非对角 L1），<=0 关闭。')
    p.add_argument('--sc_entropy_weight', type=float, default=1e-3,
                   help='SC 软先验行熵正则权重（抑制模糊均匀图），<=0 关闭。')
    p.add_argument('--sc_temporal_weight', type=float, default=0.0,
                   help='逐窗功能图时间一致性正则权重（>0 时才生成 A_seq），默认关闭。')

    # ---------- Phase 3：预测头未来查询 / 幅值重参数化 / 多尺度去重 ----------
    p.add_argument('--head_shape_mode', default='query',
                   choices=['flatten', 'query'],
                   help='形状分支形态：flatten=展平线性（旧）；'
                        'query=FutureQueryDecoder 未来查询交叉注意力（默认）。')
    p.add_argument('--future_query_mode', default='roi',
                   choices=['roi', 'roi_window', 'roi_time'],
                   help='未来查询粒度：roi / roi_window / roi_time（3480 查询，opt-in）。')
    p.add_argument('--future_query_dim', type=int, default=32,
                   help='未来查询解码器的注意力维度（需能被 future_query_heads 整除）。')
    p.add_argument('--future_query_layers', type=int, default=1,
                   help='未来查询解码器的交叉注意力层数。')
    p.add_argument('--future_query_heads', type=int, default=4,
                   help='未来查询解码器的注意力头数。')
    p.add_argument('--head_amp_mode', default='scale_mod_trend',
                   choices=['legacy', 'scale_mod_trend'],
                   help='幅值重参数化：legacy=anchor+trend+scale*shape；'
                        'scale_mod_trend=anchor+scale*(z(trend)+z(shape))（默认）。')
    p.add_argument('--head_scale_granularity', default='window',
                   choices=['window', 'timestep'],
                   help='幅值参数粒度：window=[B,F,W\',1]；timestep=[B,F,W\',S\']。')
    p.add_argument('--head_amp_consistency_weight', type=float, default=0.01,
                   help='幅值一致性损失权重（监督 scale 等于目标残差实际幅值），<=0 关闭。')
    p.add_argument('--head_use_history_proj', type=str2bool, default=True,
                   help='预测头：是否使用扁平化历史特征分支。')
    p.add_argument('--head_use_latent_proj', type=str2bool, default=True,
                   help='预测头：是否使用扁平化潜在特征分支。')
    p.add_argument('--head_use_temporal', type=str2bool, default=True,
                   help='预测头：是否使用时间轴 depthwise 卷积分支。')
    p.add_argument('--head_use_cross_roi', type=str2bool, default=True,
                   help='预测头：是否使用 ROI 轴混合分支（False 等价 head_cross_roi=skip）。')
    p.add_argument('--head_use_win_attn', type=str2bool, default=True,
                   help='预测头：是否使用因果跨窗注意力分支。')
    p.add_argument('--head_use_revin_stats', type=str2bool, default=True,
                   help='预测头：是否使用 RevIN mean/std 统计分支（norm=False 时自动关闭）。')
    p.add_argument('--ode_window_attn', default='on', choices=['on', 'off'],
                   help='ODE 内窗口注意力分支开关（与 head_use_win_attn 构成去重 2×2 消融）。')
    p.add_argument('--ode_solver', default='rk2',
                   choices=['euler', 'rk2', 'rk4', 'adaptive', 'sde'],
                   help='ODE 积分器：rk2=RK2(Heun，默认，与旧实现数值一致)；rk4 更准但前向×2；'
                        'adaptive=dopri5 自适应步长（需 torchdiffeq）；sde 默认关闭（破坏确定性）。')
    p.add_argument('--ode_step_mode', default='learnable', choices=['fixed', 'learnable'],
                   help='ODE 步长模式：learnable=softplus 可学习（默认，初值与 fixed 等价）；'
                        'fixed=固定为 --ode_step_scale。')
    p.add_argument('--ode_step_scale', type=float, default=0.1,
                   help='ODE 步长（fixed 模式使用；learnable 模式下为 softplus 初值对应的 0.1）。')
    p.add_argument('--sde_noise_scale', type=float, default=1e-3,
                   help='SDE 噪声尺度，仅在 --ode_solver sde 时生效（默认极小，等价关闭）。')
    p.add_argument('--ode_adaptive_rtol', type=float, default=1e-3,
                   help='adaptive 求解器相对容差（仅 --ode_solver adaptive）。')
    p.add_argument('--ode_adaptive_atol', type=float, default=1e-4,
                   help='adaptive 求解器绝对容差（仅 --ode_solver adaptive）。')
    p.add_argument('--mdm_scale_scheme', default='divisor', choices=['pow2', 'divisor'],
                   help='BrainMDM 多尺度方案：pow2=旧的 2 幂次（小 L 退化）；'
                        'divisor=L//i（W=6 时给出 {2,3,6}，默认）。')
    p.add_argument('--mdm_scale_gate', default='sample', choices=['none', 'sample'],
                   help='多尺度合并方式：none=均匀平均；sample=样本级 softmax 门控（默认）。')
    p.add_argument('--mdm_scale_gate_temp_start', type=float, default=2.0,
                   help='多尺度门控温度初值（熵 warmup：高温更均匀）。')
    p.add_argument('--mdm_scale_gate_temp_end', type=float, default=1.0,
                   help='多尺度门控温度终值。')

    p.add_argument('--freeze_backbone_epochs', type=int, default=10)
    p.add_argument('--backbone_lr_scale', type=float, default=0.20)
    p.add_argument('--val_ratio', type=float, default=0.10)
    p.add_argument('--test_ratio', type=float, default=0.10)
    p.add_argument('--stratify_bins', type=int, default=5)
    p.add_argument('--cache_in_memory', type=str2bool, default=False)
    p.add_argument('--use_ema', type=str2bool, default=True)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--amp', type=str2bool, default=True)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_threads', type=int, default=4)
    p.add_argument('--pin_memory', type=str2bool, default=True)
    p.add_argument('--persistent_workers', type=str2bool, default=True)
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--eval_batch_size', type=int, default=None,
                   help='验证/测试 DataLoader 的 batch 大小，默认为 batch_size×4')
    p.add_argument('--tf32', type=str2bool, default=True,
                   help='启用 cudnn.benchmark 与 TF32（对 autocast 未覆盖的 fp32 计算加速）')
    # ---- Phase 7：不确定性 / 辅助反演 / 反事实 / RevIN 统计复用 ----
    p.add_argument('--pred_head', default='gaussian', choices=['point', 'gaussian', 'quantile'],
                   help='预测头输出形式：point=仅点预测（参数量与旧实现一致）；'
                        'gaussian=均值+对数方差（默认，损失中新增高斯 NLL 项）；'
                        'quantile=分位点回归（用 pinball 损失）。')
    p.add_argument('--pred_quantiles', default='0.1,0.5,0.9',
                   help='--pred_head quantile 时的分位点列表（逗号分隔，需 (0,1) 内严格递增）。')
    p.add_argument('--init_log_var_nll', type=float, default=2.0,
                   help='高斯 NLL / pinball 项的不确定性加权初值。取正值（默认 2.0）使初始'
                        '权重 exp(-2)≈0.135 明显小于其他项，避免概率项早期主导训练。')
    p.add_argument('--inversion_weight', type=float, default=0.0,
                   help='辅助反演损失权重（池化潜在状态回归归一化病理条件）。'
                        '默认 0 关闭：不给纯预测任务强加第二目标。')
    p.add_argument('--inversion_hidden_dim', type=int, default=128,
                   help='辅助反演头隐层维度（仅 --inversion_weight>0 时构建）。')
    p.add_argument('--intervention_mode', default='latent', choices=['latent', 'parametric'],
                   help='反事实扫掠 counterfactual_pathology_sweep / 虚拟干预使用的模式：'
                        'latent=潜在状态与条件按 intensity 混合；parametric=直接用目标条件重跑。'
                        '本训练入口不消费该参数，仅供分析脚本保持一致口径。')
    p.add_argument('--compile', type=str2bool, default=True,
                   help='torch.compile 编译模型前向（稳态约 35%% 提速；首次编译有额外耗时）')

    # ---------- 任务口径与 next_timepoint 数据协议 ----------
    p.add_argument('--task_mode', default='next_timepoint',
                   choices=['next_timepoint'],
                   help='任务口径：next_timepoint=连续 BOLD context → 下一 TR 全脑状态'
                        '（Next Brain-State Prediction，context_min..context_max 个 TR → '
                        'forecast_offsets 指定的未来 TR，默认仅 +1）。')
    p.add_argument('--random_context', type=str2bool, default=True,
                   help='训练是否随机采样 context 长度 K（False=固定为可用最大 K）。')
    p.add_argument('--random_cutoff', type=str2bool, default=True,
                   help='训练是否随机采样 cutoff t（False=固定最晚 cutoff）。')
    p.add_argument('--bold_source', default='auto', choices=['auto', 'normlize', 'windows'],
                   help='连续 BOLD 数据源：auto=优先 data/Normlize/<grp>/ROISignals_<id>.mat，'
                        '缺失时回退用 50%% 重叠滑窗重构；normlize=仅用连续序列；windows=仅用滑窗重构。')
    p.add_argument('--train_samples_per_subject', type=int, default=0,
                   help='训练每个被试每 epoch 的样本预算；0=auto（min(合法组合数, 12)）。')
    p.add_argument('--sampling_seed', type=int, default=2024,
                   help='采样/分桶随机种子（训练采样可复现；评估协议本身与随机性无关）。')
    p.add_argument('--subject_cache_size', type=int, default=256,
                   help='数据集每个 worker 的被试级缓存上限（0=不缓存）。')
    p.add_argument('--eval_fc', type=str2bool, default=True,
                   help='评估是否输出 FC 层面指标。')
    p.add_argument('--eval_spectral', type=str2bool, default=False,
                   help='评估是否输出 Welch PSD 低频段一致性指标（可选）。')
    p.add_argument('--eval_spectral_tr', type=float, default=2.0,
                   help='频谱评估使用的 TR（秒），默认 2.0 s（REST-meta-MDD 常用值）。')

    # ---------- Phase 9：Next-Timepoint Prediction（连续 BOLD → 下一 TR 全脑状态） ----------
    p.add_argument('--context_min', type=int, default=16,
                   help='next_timepoint 训练 context 长度下界 K_min（TR 数）。')
    p.add_argument('--context_max', type=int, default=64,
                   help='next_timepoint context 长度上界 K_max（= 模型 S 轴建模宽度，决定参数量）。'
                        '训练随机采样 K ∈ context_lengths 或 [context_min, context_max]。')
    p.add_argument('--context_lengths', default='',
                   help='可选：离散 context 长度集合（逗号分隔，如 16,32,64）；'
                        '空=在 [context_min, context_max] 区间内均匀随机采样。')
    p.add_argument('--prediction_target', default='delta', choices=['delta', 'absolute'],
                   help='预测头参数化：delta=以 x_t 为锚点预测 Δx（默认，'
                        'x̂ = x_t + Δx̂，抑制退化成复制 x_t）；absolute=零锚点直接预测状态。')
    p.add_argument('--causal_training', default='random_context',
                   choices=['random_context', 'full_sequence'],
                   help='训练范式：random_context=随机 context → 下一时间点（方案 B，已实现，'
                        '未来信息不进入前向）；full_sequence=GPT 式全序列 teacher-forcing'
                        '（方案 A，需要重写主干时间轴算子，当前显式报错未实现）。')
    p.add_argument('--forecast_offsets', default='',
                   help='预测偏移列表（逗号分隔，相对当前 TR 的 +Δ，需严格递增），'
                        '空=按 --enable_mtp 取 [1]（单步）或 [1,2,4,8]（MTP）。')
    p.add_argument('--enable_mtp', type=str2bool, default=False,
                   help='启用 Parallel Multi-Timepoint Prediction（同一个 hidden state 并行预测'
                        '[1,2,4,8] 多个未来偏移，prompt §二十九）。默认关闭。')
    p.add_argument('--mtp_weights', default='',
                   help='逐偏移损失权重（逗号分隔，长度须等于预测偏移数）；'
                        '空=均匀权重；MTP 示例 1.0,0.7,0.5,0.3。')
    p.add_argument('--lambda_abs', type=float, default=1.0,
                   help='next_timepoint 损失：绝对状态 L1 项权重。')
    p.add_argument('--lambda_delta', type=float, default=1.0,
                   help='next_timepoint 损失：delta（变化量 Δx = x_(t+δ) - x_t）L1 项权重。'
                        'prediction_target=delta（x_t 锚点）时该项与 lambda_abs 作用于同一'
                        '目标（数值重合，见损失 docstring）；absolute 模式下提供独立梯度。')
    p.add_argument('--lambda_pcc', type=float, default=0.1,
                   help='next_timepoint 损失：spatial PCC 惩罚项权重（1 - 逐 ROI 空间相关）。')
    p.add_argument('--lambda_nll', type=float, default=0.0,
                   help='next_timepoint 损失：概率项（高斯 NLL / pinball）权重；'
                        '默认 0 关闭——单点预测下 logvar 头的方差换算在近常数 context '
                        '上不稳定（exp(-logvar) 可达 1e3），如需概率输出请显式开启并检查。')
    p.add_argument('--enable_rollout_loss', type=str2bool, default=False,
                   help='启用短程自回归 rollout 训练损失（默认关闭；开启后每步额外'
                        ' rollout_train_steps 次前向，成本与稳定性风险上升）。')
    p.add_argument('--rollout_train_steps', type=int, default=2,
                   help='rollout 训练损失的滚动步数（prompt 建议 2 或 4）。')
    p.add_argument('--lambda_rollout', type=float, default=0.2,
                   help='rollout 损失权重（仅 --enable_rollout_loss True 时生效）。')
    p.add_argument('--eval_context_length', type=int, default=0,
                   help='评估固定 context 长度 K_eval；0=自动取 --context_max。'
                        '评估协议对每个被试使用同一 K_eval 与确定性 anchor，禁止随机位置。')
    p.add_argument('--eval_anchors_per_subject', type=int, default=16,
                   help='每个被试评估的 next-state 预测点数量（等间隔确定性选取；0=枚举全部）。')
    p.add_argument('--eval_rollout_tasks_per_subject', type=int, default=2,
                   help='每个被试参与 free rollout / FC / 频谱评估的 anchor 数（0=不做 rollout）。')
    p.add_argument('--eval_rollout_horizons', default='1,2,4,8,16',
                   help='free rollout 的评估 horizon 列表（逗号分隔）。')
    p.add_argument('--eval_rollout_steps', type=int, default=0,
                   help='单次 rollout 的滚动步数 R；0=自动取 max(horizons) 并在启用 FC/频谱时'
                        '抬到 --fc_min_length（FC 需要更长的轨迹）。')
    p.add_argument('--fc_min_length', type=int, default=32,
                   help='FC/频谱评估所需的最小 rollout 长度（prompt §二十二；短于该长度不计算 FC）。')
    p.add_argument('--load_backbone_only', type=str2bool, default=False,
                   help='finetune 时只加载主干权重（预测头 / MoE / 条件模块保持随机初始化），'
                        '并显式打印 loaded/missing/reinitialized 三类清单。')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
