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
from utils.common import parse_pred_quantiles, set_seed, str2bool
from models.neurotwin import NeuroTwin
from train.losses import (IntermediateSupervisionLoss,
                          UncertaintyWeightedHybridLoss,
                          compute_intermediate_supervision,
                          compute_inversion_loss)
from train.optim import (ModelEMA, build_optimizer, build_scheduler,
                         count_trainable_params, load_backbone_weights,
                         save_backbone_weights, set_finetune_stage)
from train.moe import (compute_expert_diversity, compute_graph_regularization,
                       compute_moe_regularization, update_router_temperature)
from models.decoder import AmplitudeConsistencyLoss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args):
    model.eval()
    # 该损失无参数，评估期就地构建，避免额外在函数间传递
    amp_loss_fn = (AmplitudeConsistencyLoss(granularity=args.head_scale_granularity)
                   if (args.head_amp_consistency_weight > 0
                       and args.head_amp_mode == 'scale_mod_trend') else None)
    meters = defaultdict(list)      # tensor 累积，epoch 末一次性同步，避免每 step .item() 造成 GPU/CPU 串行
    live = defaultdict(float)       # 进度条所需的少数关键指标
    pbar = tqdm(data_loader, total=len(data_loader), dynamic_ncols=True, leave=False)
    pbar.set_description('[Val]')

    for step, batch in enumerate(pbar, start=1):
        bx = batch['x'].to(device, non_blocking=True)
        by = batch['y'].to(device, non_blocking=True)
        bs = batch['sc'].to(device, non_blocking=True)
        bp = batch.get('pathology_score', None)
        if bp is not None: bp = bp.to(device, non_blocking=True)
        bm = batch.get('pred_mask', None)   # 仅 --variable_cutoff 时存在
        if bm is not None: bm = bm.to(device, non_blocking=True)

        outputs, aux_info = model(bx, bs, bp)
        total_loss, loss_stats = criterion(outputs, by, aux_info=aux_info, mask=bm)
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
        inv_reg, inv_stats = compute_inversion_loss(
            aux_info, args.inversion_weight, device)
        total_loss = total_loss + inv_reg
        if bm is None:
            mae = torch.abs(outputs - by).mean()
        else:
            w = bm.to(outputs.dtype).reshape(
                bm.shape[0], 1, bm.shape[1], 1).expand_as(outputs)
            mae = (torch.abs(outputs - by) * w).sum() / w.sum().clamp_min(1e-8)
        val_stats = {}
        if amp_loss_fn is not None:
            amp_val = amp_loss_fn(by, aux_info, mask=bm)
            if amp_val is not None:
                val_stats['head_amp_consistency'] = amp_val.detach()
        for k, v in {**loss_stats, **moe_stats, **graph_stats, **inv_stats,
                     **val_stats, 'metric_mae': mae.detach(),
                     'metric_total': total_loss.detach()}.items():
            meters[k].append(v.detach())
        live['loss_pcc'] += float(loss_stats['loss_pcc'].item())
        live['pcc']      += float(loss_stats['pcc'].item())
        live['metric_mae'] += float(mae.item())
        pbar.set_postfix({'PCC_Loss': f"{live['loss_pcc']/step:.4f}",
                          'PCC': f"{live['pcc']/step:.4f}",
                          'MAE': f"{live['metric_mae']/step:.4f}"})

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

    print('=' * 50)
    if args.mode == 'pretrain':
        print(f'[Pretrain] 正在启动预训练 | 预测未来窗口数: {args.pred_window}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 1] Healthy Control (HC) physics backbone pretraining')
    else:
        print(f'[Finetune] Starting finetuning | Pred windows: {args.pred_window}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 2] MDD individualized pathology expert finetuning [NeuroTwinMoE]')
    print('=' * 50)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.checkpoint_dir, 'runs', f"{args.name}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志已开启，保存路径: {log_dir}")

    dl = NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode, batch_size=args.batch_size,
        in_window=args.in_window, pred_window=args.pred_window,
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
        eval_batch_size=args.eval_batch_size, cache_subjects=args.cache_subjects,
        variable_cutoff=args.variable_cutoff,
        refresh_split_manifest=getattr(args, 'refresh_split_manifest', False),
    )
    train_data, val_data = dl.get_train(), dl.get_val()

    model = NeuroTwin(
        features=args.num_rois, in_window=args.in_window, in_seq_len=args.seq_len,
        pred_window=args.pred_window, pred_seq_len=args.seq_len,
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
        ode_window_attn=args.ode_window_attn,
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
        head_scale_granularity=args.head_scale_granularity,
        head_use_history_proj=args.head_use_history_proj,
        head_use_latent_proj=args.head_use_latent_proj,
        head_use_temporal=args.head_use_temporal,
        head_use_cross_roi=args.head_use_cross_roi,
        head_use_win_attn=args.head_use_win_attn,
        head_use_revin_stats=args.head_use_revin_stats,
        recur_mode=args.recur_mode,
        # Phase 7：概率输出头 + 辅助反演头（inversion_weight=0 时不构建）
        pred_head=args.pred_head,
        pred_quantiles=parse_pred_quantiles(args.pred_quantiles),
        inversion_weight=args.inversion_weight,
        inversion_hidden_dim=args.inversion_hidden_dim,
    ).to(device)

    if args.mode == 'finetune':
        if not os.path.exists(args.pretrained_weight):
            raise FileNotFoundError(f"未找到预训练权重: {args.pretrained_weight}")
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
    print(f"[Phase6] variable_cutoff={args.variable_cutoff} | diff_mode={args.loss_diff_mode} "
          f"| recur_mode={args.recur_mode}"
          + (f" (scheduled_sampling {args.scheduled_sampling_start}→{args.scheduled_sampling_end},"
             f" 前向成本约 ×{args.pred_window})" if args.recur_mode != 'none' else ""))
    print(f"[Phase7] pred_head={args.pred_head}"
          + (f"(quantiles={parse_pred_quantiles(args.pred_quantiles)})"
             if args.pred_head == 'quantile' else "")
          + (f" | init_log_var_nll={args.init_log_var_nll}"
             if args.pred_head != 'point' else "")
          + f" | inversion_weight={args.inversion_weight}"
          + (f"(hidden={args.inversion_hidden_dim})" if args.inversion_weight > 0 else "")
          + f" | intervention_mode={args.intervention_mode}")

    criterion = UncertaintyWeightedHybridLoss(
        init_log_var_pcc=args.init_log_var_pcc, init_log_var_mae=args.init_log_var_mae,
        init_log_var_diff=args.init_log_var_diff, init_log_var_std=args.init_log_var_std,
        clamp_log_vars=args.clamp_log_vars, log_var_min=args.log_var_min,
        log_var_max=args.log_var_max, diff_mode=args.loss_diff_mode,
        init_log_var_nll=args.init_log_var_nll,
    ).to(device)
    # 轮间（deep supervision）损失：仅当权重>0 时构建，否则 compute_intermediate_supervision 返回零
    inter_sup = (IntermediateSupervisionLoss(decay=args.refiner_inter_sup_decay).to(device)
                 if args.refiner_inter_sup_weight > 0 else None)
    # 幅值一致性损失：仅在 scale_mod_trend 且权重>0 时构建（否则该头无显式幅值参数）
    amp_loss_fn = (AmplitudeConsistencyLoss(
        granularity=args.head_scale_granularity).to(device)
        if (args.head_amp_consistency_weight > 0
            and args.head_amp_mode == 'scale_mod_trend') else None)

    optimizer = build_optimizer(model, criterion, args)
    scheduler = build_scheduler(optimizer, args)
    scaler    = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    ema       = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if args.compile and hasattr(torch, 'compile'):
        # 在 EMA 深拷贝之后包装，避免 OptimizedModule 进入 EMA；训练前向被融合编译
        model = torch.compile(model)
        print('torch.compile 已启用（实验性）')

    best_pcc_loss    = float('inf')
    patience_counter = 0
    backbone_unfrozen = (args.mode == 'pretrain')

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

        # scheduled sampling 概率线性调度（仅 --recur_mode terminal_state 生效）
        if args.recur_mode != 'none' and hasattr(model, 'set_scheduled_sampling_prob'):
            ratio = min(1.0, float(epoch) / max(1, args.train_epochs - 1))
            ss_prob = (args.scheduled_sampling_start
                       + (args.scheduled_sampling_end - args.scheduled_sampling_start) * ratio)
            model.set_scheduled_sampling_prob(ss_prob)

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
        train_meters = defaultdict(list)    # tensor 累积，epoch 末一次性同步
        ode_meters = defaultdict(list)      # ODE 审计量（纯 float，不参与反传）
        live = defaultdict(float)           # 进度条所需的少数关键指标
        pbar = tqdm(train_data, total=len(train_data), dynamic_ncols=True, leave=False)
        pbar.set_description(f"Epoch {epoch+1:03d}/{args.train_epochs:03d} [Train]")

        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            bx = batch['x'].to(device, non_blocking=True)
            by = batch['y'].to(device, non_blocking=True)
            bs = batch['sc'].to(device, non_blocking=True)
            bp = batch.get('pathology_score', None)
            if bp is not None: bp = bp.to(device, non_blocking=True)
            bm = batch.get('pred_mask', None)   # 仅 --variable_cutoff 时存在
            if bm is not None: bm = bm.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # recur_mode=terminal_state 时把未来真值窗交给模型做 scheduled sampling
                if args.recur_mode == 'none':
                    outputs, aux_info = model(bx, bs, bp)
                else:
                    outputs, aux_info = model(bx, bs, bp, future=by)
                total_loss, loss_stats = criterion(
                    outputs, by, aux_info=aux_info, mask=bm)
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
                inter_reg, inter_stats = compute_intermediate_supervision(
                    aux_info, by, inter_sup, args.refiner_inter_sup_weight, device,
                    mask=bm)
                total_loss = total_loss + inter_reg
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
            for k, v in {**loss_stats, **moe_stats, **graph_stats, **inter_stats,
                         **inv_stats, **amp_stats,
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
        val_metrics = evaluate(eval_model, val_data, criterion, device, args)
        scheduler.step()

        train_metrics = {k: torch.stack(vs).mean().item() for k, vs in train_meters.items()}
        ode_metrics = {k: sum(vs) / max(1, len(vs)) for k, vs in ode_meters.items()}
        group_lrs    = {g.get('name', f'g{i}'): g['lr']
                        for i, g in enumerate(optimizer.param_groups)}
        lr_text = ' | '.join(f"{k}: {v:.2e}" for k, v in group_lrs.items())

        tqdm.write(
            f"Epoch {epoch+1:03d} | {lr_text} | "
            f"Train Total: {train_metrics['metric_total']:.4f} | "
            f"Train PCC Loss: {train_metrics['loss_pcc']:.4f} | "
            f"Val Total: {val_metrics['metric_total']:.4f} | "
            f"Val PCC Loss: {val_metrics['loss_pcc']:.4f} | "
            f"Val PCC: {val_metrics['pcc']:.4f} | "
            f"Val MAE: {val_metrics['metric_mae']:.4f} | "
            f"logvar[pcc/mae/diff/std/nll]="
            f"{train_metrics['log_var_pcc']:.2f}/"
            f"{train_metrics['log_var_mae']:.2f}/"
            f"{train_metrics['log_var_diff']:.2f}/"
            f"{train_metrics['log_var_std']:.2f}/"
            f"{train_metrics['log_var_nll']:.2f} | "
            f"NLL: {train_metrics['loss_nll']:.4f}"
        )
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

        is_best = val_metrics['loss_pcc'] < best_pcc_loss
        if is_best:
            best_pcc_loss = val_metrics['loss_pcc']
            best_name = 'base_best.pt' if args.mode == 'pretrain' else 'finetuned_best.pt'
            save_backbone_weights(os.path.join(save_dir, best_name), eval_model,
                                  meta={'args': vars(args)})
            tqdm.write(f"New Best Model Saved! (Best Val PCC Loss: {best_pcc_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly Stopping! 连续 {args.patience} epoch 未改善。"
                           f"\n最佳 Val PCC Loss: {best_pcc_loss:.4f}")
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
        for tag, val in [
            ('Train/Total', train_metrics['metric_total']),
            ('Train/PCC_Loss', train_metrics['loss_pcc']),
            ('Train/PCC', train_metrics['pcc']),
            ('Val/Total', val_metrics['metric_total']),
            ('Val/PCC_Loss', val_metrics['loss_pcc']),
            ('Val/PCC', val_metrics['pcc']),
            ('Val/MAE', val_metrics['metric_mae']),
            ('Train/NLL', train_metrics['loss_nll']),
            ('Val/NLL', val_metrics['loss_nll']),
        ]: writer.add_scalar(tag, val, epoch)
        if 'inversion_mse' in train_metrics:
            writer.add_scalar('Train/Inversion_MSE', train_metrics['inversion_mse'], epoch)
            if 'inversion_mse' in val_metrics:
                writer.add_scalar('Val/Inversion_MSE', val_metrics['inversion_mse'], epoch)
        for lv in ('pcc', 'mae', 'diff', 'std', 'nll'):
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
    p.add_argument('--in_window', type=int, default=6)
    p.add_argument('--pred_window', type=int, default=1)
    p.add_argument('--total_windows', type=int, default=9)
    p.add_argument('--n_block', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--norm', type=str2bool, default=True)
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
    p.add_argument('--init_log_var_pcc', type=float, default=0.0)
    p.add_argument('--init_log_var_mae', type=float, default=-1.5)
    p.add_argument('--init_log_var_diff', type=float, default=-2.0)
    p.add_argument('--init_log_var_std', type=float, default=-2.0)
    p.add_argument('--loss_lr_scale', type=float, default=1.0)
    p.add_argument('--clamp_log_vars', type=str2bool, default=True)
    p.add_argument('--log_var_min', type=float, default=-6.0)
    p.add_argument('--log_var_max', type=float, default=6.0)

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
    p.add_argument('--cache_subjects', type=str2bool, default=False,
                   help='按被试缓存加载数据（实测负优化：page cache 下无收益且增加内存，保留开关备用）')

    # ---- Phase 6：长程与状态递推 ----
    p.add_argument('--variable_cutoff', type=str2bool, default=False,
                   help='可变截断：允许未来窗不足 pred_window 的样本入训（提升每被试样本数），'
                        '不足窗以零占位并由 pred_mask 在损失端屏蔽。默认关闭以保持既有样本口径。')
    p.add_argument('--loss_diff_mode', default='per_window',
                   choices=['flatten', 'per_window'],
                   help='一阶差分损失口径：per_window=保留窗口维度沿时间轴 diff（默认，'
                        '避免跨窗边界伪差分）；flatten=旧行为（沿 W×S 展平 diff）。')
    p.add_argument('--recur_mode', default='none', choices=['none', 'terminal_state'],
                   help='窗口间状态递推：none=单次前向并行预测（默认）；'
                        'terminal_state=自回归滚动 pred_window 次（成本约 ×pred_window，高风险）。')
    p.add_argument('--scheduled_sampling_start', type=float, default=0.0,
                   help='recur_mode=terminal_state 时 scheduled sampling 概率起点（0=纯自回归）。')
    p.add_argument('--scheduled_sampling_end', type=float, default=0.0,
                   help='recur_mode=terminal_state 时 scheduled sampling 概率终点（线性调度）。')
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

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
