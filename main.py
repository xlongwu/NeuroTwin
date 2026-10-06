# coding=utf-8
"""NeuroTwin-TFM 训练入口。

训练基础设施（损失/图正则/优化器/EMA）位于 train/ 包，
通用工具（set_seed/str2bool）位于 utils/common.py；
本文件保留 main 训练循环与 CLI 定义。
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
from utils.common import (parse_int_list, resolve_forecast_offsets,
                          resolve_task_dims, set_seed, str2bool)
from models.tfm import NeuroTwinTFM
from train.losses import TFMDualLoss, compute_graph_regularization, spatial_pcc
from train.optim import (ModelEMA, build_optimizer, build_scheduler,
                         check_pretrain_config_compat, count_trainable_params,
                         load_backbone_only, load_backbone_weights,
                         save_backbone_weights, set_finetune_stage)


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
    context 末位 x_t，target_mask [B,H] 标记各预测偏移是否有真值。
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


def parse_pathology_fields(fields):
    """把逗号分隔的字段字符串解析为列表；空/None 返回 None（由 Dataset 使用默认）。"""
    if fields is None:
        return None
    if isinstance(fields, (list, tuple)):
        parsed = [str(f).strip() for f in fields if str(f).strip()]
        return parsed or None
    parsed = [f.strip() for f in str(fields).split(',') if f.strip()]
    return parsed or None


@torch.no_grad()
def evaluate_tfm(model, data_loader, criterion, device, args,
                 rollout_steps=0, rollout_horizons=None):
    """TFM 的验证/测试循环：one-step 主指标 + CPM 逐 horizon。

    指标键（``metric_mae_next`` 等）供 checkpoint 判据与日志复用，另加：
      - ``cpm_mae_H{h}``：CPM 头在 t+h 处的逐元素 MAE（元素数归一）；
      - ``cpm_persistence_mae_H{h}``：persistence 基线在同一 horizon 的 MAE；
      - ``metric_cpm_mae``：CPM 全 horizon 平均 MAE。
    """
    model.eval()
    meters = defaultdict(list)
    rollout_acc = defaultdict(lambda: [0.0, 0, 0])
    cpm_acc = defaultdict(lambda: [0.0, 0, 0])       # h -> [Σ|err|, Σw, n_elem]
    rollout_hs = [int(h) for h in (rollout_horizons or []) if int(h) <= int(rollout_steps)]
    horizon = int(getattr(model, 'cpm_horizon', 0) or args.cpm_horizon)
    pbar = tqdm(data_loader, total=len(data_loader), dynamic_ncols=True, leave=False)
    pbar.set_description('[Val-TFM]')

    for batch in pbar:
        bx, by, bl, bm, bs, bp = next_point_batch(batch, device)
        fut = batch['future'].to(device, non_blocking=True)          # [B,F,R]
        fmask = batch['future_mask'].to(device, non_blocking=True)   # [B,R]
        outputs, aux_info = model(bx, bs, bp)                        # [B,F,1,1]
        cpm_pred = aux_info['cpm_pred']                              # [B,F,H]
        cpm_target = fut[:, :, :horizon]                             # [B,F,H]
        cpm_mask = fmask[:, :horizon]                                # [B,H]
        total_loss, loss_stats = criterion(outputs, by, cpm_pred,
                                           cpm_target, cpm_mask)
        graph_reg, graph_stats = compute_graph_regularization(
            aux_info, device,
            sparsity_weight=args.sc_sparsity_weight,
            entropy_weight=args.sc_entropy_weight,
            temporal_weight=args.sc_temporal_weight,
        )
        total_loss = total_loss + graph_reg

        b, f, h, _ = outputs.shape
        if bm is None:
            bm = torch.ones(b, h, device=outputs.device, dtype=outputs.dtype)
        w4 = bm.view(b, 1, h, 1)
        mae = _masked_l1_weighted(outputs, by, bm)
        rmse = (((outputs - by) ** 2) * w4.expand_as(outputs)).sum() / \
            (w4.expand_as(outputs).sum().clamp_min(1e-8))
        rmse = rmse.clamp_min(0).sqrt()
        pcc_spatial = ((spatial_pcc(outputs, by) * bm).sum()
                       / bm.sum().clamp_min(1e-8))
        x_t = bx[:, :, 0, -1:].unsqueeze(-1)                         # [B,F,1,1]
        x_prev = bx[:, :, 0, -2:-1].unsqueeze(-1)
        pers = x_t.expand(b, f, h, 1)
        trend = (x_t + (x_t - x_prev)).expand(b, f, h, 1)
        pers_mae = _masked_l1_weighted(pers, by, bm)
        trend_mae = _masked_l1_weighted(trend, by, bm)

        # CPM 逐 horizon MAE（元素数归一，与 next-state MAE 同口径）
        cpm_mae_sum = 0.0
        cpm_mae_weight = 0.0
        for hi in range(horizon):
            mh = cpm_mask[:, hi].unsqueeze(1)                        # [B,1] 广播到 ROI
            n = float(mh.sum().item())
            if n <= 0:
                continue
            err = (cpm_pred[:, :, hi] - cpm_target[:, :, hi]).abs()  # [B,F]
            wsum = float((err * mh).sum().item())
            cpm_acc[hi + 1][0] += wsum
            cpm_acc[hi + 1][1] += n
            cpm_acc[hi + 1][2] += n * f
            cpm_mae_sum += wsum
            cpm_mae_weight += n * f
            # persistence 基线在同一 horizon 的 MAE（x̂=x_t）
            pers_err = (x_t[:, :, 0, 0] - cpm_target[:, :, hi]).abs()
            cpm_acc.setdefault(f'pers_{hi + 1}', [0.0, 0, 0])
            cpm_acc[f'pers_{hi + 1}'][0] += float((pers_err * mh).sum().item())
            cpm_acc[f'pers_{hi + 1}'][1] += n
            cpm_acc[f'pers_{hi + 1}'][2] += n * f
        metric_cpm_mae = (cpm_mae_sum / cpm_mae_weight) if cpm_mae_weight > 0 else 0.0

        for k, v in {**loss_stats, **graph_stats,
                     'metric_total': total_loss.detach(),
                     'metric_mae_next': mae.detach(),
                     'metric_rmse_next': rmse.detach(),
                     'metric_pcc_next': pcc_spatial.detach(),
                     'metric_persistence_mae': pers_mae.detach(),
                     'metric_trend_mae': trend_mae.detach(),
                     'metric_cpm_mae': torch.tensor(metric_cpm_mae)}.items():
            meters[k].append(v.detach())
        pbar.set_postfix({'MAE': f"{float(mae):.4f}",
                          'PCC_sp': f"{float(pcc_spatial):.4f}",
                          'CPM': f"{metric_cpm_mae:.4f}"})

        # ---- free rollout（one-step 头自回归；中间不使用真值）----
        if rollout_hs and 'rollout_flag' in batch:
            rf = batch['rollout_flag'].bool()
            rf_gpu = rf.to(device)
            if rf_gpu.any():
                hist = bx[rf_gpu][:, :, 0, :].transpose(1, 2).contiguous()   # [n,K,F]
                roll = model.rollout_next_states(
                    hist, bs[rf_gpu], (bp[rf_gpu] if bp is not None else None),
                    steps=int(rollout_steps))                        # [n,R,F]
                fut_r = batch['future'][rf].transpose(1, 2).contiguous().to(
                    device).float()                                  # [n,R,F]
                fmask_r = batch['future_mask'][rf].to(device).float()  # [n,R]
                for rh in rollout_hs:
                    ok = fmask_r[:, rh - 1] > 0
                    k = int(ok.sum())
                    if k:
                        rollout_acc[rh][0] += float(
                            (roll[ok, rh - 1, :] - fut_r[ok, rh - 1, :]).abs().sum())
                        rollout_acc[rh][1] += k
                        rollout_acc[rh][2] += k * int(roll.shape[-1])

    for hi, (err_sum, _n_tasks, n_elem) in cpm_acc.items():
        if isinstance(hi, int) and n_elem:
            meters[f'cpm_mae_H{hi}'].append(torch.tensor(err_sum / n_elem))
    for key, (err_sum, _n_tasks, n_elem) in cpm_acc.items():
        if isinstance(key, str) and key.startswith('pers_') and n_elem:
            meters[f'cpm_persistence_mae_H{key.split("_")[1]}'].append(
                torch.tensor(err_sum / n_elem))
    for rh, (err_sum, _n_tasks, n_elem) in rollout_acc.items():
        if n_elem:
            meters[f'rollout_mae_H{rh}'].append(torch.tensor(err_sum / n_elem))
    return {k: torch.stack(vs).mean().item() for k, vs in meters.items()}


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
    # checkpoint config.args 快照保留 model_arch 键：evaluate_variant 重建与
    # pretrain 签名校验依赖它（TFM 已是唯一主干，CLI 不再暴露该开关）
    args.model_arch = 'tfm'
    if getattr(args, 'device', ''):
        device = torch.device(args.device)
    else:
        # 缺省维持历史行为（多卡机器上训练集中到 cuda:1）；可用 --device 覆盖
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    amp_enabled = (device.type == 'cuda' and args.amp)
    torch.set_num_threads(args.num_threads)
    if device.type == 'cuda' and args.tf32:
        # 输入形状固定，启用 cudnn 自动调优；TF32 加速 autocast 覆盖不到的 fp32 matmul
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    # 任务维度解析：next_timepoint（in_window=1, pred_window=1, seq_len=context_max）。
    # 模型/数据/损失共用同一口径。
    dims = resolve_task_dims(args)
    forecast_offsets = resolve_forecast_offsets(args)
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
              f'model_arch=tfm | CPM horizon: {dims["cpm_horizon"]}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 1] Healthy Control (HC) physics backbone pretraining')
    else:
        print(f'[Finetune] Starting finetuning | task_mode={dims["task_mode"]} | '
              f'model_arch=tfm | CPM horizon: {dims["cpm_horizon"]}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 2] MDD individualized pathology finetuning '
              '[NeuroTwinTFM FiLM]')
    print('=' * 50)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.checkpoint_dir, 'runs', f"{args.name}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志已开启，保存路径: {log_dir}")

    print(f"[next_timepoint] 模型维度映射: in_window←1（整段 context 作为单窗口） | "
          f"pred_window←len(forecast_offsets)={dims['pred_window']} | "
          f"seq_len←context_max={dims['seq_len']} | "
          f"forecast_offsets={forecast_offsets} | "
          f"causal_training={args.causal_training}")
    print(f"[next_timepoint] 训练 context: K∈[--context_min, --context_max]="
          f"[{args.context_min}, {args.context_max}]"
          + (f"（离散 {parse_int_list(args.context_lengths, 'context_lengths')}）"
             if args.context_lengths else "（连续区间随机采样）")
          + f" | 评估固定 K_eval={int(args.eval_context_length) or args.context_max} | "
          f"anchors/被试={args.eval_anchors_per_subject} | "
          f"rollout: R={eval_rollout_steps}, horizons={eval_rollout_horizons}, "
          f"锚点/被试={args.eval_rollout_tasks_per_subject}")

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
        train_rollout_steps=dims['cpm_horizon'],
        train_anchor_max_offset=dims['cpm_horizon'],
        eval_all_future_steps=dims['cpm_horizon'],
    )
    train_data, val_data = dl.get_train(), dl.get_val()

    model = NeuroTwinTFM(
        features=args.num_rois, context_max=dims['seq_len'],
        patch_len=args.tfm_patch_len, dim=args.tfm_dim,
        num_layers=args.tfm_layers, num_heads=args.tfm_heads,
        ff_ratio=args.tfm_ff_ratio, dropout=args.dropout,
        sc_prior_mode=args.sc_prior_mode,
        sc_lambda_mode=args.sc_lambda_mode,
        sc_lambda_init=args.sc_lambda_init,
        sc_prior_rank=args.sc_prior_rank,
        sc_delta_a=args.sc_delta_a, sc_delta_rank=args.sc_delta_rank,
        sc_sinkhorn_iters=args.sc_sinkhorn_iters,
        one_step_hidden_dim=args.tfm_one_step_hidden_dim,
        cpm_horizon=args.cpm_horizon, cpm_layers=args.cpm_layers,
        cpm_intervention=args.cpm_intervention,
        pretrain_mode=(args.mode == 'pretrain'),
        pathology_input_dim=args.pathology_input_dim,
        pathology_norm_mode=args.pathology_norm_mode,
        pathology_norm_quantiles=args.pathology_norm_quantiles,
        pathology_norm_rbf_knots=args.pathology_norm_rbf_knots,
        use_revin=args.norm,
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
        print('已进入分阶段微调: 前期仅训练病理 FiLM 条件适配器，后期逐步解冻主干。')

    if args.mode == 'finetune':
        fit_pathology_normalizer(model, dl, args)

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Parameters: {count_trainable_params(model)}")
    print(f"[TFM] dim={args.tfm_dim} layers={args.tfm_layers} heads={args.tfm_heads} "
          f"patch_len={args.tfm_patch_len} ff_ratio={args.tfm_ff_ratio} | "
          f"one_step_hidden={args.tfm_one_step_hidden_dim} | "
          f"cpm_horizon={args.cpm_horizon} cpm_layers={args.cpm_layers} "
          f"cpm_intervention={args.cpm_intervention} | revin={args.norm}")
    print(f"[SC prior] mode={args.sc_prior_mode} "
          f"| lambda_mode={args.sc_lambda_mode}(init={args.sc_lambda_init}) "
          f"| rank={args.sc_prior_rank} | delta_a={args.sc_delta_a} "
          f"| sinkhorn_iters={args.sc_sinkhorn_iters}")

    # TFM 双目标损失（方案 §20）：L_one = Huber + λ_pcc(1-PCC)；
    # L_CPM = γ 加权逐 horizon Huber。不再使用数学等价的 abs+delta 双计。
    criterion = TFMDualLoss(
        lambda_one=args.lambda_one, lambda_cpm=args.lambda_cpm,
        lambda_pcc=args.lambda_pcc, cpm_gamma=args.cpm_gamma,
        huber_delta=args.cpm_huber_delta,
    ).to(device)
    print(f"[TFM] 损失: lambda_one={args.lambda_one} "
          f"lambda_cpm={args.lambda_cpm} lambda_pcc={args.lambda_pcc} "
          f"cpm_gamma={args.cpm_gamma} huber_delta={args.cpm_huber_delta} | "
          f"cpm_horizon={args.cpm_horizon}（One-Step + CPM 双头，方案 §9/§20）")

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
                # CPM 目标：训练视图 future_steps=cpm_horizon → by_roll [B,F,H]
                total_loss, loss_stats = criterion(
                    outputs, by, aux_info['cpm_pred'], by_roll, bm_roll)
                graph_reg, graph_stats = compute_graph_regularization(
                    aux_info, device,
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
                max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if ema is not None: ema.update(model)

            for k, v in {**loss_stats, **graph_stats,
                         'metric_total': total_loss.detach()}.items():
                train_meters[k].append(v.detach())
            live['metric_total'] += float(total_loss.item())
            live['loss_one'] += float(loss_stats['loss_one'].item())
            live['loss_cpm'] += float(loss_stats['loss_cpm'].item())
            pbar.set_postfix({
                'Loss': f"{live['metric_total']/step:.4f}",
                'One': f"{live['loss_one']/step:.4f}",
                'CPM': f"{live['loss_cpm']/step:.4f}",
            })

        eval_model  = ema.ema if ema is not None else model
        val_metrics = evaluate_tfm(eval_model, val_data, criterion, device,
                                   args, rollout_steps=eval_rollout_steps,
                                   rollout_horizons=eval_rollout_horizons)
        scheduler.step()

        train_metrics = {k: torch.stack(vs).mean().item() for k, vs in train_meters.items()}
        group_lrs    = {g.get('name', f'g{i}'): g['lr']
                        for i, g in enumerate(optimizer.param_groups)}
        lr_text = ' | '.join(f"{k}: {v:.2e}" for k, v in group_lrs.items())

        # TFM 口径：one-step 主指标 + CPM 全 horizon 指标 + trivial baseline
        tqdm.write(
            f"Epoch {epoch+1:03d} | {lr_text} | "
            f"Train Total: {train_metrics['metric_total']:.4f} | "
            f"one: {train_metrics['loss_one']:.4f} | "
            f"cpm: {train_metrics['loss_cpm']:.4f} | "
            f"Val Total: {val_metrics['metric_total']:.4f} | "
            f"Val next MAE: {val_metrics['metric_mae_next']:.4f} | "
            f"Val next PCC(sp): {val_metrics['metric_pcc_next']:.4f} | "
            f"Val CPM MAE: {val_metrics['metric_cpm_mae']:.4f} | "
            f"Val persistence MAE: {val_metrics['metric_persistence_mae']:.4f} | "
            f"Val trend MAE: {val_metrics['metric_trend_mae']:.4f}"
        )
        cpm_h_log = [f"H{h}: {val_metrics[f'cpm_mae_H{h}']:.4f}"
                     for h in range(1, int(args.cpm_horizon) + 1)
                     if f'cpm_mae_H{h}' in val_metrics]
        if cpm_h_log:
            tqdm.write("[TFM/CPM] val MAE per horizon | " + ' | '.join(cpm_h_log))
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
            ('Train/One', train_metrics['loss_one']),
            ('Train/CPM', train_metrics['loss_cpm']),
            ('Val/Total', val_metrics['metric_total']),
            ('Val/Next_MAE', val_metrics['metric_mae_next']),
            ('Val/Next_RMSE', val_metrics['metric_rmse_next']),
            ('Val/Next_PCC', val_metrics['metric_pcc_next']),
            ('Val/CPM_MAE', val_metrics['metric_cpm_mae']),
            ('Val/Persistence_MAE', val_metrics['metric_persistence_mae']),
            ('Val/Trend_MAE', val_metrics['metric_trend_mae']),
        ]
        for h, key in sorted((h, f'rollout_mae_H{h}')
                             for h in (eval_rollout_horizons or [])):
            if key in val_metrics:
                tb_scalars.append((f'Val/Rollout_MAE_H{h}', val_metrics[key]))
        for tag, val in tb_scalars:
            writer.add_scalar(tag, val, epoch)
        for name, lr in group_lrs.items():
            writer.add_scalar(f'LR/{name}', lr, epoch)
        for tag, key in [('Train/Graph_Sparsity', 'graph_sparsity'),
                          ('Train/Graph_Entropy', 'graph_entropy'),
                          ('Train/Graph_Temporal', 'graph_temporal')]:
            if key in train_metrics:
                writer.add_scalar(tag, train_metrics[key], epoch)

    writer.close()
    best_n = 'base_best.pt' if args.mode == 'pretrain' else 'finetuned_best.pt'
    best_path = os.path.join(save_dir, best_n)
    print(f"\n训练完成！最佳权重已保存至: {best_path}")

    if args.mode == 'finetune' and getattr(args, 'eval_after_train', True) \
            and os.path.exists(best_path):
        # 训练结束自动评 test（闭环）：复用版本化切分 manifest。评估失败不影响训练产物。
        from experiments.evaluate_variant import evaluate as _eval_variant
        eval_dir = os.path.join(save_dir, 'eval')
        try:
            summary = _eval_variant(best_path, eval_dir, eval_splits=('test',),
                                    seed=args.seed)
            print(f"[post-train eval] test_PCC={summary.get('test_pcc')} "
                  f"test_MAE={summary.get('test_mae')}")
            print(f"[post-train eval] 详细指标见: {eval_dir}/metrics_test.json")
        except Exception as e:
            print(f"[post-train eval] 自动评估失败（训练产物不受影响）: {type(e).__name__}: {e}\n"
                  f"可手动复评: python -m experiments.evaluate_variant --ckpt {best_path} --out_dir {eval_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description='NeuroTwin-TFM: Pathology-Conditioned TimesFM-style Foundation '
                    'Model for Brain Dynamics')
    p.add_argument('--mode', default='pretrain', choices=['pretrain', 'finetune'])
    p.add_argument('--device', default='',
                   help='训练设备（如 cuda:0）。空 = 历史默认（有 GPU 时用 cuda:1）。'
                        '用于多卡并行跑消融：不同执行器指定不同 --device。')
    p.add_argument('--pretrained_weight', default='./checkpoints/neurotwin_pretrain/base_best.pt')
    p.add_argument('--pathology_input_dim', type=int, default=1)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--name', default='neurotwin')
    p.add_argument('--clinical_file', default='Rest-meta-MDD-V1V2-Merged-MDD.xlsx')
    p.add_argument('--num_rois', type=int, default=116)
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--total_windows', type=int, default=9)
    p.add_argument('--norm', type=str2bool, default=False)  # 数据已 z-score，默认关闭 BrainRevIN
    p.add_argument('--dropout', type=float, default=0.2)
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

    # ---------- 协议基础（检查点版本化 / 病理归一化） ----------
    p.add_argument('--pretrained_arch_policy', default='require_match',
                   choices=['require_match', 'warn', 'ignore'],
                   help='预训练权重架构版本不匹配时的策略：require_match 直接报错；'
                        'warn 打印并跳过不兼容键；ignore 静默跳过。')
    p.add_argument('--pretrained_skip_pattern', default='',
                   help='fnmatch 模式，命中的预训练键不加载（保留随机初始化），'
                         '用于受控消融对照，如 one_step_head.*')
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
    p.add_argument('--pathology_fields', default=None,
                   help='逗号分隔的病理字段列表，默认 None 时使用 HAMD 标量。')
    p.add_argument('--pathology_missing', default='drop',
                   choices=['drop', 'zero', 'mean'],
                   help='病理字段缺失处理策略。')
    p.add_argument('--reuse_pretrained_norm_stats', type=str2bool, default=True,
                   help='加载的检查点已带归一化统计量时是否复用（否则用 train subjects 重新拟合）。')
    p.add_argument('--adapter_lr_scale', type=float, default=1.0,
                   help='条件适配器（adapter）参数组的学习率缩放。')

    # ---------- SC 软先验（soft anatomical prior / 图正则） ----------
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
    p.add_argument('--sc_sinkhorn_iters', type=int, default=64,
                   help='对称 Sinkhorn 迭代次数（使 A_eff 同时行归一且对称；靠近均匀的图收敛慢）。')
    p.add_argument('--sc_sparsity_weight', type=float, default=1e-3,
                   help='SC 软先验稀疏正则权重（A_eff 非对角 L1），<=0 关闭。')
    p.add_argument('--sc_entropy_weight', type=float, default=1e-3,
                   help='SC 软先验行熵正则权重（抑制模糊均匀图），<=0 关闭。')
    p.add_argument('--sc_temporal_weight', type=float, default=0.0,
                   help='逐窗功能图时间一致性正则权重（>0 时才生成 A_seq），默认关闭。')

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
    p.add_argument('--compile', type=str2bool, default=True,
                   help='torch.compile 编译模型前向（稳态约 35%% 提速；首次编译有额外耗时）')

    # ---------- NeuroTwin-TFM（TimesFM-3 风格主干，方案 §30 V1） ----------
    p.add_argument('--tfm_dim', type=int, default=256,
                   help='TFM 隐藏维度 D（方案 §17 推荐 128/192/256）。')
    p.add_argument('--tfm_layers', type=int, default=4,
                   help='TFM Encoder Block 层数 N（方案 §17 推荐 4/6）。')
    p.add_argument('--tfm_heads', type=int, default=4,
                   help='TFM temporal/ROI attention 的头数。')
    p.add_argument('--tfm_patch_len', type=int, default=8,
                   help='TFM temporal patch 长度 p（G30_TFM 消融：p=8 最优，'
                        '详见 docs/Version3_docs_1003/G30_TFM消融实验总结_20261005.md）。'
                        'K 不是 p 的整数倍时开头补零对齐。')
    p.add_argument('--tfm_ff_ratio', type=int, default=4,
                   help='TFM FFN 隐层扩张倍数（ff_dim = ff_ratio × dim）。')
    p.add_argument('--tfm_one_step_hidden_dim', type=int, default=256,
                   help='TFM One-Step 状态转移头的 MLP 隐层宽度。')
    p.add_argument('--cpm_horizon', type=int, default=8,
                   help='CPM 全 horizon 预测步数 H（方案 §8：推荐 {4,8,16}，'
                        '首选 8）。')
    p.add_argument('--cpm_layers', type=int, default=2,
                   help='CPM ROI×Horizon 查询解码器的交叉注意力层数。')
    p.add_argument('--cpm_intervention', type=str2bool, default=False,
                   help='构建 future-known covariate（intervention lookahead）嵌入'
                        '接口（方案 §12.4/§13）。V1 默认关闭：无真实干预数据，'
                        '仅保留接口，不作为治疗预测任务训练（方案 §13.1）。')
    p.add_argument('--lambda_one', type=float, default=1.0,
                   help='TFM 损失：one-step Huber 项权重（方案 §20.1）。')
    p.add_argument('--lambda_cpm', type=float, default=1.0,
                   help='TFM 损失：CPM 轨迹项权重（0 可退化为纯 one-step 的 '
                        'Stage 0 基线，方案 §24）。')
    p.add_argument('--lambda_pcc', type=float, default=0.1,
                   help='损失：spatial PCC 惩罚项权重（1 - 逐 ROI 空间相关）。')
    p.add_argument('--cpm_gamma', type=float, default=1.0,
                   help='CPM 逐 horizon 损失权重衰减 γ（w_h=γ^(h-1)；1=均匀，'
                        '方案 §20.2）。须在 (0, 1] 内。')
    p.add_argument('--cpm_huber_delta', type=float, default=1.0,
                   help='TFM Huber 损失的过渡阈值 δ（原始空间 BOLD 量纲，'
                        'z-score 后单位方差取 1.0）。')

    # ---------- 任务口径与 next_timepoint 数据协议 ----------
    p.add_argument('--task_mode', default='next_timepoint',
                   choices=['next_timepoint'],
                   help='任务口径：next_timepoint=连续 BOLD context → 下一 TR 全脑状态'
                        '（Next Brain-State Prediction，context_min..context_max 个 TR → '
                        '未来 TR，默认仅 +1）。')
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

    # ---------- Next-Timepoint Prediction（连续 BOLD → 下一 TR 全脑状态） ----------
    p.add_argument('--context_min', type=int, default=16,
                   help='next_timepoint 训练 context 长度下界 K_min（TR 数）。')
    p.add_argument('--context_max', type=int, default=64,
                   help='next_timepoint context 长度上界 K_max（= 模型 S 轴建模宽度，决定参数量）。'
                        '训练随机采样 K ∈ context_lengths 或 [context_min, context_max]。')
    p.add_argument('--context_lengths', default='',
                   help='可选：离散 context 长度集合（逗号分隔，如 16,32,64）；'
                        '空=在 [context_min, context_max] 区间内均匀随机采样。')
    p.add_argument('--causal_training', default='random_context',
                   choices=['random_context', 'full_sequence'],
                   help='训练范式：random_context=随机 context → 下一时间点（方案 B，已实现，'
                        '未来信息不进入前向）；full_sequence=GPT 式全序列 teacher-forcing'
                        '（方案 A，需要重写主干时间轴算子，当前显式报错未实现）。')
    p.add_argument('--forecast_offsets', default='',
                   help='预测偏移列表（逗号分隔，相对当前 TR 的 +Δ，需严格递增）；'
                        'TFM 的 one-step 头只建模 +1 偏移，留空即 [1]，多步预测用 '
                        '--cpm_horizon 控制 CPM 头。')
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
                   help='finetune 时只加载主干权重（预测头 / 条件模块保持随机初始化），'
                        '并显式打印 loaded/missing/reinitialized 清单。')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
