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
from copy import deepcopy
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
from utils.common import set_seed, str2bool
from models.neurotwin import NeuroTwin
from train.losses import UncertaintyWeightedHybridLoss
from train.optim import (ModelEMA, build_optimizer, build_scheduler,
                         count_trainable_params, load_backbone_weights,
                         set_finetune_stage)
from train.moe import compute_moe_regularization, update_router_temperature


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args):
    model.eval()
    meters = defaultdict(float)
    pbar = tqdm(data_loader, total=len(data_loader), dynamic_ncols=True, leave=False)
    pbar.set_description('[Val]')

    for step, batch in enumerate(pbar, start=1):
        bx = batch['x'].to(device, non_blocking=True)
        by = batch['y'].to(device, non_blocking=True)
        bs = batch['sc'].to(device, non_blocking=True)
        bp = batch.get('pathology_score', None)
        if bp is not None: bp = bp.to(device, non_blocking=True)

        outputs, aux_info = model(bx, bs, bp)
        total_loss, loss_stats = criterion(outputs, by)
        moe_reg, moe_stats = compute_moe_regularization(
            aux_info, device,
            load_balance_weight=args.moe_load_balance_weight,
            entropy_weight=args.moe_entropy_weight,
            z_loss_weight=args.moe_z_loss_weight,
            diversity_weight=args.moe_diversity_weight,
        )
        total_loss = total_loss + moe_reg
        mae = torch.abs(outputs - by).mean()
        for k, v in {**loss_stats, **moe_stats,
                     'metric_mae': mae.detach(),
                     'metric_total': total_loss.detach()}.items():
            meters[k] += float(v.item())
        pbar.set_postfix({'PCC_Loss': f"{meters['loss_pcc']/step:.4f}",
                          'PCC': f"{meters['pcc']/step:.4f}",
                          'MAE': f"{meters['metric_mae']/step:.4f}"})

    return {k: v / max(1, len(data_loader)) for k, v in meters.items()}


def main(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    amp_enabled = (device.type == 'cuda' and args.amp)
    torch.set_num_threads(args.num_threads)
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
        pathology_input_dim=args.pathology_input_dim, clinical_file=args.clinical_file,
        total_windows=args.total_windows, seq_len=args.seq_len,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
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
    ).to(device)

    if args.mode == 'finetune':
        if not os.path.exists(args.pretrained_weight):
            raise FileNotFoundError(f"未找到预训练权重: {args.pretrained_weight}")
        load_backbone_weights(model, args.pretrained_weight, device)
        set_finetune_stage(model, backbone_unfrozen=False)
        print('已进入分阶段微调: 前期仅训练 MoE（MoDE），后期逐步解冻主干。')

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Parameters: {count_trainable_params(model)}")

    criterion = UncertaintyWeightedHybridLoss(
        init_log_var_pcc=args.init_log_var_pcc, init_log_var_mae=args.init_log_var_mae,
        init_log_var_diff=args.init_log_var_diff, init_log_var_std=args.init_log_var_std,
        clamp_log_vars=args.clamp_log_vars, log_var_min=args.log_var_min,
        log_var_max=args.log_var_max,
    ).to(device)

    optimizer = build_optimizer(model, criterion, args)
    scheduler = build_scheduler(optimizer, args)
    scaler    = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    ema       = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

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
        train_meters = defaultdict(float)
        pbar = tqdm(train_data, total=len(train_data), dynamic_ncols=True, leave=False)
        pbar.set_description(f"Epoch {epoch+1:03d}/{args.train_epochs:03d} [Train]")

        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            bx = batch['x'].to(device, non_blocking=True)
            by = batch['y'].to(device, non_blocking=True)
            bs = batch['sc'].to(device, non_blocking=True)
            bp = batch.get('pathology_score', None)
            if bp is not None: bp = bp.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs, aux_info = model(bx, bs, bp)
                total_loss, loss_stats = criterion(outputs, by)
                moe_reg, moe_stats = compute_moe_regularization(
                    aux_info, device,
                    load_balance_weight=args.moe_load_balance_weight,
                    entropy_weight=args.moe_entropy_weight,
                    z_loss_weight=args.moe_z_loss_weight,
                    diversity_weight=args.moe_diversity_weight,
                )
                total_loss = total_loss + moe_reg

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

            for k, v in {**loss_stats, **moe_stats, 'metric_total': total_loss.detach()}.items():
                train_meters[k] += float(v.item())
            pbar.set_postfix({
                'Loss': f"{train_meters['metric_total']/step:.4f}",
                'PCC_Loss': f"{train_meters['loss_pcc']/step:.4f}",
                'PCC': f"{train_meters['pcc']/step:.4f}",
            })

        eval_model  = ema.ema if ema is not None else model
        val_metrics = evaluate(eval_model, val_data, criterion, device, args)
        scheduler.step()

        n            = max(1, len(train_data))
        train_metrics = {k: v / n for k, v in train_meters.items()}
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
            f"logvar[pcc/mae/diff/std]="
            f"{train_metrics['log_var_pcc']:.2f}/"
            f"{train_metrics['log_var_mae']:.2f}/"
            f"{train_metrics['log_var_diff']:.2f}/"
            f"{train_metrics['log_var_std']:.2f}"
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
            torch.save(deepcopy(eval_model.state_dict()), os.path.join(save_dir, best_name))
            tqdm.write(f"New Best Model Saved! (Best Val PCC Loss: {best_pcc_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly Stopping! 连续 {args.patience} epoch 未改善。"
                           f"\n最佳 Val PCC Loss: {best_pcc_loss:.4f}")
                last_name = 'base_last.pt' if args.mode == 'pretrain' else 'finetuned_last.pt'
                torch.save((ema.ema if ema else model).state_dict(),
                           os.path.join(save_dir, last_name))
                break

        if epoch == args.train_epochs - 1:
            last_name = 'base_last.pt' if args.mode == 'pretrain' else 'finetuned_last.pt'
            torch.save((ema.ema if ema else model).state_dict(),
                       os.path.join(save_dir, last_name))

        # TensorBoard
        for tag, val in [
            ('Train/Total', train_metrics['metric_total']),
            ('Train/PCC_Loss', train_metrics['loss_pcc']),
            ('Train/PCC', train_metrics['pcc']),
            ('Val/Total', val_metrics['metric_total']),
            ('Val/PCC_Loss', val_metrics['loss_pcc']),
            ('Val/PCC', val_metrics['pcc']),
            ('Val/MAE', val_metrics['metric_mae']),
        ]: writer.add_scalar(tag, val, epoch)
        for lv in ('pcc', 'mae', 'diff', 'std'):
            writer.add_scalar(f'LossWeight/LogVar_{lv.upper()}',
                              train_metrics[f'log_var_{lv}'], epoch)
        for name, lr in group_lrs.items():
            writer.add_scalar(f'LR/{name}', lr, epoch)
        if cur_temp is not None:
            writer.add_scalar('Train/MoE_Temp', cur_temp, epoch)
            for tag, key in [('Train/MoE_LoadBalance', 'moe_load_balance'),
                              ('Train/MoE_LBT_Raw', 'moe_lbt_raw'),
                              ('Train/MoE_ZLoss', 'moe_z_loss'),
                              ('Train/MoE_Entropy', 'moe_entropy')]:
                if key in train_metrics:
                    writer.add_scalar(tag, train_metrics[key], epoch)
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
    print(f"\n训练完成！最佳权重已保存至: {os.path.join(save_dir, best_n)}")


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
    p.add_argument('--clinical_file', default='Rest-meta-MDD-HAMD-V1-V2-Merge-Normalize.xlsx')
    p.add_argument('--num_rois', type=int, default=116)
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--in_window', type=int, default=6)
    p.add_argument('--pred_window', type=int, default=3)
    p.add_argument('--total_windows', type=int, default=9)
    p.add_argument('--n_block', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--norm', type=str2bool, default=True)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--num_scales', type=int, default=3)
    p.add_argument('--ode_steps', type=int, default=6)
    p.add_argument('--ode_hidden_dim', type=int, default=256)
    p.add_argument('--stochastic_depth_rate', type=float, default=0.10)
    p.add_argument('--num_experts', type=int, default=4)
    p.add_argument('--top_k', type=int, default=2)
    p.add_argument('--train_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr_init', type=float, default=1e-6)
    p.add_argument('--lr_peak', type=float, default=1e-4)
    p.add_argument('--lr_final', type=float, default=1e-6)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--init_log_var_pcc', type=float, default=0.0)
    p.add_argument('--init_log_var_mae', type=float, default=-1.5)
    p.add_argument('--init_log_var_diff', type=float, default=-2.0)
    p.add_argument('--init_log_var_std', type=float, default=-2.0)
    p.add_argument('--loss_lr_scale', type=float, default=1.0)
    p.add_argument('--clamp_log_vars', type=str2bool, default=True)
    p.add_argument('--log_var_min', type=float, default=-6.0)
    p.add_argument('--log_var_max', type=float, default=6.0)

    # MoDE MoE 超参
    p.add_argument('--moe_load_balance_weight', type=float, default=0.01,
                   help='负载均衡损失权重。乘以 load_balancing_term（均匀时≈1.0）。典型值 0.01~0.02。')
    p.add_argument('--moe_entropy_weight', type=float, default=1e-3,
                   help='门控熵正则化权重，鼓励路由分布不过于集中。')
    p.add_argument('--moe_z_loss_weight', type=float, default=1e-3,
                   help='路由 Z-Loss 权重（ST-MoE），惩罚 logit 幅度过大。')
    p.add_argument('--moe_diversity_weight', type=float, default=0.001,
                   help='专家多样化损失权重，鼓励 batch 内路由决策的多样性。典型值 0.001~0.01。')
    p.add_argument('--moe_gate_temp_start', type=float, default=2.0,
                   help='路由温度初始值。高温使 softmax 更均匀，有利于 multinomial 探索。')
    p.add_argument('--moe_gate_temp_end', type=float, default=1.0,
                   help='路由温度终止值（backbone 解冻后线性衰减）。')
    p.add_argument('--moe_expert_hidden_dim', type=int, default=256,
                   help='所有路由专家统一隐层维度（均等容量）。')
    p.add_argument('--moe_use_shared_expert', type=str2bool, default=True,
                   help='是否启用共享专家（SharedPathologyExpert，始终激活）。')
    p.add_argument('--moe_router_cond_only', type=str2bool, default=True,
                   help='路由仅基于病理条件（不含脑信号特征）。')
    p.add_argument('--moe_use_argmax', type=str2bool, default=False,
                   help='推理时用 topk（False=默认），训练时用 multinomial 随机探索。')
    p.add_argument('--moe_inference_temperature', type=float, default=0.3,
                   help='推理时路由器的幂次缩放温度（0.2~0.5，默认 0.3），'
                        '解决"训练 multinomial 均匀 → 推理 topk 坍塌"的不一致问题。')

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

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
