#!/usr/bin/env python
# coding=utf-8
"""FC 指标基线套件（2.2 验收参照）。

为模型的 FC 指标（fc_upper_mae / edge_pcc / 网络级 MAE）提供无模型基线参照，
回答「模型 FC 重构是否优于平凡策略」：

    - persistence: 最后一帧持续（滑窗预测的标准平凡基线）
    - linear:      每 ROI 对历史窗口均值做线性趋势外推
    - mean:        历史窗口均值持续

test 集由 checkpoint 配置快照 + 版本化切分 manifest 决定，与 evaluate_variant
评估的 test 完全一致。输出 <out_dir>/fc_baselines.json。

用法:
    python -m analysis.fc_baselines --ckpt <finetuned_best.pt> \
        --out_dir <eval_dir 或新目录>
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_variant import (build_eval_loader, compute_fc_metrics,
                                          load_train_args)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def persistence_forecast(x: torch.Tensor, pred_window: int) -> torch.Tensor:
    """x: [B, F, W, S] → [B, F, pred_window, S]，重复最后一帧。"""
    return x[:, :, -1:, :].repeat(1, 1, pred_window, 1)


def mean_forecast(x: torch.Tensor, pred_window: int) -> torch.Tensor:
    """历史窗口均值持续。"""
    return x.mean(dim=2, keepdim=True).repeat(1, 1, pred_window, 1)


def linear_forecast(x: torch.Tensor, pred_window: int) -> torch.Tensor:
    """每 ROI 对 W 个窗口均值做最小二乘线性外推，时间帧内保持常数。

    x: [B, F, W, S]；窗口均值 [B, F, W] → 外推 W+1..W+pred_window 步。
    """
    b, f, w, s = x.shape
    win_mean = x.mean(dim=3)                                  # [B, F, W]
    t = torch.arange(w, dtype=win_mean.dtype, device=win_mean.device)
    t_c = t - t.mean()
    denom = (t_c ** 2).sum().clamp_min(1e-8)
    slope = (win_mean * t_c).sum(dim=2) / denom               # [B, F]
    intercept = win_mean.mean(dim=2) - slope * t.mean()       # [B, F]
    steps = torch.arange(w, w + pred_window,
                         dtype=win_mean.dtype, device=win_mean.device)
    ext = intercept.unsqueeze(-1) + slope.unsqueeze(-1) * steps  # [B, F, pred_window]
    return ext.unsqueeze(-1).expand(b, f, pred_window, s)


@torch.no_grad()
def collect(loader):
    """遍历 test loader，返回 (x_hist, y_true) numpy 数组。"""
    xs, ys = [], []
    for batch in loader:
        xs.append(batch['x'].detach().cpu().numpy())
        ys.append(batch['y'].detach().cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(description='FC 指标基线套件（persistence/linear/mean）')
    ap.add_argument('--ckpt', required=True, help='finetune checkpoint（取配置快照与切分）')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--refresh_split_manifest', action='store_true',
                    help='重新生成切分 manifest（默认复用）')
    ns = ap.parse_args()

    args = load_train_args(ns.ckpt)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, _ = build_eval_loader(args, 'test',
                                  refresh_split_manifest=ns.refresh_split_manifest)
    log.info('收集 test 数据（基线不加载模型权重，仅用同一 test 划分）...')
    x, y = collect(loader)
    pred_window = y.shape[2]
    log.info(f'test 样本 {x.shape[0]}，历史 {x.shape[2]} 窗 × {x.shape[3]} 帧，'
             f'预测 {pred_window} 窗 × {y.shape[3]} 帧')

    baselines = {
        'persistence': persistence_forecast(torch.from_numpy(x), pred_window).numpy(),
        'linear': linear_forecast(torch.from_numpy(x), pred_window).numpy(),
        'mean': mean_forecast(torch.from_numpy(x), pred_window).numpy(),
    }

    result = {'n_test_samples': int(x.shape[0]), 'n_rois': int(x.shape[1])}
    for name, pred in baselines.items():
        m = compute_fc_metrics(pred, y)
        # 点预测参照（供 PCC/MAE 对照）
        p = pred.reshape(pred.shape[0], -1)
        t = y.reshape(y.shape[0], -1)
        pcc = np.mean([np.corrcoef(pi, ti)[0, 1]
                       for pi, ti in zip(p, t)
                       if pi.std() > 1e-8 and ti.std() > 1e-8])
        m['point_pcc'] = float(pcc)
        m['point_mae'] = float(np.abs(pred - y).mean())
        result[name] = m
        log.info(f'{name}: FC_upper_MAE={m["fc_upper_mae"]:.4f} '
                 f'edge_PCC={m["edge_pcc_mean"]:.4f} point_PCC={m["point_pcc"]:.4f} '
                 f'point_MAE={m["point_mae"]:.4f}')

    out_path = out_dir / 'fc_baselines.json'
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'\n基线指标已写入: {out_path}')
    print('对照方法: evaluate_variant 输出 metrics_test.json 中同名 FC 指标。')


if __name__ == '__main__':
    main()
