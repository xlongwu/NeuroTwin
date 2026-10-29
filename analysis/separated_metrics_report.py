"""分协议汇总评估：滚动单步 vs 固定锚点多步 H1–8。

两个协议回答不同问题，指标不合并：

- rolling_one_step：每个 TR 用真实观测历史预测下一点，拼接成完整预测
  序列后计算 MAE/MSE/RMSE、时间 PCC 与整段 FC。它衡量“跟踪”能力，
  不是自主生成。
- fixed_anchor_multi_step：单个 K64 锚点一次前向输出 H1–8，未来真值不
  反馈；逐 horizon 报告 MAE/RMSE/空间 PCC/方向准确率，FC 在 8 步未来
  序列上计算。H1 单独列为 single_step。

示例::

    python -m analysis.separated_metrics_report \
        --hc-eval "results/H1-8当前结果/健康对照/指标_测试集.json" \
        --mdd-eval "results/H1-8当前结果/抑郁症/指标_测试集.json" \
        --hc-rolling-dir "results/H1-8当前结果/滚动单步_健康对照/hc" \
        --hc-fc "results/H1-8当前结果/功能连接_健康对照/功能连接汇总_测试集.json" \
        --mdd-rolling-dir "results/H1-8当前结果/滚动单步_抑郁症_微调/mdd" \
        --mdd-fc "results/H1-8当前结果/功能连接_抑郁症_微调/功能连接汇总_测试集.json" \
        --out "results/H1-8当前结果/单步与多步协议汇总.json"
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HORIZONS = tuple(f'H{k}' for k in range(1, 9))


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _rolling_summary_path(rolling_dir):
    """优先读取中文汇总；历史 summary_test.json 作为兼容回退。"""
    path = Path(rolling_dir)
    preferred = path / '滚动单步汇总_测试集.json'
    return preferred if preferred.exists() else path / 'summary_test.json'


def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def single_step(eval_report):
    """固定锚点协议的 H1：下一步点值误差与空间 pattern 相关。"""
    h1 = eval_report['head_metrics']['ODE']['H1']
    return {
        'definition': '单个 K64 锚点预测 t+1；空间 PCC 为同一步跨 116 ROI 的 Pearson 相关',
        'n_subjects': int(h1['n_subjects']), 'n_tasks': int(h1['n_tasks']),
        'mae': float(h1['MAE']), 'mse': float(h1['RMSE'] ** 2),
        'rmse': float(h1['RMSE']), 'pcc_spatial': float(h1['PCC_spatial']),
        'r2': float(h1['R2']),
        'direction_accuracy': float(h1['direction_accuracy_step']),
        'variance_ratio': float(h1['variance_ratio']),
    }


def multi_step(eval_report):
    """固定锚点协议的 H1–8：逐 horizon 与总体指标 + 8 步 FC。"""
    head = eval_report['head_metrics']['ODE']
    per_horizon = {}
    for h in HORIZONS:
        if h not in head:
            continue
        blk = head[h]
        per_horizon[h] = {
            'mae': float(blk['MAE']), 'mse': float(blk['RMSE'] ** 2),
            'rmse': float(blk['RMSE']),
            'pcc_spatial': float(blk['PCC_spatial']),
            'direction_accuracy': float(blk['direction_accuracy_step']),
        }
    overall = head['overall']
    arms = [v for k, v in per_horizon.items() if k != 'H1']
    fc = eval_report['fc_h1_8']
    return {
        'definition': ('单个 K64 锚点一次前向输出 H1–8，未来真值不反馈；'
                       '空间 PCC 为每个时点跨 116 ROI 的 Pearson 相关，'
                       'FC 在预测/真实 8 步序列上计算（单 TR 无 FC）'),
        'per_horizon': per_horizon,
        'aggregate_H1_8': {
            'mae': float(overall['MAE']), 'mse': float(overall['RMSE'] ** 2),
            'rmse': float(overall['RMSE']),
            'pcc_spatial': float(overall['PCC_spatial']),
            'r2': float(overall['R2']),
            'direction_accuracy': float(overall['direction_accuracy_step']),
            'variance_ratio': float(overall['variance_ratio']),
        },
        'aggregate_H2_8_equal_horizon_mean': {
            'mae': _mean([v['mae'] for v in arms]),
            'pcc_spatial': _mean([v['pcc_spatial'] for v in arms]),
            'direction_accuracy': _mean([v['direction_accuracy'] for v in arms]),
            'note': 'H2–H8 逐 horizon 指标的等权平均，仅用于与 H1 单步对照',
        },
        'fc_H1_8': {
            'fc_mse': float(fc['fc_mse']['mean']),
            'fc_edge_pcc': float(fc['fc_pcc']['mean']),
            'fc_group_pcc': float(fc['fc_group_pcc']),
            'freq_l1': float(fc['freq_l1']['mean']),
            'n_subjects': int(fc['fc_mse']['n_subjects']),
        },
    }


def _rolling_cache_stats(cache_dir):
    """从 dense rolling 缓存计算 MAE/MSE/RMSE、整段时间 PCC 与幅度比。"""
    mae, mse, rmse, tpcc, amp = [], [], [], [], []
    for path in sorted(Path(cache_dir).glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            if 'pred' not in z.files or 'true' not in z.files:
                continue
            pred = np.asarray(z['pred'], dtype=np.float64)
            true = np.asarray(z['true'], dtype=np.float64)
            err = np.abs(pred - true)
            mae.append(float(err.mean()))
            m2 = float((err ** 2).mean())
            mse.append(m2)
            rmse.append(float(np.sqrt(m2)))
            p = pred - pred.mean(axis=0, keepdims=True)
            t = true - true.mean(axis=0, keepdims=True)
            den = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
            r = np.divide((p * t).sum(axis=0), den,
                          out=np.full(pred.shape[1], np.nan), where=den > 0)
            tpcc.append(float(np.nanmean(r)))
            amp.append(float(np.nanmean(
                pred.std(axis=0) / np.maximum(true.std(axis=0), 1e-8))))
    return {
        'n_subjects_from_cache': len(mae),
        'mae_subject_equal': _mean(mae),
        'mse_subject_equal': _mean(mse),
        'rmse_subject_equal_mean_of_subject_rmse': _mean(rmse),
        'temporal_pcc_subject_equal': _mean(tpcc),
        'amplitude_ratio_subject_equal': _mean(amp),
    }


def rolling(summary_path, fc_path, cache_dir):
    summary = _load(summary_path)
    fc = _load(fc_path)
    return {
        'definition': ('每个 TR 用真实观测历史预测下一点并拼接为完整序列；'
                       'MAE/MSE 为逐点误差，时间 PCC 逐 ROI 沿完整序列计算，'
                       'FC 在完整预测序列上计算。不是自主生成。'),
        'n_subjects': int(summary['n_subjects_this_split']),
        'mae_subject_equal': float(summary['mae_model_subject_equal_weight']),
        'persistence_mae_subject_equal': float(
            summary['mae_persistence_subject_equal_weight']),
        'improvement_vs_persistence': float(
            summary['improvement_subject_equal_weight']),
        'delta_corr_subject_equal': float(summary['delta_corr_mean']),
        'cache': _rolling_cache_stats(cache_dir),
        'full_sequence_fc': {
            'fc_edge_pcc': float(fc['fc_pcc']['mean']),
            'fc_mae': float(fc['fc_mae']['mean']),
            'fc_rmse': float(fc['fc_rmse']['mean']),
            'n_subjects': int(fc['n_subjects']),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hc-eval', required=True)
    ap.add_argument('--mdd-eval', required=True)
    ap.add_argument('--hc-rolling-dir', required=True)
    ap.add_argument('--hc-fc', required=True)
    ap.add_argument('--mdd-rolling-dir', required=True)
    ap.add_argument('--mdd-fc', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ns = ap.parse_args()
    if ns.out.exists():
        raise FileExistsError('输出已存在；请换一个路径，避免覆盖既有结果')
    hc_eval, mdd_eval = _load(ns.hc_eval), _load(ns.mdd_eval)
    report = {
        '_meta': {
            'purpose': '把滚动单步（项目主协议）与固定锚点 H1–8（多步能力）分开报告',
            'single_step_note': ('单步是固定锚点协议的 H1，也是滚动协议的逐点预测；'
                                 '单 TR 无法定义 FC，FC 只出现在 8 步窗口或完整序列协议中'),
        },
        'protocols': {
            'rolling_one_step': {
                'HC': rolling(_rolling_summary_path(ns.hc_rolling_dir),
                              ns.hc_fc, ns.hc_rolling_dir),
                'MDD': rolling(_rolling_summary_path(ns.mdd_rolling_dir),
                               ns.mdd_fc, ns.mdd_rolling_dir),
            },
            'fixed_anchor_multi_step': {
                'HC': {'single_step_H1': single_step(hc_eval),
                       'multi_step_H1_8': multi_step(hc_eval)},
                'MDD': {'single_step_H1': single_step(mdd_eval),
                        'multi_step_H1_8': multi_step(mdd_eval)},
            },
        },
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                 allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
