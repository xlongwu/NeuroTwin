"""分协议汇总评估：滚动单步 vs 固定锚点多步 H1–32。

两个协议回答不同问题，指标不合并：

- rolling_one_step：每个 TR 用真实观测历史预测下一点，拼接成完整预测
  序列后计算 MAE/MSE/RMSE、时间 PCC 与整段 FC。它衡量“跟踪”能力，
  不是自主生成。
- fixed_anchor_multi_step：单个 K64 锚点一次前向输出 H1–32，未来真值不
  反馈；逐 horizon 报告 MAE/RMSE/空间 PCC/方向准确率，FC 在 32 步未来
  序列上计算。H1 单独列为 single_step。

"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HORIZONS = tuple(f'H{k}' for k in range(1, 33))


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
    """同一个 CPM H32 前向的 H1 截面；跨 ROI 空间相关。"""
    return {metric: values['H1']['mean']
            for metric, values in eval_report['horizon_metrics'].items() if 'H1' in values}


def multi_step(eval_report):
    """固定 K64 锚点的完整 H32 与各阶段，按被试等权。"""
    return dict(definition='CPM 一次前向输出 H1–32，未来真值不反馈',
                per_horizon=eval_report['horizon_metrics'],
                stages=eval_report['trajectory_metrics'])


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
            'purpose': '把滚动单步（追踪协议）与固定锚点 H1–32（多步能力）分开报告',
            'single_step_note': ('单步是固定锚点协议的 H1，也是滚动协议的逐点预测；'
                                 '单 TR 无法定义 FC，FC 只出现在 32 步窗口或完整序列协议中'),
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
                       'multi_step_H1_32': multi_step(hc_eval)},
                'MDD': {'single_step_H1': single_step(mdd_eval),
                        'multi_step_H1_32': multi_step(mdd_eval)},
            },
        },
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                 allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
