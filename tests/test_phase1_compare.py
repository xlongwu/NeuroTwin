# coding=utf-8
"""E0/E1/E2 对比脚本的被试配对与对齐检查。"""
import json

import pandas as pd
import pytest

from experiments.loss.compare_phase1 import build_comparison, load_arm


def _write_arm(root, name, mae_offset, subject_ids=('s1', 's2', 's3'),
               cutoffs=(10, 20, 30)):
    d = root / name
    d.mkdir(parents=True)
    (d / 'metrics_test.json').write_text('{}', encoding='utf-8')
    subj = pd.DataFrame({
        'subject_id': list(subject_ids),
        't+1_MAE': [0.10 + mae_offset, 0.12 + mae_offset, 0.14 + mae_offset],
        't+1_RMSE': [0.13 + mae_offset, 0.15 + mae_offset, 0.17 + mae_offset],
        't+1_PCC_spatial': [0.99, 0.98, 0.97],
        'ODE_H1_mae': [0.10 + mae_offset, 0.12 + mae_offset, 0.14 + mae_offset],
        'ODE_H2_mae': [0.20 + mae_offset, 0.22 + mae_offset, 0.24 + mae_offset],
        'ODE_H1_delta_mae_step': [0.10, 0.12, 0.14],
        'ODE_H2_delta_mae_step': [0.08, 0.09, 0.10],
        'ODE_H1_direction_accuracy_step': [0.80, 0.82, 0.84],
        'ODE_H2_direction_accuracy_step': [0.70, 0.72, 0.74],
        'CPM_H1_mae': [0.11 + mae_offset, 0.13 + mae_offset, 0.15 + mae_offset],
        'CPM_H2_mae': [0.21 + mae_offset, 0.23 + mae_offset, 0.25 + mae_offset],
        'CPM_H1_delta_mae_step': [0.11, 0.13, 0.15],
        'CPM_H2_delta_mae_step': [0.09, 0.10, 0.11],
        'CPM_H1_direction_accuracy_step': [0.79, 0.81, 0.83],
        'CPM_H2_direction_accuracy_step': [0.69, 0.71, 0.73],
        'DUAL_H1_gap': [0.01, 0.02, 0.03],
        'DUAL_H2_gap': [0.02, 0.03, 0.04],
        'DUAL_H1_both_fail_vs_persistence': [0.0, 0.0, 0.0],
        'DUAL_H2_both_fail_vs_persistence': [0.1, 0.0, 0.0],
        'DUAL_H1_head_error_corr': [0.9, 0.9, 0.9],
        'DUAL_H2_head_error_corr': [0.8, 0.8, 0.8],
        'DUAL_H1_oracle_mae': [0.10, 0.12, 0.14],
        'DUAL_H2_oracle_mae': [0.20, 0.22, 0.24],
        'DUAL_H1_cpm_win_rate': [0.0, 0.0, 0.0],
        'DUAL_H2_cpm_win_rate': [0.0, 0.0, 0.0],
    })
    subj.to_csv(d / 'per_subject_metrics_test.csv', index=False)
    task = pd.DataFrame({'subj_id': list(subject_ids), 'context_len': 64,
                         'cutoff': list(cutoffs)})
    task.to_csv(d / 'per_task_metrics_test.csv', index=False)
    return d


def test_compare_pairs_subjects_and_reports_improvement(tmp_path):
    arms = {
        'E0': load_arm('E0', _write_arm(tmp_path, 'E0', 0.00), 'test'),
        'E1': load_arm('E1', _write_arm(tmp_path, 'E1', -0.01), 'test'),
        'E2': load_arm('E2', _write_arm(tmp_path, 'E2', -0.02), 'test'),
    }
    table, summaries, pairs = build_comparison(arms)
    assert len(table) == 3
    assert summaries['E1']['MAE']['mean'] < summaries['E0']['MAE']['mean']
    assert summaries['E2']['ODE_MAE']['mean'] < summaries['E1']['ODE_MAE']['mean']
    e1e0 = pairs['E1-E0']['MAE']
    assert e1e0['mean_improvement'] == pytest.approx(0.01)
    assert e1e0['win_rate'] == pytest.approx(1.0)
    assert e1e0['n_subjects'] == 3


def test_compare_rejects_misaligned_anchors(tmp_path):
    a = load_arm('E0', _write_arm(tmp_path, 'E0', 0.0), 'test')
    b = load_arm('E1', _write_arm(tmp_path, 'E1', 0.0, cutoffs=(10, 20, 99)),
                 'test')
    with pytest.raises(ValueError, match='锚点集合'):
        build_comparison({'E0': a, 'E1': b})
