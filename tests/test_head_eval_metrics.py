# coding=utf-8
"""第一阶段评估指标（Delta MAE / Direction Accuracy / 双头 Gap）的闭式测试。"""
import numpy as np

from analysis.next_point_eval import (aggregate_columns, amplitude_quantile_edges,
                                      per_task_amplitude_diagnostics,
                                      per_task_dual_head_metrics,
                                      per_task_head_metrics)


def _close(a, b, tol=1e-9):
    assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=tol, equal_nan=True)


def test_delta_mae_step_uses_adjacent_prediction_differences():
    anchor = np.zeros((1, 2))
    # N=1, F=2, H=3；两个 ROI 的轨迹分别为 [1,3,5] 与 [2,4,6]
    target = np.array([[[1., 3., 5.], [2., 4., 6.]]])
    pred = np.array([[[1., 5., 9.], [2., 6., 10.]]])
    out = per_task_head_metrics(pred, target, np.ones((1, 3), bool), anchor)
    # h=1 完美；h=2/3 的相邻帧差分误差均为 |4-2|=2
    _close(out['delta_mae_step'][0], [0., 2., 2.])
    # 相对锚点的差分误差在代数上恒等于点值 MAE
    _close(out['delta_mae_anchor'][0], out['mae'][0])
    _close(out['delta_mae_anchor'][0], [0., 2., 4.])


def test_direction_accuracy_excludes_near_zero_changes():
    anchor = np.zeros((1, 2))
    # ROI0 的真实变化 0.01（低于默认死区），ROI1 的变化 1.0
    # N=1, F=2, H=1
    target = np.array([[[0.01], [1.0]]])
    # 近零 ROI 方向正确（但应被排除），大幅变化 ROI 方向错误
    pred = np.array([[[0.01], [-1.0]]])
    out = per_task_head_metrics(pred, target, np.ones((1, 1), bool), anchor)
    # 近零 ROI 被排除，只剩 ROI1，DA=0 而非 0.5
    _close(out['direction_accuracy_step'][0], [0.])
    _close(out['direction_accuracy_anchor'][0], [0.])
    out0 = per_task_head_metrics(pred, target, np.ones((1, 1), bool), anchor,
                                 direction_deadzone=0.0)
    _close(out0['direction_accuracy_step'][0], [0.5])


def test_direction_accuracy_only_counts_valid_step_pairs():
    anchor = np.zeros((1, 1))
    # N=1, F=1, H=3
    target = np.array([[[1., 3., 5.]]])
    pred = target.copy()
    mask = np.array([[1., 0., 1.]])          # h2 无效 → h3 的相邻对也不可用
    out = per_task_head_metrics(pred, target, mask, anchor)
    _close(out['delta_mae_step'][0], [0., np.nan, np.nan])
    _close(out['direction_accuracy_step'][0], [1., np.nan, np.nan])


def test_dual_head_gap_quality_and_both_fail():
    anchor = np.zeros((1, 2))
    target = np.zeros((1, 2, 2))
    ode = np.zeros((1, 2, 2))                 # 完美
    cpm = np.ones((1, 2, 2))                 # 与 ODE 相差 1
    mask = np.ones((1, 2), bool)
    out = per_task_dual_head_metrics(cpm, ode, target, mask, anchor)
    _close(out['gap'], [[1., 1.]])
    _close(out['cpm_mae'], [[1., 1.]])
    _close(out['ode_mae'], [[0., 0.]])
    # persistence = 0，ODE 不失败、CPM 失败 → both_fail=0
    _close(out['both_fail_vs_persistence'], [[0., 0.]])
    _close(out['cpm_win_rate'], [[0., 0.]])
    # 误差向量为常数，相关系数无定义
    assert np.isnan(out['head_error_corr']).all()

    # 两头一起错：都预测 2，真值 0，persistence 0
    both = np.full_like(target, 2.0)
    bad = per_task_dual_head_metrics(both, both, target, mask, anchor)
    _close(bad['gap'], [[0., 0.]])
    _close(bad['both_fail_vs_persistence'], [[1., 1.]])
    _close(bad['gap_norm'], [[0., 0.]])


def test_dual_head_oracle_and_complementarity():
    anchor = np.zeros((2, 1))
    target = np.zeros((2, 1, 1))
    # anchor0: CPM 完美、ODE 差；anchor1: 反过来
    cpm = np.array([[[0.0]], [[2.0]]])
    ode = np.array([[[2.0]], [[0.0]]])
    out = per_task_dual_head_metrics(cpm, ode, target, np.ones((2, 1), bool), anchor)
    _close(out['cpm_mae'], [[0.], [2.]])
    _close(out['ode_mae'], [[2.], [0.]])
    _close(out['oracle_mae'], [[0.], [0.]])
    _close(out['oracle_gain_vs_cpm'], [[0.], [2.]])
    _close(out['oracle_gain_vs_ode'], [[2.], [0.]])
    # 两个 anchor 各由不同头胜出
    _close(out['cpm_win_rate'], [[1.], [0.]])


def test_amplitude_diagnostics_detect_uniform_scale_compression():
    anchor = np.zeros((1, 2))
    target = np.array([[[1., 3., 6.], [2., 4., 8.]]])
    pred = 0.5 * target                       # 所有相邻帧变化幅度减半
    mask = np.ones((1, 3), bool)
    edges = amplitude_quantile_edges(target, mask, anchor, n_bins=2)
    assert edges[0] == 0.0
    assert np.isinf(edges[-1])
    out = per_task_amplitude_diagnostics(pred, target, mask, anchor,
                                         amp_edges=edges)
    _close(out['delta_std_ratio'], [0.5])
    _close(out['delta_std_ratio_h1'], [0.5])
    _close(out['delta_std_ratio_h2_8'], [0.5])
    _close(out['delta_pcc'], [1.0])
    _close(out['temporal_variance_ratio'], [0.25])
    _close(out['trajectory_variance_ratio_flat'], [0.25])
    ratio = out['amp_bin_mae'] / out['amp_bin_true_abs_mean']
    _close(ratio[np.isfinite(ratio)], np.full(np.isfinite(ratio).sum(), 0.5))


def test_amplitude_diagnostics_flags_only_multistep_compression():
    anchor = np.zeros((1, 2))
    # H1 两个 ROI 的差分分别为 1、2；H2+ 真实差分仍有变化
    target = np.array([[[1., 3., 6.], [2., 4., 8.]]])
    # H1 完全正确，但 H2+ 每步只变化 0.5
    pred = np.array([[[1., 1.5, 2.], [2., 2.5, 3.]]])
    out = per_task_amplitude_diagnostics(pred, target, np.ones((1, 3), bool), anchor)
    _close(out['delta_std_ratio_h1'], [1.0])
    assert out['delta_std_ratio_h2_8'][0] < 0.5


def test_subject_aggregation_uses_per_subject_then_group():
    pred = np.zeros((3, 1, 2))
    # N=3, F=1, H=2
    target = np.array([[[1., 1.]], [[3., 3.]], [[5., 5.]]])
    anchor = np.zeros((3, 1))
    out = per_task_head_metrics(pred, target, np.ones((3, 2), bool), anchor)
    agg = aggregate_columns(out, np.array(['s1', 's1', 's2'], dtype=object),
                            ['H1', 'H2'])
    # s1 两个 anchor 的 MAE 为 1 和 3 → 被试均值 2；s2 为 5
    assert agg['mae']['H1']['mean'] == 3.5           # (2 + 5)/2
    assert agg['mae']['H1']['n_subjects'] == 2
    assert agg['mae']['H1']['task_mean'] == 3.0      # (1+3+5)/3
