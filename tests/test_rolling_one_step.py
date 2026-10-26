# coding=utf-8
"""analysis.rolling_one_step 护栏指标（guardrail_metrics / subject_scalar_summary）的
合成数据单元测试：不依赖 checkpoint、数据目录或 GPU。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rolling_one_step import (DEFAULT_MIN_POINTS, guardrail_metrics,
                                       subject_scalar_summary)


def _rng():
    return np.random.RandomState(2024)


def test_model_equals_persistence_gives_zero_improvement():
    rng = _rng()
    n, f = 50, 4
    x_last = rng.randn(n, f)
    d_true = rng.randn(n, f) * 0.5
    true = x_last + d_true
    m = guardrail_metrics(x_last, true, x_last)      # pred = x_last（persistence）
    assert np.allclose(m['mae_model'], m['mae_persistence'], atol=1e-12)
    assert np.allclose(m['improvement'], 0.0, atol=1e-9)
    # persistence 自身：Δpred = 0 → corr 未定义（NaN），方向一致率 0
    assert np.all(np.isnan(m['delta_corr']))


def test_perfect_model_gives_unit_improvement_and_delta_corr_one():
    rng = _rng()
    n, f = 60, 3
    x_last = rng.randn(n, f)
    d_true = rng.randn(n, f)
    m = guardrail_metrics(x_last + d_true, x_last + d_true, x_last)
    assert np.allclose(m['mae_model'], 0.0, atol=1e-12)
    assert np.allclose(m['improvement'], 1.0, atol=1e-9)
    assert np.allclose(m['delta_corr'], 1.0, atol=1e-9)
    assert np.allclose(m['dir_acc'], 1.0, atol=1e-12)


def test_delta_corr_matches_theory_under_linear_relation():
    rng = _rng()
    n, f = 200, 2
    x_last = rng.randn(n, f)
    d_true = rng.randn(n, f)
    noise = rng.randn(n, f) * 0.5
    pred = x_last + 2.0 * d_true + noise
    m = guardrail_metrics(pred, x_last + d_true, x_last)
    for j in range(f):
        expected = np.corrcoef(2.0 * d_true[:, j] + noise[:, j], d_true[:, j])[0, 1]
        assert abs(m['delta_corr'][j] - expected) < 1e-10
    # 方向：2*d+noise 与 d 同号比例应明显高于随机
    assert np.all(m['dir_acc'] > 0.8)


def test_constant_series_no_nan_propagation_and_invalid_pool():
    rng = _rng()
    n, f = 40, 3
    x_last = rng.randn(n, f)
    d_true = np.zeros((n, f))              # 恒定目标：真实变化全为零
    pred_d = rng.randn(n, f) * 0.1
    m = guardrail_metrics(x_last + pred_d, x_last + d_true, x_last)
    with np.errstate(invalid='ignore'):
        assert np.all(np.isnan(m['delta_corr']))
        assert np.all(np.isnan(m['dir_acc']))
        assert np.all(np.isnan(m['improvement']))   # mae_persistence = 0
        assert np.all(m['n_valid_delta'] == 0)
        assert np.all(m['mae_model'] >= 0)          # 其余指标保持有限
        assert np.all(np.isfinite(m['mae_model']))


def test_min_points_threshold_enforced():
    rng = _rng()
    n, f = 30, 2
    x_last = rng.randn(n, f)
    d_true = rng.randn(n, f) * 2
    pred = x_last + d_true + rng.randn(n, f) * 0.1
    # 只有前 3 个点的 |d_true| > deadzone（其余压到死区内）
    d_small = d_true.copy()
    d_small[3:, :] = np.sign(d_true[3:, :]) * 1e-4
    true = x_last + d_small
    m = guardrail_metrics(pred, true, x_last, min_points=8, deadzone=0.05)
    assert np.all(m['n_valid_delta'] <= 3)
    assert np.all(np.isnan(m['delta_corr']))        # 3 < 8 → 无效池
    m2 = guardrail_metrics(pred, true, x_last, min_points=3, deadzone=0.05)
    assert np.all(np.isfinite(m2['delta_corr']))    # 放宽阈值后可计算


def test_subject_scalar_summary_blocks_and_fractions():
    rng = _rng()
    n, f = 50, 4
    x_last = rng.randn(n, f)
    d_true = rng.randn(n, f)
    m = guardrail_metrics(x_last + d_true, x_last + d_true, x_last)
    s = subject_scalar_summary(m)
    assert set(s) >= {'mae_model', 'improvement', 'delta_corr', 'dir_acc',
                      'improvement_subject', 'delta_corr_valid_fraction'}
    assert abs(s['improvement_subject'] - 1.0) < 1e-9
    assert s['delta_corr_valid_fraction'] == 1.0
    # 混入恒定 ROI：有效比例下降但标量不产生 NaN
    x2 = x_last.copy()
    x2[:, 0] = 0.0
    m2 = guardrail_metrics(np.zeros_like(x2), x2, x2)
    s2 = subject_scalar_summary(m2)
    assert np.isfinite(s2['mae_model']) and np.isfinite(s2['mae_persistence'])
    assert s2['delta_corr_valid_fraction'] < 1.0
