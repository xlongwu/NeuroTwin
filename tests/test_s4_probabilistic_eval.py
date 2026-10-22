# coding=utf-8
"""S4 概率评估指标的单元测试（手算样例 + 边界 + npz 兼容性 + 集合模式）。"""
import csv
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from analysis import next_point_eval as npe
from analysis import s4_probabilistic_eval as s4


def test_gaussian_nll_hand_computed():
    mean = np.zeros((1, 1, 2))
    std = np.ones((1, 1, 2))
    target = np.array([[[0.0, 1.0]]])
    nll = s4.gaussian_nll(mean, std, target)
    assert nll.shape == (1, 1, 2)
    assert nll[0, 0, 0] == pytest.approx(0.5 * math.log(2 * math.pi))
    assert nll[0, 0, 1] == pytest.approx(0.5 * math.log(2 * math.pi) + 0.5)


def test_gaussian_crps_hand_computed():
    mean = np.zeros((1, 1, 2))
    std = np.ones((1, 1, 2))
    target = np.array([[[0.0, 1.0]]])
    crps = s4.gaussian_crps(mean, std, target)
    # z=0: 2φ(0) − 1/√π；z=1: (2Φ(1)−1)+2φ(1)−1/√π
    c0 = 2.0 / math.sqrt(2 * math.pi) - 1.0 / math.sqrt(math.pi)
    c1 = (2 * 0.8413447460685429 - 1.0) + 2 * 0.24197072451914337 \
        - 1.0 / math.sqrt(math.pi)
    assert crps[0, 0, 0] == pytest.approx(c0, rel=1e-10)
    assert crps[0, 0, 1] == pytest.approx(c1, rel=1e-10)
    assert np.all(crps >= 0)


def test_interval_coverage_and_width():
    z80 = 1.2815515655446004
    mean = np.zeros((1, 1, 4))
    std = np.ones((1, 1, 4))
    target = np.array([[[0.0, 1.0, 1.4, -1.4]]])
    covered, width = s4.gaussian_interval(mean, std, target, 0.8)
    assert covered[0, 0, 0] == 1.0
    assert covered[0, 0, 1] == 1.0
    assert covered[0, 0, 2] == 0.0
    assert covered[0, 0, 3] == 0.0
    assert np.allclose(width, 2 * z80)


def test_zero_variance_and_all_nan_mask():
    mean = np.zeros((1, 1, 2))
    std = np.zeros((1, 1, 2))
    target = np.array([[[0.0, 1.0]]])
    nll = s4.gaussian_nll(mean, std, target)
    crps = s4.gaussian_crps(mean, std, target)
    covered, width = s4.gaussian_interval(mean, std, target, 0.8)
    assert np.isfinite(nll).all()
    assert np.isfinite(crps).all()
    assert covered[0, 0, 0] == 1.0
    assert covered[0, 0, 1] == 0.0
    assert width[0, 0, 0] > 0                       # sigma_floor
    mask = np.zeros((1, 2))
    assert np.isnan(s4.gaussian_nll(mean, std, target, mask)).all()
    assert np.isnan(s4.gaussian_crps(mean, std, target, mask)).all()
    assert np.isnan(s4.reduce_tasks(s4.gaussian_nll(mean, std, target, mask))).all()


def test_reduce_tasks_and_subject_aggregation():
    values = np.array([[[1.0, np.nan]], [[3.0, np.nan]],
                       [[5.0, np.nan]], [[7.0, np.nan]]])
    task_vals = s4.reduce_tasks(values)
    assert np.allclose(task_vals, [1, 3, 5, 7])
    subj = ['a', 'a', 'b', 'b']
    means = s4.subject_means(task_vals, subj)
    assert np.allclose(means, [2.0, 6.0])
    block = s4.summary(means)
    assert block['mean'] == pytest.approx(4.0)
    assert block['n'] == 2
    per_h = s4.reduce_tasks(values, per_horizon=True)
    assert per_h.shape == (4, 2)
    assert np.all(np.isnan(per_h[:, 1]))


def test_paired_stats_structure():
    a = np.array([1.0, 2.0, 4.0])
    b = np.array([2.0, 2.0, 2.0])
    out = s4.paired_stats(a, b)
    assert out['n'] == 3
    assert out['mean_diff'] == pytest.approx(1.0 / 3.0)
    assert 0 <= out['p'] <= 1
    assert out['ci95_low'] < out['ci95_high']


def test_quantile_edges_and_bin_means():
    mag = np.array([[[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]]])
    mask = np.ones((2, 2), dtype=bool)
    edges = s4.quantile_edges(mag, mask, 2)
    assert edges[0] == 0.0
    assert np.isinf(edges[-1])
    values = -mag
    bins = s4.bin_task_means(values, mag, mask, edges)
    assert bins.shape == (2, 2)
    # 每个任务只有 4 个元素，全局分位后单个任务可能只落入一个 Q：
    # 空 bin 必须是 NaN，而非 0（0 会把“无样本”误读成“误差为零”）。
    assert np.isfinite(bins[0, 0]) and np.isfinite(bins[1, 1])
    assert np.isnan(bins[0, 1]) and np.isnan(bins[1, 0])


def test_constant_variance_reference():
    mean = np.zeros((2, 2, 2))
    target = np.array([[[1.0, 2.0], [3.0, 4.0]],
                       [[1.0, 2.0], [3.0, 4.0]]])
    mask = np.ones((2, 2))
    var_roi_h, var_global, counts = s4.build_constant_variance(
        mean, target, mask, 2)
    assert np.allclose(var_roi_h, [[1.0, 4.0], [9.0, 16.0]])
    assert var_global == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert np.allclose(counts, 2)


def test_npz_saved_without_pickle():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'pred.npz'
        npe.save_predictions_npz(
            path,
            pred=np.zeros((2, 3, 1), dtype=np.float32),
            dynamics_std=np.ones((2, 3, 1), dtype=np.float32),
            condition_fields=np.array(['Sex', 'Age'], dtype=object),
            subj_ids=np.array(['s1', 's2']))
        with np.load(path, allow_pickle=False) as z:
            assert z['dynamics_std'].shape == (2, 3, 1)
            assert z['condition_fields'].tolist() == ['Sex', 'Age']
            assert z['condition_fields'].dtype.kind == 'U'


def test_evaluate_arrays_smoke():
    n, f, h = 6, 4, 3
    rng = np.random.default_rng(0)
    test = {
        'ode_pred': rng.normal(size=(n, f, h)).astype(np.float32),
        'dynamics_std': np.full((n, f, h), 0.8, dtype=np.float32),
        'future': rng.normal(size=(n, f, h + 2)).astype(np.float32),
        'future_mask': np.ones((n, h + 2), dtype=np.float32),
        'x_last': rng.normal(size=(n, f)).astype(np.float32),
        'subj_ids': np.array(['a', 'a', 'a', 'b', 'b', 'b']),
    }
    var_roi_h = np.full((f, h), 0.5)
    report = s4.evaluate_arrays(test, var_roi_h, 0.5, n_bins=3)
    assert report['n_subjects'] == 2
    assert report['n_tasks'] == 6
    assert report['horizon'] == h
    assert set(report['subject_level']) == set(s4.METHODS)
    assert set(report['paired_model_minus_reference']) == {
        'constant_roi_h', 'constant_global'}
    assert report['paired_model_minus_reference']['constant_roi_h']['nll']['n'] == 2
    assert len(report['amplitude_strata']['labels']) == 3
    assert report['calibration']['0.8']['model']['nominal'] == 0.8
    assert np.isfinite(report['subject_level']['model']['nll']['mean'])
    assert report['ensemble'] is None


# ══════════════════════════════════════════════════════════════════════════
#  集合（S4-1/S4-2）指标
# ══════════════════════════════════════════════════════════════════════════

def _correlated_samples(n, k, f, h, rho=0.9, seed=0):
    """AR(1) 时间相关、跨 ROI 共享因子的 [N,K,F,H] 轨迹（边际近似 N(0,1)）。"""
    rng = np.random.default_rng(seed)
    coeff = np.zeros((n, k))
    steps = []
    for _ in range(h):
        coeff = rho * coeff + math.sqrt(max(1.0 - rho ** 2, 0.0)) \
            * rng.normal(size=(n, k))
        steps.append(coeff)
    shared = np.stack(steps, axis=2)[:, :, None, :]      # [N,K,1,H]
    indep = rng.normal(size=(n, k, f, h)) * 0.2
    return (shared + indep).astype(np.float32)


def test_ensemble_crps_converges_to_analytic_gaussian():
    rng = np.random.default_rng(7)
    n, k, f, h = 3, 40000, 2, 2
    mean = rng.normal(size=(n, f, h))
    std = np.abs(rng.normal(size=(n, f, h))) + 0.3
    samples = mean[:, None] + std[:, None] * rng.normal(
        size=(n, k, f, h)).astype(np.float32)
    target = mean + std * rng.normal(size=(n, f, h)).astype(np.float32)
    emp = s4.ensemble_crps(samples, target)
    ana = s4.gaussian_crps(mean, std, target)
    rel = np.abs(emp - ana) / np.maximum(ana, 1e-6)
    assert np.median(rel) < 0.02, f'经验 CRPS 未收敛到解析值：{rel}'


def test_ensemble_interval_matches_numpy_quantile():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(5, 64, 3, 4)).astype(np.float32)
    target = rng.normal(size=(5, 3, 4)).astype(np.float32)
    cov, wid = s4.ensemble_interval(samples, target, 0.8)
    lo = np.quantile(samples, 0.10, axis=1)
    hi = np.quantile(samples, 0.90, axis=1)
    assert wid == pytest.approx(hi - lo, rel=1e-6, abs=1e-6)
    assert np.array_equal(cov == 1.0, (target >= lo) & (target <= hi))


def test_iid_shuffle_keeps_marginals_and_breaks_dependence():
    samples = _correlated_samples(8, 64, 5, 6, seed=2)
    target = np.zeros(samples.shape[:1] + samples.shape[2:], dtype=np.float32)
    shuffled = s4.iid_shuffle_samples(samples, seed=3)
    assert shuffled.shape == samples.shape
    # 边际完全不变 → CRPS/覆盖率/宽度逐元素相等
    assert np.allclose(s4.ensemble_crps(shuffled, target),
                       s4.ensemble_crps(samples, target), equal_nan=True)
    for level in (0.5, 0.8):
        c1, w1 = s4.ensemble_interval(samples, target, level)
        c2, w2 = s4.ensemble_interval(shuffled, target, level)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_allclose(w1, w2)
    # 依赖结构被摧毁 → 时间变差函数失配显著增大
    vg_model = s4.temporal_variogram(samples, target)['per_task_score']
    vg_iid = s4.temporal_variogram(shuffled, target)['per_task_score']
    assert np.nanmean(vg_iid) > np.nanmean(vg_model)


def test_energy_score_penalizes_overdispersion():
    rng = np.random.default_rng(5)
    n, k, f, h = 6, 64, 4, 5
    target = rng.normal(size=(n, f, h)).astype(np.float32)
    matched = (target[:, None, :, :]
               + 0.1 * rng.normal(size=(n, k, f, h))).astype(np.float32)
    inflated = (target[:, None, :, :]
                + 1.5 * rng.normal(size=(n, k, f, h))).astype(np.float32)
    assert (np.nanmean(s4.energy_score(matched, target))
            < np.nanmean(s4.energy_score(inflated, target)))


def test_gaussian_samples_shape_and_scale():
    mean = np.zeros((4, 2, 3), dtype=np.float32)
    std = np.full((4, 2, 3), 0.5, dtype=np.float32)
    s = s4.gaussian_samples(mean, std, seed=9, n_samples=2000)
    assert s.shape == (4, 2000, 2, 3)
    assert abs(float(s.std()) - 0.5) < 0.03


def test_evaluate_arrays_ensemble_mode():
    n, f, h, k = 8, 4, 3, 24
    rng = np.random.default_rng(11)
    samples = _correlated_samples(n, k, f, h, seed=11)
    test = {
        'ode_pred': np.zeros((n, f, h), dtype=np.float32),
        'dynamics_std': np.ones((n, f, h), dtype=np.float32),
        'future': rng.normal(size=(n, f, h)).astype(np.float32),
        'future_mask': np.ones((n, h), dtype=np.float32),
        'x_last': np.zeros((n, f), dtype=np.float32),
        'subj_ids': np.array(['a'] * 4 + ['b'] * 4),
        'dynamics_samples': samples,
    }
    report = s4.evaluate_arrays(test, np.full((f, h), 0.5), 0.5, samples=samples)
    assert set(report['subject_level']) == {
        'model', 'model_iid', 'constant_gaussian_K', 's4_0_gaussian_K',
        'constant_roi_h', 'constant_global'}
    # 集合模式无法定义 nll：保持 NaN 但不阻断其他指标
    assert math.isnan(report['subject_level']['model']['nll']['mean'])
    assert np.isfinite(report['subject_level']['model']['crps']['mean'])
    assert np.isfinite(report['subject_level']['model']['crps_fair']['mean'])
    assert set(report['paired_model_minus_reference']) == {
        'model_iid', 'constant_gaussian_K', 's4_0_gaussian_K',
        'constant_roi_h', 'constant_global'}
    ens = report['ensemble']
    assert ens['n_samples'] == k
    for name in ('model', 'model_iid', 'constant_gaussian_K', 's4_0_gaussian_K'):
        for metric in ('energy_score', 'variogram'):
            assert np.isfinite(ens[name][metric]['mean'])
        pit = ens[name]['pit']
        assert pit['n_elements'] > 0
        assert len(pit['rank_histogram_density']) == pit['n_bins']
        assert np.isfinite(ens[name]['coverage_over_ceiling']['80'])
        vgl = ens[name]['variogram_by_lag']
        assert len(vgl['gamma_pred']) == h - 1 and len(vgl['gamma_true']) == h - 1
    # 方向性判断（相关结构优于独立打乱）依赖真值本身的时间结构，见
    # test_iid_shuffle_keeps_marginals_and_breaks_dependence；此处只验证有限性
    assert set(ens) >= {'n_samples', 'model', 'model_iid', 'constant_gaussian_K',
                        'coverage_ceiling', 'paired_model_minus_iid', 'definitions'}
    # CSV 复用同一份逐元素指标，不再按解析高斯重算
    report['_var_roi_h'] = np.full((f, h), 0.5)
    report['_var_global'] = 0.5
    report['_horizon_labels'] = [f'H{i}' for i in range(1, h + 1)]
    with tempfile.TemporaryDirectory() as tmp:
        files = s4.write_csvs(report, test, test['subj_ids'], Path(tmp),
                              s4.DEFAULT_LEVELS, metrics=report['_metrics'],
                              prefix='s4_1')
        for path in files.values():
            assert Path(path).name.startswith('s4_1_')
        with open(files['subject_csv'], encoding='utf-8') as fp:
            header = next(csv.reader(fp))
        assert 'model_iid_crps' in header and 'model_iid_nll' in header
        assert 'model_crps_fair' in header and 'constant_gaussian_K_crps' in header


def test_fair_crps_removes_finite_k_bias():
    """1/K² 离散度项对有限集合系统性高估 CRPS；fair 口径应无此偏差。"""
    rng = np.random.default_rng(17)
    n = 60_000
    y = rng.standard_normal((n, 1, 1)).astype(np.float32)
    analytic = float(np.nanmean(s4.gaussian_crps(
        np.zeros((n, 1, 1), dtype=np.float32),
        np.ones((n, 1, 1), dtype=np.float32), y)))
    biased, fair = {}, {}
    for K in (8, 32, 128):
        s = rng.standard_normal((n, K, 1, 1)).astype(np.float32)
        biased[K] = float(np.nanmean(s4.ensemble_crps(s, y))) - analytic
        fair[K] = float(np.nanmean(s4.ensemble_crps(s, y, fair=True))) - analytic
    # 偏差随 1/K 衰减且方向为高估；fair 口径在 K=8 也已无偏
    assert biased[8] > biased[32] > biased[128] > 0
    assert biased[8] > 0.05
    for K in (8, 32, 128):
        assert abs(fair[K]) < 0.01, f'fair CRPS 仍有偏差: K={K} {fair[K]}'


def test_finite_k_coverage_ceiling_below_nominal_and_converges():
    small = s4.finite_k_coverage_ceiling(8, (0.5, 0.8, 0.9), n_sim=120_000)
    big = s4.finite_k_coverage_ceiling(128, (0.8,), n_sim=120_000)
    for level in ('0.5', '0.8', '0.9'):
        assert small[level]['empirical_coverage'] < float(level)
        assert small[level]['infinite_K_coverage'] == float(level)
    assert (small['0.8']['empirical_coverage']
            < big['0.8']['empirical_coverage'] <= 0.8)
    # 区间宽度同样被有限采样低估
    assert (small['0.8']['empirical_width'] < big['0.8']['empirical_width']
            < small['0.8']['infinite_K_width'])
    with pytest.raises(ValueError):
        s4.finite_k_coverage_ceiling(8, (0.8,), n_sim=100)


def test_sampled_gaussian_crps_exceeds_analytic_at_finite_k():
    """同一分布：有限 K 的经验 CRPS 高于解析 CRPS，fair 口径应回到解析水平。"""
    rng = np.random.default_rng(23)
    n = 40_000
    mean = np.zeros((n, 1, 1), dtype=np.float32)
    std = np.ones((n, 1, 1), dtype=np.float32)
    y = rng.standard_normal((n, 1, 1)).astype(np.float32)
    analytic = float(np.nanmean(s4.gaussian_crps(mean, std, y)))
    s = s4.gaussian_samples(mean, std, seed=3, n_samples=8)
    sampled = float(np.nanmean(s4.ensemble_crps(s, y)))
    fair = float(np.nanmean(s4.ensemble_crps(s, y, fair=True)))
    assert sampled > analytic
    assert abs(fair - analytic) < 0.02


def test_pit_diagnostics_separate_over_and_under_dispersed():
    rng = np.random.default_rng(19)
    n, K = 40_000, 8
    y = rng.standard_normal((n, 1, 1)).astype(np.float32)
    cases = {}
    for sigma in (0.2, 1.0, 5.0):
        s = (sigma * rng.standard_normal((n, K, 1, 1))).astype(np.float32)
        cov, _ = s4.ensemble_interval(s, y, 0.8)
        cases[sigma] = (s4.pit_diagnostics(s, y, seed=0), float(np.nanmean(cov)))
    flat = cases[1.0][0]
    # 校准模型：随机化 PIT 匹配有限 K 的精确理想矩，秩直方图接近均匀
    assert abs(flat['pit_mean'] - flat['pit_mean_ideal']) < 0.02
    assert abs(flat['pit_std'] / flat['pit_std_ideal'] - 1.0) < 0.05
    assert max(flat['rank_histogram_density']) < 1.35 / flat['n_bins']
    assert abs(flat['rank_mean'] - flat['rank_ideal_mean']) < 0.1
    # 欠散（过度自信）→ U 形（两端远高于中部）；过散 → 中部隆起
    narrow, wide = cases[0.2][0], cases[5.0][0]
    assert narrow['rank_histogram_density'][0] > 4 * narrow['rank_histogram_density'][5]
    assert wide['rank_histogram_density'][5] > 2 * wide['rank_histogram_density'][0]
    # 用上界归一化后：校准≈1，欠散明显<1，过散明显>1（原始 cov80 不可直接判读）
    ceiling = s4.finite_k_coverage_ceiling(K, (0.8,), n_sim=120_000)['0.8']
    ratios = {sig: cov / ceiling['empirical_coverage']
              for sig, (_, cov) in cases.items()}
    assert abs(ratios[1.0] - 1.0) < 0.05
    assert ratios[0.2] < 0.5 < ratios[5.0]


def test_ensemble_metrics_match_torch_prob_losses():
    torch = pytest.importorskip('torch')
    from train import prob_losses as pl

    rng = np.random.default_rng(13)
    n, k, f, h = 5, 32, 4, 6
    samples = rng.normal(size=(n, k, f, h)).astype(np.float32)
    target = rng.normal(size=(n, f, h)).astype(np.float32)
    mask = np.ones((n, h), dtype=np.float32)
    mask[1, 4:] = 0.0
    mask[2] = 0.0
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731

    crps = s4.ensemble_crps(samples, target, mask)
    assert np.isfinite(crps[2]).sum() == 0            # 全缺测任务记为 NaN
    ref = pl.ensemble_crps(t(samples), t(target), t(mask)).item()
    assert float(np.nanmean(crps)) == pytest.approx(ref, rel=1e-5)

    es = s4.energy_score(samples, target, mask)
    vg = s4.temporal_variogram(samples, target, mask)['per_task_score']
    for i in (0, 1, 3, 4):
        assert float(es[i]) == pytest.approx(
            pl.ensemble_energy_score(t(samples[i:i + 1]), t(target[i:i + 1]),
                                     t(mask[i:i + 1])).item(), rel=1e-5)
        assert float(vg[i]) == pytest.approx(
            pl.temporal_variogram_score(t(samples[i:i + 1]), t(target[i:i + 1]),
                                        t(mask[i:i + 1])).item(), rel=1e-5)
