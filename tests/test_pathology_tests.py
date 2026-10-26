# coding=utf-8
"""analysis.pathology_tests 统计核心的合成数据单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.pathology_tests import (bh_fdr, network_columns,
                                      paired_bootstrap_ci, partial_corr,
                                      site_blocked_paired_permutation,
                                      site_blocked_unpaired_permutation)


def test_paired_permutation_all_positive_min_p():
    rng = np.random.RandomState(3)
    diff = rng.uniform(0.5, 1.0, size=(40, 3))
    sites = ['S1'] * 20 + ['S2'] * 20
    p = site_blocked_paired_permutation(diff, sites, n_perm=2000, seed=2024)
    assert np.allclose(p, 1.0 / 2001)             # 全为正 → 仅全不翻转超过观测


def test_paired_permutation_null_not_rejected():
    rng = np.random.RandomState(5)
    diff = rng.randn(200, 4)                      # H0 下的对称噪声
    sites = [f'S{i % 8}' for i in range(200)]
    p = site_blocked_paired_permutation(diff, sites, n_perm=4000, seed=2024)
    assert np.all(p > 0.01)
    assert 0.2 < np.median(p) < 0.8               # p 应大致均匀


def test_paired_permutation_deterministic():
    rng = np.random.RandomState(9)
    diff = rng.randn(60, 2)
    sites = ['S1'] * 30 + ['S2'] * 30
    a = site_blocked_paired_permutation(diff, sites, n_perm=500, seed=7)
    b = site_blocked_paired_permutation(diff, sites, n_perm=500, seed=7)
    assert np.array_equal(a, b)


def test_unpaired_permutation_identical_groups_and_signal():
    rng = np.random.RandomState(11)
    a = rng.randn(100, 2)
    sites = [f'S{i % 5}' for i in range(100)]
    p_null = site_blocked_unpaired_permutation(a, a.copy(), sites, sites,
                                               n_perm=500, seed=2024)
    assert np.all(p_null > 0.4)                   # 同一组 vs 自身
    b = a + 2.0
    p_sig = site_blocked_unpaired_permutation(b, a, sites, sites,
                                              n_perm=2000, seed=2024)
    assert np.all(p_sig <= 1.0 / 2001)


def test_paired_bootstrap_ci_covers_truth():
    rng = np.random.RandomState(13)
    diff = rng.randn(500, 2) + 0.3
    lo, hi = paired_bootstrap_ci(diff, n_boot=4000, seed=2024)
    assert np.all(lo < 0.3) and np.all(hi > 0.3)
    assert np.all(lo < hi)
    # 确定性
    lo2, hi2 = paired_bootstrap_ci(diff, n_boot=4000, seed=2024)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)


def test_bh_fdr_monotone_and_known_case():
    p = np.array([0.001, 0.002, 0.01, 0.04, 0.5])
    q = bh_fdr(p)
    assert np.all(np.diff(q[p.argsort()]) >= -1e-12)   # q 单调不减
    assert q[0] <= 0.005 and q[-1] <= 1.0


def test_network_columns_partition():
    cols = network_columns(116)
    named = np.concatenate([v for k, v in cols.items() if k != 'Other'])
    assert len(np.unique(named)) == len(named)        # 命名网络不重叠
    assert len(named) + len(cols['Other']) == 116
    assert len(cols['Other']) >= 100                    # 当前 NETWORK_GROUPS 仅 14 ROI


def test_partial_corr_recovers_signal_and_blocks_confound():
    rng = np.random.RandomState(17)
    n = 400
    conf = rng.randn(n) * 3
    x = conf + rng.randn(n)
    y = 0.5 * x + conf + rng.randn(n)                 # x 与 y 的真实偏相关=0.5
    res = partial_corr(y, x, conf[:, None])
    assert abs(res['r'] - 0.5 / np.sqrt(1 + 0.5 ** 2)) < 0.1 or abs(res['r'] - 0.5) < 0.12
    assert res['p'] < 1e-3
    # 纯混杂（y 只依赖 controls）→ 偏相关应近零
    y2 = conf + rng.randn(n)
    res2 = partial_corr(y2, x, conf[:, None])
    assert abs(res2['r']) < 0.15
