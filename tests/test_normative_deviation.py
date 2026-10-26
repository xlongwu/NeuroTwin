# coding=utf-8
"""analysis.normative_deviation 的合成数据单元测试（gate 置换检验、岭参照拟合、
z 打分与网络汇总；不依赖真实数据或 GPU）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.normative_deviation import (fit_normative, network_summary,
                                          paired_signflip_p, score_normative)

IDX = {'Sex': 0, 'Age': 1, 'education': 2}


def _demo_matrix(rng, n, age_effect=None):
    """合成 hc3 人口学 + 可选年龄效应的 s [N, F]。"""
    age = rng.uniform(18, 65, size=n)
    sex = rng.choice((1.0, 2.0), size=n)
    edu = rng.uniform(6, 20, size=n)
    values = np.stack([sex, age, edu], axis=1)
    mask = np.ones_like(values, dtype=np.uint8)
    base = 0.5 + 0.01 * (age - 40)[:, None] * np.ones((1, 8))   # F=8
    noise = rng.randn(n, 8) * 0.05
    return values, mask, base, noise


def test_paired_signflip_p_perfect_and_null():
    rng = np.random.RandomState(7)
    diff = rng.uniform(0.5, 1.0, size=40)     # 全为正 → p 应为最小值 1/(n+1)
    p = paired_signflip_p(diff, n_perm=2000, seed=2024)
    assert p == 1.0 / 2001
    # 均值恰为 0 的对称 diff → 观测均值 0，p≈0.5（单侧）
    diff_sym = np.tile([1.0, -1.0], 15)
    p0 = paired_signflip_p(diff_sym, n_perm=4000, seed=2024)
    assert 0.3 < p0 < 0.7
    # 全零退化 diff：null 与 obs 恒相等 → p=1（保守，不产生 0）
    assert paired_signflip_p(np.zeros(10), n_perm=100, seed=1) == 101 / 101
    # 确定性：同 seed 同结果
    assert paired_signflip_p(diff, 500, 11) == paired_signflip_p(diff, 500, 11)


def test_gate_style_pipeline_on_synthetic():
    """模型显著优于 persistence → improvement 高且 p 极小（gate 会通过）。"""
    rng = np.random.RandomState(3)
    n = 60
    mae_model = rng.uniform(0.05, 0.1, size=n)
    mae_pers = mae_model + rng.uniform(0.3, 0.6, size=n)
    improvement = 1 - mae_model.mean() / mae_pers.mean()
    p = paired_signflip_p(mae_pers - mae_model, n_perm=5000, seed=2024)
    assert improvement > 0.8 and p <= 1 / 5001


def test_fit_recovers_linear_age_effect_and_z_centered():
    rng = np.random.RandomState(11)
    n = 400
    values, mask, base, noise = _demo_matrix(rng, n)
    s = base + noise
    fit = fit_normative(s, values, mask, IDX)
    # 年龄斜率 0.01/岁：恢复的 age_z 系数 ≈ 0.01 * scale(Age)
    layout = fit['design_layout']
    j = layout.index('age_z')
    age_scale = fit['demo']['Age']['scale']
    assert abs(fit['W'][j].mean() / age_scale - 0.01) < 5e-3
    # 训练残差近似零均值、sigma 接近噪声尺度
    assert np.allclose(fit['residuals'].mean(axis=0), 0.0, atol=1e-9)
    assert np.all(np.abs(fit['sigma_roi'] - 0.05) < 0.01)
    # conformal 分位 ≈ 0.9 分位水平（|resid| ~ |N(0,0.05)| → q≈0.064）
    assert np.all((fit['conformal_q'] > 0.05) & (fit['conformal_q'] < 0.09))

    # 新 HC 样本：z 应近似标准正态（均值≈0）
    values2, mask2, base2, noise2 = _demo_matrix(rng, 200)
    z = score_normative(base2 + noise2, values2, mask2, IDX, fit)
    assert abs(z.mean()) < 0.1 and 0.8 < z.std() < 1.2

    # 病理抬升 3σ 的 ROI：z 均值显著为正
    shifted = base2 + noise2
    shifted[:, 3] += 3 * fit['sigma_roi'][3]
    z2 = score_normative(shifted, values2, mask2, IDX, fit)
    assert abs(z2[:, 3].mean() - 3.0) < 0.2
    assert abs(z2[:, 0].mean()) < 0.1


def test_fit_layout_mismatch_rejected():
    rng = np.random.RandomState(5)
    n = 100
    values, mask, base, noise = _demo_matrix(rng, n)
    fit = fit_normative(base + noise, values, mask, IDX)
    # 改动 Sex 词表（模拟不同人口学协议）→ 设计布局变化 → 拒绝复用
    fit['demo'] = {**fit['demo'], 'Sex': {'vocab': [1.0, 2.0, 3.0], 'n_valid': 100}}
    try:
        score_normative(base + noise, values, mask, IDX, fit)
    except ValueError as exc:
        assert '设计布局不一致' in str(exc)
    else:
        raise AssertionError('布局不一致时应当报错')


def test_network_summary_groups_and_other():
    from analysis.metrics import NETWORK_GROUPS
    rng = np.random.RandomState(13)
    z = rng.randn(50, 116)
    z[:, 4] += 8.0                                   # AAL5 ∈ DMN：单 ROI 强抬升
    rows = {r['network']: r for r in network_summary(z)}
    assert 'Other' in rows
    assert all(r['n_rois'] > 0 for r in rows.values())
    assert sum(r['n_rois'] for r in rows.values()) == 116
    dmN = rows['DMN']
    others = [r['z_mean'] for n, r in rows.items() if n not in ('DMN', 'Other')]
    # DMN 均值 ≈ 8/6 + 噪声，显著高于其余命名网络
    assert abs(dmN['z_mean'] - 8.0 / dmN['n_rois']) < 0.5
    assert dmN['z_mean'] > max(others) + 0.5
    assert dmN['z_subject_mean_of_max'] > 5.0        # 单 ROI 抬升在 max 上清晰可见
