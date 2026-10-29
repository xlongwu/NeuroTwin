import numpy as np

from analysis.h18_data import calibrate_raw
from analysis.h18_group_tests import bootstrap, dispersion
from analysis.h18_short_sequence import edges, similarity


def test_raw_calibration_matches_float32_evaluation_path():
    raw = np.arange(100 * 3, dtype=np.float64).reshape(100, 3) / 7
    x = np.ascontiguousarray(raw.astype(np.float32).T)
    cal = x[:, :32]
    expected = ((x - cal.mean(axis=1, keepdims=True)) /
                np.sqrt(cal.var(axis=1, keepdims=True) + 1e-6)).T
    np.testing.assert_array_equal(calibrate_raw(raw), expected)


def test_paired_bootstrap_and_dispersion_under_identical_inputs():
    lo, hi = bootstrap(np.full((12, 3), .25))
    np.testing.assert_allclose(lo, .25)
    np.testing.assert_allclose(hi, .25)
    ratio, p = dispersion(np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 4.]))
    assert ratio == 1.0 and p == 1.0


def test_short_fc_constant_roi_is_excluded_not_filled():
    rng = np.random.RandomState(2024)
    x = rng.normal(size=(8, 116))
    x[:, 0] = 1
    fc = edges(x)
    assert len(fc) == 6670
    assert np.isnan(fc).sum() == 115
    score = similarity([fc], [fc])
    np.testing.assert_allclose(score[0, 0], 1.0)
