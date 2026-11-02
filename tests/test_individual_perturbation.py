"""Key invariants of within-subject blockwise ROI permutations."""
import numpy as np

from analysis.individual_perturbation import permutations


def test_permutation_is_reproducible_blockwise_and_preserves_last_frame():
    a = permutations(3, 4, 'hc', 116)
    assert a.shape == (116, 3, 64)
    np.testing.assert_array_equal(a, permutations(3, 4, 'hc', 116))
    assert np.all(a[..., -1] == 63)
    for block in range(8):
        start = 8 * block
        actual = np.sort(a[..., start:start+8], axis=-1)
        expected = np.broadcast_to(np.arange(start, start+8), actual.shape)
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(a, permutations(3, 4, 'mdd', 116))
    assert not np.array_equal(a, permutations(3, 5, 'hc', 116))
