import json

import numpy as np
import pytest

from analysis.fc_sequence_eval import compare_fc, evaluate_cache, fc_edges


def test_fc_matches_numpy_and_excludes_diagonal():
    x = np.random.default_rng(2024).normal(size=(64, 6))
    expected = np.corrcoef(x.T)[np.triu_indices(6, 1)]
    np.testing.assert_allclose(fc_edges(x), expected, atol=1e-12)
    result = compare_fc(x, x * np.arange(1, 7) + 3)
    assert result['fc_pcc'] == pytest.approx(1)
    assert result['fc_mae'] == pytest.approx(0, abs=1e-12)
    assert result['n_valid_edges'] == 15


def test_pattern_correlation_matches_direct_edge_comparison():
    rng = np.random.default_rng(2024)
    true = rng.normal(size=(100, 8))
    pred = true + rng.normal(size=true.shape)
    expected = np.corrcoef(fc_edges(pred), fc_edges(true))[0, 1]
    assert compare_fc(pred, true)['fc_pcc'] == pytest.approx(expected)


def test_constant_roi_edges_are_excluded_and_all_constant_is_undefined():
    x = np.random.default_rng(2024).normal(size=(64, 5))
    x[:, 0] = 1
    result = compare_fc(x, x)
    assert result['n_valid_edges'] == 6
    assert result['fc_pcc'] == pytest.approx(1)
    result = compare_fc(np.ones((64, 5)), np.ones((64, 5)))
    assert result['n_valid_edges'] == 0
    assert np.isnan(result['fc_pcc'])


@pytest.mark.parametrize('shape', [(16, 5), (64, 2), (64,)])
def test_rejects_insufficient_sequence(shape):
    with pytest.raises(ValueError):
        fc_edges(np.ones(shape))


def test_rejects_nonfinite_data_and_shape_mismatch():
    x = np.ones((64, 5))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        fc_edges(x)
    with pytest.raises(ValueError, match='shape mismatch'):
        compare_fc(x, np.ones((64, 6)))


def test_cache_uses_only_requested_split_and_rejects_missing_or_unordered(tmp_path):
    x = np.random.default_rng(2024).normal(size=(64, 5))
    split_file = tmp_path / 'split.json'
    split_file.write_text(json.dumps({'train': ['train'], 'val': ['val'], 'test': ['test']}))
    manifest = {'subjects_done': ['test', 'val'], 'group': 'hc',
                'checkpoint': 'model.pt', 'checkpoint_sha256': 'example',
                'context_length': 64, 'calibration_length': 32,
                'bold_source': 'raw', 'graph_source': 'history_fc'}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='subjects missing'):
        evaluate_cache(tmp_path, split_file, 'test')
    np.savez(tmp_path / 'test.npz', pred=x, true=x, x_last=x, cutoff=np.arange(63, 127))
    rows, summary = evaluate_cache(tmp_path, split_file, 'test')
    assert [r['subject_id'] for r in rows] == ['test']
    assert summary['fc_pcc']['mean'] == pytest.approx(1)
    np.savez(tmp_path / 'test.npz', pred=x, true=x, x_last=x, cutoff=np.arange(63, 191, 2))
    with pytest.raises(ValueError, match='consecutive'):
        evaluate_cache(tmp_path, split_file, 'test')
