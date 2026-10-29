import json
import sys

import numpy as np
import pytest

from train import pathology_fingerprint as fingerprint
from train import pathology_readout as readout


def groups():
    rng = np.random.default_rng(2024)
    result = []
    for label in (0, 1):
        n = 72
        sites = np.repeat(['A', 'B', 'C'], n // 3)
        age = rng.normal(35, 5, n)
        latent = rng.normal(size=(n, 8))
        features = latent + label * 0.8
        features[:, 0] += age * 2
        group = dict(ids=[f'{label}-{i}' for i in range(n)], label=label,
                     values=np.column_stack((np.ones(n), age, np.full(n, 12))),
                     mask=np.ones((n, 3)), idx={'Sex': 0, 'Age': 1, 'education': 2},
                     sites=sites, features=features, residual_width=4, protocol={})
        if label:
            group['clinical'] = {'HAMD': (age + latent[:, 0] * 3, np.ones(n, dtype=bool)),
                                 'HAMA': (age, np.arange(n) < 9)}
        result.append(group)
    return result


def test_site_folds_never_split_a_site_and_cover_all_rows():
    sites = np.repeat(['A', 'B', 'C', 'D'], [5, 9, 7, 6])
    valid_rows = []
    for train, valid in readout.site_folds(sites):
        assert set(sites[train]).isdisjoint(sites[valid])
        valid_rows.extend(valid)
    assert sorted(valid_rows) == list(range(len(sites)))
    assert readout.site_folds(np.array(['A', 'A'])) == []


def test_classification_cv_refits_hc_reference_without_held_out_sites(monkeypatch):
    hc, mdd = groups()
    calls = []
    original = readout.fit_reference

    def record(group):
        calls.append(set(group['sites']))
        return original(group)

    monkeypatch.setattr(readout, 'fit_reference', record)
    selected, summary = readout._select_classifier(hc, mdd, ['A', 'B', 'C'], 'combined')
    assert len(calls) == 3
    assert all(len(sites) == 2 for sites in calls)
    assert {frozenset(s) for s in calls} == {
        frozenset(['A', 'B']), frozenset(['A', 'C']), frozenset(['B', 'C'])}
    assert summary['selection_split'] == 'train'
    assert selected in (0.01, 0.1, 1.0, 10.0)


def test_clinical_correction_is_applied_at_scoring_and_labels_are_not_inputs():
    hc, mdd = groups()
    model = fingerprint.fit_model(hc, mdd, optimize=True)
    assert model['clinical_targets'] == ['HAMD']  # HAMA has only nine observed train rows.
    rows = fingerprint.score_groups(model, [mdd])
    a = fingerprint.design(mdd['values'], mdd['mask'], mdd['idx'], mdd['sites'],
                           model['demo_stats'], model['site_vocab'])
    x = fingerprint.reference_features(model, mdd, 'combined')
    corrected = x - a @ model['HAMD_feature_beta']
    expected = corrected @ model['HAMD_coef'] + model['HAMD_intercept']
    np.testing.assert_allclose([r['HAMD_predicted_residual'] for r in rows], expected)
    assert not np.allclose(expected, x @ model['HAMD_coef'] + model['HAMD_intercept'])
    mdd['clinical']['HAMD'] = (np.full(len(rows), 10000), np.zeros(len(rows), dtype=bool))
    changed = fingerprint.score_groups(model, [mdd])
    np.testing.assert_allclose([r['HAMD_predicted_residual'] for r in changed], expected)
    assert all(r['HAMD_observed_residual'] == '' for r in changed)


def test_fixed_readout_keeps_legacy_scoring_without_feature_projection():
    hc, mdd = groups()
    model = fingerprint.fit_model(hc, mdd)
    assert 'HAMD_feature_beta' not in model
    rows = fingerprint.score_groups(model, [mdd])
    x = fingerprint.reference_features(model, mdd, 'combined')
    np.testing.assert_allclose([r['HAMD_predicted_residual'] for r in rows],
                               x @ model['HAMD_coef'] + model['HAMD_intercept'])


def test_cli_roundtrip_preserves_selection_and_independent_scoring(tmp_path, monkeypatch):
    data = groups()
    checkpoint = tmp_path / 'hc.pt'
    checkpoint.write_bytes(b'fixed-hc-checkpoint')
    for stage in ('pretrain', 'finetune'):
        (tmp_path / f'subject_split_{stage}.json').write_text('{}')
    monkeypatch.setattr(fingerprint, 'load_features', lambda *args: data)
    output = tmp_path / 'output'
    argv = ['fingerprint', '--data-root', str(tmp_path), '--checkpoint', str(checkpoint),
            '--rolling-dir', str(tmp_path), '--output-dir', str(output)]
    monkeypatch.setattr(sys, 'argv', argv + ['--fit', '--optimize'])
    fingerprint.main()
    monkeypatch.setattr(sys, 'argv', argv + ['--split', 'test'])
    fingerprint.main()
    assert (output / 'val_scores.csv').read_bytes() == (output / 'test_scores.csv').read_bytes()
    with np.load(output / 'fingerprint.npz', allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        assert metadata['selection']['method'] == 'train_site_group_cv'
        assert metadata['selection']['clinical']['HAMD']['n_labeled'] == 72
        assert saved['HAMD_feature_beta'].shape[1] == 8
    metrics = json.loads((output / 'test_metrics.json').read_text())
    assert metrics['all']['HAMD']['rmse'] < metrics['all']['HAMD']['zero_residual_rmse']
    assert set(metrics['per_site']) == {'A', 'B', 'C'}


def test_symptom_summary_marks_constant_prediction_correlation_undefined():
    rows = [dict(label=i % 2, score=i, site='A', shared_train_site=True,
                 HAMD_observed_residual=float(i), HAMD_predicted_residual=0.0)
            for i in range(4)]
    result = fingerprint.summarize_scores(rows)
    assert result['all']['HAMD']['pearson'] is None
    assert result['all']['HAMD']['rmse'] == result['all']['HAMD']['zero_residual_rmse']
