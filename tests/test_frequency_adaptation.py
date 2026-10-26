"""Synthetic CPU checks; never read the real cohort or adapt production weights."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.long_horizon import frequency_adaptation as adaptation
from experiments.long_horizon import frozen_features as cache
from models.tfm import NeuroTwinTFM
from train.losses import TFMDualLoss


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(old)


def tiny_model(dropout=0.):
    model = NeuroTwinTFM(
        features=3, context_max=64, patch_len=8, dim=8, num_layers=1,
        num_heads=2, ff_ratio=2, dropout=dropout, one_step_hidden_dim=8,
        cpm_horizon=32, cpm_layers=1, dynamics_mode='graph_ode',
        graph_source='history_fc', personalized=True, profile_dim=4,
        dynamics_history_memory=True, ode_steps=1, sc_sinkhorn_iters=2,
        use_revin=True)
    with torch.no_grad():
        model.rev_norm.affine_weight.copy_(torch.tensor([.7, 1.2, 1.8]))
        model.rev_norm.affine_bias.fill_(.2)
    return model.eval()


def batch(n=3):
    x = torch.randn(n, 3, 1, 64)
    return {'x': x, 'x_last': x[:, :, 0, -1], 'sc': torch.eye(3).expand(n, -1, -1),
            'calibration_bold': x[..., :8], 'controls': torch.zeros(n, 32, 3),
            'future': torch.randn(n, 3, 32), 'future_mask': torch.ones(n, 32),
            'cutoff': torch.full((n,), 63), 'subj_id': [f's{i}' for i in range(n)]}


def settings():
    return SimpleNamespace(lambda_cpm=1., lambda_pcc=.1, cpm_huber_delta=1.,
                           lambda_temporal_diff=.05, lambda_cross_head=.02,
                           lambda_temporal_pcc=0., temporal_lags='1',
                           temporal_scale_weights='1.0', seed=2024,
                           lr_init=5e-5, weight_decay=.01, grad_clip=1.)


def criterion():
    return adaptation.fixed_cpm_objective(settings(), {})[0]


def test_frequency_phase_dc_and_gradients():
    t = torch.arange(32, dtype=torch.float64)
    target = torch.sin(2 * torch.pi * 3 * t / 32)[None, None]
    assert adaptation.complex_frequency_loss(target, target).item() == 0
    assert adaptation.complex_frequency_loss(target + 5., target).item() < 1e-6
    shifted = target.roll(3, dims=-1).clone().requires_grad_()
    torch.testing.assert_close(torch.fft.rfft(shifted).abs(), torch.fft.rfft(target).abs())
    loss = adaptation.complex_frequency_loss(shifted, target)
    assert loss.item() > .1
    loss.backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
    identical = target.float().clone().requires_grad_()
    adaptation.complex_frequency_loss(identical, identical.detach()).backward()
    assert torch.isfinite(identical.grad).all()
    with pytest.raises(ValueError, match='32'):
        adaptation.complex_frequency_loss(target[..., :16], target[..., :16])


def test_cache_replays_forward_and_rejects_real_boundaries():
    model, data = tiny_model(), batch()
    features = cache.extract_batch(model, data, 'cpu')
    assert features['tokens'].shape == (3, 3, 8, 8)
    assert features['profile'].shape == (3, 4)
    assert features['z0'].shape == (3, 3, 8)
    assert all(not t.requires_grad for t in features.values() if torch.is_tensor(t))
    with torch.no_grad():
        _, aux = model(data['x'], data['sc'], calibration_bold=data['calibration_bold'],
                       controls=data['controls'], dynamics_horizon=32)
        torch.testing.assert_close(cache.cached_cpm(model, features), aux['cpm_pred'])
        torch.testing.assert_close(features['ode_pred'], aux['dynamics_pred'])
    changed_future = {**data, 'future': data['future'] * 20 + 100}
    other = cache.extract_batch(model, changed_future, 'cpu')
    for key in ('tokens', 'z0', 'q0', 'profile', 'adj', 'cpm_pred', 'ode_pred'):
        torch.testing.assert_close(features[key], other[key])
    data['future_mask'][0, -1] = 0
    with pytest.raises(ValueError, match='完整'):
        cache.extract_batch(model, data, 'cpu')
    data['future_mask'].fill_(1)
    data['controls'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='无控制'):
        cache.extract_batch(model, data, 'cpu')


def test_fixed_objective_preserves_cpm_gradients_and_fallback_label():
    features = cache.extract_batch(tiny_model(), batch(), 'cpu')
    pred = features['cpm_pred'].clone().requires_grad_()
    full = TFMDualLoss(weighting='uncertainty', lambda_temporal_diff=.05,
                       lambda_cross_head=.02)
    with torch.no_grad():
        full.log_s['cpm'].fill_(-.6)
    learned = full.get_loss_weights()['cpm']['weight']
    fixed, info = adaptation.fixed_cpm_objective(settings(),
        {'loss_weighting': {'weights': {'cpm': learned, 'temporal_diff': .05, 'cross_head': .02}}})
    one = features['ode_pred'][:, :, :1].unsqueeze(-1)
    target = features['target']
    reference, _ = full(one, target[:, :, :1].unsqueeze(-1), pred, target,
                        dynamics_pred=features['ode_pred'], dynamics_target=target,
                        context_last=features['x_last'])
    got = adaptation.original_loss(fixed, pred, features)
    torch.testing.assert_close(torch.autograd.grad(reference, pred)[0],
                               torch.autograd.grad(got, pred)[0])
    assert 'frozen' in info['source']
    assert 'substitute' in adaptation.fixed_cpm_objective(settings(), {})[1]['source']


def test_only_cpm_updates_and_zero_frequency_arms_match_with_dropout():
    initial = tiny_model(dropout=.15)
    features = cache.extract_batch(initial, batch(), 'cpu')
    before = deepcopy(initial.state_dict())
    results = []
    for _ in range(2):
        model = deepcopy(initial)
        history = adaptation.adapt_arm(model, criterion(), features, updates=2,
            batch_size=3, microbatch=2, seed=7, lr=1e-3, weight_decay=.01,
            grad_clip=1., coefficient=0., device='cpu')
        assert len(history) == 2
        results.append(model.state_dict())
    assert any(not torch.equal(before[k], results[0][k]) for k in before if k.startswith('cpm_head.'))
    for key in before:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
        if not key.startswith('cpm_head.'):
            torch.testing.assert_close(before[key], results[0][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='300'):
        adaptation.adapt_arm(initial, criterion(), features, updates=301,
            batch_size=3, microbatch=2, seed=7, lr=1e-3, weight_decay=.01,
            grad_clip=1., coefficient=0., device='cpu')


def test_accumulation_and_calibration_rng():
    initial = tiny_model()
    features = cache.extract_batch(initial, batch(), 'cpu')
    model = deepcopy(initial)
    adaptation.freeze_for_cpm(model)
    rng_before = torch.get_rng_state().clone()
    weight, stats = adaptation.calibrate_coefficient(model, criterion(), features,
                                                    torch.arange(3), 2, 'cpu')
    assert weight * stats['frequency_grad_norm'] / stats['original_grad_norm'] == pytest.approx(.1)
    assert torch.equal(rng_before, torch.get_rng_state())
    states = []
    for microbatch in (1, 3):
        model = deepcopy(initial)
        adaptation.adapt_arm(model, criterion(), features, updates=1, batch_size=3,
            microbatch=microbatch, seed=7, lr=1e-3, weight_decay=.01,
            grad_clip=0., coefficient=weight, device='cpu')
        states.append(model.cpm_head.state_dict())
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=2e-5, rtol=1e-4)
    assert [len(x) for x in adaptation.effective_batches(5, 3, 4, 7)] == [3, 2, 3, 2]


def test_subject_metrics_keep_constant_prediction_and_equal_subjects():
    target = np.tile(np.arange(32.), (3, 2, 1))
    pred = target.copy()
    pred[0] = 0
    scores = adaptation.segment_metrics(pred, target, np.zeros((3, 2)), .05)
    assert scores['H17_32']['temporal_pcc'][0] == 0
    assert scores['H17_32']['pcc_valid_fraction'][0] == 0
    assert scores['H17_32']['pcc_target_valid_fraction'][0] == 1
    ids = [cache.task_id('a', 63), cache.task_id('b', 63), cache.task_id('b', 80)]
    summary = adaptation.summarize_subjects([scores], ids)
    assert summary['H17_32']['temporal_pcc']['mean'] == pytest.approx(.5)
    pairs = adaptation.paired_differences(summary, summary, 1)
    assert pairs['H17_32']['mae']['interval_95'] == [0., 0.]


def write_cache(root, features, fingerprint):
    root.mkdir()
    splits = {}
    for split in ('train', 'val'):
        ids = [cache.task_id(f'{split}{i}', 63) for i in range(3)]
        torch.save(features, root / f'{split}.pt')
        splits[split] = {'shards': [{'file': f'{split}.pt', 'ids': ids}],
                         'active_ids': ids, 'subjects_limit': 64, 'anchors_per_subject': 4}
    (root / 'manifest.json').write_text(json.dumps({
        'version': 1, 'checkpoint_sha256': fingerprint, 'split_hash': 'synthetic', 'splits': splits}))


def test_synthetic_pair_end_to_end_and_evaluation_only(tmp_path, monkeypatch):
    model = tiny_model()
    features = cache.extract_batch(model, batch(), 'cpu')
    checkpoint = tmp_path / 'synthetic.pt'
    checkpoint.write_bytes(b'synthetic checkpoint identity')
    root = tmp_path / 'cache'
    write_cache(root, features, cache.sha256(checkpoint))
    monkeypatch.setattr(adaptation, 'load_model', lambda *a: (deepcopy(model), settings(), {}))
    output = tmp_path / 'pair'
    adaptation.run_pair(checkpoint, root, output, 'cpu', updates=2, batch_size=3, microbatch=2)
    report = json.loads((output / 'report.json').read_text())
    assert report['progression']['status'] == 'screening_only'
    assert report['n_subjects'] == 3
    assert set(report['arms']) == {'original_ode', 'original_cpm', 'control', 'frequency'}
    assert report['experiment']['updates'] == 2
    evaluation = tmp_path / 'evaluation'
    adaptation.run_pair(checkpoint, root, evaluation, 'cpu', pair=output / 'pair.pt')
    assert not (evaluation / 'pair.pt').exists()
    assert json.loads((evaluation / 'report.json').read_text())['arms'] == report['arms']
    with pytest.raises(FileExistsError):
        adaptation.run_pair(checkpoint, root, output, 'cpu')


def test_active_cache_selection_and_dry_runs(tmp_path):
    model = tiny_model()
    features = cache.extract_batch(model, batch(), 'cpu')
    root = tmp_path / 'cache'
    write_cache(root, features, 'fingerprint')
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['splits']['val']['active_ids'] = manifest['splits']['val']['active_ids'][1:]
    path.write_text(json.dumps(manifest))
    chunks = list(cache.cache_batches(root, 'val'))
    assert len(chunks[0][0]['tokens']) == 2
    with pytest.raises(ValueError, match='SHA'):
        adaptation.validate_cache(root, 'wrong')
    missing = tmp_path / 'never-created'
    cache.main(['--out-dir', str(missing)])
    adaptation.main(['--cache', '/missing', '--out-dir', str(missing)])
    assert not missing.exists()


def test_cache_extension_reuses_shards_and_replaces_active_selection(tmp_path, monkeypatch):
    from analysis import next_point_eval

    checkpoint = tmp_path / 'identity.pt'
    checkpoint.write_bytes(b'cache extension test')
    split_file = tmp_path / 'split.json'
    split_file.write_text('{}')
    args = settings()
    args.data_root, args.split_file = str(tmp_path), str(split_file)
    args.sampling_seed, args.num_rois, args.tfm_patch_len, args.tfm_dim = 17, 3, 8, 8
    args.mode = 'finetune'
    model = tiny_model()
    monkeypatch.setattr(cache, 'load_model', lambda *a: (model, args, {}))
    monkeypatch.setattr(cache, 'load_condition_bundle', lambda *a: None)
    owner = SimpleNamespace(train_subjects=['t0', 't1'], val_subjects=['v0', 'v1'],
                            test_subjects=['untouched_test'])

    class View:
        def __init__(self, split):
            names = owner.train_subjects if split == 'train' else owner.val_subjects
            self.base = SimpleNamespace(subject_ids=names)
            self.tasks = [(pos, 63 + h, 0) for pos in range(2)
                          for h in range(args.eval_anchors_per_subject)]

        def __len__(self):
            return len(self.tasks)

        def __getitem__(self, i):
            pos, cutoff, _ = self.tasks[i]
            return {'subj_id': self.base.subject_ids[pos], 'cutoff': cutoff}

    def loader(args, split, **kwargs):
        return SimpleNamespace(dataset=View(split)), [], owner

    extracted = []

    def encode(model, data, device):
        extracted.extend(zip(data['subj_id'], data['cutoff'].tolist()))
        return {'tokens': torch.zeros(len(data['subj_id']), 3, 8, 8)}

    monkeypatch.setattr(next_point_eval, 'build_next_point_loader', loader)
    monkeypatch.setattr(cache, 'extract_batch', encode)
    root = tmp_path / 'cache'
    cache.extract(checkpoint, root, ['train', 'val'], 'cpu', subjects=1, anchors=1)
    manifest = json.loads((root / 'manifest.json').read_text())
    original_train = manifest['splits']['train']
    assert len(extracted) == 2
    cache.extract(checkpoint, root, ['val'], 'cpu', subjects=0, anchors=2, extend=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['splits']['train'] == original_train
    assert len(manifest['splits']['val']['active_ids']) == 4
    assert len(extracted) == 5  # Original val anchor encoded only once.
    assert len(extracted) == len(set(extracted))
    cache.extract(checkpoint, root, ['val'], 'cpu', subjects=1, anchors=1, extend=True)
    assert len(extracted) == 5
    assert sum(len(ids) for _, ids in cache.cache_batches(root, 'val')) == 1


def test_full_val_gate_enforces_short_and_long_guards():
    arms = {name: {} for name in ('original_ode', 'original_cpm', 'control', 'frequency')}
    for name, segments in arms.items():
        for segment in ('H1', 'H1_8', 'H17_32', 'H1_32'):
            segments[segment] = {'mae': {'mean': 1.},
                                 'temporal_pcc': {'mean': .1},
                                 'direction_accuracy_step': {'mean': .6}}
    arms['frequency']['H17_32']['temporal_pcc']['mean'] = .14
    arms['frequency']['H17_32']['direction_accuracy_step']['mean'] = .65
    assert adaptation.progression_gate(arms, False)['status'] == 'screening_only'
    assert adaptation.progression_gate(arms, True)['status'] == 'numeric_gates_passed_review_subjects'
    arms['frequency']['H1']['mae']['mean'] = 1.02
    gate = adaptation.progression_gate(arms, True)
    assert gate['status'] == 'stop' and not gate['checks']['H1_mae_guard']
    arms['control']['H17_32']['temporal_pcc']['mean'] = .15
    assert not adaptation.progression_gate(arms, True)['checks']['late_pcc_vs_control']
