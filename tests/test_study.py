"""Deterministic study contracts; all fixture subjects are explicitly synthetic."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis.run_study import load_config, main, run
from analysis.study_adapter import (
    NativeAdapter,
    Prediction,
    RidgeAdapter,
    SyntheticAdapter,
)
from analysis.study_data import (
    Subject,
    synthetic_subjects,
    validate_splits,
    window_anchors,
)
from analysis.study_metrics import (
    continuous_metrics,
    fc_matrix,
    feature_vector,
    gaussian_metrics,
    point_metrics,
)
from analysis.study_stages import (
    ReferenceView,
    clinical,
    covariate_design,
    explainability,
    ig_capability,
    normative,
    uncertainty,
)
from analysis.study_stats import (
    adjusted_group,
    bh_fdr,
    bootstrap_ci,
    comparison,
    pearson,
    subject_aggregate,
)
from analysis.twin_reference import fit_reference, score_reference


def fixture_subjects():
    c = load_config(None)
    c.update(source="synthetic", bootstrap_repeats=20)
    return synthetic_subjects(c)[0]


def test_split_subject_leakage_and_duplicate_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(dict(train=["a"], val=["a"], test=[]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits(dict(train=["a", "a"], val=[], test=[]))
    assert len(validate_splits(dict(train=["a"], val=["b"], test=["c"]))) == 3


def test_window_horizon_and_prefix_boundaries():
    s = fixture_subjects()[0]
    anchors = window_anchors(s, 16, 8, 32, 999)
    assert anchors[0] == 31 and anchors[-1] == 141
    for t in anchors:
        assert len(s.bold[t - 15 : t + 1]) == 16
        assert np.array_equal(s.bold[t + 1 : t + 9][-1], s.bold[t + 8])


def test_ridge_train_only_and_test_isolation():
    subjects = fixture_subjects()
    train = [s for s in subjects if s.split == "train"]
    model = RidgeAdapter().fit(train)
    weights = model.weights.copy()
    for s in subjects:
        if s.split == "test":
            s.bold[:] = 1e6
    assert np.array_equal(RidgeAdapter().fit(train).weights, weights)
    with pytest.raises(ValueError, match="train"):
        RidgeAdapter().fit(subjects)


def test_prediction_shape_roi_order_and_modes():
    s = fixture_subjects()[0]
    a = SyntheticAdapter()
    p = a.predict(s, 63, 8, "autoregressive")
    assert p.mean.shape == (8, len(s.roi_labels))
    np.testing.assert_allclose(p.mean[0], s.bold[63] * 0.5)
    with pytest.raises(NotImplementedError):
        a.predict(s, 63, 8, "direct_cpm")


def test_native_layout_preserves_internal_calibration_contract():
    import torch

    sample = dict(
        x=torch.zeros(4, 1, 64),
        sc=torch.eye(4),
        calibration_bold=torch.arange(128.0).reshape(4, 1, 32),
    )
    ds = SimpleNamespace(_subject_pos={"s": 0}, make_sample=lambda *args: sample)
    s = Subject(
        "s",
        "HC",
        "test",
        np.zeros((150, 4)),
        ("a", "b", "c", "d"),
        dataset=ds,
        native_id="s",
    )
    adapter = NativeAdapter.__new__(NativeAdapter)
    adapter.device = torch.device("cpu")
    x, sc, kw = adapter.tensors(s, 63, 64)
    assert x.shape == (1, 4, 1, 64) and sc.shape == (1, 4, 4)
    assert kw["calibration_bold"].shape == (1, 4, 1, 32)
    np.testing.assert_array_equal(
        kw["calibration_bold"][0, 2, 0], sample["calibration_bold"][2, 0]
    )


def test_native_raw_prefix_normalization_ignores_future():
    from utils.dataloader import NeuroTwinNextPointDataset

    raw = np.random.default_rng(1).normal(size=(150, 4)).astype("float32")
    ds = NeuroTwinNextPointDataset.__new__(NeuroTwinNextPointDataset)
    ds.bold_source = "raw"
    ds.num_rois = 4
    ds.calibration_length = 32
    ds._source_logged = True
    ds._load_raw_array = lambda sid: raw
    a, stats = ds._load_series("s", 4)
    raw[64:] += 10000
    b, changed = ds._load_series("s", 4)
    for left, right in zip(stats, changed):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a[:, :64], b[:, :64])


@pytest.mark.parametrize("dynamics_mode", ["graph_ode"])
def test_actual_tiny_model_adapter_forward_and_randomization(dynamics_mode):
    import torch

    from models.tfm import NeuroTwinTFM

    torch.manual_seed(1)
    a = NativeAdapter.__new__(NativeAdapter)
    a.device = torch.device("cpu")
    a.context = 64
    a._profiles = {}
    a.model = NeuroTwinTFM(
        features=4,
        context_max=64,
        patch_len=4,
        dim=16,
        num_layers=1,
        num_heads=2,
        cpm_horizon=8,
        cpm_layers=1,
        dropout=0.0,
        pretrain_mode=True,
        dynamics_mode=dynamics_mode,
        graph_source="history_fc",
        personalized=True,
    ).eval()
    s = fixture_subjects()[0]
    sample = dict(
        x=torch.from_numpy(s.bold[:64].T[:, None, :].copy()),
        sc=torch.eye(4),
        calibration_bold=torch.from_numpy(s.bold[:32].T[:, None, :].copy()),
    )
    s.native_id = "s"
    s.dataset = SimpleNamespace(_subject_pos={"s": 0}, make_sample=lambda *args: sample)
    for mode, h in [("one_step", 1), ("direct_cpm", 8), ("autoregressive", 4)]:
        result = a.predict(s, 63, h, mode)
        assert result.mean.shape == (h, 4) and np.isfinite(result.mean).all()
    result = a.predict(s, 63, 8, "ode_integration")
    assert result.std.shape == (8, 4) and (result.std > 0).all()
    assert len(a._profiles) == 1
    state = torch.random.get_rng_state().clone()
    old = {k: v.clone() for k, v in a.model.state_dict().items()}
    randomized = a.randomize(99)
    assert not randomized._profiles and torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(old[k], v) for k, v in a.model.state_dict().items())


def test_known_metrics_and_undefined_correlations():
    m = point_metrics(np.array([1.0, 3.0]), np.array([0.0, 1.0]))
    assert m["mae"] == 1.5 and m["rmse"] == pytest.approx(np.sqrt(2.5))
    assert np.isnan(m["pearson"])
    assert np.isnan(pearson([1, 1, 1], [1, 2, 3]))
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        point_metrics(np.ones(3), np.ones(4))


def test_subject_equal_weight_aggregation():
    d = pd.DataFrame(dict(subject_id=["a", "a", "b"], metric=[1.0, 3.0, 9.0]))
    agg = subject_aggregate(d, [], ["metric"])
    assert agg.metric.tolist() == [2.0, 9.0]
    assert agg.metric.mean() == 5.5


def test_bootstrap_is_reproducible_and_stratified():
    a = bootstrap_ci([1, 2, 3, 4], 2024, 100, ["a", "a", "b", "b"])
    assert a == bootstrap_ci([1, 2, 3, 4], 2024, 100, ["a", "a", "b", "b"])
    assert 1 <= a[0] <= a[1] <= 4


def test_bh_fdr_known_answer_and_missing():
    np.testing.assert_allclose(
        bh_fdr([0.01, 0.04, 0.03, np.nan]), [0.03, 0.04, 0.04, np.nan], equal_nan=True
    )


def test_fc_symmetry_constants_and_short_guard():
    x = np.random.default_rng(2).normal(size=(64, 4))
    x[:, 0] = 0
    fc = fc_matrix(x)
    np.testing.assert_allclose(fc, fc.T, equal_nan=True)
    assert np.isnan(fc[0]).all()
    assert np.allclose(np.diag(fc)[1:], 1.0)
    assert continuous_metrics(x[:8], x[:8], 64) is None
    m = continuous_metrics(x[:, 1:], x[:, 1:], 64)
    assert m["fc_mae"] == 0 and m["psd_log_mae"] == 0


def test_reference_short_subject_alignment_and_test_isolation():
    train = [
        s for s in fixture_subjects() if s.diagnosis == "HC" and s.split == "train"
    ]
    short = train[1]
    short.bold = short.bold[:20]
    short.metadata["Age"] = 9999
    view = ReferenceView(train)
    payload = fit_reference(view, [s.subject_id for s in train])
    assert payload["n_subjects"] == len(train) - 1
    assert short.subject_id not in payload["subjects"]
    assert payload["demo"]["Age"]["mu"] < 100
    score = score_reference(train[0].bold[:64], train[0].metadata, payload)
    assert score["normal_residual"].shape == (1, 4, 6)


def test_normative_requires_hc_train_and_only_scores_held_out(tmp_path):
    subjects = fixture_subjects()
    c = load_config(None)
    ctx = dict(subjects=subjects, config=c, out=tmp_path)
    state, _ = normative(ctx)
    assert state == "IMPLEMENTED_AND_TESTED"
    scored = set(ctx["normative"].subject_id)
    assert not scored & {s.subject_id for s in subjects if s.split != "test"}
    saved = np.load(tmp_path / "hc_reference.npz")
    assert saved["W"].shape[1] == 24


def test_clinical_missing_scales_skip(tmp_path):
    subjects = fixture_subjects()
    for s in subjects:
        s.metadata["HAMD"] = None
        s.metadata["HAMA"] = None
    ctx = dict(
        subjects=subjects,
        config={**load_config(None), "bootstrap_repeats": 20},
        out=tmp_path,
        adapter=SyntheticAdapter(),
    )
    state, _ = clinical(ctx)
    assert state == "MISSING_DATA"


def test_uncertainty_standard_deviation_semantics():
    m = gaussian_metrics(np.zeros((3, 4)), np.zeros((3, 4)), np.ones((3, 4)) * 2)
    assert m["nll"] == pytest.approx(0.5 * np.log(2 * np.pi) + np.log(2))
    assert m["width95"] == pytest.approx(2 * 1.95996398454 * 2)
    assert m["coverage95"] == 1
    with pytest.raises(ValueError):
        gaussian_metrics(np.zeros(3), np.zeros(3), np.zeros(3))
    assert (
        uncertainty(dict(adapter=SyntheticAdapter(), config={}, out=None))[0]
        == "UNSUPPORTED"
    )


def test_uncertainty_stage_saves_subject_calibration(tmp_path):
    class VarianceFixture(SyntheticAdapter):
        variance_trained = True
        metadata = dict(dynamics_mode="graph_ode")

        def predict(self, subject, anchor, horizon, mode, context=None):
            mean = np.zeros((horizon, len(subject.roi_labels)))
            return Prediction(mean, np.ones_like(mean))

    c = load_config(None)
    c["bootstrap_repeats"] = 20
    ctx = dict(
        subjects=fixture_subjects(), adapter=VarianceFixture(), config=c, out=tmp_path
    )
    state, _ = uncertainty(ctx)
    assert state == "IMPLEMENTED_AND_TESTED"
    cal = pd.read_csv(tmp_path / "uncertainty_calibration.csv")
    assert sorted(cal.nominal.unique()) == [0.5, 0.8, 0.95]
    assert cal.groupby(["subject_id", "horizon", "nominal"]).size().max() == 1


def test_paired_comparison_and_covariate_regression():
    r = comparison([1, 2, 4, 5], [1, 1, 2, 3], repeats=20, paired=True)
    assert r["effect"] == 1.25 and r["n_a"] == 4
    rng = np.random.default_rng(1)
    group = np.repeat([0, 1], 30)
    age = rng.normal(size=60)
    y = 3 * group + age + rng.normal(size=60) * 0.1
    adjusted = adjusted_group(y, group, age[:, None])
    assert adjusted["valid"] and adjusted["effect"] == pytest.approx(3, abs=0.1)


def test_group_adjustment_does_not_drop_entire_hc_for_absent_medication():
    subjects = [s for s in fixture_subjects() if s.split == "test"]
    for s in subjects:
        s.metadata["medication"] = 1.0 if s.diagnosis == "MDD" else None
    design, names = covariate_design(
        subjects, ["Age", "Sex", "medication"], require_both_groups=True
    )
    assert not any(name.startswith("medication") for name in names)
    assert "Age" in names and np.isfinite(design).all()


def test_occlusion_uses_same_roi_subset_across_subjects(tmp_path):
    adapter = NativeAdapter.__new__(NativeAdapter)
    adapter.calibration = 32

    def predict(subject, anchor, horizon, mode, context=None, history=None):
        x = subject.bold[anchor] if history is None else history[-1]
        return Prediction(np.repeat(x[None], horizon, axis=0))

    adapter.predict = predict
    c = load_config(None)
    c.update(
        bootstrap_repeats=10,
        parameter_randomization=False,
        ig_steps=0,
        explain_max_rois=2,
    )
    ctx = dict(subjects=fixture_subjects(), adapter=adapter, config=c, out=tmp_path)
    assert explainability(ctx)[0] == "IMPLEMENTED_AND_TESTED"
    frame = pd.read_csv(tmp_path / "roi_importance_by_subject.csv")
    roi_sets = frame.groupby("subject_id").roi.agg(lambda x: tuple(sorted(set(x))))
    assert len(set(roi_sets)) == 1
    assert (
        frame.groupby(["diagnosis", "roi", "horizon", "baseline"])
        .subject_id.nunique()
        .min()
        == 2
    )


def test_ig_rejects_input_dependent_detached_revin_statistics():
    from models.common import BrainRevIN

    adapter = SimpleNamespace(model=SimpleNamespace(rev_norm=BrainRevIN(4)))
    assert ig_capability(adapter, 16)["status"] == "UNSUPPORTED"
    assert "detach" in ig_capability(adapter, 16)["reason"]
    adapter.model.rev_norm = None
    assert ig_capability(adapter, 16)["status"] == "IMPLEMENTED_NOT_RUN"


def test_feature_order_and_affine_invariance():
    s = fixture_subjects()[0]
    a = feature_vector(s.bold[:64], s.roi_labels)
    b = feature_vector(s.bold[:64] * 3 + 10, s.roi_labels)
    assert list(a) == list(b)
    np.testing.assert_allclose(list(a.values()), list(b.values()), atol=1e-5)


def test_cli_synthetic_all_and_existing_output_refused(tmp_path):
    c = load_config(None)
    c.update(
        source="synthetic",
        bootstrap_repeats=10,
        prediction_modes=["autoregressive"],
        output_root=str(tmp_path),
        run_name="synthetic_test",
    )
    out, status = run(c, "all", tmp_path / "one")
    assert not [k for k, v in status.items() if v["status"] == "FAILED"], status
    assert (out / "performance_by_subject.csv").is_file()
    assert (out / "figure1_performance.png").is_file()
    assert json.loads((out / "provenance.json").read_text())["synthetic"] is True
    assert main(["--stage", "report", "--input-run", str(out)]) == 0
    with pytest.raises(FileExistsError):
        run(c, "audit", out)


def test_missing_checkpoint_recorded(tmp_path):
    c = load_config(None)
    c["checkpoint"] = str(tmp_path / "absent.pt")
    out, status = run(c, "all", tmp_path / "missing")
    assert status["inputs"]["status"] == "SKIPPED_MISSING_CHECKPOINT"
    assert (out / "report.md").is_file()


def test_config_rejects_typo_and_unsafe_name(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"horzons":[1]}')
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    path.write_text('{"run_name":"../escape"}')
    with pytest.raises(ValueError, match="run_name"):
        load_config(path)
