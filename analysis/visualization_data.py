"""Real-data computations for the PNG-only visualization entry point.

Native loading, frozen splits, prefix preprocessing and checkpoint reconstruction
come from the existing study interfaces. No fitting changes the neural model.
"""

import numpy as np
import pandas as pd
import torch
from scipy import signal, stats

from analysis.study_adapter import NativeAdapter, RidgeAdapter
from analysis.study_data import native_subjects, window_anchors
from analysis.study_metrics import fc_matrix
from analysis.study_stats import association, bh_fdr, pearson

HORIZONS = np.array([1, 2, 4, 8])


def predict_batch(adapter, tasks, horizon, batch_size=16):
    """tasks: (Subject, cutoff, optional replacement history); native ODE output."""
    means, stds = [], []
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        samples = [
            adapter.tensors(s, t, adapter.context, history=h) for s, t, h in chunk
        ]
        x = torch.cat([v[0] for v in samples])
        sc = torch.cat([v[1] for v in samples])
        kwargs = {}
        for key in samples[0][2]:
            if key == "condition":
                kwargs[key] = {
                    k: torch.cat([v[2][key][k] for v in samples])
                    for k in samples[0][2][key]
                }
            else:
                kwargs[key] = torch.cat([v[2][key] for v in samples])
        with torch.inference_mode():
            if adapter.model.personalized:
                profiles = []
                for (s, _, _), sample in zip(chunk, samples):
                    if s.subject_id not in adapter._profiles:
                        adapter._profiles[s.subject_id] = (
                            adapter.model.calibrate_profile(
                                sample[2]["calibration_bold"]
                            )
                        )
                    profiles.append(adapter._profiles[s.subject_id])
                kwargs["profile"] = torch.cat(profiles)
                kwargs.pop("calibration_bold")
            mean, std = adapter.forward_tensor(
                x, sc, kwargs, horizon, "ode_integration"
            )
        means.append(mean.cpu().numpy())
        stds.append(std.cpu().numpy())
        if len(tasks) > 256 and (start // batch_size) % 64 == 0:
            print(
                f"  ODE inference: {min(start + batch_size, len(tasks))}/{len(tasks)} windows, H={horizon}",
                flush=True,
            )
    mean, std = np.concatenate(means), np.concatenate(stds)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(
            "Nonfinite ODE trajectory or invalid native predictive standard deviation"
        )
    return mean, std


def covariates(subjects):
    """Available recorded confounds; Dataset is acquisition version, not site."""
    return np.array(
        [
            [
                s.metadata.get("Age"),
                s.metadata.get("Sex"),
                s.metadata.get("education"),
                float(s.metadata.get("Dataset") == "V2"),
            ]
            for s in subjects
        ],
        dtype=float,
    )


def column_correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide(
        (a * b).sum(0),
        denominator,
        out=np.full(a.shape[1], np.nan),
        where=denominator > 1e-20,
    )


def roi_features(x):
    """Nontrivial amplitude in prefix units, successive variation and lag-1 ACF."""
    return np.stack(
        [
            np.log(np.maximum(x.std(0, ddof=1), 1e-12)),
            np.sqrt(np.mean(np.diff(x, axis=0) ** 2, axis=0)),
            column_correlation(x[:-1], x[1:]),
        ]
    )


def fisher_edges(x):
    fc = fc_matrix(x)
    return np.arctanh(np.clip(fc[np.triu_indices(x.shape[1], 1)], -0.999999, 0.999999))


def effects(a, b):
    """MDD minus HC Hedges g, Welch p and BH q over one feature family."""
    na, nb = np.isfinite(a).sum(0), np.isfinite(b).sum(0)
    var = (
        (na - 1) * np.nanvar(a, axis=0, ddof=1)
        + (nb - 1) * np.nanvar(b, axis=0, ddof=1)
    ) / np.maximum(na + nb - 2, 1)
    g = (np.nanmean(a, 0) - np.nanmean(b, 0)) / np.sqrt(np.maximum(var, 1e-24))
    g *= 1 - 3 / np.maximum(4 * (na + nb - 2) - 1, 1)
    g = np.where((na >= 3) & (nb >= 3) & (var > 1e-12), g, np.nan)
    p = stats.ttest_ind(a, b, equal_var=False, axis=0, nan_policy="omit").pvalue
    return g, bh_fdr(np.where(np.isfinite(g), p, np.nan))


def adjusted_effects(y, subjects):
    """Vectorized HC3 OLS; standardized group coefficient, complete cases."""
    c = covariates(subjects)
    ok = np.isfinite(c).all(1) & np.isfinite(y).all(1)
    c, y = c[ok], y[ok]
    group = np.array([s.diagnosis == "MDD" for s in subjects])[ok]
    design = np.column_stack([np.ones(len(c)), group, c])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Recorded demographic adjustment design is rank deficient")
    y = (y - y.mean(0)) / y.std(0, ddof=1)
    inv = np.linalg.inv(design.T @ design)
    beta = inv @ design.T @ y
    residual = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, inv, design)
    influence = (design @ inv[:, 1])[:, None] * residual / (1 - leverage[:, None])
    se = np.sqrt((influence**2).sum(0))
    p = 2 * stats.t.sf(np.abs(beta[1] / se), len(c) - design.shape[1])
    return beta[1], bh_fdr(p), int(ok.sum())


def network_mean(values, network_labels, names):
    return np.stack(
        [np.nanmean(values[..., network_labels == n], axis=-1) for n in names], axis=-1
    )


def network_edges(edges, network_labels, names):
    i, j = np.triu_indices(len(network_labels), 1)
    pairs = [(a, b) for k, a in enumerate(names) for b in names[k:]]
    masks = [
        ((network_labels[i] == a) & (network_labels[j] == b))
        | ((network_labels[i] == b) & (network_labels[j] == a))
        for a, b in pairs
    ]
    return np.stack([np.nanmean(edges[..., m], axis=-1) for m in masks], -1), pairs


def normative(train, values, train_subjects, test_subjects):
    """HC-train demographic reference; same observed reference for generated data."""
    tc, ec = covariates(train_subjects), covariates(test_subjects)
    valid = np.isfinite(tc).all(1) & np.isfinite(train).all(1)
    design = np.column_stack([np.ones(valid.sum()), tc[valid]])
    beta = np.linalg.lstsq(design, train[valid], rcond=None)[0]
    residual = train[valid] - design @ beta
    scale = np.sqrt(
        (residual**2).sum(0) / (len(residual) - np.linalg.matrix_rank(design))
    )
    usable = np.isfinite(ec).all(1)
    z = np.full(values.shape, np.nan)
    z[usable] = (
        values[usable] - np.column_stack([np.ones(usable.sum()), ec[usable]]) @ beta
    ) / np.where(scale > 1e-10, scale, np.nan)
    return z, beta, scale, int(valid.sum())


def load_inputs(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(2024)
    np.random.seed(2024)
    base = NativeAdapter(args.hc_checkpoint, args.device)
    fine = NativeAdapter(args.mdd_checkpoint, args.device)
    if any(a.model.dynamics_mode != "graph_ode" for a in (base, fine)):
        raise ValueError("This entry point requires the existing Graph-ODE checkpoints")
    config = dict(
        data_root=str(args.data_root),
        cohorts=["HC", "MDD"],
        max_subjects=dict(train=0, val=1, test=args.max_subjects),
        seed=2024,
    )
    subjects, _ = native_subjects(config, base)
    fine_config = {
        **config,
        "cohorts": ["MDD"],
        "max_subjects": dict(train=1, val=1, test=args.max_subjects),
    }
    fine_subjects, _ = native_subjects(fine_config, fine)
    train = [s for s in subjects if s.split == "train"]
    test = [s for s in subjects if s.split == "test"]
    mdd_fine = [s for s in fine_subjects if s.split == "test"]
    atlas = pd.read_excel(args.data_root / "AAL116.xlsx")
    labels = np.asarray(test[0].roi_labels)
    networks = atlas.iloc[:, -1].astype(str).to_numpy()
    names = np.array(list(dict.fromkeys(networks)))
    if len(networks) != len(labels) or not np.array_equal(
        atlas["abbr"].astype(str), labels
    ):
        raise ValueError("Atlas row order must match native AAL116 ROI order")
    return dict(
        base=base,
        fine=fine,
        train=train,
        test=test,
        mdd_fine=mdd_fine,
        labels=labels,
        networks=networks,
        names=names,
        order=np.concatenate([np.flatnonzero(networks == n) for n in names]),
        args=args,
    )


def compute_forecasts(d):
    test, base, fine, args = d["test"], d["base"], d["fine"], d["args"]
    ridge = RidgeAdapter().fit(d["train"])
    tasks, owners, targets, persistent, ridge_pred = [], [], [], [], []
    for i, s in enumerate(test):
        for t in window_anchors(s, base.context, 8, base.calibration, args.anchors):
            tasks.append((s, t, None))
            owners.append(i)
            targets.append(s.bold[t + 1 : t + 9])
            persistent.append(np.repeat(s.bold[t][None], 8, axis=0))
            ridge_pred.append(ridge.predict(s.bold[: t + 1], 8).mean)
    print(
        f"Forecast: {len(test)} test subjects, {len(tasks)} windows; frozen train-only Ridge",
        flush=True,
    )
    pred, std = predict_batch(base, tasks, 8, args.batch_size)
    predictions = {
        "Common HC model": pred,
        "Persistence": np.array(persistent),
        "Ridge AR(1)": np.array(ridge_pred),
    }
    fine_lookup = {s.subject_id: s for s in d["mdd_fine"]}
    midx = np.array([i for i, (s, _, _) in enumerate(tasks) if s.diagnosis == "MDD"])
    fine_tasks = [
        (fine_lookup[tasks[i][0].subject_id], tasks[i][1], None) for i in midx
    ]
    fp, fs = predict_batch(fine, fine_tasks, 8, args.batch_size)
    full_fp, full_fs = np.full(pred.shape, np.nan), np.full(std.shape, np.nan)
    full_fp[midx], full_fs[midx] = fp, fs
    predictions["MDD fine-tuned"] = full_fp
    owner, target = np.array(owners), np.array(targets)
    metrics = {}
    for name, p in predictions.items():
        subject_metrics = []
        for i in range(len(test)):
            pp, yy = p[owner == i], target[owner == i]
            if not np.isfinite(pp).any():
                subject_metrics.append(np.full((8, 3), np.nan))
                continue
            delta = pp - yy
            subject_metrics.append(
                np.stack(
                    [
                        np.nanmean(np.abs(delta), (0, 2)),
                        np.sqrt(np.nanmean(delta**2, (0, 2))),
                        np.array([pearson(pp[:, h], yy[:, h]) for h in range(8)]),
                    ],
                    -1,
                )
            )
        metrics[name] = np.array(subject_metrics)[:, HORIZONS - 1]
    roi_error = np.array(
        [
            np.abs(pred[owner == i] - target[owner == i]).mean((0, 1))
            for i in range(len(test))
        ]
    )
    d.update(
        tasks=tasks,
        owner=owner,
        target=target,
        predictions=predictions,
        std=std,
        fine_std=full_fs,
        metrics=metrics,
        roi_error=roi_error,
    )


def compute_dynamics(d):
    base, test = d["base"], d["test"]
    context = base.context
    length = min(len(s.bold) - context for s in test)
    if length < 64:
        raise ValueError(
            "At least 64 consecutive unobserved future TRs are required for dynamics"
        )
    print(
        f"Dynamics: one uninterrupted {length}-TR ODE trajectory per test subject (no teacher forcing)",
        flush=True,
    )
    generated, _ = predict_batch(
        base,
        [(s, context - 1, None) for s in test],
        length,
        min(8, d["args"].batch_size),
    )
    real = np.array([s.bold[context : context + length] for s in test])
    features = {
        k: np.array([roi_features(x) for x in seq])
        for k, seq in [("Real", real), ("Generated", generated)]
    }
    edges = {
        k: np.array([fisher_edges(x) for x in seq])
        for k, seq in [("Real", real), ("Generated", generated)]
    }
    acf, psd, dfc = {}, {}, {}
    lags = np.arange(21)
    for key, seq in [("Real", real), ("Generated", generated)]:
        acf[key] = np.array(
            [
                [1.0]
                + [
                    np.nanmean(column_correlation(x[:-lag], x[lag:]))
                    for lag in lags[1:]
                ]
                for x in seq
            ]
        )
        freq, spectrum = signal.welch(
            seq, fs=1.0, axis=1, nperseg=64, detrend="constant"
        )
        psd[key] = spectrum.mean(-1)
        dfc[key] = np.array(
            [
                [fisher_edges(x[t : t + 32]) for t in range(0, length - 31, 4)]
                for x in seq
            ]
        )
    groups = np.array([s.diagnosis for s in test])
    group_results = {}
    for key in features:
        group_results[key] = dict(
            roi=[
                effects(
                    features[key][groups == "MDD", k], features[key][groups == "HC", k]
                )
                for k in range(3)
            ],
            fc=effects(edges[key][groups == "MDD"], edges[key][groups == "HC"]),
        )
    adjusted = adjusted_effects(features["Real"][:, 0], test)
    d.update(
        real=real,
        generated=generated,
        length=length,
        features=features,
        edges=edges,
        acf=acf,
        freq=freq,
        psd=psd,
        dfc=dfc,
        groups=groups,
        group_results=group_results,
        adjusted=adjusted,
    )


def compute_normative(d):
    hc_train = [s for s in d["train"] if s.diagnosis == "HC"]
    k, length = d["base"].context, d["length"]
    print(
        f"Normative reference: {len(hc_train)} HC train subjects; observed reference shared by real/generated features",
        flush=True,
    )
    amplitude = np.array([roi_features(s.bold[k : k + length])[0] for s in hc_train])
    edges = np.array([fisher_edges(s.bold[k : k + length]) for s in hc_train])
    z, ze = {}, {}
    for key in ("Real", "Generated"):
        z[key], _, _, n = normative(
            amplitude, d["features"][key][:, 0], hc_train, d["test"]
        )
        ze[key], _, _, _ = normative(edges, d["edges"][key], hc_train, d["test"])
    d.update(z=z, ze=ze, normative_n=n)


def compute_explanations(d):
    subjects, adapter = d["mdd_fine"], d["fine"]
    importance = []
    for i, s in enumerate(subjects):
        anchor = adapter.context - 1
        history = s.bold[: adapter.context]
        tasks = [(s, anchor, None)]
        for roi in range(len(d["labels"])):
            h = history.copy()
            h[:, roi] = history[:, roi].mean()
            tasks.append((s, anchor, h))
        pred, _ = predict_batch(adapter, tasks, 8, d["args"].batch_size)
        target = s.bold[anchor + 1 : anchor + 9]
        error = np.abs(pred - target).mean((1, 2))
        importance.append(error[1:] - error[0])
        if i % 25 == 0:
            print(
                f"ROI occlusion: {i + 1}/{len(subjects)} MDD subjects, all 116 ROIs",
                flush=True,
            )
    d["importance"] = np.array(importance)


def compute_clinical(d):
    m = d["groups"] == "MDD"
    subjects = [s for s in d["test"] if s.diagnosis == "MDD"]
    values = d["features"]["Real"][m, 0]
    ne, pairs = network_edges(d["edges"]["Real"][m], d["networks"], d["names"])
    deviation = np.stack(
        [
            np.sqrt(np.nanmean(d["z"]["Real"][m] ** 2, 1)),
            np.mean(np.abs(d["z"]["Real"][m]) > 2, 1),
            np.sqrt(np.nanmean(d["ze"]["Real"][m] ** 2, 1)),
        ],
        -1,
    )
    deviation[~np.isfinite(d["z"]["Real"][m]).any(1)] = np.nan
    scales = np.array(
        [[s.metadata.get(k) for k in ("HAMD", "HAMA")] for s in subjects], float
    )
    cov = covariates(subjects)
    results = []
    clinical_covariates = []
    for y in scales.T:
        complete = np.isfinite(y) & np.isfinite(cov).all(1)
        # HAMA is recorded in only one data version in this test cohort.
        # A constant version column duplicates the intercept; it is not a
        # separately estimable adjustment and must not invalidate the analysis.
        varying = np.ptp(cov[complete], axis=0) > 1e-12
        c = cov[:, varying]
        clinical_covariates.append(c)
        row = {}
        for family, xx in [
            ("ROI", values),
            ("Network FC", ne),
            ("Deviation", deviation),
        ]:
            rr = [association(x, y, covariates=c) for x in xx.T]
            row[family] = dict(
                r=np.array([v["effect"] for v in rr]),
                q=bh_fdr([v["p"] for v in rr]),
                n=np.array([v["n_a"] for v in rr]),
            )
        results.append(row)
    d.update(
        clinical=results,
        clinical_scales=scales,
        clinical_deviation=deviation,
        clinical_cov=cov,
        clinical_covariates=clinical_covariates,
        clinical_network=ne,
        network_pairs=pairs,
    )


def compute_sensitivity(d):
    adapter = d["fine"]
    results = {}
    for field in ("HAMD", "HAMA"):
        subjects = [s for s in d["mdd_fine"] if s.metadata.get(field) is not None]
        train_values = [
            s.metadata[field]
            for s in d["train"]
            if s.diagnosis == "MDD" and s.metadata.get(field) is not None
        ]
        grid = np.quantile(train_values, np.linspace(0.05, 0.95, 9))
        field_index = [f["name"] for f in adapter.bundle["stats"]["fields"]].index(
            field
        )
        spec = adapter.bundle["stats"]["fields"][field_index]
        outputs = []
        for s in subjects:
            x, sc, kwargs = adapter.tensors(s, adapter.context - 1, adapter.context)
            with torch.inference_mode():
                if adapter.model.personalized:
                    kwargs["profile"] = adapter.model.calibrate_profile(
                        kwargs.pop("calibration_bold")
                    )
                cond = kwargs["condition"]
                curves = []
                for value in grid:
                    c = {key: v.clone() for key, v in cond.items()}
                    c["cond_z"][:, field_index] = (value - spec["mu"]) / spec["scale"]
                    c["cond_mask"][:, field_index] = 1
                    kwargs["condition"] = c
                    p, _ = adapter.forward_tensor(x, sc, kwargs, 8, "ode_integration")
                    curves.append(p[0].cpu().numpy())
            outputs.append(curves)
        curves = np.array(outputs)
        delta = curves - curves[:, 4:5]
        results[field] = dict(
            grid=grid,
            delta=delta,
            curves=curves,
            sensitivity=np.sqrt(np.mean(delta**2, (1, 2))),
            n=len(subjects),
        )
        print(
            f"Clinical condition sweep: {field}, {len(subjects)} subjects with observed field, 9 train-range values",
            flush=True,
        )
    d["sensitivity"] = results
