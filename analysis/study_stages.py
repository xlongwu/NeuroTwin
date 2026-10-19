"""Study computations. Each saved inferential row represents independent subjects."""

import json

import numpy as np
import pandas as pd
import torch

from analysis import twin_reference as ref
from analysis.study_adapter import NativeAdapter, RidgeAdapter
from analysis.study_data import window_anchors
from analysis.study_metrics import (
    continuous_metrics,
    fc_matrix,
    feature_vector,
    gaussian_metrics,
    point_metrics,
)
from analysis.study_stats import (
    adjusted_group,
    association,
    bh_fdr,
    bootstrap_ci,
    comparison,
    group_matrix,
    pearson,
)


def write_table(out, name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(out / f"{name}.csv", index=False)
    return frame


def correct(frame, family_columns):
    if frame.empty:
        return frame
    frame["q"] = np.nan
    for _, block in frame.groupby(family_columns, dropna=False):
        frame.loc[block.index, "q"] = bh_fdr(block["p"])
    return frame


def performance(ctx):
    c, out, adapter, subjects = (
        ctx["config"],
        ctx["out"],
        ctx["adapter"],
        ctx["subjects"],
    )
    train = [s for s in subjects if s.split == "train"]
    ridge = RidgeAdapter().fit(train)
    np.savez(out / "ridge_train_fit.npz", weights=ridge.weights)
    rows, roi_rows, examples, skipped = [], [], [], []
    for s in subjects:
        if s.split != c["evaluation_split"]:
            continue
        for context in c["contexts"]:
            hmax = max(c["horizons"])
            anchors = window_anchors(
                s, context, hmax, adapter.calibration, c["anchors_per_subject"]
            )
            for anchor in anchors:
                history = s.bold[anchor - context + 1 : anchor + 1]
                target = s.bold[anchor + 1 : anchor + 1 + hmax]
                predictions = [
                    ("persistence", "constant", np.repeat(history[-1:], hmax, axis=0)),
                    ("ridge_ar1", "autoregressive", ridge.predict(history, hmax).mean),
                ]
                for mode in c["prediction_modes"]:
                    try:
                        horizon = 1 if mode == "one_step" else hmax
                        predictions.append(
                            (
                                "model",
                                mode,
                                adapter.predict(s, anchor, horizon, mode, context).mean,
                            )
                        )
                    except NotImplementedError as exc:
                        skipped.append(dict(mode=mode, reason=str(exc)))
                for name, mode, pred in predictions:
                    for h in c["horizons"]:
                        if h > len(pred):
                            continue
                        # H denotes exact offset t+H, not an average of offsets 1..H.
                        p, t = pred[h - 1], target[h - 1]
                        base = dict(
                            subject_id=s.subject_id,
                            diagnosis=s.diagnosis,
                            site=s.metadata.get("site"),
                            model=name,
                            prediction_mode=mode,
                            context=context,
                            horizon=h,
                            anchor=anchor,
                        )
                        rows.append({**base, **point_metrics(p, t)})
                        roi_rows.extend(
                            {
                                **base,
                                "roi": roi,
                                "abs_error": float(abs(p[i] - t[i])),
                                "squared_error": float((p[i] - t[i]) ** 2),
                            }
                            for i, roi in enumerate(s.roi_labels)
                        )
                if anchors and anchor == anchors[0] and context == c["contexts"][0]:
                    for name, mode, pred in predictions:
                        examples.extend(
                            dict(
                                subject_id=s.subject_id,
                                diagnosis=s.diagnosis,
                                model=name,
                                prediction_mode=mode,
                                offset=h + 1,
                                roi=s.roi_labels[i],
                                predicted=float(pred[h, i]),
                                observed=float(target[h, i]),
                            )
                            for h in range(len(pred))
                            for i in range(min(3, pred.shape[1]))
                        )
    raw = write_table(out, "performance_by_window", rows)
    if raw.empty:
        return "NOT_RUN", "No legal evaluation windows"
    keys = [
        "subject_id",
        "diagnosis",
        "site",
        "model",
        "prediction_mode",
        "context",
        "horizon",
    ]

    def aggregate(block):
        return pd.Series(
            dict(
                mae=block.mae.mean(),
                rmse=np.sqrt(np.mean(block.rmse**2)),
                pearson=block.pearson.mean(),
                valid_pearson=int(block.pearson.notna().sum()),
                n_windows=len(block),
            )
        )

    per = (
        raw.groupby(keys, dropna=False)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    write_table(out, "performance_by_subject", per)
    roi = (
        pd.DataFrame(roi_rows)
        .groupby(keys + ["roi"], dropna=False)[["abs_error", "squared_error"]]
        .mean()
        .reset_index()
    )
    roi["rmse"] = np.sqrt(roi.pop("squared_error"))
    write_table(out, "performance_by_roi", roi)
    summary = []
    for group, block in per.groupby(
        ["model", "prediction_mode", "context", "horizon", "diagnosis"], dropna=False
    ):
        for metric in ("mae", "rmse", "pearson"):
            lo, hi = bootstrap_ci(
                block[metric],
                c["seed"],
                c["bootstrap_repeats"],
                block["site"].fillna("unknown").to_numpy(),
            )
            summary.append(
                dict(
                    zip(
                        ["model", "prediction_mode", "context", "horizon", "diagnosis"],
                        group,
                    ),
                    metric=metric,
                    mean=block[metric].mean(),
                    ci_low=lo,
                    ci_high=hi,
                    n_subjects=len(block),
                    n_valid=int(block[metric].notna().sum()),
                    unit="subject",
                    method="site-stratified subject bootstrap",
                )
            )
    write_table(out, "performance_summary", summary)
    paired = []
    for (mode, k, h, diagnosis), block in per[per.model == "model"].groupby(
        ["prediction_mode", "context", "horizon", "diagnosis"]
    ):
        for baseline in ("persistence", "ridge_ar1"):
            b = per[
                (per.model == baseline)
                & (per.context == k)
                & (per.horizon == h)
                & (per.diagnosis == diagnosis)
            ]
            merged = block.merge(b, on="subject_id", suffixes=("_m", "_b"))
            for metric in ("mae", "rmse"):
                result = comparison(
                    merged[f"{metric}_m"],
                    merged[f"{metric}_b"],
                    c["seed"],
                    c["bootstrap_repeats"],
                    paired=True,
                )
                paired.append(
                    dict(
                        metric=metric,
                        comparison=f"model:{mode} minus {baseline}",
                        context=k,
                        horizon=h,
                        diagnosis=diagnosis,
                        family="baseline_comparison",
                        **result,
                    )
                )
    write_table(out, "baseline_comparison", correct(pd.DataFrame(paired), ["family"]))
    write_table(out, "bold_examples", examples)
    ctx["performance"] = per
    ctx["prediction_skips"] = skipped
    return (
        "IMPLEMENTED_AND_TESTED",
        f"{len(per)} subject/model/horizon rows; exact-offset metrics; {len(skipped)} unsupported calls",
    )


def dynamics(ctx):
    c, out, adapter = ctx["config"], ctx["out"], ctx["adapter"]
    h = c["dynamics_horizon"]
    if h < c["min_dynamics_length"]:
        return (
            "UNSUPPORTED",
            f"Continuous H={h} shorter than minimum {c['min_dynamics_length']}; no stitched FC/PSD/dFC",
        )
    rows, matched_features = [], []
    for s in ctx["subjects"]:
        if s.split != c["evaluation_split"]:
            continue
        anchors = window_anchors(s, c["contexts"][0], h, adapter.calibration, 1)
        for anchor in anchors:
            prediction = adapter.predict(
                s, anchor, h, "autoregressive", c["contexts"][0]
            ).mean
            target = s.bold[anchor + 1 : anchor + h + 1]
            m = continuous_metrics(
                prediction,
                target,
                c["min_dynamics_length"],
                c["dfc_window"],
                c["dfc_stride"],
                c["tr"],
            )
            rows.append(
                dict(
                    subject_id=s.subject_id,
                    diagnosis=s.diagnosis,
                    prediction_mode="autoregressive",
                    anchor=anchor,
                    **m,
                )
            )
            # Retain exact paired matrices/data for reproducible figure generation.
            np.savez(
                out / f"dynamics_{s.subject_id}.npz",
                predicted=prediction,
                observed=target,
                fc_predicted=fc_matrix(prediction),
                fc_observed=fc_matrix(target),
            )
            if h >= 64:
                for origin, x in [
                    ("predicted", prediction),
                    ("observed_matched", target),
                ]:
                    matched_features.append(
                        dict(
                            subject_id=s.subject_id,
                            diagnosis=s.diagnosis,
                            origin=origin,
                            **feature_vector(x[:64], s.roi_labels, ctx.get("networks")),
                        )
                    )
    write_table(out, "dynamics_by_subject", rows)
    ctx["matched_features"] = write_table(
        out, "matched_dynamics_features", matched_features
    )
    return (
        ("IMPLEMENTED_AND_TESTED", f"{len(rows)} continuous autoregressive segments")
        if rows
        else ("NOT_RUN", "Insufficient remaining TRs")
    )


def features(ctx):
    if "features" in ctx:
        return ctx["features"]
    rows = []
    for s in ctx["subjects"]:
        if len(s.bold) < ref.K_REF:
            continue
        rows.append(
            dict(
                subject_id=s.subject_id,
                diagnosis=s.diagnosis,
                split=s.split,
                **feature_vector(s.bold[:64], s.roi_labels, ctx.get("networks")),
            )
        )
    frame = write_table(ctx["out"], "observed_subject_features", rows)
    ctx["features"] = frame
    return frame


def covariate_design(subjects, names, require_both_groups=False):
    """No inferred site or medication codes. Complete cases only, explicit availability."""
    columns, used = [], []
    for name in names:
        values = [s.metadata.get(name) for s in subjects]
        if all(v is None for v in values):
            continue
        if require_both_groups and any(
            sum(v is not None for s, v in zip(subjects, values) if s.diagnosis == group)
            < 3
            for group in ("HC", "MDD")
        ):
            continue
        if name in ("site", "Sex", "medication"):
            valid = sorted({str(v) for v in values if v is not None})
            for code in valid[1:]:
                columns.append(
                    [np.nan if v is None else float(str(v) == code) for v in values]
                )
                used.append(f"{name}={code}")
        else:
            columns.append([np.nan if v is None else float(v) for v in values])
            used.append(name)
    return np.asarray(columns, float).T if columns else np.empty(
        (len(subjects), 0)
    ), used


def pathology(ctx):
    c, out = ctx["config"], ctx["out"]
    frame = features(ctx)
    frame = frame[frame.split == c["evaluation_split"]].reset_index(drop=True)
    a, b = frame[frame.diagnosis == "MDD"], frame[frame.diagnosis == "HC"]
    if min(len(a), len(b)) < 3:
        return "MISSING_DATA", "Need >=3 held-out HC and MDD subjects"
    cols = [x for x in frame if x.startswith(("roi/", "edge/", "network/"))]
    results = group_matrix(a[cols], b[cols], c["seed"], c["bootstrap_repeats"])
    rows = [
        dict(metric=name, comparison="MDD minus HC", family=name.split("/")[0], **r)
        for name, r in zip(cols, results)
    ]
    write_table(out, "pathology_statistics", correct(pd.DataFrame(rows), ["family"]))
    ordered = [
        next(s for s in ctx["subjects"] if s.subject_id == sid)
        for sid in frame.subject_id
    ]
    design, names = covariate_design(ordered, c["covariates"], require_both_groups=True)
    write_table(
        out,
        "pathology_covariate_availability",
        [
            dict(
                covariate=name,
                diagnosis=group,
                n_observed=sum(
                    s.metadata.get(name) is not None
                    for s in ordered
                    if s.diagnosis == group
                ),
                included_in_design=any(
                    x == name or x.startswith(name + "=") for x in names
                ),
            )
            for name in c["covariates"]
            for group in ("HC", "MDD")
        ],
    )
    if names:
        adjusted = []
        for name in cols:
            r = adjusted_group(
                frame[name], (frame.diagnosis == "MDD").astype(int), design
            )
            adjusted.append(
                dict(
                    metric=name,
                    comparison="MDD minus HC adjusted",
                    family=name.split("/")[0],
                    covariates=";".join(names),
                    **r,
                )
            )
        write_table(
            out, "pathology_adjusted", correct(pd.DataFrame(adjusted), ["family"])
        )
    matched = ctx.get("matched_features")
    if matched is None:
        file = out / "matched_dynamics_features.csv"
        matched = (
            pd.read_csv(file)
            if file.exists() and file.stat().st_size > 3
            else pd.DataFrame()
        )
    fidelity = []
    if not matched.empty:
        for origin, block in matched.groupby("origin"):
            a, b = block[block.diagnosis == "MDD"], block[block.diagnosis == "HC"]
            if min(len(a), len(b)) < 3:
                continue
            r = group_matrix(a[cols], b[cols], c["seed"], c["bootstrap_repeats"])
            fidelity.extend(
                dict(
                    metric=name,
                    origin=origin,
                    comparison="MDD minus HC",
                    family=name.split("/")[0],
                    **row,
                )
                for name, row in zip(cols, r)
            )
    fidelity = correct(pd.DataFrame(fidelity), ["origin", "family"])
    write_table(out, "model_pathology_statistics", fidelity)
    concordance = []
    if not fidelity.empty:
        pivot = fidelity.pivot(index="metric", columns="origin", values="effect")
        if {"predicted", "observed_matched"} <= set(pivot.columns):
            for family, block in pivot.groupby(pivot.index.str.split("/").str[0]):
                block = block.dropna()
                concordance.append(
                    dict(
                        family=family,
                        n_features=len(block),
                        sign_agreement=float(
                            (
                                np.sign(block.predicted)
                                == np.sign(block.observed_matched)
                            ).mean()
                        ),
                        effect_correlation=pearson(
                            block.predicted, block.observed_matched
                        ),
                        interpretation="descriptive spatial agreement, not independent disease detection",
                    )
                )
    write_table(out, "model_pathology_concordance", concordance)
    return (
        "IMPLEMENTED_AND_TESTED",
        "Observed 64-TR features; separate FDR families; matched continuous prediction effects when available (otherwise NOT_RUN)",
    )


class ReferenceView:
    def __init__(self, subjects):
        self.subjects = {s.subject_id: s for s in subjects}
        self.condition_table = self
        from utils.condition_schema import schema_for

        self.specs = schema_for("hc3")

    def align(self, ids):
        values = np.array(
            [
                [
                    self.subjects[sid].metadata.get(name)
                    for name in ("Sex", "Age", "education")
                ]
                for sid in ids
            ],
            float,
        )
        return values, np.isfinite(values).astype("uint8")

    def get_series(self, sid):
        return self.subjects[sid].bold.T


def normative(ctx):
    subjects, c, out = ctx["subjects"], ctx["config"], ctx["out"]
    train = [s for s in subjects if s.split == "train" and s.diagnosis == "HC"]
    if len(train) < c["min_reference_subjects"]:
        return (
            "MISSING_DATA",
            f"Need {c['min_reference_subjects']} train HC for reference",
        )
    payload = ref.fit_reference(ReferenceView(train), [s.subject_id for s in train])
    ref.save_reference(payload, out / "hc_reference.npz")
    rows = []
    for s in subjects:
        if s.split != c["evaluation_split"] or len(s.bold) < 64:
            continue
        scores = ref.score_reference(s.bold[:64], s.metadata, payload)
        for i, roi in enumerate(s.roi_labels):
            for j, name in enumerate(ref.COMPONENT_NAMES):
                rows.append(
                    dict(
                        subject_id=s.subject_id,
                        diagnosis=s.diagnosis,
                        roi=roi,
                        component=name,
                        z=float(scores["normal_residual"][0, i, j]),
                        distance=float(scores["normal_distance"][0, i]),
                    )
                )
    ctx["normative"] = write_table(out, "normative_subject_deviations", rows)
    write_table(
        out,
        "normative_reference_summary",
        [
            dict(
                roi=roi,
                component=name,
                residual_scale=float(payload["s_ref"][i, j]),
                reference_intercept=float(
                    payload["W"][0].reshape(len(train[0].roi_labels), -1)[i, j]
                ),
                near_degenerate=bool(payload["s_ref"][i, j] <= 0.00101),
                n_train_hc=payload["n_subjects"],
                reference_kind="demographic ridge on six fixed 64-TR descriptors; not calibrated percentiles",
            )
            for i, roi in enumerate(train[0].roi_labels)
            for j, name in enumerate(ref.COMPONENT_NAMES)
        ],
    )
    return (
        "IMPLEMENTED_AND_TESTED",
        f"{len(train)} train HC; native twin_reference reused; held-out scores only",
    )


def clinical(ctx):
    frame = features(ctx)
    frame = frame[
        (frame.split == ctx["config"]["evaluation_split"]) & (frame.diagnosis == "MDD")
    ].reset_index(drop=True)
    if frame.empty:
        return "MISSING_DATA", "No held-out MDD"
    subjects = [
        next(s for s in ctx["subjects"] if s.subject_id == sid)
        for sid in frame.subject_id
    ]
    design, names = covariate_design(subjects, ctx["config"]["covariates"])
    candidates = {
        col: frame[col].to_numpy()
        for col in frame
        if col.startswith(("roi/", "network/"))
    }
    if "normative" in ctx:
        dev = ctx["normative"].groupby("subject_id").distance.mean()
        candidates["normative/mean_distance"] = dev.reindex(frame.subject_id).to_numpy()
    if "performance" in ctx:
        perf = ctx["performance"]
        for keys, block in perf.groupby(
            ["model", "prediction_mode", "context", "horizon"]
        ):
            candidates["prediction_error/" + str(keys)] = (
                block.set_index("subject_id").mae.reindex(frame.subject_id).to_numpy()
            )
    rows = []
    c = ctx["config"]
    for scale in ("HAMD", "HAMA"):
        y = np.array([s.metadata.get(scale) for s in subjects], float)
        if np.isfinite(y).sum() < 6:
            continue
        for name, x in candidates.items():
            for adjusted in (False, True) if names else (False,):
                r = association(
                    x,
                    y,
                    c["seed"],
                    c["bootstrap_repeats"],
                    design if adjusted else None,
                )
                rows.append(
                    dict(
                        metric=name,
                        comparison=scale,
                        family=f"{scale}/{name.split('/')[0]}/{adjusted}",
                        covariates=";".join(names) if adjusted else "",
                        evidence="conditioning-dependent exploratory"
                        if name.startswith("prediction_error")
                        and ctx["adapter"].metadata.get("condition_group") == "mdd8"
                        else "cross-sectional observed feature association",
                        **r,
                    )
                )
    write_table(
        ctx["out"], "clinical_associations", correct(pd.DataFrame(rows), ["family"])
    )
    return (
        (
            "IMPLEMENTED_AND_TESTED",
            f"{len(rows)} associations; complete cases; no future symptom prediction",
        )
        if rows
        else ("MISSING_DATA", "No clinical scale with >=6 finite MDD values")
    )


def ig_capability(adapter, steps):
    if steps <= 0:
        return dict(status="NOT_RUN", reason="IG disabled by configuration")
    if adapter.model.rev_norm is not None:
        return dict(
            status="UNSUPPORTED",
            reason="BrainRevIN._get_statistics detaches input-dependent mean/std; autograd omits those derivatives, invalidating IG completeness. Use occlusion; training/inference model unchanged.",
        )
    return dict(
        status="IMPLEMENTED_NOT_RUN",
        reason="Differentiable path eligible; numerical completeness still required",
    )


def explainability(ctx):
    adapter, c, out = ctx["adapter"], ctx["config"], ctx["out"]
    if not isinstance(adapter, NativeAdapter):
        return (
            "UNSUPPORTED",
            "Synthetic fixture has no trainable graph for IG/randomization; performance smoke remains valid",
        )
    mode, h = c["explain_mode"], max(c["horizons"])
    rng = np.random.default_rng(c["seed"])
    selected = []
    for diagnosis in c["cohorts"]:
        group = [
            s
            for s in ctx["subjects"]
            if s.split == c["evaluation_split"] and s.diagnosis == diagnosis
        ]
        selected.extend(group[: c["explain_subjects_per_cohort"]])
    randomized = adapter.randomize(c["seed"]) if c["parameter_randomization"] else None
    ig_state = ig_capability(adapter, c["ig_steps"])
    rows, ig_rows, sanity = [], [], []
    roi_indices = (
        np.arange(len(selected[0].roi_labels)) if selected else np.array([], dtype=int)
    )
    if c["explain_max_rois"]:
        roi_indices = np.sort(
            rng.choice(roi_indices, min(len(roi_indices), c["explain_max_rois"]), False)
        )
    (out / "explainability_roi_scope.json").write_text(
        json.dumps(
            [selected[0].roi_labels[i] for i in roi_indices] if selected else []
        ),
        encoding="utf-8",
    )
    for s in selected:
        context = c["contexts"][0]
        anchors = window_anchors(s, context, h, adapter.calibration, 2)
        for anchor in anchors:
            hist = s.bold[anchor - context + 1 : anchor + 1].copy()
            target = s.bold[anchor + 1 : anchor + h + 1]
            original = adapter.predict(s, anchor, h, mode, context).mean
            randomized_original = (
                randomized.predict(s, anchor, h, mode, context).mean
                if randomized
                else None
            )
            train_mean = np.mean(
                [
                    t.bold[: adapter.calibration].mean(0)
                    for t in ctx["subjects"]
                    if t.split == "train"
                ],
                axis=0,
            )
            for baseline in ("history_mean", "train_prefix_mean"):
                replacement = hist.mean(0) if baseline == "history_mean" else train_mean
                for i in roi_indices:
                    changed = hist.copy()
                    changed[:, i] = replacement[i]
                    pred = adapter.predict(
                        s, anchor, h, mode, context, history=changed
                    ).mean
                    random_pred = (
                        randomized.predict(
                            s, anchor, h, mode, context, history=changed
                        ).mean
                        if randomized
                        else None
                    )
                    for offset in c["horizons"]:
                        effect = (
                            np.abs(pred[offset - 1] - target[offset - 1]).mean()
                            - np.abs(original[offset - 1] - target[offset - 1]).mean()
                        )
                        rows.append(
                            dict(
                                subject_id=s.subject_id,
                                diagnosis=s.diagnosis,
                                roi=s.roi_labels[i],
                                horizon=offset,
                                anchor=anchor,
                                baseline=baseline,
                                importance=float(effect),
                                prediction_mode=mode,
                            )
                        )
                        if random_pred is not None:
                            reffect = (
                                np.abs(
                                    random_pred[offset - 1] - target[offset - 1]
                                ).mean()
                                - np.abs(
                                    randomized_original[offset - 1] - target[offset - 1]
                                ).mean()
                            )
                            sanity.append(
                                dict(
                                    subject_id=s.subject_id,
                                    roi=s.roi_labels[i],
                                    horizon=offset,
                                    anchor=anchor,
                                    baseline=baseline,
                                    trained_effect=effect,
                                    randomized_effect=reffect,
                                    randomization="partial registered-module reset; free parameters retained",
                                )
                            )
            if ig_state["status"] == "IMPLEMENTED_NOT_RUN":
                x, sc, kwargs = adapter.tensors(s, anchor, context)
                baseline = x.mean(-1, keepdim=True).expand_as(x).clone()
                gradients = []
                for alpha in np.linspace(0, 1, c["ig_steps"] + 1):
                    z = (
                        (baseline + float(alpha) * (x - baseline))
                        .detach()
                        .requires_grad_(True)
                    )
                    mean, _ = adapter.forward_tensor(z, sc, kwargs, h, mode)
                    # Scalar target: spatial mean at exact last requested horizon.
                    gradients.append(
                        torch.autograd.grad(mean[:, -1].mean(), z)[0].detach()
                    )
                grad = (torch.stack(gradients[:-1]) + torch.stack(gradients[1:])).mean(
                    0
                ) * 0.5
                attr = (x - baseline) * grad
                with torch.no_grad():
                    actual = adapter.forward_tensor(x, sc, kwargs, h, mode)[0][
                        :, -1
                    ].mean()
                    baseval = adapter.forward_tensor(baseline, sc, kwargs, h, mode)[0][
                        :, -1
                    ].mean()
                difference = float(actual - baseval)
                residual = float(attr.sum()) - difference
                ig_rows.extend(
                    dict(
                        subject_id=s.subject_id,
                        anchor=anchor,
                        roi=roi,
                        attribution=float(attr[0, i].sum()),
                        target="mean BOLD at last horizon",
                        horizon=h,
                        completeness_residual=residual,
                        relative_residual=abs(residual) / max(abs(difference), 1e-8),
                        completeness_valid=abs(residual)
                        <= max(0.001, 0.05 * abs(difference)),
                        steps=c["ig_steps"],
                        baseline="history temporal mean",
                    )
                    for i, roi in enumerate(s.roi_labels)
                )
                time_attr = attr[0].sum(0).squeeze(0).cpu().numpy()
                write_table(
                    out,
                    f"ig_time_{s.subject_id}_{anchor}",
                    [
                        dict(context_index=i, attribution=float(v))
                        for i, v in enumerate(time_attr)
                    ],
                )
    raw = write_table(out, "roi_occlusion_windows", rows)
    if raw.empty:
        return "NOT_RUN", "No legal explainability windows"
    per = (
        raw.groupby(
            ["subject_id", "diagnosis", "roi", "horizon", "baseline"], dropna=False
        )
        .importance.mean()
        .reset_index()
    )
    write_table(out, "roi_importance_by_subject", per)
    summary = []
    for keys, block in per.groupby(["diagnosis", "roi", "horizon", "baseline"]):
        lo, hi = bootstrap_ci(block.importance, c["seed"], c["bootstrap_repeats"])
        summary.append(
            dict(
                zip(["diagnosis", "roi", "horizon", "baseline"], keys),
                importance=block.importance.mean(),
                ci_low=lo,
                ci_high=hi,
                n_subjects=len(block),
                unit="subject",
            )
        )
    write_table(out, "roi_importance_summary", summary)
    if ctx.get("networks"):
        network = per.copy()
        network["network"] = network.roi.map(ctx["networks"])
        write_table(
            out,
            "network_importance_by_subject",
            network.dropna(subset=["network"])
            .groupby(["subject_id", "diagnosis", "network", "horizon", "baseline"])
            .importance.mean()
            .reset_index(),
        )
    group_comparisons = []
    for keys, block in per.groupby(["roi", "horizon", "baseline"]):
        r = comparison(
            block.loc[block.diagnosis == "MDD", "importance"],
            block.loc[block.diagnosis == "HC", "importance"],
            c["seed"],
            c["bootstrap_repeats"],
        )
        group_comparisons.append(
            dict(
                metric=keys[0],
                horizon=keys[1],
                baseline=keys[2],
                comparison="MDD minus HC occlusion effect",
                family="importance group comparison",
                **r,
            )
        )
    write_table(
        out,
        "importance_group_comparison",
        correct(pd.DataFrame(group_comparisons), ["family"]),
    )
    write_table(out, "integrated_gradients", ig_rows)
    if ig_rows:
        valid = all(row["completeness_valid"] for row in ig_rows)
        ig_state = dict(
            status="IMPLEMENTED_AND_TESTED" if valid else "FAILED",
            reason="Numerical completeness passed"
            if valid
            else "IG completeness failed; do not interpret invalid attribution rows",
        )
    write_table(out, "ig_status", [ig_state])
    sanity = write_table(out, "parameter_randomization", sanity)
    stability = []
    for keys, block in raw.groupby(["subject_id", "horizon", "baseline"]):
        pivot = block.pivot(index="roi", columns="anchor", values="importance")
        stability.append(
            dict(
                subject_id=keys[0],
                horizon=keys[1],
                baseline=keys[2],
                anchor_correlation=pearson(pivot.iloc[:, 0], pivot.iloc[:, -1]),
                n_rois=len(pivot),
            )
        )
    for keys, block in per.groupby(["subject_id", "horizon"]):
        pivot = block.pivot(index="roi", columns="baseline", values="importance")
        stability.append(
            dict(
                subject_id=keys[0],
                horizon=keys[1],
                baseline="baseline sensitivity",
                baseline_correlation=pearson(pivot.iloc[:, 0], pivot.iloc[:, -1]),
                n_rois=len(pivot),
            )
        )
    if not sanity.empty:
        for keys, block in sanity.groupby(["subject_id", "horizon", "baseline"]):
            means = block.groupby("roi")[["trained_effect", "randomized_effect"]].mean()
            stability.append(
                dict(
                    subject_id=keys[0],
                    horizon=keys[1],
                    baseline=keys[2],
                    randomized_correlation=pearson(
                        means.trained_effect, means.randomized_effect
                    ),
                    n_rois=len(means),
                )
            )
    write_table(out, "attribution_stability", stability)
    # Spatial consistency uses disjoint validation subjects, not the explained test subjects.
    validation = features(ctx)
    validation = validation[validation.split == "val"]
    consistency = []
    if (validation.diagnosis == "HC").sum() >= 3 and (
        validation.diagnosis == "MDD"
    ).sum() >= 3:
        for keys, block in per.groupby(["diagnosis", "horizon", "baseline"]):
            importance, effects = [], []
            for roi, rb in block.groupby("roi"):
                col = f"roi/{roi}/acf1"
                a = validation.loc[validation.diagnosis == "MDD", col]
                b = validation.loc[validation.diagnosis == "HC", col]
                from analysis.study_stats import hedges_g

                effects.append(abs(hedges_g(a.to_numpy(), b.to_numpy())))
                importance.append(rb.importance.mean())
            consistency.append(
                dict(
                    diagnosis=keys[0],
                    horizon=keys[1],
                    baseline=keys[2],
                    correlation=pearson(importance, effects),
                    n_rois=len(effects),
                    unit="ROI spatial descriptive comparison",
                    pathology_source="disjoint validation subjects acf1 Hedges g",
                    p=np.nan,
                )
            )
    write_table(out, "importance_pathology_consistency", consistency)
    return (
        "IMPLEMENTED_AND_TESTED",
        f"Occlusion {len(per)} rows; IG {len(ig_rows)} rows; spatial ranks are not p-values; limited ROI/sample budget recorded",
    )


def sensitivity(ctx):
    adapter, c, out = ctx["adapter"], ctx["config"], ctx["out"]
    if (
        not isinstance(adapter, NativeAdapter)
        or adapter.metadata.get("condition_group") != "mdd8"
    ):
        return (
            "UNSUPPORTED",
            "Model does not take HAMD/HAMA fields; not a clinical-condition ablation",
        )
    from utils.condition_schema import schema_for

    specs = schema_for("mdd8")
    rows, scale_status = [], []
    rng = np.random.default_rng(c["seed"])
    train = [s for s in ctx["subjects"] if s.split == "train" and s.diagnosis == "MDD"]
    candidates = [
        s
        for s in ctx["subjects"]
        if s.split == c["evaluation_split"] and s.diagnosis == "MDD"
    ]
    for scale in ("HAMD", "HAMA"):
        tests = [s for s in candidates if s.metadata.get(scale) is not None][
            : c["explain_subjects_per_cohort"]
        ]
        values = np.array([s.metadata.get(scale) for s in train], float)
        values = values[np.isfinite(values)]
        if len(values) < 6:
            scale_status.append(
                dict(
                    scale=scale,
                    status="MISSING_DATA",
                    reason="Fewer than 6 observed selected training values",
                    n_train=len(values),
                )
            )
            continue
        j = [s.name for s in specs].index(scale)
        st = tests[0].dataset.condition_stats.get(scale) if tests else None
        if st is None or not st.fitted:
            scale_status.append(
                dict(
                    scale=scale,
                    status="MISSING_DATA",
                    reason="No fitted training standardization",
                    n_train=len(values),
                )
            )
            continue
        before = len(rows)
        for s in tests:
            anchors = window_anchors(
                s, c["contexts"][0], max(c["horizons"]), adapter.calibration, 1
            )
            for anchor in anchors:
                _, _, kw = adapter.tensors(s, anchor, c["contexts"][0])
                if not bool(kw["condition"]["cond_mask"][0, j]):
                    continue
                original = adapter.predict(
                    s, anchor, max(c["horizons"]), c["explain_mode"], c["contexts"][0]
                ).mean
                settings = [
                    ("quantile", float(v)) for v in np.quantile(values, [0.1, 0.5, 0.9])
                ]
                settings += [
                    ("train_subject_permutation", float(v))
                    for v in rng.permutation(values)[:3]
                ]
                for kind, v in settings:
                    cond = {key: val.clone() for key, val in kw["condition"].items()}
                    cond["cond_z"][0, j] = (v - st.mu) / st.scale
                    pred = adapter.predict(
                        s,
                        anchor,
                        max(c["horizons"]),
                        c["explain_mode"],
                        c["contexts"][0],
                        conditions=cond,
                    ).mean
                    rows.append(
                        dict(
                            subject_id=s.subject_id,
                            scale=scale,
                            condition_value=v,
                            control=kind,
                            mean_abs_output_change=float(
                                np.abs(pred - original).mean()
                            ),
                            mean_output=float(pred.mean()),
                            training_n=len(values),
                            training_min=float(values.min()),
                            training_max=float(values.max()),
                            label="Clinical-conditioning sensitivity",
                            interpretation="model response; no treatment effect",
                        )
                    )
        scale_status.append(
            dict(
                scale=scale,
                status="IMPLEMENTED_AND_TESTED"
                if len(rows) > before
                else "MISSING_DATA",
                reason="fixed-BOLD model response"
                if len(rows) > before
                else "Selected test subjects lack observed condition",
                n_train=len(values),
                n_responses=len(rows) - before,
            )
        )
    write_table(out, "clinical_conditioning_sensitivity", rows)
    write_table(out, "sensitivity_scale_status", scale_status)
    return (
        ("IMPLEMENTED_AND_TESTED", f"{len(rows)} fixed-BOLD condition responses")
        if rows
        else (
            "MISSING_DATA",
            "Insufficient observed train scale values or missing test conditions",
        )
    )


def uncertainty(ctx):
    adapter, c, out = ctx["adapter"], ctx["config"], ctx["out"]
    if adapter.metadata["dynamics_mode"] != "graph_ode" or not adapter.variance_trained:
        return "UNSUPPORTED", "No checkpoint with configured trained ODE variance head"
    rows, calibration = [], []
    for s in ctx["subjects"]:
        if s.split != c["evaluation_split"]:
            continue
        for anchor in window_anchors(
            s,
            c["contexts"][0],
            max(c["horizons"]),
            adapter.calibration,
            c["anchors_per_subject"],
        ):
            pred = adapter.predict(
                s, anchor, max(c["horizons"]), "ode_integration", c["contexts"][0]
            )
            target = s.bold[anchor + 1 : anchor + 1 + len(pred.mean)]
            for h in c["horizons"]:
                rows.append(
                    dict(
                        subject_id=s.subject_id,
                        diagnosis=s.diagnosis,
                        horizon=h,
                        anchor=anchor,
                        **gaussian_metrics(
                            pred.mean[h - 1], target[h - 1], pred.std[h - 1]
                        ),
                    )
                )
                for coverage in (0.5, 0.8, 0.95):
                    from scipy.stats import norm

                    q = norm.ppf((1 + coverage) / 2)
                    calibration.append(
                        dict(
                            subject_id=s.subject_id,
                            diagnosis=s.diagnosis,
                            horizon=h,
                            nominal=coverage,
                            observed=float(
                                (
                                    np.abs(target[h - 1] - pred.mean[h - 1])
                                    <= q * pred.std[h - 1]
                                ).mean()
                            ),
                        )
                    )
    write_table(out, "uncertainty_by_window", rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "NOT_RUN", "No uncertainty evaluation windows"
    per = (
        frame.groupby(["subject_id", "diagnosis", "horizon"])[
            ["nll", "coverage95", "width95", "error_std_correlation"]
        ]
        .mean()
        .reset_index()
    )
    write_table(out, "uncertainty_by_subject", per)
    cal = (
        pd.DataFrame(calibration)
        .groupby(["subject_id", "diagnosis", "horizon", "nominal"])["observed"]
        .mean()
        .reset_index()
    )
    write_table(out, "uncertainty_calibration", cal)
    rows = []
    for keys, block in per.groupby(["diagnosis", "horizon"]):
        for metric in ("nll", "coverage95", "width95"):
            lo, hi = bootstrap_ci(block[metric], c["seed"], c["bootstrap_repeats"])
            rows.append(
                dict(
                    diagnosis=keys[0],
                    horizon=keys[1],
                    metric=metric,
                    mean=block[metric].mean(),
                    ci_low=lo,
                    ci_high=hi,
                    n_subjects=len(block),
                )
            )
    write_table(out, "uncertainty_summary", rows)
    return (
        "IMPLEMENTED_AND_TESTED",
        "Native dynamics_std (not logvar); marginal coverage only, no clinical confidence claim",
    )
