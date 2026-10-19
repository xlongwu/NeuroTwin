"""Standalone table-driven paper figures (PNG + PDF), with explicit synthetic labels.

Style reference: plot-is-all-you-need/oa-tech-pone-0258309-f04: white background,
visible individual observations, thin spines. Box whiskers are not confidence intervals.
Reproduce: python -m analysis.run_study --stage report --input-run RESULTS_DIRECTORY
"""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"HC": "#0072B2", "MDD": "#D55E00"}


def read(out, name):
    path = out / f"{name}.csv"
    if not path.exists() or path.stat().st_size < 3:
        return pd.DataFrame()
    return pd.read_csv(path)


def render(out, synthetic=False):
    out = Path(out)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    saved = []

    def save(fig, name, title):
        fig.suptitle(
            title + (" | SYNTHETIC TEST DATA" if synthetic else ""), fontsize=11
        )
        bottom = (
            0.18
            if name == "figure1_performance"
            else 0.14
            if name == "figure5_explainability"
            else 0
        )
        fig.tight_layout(rect=(0, bottom, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    p = read(out, "performance_summary")
    if not p.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for ax, metric in zip(axes, ("mae", "rmse")):
            for keys, block in p[p.metric == metric].groupby(
                ["diagnosis", "model", "prediction_mode", "context"]
            ):
                group, model, mode, context = keys
                block = block.sort_values("horizon")
                label = f"{group} {model}/{mode} K={context} n={int(block.n_subjects.iloc[0])}"
                styles = {
                    "autoregressive": ("-", "o"),
                    "ode_integration": ("-.", "s"),
                    "direct_cpm": (":", "^"),
                    "one_step": ("", "X"),
                    "constant": ("--", "v"),
                }
                line, marker = styles[mode] if model != "ridge_ar1" else ("--", "D")
                ax.plot(
                    block.horizon,
                    block["mean"],
                    marker=marker,
                    markersize=3,
                    color=COLORS.get(group),
                    linestyle=line,
                    alpha=1,
                    label=label,
                )
                ax.fill_between(
                    block.horizon,
                    block.ci_low,
                    block.ci_high,
                    color=COLORS.get(group),
                    alpha=0.06,
                )
            ax.set(
                xlabel="Exact forecast offset (TR)",
                ylabel=metric.upper() + " (calibrated BOLD)",
            )
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=5, loc="lower center", ncol=3, frameon=False
        )
        save(
            fig,
            "figure1_performance",
            "Forecast performance | subject bootstrap 95% CI",
        )
        per = read(out, "performance_by_subject")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        data = []
        labels = []
        per = per[
            (per.context == per.context.max()) & (per.horizon == per.horizon.max())
        ]
        for keys, b in per.groupby(["diagnosis", "prediction_mode", "model"]):
            # Each dot is one subject; restrict context and horizon, never pool windows.
            data.append(b.mae.to_numpy())
            labels.append(f"{keys[0]}\n{keys[2]}\n{keys[1]}\nn={len(b)}")
        ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#87ceeb", "alpha": 0.5},
        )
        rng = np.random.default_rng(2024)
        for i, v in enumerate(data):
            ax.scatter(
                i + 1 + rng.uniform(-0.08, 0.08, len(v)), v, s=8, color="#2b438d"
            )
        ax.set(ylabel="Subject MAE (calibrated BOLD)")
        ax.tick_params(axis="x", labelsize=6)
        save(
            fig,
            "figure1_subject_distributions",
            f"Held-out subjects H={int(per.horizon.max())} | box whiskers: 1.5 IQR",
        )
        roi = read(out, "performance_by_roi")
        if not roi.empty:
            b = roi[(roi.model == "model") & (roi.horizon == roi.horizon.max())]
            matrix = b.groupby(["roi", "prediction_mode"]).abs_error.mean().unstack()
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(matrix.T, aspect="auto", cmap="viridis")
            ax.set(
                xlabel="ROI order (table labels)",
                ylabel="Prediction mode",
                yticks=range(len(matrix.columns)),
                yticklabels=matrix.columns,
            )
            fig.colorbar(im, ax=ax, label="Subject-mean MAE (calibrated BOLD)")
            save(fig, "figure1_roi_errors", "ROI error distribution | subject means")
    examples = read(out, "bold_examples")
    files = sorted(out.glob("dynamics_*.npz"))
    if files:
        with np.load(files[0]) as d:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
            axes[0].plot(d["observed"][:, 0], color="#333333", label="Observed")
            axes[0].plot(d["predicted"][:, 0], color="#0072B2", label="Autoregressive")
            axes[0].set(xlabel="Future offset (TR)", ylabel="Calibrated BOLD")
            axes[0].legend(fontsize=7)
            for ax, key, title in zip(
                axes[1:],
                ("fc_observed", "fc_predicted"),
                ("Observed FC", "Predicted FC"),
            ):
                im = ax.imshow(d[key], vmin=-1, vmax=1, cmap="RdBu_r")
                ax.set_title(title)
            fig.colorbar(im, ax=axes[2], label="Pearson r")
            save(
                fig,
                "figure2_dynamics",
                "Continuous autoregressive segment | first deterministic test subject (n=1)",
            )
    elif not examples.empty:
        b = examples[
            (examples.subject_id == examples.subject_id.iloc[0])
            & (examples.roi == examples.roi.iloc[0])
        ]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for keys, g in b.groupby(["model", "prediction_mode"]):
            ax.plot(g.offset, g.predicted, label="/".join(keys))
        g = b.drop_duplicates("offset")
        ax.plot(g.offset, g.observed, color="black", label="Observed")
        ax.set(xlabel="Future offset (TR)", ylabel="Calibrated BOLD")
        ax.legend(fontsize=7)
        save(
            fig,
            "figure2_bold_example",
            "Short-horizon example | FC/PSD/dFC not estimated",
        )
    path = read(out, "pathology_statistics")
    if not path.empty:
        roi = path[path.family == "roi"].copy()
        roi["roi"] = roi.metric.str.split("/").str[1]
        roi["component"] = roi.metric.str.split("/").str[2]
        matrix = roi.pivot(index="component", columns="roi", values="effect")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        limit = max(float(np.nanmax(np.abs(matrix))), 0.01)
        im = axes[0].imshow(
            matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axes[0].set(
            xlabel="ROI (labels in table)",
            yticks=range(len(matrix)),
            yticklabels=matrix.index,
            title="ROI descriptor Hedges g",
        )
        fig.colorbar(im, ax=axes[0], label="MDD minus HC")
        edge = path[path.family == "edge"]
        labels = sorted(
            set(edge.metric.str.split("/").str[1])
            | set(edge.metric.str.split("/").str[2])
        )
        if labels:
            arr = np.full((len(labels), len(labels)), np.nan)
            positions = {v: i for i, v in enumerate(labels)}
            for row in edge.itertuples():
                _, a, b = row.metric.split("/")
                arr[positions[a], positions[b]] = arr[positions[b], positions[a]] = (
                    row.effect
                )
            limit = max(float(np.nanmax(np.abs(arr))), 0.01)
            im = axes[1].imshow(arr, cmap="RdBu_r", vmin=-limit, vmax=limit)
            fig.colorbar(im, ax=axes[1], label="Edge Hedges g")
            axes[1].set_title(f"FC edge effects | q<0.05: {int((edge.q < 0.05).sum())}")
        save(
            fig,
            "figure3_pathology",
            f"Observed effects | MDD n={int(path.n_a.max())}, HC n={int(path.n_b.max())} | unadjusted; FDR tables separate",
        )
    fidelity = read(out, "model_pathology_statistics")
    if not fidelity.empty:
        matrix = fidelity.pivot(
            index="metric", columns="origin", values="effect"
        ).dropna()
        if {"predicted", "observed_matched"} <= set(matrix.columns):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            for ax, family in zip(axes, ("roi", "edge")):
                b = matrix[matrix.index.str.startswith(family + "/")]
                ax.scatter(
                    b.observed_matched, b.predicted, s=4, alpha=0.3, color="#0072B2"
                )
                values = b[["observed_matched", "predicted"]].to_numpy()
                if values.size:
                    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
                    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.7)
                ax.set(
                    xlabel="Matched observed Hedges g",
                    ylabel="Predicted Hedges g",
                    title=f"{family}: descriptive spatial agreement",
                )
            save(
                fig,
                "figure3_model_fidelity",
                f"Continuous prediction pathology fidelity | MDD n={int(fidelity.n_a.max())}, HC n={int(fidelity.n_b.max())}",
            )
    norm = read(out, "normative_subject_deviations")
    if not norm.empty:
        matrix = norm.groupby(["subject_id", "roi"]).distance.mean().unstack()
        groups = (
            norm[["subject_id", "diagnosis"]]
            .drop_duplicates()
            .sort_values(["diagnosis", "subject_id"])
        )
        matrix = matrix.reindex(groups.subject_id)
        matrix.to_csv(out / "figure4_source_matrix.csv")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set(
            xlabel="ROI (labels in table)",
            ylabel=f"Held-out subject (n={len(matrix)})",
            yticks=range(len(matrix)),
            yticklabels=[f"{g} {i + 1}" for i, g in enumerate(groups.diagnosis)],
        )
        ax.tick_params(axis="y", labelsize=6)
        fig.colorbar(im, ax=ax, label="RMS standardized residual")
        save(
            fig,
            "figure4_normative",
            "HC-train demographic reference | individual deviation, not diagnosis",
        )
    imp = read(out, "roi_importance_summary")
    if not imp.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for keys, b in imp[imp.baseline == "history_mean"].groupby(
            ["diagnosis", "horizon"]
        ):
            b = b.sort_values("roi")
            line, marker = {
                1: ("-", "o"),
                2: ("--", "s"),
                4: ("-.", "^"),
                8: (":", "D"),
            }.get(keys[1], ("-", "o"))
            ax.plot(
                b.roi,
                b.importance,
                marker=marker,
                linestyle=line,
                color=COLORS[keys[0]],
                ms=3,
                label=f"{keys[0]} H={keys[1]} n={int(b.n_subjects.iloc[0])}",
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(ylabel="Occlusion change in MAE", xlabel="Evaluated ROI subset")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=6, ncol=4, loc="lower center", frameon=False
        )
        save(
            fig,
            "figure5_explainability",
            "Model input dependence | ranks do not imply significance",
        )
    clinical = read(out, "clinical_associations")
    sensitivity = read(out, "clinical_conditioning_sensitivity")
    if not clinical.empty or not sensitivity.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        if not clinical.empty:
            # Deterministic first features; never select by p or absolute correlation.
            b = clinical[
                (clinical.comparison == "HAMD") & clinical.covariates.isna()
            ].head(12)
            axes[0].barh(range(len(b)), b.effect, color="#0072B2")
            axes[0].set(
                yticks=range(len(b)),
                yticklabels=b.metric.str.replace("roi/", "", regex=False),
                xlabel="Spearman r",
                title=f"Observed HAMD association (n={int(b.n_a.max())})"
                if len(b)
                else "No unadjusted HAMD associations",
            )
            axes[0].tick_params(axis="y", labelsize=6)
        else:
            axes[0].text(0.1, 0.5, "MISSING_DATA: clinical association")
            axes[0].set_axis_off()
        if not sensitivity.empty:
            for keys, b in sensitivity[sensitivity.control == "quantile"].groupby(
                ["subject_id", "scale"]
            ):
                axes[1].plot(
                    b.condition_value,
                    b.mean_abs_output_change,
                    marker="o",
                    label=keys[1],
                )
            axes[1].set(
                xlabel="Clinical condition (training range)",
                ylabel="Mean absolute model output change",
                title="Clinical-conditioning sensitivity",
            )
        else:
            axes[1].text(0.1, 0.5, "UNSUPPORTED: no clinical condition input")
            axes[1].set_axis_off()
        save(
            fig,
            "figure6_clinical",
            "Cross-sectional association and model response are separate evidence",
        )
    (out / "figure_rendering.json").write_text(
        json.dumps(
            dict(
                plotting_source_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                style_reference="plot-is-all-you-need/oa-tech-pone-0258309-f04",
                matplotlib=matplotlib.__version__,
                synthetic=synthetic,
                figures=saved,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return saved
