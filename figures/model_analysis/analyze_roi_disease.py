"""Analyze MDD-HC ROI connectivity effects and create publication figures.

Inputs are the existing subject-level Correlation MAT files, clinical workbooks,
and AAL116.xlsx. The script never edits source data or checkpoints. It does not
run training or test-set inference. Validation-set model attribution is reported
only when the checkpoint's saved clinical source hash and alias map match the
current source exactly; a mismatch blocks that analysis.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from textwrap import fill
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "neurotwin_tfm_finetune" / "finetuned_best.pt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CLINICAL_FILES = {
    "HC": "Rest-meta-MDD-V1V2-Merged-HC.xlsx",
    "MDD": "Rest-meta-MDD-V1V2-Merged-MDD.xlsx",
}
CORRELATION_KEY = "ROISignals_Correlation"
SIGNAL_KEY = "Data"
SITE_PATTERN = re.compile(r"^(IS\d+|S\d+)-", flags=re.IGNORECASE)

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.7,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}
PALETTE = {
    "model": "#4A6FA5",
    "persistence": "#7D7D7D",
    "trend": "#5A8F7B",
    "accent": "#8E6B8A",
    "grid": "#E6E6E6",
    "neutral": "#777777",
}
NETWORK_COLORS = {
    "SMN": "#4A6FA5",
    "DAN": "#5A8F7B",
    "LN": "#C45C3E",
    "DMN": "#8E6B8A",
    "SN": "#B7950B",
    "FN": "#557F95",
    "VN": "#7D7D7D",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def _clean_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2.5, width=0.6, color="#555555")
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.07, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    values = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(values)
    ranked = values[order] * values.size / np.arange(1, values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    q[valid] = adjusted
    return q


def _load_roi_table(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    if raw.shape[0] != 116 or raw.shape[1] < 5:
        raise ValueError(f"AAL116 workbook must contain 116 rows and >=5 columns; got {raw.shape}")
    roi_ids = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy()
    if not np.array_equal(roi_ids, np.arange(1, 117)):
        raise ValueError("AAL116 ROI IDs must be unique and ordered exactly 1..116")
    roi = pd.DataFrame({
        "roi_index": roi_ids.astype(int),
        "name_cn": raw.iloc[:, 1].fillna("").astype(str).str.strip(),
        "name_en": raw.iloc[:, 2].fillna("").astype(str).str.strip(),
        "abbr": raw.iloc[:, 3].fillna("").astype(str).str.strip(),
        "network": raw.iloc[:, 4].fillna("").astype(str).str.strip(),
    })
    if roi["abbr"].eq("").any() or roi["network"].eq("").any():
        raise ValueError("AAL116 abbreviation or network labels contain empty values")
    return roi


def _site_from_id(subject_id: str) -> str:
    match = SITE_PATTERN.match(subject_id)
    if not match:
        raise ValueError(f"Cannot derive acquisition site from subject ID: {subject_id}")
    return match.group(1).upper()


def _read_clinical(group: str) -> pd.DataFrame:
    path = DATA_ROOT / CLINICAL_FILES[group]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {group} clinical workbook: {path}")
    df = pd.read_excel(path)
    edu_column = "education" if "education" in df.columns else "Education (years)"
    required = {"ID", "Sex", "Age", edu_column}
    if group == "MDD":
        required.add("HAMD")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
    columns = ["ID", "Sex", "Age", edu_column] + (["HAMD"] if group == "MDD" else [])
    df = df[columns].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    if df["ID"].duplicated().any():
        raise ValueError(f"Duplicate subject IDs in clinical workbook: {path.name}")
    df = df.rename(columns={edu_column: "Education"})
    for col in ["Sex", "Age", "Education"] + (["HAMD"] if group == "MDD" else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan
    df["group"] = group
    return df


def _list_correlation_records(clinical: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for group in ("HC", "MDD"):
        folder = DATA_ROOT / "Correlation" / group
        files = sorted(folder.glob("ROISignals_Correlation_*.mat"))
        if not files:
            raise FileNotFoundError(f"No correlation matrices found in {folder}")
        ids = [f.stem.removeprefix("ROISignals_Correlation_") for f in files]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate correlation-matrix subject IDs in {folder}")
        meta = clinical[group].set_index("ID", drop=False)
        matched = 0
        for path, subject_id in zip(files, ids):
            if subject_id not in meta.index:
                records.append({"group": group, "subject_id": subject_id,
                                "site": _site_from_id(subject_id), "mat_path": path,
                                "clinical_match": False})
                continue
            matched += 1
            row = meta.loc[subject_id]
            records.append({
                "group": group,
                "subject_id": subject_id,
                "site": _site_from_id(subject_id),
                "mat_path": path,
                "clinical_match": True,
                "Sex": row["Sex"],
                "Age": row["Age"],
                "Education": row["Education"],
                "HAMD": row.get("HAMD", np.nan),
            })
        counts[group] = {
            "correlation_files": len(files),
            "unique_correlation_subjects": len(set(ids)),
            "clinical_table_rows": int(len(clinical[group])),
            "exact_id_matches": int(matched),
            "correlation_ids_without_clinical_row": int(len(files) - matched),
        }
    frame = pd.DataFrame(records)
    return frame, counts


def _select_primary_cohort(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matched = records.loc[records["clinical_match"]].copy()
    sites = {
        group: set(matched.loc[matched.group == group, "site"].unique())
        for group in ("HC", "MDD")
    }
    shared_sites = sites["HC"] & sites["MDD"]
    common = matched.loc[matched.site.isin(shared_sites)].copy()
    covariate_complete = common[["Age", "Sex", "Education"]].notna().all(axis=1)
    complete = common.loc[covariate_complete].copy()
    group_presence = complete.groupby("site")["group"].nunique()
    analysis_sites = set(group_presence[group_presence == 2].index)
    final = complete.loc[complete.site.isin(analysis_sites)].copy()
    final = final.sort_values(["group", "site", "subject_id"]).reset_index(drop=True)
    if final.empty or final["group"].nunique() != 2:
        raise ValueError("No complete-case subjects from both diagnoses remain at shared sites")
    summary = {
        "hc_sites_with_matched_rows": len(sites["HC"]),
        "mdd_sites_with_matched_rows": len(sites["MDD"]),
        "shared_sites_before_complete_case": len(shared_sites),
        "matched_subjects_at_shared_sites": int(len(common)),
        "complete_case_subjects_before_site_pair_filter": int(len(complete)),
        "analysis_sites_with_both_diagnoses": int(len(analysis_sites)),
        "complete_case_subjects_after_site_pair_filter": int(len(final)),
        "n_hc": int((final.group == "HC").sum()),
        "n_mdd": int((final.group == "MDD").sum()),
        "n_site_clusters": int(final.site.nunique()),
        "excluded_for_covariate_missing_at_shared_sites": int((~covariate_complete).sum()),
        "covariates": ["site fixed effects", "Age", "Sex", "Education"],
        "excluded_covariate": "Dataset field (not harmonized across groups)",
    }
    if summary["n_site_clusters"] < 2:
        raise ValueError("Site-cluster robust inference requires at least two site clusters")
    return final, summary


def _load_connectivity_features(
    cohort: pd.DataFrame, roi_count: int, signal_checks_per_group: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    upper_i, upper_j = np.triu_indices(roi_count, k=1)
    n_subjects = len(cohort)
    roi_features = np.empty((n_subjects, roi_count), dtype=np.float32)
    edge_features = np.empty((n_subjects, len(upper_i)), dtype=np.float32)
    checked: dict[str, int] = {"HC": 0, "MDD": 0}
    max_validation_error = 0.0
    validation_errors: list[float] = []

    for row_index, row in cohort.iterrows():
        data = loadmat(row.mat_path, variable_names=[CORRELATION_KEY])
        if CORRELATION_KEY not in data:
            raise KeyError(f"{row.mat_path} lacks {CORRELATION_KEY}")
        corr = np.asarray(data[CORRELATION_KEY], dtype=np.float64)
        if corr.shape != (roi_count, roi_count):
            raise ValueError(f"{row.mat_path} has {corr.shape}, expected {(roi_count, roi_count)}")
        if not np.isfinite(corr).all():
            raise ValueError(f"Non-finite correlation entries in {row.mat_path}")
        symmetry_error = float(np.max(np.abs(corr - corr.T)))
        if symmetry_error > 1e-5:
            raise ValueError(f"Asymmetric correlation matrix ({symmetry_error:g}) in {row.mat_path}")
        corr = (corr + corr.T) * 0.5
        z = np.arctanh(np.clip(corr, -1.0 + 1e-7, 1.0 - 1e-7))
        np.fill_diagonal(z, np.nan)
        roi_features[row_index] = np.nanmean(z, axis=1).astype(np.float32)
        edge_features[row_index] = z[upper_i, upper_j].astype(np.float32)

        if checked[row.group] < signal_checks_per_group:
            signal_path = (DATA_ROOT / "Normlize" / row.group /
                           f"ROISignals_{row.subject_id}.mat")
            if not signal_path.is_file():
                raise FileNotFoundError(f"Cannot verify ROI columns; missing {signal_path}")
            signal_data = loadmat(signal_path, variable_names=[SIGNAL_KEY])
            if SIGNAL_KEY not in signal_data:
                raise KeyError(f"{signal_path} lacks {SIGNAL_KEY}")
            signals = np.asarray(signal_data[SIGNAL_KEY], dtype=np.float64)
            if signals.shape[1] < roi_count:
                raise ValueError(f"{signal_path} has only {signals.shape[1]} ROI columns")
            expected = np.corrcoef(signals[:150, :roi_count].T)
            expected = np.nan_to_num(expected)
            error = float(np.max(np.abs(expected - corr)))
            validation_errors.append(error)
            max_validation_error = max(max_validation_error, error)
            checked[row.group] += 1

    if max_validation_error > 1e-5:
        raise ValueError(
            "Correlation files do not match the first 150x116 normalized signal values "
            f"for spot checks (max abs difference {max_validation_error:g})")
    info = {
        "roi_count": roi_count,
        "edge_count_unique_upper_triangle": int(len(upper_i)),
        "signal_correlation_spot_checks_by_group": checked,
        "signal_to_correlation_max_abs_difference": max_validation_error,
        "roi_order_evidence": (
            "AAL IDs are exactly 1..116; project correlation code takes the first 116 "
            "normalized signal columns and computes corrcoef(signal.T). Correlation MAT "
            "files do not contain embedded ROI labels, so names follow that saved column order."
        ),
    }
    return roi_features, edge_features, info


def _design_matrix(frame: pd.DataFrame, predictor: str = "diagnosis") -> tuple[np.ndarray, int, np.ndarray, list[str]]:
    df = frame.reset_index(drop=True).copy()
    n = len(df)
    age_sd = float(df["Age"].std(ddof=0))
    edu_sd = float(df["Education"].std(ddof=0))
    if age_sd <= 0 or edu_sd <= 0:
        raise ValueError("Age and Education must vary within the analysis cohort")
    design = pd.DataFrame({
        "Intercept": np.ones(n),
        "Age_z": (df["Age"].to_numpy(dtype=float) - df["Age"].mean()) / age_sd,
        "Education_z": (df["Education"].to_numpy(dtype=float) - df["Education"].mean()) / edu_sd,
    })
    sex = pd.get_dummies(df["Sex"].astype(str), prefix="Sex", drop_first=True, dtype=float)
    site = pd.get_dummies(df["site"].astype(str), prefix="site", drop_first=True, dtype=float)
    design = pd.concat([design, sex.reset_index(drop=True), site.reset_index(drop=True)], axis=1)
    if predictor == "diagnosis":
        design.insert(1, "diagnosis_MDD", (df["group"] == "MDD").astype(float))
        coefficient_index = int(design.columns.get_loc("diagnosis_MDD"))
    elif predictor == "HAMD":
        if "HAMD" not in df or df["HAMD"].isna().any():
            raise ValueError("HAMD analysis requires complete MDD HAMD scores")
        hamd_sd = float(df["HAMD"].std(ddof=0))
        if hamd_sd <= 0:
            raise ValueError("HAMD score must vary within the MDD cohort")
        design.insert(1, "HAMD_z", (df["HAMD"].to_numpy(dtype=float) - df["HAMD"].mean()) / hamd_sd)
        coefficient_index = int(design.columns.get_loc("HAMD_z"))
    else:
        raise ValueError(f"Unsupported predictor: {predictor}")
    X = design.to_numpy(dtype=np.float64)
    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise ValueError(f"Design matrix is rank deficient: rank={rank}, columns={X.shape[1]}")
    clusters = df["site"].astype(str).to_numpy()
    return X, coefficient_index, clusters, list(design.columns)


def _fit_cluster_robust(
    X: np.ndarray, outcomes: np.ndarray, clusters: np.ndarray, coefficient_index: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients with CR1 site-clustered uncertainty for one coefficient."""
    Y = np.asarray(outcomes)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Outcome/design row mismatch: {Y.shape[0]} != {X.shape[0]}")
    xtx_inv = np.linalg.pinv(X.T @ X)
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Design matrix became rank deficient during regression")
    unique_clusters, cluster_codes = np.unique(clusters, return_inverse=True)
    n, p = X.shape
    g = len(unique_clusters)
    if g < 2 or n <= p:
        raise ValueError(f"Insufficient sample/site degrees of freedom: n={n}, p={p}, sites={g}")
    one_hot = np.eye(g, dtype=np.float64)[cluster_codes]
    contrast_weight = X @ xtx_inv[:, coefficient_index]
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    df = g - 1
    critical = float(student_t.ppf(0.975, df))

    m = Y.shape[1]
    beta = np.empty(m, dtype=np.float64)
    se = np.empty(m, dtype=np.float64)
    pvalue = np.empty(m, dtype=np.float64)
    lower = np.empty(m, dtype=np.float64)
    upper = np.empty(m, dtype=np.float64)
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        y = Y[:, start:stop].astype(np.float64, copy=False)
        coefficients = xtx_inv @ (X.T @ y)
        residual = y - X @ coefficients
        estimate = coefficients[coefficient_index]
        score = one_hot.T @ (contrast_weight[:, None] * residual)
        variance = correction * np.sum(score * score, axis=0)
        errors = np.sqrt(np.maximum(variance, 0.0))
        statistic = np.divide(estimate, errors, out=np.zeros_like(estimate), where=errors > 0)
        pv = 2.0 * student_t.sf(np.abs(statistic), df)
        pv[(errors == 0) & (estimate != 0)] = 0.0
        beta[start:stop] = estimate
        se[start:stop] = errors
        pvalue[start:stop] = pv
        lower[start:stop] = estimate - critical * errors
        upper[start:stop] = estimate + critical * errors
    return beta, se, lower, upper, pvalue


def _hedges_g(outcomes: np.ndarray, groups: np.ndarray) -> np.ndarray:
    hc = outcomes[groups == "HC"].astype(np.float64)
    mdd = outcomes[groups == "MDD"].astype(np.float64)
    n0, n1 = hc.shape[0], mdd.shape[0]
    mean0, mean1 = hc.mean(axis=0), mdd.mean(axis=0)
    var0, var1 = hc.var(axis=0, ddof=1), mdd.var(axis=0, ddof=1)
    pooled = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
    d = np.divide(mean1 - mean0, pooled, out=np.full_like(mean0, np.nan), where=pooled > 0)
    correction = 1.0 - 3.0 / (4.0 * (n0 + n1) - 9.0)
    return correction * d


def _raw_group_summary(outcomes: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, ...]:
    hc = outcomes[groups == "HC"].astype(np.float64)
    mdd = outcomes[groups == "MDD"].astype(np.float64)
    return (hc.mean(axis=0), hc.std(axis=0, ddof=1),
            mdd.mean(axis=0), mdd.std(axis=0, ddof=1))


def _network_color(network: str) -> str:
    if network in NETWORK_COLORS:
        return NETWORK_COLORS[network]
    return "#8A8A8A"


def _plot_roi_effects(roi_effects: pd.DataFrame, cohort: pd.DataFrame) -> None:
    n_show = min(30, len(roi_effects))
    show = roi_effects.reindex(roi_effects["adjusted_beta"].abs().nlargest(n_show).index)
    show = show.sort_values("adjusted_beta").reset_index(drop=True)
    height = max(5.4, 0.19 * n_show + 1.25)
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(7.2, height))
        fig.subplots_adjust(left=0.25, right=0.98, top=0.92, bottom=0.23)
        y = np.arange(len(show))
        for yi, row in enumerate(show.itertuples(index=False)):
            color = _network_color(row.network)
            marker = "s" if row.q_value < 0.05 else "o"
            ax.errorbar(row.adjusted_beta, yi,
                        xerr=[[row.adjusted_beta - row.ci_low], [row.ci_high - row.adjusted_beta]],
                        fmt=marker, color=color,
                        markersize=3.5 if row.q_value < 0.05 else 2.8,
                        markeredgecolor="#333333" if row.q_value < 0.05 else color,
                        markeredgewidth=0.35 if row.q_value < 0.05 else 0.0,
                        ecolor=color, elinewidth=0.7, capsize=1.8, zorder=3)
        ax.axvline(0, color="#555555", linewidth=0.7, linestyle="--", zorder=1)
        ax.set_yticks(y, [f"{r.abbr}  ·  {r.network}" for r in show.itertuples(index=False)])
        ax.set_xlabel("Adjusted MDD − HC difference in mean Fisher-z connectivity")
        ax.set_title("Largest absolute ROI-level disease effects", loc="left", pad=7)
        ax.text(0.99, 1.03,
                f"{n_show} of 116 ROIs · site-clustered 95% CI · BH q < 0.05 marked □",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#444444")
        ax.invert_yaxis()
        _clean_axes(ax)
        networks = list(dict.fromkeys(show["network"].tolist()))
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=_network_color(net),
                              markersize=3.5, label=net) for net in networks]
        ax.legend(handles=handles, title="AAL network", ncol=min(7, len(handles)),
                  loc="upper center", bbox_to_anchor=(0.5, -0.17), handletextpad=0.35,
                  columnspacing=0.8)
        fig.text(0.25, 0.025,
                 f"Complete cases at {cohort.site.nunique()} shared sites; "
                 f"HC n={(cohort.group == 'HC').sum()}, MDD n={(cohort.group == 'MDD').sum()}. "
                 "Positive values indicate higher average connectivity in MDD.",
                 ha="left", va="bottom", fontsize=5.7, color="#444444")
        _save(fig, "roi_disease_effects")


def _plot_edge_heatmap(edge_effects: pd.DataFrame, roi: pd.DataFrame) -> None:
    n = len(roi)
    effects = np.zeros((n, n), dtype=float)
    qvalues = np.ones((n, n), dtype=float)
    i = edge_effects["roi_i"].to_numpy(dtype=int) - 1
    j = edge_effects["roi_j"].to_numpy(dtype=int) - 1
    effects[i, j] = edge_effects["adjusted_beta"].to_numpy(dtype=float)
    effects[j, i] = effects[i, j]
    qvalues[i, j] = edge_effects["q_value"].to_numpy(dtype=float)
    qvalues[j, i] = qvalues[i, j]
    np.fill_diagonal(effects, np.nan)
    network_order = list(dict.fromkeys(roi["network"].tolist()))
    network_rank = {network: rank for rank, network in enumerate(network_order)}
    sort_idx = np.array(sorted(range(n), key=lambda k: (network_rank[roi.iloc[k].network], k)))
    reordered = effects[np.ix_(sort_idx, sort_idx)]
    reordered_q = qvalues[np.ix_(sort_idx, sort_idx)]
    finite = np.abs(reordered[np.isfinite(reordered)])
    limit = float(np.quantile(finite, 0.99)) if finite.size else 0.1
    limit = max(limit, 0.01)
    boundaries = []
    centers = []
    labels = []
    start = 0
    reordered_networks = roi.iloc[sort_idx]["network"].tolist()
    for net in network_order:
        size = sum(value == net for value in reordered_networks)
        if not size:
            continue
        centers.append(start + (size - 1) / 2)
        labels.append(net)
        start += size
        boundaries.append(start - 0.5)
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(7.2, 6.35))
        fig.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.12)
        image = ax.imshow(np.ma.masked_invalid(reordered), cmap="RdBu_r", vmin=-limit,
                          vmax=limit, interpolation="nearest", aspect="equal")
        for boundary in boundaries[:-1]:
            ax.axhline(boundary, color="white", linewidth=0.85)
            ax.axvline(boundary, color="white", linewidth=0.85)
        sig_i, sig_j = np.where((reordered_q < 0.05) & ~np.eye(n, dtype=bool))
        if sig_i.size:
            ax.scatter(sig_j, sig_i, s=1.0, c="#202020", marker=".", linewidths=0,
                       rasterized=True, zorder=3)
        ax.set_xticks(centers, labels, rotation=45, ha="left")
        ax.set_yticks(centers, labels)
        ax.tick_params(axis="both", length=0, labelsize=6)
        ax.set_xlabel("AAL network block")
        ax.set_ylabel("AAL network block")
        ax.set_title("Adjusted edge-wise MDD − HC connectivity differences", loc="left", pad=8)
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
        cbar.set_label("Adjusted difference in Fisher-z edge connectivity", fontsize=6.5)
        cbar.ax.tick_params(labelsize=5.5, length=2)
        fig.text(0.12, 0.025,
                 f"6,670 unique edges; color scale clipped at the 99th percentile (±{limit:.3f}). "
                 "Diagonal omitted; black dots mark BH q < 0.05. One row per participant; site-clustered SEs.",
                 ha="left", va="bottom", fontsize=5.6, color="#444444")
        _save(fig, "mdd_hc_edge_effects")


def _plot_importance_disease(roi_effects: pd.DataFrame, provenance: dict[str, Any]) -> None:
    n_show = min(18, len(roi_effects))
    show = roi_effects.reindex(roi_effects["adjusted_beta"].abs().nlargest(n_show).index)
    show = show.sort_values("adjusted_beta").reset_index(drop=True)
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
        fig.subplots_adjust(left=0.19, right=0.98, top=0.84, bottom=0.19, wspace=0.42)
        ax = axes[0]
        y = np.arange(len(show))
        for yi, row in enumerate(show.itertuples(index=False)):
            color = _network_color(row.network)
            ax.errorbar(row.adjusted_beta, yi,
                        xerr=[[row.adjusted_beta - row.ci_low], [row.ci_high - row.adjusted_beta]],
                        fmt="o", color=color, markersize=2.8,
                        ecolor=color, elinewidth=0.65, capsize=1.5, zorder=3)
        ax.axvline(0, color="#555555", linewidth=0.7, linestyle="--")
        ax.set_yticks(y, show["abbr"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Adjusted MDD − HC effect")
        ax.set_title("Disease association", loc="left", pad=6)
        _clean_axes(ax)
        _panel_label(ax, "A")

        ax = axes[1]
        ax.set_axis_off()
        mismatch = provenance.get("mismatch_fields", [])
        saved = provenance.get("saved", {})
        current = provenance.get("current", {})
        reason = "\n".join(f"• {field}" for field in mismatch) or provenance.get("status", "unavailable")
        expected_hash = str(saved.get("merged_sha256", "not recorded"))
        current_hash = str(current.get("merged_sha256", "not available"))
        expected_map = saved.get("resolved_source_map", {})
        current_map = current.get("resolved_source_map", {})
        expected_med = " + ".join(expected_map.get("medication", ["not recorded"]))
        current_med = " + ".join(current_map.get("medication", ["not available"]))
        ax.text(0.02, 0.97, "Model predictive dependence", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, fontweight="bold")
        ax.text(0.02, 0.82, "NOT ESTIMATED", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold", color=PALETTE["accent"])
        detail_lines = [
            "Validation ROI-history permutation ΔMAE was blocked by checkpoint provenance.",
            "",
            f"Mismatch fields: {reason.replace('• ', '').replace(chr(10), ', ')}",
            "",
            f"Saved merged SHA-256: {expected_hash[:16]}…",
            f"Current merged SHA-256: {current_hash[:16]}…",
            "",
            f"Saved medication source: {expected_med}",
            f"Current medication source: {current_med}",
            "",
            "No model-importance values are plotted or inferred.",
        ]
        details = "\n".join(fill(line, width=43) if line else "" for line in detail_lines)
        ax.text(0.02, 0.68, details, transform=ax.transAxes, ha="left", va="top",
                fontsize=5.2, color="#333333", linespacing=1.35, wrap=True)
        _panel_label(axes[1], "B")
        fig.suptitle("Disease effects and model dependence are separate evidence", 
                     fontsize=8.5, fontweight="bold", y=0.98)
        fig.text(0.19, 0.035,
                 "Panel A is an adjusted group association; Panel B records a provenance-blocked model analysis. "
                 "Neither panel implies causality.",
                 ha="left", va="bottom", fontsize=5.6, color="#444444")
        _save(fig, "roi_importance_vs_disease")


def _model_provenance_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)) if CHECKPOINT.exists() else None,
        "status": "unavailable",
        "mismatch_fields": [],
        "inference_run": False,
        "note": "Validation ROI permutation is gated by exact clinical-table provenance.",
    }
    if not CHECKPOINT.is_file():
        result["note"] = f"Checkpoint not found: {CHECKPOINT}"
        return result
    try:
        import torch
        from utils.dataloader import load_condition_table

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        condition = (checkpoint.get("config", {}).get("condition") or {})
        saved_meta = condition.get("meta") or {}
        args = checkpoint.get("config", {}).get("args") or {}
        clinical_file = str(args.get("clinical_file") or saved_meta.get("condition_file") or "")
        saved_map = saved_meta.get("resolved_source_map") or saved_meta.get("source_map") or {}
        if not clinical_file:
            result["note"] = "Checkpoint has no saved clinical_file; attribution cannot be reconstructed."
            return result
        table = load_condition_table(str(DATA_ROOT), clinical_file, "mdd8")
        current_map = table.audit.alias_sources
        saved_sha = saved_meta.get("merged_sha256")
        current_sha = table.merged_sha256
        hash_matches = None if not saved_sha else saved_sha == current_sha
        source_map_matches = None if not saved_map else saved_map == current_map
        mismatches = []
        if hash_matches is False:
            mismatches.append("merged_sha256")
        if source_map_matches is False:
            mismatches.append("resolved_source_map")
        result.update({
            "status": "blocked_provenance_mismatch" if mismatches else "provenance_match_not_run",
            "mismatch_fields": mismatches,
            "merged_sha256_matches": hash_matches,
            "resolved_source_map_matches": source_map_matches,
            "saved": {
                "clinical_file": clinical_file,
                "merged_sha256": saved_sha,
                "resolved_source_map": saved_map,
            },
            "current": {
                "clinical_file": clinical_file,
                "merged_sha256": current_sha,
                "resolved_source_map": current_map,
            },
            "local_workbooks_with_checkpoint_filename": [
                p.name for p in DATA_ROOT.glob("*.xlsx")
                if p.name == clinical_file
            ],
            "inference_run": False,
            "note": (
                "Validation inference and ROI permutation were not run because the saved "
                "clinical-table hash/source map do not match the current workbook. The "
                "checkpoint's provenance gate was not bypassed."
                if mismatches else
                "Clinical-table provenance matches. Model inference was not run by this "
                "group-level analysis script."
            ),
        })
    except Exception as exc:  # keep disease analysis independent; preserve exact reason
        result["status"] = "unavailable_preflight_error"
        result["note"] = f"Provenance preflight could not complete: {type(exc).__name__}: {exc}"
    return result


def _save_tables_and_summary(
    cohort: pd.DataFrame, roi_values: np.ndarray, edge_values: np.ndarray,
    aal: pd.DataFrame, cohort_summary: dict[str, Any],
    validation: dict[str, Any], provenance: dict[str, Any], design_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = cohort["group"].to_numpy()
    X, diagnosis_index, clusters, _ = _design_matrix(cohort, "diagnosis")
    node_beta, node_se, node_low, node_high, node_p = _fit_cluster_robust(
        X, roi_values, clusters, diagnosis_index)
    node_hc, node_hc_sd, node_mdd, node_mdd_sd = _raw_group_summary(roi_values, groups)
    node_g = _hedges_g(roi_values, groups)
    node_q = _bh_fdr(node_p)
    roi_effects = aal.copy()
    roi_effects["n_hc"] = int((groups == "HC").sum())
    roi_effects["n_mdd"] = int((groups == "MDD").sum())
    roi_effects["mean_hc_fisher_z"] = node_hc
    roi_effects["sd_hc_fisher_z"] = node_hc_sd
    roi_effects["mean_mdd_fisher_z"] = node_mdd
    roi_effects["sd_mdd_fisher_z"] = node_mdd_sd
    roi_effects["raw_mean_difference"] = node_mdd - node_hc
    roi_effects["hedges_g_unadjusted"] = node_g
    roi_effects["adjusted_beta"] = node_beta
    roi_effects["cluster_se"] = node_se
    roi_effects["ci_low"] = node_low
    roi_effects["ci_high"] = node_high
    roi_effects["p_value"] = node_p
    roi_effects["q_value"] = node_q
    roi_effects["fdr_significant"] = node_q < 0.05
    roi_effects["model_delta_mae"] = np.nan
    roi_effects["model_importance_status"] = provenance["status"]
    roi_effects = roi_effects.sort_values("adjusted_beta", key=lambda s: s.abs(), ascending=False)
    roi_effects.to_csv(OUTPUT_DIR / "roi_disease_effects.csv", index=False, encoding="utf-8-sig")

    upper_i, upper_j = np.triu_indices(len(aal), k=1)
    edge_beta, edge_se, edge_low, edge_high, edge_p = _fit_cluster_robust(
        X, edge_values, clusters, diagnosis_index)
    edge_hc, edge_hc_sd, edge_mdd, edge_mdd_sd = _raw_group_summary(edge_values, groups)
    edge_g = _hedges_g(edge_values, groups)
    edge_q = _bh_fdr(edge_p)
    edge_effects = pd.DataFrame({
        "roi_i": upper_i + 1,
        "roi_j": upper_j + 1,
        "roi_i_abbr": aal.iloc[upper_i]["abbr"].to_numpy(),
        "roi_j_abbr": aal.iloc[upper_j]["abbr"].to_numpy(),
        "network_i": aal.iloc[upper_i]["network"].to_numpy(),
        "network_j": aal.iloc[upper_j]["network"].to_numpy(),
        "n_hc": int((groups == "HC").sum()),
        "n_mdd": int((groups == "MDD").sum()),
        "mean_hc_fisher_z": edge_hc,
        "sd_hc_fisher_z": edge_hc_sd,
        "mean_mdd_fisher_z": edge_mdd,
        "sd_mdd_fisher_z": edge_mdd_sd,
        "raw_mean_difference": edge_mdd - edge_hc,
        "hedges_g_unadjusted": edge_g,
        "adjusted_beta": edge_beta,
        "cluster_se": edge_se,
        "ci_low": edge_low,
        "ci_high": edge_high,
        "p_value": edge_p,
        "q_value": edge_q,
        "fdr_significant": edge_q < 0.05,
    })
    edge_effects.to_csv(OUTPUT_DIR / "edge_disease_effects.csv", index=False, encoding="utf-8-sig")

    mdd_mask = groups == "MDD"
    mdd_cohort = cohort.loc[mdd_mask].reset_index(drop=True)
    mdd_node_values = roi_values[mdd_mask]
    hamd_complete = mdd_cohort["HAMD"].notna().to_numpy()
    mdd_cohort = mdd_cohort.loc[hamd_complete].reset_index(drop=True)
    mdd_node_values = mdd_node_values[hamd_complete]
    hamd_X, hamd_index, hamd_clusters, hamd_columns = _design_matrix(mdd_cohort, "HAMD")
    hamd_beta_sd, hamd_se_sd, hamd_low_sd, hamd_high_sd, hamd_p = _fit_cluster_robust(
        hamd_X, mdd_node_values, hamd_clusters, hamd_index)
    hamd_sd = float(mdd_cohort["HAMD"].std(ddof=0))
    hamd_effects = aal.copy()
    hamd_effects["n_mdd"] = int(len(mdd_cohort))
    hamd_effects["hamd_mean"] = float(mdd_cohort["HAMD"].mean())
    hamd_effects["beta_per_hamd_sd"] = hamd_beta_sd
    hamd_effects["beta_per_hamd_point"] = hamd_beta_sd / hamd_sd
    hamd_effects["cluster_se_per_hamd_sd"] = hamd_se_sd
    hamd_effects["ci_low_per_hamd_sd"] = hamd_low_sd
    hamd_effects["ci_high_per_hamd_sd"] = hamd_high_sd
    hamd_effects["p_value"] = hamd_p
    hamd_effects["q_value"] = _bh_fdr(hamd_p)
    hamd_effects["fdr_significant"] = hamd_effects["q_value"] < 0.05
    hamd_effects = hamd_effects.sort_values("beta_per_hamd_sd", key=lambda s: s.abs(), ascending=False)
    hamd_effects.to_csv(OUTPUT_DIR / "roi_hamd_associations.csv", index=False, encoding="utf-8-sig")

    comparison = roi_effects.copy()
    comparison["model_delta_mae"] = np.nan
    comparison["model_importance_status"] = provenance["status"]
    comparison.to_csv(OUTPUT_DIR / "roi_importance_vs_disease.csv", index=False, encoding="utf-8-sig")

    summary = {
        "analysis": "MDD-HC subject-level resting-state functional connectivity",
        "primary_roi_metric": "mean of 115 off-diagonal Fisher-z transformed ROI correlations",
        "secondary_edge_metric": "pairwise Fisher-z transformed ROI correlation",
        "inference": "OLS adjusted for diagnosis/Age/Sex/Education/site fixed effects; CR1 site-clustered 95% CI and p-values; BH-FDR within ROI, edge, and exploratory HAMD families",
        "roi_and_signal_validation": validation,
        "cohort": cohort_summary,
        "matched_source_counts": cohort_summary.get("matched_source_counts", {}),
        "model_roi_importance": provenance,
        "roi_fdr_significant_count": int(roi_effects["fdr_significant"].sum()),
        "edge_fdr_significant_count": int(edge_effects["fdr_significant"].sum()),
        "hamd_fdr_significant_count": int(hamd_effects["fdr_significant"].sum()),
        "hamd_exploratory_n": int(len(mdd_cohort)),
        "hamd_design_columns": hamd_columns,
        "primary_design_columns": design_columns,
        "top_roi_effects": roi_effects[["roi_index", "abbr", "network", "adjusted_beta", "ci_low", "ci_high", "p_value", "q_value"]].head(10).to_dict("records"),
        "limitations": [
            "The input correlation files do not embed ROI names; names are aligned by the project's first-116-column preprocessing order.",
            "AAL116.xlsx contains labels and networks but no MNI coordinates, so no anatomical surface localization is produced.",
            "Per-subject full-series z-scoring prevents interpretation of BOLD means or SDs as absolute MDD-HC amplitude differences.",
            "The group contrast is observational and does not imply causal disease mechanisms.",
            "No treatment pre/post outcomes are used; virtual stimulation or real treatment response is not inferred.",
        ],
    }
    (OUTPUT_DIR / "analysis_cohort_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return roi_effects, edge_effects, hamd_effects, summary


def run_analysis(verify_signal_samples: int = 5) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(NATURE_RC):
        aal = _load_roi_table(DATA_ROOT / "AAL116.xlsx")
        clinical = {group: _read_clinical(group) for group in ("HC", "MDD")}
        records, source_counts = _list_correlation_records(clinical)
        cohort, cohort_summary = _select_primary_cohort(records)
        roi_values, edge_values, validation = _load_connectivity_features(
            cohort, len(aal), signal_checks_per_group=verify_signal_samples)
        cohort_summary["matched_source_counts"] = source_counts
        cohort_summary["subject_unit"] = "one participant, one correlation matrix"
        X, _, _, design_columns = _design_matrix(cohort, "diagnosis")
        provenance = _model_provenance_status()
        (OUTPUT_DIR / "model_roi_importance_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        roi_effects, edge_effects, hamd_effects, summary = _save_tables_and_summary(
            cohort, roi_values, edge_values, aal, cohort_summary, validation,
            provenance, design_columns)
        provenance = summary["model_roi_importance"]
        _plot_roi_effects(roi_effects, cohort)
        _plot_edge_heatmap(edge_effects, aal)
        _plot_importance_disease(roi_effects, provenance)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-signal-samples", type=int, default=5,
                        help="per-group spot checks against normalized signal columns (default: 5)")
    args = parser.parse_args()
    summary = run_analysis(verify_signal_samples=args.verify_signal_samples)
    print(json.dumps({
        "cohort": summary["cohort"],
        "roi_fdr_significant_count": summary["roi_fdr_significant_count"],
        "edge_fdr_significant_count": summary["edge_fdr_significant_count"],
        "hamd_fdr_significant_count": summary["hamd_fdr_significant_count"],
        "model_roi_importance_status": summary["model_roi_importance"]["status"],
        "output_dir": str(OUTPUT_DIR),
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
