"""Create publication figures from the saved NeuroTwin test evaluation.

This script is read-only with respect to checkpoints and data. It consumes the
existing metrics_test.json and per_subject_metrics_test.csv; it does not run
inference or regenerate test predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Match the Nature-style settings and muted palette in analysis/visualizer.py.
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
    "legend.fontsize": 6,
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
    "ar1": "#C45C3E",
    "accent": "#8E6B8A",
    "light": "#B0B0B0",
    "grid": "#E6E6E6",
}
METHODS = [
    ("model", "TFM"),
    ("persistence", "Persistence"),
    ("trend", "Trend"),
    ("ar1", "AR(1)"),
]


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2.5, width=0.6, color="#555555")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def plot_performance(metrics: dict, output_dir: Path) -> None:
    one_step = metrics["next_state"]["model"]["t+1"]
    baselines = metrics["baselines"]
    names = [label for _, label in METHODS]
    colors = [PALETTE[key] for key, _ in METHODS]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.75))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.18,
                        wspace=0.34, hspace=0.47)
    fig.suptitle("NeuroTwin test-set performance", fontsize=9, fontweight="bold", y=0.98)

    for ax, metric, key, ylabel, panel in [
        (axes[0, 0], "MAE", "MAE_std", "MAE (lower is better)", "A"),
        (axes[0, 1], "PCC_spatial", "PCC_spatial_std", "Spatial PCC (higher is better)", "B"),
    ]:
        values = [one_step[metric]] + [baselines[k]["t+1"][metric] for k, _ in METHODS[1:]]
        errors = [one_step[key]] + [baselines[k]["t+1"][key] for k, _ in METHODS[1:]]
        bars = ax.bar(np.arange(len(names)), values, color=colors, width=0.68,
                      edgecolor="none", zorder=2)
        ax.errorbar(np.arange(len(names)), values, yerr=errors, fmt="none",
                    ecolor="#333333", elinewidth=0.65, capsize=2, capthick=0.65, zorder=3)
        ax.set_xticks(np.arange(len(names)), names, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.55, len(names) - 0.45)
        ax.set_ylim(bottom=0)
        ax.set_title("One-step forecast · t+1", loc="left", pad=5)
        _clean_axes(ax)
        _panel_label(ax, panel)
        span = max(values) - min(values)
        offset = max(span * 0.025, max(values) * 0.025)
        for bar, value, error in zip(bars, values, errors):
            ax.text(bar.get_x() + bar.get_width() / 2, value + error + offset,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=5.5)

    rollout = metrics["rollout"]
    horizons = ["H1", "H2", "H4", "H8", "H16"]
    x = np.arange(len(horizons))
    for ax, metric, std_key, ylabel, panel in [
        (axes[1, 0], "MAE", "MAE_std", "MAE (lower is better)", "C"),
        (axes[1, 1], "PCC_spatial", "PCC_spatial_std", "Spatial PCC (higher is better)", "D"),
    ]:
        for method, label in [("model", "TFM"), ("persistence", "Persistence")]:
            block = rollout["model"] if method == "model" else rollout["baselines"][method]
            ys = np.array([block[h][metric] for h in horizons], dtype=float)
            sd = np.array([block[h][std_key] for h in horizons], dtype=float)
            ax.plot(x, ys, marker="o", markersize=3.2, color=PALETTE[method], label=label)
            ax.fill_between(x, ys - sd, ys + sd, color=PALETTE[method], alpha=0.13, linewidth=0)
        ax.set_xticks(x, horizons)
        ax.set_xlabel("Rollout horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title("Autoregressive rollout · 450 tasks", loc="left", pad=5)
        _clean_axes(ax)
        _panel_label(ax, panel)
        ax.legend(loc="best", handlelength=1.4)

    caveat = (
        f"Test set: {metrics['protocol']['n_subjects']} subjects; "
        f"{metrics['protocol']['n_tasks']:,} one-step tasks. Shading/error bars: reported SD. "
        "Evaluation is conditional on the saved preprocessing: each ROI BOLD series was z-scored over its full time series."
    )
    fig.text(0.09, 0.035, caveat, ha="left", va="bottom", fontsize=5.6, color="#444444", wrap=True)
    _save(fig, output_dir, "model_performance")


def _hamd_quintiles(subject: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    frame = subject[["hamd", "t+1_MAE"]].dropna().copy()
    frame["quintile"] = pd.qcut(frame["hamd"], q=5, labels=False, duplicates="drop")
    summary = []
    for q, group in frame.groupby("quintile", observed=True):
        summary.append({
            "q": int(q),
            "hamd_min": float(group["hamd"].min()),
            "hamd_max": float(group["hamd"].max()),
            "n": int(len(group)),
            "values": group["t+1_MAE"].to_numpy(),
            "hamd_median": float(group["hamd"].median()),
            "mae_median": float(group["t+1_MAE"].median()),
        })
    return frame, summary


def plot_clinical_pathology(subject: pd.DataFrame, output_dir: Path) -> dict:
    frame, groups = _hamd_quintiles(subject)
    rho, p_value = spearmanr(frame["hamd"], frame["t+1_MAE"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.26, wspace=0.34)
    fig.suptitle("Clinical phenotype and prediction error", fontsize=9, fontweight="bold", y=0.97)

    ax = axes[0]
    ax.scatter(frame["hamd"], frame["t+1_MAE"], s=13, color=PALETTE["model"],
               alpha=0.48, edgecolors="none", rasterized=True, zorder=2)
    med_x = [g["hamd_median"] for g in groups]
    med_y = [g["mae_median"] for g in groups]
    ax.plot(med_x, med_y, marker="D", markersize=3.3, color=PALETTE["accent"],
            linewidth=1.0, label="Quintile median", zorder=3)
    ax.axhline(frame["t+1_MAE"].median(), color=PALETTE["light"], linewidth=0.8,
               linestyle="--", label="Cohort median")
    ax.set_xlabel("HAMD score")
    ax.set_ylabel("Subject-level t+1 MAE")
    ax.set_title("Continuous HAMD association", loc="left", pad=5)
    ax.text(0.04, 0.96, f"Spearman ρ = {rho:.3f}\np = {p_value:.3f}; n = {len(frame)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax.legend(loc="lower right", handlelength=1.3)
    _clean_axes(ax)
    _panel_label(ax, "A")

    ax = axes[1]
    box_data = [g["values"] for g in groups]
    bp = ax.boxplot(box_data, positions=np.arange(1, len(groups) + 1), widths=0.58,
                    patch_artist=True, showfliers=False, medianprops={"color": "#252525", "linewidth": 1.0},
                    whiskerprops={"color": "#777777", "linewidth": 0.7},
                    capprops={"color": "#777777", "linewidth": 0.7},
                    boxprops={"edgecolor": "#777777", "linewidth": 0.7})
    shades = ["#DCE4EF", "#C5D2E3", "#AABCD4", "#8CA5C5", PALETTE["model"]]
    for patch, shade in zip(bp["boxes"], shades):
        patch.set_facecolor(shade)
        patch.set_alpha(0.95)
    labels = [f"Q{g['q'] + 1}\n{g['hamd_min']:g}–{g['hamd_max']:g}\nn={g['n']}" for g in groups]
    ax.set_xticks(np.arange(1, len(groups) + 1), labels)
    ax.set_xlabel("HAMD quintile · observed range")
    ax.set_ylabel("Subject-level t+1 MAE")
    ax.set_title("Error across cohort-specific quintiles", loc="left", pad=5)
    _clean_axes(ax)
    _panel_label(ax, "B")

    fig.text(0.09, 0.055,
             "HAMD quintiles are cohort-relative groups, not diagnostic cutoffs. Whiskers extend to the most extreme observation within 1.5×IQR.",
             ha="left", va="bottom", fontsize=5.7, color="#444444")
    _save(fig, output_dir, "clinical_pathology")
    return {"n": int(len(frame)), "spearman_rho": float(rho), "spearman_p": float(p_value),
            "quintiles": [{k: g[k] for k in ("q", "hamd_min", "hamd_max", "n", "mae_median")} for g in groups]}


def plot_individual_subject_gain(subject: pd.DataFrame, output_dir: Path) -> dict:
    comparisons = [
        ("persistence_t+1_MAE", "Persistence", PALETTE["persistence"]),
        ("trend_t+1_MAE", "Trend", PALETTE["trend"]),
        ("ar1_t+1_MAE", "AR(1)", PALETTE["ar1"]),
    ]
    frame = subject.dropna(subset=["t+1_MAE", *(col for col, _, _ in comparisons)]).copy()
    deltas = [frame["t+1_MAE"].to_numpy() - frame[col].to_numpy()
              for col, _, _ in comparisons]
    all_values = np.concatenate(deltas)
    left = min(float(all_values.min()) - 0.06, -0.10)
    right = max(float(all_values.max()) + 0.08, 0.12)

    fig, ax = plt.subplots(figsize=(6.7, 2.65))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.81, bottom=0.26)
    fig.suptitle("Individual-level advantage over simple baselines", fontsize=9,
                 fontweight="bold", y=0.97)
    rng = np.random.default_rng(20261016)
    rows = []
    for y, ((_, label, color), values) in enumerate(zip(comparisons, deltas)):
        ax.scatter(values, y + rng.uniform(-0.13, 0.13, size=len(values)),
                   s=9, color=color, alpha=0.38, edgecolors="none", rasterized=True, zorder=2)
        bp = ax.boxplot([values], positions=[y], widths=0.36, orientation="horizontal",
                        patch_artist=True, showfliers=False, manage_ticks=False,
                        medianprops={"color": "#222222", "linewidth": 1.0},
                        whiskerprops={"color": "#666666", "linewidth": 0.7},
                        capprops={"color": "#666666", "linewidth": 0.7},
                        boxprops={"edgecolor": "#666666", "linewidth": 0.7})
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.48)
        wins = int(np.sum(values < 0))
        rows.append({"baseline": label, "n": int(len(values)), "wins": wins,
                     "delta_median": float(np.median(values)),
                     "delta_q25": float(np.quantile(values, 0.25)),
                     "delta_q75": float(np.quantile(values, 0.75))})
        ax.text(min(0.015, right - 0.05), y, f"{wins}/{len(values)} lower",
                ha="left", va="center", fontsize=5.8, color="#333333")

    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(left, right)
    ax.set_yticks(np.arange(len(comparisons)), [name for _, name, _ in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel("Subject-wise ΔMAE (TFM − baseline; negative favors TFM)")
    ax.set_title("Paired subject means across the saved test anchors", loc="left", pad=5)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2.5, width=0.6, color="#555555")
    _panel_label(ax, "A")

    fig.text(0.17, 0.055,
             "Each point is one subject. This export has 17 tasks per subject: 16 configured forecast anchors plus one supplemental non-overlapping rollout anchor.",
             ha="left", va="bottom", fontsize=5.6, color="#444444")
    _save(fig, output_dir, "individual_subject_gain")
    return {"n_subjects": int(len(frame)), "comparisons": rows}


def plot_network_behavior(metrics: dict, output_dir: Path) -> None:
    fc = metrics["fc"]
    groups = ["within", "between"]
    method_keys = [key for key, _ in METHODS]
    method_names = [label for _, label in METHODS]
    x = np.arange(len(groups))
    width = 0.18
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.34, wspace=0.34)
    fig.suptitle("Network-level functional connectivity behavior", fontsize=9, fontweight="bold", y=0.97)

    for ax, metric, error_key, ylabel, title, panel in [
        (axes[0], "fc_mae", "fc_mae_subject_std", "FC MAE (lower is better)", "Connectivity magnitude error", "A"),
        (axes[1], "edge_pcc_mean", "edge_pcc_std", "Edge-wise PCC (higher is better)", "Connectivity edge agreement", "B"),
    ]:
        _clean_axes(ax)
        for i, method in enumerate(method_keys):
            values, errors = [], []
            for group in groups:
                block = fc[method]["network"][group]
                value = block[metric]
                if metric == "fc_mae":
                    value = block["fc_mae_subject_mean"]
                values.append(value)
                errors.append(block[error_key])
            ax.bar(x + offsets[i], values, width=width * 0.9,
                   color=PALETTE[method], edgecolor="none", label=method_names[i], zorder=2)
            ax.errorbar(x + offsets[i], values, yerr=errors, fmt="none", ecolor="#333333",
                        elinewidth=0.55, capsize=1.7, capthick=0.55, zorder=3)
        ax.set_xticks(x, ["Within network\n1,222 edges", "Between networks\n5,448 edges"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=5)
        _panel_label(ax, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.19), ncol=4,
               columnspacing=1.2, handlelength=1.1, frameon=False)
    axes[1].axhline(0, color="#777777", linewidth=0.6, linestyle="--", zorder=1)

    n_tasks = fc["model"]["n_tasks"]
    fig.text(0.09, 0.035,
             f"AAL116 network partition; FC evaluated over 32-step rollouts ({n_tasks} tasks). Error bars: reported SD. "
             "Output-level behavior only; ROI feature importance is unavailable in the saved test evaluation.",
             ha="left", va="bottom", fontsize=5.6, color="#444444", wrap=True)
    _save(fig, output_dir, "network_behavior")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path,
                        default=repo_root / "checkpoints" / "neurotwin_tfm_finetune" / "eval")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    metrics_path = args.eval_dir / "metrics_test.json"
    subject_path = args.eval_dir / "per_subject_metrics_test.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    subject = pd.read_csv(subject_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(NATURE_RC)
    plot_performance(metrics, args.out_dir)
    pathology = plot_clinical_pathology(subject, args.out_dir)
    individual_gain = plot_individual_subject_gain(subject, args.out_dir)
    plot_network_behavior(metrics, args.out_dir)

    print(f"Saved PNG/PDF figures to {args.out_dir}")
    print(json.dumps({"clinical_pathology": pathology,
                      "individual_subject_gain": individual_gain,
                      "feature_importance_available": metrics.get("feature_importance", {}).get("available", False),
                      "test_subjects": metrics["protocol"]["n_subjects"],
                      "test_tasks": metrics["protocol"]["n_tasks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
