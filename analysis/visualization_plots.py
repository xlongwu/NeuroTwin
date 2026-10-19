"""Paper-style multipanel figures; only 300-DPI PNGs are written.

Visual references: plot-is-all-you-need / pred-vs-obs-model-grid (navy,
muted rose, fine grids and identity plots), oa-bio-plos-pcbi-1008657-f02
(paired heatmaps, aligned subject order and mean/interval panels).
"""

import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.study_stats import pearson
from analysis.visualization_data import HORIZONS, effects, network_edges, network_mean

HC, MDD = "#749ac6", "#cd7f6d"
REAL, GEN, GOLD, GREY = "#222d5a", "#cd7f6d", "#bb913d", "#98a1ac"
GROUP_COLORS = {"HC": HC, "MDD": MDD}
FEATURES = ["Log temporal SD", "Successive RMS", "Lag-1 ACF"]


def style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#59636e",
            "text.color": "#202832",
            "axes.labelcolor": "#202832",
            "grid.color": "#e3e7eb",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def figure(title, rows=1, cols=1, size=(13.5, 8), **kwargs):
    fig, ax = plt.subplots(
        rows, cols, figsize=size, squeeze=False, layout="constrained", **kwargs
    )
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.02, ha="left")
    for i, a in enumerate(ax.flat):
        a.text(
            -0.10,
            1.06,
            string.ascii_uppercase[i],
            transform=a.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    return fig, ax


def save(d, fig, filename, note):
    fig.supxlabel(note, fontsize=8, color="#59636e", ha="left", x=0.02)
    path = d["args"].output / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def ci(values, repeats=1000):
    x = np.asarray(values, float)
    x = x[
        np.isfinite(x).all(axis=tuple(range(1, x.ndim)))
        if x.ndim > 1
        else np.isfinite(x)
    ]
    if not len(x):
        return np.full(values.shape[1:], np.nan), np.full(
            (2, *values.shape[1:]), np.nan
        )
    index = np.random.default_rng(2024).integers(len(x), size=(repeats, len(x)))
    return x.mean(0), np.quantile(x[index].mean(1), [0.025, 0.975], axis=0)


def band(ax, x, values, color, label, ls="-"):
    mean, interval = ci(values)
    ax.plot(x, mean, color=color, lw=1.7, label=label, ls=ls)
    ax.fill_between(x, interval[0], interval[1], color=color, alpha=0.15, linewidth=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def heat(
    fig, ax, values, title, vmin, vmax, cmap="RdBu_r", xlabel="", ylabel="", ticks=None
):
    im = ax.imshow(
        values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto", interpolation="nearest"
    )
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if ticks is not None:
        ax.set_xticks(np.arange(len(ticks)), ticks, rotation=90, fontsize=6)
    cb = fig.colorbar(
        im, ax=ax, fraction=0.035, pad=0.025, extend="both" if vmin < 0 else "neither"
    )
    cb.ax.tick_params(labelsize=8)
    return im


def roi_heat(
    d,
    fig,
    ax,
    values,
    title,
    vmin,
    vmax,
    rows,
    cmap="RdBu_r",
    significance=None,
    label_stride=1,
):
    order = d["order"]
    heat(fig, ax, values[:, order], title, vmin, vmax, cmap, ticks=d["labels"][order])
    if label_stride > 1:
        selected = np.arange(0, len(order), label_stride)
        ax.set_xticks(selected, d["labels"][order][selected], rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)), rows)
    bounds = np.cumsum([sum(d["networks"] == n) for n in d["names"]])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=1)
    for a, b, n in zip(np.r_[0, bounds[:-1]], bounds, d["names"]):
        ax.text(
            (a + b - 1) / 2,
            1.02,
            n,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
    if significance is not None:
        y, x = np.where(significance[:, order] < 0.05)
        ax.scatter(x, y, marker=".", c="black", s=8)


def scatter(ax, x, y, color=REAL, title="", xlabel="", ylabel="", identity=False):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    ax.scatter(x, y, s=15, color=color, alpha=0.55, linewidths=0)
    if len(x) > 2 and np.std(x) > 1e-10:
        fit = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, np.polyval(fit, xx), color=GOLD, lw=1.5)
    if identity and len(x):
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=0.9)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.text(
        0.03,
        0.97,
        f"r = {pearson(x, y):.2f}  |  n = {len(x)}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(fc="white", ec="none", alpha=0.8),
    )
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)


def representative(d, group):
    idx = np.flatnonzero(d["groups"] == group)
    score = d["metrics"]["Common HC model"][idx, 0, 0]
    return idx[np.argsort(score)[len(idx) // 2]]


def forecast_figures(d):
    fig, ax = figure("Forecast accuracy across horizons", 2, 3, size=(14, 8))
    styles = {
        "Common HC model": (REAL, "-"),
        "Persistence": (GREY, "--"),
        "Ridge AR(1)": (GOLD, "-."),
        "MDD fine-tuned": (MDD, "-"),
    }
    for r, group in enumerate(("HC", "MDD")):
        mask = d["groups"] == group
        for j, name in enumerate(("MAE", "RMSE", "Pearson r")):
            a = ax[r, j]
            for model, (color, ls) in styles.items():
                if group == "HC" and model == "MDD fine-tuned":
                    continue
                band(a, HORIZONS, d["metrics"][model][mask, :, j], color, model, ls)
            a.set(
                title=f"{group}  |  n = {mask.sum()}",
                xlabel="Forecast horizon (TR)",
                ylabel=name,
                xticks=HORIZONS,
            )
            if j == 0:
                a.legend(loc="upper left", fontsize=7)
    save(
        d,
        fig,
        "01_forecast_performance.png",
        "Frozen test splits; 64-TR history; scores at each horizon, then equal subject averages. Shading: 95% subject bootstrap CI.\n"
        "Common HC checkpoint is shared across groups; MDD fine-tuned checkpoint is shown separately. Ridge: per-ROI AR(1), pooled train subjects only.",
    )

    err = np.stack([d["roi_error"][d["groups"] == g].mean(0) for g in ("HC", "MDD")])
    fig, ax = figure(
        "Where prediction errors occur",
        2,
        2,
        size=(14, 15),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    vmax = err.max()
    for row, subset in enumerate(np.array_split(d["order"], 2)):
        heat(
            fig,
            ax[row, 0],
            err[:, subset].T,
            "AAL116 ROI error (H1–8 mean)",
            0,
            vmax,
            "YlGnBu",
        )
        ax[row, 0].set_yticks(range(len(subset)), d["labels"][subset], fontsize=7)
        ax[row, 0].set_xticks([0, 1], ["HC", "MDD"])
    net = network_mean(d["roi_error"], d["networks"], d["names"])
    for k, group in enumerate(("HC", "MDD")):
        mean, interval = ci(net[d["groups"] == group])
        x = np.arange(len(mean)) + (k - 0.5) * 0.24
        ax[0, 1].errorbar(
            x,
            mean,
            yerr=np.maximum(np.stack([mean - interval[0], interval[1] - mean]), 0),
            fmt="o",
            color=GROUP_COLORS[group],
            capsize=3,
            label=group,
        )
    ax[0, 1].set(
        xticks=range(len(d["names"])),
        xticklabels=d["names"],
        ylabel="MAE",
        title="Network mean error",
    )
    ax[0, 1].legend()
    for k, field in enumerate(("Sex", "Dataset")):
        vals = [s.metadata.get(field) for s in d["test"]]
        for group in ("HC", "MDD"):
            cats = [1.0, 2.0] if field == "Sex" else ["V1", "V2"]
            for c, cat in enumerate(cats):
                selected = (d["groups"] == group) & (np.asarray(vals) == cat)
                a = d["roi_error"][selected].mean(1)
                if len(a):
                    x = k * 3 + c + (-0.10 if group == "HC" else 0.10)
                    mean, interval = ci(a)
                    ax[1, 1].errorbar(
                        x,
                        mean,
                        yerr=np.maximum(
                            [[mean - interval[0]], [interval[1] - mean]], 0
                        ),
                        fmt="o",
                        color=GROUP_COLORS[group],
                        capsize=3,
                    )
                    ax[1, 1].annotate(
                        f"n={len(a)}",
                        (x, mean),
                        xytext=(0, 9 if group == "HC" else -15),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
    ax[1, 1].set(
        xticks=[0, 1, 3, 4],
        xticklabels=["Sex 1", "Sex 2", "V1", "V2"],
        ylabel="Subject mean MAE",
        title="Recorded subgroup summaries",
    )
    save(
        d,
        fig,
        "02_roi_prediction_error.png",
        "Common HC model; all 116 ROI labels in network order. Both ROI panels share one color scale. Network/subgroup bars: 95% subject bootstrap CI.\nSex codes follow the clinical table; V1/V2 are data versions, not verified scanner sites.",
    )

    fig, ax = figure("Observed and predicted BOLD", 2, 3, size=(15, 7))
    for row, group in enumerate(("HC", "MDD")):
        i = representative(d, group)
        errors = d["roi_error"][i]
        rois = np.argsort(errors)[
            [len(errors) // 4, len(errors) // 2, 3 * len(errors) // 4]
        ]
        indices = np.flatnonzero(d["owner"] == i)
        times = np.array([d["tasks"][j][1] + 1 for j in indices])
        for col, roi in enumerate(rois):
            a = ax[row, col]
            a.plot(
                np.arange(64, 150),
                d["test"][i].bold[64:, roi],
                c=REAL,
                lw=1.3,
                label="Observed",
            )
            for key, color in [("Common HC model", HC), ("MDD fine-tuned", MDD)]:
                if group == "HC" and key == "MDD fine-tuned":
                    continue
                a.plot(
                    times,
                    d["predictions"][key][indices, 0, roi],
                    "o",
                    color=color,
                    ms=3,
                    label=f"{key}, H1",
                )
                j = indices[len(indices) // 2]
                a.plot(
                    np.arange(d["tasks"][j][1] + 1, d["tasks"][j][1] + 9),
                    d["predictions"][key][j, :, roi],
                    color=color,
                    lw=2,
                    ls="--",
                    label="8-TR forecast" if key == "Common HC model" else None,
                )
            a.set(
                title=f"{group} subject {i + 1:03d}  |  {d['labels'][roi]}",
                xlabel="Scan time (TR, zero based)",
                ylabel="Prefix-standardized BOLD",
            )
            if col == 0:
                a.legend(fontsize=7)
    save(
        d,
        fig,
        "03_bold_prediction_examples.png",
        "Subjects selected by median H1 MAE within each group; ROIs selected at the 25th, 50th and 75th error percentiles.\nMarkers: independent H1 forecasts from real history; dashed line: one uninterrupted 8-TR forecast. Anonymous subject indices match the analysis cohort.",
    )


def dynamics_figures(d):
    fig, ax = figure("Temporal dynamics beyond short forecasts", 2, 3, size=(14, 8))
    freq = d["freq"]
    for row, group in enumerate(("HC", "MDD")):
        m = d["groups"] == group
        for key, color in [("Real", REAL), ("Generated", GEN)]:
            band(ax[row, 0], np.arange(21), d["acf"][key][m], color, key)
            band(ax[row, 1], freq[1:], d["psd"][key][m, 1:], color, key)
            bands = [(0, 0.05), (0.05, 0.15), (0.15, 0.501)]
            powers = np.stack(
                [
                    d["psd"][key][m][:, (freq >= lo) & (freq < hi)].sum(1)
                    for lo, hi in bands
                ],
                -1,
            )
            powers /= d["psd"][key][m].sum(1)[:, None]
            band(ax[row, 2], np.arange(3), powers, color, key)
        ax[row, 0].set(
            title=f"{group}: ROI-average ACF",
            xlabel="Lag (TR)",
            ylabel="Autocorrelation",
        )
        ax[row, 0].legend()
        ax[row, 1].set(
            title=f"{group}: Welch power spectrum",
            xlabel="Frequency (cycles/TR)",
            ylabel="PSD (BOLD² / cycles/TR)",
            yscale="log",
        )
        ax[row, 2].set(
            title=f"{group}: relative spectral power",
            ylabel="Fraction of spectral power",
            xticks=range(3),
            xticklabels=["0–0.05", "0.05–0.15", "0.15–0.50"],
        )
        ax[row, 2].set_xlabel("Frequency band (cycles/TR)")
    save(
        d,
        fig,
        "04_temporal_dynamics.png",
        f"Common HC model; matched observed/generated {d['length']}-TR segments after the same 64-TR history. Shading: 95% subject bootstrap CI.\nOne continuous ODE integration, no future observations; horizons beyond trained H1–8 are extrapolation. TR duration unverified: no conversion to Hz.",
    )

    fig, ax = figure("Static functional connectivity fidelity", 2, 3, size=(13.5, 9))
    order = d["order"]
    tri = np.triu_indices(len(order), 1)
    matrices = {}
    for group in ("HC", "MDD"):
        m = d["groups"] == group
        for key in ("Real", "Generated"):
            mean_z = np.nanmean(d["edges"][key][m], 0)
            mat = np.eye(len(order))
            mat[tri] = mat[(tri[1], tri[0])] = np.tanh(mean_z)
            matrices[group, key] = mat[np.ix_(order, order)]
    lim = max(
        np.nanmax(np.abs(matrices[g, "Generated"] - matrices[g, "Real"]))
        for g in ("HC", "MDD")
    )
    for row, group in enumerate(("HC", "MDD")):
        a, b = matrices[group, "Real"], matrices[group, "Generated"]
        for col, (matrix, title, low, high) in enumerate(
            [
                (a, "Observed", -1, 1),
                (b, "Generated", -1, 1),
                (b - a, "Generated − observed", -lim, lim),
            ]
        ):
            heat(fig, ax[row, col], matrix, f"{group}: {title}", low, high)
            network_ticks(d, ax[row, col])
    save(
        d,
        fig,
        "05_functional_connectivity.png",
        "Subject FC averaged in Fisher-z space and transformed back to r; AAL116 ROIs in network order. FC scales and difference scales are shared across cohorts.\nCommon HC model; continuous 86-TR generated segments. Diagonal is structural self-correlation and excluded from edge analyses.",
    )

    fig, ax = figure("Dynamic functional connectivity", 2, 3, size=(14, 8))
    starts = np.arange(d["dfc"]["Real"].shape[1]) * 4 + 16 + 64
    for row, group in enumerate(("HC", "MDD")):
        m = d["groups"] == group
        for key, color in [("Real", REAL), ("Generated", GEN)]:
            seq = d["dfc"][key][m]
            strength = np.nanmean(np.abs(np.tanh(seq)), 2)
            band(ax[row, 0], starts, strength, color, key)
            variation = np.nanstd(seq, axis=1).mean(1)
            x = np.sort(variation[np.isfinite(variation)])
            ax[row, 1].step(
                x, np.arange(1, len(x) + 1) / len(x), where="post", c=color, label=key
            )
            # Each line contributes one subject summary, not pseudo-replicated windows.
        a = np.nanstd(d["dfc"]["Real"][m], 1).mean(1)
        b = np.nanstd(d["dfc"]["Generated"][m], 1).mean(1)
        scatter(
            ax[row, 2],
            a,
            b,
            GROUP_COLORS[group],
            f"{group}: subject dFC variability",
            "Observed edge SD (Fisher z)",
            "Generated edge SD (Fisher z)",
            True,
        )
        ax[row, 0].set(
            title=f"{group}: window mean |FC|",
            xlabel="Window center (scan TR)",
            ylabel="Mean |r|",
        )
        ax[row, 0].legend()
        ax[row, 1].set(
            title=f"{group}: between-window variability",
            xlabel="Subject mean edge SD (Fisher z)",
            ylabel="Cumulative subject fraction",
        )
        ax[row, 1].legend()
    save(
        d,
        fig,
        "06_dynamic_connectivity.png",
        "32-TR sliding windows, stride 4 TR, within uninterrupted matched observed/generated segments; no pooling of windows as independent subjects.\nCommon HC model; shaded curves show 95% subject bootstrap CI. Finite-window dFC is descriptive and may include sampling variability.",
    )


def network_ticks(d, ax):
    sizes = np.array([sum(d["networks"] == n) for n in d["names"]])
    bounds = np.cumsum(sizes)
    mid = bounds - sizes / 2 - 0.5
    ax.set_xticks(mid, d["names"], rotation=45, ha="right")
    ax.set_yticks(mid, d["names"])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=0.45)
        ax.axhline(b - 0.5, color="white", lw=0.45)


def pathology_figures(d):
    result = d["group_results"]
    labels = d["labels"]
    roi = np.array([[r[0] for r in result[k]["roi"]] for k in ("Real", "Generated")])
    lim = max(0.5, np.nanmax(np.abs(roi)))
    fig, ax = figure(
        "MDD–HC differences in ROI activity",
        3,
        1,
        size=(17, 10),
        gridspec_kw={"height_ratios": [1, 1, 1.1]},
    )
    for row, key in enumerate(("Real", "Generated")):
        q = np.array([r[1] for r in result[key]["roi"]])
        roi_heat(
            d,
            fig,
            ax[row, 0],
            roi[row],
            f"{key}: Hedges g (MDD − HC)",
            -lim,
            lim,
            FEATURES,
            significance=q,
        )
    adjusted, q, n = d["adjusted"]
    roi_heat(
        d,
        fig,
        ax[2, 0],
        adjusted[None],
        f"Observed log SD: adjusted standardized group coefficient (complete-case n = {n})",
        -lim,
        lim,
        ["Adjusted"],
        significance=q[None],
    )
    save(
        d,
        fig,
        "07_mdd_hc_roi_differences.png",
        "Dots: BH q < 0.05, separately across 116 ROIs for each feature/source. Hedges g: subject-level Welch comparison; positive = MDD higher.\nAdjusted panel: age, sex, education and recorded data version, HC3 errors. Site/FD unavailable. Generated groups share the same HC checkpoint; cross-sectional associations only.",
    )

    n = len(labels)
    tri = np.triu_indices(n, 1)
    order = d["order"]
    fig, ax = figure("MDD–HC functional connectivity differences", 2, 2, size=(12, 10))
    lim = max(
        0.3, *[np.nanmax(np.abs(result[k]["fc"][0])) for k in ("Real", "Generated")]
    )
    net_mats = {}
    for row, key in enumerate(("Real", "Generated")):
        g, q = result[key]["fc"]
        mat = np.zeros((n, n))
        mat[tri] = mat[(tri[1], tri[0])] = g
        heat(fig, ax[row, 0], mat[np.ix_(order, order)], f"{key}: Hedges g", -lim, lim)
        network_ticks(d, ax[row, 0])
        sig = np.zeros((n, n))
        sig[tri] = sig[(tri[1], tri[0])] = q < 0.05
        heat(
            fig,
            ax[row, 1],
            sig[np.ix_(order, order)],
            f"{key}: FDR-significant edges ({(q < 0.05).sum():,})",
            0,
            1,
            "Greys",
        )
        network_ticks(d, ax[row, 1])
        net_mats[key], pairs = network_edges(d["edges"][key], d["networks"], d["names"])
    d["pathology_network"] = {
        key: effects(v[d["groups"] == "MDD"], v[d["groups"] == "HC"])
        for key, v in net_mats.items()
    }
    save(
        d,
        fig,
        "08_mdd_hc_fc_differences.png",
        "MDD minus HC; Welch tests on one Fisher-z FC vector per subject, BH correction across all 6,670 unique edges separately for real/generated data.\nBlack cells: q < 0.05; zero diagonal is excluded from inference. Common HC checkpoint; unadjusted descriptive group effects, not causal or MDD-specific evidence.",
    )

    fig, ax = figure(
        "Does the model preserve the group-difference pattern?", 2, 3, size=(14, 8)
    )
    for j in range(3):
        scatter(
            ax[0, j],
            roi[0, j],
            roi[1, j],
            REAL,
            FEATURES[j],
            "Observed Hedges g",
            "Generated Hedges g",
            True,
        )
    scatter(
        ax[1, 0],
        result["Real"]["fc"][0],
        result["Generated"]["fc"][0],
        HC,
        "FC effect pattern (all edges)",
        "Observed Hedges g",
        "Generated Hedges g",
        True,
    )
    scatter(
        ax[1, 1],
        d["pathology_network"]["Real"][0],
        d["pathology_network"]["Generated"][0],
        MDD,
        "Network-pair FC effects",
        "Observed Hedges g",
        "Generated Hedges g",
        True,
    )
    vals = np.stack(
        [network_mean(roi[k, 0], d["networks"], d["names"]) for k in range(2)]
    )
    heat(
        fig,
        ax[1, 2],
        vals,
        "Network mean ROI-amplitude effect",
        -lim,
        lim,
        ticks=d["names"],
    )
    ax[1, 2].set_yticks([0, 1], ["Observed", "Generated"])
    save(
        d,
        fig,
        "09_pathology_pattern_reproduction.png",
        "All observed/generated features use matched future segments and one shared HC checkpoint. Pearson r summarizes spatial agreement, not independent-edge significance.\nEdges/ROIs are spatially dependent; no p-value treats them as independent samples. Group effects remain descriptive cross-sectional contrasts.",
    )


def normative_figures(d):
    m = d["groups"] == "MDD"
    indices = np.flatnonzero(m)
    zr, zp = d["z"]["Real"][m], d["z"]["Generated"][m]
    severity = np.sqrt(np.nanmean(zr**2, 1))
    valid = np.flatnonzero(np.isfinite(severity))
    ranked = valid[np.argsort(severity[valid])]
    chosen = ranked[[len(ranked) // 4, len(ranked) // 2, 3 * len(ranked) // 4]]
    fig, ax = figure(
        "Individual departures from the HC reference",
        3,
        3,
        size=(16, 10),
        gridspec_kw={"width_ratios": [1.4, 1, 1]},
    )
    for row, index in enumerate(chosen):
        vals = np.stack([zr[index], zp[index]])
        roi_heat(
            d,
            fig,
            ax[row, 0],
            vals,
            f"MDD subject {indices[index] + 1:03d}",
            -3,
            3,
            ["Observed", "Generated"],
            label_stride=4,
        )
        nv = network_mean(vals, d["networks"], d["names"])
        heat(fig, ax[row, 1], nv, "Mean signed ROI deviation", -3, 3, ticks=d["names"])
        ax[row, 1].set_yticks([0, 1], ["Observed", "Generated"])
        edge = d["ze"]["Real"][indices[index]]
        top = np.argsort(np.abs(edge))[-8:][::-1]
        i, j = np.triu_indices(len(d["labels"]), 1)
        names = [f"{d['labels'][i[k]]}–{d['labels'][j[k]]}" for k in top]
        ax[row, 2].barh(
            range(8), edge[top], color=[MDD if v > 0 else HC for v in edge[top]]
        )
        ax[row, 2].set_yticks(range(8), names, fontsize=7)
        ax[row, 2].invert_yaxis()
        ax[row, 2].set(
            xlabel="Observed FC deviation z", title="Largest absolute edge deviations"
        )
        ax[row, 2].axvline(0, color=GREY, lw=0.8)
    save(
        d,
        fig,
        "10_individual_normative_deviation.png",
        f"HC train reference n = {d['normative_n']}; age/sex/education/data-version regression; residual SD in HC train defines z. Same observed reference used for generated values.\nIndividuals chosen at 25th/50th/75th observed deviation burden. Color clipped to ±3; z is a descriptive standardized residual, not a calibrated clinical percentile or diagnosis.",
    )

    order_subject = ranked
    fig, ax = figure("Heterogeneity of MDD deviation patterns", 2, 1, size=(17, 10))
    for row, (key, values) in enumerate([("Observed", zr), ("Generated", zp)]):
        roi_heat(
            d,
            fig,
            ax[row, 0],
            values[order_subject],
            f"{key}: {len(order_subject)} MDD subjects, identical row order",
            -3,
            3,
            [],
            cmap="RdBu_r",
        )
        tick = np.linspace(0, len(order_subject) - 1, 8).astype(int)
        ax[row, 0].set_yticks(
            tick, [f"{indices[order_subject[j]] + 1:03d}" for j in tick]
        )
        ax[row, 0].set_ylabel("Anonymous subject index")
    save(
        d,
        fig,
        "11_mdd_deviation_heatmap.png",
        "Rows sorted once by observed RMS ROI deviation and kept paired across panels; ROIs in network order. Identical ±3 color scale.\nComplete demographic cases only. Reference fitted exclusively to observed HC training subjects; generated data are never renormalized to their own HC distribution.",
    )

    fig, ax = figure(
        "Real–generated agreement in individual deviation", 2, 2, size=(12.5, 9)
    )
    scatter(
        ax[0, 0],
        zr[valid].ravel(),
        zp[valid].ravel(),
        HC,
        "ROI deviations",
        "Observed z",
        "Generated z",
        True,
    )
    scatter(
        ax[0, 1],
        severity[valid],
        np.sqrt(np.nanmean(zp[valid] ** 2, 1)),
        MDD,
        "Overall ROI deviation burden",
        "Observed RMS z",
        "Generated RMS z",
        True,
    )
    er, ep = d["ze"]["Real"][m], d["ze"]["Generated"][m]
    rroi = np.array([pearson(a, b) for a, b in zip(zr, zp)])
    redge = np.array([pearson(a, b) for a, b in zip(er, ep)])
    for v, color, label in [(rroi, HC, "ROI pattern"), (redge, MDD, "FC edge pattern")]:
        x = np.sort(v[np.isfinite(v)])
        ax[1, 0].step(
            x, np.arange(1, len(x) + 1) / len(x), c=color, label=label, where="post"
        )
    ax[1, 0].set(
        xlabel="Within-subject spatial Pearson r",
        ylabel="Cumulative subject fraction",
        title="Individual pattern agreement",
    )
    ax[1, 0].legend()
    scatter(
        ax[1, 1],
        np.sqrt(np.nanmean(er**2, 1)),
        np.sqrt(np.nanmean(ep**2, 1)),
        GOLD,
        "FC deviation burden",
        "Observed edge RMS z",
        "Generated edge RMS z",
        True,
    )
    save(
        d,
        fig,
        "12_predicted_vs_real_deviation.png",
        "Shared observed HC reference; only finite paired values are plotted. Scatter r across repeated ROIs is descriptive; no pseudo-replicated inferential p-values.\nCDFs and burden plots summarize individual subjects. Generated trajectories are open-loop H1–86 extrapolations from the shared HC model.",
    )


def explanation_figures(d):
    imp, labels = d["importance"], d["labels"]
    mean, interval = ci(imp)
    rank = np.argsort(mean)[::-1]
    fig, ax = figure(
        "ROI occlusion importance",
        1,
        3,
        size=(18, 14),
        gridspec_kw={"width_ratios": [1, 1, 1.3]},
    )
    for col, selected in enumerate(np.array_split(rank, 2)):
        a = ax[0, col]
        y = np.arange(len(selected))
        a.barh(
            y,
            mean[selected],
            color=[MDD if v >= 0 else HC for v in mean[selected]],
            height=0.7,
        )
        a.errorbar(
            mean[selected],
            y,
            xerr=np.maximum(
                np.stack(
                    [
                        mean[selected] - interval[0, selected],
                        interval[1, selected] - mean[selected],
                    ]
                ),
                0,
            ),
            fmt="none",
            ecolor="#586574",
            lw=0.7,
            capsize=1,
        )
        a.set_yticks(
            y,
            [
                f"{j + col * len(rank) // 2 + 1:03d}  {labels[i]}"
                for j, i in enumerate(selected)
            ],
            fontsize=8,
        )
        a.invert_yaxis()
        a.axvline(0, color=GREY, lw=0.7)
        a.set(
            xlabel="Increase in H1–8 MAE",
            title="Ranks 1–58" if col == 0 else "Ranks 59–116",
        )
    lim = np.max(np.abs(imp))
    heat(fig, ax[0, 2], imp[:, d["order"]], "MDD subject × ROI importance", -lim, lim)
    ax[0, 2].set(xlabel="ROI (network order)", ylabel="MDD subject")
    ax[0, 2].set_xticks([])
    save(
        d,
        fig,
        "13_roi_importance.png",
        "MDD fine-tuned checkpoint; every test subject, all 116 ROIs; one fixed 64-TR history each. Occlusion replaces one ROI history with its temporal mean.\nOriginal calibration profile and clinical conditions held fixed; history-derived graph recomputed. Signed ΔMAE includes all output ROIs; negative values indicate improved prediction. Bars: 95% subject CI.",
    )

    net = network_mean(imp, d["networks"], d["names"])
    fig, ax = figure("Network contributions to prediction", 1, 2, size=(12, 6))
    rng = np.random.default_rng(2024)
    for j, name in enumerate(d["names"]):
        x = net[:, j]
        ax[0, 0].scatter(
            j + rng.uniform(-0.15, 0.15, len(x)), x, s=9, color=HC, alpha=0.3
        )
        mu, interval = ci(x)
        ax[0, 0].errorbar(
            j,
            mu,
            yerr=np.maximum([[mu - interval[0]], [interval[1] - mu]], 0),
            fmt="o",
            c=MDD,
            capsize=4,
        )
    ax[0, 0].set(
        xticks=range(len(d["names"])),
        xticklabels=d["names"],
        ylabel="Mean ROI occlusion ΔMAE",
        title="Subject distribution and mean 95% CI",
    )
    ax[0, 0].axhline(0, c=GREY, lw=0.8)
    lim = np.max(np.abs(net))
    heat(fig, ax[0, 1], net, "Subject × network", -lim, lim, ticks=d["names"])
    ax[0, 1].set_ylabel("MDD test subject")
    save(
        d,
        fig,
        "14_network_importance.png",
        "AAL116.xlsx network mapping; mean importance per ROI in each network (not a sum biased by network size). Dots represent subjects.\nMDD fine-tuned checkpoint; importance describes model reliance under mean occlusion, not causal brain-region function.",
    )

    pathology = np.abs(d["group_results"]["Real"]["roi"][0][0])
    deviation = np.nanmean(np.abs(d["z"]["Real"][d["groups"] == "MDD"]), 0)
    fig, ax = figure("Model reliance versus observed abnormality", 2, 2, size=(12, 9))
    scatter(
        ax[0, 0],
        pathology,
        mean,
        HC,
        "Group effect and importance",
        "|Observed log-SD Hedges g|",
        "Mean occlusion ΔMAE",
    )
    scatter(
        ax[0, 1],
        deviation,
        mean,
        MDD,
        "Normative deviation and importance",
        "Mean observed MDD |z|",
        "Mean occlusion ΔMAE",
    )
    mdd_indices = np.flatnonzero(d["groups"] == "MDD")
    spatial = np.array(
        [pearson(v, np.abs(d["z"]["Real"][i])) for v, i in zip(imp, mdd_indices)]
    )
    x = np.sort(spatial[np.isfinite(spatial)])
    ax[1, 0].step(x, np.arange(1, len(x) + 1) / len(x), color=REAL, where="post")
    ax[1, 0].set(
        xlabel="Within-subject spatial r",
        ylabel="Cumulative subject fraction",
        title="Individual importance–deviation relationship",
    )
    vals = np.stack([mean, pathology, deviation])
    # Each row has different physical units: standardized only for visual spatial comparison.
    vals = (vals - vals.mean(1)[:, None]) / vals.std(1)[:, None]
    roi_heat(
        d,
        fig,
        ax[1, 1],
        vals,
        "Spatial patterns (row-standardized)",
        -3,
        3,
        ["Importance", "|Group effect|", "Mean |z|"],
        label_stride=4,
    )
    save(
        d,
        fig,
        "15_importance_vs_pathology.png",
        "Pathology and deviation features come exclusively from observed BOLD. Importance comes from the MDD fine-tuned model.\nSpatial correlations are descriptive and have no independence-based p-values; shared spatial structure does not establish disease specificity.",
    )


def clinical_figures(d):
    fig, ax = figure(
        "Clinical symptoms and observed brain features", 2, 1, size=(16, 8)
    )
    r = np.array([v["ROI"]["r"] for v in d["clinical"]])
    q = np.array([v["ROI"]["q"] for v in d["clinical"]])
    roi_heat(
        d,
        fig,
        ax[0, 0],
        r,
        "Partial Spearman: ROI log temporal SD",
        -1,
        1,
        ["HAMD", "HAMA"],
        significance=q,
    )
    nr = np.array([v["Network FC"]["r"] for v in d["clinical"]])
    nq = np.array([v["Network FC"]["q"] for v in d["clinical"]])
    pairs = [f"{a}–{b}" for a, b in d["network_pairs"]]
    heat(
        fig,
        ax[1, 0],
        nr,
        "Partial Spearman: network-pair Fisher-z FC",
        -1,
        1,
        ticks=pairs,
    )
    ax[1, 0].set_yticks([0, 1], ["HAMD", "HAMA"])
    y, x = np.where(nq < 0.05)
    ax[1, 0].scatter(x, y, c="black", s=10)
    ns = ", ".join(
        f"{name}: complete-case n={d['clinical'][i]['ROI']['n'][0]}"
        for i, name in enumerate(("HAMD", "HAMA"))
    )
    save(
        d,
        fig,
        "16_clinical_roi_associations.png",
        ns
        + "; dots: BH q < 0.05 within each scale/feature family.\nObserved BOLD only; age/sex/education adjusted, data version adjusted where variable (HAMA is single-version). Site/FD unavailable; missing scores not imputed.",
    )

    fig, ax = figure(
        "Clinical symptoms and normative deviation burden", 2, 3, size=(14, 8)
    )
    descriptors = ["ROI RMS z", "Fraction of ROIs with |z| > 2", "FC edge RMS z"]
    for row, scale in enumerate(("HAMD", "HAMA")):
        cov = d["clinical_covariates"][row]
        y = d["clinical_scales"][:, row]
        for col in range(3):
            x = d["clinical_deviation"][:, col]
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(cov).all(1)
            # Rank-residual axes correspond exactly to the partial Spearman estimate.
            design = np.column_stack([np.ones(ok.sum()), cov[ok]])
            xx, yy = stats.rankdata(x[ok]), stats.rankdata(y[ok])
            xx -= design @ np.linalg.lstsq(design, xx, rcond=None)[0]
            yy -= design @ np.linalg.lstsq(design, yy, rcond=None)[0]
            scatter(
                ax[row, col],
                xx,
                yy,
                MDD,
                f"{scale} × {descriptors[col]}",
                f"{descriptors[col]} (adjusted rank)",
                f"{scale} (adjusted rank)",
            )
            q = d["clinical"][row]["Deviation"]["q"][col]
            ax[row, col].text(
                0.97,
                0.03,
                f"BH q = {q:.3f}",
                transform=ax[row, col].transAxes,
                ha="right",
                fontsize=9,
            )
    save(
        d,
        fig,
        "17_clinical_deviation_associations.png",
        "Complete-case MDD test subjects; rank residuals after age/sex/education adjustment and data version where variable (HAMA is single-version); line: residual fit.\nBH correction across three deviation summaries separately per scale. Observed BOLD only; cross-sectional association, not prediction of symptom change.",
    )


def sensitivity_figures(d):
    fig, ax = figure(
        "Computational sensitivity to clinical conditions", 2, 2, size=(13, 8)
    )
    for row, field in enumerate(("HAMD", "HAMA")):
        s = d["sensitivity"][field]
        delta, grid = s["delta"], s["grid"]
        band(
            ax[row, 0],
            grid,
            np.sqrt(np.mean(delta**2, (2, 3))),
            MDD,
            "All-ROI output RMS change",
        )
        ax[row, 0].set(
            title=f"{field} sweep  |  n = {s['n']}",
            xlabel=f"{field} condition value",
            ylabel="Prediction change (BOLD units)",
        )
        strongest = np.argsort(s["sensitivity"].mean(0))[-3:][::-1]
        for roi, color in zip(strongest, [REAL, HC, MDD]):
            band(ax[row, 1], grid, delta[..., roi].mean(2), color, d["labels"][roi])
        ax[row, 1].set(
            title="Top three condition-sensitive ROIs",
            xlabel=f"{field} condition value",
            ylabel="Signed H1–8 mean change",
        )
        ax[row, 1].legend()
    save(
        d,
        fig,
        "18_clinical_condition_sensitivity.png",
        "MDD fine-tuned checkpoint; fixed 64-TR histories, profile and other condition values; vary one observed symptom field at a time.\nGrid: training-score 5th–95th percentiles; output change relative to the median-grid condition. Shading: 95% subject CI. Computational sensitivity only, no symptom-change or intervention inference.",
    )

    values = np.array(
        [d["sensitivity"][f]["sensitivity"].mean(0) for f in ("HAMD", "HAMA")]
    )
    fig, ax = figure("Condition-sensitive ROIs and networks", 2, 1, size=(16, 7))
    vmax = values.max()
    roi_heat(
        d,
        fig,
        ax[0, 0],
        values,
        "Output RMS sensitivity (H1–8, full condition grid)",
        0,
        vmax,
        ["HAMD", "HAMA"],
        "YlGnBu",
    )
    net = network_mean(values, d["networks"], d["names"])
    heat(
        fig,
        ax[1, 0],
        net,
        "Network mean of ROI sensitivities",
        0,
        vmax,
        "YlGnBu",
        ticks=d["names"],
    )
    ax[1, 0].set_yticks([0, 1], ["HAMD", "HAMA"])
    save(
        d,
        fig,
        "19_condition_sensitive_networks.png",
        "Sensitivity = root-mean-square output change relative to median condition over training-range grid and H1–8; then averaged over eligible subjects.\nHAMD/HAMA use their own observed-score subsets and empirical ranges; magnitudes are grid-dependent. Same color scale; model computation only.",
    )


def uncertainty_figures(d):
    fig, ax = figure("Native Graph-ODE predictive uncertainty", 2, 3, size=(14, 8))
    for row, group in enumerate(("HC", "MDD")):
        subject = representative(d, group)
        windows = np.flatnonzero(d["owner"] == subject)
        j = windows[len(windows) // 2]
        model = "Common HC model" if group == "HC" else "MDD fine-tuned"
        pp = d["predictions"][model]
        ss = d["std"] if group == "HC" else d["fine_std"]
        rois = np.argsort(d["roi_error"][subject])[
            [len(d["labels"]) // 3, 2 * len(d["labels"]) // 3]
        ]
        for col, roi in enumerate(rois):
            a = ax[row, col]
            t = np.arange(1, 9)
            a.fill_between(
                t,
                pp[j, :, roi] - 1.96 * ss[j, :, roi],
                pp[j, :, roi] + 1.96 * ss[j, :, roi],
                color=GROUP_COLORS[group],
                alpha=0.25,
                label="Nominal 95% interval",
            )
            a.plot(
                t, pp[j, :, roi], c=GROUP_COLORS[group], lw=1.8, label="Predicted mean"
            )
            a.plot(t, d["target"][j, :, roi], "o-", c=REAL, ms=3, label="Observed")
            a.set(
                title=f"{group} subject {subject + 1:03d}  |  {d['labels'][roi]}",
                xlabel="Forecast horizon (TR)",
                ylabel="Prefix-standardized BOLD",
            )
            if col == 0:
                a.legend(fontsize=7)
        for g in ("HC", "MDD"):
            ids = np.flatnonzero(d["groups"] == g)
            sigma = np.array([np.mean(d["std"][d["owner"] == i], (0, 2)) for i in ids])
            band(ax[row, 2], np.arange(1, 9), sigma, GROUP_COLORS[g], g)
        ax[row, 2].set(
            title="Shared HC model: mean predictive SD",
            xlabel="Forecast horizon (TR)",
            ylabel="Predicted SD",
        )
        ax[row, 2].legend()
        if row == 1:
            ax[row, 2].clear()
            ax[row, 2].text(
                -0.10,
                1.06,
                "F",
                transform=ax[row, 2].transAxes,
                fontsize=12,
                fontweight="bold",
                va="bottom",
            )
            fine_ids = np.flatnonzero(d["groups"] == "MDD")
            for name, sigma, color in [
                ("Common HC model", d["std"], HC),
                ("MDD fine-tuned", d["fine_std"], MDD),
            ]:
                values = np.array(
                    [np.mean(sigma[d["owner"] == i], (0, 2)) for i in fine_ids]
                )
                band(ax[row, 2], np.arange(1, 9), values, color, name)
            ax[row, 2].set(
                title="MDD checkpoint comparison",
                xlabel="Forecast horizon (TR)",
                ylabel="Predicted SD",
            )
            ax[row, 2].legend()
    save(
        d,
        fig,
        "20_prediction_uncertainty.png",
        "Native trained variance head only; data-scale standard deviations returned by the existing RevIN inverse. Gaussian intervals: mean ±1.96 SD.\nRepresentative subjects/ROIs chosen by prediction-error quantiles; one 8-TR forecast each. These intervals do not add parameter uncertainty. Shading in SD summaries: 95% subject CI.",
    )

    fig, ax = figure("Predictive interval calibration", 2, 3, size=(14, 8))
    nominal = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    for row, (name, sigma) in enumerate(
        [("Common HC model", d["std"]), ("MDD fine-tuned", d["fine_std"])]
    ):
        pred = d["predictions"][name]
        for group in ("HC", "MDD"):
            if row == 1 and group == "HC":
                continue
            ids = np.flatnonzero(d["groups"] == group)
            coverage, err, sd, per_h = [], [], [], []
            for i in ids:
                ix = d["owner"] == i
                z = np.abs(d["target"][ix] - pred[ix]) / sigma[ix]
                coverage.append(
                    [(z <= stats.norm.ppf((1 + p) / 2)).mean() for p in nominal]
                )
                per_h.append((z <= 1.95996398454).mean((0, 2)))
                err.append(np.abs(d["target"][ix] - pred[ix]).mean())
                sd.append(sigma[ix].mean())
            band(ax[row, 0], nominal, np.array(coverage), GROUP_COLORS[group], group)
            band(
                ax[row, 1], np.arange(1, 9), np.array(per_h), GROUP_COLORS[group], group
            )
            ax[row, 2].scatter(
                sd,
                err,
                s=15,
                color=GROUP_COLORS[group],
                alpha=0.55,
                label=f"{group}: r={pearson(sd, err):.2f}",
            )
        ax[row, 0].plot([0.5, 1], [0.5, 1], "--", c=GREY)
        ax[row, 0].set(
            title=name,
            xlabel="Nominal coverage",
            ylabel="Observed subject-mean coverage",
            ylim=(0, 1.02),
        )
        ax[row, 0].legend()
        ax[row, 1].axhline(0.95, ls="--", c=GREY)
        ax[row, 1].set(
            title="Nominal 95% interval",
            xlabel="Forecast horizon (TR)",
            ylabel="Observed coverage",
            ylim=(0, 1.02),
        )
        ax[row, 1].legend()
        ax[row, 2].set(
            title="Error versus predicted uncertainty",
            xlabel="Subject mean predicted SD",
            ylabel="Subject MAE",
        )
        ax[row, 2].legend()
    save(
        d,
        fig,
        "21_uncertainty_calibration.png",
        "Each subject contributes equally after averaging its windows/ROIs; 95% subject bootstrap CI. Frozen test data are used only to evaluate calibration.\nGaussian interpretation of the existing variance head; no test-set recalibration or additional uncertainty model. HC/MDD comparison uses one common checkpoint.",
    )
