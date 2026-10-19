"""Metrics on [time, ROI] arrays and explicit continuous forecast segments."""

import numpy as np
from scipy.signal import welch

from analysis.study_stats import pearson
from analysis.twin_reference import COMPONENT_NAMES, describe_history


def point_metrics(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if (
        pred.shape != target.shape
        or not np.isfinite(pred).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Prediction/target must have equal finite shapes")
    d = pred - target
    return dict(
        mae=float(np.abs(d).mean()),
        rmse=float(np.sqrt((d * d).mean())),
        pearson=pearson(pred, target),
        n_values=d.size,
    )


def fc_matrix(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("FC requires finite [time, ROI] with at least 3 time points")
    centered = x - x.mean(axis=0)
    norm = np.sqrt((centered**2).sum(axis=0))
    valid = norm > 1e-12
    fc = centered.T @ centered / np.maximum(np.outer(norm, norm), 1e-24)
    fc[~valid, :] = np.nan
    fc[:, ~valid] = np.nan
    return np.clip(fc, -1, 1)


def feature_vector(history, roi_labels, networks=None):
    """Reuse exactly the reference's fixed 64-TR affine-invariant descriptors."""
    d = describe_history(history[:64])[0]
    features = {
        f"roi/{label}/{name}": float(d[i, j])
        for i, label in enumerate(roi_labels)
        for j, name in enumerate(COMPONENT_NAMES)
    }
    fc = fc_matrix(history[:64])
    for i, j in zip(*np.triu_indices(len(roi_labels), 1)):
        features[f"edge/{roi_labels[i]}/{roi_labels[j]}"] = float(
            np.arctanh(np.clip(fc[i, j], -0.999999, 0.999999))
        )
    if networks:
        for left in sorted(set(networks.values())):
            for right in sorted(set(networks.values())):
                if right < left:
                    continue
                edges = [
                    features[f"edge/{roi_labels[i]}/{roi_labels[j]}"]
                    for i, j in zip(*np.triu_indices(len(roi_labels), 1))
                    if sorted(
                        [networks.get(roi_labels[i]), networks.get(roi_labels[j])],
                        key=str,
                    )
                    == sorted([left, right])
                ]
                if edges:
                    features[f"network/{left}/{right}"] = float(np.nanmean(edges))
    return features


def continuous_metrics(
    pred, target, min_length=64, dfc_window=32, dfc_stride=8, tr=None
):
    if pred.shape != target.shape:
        raise ValueError("Continuous segment shapes differ")
    if len(pred) < min_length:
        return None
    if min_length < 32 or dfc_window < 16 or dfc_stride < 1:
        raise ValueError("Invalid minimum length or dFC settings")
    a, b = fc_matrix(pred), fc_matrix(target)
    ix = np.triu_indices(a.shape[0], 1)
    valid = np.isfinite(a[ix]) & np.isfinite(b[ix])
    row = dict(
        length=len(pred),
        fc_mae=float(np.abs(a[ix][valid] - b[ix][valid]).mean())
        if valid.any()
        else np.nan,
        fc_similarity=pearson(a[ix], b[ix]),
        n_edges=int(valid.sum()),
        mean_bias=float(np.abs(pred.mean(0) - target.mean(0)).mean()),
        variance_bias=float(np.abs(pred.var(0) - target.var(0)).mean()),
    )
    for lag in (1, 2, 4):
        row[f"acf{lag}_mae"] = float(
            np.nanmean(
                [
                    abs(
                        pearson(pred[:-lag, i], pred[lag:, i])
                        - pearson(target[:-lag, i], target[lag:, i])
                    )
                    for i in range(pred.shape[1])
                ]
            )
        )
    freq, pp = welch(pred, fs=1 / (tr or 1), axis=0, nperseg=min(64, len(pred)))
    _, pt = welch(target, fs=1 / (tr or 1), axis=0, nperseg=min(64, len(target)))
    row["psd_log_mae"] = float(np.abs(np.log(pp + 1e-12) - np.log(pt + 1e-12)).mean())
    row["frequency_unit"] = "Hz" if tr else "cycles/TR"
    if len(pred) >= 2 * dfc_window:
        row["dfc_mae"] = float(
            np.nanmean(
                [
                    np.abs(
                        fc_matrix(pred[t : t + dfc_window])[ix]
                        - fc_matrix(target[t : t + dfc_window])[ix]
                    ).mean()
                    for t in range(0, len(pred) - dfc_window + 1, dfc_stride)
                ]
            )
        )
    return row


def gaussian_metrics(mean, target, std):
    if (
        std.shape != mean.shape
        or mean.shape != target.shape
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError(
            "Uncertainty must be finite positive standard deviation, same shape as mean"
        )
    z = (target - mean) / std
    return dict(
        nll=float((0.5 * np.log(2 * np.pi) + np.log(std) + 0.5 * z * z).mean()),
        coverage95=float((np.abs(z) <= 1.95996398454).mean()),
        width95=float((2 * 1.95996398454 * std).mean()),
        error_std_correlation=pearson(np.abs(target - mean), std),
    )
