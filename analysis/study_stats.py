"""Subject-level inference. NaN denotes an undefined statistic, never zero evidence."""

import numpy as np
from scipy import stats


def pearson(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def bootstrap_ci(x, seed=2024, repeats=1000, strata=None):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    labels = np.asarray(strata)[ok] if strata is not None else None
    x = x[ok]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    groups = (
        [np.arange(len(x))]
        if labels is None
        else [np.flatnonzero(labels == g) for g in np.unique(labels)]
    )
    means = [
        np.concatenate([rng.choice(x[g], len(g), replace=True) for g in groups]).mean()
        for _ in range(repeats)
    ]
    return tuple(np.quantile(means, [0.025, 0.975]))


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    order = idx[np.argsort(p[idx])]
    if len(order):
        q[order] = np.minimum(
            1,
            np.minimum.accumulate(
                (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
            )[::-1],
        )
    return q


def subject_aggregate(frame, keys, columns):
    """One equal-weight row per subject within the requested comparison keys."""
    return (
        frame.groupby(["subject_id"] + list(keys), dropna=False)[columns]
        .mean()
        .reset_index()
    )


def hedges_g(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if min(len(a), len(b)) < 2:
        return np.nan
    df = len(a) + len(b) - 2
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / df)
    return (
        (a.mean() - b.mean()) / pooled * (1 - 3 / (4 * df - 1))
        if pooled > 1e-12
        else np.nan
    )


def comparison(a, b, seed=2024, repeats=1000, paired=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    else:
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    row = dict(
        unit="subject",
        n_a=len(a),
        n_b=len(b),
        covariates="",
        method="paired t / mean difference" if paired else "Welch t / Hedges g",
        valid=min(len(a), len(b)) >= 3,
    )
    row.update(effect=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    if not row["valid"]:
        return row
    rng = np.random.default_rng(seed)
    if paired:
        d = a - b
        row["effect"] = d.mean()
        row["ci_low"], row["ci_high"] = bootstrap_ci(d, seed, repeats)
        row["p"] = (
            float(stats.ttest_rel(a, b).pvalue)
            if d.std() > 1e-12
            else (1.0 if d.mean() == 0 else np.nan)
        )
    else:
        row["effect"] = hedges_g(a, b)
        vals = [
            hedges_g(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
            for _ in range(repeats)
        ]
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            row["ci_low"], row["ci_high"] = np.quantile(vals, [0.025, 0.975])
        if max(a.std(), b.std()) > 1e-12:
            row["p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    row["valid"] = bool(np.isfinite(row["effect"]))
    return row


def association(x, y, seed=2024, repeats=1000, covariates=None):
    """Spearman or partial Spearman with complete-case OLS residualization."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.empty((len(x), 0)) if covariates is None else np.asarray(covariates, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c).all(axis=1)
    x, y, c = x[ok], y[ok], c[ok]
    row = dict(
        unit="subject",
        n_a=len(x),
        n_b=0,
        effect=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        p=np.nan,
        valid=False,
        method="partial Spearman / approximate Fisher-z CI"
        if c.shape[1]
        else "Spearman / approximate Fisher-z CI",
    )

    def estimate(u, v, z):
        u, v = stats.rankdata(u), stats.rankdata(v)
        design = np.column_stack([np.ones(len(u)), z])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            return np.nan
        u = u - design @ np.linalg.lstsq(design, u, rcond=None)[0]
        v = v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
        return pearson(u, v)

    if len(x) < max(6, c.shape[1] + 4):
        return row
    r = estimate(x, y, c)
    if not np.isfinite(r):
        return row
    row.update(effect=r, valid=True)
    df = len(x) - c.shape[1] - 2
    row["p"] = float(2 * stats.t.sf(abs(r) * np.sqrt(df / max(1 - r * r, 1e-12)), df))
    # Correlation CI is explicitly asymptotic (not a subject bootstrap CI).
    # Group/performance CIs use the configured subject bootstrap separately.
    margin = 1.95996398454 / np.sqrt(len(x) - c.shape[1] - 3)
    row["ci_low"], row["ci_high"] = np.tanh(
        np.arctanh(np.clip(r, -0.999999, 0.999999)) + np.array([-margin, margin])
    )
    return row


def adjusted_group(y, group, covariates):
    """Complete-case OLS group coefficient, HC3 robust standard error."""
    y, group, c = (
        np.asarray(y, float),
        np.asarray(group, float),
        np.asarray(covariates, float),
    )
    ok = np.isfinite(y) & np.isfinite(group) & np.isfinite(c).all(axis=1)
    y, group, c = y[ok], group[ok], c[ok]
    a = np.column_stack([np.ones(len(y)), group, c])
    row = dict(
        unit="subject",
        n_a=int((group == 1).sum()),
        n_b=int((group == 0).sum()),
        effect=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        p=np.nan,
        valid=False,
        method="OLS group coefficient / HC3",
    )
    if (
        min(row["n_a"], row["n_b"]) < 3
        or len(y) <= a.shape[1] + 2
        or np.linalg.matrix_rank(a) < a.shape[1]
    ):
        return row
    inv = np.linalg.pinv(a.T @ a)
    beta = inv @ a.T @ y
    residual = y - a @ beta
    leverage = np.einsum("ij,jk,ik->i", a, inv, a)
    meat = a.T @ (((residual / np.maximum(1 - leverage, 1e-8)) ** 2)[:, None] * a)
    se = np.sqrt(max((inv @ meat @ inv)[1, 1], 0))
    if se < 1e-12:
        return row
    df = len(y) - a.shape[1]
    margin = stats.t.ppf(0.975, df) * se
    row.update(
        effect=float(beta[1]),
        ci_low=float(beta[1] - margin),
        ci_high=float(beta[1] + margin),
        p=float(2 * stats.t.sf(abs(beta[1] / se), df)),
        valid=True,
    )
    return row


def group_matrix(a, b, seed=2024, repeats=1000):
    """Vectorized subject bootstrap across fixed features, retaining dependence."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def effect(u, v):
        na, nb = np.isfinite(u).sum(0), np.isfinite(v).sum(0)
        var = (
            (na - 1) * np.nanvar(u, axis=0, ddof=1)
            + (nb - 1) * np.nanvar(v, axis=0, ddof=1)
        ) / np.maximum(na + nb - 2, 1)
        g = (
            (np.nanmean(u, 0) - np.nanmean(v, 0))
            / np.sqrt(np.maximum(var, 1e-24))
            * (1 - 3 / np.maximum(4 * (na + nb - 2) - 1, 1))
        )
        return np.where((na >= 3) & (nb >= 3) & (var > 1e-12), g, np.nan)

    g = effect(a, b)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            effect(
                a[rng.integers(len(a), size=len(a))],
                b[rng.integers(len(b), size=len(b))],
            )
            for _ in range(repeats)
        ]
    )
    ci = np.full((2, a.shape[1]), np.nan)
    valid_columns = np.isfinite(boot).any(axis=0)
    if valid_columns.any():
        ci[:, valid_columns] = np.nanquantile(
            boot[:, valid_columns], [0.025, 0.975], axis=0
        )
    p = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit").pvalue
    return [
        dict(
            unit="subject",
            n_a=int(np.isfinite(a[:, j]).sum()),
            n_b=int(np.isfinite(b[:, j]).sum()),
            effect=float(g[j]),
            ci_low=float(ci[0, j]),
            ci_high=float(ci[1, j]),
            p=float(p[j]) if np.isfinite(g[j]) else np.nan,
            valid=bool(np.isfinite(g[j])),
            method="Welch t / Hedges g / subject bootstrap",
            covariates="",
        )
        for j in range(a.shape[1])
    ]
