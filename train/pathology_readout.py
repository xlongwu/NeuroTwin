"""Training-only site-grouped model selection for frozen HC fingerprints."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from train.pathology_fingerprint import design, fit_reference, reference_features


def site_weights(sites: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """Give each site, or each (site, diagnosis) cell, the same total weight."""
    weights = np.empty(len(sites), dtype=np.float64)
    for site in np.unique(sites):
        rows = sites == site
        for label in (np.unique(labels[rows]) if labels is not None else [None]):
            cell = rows if labels is None else rows & (labels == label)
            weights[cell] = 1.0 / cell.sum()
    return weights / weights.mean()


def _subset(group: dict, rows: np.ndarray) -> dict:
    return {**group, 'ids': np.asarray(group['ids'])[rows].tolist(),
            **{key: group[key][rows] for key in ('values', 'mask', 'sites', 'features')},
            'clinical': {name: (y[rows], observed[rows])
                         for name, (y, observed) in group.get('clinical', {}).items()}}


def site_folds(sites: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    count = len(np.unique(sites))
    if count < 2:
        return []
    return list(GroupKFold(n_splits=min(3, count)).split(np.zeros(len(sites)), groups=sites))


def adjust_clinical(a: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """Fit the same training-only nuisance projection to features and symptom labels."""
    penalty = np.eye(a.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(a.T @ (weights[:, None] * a) + penalty,
                                   a.T @ (weights[:, None] * np.column_stack((y, x))))
    return coefficients[:, 0], coefficients[:, 1:]


def _select_classifier(hc: dict, mdd: dict, shared: list[str], feature_mode: str):
    groups = (hc, mdd)
    labels = np.concatenate([np.full(len(g['ids']), g['label']) for g in groups])
    sites = np.concatenate([g['sites'] for g in groups])
    eligible = np.flatnonzero(np.isin(sites, shared))
    folds = site_folds(sites[eligible])
    scores = {c: [] for c in (0.01, 0.1, 1.0, 10.0)}
    for train, valid in folds:
        train, valid = eligible[train], eligible[valid]
        held_out = np.unique(sites[valid])
        # Refit every reference statistic without the held-out sites, including HC scales.
        reference = fit_reference(_subset(hc, ~np.isin(hc['sites'], held_out)))
        x = np.concatenate([reference_features(reference, g, feature_mode) for g in groups])
        weights = site_weights(sites[train], labels[train])
        for c, values in scores.items():
            estimator = LogisticRegression(C=c, max_iter=1000, random_state=2024)
            estimator.fit(x[train], labels[train], sample_weight=weights)
            pred = estimator.decision_function(x[valid])
            for site in held_out:
                rows = sites[valid] == site
                values.append(roc_auc_score(labels[valid][rows], pred[rows]))
    means = {str(c): float(np.mean(values)) for c, values in scores.items() if values}
    chosen = max(scores, key=lambda c: np.mean(scores[c])) if folds else 1.0
    return chosen, dict(folds=len(folds), mean_site_auc=means,
                        selected_C=chosen, selection_split='train')


def _clinical_head(a: np.ndarray, x: np.ndarray, y: np.ndarray, sites: np.ndarray):
    folds = site_folds(sites)
    errors = {alpha: [] for alpha in (0.1, 1.0, 10.0, 100.0)}
    null_errors = []
    for train, valid in folds:
        weights = site_weights(sites[train])
        beta, feature_beta = adjust_clinical(a[train], x[train], y[train], weights)
        xt, xv = x[train] - a[train] @ feature_beta, x[valid] - a[valid] @ feature_beta
        yt, yv = y[train] - a[train] @ beta, y[valid] - a[valid] @ beta
        null_errors.extend(float(np.mean(yv[sites[valid] == site] ** 2))
                           for site in np.unique(sites[valid]))
        for alpha, values in errors.items():
            estimator = Ridge(alpha=alpha, solver='lsqr')
            estimator.fit(xt, yt, sample_weight=weights)
            squared = (estimator.predict(xv) - yv) ** 2
            values.extend(float(np.mean(squared[sites[valid] == site]))
                          for site in np.unique(sites[valid]))
    mse = {str(alpha): float(np.mean(values)) for alpha, values in errors.items() if values}
    chosen = min(errors, key=lambda alpha: np.mean(errors[alpha])) if folds else 1.0
    weights = site_weights(sites)
    beta, feature_beta = adjust_clinical(a, x, y, weights)
    estimator = Ridge(alpha=chosen, solver='lsqr')
    estimator.fit(x - a @ feature_beta, y - a @ beta, sample_weight=weights)
    selection = dict(folds=len(folds), mean_site_mse=mse, selected_alpha=chosen,
                     null_mean_site_mse=float(np.mean(null_errors)) if null_errors else None,
                     n_labeled=len(y), n_sites=len(np.unique(sites)), selection_split='train')
    return beta, feature_beta, estimator, selection


def fit_optimized_readout(hc: dict, mdd: dict, feature_mode: str) -> dict:
    """Select regularization by unseen training sites, then refit on all eligible train rows."""
    shared = sorted(set(hc['sites']) & set(mdd['sites']))
    if not shared:
        raise ValueError('训练集没有同时包含 HC 和 MDD 的站点')
    c, classifier_selection = _select_classifier(hc, mdd, shared, feature_mode)
    reference = fit_reference(hc)
    xs = [reference_features(reference, g, feature_mode) for g in (hc, mdd)]
    x = np.concatenate(xs)
    labels = np.concatenate([np.full(len(g['ids']), g['label']) for g in (hc, mdd)])
    sites = np.concatenate([g['sites'] for g in (hc, mdd)])
    keep = np.isin(sites, shared)
    estimator = LogisticRegression(C=c, max_iter=1000, random_state=2024)
    estimator.fit(x[keep], labels[keep], sample_weight=site_weights(sites[keep], labels[keep]))
    model = dict(**reference, coef=estimator.coef_[0], intercept=estimator.intercept_[0],
                 shared_sites=shared, feature_mode=feature_mode, clinical_targets=[],
                 selection=dict(version=2, method='train_site_group_cv',
                                classifier=classifier_selection, clinical={}))
    a = design(mdd['values'], mdd['mask'], mdd['idx'], mdd['sites'],
               model['demo_stats'], model['site_vocab'])
    for name, (values, observed) in mdd.get('clinical', {}).items():
        eligible = observed & np.isfinite(values) & np.isin(mdd['sites'], shared)
        if eligible.sum() < 10:
            print(f'{name}: 训练集有效病例不足 10，未拟合症状头')
            continue
        beta, feature_beta, head, selected = _clinical_head(
            a[eligible], xs[1][eligible], values[eligible].astype(float), mdd['sites'][eligible])
        model.update({f'{name}_beta': beta, f'{name}_feature_beta': feature_beta,
                      f'{name}_coef': head.coef_, f'{name}_intercept': head.intercept_})
        model['clinical_targets'].append(name)
        model['selection']['clinical'][name] = selected
    return model
