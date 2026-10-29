"""Train a site-adjusted supervised fingerprint from HC prediction residuals and FC.

Requires the existing rolling_one_step per-subject caches. The reference and
classifier are fitted only on training subjects; val/test are scored separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from analysis.normative_deviation import (HC_MERGED_FILE, MDD_MERGED_FILE,
                                         load_demo_matrix, split_subjects)
from analysis.rolling_one_step import GROUP_SUBDIR, manifest_path
from analysis.twin_reference import _demo_design_rows, fit_demo_stats
from utils.condition_schema import field_names
from utils.dataloader import load_condition_table
from utils.site_subset import parse_site


def scan_features(path: Path):
    with np.load(path, allow_pickle=False) as data:
        true = np.asarray(data['true'], dtype=np.float64)
        error = np.asarray(data['pred'], dtype=np.float64) - true
        cutoff = np.asarray(data['cutoff'])
    if error.ndim != 2 or error.shape[0] < 2 or cutoff.shape != (len(error),):
        raise ValueError(f'滚动预测缓存形状不正确: {path}')
    consecutive = np.diff(cutoff) == 1
    if not consecutive.all():
        raise ValueError(f'需要连续锚点计算误差变化；请使用 --anchor_stride 1: {path}')
    residual = np.concatenate((error.mean(0), error.std(0), np.abs(error).mean(0),
                               np.abs(np.diff(error, axis=0)).mean(0)))
    centered = true - true.mean(axis=0, keepdims=True)
    standardized = centered / (centered.std(axis=0, keepdims=True) + 1e-6)
    fc = standardized.T @ standardized / len(true)
    return residual, fc[np.triu_indices(fc.shape[0], 1)]


def load_features(root: Path, checkpoint: Path, rolling: Path, split: str):
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    groups = []
    seen = {}
    all_splits = {g: split_subjects(root, g) for g in ('hc', 'mdd')}
    hc_all = set().union(*(set(v) for v in all_splits['hc'].values()))
    mdd_all = set().union(*(set(v) for v in all_splits['mdd'].values()))
    if hc_all & mdd_all:
        raise ValueError('HC 与 MDD 存在重复的文件 ID，无法保证划分独立')
    for group, label, table in [('hc', 0, HC_MERGED_FILE), ('mdd', 1, MDD_MERGED_FILE)]:
        splits = all_splits[group]
        if any(set(splits[a]) & set(splits[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
            raise ValueError(f'{group} 的被试划分重叠')
        ids = splits[split]
        if len(ids) != len(set(ids)):
            raise ValueError(f'{group} {split} 有重复被试')
        seen[group] = set(ids)
        manifest = json.loads(manifest_path(rolling / GROUP_SUBDIR[group]).read_text(encoding='utf-8'))
        if manifest['checkpoint_sha256'] != digest:
            raise ValueError(f'{group} 滚动缓存与指定的健康模型权重不一致')
        protocol = {key: manifest[key] for key in
                    ('context_length', 'calibration_length', 'bold_source', 'graph_source', 'anchor_rule')}
        features_by_subject = [scan_features(
            rolling / GROUP_SUBDIR[group] / f'{s}.npz') for s in ids]
        residuals = np.stack([item[0] for item in features_by_subject])
        fc = np.stack([item[1] for item in features_by_subject])
        values, mask, idx = load_demo_matrix(root, table, ids)
        clinical = {}
        if group == 'mdd':
            table_data = load_condition_table(str(root), table, 'mdd8')
            clinical_values, clinical_mask = table_data.align(ids)
            names = list(field_names(table_data.specs))
            for name in ('HAMD', 'HAMA'):
                column = names.index(name)
                clinical[name] = (clinical_values[:, column], clinical_mask[:, column].astype(bool))
        features = np.concatenate((residuals, fc), axis=1)
        if not np.isfinite(features).all():
            raise ValueError(f'{group} {split} 特征包含非有限值')
        groups.append(dict(ids=ids, label=label, values=values, mask=mask, idx=idx,
                           sites=np.array([parse_site(s) for s in ids]),
                           features=features, residual_width=residuals.shape[1], protocol=protocol,
                           clinical=clinical))
    if seen['hc'] & seen['mdd']:
        raise ValueError('HC 与 MDD 存在重复的文件 ID')
    if groups[0]['protocol'] != groups[1]['protocol']:
        raise ValueError('HC 和 MDD 的滚动缓存协议不同')
    return groups


def design(values, mask, idx, sites, demo_stats, site_vocab):
    demo, _ = _demo_design_rows(values, mask, idx, demo_stats)
    site = (sites[:, None] == np.asarray(site_vocab)[None, :]).astype(np.float64)
    return np.concatenate((demo, site), axis=1)


def select_block(x, residual_width, mode):
    if mode == 'residual':
        return x[:, :residual_width]
    if mode == 'fc':
        return x[:, residual_width:]
    if mode == 'combined':
        return x
    raise ValueError(f'未知特征模式: {mode}')


def fit_reference(hc):
    """Fit demographic/site adjustment and feature scales using HC training rows only."""
    demo_stats = fit_demo_stats(hc['values'], hc['mask'], hc['idx'])
    site_vocab = sorted(set(hc['sites']))
    a = design(hc['values'], hc['mask'], hc['idx'], hc['sites'], demo_stats, site_vocab)
    penalty = np.eye(a.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    w = np.linalg.solve(a.T @ a + penalty, a.T @ hc['features'])
    hc_resid = hc['features'] - a @ w
    scale = np.sqrt(np.mean(hc_resid ** 2, axis=0) + 1e-6)
    n_res = hc['residual_width']
    n_fc = hc['features'].shape[1] - n_res
    block_scale = np.concatenate((np.full(n_res, np.sqrt(n_res)),
                                  np.full(n_fc, np.sqrt(n_fc))))
    return dict(w=w, scale=scale, block_scale=block_scale, demo_stats=demo_stats,
                site_vocab=site_vocab, residual_width=n_res, protocol=hc['protocol'])


def reference_features(model, group, feature_mode):
    a = design(group['values'], group['mask'], group['idx'], group['sites'],
               model['demo_stats'], model['site_vocab'])
    x = (group['features'] - a @ model['w']) / model['scale'] / model['block_scale']
    return select_block(x, model['residual_width'], feature_mode)


def fit_model(hc, mdd, feature_mode='combined', *, optimize=False):
    if optimize:
        from train.pathology_readout import fit_optimized_readout
        return fit_optimized_readout(hc, mdd, feature_mode)
    reference = fit_reference(hc)
    xs, labels, sites = [], [], []
    for g in (hc, mdd):
        xs.append(reference_features(reference, g, feature_mode))
        labels.extend([g['label']] * len(g['ids']))
        sites.extend(g['sites'])
    x = np.concatenate(xs)
    labels, sites = np.asarray(labels), np.asarray(sites)
    shared = sorted(set(hc['sites']) & set(mdd['sites']))
    keep = np.isin(sites, shared)
    if not shared:
        raise ValueError('训练集没有同时包含 HC 和 MDD 的站点')
    weights = np.zeros(len(labels))
    for site in shared:
        for group in (0, 1):
            selected = (sites == site) & (labels == group)
            weights[selected] = 1.0 / selected.sum()
    weights[keep] /= weights[keep].mean()
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=2024)
    classifier.fit(x[keep], labels[keep], sample_weight=weights[keep])
    model = dict(**reference, coef=classifier.coef_[0], intercept=classifier.intercept_[0],
                 shared_sites=shared, clinical_targets=[],
                 feature_mode=feature_mode)
    mdd_design = design(mdd['values'], mdd['mask'], mdd['idx'], mdd['sites'],
                        reference['demo_stats'], reference['site_vocab'])
    mdd_x = xs[1]
    for name, (values, observed) in mdd.get('clinical', {}).items():
        eligible = observed & np.isfinite(values) & np.isin(mdd['sites'], shared)
        if eligible.sum() < 10:
            print(f'{name}: 训练集有效病例不足 10，未拟合症状头')
            continue
        a, y, s = mdd_design[eligible], values[eligible].astype(np.float64), mdd['sites'][eligible]
        sw = np.array([1.0 / np.sum(s == site) for site in s])
        sw /= sw.mean()
        penalty = np.eye(a.shape[1]) * 10.0
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(a.T @ (sw[:, None] * a) + penalty, a.T @ (sw * y))
        regressor = Ridge(alpha=1.0, solver='lsqr')
        regressor.fit(mdd_x[eligible], y - a @ beta, sample_weight=sw)
        model[f'{name}_beta'] = beta
        model[f'{name}_coef'] = regressor.coef_
        model[f'{name}_intercept'] = regressor.intercept_
        model['clinical_targets'].append(name)
    return model


def score_groups(model, groups):
    rows = []
    for g in groups:
        a = design(g['values'], g['mask'], g['idx'], g['sites'],
                   model['demo_stats'], model['site_vocab'])
        normalized = (g['features'] - a @ model['w']) / model['scale'] / model['block_scale']
        x = select_block(normalized, model['residual_width'], model['feature_mode'])
        scores = x @ model['coef'] + model['intercept']
        clinical_x = {name: (x - a @ model[f'{name}_feature_beta']
                             if f'{name}_feature_beta' in model else x)
                      for name in model.get('clinical_targets', [])}
        for i, (subject, site, score) in enumerate(zip(g['ids'], g['sites'], scores)):
            row = dict(subject=subject, group='hc' if g['label'] == 0 else 'mdd',
                       label=g['label'], site=site,
                       shared_train_site=site in model['shared_sites'], score=float(score))
            for name in model.get('clinical_targets', []):
                row[f'{name}_predicted_residual'] = ''
                row[f'{name}_observed_residual'] = ''
                if g['label'] == 1:
                    row[f'{name}_predicted_residual'] = float(clinical_x[name][i] @ model[f'{name}_coef'] + model[f'{name}_intercept'])
                    values, observed = g['clinical'][name]
                    if observed[i] and np.isfinite(values[i]):
                        row[f'{name}_observed_residual'] = float(values[i] - a[i] @ model[f'{name}_beta'])
            rows.append(row)
    return rows


def summarize_scores(rows):
    """Report discrimination and symptom error separately for sites seen in both train groups."""
    def summarize(selected):
        result = {'n': len(selected)}
        if len({row['label'] for row in selected}) == 2:
            result['auc'] = float(roc_auc_score([r['label'] for r in selected],
                                               [r['score'] for r in selected]))
        for name in ('HAMD', 'HAMA'):
            observed = [r for r in selected if r.get(f'{name}_observed_residual', '') != '']
            if not observed:
                continue
            y = np.array([r[f'{name}_observed_residual'] for r in observed])
            p = np.array([r[f'{name}_predicted_residual'] for r in observed])
            corr = (float(np.corrcoef(y, p)[0, 1])
                    if len(y) >= 3 and y.std() > 1e-12 and p.std() > 1e-12 else None)
            result[name] = dict(n=len(y), pearson=corr,
                                rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                                zero_residual_rmse=float(np.sqrt(np.mean(y ** 2))))
        return result

    return dict(all=summarize(rows),
                shared_train_sites=summarize([r for r in rows if r['shared_train_site']]),
                per_site={site: summarize([r for r in rows if r['site'] == site])
                          for site in sorted({r['site'] for r in rows})})


def write_scores(path, rows):
    with path.open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_scores(rows)
    for scope in ('all', 'shared_train_sites'):
        metrics = summary[scope]
        if 'auc' in metrics:
            print(f"{scope}: n={metrics['n']} AUC={metrics['auc']:.4f}")
        for name in ('HAMD', 'HAMA'):
            if name in metrics:
                print(f"{scope} {name}: {metrics[name]}")
    metric_path = path.with_name(path.stem.removesuffix('_scores') + '_metrics.json')
    with metric_path.open('x', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rolling-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--features', choices=['combined', 'residual', 'fc'], default='combined',
                        help='拟合时选择特征组；评价时使用模型内保存的设置。')
    parser.add_argument('--fit', action='store_true', help='先用 HC/MDD train 拟合新模型')
    parser.add_argument('--optimize', action='store_true',
                        help='拟合时采用训练集站点分组交叉验证及症状特征混杂校正')
    args = parser.parse_args()
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model_path = args.output_dir / 'fingerprint.npz'
    if args.fit:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        model = fit_model(*load_features(args.data_root, args.checkpoint, args.rolling_dir, 'train'),
                          feature_mode=args.features, optimize=args.optimize)
        meta = {k: model.pop(k) for k in ('demo_stats', 'site_vocab', 'shared_sites',
                                         'residual_width', 'protocol', 'clinical_targets', 'feature_mode')}
        meta['checkpoint_sha256'] = digest
        if 'selection' in model:
            meta['selection'] = model.pop('selection')
        meta['splits_sha256'] = {g: hashlib.sha256((args.data_root / name).read_bytes()).hexdigest()
                                 for g, name in [('hc', 'subject_split_pretrain.json'), ('mdd', 'subject_split_finetune.json')]}
        np.savez_compressed(model_path, **model, metadata=json.dumps(meta))
        model.update(meta)
    else:
        with np.load(model_path, allow_pickle=False) as data:
            model = {k: data[k].copy() for k in data.files if k != 'metadata'}
            model.update(json.loads(str(data['metadata'])))
        if model['checkpoint_sha256'] != digest:
            raise ValueError('病理模型与健康 checkpoint 不匹配')
        for group, name in [('hc', 'subject_split_pretrain.json'), ('mdd', 'subject_split_finetune.json')]:
            if hashlib.sha256((args.data_root / name).read_bytes()).hexdigest() != model['splits_sha256'][group]:
                raise ValueError(f'{group} 数据划分与病理模型训练时不同')
    groups = load_features(args.data_root, args.checkpoint, args.rolling_dir, args.split)
    if groups[0]['protocol'] != model['protocol']:
        raise ValueError('评价缓存协议与病理模型训练时不同')
    write_scores(args.output_dir / f'{args.split}_scores.csv', score_groups(model, groups))


if __name__ == '__main__':
    main()
