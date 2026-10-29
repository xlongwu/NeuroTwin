"""Prespecified matched and unmatched comparisons of HC-reference z scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pathology_tests import bh_fdr, site_blocked_paired_permutation, site_blocked_unpaired_permutation
from utils.site_subset import parse_site

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/H1-8分析_20261028/02_病理分析'
N = 10000
SEED = 2024


def load(group, split):
    with np.load(OUT / '评分缓存' / f'z_matrix_{group}_{split}.npz') as d:
        return d['subjects'].astype(str), d['z'].astype(float)


def bootstrap(diff):
    rng = np.random.RandomState(SEED)
    means = np.empty((N, diff.shape[1]))
    for start in range(0, N, 100):
        end = min(start + 100, N)
        idx = rng.randint(0, len(diff), (end - start, len(diff)))
        counts = np.zeros((end-start, len(diff)), dtype=np.float64)
        np.add.at(counts, (np.arange(end-start)[:, None], idx), 1)
        means[start:end] = counts @ diff / len(diff)
    return np.percentile(means, [2.5, 97.5], axis=0)


def dispersion(mdd, hc):
    observed = mdd.var() / hc.var()
    threshold = abs(np.log(observed))
    rng = np.random.RandomState(SEED)
    exceed = 0
    for start in range(0, N, 200):
        swaps = rng.randint(0, 2, (min(200, N-start), len(mdd))).astype(bool)
        pm = np.where(swaps, hc, mdd)
        ph = np.where(swaps, mdd, hc)
        ratios = pm.var(axis=1) / ph.var(axis=1)
        exceed += int((np.abs(np.log(ratios)) >= threshold - 1e-12).sum())
    return float(observed), (exceed + 1) / (N + 1)


def main():
    control = json.loads((OUT / '原始_健康负控_20261028.json').read_text())
    if not control['passed_roi_negative_control']:
        raise RuntimeError('HC 负控未通过，停止病理组间分析')
    ids, vals = {}, {}
    for group in ('hc', 'mdd'):
        split_file = ROOT / 'data' / ('subject_split_pretrain.json' if group == 'hc' else 'subject_split_finetune.json')
        expected = json.loads(split_file.read_text())
        for split in ('train', 'val', 'test'):
            ids[group, split], vals[group, split] = load(group, split)
            if (len(ids[group, split]) != len(expected[split])
                    or set(ids[group, split]) != set(expected[split])
                    or vals[group, split].shape != (len(ids[group, split]), 116)
                    or not np.isfinite(vals[group, split]).all()):
                raise ValueError(f'{group} {split} 评分与划分或维度不一致')
    z = {sid: value for key in ids for sid, value in zip(ids[key], vals[key])}
    pairs = json.loads((ROOT / 'data/splits/site_matched_subset.json').read_text())['pairs']
    if len({p['hc'] for p in pairs}) != len(pairs) or len({p['mdd'] for p in pairs}) != len(pairs):
        raise ValueError('匹配清单不是一对一')
    missing = [p for p in pairs if p['mdd'] not in z or p['hc'] not in z]
    if missing:
        raise ValueError(f'匹配被试缺少 z 评分：{len(missing)} 对')
    mdd = np.stack([z[p['mdd']] for p in pairs])
    hc = np.stack([z[p['hc']] for p in pairs])
    diff = mdd - hc
    sites = [p['site'] for p in pairs]
    atlas = pd.read_excel(ROOT / 'data/AAL116.xlsx')
    network = atlas['对应网络'].to_numpy()
    names = list(dict.fromkeys(network))
    netdiff = np.column_stack([diff[:, network == n].mean(1) for n in names])
    p = site_blocked_paired_permutation(diff, sites, n_perm=N, seed=SEED)
    q = bh_fdr(p)
    lo, hi = bootstrap(diff)
    table = pd.DataFrame({'roi': np.arange(1, 117), 'label': atlas['micro命名'],
                          'network': network, 'delta_z': diff.mean(0), 'ci_low': lo,
                          'ci_high': hi, 'p': p, 'q': q, 'n_pairs': len(pairs)})
    table.loc[24, 'label'], table.loc[25, 'label'] = 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R'
    table.to_csv(OUT / '表09_组间偏离_滚动单步_匹配ROI.csv', index=False)
    pn = site_blocked_paired_permutation(netdiff, sites, n_perm=N, seed=SEED)
    qn = bh_fdr(pn)
    ln, hn = bootstrap(netdiff)
    pd.DataFrame({'network': names, 'delta_z': netdiff.mean(0), 'ci_low': ln, 'ci_high': hn,
                  'p': pn, 'q': qn, 'n_pairs': len(pairs)}).to_csv(
                      OUT / '表10_组间偏离_滚动单步_匹配网络.csv', index=False)
    dm, dh = np.abs(mdd).mean(1), np.abs(hc).mean(1)
    ratio, dispersion_p = dispersion(dm, dh)
    whole = diff.mean(axis=1, keepdims=True)
    whole_p = site_blocked_paired_permutation(whole, sites, n_perm=N, seed=SEED)
    whole_lo, whole_hi = bootstrap(whole)
    descriptions = []
    for split in ('all', 'test'):
        groups = {}
        for group in ('hc', 'mdd'):
            selected = ('train', 'val', 'test') if split == 'all' else ('test',)
            group_ids = np.concatenate([ids[group, s] for s in selected])
            group_values = np.concatenate([vals[group, s] for s in selected])
            groups[group] = (group_ids, group_values)
        ih, h = groups['hc']
        im, m = groups['mdd']
        ph = site_blocked_unpaired_permutation(m, h, [parse_site(s) for s in im],
                                                [parse_site(s) for s in ih], n_perm=N, seed=SEED)
        qh = bh_fdr(ph)
        hs = np.array([parse_site(s) for s in ih])
        ms = np.array([parse_site(s) for s in im])
        shared = set(hs) & set(ms)
        descriptions.append({'split': split, 'n_hc': len(ih), 'n_mdd': len(im),
                             'n_hc_shared_sites': int(np.isin(hs, list(shared)).sum()),
                             'n_mdd_shared_sites': int(np.isin(ms, list(shared)).sum()),
                             'n_roi_fdr': int((qh < .05).sum())})
        pd.DataFrame({'roi': np.arange(1, 117), 'delta_unadjusted': m.mean(0)-h.mean(0),
                      'p_site_blocked': ph, 'q': qh, 'n_hc': len(ih), 'n_mdd': len(im)}).to_csv(
                          OUT / f'表11_组间偏离_滚动单步_未匹配{split}.csv', index=False)
    report = {'n_pairs': len(pairs), 'n_sites': len(set(sites)), 'n_perm': N, 'n_boot': N,
              'seed': SEED, 'roi_significant': int((q < .05).sum()),
              'network_significant': int((qn < .05).sum()),
              'max_abs_delta_z': float(np.abs(diff.mean(0)).max()),
              'whole_brain': {'delta_z': float(whole.mean()), 'p': float(whole_p[0]),
                              'ci_low': float(whole_lo[0]), 'ci_high': float(whole_hi[0])},
              'dispersion_variance_ratio': ratio, 'dispersion_p': dispersion_p,
              'dispersion_test': 'two-sided matched-pair label swaps; absolute log variance ratio',
              'unmatched': descriptions,
              'caveat': 'matched pairs span train/val/test; HC train scores are cross-fitted, not independent test-only'}
    (OUT / '原始_组间偏离_20261028.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
