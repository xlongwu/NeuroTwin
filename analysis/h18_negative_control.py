"""HC validation-versus-test negative control for the H1–8 analysis plan."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pathology_tests import bh_fdr, site_blocked_unpaired_permutation
from utils.site_subset import parse_site

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/H1-8分析_20261028/02_病理分析'


def load(group):
    with np.load(OUT / '评分缓存' / f'z_matrix_hc_{group}.npz') as d:
        subjects, z = d['subjects'].astype(str), d['z'].astype(float)
    return subjects, z, np.array([parse_site(s) for s in subjects])


def contrast(a, b, sa, sb):
    total = len(a) + len(b)
    value = np.zeros(a.shape[1])
    for site in set(sa) & set(sb):
        ma, mb = sa == site, sb == site
        value += ((ma.sum() + mb.sum()) / total) * (a[ma].mean(0) - b[mb].mean(0))
    return value


def main():
    ida, a, sa = load('val')
    idb, b, sb = load('test')
    atlas = pd.read_excel(ROOT / 'data/AAL116.xlsx')
    assert a.shape == (len(ida), 116) and b.shape == (len(idb), 116)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert set(ida).isdisjoint(idb)
    p = site_blocked_unpaired_permutation(a, b, sa, sb, n_perm=10000, seed=2024)
    q = bh_fdr(p)
    roi = pd.DataFrame({'roi': np.arange(1, 117), 'label': atlas['micro命名'],
                        'network': atlas['对应网络'], 'delta_z_site_weighted': contrast(a, b, sa, sb),
                        'p': p, 'q': q, 'significant': q < .05})
    roi.loc[24, 'label'], roi.loc[25, 'label'] = 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R'
    roi.to_csv(OUT / '表07_健康负控_滚动单步_ROI.csv', index=False)
    networks = list(dict.fromkeys(atlas['对应网络']))
    av = np.column_stack([a[:, atlas['对应网络'].to_numpy() == n].mean(1) for n in networks])
    bv = np.column_stack([b[:, atlas['对应网络'].to_numpy() == n].mean(1) for n in networks])
    pn = site_blocked_unpaired_permutation(av, bv, sa, sb, n_perm=10000, seed=2024)
    qn = bh_fdr(pn)
    pd.DataFrame({'network': networks, 'delta_z_site_weighted': contrast(av, bv, sa, sb),
                  'p': pn, 'q': qn, 'significant': qn < .05}).to_csv(
                      OUT / '表08_健康负控_滚动单步_网络.csv', index=False)
    common = set(sa) & set(sb)
    report = {'comparison': 'HC val vs HC test', 'n_val': len(ida), 'n_test': len(idb),
              'n_shared_sites': len(common), 'n_val_in_shared_sites': int(np.isin(sa, list(common)).sum()),
              'n_test_in_shared_sites': int(np.isin(sb, list(common)).sum()),
              'n_perm': 10000, 'seed': 2024, 'fdr_q': .05,
              'roi_significant': int((q < .05).sum()), 'network_significant': int((qn < .05).sum()),
              'passed_roi_negative_control': bool((q < .05).sum() == 0),
              'statistic': 'within-site mean difference weighted by site size / total cohort size; val minus test',
              'single_group_sites': 'excluded from permutation statistic'}
    (OUT / '原始_健康负控_20261028.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
