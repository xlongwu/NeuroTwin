"""Exploratory complete-case partial correlations for matched MDD subjects."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils.dataloader import load_condition_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/H1-8分析_20261028/02_病理分析'


def main():
    control = json.loads((OUT / '原始_健康负控_20261028.json').read_text())
    if not control['passed_roi_negative_control']:
        raise RuntimeError('HC 负控未通过，停止临床关联分析')
    pairs = json.loads((ROOT / 'data/splits/site_matched_subset.json').read_text())['pairs']
    z = {}
    for split in ('train', 'val', 'test'):
        with np.load(OUT / '评分缓存' / f'z_matrix_mdd_{split}.npz') as d:
            z.update({str(s): v for s, v in zip(d['subjects'], d['z'])})
    table = load_condition_table(str(ROOT / 'data'), 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx', 'mdd8')
    cols = {spec.name: i for i, spec in enumerate(table.specs)}
    atlas = pd.read_excel(ROOT / 'data/AAL116.xlsx')
    networks = atlas['对应网络'].to_numpy()
    rows = []
    for score in ('HAMD', 'HAMA'):
        selected = []
        for pair in pairs:
            sid = pair['mdd']
            idx = table.row_of_id.get(sid, -1)
            if idx < 0 or sid not in z:
                raise ValueError(f'缺少匹配被试 {sid} 的条件表行或 z 值')
            if all(table.mask[idx, cols[k]] for k in ('Age', 'Sex', score)):
                v = table.values[idx]
                selected.append((sid, pair['site'], float(v[cols['Age']]),
                                 float(v[cols['Sex']]), float(v[cols[score]])))
        sites = sorted({x[1] for x in selected})
        sex = sorted({x[3] for x in selected})
        if len(sex) != 2:
            raise ValueError(f'{score}: Sex 类别数异常：{sex}')
        c = np.array([[1., age, float(s == sex[1])] + [float(site == a) for a in sites[1:]]
                      for _, site, age, s, _ in selected])
        x = np.array([item[4] for item in selected])
        rx = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
        rank = np.linalg.matrix_rank(c)
        dof = len(x) - rank - 1
        if dof <= 0 or rx.std() <= 1e-12:
            raise ValueError(f'{score}: 偏相关自由度不足或临床评分残差为常量')
        for network in dict.fromkeys(networks):
            y = np.array([z[sid][networks == network].mean() for sid, *_ in selected])
            ry = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
            if ry.std() <= 1e-12:
                raise ValueError(f'{score} {network}: 偏离残差为常量')
            r = float(np.corrcoef(rx, ry)[0, 1])
            t = r * np.sqrt(dof / max(1 - r*r, 1e-15))
            rows.append({'score': score, 'network': network, 'n': len(x),
                         'n_missing_or_excluded': len(pairs)-len(x), 'n_sites': len(sites),
                         'dof': dof, 'partial_r': r, 'p_uncorrected': float(2*stats.t.sf(abs(t), dof))})
    pd.DataFrame(rows).to_csv(OUT / '表12_临床关联_滚动单步_网络.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
