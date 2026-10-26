# coding=utf-8
"""utils/site_subset 的确定性单元测试（不依赖真实数据目录）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.site_subset import (build_matched_subset, build_site_counts,
                               match_within_site, parse_site)

import pytest


def test_parse_site_first_segment():
    assert parse_site('IS001-1-0001') == 'IS001'
    assert parse_site('S20-2-0133') == 'S20'
    assert parse_site('IS014-1-0002 ') == 'IS014'
    with pytest.raises(ValueError):
        parse_site('')
    with pytest.raises(ValueError):
        parse_site('-1-0001')


def _demo(age, sex=1.0):
    return {'Sex': sex, 'Age': age, 'education': None}


def test_build_site_counts_and_keep_flag():
    hc = ['S1-2-0001', 'S1-2-0002', 'S2-2-0001']
    mdd = ['S1-1-0001', 'S2-1-0001']
    rows = build_site_counts(hc, mdd, {'S1-2-0001': _demo(30)}, {'S1-1-0001': _demo(31)},
                             min_per_group=2)
    by_site = {r['site']: r for r in rows}
    assert by_site['S1']['n_hc'] == 2 and by_site['S1']['n_mdd'] == 1
    assert by_site['S1']['keep'] is False          # MDD 只有 1 人 < 2
    assert by_site['S2']['keep'] is False
    rows10 = build_site_counts(hc, mdd, {}, {}, min_per_group=1)
    assert {r['site']: r['keep'] for r in rows10} == {'S1': True, 'S2': True}


def test_match_within_site_rules():
    hc = [('hc_a', _demo(30.0, sex=1)), ('hc_b', _demo(30.5, sex=1)),
          ('hc_c', _demo(40.0, sex=2)), ('hc_d', _demo(31.0, sex=1))]
    mdd = [('mdd_1', _demo(30.2, sex=1)),   # 最近 hc_a(0.2) < hc_d(0.8)
           ('mdd_2', _demo(30.6, sex=1)),   # 剩余中最近 hc_d(0.4)（hc_b 0.1 也近）
           ('mdd_3', _demo(38.0, sex=2)),   # Sex=2 只有 hc_c，卡钳 5 内
           ('mdd_4', _demo(50.0, sex=1))]   # 无卡钳内候选 → 不配对
    pairs = match_within_site(hc, mdd, caliper=5.0)
    got = dict(pairs)
    # mdd_2 在 hc_a 已被占用后：候选 hc_b(0.1) 与 hc_d(0.4)，最近是 hc_b
    assert got['mdd_1'] == 'hc_a'
    assert got['mdd_2'] == 'hc_b'
    assert got['mdd_3'] == 'hc_c'
    assert 'mdd_4' not in got
    used_hc = [h for _, h in pairs]
    assert len(used_hc) == len(set(used_hc))       # 无放回


def test_match_within_site_deterministic_and_tiebreak():
    hc = [('hc_1', _demo(30.0)), ('hc_2', _demo(30.0))]
    mdd = [('mdd_1', _demo(30.0)), ('mdd_2', _demo(30.0))]
    p1 = match_within_site(hc, mdd)
    p2 = match_within_site(list(reversed(hc)), list(reversed(mdd)))
    assert p1 == p2 == [('mdd_1', 'hc_1'), ('mdd_2', 'hc_2')]   # 平局取 ID 序最小


def test_build_matched_subset_structure_and_exclusions():
    hc = [f'S1-2-{i:04d}' for i in range(1, 4)] + ['S9-2-0001']
    mdd = [f'S1-1-{i:04d}' for i in range(1, 4)] + ['S9-1-0001']
    hc_demo = {s: _demo(30.0 + i) for i, s in enumerate(hc)}
    mdd_demo = {s: _demo(30.0 + i) for i, s in enumerate(mdd)}
    subset = build_matched_subset(hc, mdd, hc_demo, mdd_demo,
                                  min_per_group=2, caliper=5.0)
    assert subset['n_kept_sites'] == 1 and subset['kept_sites'] == ['S1']
    assert [d['site'] for d in subset['dropped_sites']] == ['S9']
    assert subset['n_pairs'] == 3
    assert all(p['pair_id'].startswith('S1#') for p in subset['pairs'])
    hc_used = [p['hc'] for p in subset['pairs']]
    mdd_used = [p['mdd'] for p in subset['pairs']]
    assert len(set(hc_used)) == 3 and len(set(mdd_used)) == 3
    # 卡钳内匹配：年龄差均 ≤ 5
    for p in subset['pairs']:
        assert abs(p['mdd_demo']['Age'] - p['hc_demo']['Age']) <= 5.0
        assert p['mdd_demo']['Sex'] == p['hc_demo']['Sex']
    assert json.dumps(subset, sort_keys=True) == json.dumps(
        build_matched_subset(hc, mdd, hc_demo, mdd_demo, 2, 5.0), sort_keys=True)


def test_missing_demo_excluded_from_matching():
    hc = ['S1-2-0001', 'S1-2-0002']
    mdd = ['S1-1-0001', 'S1-1-0002']
    hc_demo = {'S1-2-0001': _demo(30)}        # S1-2-0002 缺人口学
    mdd_demo = {'S1-1-0001': _demo(30), 'S1-1-0002': _demo(31)}
    subset = build_matched_subset(hc, mdd, hc_demo, mdd_demo,
                                  min_per_group=2, caliper=5.0)
    # S1-2-0002 缺人口学但仍在影像计数里 → 站点保留；匹配只用有效者
    assert subset['n_pairs'] == 1
    assert subset['pairs'][0]['hc'] == 'S1-2-0001'
    assert 'S1-1-0002' in subset['unmatched_mdd_in_kept_sites']
