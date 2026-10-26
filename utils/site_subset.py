# coding=utf-8
"""站点解析与 HC/MDD 匹配子集（v2 研究计划 §4）。

- 站点码 = 被试 ID 的首段（``IS001-1-0001 → IS001``），与既有
  ``utils/site_stratified_split.py`` 的解析规则同源；被试清单从
  ``Raw/<group>/ROISignals_*.mat`` 枚举（与 raw 数据集同一队列口径）。
- **计数表**（``--emit-count-table``）：全队列 site×group 计数与人口学有效数，
  写 ``data/splits/site_count_table.csv`` + ``.json``，供 V2 人工核对解析规则；
  若计数表证据显示中间段才是站点码，以证据修正本模块并记录，重跑只影响
  匹配产物，不污染上游缓存。
- **匹配**（``--match``）：保留 HC/MDD 人数均 ≥ ``--min_per_group`` 的站点；
  站内按 Age/Sex 做 1:1 最近邻**无放回**匹配（卡钳 Age±``--caliper`` 岁、
  Sex 严格同层；MDD 侧按 (age, id) 排序依次配对，同龄平局取 ID 序最小的 HC，
  全程确定性可复现）。产物 ``data/splits/site_matched_subset.json``。

人口学（Sex/Age/education）只从 merged 表读取，缺 Age 或 Sex 的被试无法参与
匹配（计数后排除，不插补）；education 仅随表携带，不参与匹配约束。

用法::

    python -m utils.site_subset --emit-count-table
    python -m utils.site_subset --match [--min-per-group 10] [--caliper 5]
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

HC_MERGED_FILE = 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'
MDD_MERGED_FILE = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'
COUNT_TABLE_JSON = 'site_count_table.json'
COUNT_TABLE_CSV = 'site_count_table.csv'
MATCHED_SUBSET_JSON = 'site_matched_subset.json'
DEMO_FIELDS = ('Sex', 'Age', 'education')


def parse_site(subject_id: str) -> str:
    """被试 ID → 站点码（首段；空段直接报错，不猜测）。"""
    sid = str(subject_id).strip()
    if not sid:
        raise ValueError(f'被试 ID 为空，无法解析站点码: {subject_id!r}')
    site = sid.split('-')[0].strip()
    if not site:
        raise ValueError(f'被试 ID 首段为空，无法解析站点码: {subject_id!r}')
    return site


def scan_raw_subjects(raw_dir: Path) -> List[str]:
    """枚举 ``Raw/<group>/ROISignals_*.mat`` 的被试 ID（排序，与数据集口径一致）。"""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f'原始数据目录不存在: {raw_dir}')
    ids = sorted(p.stem.replace('ROISignals_', '').strip()
                 for p in raw_dir.glob('ROISignals_*.mat'))
    if not ids:
        raise ValueError(f'{raw_dir} 下没有 ROISignals_*.mat 被试文件')
    return ids


def load_demo(data_root: Path, merged_file: str,
              subject_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, float]],
                                                  List[str]]:
    """从 merged 表读取被试人口学。

    Returns:
        (demo, missing): ``demo`` 只包含 Sex/Age 均有效且在被试清单内的被试
        （education 可缺失，随表携带）；``missing`` 为无法参与匹配的被试
        （不在表中，或 Sex/Age 缺失/不可解析）。
    """
    import numpy as np
    import pandas as pd

    path = Path(data_root) / merged_file
    if not path.exists():
        raise FileNotFoundError(f'人口学 merged 表不存在: {path}')
    df = pd.read_excel(path)
    for col in ('ID', 'Sex', 'Age'):
        if col not in df.columns:
            raise ValueError(f'{path} 缺少必需列 {col}（现有列含 ID/Sex/Age）')
    df = df[['ID', *DEMO_FIELDS]].copy()
    df['ID'] = df['ID'].astype(str).str.strip()
    wanted = set(str(s) for s in subject_ids)
    rows = df[df['ID'].isin(wanted)]
    demo: Dict[str, Dict[str, float]] = {}
    missing: List[str] = []
    seen = set(rows['ID'])
    for _, row in rows.iterrows():
        sid = str(row['ID'])
        try:
            sex = float(row['Sex'])
            age = float(row['Age'])
            edu = row.get('education')
            edu_v = None if edu is None or pd.isna(edu) else float(edu)
        except (TypeError, ValueError):
            missing.append(sid)
            continue
        if not (np.isfinite(sex) and np.isfinite(age)):
            missing.append(sid)
            continue
        demo[sid] = {'Sex': sex, 'Age': age, 'education': edu_v}
    missing.extend(sorted(wanted - seen))
    return demo, sorted(set(missing))


def build_site_counts(hc_ids: Sequence[str], mdd_ids: Sequence[str],
                      hc_demo: Dict[str, Dict[str, float]],
                      mdd_demo: Dict[str, Dict[str, float]],
                      min_per_group: int = 10) -> List[Dict]:
    """site×group 计数表（按站点码排序；keep = 两组影像人数均 ≥ min_per_group）。"""
    sites = sorted({parse_site(s) for s in hc_ids} | {parse_site(s) for s in mdd_ids})
    rows = []
    for site in sites:
        hc_n = sum(1 for s in hc_ids if parse_site(s) == site)
        mdd_n = sum(1 for s in mdd_ids if parse_site(s) == site)
        hc_ok = sum(1 for s in hc_demo if parse_site(s) == site)
        mdd_ok = sum(1 for s in mdd_demo if parse_site(s) == site)
        rows.append({
            'site': site,
            'n_hc': hc_n, 'n_mdd': mdd_n,
            'n_hc_demo_valid': hc_ok, 'n_mdd_demo_valid': mdd_ok,
            'keep': bool(hc_n >= min_per_group and mdd_n >= min_per_group),
        })
    return rows


def match_within_site(hc_entries: List[Tuple[str, Dict[str, float]]],
                      mdd_entries: List[Tuple[str, Dict[str, float]]],
                      caliper: float = 5.0) -> List[Tuple[str, str]]:
    """站内 1:1 最近邻无放回匹配，返回 [(mdd_id, hc_id), ...]。

    规则（确定性，无随机数）：MDD 侧按 (Age, id) 升序逐个配对；每个 MDD 在
    同层（Sex 相等）且 |ΔAge| ≤ caliper 的未用 HC 中选 |ΔAge| 最小者，
    平局取 HC ID 字典序最小。education 不参与约束。
    """
    mdd_sorted = sorted(mdd_entries, key=lambda kv: (float(kv[1]['Age']), kv[0]))
    available = {sid: demo for sid, demo in hc_entries}
    pairs: List[Tuple[str, str]] = []
    for mdd_id, md in mdd_sorted:
        candidates = [
            (abs(float(hd['Age']) - float(md['Age'])), hc_id)
            for hc_id, hd in available.items()
            if float(hd['Sex']) == float(md['Sex'])
            and abs(float(hd['Age']) - float(md['Age'])) <= float(caliper)
        ]
        if not candidates:
            continue
        candidates.sort()                     # (ΔAge, hc_id)：平局按 ID 序
        _, best = candidates[0]
        pairs.append((mdd_id, best))
        del available[best]                   # 无放回
    return pairs


def build_matched_subset(hc_ids: Sequence[str], mdd_ids: Sequence[str],
                         hc_demo: Dict[str, Dict[str, float]],
                         mdd_demo: Dict[str, Dict[str, float]],
                         min_per_group: int = 10, caliper: float = 5.0) -> dict:
    """按 v2 §4 规则构建匹配子集（详见模块 docstring）。"""
    counts = build_site_counts(hc_ids, mdd_ids, hc_demo, mdd_demo,
                               min_per_group=min_per_group)
    kept_sites = [r['site'] for r in counts if r['keep']]
    dropped = [{'site': r['site'], 'n_hc': r['n_hc'], 'n_mdd': r['n_mdd']}
               for r in counts if not r['keep']]

    pairs_out: List[dict] = []
    unmatched_mdd: List[str] = []
    for site in kept_sites:
        hc_entries = sorted((sid, d) for sid, d in hc_demo.items()
                            if parse_site(sid) == site)
        mdd_entries = sorted((sid, d) for sid, d in mdd_demo.items()
                             if parse_site(sid) == site)
        pairs = match_within_site(hc_entries, mdd_entries, caliper=caliper)
        matched_mdd = {m for m, _ in pairs}
        unmatched_mdd.extend(sorted(sid for sid, _ in mdd_entries
                                    if sid not in matched_mdd))
        for k, (mdd_id, hc_id) in enumerate(pairs, start=1):
            pairs_out.append({
                'pair_id': f'{site}#{k:04d}',
                'site': site,
                'mdd': mdd_id, 'hc': hc_id,
                'mdd_demo': mdd_demo[mdd_id], 'hc_demo': hc_demo[hc_id],
            })
    return {
        'rules': {
            'site_parse': "ID 首段（'-' 分隔）",
            'min_per_group': int(min_per_group),
            'caliper_age': float(caliper),
            'sex': '严格同层',
            'design': '站内 1:1 最近邻无放回；MDD 按 (Age,id) 序，平局取 HC ID 最小',
        },
        'n_kept_sites': len(kept_sites),
        'kept_sites': kept_sites,
        'dropped_sites': dropped,
        'n_pairs': len(pairs_out),
        'pairs': pairs_out,
        'unmatched_mdd_in_kept_sites': unmatched_mdd,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    import csv
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', type=Path, default=Path('data'))
    ap.add_argument('--out-dir', type=Path, default=Path('data/splits'))
    ap.add_argument('--min-per-group', type=int, default=10,
                    help='站点保留阈值：HC 与 MDD 影像人数均 ≥ 该值（默认 10）')
    ap.add_argument('--caliper', type=float, default=5.0,
                    help='Age 匹配卡钳（± 岁，默认 5）')
    ap.add_argument('--emit-count-table', action='store_true',
                    help='产出 site×group 计数表（V2 人工核对用）')
    ap.add_argument('--match', action='store_true',
                    help='构建站点匹配子集 site_matched_subset.json')
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    if not (ns.emit_count_table or ns.match):
        ap.error('至少指定 --emit-count-table 或 --match 之一')

    hc_ids = scan_raw_subjects(ns.data_root / 'Raw' / 'HC')
    mdd_ids = scan_raw_subjects(ns.data_root / 'Raw' / 'MDD')
    log.info('队列 | HC=%d MDD=%d（Raw 文件枚举）', len(hc_ids), len(mdd_ids))
    hc_demo, hc_missing = load_demo(ns.data_root, HC_MERGED_FILE, hc_ids)
    mdd_demo, mdd_missing = load_demo(ns.data_root, MDD_MERGED_FILE, mdd_ids)
    log.info('人口学有效 | HC=%d（缺 %d） MDD=%d（缺 %d）',
             len(hc_demo), len(hc_missing), len(mdd_demo), len(mdd_missing))

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    if ns.emit_count_table:
        rows = build_site_counts(hc_ids, mdd_ids, hc_demo, mdd_demo,
                                 min_per_group=ns.min_per_group)
        payload = {
            'site_parse': "ID 首段（'-' 分隔）；队列来源 Raw/<group>/ROISignals_*.mat",
            'min_per_group': int(ns.min_per_group),
            'n_sites': len(rows),
            'counts': rows,
            'hc_demo_missing': hc_missing,
            'mdd_demo_missing': mdd_missing,
        }
        (ns.out_dir / COUNT_TABLE_JSON).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        _write_csv(ns.out_dir / COUNT_TABLE_CSV, rows)
        keep = [r for r in rows if r['keep']]
        log.info('计数表已写入 %s / %s | 站点 %d 个，满足 ≥%d/组 的 %d 个',
                 ns.out_dir / COUNT_TABLE_JSON, ns.out_dir / COUNT_TABLE_CSV,
                 len(rows), ns.min_per_group, len(keep))

    if ns.match:
        subset = build_matched_subset(hc_ids, mdd_ids, hc_demo, mdd_demo,
                                      min_per_group=ns.min_per_group,
                                      caliper=ns.caliper)
        (ns.out_dir / MATCHED_SUBSET_JSON).write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        log.info('匹配子集已写入 %s | 保留站点 %d | 匹配对 %d | 站内未配对 MDD %d',
                 ns.out_dir / MATCHED_SUBSET_JSON, subset['n_kept_sites'],
                 subset['n_pairs'], len(subset['unmatched_mdd_in_kept_sites']))


if __name__ == '__main__':
    main()
