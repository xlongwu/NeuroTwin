#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""站点分层被试级划分：将 ROISignals_window 下的被试按 (数据集版本, 站点) 分层，划分为 train/val/test（默认 8:1:1）。

- 划分单元为被试（subject-level），窗口文件不跨集合；
- 站点由 ID 前缀解析（如 S20 / IS001），数据集版本由站点码体系推断（S*=V1，IS*=V2）；
- 站内用 Hamilton 最大余数法分配名额，保证全局比例精确；
- 产出 JSON（train/val/test ID 清单 + 站点统计）与 CSV（id→split 映射），供 dataloader 通过 --split_file 复用。

用法：
    python scripts/site_stratified_split.py --data_root data --groups MDD HC --ratio 8:1:1 --seed 2024
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

WINDOW_PREFIX = 'ROISignals_'


def scan_subjects(window_dir: Path) -> dict:
    """扫描某组目录，返回 {subj_id: 窗口文件数}。"""
    subjects = defaultdict(int)
    for f in window_dir.glob(f'{WINDOW_PREFIX}*.mat'):
        subj = f.stem[len(WINDOW_PREFIX):].rsplit('-', 1)[0]
        subjects[subj] += 1
    return dict(subjects)


def parse_dataset(site: str) -> str:
    """由站点码推断数据集版本：IS*=V2，S*=V1。"""
    if re.fullmatch(r'IS\d+', site):
        return 'V2'
    if re.fullmatch(r'S\d+', site):
        return 'V1'
    return 'unknown'


def hamilton_allocate(size: int, ratios) -> list:
    """Hamilton 最大余数法：把 size 个名额按 ratios 分配为整数名额。"""
    quotas = [size * r for r in ratios]
    alloc = [int(q) for q in quotas]
    rem = size - sum(alloc)
    order = sorted(range(len(ratios)), key=lambda i: (-(quotas[i] - alloc[i]), i))
    for i in order[:rem]:
        alloc[i] += 1
    return alloc


def stratified_split(subjects: dict, ratios, seed: int):
    """按 (dataset, site) 分层划分。返回 alloc、每站点统计、全局目标。"""
    rng = random.Random(seed)
    site_map = defaultdict(list)
    for sid in subjects:
        site = sid.split('-')[0]
        site_map[(parse_dataset(site), site)].append(sid)

    setnames = ['train', 'val', 'test']
    alloc = {s: [] for s in setnames}
    for key in sorted(site_map):
        sids = sorted(site_map[key])
        rng.shuffle(sids)
        n_tr, n_va, n_te = hamilton_allocate(len(sids), ratios)
        alloc['train'] += sids[:n_tr]
        alloc['val'] += sids[n_tr:n_tr + n_va]
        alloc['test'] += sids[n_tr + n_va:n_tr + n_va + n_te]

    # 全局精确配额：当站内取整造成累计偏差时，在集合间微调
    total = len(subjects)
    target = dict(zip(setnames, hamilton_allocate(total, ratios)))

    def site_total(sid):
        site = sid.split('-')[0]
        return len(site_map[(parse_dataset(site), site)])

    for _ in range(total):
        diff = {s: len(alloc[s]) - target[s] for s in setnames}
        src = max(setnames, key=lambda s: (diff[s], s))
        dst = min(setnames, key=lambda s: (diff[s], s))
        if diff[src] <= 0 or diff[dst] >= 0:
            break
        # 优先从大站点移动，尽量不掏空小站点在训练集的覆盖
        sid = max(alloc[src], key=site_total)
        alloc[src].remove(sid)
        alloc[dst].append(sid)

    per_site = defaultdict(lambda: {'total': 0, 'train': 0, 'val': 0, 'test': 0})
    for s in setnames:
        for sid in alloc[s]:
            site = sid.split('-')[0]
            k = (parse_dataset(site), site)
            per_site[k]['total'] += 1
            per_site[k][s] += 1
    return alloc, dict(per_site), target


def main():
    p = argparse.ArgumentParser(description='站点分层被试级 train/val/test 划分')
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--groups', nargs='+', default=['MDD', 'HC'], choices=['MDD', 'HC'])
    p.add_argument('--ratio', type=str, default='8:1:1', help='train:val:test，如 8:1:1')
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--min_windows', type=int, default=9, help='被试纳入所需的最少窗口文件数')
    p.add_argument('--out_dir', type=str, default=None, help='输出目录，默认 <data_root>/splits')
    p.add_argument('--make_dirs', action='store_true',
                   help='同时生成实体目录 <data_root>/ROISignals_window_split/<组>/<split>/（硬链接，失败时回退复制）')
    args = p.parse_args()

    ratios = [float(x) for x in args.ratio.split(':')]
    ratios = [r / sum(ratios) for r in ratios]
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)

    for group in args.groups:
        window_dir = data_root / 'ROISignals_window' / group
        subjects = scan_subjects(window_dir)
        eligible = {sid: n for sid, n in subjects.items() if n >= args.min_windows}
        excluded = {sid: n for sid, n in subjects.items() if n < args.min_windows}
        if not eligible:
            print(f'[{group}] 无有效被试，跳过')
            continue

        alloc, per_site, target = stratified_split(eligible, ratios, args.seed)

        tag = f'{group}_seed{args.seed}'
        payload = {
            'group': group,
            'ratio': args.ratio,
            'seed': args.seed,
            'min_windows': args.min_windows,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'counts': {s: len(alloc[s]) for s in ('train', 'val', 'test')},
            'excluded_subjects': excluded,
            'per_site': {f'{ds}|{site}': st for (ds, site), st in sorted(per_site.items())},
            'sets': alloc,
        }
        json_path = out_dir / f'site_stratified_split_{tag}.json'
        csv_path = out_dir / f'site_stratified_split_{tag}.csv'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('id,dataset,site,n_windows,split\n')
            for s in ('train', 'val', 'test'):
                for sid in sorted(alloc[s]):
                    site = sid.split('-')[0]
                    f.write(f'{sid},{parse_dataset(site)},{site},{eligible[sid]},{s}\n')

        # 汇总输出
        n_sets = [len(alloc[s]) for s in ('train', 'val', 'test')]
        cov = {s: len({sid.split('-')[0] for sid in alloc[s]}) for s in ('train', 'val', 'test')}
        print(f'[{group}] 有效被试 {len(eligible)}（排除 {len(excluded)}）| '
              f'train/val/test = {n_sets[0]}/{n_sets[1]}/{n_sets[2]}（目标 {target["train"]}/{target["val"]}/{target["test"]}）')
        print(f'      站点覆盖: train {cov["train"]} / val {cov["val"]} / test {cov["test"]} '
              f'(共 {len(per_site)} 个站点)')
        ds_stat = defaultdict(lambda: [0, 0, 0])
        for s, i in zip(('train', 'val', 'test'), range(3)):
            for sid in alloc[s]:
                ds_stat[parse_dataset(sid.split('-')[0])][i] += 1
        for ds, st in sorted(ds_stat.items()):
            print(f'      {ds}: train {st[0]} / val {st[1]} / test {st[2]}')
        print(f'      输出: {json_path.name} + {csv_path.name}')

        if args.make_dirs:
            split_root = data_root / 'ROISignals_window_split' / group
            n_linked = 0
            for s in ('train', 'val', 'test'):
                d = split_root / s
                d.mkdir(parents=True, exist_ok=True)
                for sid in alloc[s]:
                    for src in sorted(window_dir.glob(f'{WINDOW_PREFIX}{sid}-*.mat')):
                        dst = d / src.name
                        if dst.exists():
                            continue
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy2(src, dst)
                        n_linked += 1
            print(f'      实体目录: {split_root}（链接/复制 {n_linked} 个窗口文件）')


if __name__ == '__main__':
    sys.exit(main())
