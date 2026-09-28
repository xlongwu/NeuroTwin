# coding=utf-8
"""消融实验结果集中登记：registry.jsonl（追加式主档）+ registry.csv（透视用）。

主键为 (exp_id, seed)；upsert_record 以主键覆盖旧记录，保证重跑幂等。
"""
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

from experiments.base_config import ROOT

RESULTS_DIR = ROOT / 'results' / 'ablation'
REGISTRY_JSONL = RESULTS_DIR / 'registry.jsonl'
REGISTRY_CSV = RESULTS_DIR / 'registry.csv'

# 登记字段（固定列序，csv 表头与此一致）
FIELDS = [
    'exp_id', 'group', 'doc_ref', 'priority', 'seed', 'mode', 'status',
    'git_commit', 'started_at', 'finished_at', 'cmd',
    'val_pcc', 'val_mae', 'test_pcc', 'test_mae', 'test_r2', 'test_picp',
    'test_picp_cal',
    'test_fc_upper_mae', 'test_edge_pcc', 'test_subj_pcc', 'test_n_params',
    'log_path', 'note',
]


def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_git_commit():
    """取当前 git commit 短哈希；非 git 环境返回 'unknown'。"""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def load_registry():
    """读取全部登记记录；文件不存在返回空列表。"""
    if not REGISTRY_JSONL.exists():
        return []
    records = []
    for line in REGISTRY_JSONL.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # 跳过损坏行（如进程中断造成的半行），不让单行污染整表
            continue
    return records


def _write_csv(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            writer.writerow({k: ('' if r.get(k) is None else r.get(k))
                             for k in FIELDS})


def upsert_record(record):
    """按 (exp_id, seed) 主键插入或覆盖记录，并重写 jsonl 与 csv。"""
    missing = [k for k in ('exp_id', 'seed') if k not in record]
    if missing:
        raise ValueError(f'登记记录缺少主键字段：{missing}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key = (record['exp_id'], str(record['seed']))
    records = [r for r in load_registry()
               if (r.get('exp_id'), str(r.get('seed'))) != key]
    records.append(record)
    with REGISTRY_JSONL.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    _write_csv(records)


def is_done(exp_id, seed):
    """该 (exp_id, seed) 是否已有 success 记录（用于幂等跳过）。"""
    for r in load_registry():
        if r.get('exp_id') == exp_id and str(r.get('seed')) == str(seed):
            if r.get('status') == 'success':
                return True
    return False
