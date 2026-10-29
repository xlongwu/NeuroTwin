from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=Path('checkpoints/neurotwin_pretrain/base_best.pt')
RUN='独立测试集_20261027'
DATA=ROOT/'results/数据'/RUN
TABLES=ROOT/'results/表格'/RUN
FIGURES=ROOT/'results/可视化'/RUN
SPLITS={'hc':'subject_split_pretrain.json','mdd':'subject_split_finetune.json'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def test_ids(group: str) -> list[str]:
    split=read_json(ROOT/'data'/SPLITS[group])
    ids=list(split['test'])
    if len(set(ids))!=len(ids) or set(ids)&(set(split['train'])|set(split['val'])):
        raise ValueError(f'Invalid independent split: {group}')
    return ids


def check_ids(ids,group: str,complete: bool=True) -> None:
    ids=[str(s) for s in ids]
    expected=set(test_ids(group))
    if len(set(ids))!=len(ids) or not set(ids)<=expected or (complete and set(ids)!=expected):
        raise ValueError(f'Non-test, duplicated, or incomplete subjects: {group}')


def sample_path(group: str,sid: str,data_dir: Path=DATA) -> Path:
    return data_dir/'自主生成'/group/f'{sid}.npz'


def samples(group: str,data_dir: Path=DATA) -> list[dict]:
    result=[]
    for sid in test_ids(group):
        with np.load(sample_path(group,sid,data_dir)) as d:
            result.append({k:d[k].copy() for k in d.files})
    return result
