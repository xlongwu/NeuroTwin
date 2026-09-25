# coding=utf-8
"""消融结果对比报告：读 registry，按 doc_ref 分组输出 Markdown。

用法：
  python -m experiments.compare                # 全部记录
  python -m experiments.compare --group G14_MOE

基线参照：exp_id=baseline 的记录（多 seed 取均值）；缺失时 delta 列为 nan。
"""
import argparse
from collections import defaultdict
from pathlib import Path

from experiments import registry

BASELINE_ID = 'baseline'
METRIC_KEYS = ('val_pcc', 'val_mae', 'test_pcc', 'test_mae', 'test_r2',
               'test_picp')


def _fmt(x):
    if x is None:
        return '-'
    try:
        return f'{float(x):.4f}'
    except (TypeError, ValueError):
        return str(x)


def _mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_report(records):
    """生成 Markdown 文本：按 doc_ref 分组，变体 x seed 行 + 相对基线 delta。"""
    base_rows = [r for r in records if r.get('exp_id') == BASELINE_ID
                 and r.get('status') == 'success']
    ref = {k: _mean([r.get(k) for r in base_rows]) for k in METRIC_KEYS}

    groups = defaultdict(list)
    for r in records:
        groups[(r.get('doc_ref', '?'), r.get('group', '?'))].append(r)

    lines = ['# NeuroTwin 消融对比报告', '',
             f'数据源：`{registry.REGISTRY_JSONL.name}` | '
             f'记录数：{len(records)}', '']
    if base_rows:
        lines.append('基线（baseline 多 seed 均值）：'
                     + ', '.join(f'{k}={_fmt(v)}' for k, v in ref.items()))
    else:
        lines.append('> 警告：registry 中没有成功的 baseline 记录，'
                     'delta 列不可用。请先运行 --only baseline。')
    lines.append('')

    for (doc_ref, group) in sorted(groups, key=lambda x: (str(x[0]), str(x[1]))):
        lines.append(f'## {doc_ref} / {group}')
        lines.append('')
        lines.append('| exp_id | seed | status | val_PCC | val_MAE | test_PCC |'
                     ' test_MAE | R2 | PICP | dPCC | dMAE |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in sorted(groups[(doc_ref, group)],
                        key=lambda x: (x.get('exp_id'), str(x.get('seed')))):
            d_pcc = d_mae = None
            if r.get('status') == 'success' and ref['test_pcc'] is not None:
                if r.get('test_pcc') is not None:
                    d_pcc = float(r['test_pcc']) - ref['test_pcc']
                if r.get('test_mae') is not None:
                    d_mae = float(r['test_mae']) - ref['test_mae']
            lines.append(
                f"| {r.get('exp_id')} | {r.get('seed')} | {r.get('status')} "
                f"| {_fmt(r.get('val_pcc'))} | {_fmt(r.get('val_mae'))} "
                f"| {_fmt(r.get('test_pcc'))} | {_fmt(r.get('test_mae'))} "
                f"| {_fmt(r.get('test_r2'))} | {_fmt(r.get('test_picp'))} "
                f"| {_fmt(d_pcc)} | {_fmt(d_mae)} |")
        lines.append('')

    # 附加指标提示：成功记录的 eval 目录下有完整 metrics_*.json
    lines.append('附加指标（HAMD 分层 / shuffled 负对照 / gate 统计 / FC）：'
                 '见 `results/ablation/eval/<exp_id>_s<seed>/metrics_<split>.json`。')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description='消融结果对比报告')
    ap.add_argument('--group', default=None, help='只输出指定实验组')
    ap.add_argument('--out', default=None, help='输出 Markdown 路径（默认按 group 命名）')
    ns = ap.parse_args()

    records = registry.load_registry()
    if ns.group:
        records = [r for r in records if r.get('group') == ns.group]
    if not records:
        print('registry 为空，先运行 run_experiments.py 产生记录。')
        return 0

    report = build_report(records)
    out_path = (Path(ns.out) if ns.out
                else registry.RESULTS_DIR / f"compare_{ns.group or 'all'}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
    print(report)
    print(f'报告已写入：{out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
