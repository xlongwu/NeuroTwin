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
               'test_picp', 'test_picp_cal', 'test_fc_upper_mae',
               'test_edge_pcc', 'test_subj_pcc', 'test_n_params')


def _fmt(x):
    if x is None:
        return '-'
    try:
        return f'{float(x):.4f}'
    except (TypeError, ValueError):
        return str(x)


def _fmt_params(x):
    if x is None:
        return '-'
    try:
        return f'{float(x) / 1e6:.2f}M'
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
                     ' test_MAE | R2 | PICP | PICP_cal | FC_MAE | edgePCC |'
                     ' subjPCC | params | dPCC | dMAE |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
                     '---|---|')
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
                f"| {_fmt(r.get('test_picp_cal'))} "
                f"| {_fmt(r.get('test_fc_upper_mae'))} "
                f"| {_fmt(r.get('test_edge_pcc'))} "
                f"| {_fmt(r.get('test_subj_pcc'))} "
                f"| {_fmt_params(r.get('test_n_params'))} "
                f"| {_fmt(d_pcc)} | {_fmt(d_mae)} |")
        lines.append('')

    # 附加指标提示：成功记录的 eval 目录下有完整 metrics_*.json 与被试级 CSV
    lines.append('附加指标（HAMD 分层 / shuffled 负对照 / gate 统计 / FC 边级与'
                 '网络级 / 被试级聚合）：见 '
                 '`results/ablation/eval/<exp_id>_s<seed>/metrics_<split>.json` '
                 '与 `subject_metrics_<split>.csv`。')
    return '\n'.join(lines) + '\n'


def paired_subject_tests(pairs):
    """被试级配对统计检验：对齐 subject_metrics_test.csv 的逐被试 PCC 后检验。

    Args:
        pairs: [(exp_id, eval_dir), ...]，eval_dir 为 evaluate_variant 输出目录。

    Returns:
        (lines, ok)：markdown 行列表与是否存在可检验对。
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    series = {}
    for exp_id, eval_dir in pairs:
        csv_path = Path(eval_dir) / 'subject_metrics_test.csv'
        if not csv_path.exists():
            print(f'警告: {exp_id} 缺少 {csv_path}，跳过')
            continue
        df = pd.read_csv(csv_path)
        if 'subj_id' not in df.columns or 'pcc_mean' not in df.columns:
            print(f'警告: {exp_id} 的 CSV 缺少 subj_id/pcc_mean 列，跳过')
            continue
        series[exp_id] = df.set_index('subj_id')['pcc_mean']

    if len(series) < 2:
        return (['需要至少两个含 subject_metrics_test.csv 的评估目录。',
                 '用法: python -m experiments.compare --paired_tests '
                 'id1=<eval_dir1> id2=<eval_dir2> [...]'], False)

    lines = ['# 被试级配对统计检验（test, 逐被试 PCC）', '',
             '样本级滑窗样本非独立，检验以被试为统计单位（2.1 节协议）。', '']
    ids = list(series)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = series[ids[i]], series[ids[j]]
            common = a.index.intersection(b.index)
            if len(common) < 8:
                lines.append(f'## {ids[i]} vs {ids[j]}：共同被试不足（{len(common)}），跳过')
                lines.append('')
                continue
            da, db = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            diff = da - db
            t_stat, t_p = stats.ttest_rel(da, db)
            try:
                w_stat, w_p = stats.wilcoxon(da, db)
            except ValueError:  # 全零差值等退化情形
                w_stat, w_p = float('nan'), float('nan')
            lines.append(f'## {ids[i]} vs {ids[j]}（n={len(common)} 被试）')
            lines.append('')
            lines.append(f'- PCC 均值差（{ids[i]} − {ids[j]}）: {diff.mean():+.4f}'
                         f' ± {diff.std(ddof=1) if len(common) > 1 else float("nan"):.4f}')
            lines.append(f'- 配对 t 检验: t={t_stat:.3f}, p={t_p:.4g}')
            lines.append(f'- Wilcoxon 符号秩: W={w_stat:.1f}, p={w_p:.4g}')
            sig = '显著' if min(t_p, w_p) < 0.05 else '不显著'
            lines.append(f'- 结论: α=0.05 下{sig}（两检验取更保守者供参考，'
                         '多重比较未校正，正式结论需纳入校正）')
            lines.append('')
    return lines, True


def main():
    ap = argparse.ArgumentParser(description='消融结果对比报告')
    ap.add_argument('--group', default=None, help='只输出指定实验组')
    ap.add_argument('--out', default=None, help='输出 Markdown 路径（默认按 group 命名）')
    ap.add_argument('--paired_tests', nargs='*', metavar='EXP_ID=EVAL_DIR',
                    help='被试级配对检验：给定各变体的 evaluate_variant 输出目录，'
                         '对 subject_metrics_test.csv 的逐被试 PCC 做配对 t / Wilcoxon 检验')
    ns = ap.parse_args()

    if ns.paired_tests:
        pairs = []
        for item in ns.paired_tests:
            if '=' not in item:
                print(f'格式错误（需 EXP_ID=EVAL_DIR）: {item}')
                return 2
            exp_id, eval_dir = item.split('=', 1)
            pairs.append((exp_id, eval_dir))
        lines, ok = paired_subject_tests(pairs)
        report = '\n'.join(lines) + '\n'
        out_path = (Path(ns.out) if ns.out
                    else registry.RESULTS_DIR / 'paired_tests.md')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding='utf-8')
        print(report)
        print(f'检验报告已写入：{out_path}')
        return 0 if ok else 1

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
