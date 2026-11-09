"""读取V2 正式 H32 评价产物，导出逐时距及阶段汇总。"""
import argparse
import json
from pathlib import Path

import pandas as pd

from utils.common import split_label


def load_eval_artifacts(eval_dir, split='test'):
    directory = Path(eval_dir)
    label = split_label(split)
    metrics = json.loads((directory / f'指标_{label}.json').read_text(encoding='utf-8'))
    if metrics['protocol']['model'] != 'V2' or metrics['protocol']['horizon'] != 32:
        raise ValueError('需要V2 的 V2 H32 评价结果')
    tables = {}
    for name in ('逐时距指标', '分阶段指标', '逐被试分阶段指标'):
        path = directory / '表格' / f'{name}_{label}.csv'
        frame = pd.read_csv(path)
        required = {'模型', '预测范围', 'mae', 'rmse'}
        if name == '逐被试分阶段指标':
            required.add('被试')
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{path} 缺少列: {sorted(missing)}')
        tables[name] = frame
    return dict(metrics=metrics, **tables)


def baseline_comparison(metrics):
    rows = []
    for name, report in {'V2': metrics, **metrics['baselines']}.items():
        row = {'模型': name, '预测范围': 'H1_32'}
        for metric, values in report['trajectory_metrics'].items():
            if 'H1_32' in values:
                row[metric] = values['H1_32']['mean']
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(eval_dir, out_dir, split='test'):
    data = load_eval_artifacts(eval_dir, split)
    out = Path(out_dir)
    tables = out / '表格'
    tables.mkdir(parents=True, exist_ok=True)
    for name in ('逐时距指标', '分阶段指标', '逐被试分阶段指标'):
        data[name].to_csv(tables / f'{name}.csv', index=False)
    comparison = baseline_comparison(data['metrics'])
    comparison.to_csv(tables / '完整时距模型与基线.csv', index=False)
    protocol = data['metrics']['protocol']
    lines = ['# V2 预测评价', '',
             f"划分：{split}；被试：{protocol['n_subjects']}；锚点：{protocol['n_anchors']}。", '',
             '逐时距 PCC 沿 ROI 计算；时间 PCC 沿阶段时间轴计算。指标先按锚点计算，'
             '再在被试内平均，最后跨被试等权。', '',
             '完整时距结果（H1–32）：', '',
             '| 模型 | MAE | 时间 PCC | FC MSE | 频域 L1 |',
             '|---|---:|---:|---:|---:|']
    for row in comparison.to_dict('records'):
        lines.append(f"| {row['模型']} | {row['mae']:.6f} | {row['temporal_pcc']:.6f} | "
                     f"{row['fc_mse']:.6f} | {row['freq_l1']:.6f} |")
    lines.extend(['', '逐时距、阶段及逐被试明细见表格目录。', ''])
    (out / '评价说明.md').write_text('\n'.join(lines), encoding='utf-8')
    return data



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', choices=['val', 'test', 'train'], default='test')
    args = parser.parse_args()
    write_report(args.eval_dir, args.out_dir, args.split)


if __name__ == '__main__':
    main()
