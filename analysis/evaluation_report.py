"""读取方案 B 正式 H32 评价产物，导出逐时距及阶段汇总。"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

from utils.common import split_label


def load_eval_artifacts(eval_dir, split='test'):
    directory = Path(eval_dir)
    label = split_label(split)
    metrics = json.loads((directory / f'指标_{label}.json').read_text(encoding='utf-8'))
    if metrics['protocol']['model'] != 'CPM' or metrics['protocol']['horizon'] != 32:
        raise ValueError('需要方案 B 的 CPM H32 评价结果')
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
    for name, report in {'CPM': metrics, **metrics['baselines']}.items():
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
    lines = ['# 方案 B 预测评价', '',
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


def _paired_arrays(original_dir, optimized_dir, split):
    """按被试/锚点对齐真实评价缓存，禁止使用不同历史或真值画模型对比。"""
    arrays = []
    for directory in (original_dir, optimized_dir):
        with np.load(Path(directory)/f'预测数组_{split_label(split)}.npz') as archive:
            data = {key: archive[key] for key in ('subjects', 'cutoff', 'history', 'target', 'mask', 'anchor', 'pred')}
        order = np.lexsort((data['cutoff'], data['subjects']))
        data = {key: value[order] for key, value in data.items()}
        keys = list(zip(data['subjects'], data['cutoff']))
        if len(set(keys)) != len(keys):
            raise ValueError('对比缓存含重复被试/锚点')
        arrays.append(data)
    for key in ('subjects', 'cutoff', 'history', 'target', 'mask', 'anchor'):
        if not np.array_equal(arrays[0][key], arrays[1][key]):
            raise ValueError(f'原B与新模型的{key}不一致，不能进行同协议比较')
    return arrays


def write_comparison(original_dir, optimized_dir, out_dir, split='test'):
    """复用已有指标和预测缓存，生成同队列原B/新模型的四类PDF图。"""
    reports = [load_eval_artifacts(d, split) for d in (original_dir, optimized_dir)]
    protocols = [r['metrics']['protocol'] for r in reports]
    for key in ('split_hash', 'mode', 'horizon', 'context_length', 'tr_seconds', 'calibration_length', 'normalization'):
        if protocols[0].get(key) is None or protocols[0].get(key) != protocols[1].get(key):
            raise ValueError(f'原B与新模型评价协议缺少或不匹配: {key}；请用当前evaluate --save_arrays重新评价')
    if protocols[0].get('multiscale_phase') or not protocols[1].get('multiscale_phase'):
        raise ValueError('需要原B与已启用multiscale_phase的新模型评价结果')
    arrays = _paired_arrays(original_dir, optimized_dir, split)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('比较输出目录非空，请使用新目录以保留已有结果')
    for name in ('图片', '表格', '说明'):
        (out/name).mkdir(parents=True, exist_ok=True)
    names, colors = ('原B', '多尺度相位'), ('#6592ac', '#2b438d')
    for table in ('逐时距指标', '分阶段指标', '逐被试分阶段指标'):
        frames = []
        for name, report in zip(names, reports):
            frame = report[table].copy()
            frame['模型'] = name
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(out/'表格'/f'{table}对比.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = [f.name for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if fonts:
        plt.rcParams['font.family'] = fonts[0]
    plt.rcParams.update({'pdf.fonttype': 3, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    cohort = 'HC' if protocols[0]['mode'] == 'pretrain' else 'MDD'
    tr = protocols[0]['tr_seconds']
    stages = ('H1_8', 'H9_16', 'H17_32', 'H1_32')

    def stage_values(report, key):
        return [report['metrics']['trajectory_metrics'][key][s]['mean'] for s in stages]

    def save(fig, name):
        fig.savefig(out/'图片'/f'{cohort}_{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for name, color, report in zip(names, colors, reports):
        frame = report['逐时距指标'].copy()
        frame['时距'] = frame['预测范围'].str.removeprefix('H').astype(int)
        frame = frame.sort_values('时距')
        for ax, metric in zip(axes, ('mae', 'rmse', 'spatial_pcc')):
            ax.plot(frame['时距'], frame[metric], label=name, color=color)
            ax.set(xlabel='预测时距（TR）', ylabel=metric)
            ax.set_xticks([1, 8, 16, 24, 32])
    axes[0].legend()
    fig.suptitle(f'{cohort}｜逐horizon预测性能')
    save(fig, '逐时距预测性能')

    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for i, (name, color, report) in enumerate(zip(names, colors, reports)):
        ax.bar(np.arange(4)+(i-.5)*.35, stage_values(report, 'temporal_pcc'),
               width=.35, color=color, label=name)
    ax.axhline(0, color='gray', lw=.6)
    ax.set(xticks=np.arange(4), xticklabels=[s.replace('_', '–') for s in stages],
           ylabel='Temporal PCC', title=f'{cohort}｜真实未来轨迹相关性')
    ax.legend()
    save(fig, '未来轨迹相关性')

    ref, new = arrays
    chosen_subjects = sorted(set(ref['subjects']))
    chosen_subjects = [chosen_subjects[i] for i in sorted(set([0, len(chosen_subjects)//2, len(chosen_subjects)-1]))]
    f, k = ref['history'].shape[1:]
    rois = sorted(set([0, f//2, f-1]))
    fig, axes = plt.subplots(len(chosen_subjects), len(rois), figsize=(13, 3*len(chosen_subjects)),
                             squeeze=False, layout='constrained')
    representatives = []
    for row, sid in enumerate(chosen_subjects):
        idx = int(np.flatnonzero(ref['subjects'] == sid)[0])
        for col, roi in enumerate(rois):
            ax = axes[row, col]
            ax.plot(np.arange(-k+1, 1)*tr, ref['history'][idx, roi], color='#999999', label='真实历史')
            future_t = np.arange(33)*tr
            for values, label, color in [(ref['target'], '真实未来', 'black'),
                                          (ref['pred'], names[0], colors[0]), (new['pred'], names[1], colors[1])]:
                ax.plot(future_t, np.r_[ref['anchor'][idx, roi], values[idx, roi]], color=color, label=label)
            ax.axvline(0, ls='--', color='gray')
            ax.set(title=f'{sid}｜ROI {roi+1}', xlabel='相对锚点时间（秒）', ylabel='校准BOLD')
            representatives.append({'被试': sid, '锚点cutoff': int(ref['cutoff'][idx]), 'ROI': roi+1})
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{cohort}｜固定排序首/中/末被试，最早锚点，固定首/中/末ROI')
    save(fig, '真实与预测轨迹')
    pd.DataFrame(representatives).to_csv(out/'表格'/'代表序列选择.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for name, color, report in zip(names, colors, reports):
        for ax, key in [(axes[0, 0], 'std_ratio'), (axes[0, 1], 'delta_pcc'), (axes[1, 1], 'mae')]:
            ax.plot(range(4), stage_values(report, key), marker='o', label=name, color=color)
            ax.set(xticks=range(4), xticklabels=[s.replace('_', '–') for s in stages], ylabel=key)
    axes[0, 0].axhline(1, ls='--', color='gray', lw=.7)
    freq = np.fft.rfftfreq(32, tr)
    for values, name, color in [(ref['target'], '真实未来', 'black'), (ref['pred'], names[0], colors[0]),
                                 (new['pred'], names[1], colors[1])]:
        power = np.abs(np.fft.rfft(values-values.mean(-1, keepdims=True), axis=-1))**2
        power[..., 1:-1] *= 2
        density = power / np.maximum(power.sum(-1, keepdims=True)*(freq[1]-freq[0]), 1e-12)
        per_anchor = density.mean(1)
        equal_subject = np.stack([per_anchor[ref['subjects'] == sid].mean(0)
                                  for sid in np.unique(ref['subjects'])]).mean(0)
        axes[1, 0].plot(freq, equal_subject, color=color, label=name)
    axes[1, 0].set(xlabel='频率（Hz）', ylabel='归一化功率谱（1/Hz）')
    axes[1, 0].legend()
    axes[0, 0].legend()
    fig.suptitle(f'{cohort}｜动态与频谱保真（先ROI、锚点、被试内平均，跨被试等权）')
    save(fig, '动态与频谱保真')
    (out/'说明'/'对比记录.json').write_text(json.dumps(
        {'原B': protocols[0], '多尺度相位': protocols[1],
         '代表规则': '按被试ID排序首/中/末；最早锚点；ROI首/中/末，未按预测效果选择'},
        ensure_ascii=False, indent=2)+'\n')
    lines = ['# 同协议模型比较', '', f'队列：{cohort}；划分：{split}。', '',
             '逐时距PCC沿ROI计算；阶段Temporal PCC沿时间计算。所有指标跨被试等权。', '',
             '| 模型 | 范围 | MAE | RMSE | Temporal PCC |', '|---|---|---:|---:|---:|']
    for name, report in zip(names, reports):
        metrics = report['metrics']['trajectory_metrics']
        for stage in stages:
            vals = [metrics[key][stage]['mean'] for key in ('mae', 'rmse', 'temporal_pcc')]
            lines.append(f'| {name} | {stage} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} |')
    lines.extend(['', 'Std Ratio、Delta PCC、Delta MAE、PSD Shape PCC、Log-PSD MAE、FC MSE和FC Pattern PCC见表格。',
                  '轨迹样本按固定规则选择；预测边界为0秒。四类图件仅保存PDF。', ''])
    (out/'说明'/'模型比较.md').write_text('\n'.join(lines))
    return reports


def write_experiment_summary(experiment_dir, out_dir, split='test'):
    """汇总已完成的HC/MDD实验，不重新推理、选模或拟合参数。"""
    from train.training_state import read_training_state_metadata
    root, out = Path(experiment_dir), Path(out_dir)
    for name in ('表格', '说明'):
        (out/name).mkdir(parents=True, exist_ok=True)
    stages = ('H1_8', 'H9_16', 'H17_32', 'H1_32')
    dynamic_names = ('std_ratio', 'delta_pcc', 'delta_mae', 'psd_shape_pcc',
                     'log_psd_mae', 'fc_mse', 'fc_pcc')
    stage_rows, dynamic_rows, horizon_rows, training_rows, sources = [], [], [], [], []
    for cohort, old_name, new_name in (('HC', '原B健康评价', '新模型健康评价'),
                                      ('MDD', '原B抑郁症评价', '新模型抑郁症评价')):
        original, new = [load_eval_artifacts(root/name, split) for name in (old_name, new_name)]
        a, b = original['metrics'], new['metrics']
        if a['protocol'].get('multiscale_phase') or not b['protocol'].get('multiscale_phase'):
            raise ValueError(f'{cohort}需要原B与多尺度相位模型的对应评价')
        for key in ('split_hash', 'mode', 'horizon', 'context_length', 'tr_seconds',
                    'calibration_length', 'normalization', 'n_subjects', 'n_anchors'):
            if a['protocol'].get(key) is None or a['protocol'][key] != b['protocol'].get(key):
                raise ValueError(f'{cohort}比较协议缺少或不匹配: {key}')
        for stage in stages:
            row = {'队列': cohort, '预测范围': stage}
            for key in ('mae', 'rmse', 'temporal_pcc'):
                old, value = [r['trajectory_metrics'][key][stage]['mean'] for r in (a, b)]
                row[f'原B_{key}'], row[f'新模型_{key}'] = old, value
                row[f'{key}_变化'] = value-old
                if key != 'temporal_pcc':
                    row[f'{key}_下降百分比'] = 100*(old-value)/old
            stage_rows.append(row)
        for key in dynamic_names:
            old, value = [r['trajectory_metrics'][key]['H1_32']['mean'] for r in (a, b)]
            dynamic_rows.append({'队列': cohort, '指标': key, '原B': old, '新模型': value, '变化': value-old})
        merged = original['逐时距指标'].merge(new['逐时距指标'], on='预测范围', suffixes=('_原B', '_新模型'), validate='one_to_one')
        expected = {f'H{i}' for i in range(1, 33)}
        if len(merged) != 32 or set(merged['预测范围']) != expected:
            raise ValueError(f'{cohort}逐时距表必须完整覆盖H1–32')
        for _, pair in merged.iterrows():
            row = {'队列': cohort, '预测范围': pair['预测范围']}
            for key in ('mae', 'rmse', 'spatial_pcc'):
                row[f'原B_{key}'], row[f'新模型_{key}'] = pair[f'{key}_原B'], pair[f'{key}_新模型']
                row[f'{key}_变化'] = pair[f'{key}_新模型']-pair[f'{key}_原B']
            horizon_rows.append(row)
        checkpoint = Path(b['protocol']['checkpoint']['path'])
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[1]/checkpoint
        run = checkpoint.parent
        history = [json.loads(line) for line in (run/'验证训练记录.jsonl').read_text().splitlines()]
        best = min(history, key=lambda row: row['val']['metric_selection_h32'])
        progress = read_training_state_metadata(run/'training_state.pt')['progress']
        training_rows.append({'队列': cohort, '完成轮数': progress['epoch'], 'best轮数': best['epoch'],
                              '早停': progress['early_stopped'], '末轮主干已解冻': progress['backbone_unfrozen'],
                              '测试被试数': b['protocol']['n_subjects'], '测试锚点数': b['protocol']['n_anchors'],
                              '权重路径': b['protocol']['checkpoint']['path'],
                              '权重SHA256': b['protocol']['checkpoint']['sha256']})
        sources.extend([(f'{cohort}原B测试', root/old_name/f'指标_{split_label(split)}.json'),
                        (f'{cohort}新模型测试', root/new_name/f'指标_{split_label(split)}.json'),
                        (f'{cohort}训练验证记录', run/'验证训练记录.jsonl')])
    frames = dict(分阶段主指标汇总=pd.DataFrame(stage_rows), 动态保真指标汇总=pd.DataFrame(dynamic_rows),
                  逐时距变化汇总=pd.DataFrame(horizon_rows), 训练与权重汇总=pd.DataFrame(training_rows))
    for name, frame in frames.items():
        frame.to_csv(out/'表格'/f'{name}.csv', index=False, encoding='utf-8-sig')
    full = {r['队列']: r for r in stage_rows if r['预测范围'] == 'H1_32'}
    short = {r['队列']: r for r in stage_rows if r['预测范围'] == 'H1_8'}
    lines = ['# 多尺度相位预测实施结果总结', '', '## 结论摘要', '',
             f"同协议测试中，HC、MDD完整H32的MAE分别下降{full['HC']['mae_下降百分比']:.2f}%、"
             f"{full['MDD']['mae_下降百分比']:.2f}%，Temporal PCC分别增加"
             f"{full['HC']['temporal_pcc_变化']:.4f}、{full['MDD']['temporal_pcc_变化']:.4f}。",
             f"收益主要在H1–8：MAE分别下降{short['HC']['mae_下降百分比']:.2f}%、"
             f"{short['MDD']['mae_下降百分比']:.2f}%。H17–32改善较小，振幅收缩尚未解决。", '',
             '本报告汇总已完成的固定配置实验，没有使用测试集追加调参、重选权重或启动新训练。', '',
             '## 实际训练与选模', '',
             '| 队列 | 完成轮数 | best轮数 | 早停 | 末轮已解冻 | 测试被试/锚点 |',
             '|---|---:|---:|---|---|---|']
    for row in training_rows:
        lines.append(f"| {row['队列']} | {row['完成轮数']} | {row['best轮数']} | {row['早停']} | "
                     f"{row['末轮主干已解冻']} | {row['测试被试数']}/{row['测试锚点数']} |")
    lines.extend(['', '两阶段均在第11轮解冻并继续联合训练；最终best均来自第10轮冻结阶段末，'
                  '解冻后的权重没有获得更好的验证选模分数。保存并测试的是EMA best，未用last替代。', '',
                  '## 分阶段测试结果', '', '数值为原B→新模型；时间PCC逐ROI沿阶段时间轴计算。', '',
                  '| 队列 | 范围 | MAE | RMSE | Temporal PCC | MAE下降 |',
                  '|---|---|---:|---:|---:|---:|'])
    for row in stage_rows:
        values = [f"{row['原B_'+key]:.6f} → {row['新模型_'+key]:.6f}" for key in ('mae', 'rmse', 'temporal_pcc')]
        lines.append(f"| {row['队列']} | {row['预测范围']} | {' | '.join(values)} | {row['mae_下降百分比']:.2f}% |")
    lines.extend(['', '## 逐时距结果并非每点都改善', '',
                  '| 队列 | MAE改善时距数 | RMSE改善时距数 | Spatial PCC改善时距数 | MAE退化时距 |',
                  '|---|---:|---:|---:|---|'])
    for cohort, sub in frames['逐时距变化汇总'].groupby('队列'):
        worse = sub[sub['mae_变化'] > 0]
        lines.append(f"| {cohort} | {(sub['mae_变化']<0).sum()}/32 | {(sub['rmse_变化']<0).sum()}/32 | "
                     f"{(sub['spatial_pcc_变化']>0).sum()}/32 | {', '.join(worse['预测范围']) or '无'} |")
    lines.extend(['', '单horizon的PCC是沿ROI计算的Spatial PCC，不能称为Temporal PCC。'
                  'MDD部分远期点MAE略有退化，因此不能把阶段平均改善表述为H1–32每个点均改善。', '',
                  '## 完整H32动态保真', '', '| 队列 | 指标 | 原B | 新模型 | 变化 |', '|---|---|---:|---:|---:|'])
    for row in dynamic_rows:
        lines.append(f"| {row['队列']} | {row['指标']} | {row['原B']:.6f} | {row['新模型']:.6f} | {row['变化']:+.6f} |")
    lines.extend(['', 'Std Ratio以接近1为理想；两队列的新模型Std Ratio略低，幅值收缩没有缓解。'
                  'Delta PCC/MAE与FC指标有所改善，PSD Shape PCC变化很小且MDD略降。'
                  '频谱相似性不能代替真实轨迹对齐，Temporal/Delta PCC也不直接等于专门的相位误差。', '',
                  '## 解释边界', '',
                  '本轮同时改变结构、损失、增强和训练过程，未做同预算B续训对照或模块消融；'
                  '结果属于整套方案，不能单独归因于相位分支。只完成一组固定随机种子，未声称统计显著性或跨随机种子稳定收益。'
                  '现有结果支持短程改善，但未证明解决长程相位外推和振幅收缩。', '',
                  '## 结构与图表', '',
                  '[当前模型结构](../../../../docs/复现说明/多尺度相位模型结构.md)；'
                  '[实施与运行命令](../../../../docs/复现说明/多尺度相位预测实施说明.md)。', '',
                  '[HC完整图表](../健康模型比较/说明/模型比较.md)；[MDD完整图表](../抑郁症模型比较/说明/模型比较.md)。'
                  '每个队列包含逐时距、阶段时间相关性、固定样本轨迹、动态/频谱四张PDF；原始表格与图片分开保存。', '',
                  '## 来源与口径', '',
                  '数据先按ROI和锚点计算，在被试内平均，最后跨被试等权。'
                  '模型对比使用相同被试、锚点、校准历史、未来真值及mask；训练验证只用于选模，test用于最终评价。', ''])
    for i, (label, path) in enumerate(sources, 1):
        relative = Path(os.path.relpath(path.resolve(), (out/'说明').resolve())).as_posix()
        lines.append(f'{i}. [{label}]({relative})')
    (out/'说明'/'实施结果总结.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--eval_dir')
    source.add_argument('--experiment_dir', help='已完成的HC/MDD实验目录，汇总结果而不重新推理')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', choices=['val', 'test', 'train'], default='test')
    parser.add_argument('--original_eval', default='', help='原B同协议评价目录；提供时生成四类对比PDF，两个评价均须--save_arrays')
    args = parser.parse_args()
    if args.experiment_dir:
        write_experiment_summary(args.experiment_dir, args.out_dir, args.split)
    elif args.original_eval:
        write_comparison(args.original_eval, args.eval_dir, args.out_dir, args.split)
    else:
        write_report(args.eval_dir, args.out_dir, args.split)


if __name__ == '__main__':
    main()
