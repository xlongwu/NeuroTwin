"""Analyze actual E2 artifacts without altering checkpoints, data or evaluator code.

Run from repository root: PYTHONPATH=. python <this script>. Figures follow the plain
multi-curve panel structure of gallery reference oa-tech-pone-0116670-f16;
data, labels and conclusions are entirely from the current NeuroTwin run.
"""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

from utils.dataloader import read_mat_array


RUN = Path('checkpoints/h32_spectral/E2')
OUT = Path(__file__).resolve().parents[1]
EVAL = RUN / 'neurotwin_finetune/eval'
report = json.loads((EVAL / '指标_测试集.json').read_text())
tasks = pd.read_csv(EVAL / '逐任务指标_测试集.csv')
subjects = pd.read_csv(EVAL / '逐被试指标_测试集.csv')
stages = pd.read_csv(EVAL / '长程分阶段指标_测试集.csv')
split = json.loads(Path('data/subject_split_finetune.json').read_text())
best = torch.load(RUN / 'neurotwin_finetune/finetuned_best.pt', map_location='cpu', weights_only=False)
args = best['config']['args']
assert set(tasks.subj_id) == set(subjects.subject_id) == set(split['test'])
assert tasks.groupby('subj_id').size().eq(8).all()
assert not tasks.duplicated(['subj_id', 'cutoff']).any()
assert len(subjects) == report['protocol']['n_subjects'] == 225
assert not stages.duplicated(['被试', '预测阶段']).any()

# Reproduce exactly the Raw calibration used by utils/dataloader.py.
# Only past context enters the history-mean baseline; future is used for scoring.
history_rows, raw_sources = [], []
for sid, group in tasks.groupby('subj_id', sort=True):
    raw_path = Path(args['data_root']) / 'Raw/MDD' / f'ROISignals_{sid}.mat'
    raw_sources.append(str(raw_path))
    arr = read_mat_array(raw_path, preferred_keys=['ROISignals', 'Data']).astype(np.float32)
    x = np.ascontiguousarray(arr[:, :args['num_rois']].T)
    cal = x[:, :args['calibration_length']]
    series = ((x - cal.mean(1, keepdims=True)) / np.sqrt(cal.var(1, keepdims=True) + 1e-6)).astype(np.float32)
    for row in group.itertuples(index=False):
        t, k = int(row.cutoff), int(row.context_len)
        future = series[:, t + 1:t + 33]
        assert future.shape == (116, 32) and t - k + 1 >= 0
        pred = series[:, t - k + 1:t + 1].mean(-1, keepdims=True)
        errors = np.abs(future - pred).mean(0)
        record = {'被试': sid, '锚点': t, '持久性H1误差': float(np.abs(future[:, 0] - series[:, t]).mean())}
        record.update({f'H{h + 1}': float(value) for h, value in enumerate(errors)})
        history_rows.append(record)
history_tasks = pd.DataFrame(history_rows)
history_subjects = history_tasks.groupby('被试').mean(numeric_only=True)
assert np.isclose(history_subjects['持久性H1误差'].mean(), report['baselines']['persistence']['t+1']['MAE'], atol=1e-6)
history_tasks.to_csv(OUT / '表格/历史均值基线逐锚点误差.csv', index=False)

per_h = []
for h in range(1, 33):
    ode = tasks.groupby('subj_id')[f'ODE_H{h}_mae'].mean().mean()
    rollout = report['rollout']['model'][f'H{h}']
    assert np.isclose(ode, rollout['MAE'], atol=2e-6)
    per_h.append({'预测步': h, 'E2_MAE': ode,
                  'AR1_MAE': report['rollout']['baselines']['ar1'][f'H{h}']['MAE'],
                  '历史均值_MAE': history_subjects[f'H{h}'].mean(),
                  '持久性_MAE': report['rollout']['baselines']['persistence'][f'H{h}']['MAE'],
                  'E2_空间PCC': rollout['PCC_spatial']})
per_h = pd.DataFrame(per_h)
per_h.to_csv(OUT / '表格/逐步预测与基线对照.csv', index=False)

stage_rows, paired_rows = [], []
for label, start, end in [('H1', 1, 1), ('H1_8', 1, 8), ('H9_16', 9, 16), ('H17_32', 17, 32), ('H1_32', 1, 32)]:
    part = per_h[(per_h['预测步'] >= start) & (per_h['预测步'] <= end)]
    row = {'阶段': label, **{c: part[c].mean() for c in ('E2_MAE', 'AR1_MAE', '历史均值_MAE', '持久性_MAE')}}
    for metric, values in report['long_horizon']['stages'].items():
        if label in values:
            row[metric] = values[label]['mean']
            assert np.isclose(stages.loc[stages['预测阶段'] == label, metric].mean(), row[metric], atol=1e-6)
    assert np.isclose(row['E2_MAE'], row['mae'], atol=1e-6)
    row['相对AR1改善百分比'] = 100 * (row['AR1_MAE'] - row['E2_MAE']) / row['AR1_MAE']
    row['相对历史均值改善百分比'] = 100 * (row['历史均值_MAE'] - row['E2_MAE']) / row['历史均值_MAE']
    stage_rows.append(row)
    model_subject = tasks.groupby('subj_id')[[f'ODE_H{h}_mae' for h in range(start, end + 1)]].mean().mean(axis=1)
    baseline_subject = history_subjects[[f'H{h}' for h in range(start, end + 1)]].mean(axis=1)
    for sid in model_subject.index:
        paired_rows.append({'被试': sid, '阶段': label, 'E2_MAE': model_subject[sid],
                            '历史均值_MAE': baseline_subject[sid],
                            '模型减基线MAE': model_subject[sid] - baseline_subject[sid]})
stage_table = pd.DataFrame(stage_rows)
stage_table.to_csv(OUT / '表格/分阶段预测结果与基线对照.csv', index=False)
paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT / '表格/逐被试历史均值配对比较.csv', index=False)

curve_rows = []
number = r'([-+\d.eE]+)'
for stage, name in [('HC预训练', '预训练.log'), ('MDD微调', '微调.log')]:
    current = None
    for line in (RUN / name).read_text(errors='replace').replace('\r', '\n').splitlines():
        match = re.match(r'Epoch (\d+) \|.*Train Total: ' + number + r' \| rec: ' + number + r' \| fc: ' + number + r' \| freq: ' + number, line)
        if match:
            current = {'阶段': stage, 'epoch': int(match[1]), '训练总损失': float(match[2]),
                       '训练重建MAE': float(match[3]), '训练FC_MSE': float(match[4]), '训练频域L1': float(match[5])}
        if current is not None and line.startswith('[Val]'):
            for key, expr in [('验证MAE', r'H1–32 MAE='), ('验证FC_MSE', r'FC MSE='), ('验证总损失', r'total=')]:
                current[key] = float(re.search(expr + number, line)[1])
            curve_rows.append(current)
            current = None
curves = pd.DataFrame(curve_rows)
assert curves.groupby('阶段').size().to_dict() == {'HC预训练': 150, 'MDD微调': 57}
curves.to_csv(OUT / '表格/训练与验证曲线.csv', index=False)

# Parts borrowed from the reference: plain aligned panels, thin curves and an
# unframed legend. Distinct line styles remain readable without color.
font_name = next(f.name for f in font_manager.fontManager.ttflist if f.name.startswith('Noto Sans CJK'))
plt.rcParams.update({'font.family': font_name, 'pdf.fonttype': 42, 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
red, blue, brown, grey = '#ff2732', '#5070a6', '#6b3f41', '#777777'
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
ax = axes[0, 0]
for column, label, color, ls in [('E2_MAE', 'E2', red, '-'), ('AR1_MAE', 'AR(1)', blue, '--'),
                                  ('历史均值_MAE', '历史均值', brown, '-.'), ('持久性_MAE', '持久性', grey, ':')]:
    ax.plot(per_h['预测步'], per_h[column], color=color, ls=ls, lw=1.8, label=label)
ax.axvspan(21.5, 32.5, color=grey, alpha=.08)
ax.set(xlim=(1, 32), ylim=(0, 1.6), xlabel='预测步长 / TR', ylabel='MAE（固定前缀标准化 BOLD）', title='A  第22步起，E2误差高于AR(1)')
ax.legend(frameon=False, fontsize=9, ncol=2)
ax = axes[0, 1]
ax.plot(per_h['预测步'], per_h['E2_空间PCC'], color=red, lw=1.8)
ax.axvline(8.5, color=grey, ls=':', lw=1)
ax.axvline(16.5, color=grey, ls=':', lw=1)
ax.set(xlim=(1, 32), ylim=(0, 1.05), xlabel='预测步长 / TR', ylabel='逐时刻空间 PCC', title='B  空间预测对应随步长衰减')
ax = axes[1, 0]
selected = stage_table[stage_table['阶段'].isin(['H1_8', 'H9_16', 'H17_32'])]
idx = np.arange(3)
ax.plot(idx, selected['temporal_pcc'], color=red, marker='o', lw=1.8, label='时间 PCC')
ax.plot(idx, selected['std_ratio'], color=blue, marker='s', ls='--', lw=1.8, label='预测/真实时间标准差比')
for x, value in zip(idx, selected['temporal_pcc']):
    ax.text(x, value + .045, f'{value:.3f}', ha='center', color=red)
ax.axhline(1, color=grey, ls=':', lw=1)
ax.set(xticks=idx, xticklabels=['H1–8', 'H9–16', 'H17–32'], ylim=(0, 1.14), ylabel='指标值（无量纲）', title='C  远期动态有振幅，但与真值的时间对应很弱')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax = axes[1, 1]
fine = curves[curves['阶段'] == 'MDD微调']
ax.plot(fine.epoch, fine['验证MAE'], color=blue, lw=1.8)
ax.axvline(45, color=red, ls='--', lw=1.2, label='联合判据最优：第45轮')
ax.set(xlim=(1, 57), xlabel='微调 epoch', ylabel='验证集 H1–32 MAE', title='D  微调后期误差进入平台并轻微回升')
ax.legend(frameon=False, fontsize=9)
fig.suptitle('E2 长程预测结果诊断｜测试集225名被试、1800个固定锚点', fontsize=14)
fig.savefig(OUT / '图片/长程预测结果诊断.pdf')
fig.savefig('/tmp/NeuroTwin_E2_结果分析预览.png', dpi=135)
plt.close(fig)

far = paired[paired['阶段'] == 'H17_32']
far_dynamics = stages[stages['预测阶段'] == 'H17_32']
sources = [EVAL / '指标_测试集.json', EVAL / '逐任务指标_测试集.csv', EVAL / '长程分阶段指标_测试集.csv',
           Path('data/subject_split_finetune.json'), RUN / 'neurotwin_finetune/finetuned_best.pt']
summary = {'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
           'raw_sources': raw_sources, 'history_baseline': '每个锚点之前含锚点的K64历史均值，按训练前32TR固定mu0/s0标准化；不使用未来估计参数',
           'test_subjects': len(subjects), 'test_tasks': len(tasks),
           'best_finetune_epoch': best['config']['epoch_1based'],
           'far_model_beats_history_subjects': int((far['模型减基线MAE'] < 0).sum()),
           'far_subjects': len(far), 'far_pcc_nonpositive_subjects': int((far_dynamics.temporal_pcc <= 0).sum()),
           'far_std_ratio_below_one_subjects': int((far_dynamics.std_ratio < 1).sum()),
           'checks': ['test IDs equal split manifest', '8 unique complete anchors per subject',
                      'CSV means reconcile with JSON', 'raw persistence H1 matches report',
                      'ODE forward and state rollout MAE agree within 2e-6'],
           'limitations': ['E0/E1/E3 not trained; spectral-branch efficacy unisolated',
                           'Welch spectral branch has layout/mask defect; unavailable is not a result',
                           'protocol Normlize caveat is stale for this Raw/prefix-calibrated run',
                           'HC first six epochs used a 30-epoch LR schedule before extension to 150']}
(OUT / '分析记录.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(stage_table[['阶段', 'E2_MAE', 'AR1_MAE', '历史均值_MAE', 'temporal_pcc', 'std_ratio']].to_string(index=False))
print('far wins over history mean:', summary['far_model_beats_history_subjects'], '/', len(far))
print('output:', OUT)
