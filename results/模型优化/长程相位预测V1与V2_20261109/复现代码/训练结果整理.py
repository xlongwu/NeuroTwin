"""只读汇总已完成的V0/V1/V2验证记录与best权重，不训练、不评价测试集。

仓库根目录运行：python results/模型优化/长程相位预测V1与V2_20261109/复现代码/训练结果整理.py
"""
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurotwin-result-matplotlib')

import numpy as np
import pandas as pd
import torch

from model import load_model_checkpoint, verify_checkpoint_contract, load_model_config
from train.optim import read_checkpoint
from train.training_state import read_training_state_metadata

VERSIONS = {'V0': '多尺度相位预测_20261107', 'V1': '长程相位预测_V1_20261108',
            'V2': '多尺度相位协同预测方案_1108'}
CONFIGS = {'V0': 'configs/neurotwin_multiscale_phase.json',
           **{v: f'configs/neurotwin_multiscale_phase_{v.lower()}.json' for v in ('V1', 'V2')}}
COHORTS = {'HC': 'pretrain', 'MDD': 'finetune'}
STAGES = ('h1_8', 'h9_16', 'h17_32', 'h1_32')
LABELS = ('H1–8', 'H9–16', 'H17–32', 'H1–32')
DYNAMIC = {'std_ratio': 'Std Ratio', 'delta_pcc': 'Delta PCC', 'delta_mae': 'Delta MAE',
           'psd_shape_pcc': 'PSD Shape PCC', 'log_psd_mae': 'Log-PSD MAE',
           'fc_mse': 'FC MSE', 'fc_pcc': 'FC Pattern PCC'}
COLORS = {'V0': '#6592ac', 'V1': '#2a8075', 'V2': '#c47739'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(frame: pd.DataFrame) -> list[str]:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.6f}'
        return str(value)
    return ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '|' + '|'.join('---' for _ in frame.columns) + '|'] + [
            '| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]


def collect():
    torch.set_num_threads(1)
    torch.manual_seed(2024)
    pipeline_path = ROOT/'checkpoints/长程相位预测_20261108_运行记录/训练状态.json'
    pipeline = json.loads(pipeline_path.read_text())
    if pipeline['status'] != 'complete' or len(pipeline['stages']) != 4:
        raise ValueError('四阶段训练尚未完成')
    histories, bests, manifests = {}, {}, []
    training, stages, dynamics, horizons, curves, budgets, pairs = [], [], [], [], [], [], []
    for cohort, mode in COHORTS.items():
        protocol, inheritance = None, {}
        for version, directory in VERSIONS.items():
            run = ROOT/'checkpoints'/directory/f'neurotwin_{mode}'
            log = run/'验证训练记录.jsonl'
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            if [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
                raise ValueError(f'轮次不连续: {run}')
            best = min(rows, key=lambda r: r['val']['metric_selection_h32'])
            weight = run/('base_best.pt' if mode == 'pretrain' else 'finetuned_best.pt')
            raw = read_checkpoint(weight)
            meta = raw['config']
            progress = read_training_state_metadata(run/'training_state.pt')['progress']
            if meta['epoch_1based'] != best['epoch'] or meta['val_metrics'] != best['val']:
                raise ValueError(f'best权重与完整日志选模结果不一致: {weight}')
            if progress['epoch'] != len(rows) or not progress['backbone_unfrozen']:
                raise ValueError(f'训练状态与完成轮数不一致: {run}')
            cfg = load_model_config(ROOT/CONFIGS[version])
            if sha(weight) != cfg['weights'][mode]['sha256']:
                raise ValueError(f'已登记权重哈希不一致: {weight}')
            model, args, bundle = load_model_checkpoint(weight, 'cpu')
            verify_checkpoint_contract(args, cfg)
            with torch.no_grad():
                prediction, _ = model(torch.randn(1, 116, 1, 64))
            if prediction.shape != (1, 116, 32) or not torch.isfinite(prediction).all():
                raise ValueError(f'严格加载后最小前向失败: {weight}')
            current = (bundle['split_hash'], args.eval_context_length, args.eval_anchors_per_subject,
                       args.calibration_length, best['val']['metric_mae_h1_32_n_subjects'])
            if protocol is not None and protocol != current:
                raise ValueError(f'验证集比较口径不同: {cohort} {version}')
            protocol = current
            for r in rows:
                val = r['val']
                if not math.isclose(val['metric_selection_h32'], val['metric_mae_h1_32']
                                    + args.selection_corr_weight*(1-val['metric_temporal_pcc_h1_32']), abs_tol=1e-12):
                    raise ValueError('选模指标计算不一致')
                if version != 'V0' and (r['val'] != r['val_ema'] or not r['val_raw']):
                    raise ValueError('缺少双权重验证记录')
                curves.append({'队列': cohort, '版本': version, '轮数': r['epoch'],
                               **{f'EMA_{k}': val[f'metric_{k}'] for k in
                                  ('selection_h32', 'mae_h1_32', 'temporal_pcc_h1_32')},
                               **{f'原始_{k}': r.get('val_raw', {}).get(f'metric_{k}') for k in
                                  ('selection_h32', 'mae_h1_32', 'temporal_pcc_h1_32')}})
            state = next((s for s in pipeline['stages'] if s['version'] == int(version[-1])
                          and s['mode'] == mode), None) if version != 'V0' else None
            minutes = ((datetime.fromisoformat(state['finished_at'])
                        - datetime.fromisoformat(state['started_at'])).total_seconds()/60) if state else None
            if state and (state['sha256'] != sha(weight) or state['epochs'] != len(rows) or state['exit_code'] != 0):
                raise ValueError('流水线退出状态或权重记录不匹配')
            training.append({'队列': cohort, '版本': version, '完成轮数': len(rows), 'best轮数': best['epoch'],
                             '早停': progress['early_stopped'], '验证被试数': current[-1],
                             'split_hash': current[0], '参数量': sum(p.numel() for p in model.parameters()),
                             '耗时分钟': minutes, '权重路径': str(weight.relative_to(ROOT)), 'SHA256': sha(weight)})
            inheritance[version] = meta['initialization']
            histories[cohort, version], bests[cohort, version] = rows, best
            manifests.append({'队列': cohort, '版本': version, '权重': str(weight.relative_to(ROOT)),
                              '权重SHA256': sha(weight), '训练记录': str(log.relative_to(ROOT)),
                              '训练记录SHA256': sha(log), 'best轮数': best['epoch'],
                              '完成轮数': len(rows), 'split_hash': current[0],
                              '严格加载与CPU前向': True})
            for stage, label in zip(STAGES, LABELS):
                stages.append({'队列': cohort, '版本': version, '预测范围': label,
                               **{k: best['val'][f'metric_{k}_{stage}'] for k in ('mae', 'rmse', 'temporal_pcc')}})
                dynamics.append({'队列': cohort, '版本': version, '预测范围': label,
                                 **{k: best['val'][f'metric_{k}_{stage}'] for k in DYNAMIC}})
            for h in range(1, 33):
                horizons.append({'队列': cohort, '版本': version, '时距': h,
                                 **{k: best['val'][f'metric_{k}_h{h}'] for k in ('mae', 'rmse', 'spatial_pcc')}})
            for budget in (10, 22):
                selected = min((r for r in rows if r['epoch'] <= budget),
                               key=lambda r: r['val']['metric_selection_h32'])
                budgets.append({'队列': cohort, '版本': version, '训练预算轮数': budget,
                                '预算内best轮数': selected['epoch'],
                                **{k: selected['val'][f'metric_{k}'] for k in
                                   ('mae_h1_32', 'temporal_pcc_h1_32', 'selection_h32')}})
        if mode == 'pretrain':
            if len({inheritance[v]['source_sha256'] for v in VERSIONS}) != 1:
                raise ValueError('三个HC实验的B初始化权重不同')
    for cohort in COHORTS:
        for old, new in (('V0', 'V1'), ('V0', 'V2'), ('V1', 'V2')):
            a, b = bests[cohort, old]['val'], bests[cohort, new]['val']
            for stage, label in zip(STAGES, LABELS):
                x, y = a[f'metric_mae_{stage}'], b[f'metric_mae_{stage}']
                pairs.append({'队列': cohort, '对比': f'{new}相对{old}', '预测范围': label,
                              '基准MAE': x, '新MAE': y, 'MAE下降百分比': 100*(x-y)/x,
                              'Temporal PCC变化': b[f'metric_temporal_pcc_{stage}']-a[f'metric_temporal_pcc_{stage}']})
    frames = dict(训练与权重汇总=pd.DataFrame(training), 分阶段主指标=pd.DataFrame(stages),
                  动态指标=pd.DataFrame(dynamics), 逐时距主指标=pd.DataFrame(horizons),
                  验证曲线=pd.DataFrame(curves), 预算内验证最佳=pd.DataFrame(budgets),
                  分阶段改善率=pd.DataFrame(pairs))
    for name, frame in frames.items():
        frame.to_csv(OUT/'表格'/f'{name}.csv', index=False, encoding='utf-8-sig')
    evidence = {'整理时间': datetime.now().astimezone().isoformat(timespec='seconds'),
                '训练结束时间': pipeline['finished_at'], '评价范围': '已有验证日志；未运行测试集',
                '流水线状态': pipeline['status'], '权重与日志核对': manifests,
                '比较口径': '相同split_hash、K64、每被试8锚点、固定32TR校准；先锚点再被试等权',
                '显著性检验': '未执行：仅有单个种子和汇总验证日志，不能从中推导置信区间',
                '运行命令': 'python '+str(Path(__file__).relative_to(ROOT))}
    (OUT/'说明/核对记录.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    return histories, bests, frames, pipeline


def draw(histories, bests):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = [f.name for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if not fonts:
        raise RuntimeError('缺少可用中文字体')
    plt.rcParams.update({'font.family': fonts[0], 'pdf.fonttype': 3, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    def save(fig, name):
        fig.savefig(OUT/'图片'/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), layout='constrained')
    for row, cohort in enumerate(COHORTS):
        for col, (metric, label) in enumerate([('selection_h32', '选模分数（越低越好）'),
                                               ('mae_h1_32', '完整H32 MAE'),
                                               ('temporal_pcc_h1_32', '完整H32 Temporal PCC')]):
            ax = axes[row, col]
            for version in VERSIONS:
                rs = histories[cohort, version]
                ax.plot([r['epoch'] for r in rs], [r['val'][f'metric_{metric}'] for r in rs],
                        color=COLORS[version], label=version+' EMA', lw=1.6)
                if version != 'V0':
                    ax.plot([r['epoch'] for r in rs], [r['val_raw'][f'metric_{metric}'] for r in rs],
                            color=COLORS[version], ls=':', alpha=.45, lw=.8)
                b = bests[cohort, version]
                ax.scatter(b['epoch'], b['val'][f'metric_{metric}'], marker='*', s=65,
                           color=COLORS[version], zorder=5)
            ax.axvline(10.5, color='#777777', ls='--', lw=.8)
            ax.set(title=f'{cohort}｜{label}', xlabel='训练轮数', ylabel=label)
            if col == 0:
                ax.legend(fontsize=8)
    fig.suptitle('训练验证过程｜实线：EMA，点线：原始权重；星号：验证选出的best；虚线：解冻边界')
    save(fig, '训练与双权重验证曲线')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout='constrained')
    for row, cohort in enumerate(COHORTS):
        base = bests[cohort, 'V0']['val']
        for i, version in enumerate(('V1', 'V2')):
            metrics = bests[cohort, version]['val']
            changes = [100*(base[f'metric_mae_{s}']-metrics[f'metric_mae_{s}'])/base[f'metric_mae_{s}'] for s in STAGES]
            bars = axes[row, 0].bar(np.arange(4)+(i-.5)*.36, changes, width=.36, color=COLORS[version], label=version)
            axes[row, 0].bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
            axes[row, 1].plot(LABELS, [metrics[f'metric_temporal_pcc_{s}']-base[f'metric_temporal_pcc_{s}'] for s in STAGES],
                              marker='o', color=COLORS[version], label=version)
        for col in range(2):
            axes[row, col].axhline(0, color='#888888', lw=.7)
            axes[row, col].set_title(cohort)
            axes[row, col].legend(fontsize=8)
        axes[row, 0].set(xticks=range(4), xticklabels=LABELS, ylabel='相对V0的MAE下降（%）')
        axes[row, 0].margins(y=.17)
        axes[row, 1].set_ylabel('相对V0的Temporal PCC增量')
    fig.suptitle('分阶段验证改善｜收益集中在短程；不同指标分开显示')
    save(fig, '分阶段验证改善')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout='constrained')
    h = np.arange(1, 33)
    for row, cohort in enumerate(COHORTS):
        vals = {v: np.array([bests[cohort, v]['val'][f'metric_mae_h{i}'] for i in h]) for v in VERSIONS}
        for v in ('V1', 'V2'):
            axes[row, 0].plot(h, 100*(vals['V0']-vals[v])/vals['V0'], color=COLORS[v], label=v)
        change = 100*(vals['V1']-vals['V2'])/vals['V1']
        axes[row, 1].plot(h, change, color=COLORS['V2'], label='V2相对V1')
        axes[row, 1].scatter(h[change < 0], change[change < 0], color='#b44c4c', s=25, label='MAE退化')
        for col in range(2):
            axes[row, col].axhline(0, color='#777777', lw=.8)
            axes[row, col].set(title=cohort, xlabel='预测时距（TR）', xticks=[1, 8, 16, 24, 32])
            axes[row, col].legend(fontsize=8)
        axes[row, 0].set_ylabel('相对V0的MAE下降（%）')
        axes[row, 1].set_ylabel('V2相对V1的MAE下降（%）')
    fig.suptitle('逐时距验证改善｜正数为改善，负数为退化；两列纵轴量级不同')
    save(fig, '逐时距验证改善与退化')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for row, cohort in enumerate(COHORTS):
        for col, metric in enumerate(('std_ratio', 'delta_pcc', 'psd_shape_pcc')):
            for version in VERSIONS:
                axes[row, col].plot(LABELS, [bests[cohort, version]['val'][f'metric_{metric}_{s}'] for s in STAGES],
                                    marker='o', color=COLORS[version], label=version)
            axes[row, col].set(title=f'{cohort}｜{DYNAMIC[metric]}', ylabel=DYNAMIC[metric])
            if col == 0:
                axes[row, col].legend(fontsize=8)
    fig.suptitle('动态与频谱验证指标｜Std Ratio仅为辅助指标，不要求强制达到1')
    save(fig, '动态与频谱验证比较')


def report(histories, bests, frames, pipeline):
    def pair(cohort, comparison, stage):
        f = frames['分阶段改善率']
        return f[(f['队列'] == cohort) & (f['对比'] == comparison) & (f['预测范围'] == stage)].iloc[0]
    lines = ['# 长程相位预测 V1/V2 训练结果总结', '',
             '整理日期：2026-11-09。四阶段于 '+pipeline['finished_at']+' 完成。', '',
             '## 结论摘要', '',
             '本报告只整理已有验证集记录，不重新训练、推理或评价测试集。三个版本采用相同被试划分、K64/H32、校准及锚点口径；HC验证222人，MDD验证225人。V0为已训练多尺度相位模型，本文不把其历史测试指标混入验证比较。', '']
    for cohort in COHORTS:
        p1, p2, extra = [pair(cohort, c, 'H1–32') for c in ('V1相对V0', 'V2相对V0', 'V2相对V1')]
        long = pair(cohort, 'V2相对V0', 'H17–32')
        lines.append(f'- {cohort}：V1完整H32 MAE较V0下降{p1["MAE下降百分比"]:.2f}%，V2下降{p2["MAE下降百分比"]:.2f}%；V2较同预算V1再下降{extra["MAE下降百分比"]:.3f}%。但V2的H17–32 MAE较V0仅下降{long["MAE下降百分比"]:.2f}%，远期瓶颈仍在。')
    lines += ['', 'V1相较V0的主要变化是训练方式与动态通路初始化；V2在V1设置上增加局部—全局频率状态和未来时距交互。V1/V2均从相同B权重独立开始HC训练，各自HC best再初始化各自MDD。V1/V2的全部四阶段均完成150轮，未触发早停，best在第149或150轮；本轮没有据此追加训练预算。', '',
              '## 完整 H32 验证结果', '']
    full = frames['分阶段主指标'].query('预测范围 == "H1–32"')
    lines += markdown(full[['队列', '版本', 'mae', 'rmse', 'temporal_pcc']])
    lines += ['', '选模使用完整H32 MAE加0.15倍的时间相关性损失，选择EMA best；不根据分阶段、逐时距或测试结果另行挑选checkpoint。', '',
              '## 短、中、长程分别改善多少', '',
              '下表百分比为MAE相对下降，正数表示改善。阶段Temporal PCC完整数值见独立CSV。', '']
    merged = []
    for cohort in COHORTS:
        for stage in LABELS:
            merged.append({'队列': cohort, '预测范围': stage,
                           **{c+'（%）': pair(cohort, c, stage)['MAE下降百分比']
                              for c in ('V1相对V0', 'V2相对V0', 'V2相对V1')}})
    lines += markdown(pd.DataFrame(merged))
    lines += ['', '[分阶段验证改善（PDF）](../图片/分阶段验证改善.pdf)', '',
              '## 动态幅值与频谱表现', '',
              'Std Ratio在完整H32上的回升与Delta PCC提升同时出现，说明验证集动态保真有所改善，但远期幅值依然明显收缩；不能把Std Ratio趋近1作为独立优化目标。', '']
    dyn = []
    for cohort in COHORTS:
        for metric, name in DYNAMIC.items():
            dyn.append({'队列': cohort, '指标': name, **{v: bests[cohort, v]['val'][f'metric_{metric}_h1_32'] for v in VERSIONS}})
    lines += markdown(pd.DataFrame(dyn))
    lines += ['', 'V2并非所有辅助指标都更好：完整H32的PSD Shape PCC在HC/MDD上均略低于V1，尽管Log-PSD MAE进一步降低。频谱形状与幅值误差反映不同方面，不能互相替代。', '',
              '[动态与频谱验证比较（PDF）](../图片/动态与频谱验证比较.pdf)', '',
              '## V2 的额外收益与代价', '',
              'V2的完整H32 MAE和Temporal PCC均略优于同预算V1，但增益很小。按逐时距MAE核对：', '']
    for cohort in COHORTS:
        a, b = [bests[cohort, v]['val'] for v in ('V1', 'V2')]
        worse = [h for h in range(1, 33) if b[f'metric_mae_h{h}'] > a[f'metric_mae_h{h}']]
        relative_v0 = [h for h in range(1, 33) if b[f'metric_mae_h{h}'] > bests[cohort, 'V0']['val'][f'metric_mae_h{h}']]
        lines.append(f'- {cohort}：V2相对V1退化时距为'+('、'.join('H'+str(h) for h in worse) if worse else '无')+f'；V2相对V0退化时距为'+('、'.join('H'+str(h) for h in relative_v0) if relative_v0 else '无')+'。')
    lines += ['', '[逐时距验证改善（PDF）](../图片/逐时距验证改善与退化.pdf)', '',
              '实际单GPU训练耗时包含数据加载、初始化、原始权重与EMA两遍验证，不是独立性能基准：', '']
    training = frames['训练与权重汇总']
    lines += markdown(training[training['版本'] != 'V0'][['队列', '版本', '完成轮数', 'best轮数', '参数量', '耗时分钟']])
    lines.append('')
    for cohort in COHORTS:
        times = {v: float(training[(training['队列'] == cohort) & (training['版本'] == v)]['耗时分钟'].iloc[0]) for v in ('V1', 'V2')}
        lines.append(f'{cohort}的V2阶段壁钟耗时较V1增加{100*(times["V2"]/times["V1"]-1):.1f}%。')
    lines += ['', '因此，V1可作为计算成本较低的后续参照；V2是验证指标略优的候选，现有证据尚不足以支持其长程能力出现实质突破。', '',
              '## 联合训练与 EMA', '',
              'V0两个阶段的best均在第10轮，22轮早停；V1/V2在解冻后继续刷新best。当前实现本就从解冻后累计patience，本轮未重复修改这一规则。', '',
              '原始权重与EMA在新模型best轮次的验证表现：', '']
    dual = []
    for cohort in COHORTS:
        for version in ('V1', 'V2'):
            b = bests[cohort, version]
            dual.append({'队列': cohort, '版本': version, 'best轮数': b['epoch'],
                         'EMA_MAE': b['val']['metric_mae_h1_32'],
                         '原始_MAE': b['val_raw']['metric_mae_h1_32'],
                         'EMA_PCC': b['val']['metric_temporal_pcc_h1_32'],
                         '原始_PCC': b['val_raw']['metric_temporal_pcc_h1_32']})
    lines += markdown(pd.DataFrame(dual))
    lines += ['', '[训练与双权重验证曲线（PDF）](../图片/训练与双权重验证曲线.pdf)', '',
              '前22轮内按相同指标选出的验证best如下。HC可用于观察相同阶段预算下的差异；MDD仍继承各自已经训练完成的HC best，不能当作全流程等预算比较。', '']
    lines += markdown(frames['预算内验证最佳'].query('训练预算轮数 == 22')[
        ['队列', '版本', '预算内best轮数', 'mae_h1_32', 'temporal_pcc_h1_32']])
    lines += ['',
              '## 比较边界与解释', '',
              '- V1/V2与V0的训练上限均为150轮，但V0实际22轮早停，而新模型均完成150轮。相对V0的提升包含额外有效训练，不能单独归因于EMA、解冻范围或初始化。前10/22轮预算内的验证best已单独导出；MDD起点来自各自HC best，因此即使MDD轮数相同，也不等于HC→MDD全流程预算相同。',
              '- V1/V2的实际阶段预算均为150轮，数据、损失、增强及B初始化相同，可用于衡量两项新增结构的整体增量；没有分别消融局部相位模块与未来时距模块，不能拆分归因。',
              '- 原始权重和EMA在best附近接近，但没有EMA-only对照，不能断言上一轮停滞由EMA滞后造成。',
              '- 仅单随机种子；现有日志是跨被试汇总，不包含每名被试的配对误差，未进行显著性检验、置信区间或稳定性验证。',
              '- 验证集参与选模，不能视为独立泛化证明；已经查看过的测试集未用于本轮选择，本次也未新增测试评价。', '',
              '## 权重与复现', '',
              'V1/V2配置已登记四份实际EMA best的epoch和SHA256。核对记录包含全部六份V0/V1/V2权重与日志哈希、划分信息、严格加载及CPU最小前向检查；原B与V0权重没有修改。', '',
              '[训练与权重汇总](../表格/训练与权重汇总.csv) · [核对记录](核对记录.json) · [复现脚本](../复现代码/训练结果整理.py)', '',
              '仓库根目录重建本报告、CSV和四张PDF：', '', '```bash',
              'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python '+str(Path(__file__).relative_to(ROOT)), '```', '',
              '严格加载已登记权重，例如：', '', '```bash',
              'python model.py --model_config configs/neurotwin_multiscale_phase_v2.json --mode finetune --device cpu', '```', '',
              '## 来源', '']
    for cohort, mode in COHORTS.items():
        for version, directory in VERSIONS.items():
            rel = '../../../../checkpoints/'+directory+f'/neurotwin_{mode}/验证训练记录.jsonl'
            lines.append(f'- [{cohort} {version}完整训练验证记录]({rel})')
    lines += ['- [四阶段完成状态与权重哈希](../../../../checkpoints/长程相位预测_20261108_运行记录/训练状态.json)', '']
    (OUT/'说明/训练结果总结.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    histories, bests, frames, pipeline = collect()
    draw(histories, bests)
    report(histories, bests, frames, pipeline)
    print('已完成：7份CSV、4张PDF、结果总结和核对记录；未运行训练或测试集评价。')
