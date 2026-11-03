"""核验并比较原 H1–8 模型与个体化模型；仅评估，不修改模型或训练配置。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.checkpoint import load_checkpoint_compat
from analysis.evaluate import (build_model_from_args, evaluate_next_point,
                               load_condition_bundle, load_train_args)
from analysis.next_point_eval import build_next_point_loader
from main import condition_batch, dynamics_extras, next_point_batch


ROOT = Path(__file__).resolve().parents[1]
OLD_RESULTS = ROOT / 'results/当前模型分析结果/多步预测分析结果'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index('subject_id').sort_index()
    if not frame.index.is_unique:
        raise ValueError(f'逐被试表存在重复 ID: {path}')
    frame['ODE_H1至8_mae'] = frame[[f'ODE_H{h}_mae' for h in range(1, 9)]].mean(axis=1)
    frame['持久性_H1至8_mae'] = frame[[f'DUAL_H{h}_persistence_mae' for h in range(1, 9)]].mean(axis=1)
    return frame


def matched_tasks(old_path: Path, new_path: Path) -> dict:
    keys = ['subj_id', 'context_len', 'cutoff']
    old, new = [pd.read_csv(p).set_index(keys).sort_index() for p in (old_path, new_path)]
    if not old.index.is_unique or not old.index.equals(new.index):
        raise ValueError('新旧评估锚点不完全一致，不能直接配对比较')
    columns = [f'DUAL_H{h}_persistence_mae' for h in range(1, 9)]
    diff = float(np.max(np.abs(old[columns].to_numpy() - new[columns].to_numpy())))
    if diff > 1e-6:
        raise ValueError(f'新旧持久性基线不同，需检查输入/真值口径: {diff}')
    return {'锚点完全匹配': True, '锚点数': len(old), '持久性基线最大差': diff}


@torch.inference_mode()
def check_saved_predictions(checkpoint: Path, task_path: Path, device: torch.device) -> dict:
    """用真实权重重算首个 batch 的 H1/H8，排除缓存来源或预测头混用。"""
    args = load_train_args(checkpoint)
    args.data_root = str(ROOT / 'data')
    args.num_workers, args.eval_batch_size = 0, 8
    bundle = load_condition_bundle(checkpoint)
    loader, _, _ = build_next_point_loader(args, 'test', condition_bundle=bundle)
    model = build_model_from_args(args, device, bundle).eval()
    load_checkpoint_compat(model, str(checkpoint), device, strict=True)
    batch = next(iter(loader))
    x, _, _, _, graph, _ = next_point_batch(batch, device)
    cal, controls = dynamics_extras(batch, device)
    _, aux = model(x, graph, condition=condition_batch(batch, device),
                   calibration_bold=cal, controls=controls, with_aux=False)
    error = (aux['dynamics_pred'] - batch['future'].to(device)).abs().mean(dim=1).cpu().numpy()
    saved = pd.read_csv(task_path).set_index(['subj_id', 'context_len', 'cutoff'])
    differences = []
    for i, sid in enumerate(batch['subj_id']):
        row = saved.loc[(sid, int(batch['context_len'][i]), int(batch['cutoff'][i]))]
        differences.extend(abs(float(error[i, h-1]) - float(row[f'ODE_H{h}_mae'])) for h in (1, 8))
    maximum = max(differences)
    if maximum > 1e-3:
        raise ValueError(f'真实权重与缓存不符: {checkpoint}, H1/H8 MAE 最大差 {maximum}')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'checkpoint': str(checkpoint.relative_to(ROOT)), 'batch_size': len(error),
            'H1和H8_MAE最大差': maximum, '容差': 1e-3, '通过': True}


def paired_summary(old: pd.DataFrame, new: pd.DataFrame, cohort: str) -> tuple:
    if not old.index.equals(new.index):
        raise ValueError(f'{cohort} 新旧测试被试不一致')
    definitions = [('单步MAE', 't+1_MAE', 'down'), ('单步RMSE', 't+1_RMSE', 'down'),
                   ('单步空间PCC', 't+1_PCC_spatial', 'up'),
                   ('H1至8_MAE', 'ODE_H1至8_mae', 'down'), ('H8_MAE', 'ODE_H8_mae', 'down'),
                   ('H8_空间PCC', 'ODE_H8_spatial_pcc', 'up')]
    rng = np.random.default_rng(2024)
    indices = rng.integers(0, len(old), size=(5000, len(old)))
    rows, paired = [], pd.DataFrame({'被试': old.index, '队列': cohort})
    for label, column, direction in definitions:
        a, b = old[column].to_numpy(), new[column].to_numpy()
        delta = b - a
        lo, hi = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        rows.append({'队列': cohort, '指标': label, '方向': direction, '修改前': a.mean(),
                     '修改后': b.mean(), '差值_新减旧': delta.mean(),
                     '相对变化百分比': 100 * (b.mean() / a.mean() - 1) if direction == 'down' else np.nan,
                     '差值95CI下界': lo, '差值95CI上界': hi, '被试数': len(old),
                     '改善被试数': int((delta < 0).sum() if direction == 'down' else (delta > 0).sum())})
        paired[label + '_修改前'], paired[label + '_修改后'] = a, b
        paired[label + '_差值'] = delta
    return rows, paired


def plot_comparison(frames: dict, output: Path) -> None:
    """借鉴 oa-geo-bg-21-3305-2024-f02 的横排曲线与红棕配色；仅保存 PDF。"""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    if 'Noto Sans CJK JP' in names:
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams.update({'pdf.fonttype': 3, 'axes.unicode_minus': False, 'font.size': 10})
    colors = {'修改前': '#643c33', '修改后': '#df5252', '持久性': '#888888'}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    h = np.arange(1, 9)
    for row, (cohort, (old, new)) in enumerate(frames.items()):
        for label, frame in (('修改前', old), ('修改后', new)):
            mae = frame[[f'ODE_H{i}_mae' for i in h]].mean().to_numpy()
            pcc = frame[[f'ODE_H{i}_spatial_pcc' for i in h]].mean().to_numpy()
            axes[row, 0].plot(h, mae, 'o-', label=label, color=colors[label], ms=4)
            axes[row, 1].plot(h, pcc, 'o-', label=label, color=colors[label], ms=4)
        persistence = new[[f'DUAL_H{i}_persistence_mae' for i in h]].mean().to_numpy()
        axes[row, 0].plot(h, persistence, '--', label='持久性', color=colors['持久性'])
        axes[row, 2].scatter(old['ODE_H1至8_mae'], new['ODE_H1至8_mae'], s=13, alpha=.6,
                             color=colors['修改后'], edgecolors='none')
        bounds = [0, max(old['ODE_H1至8_mae'].max(), new['ODE_H1至8_mae'].max()) * 1.03]
        axes[row, 2].plot(bounds, bounds, '--', color='#888888', lw=1)
        axes[row, 0].set(title=cohort + '：逐时距 MAE', xlabel='预测时距 H', ylabel='MAE（越低越好）', xticks=h)
        axes[row, 1].set(title=cohort + '：逐时距空间相关', xlabel='预测时距 H', ylabel='空间 PCC（越高越好）', xticks=h)
        axes[row, 2].set(title=cohort + f'：被试配对（n={len(old)}）',
                         xlabel='修改前 H1–8 MAE', ylabel='修改后 H1–8 MAE', xlim=bounds, ylim=bounds)
        for ax in axes[row]:
            ax.spines[['top', 'right']].set_visible(False)
        axes[row, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('修改前后测试集对比：K64，8 个相同锚点，无未来真值反馈', fontsize=13)
    fig.savefig(output / '修改前后预测表现对比.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    ns = parser.parse_args()
    run, out = ns.new_run.resolve(), ns.out_dir.resolve()
    for folder in ('表格', '图件', '中间产物'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    device = torch.device(ns.device)
    torch.set_num_threads(4)
    config = read_json(ROOT / 'configs/neurotwin.json')
    provenance = {'旧结果已与权重SHA256核对': True, '实际权重预测核验': [], '队列核验': {},
                  'bootstrap': '被试级配对重采样5000次，numpy.default_rng(2024)，差值均值95%百分位区间',
                  '比较限制': '两套完整配方对比；结构、损失、训练预算和MDD条件/冻结策略同时变化，不能归因于单个模块'}
    old_files, new_files = {}, {}
    for cohort, mode, folder in (('HC', 'pretrain', '健康对照'), ('MDD', 'finetune', '抑郁症')):
        old_ckpt = ROOT / 'checkpoints/1031' / ('neurotwin_' + mode) / ('base_best.pt' if mode == 'pretrain' else 'finetuned_best.pt')
        expected = config['weights'][mode]['sha256']
        if sha256(old_ckpt) != expected:
            raise ValueError(f'{cohort} 旧权重与正式基线SHA256不符')
        metadata = read_json(OLD_RESULTS / '中间产物/H1-8当前结果/评估汇总/指标总表.json')['meta']
        if metadata['checkpoints'][cohort + '_sha256'] != expected:
            raise ValueError('历史评估记录与旧权重不一致')
        old_files[cohort] = (OLD_RESULTS / '中间产物/H1-8当前结果' / folder / '指标_测试集.json',
                             OLD_RESULTS / '表格/H1-8当前结果' / folder)
        new_ckpt = run / ('neurotwin_B2_pretrain/base_best.pt' if mode == 'pretrain' else 'neurotwin_B3_finetune/finetuned_best.pt')
        new_dir = out / '中间产物/新模型HC测试' if mode == 'pretrain' else run / 'neurotwin_B3_finetune/eval'
        if mode == 'pretrain':
            args = load_train_args(new_ckpt)
            args.data_root, args.num_workers, args.eval_batch_size = str(ROOT / 'data'), 0, 64
            evaluate_next_point(str(new_ckpt), str(new_dir), args, device=device,
                                eval_rollout=False, eval_fc=False, eval_spectral=False)
        new_files[cohort] = (new_dir / '指标_测试集.json', new_dir)
        # MDD 两个权重都核验；HC 新权重已完整重新评估。
        if cohort == 'MDD':
            for ckpt, table in ((old_ckpt, old_files[cohort][1]), (new_ckpt, new_dir)):
                provenance['实际权重预测核验'].append(check_saved_predictions(
                    ckpt, table / '逐任务指标_测试集.csv', device))
        old_meta = torch.load(old_ckpt, map_location='cpu', weights_only=False)['config']
        new_meta = torch.load(new_ckpt, map_location='cpu', weights_only=False)['config']
        if old_meta['condition']['split_hash'] != new_meta['condition']['split_hash']:
            raise ValueError(f'{cohort} 训练/验证/测试划分哈希不同')
        provenance[cohort + '模型'] = {
            '旧权重': str(old_ckpt.relative_to(ROOT)), '旧SHA256': expected,
            '新权重': str(new_ckpt.relative_to(ROOT)), '新SHA256': sha256(new_ckpt),
            '旧最佳轮次': old_meta.get('epoch_1based'), '新最佳轮次': new_meta.get('epoch_1based'),
            'split_hash': new_meta['condition']['split_hash']}
    summaries, subjects, horizons, frames = [], [], [], {}
    for cohort in ('HC', 'MDD'):
        old_json, old_dir = old_files[cohort]
        new_json, new_dir = new_files[cohort]
        a, b = [subject_metrics(p / '逐被试指标_测试集.csv') for p in (old_dir, new_dir)]
        provenance['队列核验'][cohort] = matched_tasks(old_dir / '逐任务指标_测试集.csv', new_dir / '逐任务指标_测试集.csv')
        rows, paired = paired_summary(a, b, cohort)
        summaries.extend(rows)
        subjects.append(paired)
        old, new = read_json(old_json), read_json(new_json)
        for key, label, direction in (('fc_mse', 'H1至8_FC_MSE', 'down'),
                                      ('fc_pcc', 'H1至8_FC_PCC', 'up'), ('freq_l1', '频域L1', 'down')):
            x, y = old['fc_h1_8'][key]['mean'], new['fc_h1_8'][key]['mean']
            summaries.append({'队列': cohort, '指标': label, '方向': direction,
                '修改前': x, '修改后': y, '差值_新减旧': y - x,
                '相对变化百分比': 100 * (y / x - 1) if direction == 'down' else np.nan, '被试数': len(a)})
        for h in range(1, 9):
            horizons.append({'队列': cohort, '时距': h,
                **{name + '_MAE': frame[f'ODE_H{h}_mae'].mean() for name, frame in [('修改前', a), ('修改后', b)]},
                **{name + '_空间PCC': frame[f'ODE_H{h}_spatial_pcc'].mean() for name, frame in [('修改前', a), ('修改后', b)]},
                '持久性_MAE': b[f'DUAL_H{h}_persistence_mae'].mean(),
                '修改前_差分幅度比': a[f'ODE_H{h}_delta_magnitude_ratio'].mean(),
                '修改后_差分幅度比': b[f'ODE_H{h}_delta_magnitude_ratio'].mean()})
        frames[cohort] = a, b
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(out / '表格/修改前后指标对比.csv', index=False)
    pd.concat(subjects, ignore_index=True).to_csv(out / '表格/修改前后逐被试配对.csv', index=False)
    pd.DataFrame(horizons).to_csv(out / '表格/修改前后逐时距指标.csv', index=False)
    (out / '中间产物/对比来源与核验.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    plot_comparison(frames, out / '图件')
    print(comparison.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
