"""固定 H1–8 协议下导出 Profile 交换、扫描内稳定性与病理表征表格。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from analysis.checkpoint import load_checkpoint_compat
from analysis.evaluate import build_model_from_args, load_condition_bundle, load_train_args
from main import condition_batch, condition_loader_kwargs, dynamics_extras, next_point_batch
from utils.common import set_seed
from utils.dataloader import CONDITION_FILE_BY_MODE, NeuroTwinDataLoader


@torch.no_grad()
def evaluate_profiles(model, loader, dataset, subjects: list[str], device,
                      clinical_readout: bool = False) -> tuple:
    """Profile 交换保持历史与临床条件不变；候选供体按被试排序循环配对。"""
    model.eval()
    if len(subjects) < 2:
        raise ValueError('Profile 交换至少需要两名不同被试')
    profiles, profiles_b, stability = {}, {}, []
    length = dataset.calibration_length
    for sid in sorted(subjects):
        series = dataset.get_series(sid)
        cal = torch.tensor(series[:, None, :length].copy(), device=device).unsqueeze(0)
        p = model.calibrate_profile(cal)
        profiles[sid] = p
        if series.shape[1] >= 2 * length:
            cal_b = torch.tensor(series[:, None, length:2 * length].copy(), device=device).unsqueeze(0)
            pb = model.calibrate_profile(cal_b)
            profiles_b[sid] = pb
            row = {'被试': sid, 'Profile余弦相似度': F.cosine_similarity(p, pb).item()}
            pa, pp = model.generate_personal_parameters(p), model.generate_personal_parameters(pb)
            if pa is not None:
                for key, label in (('coefficients', '个体系数'), ('kappa_delta', '区域扩散修正'),
                                   ('graph_coefficients', '个体图系数')):
                    row[label + '均方差'] = (pa[key] - pp[key]).square().mean().item()
            stability.append(row)
    ordered = sorted(subjects)
    donors = {sid: ordered[(i + 1) % len(ordered)] for i, sid in enumerate(ordered)}
    metrics = defaultdict(lambda: defaultdict(list))
    parameters = defaultdict(list)
    for batch in loader:
        x, _, _, _, graph, _ = next_point_batch(batch, device)
        _, control = dynamics_extras(batch, device)
        ids = [str(sid) for sid in batch['subj_id']]
        p = torch.cat([profiles[sid] for sid in ids])
        donor = torch.cat([profiles[donors[sid]] for sid in ids])
        kwargs = dict(condition=condition_batch(batch, device), controls=control,
                      dynamics_horizon=8, with_aux=False)
        _, own = model(x, graph, profile=p, **kwargs)
        _, swapped = model(x, graph, profile=donor, **kwargs)
        _, zeroed = model(x, graph, profile=torch.zeros_like(p), **kwargs)
        variants = [('原Profile', own), ('交换Profile', swapped), ('零Profile', zeroed)]
        if model.one_step_head.pathology_dynamics is not None:
            _, healthy = model(x, graph, profile=p, pathology_active=False, **kwargs)
            variants.append(('健康基础', healthy))
        target, mask = batch['future'].to(device), batch['future_mask'].to(device)
        for i, sid in enumerate(ids):
            for name, aux in variants:
                error = (aux['dynamics_pred'][i] - target[i]).abs().mean(0)
                for h in range(8):
                    if mask[i, h]:
                        metrics[sid][f'{name}_H{h+1}_MAE'].append(error[h].item())
                metrics[sid][name + '_H1至8_MAE'].append(
                    (error * mask[i]).sum().item() / mask[i].sum().item())
            path = own.get('pathology_parameters')
            if path is not None:
                parameters[sid].append(torch.cat((path['embedding'][i],
                    path['coefficients'][i], path['operator_energy'][i:i+1],
                    path['disease_logit'][i:i+1].sigmoid(),
                    path['clinical_prediction'][i:i+1])).cpu().numpy())
    rows = []
    for sid, values in metrics.items():
        row = {'被试': sid, 'Profile供体': donors[sid],
               **{key: float(np.mean(v)) for key, v in values.items()}}
        row['交换MAE增量'] = row['交换Profile_H1至8_MAE'] - row['原Profile_H1至8_MAE']
        row['零ProfileMAE增量'] = row['零Profile_H1至8_MAE'] - row['原Profile_H1至8_MAE']
        if '健康基础_H1至8_MAE' in row:
            row['病理增量MAE改善'] = row['健康基础_H1至8_MAE'] - row['原Profile_H1至8_MAE']
        rows.append(row)
    stability_frame = pd.DataFrame(stability)
    if profiles_b:
        valid = sorted(profiles_b)
        a = F.normalize(torch.cat([profiles[sid] for sid in valid]), dim=-1)
        b = F.normalize(torch.cat([profiles_b[sid] for sid in valid]), dim=-1)
        retrieval = (a @ b.T).argmax(-1).cpu().numpy()
        hits = {sid: float(index == i) for i, (sid, index) in enumerate(zip(valid, retrieval))}
        stability_frame['扫描内检索命中'] = stability_frame['被试'].map(hits)
    parameter_rows = []
    for sid in sorted(profiles):
        row = {'被试': sid}
        for i, value in enumerate(profiles[sid][0].cpu().tolist()):
            row[f'Profile_{i+1}'] = value
        personal = model.generate_personal_parameters(profiles[sid])
        if personal is not None:
            for key, label in (('coefficients', '个体系数'), ('kappa_delta', '扩散logit修正'),
                               ('gamma', '衰减率'), ('graph_coefficients', '图系数')):
                for i, value in enumerate(personal[key][0].cpu().tolist()):
                    row[f'{label}_{i+1}'] = value
        if sid in parameters:
            values = np.mean(parameters[sid], axis=0)
            path = model.one_step_head.pathology_dynamics
            d = path.disease_head.in_features
            rank = path.coefficients.out_features
            labels = ([f'病理表征_{i+1}' for i in range(d)]
                      + [f'病理系数_{i+1}' for i in range(rank)]
                      + ['病理算子平方范数', 'MDD辅助概率', '临床辅助预测'])
            if not clinical_readout:
                labels, values = labels[:-1], values[:-1]
            row.update(zip(labels, map(float, values)))
        parameter_rows.append(row)
    return pd.DataFrame(rows), stability_frame, pd.DataFrame(parameter_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--cohort', choices=['hc', 'mdd'])
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--split-file', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--anchors', type=int, default=8)
    ns = parser.parse_args()
    args = load_train_args(ns.checkpoint)
    set_seed(args.seed)
    if ns.cohort:
        args.mode = 'pretrain' if ns.cohort == 'hc' else 'finetune'
    args.data_root = ns.data_root
    args.split_file = ns.split_file or ('' if ns.cohort else getattr(args, 'split_file', ''))
    args.condition_file = CONDITION_FILE_BY_MODE[args.mode]
    bundle = load_condition_bundle(ns.checkpoint)
    model = build_model_from_args(args, ns.device, bundle)
    load_checkpoint_compat(model, ns.checkpoint, ns.device, strict=True)
    dl = NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode, batch_size=args.batch_size,
        num_workers=0, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        bold_source=args.bold_source, num_rois=args.num_rois, graph_source=args.graph_source,
        context_min=args.context_min, context_max=args.context_max,
        calibration_length=args.calibration_length, eval_context_length=args.context_max,
        eval_anchors_per_subject=ns.anchors, eval_rollout_tasks_per_subject=0,
        eval_rollout_steps=0, eval_all_future_steps=8, eval_anchor_min_future_steps=8,
        **condition_loader_kwargs(args, bundle))
    view = getattr(dl, ns.split + '_dataset')
    # train 评估也使用固定锚点，不使用随机增强训练视图。
    if ns.split == 'train':
        from utils.dataloader import NextPointEvalView
        view = NextPointEvalView(dl.base_dataset, view.subject_indices, args.context_max,
            anchors_per_subject=ns.anchors, all_future_steps=8, anchor_min_future_steps=8)
    loader = dl._make_bucketed_loader(view, args.batch_size, shuffle=False)
    subjects = getattr(dl, ns.split + '_subjects')
    predictions, stability, parameters = evaluate_profiles(
        model, loader, dl.base_dataset, subjects, ns.device,
        clinical_readout=getattr(args, 'lambda_clinical', 0) > 0)
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(ns.out_dir / '个体交换预测指标.csv', index=False)
    stability.to_csv(ns.out_dir / '扫描内稳定性指标.csv', index=False)
    parameters.to_csv(ns.out_dir / '个体与病理参数.csv', index=False)
    summary = {'被试数': len(predictions), '交换MAE增量': float(predictions['交换MAE增量'].mean()),
               '零ProfileMAE增量': float(predictions['零ProfileMAE增量'].mean()),
               'checkpoint': str(ns.checkpoint), 'cohort': args.mode, 'split': ns.split,
               '供体规则': '同队列同划分内按被试排序循环配对；不代表混杂控制',
               '稳定性范围': '同扫描两个非重叠窗口；不代表跨访视可重复性',
               '临床辅助预测已训练': getattr(args, 'lambda_clinical', 0) > 0,
               '病理解释': 'BOLD-only 辅助表征；需另行控制站点、人口学并比较静态FC'}
    (ns.out_dir / '个体化评估汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
