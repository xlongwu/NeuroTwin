# coding=utf-8
"""统一评估/分析入口：从 checkpoint 重建完整模型 → 评估 → 落盘指标。

唯一评估 CLI（已合并原 analysis/run_comprehensive.py 的全部能力）：
  - 基础闭环（默认）：PCC/MAE/R2/PICP + HAMD 分层 + shuffled-HAMD 负对照
    + gate 统计 + FC 边级/网络级指标 + 被试级聚合 + 参数量
  - 可选开关：--visualize（可视化）、--feature_importance（置换重要性 +
    FDR 显著脑区导出）、--save_arrays（预测数组）、expert-HAMD 分层分析
模型按训练时保存的 config.args 快照自省重建，结构性变体不会权重错配。
默认只评估 test（--splits 可显式加 val）；多 seed 复验暂不做，固定当前 seed。

关键点（为什么不用 build_model_and_loader）：
  analysis/checkpoint.py:build_model_and_loader 构造 NeuroTwin 时只传部分
  参数，结构性变体（head_shape_mode/moe_experts_mode/sc_prior_mode 等）会
  按默认结构建模导致权重错配。训练结束时 save_backbone_weights 已把完整
  vars(args) 存进 ckpt['config']['args']（train/optim.py），因此这里从该
  快照 + inspect.signature 自省重建与训练完全一致的模型，再 strict 加载。
"""
import argparse
import inspect
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from analysis.analyzer import ModelAnalyzer
from analysis.metrics import (NETWORK_GROUPS, _gaussian_z, compute_calibration_metrics,
                              export_significance_to_xlsx, fit_conformal_scale,
                              load_hamd_data)
from analysis.visualizer import VisualizationGenerator
from models.neurotwin import NeuroTwin
from train.losses import UncertaintyWeightedHybridLoss
from utils.common import parse_pred_quantiles, set_seed
from utils.dataloader import NeuroTwinDataLoader

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# NeuroTwin 构造参数名与 argparse 参数名不同名的映射（其余同名直取）
_KWARG_ALIAS = {
    'features': 'num_rois',
    'in_seq_len': 'seq_len',
    'pred_seq_len': 'seq_len',
    'adapter_alpha': 'alpha',
    # 评估期路由温度：训练结束时已退火到 temp_end（与 analysis/checkpoint.py 一致）
    'moe_gate_temperature': 'moe_gate_temp_end',
}
# 由本函数显式控制的参数（不从 args 取）
_KWARG_FIXED = {'pretrain_mode', 'pred_quantiles', 'refiner_return_rounds',
                'pred_logvar_init'}


def load_train_args(ckpt_path):
    """从 checkpoint 读取训练时保存的完整参数快照。"""
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = raw.get('config') if isinstance(raw, dict) else None
    saved = cfg.get('args') if isinstance(cfg, dict) else None
    if not isinstance(saved, dict):
        raise ValueError(
            f'{ckpt_path} 缺少 config.args 配置快照，无法重建模型；'
            f'请确认该权重由当前版本 main.py 保存。')
    return Namespace(**saved)


def build_model_from_args(args, device):
    """按 inspect.signature 自省从参数快照重建与训练一致的 NeuroTwin。"""
    sig = inspect.signature(NeuroTwin.__init__).parameters
    kwargs = {}
    for name, param in sig.items():
        if name == 'self':
            continue
        if name in _KWARG_FIXED:
            continue
        if name in _KWARG_ALIAS:
            kwargs[name] = getattr(args, _KWARG_ALIAS[name])
        elif hasattr(args, name):
            kwargs[name] = getattr(args, name)
        else:
            # 快照中缺失的参数（旧 checkpoint）回落模型默认值
            if param.default is inspect.Parameter.empty:
                raise ValueError(f'参数快照缺少必填构造参数：{name}')
    kwargs['pretrain_mode'] = False
    kwargs['pred_quantiles'] = parse_pred_quantiles(args.pred_quantiles)
    return NeuroTwin(**kwargs).to(device)


def build_eval_loader(args, eval_split, refresh_split_manifest=False):
    """构建评估 DataLoader（与训练时同一被试级 8:1:1 划分，版本化 manifest 复用）。"""
    from main import parse_pathology_fields
    loader = NeuroTwinDataLoader(
        data_root=args.data_root, mode='finetune', batch_size=args.batch_size,
        in_window=args.in_window, pred_window=args.pred_window,
        pathology_input_dim=args.pathology_input_dim,
        pathology_fields=parse_pathology_fields(args.pathology_fields),
        pathology_missing=args.pathology_missing,
        clinical_file=args.clinical_file,
        total_windows=args.total_windows, seq_len=args.seq_len,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        eval_batch_size=args.eval_batch_size,
        cache_subjects=args.cache_subjects,
        variable_cutoff=args.variable_cutoff,
        refresh_split_manifest=bool(getattr(args, 'refresh_split_manifest', False)
                                    or refresh_split_manifest),
    )
    if eval_split == 'test':
        return loader.get_test(), loader.get_test_subjects()
    return loader.get_val(), loader.get_val_subjects()


class ShuffledPathologyLoader:
    """被试级病理评分置换包装器：每个被试被随机指派另一个被试的 HAMD。

    用于 shuffled-HAMD 负对照：模型结构不变、评分信息被打乱，若指标仍不
    下降，说明"病理条件化收益"不成立。
    """

    def __init__(self, loader, subj_to_score):
        self.loader = loader
        self.subj_to_score = subj_to_score

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            ps = batch.get('pathology_score', None)
            subj = batch.get('subj_id', None)
            if ps is not None and subj is not None:
                vals = torch.tensor(
                    [self.subj_to_score[str(s)] for s in subj], dtype=ps.dtype)
                batch['pathology_score'] = vals.reshape(ps.shape)
            yield batch


def build_subject_score_map(subj_ids, pathology_scores):
    """从评估 artifacts 建立被试→HAMD 映射（同被试评分应一致）。"""
    mapping = {}
    for s, h in zip(subj_ids, pathology_scores):
        s = str(s)
        if np.isnan(h):
            continue
        if s in mapping and abs(mapping[s] - float(h)) > 1e-4:
            raise ValueError(f'被试 {s} 的病理评分在样本间不一致：'
                             f'{mapping[s]} vs {h}')
        mapping[s] = float(h)
    return mapping


def shuffled_subject_map(subj_to_score, seed=2024):
    """随机置换被试→评分映射（负对照用）。"""
    rng = np.random.default_rng(seed)
    subjects = sorted(subj_to_score)
    scores = [subj_to_score[s] for s in subjects]
    perm = rng.permutation(len(subjects))
    return {s: scores[i] for s, i in zip(subjects, perm)}


def hamd_stratified_metrics(sample_pcc, sample_mae, hamd, subj_ids, n_bins=4):
    """被试级 HAMD 分箱 PCC/MAE：按被试聚合样本，再按 HAMD 分位数分箱。"""
    per_subj = {}
    for pcc, mae, h, s in zip(sample_pcc, sample_mae, hamd, subj_ids):
        d = per_subj.setdefault(str(s), {'pcc': [], 'mae': [], 'hamd': None})
        d['pcc'].append(float(pcc))
        d['mae'].append(float(mae))
        if not np.isnan(h):
            d['hamd'] = float(h)
    subjects = [(d['hamd'], float(np.mean(d['pcc'])), float(np.mean(d['mae'])))
                for d in per_subj.values() if d['hamd'] is not None]
    if not subjects:
        return {'available': False, 'reason': '无有效 HAMD 评分'}
    arr = np.array(subjects, dtype=np.float64)  # [N, 3]: hamd, pcc, mae
    edges = np.quantile(arr[:, 0], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    bins = []
    for i in range(n_bins):
        m = (arr[:, 0] >= edges[i]) & (arr[:, 0] < edges[i + 1])
        if not m.any():
            continue
        bins.append({
            'bin': i, 'hamd_min': float(edges[i]), 'hamd_max': float(edges[i + 1]),
            'n_subjects': int(m.sum()),
            'pcc_mean': float(arr[m, 1].mean()), 'pcc_std': float(arr[m, 1].std()),
            'mae_mean': float(arr[m, 2].mean()), 'mae_std': float(arr[m, 2].std()),
        })
    corr_pcc = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    corr_mae = float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1])
    return {'available': True, 'n_subjects': int(arr.shape[0]),
            'bins': bins,
            'corr_hamd_pcc': corr_pcc, 'corr_hamd_mae': corr_mae}


def subject_level_metrics(sample_pcc, sample_mae, subj_ids):
    """被试级聚合指标（2.1 统计单位 / 2.5 失败分析维度）。

    滑窗样本先按被试聚合（PCC/MAE 取均值），再统计分布与低 PCC 被试，
    避免把同一被试的多个滑窗当作独立重复。
    """
    per = {}
    for pcc, mae, s in zip(sample_pcc, sample_mae, subj_ids):
        d = per.setdefault(str(s), {'pcc': [], 'mae': []})
        d['pcc'].append(float(pcc))
        d['mae'].append(float(mae))
    if not per:
        return {'n_subjects': 0}
    subj_pcc = np.array([np.mean(d['pcc']) for d in per.values()], dtype=np.float64)
    subj_mae = np.array([np.mean(d['mae']) for d in per.values()], dtype=np.float64)
    low = subj_pcc < 0.5
    k = max(1, int(round(0.1 * len(subj_pcc))))
    worst = np.sort(subj_pcc)[:k]
    return {'n_subjects': int(len(subj_pcc)),
            'subj_pcc_mean': float(subj_pcc.mean()),
            'subj_pcc_std': float(subj_pcc.std()),
            'subj_pcc_median': float(np.median(subj_pcc)),
            'subj_mae_mean': float(subj_mae.mean()),
            'subj_mae_std': float(subj_mae.std()),
            'n_pcc_below_0.5': int(low.sum()),
            'low_pcc_ratio': float(low.mean()),
            'worst10_pcc_mean': float(worst.mean()),
            'samples_per_subject_mean': float(
                np.mean([len(d['pcc']) for d in per.values()]))}


def gate_statistics(gate_weights, eps=1e-8):
    """专家门控统计：均值/方差/归一化熵/top1 分配分布。

    gate_weights 期望 [N, E]（analyzer 已按样本 concat）；其他形状按最后一维
    展开为专家维。
    """
    if gate_weights is None:
        return {'available': False}
    g = np.asarray(gate_weights, dtype=np.float64)
    if g.ndim == 1:
        g = g[:, None]
    if g.ndim > 2:
        g = g.reshape(-1, g.shape[-1])
    n_experts = g.shape[1]
    p = g / (g.sum(axis=1, keepdims=True) + eps)
    entropy = -(p * np.log(p + eps)).sum(axis=1) / max(1.0, np.log(n_experts))
    top1 = p.argmax(axis=1)
    top1_dist = {f'E{i}': int((top1 == i).sum()) for i in range(n_experts)}
    return {'available': True, 'n_samples': int(g.shape[0]),
            'expert_mean': [float(x) for x in g.mean(axis=0)],
            'expert_std': [float(x) for x in g.std(axis=0)],
            'normalized_entropy_mean': float(entropy.mean()),
            'top1_assignment': top1_dist}


def _roi_network_labels(n_rois):
    """ROI 索引(0 基)→网络标签；未命名脑区归入 Other（NETWORK_GROUPS 为 1 基 AAL 索引）。"""
    labels = ['Other'] * n_rois
    for net, regions in NETWORK_GROUPS.items():
        for r in regions:
            if 1 <= r <= n_rois:
                labels[r - 1] = net
    return labels


def compute_fc_metrics(pred, target, eps=1e-6):
    """FC 指标（1.2/2.2 验收）：样本级 Pearson FC 的上三角 MAE/RMSE、edge-PCC，
    以及命名网络内/网络间边 MAE（含 Other 归组的边不参与网络级统计）。

    pred/target: [N, F, W, S]（或可展平为 [N, F, T] 的形状）。
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n, f = p.shape[0], p.shape[1]
    p = p.reshape(n, f, -1)
    t = t.reshape(n, f, -1)

    def _fc(x):
        x = x - x.mean(axis=-1, keepdims=True)
        x = x / (x.std(axis=-1, keepdims=True) + eps)
        return np.einsum('nft,ngt->nfg', x, x) / x.shape[-1]

    fc_p, fc_t = _fc(p), _fc(t)
    iu = np.triu_indices(f, k=1)
    a = fc_p[:, iu[0], iu[1]]
    b = fc_t[:, iu[0], iu[1]]
    diff = a - b  # [N, E]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = (np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0)) + eps)
    edge_pcc = (a * b).sum(axis=0) / denom  # 每条边跨样本相关
    result = {'fc_upper_mae': float(np.abs(diff).mean()),
              'fc_upper_rmse': float(np.sqrt((diff ** 2).mean())),
              'edge_pcc_mean': float(edge_pcc.mean()),
              'edge_pcc_std': float(edge_pcc.std())}
    # 网络级指标：仅统计两端均属于命名网络（非 Other）的边
    labels = _roi_network_labels(f)
    named = [i for i, lb in enumerate(labels) if lb != 'Other']
    if len(named) >= 2:
        edge_idx = {(i, j): k for k, (i, j) in enumerate(zip(iu[0], iu[1]))}
        within, between = [], []
        for x in range(len(named)):
            for y in range(x + 1, len(named)):
                k = edge_idx.get((named[x], named[y]))
                if k is None:
                    continue
                (within if labels[named[x]] == labels[named[y]]
                 else between).append(k)
        if within:
            result['fc_within_mae'] = float(np.abs(diff[:, within]).mean())
            result['n_edges_within'] = len(within)
        if between:
            result['fc_between_mae'] = float(np.abs(diff[:, between]).mean())
            result['n_edges_between'] = len(between)
    return result


def evaluate(ckpt_path, out_dir, eval_splits=('test',), seed=2024,
             run_shuffled=True, device=None, visualize=False,
             compute_feature_importance=False, save_arrays=False,
             expert_assignment_method='top1', feature_importance_samples=50,
             feature_importance_method='balanced',
             feature_importance_permutations=200, calibrate=True,
             refresh_split_manifest=False):
    """评估单个 checkpoint，写 metrics_<split>[_shuffled].json、样本/被试级
    csv，并按开关生成 expert-HAMD 分析、可视化、特征重要性、预测数组。

    默认只评估 test；需要 val 时显式传 eval_splits。返回供 registry 使用的
    摘要 dict（test_pcc/test_mae/test_fc_upper_mae/...）。

    calibrate=True（默认）时启用 conformal 方差校准：在 val 上拟合区间缩放
    因子 q（fit_conformal_scale），对各 split 应用 ``μ ± q·σ`` 得到
    PICP_cal/MPIW_cal（写入 metrics_<split>.json 与 summary）。val 不在
    eval_splits 中时自动补评；checkpoint 无概率头输出时跳过校准。
    """
    ckpt_path = str(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    if calibrate and 'val' not in eval_splits:
        # conformal 拟合集：校准因子的覆盖保证来自留出的 val
        log.info('conformal 校准需要 val 拟合，自动加入 val 评估')
        eval_splits = ('val',) + tuple(eval_splits)
    calib_cache = {}

    args = load_train_args(ckpt_path)
    model = build_model_from_args(args, device)
    # strict 加载：配置重建不完整时立即暴露，不允许静默随机权重评估
    from analysis.checkpoint import load_checkpoint_compat
    model, compat = load_checkpoint_compat(model, ckpt_path, device, strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')

    criterion = UncertaintyWeightedHybridLoss(
        init_log_var_pcc=args.init_log_var_pcc,
        init_log_var_mae=args.init_log_var_mae,
        init_log_var_diff=args.init_log_var_diff,
        init_log_var_std=args.init_log_var_std,
        clamp_log_vars=args.clamp_log_vars,
        log_var_min=args.log_var_min, log_var_max=args.log_var_max,
        diff_mode=args.loss_diff_mode,
        init_log_var_nll=args.init_log_var_nll,
    ).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    analyzer = ModelAnalyzer(model, device)
    eval_kwargs = dict(
        moe_load_balance_weight=args.moe_load_balance_weight,
        moe_entropy_weight=args.moe_entropy_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
    )
    # expert-HAMD 分层分析用的量表数据；文件缺失时显式告警并跳过该分析
    hamd_path = Path(args.data_root) / args.clinical_file
    if hamd_path.exists():
        hamd_data = load_hamd_data(str(hamd_path), str(hamd_path))
    else:
        hamd_data = {}
        log.warning(f'未找到临床评分表 {hamd_path}，expert-HAMD 分层分析将跳过')

    summary = {}
    for split in eval_splits:
        data_loader, _ = build_eval_loader(args, split,
                                           refresh_split_manifest=refresh_split_manifest)
        metrics, artifacts = analyzer.evaluate(data_loader, criterion, **eval_kwargs)
        # 自定义附加指标
        pred, target = artifacts['pred'], artifacts['target']
        sample_mae = np.abs(pred - target).reshape(pred.shape[0], -1).mean(axis=1)
        metrics['hamd_stratified'] = hamd_stratified_metrics(
            artifacts['sample_pcc'], sample_mae,
            artifacts['pathology_scores'], artifacts['subj_ids'])
        metrics['gate_stats'] = gate_statistics(artifacts['gate_weights'])
        metrics['fc_metrics'] = compute_fc_metrics(pred, target)
        metrics['subject_level'] = subject_level_metrics(
            artifacts['sample_pcc'], sample_mae, artifacts['subj_ids'])
        metrics['n_params'] = n_params
        # 负对照：打乱被试级 HAMD 后重评（PCC/MAE 应显著下降）
        if run_shuffled and 'pathology_score' in _probe_batch_keys(data_loader):
            s2s = build_subject_score_map(
                artifacts['subj_ids'], artifacts['pathology_scores'])
            if s2s:
                shuffled_loader = ShuffledPathologyLoader(
                    data_loader, shuffled_subject_map(s2s, seed=seed))
                sh_metrics, _ = analyzer.evaluate(
                    shuffled_loader, criterion, **eval_kwargs)
                for k in ('PCC', 'MAE', 'R2'):
                    if k in sh_metrics:
                        metrics[f'shuffled_{k}'] = sh_metrics[k]
                metrics['shuffled_delta_PCC'] = (
                    metrics.get('shuffled_PCC', float('nan'))
                    - metrics.get('PCC', float('nan')))

        with (out_dir / f'metrics_{split}.json').open('w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
        if calibrate and (artifacts.get('pred_logvar') is not None
                          or artifacts.get('lower') is not None):
            calib_cache[split] = {
                'pred': artifacts['pred'], 'target': artifacts['target'],
                'pred_logvar': artifacts.get('pred_logvar'),
                'lower': artifacts.get('lower'), 'upper': artifacts.get('upper'),
            }
        _write_sample_csv(out_dir / f'sample_metrics_{split}.csv',
                          artifacts['subj_ids'], artifacts['pathology_scores'],
                          artifacts['sample_pcc'], sample_mae)
        _write_subject_csv(out_dir / f'subject_metrics_{split}.csv',
                           artifacts['subj_ids'], artifacts['pathology_scores'],
                           artifacts['sample_pcc'], sample_mae)

        # ---- 可选重分析能力（自 analysis/run_comprehensive.py 合并） ----
        expert_hamd_analysis = {}
        if hamd_data and artifacts['gate_weights'] is not None:
            expert_hamd_analysis = analyzer.analyze_expert_hamd_distribution(
                subj_ids=artifacts['subj_ids'],
                gate_weights=artifacts['gate_weights'],
                hamd_data=hamd_data,
                assignment_method=expert_assignment_method)
            if expert_hamd_analysis:
                with (out_dir / f'expert_hamd_analysis_{split}.json').open(
                        'w', encoding='utf-8') as f:
                    json.dump(expert_hamd_analysis, f, ensure_ascii=False,
                              indent=2, default=float)
        if visualize:
            (out_dir / 'visualizations').mkdir(exist_ok=True)
            viz = VisualizationGenerator(out_dir / 'visualizations')
            viz.plot_prediction_scatter(artifacts['pred'], artifacts['target'])
            viz.plot_error_distribution(artifacts['pred'], artifacts['target'])
            viz.plot_sample_pcc_distribution(artifacts['sample_pcc'])
            viz.plot_roi_heatmap(artifacts['roi_metrics'])
            viz.plot_temporal_metrics(artifacts['temporal_metrics'])
            viz.plot_moe_gate_analysis(artifacts['gate_weights'], metrics)
            if expert_hamd_analysis:
                viz.plot_expert_hamd_distribution(expert_hamd_analysis)
            for i in range(min(3, artifacts['pred'].shape[0])):
                for r in range(0, artifacts['pred'].shape[1], 20):
                    viz.plot_temporal_prediction(
                        artifacts['pred'], artifacts['target'], i, r)
            if not np.all(np.isnan(artifacts['pathology_scores'])):
                viz.plot_pathology_correlation(
                    artifacts['pathology_scores'], artifacts['sample_pcc'])
            viz.plot_comprehensive_dashboard(metrics, artifacts)
            log.info(f'[{split}] 已生成 {viz.fig_count} 张可视化')
        if compute_feature_importance:
            importance = analyzer.compute_feature_importance(
                data_loader, n_samples=feature_importance_samples,
                n_permutations=feature_importance_permutations,
                method=feature_importance_method)
            save_result = {
                'roi_importance': importance['roi_importance'].tolist(),
                'roi_pvalues': importance.get('roi_pvalues', []).tolist(),
                'roi_std_errors': importance.get('roi_std_errors', []).tolist(),
                'roi_importance_rank': importance['roi_importance_rank'].tolist(),
                'baseline_mae': importance['baseline_mae'],
                'method': importance.get('method', feature_importance_method),
            }
            with (out_dir / f'feature_importance_{split}.json').open(
                    'w', encoding='utf-8') as f:
                json.dump(save_result, f, ensure_ascii=False, indent=2,
                          default=float)
            if visualize:
                viz.plot_feature_importance(importance)
                viz.plot_significant_brain_regions(importance, fdr_alpha=0.001)
            regions_dir = out_dir / 'brain_regions'
            regions_dir.mkdir(exist_ok=True)
            export_significance_to_xlsx(
                importance,
                output_path=regions_dir / f'region_significance_{split}.xlsx',
                fdr_alpha=0.05,
            )
        if save_arrays:
            np.savez_compressed(
                out_dir / f'predictions_{split}.npz',
                pred=artifacts['pred'], target=artifacts['target'],
                sample_pcc=artifacts['sample_pcc'],
                gate_weights=(artifacts['gate_weights']
                              if artifacts['gate_weights'] is not None else []),
                subj_ids=artifacts['subj_ids'],
                pathology_scores=artifacts['pathology_scores'],
                roi_mae=artifacts['roi_metrics']['roi_mae'],
                roi_pcc=artifacts['roi_metrics']['roi_pcc'],
            )

        summary[f'{split}_pcc'] = metrics.get('PCC')
        summary[f'{split}_mae'] = metrics.get('MAE')
        if split == 'test':
            summary['test_r2'] = metrics.get('R2')
            summary['test_picp'] = metrics.get('PICP')
            summary['test_fc_upper_mae'] = metrics['fc_metrics'].get('fc_upper_mae')
            summary['test_edge_pcc'] = metrics['fc_metrics'].get('edge_pcc_mean')
            summary['test_subj_pcc'] = metrics['subject_level'].get('subj_pcc_mean')
            summary['test_n_params'] = n_params

    if calibrate and calib_cache:
        # conformal 方差校准：val 拟合缩放因子 q，各 split 应用 μ ± q·σ。
        # 高斯 NLL 训练的方差估计系统性上偏导致 PICP 过覆盖（名义 90% 实测
        # 0.93-0.96），校准后区间在拟合集上保证边际覆盖 ≥ 1-α。
        z = _gaussian_z(0.1)  # 与 compute_calibration_metrics 默认 alpha=0.1 一致
        fit_split = 'val' if 'val' in calib_cache else next(iter(calib_cache))
        if fit_split != 'val':
            log.warning('缺少 val 拟合集，改用 %s 拟合校准因子（覆盖保证仅在拟合集上成立）',
                        fit_split)
        fc = calib_cache[fit_split]
        q = fit_conformal_scale(fc['pred'], fc['target'],
                                pred_logvar=fc['pred_logvar'],
                                lower=fc['lower'], upper=fc['upper'])
        log.info('conformal 校准：q=%.4f（原始 z=%.4f，q<z 说明原区间过宽）', q, z)
        for split, c in calib_cache.items():
            if c['pred_logvar'] is not None:
                sigma = np.exp(0.5 * np.asarray(c['pred_logvar'], dtype=np.float64))
            else:  # 分位数头：由原始端点反推有效 σ
                sigma = (np.asarray(c['upper'], dtype=np.float64)
                         - np.asarray(c['lower'], dtype=np.float64)) / (2.0 * z)
            lower_c = c['pred'] - q * sigma
            upper_c = c['pred'] + q * sigma
            cal = compute_calibration_metrics(
                c['pred'], c['target'], lower=lower_c, upper=upper_c, alpha=0.1)
            mpath = out_dir / f'metrics_{split}.json'
            try:
                m = json.loads(mpath.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as e:
                log.warning('读取 %s 失败（%s），校准指标仅写入 summary', mpath, e)
                m = {}
            m.update({'PICP_cal': cal['PICP'], 'MPIW_cal': cal['MPIW'],
                      'MPIW_norm_cal': cal['MPIW_norm'],
                      'calib_q': float(q), 'calib_fit_split': fit_split})
            mpath.write_text(json.dumps(m, ensure_ascii=False, indent=2,
                                        default=float), encoding='utf-8')
            summary[f'{split}_picp_cal'] = cal['PICP']
            summary['calib_q'] = float(q)
    return summary


def _probe_batch_keys(data_loader):
    """探测一个 batch 的键集合（判断数据集是否提供 pathology_score）。"""
    for batch in data_loader:
        return list(batch.keys())
    return []


def _write_sample_csv(path, subj_ids, hamd, pcc, mae):
    import csv as _csv
    with path.open('w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['subj_id', 'hamd', 'sample_pcc', 'sample_mae'])
        for s, h, p, m in zip(subj_ids, hamd, pcc, mae):
            w.writerow([s, '' if np.isnan(h) else float(h), float(p), float(m)])


def _write_subject_csv(path, subj_ids, hamd, pcc, mae):
    """被试级聚合 CSV（2.1：统计单位是被试，滑窗样本先聚合再落盘）。"""
    import csv as _csv
    per = {}
    for s, h, p, m in zip(subj_ids, hamd, pcc, mae):
        d = per.setdefault(str(s), {'hamd': float('nan'), 'pcc': [], 'mae': []})
        d['pcc'].append(float(p))
        d['mae'].append(float(m))
        if not np.isnan(h):
            d['hamd'] = float(h)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['subj_id', 'hamd', 'n_samples',
                    'pcc_mean', 'pcc_std', 'mae_mean'])
        for s in sorted(per):
            d = per[s]
            w.writerow([s, '' if np.isnan(d['hamd']) else d['hamd'],
                        len(d['pcc']), float(np.mean(d['pcc'])),
                        float(np.std(d['pcc'])), float(np.mean(d['mae']))])


def main():
    ap = argparse.ArgumentParser(description='单 checkpoint 轻量评估')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--splits', default='test',
                    help='评估划分（逗号分隔）；默认只评 test，需要 val 时显式传入')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--no_shuffled', action='store_true')
    ap.add_argument('--visualize', action='store_true',
                    help='生成评估可视化（散射/误差/ROI 热图/门控/仪表板等）')
    ap.add_argument('--feature_importance', action='store_true',
                    help='置换特征重要性 + FDR 显著脑区 xlsx 导出')
    ap.add_argument('--save_arrays', action='store_true',
                    help='保存预测数组 predictions_<split>.npz')
    ap.add_argument('--no_calibrate', dest='calibrate', action='store_false',
                    help='跳过 conformal 方差校准（默认开启：val 拟合 q → 各 split 应用）')
    ap.add_argument('--refresh_split_manifest', action='store_true',
                    help='重新生成 subject_split_finetune.json 切分 manifest（默认复用）')
    ap.add_argument('--expert_assignment_method', default='top1',
                    choices=['top1', 'threshold'],
                    help='expert-HAMD 分析的样本-专家指派方法')
    ap.add_argument('--feature_importance_samples', type=int, default=50)
    ap.add_argument('--feature_importance_method', default='balanced',
                    choices=['fast', 'balanced', 'precise', 'permutation'],
                    help='特征重要性方法: fast(~1s), balanced(~5min,推荐), '
                         'precise(~25min)')
    ap.add_argument('--feature_importance_permutations', type=int, default=200)
    ns = ap.parse_args()
    summary = evaluate(ns.ckpt, ns.out_dir, tuple(ns.splits.split(',')),
                       seed=ns.seed, run_shuffled=not ns.no_shuffled,
                       visualize=ns.visualize,
                       compute_feature_importance=ns.feature_importance,
                       save_arrays=ns.save_arrays,
                       expert_assignment_method=ns.expert_assignment_method,
                       feature_importance_samples=ns.feature_importance_samples,
                       feature_importance_method=ns.feature_importance_method,
                       feature_importance_permutations=ns.feature_importance_permutations,
                       calibrate=ns.calibrate,
                       refresh_split_manifest=ns.refresh_split_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
