# coding=utf-8
"""变体轻量评估：从 checkpoint 重建完整模型 → 评 val/test → 落盘指标。

与 analysis/run_comprehensive.py（重评估）不同，本模块只做框架登记所需的
最小闭环：PCC/MAE/R2/PICP + HAMD 分层 + shuffled-HAMD 负对照 + gate 统计
+ FC 边级指标，不生成任何可视化。

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
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from analysis.analyzer import ModelAnalyzer
from models.neurotwin import NeuroTwin
from train.losses import UncertaintyWeightedHybridLoss
from utils.common import parse_pred_quantiles, set_seed
from utils.dataloader import NeuroTwinDataLoader

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


def build_eval_loader(args, eval_split):
    """构建评估 DataLoader（与训练时同一被试级 8:1:1 划分）。"""
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


def compute_fc_metrics(pred, target, eps=1e-6):
    """FC 边级指标（1.2 验收）：样本级 Pearson FC 的上三角 MAE 与 edge-PCC。

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
    diff = fc_p[:, iu[0], iu[1]] - fc_t[:, iu[0], iu[1]]  # [N, E]
    mae_fc = float(np.abs(diff).mean())
    a = fc_p[:, iu[0], iu[1]]
    b = fc_t[:, iu[0], iu[1]]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = (np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0)) + eps)
    edge_pcc = (a * b).sum(axis=0) / denom  # 每条边跨样本相关
    return {'fc_upper_mae': mae_fc,
            'edge_pcc_mean': float(edge_pcc.mean()),
            'edge_pcc_std': float(edge_pcc.std())}


def evaluate(ckpt_path, out_dir, eval_splits=('val', 'test'), seed=2024,
             run_shuffled=True, device=None):
    """评估单个 checkpoint，写 metrics_<split>[_shuffled].json 与样本级 csv。

    返回供 registry 使用的摘要 dict（val_pcc/val_mae/test_pcc/...）。
    """
    ckpt_path = str(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

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
    analyzer = ModelAnalyzer(model, device)
    eval_kwargs = dict(
        moe_load_balance_weight=args.moe_load_balance_weight,
        moe_entropy_weight=args.moe_entropy_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
    )

    summary = {}
    for split in eval_splits:
        data_loader, _ = build_eval_loader(args, split)
        metrics, artifacts = analyzer.evaluate(data_loader, criterion, **eval_kwargs)
        # 自定义附加指标
        pred, target = artifacts['pred'], artifacts['target']
        sample_mae = np.abs(pred - target).reshape(pred.shape[0], -1).mean(axis=1)
        metrics['hamd_stratified'] = hamd_stratified_metrics(
            artifacts['sample_pcc'], sample_mae,
            artifacts['pathology_scores'], artifacts['subj_ids'])
        metrics['gate_stats'] = gate_statistics(artifacts['gate_weights'])
        metrics['fc_metrics'] = compute_fc_metrics(pred, target)
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
        _write_sample_csv(out_dir / f'sample_metrics_{split}.csv',
                          artifacts['subj_ids'], artifacts['pathology_scores'],
                          artifacts['sample_pcc'], sample_mae)
        summary[f'{split}_pcc'] = metrics.get('PCC')
        summary[f'{split}_mae'] = metrics.get('MAE')
        if split == 'test':
            summary['test_r2'] = metrics.get('R2')
            summary['test_picp'] = metrics.get('PICP')
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


def main():
    ap = argparse.ArgumentParser(description='单 checkpoint 轻量评估')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--splits', default='val,test')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--no_shuffled', action='store_true')
    ns = ap.parse_args()
    summary = evaluate(ns.ckpt, ns.out_dir, tuple(ns.splits.split(',')),
                       seed=ns.seed, run_shuffled=not ns.no_shuffled)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
