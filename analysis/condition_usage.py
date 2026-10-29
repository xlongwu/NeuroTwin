# coding=utf-8
"""v2 B5：FiLM 条件使用量量化（冻结权重，只读）。

问题：模型对 checkpoint 规定的条件字段（HC 为 hc3，MDD 为 mdd8）的实际使用量有多大？
若近零，则「条件使用近零」即为最终结论，不再开敏感性分析支线（v2 §6.6）。

对指定组和划分的每名被试（全部 K=64 锚点，lag-1）：

1. **FiLM 幅度**（ODE 头）：``‖γ(c)‖₂``、``‖β(c)‖₂``（``one_step_head.film``
   直接消费条件嵌入 c）；
2. **AdaRMS 生成器幅度**（编码器侧）：``model.cond_norm(c)`` 各层 γ/β 参数的
   平均 L2 范数（未启用 ``cond_norm`` 时记 null）；
3. **输出敏感度**：``Δpred = mean |pred(c) − pred(c=0)|``（同锚点两次前向；
   c=0 定义为全缺失编码：z=0、code=Unknown、mask=0），同时报告相对量
   ``Δpred / mean|true − x_last|``（信号尺度）与两种条件下的 lag-1 MAE。

口径：所有量为描述性统计；Δpred 反映的是「条件通路的输出贡献」，不是临床
效应方向。产物 ``results/pathology_analysis/condition_usage.json`` + 逐被试 CSV。

用法::

    python -m analysis.condition_usage \
      --ckpt checkpoints/h1_8_v2/neurotwin_pretrain/base_best.pt \
      --split val --device cuda:0
    # MDD 使用微调权重与 mdd8 条件：
    python -m analysis.condition_usage --ckpt checkpoints/neurotwin_finetune/finetuned_best.pt \
      --group mdd --split test --device cuda:0 --out_json <新结果.json> --out_csv <新结果.csv>
"""
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_OUT_JSON = Path('results/pathology_analysis/condition_usage.json')
DEFAULT_OUT_CSV = Path('results/pathology_analysis/condition_usage.csv')


def zero_condition(batch_cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """c=0：全缺失编码（z=0、code=Unknown(0)、mask=0）。"""
    return {'cond_z': torch.zeros_like(batch_cond['cond_z']),
            'cond_code': torch.zeros_like(batch_cond['cond_code']),
            'cond_mask': torch.zeros_like(batch_cond['cond_mask'])}


def film_norms(model, cond_emb: torch.Tensor) -> Dict[str, float]:
    """条件嵌入 [B, C] → ODE 头 FiLM 的 ‖γ‖/‖β‖（批内均值）。"""
    head = getattr(model, 'one_step_head', None)
    film = getattr(head, 'film', None) if head is not None else None
    if film is None or cond_emb is None:
        return {'ode_film_gamma': float('nan'), 'ode_film_beta': float('nan')}
    with torch.no_grad():
        scale, shift = film(cond_emb)
    return {'ode_film_gamma': float(scale.norm(dim=-1).mean()),
            'ode_film_beta': float(shift.norm(dim=-1).mean())}


def adarms_norms(model, cond_emb: torch.Tensor) -> Dict[str, float]:
    """条件嵌入 → 编码器侧 AdaRMS 生成器各层 γ/β 参数的平均 L2 范数。"""
    gen = getattr(model, 'cond_norm', None)
    if gen is None or cond_emb is None:
        return {'encoder_adarms_gamma': float('nan'),
                'encoder_adarms_beta': float('nan')}
    with torch.no_grad():
        out = gen(cond_emb)                     # Dict[int, (g..., b..., ...)]
    gs, bs = [], []
    for params in out.values():
        # ConditionNormGenerator.forward 返回每层 (γ参数组, β参数组, 其余)；
        # 参数组内元素为可广播张量，逐元素取 L2 后汇总
        for group, acc in ((params[0], gs), (params[1], bs)):
            for t in group:
                acc.append(float(t.reshape(t.shape[0], -1).norm(dim=-1).mean()))
    if not gs:
        return {'encoder_adarms_gamma': float('nan'),
                'encoder_adarms_beta': float('nan')}
    return {'encoder_adarms_gamma': float(np.mean(gs)),
            'encoder_adarms_beta': float(np.mean(bs))}


@torch.no_grad()
def run_condition_usage(model, loader, subjects, view, device, batch_size: int,
                        num_workers: int) -> List[dict]:
    """逐被试计算 FiLM 幅度与 Δpred（同锚点两次前向）。"""
    from main import dynamics_extras
    from torch.utils.data import DataLoader
    model.eval()

    acc: Dict[str, Dict[str, list]] = {}
    dl = DataLoader(view, batch_size=int(batch_size), shuffle=False,
                    num_workers=int(num_workers))

    def _forward(batch, cond_override=None):
        x = batch['x'].to(device, non_blocking=True)
        sc = batch['sc'].to(device, non_blocking=True)
        cond = None
        if 'cond_z' in batch:
            raw = batch if cond_override is None else None
            cond = ({'cond_z': batch['cond_z'].to(device, non_blocking=True),
                     'cond_code': batch['cond_code'].to(device, non_blocking=True),
                     'cond_mask': batch['cond_mask'].to(device, non_blocking=True)}
                    if cond_override is None else
                    {k: v.to(device, non_blocking=True)
                     for k, v in cond_override.items()})
        cal, _ = dynamics_extras(batch, device)
        _, aux = model(x, sc, None, condition=cond, controls=None,
                       calibration_bold=cal, dynamics_horizon=1)
        return aux['dynamics_pred'][:, :, 0].float().cpu().numpy()

    for batch in dl:
        if 'cond_z' not in batch:
            raise ValueError('checkpoint 未启用条件注入（无 cond_z），B5 无对象。')
        pred_c = _forward(batch)
        pred_0 = _forward(batch, cond_override=zero_condition(batch))
        true = batch['y'][:, :, 0].float().numpy()
        x_last = batch['x_last'].float().numpy()
        # 条件嵌入（与 forward 内部同路的 _prepare_condition）
        cond_emb = model._prepare_condition(
            None, {'cond_z': batch['cond_z'].to(device),
                   'cond_code': batch['cond_code'].to(device),
                   'cond_mask': batch['cond_mask'].to(device)}, None)
        for i, sid in enumerate(batch['subj_id']):
            sid = str(sid)
            if sid not in acc:
                # Conditions are subject-level; a batch may contain multiple subjects.
                acc[sid] = {'dp': [], 'sig': [], 'mae_c': [], 'mae_0': [],
                            'film': film_norms(model, cond_emb[i:i + 1]),
                            'adarms': adarms_norms(model, cond_emb[i:i + 1])}
            e = acc[sid]
            e['dp'].append(float(np.abs(pred_c[i] - pred_0[i]).mean()))
            e['sig'].append(float(np.abs(true[i] - x_last[i]).mean()))
            e['mae_c'].append(float(np.abs(pred_c[i] - true[i]).mean()))
            e['mae_0'].append(float(np.abs(pred_0[i] - true[i]).mean()))

    rows = []
    for sid in [str(s) for s in subjects]:
        if sid not in acc:
            raise ValueError(f'视图中没有被试 {sid} 的锚点。')
        e = acc[sid]
        ad = e['adarms']
        dp, sig = float(np.mean(e['dp'])), float(np.mean(e['sig']))
        rows.append({
            'subject': sid,
            'ode_film_gamma': e['film']['ode_film_gamma'],
            'ode_film_beta': e['film']['ode_film_beta'],
            'encoder_adarms_gamma': ad['encoder_adarms_gamma'],
            'encoder_adarms_beta': ad['encoder_adarms_beta'],
            'delta_pred_abs': dp,
            'delta_pred_relative': dp / sig if sig > 1e-12 else float('nan'),
            'mae_cond': float(np.mean(e['mae_c'])),
            'mae_zero_cond': float(np.mean(e['mae_0'])),
        })
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.checkpoint import load_checkpoint_compat
    from analysis.rolling_one_step import (_build_loader_with_condition,
                                           build_dense_view,
                                           get_split_subjects)
    from analysis.evaluate import (build_model_from_args,
                                              load_condition_bundle,
                                              load_train_args)
    from main import condition_loader_kwargs
    from utils.common import set_seed

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--group', default='hc', choices=['hc', 'mdd'])
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--context_length', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--device', default='')
    ap.add_argument('--data_root', type=Path, default=None)
    ap.add_argument('--out_json', type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument('--out_csv', type=Path, default=DEFAULT_OUT_CSV)
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    device = torch.device(ns.device) if ns.device else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    args = load_train_args(ns.ckpt)
    bundle = load_condition_bundle(ns.ckpt)
    data_root = Path(ns.data_root) if ns.data_root else Path(args.data_root)
    set_seed(int(args.seed))
    model = build_model_from_args(args, device, condition_bundle=bundle)
    model, compat = load_checkpoint_compat(model, ns.ckpt, device, strict=True)
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f'权重加载不完整：{compat}')

    loader = _build_loader_with_condition(
        args, ns.group, data_root, ns.context_length, ns.batch_size, ns.num_workers,
        condition_loader_kwargs(args, bundle))
    subjects = get_split_subjects(loader, ns.split)
    log.info('[B5] group=%s split=%s | 被试 %d | 设备 %s',
             ns.group, ns.split, len(subjects), device)
    view = build_dense_view(loader, subjects, ns.context_length)
    rows = run_condition_usage(model, loader, subjects, view, device,
                               ns.batch_size, ns.num_workers)

    def col(key):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        ok = np.isfinite(vals)
        return {'mean': float(vals[ok].mean()) if ok.any() else None,
                'median': float(np.median(vals[ok])) if ok.any() else None,
                'max': float(vals[ok].max()) if ok.any() else None,
                'n_valid': int(ok.sum())}

    summary = {k: col(k) for k in (
        'ode_film_gamma', 'ode_film_beta', 'encoder_adarms_gamma',
        'encoder_adarms_beta', 'delta_pred_abs', 'delta_pred_relative',
        'mae_cond', 'mae_zero_cond')}
    rel = summary['delta_pred_relative']
    if rel['median'] is None:
        raise ValueError('条件敏感度的信号尺度均无效，无法计算相对敏感度。')
    near_zero = bool(rel['median'] < 0.01)
    payload = {
        'analysis': 'B5 FiLM 条件使用量（v2 §6.6）',
        'checkpoint': str(ns.ckpt), 'group': ns.group, 'split': ns.split,
        'n_subjects': len(rows),
        'zero_condition_def': '全缺失编码：cond_z=0、cond_code=Unknown(0)、cond_mask=0',
        'summary': summary,
        'near_zero_threshold': 0.01,
        'condition_usage_near_zero': near_zero,
        'conclusion': ('条件使用近零（Δpred/信号尺度 中位数 < 1%）→ 按 v2 §6.6 为'
                       '最终结论，不再开敏感性分析支线'
                       if near_zero else
                       '条件使用非零；报告幅度，不作为单独的解释支线展开'),
        'interpretation_note': 'Δpred 是条件通路的输出贡献（描述性），'
                               '不是临床效应方向或因果证据。',
    }
    ns.out_json.parent.mkdir(parents=True, exist_ok=True)
    ns.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    with ns.out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info('[B5] 完成 | Δpred/信号尺度 中位=%.5f 均值=%.5f | 近零判定=%s',
             rel['median'], rel['mean'], near_zero)
    print(json.dumps({k: payload[k] for k in (
        'n_subjects', 'summary', 'condition_usage_near_zero', 'conclusion')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
