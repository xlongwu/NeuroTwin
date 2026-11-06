"""CPM 字段条件使用量：FiLM 幅度、全缺失对照与预测残差。"""
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
log = logging.getLogger(__name__)
DEFAULT_OUT_JSON = Path('results/方案B/病理分析/condition_usage.json')
DEFAULT_OUT_CSV = Path('results/方案B/病理分析/condition_usage.csv')

def zero_condition(batch_cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """c=0：全缺失编码（z=0、code=Unknown(0)、mask=0）。"""
    return {'cond_z': torch.zeros_like(batch_cond['cond_z']), 'cond_code': torch.zeros_like(batch_cond['cond_code']), 'cond_mask': torch.zeros_like(batch_cond['cond_mask'])}

def film_norms(model, cond_emb: torch.Tensor) -> Dict[str, float]:
    """条件嵌入 [B, C] → CPM 头 FiLM 的 ‖γ‖/‖β‖（批内均值）。"""
    head = getattr(model, 'cpm_head', None)
    film = getattr(head, 'film', None) if head is not None else None
    if film is None or cond_emb is None:
        return {'cpm_film_gamma': float('nan'), 'cpm_film_beta': float('nan')}
    with torch.no_grad():
        scale, shift = film(cond_emb)
    return {'cpm_film_gamma': float(scale.norm(dim=-1).mean()), 'cpm_film_beta': float(shift.norm(dim=-1).mean())}


@torch.no_grad()
def run_condition_usage(model, loader, subjects, view, device, batch_size: int, num_workers: int) -> List[dict]:
    """逐被试计算 FiLM 幅度与 Δpred（同锚点两次前向）。"""
    from torch.utils.data import DataLoader
    model.eval()
    acc: Dict[str, Dict[str, list]] = {}
    dl = DataLoader(view, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))

    def _forward(batch, cond_override=None):
        x = batch['x'].to(device, non_blocking=True)
        cond = None
        if 'cond_z' in batch:
            cond = {'cond_z': batch['cond_z'].to(device, non_blocking=True), 'cond_code': batch['cond_code'].to(device, non_blocking=True), 'cond_mask': batch['cond_mask'].to(device, non_blocking=True)} if cond_override is None else {k: v.to(device, non_blocking=True) for k, v in cond_override.items()}
        aux_prediction, aux = model(x, condition=cond)
        return aux_prediction[:, :, 0].float().cpu().numpy()
    for batch in dl:
        if 'cond_z' not in batch:
            raise ValueError('checkpoint 未启用条件注入（无 cond_z），B5 无对象。')
        pred_c = _forward(batch)
        pred_0 = _forward(batch, cond_override=zero_condition(batch))
        true = batch['y'][:, :, 0].float().numpy()
        x_last = batch['x_last'].float().numpy()
        cond_emb = model._prepare_condition({'cond_z': batch['cond_z'].to(device), 'cond_code': batch['cond_code'].to(device), 'cond_mask': batch['cond_mask'].to(device)}, None)
        for i, sid in enumerate(batch['subj_id']):
            sid = str(sid)
            if sid not in acc:
                acc[sid] = {'dp': [], 'sig': [], 'mae_c': [], 'mae_0': [], 'film': film_norms(model, cond_emb[i:i + 1])}
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
        dp, sig = (float(np.mean(e['dp'])), float(np.mean(e['sig'])))
        rows.append({'subject': sid, 'cpm_film_gamma': e['film']['cpm_film_gamma'], 'cpm_film_beta': e['film']['cpm_film_beta'], 'delta_pred_abs': dp, 'delta_pred_relative': dp / sig if sig > 1e-12 else float('nan'), 'mae_cond': float(np.mean(e['mae_c'])), 'mae_zero_cond': float(np.mean(e['mae_0']))})
    return rows

def main(argv: Optional[List[str]]=None) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.rolling_one_step import _build_loader_with_condition, build_dense_view, get_split_subjects
    from model import load_model_checkpoint
    from main import condition_loader_kwargs
    from utils.common import set_seed
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    device = torch.device(ns.device) if ns.device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, args, bundle = load_model_checkpoint(ns.ckpt, device)
    data_root = Path(ns.data_root) if ns.data_root else Path(args.data_root)
    set_seed(int(args.seed))
    loader = _build_loader_with_condition(args, ns.group, data_root, ns.context_length, ns.batch_size, ns.num_workers, condition_loader_kwargs(args, bundle))
    subjects = get_split_subjects(loader, ns.split)
    log.info('[B5] group=%s split=%s | 被试 %d | 设备 %s', ns.group, ns.split, len(subjects), device)
    view = build_dense_view(loader, subjects, ns.context_length)
    rows = run_condition_usage(model, loader, subjects, view, device, ns.batch_size, ns.num_workers)

    def col(key):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        ok = np.isfinite(vals)
        return {'mean': float(vals[ok].mean()) if ok.any() else None, 'median': float(np.median(vals[ok])) if ok.any() else None, 'max': float(vals[ok].max()) if ok.any() else None, 'n_valid': int(ok.sum())}
    summary = {k: col(k) for k in ('cpm_film_gamma', 'cpm_film_beta', 'delta_pred_abs', 'delta_pred_relative', 'mae_cond', 'mae_zero_cond')}
    rel = summary['delta_pred_relative']
    if rel['median'] is None:
        raise ValueError('条件敏感度的信号尺度均无效，无法计算相对敏感度。')
    near_zero = bool(rel['median'] < 0.01)
    payload = {'analysis': 'B5 FiLM 条件使用量（v2 §6.6）', 'checkpoint': str(ns.ckpt), 'group': ns.group, 'split': ns.split, 'n_subjects': len(rows), 'zero_condition_def': '全缺失编码：cond_z=0、cond_code=Unknown(0)、cond_mask=0', 'summary': summary, 'near_zero_threshold': 0.01, 'condition_usage_near_zero': near_zero, 'conclusion': '条件使用近零（Δpred/信号尺度 中位数 < 1%）→ 按 v2 §6.6 为最终结论，不再开敏感性分析支线' if near_zero else '条件使用非零；报告幅度，不作为单独的解释支线展开', 'interpretation_note': 'Δpred 是条件通路的输出贡献（描述性），不是临床效应方向或因果证据。'}
    ns.out_json.parent.mkdir(parents=True, exist_ok=True)
    ns.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with ns.out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info('[B5] 完成 | Δpred/信号尺度 中位=%.5f 均值=%.5f | 近零判定=%s', rel['median'], rel['mean'], near_zero)
    print(json.dumps({k: payload[k] for k in ('n_subjects', 'summary', 'condition_usage_near_zero', 'conclusion')}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
