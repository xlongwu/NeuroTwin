"""[存档·描述性] H32 全 val 冻结推理缓存与 MAE/MSE/FC 指标汇总（--execute 执行）。

v2 重定向（2026-10-25）后的角色声明：

- 本模块属于 **H32 存档线**：消费 `archive/checkpoints/h32_best.pt` 的
  teacher-forced 全 horizon 预测，MSE/FC 汇总只作**描述性存档输出**，
  不再作为研究主线，也不引用 h32_best 产生新结论。
- v2 护栏指标族（滚动单步 lag-1：improvement / delta_corr / dir_acc 等）
  由 `analysis/rolling_one_step.py` 产生；偏离信号与门槛检查见
  `analysis/normative_deviation.py`。两者不可与本模块的 teacher-forced
  口径混比。

按《预测指标优化与病理可解释性分析实施方案》§4 实现：复用 checkpoint 的
正式评价协议（225人×16锚点，K64/H32），一次推理缓存 ODE/CPM 预测与真值，
再按统一层级（锚点→被试内平均→被试等权）计算分段 MAE/MSE、RMSE 两种口径、
FC 三项指标与辅助时间指标。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoint import load_checkpoint_compat
from analysis.metrics import NETWORK_GROUPS, get_aal116_labels
from analysis.next_point_eval import build_next_point_loader, run_inference
from analysis.evaluate import (build_model_from_args, load_condition_bundle,
                                          load_train_args)
from main import split_hash
from utils.common import set_seed

DEFAULT_CHECKPOINT = Path('archive/checkpoints/h32_best.pt')
SEGMENTS = (('H1', 0, 1), ('H1_8', 0, 8), ('H9_16', 8, 16), ('H17_32', 16, 32), ('H1_32', 0, 32))
EPS = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hierarchy(values: np.ndarray, subjects: np.ndarray) -> float:
    """锚点级指标数组 → 被试内平均 → 被试等权。"""
    return float(np.mean([np.mean(values[subjects == sid]) for sid in sorted(set(subjects))]))


def per_subject(values: np.ndarray, subjects: np.ndarray) -> dict:
    return {sid: float(np.mean(values[subjects == sid])) for sid in sorted(set(subjects))}


def hierarchy_nan(values: np.ndarray, subjects: np.ndarray) -> float:
    """同 hierarchy，但忽略 NaN（如常数目标上未定义的 PCC）。"""
    per = []
    for sid in sorted(set(subjects)):
        v = values[subjects == sid]
        v = v[np.isfinite(v)]
        if len(v):
            per.append(float(v.mean()))
    return float(np.mean(per)) if per else float('nan')


def temporal_pcc(pred: np.ndarray, target: np.ndarray, lo: int, hi: int) -> tuple:
    """[N,F,H]→逐锚点ROI时间PCC。常数预测在非常数目标上记0；常数目标排除并计数。"""
    p, t = pred[..., lo:hi].astype(np.float64), target[..., lo:hi].astype(np.float64)
    pc = p - p.mean(-1, keepdims=True)
    tc = t - t.mean(-1, keepdims=True)
    ps = np.sqrt((pc ** 2).mean(-1) + EPS)
    ts = np.sqrt((tc ** 2).mean(-1) + EPS)
    valid = ts > 1e-6
    r = (pc * tc).mean(-1) / (ps * ts)
    r = np.where((ps <= 1e-6) & valid, 0.0, r)
    if not valid.any():
        return np.full(p.shape[:2], np.nan), 0, 1.0
    return np.where(valid, r, np.nan), int(valid.sum()), float(1.0 - valid.mean())


def fc_matrix(series: np.ndarray) -> np.ndarray:
    """[F,H] → signed Pearson FC；常数ROI产生NaN行，由调用方按有效性排除。"""
    return np.corrcoef(series)


def anchor_fc_metrics(pred: np.ndarray, target: np.ndarray, iu: tuple) -> dict:
    """逐锚点FC三项：上三角边 |FC| 误差与边模式相关；无效锚点记NaN。"""
    out = {'fc_mae': [], 'fc_mse': [], 'fc_edge_pcc': [], 'valid': []}
    for p, t in zip(pred, target):
        cp, ct = fc_matrix(p), fc_matrix(t)
        e1, e2 = cp[iu], ct[iu]
        ok = np.isfinite(e1) & np.isfinite(e2)
        if ok.sum() < 100 or np.std(e1[ok]) < EPS or np.std(e2[ok]) < EPS:
            out['valid'].append(False)
            out['fc_mae'].append(np.nan)
            out['fc_mse'].append(np.nan)
            out['fc_edge_pcc'].append(np.nan)
            continue
        diff = e1[ok] - e2[ok]
        out['valid'].append(True)
        out['fc_mae'].append(float(np.abs(diff).mean()))
        out['fc_mse'].append(float((diff ** 2).mean()))
        out['fc_edge_pcc'].append(float(np.corrcoef(e1[ok], e2[ok])[0, 1]))
    return {k: np.asarray(v) for k, v in out.items()}


def headline_metrics(pred: np.ndarray, target: np.ndarray, subjects: np.ndarray,
                     iu: tuple) -> dict:
    """§4.1 指标表全部条目；层级统一为锚点→被试内→被试等权。"""
    err = np.abs(pred.astype(np.float64) - target)
    sq = err ** 2
    m = {'n_anchors': int(len(pred)), 'n_subjects': int(len(set(subjects)))}
    for name, lo, hi in SEGMENTS:
        m[f'MAE_{name}'] = hierarchy(err[:, :, lo:hi].mean((1, 2)), subjects)
        m[f'MSE_{name}'] = hierarchy(sq[:, :, lo:hi].mean((1, 2)), subjects)
        # RMSE两种口径名称分立：总体=被试等权MSE开方；逐被试=先开方再平均
        m[f'RMSE_overall_{name}'] = float(np.sqrt(m[f'MSE_{name}']))
        m[f'RMSE_subjmean_{name}'] = hierarchy(
            np.sqrt(sq[:, :, lo:hi].mean((1, 2))), subjects)
    # FC（H1–32窗）
    fc = anchor_fc_metrics(pred, target, iu)
    valid = fc['valid'].astype(bool)
    m['FC_valid_fraction'] = float(valid.mean())
    if valid.any():
        for key in ('fc_mae', 'fc_mse', 'fc_edge_pcc'):
            m[f'FC_{key.split("_", 1)[1]}'] = hierarchy(fc[key][valid], subjects[valid])
        m['FC_rmse'] = float(np.sqrt(hierarchy(fc['fc_mse'][valid], subjects[valid])))
    # 辅助时间指标
    for name, lo, hi in (('H1_32', 0, 32), ('H17_32', 16, 32)):
        r, n_valid, frac_const = temporal_pcc(pred, target, lo, hi)
        with np.errstate(invalid='ignore'):
            per_anchor = np.nanmean(r.reshape(len(pred), -1), axis=1)
        m[f'PCC_{name}'] = hierarchy(per_anchor, subjects)
        m[f'PCC_{name}_target_const_fraction'] = frac_const
    pd_sig = np.sign(np.diff(pred, axis=-1))
    td_sig = np.sign(np.diff(target, axis=-1))
    nonzero = td_sig != 0
    dir_acc = (pd_sig == td_sig)[nonzero]
    step_ids = np.repeat(np.arange(len(pred)), nonzero.reshape(len(pred), -1).sum(1))
    m['direction_accuracy'] = hierarchy(dir_acc, step_ids)
    std_ratio = pred.std(-1) / (target.std(-1) + EPS)
    m['std_ratio_H1_32'] = hierarchy(std_ratio.mean(1), subjects)
    m['bias_H1_32'] = hierarchy((pred - target).mean((-1, -2)), subjects)
    diff_rms_ratio = np.sqrt((np.diff(pred, axis=-1) ** 2).mean(-1) + EPS) / (
        np.sqrt((np.diff(target, axis=-1) ** 2).mean(-1) + EPS))
    m['diff_rms_ratio_H1_32'] = hierarchy(diff_rms_ratio.mean(1), subjects)
    return m


def per_subject_table(pred: np.ndarray, target: np.ndarray, subjects: np.ndarray,
                      iu: tuple) -> list:
    err = np.abs(pred.astype(np.float64) - target)
    fc = anchor_fc_metrics(pred, target, iu)
    r32, _, _ = temporal_pcc(pred, target, 0, 32)
    with np.errstate(invalid='ignore'):
        pcc32 = np.nanmean(r32.reshape(len(pred), -1), axis=1)
    rows = []
    for sid in sorted(set(subjects)):
        sel = subjects == sid
        fcs = fc['fc_edge_pcc'][sel & fc['valid'].astype(bool)]
        rows.append({
            'subject': sid, 'n_anchors': int(sel.sum()),
            'MAE_H1_32': float(err[sel].mean()),
            'MSE_H1_32': float((err[sel] ** 2).mean()),
            'FC_edge_pcc': float(fcs.mean()) if len(fcs) else float('nan'),
            'PCC_H1_32': float(np.nanmean(pcc32[sel])),
            'std_ratio': float((pred[sel].std(-1) / (target[sel].std(-1) + EPS)).mean()),
            'bias': float((pred[sel] - target[sel]).mean()),
        })
    return rows


def per_roi_table(pred: np.ndarray, target: np.ndarray, subjects: np.ndarray) -> list:
    err = np.abs(pred.astype(np.float64) - target)
    r32, _, _ = temporal_pcc(pred, target, 0, 32)
    labels = get_aal116_labels()
    rows = []
    for f in range(pred.shape[1]):
        with np.errstate(invalid='ignore'):
            pcc_f = r32[:, f]
        rows.append({
            'roi_index': f, 'label': labels[f] if f < len(labels) else f'AAL{f+1}',
            'MAE_H1_32': hierarchy(err[:, f, :].mean(1), subjects),
            'MAE_H17_32': hierarchy(err[:, f, 16:].mean(1), subjects),
            'MSE_H1_32': hierarchy((err[:, f, :] ** 2).mean(1), subjects),
            'PCC_H1_32': hierarchy_nan(pcc_f, subjects),
            'std_ratio': hierarchy((pred[:, f, :].std(-1) / (target[:, f, :].std(-1) + EPS)), subjects),
            'bias': hierarchy((pred[:, f, :] - target[:, f, :]).mean(1), subjects),
        })
    return rows


def network_fc_table(pred: np.ndarray, target: np.ndarray, iu: tuple) -> list:
    """按既有 NETWORK_GROUPS 汇总边级 FC MAE（组内/组间/含未分组ROI）。"""
    n_roi = pred.shape[1]
    group_of = np.full(n_roi, -1, int)
    for gi, members in enumerate(NETWORK_GROUPS.values()):
        for aal_id in members:
            if 0 < aal_id <= n_roi:
                group_of[aal_id - 1] = gi
    i, j = iu
    gi, gj = group_of[i], group_of[j]
    kind = np.where((gi >= 0) & (gj >= 0) & (gi == gj), 'within',
                    np.where((gi >= 0) & (gj >= 0), 'between', 'unassigned'))
    rows = []
    for label in ('within', 'between', 'unassigned'):
        sel = kind == label
        if not sel.any():
            continue
        errs = []
        for p, t in zip(pred, target):
            cp, ct = fc_matrix(p), fc_matrix(t)
            e1, e2 = cp[iu][sel], ct[iu][sel]
            ok = np.isfinite(e1) & np.isfinite(e2)
            if ok.any():
                errs.append(float(np.abs(e1[ok] - e2[ok]).mean()))
        rows.append({'edge_class': label, 'n_edges': int(sel.sum()),
                     'FC_mae': float(np.mean(errs)) if errs else float('nan')})
    return rows


def run(checkpoint: Path, out_dir: Path, split: str, device: str,
        data_root=None, num_workers: int = 0) -> dict:
    cache_path = out_dir / 'prediction_cache.pt'
    if cache_path.exists():
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
        print(f'复用已有推理缓存: {cache_path}')
    else:
        args = load_train_args(checkpoint)
        set_seed(args.seed)
        if data_root is not None:
            args.data_root = str(data_root)
        args.num_workers = num_workers
        args.eval_rollout_steps = 32
        bundle = load_condition_bundle(checkpoint)
        model = build_model_from_args(args, torch.device(device), condition_bundle=bundle)
        model, _ = load_checkpoint_compat(model, str(checkpoint), torch.device(device),
                                          strict=True)
        loader, _, owner = build_next_point_loader(args, split, condition_bundle=bundle)
        meta = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
        current_hash = split_hash(owner, args)
        if current_hash != meta.get('condition', {}).get('split_hash'):
            raise ValueError('当前数据划分与 checkpoint 不一致')
        data = run_inference(model, loader, torch.device(device), offsets=[1],
                             rollout_steps=0, eval_rollout=False, collect_cpm=True,
                             dynamics_horizon=32)
        if not (data['future_mask'][:, :32] > 0).all():
            raise ValueError('正式汇总要求完整 H32 锚点')
        payload = {k: data[k] for k in ('ode_pred', 'cpm_pred', 'future', 'subj_ids',
                                        'cutoff', 'future_mask')}
        del model, loader, owner, data
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        print(f'推理缓存已保存: {cache_path}')

    subjects = payload['subj_ids'].astype(str)
    iu = np.triu_indices(payload['ode_pred'].shape[1], 1)
    report = {
        'plan_section': '[存档·描述性] H32 teacher-forced 指标表（v2 主线见 '
                        'analysis/rolling_one_step.py）',
        'interpretation': 'teacher-forced 口径的描述性存档输出，不代表 v2 滚动'
                          '单步护栏指标，不可与 rolling_one_step 结果混比',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'split': split, 'aggregation': '锚点内ROI/时间平均 → 被试内平均 → 被试等权',
        'aggregation_note': 'FC边模式相关为逐锚点计算后同层级汇总；PCC常数规则见方案',
        'direction_accuracy_def': 'sign(Δpred)==sign(Δtarget)，仅统计Δtarget≠0的步',
        'heads': {
            'ODE': headline_metrics(payload['ode_pred'], payload['future'], subjects, iu),
            'CPM': headline_metrics(payload['cpm_pred'], payload['future'], subjects, iu),
        },
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + '\n')

    def write_csv(name: str, rows: list) -> None:
        import csv
        if not rows:
            return
        with (out_dir / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for head, key in (('ode', 'ode_pred'), ('cpm', 'cpm_pred')):
        write_csv(f'per_subject_{head}.csv',
                  per_subject_table(payload[key], payload['future'], subjects, iu))
        write_csv(f'per_roi_{head}.csv',
                  per_roi_table(payload[key], payload['future'], subjects))
        write_csv(f'network_fc_{head}.csv',
                  network_fc_table(payload[key], payload['future'], iu))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('results/prediction_optimization/val_full'))
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(f'Dry run: {args.split} 全锚点冻结推理与指标汇总 → {args.out_dir}；'
              '加 --execute 执行')
        return
    report = run(args.checkpoint, args.out_dir, args.split, args.device,
                 args.data_root, args.num_workers)
    print(json.dumps(report['heads'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
