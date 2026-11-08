"""方案 B 固定锚点 H32 评价：逐时距、阶段轨迹、基线及 ROI 归因。"""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from main import build_data_loader, condition_loader_kwargs, forecast_batch
from train.forecast_metrics import H32_STAGE_LABELS, h32_stage_metrics
from utils.common import split_label

EPS=1e-8
DEFAULT_DIRECTION_DEADZONE=0.05

def _subject_mean(values: np.ndarray, subj_idx: np.ndarray) -> np.ndarray:
    """把任务级数值按被试聚合为被试均值 [n_unique]。"""
    uniq = np.unique(subj_idx)
    return np.array([values[subj_idx == s].mean() for s in uniq], dtype=np.float64)


def per_task_next_state(pred, target, mask, anchor=None):
    """next-state 逐 (task, offset) 指标数组，形状均为 [N, H]（无效位置 NaN）。

    pred/target: [N, F, H]（单个时间点预测，原始空间）；mask: [N, H]。
    指标：mae / rmse / spatial_pcc（沿 ROI 维相关）/ r2（相对 ROI 方差）。
    若给定 ``anchor`` [N, F]（context 末位 = 预测锚点 x_t），额外统计
    ``delta_direction``：逐 ROI 变化方向 sign(x̂−x_t) 与 sign(x_true−x_t) 的一致比例
    （prompt §二十；delta MAE 与 MAE 在 delta 口径下恒等，故不重复报告）。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致：{pred.shape} vs {target.shape}")
    n, f, h = pred.shape
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.ones((n, h), bool))
    if valid.shape != (n, h):
        raise ValueError(f"mask 形状 {valid.shape} 应为 {(n, h)}")

    err = pred - target
    mae = np.abs(err).mean(axis=1)                       # [N,H]
    rmse = np.sqrt((err ** 2).mean(axis=1))
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1)) + EPS
    pcc = (pc * tc).sum(axis=1) / denom
    r2 = 1.0 - (err ** 2).sum(axis=1) / ((tc ** 2).sum(axis=1) + EPS)

    arrays = {'mae': mae, 'rmse': rmse, 'spatial_pcc': pcc, 'r2': r2}
    if anchor is not None:
        a = np.asarray(anchor, dtype=pred.dtype)[:, :, None]     # [N,F,1]
        agree = np.sign(pred - a) == np.sign(target - a)         # [N,F,H]
        arrays['delta_direction'] = agree.mean(axis=1)           # [N,H]
    for k in arrays:
        arrays[k] = np.where(valid, arrays[k], np.nan)
    return arrays


def per_task_head_metrics(pred, target, mask, anchor,
                          direction_deadzone=DEFAULT_DIRECTION_DEADZONE):
    """CPM 预测或相同锚点基线的逐 horizon 指标 [N, H]。

    与 ``per_task_next_state`` 的区别是显式区分两种差分口径：

    - ``delta_mae_anchor``：``|(ŷ_h − x_t) − (y_h − x_t)|``，在数学上恒等于
      ``MAE``（共同锚点抵消），保留该列只为证明/核对这一点；
    - ``delta_mae_step``：相邻帧一阶差分误差。``Δŷ_1=ŷ_1−x_t``、
      ``Δŷ_h=ŷ_h−ŷ_(h−1)``（真值同理），与训练侧
      ``temporal_difference_huber`` 完全同口径。只有该指标能反映模型是否学到
      局部时间动力学，而不是单纯复制锚点。

    ``direction_accuracy_step`` / ``direction_accuracy_anchor`` 分别对上述两种
    差分统计 sign 一致率；真实差分绝对值 <= ``direction_deadzone`` 的样本不计入，
    避免近零噪声左右方向指标。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致：{pred.shape} vs {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"pred/target 期望 [N,F,H]，收到 {pred.shape}")
    n, f, h = pred.shape
    if anchor is None:
        raise ValueError("方向/差分指标需要 anchor [N,F]（context 末位 x_t）")
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (n, f):
        raise ValueError(f"anchor 形状应为 {(n, f)}，收到 {anchor.shape}")
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.ones((n, h), bool))
    if valid.shape != (n, h):
        raise ValueError(f"mask 形状 {valid.shape} 应为 {(n, h)}")

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    mae = np.abs(err).mean(axis=1)
    rmse = np.sqrt((err ** 2).mean(axis=1))
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1)) + EPS
    spatial_pcc = (pc * tc).sum(axis=1) / denom
    r2 = 1.0 - (err ** 2).sum(axis=1) / ((tc ** 2).sum(axis=1) + EPS)
    var_ratio = np.where(
        target.var(axis=1) > EPS, pred.var(axis=1) / (target.var(axis=1) + EPS),
        np.nan)

    # 相邻帧差分：h=1 用真实锚点，h>=2 用各自轨迹的前一帧
    prev_p = np.concatenate([anchor[:, :, None], pred[:, :, :-1]], axis=2)
    prev_t = np.concatenate([anchor[:, :, None], target[:, :, :-1]], axis=2)
    d_pred = pred - prev_p
    d_true = target - prev_t
    step_valid = valid.copy()
    if h > 1:
        step_valid[:, 1:] &= valid[:, :-1]
    d_err = np.abs(d_pred - d_true)
    delta_mae_step = np.where(
        step_valid, d_err.mean(axis=1), np.nan)
    # 数学恒等列：Δ 相对锚点，与点值 MAE 相同
    delta_mae_anchor = mae.copy()

    eligible_step = (step_valid[:, None, :]
                     & (np.abs(d_true) > float(direction_deadzone)))
    agree_step = (np.sign(d_pred) == np.sign(d_true)) & eligible_step
    n_eligible_step = eligible_step.sum(axis=1)
    da_step = np.where(
        n_eligible_step > 0,
        agree_step.sum(axis=1) / np.maximum(n_eligible_step, 1), np.nan)

    h_delta = pred - anchor[:, :, None]
    h_delta_true = target - anchor[:, :, None]
    eligible_anchor = (valid[:, None, :]
                       & (np.abs(h_delta_true) > float(direction_deadzone)))
    agree_anchor = (np.sign(h_delta) == np.sign(h_delta_true)) & eligible_anchor
    n_eligible_anchor = eligible_anchor.sum(axis=1)
    da_anchor = np.where(
        n_eligible_anchor > 0,
        agree_anchor.sum(axis=1) / np.maximum(n_eligible_anchor, 1), np.nan)

    # 变化幅度比：预测一阶差分 / 真实一阶差分（<1 表示趋平，>1 表示过冲）
    mag_p = np.abs(d_pred).mean(axis=1)
    mag_t = np.abs(d_true).mean(axis=1)
    mag_ratio = np.where(step_valid & (mag_t > EPS), mag_p / (mag_t + EPS), np.nan)

    arrays = {
        'mae': mae, 'rmse': rmse, 'spatial_pcc': spatial_pcc, 'r2': r2,
        'variance_ratio': var_ratio,
        'delta_mae_anchor': delta_mae_anchor,
        'delta_mae_step': delta_mae_step,
        'delta_direction': da_step,
        'direction_accuracy_step': da_step,
        'direction_accuracy_anchor': da_anchor,
        'delta_magnitude_ratio': mag_ratio,
    }
    for key in ('mae', 'rmse', 'spatial_pcc', 'r2', 'delta_mae_anchor'):
        arrays[key] = np.where(valid, arrays[key], np.nan)
    return arrays


def _step_deltas(pred, target, anchor):
    """相邻帧一阶差分（与 temporal_difference_huber 同口径）。"""
    prev_p = np.concatenate([anchor[:, :, None], pred[:, :, :-1]], axis=2)
    prev_t = np.concatenate([anchor[:, :, None], target[:, :, :-1]], axis=2)
    return pred - prev_p, target - prev_t


def _step_valid(mask):
    """相邻帧差分有效性：h 与 h−1 的真值都有效；h=1 只需自身有效。"""
    mask = np.asarray(mask, dtype=bool)
    valid = mask.copy()
    if mask.shape[1] > 1:
        valid[:, 1:] &= mask[:, :-1]
    return valid


def amplitude_quantile_edges(target, mask, anchor, n_bins=5):
    """用真实相邻帧变化幅度的分位数确定幅度分层边界。

    模型与基线使用相同的 test 锚点与真值；返回长度
    ``n_bins+1``（首边界 0、末边界 +inf），供 ``per_task_amplitude_diagnostics``
    对模型与基线做同一套分层。
    """
    if int(n_bins) < 1:
        raise ValueError(f'n_bins 必须 >= 1，收到 {n_bins}')
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    mask = (np.asarray(mask, dtype=bool) if mask is not None
            else np.ones(target.shape[:2], bool))
    _, dt = _step_deltas(target, target, anchor)
    valid = _step_valid(mask)
    vals = np.abs(dt)[np.broadcast_to(valid[:, None, :], dt.shape)]
    if vals.size == 0:
        raise ValueError('真实差分没有有效元素，无法确定幅度分层边界')
    edges = np.quantile(vals, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[0] = 0.0
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges.tolist()


def per_task_amplitude_diagnostics(pred, target, mask, anchor,
                                   amp_edges=None, n_bins=5):
    """逐 anchor 的幅度/动态诊断，回答“方向对但幅度是否被压缩”。

    输出的标量数组均为 ``[N]``，逐 horizon 数组为 ``[N,H]``，分层数组为
    ``[N,B]``：

    - ``delta_std_ratio``：``std(Δŷ)/std(Δy)``（H1–H8 × 全部 ROI 的有效元素），
      <1 表示变化幅度整体偏小；
    - ``delta_std_ratio_h1`` / ``_h2_8``：分别看单步与多步推进的幅度比。
      H1≈1 而 H2–H8<1 说明压缩来自状态推进/条件均值，而非单步 readout；
    - ``delta_pcc`` / ``delta_pcc_h2_8``：``Corr(Δŷ, Δy)``，衡量是否跟随变化过程；
    - ``temporal_variance_ratio``：逐 ROI 在有效未来轨迹上的
      ``Var(ŷ_t..)/Var(y_t..)`` 再对 ROI 平均；<1 表示轨迹方差塌缩；
    - ``trajectory_variance_ratio_flat``：把 H×ROI 展平后的方差比（对照口径）；
    - ``amp_bin_mae``：按真实 ``|Δy|`` 分位分层的 Delta MAE（默认 5 层），
      可直接检查“大幅变化是否尤其欠拟合”。

    ``amp_edges`` 由 :func:`amplitude_quantile_edges` 在真值上预先算好，保证
    模型与基线、不同实验臂使用完全相同的分层边界。
    """
    if pred.shape != target.shape:
        raise ValueError(f'pred/target 形状不一致：{pred.shape} vs {target.shape}')
    if pred.ndim != 3:
        raise ValueError(f'pred/target 期望 [N,F,H]，收到 {pred.shape}')
    n, f, h = pred.shape
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (n, f):
        raise ValueError(f'anchor 形状应为 {(n, f)}，收到 {anchor.shape}')
    mask = (np.asarray(mask, dtype=bool) if mask is not None
            else np.ones((n, h), bool))
    if mask.shape != (n, h):
        raise ValueError(f'mask 形状 {mask.shape} 应为 {(n, h)}')
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dp, dt = _step_deltas(pred, target, anchor)
    step_valid = _step_valid(mask)
    if amp_edges is None:
        amp_edges = amplitude_quantile_edges(target, mask, anchor, n_bins=n_bins)
    amp_edges = np.asarray(amp_edges, dtype=np.float64)
    nb = len(amp_edges) - 1

    keys = ('delta_std_ratio', 'delta_pcc', 'temporal_variance_ratio',
            'trajectory_variance_ratio_flat',
            'delta_std_ratio_h1', 'delta_std_ratio_h2_8',
            'delta_pcc_h1', 'delta_pcc_h2_8',
            'mean_abs_pred_delta', 'mean_abs_true_delta')
    out = {k: np.full(n, np.nan, dtype=np.float64) for k in keys}
    out['delta_std_ratio_h'] = np.full((n, h), np.nan)
    out['delta_pcc_h'] = np.full((n, h), np.nan)
    out['amp_bin_mae'] = np.full((n, nb), np.nan)
    out['amp_bin_count'] = np.zeros((n, nb), dtype=np.float64)
    out['amp_bin_true_abs_mean'] = np.full((n, nb), np.nan)

    for i in range(n):
        valid = step_valid[i]
        if not valid.any():
            continue
        m3 = np.broadcast_to(valid[None, :], (f, h))
        x = dp[i][m3]
        y = dt[i][m3]
        sx, sy = float(np.std(x)), float(np.std(y))
        if sy > EPS:
            out['delta_std_ratio'][i] = sx / sy
        if sx > EPS and sy > EPS:
            out['delta_pcc'][i] = float(np.corrcoef(x, y)[0, 1])
        out['mean_abs_pred_delta'][i] = float(np.abs(x).mean())
        out['mean_abs_true_delta'][i] = float(np.abs(y).mean())

        # 轨迹方差比：逐 ROI 在有效 horizon 上的时间方差
        vh = mask[i]
        if vh.sum() >= 3:
            pv = pred[i][:, vh]
            tv = target[i][:, vh]
            vp, vt = pv.var(axis=1), tv.var(axis=1)
            ok = vt > EPS
            if ok.any():
                out['temporal_variance_ratio'][i] = float(
                    np.mean(vp[ok] / (vt[ok] + EPS)))
            fp = pred[i][:, vh].reshape(-1)
            ft = target[i][:, vh].reshape(-1)
            if ft.var() > EPS:
                out['trajectory_variance_ratio_flat'][i] = float(
                    fp.var() / (ft.var() + EPS))

        # H1 与 H2-8 分开看压缩发生在哪一段
        for key, sel in (('h1', np.array([0])),
                         ('h2_8', np.arange(1, h))):
            if h == 1 and key == 'h2_8':
                continue
            sel = sel[valid[sel]]
            if sel.size == 0:
                continue
            xx = dp[i][:, sel].reshape(-1)
            yy = dt[i][:, sel].reshape(-1)
            syy = float(np.std(yy))
            if syy > EPS:
                out[f'delta_std_ratio_{key}'][i] = float(np.std(xx)) / syy
            if xx.std() > EPS and syy > EPS:
                out[f'delta_pcc_{key}'][i] = float(np.corrcoef(xx, yy)[0, 1])

        # 逐 horizon 的幅度比与 PCC（跨 ROI）
        for hi in range(h):
            if not valid[hi]:
                continue
            xx = dp[i][:, hi]
            yy = dt[i][:, hi]
            if yy.std() > EPS:
                out['delta_std_ratio_h'][i, hi] = xx.std() / yy.std()
            if xx.std() > EPS and yy.std() > EPS:
                out['delta_pcc_h'][i, hi] = float(np.corrcoef(xx, yy)[0, 1])

        # 按真实变化幅度分位分层
        mag = np.abs(y)
        err = np.abs(x - y)
        which = np.digitize(mag, amp_edges[1:-1], right=False)
        for b in range(nb):
            sel = which == b
            cnt = int(sel.sum())
            out['amp_bin_count'][i, b] = cnt
            if cnt:
                out['amp_bin_mae'][i, b] = float(err[sel].mean())
                out['amp_bin_true_abs_mean'][i, b] = float(mag[sel].mean())
    return out


def aggregate_columns(arrays, subj_idx, labels):
    """把 [N, C] 指标数组聚合为 {metric: {label: {...}}}（被试级为主口径）。

    每个 label 报告：被试级 mean/std/median（先被试内平均、再跨被试统计）+
    任务级 pooled mean + 任务数/被试数。
    """
    out = {}
    for metric, arr in arrays.items():
        block = {}
        for c, label in enumerate(labels):
            col = arr[:, c]
            ok = ~np.isnan(col)
            if not ok.any():
                continue
            vals = col[ok]
            subj_vals = _subject_mean(vals, subj_idx[ok])
            block[str(label)] = {
                'mean': float(subj_vals.mean()),
                'std': float(subj_vals.std()),
                'median': float(np.median(subj_vals)),
                'task_mean': float(vals.mean()),
                'n_tasks': int(ok.sum()),
                'n_subjects': int(len(subj_vals)),
            }
        if block:
            out[metric] = block
    return out


def fit_ar1_from_train(base_dataset, train_subjects):
    """用 **train subjects** 的连续序列拟合逐 ROI AR(1)：x_(t+1) = a_i·x_t + b_i。

    最小二乘闭式解（逐 ROI 汇总 Σx/Σy/Σx²/Σxy 后一次求解），不使用任何
    val/test 被试；返回 {'a': [F], 'b': [F], 'n_pairs': int}。
    """
    sx = sy = sxx = sxy = None
    n_pairs = 0
    for sid in train_subjects:
        series = np.asarray(base_dataset.get_series(sid), dtype=np.float64)
        if series.ndim != 2 or series.shape[1] < 2:
            continue
        x, y = series[:, :-1], series[:, 1:]
        if sx is None:
            f = series.shape[0]
            sx = np.zeros(f); sy = np.zeros(f)
            sxx = np.zeros(f); sxy = np.zeros(f)
        sx += x.sum(axis=1); sy += y.sum(axis=1)
        sxx += (x * x).sum(axis=1); sxy += (x * y).sum(axis=1)
        n_pairs += x.shape[1]
    if sx is None or n_pairs == 0:
        raise ValueError("AR(1) 拟合失败：训练集没有可用的连续序列。")
    denom = n_pairs * sxx - sx * sx
    a = np.where(np.abs(denom) > EPS, (n_pairs * sxy - sx * sy) / denom, 0.0)
    b = (sy - a * sx) / float(n_pairs)
    return {'a': a.astype(np.float32), 'b': b.astype(np.float32),
            'n_pairs': int(n_pairs)}


def _trajectory_baselines(x_last, x_prev, ar1, horizon, history_mean=None):
    """构造 persistence / trend / AR(1) 的全 horizon 轨迹 [N,F,H]。"""
    x_last = np.asarray(x_last, dtype=np.float64)
    x_prev = np.asarray(x_prev, dtype=np.float64)
    h = int(horizon)
    n, f = x_last.shape
    k = np.arange(1, h + 1, dtype=np.float64)
    out = {
        'persistence': np.repeat(x_last[:, :, None], h, axis=2),
        'trend': (x_last[:, :, None]
                  + k[None, None, :] * (x_last - x_prev)[:, :, None]),
    }
    if history_mean is not None:
        out['history_mean'] = np.repeat(np.asarray(history_mean)[:, :, None], h, axis=2)
    if ar1 is not None:
        a = np.asarray(ar1['a'], dtype=np.float64).reshape(1, f)
        b = np.asarray(ar1['b'], dtype=np.float64).reshape(1, f)
        cur = x_last.copy()
        steps = []
        for _ in range(h):
            cur = a * cur + b
            steps.append(cur.copy())
        out['ar1'] = np.stack(steps, axis=2)
    return out


def bh_fdr(pvalues):
    """Benjamini–Hochberg FDR 校正，返回与输入同序的 q 值（含单调化）。"""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind='mergesort')
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]          # 保证 q 单调不减
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def build_next_point_loader(args,split,condition_bundle=None):
    kwargs=condition_loader_kwargs(args,condition_bundle)
    loader=build_data_loader(args,kwargs['condition_base_stats'],inherit_all=True)
    if split=='test':return loader.get_test(),loader.get_test_subjects(),loader
    if split=='val':return loader.get_val(),loader.get_val_subjects(),loader
    if split=='train':
        from torch.utils.data import DataLoader
        from utils.dataloader import NextPointEvalView
        subjects=loader.get_train_subjects()
        view=NextPointEvalView(loader.base_dataset,loader._expand_subjects_to_sample_indices(subjects),
            context_length=args.eval_context_length,anchors_per_subject=args.eval_anchors_per_subject,
            rollout_tasks_per_subject=0,rollout_steps=0,all_future_steps=args.cpm_horizon,
            anchor_min_future_steps=args.cpm_horizon)
        options=dict(batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers,
                     pin_memory=args.pin_memory)
        if args.num_workers:options.update(persistent_workers=args.persistent_workers,prefetch_factor=args.prefetch_factor)
        return DataLoader(view,**options),subjects,loader
    raise ValueError(f'未知 split: {split}')


@torch.no_grad()
def run_inference(model,data_loader,device,save_history=False):
    model.eval()
    chunks=defaultdict(list)
    for batch in tqdm(data_loader,desc='B H32 inference',leave=False):
        x,target,mask,anchor,cond=forecast_batch(batch,device)
        pred,_=model(x,condition=cond)
        for name,tensor in dict(pred=pred,target=target,mask=mask,anchor=anchor,
                                previous=x[:,:,0,-2],history_mean=x[:,:,0].mean(-1)).items():
            chunks[name].append(tensor.float().cpu().numpy())
        if save_history:
            chunks['history'].append(x[:, :, 0, :].float().cpu().numpy())
        chunks['subjects'].extend(map(str,batch['subj_id']))
        chunks['cutoff'].append(batch['cutoff'].numpy())
    if not chunks['subjects']:raise ValueError('评估集合为空')
    return {k:np.asarray(v,dtype=str) if k=='subjects' else np.concatenate(v)
            for k,v in chunks.items()}


def _write_table(path,arrays,subjects,labels,by_subject=False,model_name='CPM'):
    rows=[]
    if by_subject:
        for sid in np.unique(subjects):
            for i,label in enumerate(labels):
                row={'模型':model_name,'被试':sid,'预测范围':label}
                for metric,arr in arrays.items():
                    values=arr[subjects==sid,i]
                    values=values[np.isfinite(values)]
                    row[metric]=float(values.mean()) if values.size else None
                rows.append(row)
    else:
        agg=aggregate_columns(arrays,subjects,labels)
        for label in labels:
            row={'模型':model_name,'预测范围':label}
            for metric,block in agg.items():
                row[metric]=block.get(label,{}).get('mean')
            rows.append(row)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@torch.no_grad()
def evaluate_split(model,data_loader,args,device,out_dir,split,ar1=None,save_arrays=False,
                   compute_importance=False,importance_max_tasks=256,
                   checkpoint_identity=None,split_identity=None):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    tables=out_dir/'表格';tables.mkdir(exist_ok=True)
    data=run_inference(model,data_loader,device,save_history=save_arrays)
    pred,target,mask,anchor,subjects=(data[k] for k in ('pred','target','mask','anchor','subjects'))
    labels=[f'H{i}' for i in range(1,args.cpm_horizon+1)]
    steps=per_task_head_metrics(pred,target,mask,anchor)
    stages={k:v.numpy() for k,v in h32_stage_metrics(
        torch.as_tensor(pred),torch.as_tensor(target),torch.as_tensor(mask),
        torch.as_tensor(anchor),include_local=True).items()}
    report=dict(protocol=dict(model='CPM',horizon=args.cpm_horizon,
        context_length=args.eval_context_length,split=split,
        mode=args.mode, tr_seconds=getattr(args, 'tr_seconds', 2.),
        calibration_length=args.calibration_length,
        multiscale_phase=getattr(args, 'multiscale_phase', False),
        checkpoint=checkpoint_identity, split_hash=split_identity,
        n_subjects=len(np.unique(subjects)),n_anchors=len(subjects),
        aggregation='per anchor → within subject mean → equal subject mean',
        normalization='Raw first-32TR calibration; context-only RevIN',
        pcc='spatial PCC along ROI at each horizon; temporal PCC along time within each stage'),
        horizon_metrics=aggregate_columns(steps,subjects,labels),
        trajectory_metrics=aggregate_columns(stages,subjects,H32_STAGE_LABELS))
    baselines=_trajectory_baselines(anchor,data['previous'],ar1,args.cpm_horizon,data['history_mean'])
    # trend 保留为可解释的外推对照；Persistence/历史均值/AR(1) 与模型共用锚点。
    report['baselines']={}
    for name,trajectory in baselines.items():
        bs=per_task_head_metrics(trajectory,target,mask,anchor)
        bt={k:v.numpy() for k,v in h32_stage_metrics(
            torch.as_tensor(trajectory),torch.as_tensor(target),torch.as_tensor(mask),
            torch.as_tensor(anchor),include_local=True).items()}
        report['baselines'][name]=dict(horizon_metrics=aggregate_columns(bs,subjects,labels),
            trajectory_metrics=aggregate_columns(bt,subjects,H32_STAGE_LABELS))
        _write_table(tables/f'{name}逐时距指标_{split_label(split)}.csv',bs,subjects,labels,model_name=name)
    _write_table(tables/f'逐时距指标_{split_label(split)}.csv',steps,subjects,labels)
    _write_table(tables/f'分阶段指标_{split_label(split)}.csv',stages,subjects,H32_STAGE_LABELS)
    _write_table(tables/f'逐被试分阶段指标_{split_label(split)}.csv',stages,subjects,H32_STAGE_LABELS,by_subject=True)
    if compute_importance:
        report['roi_importance']=roi_permutation_importance(model,data_loader,device,importance_max_tasks)
    if save_arrays:np.savez_compressed(out_dir/f'预测数组_{split_label(split)}.npz',**data)
    (out_dir/f'指标_{split_label(split)}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


@torch.no_grad()
def roi_permutation_importance(model,data_loader,device,max_tasks=256,seed=2024):
    """打乱指定 ROI 的历史时间顺序，比较完整 H32 MAE；不使用未来构图。"""
    model.eval();generator=torch.Generator().manual_seed(seed)
    errors=[];changes=[];subjects=[];count=0
    for batch in data_loader:
        x,target,mask,anchor,cond=forecast_batch(batch,device)
        if max_tasks and count+x.shape[0]>max_tasks:
            n=max_tasks-count
            x,target,mask=x[:n],target[:n],mask[:n]
            if cond:cond={k:v[:n] for k,v in cond.items()}
        if not len(x):break
        baseline,_=model(x,condition=cond)
        weight=mask[:,None,:]
        def mae(pred):return ((pred-target).abs()*weight).sum((1,2))/(mask.sum(-1).clamp_min(1)*pred.shape[1])
        base=mae(baseline)
        rows=[]
        for roi in range(x.shape[1]):
            altered=x.clone()
            order=torch.randperm(x.shape[-1],generator=generator).to(device)
            altered[:,roi]=altered[:,roi].index_select(-1,order)
            prediction,_=model(altered,condition=cond)
            rows.append((mae(prediction)-base).cpu().numpy())
        changes.append(np.stack(rows,axis=1));errors.append(base.cpu().numpy())
        subjects.extend(map(str,batch['subj_id'][:len(x)]));count+=len(x)
        if max_tasks and count>=max_tasks:break
    delta=np.concatenate(changes);subjects=np.asarray(subjects)
    per_subject=np.stack([delta[subjects==sid].mean(0) for sid in np.unique(subjects)])
    return dict(method='within-history ROI temporal permutation',seed=seed,n_tasks=count,
                n_subjects=len(per_subject),mae_increase=per_subject.mean(0).tolist())


def summarize(report):
    metrics=report['trajectory_metrics']
    return {name:metrics[name]['H1_32']['mean'] for name in
            ('mae','rmse','temporal_pcc','delta_pcc','std_ratio','fc_mse','fc_pcc','freq_l1')}
