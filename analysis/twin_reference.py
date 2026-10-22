# coding=utf-8
"""健康人口学参照：固定描述量上的人口学岭回归（方案 §7）。

不往 TFM 增加健康分类 loss，不用随训练漂移的 latent 定义病理。参照由三部分组成：

1. :func:`describe_history`：K_ref=64 的数据尺度 BOLD → 六个固定描述量 [B,F,6]
   （acf1/acf2/low_power/spectral_centroid/diff_rms/signed_fc_mean）。函数为
   仿射不变口径（沿时间去均值 + RMS 标准化），HC 拟合、患者分析与 outcome
   特征共用同一实现。
2. :func:`fit_reference`：每个 HC 训练被试只取**最早 64 个观测**的一份描述量，
   对 intercept + Age z + Age z² + education z + Sex one-hot + 三字段有效 mask
   的设计矩阵做 α=1 岭回归（solve，不显式求逆），保存 W、逐 ROI/分量残差尺度、
   设计布局、人口学统计量、K_ref 与分量名。
3. :func:`score_reference`：患者描述量相对人口学预测的标准化偏离
   ``r_ref=(D-Ŷ)/s_ref`` 与 ROI 距离 ``d_ref=sqrt(mean_j r_ref²)``。

解释边界（方案 §7.2）：输出是人口学条件下的标准化动态偏离，**不是**疾病诊断
概率、治疗目标或因果病灶；HC 训练残差尺度也不是校准的正常人百分位；代码不设
诊断阈值；当前临床条件变化不改变同一真实 BOLD 的参照输出。

命令行::

    python -m analysis.twin_reference fit  --data_root ... --mode pretrain \
        --condition_group hc3 --condition_file ... --out hc_reference.npz
    python -m analysis.twin_reference score --reference hc_reference.npz \
        --history bold.npy --age 32 --education 12 --sex 1 --out score.npz

频率单位一律为 cycles/sample（TR 未知，不换算成 Hz）。
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: 参照分析的固定历史长度（不足该长度不执行参照分析，普通预测仍可运行）
K_REF = 64
#: 六分量固定顺序（outcome 特征与 NPZ 布局都依赖该顺序）
COMPONENT_NAMES = ('acf1', 'acf2', 'low_power', 'spectral_centroid',
                   'diff_rms', 'signed_fc_mean')
#: low_power 的频带上界（cycles/sample；rfftfreq 的 d=1）
LOW_BAND_CYCLES = 0.125
EPS = 1e-6
RIDGE_ALPHA = 1.0
#: 设计矩阵布局（列名固定，NPZ 里以 JSON 保存）
DESIGN_INTERCEPT = 'intercept'
DESIGN_AGE_Z = 'age_z'
DESIGN_AGE_Z2 = 'age_z_sq'
DESIGN_EDU_Z = 'education_z'
DESIGN_SEX_PREFIX = 'sex_code_'
DESIGN_MASK_AGE = 'mask_age'
DESIGN_MASK_EDU = 'mask_education'
DESIGN_MASK_SEX = 'mask_sex'


# ══════════════════════════════════════════════════════════════════════════
#  §7.1 统一描述量
# ══════════════════════════════════════════════════════════════════════════

def describe_history(history) -> np.ndarray:
    """数据尺度 BOLD 历史 → 六个固定描述量。

    Args:
        history: ``[B, 64, F]``（或 ``[64, F]`` 单样本）。数据尺度或逐 ROI
            仿射变换后的序列均可：第一步沿时间去均值 + RMS 标准化使结果对
            逐 ROI 仿射变换不变（不从归一化信号推断原始强度异常）。

    Returns:
        ``[B, F, 6]``，分量顺序固定为
        ``acf1, acf2, low_power, spectral_centroid, diff_rms, signed_fc_mean``。

    数值约定（方案 §7.1）：所有分母加 ``eps=1e-6``（含常量序列与全零差分）；
    频率单位 cycles/sample；low_power 分母为全部正频率功率。
    """
    x = np.asarray(history, dtype=np.float64)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[1] != K_REF:
        raise ValueError(
            f'describe_history 期望 [B,{K_REF},F]，收到 {tuple(x.shape)}；'
            '不同长度不混入同一参照，调用方必须显式提供最近 64 个真实观测。')
    b, t, f = x.shape

    # 沿时间去均值并按当前历史 RMS 标准化 → r [B,T,F]
    xc = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt((xc ** 2).mean(axis=1, keepdims=True) + EPS)
    r = xc / rms

    comps: List[np.ndarray] = []
    # 1/2. acf1 / acf2：lag=k 两片段各自去均值后的中心化相关，分母加 eps
    for lag in (1, 2):
        a = r[:, :t - lag, :]
        c = r[:, lag:, :]
        ac = a - a.mean(axis=1, keepdims=True)
        cc = c - c.mean(axis=1, keepdims=True)
        num = (ac * cc).sum(axis=1)
        den = (np.sqrt((ac ** 2).sum(axis=1) + EPS)
               * np.sqrt((cc ** 2).sum(axis=1) + EPS))
        comps.append(num / den)

    # 频谱（正频率；单位 cycles/sample，d=1）
    spec = np.fft.rfft(r, axis=1)                       # [B, T//2+1, F]
    power = spec.real ** 2 + spec.imag ** 2
    freqs = np.fft.rfftfreq(t, d=1.0)                   # cycles/sample
    pos = freqs > 0
    pos_power = power[:, pos, :]                        # [B, P, F]
    total_pos = pos_power.sum(axis=1)                   # [B, F]

    # 3. low_power：(0, 0.125] cycles/sample 功率 / 全部正频率功率
    band = (freqs > 0) & (freqs <= LOW_BAND_CYCLES)
    comps.append(power[:, band, :].sum(axis=1) / (total_pos + EPS))

    # 4. spectral_centroid：正频率功率加权频率中心（cycles/sample）
    centroid = (pos_power * freqs[pos][None, :, None]).sum(axis=1)
    comps.append(centroid / (total_pos + EPS))

    # 5. diff_rms：相邻 r 差分平方的均值开方
    d = np.diff(r, axis=1)
    comps.append(np.sqrt((d ** 2).mean(axis=1) + EPS))

    # 6. signed_fc_mean：与其他 ROI 的 signed Pearson 相关均值（对角不参与）
    rc = r - r.mean(axis=1, keepdims=True)
    cov = np.einsum('btf,bts->bfs', rc, rc)             # [B, F, F] 同一 t 配对
    den = np.sqrt((rc ** 2).sum(axis=1) + EPS)          # [B, F]
    fc = cov / (den[:, :, None] * den[:, None, :])
    offdiag_sum = fc.sum(axis=-1) - np.diagonal(fc, axis1=1, axis2=2)
    comps.append(offdiag_sum / (f - 1))

    out = np.stack(comps, axis=-1).astype(np.float64)   # [B, F, 6]
    assert out.shape == (b, f, len(COMPONENT_NAMES))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  人口学设计矩阵
# ══════════════════════════════════════════════════════════════════════════

def _demo_field_indices(specs) -> Dict[str, int]:
    """从条件表字段规格中定位 Sex/Age/education 的列下标（缺一即报错）。"""
    from utils.condition_schema import field_names
    names = list(field_names(tuple(specs)))
    out = {}
    for name in ('Sex', 'Age', 'education'):
        if name not in names:
            raise ValueError(
                f'条件表缺少参照必需字段 {name}（现有字段={names}）；'
                '健康参照只用 Age/education/Sex，不使用 HAMD/HAMA/诊断/ID。')
        out[name] = names.index(name)
    return out


def fit_demo_stats(values: np.ndarray, mask: np.ndarray,
                   idx: Dict[str, int]) -> dict:
    """训练集人口学统计量：Age/education 的 z 标准化参数与 Sex 已登记词表。

    只用训练有效观测；字段全缺失时 mu=0、scale=1（z 恒 0，mask=0）。 Sex 词表
    为训练有效观测中出现的类别码升序（不含缺失）。
    """
    def _cont(name: str):
        v = values[:, idx[name]].astype(np.float64)
        m = mask[:, idx[name]].astype(bool)
        valid = v[m]
        if valid.size == 0:
            return {'mu': 0.0, 'scale': 1.0, 'n_valid': 0}
        mu = float(valid.mean())
        sd = float(valid.std())
        return {'mu': mu, 'scale': sd if sd > EPS else 1.0, 'n_valid': int(valid.size)}

    sex_v = values[:, idx['Sex']].astype(np.float64)
    sex_m = mask[:, idx['Sex']].astype(bool)
    vocab = sorted(float(c) for c in np.unique(sex_v[sex_m]))
    return {
        'Age': _cont('Age'),
        'education': _cont('education'),
        'Sex': {'vocab': vocab, 'n_valid': int(sex_m.sum())},
    }


def _demo_design_rows(values: np.ndarray, mask: np.ndarray,
                      idx: Dict[str, int], stats: dict) -> tuple:
    """批量构造人口学设计矩阵 A [N, d_a] 与列布局。

    缺失规则（方案 §7.2）：连续字段缺失 z=0、mask=0；Sex 缺失 one-hot 全零、
    mask=0。列顺序：intercept, age_z, age_z_sq, education_z, sex one-hot,
    mask_age, mask_education, mask_sex。
    """
    layout: List[str] = [DESIGN_INTERCEPT, DESIGN_AGE_Z, DESIGN_AGE_Z2,
                         DESIGN_EDU_Z]
    vocab = list(stats['Sex']['vocab'])
    layout += [f'{DESIGN_SEX_PREFIX}{c:g}' for c in vocab]
    layout += [DESIGN_MASK_AGE, DESIGN_MASK_EDU, DESIGN_MASK_SEX]

    n = values.shape[0]
    a = np.zeros((n, len(layout)), dtype=np.float64)
    a[:, 0] = 1.0
    age = values[:, idx['Age']].astype(np.float64)
    age_m = mask[:, idx['Age']].astype(bool)
    edu = values[:, idx['education']].astype(np.float64)
    edu_m = mask[:, idx['education']].astype(bool)
    sex = values[:, idx['Sex']].astype(np.float64)
    sex_m = mask[:, idx['Sex']].astype(bool)

    age_z = np.where(age_m, (age - stats['Age']['mu']) / stats['Age']['scale'], 0.0)
    edu_z = np.where(edu_m, (edu - stats['education']['mu']) / stats['education']['scale'], 0.0)
    a[:, 1] = age_z
    a[:, 2] = age_z ** 2
    a[:, 3] = edu_z
    code_to_col = {c: 4 + i for i, c in enumerate(vocab)}
    for i, c in enumerate(sex):
        if sex_m[i] and float(c) in code_to_col:
            a[i, code_to_col[float(c)]] = 1.0
    a[:, 4 + len(vocab) + 0] = age_m.astype(np.float64)
    a[:, 4 + len(vocab) + 1] = edu_m.astype(np.float64)
    a[:, 4 + len(vocab) + 2] = sex_m.astype(np.float64)
    return a, layout


def _single_demo_row(demo: Dict[str, Optional[float]], stats: dict):
    """把 ``{'Age': ..., 'education': ..., 'Sex': ...}``（None=缺失）转成设计行。"""
    values = np.array([[float('nan') if demo.get(k) is None else float(demo[k])
                        for k in ('Sex', 'Age', 'education')]], dtype=np.float64)
    mask = np.array([[0 if demo.get(k) is None else 1
                      for k in ('Sex', 'Age', 'education')]], dtype=np.uint8)
    idx = {'Sex': 0, 'Age': 1, 'education': 2}
    a, _ = _demo_design_rows(values, mask, idx, stats)
    return a[0]


# ══════════════════════════════════════════════════════════════════════════
#  §7.2 拟合与推理
# ══════════════════════════════════════════════════════════════════════════

def fit_reference(dataset, train_subjects: Sequence[str],
                  stats: Optional[dict] = None) -> dict:
    """从数据集拟合健康参照。

    Args:
        dataset: 具备 ``get_series(subj)->[F,T]`` 与
            ``condition_table``（``align(ids)->(values, mask)``）的数据集
            （NeuroTwinNextPointDataset 或其视图的 base_dataset）。
        train_subjects: **仅 train 被试**（禁止 val/test 泄漏）。
        stats: 可选的既有人口学统计量（:func:`fit_demo_stats` 输出）；
            None 时由 train 被试拟合（train-only，无泄漏）。

    Returns:
        payload dict（``W / s_ref / design_layout / demo / K_ref / components /
        n_subjects / n_features``），可直接交给 :func:`save_reference`。
    """
    table = getattr(dataset, 'condition_table', None)
    if table is None:
        raise ValueError(
            '健康参照需要 merged 条件表（Age/education/Sex）：'
            '请以 condition_group 开启的方式构建数据集（HC 用 hc3）。')
    idx = _demo_field_indices(table.specs)
    subjects = [str(s) for s in train_subjects]
    if not subjects:
        raise ValueError('train_subjects 为空，无法拟合健康参照。')

    values, mask = table.align(subjects)
    if stats is None:
        stats = fit_demo_stats(values, mask, idx)

    # 每个训练被试只取最早 64 个观测的一份描述量（避免扫描窗口数决定权重）
    descs, kept = [], []
    skipped = []
    for sid in subjects:
        series = np.asarray(dataset.get_series(sid), dtype=np.float64)
        if series.ndim != 2 or series.shape[1] < K_REF:
            skipped.append((sid, int(series.shape[1]) if series.ndim == 2 else -1))
            continue
        hist = series[:, :K_REF].T                        # [64, F]
        descs.append(describe_history(hist)[0])           # [F, 6]
        kept.append(sid)
    if skipped:
        log.warning('[twin_reference] %d 个 train 被试观测不足 %d TR，跳过（例如 %s）',
                    len(skipped), K_REF, skipped[:3])
    if len(descs) < 2:
        raise ValueError(
            f'观测 >= {K_REF} TR 的 train 被试只有 {len(descs)} 个，不足以拟合参照；'
            '不足 64 个真实观测时不执行参照分析（普通预测不受影响）。')

    d = np.stack(descs, axis=0)                           # [N, F, 6]
    n, n_feat, n_comp = d.shape
    y = d.reshape(n, n_feat * n_comp)                     # 行主序展开 (F,6)

    a, layout = _demo_design_rows(values, mask, idx, stats)

    # 岭回归：W=(AᵀA+αR)⁻¹AᵀY，用 solve 不显式求逆；α=1，intercept 不惩罚
    r_pen = np.ones(a.shape[1], dtype=np.float64)
    r_pen[0] = 0.0
    lhs = a.T @ a + RIDGE_ALPHA * np.diag(r_pen)
    rhs = a.T @ y
    w = np.linalg.solve(lhs, rhs)                         # [d_a, 6F]

    resid = y - a @ w
    s_ref = np.sqrt((resid ** 2).mean(axis=0) + 1e-6)     # [6F]（方案 §7.2）
    s_ref = s_ref.reshape(n_feat, n_comp)

    payload = {
        'W': w,
        's_ref': s_ref.astype(np.float64),
        'design_layout': layout,
        'demo': stats,
        'K_ref': K_REF,
        'components': list(COMPONENT_NAMES),
        'n_subjects': int(n),
        'n_features': int(n_feat),
        'subjects': kept,
    }
    log.info('[twin_reference] 拟合完成 | train 被试=%d | F=%d | 设计列=%d | '
             '残差尺度中位数=%.4f', n, n_feat, len(layout),
             float(np.median(s_ref)))
    return payload


def score_reference(history, demo: Dict[str, Optional[float]],
                    payload: dict) -> dict:
    """患者参照评分：标准化动态偏离与 ROI 距离。

    Args:
        history: ``[B, 64, F]``（或 ``[64, F]``）数据尺度 BOLD（最近 64 个真实观测）。
        demo: ``{'Age': float|None, 'education': float|None, 'Sex': float|None}``。
        payload: :func:`fit_reference` 的输出（或 :func:`load_reference` 的结果）。

    Returns:
        dict：``normal_residual [B,F,6]``、``normal_distance [B,F]``、
        ``descriptors [B,F,6]``。输出是标准化动态偏离，不是诊断概率。
    """
    for key in ('W', 's_ref', 'demo'):
        if key not in payload:
            raise ValueError(f'参照 payload 缺少 {key}；请用 fit_reference 重新拟合。')
    k_ref = int(payload.get('K_ref', K_REF))
    d = describe_history(history) if k_ref == K_REF else None
    if d is None:
        raise ValueError(f'参照 K_ref={k_ref} 与当前实现 K_ref={K_REF} 不一致。')
    a = _single_demo_row(demo, payload['demo'])
    y_hat = (a @ payload['W']).reshape(d.shape[1], d.shape[2])   # [F, 6]
    r_ref = (d - y_hat[None, ...]) / payload['s_ref'][None, ...]
    dist = np.sqrt((r_ref ** 2).mean(axis=-1))                   # [B, F]
    return {'normal_residual': r_ref, 'normal_distance': dist,
            'descriptors': d}


# ══════════════════════════════════════════════════════════════════════════
#  NPZ 持久化
# ══════════════════════════════════════════════════════════════════════════

def save_reference(payload: dict, path) -> str:
    """参照 payload → NPZ（W、s_ref、布局、统计量、K_ref、分量名）。"""
    path = str(path)
    meta = {
        'design_layout': json.dumps(payload['design_layout']),
        'components': json.dumps(list(payload['components'])),
        'demo': json.dumps(payload['demo']),
        'K_ref': int(payload['K_ref']),
        'n_subjects': int(payload['n_subjects']),
        'n_features': int(payload['n_features']),
    }
    np.savez(path, W=payload['W'], s_ref=payload['s_ref'], **meta)
    return path


def load_reference(path) -> dict:
    """NPZ → payload dict（与 :func:`fit_reference` 输出同构）。"""
    raw = np.load(str(path), allow_pickle=False)
    return {
        'W': raw['W'],
        's_ref': raw['s_ref'],
        'design_layout': json.loads(str(raw['design_layout'])),
        'components': json.loads(str(raw['components'])),
        'demo': json.loads(str(raw['demo'])),
        'K_ref': int(raw['K_ref']),
        'n_subjects': int(raw['n_subjects']),
        'n_features': int(raw['n_features']),
    }


# ══════════════════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════════════════

def _build_dataset(args):
    from utils.dataloader import NeuroTwinDataLoader
    loader = NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode,
        condition_group=(args.condition_group or None),
        condition_file=(args.condition_file or None),
        split_file=(args.split_file or None),
        seed=args.seed,
        task_mode='next_timepoint',
        bold_source=args.bold_source,
        calibration_length=args.calibration_length,
        num_rois=args.num_rois,
        context_min=16, context_max=64,
    )
    return loader


def main():
    ap = argparse.ArgumentParser(
        description='健康人口学参照：fit（HC train 被试）/ score（单被试偏离）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_fit = sub.add_parser('fit', help='从已有 split 的 train 被试拟合参照')
    ap_fit.add_argument('--data_root', required=True)
    ap_fit.add_argument('--mode', default='pretrain', choices=['pretrain', 'finetune'])
    ap_fit.add_argument('--condition_group', default='hc3',
                        help='HC 参照使用 hc3（Sex/Age/education + 五字段）')
    ap_fit.add_argument('--condition_file', default='')
    ap_fit.add_argument('--split_file', default='',
                        help='复用已有被试划分 manifest（与训练一致）')
    ap_fit.add_argument('--bold_source', default='auto',
                        choices=['auto', 'normlize', 'windows', 'raw'])
    ap_fit.add_argument('--calibration_length', type=int, default=0)
    ap_fit.add_argument('--num_rois', type=int, default=116)
    ap_fit.add_argument('--seed', type=int, default=42)
    ap_fit.add_argument('--out', required=True, help='参照 NPZ 输出路径')

    ap_score = sub.add_parser('score', help='用已拟合参照给单被试历史评分')
    ap_score.add_argument('--reference', required=True, help='fit 产出的参照 NPZ')
    ap_score.add_argument('--history', required=True,
                          help='NPZ（键 bold，[T,F]，T>=64）或 NPY [64,F]')
    ap_score.add_argument('--age', type=float, default=None)
    ap_score.add_argument('--education', type=float, default=None)
    ap_score.add_argument('--sex', type=float, default=None)
    ap_score.add_argument('--out', required=True, help='评分 NPZ 输出路径')

    args = ap.parse_args()
    if args.cmd == 'fit':
        loader = _build_dataset(args)
        payload = fit_reference(loader.base_dataset, loader.get_train_subjects())
        save_reference(payload, args.out)
        print(json.dumps({'out': str(args.out),
                          'n_subjects': payload['n_subjects'],
                          'n_features': payload['n_features']},
                         ensure_ascii=False))
        return

    payload = load_reference(args.reference)
    p = Path(args.history)
    if p.suffix == '.npz':
        arr = np.load(str(p))['bold']
    else:
        arr = np.load(str(p))
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.shape[1] != K_REF:
        arr = arr[:, -K_REF:, :]          # 显式取最近 64 个真实观测
        log.info('[twin_reference] 输入序列长于 %d TR，取最近 %d 个观测', K_REF, K_REF)
    demo = {'Age': args.age, 'education': args.education, 'Sex': args.sex}
    result = score_reference(arr, demo, payload)
    np.savez(str(args.out),
             normal_residual=result['normal_residual'],
             normal_distance=result['normal_distance'],
             descriptors=result['descriptors'])
    print(json.dumps({'out': str(args.out),
                      'n_samples': int(result['normal_distance'].shape[0]),
                      'distance_mean': float(result['normal_distance'].mean())},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
