# coding=utf-8
"""v2 健康参照拟合与规范性偏离（B3）+ 校准质量门槛检查（B4）。

输入：``analysis.rolling_one_step`` 的逐被试 ROI-MAE 矩阵
（``<rolling_dir>/<group>/逐被试_ROI误差_<split>.npz`` → ``mae_model``，
历史产物 ``per_subject_roi_mae_<split>.npz`` 仍作为兼容回退，
即逐 ROI 的 ``s_i,roi = mean_t |err_lag1|``）与人口学 merged 表（hc3 三字段）。

- ``gate``（B4，事先写死的通过标准）：HC val 上模型 lag-1 MAE 相对 persistence
  改善 ≥ ``--min-improvement``（默认 1%）**且**被试级配对置换检验
  p < ``--alpha``（默认 0.01，符号翻转 10,000 次，seed=2024）。
  不通过 → 输出 ``passed=false`` 并提示停止偏离分析（v2 计划 §7 触发器 T1，
  唯一预登记的方向重议点）。产物 ``results/normative_reference/gate_report.json``。
- ``fit``（B3 第一阶段）：HC **train** 被试的 s 对人口学设计矩阵做 α=1 岭回归
  （复用 ``analysis/twin_reference.py`` 的设计布局：intercept + Age z + Age z² +
  education z + Sex one-hot + 缺失 mask），得 ``μ_roi(x)``、残差 ``σ_roi`` 与
  逐 ROI 残差绝对值的 conformal 分位数（α=0.1 → 0.9 分位）。
  产物 ``results/normative_reference/fit.npz``（W / sigma / conformal_q / meta）。
- ``score``（B3 第二阶段）：对 HC val（规范集）、HC test（负控）与 MDD 全队列
  计算 ``z_i,roi = (s_i − μ_roi(x_i)) / σ_roi``；输出逐被试 z 表
  （``per_subject_z_<group>_<split>.csv``，宽表 116 列）、z 矩阵 npz 与
  网络级汇总（复用 ``analysis/metrics.py`` 的 NETWORK_GROUPS）。
  z 是人口学条件下的标准化偏离，不是诊断概率（v2 计划 §3 口径声明）。

用法::

    python -m analysis.normative_deviation gate \
      --rolling-dir results/rolling_eval --split val
    python -m analysis.normative_deviation fit \
      --rolling-dir results/rolling_eval
    python -m analysis.normative_deviation score \
      --rolling-dir results/rolling_eval \
      --out-dir results/pathology_analysis/run1/z_maps
"""
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.metrics import NETWORK_GROUPS
from analysis.rolling_one_step import GROUP_SUBDIR, manifest_path
from utils.common import split_label

log = logging.getLogger(__name__)

HC_MERGED_FILE = 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'
MDD_MERGED_FILE = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'
RIDGE_ALPHA = 1.0          # 与 twin_reference 一致；intercept 不惩罚
DEFAULT_CONFORMAL_ALPHA = 0.1
EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════
#  数据读取
# ══════════════════════════════════════════════════════════════════════════

def roi_mae_path(rolling_dir: Path, group: str, split: str) -> Path:
    """优先使用中文 ROI-MAE 产物，历史英文产物作为兼容回退。"""
    group_dir = Path(rolling_dir) / GROUP_SUBDIR[group]
    preferred = group_dir / f'逐被试_ROI误差_{split_label(split)}.npz'
    return preferred if preferred.exists() else group_dir / f'per_subject_roi_mae_{split}.npz'


def load_roi_mae(rolling_dir: Path, group: str, split: str) -> Tuple[List[str],
                                                                     np.ndarray]:
    """读取某 group×split 的被试清单与 ``mae_model [N, F]``（缺失即报错）。"""
    path = roi_mae_path(rolling_dir, group, split)
    if not path.exists():
        raise FileNotFoundError(
            f'缺少 {path}；请先对该 group/split 运行 analysis.rolling_one_step。')
    with np.load(path) as z:
        subjects = [str(s) for s in z['subjects']]
        mae = np.asarray(z['mae_model'], dtype=np.float64)
    if mae.ndim != 2 or len(subjects) != mae.shape[0]:
        raise ValueError(f'{path} 内容形状异常：subjects={len(subjects)} mae={mae.shape}')
    return subjects, mae


def load_demo_matrix(data_root: Path, merged_file: str,
                     subjects: Sequence[str]) -> Tuple[np.ndarray, np.ndarray,
                                                       Dict[str, int]]:
    """被试清单 → hc3 人口学 (values [N,3], mask [N,3], 字段下标)。

    字段顺序固定为 ``twin_reference`` 约定的 Sex/Age/education 布局。
    """
    from utils.dataloader import load_condition_table
    table = load_condition_table(str(data_root), merged_file, 'hc3')
    from analysis.twin_reference import _demo_field_indices
    idx = _demo_field_indices(table.specs)
    order = ('Sex', 'Age', 'education')
    values, mask = table.align([str(s) for s in subjects])
    v = np.stack([values[:, idx[k]] for k in order], axis=1).astype(np.float64)
    m = np.stack([mask[:, idx[k]] for k in order], axis=1).astype(np.uint8)
    return v, m, {k: i for i, k in enumerate(order)}


def split_subjects(data_root: Path, group: str) -> Dict[str, List[str]]:
    """读取组内 train/val/test 划分（HC=预训练 manifest；MDD=finetune 划分）。"""
    name = 'subject_split_pretrain.json' if group == 'hc' \
        else 'subject_split_finetune.json'
    path = Path(data_root) / name
    if not path.exists():
        raise FileNotFoundError(f'划分文件不存在: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {k: [str(s) for s in payload[k]] for k in ('train', 'val', 'test')}


# ══════════════════════════════════════════════════════════════════════════
#  B4 门槛检查
# ══════════════════════════════════════════════════════════════════════════

def paired_signflip_p(diff: np.ndarray, n_perm: int = 10000, seed: int = 2024) -> float:
    """被试级配对置换（符号翻转）单侧 p 值：H0: E[diff]=0 vs H1: E[diff]>0。

    ``diff`` = mae_persistence − mae_model（正值 = 模型更好）。
    p 使用 ``(计数+1)/(n_perm+1)`` 平滑，保证 0 次超过时 p=1/(n+1) 而非 0。
    """
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    if diff.size < 2:
        raise ValueError(f'配对置换至少需要 2 个被试，收到 {diff.size}')
    obs = float(diff.mean())
    rng = np.random.RandomState(int(seed))
    signs = rng.choice((-1.0, 1.0), size=(int(n_perm), diff.size))
    null = (signs * diff[None, :]).mean(axis=1)
    exceed = int((null >= obs - 1e-12).sum())
    return float((exceed + 1) / (int(n_perm) + 1))


def run_gate(rolling_dir: Path, data_root: Path, split: str = 'val',
             min_improvement: float = 0.01, alpha: float = 0.01,
             n_perm: int = 10000, seed: int = 2024) -> dict:
    """B4 门槛：HC <split> 上 lag-1 模型 vs persistence（判据事先写死）。"""
    path = roi_mae_path(rolling_dir, 'hc', split)
    if not path.exists():
        raise FileNotFoundError(f'缺少 {path}；请先运行 rolling_one_step --group hc。')
    with np.load(path) as z:
        subjects = [str(s) for s in z['subjects']]
        mae_m = np.asarray(z['mae_model'], dtype=np.float64)
        mae_p = np.asarray(z['mae_persistence'], dtype=np.float64)

    # 只保留属于该 split 的被试（组目录缓存可能含多 split）
    members = set(split_subjects(data_root, 'hc')[split])
    keep = [i for i, s in enumerate(subjects) if s in members]
    if len(keep) < 2:
        raise ValueError(
            f'{path} 中属于 {split} 的被试只有 {len(keep)} 个（缓存与划分不匹配？'
            '请对完整 split 运行 rolling_one_step 后再执行 gate。）')
    subjects = [subjects[i] for i in keep]
    mae_m, mae_p = mae_m[keep], mae_p[keep]

    subj_model = mae_m.mean(axis=1)                 # 被试等权
    subj_pers = mae_p.mean(axis=1)
    improvement = float(1.0 - subj_model.mean() / subj_pers.mean())
    diff = subj_pers - subj_model
    p_value = paired_signflip_p(diff, n_perm=n_perm, seed=seed)
    passed = bool(improvement >= float(min_improvement) and p_value < float(alpha))
    report = {
        'gate': 'B4 校准质量门槛（v2 §7 触发器 T1）',
        'split': split, 'group': 'hc',
        'n_subjects': len(subjects),
        'criteria': {'min_improvement': float(min_improvement),
                     'alpha': float(alpha), 'n_perm': int(n_perm),
                     'seed': int(seed),
                     'test': '被试级配对符号翻转置换（单侧，H1: 模型 MAE 更低）'},
        'mae_model_subject_equal_weight': float(subj_model.mean()),
        'mae_persistence_subject_equal_weight': float(subj_pers.mean()),
        'improvement': improvement,
        'permutation_p': p_value,
        'passed': passed,
        'action_on_fail': '停止偏离分析，回到方向重议（唯一预登记例外）；'
                          '不得继续 fit/score 与组间检验。',
        'source': str(path),
    }
    return report


# ══════════════════════════════════════════════════════════════════════════
#  B3 第一阶段：参照拟合
# ══════════════════════════════════════════════════════════════════════════

def fit_normative(s: np.ndarray, values: np.ndarray, mask: np.ndarray,
                  idx: Dict[str, int], conformal_alpha: float = 0.1) -> dict:
    """对 HC train 的 ``s [N, F]`` 拟合人口学条件化参照（纯函数，可测试）。

    Returns:
        dict: ``W [d, F]``、``sigma_roi [F]``、``conformal_q [F]``、
        ``design_layout``、``demo``（人口学统计量）、``residuals [N, F]``。
    """
    from analysis.twin_reference import (_demo_design_rows, fit_demo_stats,
                                         _demo_field_indices)
    if s.ndim != 2:
        raise ValueError(f's 期望 [N, F]，收到 {s.shape}')
    if s.shape[0] != values.shape[0]:
        raise ValueError(f's 行数 {s.shape[0]} != 人口学行数 {values.shape[0]}')
    stats = fit_demo_stats(values, mask, idx)
    a, layout = _demo_design_rows(values, mask, idx, stats)

    r_pen = np.ones(a.shape[1], dtype=np.float64)
    r_pen[0] = 0.0                                   # intercept 不惩罚
    lhs = a.T @ a + RIDGE_ALPHA * np.diag(r_pen)
    w = np.linalg.solve(lhs, a.T @ s)                # [d, F]
    resid = s - a @ w
    sigma = np.sqrt((resid ** 2).mean(axis=0) + 1e-6)            # [F]
    q_level = 1.0 - float(conformal_alpha)
    conformal_q = np.quantile(np.abs(resid), q_level, axis=0)   # [F]
    return {'W': w, 'sigma_roi': sigma, 'conformal_q': conformal_q,
            'design_layout': layout, 'demo': stats, 'residuals': resid}


def score_normative(s: np.ndarray, values: np.ndarray, mask: np.ndarray,
                    idx: Dict[str, int], fit: dict) -> np.ndarray:
    """新被试 ``s [N, F]`` + 人口学 → ``z [N, F]``（μ 由设计行预测，σ 来自参照）。"""
    from analysis.twin_reference import _demo_design_rows
    a, layout = _demo_design_rows(values, mask, idx, fit['demo'])
    if layout != fit['design_layout']:
        raise ValueError(
            f'设计布局不一致：参照 {fit["design_layout"]} vs 当前 {layout}；'
            'Sex 词表或字段顺序变化都会禁止复用旧参照。')
    mu = a @ fit['W']                                # [N, F]
    return (s - mu) / fit['sigma_roi'][None, :]


def score_crossfit(rolling_dir: Path, data_root: Path, k: int = 10,
                   conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
                   seed: int = 2024) -> Tuple[List[str], np.ndarray]:
    """HC train 的 K 折 cross-fitted z（折外参照，避免 in-sample 收缩）。

    每个被试的 μ/σ 来自**不含该被试**的 K−1 折重新拟合；折分配按被试清单
    排序后轮转（确定性，与 seed 无关）。返回 (subjects, z)。
    """
    subjects, s = load_roi_mae(rolling_dir, 'hc', 'train')
    values, mask, idx = load_demo_matrix(data_root, HC_MERGED_FILE, subjects)
    order = np.argsort(np.asarray(subjects))             # 确定性折分配
    fold = np.empty(len(subjects), dtype=int)
    for i, pos in enumerate(order):
        fold[pos] = i % int(k)
    z = np.full_like(s, np.nan, dtype=np.float64)
    for f in range(int(k)):
        te = fold == f
        tr = ~te
        fit = fit_normative(s[tr], values[tr], mask[tr], idx,
                            conformal_alpha=conformal_alpha)
        z[te] = score_normative(s[te], values[te], mask[te], idx, fit)
    assert np.isfinite(z).all()
    return subjects, z





# ══════════════════════════════════════════════════════════════════════════
#  网络级汇总
# ══════════════════════════════════════════════════════════════════════════

def network_summary(z: np.ndarray, n_rois: int = 116) -> List[dict]:
    """z [N, F] → 各 AAL 网络的 mean/median/std（未命名 ROI 归 Other）。"""
    labels = ['Other'] * n_rois
    for net, regions in NETWORK_GROUPS.items():
        for r in regions:
            if 1 <= r <= n_rois:
                labels[r - 1] = net
    rows = []
    for net in sorted(set(labels)):
        cols = [j for j, lb in enumerate(labels) if lb == net]
        block = z[:, cols]
        rows.append({'network': net, 'n_rois': len(cols),
                     'z_mean': float(block.mean()),
                     'z_median': float(np.median(block)),
                     'z_std': float(block.std()),
                     'z_subject_mean_of_max': float(
                         np.abs(block).max(axis=1).mean())})
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def _cmd_gate(ns) -> None:
    report = run_gate(ns.rolling_dir, ns.data_root, split=ns.split,
                      min_improvement=ns.min_improvement, alpha=ns.alpha,
                      n_perm=ns.n_perm, seed=ns.seed)
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8')
    log.info('[gate] %s | improvement=%.4f p=%.5f → %s', out,
             report['improvement'], report['permutation_p'],
             'PASSED' if report['passed'] else 'FAILED（停止偏离分析，回方向重议）')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _cmd_fit(ns) -> None:
    subjects, s = load_roi_mae(ns.rolling_dir, 'hc', 'train')
    members = set(split_subjects(ns.data_root, 'hc')['train'])
    keep = [i for i, sid in enumerate(subjects) if sid in members]
    subjects = [subjects[i] for i in keep]
    s = s[keep]
    if len(subjects) < 10:
        raise ValueError(f'HC train 缓存被试过少（{len(subjects)}），无法拟合参照。')
    values, mask, idx = load_demo_matrix(ns.data_root, HC_MERGED_FILE, subjects)
    fit = fit_normative(s, values, mask, idx, conformal_alpha=ns.conformal_alpha)

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path(
        Path(ns.rolling_dir) / GROUP_SUBDIR['hc']).read_text(encoding='utf-8'))
    meta = {'design_layout': json.dumps(fit['design_layout']),
            'demo': json.dumps(fit['demo']),
            'conformal_alpha': float(ns.conformal_alpha),
            'ridge_alpha': RIDGE_ALPHA,
            'n_subjects': int(len(subjects)), 'n_rois': int(s.shape[1]),
            'subjects': json.dumps(subjects),
            'source_split': 'hc train（subject_split_pretrain.json）',
            'manifest_protocol': json.dumps(
                {k: manifest.get(k) for k in ('checkpoint_sha256', 'context_length',
                                              'anchor_rule', 'lag_output')})}
    np.savez(out, W=fit['W'], sigma_roi=fit['sigma_roi'],
             conformal_q=fit['conformal_q'], **meta)
    log.info('[fit] 参照已写入 %s | train 被试=%d | 设计列=%d | σ 中位数=%.4f '
             '| conformal q(%.2f) 中位数=%.4f', out, len(subjects),
             len(fit['design_layout']), float(np.median(fit['sigma_roi'])),
             1 - ns.conformal_alpha, float(np.median(fit['conformal_q'])))
    print(json.dumps({'out': str(out), 'n_subjects': len(subjects),
                      'n_features': int(s.shape[1]),
                      'sigma_median': float(np.median(fit['sigma_roi']))},
                     ensure_ascii=False))


def _load_fit(path: Path) -> dict:
    raw = np.load(str(path), allow_pickle=False)
    return {'W': raw['W'], 'sigma_roi': raw['sigma_roi'],
            'conformal_q': raw['conformal_q'],
            'design_layout': json.loads(str(raw['design_layout'])),
            'demo': json.loads(str(raw['demo']))}


def _cmd_score(ns) -> None:
    fit = _load_fit(ns.reference)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = {'hc': HC_MERGED_FILE, 'mdd': MDD_MERGED_FILE}

    # HC train：cross-fitted z（折外参照），避免参照拟合集自身 in-sample 收缩
    if roi_mae_path(ns.rolling_dir, 'hc', 'train').exists():
        subjects, z = score_crossfit(ns.rolling_dir, ns.data_root, k=ns.crossfit)
        _write_z_outputs(out_dir, 'hc', 'train', subjects, z)
        log.info('[score] hc train(crossfit%d) | n=%d | z mean=%.4f | '
                 '|z|max(被试均值)=%.3f', ns.crossfit, len(subjects),
                 float(z.mean()), float(np.abs(z).max(axis=1).mean()))

    for group, split in [('hc', 'val'), ('hc', 'test'),
                         ('mdd', 'train'), ('mdd', 'val'), ('mdd', 'test')]:
        path = roi_mae_path(ns.rolling_dir, group, split)
        if not path.exists():
            log.warning('[score] 缺少 %s，跳过（先运行 rolling_one_step）', path)
            continue
        subjects, s = load_roi_mae(ns.rolling_dir, group, split)
        values, mask, idx = load_demo_matrix(ns.data_root, merged[group], subjects)
        z = score_normative(s, values, mask, idx, fit)          # [N, F]
        _write_z_outputs(out_dir, group, split, subjects, z)
        log.info('[score] %s %s | n=%d | z mean=%.4f | |z|max(被试均值)=%.3f',
                 group, split, len(subjects), float(z.mean()),
                 float(np.abs(z).max(axis=1).mean()))
    log.info('[score] z_maps 已写入 %s（z 为人口学条件化标准化偏离，非诊断概率）',
             out_dir)


def _write_z_outputs(out_dir: Path, group: str, split: str,
                     subjects: List[str], z: np.ndarray) -> None:
    n_rois = z.shape[1]
    np.savez_compressed(out_dir / f'z_matrix_{group}_{split}.npz',
                        subjects=np.asarray(subjects), z=z.astype(np.float32))
    csv_path = out_dir / f'per_subject_z_{group}_{split}.csv'
    header = ['subject'] + [f'roi_{j + 1:03d}' for j in range(n_rois)]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sid, row in zip(subjects, z):
            writer.writerow([sid] + [f'{v:.6f}' for v in row])


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data-root', type=Path, default=Path('data'))
    common.add_argument('--rolling-dir', type=Path, required=True,
                        help='rolling_one_step 的输出根目录（含 <group>/ 子目录）')

    g = sub.add_parser('gate', parents=[common],
                       help='B4 校准质量门槛（HC val 上 model vs persistence）')
    g.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    g.add_argument('--min-improvement', type=float, default=0.01)
    g.add_argument('--alpha', type=float, default=0.01)
    g.add_argument('--n-perm', type=int, default=10000)
    g.add_argument('--seed', type=int, default=2024)
    g.add_argument('--out', type=Path,
                   default=Path('results/normative_reference/gate_report.json'))
    g.set_defaults(func=_cmd_gate)

    f = sub.add_parser('fit', parents=[common],
                       help='B3 第一阶段：HC train 参照拟合')
    f.add_argument('--conformal-alpha', type=float, default=DEFAULT_CONFORMAL_ALPHA)
    f.add_argument('--out', type=Path,
                   default=Path('results/normative_reference/fit.npz'))
    f.set_defaults(func=_cmd_fit)

    sc = sub.add_parser('score', parents=[common],
                        help='B3 第二阶段：z 偏离打分（HC train crossfit + HC val/test + MDD 全队列）')
    sc.add_argument('--reference', type=Path,
                    default=Path('results/normative_reference/fit.npz'))
    sc.add_argument('--out-dir', type=Path,
                    default=Path('results/pathology_analysis/run1/z_maps'))
    sc.add_argument('--crossfit', type=int, default=10,
                    help='HC train cross-fitted z 的折数（默认 10；0 = 跳过 hc train）')
    sc.set_defaults(func=_cmd_score)

    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    ns.func(ns)


if __name__ == '__main__':
    main()
