# coding=utf-8
"""独立临床结局读出训练入口（方案 §8）。

一个患者/访视对一个样本；冻结动态模型与健康参照，``no_grad`` 提取 83 维特征
（全部来自基线信息与基线模拟），训练 :class:`models.twin_outcome.ClinicalOutcomeHead`。

配对 CSV 固定字段（顺序严格）::

    subject_id,baseline_visit_id,followup_visit_id,baseline_bold_path,
    control_path,followup_days,baseline_hamd,followup_hamd,
    baseline_hama,followup_hama,split

- baseline NPZ 键 ``bold``，``[T,F]``，T>=64（取最早 64 TR；profile 用其中最早
  32 TR）；control NPZ 键 ``u``，``[8,F]``。no-treatment 样本必须显式提供全零
  控制 NPZ，不把缺 control_path 猜成 sham。
- 八字段基线条件由现有 merged 表 + 动态模型保存的统计量按 subject_id 对齐。
- 同一 subject_id 不得跨 train/val/test（只做这项直接防泄漏检查）。
- 随访天数只作为 outcome 特征，不传入 TR 积分器。

用法::

    python -m train.twin_outcome --pairs pairs.csv \
        --dynamics_checkpoint <mdd_cat_twin>/best.pt \
        --reference hc_reference.npz --out_dir runs/twin_outcome

缺真实配对数据不阻断代码交付；本入口在无数据时无法拟合权重。
"""
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from models.twin_outcome import (FEATURE_BLOCKS, FEATURE_DIM,
                                 ClinicalOutcomeHead, OutcomeStandardizer,
                                 assemble_features, masked_gaussian_nll)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

PAIR_COLUMNS = ['subject_id', 'baseline_visit_id', 'followup_visit_id',
                'baseline_bold_path', 'control_path', 'followup_days',
                'baseline_hamd', 'followup_hamd', 'baseline_hama',
                'followup_hama', 'split']
OUTCOME_HORIZON = 8          # 基线模拟步数（早期响应特征，方案 §8.1）
BASELINE_TR = 64
PROFILE_TR = 32
SCORE_FIELDS = ('hamd', 'hama')   # CSV 列后缀 → 目标分量顺序


# ══════════════════════════════════════════════════════════════════════════
#  配对数据
# ══════════════════════════════════════════════════════════════════════════

def _parse_score(cell: str) -> Optional[float]:
    cell = (cell or '').strip()
    return float(cell) if cell else None


def load_pairs(path: str) -> List[dict]:
    """读取配对 CSV；表头字段与顺序必须与 PAIR_COLUMNS 完全一致。"""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if header != PAIR_COLUMNS:
            raise ValueError(
                f'配对 CSV 表头必须严格为 {PAIR_COLUMNS}，收到 {header}')
        rows = []
        for i, row in enumerate(reader):
            sid = (row['subject_id'] or '').strip()
            if not sid:
                raise ValueError(f'第 {i + 2} 行 subject_id 为空')
            split = (row['split'] or '').strip()
            if split not in ('train', 'val', 'test'):
                raise ValueError(f'第 {i + 2} 行 split 非法：{split!r}')
            days = (row['followup_days'] or '').strip()
            if not days:
                raise ValueError(f'第 {i + 2} 行 followup_days 缺失（outcome 特征必需）')
            days_v = float(days)
            if days_v < 0 or not np.isfinite(days_v):
                raise ValueError(f'第 {i + 2} 行 followup_days 非法：{days_v}')
            ctrl = (row['control_path'] or '').strip()
            if not ctrl:
                raise ValueError(
                    f'第 {i + 2} 行 control_path 为空：no-treatment 样本必须显式'
                    '提供全零控制 NPZ，不把缺路径猜成 sham。')
            scores = {k: _parse_score(row[f'baseline_{k}'])
                      for k in SCORE_FIELDS}
            scores.update({f'followup_{k}': _parse_score(row[f'followup_{k}'])
                           for k in SCORE_FIELDS})
            rows.append({
                'subject_id': sid,
                'baseline_visit_id': (row['baseline_visit_id'] or '').strip(),
                'followup_visit_id': (row['followup_visit_id'] or '').strip(),
                'baseline_bold_path': (row['baseline_bold_path'] or '').strip(),
                'control_path': ctrl,
                'followup_days': days_v,
                'scores': scores,
                'split': split,
                '_line': i + 2,
            })
    if not rows:
        raise ValueError(f'{path} 无任何配对样本')
    # 防泄漏：同一 subject_id 不得跨 split（只做这项直接检查，不设计新划分系统）
    seen: Dict[str, str] = {}
    for r in rows:
        prev = seen.setdefault(r['subject_id'], r['split'])
        if prev != r['split']:
            raise ValueError(
                f"subject {r['subject_id']} 同时出现在 {prev} 与 {r['split']}："
                '同一被试的访视对必须落在同一 split。')
    return rows


def load_baseline_bold(path: str, num_rois: int) -> np.ndarray:
    """基线 NPZ（键 bold，[T,F]，T>=64）。"""
    raw = np.load(path)
    if 'bold' not in raw:
        raise KeyError(f'{path} 缺少键 bold（现有键={list(raw.keys())}）')
    arr = np.asarray(raw['bold'], dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'{path} bold 期望 [T,F]，收到 {arr.shape}')
    if arr.shape[0] < BASELINE_TR:
        raise ValueError(f'{path} 只有 {arr.shape[0]} 个 TR，基线至少需要 {BASELINE_TR}')
    if arr.shape[1] != num_rois:
        raise ValueError(f'{path} ROI 数 {arr.shape[1]} 与动态模型 num_rois='
                         f'{num_rois} 不一致')
    if not np.isfinite(arr).all():
        raise ValueError(f'{path} bold 含非有限值')
    return arr


def load_control(path: str, num_rois: int) -> np.ndarray:
    """控制 NPZ（键 u，[8,F]，逐区间有效驱动）。"""
    raw = np.load(path)
    if 'u' not in raw:
        raise KeyError(f'{path} 缺少键 u（现有键={list(raw.keys())}）')
    arr = np.asarray(raw['u'], dtype=np.float64)
    if arr.shape != (OUTCOME_HORIZON, num_rois):
        raise ValueError(f'{path} u 期望 [{OUTCOME_HORIZON},{num_rois}]，收到 {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError(f'{path} u 含非有限值')
    return arr


# ══════════════════════════════════════════════════════════════════════════
#  冻结动态模型 + 参照 → 特征提取
# ══════════════════════════════════════════════════════════════════════════

class OutcomeFeatureExtractor:
    """冻结动态模型 + 健康参照 → 每个访视对 83 维特征（no_grad）。"""

    def __init__(self, ckpt_path: str, reference_path: str,
                 condition_file: Optional[str] = None, device=None):
        from experiments.evaluate_variant import (build_model_from_args,
                                                  load_condition_bundle,
                                                  load_train_args)
        from analysis.checkpoint import load_checkpoint_compat
        from analysis.twin_reference import load_reference
        from utils.dataloader import load_condition_table
        from utils.condition_schema import ConditionStats, schema_for

        self.device = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = load_train_args(ckpt_path)
        self.bundle = load_condition_bundle(ckpt_path)
        self.model = build_model_from_args(self.args, self.device,
                                           condition_bundle=self.bundle)
        self.model, compat = load_checkpoint_compat(self.model, ckpt_path,
                                                    self.device, strict=True)
        if compat.get('missing') or compat.get('unexpected'):
            raise RuntimeError(f'动态模型权重加载不完整：{compat}')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.model.dynamics_mode != 'graph_ode' or not self.model.personalized:
            raise ValueError(
                'outcome 读出首版要求 graph_ode + personalized 的动态 checkpoint'
                f'（当前 dynamics_mode={self.model.dynamics_mode}, '
                f'personalized={self.model.personalized}）。')
        if getattr(self.model, 'conditioner', None) is None:
            raise ValueError('outcome 需要 C=32 的八字段条件编码器，'
                             '该 checkpoint 未构建 conditioner（arm=none）。')

        self.num_rois = int(self.args.num_rois)
        self.calibration_length = int(getattr(self.args, 'calibration_length', 0) or 0)
        if self.calibration_length < 1:
            raise ValueError('动态 checkpoint 的 calibration_length 必须 >= 1 '
                             '（raw 前缀校准口径）。')
        self.reference = load_reference(reference_path)
        if int(self.reference['n_features']) != self.num_rois:
            raise ValueError('参照 F 与动态模型 num_rois 不一致：'
                             f"{self.reference['n_features']} vs {self.num_rois}")

        group = ((self.bundle or {}).get('config') or {}).get('group')
        cond_file = condition_file or getattr(self.args, 'condition_file', None)
        if not group or not cond_file:
            raise ValueError('无法确定条件表：bundle.config.group 与 '
                             '--condition_file（缺省取 checkpoint 训练值）均需存在。')
        data_root = getattr(self.args, 'data_root')
        self.table = load_condition_table(data_root, cond_file, group)
        self.cond_stats = ConditionStats.from_dict(
            schema_for(group), (self.bundle or {}).get('stats') or {})
        from analysis.twin_reference import _demo_field_indices
        self.demo_idx = _demo_field_indices(self.table.specs)
        self.seed = int(getattr(self.args, 'seed', 2024))

    # ---- 条件与人口学 ----
    @torch.no_grad()
    def condition_for(self, subject_id: str) -> tuple:
        """八字段基线条件（merged 表 + 动态模型统计量对齐）→ (cond dict, c [1,32])。"""
        values, mask = self.table.align([str(subject_id)])
        z, emb = self.cond_stats.encode(values, mask)
        cond = {
            'cond_z': torch.from_numpy(z).to(self.device),
            'cond_code': torch.from_numpy(emb).to(self.device),
            'cond_mask': torch.from_numpy(mask.astype(np.float32)).to(self.device),
        }
        c = self.model._prepare_condition(None, condition=cond,
                                          condition_epoch=None)   # [1,C]
        return cond, c

    @torch.no_grad()
    def demo_for(self, subject_id: str) -> Dict[str, Optional[float]]:
        values, mask = self.table.align([str(subject_id)])
        out: Dict[str, Optional[float]] = {}
        for name, key in (('Age', 'Age'), ('education', 'education'), ('Sex', 'Sex')):
            j = self.demo_idx[key]
            out[name] = (float(values[0, j]) if mask[0, j] > 0
                         and np.isfinite(values[0, j]) else None)
        return out

    # ---- 单对特征 ----
    @torch.no_grad()
    def extract_from_arrays(self, bold_raw: np.ndarray, u: np.ndarray,
                            followup_days: float, subject_id: str) -> np.ndarray:
        """已加载的原始数组 → 83 维特征（推理侧 :mod:`inference_twin` 复用同一拼接路径）。

        bold_raw ``[T,F]`` 原始尺度（T>=64）；u ``[8,F]``。基线取最早 64 TR；
        profile 用其中最早 32 TR；从第 64 TR 末状态模拟 H=8（有/无控制两条
        轨迹）；参照也用同一基线 64 TR。真实随访影像/随访条件不进入任何输入。
        """
        from analysis.twin_reference import describe_history, score_reference

        sid = str(subject_id)
        hist_raw = np.asarray(bold_raw, dtype=np.float64)[:BASELINE_TR]   # [64,F]
        if hist_raw.shape[0] < BASELINE_TR:
            raise ValueError(f'基线只有 {hist_raw.shape[0]} 个 TR，至少需要 {BASELINE_TR}')
        cal = hist_raw[:self.calibration_length]
        mu0 = cal.mean(axis=0)                                     # [F]
        s0 = np.sqrt(cal.var(axis=0) + 1e-6)                       # [F]
        hist = (hist_raw - mu0) / s0                               # [64,F] 数据尺度

        f_dim = self.num_rois
        cal32 = torch.from_numpy(
            hist[:PROFILE_TR].T.astype(np.float32))[None, :, None, :].to(self.device)
        profile = self.model.calibrate_profile(cal32)[0].cpu().numpy()   # [32]

        cond, c = self.condition_for(sid)
        cond_cpu = {k: v for k, v in cond.items()}
        c_vec = c[0].cpu().numpy()                                       # [32]

        ref = score_reference(hist, self.demo_for(sid), self.reference)
        ref_summary = np.abs(ref['normal_residual'][0]).mean(axis=0)     # [6]
        desc_summary = describe_history(hist)[0].mean(axis=0)            # [6]

        u = np.asarray(u, dtype=np.float64)
        if u.shape != (OUTCOME_HORIZON, f_dim):
            raise ValueError(f'控制 u 期望 [{OUTCOME_HORIZON},{f_dim}]，收到 {u.shape}')
        u_t = torch.from_numpy(u.astype(np.float32))[None].to(self.device)  # [1,8,F]
        hist_t = torch.from_numpy(hist.T.astype(np.float32))[None].to(self.device)
        sc_eye = torch.eye(f_dim, device=self.device)[None]          # 占位（history_fc）
        kw = dict(sc_matrix=sc_eye, pathology_score=None, steps=OUTCOME_HORIZON,
                  condition=cond_cpu, calibration_bold=cal32)
        traj_u = self.model.rollout_next_states(hist_t, controls=u_t, **kw)
        traj_0 = self.model.rollout_next_states(hist_t, **kw)        # 零控制分支
        diff = (traj_u - traj_0)[0].cpu().numpy()                    # [8,F]

        diff_summary = np.array([
            diff.mean(),
            np.sqrt((diff ** 2).mean()),
            np.abs(diff[-1]).mean(),
        ])
        ctrl_summary = np.array([
            np.abs(u).mean(),
            np.abs(u).max(),
            float((np.abs(u) > 0).any(axis=0).mean()),
        ])

        feat = assemble_features(profile, c_vec, desc_summary, ref_summary,
                                 diff_summary, ctrl_summary, followup_days)
        return feat

    def extract(self, row: dict) -> tuple:
        """一个访视对 → (feat83 [83], baseline_scores [2])。"""
        bold = load_baseline_bold(row['baseline_bold_path'], self.num_rois)
        u = load_control(row['control_path'], self.num_rois)
        feat = self.extract_from_arrays(bold, u, row['followup_days'],
                                        row['subject_id'])
        base = np.array([row['scores']['hamd'], row['scores']['hama']],
                        dtype=np.float64)
        return feat, base


# ══════════════════════════════════════════════════════════════════════════
#  训练
# ══════════════════════════════════════════════════════════════════════════

def build_pairs_targets(rows: List[dict]) -> tuple:
    """配对行 → (targets [N,2], mask [N,2])；mask_j 要求基线和随访都存在。"""
    n = len(rows)
    targets = np.zeros((n, 2), dtype=np.float64)
    mask = np.zeros((n, 2), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, k in enumerate(SCORE_FIELDS):
            b, fu = r['scores'][k], r['scores'][f'followup_{k}']
            if b is not None and fu is not None:
                targets[i, j] = fu - b            # followup - baseline
                mask[i, j] = 1.0
    return targets, mask


def main():
    ap = argparse.ArgumentParser(description='独立临床结局读出训练（方案 §8）')
    ap.add_argument('--pairs', required=True, help='配对 CSV（字段严格见模块 docstring）')
    ap.add_argument('--dynamics_checkpoint', required=True,
                    help='graph_ode + personalized 动态 checkpoint（冻结）')
    ap.add_argument('--reference', required=True, help='健康参照 NPZ（twin_reference fit 产出）')
    ap.add_argument('--condition_file', default='',
                    help='八字段条件 merged 表；缺省用动态 checkpoint 训练时的表')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0,
                    help='0=沿用动态 checkpoint 保存的 seed')
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    torch.manual_seed(0)
    extractor = OutcomeFeatureExtractor(
        args.dynamics_checkpoint, args.reference,
        condition_file=(args.condition_file or None),
        device=(torch.device(args.device) if args.device else None))
    seed = args.seed if args.seed else extractor.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = extractor.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pairs(args.pairs)
    splits = {s: [r for r in rows if r['split'] == s]
              for s in ('train', 'val', 'test')}
    log.info('配对样本：train=%d val=%d test=%d', *(len(splits[s]) for s in splits))

    # ---- no_grad 特征提取（一次缓存全部）----
    feats, bases, masks = {}, {}, {}
    for s, srows in splits.items():
        if not srows:
            continue
        fs, bs = [], []
        for r in srows:
            f, b = extractor.extract(r)
            fs.append(f)
            bs.append(b)
        feats[s] = np.stack(fs, axis=0)
        bases[s] = np.stack(bs, axis=0)
        base_mu = np.stack([np.array([r['scores']['hamd'],
                                      r['scores']['hama']], dtype=np.float64)
                            for r in srows], axis=0)
        # mask_j = 基线与随访都存在
        masks[s] = np.stack([
            np.array([1.0 if (r['scores']['hamd'] is not None
                              and r['scores'][f'followup_hamd'] is not None) else 0.0,
                      1.0 if (r['scores']['hama'] is not None
                              and r['scores'][f'followup_hama'] is not None) else 0.0])
            for r in srows], axis=0)
        del base_mu
    if 'train' not in feats or len(feats['train']) == 0:
        raise ValueError('配对 CSV 中没有 train 样本，无法拟合统计量。')

    deltas_train, mask_train = build_pairs_targets(splits['train'])
    standardizer = OutcomeStandardizer.fit(feats['train'], deltas_train, mask_train)
    feats_std = {s: standardizer.transform_features(v) for s, v in feats.items()}
    targets_std = {s: standardizer.transform_delta(build_pairs_targets(splits[s])[0])
                   for s in feats}
    for s in ('val', 'test'):
        if s in masks and not masks[s].any():
            log.warning('%s split 无任何有效标签（仅记录，不阻断）', s)

    head = ClinicalOutcomeHead(in_dim=FEATURE_DIM, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def run_epoch(split, train_mode):
        ds_feats = feats_std[split]
        ds_t = targets_std[split]
        ds_m = masks[split]
        n = len(ds_feats)
        idx_all = np.arange(n)
        rng = np.random.default_rng(seed + (0 if train_mode else 1))
        if train_mode:
            rng.shuffle(idx_all)
        head.train(train_mode)
        total, count = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for lo in range(0, n, args.batch_size):
                sel = idx_all[lo:lo + args.batch_size]
                x = torch.from_numpy(ds_feats[sel]).to(device)
                t = torch.from_numpy(ds_t[sel]).to(device)
                m = torch.from_numpy(ds_m[sel]).to(device)
                d_hat, lv = head(x)
                loss = masked_gaussian_nll(d_hat, lv, t, m)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total += float(loss.item()) * len(sel)
                count += len(sel)
        return total / max(count, 1)

    best_val = float('inf')
    history = []
    for epoch in range(args.epochs):
        tr = run_epoch('train', True)
        va = run_epoch('val', False) if 'val' in feats else tr
        history.append({'epoch': epoch + 1, 'train_nll': tr, 'val_nll': va})
        if va < best_val:
            best_val = va
            payload = {
                'arch': 'twin_outcome_v1',
                'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                'standardizer': standardizer.to_dict(),
                'config': {
                    'feature_blocks': [[n, d] for n, d in FEATURE_BLOCKS],
                    'in_dim': FEATURE_DIM, 'hidden_dim': 64,
                    'dynamics_checkpoint': str(args.dynamics_checkpoint),
                    'reference': str(args.reference),
                    'condition_file': (args.condition_file
                                       or getattr(extractor.args, 'condition_file', '')),
                    'seed': seed, 'lr': args.lr, 'weight_decay': args.weight_decay,
                    'batch_size': args.batch_size, 'epochs': args.epochs,
                },
                'history': history,
            }
            torch.save(payload, out_dir / 'outcome_best.pt')
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            log.info('epoch %d | train NLL=%.4f | val NLL=%.4f | best=%.4f',
                     epoch + 1, tr, va, best_val)
    torch.save(payload, out_dir / 'outcome_last.pt')
    with open(out_dir / 'outcome_history.json', 'w', encoding='utf-8') as f:
        json.dump({'history': history, 'best_val_nll': best_val,
                   'n_pairs': {s: len(v) for s, v in splits.items()}},
                  f, ensure_ascii=False, indent=2)
    log.info('完成：best val NLL=%.4f → %s', best_val, out_dir / 'outcome_best.pt')


if __name__ == '__main__':
    main()
