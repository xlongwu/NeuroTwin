# coding=utf-8
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from utils.augmentation import BrainSignalAugmentation


def load_clinical_scores(data_root, clinical_file: str, fields: List[str],
                         missing: str) -> Dict[str, np.ndarray]:
    """读取临床评分表 → {subj_id: np.ndarray[D]}（pretrain 不使用该表）。

    口径与既有实现保持一致（ID 去空格、缺失策略 drop/mean/zero），供数据集与
    离线转换脚本共用，避免两份实现漂移。
    """
    if missing not in ('drop', 'zero', 'mean'):
        raise ValueError(f"pathology_missing 仅支持 drop/zero/mean，收到 '{missing}'")
    clinical_path = Path(data_root) / clinical_file
    if not clinical_path.exists():
        raise FileNotFoundError(f"未找到临床评分文件：{clinical_path}")

    df = pd.read_excel(clinical_path)
    required_cols = {'ID', *fields}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"clinical file must contain columns {required_cols}, got {set(df.columns)}")

    df = df[['ID', *fields]].copy()
    df['ID'] = df['ID'].astype(str).str.strip()
    value_cols = list(fields)
    if missing == 'drop':
        df = df.dropna(subset=value_cols)
    elif missing == 'mean':
        df[value_cols] = df[value_cols].fillna(df[value_cols].mean(numeric_only=True))
        df = df.dropna(subset=value_cols)
    else:  # zero
        df[value_cols] = df[value_cols].fillna(0.0)
    values = df[value_cols].to_numpy(dtype=np.float32)
    out = {k: v for k, v in zip(df['ID'].tolist(), list(values))}
    print(
        f"成功加载临床评分表，共包含 {len(out)} 个被试的 "
        f"{'/'.join(value_cols)} 数据（缺失策略: {missing}）。"
    )
    return out


def _extract_mat(mat_dict, preferred_keys=None):
    """从 loadmat 字典中取出目标矩阵（优先指定键，其次唯一候选）。

    由原数据集内部的 _extract_mat 方法提取为模块级函数，供数据集与
    utils/convert_mat_to_npz.py 复用，保证转换与训练读取口径一致。
    """
    if preferred_keys is not None:
        for key in preferred_keys:
            if key in mat_dict and isinstance(mat_dict[key], np.ndarray):
                return mat_dict[key]

    valid = [
        (k, v) for k, v in mat_dict.items()
        if not k.startswith('__') and isinstance(v, np.ndarray)
    ]

    if len(valid) == 1:
        return valid[0][1]
    if len(valid) == 0:
        raise ValueError("未找到有效的矩阵数据")
    raise ValueError(f"mat 文件中存在多个候选变量: {[k for k, _ in valid]}，请显式指定键名")


def read_mat_array(path, preferred_keys=None) -> np.ndarray:
    """读取单个 mat 文件并返回目标矩阵（无缓存版本，供数据集/离线转换复用）。"""
    mat = sio.loadmat(path)
    arr = _extract_mat(mat, preferred_keys=preferred_keys)
    return np.asarray(arr)


# ══════════════════════════════════════════════════════════════════════════
#  Next-Timepoint 任务（task_mode='next_timepoint'）
#
#  Raw ROI BOLD [F, T] → 每个被试一条**连续序列**（不再切窗、不做 window averaging）
#      训练：随机 (K, t) → history = series[:, t-K+1 : t+1]   [F, 1, K]
#                          target  = series[:, t + offsets]     [F, H]
#      其中整段 context 作为**单个窗口**（W=1）、K 个 TR 作为该窗口的时间轴（S=K），
#      因此主干（BrainMDM 时间卷积 / GraphODE / 预测头）无需改写，只依赖 S 轴的
#      变长前缀切片；同一 batch 内 K 一致（按 K 分桶），不引入 padding。
#      offsets = forecast_offsets（默认 [1] = 预测下一个 TR；MTP 时如 [1,2,4,8]）。
# ══════════════════════════════════════════════════════════════════════════


class NeuroTwinNextPointDataset(Dataset):
    """next_timepoint 任务的被试级连续 BOLD 数据集。

    每个被试直接读取一条**连续 ROI BOLD 序列** [F, T]（不再切窗、不做 window
    averaging），按 TR 直接索引：

      - context 为**单个窗口** W=1、其时间轴 S=K（K 个连续 TR 作为窗口内时间轴）；
      - 目标为 ``t + offsets`` 的真值（默认 offsets=[1] 即预测下一个 TR；MTP 时可如
        [1,2,4,8]），预测严格晚于 context 末位 x_t，context 不含任何目标；
      - 归一化统计量（BrainRevIN）只由历史 context 估计，未来 target 不参与。

    自管理：连续序列读取（Normlize 优先，缺失时按 50% 重叠滑窗重构）、SC 清洗、
    被试级缓存、ROI 顺序校验。
    """

    BOLD_SOURCES = ('auto', 'normlize', 'windows')

    def __init__(
        self,
        data_root,
        mode: str = 'pretrain',
        context_min: int = 16,
        context_max: int = 64,
        context_lengths: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        pathology_input_dim: int = 1,
        pathology_fields: Optional[List[str]] = None,
        pathology_missing: str = 'drop',
        clinical_file: str = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx',
        total_windows: int = 9,
        seq_len: int = 30,
        bold_source: str = 'auto',
        subject_cache_size: int = 256,
        cache_in_memory: bool = False,
    ):
        self.data_root = Path(data_root)
        self.mode = mode
        self.context_min = int(context_min)
        self.context_max = int(context_max)
        self.context_lengths = ([int(k) for k in context_lengths]
                                if context_lengths else None)
        self.offsets = tuple(int(o) for o in (offsets or [1]))
        if not self.offsets or min(self.offsets) < 1:
            raise ValueError(f"forecast_offsets 必须为正整数，收到 {self.offsets}")
        self.max_offset = max(self.offsets)
        if self.context_min < 1 or self.context_max < self.context_min:
            raise ValueError(
                f"context_min/context_max 非法：{self.context_min}/{self.context_max}")

        self.total_windows = int(total_windows)
        self.source_window_len = int(seq_len)
        if bold_source not in self.BOLD_SOURCES:
            raise ValueError(f"bold_source 仅支持 {self.BOLD_SOURCES}，收到 '{bold_source}'")
        self.bold_source = bold_source
        self.cache_in_memory = cache_in_memory
        self.subject_cache_size = int(subject_cache_size)

        self.pathology_input_dim = pathology_input_dim
        # 病理条件字段（默认仅 HAMD 总分；条目级缺失率高，向量路径需显式开启）
        self.pathology_fields = list(pathology_fields) if pathology_fields else ['HAMD']
        if pathology_missing not in ('drop', 'zero', 'mean'):
            raise ValueError(
                f"pathology_missing 仅支持 drop/zero/mean，收到 '{pathology_missing}'")
        self.pathology_missing = pathology_missing
        if self.mode == 'finetune' and self.pathology_input_dim != len(self.pathology_fields):
            raise ValueError(
                f"pathology_input_dim={self.pathology_input_dim} 与 pathology_fields="
                f"{self.pathology_fields}（宽度 {len(self.pathology_fields)}）不一致。")

        self.clinical_dict: Dict[str, np.ndarray] = {}
        if self.mode == 'finetune':
            self.clinical_dict = load_clinical_scores(
                self.data_root, clinical_file, self.pathology_fields,
                self.pathology_missing)

        group_folder = 'HC' if mode == 'pretrain' else 'MDD'
        self.window_dir = self.data_root / 'ROISignals_window' / group_folder
        self.sc_dir = self.data_root / 'Mask' / group_folder
        # 聚合 npz 缓存目录（由 utils/convert_mat_to_npz.py 生成）：
        # 每被试一个 npz（sc + windows [W,F,S]），存在则优先读取，消除小 mat 文件 I/O
        self.npz_dir = self.data_root / 'npz_cache' / group_folder
        self.npz_dir = self.npz_dir if self.npz_dir.is_dir() else None
        self.normlize_dir = self.data_root / 'Normlize' / group_folder

        self.subject_ids: List[str] = []
        self._subject_pos: Dict[str, int] = {}
        self.subject_series_len: Dict[str, int] = {}
        self.subject_n_tr: Dict[str, int] = {}
        self.subject_to_score: Dict[str, float] = {}
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._mat_cache: Dict[str, np.ndarray] = {}
        self._source_logged = False
        self._build_index()

    # ------------------------------------------------------------------
    # 索引与数据加载
    # ------------------------------------------------------------------
    def _normlize_path(self, subj_id: str) -> Path:
        return self.normlize_dir / f"ROISignals_{subj_id}.mat"

    def _window_paths(self, subj_id: str) -> List[Path]:
        return [self.window_dir / f"ROISignals_{subj_id}-{i}.mat"
                for i in range(1, self.total_windows + 1)]

    def _probe_series_length(self, subj_id: str) -> Optional[int]:
        """探测单被试连续序列长度 T（不读取全部数据，仅取 shape）。

        优先 Normlize 连续序列（原始 ROI BOLD，[T, F] 或 [F, T]）；不可用时按
        ``ROISignals_window`` 的 50% 重叠布局推算：偶数序号窗（1,3,5,...）恰好
        覆盖 0:S, S:2S, ... 的连续片段（见 data/utils/window_stride.py）。
        """
        if self.bold_source in ('auto', 'normlize'):
            path = self._normlize_path(subj_id)
            if path.exists():
                arr = self._read_mat_array(
                    path, preferred_keys=['ROISignals', 'Data', 'bold'])
                return int(max(arr.shape))
        if self.bold_source == 'normlize':
            return None
        window_paths = self._window_paths(subj_id)
        if not all(p.exists() for p in window_paths):
            return None
        n_used = (self.total_windows + 1) // 2
        return n_used * self.source_window_len

    def _build_index(self):
        """按 TR 直接索引：每个被试一条连续序列 [F, T]，T 即该被试的 TR 总数。"""
        sc_files = sorted(self.sc_dir.glob('Mask_*.mat'), key=lambda p: p.stem)
        skipped_clinical = skipped_missing = 0
        too_short: List[str] = []
        min_required = self.context_min + 1  # 至少 context_min 段历史 + 1 个目标 TR

        for sc_file in sc_files:
            subj_id = str(sc_file.stem.replace('Mask_', '')).strip()
            if self.mode == 'finetune' and subj_id not in self.clinical_dict:
                skipped_clinical += 1
                continue
            series_len = self._probe_series_length(subj_id)
            if series_len is None or series_len <= 0:
                skipped_missing += 1
                continue
            if series_len < min_required:
                too_short.append(f"{subj_id}(T={series_len})")
                continue
            self._subject_pos[subj_id] = len(self.subject_ids)
            self.subject_ids.append(subj_id)
            self.subject_series_len[subj_id] = series_len
            self.subject_n_tr[subj_id] = series_len
            if self.mode == 'finetune':
                # 分层划分使用首个字段（默认 HAMD 总分）作为标量分层依据
                self.subject_to_score[subj_id] = float(self.clinical_dict[subj_id][0])

        if not self.subject_ids:
            raise ValueError(
                f"next_timepoint 数据集为空：mode={self.mode}, context_min={self.context_min}"
                f"（需 T >= {min_required}）。"
                "请检查 data_root 下的 Normlize / ROISignals_window / Mask 目录。")

        series_lens = sorted(self.subject_n_tr.values())
        print(
            f"[next_timepoint] 数据集构建完毕 | 模式: {self.mode} | 有效被试数: "
            f"{len(self.subject_ids)} | TR 数 min/median/max="
            f"{series_lens[0]}/{int(np.median(series_lens))}/{series_lens[-1]}"
        )
        print(
            f"[next_timepoint] 数据源={self.bold_source} | 跳过: "
            f"无临床评分 {skipped_clinical} | 数据缺失 {skipped_missing} | "
            f"序列过短 {len(too_short)}"
            + (f"（例如 {too_short[:3]}）" if too_short else "")
        )

    @staticmethod
    def _normalize_sc(sc_matrix: np.ndarray) -> np.ndarray:
        """SC 清洗口径：nan→0、对称化、clip(0)、log1p、按正值 99 分位归一、clip(0,1)。"""
        sc = np.asarray(sc_matrix, dtype=np.float32)
        sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
        sc = 0.5 * (sc + sc.T)
        sc = np.clip(sc, a_min=0.0, a_max=None)
        sc = np.log1p(sc)

        positive = sc[sc > 0]
        if positive.size > 0:
            scale = np.percentile(positive, 99)
            if scale > 1e-8:
                sc = sc / scale
        sc = np.clip(sc, 0.0, 1.0)
        return sc.astype(np.float32)

    def _read_mat_array(self, path: Path, preferred_keys=None) -> np.ndarray:
        key = str(path)
        if key in self._mat_cache:
            return self._mat_cache[key]
        arr = read_mat_array(path, preferred_keys=preferred_keys)
        arr = np.asarray(arr)
        if self.cache_in_memory:
            self._mat_cache[key] = arr
        return arr

    def _load_series(self, subj_id: str, n_rois: int) -> np.ndarray:
        """返回连续 ROI BOLD 序列 [F, T]（float32）。"""
        def _orient(arr: np.ndarray, src: str) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{subj_id} 的 BOLD 期望二维，收到 {arr.shape}（{src}）")
            if arr.shape[0] == n_rois and arr.shape[1] != n_rois:
                return arr
            if arr.shape[1] == n_rois and arr.shape[0] != n_rois:
                return arr.T
            raise ValueError(
                f"{subj_id} 的 BOLD 形状 {arr.shape} 无法与 ROI 数 {n_rois} 对齐（{src}）")

        if self.bold_source in ('auto', 'normlize'):
            path = self._normlize_path(subj_id)
            if path.exists():
                if not self._source_logged:
                    print(f"[next_timepoint] 连续 BOLD 数据源: {path.parent}（[T,F] → [F,T]）")
                    self._source_logged = True
                return _orient(self._read_mat_array(
                    path, preferred_keys=['ROISignals', 'Data', 'bold']), 'Normlize')

        # fallback：从 50% 重叠滑窗重构连续序列（取偶数序号窗 = 0:S, S:2S, ...）
        if self.source_window_len % 2 != 0:
            raise ValueError(
                f"窗口长度 seq_len={self.source_window_len} 非偶数，无法按 50% 重叠约定重构；"
                "请确认 bold_source 或 --seq_len。")
        n_used = (self.total_windows + 1) // 2
        segments: List[np.ndarray] = []
        npz_path = self.npz_dir / f"{subj_id}.npz" if self.npz_dir is not None else None
        if npz_path is not None and npz_path.exists():
            with np.load(npz_path) as z:
                windows = np.asarray(z['windows'])
            if windows.shape[0] != self.total_windows:
                raise ValueError(
                    f"{subj_id} 的 npz windows 形状异常 {windows.shape}（期望 W={self.total_windows}）")
            for i in range(0, self.total_windows, 2):
                segments.append(windows[i])           # [F, S]
        else:
            for i in range(1, self.total_windows + 1, 2):
                path = self.window_dir / f"ROISignals_{subj_id}-{i}.mat"
                if not path.exists():
                    raise FileNotFoundError(f"{subj_id} 缺少滑窗文件 {path.name}")
                arr = _orient(self._read_mat_array(
                    path, preferred_keys=['ROISignals', 'data', 'bold']), 'windows')
                if arr.shape[1] != self.source_window_len:
                    raise ValueError(
                        f"{subj_id} 窗口 {path.name} 长度 {arr.shape[1]} != seq_len "
                        f"{self.source_window_len}")
                segments.append(arr)
        series = np.concatenate(segments[:n_used], axis=1)
        if not self._source_logged:
            print("[next_timepoint] 未找到 Normlize 连续序列，回退用 50% 重叠滑窗重构"
                  "（偶数序号窗拼接，布局见 data/utils/window_stride.py）")
            self._source_logged = True
        return series.astype(np.float32, copy=False)

    def _get_subject(self, subj_id: str) -> Dict[str, np.ndarray]:
        cached = self._cache.get(subj_id, None)
        if cached is not None:
            return cached

        sc_path = self.sc_dir / f"Mask_{subj_id}.mat"
        sc_matrix = self._read_mat_array(
            sc_path, preferred_keys=['SC', 'sc_matrix', 'mask', 'Mask'])
        if sc_matrix.ndim != 2 or sc_matrix.shape[0] != sc_matrix.shape[1]:
            raise ValueError(f"SC/mask must be square [F, F], got {sc_matrix.shape} for {subj_id}")
        sc_matrix = self._normalize_sc(sc_matrix)
        n_rois = int(sc_matrix.shape[0])

        series = self._load_series(subj_id, n_rois)
        if series.shape[0] != n_rois:
            raise ValueError(
                f"ROI 顺序/数量不一致：{subj_id} BOLD {series.shape[0]} vs SC {n_rois}；"
                "next_timepoint 要求 BOLD ROI i 与 SC ROI i 严格对应。")
        T = int(series.shape[1])
        expected = self.subject_series_len.get(subj_id, None)
        if expected is not None and T != expected:
            raise ValueError(
                f"{subj_id} 实际序列长度 {T} 与索引阶段探测的 {expected} 不一致："
                "请检查数据目录是否在两次运行之间发生变化。")
        if T < self.context_min + 1:
            raise ValueError(
                f"{subj_id} 只有 {T} 个 TR，不足以构成 context_min+1="
                f"{self.context_min + 1} 的最小样本。")

        entry: Dict[str, np.ndarray] = {
            'series': np.ascontiguousarray(series.astype(np.float32, copy=True)),  # [F, T]
            'sc': sc_matrix,                            # [F, F]
            'subj_id': subj_id,
            'series_len': T,
        }
        if self.mode == 'finetune':
            entry['pathology_score'] = np.asarray(
                self.clinical_dict[subj_id], dtype=np.float32)

        if self.subject_cache_size > 0:
            if len(self._cache) >= self.subject_cache_size:
                # 简单 FIFO 淘汰：只控制内存上界，不追求命中率
                self._cache.pop(next(iter(self._cache)))
            self._cache[subj_id] = entry
        return entry

    # ------------------------------------------------------------------
    # Dataset 接口：返回“完整被试序列”，不在离线阶段固定 x/y
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        return dict(self._get_subject(self.subject_ids[idx]))

    # ------------------------------------------------------------------
    # 只读访问器（供 split / 视图构造使用）
    # ------------------------------------------------------------------
    def get_subject_ids(self) -> List[str]:
        return list(self.subject_ids)

    def get_subject_indices(self, subj_id: str) -> List[int]:
        """本数据集以“被试”为单位（每个被试一条连续序列），索引即 subject_ids 位置。"""
        pos = self._subject_pos.get(str(subj_id), None)
        return [] if pos is None else [pos]

    def get_pathology_scalar_by_subject(self, subj_id: str) -> Optional[float]:
        """返回首个病理字段的标量值（默认 HAMD 总分），仅供分层/统计使用。"""
        if self.mode != 'finetune':
            return None
        score = self.subject_to_score.get(str(subj_id), None)
        return None if score is None else float(score)

    def get_pathology_vector_by_subject(self, subj_id: str) -> Optional[np.ndarray]:
        """返回完整病理条件向量 [D]（默认 [HAMD]）。"""
        if self.mode != 'finetune':
            return None
        vec = self.clinical_dict.get(str(subj_id), None)
        return None if vec is None else np.asarray(vec, dtype=np.float32)

    def n_tr(self, subj_id: str) -> int:
        """该被试的 TR 总数 T（连续序列长度）。"""
        return int(self.subject_n_tr[subj_id])

    def get_series(self, subj_id: str) -> np.ndarray:
        """返回被试的连续 BOLD 序列 [F, T]（只读）。

        供评估侧拟合 AR(1) baseline 等只读用途；数据来源与训练完全一致
        （Normlize 连续序列，缺失时按 50% 重叠滑窗重构）。
        """
        return self._get_subject(str(subj_id))['series']

    def legal_context_lengths(self, subj_id: str) -> List[int]:
        """合法 context 长度：K <= min(context_max, T-1)（至少 1 个未来 TR）。"""
        n = self.n_tr(subj_id)
        upper = min(self.context_max, n - 1)
        if self.context_lengths:
            return [k for k in self.context_lengths if k <= upper]
        return list(range(self.context_min, upper + 1)) if upper >= self.context_min else []

    def legal_anchor_range(self, subj_id: str, context_len: int,
                           max_offset: int = None) -> Tuple[int, int]:
        """合法预测位置区间（含端点，0 基 cutoff t）：t-K+1 >= 0 且 t+max_offset <= T-1。"""
        n = self.n_tr(subj_id)
        off = int(max_offset) if max_offset else self.max_offset
        return context_len - 1, n - 1 - off

    def legal_task_count(self, subj_id: str) -> int:
        """该被试的合法 (K, t) 组合数（训练视图的样本预算）。"""
        total = 0
        for k in self.legal_context_lengths(subj_id):
            lo, hi = self.legal_anchor_range(subj_id, k)
            if hi >= lo:
                total += hi - lo + 1
        return total

    # ------------------------------------------------------------------
    # 样本切片（视图共用）
    # ------------------------------------------------------------------
    def make_sample(self, subj_pos: int, context_len: int, t: int,
                    future_steps: int = 0) -> Dict:
        """按 (K, t) 切出一个 next-timepoint 样本（0 基 TR 序号）。

        Returns dict:
            x           [F, 1, K]   context（单窗口 W=1、时间轴 S=K，主干内部布局）
            y           [F, H]      x_(t+offset) 真值（H = len(forecast_offsets)）
            x_last      [F]         context 末位 x_t（delta 重建与 baseline 使用）
            future      [F, R]      t+1..t+R 的真值（仅 future_steps>0，供 rollout 评分）
            sc / subj_id / context_len / cutoff / target_mask [H]
        """
        subj_id = self.subject_ids[subj_pos]
        n = self.n_tr(subj_id)
        if context_len < 1 or context_len > self.context_max:
            raise ValueError(f"非法 context 长度 K={context_len}（{subj_id}）")
        lo, hi = self.legal_anchor_range(subj_id, context_len)
        if not (lo <= t <= hi):
            raise ValueError(
                f"样本越界：{subj_id} K={context_len}, t={t} 不在合法区间 [{lo}, {hi}]"
                f"（T={n}, max_offset={self.max_offset}）")
        entry = self._get_subject(subj_id)
        series = entry['series']                       # [F, T]
        x = series[:, t - context_len + 1: t + 1]      # [F, K]
        target_idx = np.asarray([t + o for o in self.offsets], dtype=np.int64)
        out = {
            'x': torch.from_numpy(np.ascontiguousarray(
                np.asarray(x[:, None, :], dtype=np.float32))),        # [F,1,K]
            'y': torch.from_numpy(np.ascontiguousarray(
                np.asarray(series[:, target_idx], dtype=np.float32))),  # [F,H]
            'x_last': torch.from_numpy(np.ascontiguousarray(
                np.asarray(series[:, t], dtype=np.float32))),           # [F]
            'sc': torch.from_numpy(entry['sc']),
            'subj_id': subj_id,
            'context_len': int(context_len),
            'cutoff': int(t),
            'target_mask': torch.ones(len(self.offsets), dtype=torch.float32),
        }
        if future_steps > 0:
            future_end = min(int(future_steps), n - 1 - t)
            future = np.zeros((series.shape[0], int(future_steps)), dtype=np.float32)
            if future_end > 0:
                future[:, :future_end] = series[:, t + 1: t + 1 + future_end]
            mask = np.zeros(int(future_steps), dtype=np.float32)
            mask[:future_end] = 1.0
            out['future'] = torch.from_numpy(future)
            out['future_mask'] = torch.from_numpy(mask)
        if self.mode == 'finetune':
            out['pathology_score'] = torch.tensor(
                np.asarray(entry['pathology_score'], dtype=np.float32),
                dtype=torch.float32)
        return out


class NextPointTrainView(Dataset):
    """next_timepoint 训练视图：动态采样 (K, t)，不落盘预切样本。

    - 每个 item 绑定 ``(subject, K)``：K 在构造期由 RNG 从 ``context_lengths``
      （或 [context_min, context_max] 连续区间）抽取，fork 安全，供按 K 分桶的
      batch sampler 使用；
    - ``t`` 在 ``__getitem__`` 内由 worker 局部 RNG 采样于
      ``[K-1, T-1-max_offset]``（保证全部 forecast_offsets 都有真值），
      因此同一 item 在不同 epoch 会得到不同的 (K, t)；
    - 样本预算 ``samples_per_subject``：>0 时按该值取样；=0（auto）时取
      ``min(合法 (K,t) 组合数, AUTO_SAMPLES_PER_SUBJECT_CAP)``，防止单 epoch
      计算量随 T 线性膨胀。
    """

    #: auto 预算上限（与轨迹训练视图一致，控制单 epoch 计算量）
    AUTO_SAMPLES_PER_SUBJECT_CAP = 12

    def __init__(self, base: NeuroTwinNextPointDataset, subject_indices: List[int],
                 samples_per_subject: int = 0, seed: int = 2024,
                 augmentation: Optional[BrainSignalAugmentation] = None,
                 random_context: bool = True, random_cutoff: bool = True,
                 future_steps: int = 0):
        self.base = base
        self.subject_indices = list(subject_indices)
        self.augmentation = augmentation
        self.random_context = bool(random_context)
        self.random_cutoff = bool(random_cutoff)
        self.seed = int(seed)
        # >0 时额外返回未来 R 步真值（供 --enable_rollout_loss 的短程 rollout 监督）
        self.future_steps = int(future_steps)

        rng = np.random.RandomState(seed)
        table: List[Tuple[int, int]] = []
        for subj_pos in self.subject_indices:
            subj_id = base.subject_ids[subj_pos]
            legal = base.legal_context_lengths(subj_id)
            if not legal:
                continue
            budget = int(samples_per_subject) if samples_per_subject > 0 \
                else min(base.legal_task_count(subj_id),
                         self.AUTO_SAMPLES_PER_SUBJECT_CAP)
            budget = max(1, budget)
            for _ in range(budget):
                K = int(legal[rng.randint(len(legal))]) if self.random_context \
                    else int(min(legal[-1], base.context_max))
                table.append((subj_pos, K))
        if not table:
            raise ValueError(
                "next_timepoint 训练视图为空：没有任何被试能构成合法 (K, t) 组合。")
        self.table = table
        self.labels = [K for _, K in table]
        print(
            f"[next_timepoint] 训练视图 | 被试数={len(self.subject_indices)} | "
            f"样本数={len(table)} | K 分布="
            f"{ {K: self.labels.count(K) for K in sorted(set(self.labels))} } | "
            f"random_context={self.random_context} | random_cutoff={self.random_cutoff}"
        )

    def __len__(self):
        return len(self.table)

    def lengths(self) -> List[int]:
        """每个 item 的 context 长度（供 LengthBucketBatchSampler 分桶）。"""
        return list(self.labels)

    def __getitem__(self, idx):
        subj_pos, K = self.table[idx]
        subj_id = self.base.subject_ids[subj_pos]
        lo, hi = self.base.legal_anchor_range(subj_id, K)
        t = int(np.random.randint(lo, hi + 1)) if self.random_cutoff else int(hi)
        sample = self.base.make_sample(subj_pos, K, t,
                                       future_steps=self.future_steps)
        if self.augmentation is not None:
            sample['x'] = self.augmentation(sample['x'])
        return sample


class NextPointEvalView(Dataset):
    """next_timepoint 评估视图：固定 protocol、无随机性。

    对每个被试固定 ``context 长度 = eval_context_length``，在合法预测区间内
    **等间隔确定性**选取至多 ``anchors_per_subject`` 个 anchor（0 = 枚举全部），
    另外标记至多 ``rollout_tasks_per_subject`` 个 anchor 作为 free rollout /
    FC / 频谱评估点（限定在 ``[K-1, T-1-rollout_steps]`` 内，保证完整 R 步真值）。
    每次评估的 (subject, K, t) 完全一致，且与训练采样无关。
    """

    def __init__(self, base: NeuroTwinNextPointDataset, subject_indices: List[int],
                 context_length: int, anchors_per_subject: int = 16,
                 rollout_tasks_per_subject: int = 0, rollout_steps: int = 0):
        self.base = base
        self.subject_indices = list(subject_indices)
        self.context_length = int(context_length)
        self.anchors_per_subject = int(anchors_per_subject)
        self.rollout_tasks_per_subject = int(rollout_tasks_per_subject)
        self.rollout_steps = int(rollout_steps)

        tasks: List[Tuple[int, int, int]] = []       # (subj_pos, t, rollout_flag)
        per_subject: Dict[str, int] = defaultdict(int)
        n_rollout = 0
        for subj_pos in self.subject_indices:
            subj_id = base.subject_ids[subj_pos]
            lo, hi = base.legal_anchor_range(subj_id, self.context_length)
            if hi < lo:
                continue
            anchors = list(range(lo, hi + 1))
            if self.anchors_per_subject > 0 and len(anchors) > self.anchors_per_subject:
                idx = np.linspace(0, len(anchors) - 1,
                                  self.anchors_per_subject).round().astype(int)
                anchors = [anchors[i] for i in sorted(set(idx.tolist()))]
            roll_set = set()
            if self.rollout_tasks_per_subject > 0 and self.rollout_steps > 0:
                r_lo, r_hi = base.legal_anchor_range(
                    subj_id, self.context_length, max_offset=self.rollout_steps)
                if r_hi >= r_lo:
                    r_anchors = list(range(r_lo, r_hi + 1))
                    if len(r_anchors) > self.rollout_tasks_per_subject:
                        idx = np.linspace(0, len(r_anchors) - 1,
                                          self.rollout_tasks_per_subject
                                          ).round().astype(int)
                        r_anchors = [r_anchors[i] for i in sorted(set(idx.tolist()))]
                        roll_set = set(r_anchors)
            if not anchors:
                continue
            for t in anchors:
                tasks.append((subj_pos, t, 1 if t in roll_set else 0))
            # rollout 锚点若不在 next-state 网格上则补进任务表（本身也是合法预测点），
            # 保证每个被试的 rollout 任务数确定且可复现
            for t in sorted(roll_set - set(anchors)):
                tasks.append((subj_pos, t, 1))
            n_rollout += len(roll_set)
            per_subject[subj_id] = len(anchors) + len(roll_set - set(anchors))

        if not tasks:
            raise ValueError(
                f"next_timepoint 评估视图为空：没有任何被试能构成 K={self.context_length} 的评估任务。")
        self.tasks = tasks
        self.labels = [self.context_length] * len(tasks)
        print(
            f"[next_timepoint] 评估视图 | 被试数={len(per_subject)} | 任务数={len(tasks)} | "
            f"每被试 anchor 数 min/median/max="
            f"{min(per_subject.values())}/{int(np.median(list(per_subject.values())))}/"
            f"{max(per_subject.values())} | K={self.context_length} | "
            f"rollout 锚点数={n_rollout}（R={self.rollout_steps}）"
        )

    def __len__(self):
        return len(self.tasks)

    def lengths(self) -> List[int]:
        return list(self.labels)

    def __getitem__(self, idx):
        subj_pos, t, rollout_flag = self.tasks[idx]
        sample = self.base.make_sample(
            subj_pos, self.context_length, t,
            future_steps=(self.rollout_steps if rollout_flag else 0))
        if 'future' not in sample:
            # 非 rollout 锚点也返回同形张量（全零 + 掩码全 0），保证 batch collate 一致；
            # 这些行不会参与 rollout/FC/频谱指标（rollout_flag=0 会被显式筛掉）
            n_rois = int(sample['sc'].shape[0])
            steps = max(1, int(self.rollout_steps))
            sample['future'] = torch.zeros(n_rois, steps)
            sample['future_mask'] = torch.zeros(steps)
        sample['rollout_flag'] = torch.tensor(float(rollout_flag))
        return sample


class LengthBucketBatchSampler:
    """按 context 长度 K 分桶的 batch sampler。

    同一 batch 内所有样本的 K 相同，因此 ``x [B,F,1,K]`` 时间轴天然对齐，不需要
    在时间轴上做 padding（也就不会把 padding 污染引入 RevIN 统计与图卷积）。
    batch 内部与 batch 之间在每个 epoch 重新打乱（``set_epoch``）。
    """

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True,
                 seed: int = 2024, drop_last: bool = False):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._batches: List[List[int]] = self._build()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = self._build()

    def _build(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, L in enumerate(self.lengths):
            groups[int(L)].append(idx)
        batches: List[List[int]] = []
        for L in sorted(groups):
            ids = groups[L]
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                chunk = ids[i:i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class NeuroTwinDataLoader:
    """
    被试级划分的 NeuroTwin DataLoader 构建器（next_timepoint 任务）。

    恒为被试级划分：
    - 同一被试的连续 BOLD 序列只出现在 train/val/test 之一。
    - Finetune 模式按被试病理评分分层，而非按样本。
    - 默认划分比例 8:1:1。
    """
    def __init__(
        self,
        data_root,
        mode: str = 'pretrain',
        batch_size: int = 8,
        eval_batch_size: Optional[int] = None,
        pathology_input_dim: int = 1,
        pathology_fields: Optional[List[str]] = None,
        pathology_missing: str = 'drop',
        clinical_file: str = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx',
        total_windows: int = 9,
        seq_len: int = 30,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_bins: int = 5,
        cache_in_memory: Optional[bool] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        refresh_split_manifest: bool = False,
        # ---- Next-Timepoint 任务（task_mode='next_timepoint'）----
        task_mode: str = 'next_timepoint',
        bold_source: str = 'auto',
        random_context: bool = True,
        random_cutoff: bool = True,
        train_samples_per_subject: int = 0,
        subject_cache_size: int = 256,
        sampling_seed: int = 2024,
        context_min: int = 16,
        context_max: int = 64,
        context_lengths: Optional[List[int]] = None,
        forecast_offsets: Optional[List[int]] = None,
        eval_context_length: int = 64,
        eval_anchors_per_subject: int = 16,
        eval_rollout_tasks_per_subject: int = 2,
        eval_rollout_steps: int = 32,
        train_rollout_steps: int = 0,
    ):
        task_mode = str(task_mode)
        if task_mode != 'next_timepoint':
            raise ValueError(
                f"task_mode 仅支持 'next_timepoint'，收到 '{task_mode}'")
        self.task_mode = task_mode
        self.bold_source = bold_source
        self.random_context = bool(random_context)
        self.random_cutoff = bool(random_cutoff)
        self.train_samples_per_subject = int(train_samples_per_subject)
        self.subject_cache_size = int(subject_cache_size)
        self.sampling_seed = int(sampling_seed)
        self.context_min = int(context_min)
        self.context_max = int(context_max)
        self.context_lengths = ([int(k) for k in context_lengths]
                                if context_lengths else None)
        self.forecast_offsets = ([int(o) for o in forecast_offsets]
                                 if forecast_offsets else [1])
        self.eval_context_length = int(eval_context_length)
        self.eval_anchors_per_subject = int(eval_anchors_per_subject)
        self.eval_rollout_tasks_per_subject = int(eval_rollout_tasks_per_subject)
        self.eval_rollout_steps = int(eval_rollout_steps)
        self.train_rollout_steps = int(train_rollout_steps)
        self.seed = seed
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size * 4
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.loader_generator = torch.Generator().manual_seed(seed)
        self.refresh_split_manifest = refresh_split_manifest

        if cache_in_memory is None:
            cache_in_memory = (num_workers == 0)

        self.base_dataset = NeuroTwinNextPointDataset(
            data_root=data_root,
            mode=mode,
            context_min=self.context_min,
            context_max=self.context_max,
            context_lengths=self.context_lengths,
            offsets=self.forecast_offsets,
            pathology_input_dim=pathology_input_dim,
            pathology_fields=pathology_fields,
            pathology_missing=pathology_missing,
            clinical_file=clinical_file,
            total_windows=total_windows,
            seq_len=seq_len,
            bold_source=self.bold_source,
            subject_cache_size=self.subject_cache_size,
            cache_in_memory=cache_in_memory,
        )

        train_indices, val_indices, test_indices, train_subjects, val_subjects, test_subjects = self._split_indices_by_subject(
            mode=mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_bins=stratify_bins,
        )

        augmentation = BrainSignalAugmentation(
            noise_std=0.03,
            scale_std=0.05,
            channel_drop_prob=0.03,
            time_mask_prob=0.15,
            max_mask_ratio=0.10,
        )

        # next-timepoint：训练视图动态采样 (K, t)，评估视图固定 K 与 anchor
        self.train_dataset = NextPointTrainView(
            self.base_dataset, train_indices,
            samples_per_subject=self.train_samples_per_subject,
            seed=self.sampling_seed, augmentation=augmentation,
            random_context=self.random_context, random_cutoff=self.random_cutoff,
            future_steps=self.train_rollout_steps)
        self.val_dataset = NextPointEvalView(
            self.base_dataset, val_indices,
            context_length=self.eval_context_length,
            anchors_per_subject=self.eval_anchors_per_subject,
            rollout_tasks_per_subject=self.eval_rollout_tasks_per_subject,
            rollout_steps=self.eval_rollout_steps)
        self.test_dataset = NextPointEvalView(
            self.base_dataset, test_indices,
            context_length=self.eval_context_length,
            anchors_per_subject=self.eval_anchors_per_subject,
            rollout_tasks_per_subject=self.eval_rollout_tasks_per_subject,
            rollout_steps=self.eval_rollout_steps)

        # 持久化被试划分结果：供归一化统计量拟合、分析脚本与实验记录复用
        self.train_subjects: List[str] = sorted(train_subjects)
        self.val_subjects: List[str] = sorted(val_subjects)
        self.test_subjects: List[str] = sorted(test_subjects)

        overlap_subjects = sorted(set(train_subjects) & set(val_subjects))
        if overlap_subjects:
            raise RuntimeError(
                f"subject-level split 失败：发现 {len(overlap_subjects)} 个被试同时出现在 train 和 val 中，例如 {overlap_subjects[:5]}"
            )
        overlap_subjects = sorted(set(train_subjects) & set(test_subjects))
        if overlap_subjects:
            raise RuntimeError(
                f"subject-level split 失败：发现 {len(overlap_subjects)} 个被试同时出现在 train 和 test 中，例如 {overlap_subjects[:5]}"
            )
        overlap_subjects = sorted(set(val_subjects) & set(test_subjects))
        if overlap_subjects:
            raise RuntimeError(
                f"subject-level split 失败：发现 {len(overlap_subjects)} 个被试同时出现在 val 和 test 中，例如 {overlap_subjects[:5]}"
            )

        print(
            "Train/Val/Test split 完成 | "
            f"train_subjects={len(train_subjects)} | val_subjects={len(val_subjects)} | test_subjects={len(test_subjects)} | "
            f"train_samples={len(self.train_dataset)} | val_samples={len(self.val_dataset)} | test_samples={len(self.test_dataset)}"
        )

        if mode == 'finetune' and len(train_subjects) > 0 and len(val_subjects) > 0:
            train_scores = [self.base_dataset.get_pathology_scalar_by_subject(sid) for sid in train_subjects]
            val_scores = [self.base_dataset.get_pathology_scalar_by_subject(sid) for sid in val_subjects]
            test_scores = [self.base_dataset.get_pathology_scalar_by_subject(sid) for sid in test_subjects]
            train_scores = [s for s in train_scores if s is not None]
            val_scores = [s for s in val_scores if s is not None]
            test_scores = [s for s in test_scores if s is not None]
            if train_scores and val_scores:
                print(
                    "HAMD 分布 | "
                    f"train mean={np.mean(train_scores):.3f}, std={np.std(train_scores):.3f} | "
                    f"val mean={np.mean(val_scores):.3f}, std={np.std(val_scores):.3f} | "
                    f"test mean={np.mean(test_scores):.3f}, std={np.std(test_scores):.3f}"
                )

    def _expand_subjects_to_sample_indices(self, subject_ids: List[str]) -> List[int]:
        indices: List[int] = []
        for sid in subject_ids:
            indices.extend(self.base_dataset.get_subject_indices(sid))
        return indices

    @staticmethod
    def _allocate_val_counts(group_sizes: Dict[int, int], target_total: int, val_ratio: float) -> Dict[int, int]:
        allocations: Dict[int, int] = {}
        fractional_parts: List[Tuple[float, int]] = []

        for group_id, size in group_sizes.items():
            raw = size * val_ratio
            base = int(np.floor(raw))
            base = min(max(base, 0), size)
            allocations[group_id] = base
            fractional_parts.append((raw - base, group_id))

        current_total = sum(allocations.values())
        remaining = target_total - current_total

        if remaining > 0:
            for _, group_id in sorted(fractional_parts, key=lambda x: (-x[0], x[1])):
                if remaining <= 0:
                    break
                if allocations[group_id] < group_sizes[group_id]:
                    allocations[group_id] += 1
                    remaining -= 1

        elif remaining < 0:
            for _, group_id in sorted(fractional_parts, key=lambda x: (x[0], x[1])):
                if remaining >= 0:
                    break
                if allocations[group_id] > 0:
                    allocations[group_id] -= 1
                    remaining += 1

        return allocations

    def _split_subject_ids(self, mode: str, val_ratio: float, test_ratio: float, stratify_bins: int) -> Tuple[List[str], List[str], List[str]]:
        """将受试者划分为训练集、验证集和测试集（8:1:1）。

        版本化固定：切分结果落盘到 ``<data_root>/subject_split_<mode>.json``
        （含 seed/比例/分箱等配置指纹）。同配置的后续运行直接复用名单，
        防止代码演化导致的切分漂移（跨实验 test 集一致性）；配置变化或
        被试集合变化时默认报错，需显式 ``refresh_split_manifest=True`` 重新生成。
        """
        manifest_path = self._split_manifest_path(mode)
        meta = {'mode': mode, 'seed': int(self.seed), 'val_ratio': float(val_ratio),
                'test_ratio': float(test_ratio), 'stratify_bins': int(stratify_bins),
                'n_subjects': len(self.base_dataset.get_subject_ids())}
        if manifest_path is not None and manifest_path.exists() \
                and not self.refresh_split_manifest:
            try:
                old = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(
                    f'切分 manifest 损坏（{manifest_path}）：{e}。'
                    '删除该文件或传 refresh_split_manifest=True 重新生成。') from e
            mismatch = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
            if mismatch:
                raise RuntimeError(
                    f'切分 manifest 配置与当前运行不一致（{manifest_path}）：{mismatch}。'
                    '确认数据或配置变更后传 refresh_split_manifest=True 重新生成；'
                    '若非有意变更，请回退当前运行的 seed/比例设置以复用固定切分。')
            cur = set(self.base_dataset.get_subject_ids())
            train, val, test = old.get('train'), old.get('val'), old.get('test')
            if train and val and test and (set(train) | set(val) | set(test)) == cur:
                print(f'[split] 复用版本化切分 manifest: {manifest_path}')
                return sorted(train), sorted(val), sorted(test)
            raise RuntimeError(
                f'切分 manifest 与当前被试集合不一致（{manifest_path}），'
                '数据目录可能已变化。删除该文件或传 refresh_split_manifest=True 重新生成。')

        train, val, test = self._split_subject_ids_impl(mode, val_ratio, test_ratio,
                                                        stratify_bins)
        if manifest_path is not None:
            manifest_path.write_text(
                json.dumps({**meta, 'train': train, 'val': val, 'test': test},
                           ensure_ascii=False, indent=2),
                encoding='utf-8')
            print(f'[split] 版本化切分已写入: {manifest_path} '
                  f'(train/val/test = {len(train)}/{len(val)}/{len(test)})')
        return train, val, test

    def _split_manifest_path(self, mode: str):
        root = getattr(self.base_dataset, 'data_root', None)
        if not root:
            return None
        return Path(str(root)) / f'subject_split_{mode}.json'

    def _split_subject_ids_impl(self, mode: str, val_ratio: float, test_ratio: float,
                                stratify_bins: int) -> Tuple[List[str], List[str], List[str]]:
        subject_ids = self.base_dataset.get_subject_ids()
        rng = random.Random(self.seed)

        total_subjects = len(subject_ids)
        if total_subjects == 0:
            raise ValueError("数据集为空，请检查 data_root 与文件组织结构。")
        if total_subjects < 3:
            raise ValueError("subject-level 三分割至少需要 3 个有效被试。")

        shuffled_subjects = list(subject_ids)
        rng.shuffle(shuffled_subjects)

        # 计算各集合的目标数量
        target_val_subjects = max(1, min(total_subjects - 2, int(round(total_subjects * val_ratio))))
        target_test_subjects = max(1, min(total_subjects - target_val_subjects - 1, int(round(total_subjects * test_ratio))))

        if mode != 'finetune':
            # 非finetune模式：直接按比例划分
            test_subjects = shuffled_subjects[:target_test_subjects]
            val_subjects = shuffled_subjects[target_test_subjects:target_test_subjects + target_val_subjects]
            train_subjects = shuffled_subjects[target_test_subjects + target_val_subjects:]
            return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

        scores = [self.base_dataset.get_pathology_scalar_by_subject(sid) for sid in subject_ids]
        if any(score is None for score in scores):
            raise ValueError("finetune 模式下存在缺失 pathology_score 的被试，无法做 subject-level stratified split。")

        score_values = [float(s) for s in scores if s is not None]
        unique_scores = len(set(score_values))
        if unique_scores <= 1:
            test_subjects = shuffled_subjects[:target_test_subjects]
            val_subjects = shuffled_subjects[target_test_subjects:target_test_subjects + target_val_subjects]
            train_subjects = shuffled_subjects[target_test_subjects + target_val_subjects:]
            return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

        # 分层划分
        n_bins = max(2, min(stratify_bins, unique_scores, total_subjects))
        score_series = pd.Series(score_values)
        bin_ids = pd.qcut(score_series, q=n_bins, labels=False, duplicates='drop')
        bin_list = [int(b) for b in bin_ids.tolist()]

        bin_to_subjects: Dict[int, List[str]] = defaultdict(list)
        for sid, bid in zip(subject_ids, bin_list):
            bin_to_subjects[bid].append(sid)

        for bid in bin_to_subjects:
            rng.shuffle(bin_to_subjects[bid])

        group_sizes = {bid: len(sids) for bid, sids in bin_to_subjects.items()}

        # 先分配test集
        test_allocations = self._allocate_val_counts(group_sizes, target_test_subjects, test_ratio)
        # 再分配val集（从剩余中分配）
        remaining_sizes = {bid: group_sizes[bid] - test_allocations.get(bid, 0) for bid in group_sizes}
        val_allocations = self._allocate_val_counts(remaining_sizes, target_val_subjects, val_ratio / (1 - test_ratio))

        train_subjects: List[str] = []
        val_subjects: List[str] = []
        test_subjects: List[str] = []

        for bid in sorted(bin_to_subjects.keys()):
            subjects = bin_to_subjects[bid]
            n_test = min(test_allocations.get(bid, 0), len(subjects))
            n_val = min(val_allocations.get(bid, 0), len(subjects) - n_test)

            test_subjects.extend(subjects[:n_test])
            val_subjects.extend(subjects[n_test:n_test + n_val])
            train_subjects.extend(subjects[n_test + n_val:])

        # 确保每个集合至少有一个样本
        if len(test_subjects) == 0:
            test_subjects = [train_subjects.pop()]
        if len(val_subjects) == 0:
            if len(train_subjects) > 0:
                val_subjects = [train_subjects.pop()]
            else:
                val_subjects = [test_subjects.pop()]
        if len(train_subjects) == 0:
            train_subjects = [val_subjects.pop()]

        rng.shuffle(train_subjects)
        rng.shuffle(val_subjects)
        rng.shuffle(test_subjects)
        return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

    def _split_indices_by_subject(self, mode: str, val_ratio: float, test_ratio: float, stratify_bins: int):
        train_subjects, val_subjects, test_subjects = self._split_subject_ids(
            mode=mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_bins=stratify_bins,
        )
        train_indices = self._expand_subjects_to_sample_indices(train_subjects)
        val_indices = self._expand_subjects_to_sample_indices(val_subjects)
        test_indices = self._expand_subjects_to_sample_indices(test_subjects)
        return train_indices, val_indices, test_indices, train_subjects, val_subjects, test_subjects

    def _worker_init_fn(self, worker_id: int):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _make_bucketed_loader(self, dataset, batch_size: int, shuffle: bool):
        """按 context 长度 K 分桶的 batch_sampler 取代 shuffle（同 batch 内 K 一致）。"""
        if dataset.lengths() is None:
            raise ValueError("视图缺少 lengths()，无法构建分桶 sampler。")
        sampler = LengthBucketBatchSampler(
            dataset.lengths(), batch_size=batch_size, shuffle=shuffle,
            seed=self.sampling_seed)
        kwargs = dict(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            generator=self.loader_generator,
            persistent_workers=self.persistent_workers,
        )
        if self.prefetch_factor is not None:
            kwargs['prefetch_factor'] = self.prefetch_factor
        return DataLoader(**kwargs)

    def get_train(self):
        return self._make_bucketed_loader(self.train_dataset, self.batch_size, shuffle=True)

    def get_val(self):
        # 评估无反传，显存压力小，可用更大 batch 减少调度与 collate 次数
        return self._make_bucketed_loader(self.val_dataset, self.eval_batch_size,
                                          shuffle=False)

    def get_test(self):
        return self._make_bucketed_loader(self.test_dataset, self.eval_batch_size,
                                          shuffle=False)

    # ------------------------------------------------------------------
    # 只读访问器：被试划分与病理条件（供归一化统计量拟合 / 分析脚本使用）
    # ------------------------------------------------------------------
    def get_train_subjects(self) -> List[str]:
        return list(self.train_subjects)

    def get_val_subjects(self) -> List[str]:
        return list(self.val_subjects)

    def get_test_subjects(self) -> List[str]:
        return list(self.test_subjects)

    def get_train_pathology_scores(self) -> Optional[np.ndarray]:
        """训练集被试的病理条件矩阵 [N_train, D]；非 finetune 模式返回 None。

        仅包含 train subjects，确保归一化统计量不泄露 val/test 信息。
        """
        if self.base_dataset.mode != 'finetune':
            return None
        rows = [
            vec for vec in (
                self.base_dataset.get_pathology_vector_by_subject(sid)
                for sid in self.train_subjects
            )
            if vec is not None
        ]
        if not rows:
            return None
        return np.stack(rows, axis=0).astype(np.float32)
