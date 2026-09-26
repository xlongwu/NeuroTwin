# coding=utf-8
import copy
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


class NeuroTwinDataset(Dataset):
    """
    Unified NeuroTwin dataset (subject-aware).

    pretrain:
        返回 HC 的 x, y, sc
    finetune:
        返回 MDD 的 x, y, sc, pathology_score

    修复点：
    1. 显式维护 subj_id -> sample_indices 映射，供 DataLoader 做被试级切分。
    2. build_index 日志同时报告 subject 数与 sample 数，避免把 sample 数误写成“有效被试数”。
    3. 保持 __getitem__ 接口不变，兼容现有 main.py / model / trainer。
    """
    def __init__(
        self,
        data_root,
        mode: str = 'pretrain',
        in_window: int = 6,
        pred_window: int = 3,
        pathology_input_dim: int = 1,
        pathology_fields: Optional[List[str]] = None,
        pathology_missing: str = 'drop',
        clinical_file: str = 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx',
        total_windows: int = 9,
        seq_len: int = 30,
        cache_in_memory: bool = False,
        # 实测（R1/R2 对照）：page cache 下 loadmat 重复读并非瓶颈，被试级缓存
        # 无稳态收益且增加 worker 内存（×num_workers）与启动成本，故默认关闭
        cache_subjects: bool = False,
        # 可变截断：允许未来窗不足 pred_window 的样本入训，提升每被试样本数
        # （pred_window=3 且 total_windows=9 / in_window=6 时固定截断仅 1 样本/被试）。
        # 不足部分在 y 中以零占位，并由 pred_mask 标记为无效窗口，损失端按掩码加权。
        variable_cutoff: bool = False,
    ):
        self.data_root = Path(data_root)
        self.mode = mode
        self.in_window = in_window
        self.pred_window = pred_window
        self.pathology_input_dim = pathology_input_dim
        # 病理条件字段（默认仅 HAMD 总分；条目级缺失率高，向量路径需显式开启）
        self.pathology_fields = list(pathology_fields) if pathology_fields else ['HAMD']
        if pathology_missing not in ('drop', 'zero', 'mean'):
            raise ValueError(
                f"pathology_missing 仅支持 drop/zero/mean，收到 '{pathology_missing}'")
        self.pathology_missing = pathology_missing
        self.total_windows = total_windows
        self.seq_len = seq_len
        self.cache_in_memory = cache_in_memory
        # 被试级缓存：同一被试的 Mask/9 窗只加载与归一化一次，
        # 消除滑窗样本间约 2/3~90% 的重复读盘（多 worker 下每进程各持一份）
        self.cache_subjects = cache_subjects
        self.variable_cutoff = bool(variable_cutoff)
        self._subject_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._mat_cache: Dict[str, np.ndarray] = {}

        # 可变截断下至少保留 1 个未来窗即可成样本，因此约束放宽到 in_window + 1
        min_pred = 1 if self.variable_cutoff else self.pred_window
        if self.in_window + min_pred > self.total_windows:
            raise ValueError(
                f"in_window + pred_window must be <= {self.total_windows}, got {self.in_window} + {self.pred_window}"
            )

        if self.mode == 'finetune' and self.pathology_input_dim != len(self.pathology_fields):
            raise ValueError(
                f"pathology_input_dim={self.pathology_input_dim} 与 pathology_fields="
                f"{self.pathology_fields}（宽度 {len(self.pathology_fields)}）不一致。")

        self.clinical_dict: Dict[str, np.ndarray] = {}
        if self.mode == 'finetune':
            clinical_path = self.data_root / clinical_file
            if not clinical_path.exists():
                raise FileNotFoundError(f"未找到临床评分文件：{clinical_path}")

            df = pd.read_excel(clinical_path)
            required_cols = {'ID', *self.pathology_fields}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"clinical file must contain columns {required_cols}, got {set(df.columns)}")

            df = df[['ID', *self.pathology_fields]].copy()
            df['ID'] = df['ID'].astype(str).str.strip()
            value_cols = self.pathology_fields
            if self.pathology_missing == 'drop':
                df = df.dropna(subset=value_cols)
            elif self.pathology_missing == 'mean':
                df[value_cols] = df[value_cols].fillna(df[value_cols].mean(numeric_only=True))
                df = df.dropna(subset=value_cols)
            else:  # zero
                df[value_cols] = df[value_cols].fillna(0.0)
            values = df[value_cols].to_numpy(dtype=np.float32)
            self.clinical_dict = {
                k: v for k, v in zip(df['ID'].tolist(), list(values))
            }
            print(
                f"成功加载临床评分表，共包含 {len(self.clinical_dict)} 个被试的 "
                f"{'/'.join(value_cols)} 数据（缺失策略: {self.pathology_missing}）。"
            )

        group_folder = 'HC' if mode == 'pretrain' else 'MDD'
        self.window_dir = self.data_root / 'ROISignals_window' / group_folder
        self.sc_dir = self.data_root / 'Mask' / group_folder
        # 聚合 npz 缓存目录（由 scripts/convert_mat_to_npz.py 生成）：
        # 每被试一个 npz（sc + windows [W,F,S]），存在则优先读取，消除小 mat 文件 I/O
        self.npz_dir = self.data_root / 'npz_cache' / group_folder
        self.npz_dir = self.npz_dir if self.npz_dir.is_dir() else None

        self.samples: List[Dict] = []
        self.subject_ids: List[str] = []
        self.subject_to_indices: Dict[str, List[int]] = {}
        self.subject_to_score: Dict[str, float] = {}
        self._build_index()

    def _build_index(self):
        sc_files = sorted(self.sc_dir.glob('Mask_*.mat'), key=lambda p: p.stem)
        subject_to_indices: Dict[str, List[int]] = defaultdict(list)
        valid_subjects: List[str] = []

        min_pred = 1 if self.variable_cutoff else self.pred_window
        max_start = self.total_windows - self.in_window - min_pred + 1
        if max_start <= 0:
            raise ValueError(
                f"无法切出任何样本：total_windows={self.total_windows}, in_window={self.in_window}, pred_window={self.pred_window}"
            )

        for sc_file in sc_files:
            subj_id = str(sc_file.stem.replace('Mask_', '')).strip()

            if self.mode == 'finetune' and subj_id not in self.clinical_dict:
                print(f"跳过被试 {subj_id}（临床表中无 HAMD 评分）")
                continue

            window_files = [
                self.window_dir / f"ROISignals_{subj_id}-{i}.mat"
                for i in range(1, self.total_windows + 1)
            ]
            if not all(f.exists() for f in window_files):
                print(f"跳过被试 {subj_id}（滑动窗口文件不足 {self.total_windows} 个）")
                continue

            valid_subjects.append(subj_id)
            if self.mode == 'finetune':
                # 分层划分使用首个字段（默认 HAMD 总分）作为标量分层依据
                self.subject_to_score[subj_id] = float(self.clinical_dict[subj_id][0])

            for start_idx in range(max_start):
                sample = {
                    'subj_id': subj_id,
                    'sc_path': sc_file,
                    'window_paths': window_files,
                    'start_idx': start_idx,
                    # 该起点实际可用的未来窗数（<= pred_window）；固定截断时恒为 pred_window
                    'pred_len': min(self.pred_window,
                                    self.total_windows - start_idx - self.in_window),
                }
                if self.mode == 'finetune':
                    sample['pathology_score'] = np.asarray(
                        self.clinical_dict[subj_id], dtype=np.float32)
                sample_idx = len(self.samples)
                self.samples.append(sample)
                subject_to_indices[subj_id].append(sample_idx)

        self.subject_ids = sorted(valid_subjects)
        self.subject_to_indices = {sid: subject_to_indices[sid] for sid in self.subject_ids}

        print(
            f"构建完毕 | 模式: {self.mode} | 有效被试数: {len(self.subject_ids)} | 样本数: {len(self.samples)}"
            f" | variable_cutoff: {self.variable_cutoff}"
        )

    def get_subject_ids(self) -> List[str]:
        return list(self.subject_ids)

    def get_subject_indices(self, subj_id: str) -> List[int]:
        return list(self.subject_to_indices.get(subj_id, []))

    def get_pathology_scalar_by_index(self, idx: int) -> Optional[float]:
        """返回首个病理字段的标量值（默认 HAMD 总分），仅供分层/统计使用。"""
        if self.mode != 'finetune':
            return None
        return float(np.asarray(self.samples[idx]['pathology_score']).reshape(-1)[0])

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

    def __len__(self):
        return len(self.samples)

    def _extract_mat(self, mat_dict, preferred_keys=None):
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

    def _read_mat_array(self, path: Path, preferred_keys=None) -> np.ndarray:
        key = str(path)
        if self.cache_in_memory and key in self._mat_cache:
            return self._mat_cache[key].copy()

        mat = sio.loadmat(path)
        arr = self._extract_mat(mat, preferred_keys=preferred_keys)
        arr = np.asarray(arr)

        if self.cache_in_memory:
            self._mat_cache[key] = arr.copy()
        return arr

    @staticmethod
    def _normalize_sc(sc_matrix: np.ndarray) -> np.ndarray:
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

    def _load_subject_from_npz(self, npz_path: Path) -> Dict[str, np.ndarray]:
        """从聚合 npz 读取单被试数据（sc 为原始 Mask，windows 已统一为 [W, F, S] float32）。"""
        with np.load(npz_path) as z:
            sc_raw = np.asarray(z['sc'])
            windows = np.asarray(z['windows'])
        sc_matrix = self._normalize_sc(sc_raw)
        if windows.ndim != 3 or windows.shape[0] != self.total_windows:
            raise ValueError(f"npz windows 形状异常 {windows.shape}（期望 [W={self.total_windows}, F, S]）: {npz_path.name}")
        if windows.shape[1] != sc_matrix.shape[0] or windows.shape[2] != self.seq_len:
            raise ValueError(f"npz 维度与 SC/seq_len 不匹配: {npz_path.name} windows={windows.shape}")
        return {'sc': sc_matrix, 'windows': windows.astype(np.float32, copy=False)}

    def _load_subject(self, subj_id: str, sc_path: Path, window_paths: List[Path]) -> Dict[str, np.ndarray]:
        """加载并预处理单个被试的全部数据：归一化 SC + 堆叠 9 窗 [W_total, F, S]。"""
        if self.npz_dir is not None:
            npz_path = self.npz_dir / f"{subj_id}.npz"
            if npz_path.exists():
                return self._load_subject_from_npz(npz_path)

        sc_matrix = self._read_mat_array(
            sc_path,
            preferred_keys=['SC', 'sc_matrix', 'mask', 'Mask']
        )
        if sc_matrix.ndim != 2 or sc_matrix.shape[0] != sc_matrix.shape[1]:
            raise ValueError(f"SC/mask must be square [F, F], got {sc_matrix.shape} for {subj_id}")
        sc_matrix = self._normalize_sc(sc_matrix)

        bold_windows = []
        for w_path in window_paths:
            mat_data = self._read_mat_array(w_path, preferred_keys=['ROISignals', 'data', 'bold'])
            if mat_data.ndim != 2:
                raise ValueError(f"BOLD window must be 2D, got {mat_data.shape} in {w_path.name}")

            if mat_data.shape[0] == self.seq_len:
                mat_data = mat_data.T
            elif mat_data.shape[1] == self.seq_len:
                pass
            else:
                raise ValueError(f"Cannot infer BOLD layout from shape {mat_data.shape} in {w_path.name}")

            bold_windows.append(np.asarray(mat_data, dtype=np.float32))

        bold_windows = np.stack(bold_windows, axis=0)  # [W_total, F, S]
        if bold_windows.shape[1] != sc_matrix.shape[0]:
            raise ValueError(
                f"Node count mismatch for {subj_id}: BOLD has {bold_windows.shape[1]} nodes, SC has {sc_matrix.shape[0]}"
            )
        return {'sc': sc_matrix, 'windows': bold_windows}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start_idx = sample['start_idx']

        entry = None
        if self.cache_subjects:
            entry = self._subject_cache.get(sample['subj_id'])
        if entry is None:
            entry = self._load_subject(sample['subj_id'], sample['sc_path'], sample['window_paths'])
            if self.cache_subjects:
                self._subject_cache[sample['subj_id']] = entry

        sc_matrix = entry['sc']
        bold_windows = entry['windows']

        x_bold = bold_windows[start_idx: start_idx + self.in_window].transpose(1, 0, 2)  # [F, in_W, S]
        # 可变截断下未来窗可能不足 pred_window：不足部分补零占位，
        # 并由 pred_mask 标记无效窗口，损失端按掩码加权（零值不会被监督）
        pred_len = int(sample.get('pred_len', self.pred_window))
        pred_len = max(1, min(pred_len, self.pred_window))
        y_bold = bold_windows[
            start_idx + self.in_window: start_idx + self.in_window + pred_len
        ].transpose(1, 0, 2)  # [F, pred_len, S]
        if pred_len < self.pred_window:
            pad = np.zeros(
                (y_bold.shape[0], self.pred_window - pred_len, y_bold.shape[2]),
                dtype=y_bold.dtype)
            y_bold = np.concatenate([y_bold, pad], axis=1)

        out = {
            'x': torch.tensor(x_bold, dtype=torch.float32),
            'y': torch.tensor(y_bold, dtype=torch.float32),
            'sc': torch.tensor(sc_matrix, dtype=torch.float32),
            'subj_id': sample['subj_id'],
        }
        if self.variable_cutoff:
            mask = np.zeros(self.pred_window, dtype=np.float32)
            mask[:pred_len] = 1.0
            out['pred_mask'] = torch.from_numpy(mask)
        if self.mode == 'finetune':
            out['pathology_score'] = torch.tensor(
                np.asarray(sample['pathology_score'], dtype=np.float32))
        return out


class DatasetView(Dataset):
    """
    基于同一个底层 dataset 的训练/验证视图。
    训练视图可单独加增强，避免重复构造两份完整数据集。
    """
    def __init__(self, base_dataset: NeuroTwinDataset, indices, augmentation: Optional[BrainSignalAugmentation] = None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.base_dataset[self.indices[idx]]
        if self.augmentation is None:
            return sample

        sample = {
            k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in sample.items()
        }
        sample['x'] = self.augmentation(sample['x'])
        return sample


class NeuroTwinDataLoader:
    """
    Subject-level split DataLoader builder for NeuroTwin.

    Uses subject-level partitioning by default:
    - All start_idx samples of the same subj_id appear in only one of train/val/test.
    - Finetune mode stratification is by subject pathology_score, not by sample.
    - Default split ratio is 8:1:1 for train, val, test.
    """
    def __init__(
        self,
        data_root,
        mode: str = 'pretrain',
        batch_size: int = 8,
        eval_batch_size: Optional[int] = None,
        in_window: int = 6,
        pred_window: int = 3,
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
        cache_subjects: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        split_by_subject: bool = True,
        variable_cutoff: bool = False,
        refresh_split_manifest: bool = False,
    ):
        self.seed = seed
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size * 4
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.loader_generator = torch.Generator().manual_seed(seed)
        self.split_by_subject = split_by_subject
        self.refresh_split_manifest = refresh_split_manifest

        if cache_in_memory is None:
            cache_in_memory = (num_workers == 0)

        self.base_dataset = NeuroTwinDataset(
            data_root=data_root,
            mode=mode,
            in_window=in_window,
            pred_window=pred_window,
            pathology_input_dim=pathology_input_dim,
            pathology_fields=pathology_fields,
            pathology_missing=pathology_missing,
            clinical_file=clinical_file,
            total_windows=total_windows,
            seq_len=seq_len,
            cache_in_memory=cache_in_memory,
            cache_subjects=cache_subjects,
            variable_cutoff=variable_cutoff,
        )

        if self.split_by_subject:
            train_indices, val_indices, test_indices, train_subjects, val_subjects, test_subjects = self._split_indices_by_subject(
                mode=mode,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify_bins=stratify_bins,
            )
        else:
            train_indices, val_indices, test_indices = self._split_indices_by_sample(
                mode=mode,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify_bins=stratify_bins,
            )
            train_subjects = sorted({self.base_dataset.samples[i]['subj_id'] for i in train_indices})
            val_subjects = sorted({self.base_dataset.samples[i]['subj_id'] for i in val_indices})
            test_subjects = sorted({self.base_dataset.samples[i]['subj_id'] for i in test_indices})

        augmentation = BrainSignalAugmentation(
            noise_std=0.03,
            scale_std=0.05,
            channel_drop_prob=0.03,
            time_mask_prob=0.15,
            max_mask_ratio=0.10,
        )

        self.train_dataset = DatasetView(self.base_dataset, train_indices, augmentation=augmentation)
        self.val_dataset = DatasetView(self.base_dataset, val_indices, augmentation=None)
        self.test_dataset = DatasetView(self.base_dataset, test_indices, augmentation=None)

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

    def _split_indices_by_sample(self, mode: str, val_ratio: float, test_ratio: float, stratify_bins: int):
        """
        保留旧逻辑作为可选回退，不推荐正式实验使用。
        支持三分割：训练集、验证集、测试集
        """
        total_size = len(self.base_dataset)
        indices = list(range(total_size))
        rng = random.Random(self.seed)

        if total_size == 0:
            raise ValueError("数据集为空，请检查 data_root 与文件组织结构。")
        if total_size < 3:
            raise ValueError("sample-level 三分割至少需要 3 个样本。")

        if mode != 'finetune':
            rng.shuffle(indices)
            test_size = max(1, int(round(total_size * test_ratio)))
            val_size = max(1, int(round(total_size * val_ratio)))
            test_indices = indices[:test_size]
            val_indices = indices[test_size:test_size + val_size]
            train_indices = indices[test_size + val_size:]
            return train_indices, val_indices, test_indices

        scores = [self.base_dataset.get_pathology_scalar_by_index(i) for i in indices]
        unique_scores = len(set(scores))
        if unique_scores <= 1:
            rng.shuffle(indices)
            test_size = max(1, int(round(total_size * test_ratio)))
            val_size = max(1, int(round(total_size * val_ratio)))
            test_indices = indices[:test_size]
            val_indices = indices[test_size:test_size + val_size]
            train_indices = indices[test_size + val_size:]
            return train_indices, val_indices, test_indices

        n_bins = max(2, min(stratify_bins, unique_scores, total_size))
        score_series = pd.Series(scores)
        bin_ids = pd.qcut(score_series, q=n_bins, labels=False, duplicates='drop')

        train_indices, val_indices, test_indices = [], [], []
        for bin_id in sorted(set(bin_ids.tolist())):
            bin_group = [idx for idx, bid in zip(indices, bin_ids.tolist()) if bid == bin_id]
            rng.shuffle(bin_group)
            test_size = max(1, int(round(len(bin_group) * test_ratio))) if len(bin_group) > 2 else 0
            val_size = max(1, int(round(len(bin_group) * val_ratio))) if len(bin_group) - test_size > 1 else 0
            test_indices.extend(bin_group[:test_size])
            val_indices.extend(bin_group[test_size:test_size + val_size])
            train_indices.extend(bin_group[test_size + val_size:])

        # 确保每个集合至少有一个样本
        if len(test_indices) == 0:
            if len(train_indices) > 0:
                test_indices = [train_indices.pop()]
            else:
                test_indices = [val_indices.pop()]
        if len(val_indices) == 0:
            if len(train_indices) > 0:
                val_indices = [train_indices.pop()]
            else:
                val_indices = [test_indices.pop()]
        if len(train_indices) == 0:
            if len(val_indices) > 0:
                train_indices = [val_indices.pop()]
            else:
                train_indices = [test_indices.pop()]

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
        return train_indices, val_indices, test_indices

    def _worker_init_fn(self, worker_id: int):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _make_loader(self, dataset: Dataset, shuffle: bool, batch_size: Optional[int] = None):
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle,
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
        return self._make_loader(self.train_dataset, shuffle=True)

    def get_val(self):
        # 评估无反传，显存压力小，可用更大 batch 减少调度与 collate 次数
        return self._make_loader(self.val_dataset, shuffle=False,
                                 batch_size=self.eval_batch_size)

    def get_test(self):
        return self._make_loader(self.test_dataset, shuffle=False,
                                 batch_size=self.eval_batch_size)

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
