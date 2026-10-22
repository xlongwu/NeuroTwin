# coding=utf-8
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from utils.augmentation import BrainSignalAugmentation
from utils.condition_schema import FieldSpec


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


# ══════════════════════════════════════════════════════════════════════════
#  条件注入表（v1 revised 20261008）：**只**读取 merged 文件
# ══════════════════════════════════════════════════════════════════════════

#: 阶段 → 条件 merged 文件（路径 = data_root + 文件名；禁止任何其他 xlsx）
CONDITION_FILE_BY_MODE = {
    'pretrain': 'Rest-meta-MDD-V1V2-Merged-HC.xlsx',
    'finetune': 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx',
}
#: 阶段 → 条件字段分组
CONDITION_GROUP_BY_MODE = {'pretrain': 'hc3', 'finetune': 'mdd8'}
#: 显式分层键（**不随条件 fields[0] 变化**）；MDD 保持历史 HAMD 口径
STRATIFY_KEY_BY_MODE = {'finetune': 'HAMD'}
#: 仅用于划分审计 / 结果分层的辅助列（绝不进入条件向量）
AUX_AUDIT_COLUMNS = ('Dataset', 'site')


@dataclass
class ConditionTable:
    """merged 条件表的一次性加载结果（值、有效性 mask、ID 对齐与审计）。

    Args:
        values: ``[N_merged, K]`` float32；缺失格为 ``nan``（不做任何填补）。
        mask: ``[N_merged, K]`` uint8，:math:`m_{ij}` = 单元格可解析为有限数值。
        ids: merged 行的被试 ID（只用于匹配）。
        row_of_id: ID → merged 行下标。
        audit: :class:`~utils.condition_schema.LoadAudit`（哨兵计数、缺失原因、
            实际解析到的来源列）。
        aux: 审计用辅助列（如 ``Dataset`` 的 V1/V2），不进条件向量。
        merged_sha256: merged 文件快照哈希（写入 checkpoint）。
        missing_row_subjects: 影像队列中在 merged 无整行记录的被试（全字段缺失）。
    """

    group: str
    file_name: str
    values: np.ndarray
    mask: np.ndarray
    ids: List[str]
    row_of_id: Dict[str, int]
    audit: object
    aux: Dict[str, np.ndarray]
    merged_sha256: str
    specs: Tuple[FieldSpec, ...]
    missing_row_subjects: List[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int(self.values.shape[1])

    def rows_for(self, subject_ids: Sequence[str]) -> np.ndarray:
        """被试 ID → merged 行下标；无整行记录者映射到 ``-1``（全缺失分支）。"""
        return np.asarray([self.row_of_id.get(str(s), -1) for s in subject_ids],
                          dtype=np.int64)

    def align(self, subject_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """按影像队列取 ``(values, mask)``；缺行被试得到全 ``nan`` + 全 0 mask。"""
        rows = self.rows_for(subject_ids)
        v = np.full((len(subject_ids), self.k), np.nan, dtype=np.float32)
        m = np.zeros((len(subject_ids), self.k), dtype=np.uint8)
        hit = rows >= 0
        if hit.any():
            v[hit] = self.values[rows[hit]]
            m[hit] = self.mask[rows[hit]]
        return v, m


def load_condition_table(data_root, file_name: str, group: str,
                         id_column: str = 'ID') -> ConditionTable:
    """加载条件 merged 表（唯一允许的条件数据来源），并读取审计辅助列。

    显式拒绝任何其他文件：本函数不接受目录外的路径，也不会因为目标缺失而回退
    到单独的 V1/V2 源表（方案 §2：禁止回读补齐/覆盖/恢复编码/创建 mask）。
    """
    from utils.condition_schema import (field_names, file_sha256,
                                        load_merged_conditions, schema_for)

    path = Path(data_root) / file_name
    if not path.exists():
        raise FileNotFoundError(
            f'条件数据只允许读取 merged 文件，但 {path} 不存在；'
            f'禁止改用单独的 V1/V2 源表（{file_name} 之外的文件不作为条件来源）。')
    specs = schema_for(group)
    values, mask, ids, audit = load_merged_conditions(path, specs, id_column=id_column)
    df = pd.read_excel(path)
    aux: Dict[str, np.ndarray] = {}
    wanted = [c for c in AUX_AUDIT_COLUMNS
              if str(c).strip().casefold() in
              {str(x).strip().casefold() for x in df.columns}]
    if wanted:
        norm = {str(x).strip().casefold(): x for x in df.columns}
        for col in wanted:
            aux[col] = df[norm[str(col).strip().casefold()]].astype(str).str.strip().values
    return ConditionTable(
        group=group, file_name=file_name, values=values, mask=mask, ids=ids,
        row_of_id={sid: i for i, sid in enumerate(ids)}, audit=audit, aux=aux,
        merged_sha256=file_sha256(path), specs=tuple(specs))


def attach_missing_rows(table: ConditionTable, subject_ids: Sequence[str]) -> ConditionTable:
    """记录影像队列中 merged 无整行记录的被试（保留样本，条件全缺失）。"""
    table.missing_row_subjects = [str(s) for s in subject_ids if str(s) not in table.row_of_id]
    return table


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
#      因此主干无需改写，只依赖 S 轴的
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

    BOLD_SOURCES = ('auto', 'normlize', 'windows', 'raw')

    #: graph_source：legacy_mask = 现有 Mask_<id>.mat 外部图；history_fc = 当前可见
    #: 历史 signed Pearson 绝对值功能耦合图（模型侧计算，loader 只给单位图占位）。
    GRAPH_SOURCES = ('legacy_mask', 'history_fc')

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
        # ---- 条件注入（v1 revised 20261008）：None = 关闭，走 legacy 数值通路 ----
        condition_group: Optional[str] = None,
        condition_file: Optional[str] = None,
        condition_control: str = 'none',
        total_windows: int = 9,
        seq_len: int = 30,
        bold_source: str = 'auto',
        subject_cache_size: int = 256,
        cache_in_memory: bool = False,
        # ---- graph_ode 路线（方案 §4）：旧调用完全不传时行为与历史一致 ----
        num_rois: int = 116,
        graph_source: str = 'legacy_mask',
        calibration_length: int = 0,          # 0 = 关闭（legacy）；raw 分支必须 >=1
        control_manifest: Optional[str] = None,
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

        # ---- graph_ode 路线参数（方案 §4.1–§4.3）----
        if graph_source not in self.GRAPH_SOURCES:
            raise ValueError(
                f"graph_source 仅支持 {self.GRAPH_SOURCES}，收到 '{graph_source}'")
        self.graph_source = str(graph_source)
        self.num_rois = int(num_rois)
        if self.num_rois < 1:
            raise ValueError(f"num_rois 必须 >= 1，收到 {self.num_rois}")
        self.calibration_length = int(calibration_length)
        if self.bold_source == 'raw' and self.calibration_length < 1:
            raise ValueError(
                "bold_source='raw' 必须显式传 calibration_length >= 1"
                "（Raw 标准化与 profile 使用同一前缀；新路线固定 32）")
        self.control_manifest = str(control_manifest) if control_manifest else None
        # 逐被试控制序列 {subj_id: u [T,F] float32}；manifest 未提供时保持 None
        self.controls_by_subject: Optional[Dict[str, np.ndarray]] = None

        self.pathology_input_dim = pathology_input_dim
        # 病理条件字段（默认仅 HAMD 总分；条目级缺失率高，向量路径需显式开启）
        self.pathology_fields = list(pathology_fields) if pathology_fields else ['HAMD']
        if pathology_missing not in ('drop', 'zero', 'mean'):
            raise ValueError(
                f"pathology_missing 仅支持 drop/zero/mean，收到 '{pathology_missing}'")
        self.pathology_missing = pathology_missing
        if self.mode == 'finetune' and self.pathology_input_dim != len(self.pathology_fields):
            raise ValueError(
                f"pathology_input_dim={pathology_input_dim} 与 pathology_fields="
                f"{self.pathology_fields}（宽度 {len(self.pathology_fields)}）不一致。")

        # ---- 条件注入通路 ----
        # 开启后：条件只来自 merged 文件，且**不因临床缺失或整行缺失删除被试**；
        # legacy 数值通路（clinical_dict / pathology_score）同时关闭，避免同一
        # 样本存在两套互相冲突的条件输入。
        self.condition_group = str(condition_group) if condition_group else None
        self.condition_enabled = self.condition_group is not None
        if self.condition_enabled and self.condition_group not in ('hc3', 'mdd8'):
            raise ValueError(
                f"condition_group 仅支持 hc3/mdd8，收到 '{self.condition_group}'")
        self.condition_file = (condition_file or CONDITION_FILE_BY_MODE.get(self.mode)
                               if self.condition_enabled else None)
        # 信息/容量对照（方案 §12）：none | static_new（新增组值与 mask 都固定）
        # | mask_only（新增组值固定但保留真实 mask，检验缺失模式的独立贡献）
        if condition_control not in ('none', 'static_new', 'mask_only'):
            raise ValueError(
                "condition_control 仅支持 none/static_new/mask_only，收到 "
                f"'{condition_control}'")
        self.condition_control = str(condition_control)
        if self.condition_enabled and self.condition_file is None:
            raise ValueError(
                f'mode={self.mode} 没有登记的条件 merged 文件，请显式传 --condition_file')

        self.clinical_dict: Dict[str, np.ndarray] = {}
        if self.mode == 'finetune' and not self.condition_enabled:
            self.clinical_dict = load_clinical_scores(
                self.data_root, clinical_file, self.pathology_fields,
                self.pathology_missing)

        self.condition_table: Optional[ConditionTable] = None
        self.cond_values: Optional[np.ndarray] = None      # [N_subj, K]（缺失=nan）
        self.cond_mask: Optional[np.ndarray] = None        # [N_subj, K] uint8
        self.cond_z: Optional[np.ndarray] = None           # 拟合后填充
        self.cond_code: Optional[np.ndarray] = None
        self.condition_stats = None

        group_folder = 'HC' if mode == 'pretrain' else 'MDD'
        self.window_dir = self.data_root / 'ROISignals_window' / group_folder
        self.sc_dir = self.data_root / 'Mask' / group_folder
        # 聚合 npz 缓存目录（由 utils/convert_mat_to_npz.py 生成）：
        # 每被试一个 npz（sc + windows [W,F,S]），存在则优先读取，消除小 mat 文件 I/O
        self.npz_dir = self.data_root / 'npz_cache' / group_folder
        self.npz_dir = self.npz_dir if self.npz_dir.is_dir() else None
        self.normlize_dir = self.data_root / 'Normlize' / group_folder
        # raw 分支目录（方案 §4.1）：Raw/HC|MDD/ROISignals_<id>.mat
        self.raw_dir = self.data_root / 'Raw' / group_folder

        self.subject_ids: List[str] = []
        self._subject_pos: Dict[str, int] = {}
        self.subject_series_len: Dict[str, int] = {}
        self.subject_n_tr: Dict[str, int] = {}
        self.subject_to_score: Dict[str, float] = {}
        #: 影像队列里在临床表中**无整行记录**的被试（保留，不删人）
        self.no_clinical_subjects: List[str] = []
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._mat_cache: Dict[str, np.ndarray] = {}
        self._source_logged = False
        self._build_index()

    def attach_condition_stats(self, stats) -> None:
        """附加 train-only 统计量并**一次性**编码整个队列（z / 类别索引）。

        统计量必须由 DataLoader 在冻结划分之后用 train 被试拟合；本方法只做变换，
        不再拟合，避免 val/test 影响统计量（验收 E）。
        """
        from utils.condition_schema import field_names
        table = self.condition_table
        if table is None:
            raise RuntimeError('条件表尚未加载，无法附加统计量。')
        names = field_names(table.specs)
        if names != [s.name for s in stats.specs]:
            raise ValueError(
                f'统计量字段顺序与条件 schema 不一致：{names} vs '
                f'{[s.name for s in stats.specs]}（顺序参与 layout_hash，不允许重排）')
        # 用 merged 的**队列对齐**矩阵（缺行被试为全缺失）做编码
        z, code = stats.encode(self.cond_values, self.cond_mask)
        self.condition_stats = stats
        self.cond_z, self.cond_code = z, code
        print(f'[condition] train-only 统计量已附加 | 连续字段 '
              + ', '.join(f'{s.name}: mu={stats.get(s.name).mu:.3f} '
                          f'scale={stats.get(s.name).scale:.3f} '
                          f'({stats.get(s.name).scale_source}, n={stats.get(s.name).n_valid})'
                          for s in table.specs if s.is_continuous))
        for s in table.specs:
            if not s.is_continuous:
                print(f'[condition] {s.name} 类别词表={list(stats.get(s.name).codes)} '
                      f'+ Unknown（n_valid={stats.get(s.name).n_valid}）')
        self._apply_condition_control()

    def _apply_condition_control(self) -> None:
        r"""新增组（MDD 五字段）的信息 / 容量对照变换（方案 §12）。

        - ``static_new``：新增五字段全部取**训练常量**（连续 :math:`z=0`，即训练
          中位数；类别 → 该字段已登记的第一个有效类别码，**不是** Unknown），并把
          新增组 mask **固定为全 1**。这会同时取消「新增值的个体差异」与「新增缺失
          模式的个体差异」，报告时必须如此标注。
        - ``mask_only``：值同样固定到常量，但**保留真实 mask**，用来检验收益是否
          只由缺失模式解释。分类保留 mask，不把整块改成 Unknown 而丢失观测状态。

        共享三字段（Sex/Age/education）在两种对照下都保持逐人真实值不变。
        """
        from utils.condition_schema import MDD_NEW_GROUP, group_indices
        if self.condition_control == 'none':
            return
        table = self.condition_table
        new_idx = group_indices(table.specs, MDD_NEW_GROUP)
        stats = self.condition_stats
        fixed_code, const_raw = {}, {}
        for j, spec in enumerate(table.specs):
            if j not in new_idx or spec.is_continuous:
                continue
            codes = stats.get(spec.name).codes
            if not codes:
                raise RuntimeError(
                    f'对照 {self.condition_control} 需要字段 {spec.name} 的合法常量，'
                    f'但其类别词表为空（训练集无有效观测）；无法构造「保留观测状态」'
                    f'的对照，请显式登记类别码后重试。')
            fixed_code[j] = 1                  # codes[0] → 内部索引 1（0 留给 Unknown）
            const_raw[j] = float(codes[0])
        for j in new_idx:
            spec = table.specs[j]
            if spec.is_continuous:
                self.cond_z[:, j] = 0.0        # z=0 ⇔ 训练填补中心（中位数）
                self.cond_values[:, j] = np.float32(stats.get(spec.name).mu)
            else:
                self.cond_code[:, j] = fixed_code[j]
                self.cond_values[:, j] = np.float32(const_raw[j])
            if self.condition_control == 'static_new':
                self.cond_mask[:, j] = 1
        mask_note = ('全部固定为 1（同时取消值与缺失模式的个体差异）'
                     if self.condition_control == 'static_new'
                     else '保持真实（检验缺失模式的独立贡献）')
        print(f'[condition control] {self.condition_control}：新增组 {MDD_NEW_GROUP} '
              f'的值已固定为训练常量；mask {mask_note}；共享三字段保持逐人真实值')

    def condition_matrix_for(self, subject_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        """按被试 ID 取条件矩阵（供诊断/置换对照与逐被试评估使用）。"""
        pos = [self._subject_pos[str(s)] for s in subject_ids
               if str(s) in self._subject_pos]
        return {
            'z': self.cond_z[pos], 'code': self.cond_code[pos],
            'mask': self.cond_mask[pos], 'raw': self.cond_values[pos],
            'subject_index': np.asarray(pos, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # 索引与数据加载
    # ------------------------------------------------------------------
    def _normlize_path(self, subj_id: str) -> Path:
        return self.normlize_dir / f"ROISignals_{subj_id}.mat"

    def _window_paths(self, subj_id: str) -> List[Path]:
        return [self.window_dir / f"ROISignals_{subj_id}-{i}.mat"
                for i in range(1, self.total_windows + 1)]

    def _raw_path(self, subj_id: str) -> Path:
        return self.raw_dir / f"ROISignals_{subj_id}.mat"

    def _load_raw_array(self, subj_id: str) -> np.ndarray:
        """读取 raw 原始序列并按该来源明确的 [T, F] 布局解释（方案 §4.1）。

        不使用 ``max(shape)`` 猜时间轴：Raw 的 ROISignals 约定第 0 维为 TR、
        第 1 维为 ROI。缺失或形状非法直接报告路径，不回退 Normlize/窗口缓存。
        """
        path = self._raw_path(subj_id)
        if not path.exists():
            raise FileNotFoundError(f"raw 分支缺少原始序列文件: {path}")
        arr = self._read_mat_array(path, preferred_keys=['ROISignals', 'Data'])
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"{subj_id} 的 Raw BOLD 期望二维 [T, F]，收到 {arr.shape}（{path}）")
        if arr.shape[1] < self.num_rois:
            raise ValueError(
                f"{subj_id} 的 Raw BOLD 仅 {arr.shape[1]} 列 ROI，少于 "
                f"num_rois={self.num_rois}，无法取前 {self.num_rois} 列（{path}）")
        return arr

    def _probe_series_length(self, subj_id: str) -> Optional[int]:
        """探测单被试连续序列长度 T（不读取全部数据，仅取 shape）。

        优先 Normlize 连续序列（原始 ROI BOLD，[T, F] 或 [F, T]，按 num_rois
        判向——时间轴长度可能小于 ROI 数，不用 max(shape) 猜）；不可用时按
        ``ROISignals_window`` 的 50% 重叠布局推算：偶数序号窗（1,3,5,...）恰好
        覆盖 0:S, S:2S, ... 的连续片段（见 data/utils/window_stride.py）。
        raw 分支从 Raw 的明确时间轴取完整长度（不截断）。
        """
        if self.bold_source == 'raw':
            path = self._raw_path(subj_id)
            if not path.exists():
                return None
            arr = self._load_raw_array(subj_id)
            return int(arr.shape[0])
        if self.bold_source in ('auto', 'normlize'):
            path = self._normlize_path(subj_id)
            if path.exists():
                arr = np.asarray(self._read_mat_array(
                    path, preferred_keys=['ROISignals', 'Data', 'bold']))
                if arr.ndim != 2:
                    raise ValueError(
                        f"{subj_id} 的 Normlize BOLD 期望二维，收到 {arr.shape}（{path}）")
                if arr.shape[1] == self.num_rois and arr.shape[0] != self.num_rois:
                    return int(arr.shape[0])                        # [T, F]
                if arr.shape[0] == self.num_rois and arr.shape[1] != self.num_rois:
                    return int(arr.shape[1])                        # [F, T]
                raise ValueError(
                    f"{subj_id} 的 BOLD 形状 {arr.shape} 无法与 ROI 数 "
                    f"{self.num_rois} 对齐（{path}）")
        if self.bold_source == 'normlize':
            return None
        window_paths = self._window_paths(subj_id)
        if not all(p.exists() for p in window_paths):
            return None
        n_used = (self.total_windows + 1) // 2
        return n_used * self.source_window_len

    def _build_index(self):
        """按 TR 直接索引：每个被试一条连续序列 [F, T]，T 即该被试的 TR 总数。

        条件注入开启时，队列**只由影像有效性决定**（BOLD/SC 存在且长度足够）；
        在 merged 中缺整行或字段缺失的被试全部保留，条件记为缺失（方案 §4）。
        legacy 数值通路（``condition_group=None``）同样**不因临床缺行删人**：
        所有对照臂必须共用同一完整影像队列与同一份 split（验收 F），否则
        「条件开 / 条件关」两次运行会得到不同 cohort 与不同划分。

        raw 分支（方案 §4.1）从 ``Raw/<group>/ROISignals_*.mat`` 枚举被试，
        不要求 Mask 存在（history_fc 的图来自当前可见历史）；legacy_mask 图在
        ``_get_subject`` 按需读取，缺失时直接报路径错误。
        """
        if self.bold_source == 'raw':
            if not self.raw_dir.is_dir():
                raise FileNotFoundError(
                    f"bold_source='raw' 需要 Raw 目录: {self.raw_dir}")
            source_files = sorted(self.raw_dir.glob('ROISignals_*.mat'),
                                  key=lambda p: p.stem)
            prefix = 'ROISignals_'
        else:
            source_files = sorted(self.sc_dir.glob('Mask_*.mat'), key=lambda p: p.stem)
            prefix = 'Mask_'
        skipped_missing = 0
        missing_examples: List[str] = []
        no_clinical: List[str] = []
        too_short: List[str] = []
        # raw 路线（calibration_length>0）：锚点下界 max(K-1, calib-1)，故被试
        # 至少需要 max(context_min, calibration_length) + max_offset 个 TR；
        # legacy 路线保持历史口径 context_min + 1 不变。
        if self.calibration_length > 0:
            min_required = max(self.context_min, self.calibration_length) \
                + self.max_offset
        else:
            min_required = self.context_min + 1  # 至少 context_min 段历史 + 1 个目标 TR
        self._min_series_len = min_required

        for src_file in source_files:
            subj_id = str(src_file.stem.replace(prefix, '')).strip()
            has_clinical = subj_id in self.clinical_dict
            if self.mode == 'finetune' and not self.condition_enabled and not has_clinical:
                # 只登记，不删人：评分缺失按「无临床信息」编码（见 _get_subject）
                no_clinical.append(subj_id)
            series_len = self._probe_series_length(subj_id)
            if series_len is None or series_len <= 0:
                skipped_missing += 1
                if self.bold_source == 'raw':
                    missing_examples.append(str(self._raw_path(subj_id)))
                continue
            if series_len < min_required:
                too_short.append(f"{subj_id}(T={series_len})")
                continue
            self._subject_pos[subj_id] = len(self.subject_ids)
            self.subject_ids.append(subj_id)
            self.subject_series_len[subj_id] = series_len
            self.subject_n_tr[subj_id] = series_len
            if self.mode == 'finetune' and not self.condition_enabled and has_clinical:
                # 分层划分使用首个字段（默认 HAMD 总分）作为标量分层依据
                self.subject_to_score[subj_id] = float(self.clinical_dict[subj_id][0])

        if not self.subject_ids:
            raise ValueError(
                f"next_timepoint 数据集为空：mode={self.mode}, context_min={self.context_min}"
                f"（需 T >= {min_required}）。"
                "请检查 data_root 下的 Normlize / ROISignals_window / Mask 目录。")

        if self.condition_enabled:
            self._load_condition_columns()
        else:
            self.no_clinical_subjects = no_clinical

        series_lens = sorted(self.subject_n_tr.values())
        print(
            f"[next_timepoint] 数据集构建完毕 | 模式: {self.mode} | 有效被试数: "
            f"{len(self.subject_ids)} | TR 数 min/median/max="
            f"{series_lens[0]}/{int(np.median(series_lens))}/{series_lens[-1]}"
        )
        print(
            f"[next_timepoint] 数据源={self.bold_source} | 图来源={self.graph_source}"
            f" | 校准长度={self.calibration_length if self.calibration_length > 0 else 'off'}"
            f" | 临床表无评分被试="
            f"{len(self.no_clinical_subjects)}（**保留**，按无临床信息编码）| 跳过: "
            f"数据缺失 {skipped_missing} | "
            f"序列过短 {len(too_short)}（需 T >= {min_required}）"
            + (f"（例如 {too_short[:3]}）" if too_short else "")
        )
        if missing_examples:
            print(f"[next_timepoint] raw 缺失文件示例: {missing_examples[:3]}")

        # 控制 manifest（方案 §4.3）：在索引建立后加载，逐条校验文件存在性与
        # 队列内被试的 T/F 匹配；无 manifest 时 controls 保持 None（不伪造刺激）。
        if self.control_manifest is not None:
            self._load_control_manifest(self.control_manifest)

    def _load_control_manifest(self, manifest_path: str) -> None:
        """读取控制 manifest CSV 并校验每个条目（方案 §4.3）。

        CSV 两列 ``subject_id,controls_path``；相对路径相对 manifest 所在目录。
        每个 NPZ 只有必需数组 ``u [T, F]``：u[t] 对应到达观测 t 的区间，因此
        锚点 t 的未来切片是 ``u[t+1:t+H+1]``。要求：

        - 条目文件缺失、NPZ 缺 ``u`` 键、维度/宽度/长度不匹配或含非有限值 →
          **明确报错**（不用无刺激代替缺文件，不静默跳过）；
        - T 与该被试的完整序列长度严格一致；
        - 控制幅值不使用 BOLD 均值/方差标准化，按上游定义的原值进入模型。
        """
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"control_manifest 不存在: {path}")
        df = pd.read_csv(path)
        cols = [c.strip() for c in df.columns]
        if len(cols) < 2 or cols[0] != 'subject_id' or cols[1] != 'controls_path':
            raise ValueError(
                f"control_manifest 需要表头 'subject_id,controls_path'，收到 {cols}（{path}）")
        controls: Dict[str, np.ndarray] = {}
        base_dir = path.parent
        for _, row in df.iterrows():
            subj_id = str(row['subject_id']).strip()
            rel = str(row['controls_path']).strip()
            npz_path = Path(rel) if Path(rel).is_absolute() else base_dir / rel
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"control_manifest 条目 {subj_id} 的控制文件缺失: {npz_path}（{path}）")
            with np.load(npz_path) as z:
                if 'u' not in z:
                    raise ValueError(
                        f"{subj_id} 的控制 NPZ 缺少 'u' 键（{npz_path}）")
                u = np.asarray(z['u'], dtype=np.float32)
            if u.ndim != 2:
                raise ValueError(
                    f"{subj_id} 的控制 u 期望二维 [T, F]，收到 {u.shape}（{npz_path}）")
            if u.shape[1] != self.num_rois:
                raise ValueError(
                    f"{subj_id} 的控制 u 末维 {u.shape[1]} 与 num_rois={self.num_rois} "
                    f"不一致（{npz_path}）")
            if not np.isfinite(u).all():
                raise ValueError(f"{subj_id} 的控制 u 含非有限值（{npz_path}）")
            expected_t = self.subject_series_len.get(subj_id, None)
            if expected_t is not None and u.shape[0] != expected_t:
                raise ValueError(
                    f"{subj_id} 的控制 u 长度 {u.shape[0]} 与序列 T={expected_t} "
                    f"不一致（{npz_path}；序列长度按完整时间轴口径）")
            controls[subj_id] = u
        queue_hits = sum(1 for s in controls if s in self._subject_pos)
        print(f"[controls] manifest 已加载: {path} | 条目={len(controls)} "
              f"| 命中当前队列={queue_hits} | u 布局 [T,F]，区间 u[t] 到达观测 t")
        self.controls_by_subject = controls

    def _load_condition_columns(self) -> None:
        """加载 merged 条件表并与影像队列对齐（缺行被试 → 全字段缺失）。"""
        table = load_condition_table(self.data_root, self.condition_file,
                                     self.condition_group)
        attach_missing_rows(table, self.subject_ids)
        self.condition_table = table
        v, m = table.align(self.subject_ids)
        self.cond_values, self.cond_mask = v, m
        audit = table.audit
        print(
            f"[condition] merged={table.file_name} group={table.group} "
            f"K={table.k} 行={audit.n_rows} | 影像队列={len(self.subject_ids)} "
            f"（merged 无整行 {len(table.missing_row_subjects)} 人，保留并记全字段缺失）")
        print(f"[condition] 有效观测数/被试队列 = {audit.n_valid}")
        print(f"[condition] 实际来源列 = {audit.alias_sources}")
        for name, info in audit.sentinel_counts.items():
            if info.get('remap_applied'):
                print(f"[condition] {name} 哨兵检测 原始={info['raw_sentinel_counts']} "
                      f"映射后={info['mapped_sentinel_counts']}（-1→2、-9999→0，mask 恒为 1）")
        if table.missing_row_subjects:
            print(f"[condition] 无表行被试示例: {table.missing_row_subjects[:5]}")

    def stratify_value_by_subject(self, subj_id: str,
                                  key: Optional[str]) -> Optional[float]:
        """按**显式分层键**取被试的分层数值；缺失返回 ``None``。

        分层键由调用方指定（MDD 保持 HAMD），不随条件字段顺序 ``fields[0]`` 变化；
        键缺失时由调用方使用预注册的缺失分桶 + 确定性分配，**不因人**。
        """
        if key is None:
            return None
        if self.condition_enabled:
            table = self.condition_table
            names = [s.name for s in table.specs]
            if key not in names:
                raise ValueError(
                    f'分层键 {key} 不在条件 schema {names} 中；分层键必须显式登记。')
            row = table.row_of_id.get(str(subj_id), -1)
            if row < 0:
                return None
            j = names.index(key)
            if not int(table.mask[row, j]):
                return None
            val = float(table.values[row, j])
            return val if np.isfinite(val) else None
        if key != self.pathology_fields[0]:
            raise ValueError(
                f'legacy 数值通路只提供 {self.pathology_fields[0]} 作为分层标量，'
                f'无法按 {key} 分层；请开启条件注入（--condition_group）。')
        return self.subject_to_score.get(str(subj_id), None)

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

    def _load_series(self, subj_id: str, n_rois: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """返回 (连续 ROI BOLD 序列 [F, T], 校准统计量或 None)。

        raw 分支序列为校准后数据尺度 X，校准统计量为 (mu0, s0)，各 [F, 1]
        （每 ROI 只用最早 calibration_length 个 TR 估计，方案 §4.1）；
        其余分支 cal_stats=None（无前缀校准概念）。
        """
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

        if self.bold_source == 'raw':
            # raw 分支（方案 §4.1）：明确 [T, F] 布局取前 num_rois 列（完整
            # 时间轴，不截断），每 ROI 只用最早 calibration_length 个 TR 计算
            # 固定 mu0/s0 后标准化整个序列；缺失直接报路径（_load_raw_array
            # 内），不回退旧缓存。
            arr = self._load_raw_array(subj_id)
            n_tr = int(arr.shape[0])
            if n_tr < self.calibration_length:
                raise ValueError(
                    f"{subj_id} 的 Raw BOLD 只有 {n_tr} 个 TR，少于 "
                    f"calibration_length={self.calibration_length}（{self._raw_path(subj_id)}）")
            x = np.ascontiguousarray(arr[:n_tr, :self.num_rois].T)      # [F, T]
            cal = x[:, :self.calibration_length]
            mu0 = cal.mean(axis=1, keepdims=True)                       # [F, 1]
            s0 = np.sqrt(cal.var(axis=1, keepdims=True) + 1e-6)         # [F, 1]
            cal_stats = (mu0.astype(np.float32), s0.astype(np.float32))
            if not self._source_logged:
                print(f"[next_timepoint] raw 数据源: {self.raw_dir}"
                      f"（[T,F] → 前 {self.num_rois} 列 → [F,T]，完整时间轴不截断，"
                      f"最早 {self.calibration_length} TR 固定 mu0/s0 校准）")
                self._source_logged = True
            series = ((x - mu0) / s0).astype(np.float32, copy=False)
            return series, cal_stats

        if self.bold_source in ('auto', 'normlize'):
            path = self._normlize_path(subj_id)
            if path.exists():
                if not self._source_logged:
                    print(f"[next_timepoint] 连续 BOLD 数据源: {path.parent}（[T,F] → [F,T]）")
                    self._source_logged = True
                return _orient(self._read_mat_array(
                    path, preferred_keys=['ROISignals', 'Data', 'bold']),
                    'Normlize'), None

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
        return series.astype(np.float32, copy=False), None

    def _get_subject(self, subj_id: str) -> Dict[str, np.ndarray]:
        cached = self._cache.get(subj_id, None)
        if cached is not None:
            return cached

        cal_stats = None
        if self.bold_source == 'raw':
            # raw 分支：ROI 数由 num_rois 参数提供（方案 §4.1），不依赖 Mask 推断。
            n_rois = self.num_rois
            if self.graph_source == 'legacy_mask':
                sc_path = self.sc_dir / f"Mask_{subj_id}.mat"
                if not sc_path.exists():
                    raise FileNotFoundError(
                        f"graph_source='legacy_mask' 需要 mask 文件: {sc_path}")
                sc_matrix = self._read_mat_array(
                    sc_path, preferred_keys=['SC', 'sc_matrix', 'mask', 'Mask'])
                if sc_matrix.ndim != 2 or sc_matrix.shape[0] != sc_matrix.shape[1]:
                    raise ValueError(
                        f"SC/mask must be square [F, F], got {sc_matrix.shape} for {subj_id}")
                sc_matrix = self._normalize_sc(sc_matrix)
            else:
                # history_fc：单位图占位以兼容现有 batch 拆包；模型实际图来自
                # 当前可见历史（_history_fc_adj），不会把占位当实际图。
                sc_matrix = np.eye(self.num_rois, dtype=np.float32)
        else:
            sc_path = self.sc_dir / f"Mask_{subj_id}.mat"
            sc_matrix = self._read_mat_array(
                sc_path, preferred_keys=['SC', 'sc_matrix', 'mask', 'Mask'])
            if sc_matrix.ndim != 2 or sc_matrix.shape[0] != sc_matrix.shape[1]:
                raise ValueError(f"SC/mask must be square [F, F], got {sc_matrix.shape} for {subj_id}")
            sc_matrix = self._normalize_sc(sc_matrix)
            n_rois = int(sc_matrix.shape[0])

        series, cal_stats = self._load_series(subj_id, n_rois)
        if series.shape[0] != n_rois:
            raise ValueError(
                f"ROI 顺序/数量不一致：{subj_id} BOLD {series.shape[0]} vs 期望 {n_rois}；"
                "next_timepoint 要求 BOLD ROI i 与图 ROI i 严格对应。")
        T = int(series.shape[1])
        expected = self.subject_series_len.get(subj_id, None)
        if expected is not None and T != expected:
            raise ValueError(
                f"{subj_id} 实际序列长度 {T} 与索引阶段探测的 {expected} 不一致："
                "请检查数据目录是否在两次运行之间发生变化。")
        if T < self._min_series_len:
            raise ValueError(
                f"{subj_id} 只有 {T} 个 TR，不足以构成最小样本（需 T >= "
                f"{self._min_series_len}）。")

        entry: Dict[str, np.ndarray] = {
            'series': np.ascontiguousarray(series.astype(np.float32, copy=True)),  # [F, T]
            'sc': sc_matrix,                            # [F, F]
            'subj_id': subj_id,
            'series_len': T,
        }
        if cal_stats is not None:
            # raw 前缀校准统计量（固定 mu0/s0，各 [F,1]）：供推理侧把新原始观测
            # 转换到同一数据尺度（inference_twin 复用，不再重新估计）。
            entry['cal_mu0'] = cal_stats[0]
            entry['cal_s0'] = cal_stats[1]
        if self.mode == 'finetune' and not self.condition_enabled:
            vec = self.clinical_dict.get(subj_id, None)
            if vec is None:
                # 临床表无该被试：为保住「所有臂共用一个完整影像队列」（验收 F）
                # 仍然保留此人，评分按无信息常量 0 编码（与 pathology_missing='zero'
                # 同口径）；分层标量保持缺失（None），由缺失桶确定性分配。
                vec = np.zeros(self.pathology_input_dim, dtype=np.float32)
            entry['pathology_score'] = np.asarray(vec, dtype=np.float32)

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
        （raw 分支为校准后序列，否则 Normlize 连续序列，缺失时按 50% 重叠滑窗重构）。
        """
        return self._get_subject(str(subj_id))['series']

    def get_calibration_stats(self, subj_id: str):
        """返回该被试的固定前缀校准统计量 (mu0, s0)，各 [F, 1]；非 raw 分支为 None。

        供推理侧把新原始观测用**同一扫描**的 mu0/s0 转换到数据尺度，
        不重新估计统计量（方案 §4.1）。
        """
        entry = self._get_subject(str(subj_id))
        if 'cal_mu0' not in entry:
            return None
        return entry['cal_mu0'], entry['cal_s0']

    def get_controls(self, subj_id: str) -> Optional[np.ndarray]:
        """返回该被试的控制序列 u [T, F]；无 manifest 或该被试无条目时为 None。"""
        if self.controls_by_subject is None:
            return None
        return self.controls_by_subject.get(str(subj_id), None)

    def legal_context_lengths(self, subj_id: str) -> List[int]:
        """合法 context 长度：K <= min(context_max, T-1)（至少 1 个未来 TR）。

        raw 路线（calibration_length>0）锚点下界为 max(K-1, calib-1)，故还需
        K <= T - max_offset 才存在合法锚点；否则训练视图会采样到空锚点区间。
        """
        n = self.n_tr(subj_id)
        upper = min(self.context_max, n - 1)
        if self.calibration_length > 0:
            upper = min(upper, n - self.max_offset)
        if self.context_lengths:
            return [k for k in self.context_lengths if k <= upper]
        return list(range(self.context_min, upper + 1)) if upper >= self.context_min else []

    def legal_anchor_range(self, subj_id: str, context_len: int,
                           max_offset: int = None) -> Tuple[int, int]:
        """合法预测位置区间（含端点，0 基 cutoff t）：t-K+1 >= 0 且 t+max_offset <= T-1。

        raw 路线（方案 §4.1）下界改为 max(K-1, calibration_length-1)：未来样本
        不参与基线/profile 校准，首个预测至少是第 33 个 TR（calibration_length=32）。
        """
        n = self.n_tr(subj_id)
        off = int(max_offset) if max_offset else self.max_offset
        lo = context_len - 1
        if self.calibration_length > 0:
            lo = max(lo, self.calibration_length - 1)
        return lo, n - 1 - off

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
        raw 路线附加：
            calibration_bold [F, 1, calibration_length]  同一扫描最早校准前缀（数据尺度）
            controls         [R, F]      u[t+1:t+R+1]，缺尾区间填零（有 manifest 时）
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
        if self.calibration_length > 0:
            # 固定基线校准前缀（数据尺度 X）：独立于当前锚点，profile 与 Raw
            # 标准化使用同一前缀；锚点之后的样本不参与校准。
            cal = series[:, :self.calibration_length]                  # [F, L]
            out['calibration_bold'] = torch.from_numpy(np.ascontiguousarray(
                np.asarray(cal[:, None, :], dtype=np.float32)))        # [F,1,L]
        if self.controls_by_subject is not None:
            u = self.controls_by_subject.get(subj_id, None)
            h = int(future_steps)
            ctrl = np.zeros((h, self.num_rois), dtype=np.float32)
            if u is not None and h > 0:
                # u[t] 对应到达观测 t 的区间；锚点 t 的未来区间切片 u[t+1:t+H+1]，
                # 缺尾区间随 future_mask 填零（方案 §4.3）。
                avail = max(0, min(h, int(u.shape[0]) - (t + 1)))
                if avail > 0:
                    ctrl[:avail] = u[t + 1: t + 1 + avail]
            out['controls'] = torch.from_numpy(ctrl)                   # [H, F]
        if future_steps > 0:
            future_end = min(int(future_steps), n - 1 - t)
            future = np.zeros((series.shape[0], int(future_steps)), dtype=np.float32)
            if future_end > 0:
                future[:, :future_end] = series[:, t + 1: t + 1 + future_end]
            mask = np.zeros(int(future_steps), dtype=np.float32)
            mask[:future_end] = 1.0
            out['future'] = torch.from_numpy(future)
            out['future_mask'] = torch.from_numpy(mask)
        if self.condition_enabled:
            if self.cond_z is None:
                raise RuntimeError(
                    '条件统计量尚未附加（attach_condition_stats 未调用）；'
                    'train-only 统计量必须在冻结划分后拟合，不能靠运行时默认值猜测。')
            out['cond_z'] = torch.from_numpy(np.ascontiguousarray(
                self.cond_z[subj_pos], dtype=np.float32))              # [K]
            out['cond_code'] = torch.from_numpy(np.ascontiguousarray(
                self.cond_code[subj_pos], dtype=np.int64))             # [K]
            out['cond_mask'] = torch.from_numpy(np.ascontiguousarray(
                self.cond_mask[subj_pos], dtype=np.float32))           # [K]
            # 原值（替换后类别码 / 连续原始值，缺失=nan）只供诊断与置换对照，
            # 不直接进模型；模型消费 z/code/mask。
            out['cond_raw'] = torch.from_numpy(np.ascontiguousarray(
                self.cond_values[subj_pos], dtype=np.float32))          # [K]
            out['subject_index'] = torch.tensor(int(subj_pos), dtype=torch.long)
        elif self.mode == 'finetune':
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
                 future_steps: int = 0, anchor_max_offset: int = 0):
        self.base = base
        self.subject_indices = list(subject_indices)
        self.augmentation = augmentation
        self.random_context = bool(random_context)
        self.random_cutoff = bool(random_cutoff)
        self.seed = int(seed)
        # >0 时额外返回未来 R 步真值（供 --enable_rollout_loss 的短程 rollout 监督，
        # 以及 TFM 的 CPM 全 horizon 监督）
        self.future_steps = int(future_steps)
        # >0 时锚点采样区间改用该 max_offset（保证 t+anchor_max_offset 处仍有真值，
        # 供 CPM 全 horizon 监督；0 = 沿用 base.max_offset，legacy 行为不变）
        self.anchor_max_offset = int(anchor_max_offset)

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
        lo, hi = self.base.legal_anchor_range(
            subj_id, K,
            max_offset=(self.anchor_max_offset or None))
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
                 rollout_tasks_per_subject: int = 0, rollout_steps: int = 0,
                 all_future_steps: int = 0):
        self.base = base
        self.subject_indices = list(subject_indices)
        self.context_length = int(context_length)
        self.anchors_per_subject = int(anchors_per_subject)
        self.rollout_tasks_per_subject = int(rollout_tasks_per_subject)
        self.rollout_steps = int(rollout_steps)
        # >0 时**所有** anchor 都返回未来真值（供 TFM 的 CPM 逐 horizon 指标；
        # 与 rollout_steps 取大者统一宽度，0 = legacy 行为）
        self.all_future_steps = int(all_future_steps)

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
        # 所有 anchor 返回同一宽度的 future（batch collate 要求维度一致）；
        # rollout_steps（rollout/FC 用，通常更长）与 all_future_steps（CPM 用）取大者
        future_steps = max(self.rollout_steps, self.all_future_steps)
        sample = self.base.make_sample(subj_pos, self.context_length, t,
                                       future_steps=future_steps)
        if 'future' not in sample:
            # future_steps=0 的情形（legacy 且非 rollout 锚点）也返回同形张量
            # （全零 + 掩码全 0），保证 batch collate 一致；这些行不参与
            # rollout/FC/频谱/CPM 指标（显式筛掉）
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
        # ---- 条件注入（v1 revised 20261008）----
        condition_group: Optional[str] = None,
        condition_file: Optional[str] = None,
        split_file: Optional[str] = None,
        stratify_key: Optional[str] = None,
        condition_base_stats=None,
        condition_extra_codes: Optional[Dict[str, Sequence[float]]] = None,
        condition_stats_mode: str = 'fit',
        condition_control: str = 'none',
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
        train_anchor_max_offset: int = 0,
        eval_all_future_steps: int = 0,
        # ---- graph_ode 路线（方案 §4）：透传给 next-timepoint 数据集 ----
        num_rois: int = 116,
        graph_source: str = 'legacy_mask',
        calibration_length: int = 0,
        control_manifest: Optional[str] = None,
    ):
        task_mode = str(task_mode)
        if task_mode != 'next_timepoint':
            raise ValueError(
                f"task_mode 仅支持 'next_timepoint'，收到 '{task_mode}'")
        self.task_mode = task_mode
        self.mode = str(mode)
        self.condition_group = str(condition_group) if condition_group else None
        self.condition_file = condition_file
        self.split_file = str(split_file) if split_file else None
        self.condition_base_stats = condition_base_stats
        self.condition_extra_codes = dict(condition_extra_codes or {})
        if condition_stats_mode not in ('fit', 'inherit'):
            raise ValueError(
                f"condition_stats_mode 仅支持 fit/inherit，收到 '{condition_stats_mode}'")
        self.condition_stats_mode = str(condition_stats_mode)
        if condition_control not in ('none', 'static_new', 'mask_only'):
            raise ValueError(
                "condition_control 仅支持 none/static_new/mask_only，收到 "
                f"'{condition_control}'")
        self.condition_control = str(condition_control)
        # 分层键显式固定（MDD 保持 HAMD），不随条件字段顺序变化
        self.stratify_key = (str(stratify_key) if stratify_key
                             else STRATIFY_KEY_BY_MODE.get(self.mode))
        self.condition_stats = None
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
        # TFM 的 CPM 支持：训练锚点按 cpm_horizon 右移上界、评估全 anchor 带 future
        self.train_anchor_max_offset = int(train_anchor_max_offset)
        self.eval_all_future_steps = int(eval_all_future_steps)
        self.num_rois = int(num_rois)
        self.graph_source = str(graph_source)
        self.calibration_length = int(calibration_length)
        self.control_manifest = str(control_manifest) if control_manifest else None
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
            condition_group=self.condition_group,
            condition_file=self.condition_file,
            condition_control=self.condition_control,
            total_windows=total_windows,
            seq_len=seq_len,
            bold_source=self.bold_source,
            subject_cache_size=self.subject_cache_size,
            cache_in_memory=cache_in_memory,
            num_rois=self.num_rois,
            graph_source=self.graph_source,
            calibration_length=self.calibration_length,
            control_manifest=self.control_manifest,
        )

        train_indices, val_indices, test_indices, train_subjects, val_subjects, test_subjects = self._split_indices_by_subject(
            mode=mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_bins=stratify_bins,
            stratify_key=self.stratify_key,
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
            future_steps=self.train_rollout_steps,
            anchor_max_offset=self.train_anchor_max_offset)
        self.val_dataset = NextPointEvalView(
            self.base_dataset, val_indices,
            context_length=self.eval_context_length,
            anchors_per_subject=self.eval_anchors_per_subject,
            rollout_tasks_per_subject=self.eval_rollout_tasks_per_subject,
            rollout_steps=self.eval_rollout_steps,
            all_future_steps=self.eval_all_future_steps)
        self.test_dataset = NextPointEvalView(
            self.base_dataset, test_indices,
            context_length=self.eval_context_length,
            anchors_per_subject=self.eval_anchors_per_subject,
            rollout_tasks_per_subject=self.eval_rollout_tasks_per_subject,
            rollout_steps=self.eval_rollout_steps,
            all_future_steps=self.eval_all_future_steps)

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

        # 划分冻结**之后**才拟合预处理统计量（train-only；方案 §4 / 验收 E）
        if self.base_dataset.condition_enabled:
            self._fit_condition_stats()

        if mode == 'finetune' and len(train_subjects) > 0 and len(val_subjects) > 0:
            key = self.stratify_key
            def _vals(sids):
                if self.base_dataset.condition_enabled:
                    return [v for v in (self.base_dataset.stratify_value_by_subject(s, key)
                                        for s in sids) if v is not None]
                return [s for s in (self.base_dataset.get_pathology_scalar_by_subject(sid)
                                    for sid in sids) if s is not None]
            train_scores = _vals(train_subjects)
            val_scores = _vals(val_subjects)
            test_scores = _vals(test_subjects)
            if train_scores and val_scores:
                print(
                    f"{key} 分布 | "
                    f"train mean={np.mean(train_scores):.3f}, std={np.std(train_scores):.3f} | "
                    f"val mean={np.mean(val_scores):.3f}, std={np.std(val_scores):.3f} | "
                    f"test mean={np.mean(test_scores):.3f}, std={np.std(test_scores):.3f}"
                )

    def _fit_condition_stats(self) -> None:
        """冻结划分**之后**用 train 被试的有效观测拟合 train-only 统计量。

        - 连续字段：:math:`\\mu_j`=train 有效观测中位数，:math:`s_j=\\operatorname{IQR}/1.349`
          （退化回退 std，再回退 1）；
        - 类别字段：类别词表只在 train 有效观测上登记，外加
          ``condition_extra_codes`` 显式预注册的码（不允许运行时静默扩表）；
        - HC→MDD：给出 ``condition_base_stats`` 时，Age/education 的 mu/scale 与类别
          协议**逐字段继承** HC-train，只对新增五字段拟合；绝不调用一次通用 fit
          覆盖继承统计量（方案 §4，验收 E）。
        """
        from utils.condition_schema import (DEMO_GROUP, MDD_NEW_GROUP,
                                            audit_unseen_codes,
                                            fit_condition_stats, layout_hash)
        ds = self.base_dataset
        table = ds.condition_table
        specs = table.specs
        train_rows = np.asarray([ds._subject_pos[s] for s in self.train_subjects],
                                dtype=np.int64)
        if train_rows.size == 0:
            raise RuntimeError('训练被试为空，无法拟合 train-only 统计量。')
        v, m = ds.cond_values, ds.cond_mask
        base = self.condition_base_stats

        if self.condition_stats_mode == 'inherit':
            # 独立评估/推理：统计量**必须**逐位复用 checkpoint 内的 train-only 快照，
            # 不能在这里重新拟合（否则评估编码与训练编码可能因队列/划分漂移而不一致）
            if base is None:
                raise ValueError(
                    "condition_stats_mode='inherit' 需要 condition_base_stats"
                    '（通常来自 checkpoint 的 config.condition.stats）')
            stats = fit_condition_stats(specs, v, m, train_rows, only_fields=[], base=base)
            self.condition_stats = stats
            ds.attach_condition_stats(stats)
            print('[condition] inherit 模式：全部字段逐位复用传入统计量，未重新拟合')
            return
        if base is not None:
            if table.group != 'mdd8':
                raise ValueError(
                    f'condition_base_stats 仅用于 mdd8 继承 HC 统计量，收到 '
                    f'group={table.group}')
            # 继承路径同样要接收 extra_codes：新增五字段的预注册码只有在这里进词表，
            # 否则下方 audit 报错建议的 --condition_extra_codes 传进来会被静默丢弃。
            demo = set(DEMO_GROUP)
            shared_reg = sorted(demo & set(self.condition_extra_codes))
            if shared_reg:
                raise ValueError(
                    f'--condition_extra_codes 含继承字段 {shared_reg}：Sex/education 的类别'
                    '协议由本轮 HC-train 冻结并逐位继承（方案 §四），MDD 侧不得扩大它。'
                    '确实需要新增码时，必须在 HC 试次登记该码并重新训练 HC，再继承新快照。')
            stats = fit_condition_stats(specs, v, m, train_rows,
                                        only_fields=list(MDD_NEW_GROUP), base=base,
                                        extra_codes=self.condition_extra_codes)
            for name in DEMO_GROUP:
                got, src = stats.get(name), base.get(name)
                if (got.fitted, got.mu, got.scale, got.codes) != (
                        src.fitted, src.mu, src.scale, src.codes):
                    raise AssertionError(
                        f'继承字段 {name} 的统计量被覆盖：{got} vs {src}。'
                        'HC→MDD 必须逐字段继承 Age/education 的 mu/scale 与类别协议。')
            print(f'[condition] 继承 HC-train 统计量：{list(DEMO_GROUP)}；'
                  f'仅在 MDD-train 拟合 {list(MDD_NEW_GROUP)}')
        else:
            if table.group == 'mdd8':
                print('[condition] 警告: 未提供 HC 统计量（condition_base_stats=None），'
                      '全部字段在 MDD-train 拟合；这是「无继承」变体，与 HC→MDD '
                      '迁移实验不同口径，比较时须显式区分。')
            stats = fit_condition_stats(specs, v, m, train_rows)

        unfitted = [s.name for s in specs if s.is_continuous
                    and not stats.get(s.name).fitted]
        if unfitted:
            print(f'[condition] 全训练缺失字段（fitted=false, n_valid=0，编码走 z=0/m=0 '
                  f'分支，不从 val/test 取值）：{unfitted}')

        unseen = audit_unseen_codes(specs, stats, v, m, train_rows)
        if unseen:
            def _fmt_code(c: float) -> str:
                return str(int(c)) if float(c).is_integer() else repr(c)

            # 继承来的共享字段不能靠 --condition_extra_codes 补码（那个开关对
            # only_fields=MDD_NEW_GROUP 的拟合是空转），必须回到 HC 侧登记重训；
            # 否则报错会建议一个对本路径无效的修复，用户按提示改完仍然报同一个错。
            if base is not None:
                shared_hits = {k: vv for k, vv in unseen.items() if k in set(DEMO_GROUP)}
                if shared_hits:
                    raise ValueError(
                        f'继承字段出现未登记的有效类别码：{shared_hits}。这些字段的词表由本轮 '
                        'HC-train 冻结并逐位继承（方案 §四），MDD 侧禁止扩大；'
                        '正确处置是在 HC 试次用 --condition_extra_codes 登记该码并重新训练 HC，'
                        '再继承新的统计量快照。')
            suggestion = ';'.join(
                f'{k}:{",".join(_fmt_code(c) for c in vv)}' for k, vv in unseen.items())
            raise ValueError(
                f'训练集之外存在未登记的有效类别码：{unseen}。协议要求显式登记而不是'
                f'伪装成缺失；请确认后传 --condition_extra_codes "{suggestion}"'
                f'（运行时禁止静默扩大 Embedding，否则 checkpoint 不可重建）。')

        self.condition_stats = stats
        ds.attach_condition_stats(stats)
        print('[condition] layout_hash='
              f'{layout_hash(specs, stats=stats)} | merged_sha256='
              f'{table.merged_sha256[:16]}… | 实际来源列={table.audit.alias_sources}')

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

    def _split_subject_ids(self, mode: str, val_ratio: float, test_ratio: float,
                           stratify_bins: int,
                           stratify_key: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
        """将受试者划分为训练集、验证集和测试集（8:1:1）。

        优先级：显式 ``split_file`` > 版本化 manifest > 现场生成并落盘。

        版本化固定：切分结果落盘到 ``<data_root>/subject_split_<mode>.json``
        （含 seed/比例/分箱/分层键等配置指纹）。同配置的后续运行直接复用名单，
        防止代码演化导致的切分漂移（跨实验 test 集一致性）；配置变化或被试集合
        变化时默认报错，需显式 ``refresh_split_manifest=True`` 重新生成。

        所有对照臂（无条件 / 各条件臂）共用同一份 cohort 与 split：cohort 由影像
        有效性决定，**不因条件缺失删人**（验收 F）。
        """
        cohort = set(self.base_dataset.get_subject_ids())
        if self.split_file:
            return self._load_split_file(self.split_file, cohort)

        manifest_path = self._split_manifest_path(mode)
        meta = {'mode': mode, 'seed': int(self.seed), 'val_ratio': float(val_ratio),
                'test_ratio': float(test_ratio), 'stratify_bins': int(stratify_bins),
                'n_subjects': len(cohort)}
        if stratify_key is not None:
            meta['stratify_key'] = str(stratify_key)
        if manifest_path is not None and manifest_path.exists() \
                and not self.refresh_split_manifest:
            try:
                old = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(
                    f'切分 manifest 损坏（{manifest_path}）：{e}。'
                    '删除该文件或传 refresh_split_manifest=True 重新生成。') from e
            if stratify_key is not None and 'stratify_key' not in old:
                # 历史 manifest 未记录分层键；其实际口径即模式默认键，显式声明后复用
                old = {**old, 'stratify_key': stratify_key}
                print(f'[split] manifest 无 stratify_key 字段，按模式默认键 '
                      f'"{stratify_key}" 解释（与历史切分口径一致）。')
            mismatch = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
            if mismatch:
                raise RuntimeError(
                    f'切分 manifest 配置与当前运行不一致（{manifest_path}）：{mismatch}。'
                    '确认数据或配置变更后传 refresh_split_manifest=True 重新生成；'
                    '若非有意变更，请回退当前运行的 seed/比例设置以复用固定切分。')
            cur = cohort
            train, val, test = old.get('train'), old.get('val'), old.get('test')
            if train and val and test and (set(train) | set(val) | set(test)) == cur:
                print(f'[split] 复用版本化切分 manifest: {manifest_path}')
                return sorted(train), sorted(val), sorted(test)
            raise RuntimeError(
                f'切分 manifest 与当前被试集合不一致（{manifest_path}），'
                '数据目录可能已变化。删除该文件或传 refresh_split_manifest=True 重新生成。')

        train, val, test = self._split_subject_ids_impl(mode, val_ratio, test_ratio,
                                                        stratify_bins, stratify_key)
        if manifest_path is not None:
            manifest_path.write_text(
                json.dumps({**meta, 'train': train, 'val': val, 'test': test},
                           ensure_ascii=False, indent=2),
                encoding='utf-8')
            print(f'[split] 版本化切分已写入: {manifest_path} '
                  f'(train/val/test = {len(train)}/{len(val)}/{len(test)})')
        return train, val, test

    @staticmethod
    def _load_split_file(path: str, cohort: set) -> Tuple[List[str], List[str], List[str]]:
        """读取并校验外部被试划分文件（互斥、重复、覆盖、队列成员）。

        校验失败一律报错：划分是实验对照的地基，静默容忍会让不同臂用不同队列，
        从而使「条件 vs 无条件」的比较失去意义。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'--split_file 指定的划分文件不存在: {path}')
        payload = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f'--split_file 需为含 train/val/test 键的 JSON 对象，收到 {type(payload)}')
        parts = {}
        for key in ('train', 'val', 'test'):
            v = payload.get(key, None)
            if not isinstance(v, (list, tuple)) or not v:
                raise ValueError(f'--split_file 缺少非空列表键 "{key}"（{path}）')
            parts[key] = [str(s).strip() for s in v]
        tr, va, te = parts['train'], parts['val'], parts['test']
        pairs = {'train∩val': set(tr) & set(va), 'train∩test': set(tr) & set(te),
                 'val∩test': set(va) & set(te)}
        overlap = {k: sorted(v)[:5] for k, v in pairs.items() if v}
        if overlap:
            raise ValueError(f'--split_file 集合不互斥（{path}）：{overlap}')
        dup = {k: sorted({s for s in v if v.count(s) > 1})[:5]
               for k, v in parts.items()}
        dup = {k: v for k, v in dup.items() if v}
        if dup:
            raise ValueError(f'--split_file 存在重复被试（{path}）：{dup}')
        union = set(tr) | set(va) | set(te)
        unknown = sorted(union - cohort)[:5]
        if unknown:
            raise ValueError(
                f'--split_file 含当前影像队列之外的被试（{path}），例如 {unknown}。')
        uncovered = sorted(cohort - union)
        if uncovered:
            raise ValueError(
                f'--split_file 未覆盖影像队列中的 {len(uncovered)} 个被试（{path}），'
                f'例如 {uncovered[:5]}。所有对照臂必须共用同一完整队列；'
                f'请补齐该划分文件或按预先固定规则重新生成完整队列划分。')
        print(f'[split] 复用外部划分文件（已校验互斥/无重复/全覆盖）: {path} '
              f'| train/val/test = {len(tr)}/{len(va)}/{len(te)}')
        return sorted(tr), sorted(va), sorted(te)

    def _split_manifest_path(self, mode: str):
        root = getattr(self.base_dataset, 'data_root', None)
        if not root:
            return None
        return Path(str(root)) / f'subject_split_{mode}.json'

    def _split_subject_ids_impl(self, mode: str, val_ratio: float, test_ratio: float,
                                stratify_bins: int,
                                stratify_key: Optional[str] = None
                                ) -> Tuple[List[str], List[str], List[str]]:
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

        if mode != 'finetune' or stratify_key is None:
            # 非分层模式：直接按比例划分
            test_subjects = shuffled_subjects[:target_test_subjects]
            val_subjects = shuffled_subjects[target_test_subjects:target_test_subjects + target_val_subjects]
            train_subjects = shuffled_subjects[target_test_subjects + target_val_subjects:]
            return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

        # 分层键**显式指定**（MDD 保持 HAMD），不随条件 fields[0] 变化
        values = [self.base_dataset.stratify_value_by_subject(sid, stratify_key)
                  for sid in subject_ids]
        missing = [sid for sid, v in zip(subject_ids, values) if v is None]
        score_map = {sid: float(v) for sid, v in zip(subject_ids, values) if v is not None}
        score_values = sorted(score_map.values())
        unique_scores = len(set(score_values))
        if unique_scores <= 1:
            test_subjects = shuffled_subjects[:target_test_subjects]
            val_subjects = shuffled_subjects[target_test_subjects:target_test_subjects + target_val_subjects]
            train_subjects = shuffled_subjects[target_test_subjects + target_val_subjects:]
            return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

        # 分层划分（分层键缺失者进预注册的「缺失桶」并按确定性顺序分配，不删人）
        n_bins = max(2, min(stratify_bins, unique_scores, total_subjects))
        score_series = pd.Series(score_values)
        bin_edges = pd.qcut(score_series, q=n_bins, labels=False, duplicates='drop')
        bins_sorted = [int(b) for b in bin_edges.tolist()]
        sid_sorted = [sid for sid, _ in sorted(score_map.items(),
                                               key=lambda kv: float(kv[1]))]
        sid_bin = dict(zip(sid_sorted, bins_sorted))
        MISSING_BIN = -1
        for sid in missing:
            sid_bin[sid] = MISSING_BIN
        if missing:
            print(f"[split] 分层键 '{stratify_key}' 缺失 {len(missing)} 人 → 进入缺失桶 "
                  f"({MISSING_BIN})，按确定性顺序分配，**不因人**（预注册规则）。")

        bin_to_subjects: Dict[int, List[str]] = defaultdict(list)
        for sid in subject_ids:
            bin_to_subjects[sid_bin[sid]].append(sid)

        # 每个桶内部按 ID 排序后再用同一 seed 打乱：与 dict 迭代顺序无关，可复现
        for bid in bin_to_subjects:
            bin_to_subjects[bid] = sorted(bin_to_subjects[bid])
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

    def _split_indices_by_subject(self, mode: str, val_ratio: float, test_ratio: float,
                                  stratify_bins: int,
                                  stratify_key: Optional[str] = None):
        train_subjects, val_subjects, test_subjects = self._split_subject_ids(
            mode=mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_bins=stratify_bins,
            stratify_key=stratify_key,
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
        if self.base_dataset.condition_enabled:
            # 条件通路使用 ConditionStats（train-only）；旧接口显式拒绝，
            # 避免两条条件链路在同一实验里同时生效
            raise RuntimeError(
                '条件注入已开启（condition_group），旧 get_train_pathology_scores '
                '接口不适用；请使用 get_condition_stats() / condition_matrix_for()。')
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

    # ------------------------------------------------------------------
    # 条件注入访问器（train-only 统计量 / merged 审计 / 逐被试矩阵）
    # ------------------------------------------------------------------
    def get_condition_stats(self):
        """返回 train-only 拟合的 :class:`ConditionStats`（未开启条件时为 None）。"""
        return self.condition_stats

    def get_condition_meta(self) -> dict:
        """checkpoint 需要持久化的条件协议元数据（哈希、来源列、审计）。"""
        ds = self.base_dataset
        if not ds.condition_enabled:
            return {'condition_enabled': False}
        from utils.condition_schema import layout_hash
        table = ds.condition_table
        return {
            'condition_enabled': True,
            'condition_group': ds.condition_group,
            'condition_file': table.file_name,
            'merged_sha256': table.merged_sha256,
            'fields': [s.name for s in table.specs],
            'source_map': table.audit.alias_sources,
            # 结构哈希（字段/顺序/类型/注册来源/词表）；解析到的实际列与文件哈希
            # 属于溯源信息，单独记录并独立校验，避免两处哈希口径不一致
            'layout_hash': layout_hash(table.specs, stats=self.condition_stats),
            'resolved_source_map': table.audit.alias_sources,
            'condition_audit': table.audit.to_dict(),
            'n_missing_row_subjects': len(table.missing_row_subjects),
            'split_file': self.split_file,
            'stratify_key': self.stratify_key,
        }

    def condition_matrix_for(self, subject_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        """逐被试条件矩阵（z/code/mask/raw/subject_index），供评估与置换诊断用。"""
        return self.base_dataset.condition_matrix_for(subject_ids)
