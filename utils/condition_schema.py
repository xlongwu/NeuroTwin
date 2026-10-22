# coding=utf-8
"""条件注入的字段 schema、清洗、类别协议、缺失判定与布局哈希。

本模块是「条件数据来源」的唯一事实来源：字段白名单、merged 文件内的列映射、
数值缺失判定、``episode``/``medication`` 的编码替换、类别词表、有效性 mask
语义、连续/类别分组切片与 ``layout_hash`` 全部在这里生成，其他模块只消费
本模块的输出，不允许各自猜列序。

协议要点（docs/Version3_docs_1003/NeuroTwin_条件注入实验方案.md）：

- 条件数据**只**读取 merged 文件（``Rest-meta-MDD-V1V2-Merged-{HC,MDD}.xlsx``），
  禁止回读单独的 V1/V2 源表补齐、覆盖或恢复编码；
- 所有可解析的有限数值都是有效值（含 ``0``、``-1``、``-9999``），有效性由
  :math:`m_{ij}=\\mathbf{1}[\\text{可解析为有限数值}]` 决定，不按临床范围、
  正负或当前类别词表把数值改判为缺失；
- ``episode``/``medication`` 的 :math:`T(x)`（:math:`T(-1)=2`、
  :math:`T(-9999)=0`）只是**编码替换**，不改变有效性，且对已更正数据幂等；
- ``HAMA`` 只允许精确名称为 ``HAMA`` 的总分列，``HAMATotal`` 与任何条目列
  都不参与（模糊匹配不会命中它们，因为匹配要求规范化后全等）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 条件协议版本；语义变化时必须递增，checkpoint 记录该值以拒绝旧 mask 语义。
SCHEMA_VERSION = 'condition_v1_20261008'

# ══════════════════════════════════════════════════════════════════════════
#  随机来源登记处（方案 §四：split seed / model seed / sampling seed /
#  Condition Dropout seed 两两分离）
#
#  **种子本身不是对比工具**：``main.py`` 的 ``--sampling_seed`` /
#  ``--cond_dropout_seed`` 与 ``models/conditioning.py`` 的 ``dropout_seed`` 默认值
#  都要引用同一个登记值。留在别处就会有两份字面量，而两份字面量必然漂移（历史上
#  ``--seed`` 与 ``--sampling_seed`` 的默认值都写成 2024，两条随机流因此并成一条，
#  ``main.fresh_study_gate`` 会直接拒绝这种同值）。划分 seed 仍冻结在
#  ``data/subject_split_*.json`` 的 ``seed`` 字段里，训练期只读名单不产生随机流。
# ══════════════════════════════════════════════════════════════════════════

#: 采样/分桶随机种子（``utils/dataloader.py`` 的 sampler 流）。
SAMPLING_SEED = 20261009

#: Condition Dropout 种子（受试者级、每 epoch 固定；与模型 seed 和采样 seed 分离）。
DROPOUT_SEED = 20261008

#: 模型初始化 seed 的登记值（``main.py --seed``）；必须与上面两个都不同。
MODEL_SEED = 2024

#: 文本缺失标记（先 strip + casefold 后比较）。
MISSING_TEXTS = frozenset({'', 'null', 'none', 'nan', 'na', 'n/a', '<na>', 'nat'})

#: 编码替换表：仅这两个字段执行，其他字段数值原样保留。
CODE_REMAP_FIELDS = ('episode', 'medication')
CODE_REMAP: Dict[float, float] = {-1.0: 2.0, -9999.0: 0.0}

#: 检测协议关注的哨兵数值（只审计与替换，不当作缺失）。
SENTINEL_VALUES = (-9999.0, -1.0)

CONTINUOUS = 'continuous'
CATEGORICAL = 'categorical'


@dataclass(frozen=True)
class FieldSpec:
    """单个条件字段的规范定义。

    Args:
        name: 规范字段名（模型与文档使用的唯一标识）。
        kind: ``continuous`` 或 ``categorical``。
        sources: merged 文件内允许的来源列名（按注册顺序解析）。长度 >1 表示
            同一行内在这些列上取可解析数值并要求一致（冲突时报错），
            不引入「来自哪一列」的身份特征。
    """

    name: str
    kind: str
    sources: Tuple[str, ...]

    @property
    def is_continuous(self) -> bool:
        return self.kind == CONTINUOUS


#: HC 条件字段白名单（K_HC=3）。
HC_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec('Sex', CATEGORICAL, ('Sex',)),
    FieldSpec('Age', CONTINUOUS, ('Age',)),
    FieldSpec('education', CONTINUOUS, ('education',)),
)

#: MDD 条件字段白名单（K_MDD=8），前三项与 HC 完全一致（共享前缀）。
MDD_FIELDS: Tuple[FieldSpec, ...] = HC_FIELDS + (
    FieldSpec('episode', CATEGORICAL, ('episode',)),
    FieldSpec('illness', CONTINUOUS, ('illness',)),
    # merged 表在 2026-10-08 已提供精确名称 `medication`（原 `On medication?`
    # 列的规范名），`OnMedication` 为同表内的互补列；三个来源都显式登记，
    # 同行人冲突时报错，不静默切换来源。
    FieldSpec('medication', CATEGORICAL, ('medication', 'On medication?', 'OnMedication')),
    FieldSpec('HAMD', CONTINUOUS, ('HAMD',)),
    FieldSpec('HAMA', CONTINUOUS, ('HAMA',)),
)

#: 条件分组：HC 共享三字段 / MDD 新增五字段。
DEMO_GROUP = ('Sex', 'Age', 'education')
MDD_NEW_GROUP = ('episode', 'illness', 'medication', 'HAMD', 'HAMA')

GROUP_FIELDS: Dict[str, Tuple[FieldSpec, ...]] = {
    'hc3': HC_FIELDS,
    'mdd8': MDD_FIELDS,
}


def schema_for(group: str) -> Tuple[FieldSpec, ...]:
    """按分组名返回字段规格（``hc3`` / ``mdd8``）。"""
    if group not in GROUP_FIELDS:
        raise ValueError(f"未知条件分组 '{group}'，可选 {sorted(GROUP_FIELDS)}")
    return GROUP_FIELDS[group]


def field_names(specs: Sequence[FieldSpec]) -> List[str]:
    return [f.name for f in specs]


def continuous_indices(specs: Sequence[FieldSpec]) -> List[int]:
    return [i for i, f in enumerate(specs) if f.is_continuous]


def categorical_indices(specs: Sequence[FieldSpec]) -> List[int]:
    return [i for i, f in enumerate(specs) if not f.is_continuous]


def group_indices(specs: Sequence[FieldSpec], group: Sequence[str]) -> List[int]:
    """按名称列表取字段下标；名称未注册时立即报错（不静默漏字段）。"""
    names = field_names(specs)
    missing = [g for g in group if g not in names]
    if missing:
        raise ValueError(f'字段 {missing} 不在 schema 中（当前 {names}）')
    return [names.index(g) for g in group]


# ══════════════════════════════════════════════════════════════════════════
#  单元格解析
# ══════════════════════════════════════════════════════════════════════════

def parse_cell(value) -> Tuple[Optional[float], str]:
    """把 merged 单元格解析为 ``(有限数值 or None, 原因)``。

    规则（v1 revised §1.1）：只有真正的空值/文本缺失标记/不可解析内容/非有限
    数值才是缺失；``0``、``-1``、``-9999`` 与数值字符串都是有效数值。禁止用
    真值判断（``if not value``）判空，否则有效 0 会被误删。

    Returns:
        ``(float, reason)`` 表示有效数值； ``(None, reason)`` 表示缺失，
        reason ∈ {empty, na_type, missing_text, unparseable_text, non_numeric,
        non_finite, bool_as_number, numeric, numeric_string}。
    """
    if value is None:
        return None, 'empty'
    if isinstance(value, (bool, np.bool_)):
        return float(int(value)), 'bool_as_number'
    if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None, 'na_type'
    if isinstance(value, float):
        return (value, 'numeric') if np.isfinite(value) else (None, 'non_finite')
    if isinstance(value, np.floating):
        f = float(value)
        return (f, 'numeric') if np.isfinite(f) else (None, 'non_finite')
    if isinstance(value, (int, np.integer)):
        return float(int(value)), 'numeric'
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in MISSING_TEXTS:
            return None, ('empty' if text == '' else 'missing_text')
        try:
            f = float(text)
        except ValueError:
            return None, 'unparseable_text'
        return (f, 'numeric_string') if np.isfinite(f) else (None, 'non_finite')
    # pandas NA / 其他标量：尽力转 float，失败记为不可解析
    try:
        if pd.isna(value):
            return None, 'na_type'
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return None, 'non_numeric'
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None, 'non_numeric'
    return (f, 'numeric') if np.isfinite(f) else (None, 'non_finite')


def remap_code(value: float) -> float:
    """``episode``/``medication`` 的指定编码替换 :math:`T(x)`（幂等）。"""
    return CODE_REMAP.get(float(value), float(value))


def applies_remap(name: str) -> bool:
    return name in CODE_REMAP_FIELDS


# ══════════════════════════════════════════════════════════════════════════
#  列解析
# ══════════════════════════════════════════════════════════════════════════

def _normalized_columns(columns: Sequence) -> Dict[str, List[str]]:
    """``规范化键 -> 实际列名列表``；大小写与首尾空白可匹配，多义时报错。"""
    out: Dict[str, List[str]] = {}
    for c in columns:
        key = str(c).strip().casefold()
        out.setdefault(key, []).append(str(c))
    return out


def resolve_source_columns(df: pd.DataFrame, specs: Sequence[FieldSpec]) -> Dict[str, List[str]]:
    """把每个规范字段解析为 merged 表中实际存在的来源列（保持注册顺序）。

    - 同一规范化键对应多个实际列 → 歧义，报错；
    - 字段的注册来源列在表中不存在 → 该来源跳过；全部不存在 → 报错，
      绝不回退到其他 xlsx 或其他列。
    """
    norm = _normalized_columns(df.columns)
    resolved: Dict[str, List[str]] = {}
    for spec in specs:
        matched: List[str] = []
        for src in spec.sources:
            key = src.strip().casefold()
            hits = norm.get(key, [])
            if len(hits) > 1:
                raise ValueError(
                    f'列名歧义：字段 {spec.name} 的来源 "{src}" 同时匹配 {hits}')
            if hits and hits[0] not in matched:
                matched.append(hits[0])
        if not matched:
            raise ValueError(
                f'merged 文件中找不到字段 {spec.name} 的任何注册来源列 '
                f'{list(spec.sources)}；禁止用其他文件或推断列补齐。')
        resolved[spec.name] = matched
    return resolved


# ══════════════════════════════════════════════════════════════════════════
#  装载：values / mask
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LoadAudit:
    """加载期审计（人数、哨兵值计数、缺失原因、别名冲突），供报告与日志使用。"""

    n_rows: int = 0
    sentinel_counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    missing_reasons: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_valid: Dict[str, int] = field(default_factory=dict)
    alias_sources: Dict[str, List[str]] = field(default_factory=dict)
    alias_overlap: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'n_rows': self.n_rows,
            'n_valid': dict(self.n_valid),
            'sentinel_counts': json.loads(json.dumps(self.sentinel_counts)),
            'missing_reasons': json.loads(json.dumps(self.missing_reasons)),
            'alias_sources': json.loads(json.dumps(self.alias_sources)),
            'alias_overlap': dict(self.alias_overlap),
            'warnings': list(self.warnings),
        }


class MedicationConflict(RuntimeError):
    """同一行多个来源列映射后取值不一致：显式报错并停止该配置。"""


def load_merged_conditions(path, specs: Sequence[FieldSpec],
                           id_column: str = 'ID') -> Tuple[np.ndarray, np.ndarray, List[str], LoadAudit]:
    """读取 merged 表 → ``(values, mask, subject_ids, audit)``。

    Args:
        path: merged xlsx 路径（唯一允许的条件数据来源）。
        specs: 字段规格（决定列映射与替换规则）。
        id_column: 被试匹配列（只用于匹配，不进条件向量）。

    Returns:
        values: ``[N, K]`` float32。有效数值为**替换后**的类别码或原始连续值；
            缺失格为 ``nan``（由下游按训练统计量填补，本函数不做任何填补）。
        mask: ``[N, K]`` uint8 有效性掩码 :math:`m_{ij}`。
        subject_ids: 与 values 行对齐的被试 ID 列表（去空白）。
    """
    df = pd.read_excel(path)
    if id_column not in df.columns:
        raise ValueError(f'{path} 缺少 ID 列（现有列：{list(df.columns)[:12]}...）')
    specs = list(specs)
    sources = resolve_source_columns(df, specs)
    audit = LoadAudit(n_rows=int(len(df)))
    audit.alias_sources = {k: list(v) for k, v in sources.items()}

    subject_ids = df[id_column].astype(str).str.strip().tolist()
    if len(set(subject_ids)) != len(subject_ids):
        dup = [s for s in set(subject_ids) if subject_ids.count(s) > 1]
        raise ValueError(f'merged 表存在重复 ID（{dup[:5]}...），无法唯一匹配条件')

    n, k = len(df), len(specs)
    values = np.full((n, k), np.nan, dtype=np.float32)
    mask = np.zeros((n, k), dtype=np.uint8)

    for j, spec in enumerate(specs):
        cols = sources[spec.name]
        parsed = [df[c].tolist() for c in cols]
        reasons: Dict[str, int] = {}
        # 哨兵计数以 float 为键（-1 与 -1.0 是同一个键），输出时再转字符串
        sentinel_before: Dict[float, int] = {v: 0 for v in SENTINEL_VALUES}
        sentinel_after: Dict[float, int] = {v: 0 for v in SENTINEL_VALUES}
        overlap = 0
        row_conflicts: List[Tuple[str, Dict[str, float]]] = []

        for i in range(n):
            cand: Dict[str, float] = {}
            for col, cells in zip(cols, parsed):
                val, reason = parse_cell(cells[i])
                if val is not None:
                    cand[col] = val
            if not cand:
                # 记录最后一个缺失原因（单来源列时即该列的原因）
                _, reason0 = parse_cell(parsed[0][i])
                reasons[reason0] = reasons.get(reason0, 0) + 1
                continue
            if len(cand) > 1:
                overlap += 1
            mapped = {c: (remap_code(v) if applies_remap(spec.name) else v)
                      for c, v in cand.items()}
            if len(set(mapped.values())) > 1:
                row_conflicts.append((subject_ids[i], mapped))
                continue
            value = float(next(iter(mapped.values())))
            values[i, j] = np.float32(value)
            mask[i, j] = 1
            if applies_remap(spec.name):
                raw = float(next(iter(cand.values())))
                if raw in sentinel_before:
                    sentinel_before[raw] += 1
                if value in sentinel_after:
                    # 已更正数据（原本就是 2/0）也计数，用于证明 T 的幂等性
                    sentinel_after[value] += 1

        if row_conflicts:
            sample = '; '.join(f'{sid}: {d}' for sid, d in row_conflicts[:5])
            raise MedicationConflict(
                f'字段 {spec.name} 在 {len(row_conflicts)} 行内多个来源列取值冲突'
                f'（映射后仍不一致），已停止该配置：{sample}')

        audit.sentinel_counts[spec.name] = {
            'raw_sentinel_counts': {str(k): int(v) for k, v in sentinel_before.items()},
            'mapped_sentinel_counts': {str(k): int(v) for k, v in sentinel_after.items()},
            'remap_applied': bool(applies_remap(spec.name))}
        audit.missing_reasons[spec.name] = dict(reasons)
        audit.n_valid[spec.name] = int(mask[:, j].sum())
        audit.alias_overlap[spec.name] = int(overlap)

    return values, mask, subject_ids, audit


# ══════════════════════════════════════════════════════════════════════════
#  统计量与类别词表（train-only）
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FieldStats:
    """单字段拟合结果。

    Args:
        fitted: 是否已用训练有效观测拟合。
        n_valid: 参与拟合的训练有效观测数（全缺失时为 0）。
        mu: 连续字段的填补中心（训练有效观测中位数）；类别字段无意义。
        scale: 稳健尺度 :math:`\\operatorname{IQR}/1.349`，退化回退 std，再回退 1。
        scale_source: ``iqr`` / ``std`` / ``ones`` / ``none``。
        codes: 类别字段按升序排列的**已登记**类别码（不含 Unknown）。
    """

    fitted: bool = False
    n_valid: int = 0
    mu: float = 0.0
    scale: float = 1.0
    scale_source: str = 'none'
    codes: Tuple[float, ...] = ()

    def vocab_size(self) -> int:
        return len(self.codes)


class ConditionStats:
    """全 schema 的 train-only 统计量与类别协议。

    ``fit`` 按字段逐个进行，绝不「统一 fit 覆盖继承统计量」：HC→MDD 迁移时用
    :meth:`inherit` 显式复制 Age/education 的统计量与词表，再只对新增字段拟合。
    """

    def __init__(self, specs: Sequence[FieldSpec], fields: Dict[str, FieldStats]):
        self.specs = list(specs)
        self.fields = dict(fields)
        self._validate()

    def _validate(self) -> None:
        names = field_names(self.specs)
        if set(self.fields) != set(names):
            raise ValueError(
                f'统计量字段与 schema 不一致：{sorted(self.fields)} vs {names}')

    @property
    def k(self) -> int:
        return len(self.specs)

    def get(self, name: str) -> FieldStats:
        if name not in self.fields:
            raise KeyError(f'字段 {name} 无统计量记录')
        return self.fields[name]

    def continuous_mu(self) -> np.ndarray:
        mu = np.zeros(self.k, dtype=np.float64)
        for j, spec in enumerate(self.specs):
            if spec.is_continuous:
                mu[j] = self.fields[spec.name].mu
        return mu

    def continuous_scale(self) -> np.ndarray:
        scale = np.ones(self.k, dtype=np.float64)
        for j, spec in enumerate(self.specs):
            if spec.is_continuous:
                scale[j] = self.fields[spec.name].scale
        return scale

    def vocab(self) -> Dict[str, List[float]]:
        return {s.name: list(self.fields[s.name].codes)
                for s in self.specs if not s.is_continuous}

    def check_codes(self, name: str, codes: Sequence[float]) -> None:
        """有效类别码必须已登记；未登记时报错，不伪装为缺失、不静默扩表。"""
        known = set(self.fields[name].codes)
        bad = sorted({float(c) for c in codes if float(c) not in known})
        if bad:
            raise ValueError(
                f'字段 {name} 出现未登记的有效类别码 {bad[:8]}（已登记 '
                f'{sorted(known)}）。必须显式更新类别协议并重新拟合，'
                '禁止运行时扩大 Embedding。')

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_continuous(values: np.ndarray, valid: np.ndarray, eps: float = 1e-12) -> FieldStats:
        x = values[valid].astype(np.float64)
        n = int(x.size)
        if n == 0:
            # 全训练缺失：显式记录不可拟合，不从 val/test 取值
            return FieldStats(fitted=False, n_valid=0, mu=0.0, scale=1.0,
                              scale_source='none')
        mu = float(np.median(x))
        q1, q3 = np.percentile(x, [25.0, 75.0])
        iqr = float(q3 - q1)
        scale, source = iqr / 1.349, 'iqr'
        if not np.isfinite(scale) or scale <= eps:
            std = float(np.std(x))
            scale, source = std, 'std'
        if not np.isfinite(scale) or scale <= eps:
            scale, source = 1.0, 'ones'
        return FieldStats(fitted=True, n_valid=n, mu=mu, scale=scale,
                          scale_source=source)

    @staticmethod
    def _fit_categorical(values: np.ndarray, valid: np.ndarray,
                         extra: Sequence[float] = ()) -> FieldStats:
        x = values[valid].astype(np.float64)
        codes = tuple(sorted({float(v) for v in x} | {float(e) for e in extra}))
        return FieldStats(fitted=True, n_valid=int(x.size), codes=codes)

    def fit(self, values: np.ndarray, mask: np.ndarray, subject_rows: np.ndarray,
            only_fields: Optional[Sequence[str]] = None,
            extra_codes: Optional[Dict[str, Sequence[float]]] = None) -> 'ConditionStats':
        """只用 ``subject_rows`` 指定的训练被试有效观测拟合指定字段。

        Args:
            values: ``[N, K]``（缺失格为 nan）。
            mask: ``[N, K]`` uint8。
            subject_rows: 训练集行下标。
            only_fields: 需要（重新）拟合的字段名；None=全部。其余字段保持原值。
            extra_codes: **显式预注册**的额外有效类别码（按字段名）。用于 merged
                中确实存在、但恰好未落在训练集的词表；词表因此在拟合期一次性
                冻结，不在运行时静默扩大 Embedding。观测计数 ``n_valid`` 仍只统计
                训练观测。
        """
        names = field_names(self.specs)
        targets = list(names if only_fields is None else only_fields)
        unknown = [t for t in targets if t not in names]
        if unknown:
            raise ValueError(f'待拟合字段不在 schema 中：{unknown}')
        rows = np.asarray(subject_rows, dtype=np.int64)
        if rows.size == 0:
            raise ValueError('训练被试为空，无法拟合 train-only 统计量。')
        extra_codes = extra_codes or {}
        stray = sorted(set(extra_codes) - set(names))
        if stray:
            raise ValueError(f'--condition_extra_codes 含未知字段：{stray}')
        for j, spec in enumerate(self.specs):
            if spec.name not in targets:
                continue
            v, m = values[rows, j], mask[rows, j].astype(bool)
            finite = np.isfinite(np.where(m, v, np.nan))
            m = m & finite
            if spec.is_continuous:
                self.fields[spec.name] = self._fit_continuous(v, m)
            else:
                self.fields[spec.name] = self._fit_categorical(
                    v, m, extra_codes.get(spec.name, ()))
        return self

    def inherit(self, other: 'ConditionStats', fields: Sequence[str]) -> 'ConditionStats':
        """从另一份统计量按字段名复制（含 mu/scale/词表），不触碰其他字段。"""
        if field_names(self.specs) != field_names(other.specs):
            raise ValueError('继承要求两份统计量的 schema 字段一致')
        for name in fields:
            self.fields[name] = other.get(name)
        return self

    def take_fields(self, other: 'ConditionStats', names: Sequence[str],
                    require_fitted: bool = True) -> 'ConditionStats':
        """从**较小 schema**（如 HC 的 hc3）按名字取字段进来（HC→MDD 继承原语）。

        与 :meth:`inherit` 的区别：这里允许 ``other`` 的字段集合是 ``self`` 的子集，
        但要求被取的字段确实已拟合，避免把 HC 里没拟合成功的字段悄悄带进 MDD。
        未被取的字段保持原状（通常随后在 MDD-train 上拟合）。
        """
        available = set(other.fields)
        missing = [n for n in names if n not in available]
        if missing:
            raise ValueError(
                f'继承要求字段 {missing} 存在于来源统计量（来源只有 {sorted(available)}）')
        for name in names:
            st = other.get(name)
            if require_fitted and not st.fitted:
                raise ValueError(
                    f'继承字段 {name} 在来源中未拟合（n_valid={st.n_valid}）；'
                    f'不能把来源的全缺失状态当作 HC-train 统计量继承过来。')
            self.fields[name] = st
        return self

    # ------------------------------------------------------------------
    def encode(self, values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """→ ``(z [N,K] float32, emb_idx [N,K] int64)``。

        连续：先按训练填补中心安全填补，再 :math:`z=(\\tilde x-\\mu)/s`；缺失格
        恒为 :math:`z=0, m=0`，真实中位数则为 :math:`z=0, m=1`。实现上先写回
        填补值再运算，避免 ``0 * NaN``。

        类别：内部索引 ``0`` = Unknown，有效码映射为 ``1..V``；Unknown 与有效
        ``code_0`` 是不同索引。缺失格 emb_idx=0，但保留 mask=0。
        """
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask).astype(np.uint8)
        if values.shape != mask.shape:
            raise ValueError(f'values/mask 形状不一致：{values.shape} vs {mask.shape}')
        if values.shape[1] != self.k:
            raise ValueError(f'字段宽度不一致：期望 K={self.k}，收到 {values.shape[1]}')

        z = np.zeros(values.shape, dtype=np.float32)
        emb = np.zeros(values.shape, dtype=np.int64)
        for j, spec in enumerate(self.specs):
            m = mask[:, j].astype(bool)
            if spec.is_continuous:
                st = self.fields[spec.name]
                filled = np.where(m, np.nan_to_num(values[:, j], nan=st.mu), st.mu)
                if not st.fitted:
                    # 全训练缺失分支：统一落到填补中心（z=0），有效性仍由 mask 表达
                    z[:, j] = np.zeros(values.shape[0], dtype=np.float32)
                    continue
                z[:, j] = ((filled - st.mu) / st.scale).astype(np.float32)
                z[~m, j] = 0.0
            else:
                lookup = {c: i + 1 for i, c in enumerate(self.fields[spec.name].codes)}
                idx = np.zeros(values.shape[0], dtype=np.int64)
                if m.any():
                    codes = values[m, j]
                    self.check_codes(spec.name, codes)
                    idx[m] = np.array([lookup[float(c)] for c in codes], dtype=np.int64)
                emb[:, j] = idx
        if not np.isfinite(z).all():
            raise ValueError('编码后连续特征存在非有限值（应由缺失判定或填补协议拦截）')
        return z, emb

    def condition_vector(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """旧「数值条件」通路使用的规范化原始值（连续 z，类别保留替换后码）。

        保持与历史 ``PathologyNormalizer`` 相同的消费方式，仅用于兼容入口。
        """
        z, _ = self.encode(values, mask)
        out = z.copy()
        for j, spec in enumerate(self.specs):
            if not spec.is_continuous:
                m = mask[:, j].astype(bool)
                col = np.zeros(values.shape[0], dtype=np.float32)
                col[m] = np.nan_to_num(values[m, j], nan=0.0).astype(np.float32)
                out[:, j] = col
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'schema_version': SCHEMA_VERSION,
            'fields': [
                {'name': s.name, 'kind': s.kind, 'sources': list(s.sources),
                 **_asdict(self.fields[s.name])}
                for s in self.specs],
        }

    @staticmethod
    def from_dict(specs: Sequence[FieldSpec], payload: dict) -> 'ConditionStats':
        if payload.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(
                f'统计量协议版本不一致：checkpoint={payload.get("schema_version")}，'
                f'当前代码={SCHEMA_VERSION}')
        fields: Dict[str, FieldStats] = {}
        for item in payload['fields']:
            fields[item['name']] = FieldStats(
                fitted=bool(item['fitted']), n_valid=int(item['n_valid']),
                mu=float(item['mu']), scale=float(item['scale']),
                scale_source=str(item['scale_source']),
                codes=tuple(float(c) for c in item['codes']))
        return ConditionStats(specs, fields)


def _asdict(st: FieldStats) -> dict:
    return {'fitted': bool(st.fitted), 'n_valid': int(st.n_valid),
            'mu': float(st.mu), 'scale': float(st.scale),
            'scale_source': st.scale_source, 'codes': [float(c) for c in st.codes]}


def fit_condition_stats(specs: Sequence[FieldSpec], values: np.ndarray, mask: np.ndarray,
                        train_rows: np.ndarray,
                        only_fields: Optional[Sequence[str]] = None,
                        base: Optional[ConditionStats] = None,
                        extra_codes: Optional[Dict[str, Sequence[float]]] = None
                        ) -> ConditionStats:
    """构造（或以 ``base`` 为起点）并拟合 train-only 统计量。"""
    stats = ConditionStats(specs, {s.name: FieldStats() for s in specs})
    if base is not None:
        stats = ConditionStats(specs, dict(base.fields))
    return stats.fit(values, mask, train_rows, only_fields=only_fields,
                     extra_codes=extra_codes)


def audit_unseen_codes(specs: Sequence[FieldSpec], stats: ConditionStats,
                       values: np.ndarray, mask: np.ndarray,
                       train_rows: np.ndarray) -> Dict[str, List[float]]:
    """找出**训练集之外**存在、但不在已冻结词表里的有效类别码。

    协议要求这类码显式登记而不是伪装成缺失；本函数只报告，不修改词表，由调用方
    通过 ``extra_codes`` 显式确认后才进入 Embedding 表。
    """
    names = field_names(specs)
    rows = set(int(r) for r in np.asarray(train_rows, dtype=np.int64).tolist())
    out: Dict[str, List[float]] = {}
    for j, spec in enumerate(specs):
        if spec.is_continuous:
            continue
        known = set(stats.get(spec.name).codes)
        seen: set = set()
        for i in range(values.shape[0]):
            if i in rows or not mask[i, j]:
                continue
            c = float(values[i, j])
            if c not in known:
                seen.add(c)
        if seen:
            out[names[j]] = sorted(seen)
    return out


def load_inherited_stats(payload: dict, target_specs: Sequence[FieldSpec],
                         share_names: Sequence[str]) -> 'ConditionStats':
    """把来源快照（通常是 HC 的 hc3 统计量）装入目标 schema 的共享字段。

    HC→MDD 的继承原语：结果是一个**未完全拟合**的 mdd8 统计量对象，共享三字段
    来自 HC-train，新增五字段留空等待 MDD-train 拟合（由
    :meth:`ConditionStats.fit` 的 ``only_fields`` 完成）。字段顺序按
    ``target_specs``，来源顺序不同也按名字对齐；名字缺失立即报错。
    """
    src_specs = [FieldSpec(item['name'], item['kind'], tuple(item['sources']))
                 for item in payload.get('fields', [])]
    if [s.name for s in src_specs] != field_names(src_specs):
        raise ValueError('来源统计量字段解析失败')
    source = ConditionStats.from_dict(src_specs, payload)
    target = ConditionStats(target_specs, {s.name: FieldStats() for s in target_specs})
    return target.take_fields(source, share_names)


def parse_extra_codes(text) -> Dict[str, List[float]]:
    """解析 ``Sex:0,3;medication:2`` 形式的显式类别登记串。"""
    if not text:
        return {}
    if isinstance(text, dict):
        return {str(k): [float(c) for c in v] for k, v in text.items()}
    out: Dict[str, List[float]] = {}
    for part in str(text).split(';'):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f'--condition_extra_codes 片段缺少冒号：{part!r}')
        key, vals = part.split(':', 1)
        codes = [float(v) for v in vals.replace(',', ' ').split() if v.strip()]
        if not codes:
            raise ValueError(f'--condition_extra_codes 字段 {key} 未给出任何类别码')
        out.setdefault(key.strip(), []).extend(codes)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  布局哈希
# ══════════════════════════════════════════════════════════════════════════

def layout_hash(specs: Sequence[FieldSpec],
                source_map: Optional[Dict[str, List[str]]] = None,
                stats: Optional[ConditionStats] = None) -> str:
    """条件向量布局的规范哈希（**只含结构**）。

    字段**顺序**、名称、类型、来源列映射与类别词表都参与哈希，因此「同宽度但
    字段顺序变化」也会得到不同哈希，加载时必须被拒绝（验收 K）。

    注意：连续字段的 ``mu``/``scale``/``scale_source``/``n_valid``/``fitted``
    **不**在这个哈希里 —— 它只回答「布局是否同一代」。HC→MDD 的「共享三字段统计量
    逐位继承」必须同时比 :func:`layout_hash`（或 :func:`subset_layout_hash`）与
    :func:`shared_stats_hash`（数值），缺任何一侧都会放过「字段顺序与词表没变、
    但在 MDD-train 上偷偷重拟合了 Age/education」的 checkpoint。
    """
    payload = {
        'schema_version': SCHEMA_VERSION,
        'fields': [{'name': s.name, 'kind': s.kind, 'sources': list(s.sources)}
                   for s in specs],
    }
    if source_map:
        payload['source_map'] = {k: sorted(v) for k, v in sorted(source_map.items())}
    if stats is not None:
        # 词表只取**本 specs 覆盖的字段**：否则 hc3 子集哈希会被 mdd8 的新增字段
        # 词表污染，HC→MDD 的共享前缀比较失去意义。
        names = set(field_names(specs))
        payload['vocab'] = {k: [float(c) for c in v]
                            for k, v in sorted(stats.vocab().items()) if k in names}
        payload['continuous'] = [s.name for s in specs if s.is_continuous]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def subset_layout_hash(specs: Sequence[FieldSpec], stats: ConditionStats,
                       names: Sequence[str]) -> str:
    """共享子集（如 HC 的 Sex/Age/education）的布局哈希。

    HC→MDD 迁移时两侧的完整 ``layout_hash`` 天然不同（K=3 vs K=8），但**共享前缀
    必须逐位一致**：字段顺序、类型、注册来源列与类别词表都参与哈希，因此 HC 的
    共享三字段哈希应等于 MDD 侧的子集哈希，否则说明继承被改写。
    """
    by_name = {s.name: s for s in specs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f'子集字段 {missing} 不在 schema 中')
    sub = [by_name[n] for n in names]
    return layout_hash(sub, stats=stats)


# ══════════════════════════════════════════════════════════════════════════
#  train-only 统计量的精确值指纹（layout_hash 的数值补集）
# ══════════════════════════════════════════════════════════════════════════

def field_fingerprint(st: FieldStats) -> Tuple[Any, ...]:
    """单字段统计量的**精确值**指纹：float 走 ``hex()``，不做任何容差比较。

    与 :func:`layout_hash` 互补：这里只装「数值」——``fitted`` / ``n_valid`` /
    ``mu`` / ``scale`` / ``scale_source`` / 已登记类别码。「最后一位 ULP 不同」也
    算不同，因为继承协议要求**逐位**保持；容差比较会让「在 MDD-train 上重新拟合
    过 Age」的统计量看起来和 HC 一样。

    与 :func:`experiments.condition_cohort._field_fingerprint` 同口径（两处必须
    保持一致，否则同一份统计量在不同闸门前会有不同的相等判定）。
    """
    return (bool(st.fitted), int(st.n_valid), float(st.mu).hex(), float(st.scale).hex(),
            str(st.scale_source), tuple(float(c).hex() for c in st.codes))


def shared_stats_hash(specs: Sequence[FieldSpec], stats: ConditionStats,
                      names: Optional[Sequence[str]] = None) -> str:
    """共享字段**数值统计量**的规范哈希 —— :func:`layout_hash` 的数值补集。

    哈希对象是 ``{schema_version, 按给定顺序逐字段的 (name, kind, field_fingerprint)}``
    的规范 JSON（``sort_keys=True``）之 sha256 前 16 位，因此：

    * **顺序敏感**：``names=['Age','education']`` 与 ``names=['education','Age']``
      必然得到不同哈希；
    * **数值精确**：``mu``/``scale``/类别码用 ``float.hex()``，``n_valid`` 用整数、
      ``fitted`` 用布尔、``scale_source`` 用字符串 —— 任何一位 ULP 的差异都会改哈希。

    与 :func:`layout_hash`（**结构**：字段名/顺序/类型/来源列 + 类别词表）服务的是
    两个不同问题：布局哈希回答「向量列语义是否同一代」，本哈希回答「train-only 的
    填补中心/尺度/观测数是否还是那一组数」。HC→MDD 的逐位继承**必须同时比这两侧**：
    只比 layout 哈希会放行「词表和列序没动、但 Age/education 已在 MDD-train 上重拟合」
    的 checkpoint；只比本哈希则会放行「列序被置换」的 checkpoint。
    """
    by_name = {s.name: s for s in specs}
    if names is None:
        sel = list(specs)
    else:
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(
                f'子集字段 {missing} 不在 schema 中（当前 {field_names(specs)}）')
        sel = [by_name[n] for n in names]
    payload = {
        'schema_version': SCHEMA_VERSION,
        'fields': [{'name': s.name, 'kind': s.kind,
                    'fingerprint': list(field_fingerprint(stats.get(s.name)))}
                   for s in sel],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def file_sha256(path) -> str:
    """merged 文件快照哈希（写入 checkpoint，用于溯源）。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
