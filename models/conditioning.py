"""Clinical field encoders, demo-plus-cat fusion, missingness handling and deterministic Condition Dropout."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from utils.condition_schema import (
    DROPOUT_SEED,
    ConditionStats,
    FieldSpec,
    categorical_indices,
    continuous_indices,
    field_names,
    group_indices,
    layout_hash,
)

ARMS = ("none", "demo", "cat")
FUSIONS = ("cat",)
DELTA_STYLES = ("mlp",)


@dataclass
class ConditioningConfig:
    """Structure and dropout settings for hc3/demo and mdd8/cat field encoding."""

    group: str = "hc3"
    arm: str = "none"
    d_demo: int = 16
    d_new: int = 16
    cond_dim: int = 32
    h_delta: int = 64
    delta_style: str = "mlp"
    dropout_p: float = 0.0
    dropout_seed: int = DROPOUT_SEED
    extra: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm 必须是 {ARMS}，收到 '{self.arm}'")
        if self.delta_style not in DELTA_STYLES:
            raise ValueError(
                f"delta_style 必须是 {DELTA_STYLES}，收到 '{self.delta_style}'"
            )
        for name in ("d_demo", "d_new", "cond_dim", "h_delta"):
            v = int(getattr(self, name))
            if v < 1:
                raise ValueError(f"{name} 必须 >= 1，收到 {v}")
            setattr(self, name, v)
        p = float(self.dropout_p)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout_p 必须在 [0, 1) 内，收到 {p}")
        self.dropout_p = p

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict) -> "ConditioningConfig":
        known = {
            k: v
            for k, v in payload.items()
            if k
            in (
                "group",
                "arm",
                "d_demo",
                "d_new",
                "cond_dim",
                "h_delta",
                "delta_style",
                "dropout_p",
                "dropout_seed",
            )
        }
        extra = {k: v for k, v in payload.items() if k not in known}
        return ConditioningConfig(**known, extra=extra)


_U53 = 1 << 53
_MASK53 = _U53 - 1
_SM_C1 = -4658895280553007687
_SM_C2 = -7723592293110705685


def mix64(state: torch.Tensor) -> torch.Tensor:
    """确定性的 64 位整数混合（splitmix64 的有符号 int64 等价写法）。

    选用整数混合而非 ``hash()``：Python 的字符串 hash 带进程级随机盐，跨进程/
    跨 epoch 不可复现，无法满足「恢复训练只要还原 (seed, epoch)」的要求。
    """
    x = state.long()
    x = (x ^ x >> 30) * _SM_C1
    x = x ^ x >> 27
    x = x * _SM_C2
    x = x ^ x >> 31
    return x


def deterministic_uniform(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """由整数键产生的确定性 :math:`U[0,1)`（同一 key 恒得同一值，跨设备一致）。

    键由 ``(condition_seed, epoch, subject_index, field_index)`` 组合而成，因此
    同一被试在同一 epoch 的所有窗口共享同一遮蔽决策，下一 epoch 自动改变；
    恢复训练只要还原 ``(seed, epoch)`` 即可完全复现。
    """
    h = mix64(keys.long() ^ int(seed) & _MASK53)
    u = (h & _MASK53).to(torch.float64) / float(_U53)
    return u.to(torch.float32)


def dropout_keys(
    subject_index: torch.Tensor, k: int, epoch: int, device: torch.device
) -> torch.Tensor:
    """``[B]`` 被试下标 → ``[B, K]`` 混合键（epoch/subject/field 位域唯一编码）。

    布局：高 32 位为 epoch，中间 24 位为被试下标，低 8 位为字段下标；因此
    ``epoch < 2^{32}``、``subject < 2^{24}``（约 1677 万，远大于本项目队列）、
    ``k < 256`` 时三元组唯一，不会互相串扰。
    """
    subj = subject_index.long().view(-1, 1)
    lo = int(subj.min()) if subj.numel() else 0
    hi = int(subj.max()) if subj.numel() else 0
    if hi >= 1 << 24 or lo < 0:
        raise ValueError(
            f"被试下标超出键布局上限（需要 0 <= idx < {1 << 24}，收到 [{lo}, {hi}]）"
        )
    if k >= 1 << 8:
        raise ValueError(f"字段数 K={k} 超出键布局上限 {1 << 8}")
    ep = int(epoch)
    if not 0 <= ep < 1 << 32:
        raise ValueError(f"epoch={ep} 超出键布局上限 {1 << 32}")
    fld = torch.arange(k, dtype=torch.long, device=device).view(1, k)
    return ((ep << 32) + (subj << 8) + fld).expand(subj.shape[0], k)


def condition_dropout(
    z: torch.Tensor,
    codes: torch.Tensor,
    mask: torch.Tensor,
    subject_index: Optional[torch.Tensor],
    epoch: Optional[int],
    p: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """采样 :math:`r_{ij}` 并返回 :math:`(z^{eff}, q^{eff}, m^{eff})`。

    被屏蔽字段：连续 → :math:`z=0, m=0`（即训练填补中心，不泄露原值）；类别 →
    该字段独立 Unknown（索引 0）且 :math:`m=0`。原本缺失的字段保持缺失。
    ``epoch=None`` 或 ``p<=0`` 时恒等返回（推理态）。
    """
    if p <= 0.0 or epoch is None or subject_index is None:
        return (z, codes, mask)
    b, k = mask.shape
    keys = dropout_keys(subject_index.to(z.device), k, epoch, z.device)
    keep = deterministic_uniform(keys, seed).to(z.device) >= p
    m_eff = mask * keep.to(mask.dtype)
    z_eff = torch.where(keep, z, torch.zeros_like(z))
    q_eff = torch.where(keep, codes, torch.zeros_like(codes))
    return (z_eff, q_eff, m_eff)


class ContinuousFieldEncoder(nn.Module):
    """连续字段的独立编码 :math:`F_j([z,m])`：``Linear(2,d) -> GELU -> Linear(d,d)``。

    参数量 :math:`d^2 + 4d`（见 :meth:`n_params_formula`）。
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = int(d)
        self.fc1 = nn.Linear(2, self.d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.d, self.d)

    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """z [B] 或 [B,1]，m 同形 → e [B, d]。"""
        h = torch.stack([z.reshape(-1), m.reshape(-1)], dim=-1).to(
            self.fc1.weight.dtype
        )
        return self.fc2(self.act(self.fc1(h)))

    @staticmethod
    def n_params_formula(d: int) -> int:
        return int(d) ** 2 + 4 * int(d)


class CategoricalFieldEncoder(nn.Module):
    """类别字段的独立 Embedding：索引 ``0`` = 该字段**专属** Unknown。

    词表大小 ``vocab + 1``（含 Unknown），参数量 :math:`(V+1)\\cdot d= n_{cat}\\cdot d`。
    有效 ``code_0`` 与 Unknown 是不同行（``utils.condition_schema`` 保证索引分配）。
    """

    def __init__(self, num_embeddings: int, d: int):
        super().__init__()
        if num_embeddings < 1:
            raise ValueError(
                f"Embedding 行数必须 >= 1（至少 Unknown），收到 {num_embeddings}"
            )
        self.num_embeddings = int(num_embeddings)
        self.d = int(d)
        self.emb = nn.Embedding(self.num_embeddings, self.d)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        flat = idx.reshape(-1).long()
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.num_embeddings):
            raise ValueError(
                f"类别索引越界：[{int(flat.min())}, {int(flat.max())}] 不在 [0, {self.num_embeddings - 1}]（0=Unknown）。有效类别未登记时必须显式更新类别协议，禁止运行时扩表。"
            )
        return self.emb(flat)


class FeatureEncoder(nn.Module):
    """给定字段子集的**独立**编码器集合（调用方按臂只传入真正消费的字段）。

    连续字段按所属组取宽度（``d_demo`` / ``d_new``），类别字段同理；输出
    ``e`` 的逐字段切片由构造顺序（``specs`` 顺序）给出，避免各处猜列序。
    入参 ``z/codes/mask`` 的列数必须与 ``specs`` 长度一致（本类只看见自己负责的列）。
    """

    def __init__(
        self, specs: Sequence[FieldSpec], stats: ConditionStats, widths: Dict[str, int]
    ):
        super().__init__()
        self.specs = list(specs)
        self.stats = stats
        self.widths = dict(widths)
        self.cont_idx = continuous_indices(self.specs)
        self.cat_idx = categorical_indices(self.specs)
        self.continuous = nn.ModuleDict()
        self.categorical = nn.ModuleDict()
        for j, spec in enumerate(self.specs):
            d = int(self.widths[spec.name])
            if spec.is_continuous:
                self.continuous[spec.name] = ContinuousFieldEncoder(d)
            else:
                n_rows = stats.get(spec.name).vocab_size() + 1
                self.categorical[spec.name] = CategoricalFieldEncoder(n_rows, d)

    def dim_of(self, name: str) -> int:
        return int(self.widths[name])

    def forward(
        self, z: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """→ 逐字段表示列表 ``[e_0, ..., e_{K-1}]``，每个 ``[B, d_j]``。"""
        if z.ndim != 2 or z.shape != mask.shape or codes.shape != mask.shape:
            raise ValueError(
                f"条件输入形状不一致：z={tuple(z.shape)} codes={tuple(codes.shape)} mask={tuple(mask.shape)}（要求同形 [B, K]）"
            )
        if z.shape[1] != len(self.specs):
            raise ValueError(
                f"字段宽度 K={z.shape[1]} 与 schema 的 {len(self.specs)} 不一致"
            )
        outs: List[torch.Tensor] = []
        for j, spec in enumerate(self.specs):
            if spec.is_continuous:
                outs.append(self.continuous[spec.name](z[:, j], mask[:, j]))
            else:
                outs.append(self.categorical[spec.name](codes[:, j]))
        return outs


def pool_phi(emb: List[torch.Tensor], fusion: str) -> torch.Tensor:
    """Concatenate the independent new-field embeddings; cat is the supported fusion."""
    if fusion == "cat":
        return torch.cat(emb, dim=-1)
    raise ValueError(f"fusion 必须是 {FUSIONS}，收到 '{fusion}'")


class DemoEncoder(nn.Module):
    """HC 共享三字段编码 :math:`E_{demo}([e_{Sex},e_{Age},e_{education},m_{demo}])`。

    ``Linear -> GELU -> Linear`` 的两层 MLP，输出宽度 C；继承自 HC checkpoint 时
    **不重新清零**。
    """

    is_conditioning_adapter = True

    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.cond_dim = int(cond_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.cond_dim),
            nn.GELU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

    def forward(self, feats: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        h = torch.cat(feats + [mask], dim=-1)
        if h.shape[-1] != self.in_dim:
            raise ValueError(
                f"E_demo 输入宽度 {h.shape[-1]} 与登记的 {self.in_dim} 不一致"
            )
        return self.net(h)


class DeltaEncoder(nn.Module):
    """Zero-initialized MLP increment from demo, concatenated new fields and masks."""

    is_conditioning_adapter = True

    def __init__(self, in_dim: int, cond_dim: int, hidden: int, style: str = "mlp"):
        super().__init__()
        if style not in DELTA_STYLES:
            raise ValueError(f"style 必须是 {DELTA_STYLES}，收到 '{style}'")
        self.in_dim, self.cond_dim, self.hidden, self.style = (
            int(in_dim),
            int(cond_dim),
            int(hidden),
            str(style),
        )
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.cond_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.out = self.net[-1]

    def forward(
        self, c_demo: torch.Tensor, phi: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat([c_demo, phi, mask], dim=-1)
        if h.shape[-1] != self.in_dim:
            raise ValueError(
                f"DeltaE 输入宽度 {h.shape[-1]} 与登记的 {self.in_dim} 不一致（融合算子或字段组发生变化？）"
            )
        return self.net(h)

    def n_params_formula(self) -> int:
        """Number of parameters in the two-layer increment MLP."""
        input_dim, hidden_dim, output_dim = self.in_dim, self.hidden, self.cond_dim
        return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim


class ConditionEncoder(nn.Module):
    """按 :class:`ConditioningConfig` 组装的条件编码链路。

    ``mdd8`` 分组下：先算 :math:`c_{demo}`（继承 HC 的 ``E_demo`` 与共享编码器），
    再按 ``arm`` 计算新增组增量；``demo`` 臂只用三字段，``none`` 臂返回 ``None``
    （表示绕过条件通路，与「全字段缺失」不同）。
    """

    is_conditioning_adapter = True

    def __init__(
        self,
        specs: Sequence[FieldSpec],
        stats: ConditionStats,
        cfg: ConditioningConfig,
        demo_group: Sequence[str],
        new_group: Sequence[str],
    ):
        super().__init__()
        self.cfg = cfg
        self.specs = list(specs)
        self.stats = stats
        self.names = field_names(self.specs)
        self.demo_group = list(demo_group)
        self.new_group = list(new_group)
        self.demo_idx = group_indices(self.specs, self.demo_group)
        self.new_idx = (
            group_indices(self.specs, self.new_group) if self.new_group else []
        )
        for g in self.new_group:
            if g not in self.names:
                raise ValueError(f"新增组字段 {g} 不在 schema {self.names} 中")
        widths = {}
        for name in self.names:
            widths[name] = cfg.d_demo if name in self.demo_group else cfg.d_new
        self.feat_names = self._used_field_names(cfg.arm)
        self.feat_idx = group_indices(self.specs, self.feat_names)
        self._feat_pos = {j: i for i, j in enumerate(self.feat_idx)}
        feat_specs = [self.specs[j] for j in self.feat_idx]
        self.features = FeatureEncoder(
            feat_specs, stats, {n: widths[n] for n in self.feat_names}
        )
        d_demo_sum = sum((widths[n] for n in self.demo_group))
        self.demo_in_dim = d_demo_sum + len(self.demo_group)
        self.demo = DemoEncoder(self.demo_in_dim, cfg.cond_dim)
        self.phi: Optional[str] = None
        self.delta: Optional[DeltaEncoder] = None
        self.delta_in_dim = 0
        if cfg.group == "mdd8" and cfg.arm in FUSIONS:
            self.phi = cfg.arm
            n_new = len(self.new_group)
            phi_w = n_new * cfg.d_new
            self.delta_in_dim = cfg.cond_dim + phi_w + n_new
            self.delta = DeltaEncoder(
                self.delta_in_dim, cfg.cond_dim, cfg.h_delta, cfg.delta_style
            )
            self._assert_delta_formulas(phi_w)

    def _used_field_names(self, arm: str) -> List[str]:
        """本臂真正消费的字段（决定 :class:`FeatureEncoder` 建哪些参数）。"""
        if arm in FUSIONS:
            used = list(self.demo_group) + [
                n for n in self.new_group if n not in self.demo_group
            ]
        else:
            used = list(self.demo_group)
        missing = [n for n in used if n not in self.names]
        if missing:
            raise ValueError(f"字段 {missing} 不在 schema {self.names} 中")
        return [n for n in self.names if n in used]

    def _assert_delta_formulas(self, phi_w: int) -> None:
        """按实际字段宽度和类别数统计条件模块参数量。"""
        n_new = len(self.new_group)
        C, H = (self.cfg.cond_dim, self.cfg.h_delta)
        expect_in = C + phi_w + n_new
        if self.delta_in_dim != expect_in:
            raise AssertionError(
                f"DeltaE 输入宽度 {self.delta_in_dim} != 公式值 {expect_in}"
            )
        if self.cfg.delta_style == "mlp":
            expect = expect_in * H + H + H * C + C
        else:
            expect = expect_in * C + C
        actual = sum((p.numel() for p in self.delta.parameters()))
        if actual != expect:
            raise AssertionError(
                f"DeltaE 参数量 {actual} != 公式值 {expect}（style={self.cfg.delta_style}）"
            )

    @property
    def enabled(self) -> bool:
        return self.cfg.arm != "none"

    def condition_dim(self) -> int:
        return int(self.cfg.cond_dim)

    def layout(self) -> dict:
        """结构自描述（写入 checkpoint，用于 strict 重建与语义冲突拦截）。"""
        return {
            "group": self.cfg.group,
            "arm": self.cfg.arm,
            "fields": self.names,
            "demo_group": self.demo_group,
            "new_group": self.new_group,
            "d_demo": self.cfg.d_demo,
            "d_new": self.cfg.d_new,
            "cond_dim": self.cfg.cond_dim,
            "h_delta": self.cfg.h_delta,
            "delta_style": self.cfg.delta_style,
            "demo_in_dim": self.demo_in_dim,
            "delta_in_dim": self.delta_in_dim,
            "vocab": {
                n: self.stats.get(n).vocab_size()
                for n in self.names
                if not self.specs[self.names.index(n)].is_continuous
            },
            "sources": {s.name: list(s.sources) for s in self.specs},
            "layout_hash": layout_hash(self.specs, stats=self.stats),
            "demo_layout_hash": layout_hash(
                [s for s in self.specs if s.name in self.demo_group], stats=self.stats
            ),
        }

    def param_report(self) -> Dict[str, int]:
        """实际参数量登记。

        只统计**本臂真正建立**的字段编码（:attr:`feat_names`）；报告口径必须与
        ``sum(p.numel() for p in self.parameters())`` 一致，否则参数预算（rho）
        会把未消费的张量算进去。
        """
        cont_names = [
            n for n in self.feat_names if self.specs[self.names.index(n)].is_continuous
        ]
        cat_names = [n for n in self.feat_names if n not in cont_names]
        feat_cont = sum(
            (
                ContinuousFieldEncoder.n_params_formula(self.features.dim_of(n))
                for n in cont_names
            )
        )
        feat_cat = sum(
            (
                self.features.categorical[n].num_embeddings * self.features.dim_of(n)
                for n in cat_names
            )
        )
        demo = sum((p.numel() for p in self.demo.parameters()))
        delta = sum((p.numel() for p in self.delta.parameters())) if self.delta else 0
        report = {
            "feature_continuous": int(feat_cont),
            "feature_categorical": int(feat_cat),
            "demo_encoder": int(demo),
            "delta_encoder": int(delta),
            "total": int(feat_cont + feat_cat + demo + delta),
            "encoded_fields": len(self.feat_names),
        }
        actual = int(sum((p.numel() for p in self.parameters())))
        if report["total"] != actual:
            raise AssertionError(
                f"条件参数量报告 {report['total']} 与实际构造 {actual} 不一致（arm={self.cfg.arm}）；登记口径不得包含未建立的字段编码。"
            )
        return report

    def forward(
        self,
        z: torch.Tensor,
        codes: torch.Tensor,
        mask: torch.Tensor,
        subject_index: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """→ 条件向量 :math:`c` ``[B, C]``；``arm='none'`` 时返回 ``None``（绕过）。

        Condition Dropout 只在训练且给出 ``(subject_index, epoch)`` 时生效，推理
        （``epoch=None`` 或 ``self.training=False``）恒按真实缺失编码。
        """
        if not self.enabled:
            return None
        if self.cfg.dropout_p > 0.0 and self.training:
            z, codes, mask = condition_dropout(
                z,
                codes,
                mask,
                subject_index,
                epoch,
                self.cfg.dropout_p,
                self.cfg.dropout_seed,
            )
        mask = mask.to(z.dtype)
        emb = self.features(
            z[:, self.feat_idx], codes[:, self.feat_idx], mask[:, self.feat_idx]
        )
        c_demo = self.demo(
            [emb[self._feat_pos[j]] for j in self.demo_idx], mask[:, self.demo_idx]
        )
        if self.delta is None:
            return c_demo
        by_new = [emb[self._feat_pos[j]] for j in self.new_idx]
        d_new = self.cfg.d_new
        for name, e in zip(self.new_group, by_new):
            if e.shape[-1] != d_new:
                raise AssertionError(
                    f"新增字段 {name} 表示宽度 {e.shape[-1]} != d_new={d_new}（cat 的拼接宽度与 DeltaE 参数公式都按 |N|·d_new 登记）"
                )
        phi = pool_phi(by_new, self.cfg.arm)
        delta = self.delta(c_demo, phi, mask[:, self.new_idx])
        return c_demo + delta


def build_condition_encoder(
    specs: Sequence[FieldSpec], stats: ConditionStats, cfg: ConditioningConfig
) -> ConditionEncoder:
    """按分组选择共享三字段 / 新增五字段的划分并构造编码器。

    ``hc3`` 只有共享三字段（无增量）；``mdd8`` 同时携带 ``E_demo`` 与新增组
    ``DeltaE``。``arm='demo'`` 表示只用继承的三字段（人口学基线），``arm='none'``
    表示绕过条件通路。``mdd8`` 的选定臂是 ``cat``；``hc3`` 没有新增字段，因此只可能
    是 ``none`` 或 ``demo``（下面这条校验就是结构性保证，不给「HC 也用了融合臂」留通道）。
    """
    from utils.condition_schema import DEMO_GROUP, MDD_NEW_GROUP

    if cfg.group == "hc3":
        demo = [n for n in DEMO_GROUP if n in field_names(specs)]
        new: List[str] = []
    elif cfg.group == "mdd8":
        demo, new = (list(DEMO_GROUP), list(MDD_NEW_GROUP))
    else:
        raise ValueError(f"group 必须是 hc3/mdd8，收到 '{cfg.group}'")
    if cfg.group == "hc3" and cfg.arm in FUSIONS:
        raise ValueError(f"hc3 分组没有新增字段，arm='{cfg.arm}' 需要 mdd8 分组")
    return ConditionEncoder(specs, stats, cfg, demo_group=demo, new_group=new)
