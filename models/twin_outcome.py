# coding=utf-8
"""独立临床结局读出：HAMD/HAMA 随访变化 MLP 头。

一个患者/访视对是一个样本；83 维特征全部来自**基线**信息与基线模拟
（profile 32 + 临床编码 c 32 + 动态描述量摘要 6 + 健康偏离摘要 6 +
有/无控制差摘要 3 + 控制摘要 3 + log1p(followup_days) 1），不输入真实随访
BOLD 或随访临床条件。MLP 固定 ``83→64→4``、GELU、无 Dropout：前两列为
standardized delta 均值，后两列为 logvar（clamp [-6,4]）。

本模块只包含头、损失与标准化统计量；特征提取（冻结动态模型 + 参照）与训练
循环在 :mod:`train.twin_outcome`。未拟合的读出不输出疗效评分。
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

#: 83 维特征的固定分段（顺序即拼接顺序，训练/推理两侧共用）
FEATURE_BLOCKS = (
    ('profile', 32),          # 基线校准码 p（前 32 TR，固定 profile）
    ('condition', 32),        # 基线八字段临床编码 c
    ('descriptor', 6),        # 基线 64 TR 描述量逐 ROI 均值
    ('reference', 6),         # 健康参照残差逐 ROI 绝对值均值
    ('control_diff', 3),      # 有/无控制早期 H=8 差轨迹摘要
    ('control', 3),           # 有效控制摘要
    ('followup_days', 1),     # log1p(followup_days)
)
FEATURE_DIM = int(sum(d for _, d in FEATURE_BLOCKS))       # 83
OUTCOME_COMPONENTS = ('HAMD', 'HAMA')                      # 目标分量固定顺序
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0


class ClinicalOutcomeHead(nn.Module):
    """83 → 64 → 4 MLP（GELU，无 Dropout）。

    输入为**标准化后**的 83 维特征（统计量由 outcome 训练对拟合，
    :class:`OutcomeStandardizer`）；输出前两列为 standardized delta 均值，
    后两列为 logvar（clamp 到 [-6,4]）。
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 64,
                 n_components: int = 2):
        super().__init__()
        if in_dim != FEATURE_DIM:
            raise ValueError(
                f'首版 outcome 特征固定 {FEATURE_DIM} 维（C=32、profile_dim=32），'
                f'收到 in_dim={in_dim}；不为其他宽度自动补零或截断。')
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_components = int(n_components)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_components),
        )

    def forward(self, x: torch.Tensor):
        """标准化特征 ``[B,83]`` → ``(delta_mean [B,2], logvar [B,2])``。"""
        out = self.net(x)
        delta_mean = out[:, :self.n_components]
        logvar = out[:, self.n_components:].clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
        return delta_mean, logvar


def masked_gaussian_nll(delta_mean: torch.Tensor, logvar: torch.Tensor,
                        target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """ masked Gaussian NLL::

        L = Σ_{n,j} m_nj [ ½ exp(-ℓ_nj)(δ̂_nj-δ_nj)² + ½ ℓ_nj ] / max(1, Σ m)

    delta_mean/logvar/target: [B,2]（standardized delta 空间）；mask: [B,2]。
    缺 HAMA 不丢弃有效 HAMD（mask 逐分量）。
    """
    m = mask.to(delta_mean.dtype)
    nll = 0.5 * torch.exp(-logvar) * (delta_mean - target) ** 2 + 0.5 * logvar
    return (nll * m).sum() / m.sum().clamp_min(1.0)


class OutcomeStandardizer:
    """83 维特征与 delta 目标的标准化统计量（仅从 outcome 训练对拟合）。

    特征尺度下限 1e-6；统计量随读出 checkpoint 保存，推理侧逐位复用。
    """

    def __init__(self, feat_mean: np.ndarray, feat_scale: np.ndarray,
                 delta_mean: np.ndarray, delta_scale: np.ndarray):
        self.feat_mean = np.asarray(feat_mean, dtype=np.float64)
        self.feat_scale = np.asarray(feat_scale, dtype=np.float64)
        self.delta_mean = np.asarray(delta_mean, dtype=np.float64)
        self.delta_scale = np.asarray(delta_scale, dtype=np.float64)
        if self.feat_mean.shape != (FEATURE_DIM,) or self.feat_scale.shape != (FEATURE_DIM,):
            raise ValueError(f'特征统计量形状应为 ({FEATURE_DIM},)')
        if self.delta_mean.shape != (2,) or self.delta_scale.shape != (2,):
            raise ValueError('delta 统计量形状应为 (2,)')

    @classmethod
    def fit(cls, feats: np.ndarray, deltas: np.ndarray,
            delta_mask: np.ndarray) -> 'OutcomeStandardizer':
        """从训练对拟合；某分量无有效训练标签时直接报错（不伪造统计量）。"""
        feats = np.asarray(feats, dtype=np.float64)
        mu = feats.mean(axis=0)
        sd = feats.std(axis=0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        d_mu = np.zeros(2)
        d_sd = np.ones(2)
        for j in range(2):
            valid = delta_mask[:, j] > 0
            if not valid.any():
                raise ValueError(
                    f'{OUTCOME_COMPONENTS[j]} 在训练集中完全没有有效标签'
                    '（基线与随访须同时存在）；必需的统计量不存在，'
                    '不输出随机模型。')
            vals = deltas[valid, j]
            d_mu[j] = vals.mean()
            sd_j = vals.std()
            d_sd[j] = sd_j if sd_j > 1e-6 else 1.0
        return cls(mu, sd, d_mu, d_sd)

    def transform_features(self, feats: np.ndarray) -> np.ndarray:
        f = (np.asarray(feats, dtype=np.float64) - self.feat_mean) / self.feat_scale
        return f.astype(np.float32)

    def transform_delta(self, delta: np.ndarray) -> np.ndarray:
        return ((np.asarray(delta, dtype=np.float64) - self.delta_mean)
                / self.delta_scale).astype(np.float32)

    def to_delta_mean_orig(self, delta_hat_std: np.ndarray) -> np.ndarray:
        """standardized delta 均值 → 原分数尺度。"""
        return (np.asarray(delta_hat_std) * self.delta_scale + self.delta_mean)

    def to_delta_std_orig(self, logvar: np.ndarray) -> np.ndarray:
        """logvar（standardized 空间）→ 原分数尺度 std。"""
        return np.exp(0.5 * np.asarray(logvar)) * self.delta_scale

    def to_dict(self) -> dict:
        return {'feat_mean': self.feat_mean.tolist(),
                'feat_scale': self.feat_scale.tolist(),
                'delta_mean': self.delta_mean.tolist(),
                'delta_scale': self.delta_scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> 'OutcomeStandardizer':
        return cls(np.asarray(payload['feat_mean'], dtype=np.float64),
                   np.asarray(payload['feat_scale'], dtype=np.float64),
                   np.asarray(payload['delta_mean'], dtype=np.float64),
                   np.asarray(payload['delta_scale'], dtype=np.float64))


def assemble_features(profile: np.ndarray, condition: np.ndarray,
                      descriptor_summary: np.ndarray,
                      reference_summary: np.ndarray,
                      control_diff_summary: np.ndarray,
                      control_summary: np.ndarray,
                      followup_days: float) -> np.ndarray:
    """按 FEATURE_BLOCKS 固定顺序拼出 83 维特征（训练/推理唯一拼接入口）。"""
    parts = [np.asarray(profile, dtype=np.float64).reshape(-1),
             np.asarray(condition, dtype=np.float64).reshape(-1),
             np.asarray(descriptor_summary, dtype=np.float64).reshape(-1),
             np.asarray(reference_summary, dtype=np.float64).reshape(-1),
             np.asarray(control_diff_summary, dtype=np.float64).reshape(-1),
             np.asarray(control_summary, dtype=np.float64).reshape(-1),
             np.array([np.log1p(float(followup_days))], dtype=np.float64)]
    for name, width, part in zip([b[0] for b in FEATURE_BLOCKS],
                                 [b[1] for b in FEATURE_BLOCKS], parts):
        if part.shape[0] != width:
            raise ValueError(f'特征块 {name} 期望 {width} 维，收到 {part.shape[0]}')
    feat = np.concatenate(parts, axis=0)
    if feat.shape[0] != FEATURE_DIM or not np.isfinite(feat).all():
        raise ValueError(f'特征拼接结果非法：shape={feat.shape}，'
                         f'finite={np.isfinite(feat).all()}')
    return feat


def predict_with_baseline(head: ClinicalOutcomeHead, feats_std: torch.Tensor,
                          baseline_scores: Optional[np.ndarray] = None,
                          standardizer: Optional[OutcomeStandardizer] = None):
    """推理包装：标准化特征 → 原分数尺度 delta_mean/delta_std [B,2]。

    baseline_scores: [B,2] 基线原始评分（HAMD/HAMA，缺失为 nan）；提供时返回
    ``followup_mean = baseline + delta_mean``（仅对有基线值的分量有效）。
    """
    head.eval()
    with torch.no_grad():
        delta_hat_std, logvar = head(feats_std)
    d_mean = standardizer.to_delta_mean_orig(delta_hat_std.cpu().numpy())
    d_std = standardizer.to_delta_std_orig(logvar.cpu().numpy())
    out = {'delta_mean': d_mean, 'delta_std': d_std}
    if baseline_scores is not None:
        base = np.asarray(baseline_scores, dtype=np.float64)
        followup = base + d_mean
        followup[np.isnan(base)] = np.nan
        out['followup_mean'] = followup
    return out
