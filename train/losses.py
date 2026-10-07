# coding=utf-8
"""TFM 双目标损失与 SC 软先验图正则。"""
import torch
import torch.nn as nn


def pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """逐 ROI 的 Pearson 相关系数，返回 [B, F]。

    pred/target 允许 [B, F, ...] 任意尾部形状，内部按最后一维之外展平。

    数值：分母的两个 L2 范数用 ``sqrt(sum(x²) + eps)`` 而非 ``sqrt(sum(x²))``。
    后者在常数序列（方差恰为 0）处局部导数无穷，即使该行的 incoming 梯度为 0，
    ``0 * inf`` 也会把整张反向图污染成 NaN（实测：mask 掉某个 (样本, horizon)
    后梯度全变 NaN）。加 eps 后 ``√`` 的入参恒 ≥ eps > 0，非退化行的数值不变。
    """
    b, f = pred.shape[0], pred.shape[1]
    pf = pred.reshape(b, f, -1)
    tf = target.reshape(b, f, -1)
    pc = pf - pf.mean(-1, keepdim=True)
    tc = tf - tf.mean(-1, keepdim=True)
    n_p = ((pc ** 2).sum(-1) + eps).sqrt()
    n_t = ((tc ** 2).sum(-1) + eps).sqrt()
    return (pc * tc).sum(-1) / (n_p * n_t + eps)


def spatial_pcc(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """逐 (样本, 偏移) 的**空间** PCC：在 ROI 维比较两种全脑 pattern。

    pred/target: [B, F, H, 1]。返回 [B, H]。

    与沿时间轴定义的 temporal PCC 语义不同：next-timepoint 的单步目标
    ``x_(t+δ) ∈ R^F`` 只有一个时间点，沿 ROI 维的相关才对应“预测的空间
    pattern 是否与真实一致”（时序轨迹口径在 rollout 指标里单独计算）。
    """
    p = pred[..., 0]
    t = target[..., 0]
    pc = p - p.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    n_p = ((pc ** 2).sum(dim=1) + eps).sqrt()
    n_t = ((tc ** 2).sum(dim=1) + eps).sqrt()
    return (pc * tc).sum(dim=1) / (n_p * n_t + eps)


def compute_graph_regularization(aux_info, device,
                                 sparsity_weight: float = 1e-3,
                                 entropy_weight: float = 1e-3,
                                 temporal_weight: float = 0.0):
    """SC 软先验的图正则，返回 (total_reg, stats)。

    total_reg = 稀疏(L1, 去掉对角) × sparsity_weight
              + 行熵（抑制模糊的均匀图） × entropy_weight
              + 时间一致性（逐窗功能图一阶差分平方） × temporal_weight

    从 aux_info['sc_prior']（SoftAnatomicalPrior 输出字典）读取：
      - 'A_eff' [B,F,F]：可微软先验（带梯度 → 正则可回传到 λ / U,V / ΔA / 掩码）
      - 'A_seq' [B,W,F,F]：仅当 --sc_temporal_weight > 0 时才生成

    未启用 SC 软先验（--sc_prior_mode scaled）或三项权重全为 0 时返回零正则，
    不影响原有训练行为。
    """
    zero = torch.zeros(1, device=device).squeeze(0)
    zero_stats = {'graph_sparsity': zero.detach(),
                  'graph_entropy': zero.detach(),
                  'graph_temporal': zero.detach()}

    if max(sparsity_weight, entropy_weight, temporal_weight) <= 0.0:
        return zero, zero_stats

    sc_prior = aux_info.get('sc_prior', None) if isinstance(aux_info, dict) else None
    if not isinstance(sc_prior, dict):
        return zero, zero_stats
    a_eff = sc_prior.get('A_eff', None)
    if not torch.is_tensor(a_eff) or a_eff.ndim != 3:
        return zero, zero_stats

    # 去掉对角线：稀疏与熵只约束 ROI 之间的连接
    off_diag = a_eff - torch.diag_embed(a_eff.diagonal(dim1=-2, dim2=-1))
    sparsity = off_diag.abs().mean()

    p_row = off_diag.clamp_min(1e-8)
    p_row = p_row / p_row.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(p_row * p_row.log()).sum(dim=-1).mean()

    temporal = zero
    a_seq = sc_prior.get('A_seq', None)
    if temporal_weight > 0 and torch.is_tensor(a_seq) and a_seq.ndim == 4 and a_seq.shape[1] > 1:
        temporal = (a_seq[:, 1:] - a_seq[:, :-1]).pow(2).mean()

    total_reg = (sparsity_weight * sparsity
                 + entropy_weight * entropy
                 + temporal_weight * temporal)
    stats = {
        'graph_sparsity': sparsity.detach(),
        'graph_entropy':  entropy.detach(),
        'graph_temporal': temporal.detach() if torch.is_tensor(temporal) else zero.detach(),
    }
    return total_reg, stats


# ══════════════════════════════════════════════════════════════════════════
#  NeuroTwin-TFM 双目标损失（方案 §19 / §20）
# ══════════════════════════════════════════════════════════════════════════


def _huber_elementwise(pred: torch.Tensor, target: torch.Tensor,
                       delta: float) -> torch.Tensor:
    """逐元素 Huber：|e|≤δ 取 0.5e²，否则 δ(|e|−0.5δ)（与 F.huber_loss 同式）。"""
    e = (pred - target).abs()
    return torch.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta))


class TFMDualLoss(nn.Module):
    """One-Step + CPM 双目标损失（方案 §9 Dual Dynamics Objective / §20）。

    数学定义（pred/target 均为原始空间；``x_t`` 为 context 末位 TR）::

        L_one = Huber(x̂_(t+1), x_(t+1)) + λ_pcc·(1 − corr_ROI(x̂_(t+1), x_(t+1)))
        L_cpm = Σ_h w_h·Huber(x̂_(t+h), x_(t+h)) / Σ_h w_h,   w_h = γ^(h−1)·mask_h
        L     = λ_one·L_one + λ_cpm·L_cpm

    设计说明（§20.1）：不使用数学等价的 ``L_abs + L_delta`` 双计项
    （prediction_target=delta 时二者数值重合），改为单一 Huber + spatial PCC；
    CPM 项逐 horizon 加权（§20.2，γ=1 即均匀，<1 时对远期 horizon 降权）。
    λ_cpm=0 可退化为纯 one-step 的 Stage 0 基线。
    """

    def __init__(self, lambda_one: float = 1.0, lambda_cpm: float = 1.0,
                 lambda_pcc: float = 0.1, cpm_gamma: float = 1.0,
                 huber_delta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if lambda_one < 0 or lambda_cpm < 0 or lambda_pcc < 0:
            raise ValueError(
                f"lambda_one/lambda_cpm/lambda_pcc 不允许负数："
                f"{lambda_one}/{lambda_cpm}/{lambda_pcc}")
        if lambda_one == 0 and lambda_cpm == 0:
            raise ValueError("lambda_one 与 lambda_cpm 不能同时为 0。")
        if cpm_gamma <= 0 or cpm_gamma > 1.0:
            raise ValueError(
                f"cpm_gamma 应在 (0, 1]（1=均匀权重），收到 {cpm_gamma}")
        if huber_delta <= 0:
            raise ValueError(f"huber_delta 必须 > 0，收到 {huber_delta}")
        self.lambda_one = float(lambda_one)
        self.lambda_cpm = float(lambda_cpm)
        self.lambda_pcc = float(lambda_pcc)
        self.cpm_gamma = float(cpm_gamma)
        self.huber_delta = float(huber_delta)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                cpm_pred: torch.Tensor, cpm_target: torch.Tensor,
                cpm_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            pred/target: [B, F, 1, 1] one-step 预测与真值（原始空间）
            cpm_pred:    [B, F, H]   CPM 全 horizon 预测（原始空间）
            cpm_target:  [B, F, H]   t+1..t+H 真值（原始空间，序列末尾截断补零）
            cpm_mask:    [B, H]      各 horizon 是否有真值（None = 全有效）
        Returns:
            (total, stats dict)
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"one-step pred/target 形状不一致：{tuple(pred.shape)} vs "
                f"{tuple(target.shape)}")
        if pred.ndim != 4 or pred.shape[-1] != 1 or pred.shape[2] != 1:
            raise ValueError(
                f"one-step pred 期望 [B,F,1,1]（TFM 的 one-step 头只建模 +1 偏移），"
                f"收到 {tuple(pred.shape)}")
        if cpm_pred.shape != cpm_target.shape:
            raise ValueError(
                f"CPM pred/target 形状不一致：{tuple(cpm_pred.shape)} vs "
                f"{tuple(cpm_target.shape)}")
        if cpm_pred.ndim != 3:
            raise ValueError(
                f"CPM pred 期望 [B,F,H]，收到 {tuple(cpm_pred.shape)}")

        stats = {}
        # ---- L_one（§20.1：Huber + spatial PCC）----
        loss_one = _huber_elementwise(pred[:, :, 0, 0], target[:, :, 0, 0],
                                      self.huber_delta).mean()
        corr_one = spatial_pcc(pred, target, self.eps).mean()      # [1] → 标量
        loss_one_pcc = 1.0 - corr_one
        total = self.lambda_one * (loss_one + self.lambda_pcc * loss_one_pcc)
        stats['loss_one'] = loss_one.detach()
        stats['loss_one_pcc'] = loss_one_pcc.detach()
        stats['pcc_one'] = corr_one.detach()

        # ---- L_CPM（§20.2：γ 加权逐 horizon Huber）----
        n_valid = cpm_pred.shape[0]
        if n_valid > 0 and self.lambda_cpm > 0:
            h = cpm_pred.shape[2]
            w = torch.ones(cpm_pred.shape[0], h, device=cpm_pred.device,
                           dtype=cpm_pred.dtype)
            if self.cpm_gamma < 1.0:
                gamma_pows = self.cpm_gamma ** torch.arange(
                    h, device=cpm_pred.device, dtype=cpm_pred.dtype)
                w = w * gamma_pows.view(1, h)
            if cpm_mask is not None:
                if tuple(cpm_mask.shape) != (cpm_pred.shape[0], h):
                    raise ValueError(
                        f"cpm_mask 形状 {tuple(cpm_mask.shape)} 与 "
                        f"[B,{h}] 不一致")
                w = w * cpm_mask.to(cpm_pred.dtype)
            # 逐元素 Huber [B,F,H] × 逐 (样本,horizon) 权重
            hub = _huber_elementwise(cpm_pred, cpm_target, self.huber_delta)
            w3 = w.view(w.shape[0], 1, h)
            loss_cpm = (hub * w3).sum() / (w3.sum().clamp_min(self.eps)
                                           * cpm_pred.shape[1])
            # 逐 horizon 明细 + spatial PCC（远期 horizon 的不确定性来源之一）
            corr_cpm = spatial_pcc(cpm_pred.unsqueeze(-1),
                                   cpm_target.unsqueeze(-1), self.eps)  # [B,H]
            for hi in range(h):
                n_h = float(w[:, hi].sum().item())
                if n_h <= 0:
                    continue
                stats[f'loss_cpm_h{hi + 1}'] = (
                    (hub[:, :, hi] * w[:, hi:hi + 1]).sum()
                    / (n_h * cpm_pred.shape[1])).detach()
                stats[f'pcc_cpm_h{hi + 1}'] = (
                    (corr_cpm[:, hi] * w[:, hi]).sum() / n_h).detach()
            loss_cpm_pcc = 1.0 - (corr_cpm * w).sum() / w.sum().clamp_min(self.eps)
            total = total + self.lambda_cpm * (loss_cpm + self.lambda_pcc * loss_cpm_pcc)
            stats['loss_cpm'] = loss_cpm.detach()
            stats['loss_cpm_pcc'] = loss_cpm_pcc.detach()
        else:
            stats['loss_cpm'] = torch.zeros((), device=pred.device)
            stats['loss_cpm_pcc'] = torch.zeros((), device=pred.device)
        stats['loss_total'] = total.detach()
        return total, stats
