# coding=utf-8
"""方案 B 的六项固定权重预测损失及共用评价函数。"""
import math
import torch
import torch.nn as nn

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


def temporal_pcc_loss(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """Mean 1-PCC over valid (sample, ROI) future trajectories [B,F,H].

    At least three observed future points and nonconstant targets are required.
    Constant predictions remain supervised. Observed context is not appended.
    Each valid ROI/trajectory has equal weight, independent of missing horizons.
    """
    if pred.ndim != 3 or pred.shape != target.shape:
        raise ValueError('temporal PCC pred/target must share [B,F,H]')
    b, _, h = pred.shape
    if mask is not None and mask.shape != (b, h):
        raise ValueError('temporal PCC mask must be [B,H]')
    # Use float32 for stable variances/norms under autocast.
    p, t = pred.float(), target.float()
    w = p.new_ones(b, 1, h) if mask is None else mask.to(p)[:, None, :]
    n = w.sum(-1)
    count = n.clamp_min(1.0)
    p_mean = (p * w).sum(-1) / count
    t_mean = (t * w).sum(-1) / count
    pc, tc = (p - p_mean[..., None]) * w, (t - t_mean[..., None]) * w
    p_ss, t_ss = pc.square().sum(-1), tc.square().sum(-1)
    valid = (n >= 3) & (t_ss / count > eps)
    denom = (p_ss + eps).sqrt() * (t_ss + eps).sqrt() + eps
    corr = ((pc * tc).sum(-1) / denom).clamp(-1.0, 1.0)
    return ((1.0 - corr) * valid).sum() / valid.sum().clamp_min(1)


def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """Masked element-wise L1 over ``[B, F, H]`` futures.

    ``mask`` is ``[B, H]`` and marks which horizons have a real target. The
    normalization is per valid element, so partial futures do not down-weight
    the remaining observations.
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'masked_l1 expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    if mask is None:
        return (pred - target).abs().mean()
    if tuple(mask.shape) != (pred.shape[0], pred.shape[2]):
        raise ValueError(
            f'masked_l1 mask must be [B,{pred.shape[2]}], got {tuple(mask.shape)}')
    w = mask.to(pred.dtype)[:, None, :]
    return ((pred - target).abs() * w).sum() / (
        w.sum().clamp_min(eps) * pred.shape[1])


def masked_l1_per_sample(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor = None,
                         eps: float = 1e-8) -> torch.Tensor:
    """Per-sample masked L1 ``[B]`` for ``[B,F,H]`` tensors."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'masked_l1_per_sample expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    err = (pred - target).abs()
    if mask is None:
        return err.mean(dim=(1, 2))
    if tuple(mask.shape) != (pred.shape[0], pred.shape[2]):
        raise ValueError(
            f'masked_l1_per_sample mask must be [B,{pred.shape[2]}], got '
            f'{tuple(mask.shape)}')
    w = mask.to(pred.dtype)[:, None, :]
    return (err * w).sum(dim=(1, 2)) / (
        w.sum(dim=(1, 2)).clamp_min(eps) * pred.shape[1])


def complete_sequence_mask(mask: torch.Tensor, horizon: int) -> torch.Tensor:
    """Samples with every one of the first ``horizon`` future steps observed."""
    if mask is None:
        raise ValueError('complete_sequence_mask requires an explicit [B,H] mask')
    if mask.ndim != 2 or mask.shape[1] < horizon:
        raise ValueError(
            f'complete_sequence_mask expects [B,H>= {horizon}], got {tuple(mask.shape)}')
    return (mask[:, :horizon] > 0).all(dim=1)


def correlation_matrix(series: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample Pearson FC matrix from ``[B, F, T]`` time series."""
    if series.ndim != 3:
        raise ValueError(f'correlation_matrix expects [B,F,T], got {tuple(series.shape)}')
    x = series.float()
    centered = x - x.mean(dim=-1, keepdim=True)
    std = (centered.square().mean(dim=-1) + eps).sqrt()
    cov = torch.bmm(centered, centered.transpose(1, 2)) / x.shape[-1]
    return cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + eps)


def fc_mse_per_sample(pred: torch.Tensor, target: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Per-sample FC-matrix MSE ``[B]``; off-diagonal edges only."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'fc_mse_per_sample expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = correlation_matrix(pred)
        t = correlation_matrix(target)
        f = pred.shape[1]
        off_diag = 1.0 - torch.eye(f, device=p.device, dtype=p.dtype)
        return ((p - t) ** 2 * off_diag).sum(dim=(1, 2)) / max(f * (f - 1), 1)


def fc_mse(pred: torch.Tensor, target: torch.Tensor,
           mask: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """Mean squared error between predicted and true Pearson FC matrices.

    Only samples with a fully observed future contribute: an FC matrix from a
    truncated window is not comparable across subjects. Off-diagonal edges are
    averaged; the diagonal is always one and carries no information.
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'fc_mse expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    if mask is None:
        valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    else:
        valid = complete_sequence_mask(mask, pred.shape[-1])
    if not bool(valid.any()):
        return pred.new_zeros(())
    return fc_mse_per_sample(pred[valid], target[valid], eps).mean()


def frequency_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 distance between orthonormal rFFT magnitudes ``[B]``."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'frequency_l1_per_sample expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = torch.fft.rfft(pred.float(), dim=-1, norm='ortho').abs()
        t = torch.fft.rfft(target.float(), dim=-1, norm='ortho').abs()
        return (p - t).abs().mean(dim=(1, 2))


def frequency_l1(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
    """L1 distance between orthonormal rFFT magnitudes of future trajectories.

    The transform is applied along time for each ROI; DC is included because a
    short BOLD window can carry a non-zero mean level that should be matched.
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'frequency_l1 expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    if mask is None:
        valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    else:
        valid = complete_sequence_mask(mask, pred.shape[-1])
    if not bool(valid.any()):
        return pred.new_zeros(())
    return frequency_l1_per_sample(pred[valid], target[valid]).mean()


def fc_pattern_pcc(pred: torch.Tensor, target: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Per-sample Pearson correlation between upper-triangle FC edges ``[B]``."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f'fc_pattern_pcc expects matching [B,F,H] tensors, got '
            f'{tuple(pred.shape)} vs {tuple(target.shape)}')
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = correlation_matrix(pred)
        t = correlation_matrix(target)
        f = pred.shape[1]
        iu = torch.triu_indices(f, f, offset=1, device=p.device)
        a, b = p[:, iu[0], iu[1]], t[:, iu[0], iu[1]]
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        denom = ((a.square().sum(dim=1) + eps).sqrt()
                 * (b.square().sum(dim=1) + eps).sqrt())
        return (a * b).sum(dim=1) / denom


def _huber_elementwise(pred: torch.Tensor, target: torch.Tensor,
                       delta: float) -> torch.Tensor:
    """逐元素 Huber：|e|≤δ 取 0.5e²，否则 δ(|e|−0.5δ)（与 F.huber_loss 同式）。"""
    e = (pred - target).abs()
    return torch.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta))


def temporal_difference_huber(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor = None,
                              context_last: torch.Tensor = None,
                              huber_delta: float = 1.0,
                              eps: float = 1e-8) -> torch.Tensor:
    """一阶时间差分 Huber（P0「时间动力学」约束）。

    Args:
        pred/target: [B, F, H] 预测与真值（同一空间，H 为预测步数）。
        mask:        [B, H] 各 horizon 是否有真值；None = 全有效。
        context_last:[B, F] context 末位 ``x_t``（真值）；None 时第一步差分不可得，
                     仅从 h=2 起监督（用于轨迹头，其 h=1 已由 one-step 项监督）。

    差分定义（第一步用真实历史末帧，其后用预测轨迹自身的前一帧）::

        Δŷ_1 = ŷ_1 − x_t ；Δŷ_h = ŷ_h − ŷ_(h−1)（h ≥ 2）
        Δy_1 = y_1 − x_t ；Δy_h = y_h − y_(h−1)（h ≥ 2）

    相邻有效性：horizon h 与其前一帧的真值须同时存在才计入（h=1 只需自身有效）。
    所有真值缺失时整项为零（按元素数归一，不引入常数项）。
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"时间差分 pred/target 形状不一致：{tuple(pred.shape)} vs "
            f"{tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"时间差分 pred 期望 [B,F,H]，收到 {tuple(pred.shape)}")
    b, f, h = pred.shape
    if context_last is not None and tuple(context_last.shape) != (b, f):
        raise ValueError(
            f"context_last 期望 [B,{f}]，收到 {tuple(context_last.shape)}")

    # 前一帧参考（预测与真值各一份）：有 x_t 时 h=1 用真值末帧，否则补零占位
    if context_last is not None:
        prev_p = torch.cat([context_last.unsqueeze(-1), pred[:, :, :-1]], dim=-1)
        prev_t = torch.cat([context_last.unsqueeze(-1), target[:, :, :-1]], dim=-1)
    else:
        zero_col = pred.new_zeros(b, f, 1)
        prev_p = torch.cat([zero_col, pred[:, :, :-1]], dim=-1)
        prev_t = torch.cat([zero_col, target[:, :, :-1]], dim=-1)

    if mask is None:
        m = pred.new_ones(b, h)
    else:
        if tuple(mask.shape) != (b, h):
            raise ValueError(
                f"时间差分 mask 形状 {tuple(mask.shape)} 与 [B,{h}] 不一致")
        m = mask.to(pred.dtype)
    # 前一帧有效性：h≥2 取决于 h−1；h=1 取决于是否提供 x_t
    prev_valid = torch.zeros_like(m)
    prev_valid[:, 1:] = m[:, :-1]
    if context_last is not None:
        prev_valid[:, 0] = 1.0
    w3 = (m * prev_valid).view(b, 1, h)

    hub = _huber_elementwise(pred - prev_p, target - prev_t, huber_delta)
    return (hub * w3).sum() / (w3.sum().clamp_min(eps) * f)


class ForecastFidelityLoss(nn.Module):
    """与已验证 B 相同的重建、FC、FFT、差分及全局/局部时间 PCC 损失。"""

    def __init__(self, lambda_rec: float, lambda_fc: float, lambda_freq: float,
                 lambda_delta: float, lambda_corr: float, lambda_local_corr: float,
                 huber_delta: float = 1.0):
        super().__init__()
        self.weights = dict(rec=lambda_rec, fc=lambda_fc, freq=lambda_freq,
                            delta=lambda_delta, corr=lambda_corr,
                            local_corr=lambda_local_corr)
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError('损失权重必须非负且有限')
        if not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError('huber_delta 必须为正且有限')
        self.huber_delta = float(huber_delta)

    def get_loss_weights(self) -> dict:
        return {name: float(weight) for name, weight in self.weights.items()}

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, context_last: torch.Tensor) -> tuple:
        if pred.ndim != 3 or pred.shape != target.shape:
            raise ValueError('预测和目标必须同为 [B,F,H]')
        if mask.shape != (pred.shape[0], pred.shape[-1]):
            raise ValueError('future mask 必须为 [B,H]')
        raw = dict(rec=masked_l1(pred, target, mask),
                   fc=fc_mse(pred, target, mask),
                   freq=frequency_l1(pred, target, mask))
        valid = (mask > 0).any().to(pred.dtype)
        complete = complete_sequence_mask(mask, pred.shape[-1]).any().to(pred.dtype)
        total = pred.sum() * 0.0
        for name, active in (('rec', valid), ('fc', complete), ('freq', complete)):
            if self.weights[name] > 0 and bool(active):
                total = total + self.weights[name] * raw[name]
        if self.weights['delta'] > 0:
            raw['delta'] = temporal_difference_huber(
                pred.float(), target.float(), mask, context_last.float(), self.huber_delta)
            total = total + self.weights['delta'] * raw['delta']
        if self.weights['corr'] > 0:
            raw['corr'] = temporal_pcc_loss(pred, target, mask)
            total = total + self.weights['corr'] * raw['corr']
        if self.weights['local_corr'] > 0:
            if pred.shape[-1] != 32:
                raise ValueError('局部时间 PCC 需要 H32')
            raw['local_corr'] = torch.stack([
                temporal_pcc_loss(pred[..., s:s+8], target[..., s:s+8], mask[:, s:s+8])
                for s in range(0, 32, 8)]).mean()
            total = total + self.weights['local_corr'] * raw['local_corr']
        stats = {f'loss_{key}': value.detach() for key, value in raw.items()}
        stats['loss_total'] = total.detach()
        return total, stats
