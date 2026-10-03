# coding=utf-8
"""不确定性加权多任务混合损失与轮间（deep supervision）损失。"""
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _window_weight(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把窗口级掩码 [B, W] 广播为与 `ref` [B, F, W, S] 同形的权重。"""
    if mask.ndim != 2:
        raise ValueError(f"pred_mask 期望 [B, W] 二维张量，收到 {tuple(mask.shape)}")
    if mask.shape[0] != ref.shape[0] or mask.shape[1] != ref.shape[2]:
        raise ValueError(
            f"pred_mask 形状 {tuple(mask.shape)} 与预测 {tuple(ref.shape)} 不匹配")
    return mask.to(ref.dtype).reshape(mask.shape[0], 1, mask.shape[1], 1)


def _masked_pearson(pred: torch.Tensor, target: torch.Tensor,
                    weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """带掩码的逐 ROI Pearson，返回 [B, F]。

    weight 为 [B, 1, W, 1] 形式的窗口掩码，广播后按展平维度做加权统计；
    weight 全 1 时与 `pearson` 数值等价。

    逐样本权重全 0（多 horizon 下该样本在该 horizon 无真值）时 ``pc = pf*w = 0``，
    这里用 ``sqrt(sum(x²) + eps)`` 保证 √ 的入参恒为正（避免 ``sqrt(0)`` 的无穷
    导数把整批梯度污染成 NaN）；该行输出仍为 0，外层再乘 sample_weight=0 后
    既不进损失也不进梯度。
    """
    b, f = pred.shape[0], pred.shape[1]
    pf = pred.reshape(b, f, -1)
    tf = target.reshape(b, f, -1)
    w = weight.expand_as(pred).reshape(b, f, -1)
    n = w.sum(-1).clamp_min(eps).unsqueeze(-1)
    pf = pf - (pf * w).sum(-1, keepdim=True) / n
    tf = tf - (tf * w).sum(-1, keepdim=True) / n
    pc, tc = pf * w, tf * w
    n_p = ((pc ** 2).sum(-1) + eps).sqrt()
    n_t = ((tc ** 2).sum(-1) + eps).sqrt()
    return (pc * tc).sum(-1) / (n_p * n_t + eps)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor,
               weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """带掩码的加权 L1（权重可为广播形式），全 1 时退化为 `F.l1_loss`。"""
    w = weight.expand_as(pred)
    return (torch.abs(pred - target) * w).sum() / w.sum().clamp_min(eps)


class GaussianNLLLoss(nn.Module):
    """高斯负对数似然（逐元素，可带窗口掩码）。

    ``0.5 * (logvar + (target - mu)^2 * exp(-logvar))``。``logvar`` 先做 clamp，
    避免训练早期方差塌缩（logvar 大幅下降）导致损失爆炸。``mu`` 与 ``logvar``
    都参与梯度，属标准 NLL；该项在总损失中的整体权重由 ``lambda_nll`` 承担。
    """

    def __init__(self, logvar_min: float = -8.0, logvar_max: float = 4.0, eps: float = 1e-6):
        super().__init__()
        if logvar_min >= logvar_max:
            raise ValueError(f"logvar_min ({logvar_min}) 必须小于 logvar_max ({logvar_max})")
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.eps = float(eps)

    def forward(self, pred, target, logvar, mask=None):
        if logvar.shape != pred.shape:
            raise ValueError(
                f"pred_logvar 形状 {tuple(logvar.shape)} 与预测 {tuple(pred.shape)} 不一致")
        lv = logvar.clamp(self.logvar_min, self.logvar_max)
        nll = 0.5 * (lv + (target - pred) ** 2 * torch.exp(-lv))
        if mask is None:
            return nll.mean()
        w = _window_weight(mask, pred)
        return (nll * w).sum() / w.sum().clamp_min(self.eps)


class QuantilePinballLoss(nn.Module):
    """分位数（pinball）损失：``sum_q mean(max(q*e, (q-1)*e))``，``e = y - mu_q``。

    ``pred_q`` 形状 [B, F, Q, W, S]，``target`` 形状 [B, F, W, S]。
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred_q, target, quantiles, mask=None):
        if pred_q.ndim != target.ndim + 1:
            raise ValueError(
                f"pred_quantiles 形状 {tuple(pred_q.shape)} 应为 {tuple(target.shape)} 前插一维 Q")
        if pred_q.shape[2] != len(quantiles):
            raise ValueError(
                f"分位数通道数 {pred_q.shape[2]} 与 quantiles={list(quantiles)} 不一致")
        q = torch.as_tensor(quantiles, dtype=pred_q.dtype,
                            device=pred_q.device).view(1, 1, -1, 1, 1)
        err = target.unsqueeze(2) - pred_q
        loss = torch.maximum(q * err, (q - 1.0) * err)
        if mask is None:
            return loss.mean()
        w = mask.to(pred_q.dtype).reshape(mask.shape[0], 1, 1, mask.shape[1], 1)
        w = w.expand_as(loss)
        return (loss * w).sum() / w.sum().clamp_min(self.eps)


class IntermediateSupervisionLoss(nn.Module):
    """轮间（deep supervision）监督：约束 Refiner 各轮的中间预测。

    动机：多轮级联细化只监督最终输出时，中间轮可能承担与最终目标无关的
    修正方向。对每轮中间预测直接施加波形监督，可让“每轮都在向真值靠近”
    成为显式约束。

    权重：第 r 轮权重 w_r = decay ** (R - 1 - r)，末轮最大（w_{R-1} = 1），
    并归一化为 sum(w) = 1，使该项与主损失量级可比。

    每轮损失：`(1 - PCC) + mae_weight * L1`。
    """

    def __init__(self, decay: float = 0.5, mae_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.decay = float(decay)
        self.mae_weight = float(mae_weight)
        self.eps = float(eps)

    def forward(self, round_preds: Sequence[torch.Tensor], target: torch.Tensor,
                mask: torch.Tensor = None):
        """round_preds: 长度 R 的中间预测列表（含最终预测或仅前置轮均可）。

        mask: 可选的窗口级掩码 [B, W]（1 有效 / 0 填充）。
        """
        if not round_preds:
            raise ValueError("round_preds 为空，无法计算轮间监督损失。")
        r_total = len(round_preds)
        weights = [self.decay ** (r_total - 1 - r) for r in range(r_total)]
        norm = float(sum(weights))

        total = target.new_zeros(())
        stats = {}
        for r, pred in enumerate(round_preds):
            if pred.shape != target.shape:
                raise ValueError(
                    f"轮间预测形状 {tuple(pred.shape)} 与目标 {tuple(target.shape)} 不一致")
            if mask is None:
                l_pcc = 1.0 - pearson(pred, target, self.eps).mean()
                l_mae = F.l1_loss(pred, target)
            else:
                w = _window_weight(mask, pred)
                l_pcc = 1.0 - _masked_pearson(pred, target, w, self.eps).mean()
                l_mae = _masked_l1(pred, target, w, self.eps)
            loss_r = l_pcc + self.mae_weight * l_mae
            total = total + (weights[r] / norm) * loss_r
            stats[f'inter_sup_r{r}_pcc'] = l_pcc.detach()
            stats[f'inter_sup_r{r}_mae'] = l_mae.detach()
        stats['inter_sup_total'] = total.detach()
        return total, stats


def compute_intermediate_supervision(aux_info, target, criterion, weight, device, mask=None):
    """从 aux_info['refiner_round_preds'] 计算加权轮间监督损失。

    weight <= 0 或不含轮间预测时返回零标量，调用方可无条件相加。
    """
    zero = torch.zeros(1, device=device).squeeze(0)
    if criterion is None or weight <= 0 or not isinstance(aux_info, dict):
        return zero, {}
    round_preds: List[torch.Tensor] = aux_info.get('refiner_round_preds', None) or []
    if not round_preds:
        return zero, {}
    loss, stats = criterion(round_preds, target, mask=mask)
    return loss * float(weight), stats



def compute_inversion_loss(aux_info, weight, device):
    """辅助反演损失：把池化潜在状态反演到（归一化后的）病理条件向量。

    weight <= 0 或模型未构建反演头时返回零标量，调用方可无条件相加。
    """
    zero = torch.zeros(1, device=device).squeeze(0)
    if weight <= 0 or not isinstance(aux_info, dict):
        return zero, {}
    pred = aux_info.get('inversion_pred', None)
    target = aux_info.get('pathology_cond', None)
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        return zero, {}
    if pred.shape != target.shape:
        raise ValueError(
            f"反演预测形状 {tuple(pred.shape)} 与病理条件 {tuple(target.shape)} 不一致")
    loss = F.mse_loss(pred, target.detach())
    return loss * float(weight), {'inversion_mse': loss.detach()}


# ══════════════════════════════════════════════════════════════════════════
#  Next-Timepoint 任务损失（task_mode='next_timepoint'）
# ══════════════════════════════════════════════════════════════════════════


def spatial_pcc(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """逐 (样本, 偏移) 的**空间** PCC：在 ROI 维比较两种全脑 pattern。

    pred/target: [B, F, H, 1]。返回 [B, H]。

    与沿时间轴定义的 temporal PCC 语义不同：next-timepoint 的单步目标
    ``x_(t+δ) ∈ R^F`` 只有一个时间点，沿 ROI 维的相关才对应“预测的空间
    pattern 是否与真实一致”（prompt §十四 明确要求区分两种口径；时序轨迹
    口径在 rollout 指标里单独计算）。
    """
    p = pred[..., 0]
    t = target[..., 0]
    pc = p - p.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    n_p = ((pc ** 2).sum(dim=1) + eps).sqrt()
    n_t = ((tc ** 2).sum(dim=1) + eps).sqrt()
    return (pc * tc).sum(dim=1) / (n_p * n_t + eps)


class NextTimepointLoss(nn.Module):
    """Next-Timepoint 主损失：absolute + delta + spatial PCC（+ 可选概率项）。

    数学定义（``pred`` 与 ``target`` 均为原始空间的 [B,F,H,1]，H = 预测偏移数；
    ``x_t`` 为 context 末位 TR）：

        L_abs   = Σ w_bh·|x̂_(t+δ) − x_(t+δ)| / Σ w_bh
        L_delta = Σ w_bh·|Δx̂_(t+δ) − Δx_(t+δ)| / Σ w_bh
                  Δx̂ = x̂ − x_t，Δx_ = x_(t+δ) − x_t
        L_pcc   = 1 − Σ w_bh·corr_ROI(x̂_(t+δ), x_(t+δ)) / Σ w_bh   （spatial PCC）
        L_total = λ_abs·L_abs + λ_delta·L_delta + λ_pcc·L_pcc
                  (+ λ_nll·NLL，仅在 --lambda_nll > 0 且模型给出 logvar/分位数时)

    说明（重要）：当预测头使用 x_t 锚点（``prediction_target=delta``，默认）时，
    ``Δx̂ = x̂ − x_t`` 与 ``L_abs`` 的误差**数值重合**（因为 x̂ = x_t + Δx̂ 是仿射
    关系），此时 λ_delta 等价于给同一目标再加权重；在 ``prediction_target=absolute``
    （零锚点）下两项才提供不同梯度：L_abs 约束绝对状态、L_delta 只约束状态变化量，
    这正是「delta vs absolute」消融的对照点。两项都会被如实记录，便于核对。

    设计取舍（实测依据见实施报告 §7）：最初尝试把 L_delta 定义在 RevIN 归一化空间
    （即按逐 ROI 的**局部** stdev 归一 ``Δx_raw / stdev_local``），但本数据的 BOLD 经
    逐 ROI 全序列 z-score 后仍存在近乎常数的时间窗（实测单窗口 stdev 最小 0.003），
    该定义会被极少数近常数 ROI 主导（实测 L_delta 被放大到 L_abs 的 ~9 倍），
    因此改为原始空间定义。

    概率项默认关闭（``lambda_nll=0``）：单时间点预测下 logvar 头在近常数 context 上
    的方差换算不稳定（exp(−logvar) 可达 1e3 量级），v1 不纳入主损失。

    权重 ``w_bh`` 由「偏移权重（--mtp_weights，单步时恒为 1）× 目标有效掩码」构成，
    因此 MTP 下某偏移缺真值（序列末尾）时自动不参与监督。
    """

    def __init__(self, lambda_abs: float = 1.0, lambda_delta: float = 1.0,
                 lambda_pcc: float = 0.1, lambda_nll: float = 0.0,
                 eps: float = 1e-8):
        super().__init__()
        if lambda_abs < 0 or lambda_delta < 0 or lambda_pcc < 0 or lambda_nll < 0:
            raise ValueError(
                f"lambda_abs/lambda_delta/lambda_pcc/lambda_nll 不允许负数："
                f"{lambda_abs}/{lambda_delta}/{lambda_pcc}/{lambda_nll}")
        if lambda_abs == 0 and lambda_delta == 0 and lambda_pcc == 0:
            raise ValueError("lambda_abs/lambda_delta/lambda_pcc 不能全为 0。")
        self.lambda_abs = float(lambda_abs)
        self.lambda_delta = float(lambda_delta)
        self.lambda_pcc = float(lambda_pcc)
        self.lambda_nll = float(lambda_nll)
        self.eps = float(eps)
        self.nll_fn = GaussianNLLLoss()
        self.pinball_fn = QuantilePinballLoss()

    @staticmethod
    def _weights(pred: torch.Tensor, mask, mtp_weights) -> torch.Tensor:
        """返回逐 (样本, 偏移) 权重 [B, H]（偏移权重 × 有效掩码）。"""
        b, _, h, _ = pred.shape
        w = torch.ones(b, h, device=pred.device, dtype=pred.dtype)
        if mask is not None:
            if tuple(mask.shape) != (b, h):
                raise ValueError(
                    f"target_mask 形状 {tuple(mask.shape)} 与预测 {tuple(pred.shape)} 不匹配")
            w = w * mask.to(pred.dtype)
        if mtp_weights is not None:
            if len(mtp_weights) != h:
                raise ValueError(
                    f"mtp_weights 长度 {len(mtp_weights)} 与预测偏移数 {h} 不一致")
            w = w * torch.as_tensor(mtp_weights, device=pred.device,
                                    dtype=pred.dtype).view(1, h)
        return w

    def forward(self, pred: torch.Tensor, target: torch.Tensor, x_last: torch.Tensor,
                aux_info=None, mask=None, mtp_weights=None, offsets=None):
        """pred/target: [B,F,H,1]；x_last: [B,F] 或 [B,F,1,1]；mask: [B,H]。"""
        if pred.shape != target.shape:
            raise ValueError(
                f"pred/target 形状不一致：{tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.ndim != 4 or pred.shape[-1] != 1:
            raise ValueError(f"next_timepoint 损失期望 [B,F,H,1]，收到 {tuple(pred.shape)}")
        xl = x_last.reshape(x_last.shape[0], x_last.shape[1], 1, 1)
        w = self._weights(pred, mask, mtp_weights)               # [B,H]
        w4 = w.view(w.shape[0], 1, w.shape[1], 1)
        n_ele = w.sum().clamp_min(self.eps) * pred.shape[1] * pred.shape[3]

        # ---- L_abs（原始空间） ----
        loss_abs = ((pred - target).abs() * w4).sum() / n_ele

        # ---- L_delta（原始空间的「变化量」误差） ----
        delta_true = target - xl                                 # [B,F,H,1]
        delta_pred = pred - xl
        loss_delta = ((delta_pred - delta_true).abs() * w4).sum() / n_ele

        # ---- L_spatial_pcc ----
        corr = spatial_pcc(pred, target, self.eps)               # [B,H]
        loss_pcc = 1.0 - (corr * w).sum() / w.sum().clamp_min(self.eps)

        total = (self.lambda_abs * loss_abs
                 + self.lambda_delta * loss_delta
                 + self.lambda_pcc * loss_pcc)

        # ---- 可选概率项（默认关闭；pred_head=gaussian / quantile） ----
        loss_nll = None
        if self.lambda_nll > 0 and isinstance(aux_info, dict):
            logvar = aux_info.get('pred_logvar', None)
            quantiles = aux_info.get('pred_quantiles', None)
            if torch.is_tensor(logvar):
                loss_nll = self.nll_fn(pred, target, logvar, mask=mask)
            elif torch.is_tensor(quantiles):
                qs = aux_info.get('pred_quantile_levels', None)
                if qs is None:
                    raise ValueError(
                        "aux_info['pred_quantiles'] 存在但缺少 'pred_quantile_levels'")
                loss_nll = self.pinball_fn(quantiles, target, qs, mask=mask)
        if loss_nll is not None:
            total = total + self.lambda_nll * loss_nll

        stats = {
            'loss_total': total.detach(),
            'loss_abs': loss_abs.detach(),
            'loss_delta': loss_delta.detach(),
            # 兼容既有训练日志/评估口径：loss_pcc = spatial PCC 惩罚项；pcc = 平均空间 PCC
            'loss_pcc': loss_pcc.detach(),
            'loss_mae': loss_abs.detach(),
            'loss_nll': (loss_nll.detach() if loss_nll is not None
                         else torch.zeros((), device=pred.device)),
            'pcc': ((corr * w).sum() / w.sum().clamp_min(self.eps)).detach(),
        }
        # ---- 逐偏移明细（MTP 日志 train/loss_t+1 .. t+8 的来源） ----
        h_max = pred.shape[2]
        offs = list(offsets) if offsets is not None else list(range(1, h_max + 1))
        if len(offs) != h_max:
            raise ValueError(f"offsets 长度 {len(offs)} 与预测偏移数 {h_max} 不一致")
        for h in range(h_max):
            w_h = w[:, h]
            n_h = float(w_h.sum().item())
            if n_h <= 0:
                continue
            w4h = w_h.view(-1, 1, 1, 1)
            stats[f'loss_off{offs[h]}'] = (
                ((pred[:, :, h:h + 1, :] - target[:, :, h:h + 1, :]).abs() * w4h
                 ).sum() / (n_h * pred.shape[1] * pred.shape[3])).detach()
            stats[f'pcc_off{offs[h]}'] = (
                (corr[:, h] * w_h).sum() / n_h).detach()
            stats[f'n_off{offs[h]}'] = torch.tensor(n_h, device=pred.device)
        return total, stats


def compute_rollout_loss(pred_rollout: torch.Tensor, target_future: torch.Tensor,
                         mask: torch.Tensor, weight: float, device):
    """可选的自回归 rollout 训练损失（--enable_rollout_loss，prompt §三十三）。

    pred_rollout: [B, R, F]（模型自由滚动的预测轨迹）
    target_future: [B, R, F]（真值）；mask: [B, R]（1 有效 / 0 超出序列末尾）
    L_rollout = Σ mask·|x̂ − x| / Σ mask
    """
    zero = torch.zeros(1, device=device).squeeze(0)
    if weight <= 0 or pred_rollout is None or target_future is None:
        return zero, {}
    if pred_rollout.shape != target_future.shape:
        raise ValueError(
            f"rollout 预测/真值形状不一致：{tuple(pred_rollout.shape)} vs "
            f"{tuple(target_future.shape)}")
    w = mask.to(pred_rollout.dtype)
    n = w.sum().clamp_min(1e-8) * pred_rollout.shape[-1]
    loss = ((pred_rollout - target_future).abs() * w.unsqueeze(-1)).sum() / n
    return loss * float(weight), {'rollout': loss.detach()}


def compute_rollout_amp_loss(pred_rollout: torch.Tensor, target_future: torch.Tensor,
                             mask: torch.Tensor, weight: float, device):
    """多步幅值约束（--lambda_rollout_amp，配合 --enable_rollout_loss 使用）。

    自由滚动下条件均值回归会使预测轨道幅值系统性收缩（方差塌缩），
    该项逐 rollout 步匹配预测与真值的状态 RMS（跨 ROI），约束每步轨道幅值。
    采用逐步相对偏差的平方（对整体尺度不敏感，且各步独立受约束）：

        rms_k  = mean_F(x²)^0.5          # [B, R] 逐样本逐步
        rel_k  = Σ_b w·(rms_p − rms_t) / Σ_b w·rms_t
        L_amp  = mean_k rel_k²

    pred_rollout: [B, R, F]；target_future: [B, R, F]；mask: [B, R]
    """
    zero = torch.zeros(1, device=device).squeeze(0)
    if weight <= 0 or pred_rollout is None or target_future is None:
        return zero, {}
    if pred_rollout.shape != target_future.shape:
        raise ValueError(
            f"rollout 幅值约束预测/真值形状不一致：{tuple(pred_rollout.shape)} vs "
            f"{tuple(target_future.shape)}")
    w = mask.to(pred_rollout.dtype)                                  # [B, R]
    rms_p = pred_rollout.pow(2).mean(-1).sqrt()                      # [B, R]
    rms_t = target_future.pow(2).mean(-1).sqrt()                     # [B, R]
    num = (w * (rms_p - rms_t)).sum(0)                               # [R]
    den = (w * rms_t).sum(0).clamp_min(1e-6)                         # [R]
    rel_sq = (num / den) ** 2                                        # [R]
    valid = w.sum(0) > 0                                             # 仅对有效步平均
    loss = rel_sq[valid].mean() if bool(valid.any()) else torch.zeros(
        (), device=pred_rollout.device)
    return loss * float(weight), {'rollout_amp': loss.detach()}


# ══════════════════════════════════════════════════════════════════════════
#  NeuroTwin-TFM 双目标损失（model_arch='tfm'，方案 §19 / §20）
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

    与 NextTimepointLoss 的区别（§20.1）：不再同时使用数学等价的
    ``L_abs + L_delta`` 双计项（prediction_target=delta 时二者数值重合），
    改为单一 Huber + spatial PCC；CPM 项逐 horizon 加权（§20.2，γ=1 即均匀，
    <1 时对远期 horizon 降权）。权重 α=β 的取值由验证集调优（§23：禁止第一版
    全部打开——此处仅两项，λ_cpm=0 可退化为纯 one-step 的 Stage 0 基线）。
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