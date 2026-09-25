# coding=utf-8
"""不确定性加权多任务混合损失与轮间（deep supervision）损失。"""
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """逐 ROI 的 Pearson 相关系数，返回 [B, F]。

    pred/target 允许 [B, F, ...] 任意尾部形状，内部按最后一维之外展平。
    """
    b, f = pred.shape[0], pred.shape[1]
    pf = pred.reshape(b, f, -1)
    tf = target.reshape(b, f, -1)
    pc = pf - pf.mean(-1, keepdim=True)
    tc = tf - tf.mean(-1, keepdim=True)
    return (pc * tc).sum(-1) / ((pc**2).sum(-1).sqrt() * (tc**2).sum(-1).sqrt() + eps)


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
    """
    b, f = pred.shape[0], pred.shape[1]
    pf = pred.reshape(b, f, -1)
    tf = target.reshape(b, f, -1)
    w = weight.expand_as(pred).reshape(b, f, -1)
    n = w.sum(-1).clamp_min(eps).unsqueeze(-1)
    pf = pf - (pf * w).sum(-1, keepdim=True) / n
    tf = tf - (tf * w).sum(-1, keepdim=True) / n
    pc, tc = pf * w, tf * w
    return (pc * tc).sum(-1) / ((pc**2).sum(-1).sqrt() * (tc**2).sum(-1).sqrt() + eps)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor,
               weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """带掩码的加权 L1（权重可为广播形式），全 1 时退化为 `F.l1_loss`。"""
    w = weight.expand_as(pred)
    return (torch.abs(pred - target) * w).sum() / w.sum().clamp_min(eps)


def _masked_std(pred: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """带掩码的逐 ROI 标准差（按展平维度统计），返回 [B, F]。"""
    b, f = pred.shape[0], pred.shape[1]
    pf = pred.reshape(b, f, -1)
    w = weight.expand_as(pred).reshape(b, f, -1)
    n = w.sum(-1).clamp_min(eps).unsqueeze(-1)
    mean = (pf * w).sum(-1, keepdim=True) / n
    var = ((pf - mean) ** 2 * w).sum(-1) / n.squeeze(-1)
    return (var + eps).sqrt()


class GaussianNLLLoss(nn.Module):
    """高斯负对数似然（逐元素，可带窗口掩码）。

    ``0.5 * (logvar + (target - mu)^2 * exp(-logvar))``。``logvar`` 先做 clamp，
    避免训练早期方差塌缩（logvar 大幅下降）导致损失爆炸。``mu`` 与 ``logvar``
    都参与梯度，属标准 NLL；该项在总损失中的整体权重由
    `UncertaintyWeightedHybridLoss` 的 ``log_var_nll``（默认低权重）承担。
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


class UncertaintyWeightedHybridLoss(nn.Module):
    """同方差不确定性加权多任务损失。

    同时优化 PCC、MAE、一阶差分与窗口内标准差四项，
    权重由可学习的 log-variance 参数自动平衡。

    一阶差分项有两种口径（``diff_mode``）：
      - ``per_window``（默认）：先保留窗口维度，再沿时间轴 ``diff``，避免
        展平后把“上一窗末 → 下一窗首”的跨窗跳变误当成一阶差分；
      - ``flatten``：旧行为，沿 ``[W × S]`` 展平维度 ``diff``。

    当 DataLoader 因 ``--variable_cutoff`` 提供窗口级掩码时（``mask`` 非 None），
    全部四项按有效窗口做加权统计，填充窗不参与监督。
    """

    def __init__(
        self,
        eps=1e-8, init_log_var_pcc=0.0, init_log_var_mae=-1.5,
        init_log_var_diff=-2.0, init_log_var_std=-2.0,
        clamp_log_vars=True, log_var_min=-6.0, log_var_max=6.0,
        diff_mode='per_window',
        init_log_var_nll=2.0,
    ):
        super().__init__()
        if diff_mode not in ('flatten', 'per_window'):
            raise ValueError(f"diff_mode 仅支持 flatten/per_window，收到 '{diff_mode}'")
        self.eps = eps
        self.clamp_log_vars = clamp_log_vars
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.diff_mode = diff_mode
        self.log_var_pcc  = nn.Parameter(torch.tensor(float(init_log_var_pcc)))
        self.log_var_mae  = nn.Parameter(torch.tensor(float(init_log_var_mae)))
        self.log_var_diff = nn.Parameter(torch.tensor(float(init_log_var_diff)))
        self.log_var_std  = nn.Parameter(torch.tensor(float(init_log_var_std)))
        # 第 5 项：概率项（Gaussian NLL 或分位数 pinball）。init_log_var_nll 取正值
        # 使初始权重 exp(-lv) 明显小于其他项（默认 exp(-2)=0.135），避免方差塌缩。
        self.log_var_nll  = nn.Parameter(torch.tensor(float(init_log_var_nll)))
        self.nll_fn = GaussianNLLLoss()
        self.pinball_fn = QuantilePinballLoss()

    def _bounded(self, x):
        return torch.clamp(x, self.log_var_min, self.log_var_max) if self.clamp_log_vars else x

    def pearson(self, pred, target):
        """保留为实例方法以兼容既有调用。"""
        return pearson(pred, target, self.eps)

    def _diff_term(self, pred, pf, target, tf, weight=None):
        """一阶差分项，按 `self.diff_mode` 选择口径。"""
        if self.diff_mode == 'per_window':
            if pred.ndim < 2 or pred.shape[-1] < 2:
                raise ValueError(
                    "diff_mode='per_window' 需要预测的最后一维为时间轴且长度 >= 2，"
                    f"收到 {tuple(pred.shape)}")
            dp, dt = torch.diff(pred, dim=-1), torch.diff(target, dim=-1)
        else:
            dp, dt = torch.diff(pf, dim=-1), torch.diff(tf, dim=-1)
        if weight is None:
            return F.l1_loss(dp, dt)
        if self.diff_mode == 'per_window':
            return _masked_l1(dp, dt, weight, self.eps)
        # flatten 模式下差分结果的最后一个元素对应后一时刻，掩码同步右移一位
        w = weight.expand_as(pred).reshape(pred.shape[0], pred.shape[1], -1)[..., 1:]
        return _masked_l1(dp, dt, w, self.eps)

    def forward(self, pred, target, aux_info=None, mask=None):
        """aux_info 为预留参数（概率头等额外监督项），当前不参与计算。

        mask: 可选的窗口级掩码 [B, W]（1 有效 / 0 填充），仅在
              `--variable_cutoff` 开启时由 DataLoader 提供。
        """
        pf = pred.reshape(pred.shape[0], pred.shape[1], -1)
        tf = target.reshape(target.shape[0], target.shape[1], -1)

        # mask 为 None 时走未掩码实现，保证既有配置的数值逐位不变
        if mask is None:
            pcc       = self.pearson(pred, target)
            loss_pcc  = 1.0 - pcc.mean()
            loss_mae  = F.l1_loss(pred, target)
            loss_diff = self._diff_term(pred, pf, target, tf)
            loss_std  = F.l1_loss(pf.std(-1, unbiased=False), tf.std(-1, unbiased=False))
        else:
            w         = _window_weight(mask, pred)
            pcc       = _masked_pearson(pred, target, w, self.eps)
            loss_pcc  = 1.0 - pcc.mean()
            loss_mae  = _masked_l1(pred, target, w, self.eps)
            loss_diff = self._diff_term(pred, pf, target, tf, weight=w)
            loss_std  = F.l1_loss(_masked_std(pred, w, self.eps),
                                  _masked_std(target, w, self.eps))

        lv_pcc, lv_mae = self._bounded(self.log_var_pcc), self._bounded(self.log_var_mae)
        lv_diff, lv_std = self._bounded(self.log_var_diff), self._bounded(self.log_var_std)
        lv_nll = self._bounded(self.log_var_nll)

        total = (torch.exp(-lv_pcc) * loss_pcc  + lv_pcc
               + torch.exp(-lv_mae) * loss_mae  + lv_mae
               + torch.exp(-lv_diff)* loss_diff + lv_diff
               + torch.exp(-lv_std) * loss_std  + lv_std)

        # 第 5 项概率损失：仅在模型提供概率输出时参与，否则该项严格为 0
        # （不加 lv_nll 常数项，避免在无概率头时把该权重推向无意义的方向）
        loss_nll = None
        if isinstance(aux_info, dict):
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
            total = total + torch.exp(-lv_nll) * loss_nll + lv_nll

        stats = {
            'loss_total': total.detach(), 'loss_pcc': loss_pcc.detach(),
            'loss_mae': loss_mae.detach(), 'loss_diff': loss_diff.detach(),
            'loss_std': loss_std.detach(),
            'loss_nll': (loss_nll.detach() if loss_nll is not None
                         else torch.zeros((), device=pred.device)),
            'weighted_pcc': (torch.exp(-lv_pcc)*loss_pcc+lv_pcc).detach(),
            'weighted_mae': (torch.exp(-lv_mae)*loss_mae+lv_mae).detach(),
            'weighted_diff': (torch.exp(-lv_diff)*loss_diff+lv_diff).detach(),
            'weighted_std': (torch.exp(-lv_std)*loss_std+lv_std).detach(),
            'weighted_nll': (torch.exp(-lv_nll)*loss_nll+lv_nll).detach()
                            if loss_nll is not None else torch.zeros((), device=pred.device),
            'pcc': pcc.mean().detach(),
            'log_var_pcc': lv_pcc.detach(), 'log_var_mae': lv_mae.detach(),
            'log_var_diff': lv_diff.detach(), 'log_var_std': lv_std.detach(),
            'log_var_nll': lv_nll.detach(),
            'precision_pcc':  torch.exp(-lv_pcc).detach(),
            'precision_mae':  torch.exp(-lv_mae).detach(),
            'precision_diff': torch.exp(-lv_diff).detach(),
            'precision_std':  torch.exp(-lv_std).detach(),
            'precision_nll':  torch.exp(-lv_nll).detach(),
        }
        return total, stats


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

        mask: 可选的窗口级掩码 [B, W]，语义同 `UncertaintyWeightedHybridLoss`。
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