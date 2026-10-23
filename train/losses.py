# coding=utf-8
"""TFM 双目标损失与 SC 软先验图正则。"""
import math

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


def compute_cpm_mask(controls: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    """从首次非零未来控制开始屏蔽 CPM 监督（方案 §9）。

    controls:    [B, H, F] 新布局逐区间控制（u[t+h] 到达区间 (t+h-1, t+h]）
    future_mask: [B, H]    各 horizon 是否有真值

    active_h = 区间内存在任何非零驱动；cpm_mask = future_mask 且尚未遇到首个
    受控区间。该 mask 只用于 CPM（无控制辅助预测），不能用于屏蔽受控动态监督
    （动态监督仍用完整 future_mask）。
    """
    if controls.ndim != 3:
        raise ValueError(f"controls 期望 [B,H,F]，收到 {tuple(controls.shape)}")
    active_h = controls.abs().sum(-1) > 0                       # [B, H]
    unseen = (active_h.cumsum(-1) == 0).to(controls.dtype)
    return future_mask.to(controls.dtype) * unseen


def _single_temporal_difference_huber(pred: torch.Tensor, target: torch.Tensor,
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


def temporal_difference_huber(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor = None,
                              context_last: torch.Tensor = None,
                              huber_delta: float = 1.0, eps: float = 1e-8,
                              temporal_mode: str = 'standard',
                              temporal_lags=(1,), temporal_scale_weights=(1.0,),
                              adaptive_delta_alpha: float = 0.5,
                              adaptive_delta_clip: float = 3.0,
                              adaptive_delta_scale: float = 1.0) -> torch.Tensor:
    """E2-compatible temporal Huber; each lag is independently normalized.

    Only the observed final context frame is available: lag k starts at h=k
    with context_last, or h=k+1 without it. Validity requires both endpoints.
    Adaptive weights use detached targets and a fixed train-only delta scale.
    """
    if temporal_mode not in ('standard', 'adaptive', 'multiscale'):
        raise ValueError('unknown temporal_mode')
    if temporal_mode == 'standard':
        return _single_temporal_difference_huber(
            pred, target, mask, context_last, huber_delta, eps)
    if pred.ndim != 3 or pred.shape != target.shape:
        raise ValueError('temporal pred/target must share [B,F,H]')
    b, f, h = pred.shape
    if mask is not None and mask.shape != (b, h):
        raise ValueError('temporal mask must be [B,H]')
    if context_last is not None and context_last.shape != (b, f):
        raise ValueError('context_last must be [B,F]')
    if temporal_mode == 'adaptive':
        if (not math.isfinite(adaptive_delta_scale) or adaptive_delta_scale <= 0
                or not math.isfinite(adaptive_delta_alpha) or adaptive_delta_alpha < 0
                or not math.isfinite(adaptive_delta_clip) or adaptive_delta_clip <= 0):
            raise ValueError('adaptive scale/clip must be positive; alpha nonnegative')
        lags, weights = (1,), (1.0,)
    else:
        lags, weights = tuple(temporal_lags), tuple(temporal_scale_weights)
        if (not lags or len(lags) != len(weights) or len(set(lags)) != len(lags)
                or any(k not in (1, 2, 4) for k in lags)
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-7)):
            raise ValueError('lags must be unique 1/2/4 with nonnegative weights summing to 1')
    m = pred.new_ones(b, h) if mask is None else mask.to(pred)
    pp, tt = pred, target
    if context_last is not None:
        pp = torch.cat((context_last[..., None], pp), -1)
        tt = torch.cat((context_last[..., None], tt), -1)
        m = torch.cat((m.new_ones(b, 1), m), -1)
    loss = pred[..., :0].sum()
    for lag, scale_weight in zip(lags, weights):
        if lag >= pp.shape[-1]:
            continue
        dp, dt = pp[..., lag:] - pp[..., :-lag], tt[..., lag:] - tt[..., :-lag]
        w = (m[:, lag:] * m[:, :-lag])[:, None, :].expand_as(dp)
        if temporal_mode == 'adaptive':
            w = w * (1.0 + adaptive_delta_alpha *
                     (dt.detach().abs() / (adaptive_delta_scale + eps)).clamp(max=adaptive_delta_clip))
        hub = _huber_elementwise(dp, dt, huber_delta)
        loss = loss + scale_weight * (hub * w).sum() / w.sum().clamp_min(eps)
    return loss


def horizon_weights(horizon: int, reference: torch.Tensor, mode: str = 'gamma',
                    gamma: float = 1.0, alpha: float = 0.5,
                    power: float = 1.0) -> torch.Tensor:
    """Shared CPM/Dynamics horizon weights, indexed on the full H1..H grid."""
    index = torch.arange(horizon, device=reference.device, dtype=reference.dtype)
    if mode == 'gamma':
        return gamma ** index
    if mode == 'increasing':
        return 1.0 + alpha * (index / max(1, horizon - 1)).pow(power)
    raise ValueError(f'unknown horizon mode: {mode}')


def cross_head_consistency(cpm_pred: torch.Tensor, dynamics_pred: torch.Tensor,
                           mask: torch.Tensor = None,
                           eps: float = 1e-8) -> torch.Tensor:
    """双预测头一致性：CPM（学生）向 Graph-ODE 轨迹（教师，stop-gradient）靠近。

    L_cons = mean_{有效 (B,F,H)} | ŷ_CPM − sg(ŷ_ODE) |（L1）。

    梯度只流回 CPM 头，不反向把辅助头的误差灌进动力学主头（方案：ODE 作 Teacher、
    CPM 作 Student）。mask（[B,H]）通常取 ``cpm_mask ∩ future_mask``：**仅在 CPM 与
    ODE 表示同一无控制预测区间时**计算；当前脚本无控制序列时两者一致。
    """
    if cpm_pred.shape != dynamics_pred.shape:
        raise ValueError(
            f"双头一致性要求同形状（[B,F,H]），收到 CPM {tuple(cpm_pred.shape)} vs "
            f"ODE {tuple(dynamics_pred.shape)}")
    if cpm_pred.ndim != 3:
        raise ValueError(f"双头一致性 pred 期望 [B,F,H]，收到 {tuple(cpm_pred.shape)}")
    b, f, h = cpm_pred.shape
    d = (cpm_pred - dynamics_pred.detach()).abs()
    if mask is None:
        w3 = cpm_pred.new_ones(b, 1, h)
    else:
        if tuple(mask.shape) != (b, h):
            raise ValueError(
                f"双头一致性 mask 形状 {tuple(mask.shape)} 与 [B,{h}] 不一致")
        w3 = mask.to(cpm_pred.dtype).view(b, 1, h)
    return (d * w3).sum() / (w3.sum().clamp_min(eps) * f)


class TFMDualLoss(nn.Module):
    """One-Step + CPM 双目标损失（方案 §9 Dual Dynamics Objective / §20）。

    数学定义（pred/target 均为原始空间；``x_t`` 为 context 末位 TR）::

        L_one = Huber(x̂_(t+1), x_(t+1)) + λ_pcc·(1 − corr_ROI(x̂_(t+1), x_(t+1)))
        L_cpm = Σ_h w_h·Huber(x̂_(t+h), x_(t+h)) / Σ_h w_h,   w_h = γ^(h−1)·mask_h
        L     = λ_one·L_one + λ_cpm·L_cpm                     （weighting='fixed'）

    weighting='uncertainty'（同方差不确定性加权，Kendall et al., CVPR 2018）：
    每个启用（λ>0）的任务项 i ∈ {one, cpm, dynamics, variance} 配一个可学习
    log 方差参数 s_i（ParameterDict ``log_s``；经 build_optimizer 进入
    weight_decay=0 的 'loss_weight' 参数组），聚合改为::

        L = Σ_i [ exp(−s_i)·L_i + s_i·valid_i ]

    初始 s_i = −log λ_i，即**初始有效权重 exp(−s_i) 与固定权重方案完全一致**
    （warm start：训练起点梯度平衡与历史配置相同，随后由不确定性机制自适应）。
    valid_i 是该任务本 batch 的有效性指示（CPM=has_valid、dynamics=has_valid_dyn、
    variance=var_w.sum()>0；one-step 恒有效）：任务整批被 mask 时整项为零、
    **不加常数 s_i**（与「全 CPM mask 时损失项必须为零、不引入常数项」的约束
    一致，也避免 s_i 在无监督信号时无界漂移）。各任务损失的内部结构
    （Huber/PCC/γ 加权/mask 语义）不变，只替换跨任务聚合权重。

    设计说明（§20.1）：不使用数学等价的 ``L_abs + L_delta`` 双计项
    （prediction_target=delta 时二者数值重合），改为单一 Huber + spatial PCC；
    CPM 项逐 horizon 加权（§20.2，γ=1 即均匀，<1 时对远期 horizon 降权；
    cpm_horizon_weight='increasing' 时改为远期递增 w_h=1+α(h−1)/(H−1)）。
    λ_cpm=0 可退化为纯 one-step 的 Stage 0 基线。

    P0 新增（默认 λ=0，不改变既有行为）：
      - ``lambda_temporal_diff``：一阶时间差分 Huber（CPM + 动力学轨迹，见
        ``temporal_difference_huber``）；
      - ``lambda_cross_head``：双头一致性 L1（CPM 学生 ← sg(Graph-ODE) 教师，
        见 ``cross_head_consistency``，``cross_head_start_epoch`` 控制预热）。
    temporal_diff / cross_head 以固定权重直接加入总损失，不进入 uncertainty 聚合。
    lambda_temporal_pcc（默认 0）另加 CPM + ODE 的未来轨迹时间 PCC，亦为固定权重。
    """

    def __init__(self, lambda_one: float = 1.0, lambda_cpm: float = 1.0,
                 lambda_pcc: float = 0.1, cpm_gamma: float = 1.0,
                 huber_delta: float = 1.0, eps: float = 1e-8,
                 lambda_dynamics: float = 0.5, lambda_variance: float = 0.05,
                 weighting: str = 'fixed',
                 lambda_temporal_diff: float = 0.0,
                 lambda_cross_head: float = 0.0,
                 cpm_horizon_weight: str = 'gamma',
                 cpm_horizon_alpha: float = 0.5,
                 cross_head_start_epoch: int = 0,
                 temporal_mode: str = 'standard',
                 temporal_lags=(1,), temporal_scale_weights=(1.0,),
                 adaptive_delta_alpha: float = 0.5,
                 adaptive_delta_clip: float = 3.0,
                 adaptive_delta_scale: float = 1.0,
                 dynamics_horizon_weight: bool = False,
                 horizon_weight_power: float = 1.0,
                 lambda_temporal_pcc: float = 0.0):
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
        if lambda_dynamics < 0 or lambda_variance < 0:
            raise ValueError(
                f"lambda_dynamics/lambda_variance 不允许负数："
                f"{lambda_dynamics}/{lambda_variance}")
        if lambda_temporal_diff < 0 or lambda_cross_head < 0:
            raise ValueError(
                f"lambda_temporal_diff/lambda_cross_head 不允许负数："
                f"{lambda_temporal_diff}/{lambda_cross_head}")
        if cpm_horizon_weight not in ('gamma', 'increasing'):
            raise ValueError(
                f"cpm_horizon_weight 只支持 'gamma' / 'increasing'，"
                f"收到 {cpm_horizon_weight!r}")
        if cpm_horizon_alpha < 0:
            raise ValueError(
                f"cpm_horizon_alpha 不允许负数，收到 {cpm_horizon_alpha}")
        if int(cross_head_start_epoch) < 0:
            raise ValueError(
                f"cross_head_start_epoch 不允许负数，收到 {cross_head_start_epoch}")
        self.lambda_one = float(lambda_one)
        self.lambda_cpm = float(lambda_cpm)
        self.lambda_pcc = float(lambda_pcc)
        self.cpm_gamma = float(cpm_gamma)
        self.huber_delta = float(huber_delta)
        self.eps = float(eps)
        self.lambda_dynamics = float(lambda_dynamics)
        self.lambda_variance = float(lambda_variance)
        self.lambda_temporal_diff = float(lambda_temporal_diff)
        self.lambda_cross_head = float(lambda_cross_head)
        if not math.isfinite(lambda_temporal_pcc) or lambda_temporal_pcc < 0:
            raise ValueError('lambda_temporal_pcc must be finite and nonnegative')
        self.lambda_temporal_pcc = float(lambda_temporal_pcc)
        self.cpm_horizon_weight = str(cpm_horizon_weight)
        self.cpm_horizon_alpha = float(cpm_horizon_alpha)
        self.cross_head_start_epoch = int(cross_head_start_epoch)
        if not math.isfinite(horizon_weight_power) or horizon_weight_power <= 0:
            raise ValueError('horizon_weight_power must be positive')
        self.dynamics_horizon_weight = bool(dynamics_horizon_weight)
        self.horizon_weight_power = float(horizon_weight_power)
        self.temporal_kwargs = dict(
            temporal_mode=temporal_mode, temporal_lags=tuple(temporal_lags),
            temporal_scale_weights=tuple(temporal_scale_weights),
            adaptive_delta_alpha=adaptive_delta_alpha,
            adaptive_delta_clip=adaptive_delta_clip,
            adaptive_delta_scale=adaptive_delta_scale)
        # Validate configuration even before the first supervised batch.
        temporal_difference_huber(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8),
                                  **self.temporal_kwargs)
        if weighting not in ('fixed', 'uncertainty'):
            raise ValueError(
                f"weighting 只支持 'fixed' / 'uncertainty'，收到 {weighting!r}")
        self.weighting = str(weighting)
        # 同方差不确定性加权的可学习 log 方差（公式见类 docstring）。s_i =
        # -log λ_i 使初始有效权重 exp(-s_i) 精确等于原固定权重；λ=0 的任务项
        # 整体禁用（不建参数，前向也不进总和），避免 -log(0) 与无意义漂移。
        self.log_s = nn.ParameterDict()
        if weighting == 'uncertainty':
            for name, lam in (('one', self.lambda_one),
                              ('cpm', self.lambda_cpm),
                              ('dynamics', self.lambda_dynamics),
                              ('variance', self.lambda_variance)):
                if lam > 0:
                    self.log_s[name] = nn.Parameter(
                        torch.tensor(-math.log(lam), dtype=torch.float32))

    def _unc_weighted(self, name: str, task_loss: torch.Tensor,
                      valid: torch.Tensor = None) -> torch.Tensor:
        """同方差不确定性加权：exp(-s_i)·L_i + s_i（valid 给出时乘有效性指示，
        任务整批无效时整项为零、不引入常数项）。"""
        s = self.log_s[name]
        weighted = torch.exp(-s) * task_loss + s
        if valid is None:
            return weighted
        return valid.to(weighted.dtype) * weighted

    def get_loss_weights(self) -> dict:
        """当前各任务项的有效权重（日志 / checkpoint 元数据用）。

        返回 ``{'mode': ..., '<task>': {'weight': float, 'log_s': float|None}}``，
        只包含启用（λ>0）的任务项；'uncertainty' 模式下 weight=exp(-log_s)。
        """
        out = {'mode': self.weighting}
        for name in ('one', 'cpm', 'dynamics', 'variance',
                     'temporal_diff', 'cross_head', 'temporal_pcc'):
            lam = float(getattr(self, f'lambda_{name}'))
            if lam <= 0:
                continue
            # 新增的 temporal_diff / cross_head 项不参与不确定性加权：
            # 永远以固定 λ 报告（与 forward 中的聚合方式一致）。
            if self.weighting == 'uncertainty' and name in self.log_s:
                s = self.log_s[name]
                out[name] = {'weight': float(torch.exp(-s.detach())),
                             'log_s': float(s.detach())}
            else:
                out[name] = {'weight': lam, 'log_s': None}
        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                cpm_pred: torch.Tensor, cpm_target: torch.Tensor,
                cpm_mask: torch.Tensor = None,
                dynamics_pred: torch.Tensor = None,
                dynamics_target: torch.Tensor = None,
                dynamics_mask: torch.Tensor = None,
                dynamics_mean_norm: torch.Tensor = None,
                dynamics_logvar_norm: torch.Tensor = None,
                dynamics_target_norm: torch.Tensor = None,
                context_last: torch.Tensor = None,
                epoch: int = None) -> tuple:
        """
        Args:
            pred/target: [B, F, 1, 1] one-step 预测与真值（原始空间）
            cpm_pred:    [B, F, H]   CPM 全 horizon 预测（原始空间）
            cpm_target:  [B, F, H]   t+1..t+H 真值（原始空间，序列末尾截断补零）
            cpm_mask:    [B, H]      各 horizon 是否有真值（None = 全有效）；
                                     graph_ode 路线传入 compute_cpm_mask 的结果
            ---- 以下为 graph_ode 动态可选输入（discrete 路线不传，行为不变）----
            dynamics_pred/target:  [B, F, H] 动态轨迹预测与真值（数据尺度）
            dynamics_mask:         [B, H]    future_mask（**不是** cpm_mask）
            dynamics_mean_norm:    [B, F, H] 归一化空间动态均值（方差损失用，内部 detach）
            dynamics_logvar_norm:  [B, F, H] 归一化空间 logvar（仅更新 VarianceHead）
            dynamics_target_norm:  [B, F, H] 用当前历史 norm_stats 归一化的未来真值
            ---- 以下为 P0 新增项（λ=0 时不参与，行为与旧版一致）----
            context_last: [B, F] context 末位 x_t（真值，原始空间）；供 CPM 的
                一阶时间差分第一步使用，None 时 CPM 差分从 h=2 起
            epoch:        当前 epoch（训练用；为 None 视为不小于 start，验证/测试
                恒启用）。仅影响 --lambda_cross_head 的 ``cross_head_start_epoch``
                预热门控
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
        L_one_comb = loss_one + self.lambda_pcc * loss_one_pcc
        if self.weighting == 'uncertainty' and 'one' in self.log_s:
            total = self._unc_weighted('one', L_one_comb)
        else:
            total = self.lambda_one * L_one_comb
        stats['loss_one'] = loss_one.detach()
        stats['loss_one_pcc'] = loss_one_pcc.detach()
        stats['pcc_one'] = corr_one.detach()

        # ---- L_CPM（§20.2：γ 加权逐 horizon Huber）----
        n_valid = cpm_pred.shape[0]
        if n_valid > 0 and self.lambda_cpm > 0:
            h = cpm_pred.shape[2]
            w = torch.ones(cpm_pred.shape[0], h, device=cpm_pred.device,
                           dtype=cpm_pred.dtype)
            w = w * horizon_weights(
                h, cpm_pred, self.cpm_horizon_weight, self.cpm_gamma,
                self.cpm_horizon_alpha, self.horizon_weight_power).view(1, h)
            if cpm_mask is not None:
                if tuple(cpm_mask.shape) != (cpm_pred.shape[0], h):
                    raise ValueError(
                        f"cpm_mask 形状 {tuple(cpm_mask.shape)} 与 "
                        f"[B,{h}] 不一致")
                w = w * cpm_mask.to(cpm_pred.dtype)
            # 逐元素 Huber [B,F,H] × 逐 (样本,horizon) 权重
            hub = _huber_elementwise(cpm_pred, cpm_target, self.huber_delta)
            w3 = w.view(w.shape[0], 1, h)
            # CPM 有效权重全零时（受控区间从首个 horizon 起全部屏蔽），Huber 和
            # PCC **整项**都为零：不能留下 1-corr 常数项把梯度推向退化解。
            has_valid = (w.sum() > 0).to(cpm_pred.dtype)
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
            loss_cpm_pcc = has_valid * (
                1.0 - (corr_cpm * w).sum() / w.sum().clamp_min(self.eps))
            L_cpm_comb = loss_cpm + self.lambda_pcc * loss_cpm_pcc
            if self.weighting == 'uncertainty' and 'cpm' in self.log_s:
                # has_valid 同时门控 exp(-s)·L 与 s：整批被 mask 时不加常数项
                total = total + self._unc_weighted('cpm', L_cpm_comb,
                                                   valid=has_valid)
            else:
                total = total + self.lambda_cpm * has_valid * L_cpm_comb
            stats['loss_cpm'] = (has_valid * loss_cpm).detach()
            stats['loss_cpm_pcc'] = loss_cpm_pcc.detach()
        else:
            stats['loss_cpm'] = torch.zeros((), device=pred.device)
            stats['loss_cpm_pcc'] = torch.zeros((), device=pred.device)

        # ---- 动态多步损失（graph_ode；h=2..H 的 masked Huber + PCC）----
        # 第一帧已有 one-step 主监督，不重复加入多步项；没有有效 h2..H 时为零。
        if dynamics_pred is not None:
            if dynamics_pred.shape != dynamics_target.shape:
                raise ValueError(
                    f"动态 pred/target 形状不一致：{tuple(dynamics_pred.shape)} vs "
                    f"{tuple(dynamics_target.shape)}")
            if dynamics_pred.ndim != 3:
                raise ValueError(
                    f"动态 pred 期望 [B,F,H]，收到 {tuple(dynamics_pred.shape)}")
            if dynamics_mean_norm is not None and dynamics_mean_norm.shape[:3] != dynamics_pred.shape[:3]:
                raise ValueError(
                    f"动态均值归一化形状 {tuple(dynamics_mean_norm.shape)} 与 "
                    f"[B,F,{dynamics_pred.shape[2]}] 不一致")
            f_dim = dynamics_pred.shape[1]
            h_dyn = dynamics_pred.shape[2]
            dyn_w = dynamics_mask.to(dynamics_pred.dtype) \
                if dynamics_mask is not None \
                else torch.ones(dynamics_pred.shape[0], h_dyn,
                                device=dynamics_pred.device,
                                dtype=dynamics_pred.dtype)
            if tuple(dyn_w.shape) != (dynamics_pred.shape[0], h_dyn):
                raise ValueError(
                    f"dynamics_mask 形状 {tuple(dyn_w.shape)} 与 "
                    f"[B,{h_dyn}] 不一致")
            if h_dyn > 1:
                if self.dynamics_horizon_weight:
                    dyn_w = dyn_w * horizon_weights(
                        h_dyn, dynamics_pred, self.cpm_horizon_weight,
                        self.cpm_gamma, self.cpm_horizon_alpha,
                        self.horizon_weight_power).view(1, h_dyn)
                mask2 = dyn_w[:, 1:]                                   # [B, H-1]
                w2 = mask2.view(dynamics_pred.shape[0], 1, h_dyn - 1)
                hub_dyn = _huber_elementwise(dynamics_pred[:, :, 1:],
                                             dynamics_target[:, :, 1:],
                                             self.huber_delta)
                loss_dyn = (hub_dyn * w2).sum() / (
                    w2.sum().clamp_min(self.eps) * f_dim)
                corr_dyn = spatial_pcc(dynamics_pred[:, :, 1:].unsqueeze(-1),
                                       dynamics_target[:, :, 1:].unsqueeze(-1),
                                       self.eps)                       # [B, H-1]
                has_valid_dyn = (w2.sum() > 0).to(dynamics_pred.dtype)
                loss_dyn_pcc = has_valid_dyn * (
                    1.0 - (corr_dyn * mask2).sum()
                    / mask2.sum().clamp_min(self.eps))
                L_dyn_comb = loss_dyn + self.lambda_pcc * loss_dyn_pcc
                loss_dynamics = has_valid_dyn * L_dyn_comb
            else:
                has_valid_dyn = dynamics_pred.new_zeros(())
                loss_dynamics = has_valid_dyn
                loss_dyn_pcc = has_valid_dyn
                L_dyn_comb = None
            if (self.weighting == 'uncertainty' and 'dynamics' in self.log_s
                    and L_dyn_comb is not None):
                # has_valid_dyn 同时门控 exp(-s)·L 与 s（无常数项泄漏）
                total = total + self._unc_weighted('dynamics', L_dyn_comb,
                                                   valid=has_valid_dyn)
            else:
                total = total + self.lambda_dynamics * loss_dynamics
            stats['loss_dynamics'] = loss_dynamics.detach()
            stats['loss_dynamics_pcc'] = loss_dyn_pcc.detach()
        else:
            stats['loss_dynamics'] = torch.zeros((), device=pred.device)
            stats['loss_dynamics_pcc'] = torch.zeros((), device=pred.device)

        # ---- 方差损失（h=1..H 对角高斯 NLL；只更新 VarianceHead）----
        # 均值与归一化目标 detach；logvar 保留梯度（VarianceHead 输入 z 轨迹已在
        # 模型侧 detach，本项不会影响共享动力学参数）。
        if dynamics_logvar_norm is not None:
            if dynamics_mean_norm is None or dynamics_target_norm is None:
                raise ValueError(
                    "传入 dynamics_logvar_norm 时必须同时提供 dynamics_mean_norm "
                    "与 dynamics_target_norm（归一化空间）")
            mu = dynamics_mean_norm.detach()
            tgt = dynamics_target_norm.detach()
            logvar = dynamics_logvar_norm
            if logvar.shape != mu.shape or tgt.shape != mu.shape:
                raise ValueError(
                    f"方差损失输入形状不一致：logvar {tuple(logvar.shape)}, "
                    f"mu {tuple(mu.shape)}, target {tuple(tgt.shape)}")
            var_w = dynamics_mask.to(logvar.dtype) \
                if dynamics_mask is not None \
                else torch.ones_like(logvar[:, 0, :])
            if tuple(var_w.shape) != logvar.shape[:1] + logvar.shape[2:]:
                raise ValueError(
                    f"dynamics_mask 形状 {tuple(var_w.shape)} 与 logvar "
                    f"[B,F,{logvar.shape[2]}] 的 [B,H] 不一致")
            nll = 0.5 * (logvar + (-logvar).exp() * (mu - tgt) ** 2)   # [B,F,H]
            loss_var = (nll * var_w.unsqueeze(1)).sum() / (
                var_w.sum().clamp_min(self.eps) * logvar.shape[1])
            if self.weighting == 'uncertainty' and 'variance' in self.log_s:
                # 全 mask 时 loss_var 恒为 0，valid 指示同时压掉 s 常数项
                valid_var = (var_w.sum() > 0).to(logvar.dtype)
                total = total + self._unc_weighted('variance', loss_var,
                                                   valid=valid_var)
            else:
                total = total + self.lambda_variance * loss_var
            stats['loss_variance'] = loss_var.detach()
        else:
            stats['loss_variance'] = torch.zeros((), device=pred.device)

        # ---- P0：一阶时间差分（时间动力学约束，固定权重项）----
        # CPM 用自身 cpm_mask，第一步差分参考真实历史末帧 x_t；动力学轨迹用
        # dynamics_mask，其 h=1 已由 one-step 项监督，差分自 h=2 起（参考预测
        # 轨迹自身的前一帧）。不受 weighting='uncertainty' 影响。
        td_cpm = pred.new_zeros(())
        td_dyn = pred.new_zeros(())
        if self.lambda_temporal_diff > 0:
            if cpm_pred.shape[0] > 0:
                td_cpm = temporal_difference_huber(
                    cpm_pred, cpm_target, mask=cpm_mask,
                    context_last=context_last,
                    huber_delta=self.huber_delta, eps=self.eps,
                    **self.temporal_kwargs)
            if dynamics_pred is not None and dynamics_pred.shape[0] > 0:
                td_dyn = temporal_difference_huber(
                    dynamics_pred, dynamics_target, mask=dynamics_mask,
                    context_last=None,
                    huber_delta=self.huber_delta, eps=self.eps,
                    **self.temporal_kwargs)
            total = total + self.lambda_temporal_diff * (td_cpm + td_dyn)
        stats['loss_temporal_diff'] = (td_cpm + td_dyn).detach()
        stats['loss_temporal_diff_cpm'] = td_cpm.detach()
        stats['loss_temporal_diff_dyn'] = td_dyn.detach()

        # ---- P0：双预测头一致性（CPM ← sg(Graph-ODE)，固定权重项）----
        # ODE 作 Teacher（stop-gradient）、CPM 作 Student：梯度只回 CPM 头。
        # 仅在 CPM 与 ODE 表示同一无控制区间（cpm_mask ∩ future_mask）内计算；
        # epoch 早于 start 时不启用（预热门控，方案 §5「前 5–10 epoch 不启用」）。
        cross = pred.new_zeros(())
        if (self.lambda_cross_head > 0
                and not (epoch is not None
                         and int(epoch) < self.cross_head_start_epoch)
                and dynamics_pred is not None and dynamics_pred.shape[0] > 0
                and cpm_pred.shape[0] > 0):
            cons_mask = cpm_mask
            if cons_mask is None:
                cons_mask = dynamics_mask
            elif dynamics_mask is not None:
                if tuple(cons_mask.shape) != tuple(dynamics_mask.shape):
                    raise ValueError(
                        f"双头一致性 cpm_mask {tuple(cons_mask.shape)} 与 "
                        f"dynamics_mask {tuple(dynamics_mask.shape)} 形状不一致")
                cons_mask = cons_mask.to(dynamics_mask.dtype) * dynamics_mask
            cross = cross_head_consistency(cpm_pred, dynamics_pred,
                                           mask=cons_mask, eps=self.eps)
            total = total + self.lambda_cross_head * cross
        stats['loss_cross_head'] = cross.detach()

        # Temporal PCC is a separate auxiliary objective; retain all E2 terms.
        tp_cpm, tp_dyn = pred.new_zeros(()), pred.new_zeros(())
        if self.lambda_temporal_pcc > 0:
            if cpm_pred.shape[0] > 0:
                tp_cpm = temporal_pcc_loss(cpm_pred, cpm_target, cpm_mask, self.eps)
            if dynamics_pred is not None and dynamics_pred.shape[0] > 0:
                tp_dyn = temporal_pcc_loss(
                    dynamics_pred, dynamics_target, dynamics_mask, self.eps)
            total = total + self.lambda_temporal_pcc * (tp_cpm + tp_dyn)
        stats['loss_temporal_pcc'] = (tp_cpm + tp_dyn).detach()
        stats['loss_temporal_pcc_cpm'] = tp_cpm.detach()
        stats['loss_temporal_pcc_dyn'] = tp_dyn.detach()

        stats['loss_total'] = total.detach()
        return total, stats
