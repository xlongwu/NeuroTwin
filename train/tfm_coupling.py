# coding=utf-8
"""TFM Router–Expert Competence Coupling（RECC）与 Router 正则（方案 §17–§25/§41–§43）。

职责划分（方案 §22/§85/§86）：
- TFMDualLoss（train/losses.py）继续只负责 One-Step + CPM 任务损失 → 训练 Expert 能力；
- 本文件的 RECC / Pair-RECC / Router 正则由 main.py 在训练循环中聚合 → 校准 Router：
    L_total = L_TFMDual + L_graph + L_router-reg + λ_recc·L_recc + λ_pair·L_pair

耦合方向（§16–§21）::

    所有 Expert 对同一样本分别预测（candidate 均含共享 HC 动力学）
        → 用真实 x_(t+1) 计算每个 Expert 的预测误差（competence）
        → 逐样本 competence 分布 q（stop-gradient，Expert 维 z-score + 温度 softmax）
        → KL(sg(q) ‖ p_cpl)，p_cpl 为 detach 输入下同一 Router 的完整 soft 分布
        → 梯度只更新 Router 参数（核心验收项 §43，由 tests/test_tfm_recc.py 守护）

与原 ERC 的 activation norm proxy 不同（§15）：TFM 场景 E=4、batch 有限，
每个样本可对全部 Expert 真实预测，直接使用 next-state 预测误差。
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.tfm_experts import build_pair_indices


def get_tfm_recc_weight(epoch: int, weight: float, start_epoch: int = 5,
                        ramp_epochs: int = 10) -> float:
    """RECC 权重 warmup 调度（方案 §24）：start 前 0 → ramp 线性增加 → 固定。

    epoch 为 0 基；start_epoch=5, ramp_epochs=10 时：epoch<5 → 0，
    epoch 5–14 → weight·(epoch−4)/10，epoch≥15 → weight。
    """
    epoch = int(epoch)
    weight = float(weight)
    if epoch < int(start_epoch) or weight <= 0.0:
        return 0.0
    ramp = int(ramp_epochs)
    if ramp <= 0 or epoch >= int(start_epoch) + ramp:
        return weight
    return weight * (epoch - int(start_epoch) + 1) / float(ramp)


def compute_expert_competence_losses(expert_pred: torch.Tensor,
                                     target: torch.Tensor,
                                     huber_delta: float = 1.0,
                                     pcc_weight: float = 0.0,
                                     eps: float = 1e-8) -> torch.Tensor:
    """逐 (样本, 专家) 的 candidate 预测误差（方案 §17/§18），返回 [B, E]（detach）。

    expert_pred [B,E,F] 与 target [B,F] 必须同空间（原始 BOLD 量纲）。
    ℓ = Huber(candidate, target)（ROI 维取均值）；pcc_weight>0 时叠加
    λ_c_pcc·(1 − PCC_ROI)。competence 只作监督信号不训练 Expert（§22），
    因此在此处整体 detach，切断向 Expert 的梯度通路。
    """
    if expert_pred.ndim != 3:
        raise ValueError(
            f"expert_pred 期望 [B,E,F]，收到 {tuple(expert_pred.shape)}")
    target2d = target.reshape(target.shape[0], -1)
    if target2d.shape[0] != expert_pred.shape[0] \
            or target2d.shape[-1] != expert_pred.shape[-1]:
        raise ValueError(
            f"target 形状 {tuple(target.shape)} 与 expert_pred "
            f"{tuple(expert_pred.shape)} 在 (B, F) 维不一致")
    e_abs = (expert_pred - target2d.unsqueeze(1)).abs()       # [B,E,F]
    ell = torch.where(e_abs <= huber_delta, 0.5 * e_abs ** 2,
                      huber_delta * (e_abs - 0.5 * huber_delta)).mean(dim=-1)
    if pcc_weight > 0:
        # PCC_ROI：逐 (样本, 专家) 在 ROI 维比较全脑 pattern（方案 §18）
        pc = expert_pred - expert_pred.mean(dim=-1, keepdim=True)
        tc = target2d - target2d.mean(dim=-1, keepdim=True)
        num = (pc * tc.unsqueeze(1)).sum(dim=-1)
        den = ((pc ** 2).sum(dim=-1) + eps).sqrt() \
            * ((tc ** 2).sum(dim=-1) + eps).sqrt()
        ell = ell + pcc_weight * (1.0 - num / (den + eps))
    return ell.detach()


def _competence_distribution(ell: torch.Tensor, tau: float,
                             eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """误差矩阵 [B, E'] → competence 分布 q（方案 §19）：Expert 维 z-score 后
    softmax(−ℓ̃/τ)。返回 (q, log_q)。"""
    mu = ell.mean(dim=-1, keepdim=True)
    sigma = ell.std(dim=-1, unbiased=False, keepdim=True)
    ell_n = (ell - mu) / (sigma + eps)
    log_q = F.log_softmax(-ell_n / float(tau), dim=-1)
    return log_q.exp(), log_q


def compute_tfm_recc(aux_info: Dict, target: torch.Tensor, tau: float = 1.0,
                     pcc_weight: float = 0.0, huber_delta: float = 1.0,
                     eps: float = 1e-8) -> Tuple[torch.Tensor, Dict]:
    """RECC-V1（方案 §20/§42）：L_RECC = KL(sg(q) ‖ p_cpl)。

    Args:
        aux_info: 需含 ``tfm_expert_pred`` [B,E,F]（原始空间）与
            ``tfm_router_probs_coupling`` [B,E]（detach 耦合分支）
        target: [B,F,1,1] 或 [B,F] 真实下一时刻 BOLD（原始空间）
        tau: competence 温度 τ_c
        pcc_weight: spatial PCC competence 项权重（V1 默认 0，方案 §18）
    Returns:
        (loss, stats)；aux 缺少必要键时返回 (0, {})（recc off 的调用路径）。
    """
    zero = torch.zeros((), device=target.device)
    expert_pred = aux_info.get('tfm_expert_pred') if isinstance(aux_info, dict) else None
    p_cpl = aux_info.get('tfm_router_probs_coupling') if isinstance(aux_info, dict) else None
    if not (torch.is_tensor(expert_pred) and torch.is_tensor(p_cpl)):
        return zero, {}
    if expert_pred.shape[0] != p_cpl.shape[0] \
            or expert_pred.shape[1] != p_cpl.shape[1]:
        raise ValueError(
            f"expert_pred {tuple(expert_pred.shape)} 与 router_probs_coupling "
            f"{tuple(p_cpl.shape)} 在 (B, E) 维不一致")

    ell = compute_expert_competence_losses(expert_pred, target, huber_delta,
                                           pcc_weight, eps)      # [B,E] detached
    q, log_q = _competence_distribution(ell, tau, eps)
    # KL(q ‖ p)：sg(q) 使梯度只流向 p_cpl（→ Router 参数，方案 §21/§43）
    log_p = p_cpl.clamp_min(eps).log()
    kl = (q * (log_q - log_p)).sum(dim=-1).mean()

    stats = {
        'recc_loss': kl.detach(),
        'recc_competence_entropy': -(q * log_q).sum(dim=-1).mean().detach(),
    }
    return kl, stats


def compute_tfm_pair_recc(aux_info: Dict, target: torch.Tensor, tau: float = 1.0,
                          pcc_weight: float = 0.0, huber_delta: float = 1.0,
                          eps: float = 1e-8) -> Tuple[torch.Tensor, Dict]:
    """RECC-V2：Top-2 Pair-Aware Coupling（方案 §26–§29）。

    q_pair 来自等权 pair candidate（Δz_ij=(Δz_i+Δz_j)/2）的真实误差；
    Router implied pair score s_ij = p_i·p_j 归一化为 π，损失
    L_pair = KL(sg(q_pair) ‖ π)。使用 equal weight（§29）避免 competence
    target 依赖当前 Router 形成自反馈。
    """
    zero = torch.zeros((), device=target.device)
    pair_pred = aux_info.get('tfm_pair_pred') if isinstance(aux_info, dict) else None
    p_cpl = aux_info.get('tfm_router_probs_coupling') if isinstance(aux_info, dict) else None
    if not (torch.is_tensor(pair_pred) and torch.is_tensor(p_cpl)):
        return zero, {}
    num_experts = p_cpl.shape[1]
    pair_idx = build_pair_indices(num_experts).to(pair_pred.device)  # [P,2]
    if pair_pred.shape[1] != pair_idx.shape[0]:
        raise ValueError(
            f"tfm_pair_pred 有 {pair_pred.shape[1]} 个 pair，与 E={num_experts} "
            f"应产生的 C(E,2)={pair_idx.shape[0]} 不一致")

    ell_pair = compute_expert_competence_losses(pair_pred, target, huber_delta,
                                                pcc_weight, eps)   # [B,P] detached
    q_pair, log_q_pair = _competence_distribution(ell_pair, tau, eps)
    # Router implied pair score：s_ij = p_i p_j，在 pair 维归一化（方案 §28）
    s_pair = p_cpl[:, pair_idx[:, 0]] * p_cpl[:, pair_idx[:, 1]]     # [B,P]
    pi = s_pair / s_pair.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = (q_pair * (log_q_pair - pi.clamp_min(eps).log())).sum(dim=-1).mean()

    stats = {
        'pair_recc_loss': kl.detach(),
        'pair_competence_entropy': -(q_pair * log_q_pair).sum(dim=-1).mean().detach(),
    }
    return kl, stats


def compute_tfm_router_regularization(aux_info: Dict,
                                      balance_weight: float = 0.01,
                                      z_weight: float = 0.001,
                                      entropy_weight: float = 0.0005,
                                      eps: float = 1e-8) -> Tuple[torch.Tensor, Dict]:
    """Router 正则（方案 §25）：load balance + z-loss + 熵（防坍缩，权重极小）。

    - L_bal = E·Σ_i importance_i·load_i：importance 为 soft 概率均值，load 为
      Top-K 选中频率（dense warmup 期恒 1，该项退化为常数、无梯度）；
    - L_z = E[(log Σ e^{a_i})²]：抑制 router logits 过度增大；
    - L_ent = −H(p)：极小权重鼓励熵，与 RECC 的 competence specialization 平衡
      （§25.3：RECC 已提供 soft target，故权重须远小于 balance/z）。

    aux 缺少 router 键（非 moe 模式）或三项权重全为 0 时返回零正则。
    """
    device = None
    for key in ('tfm_router_probs', 'tfm_router_logits', 'tfm_router_weights'):
        v = aux_info.get(key) if isinstance(aux_info, dict) else None
        if torch.is_tensor(v):
            device = v.device
            break
    zero = torch.zeros((), device=device) if device is not None \
        else torch.zeros(())
    zero_stats = {'router_balance': zero.detach(),
                  'router_zloss': zero.detach(),
                  'router_entropy': zero.detach()}
    probs = aux_info.get('tfm_router_probs') if isinstance(aux_info, dict) else None
    if not torch.is_tensor(probs):
        return zero, zero_stats
    logits = aux_info.get('tfm_router_logits')
    weights = aux_info.get('tfm_router_weights')
    total = zero
    stats = {}
    if balance_weight > 0:
        n_experts = probs.shape[1]
        importance = probs.mean(dim=0)                              # [E]
        load = (weights > 0).to(probs.dtype).mean(dim=0) \
            if torch.is_tensor(weights) else torch.ones_like(importance)
        bal = n_experts * (importance * load).sum()
        total = total + balance_weight * bal
        stats['router_balance'] = bal.detach()
    if z_weight > 0 and torch.is_tensor(logits):
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        total = total + z_weight * z
        stats['router_zloss'] = z.detach()
    if entropy_weight > 0:
        entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1).mean()
        total = total - entropy_weight * entropy
        stats['router_entropy'] = entropy.detach()
    for k in zero_stats:
        stats.setdefault(k, zero_stats[k])
    return total, stats


def compute_tfm_expert_diversity(aux_info: Dict,
                                 eps: float = 1e-8) -> Tuple[torch.Tensor, Dict]:
    """专家修正去相关正则（专家分化实验 Phase B-E2）。

    对逐专家 latent 修正 Δz_i [B,E,F,D] 在 (F·D) 维展平后做 L2 归一化，
    L_div = mean_{i<j} cos²(Δz_i, Δz_j) ∈ [0, 1]。

    动机（Phase A 诊断，2026-10-07）：B 变体中四个专家参数近正交（cos≈0.06）
    但 Δz 输出逐样本两两 |cos|≈0.62 —— 任务梯度只约束 mixture 均值，各专家
    修正收敛到共享的大幅有害方向 + 平均后才抵消的结构，candidate 误差彼此
    相同（0.4403~0.4404），competence 排名失去信息。本正则直接打散共享方向，
    使各专家修正可区分、RECC 的 competence 监督有对象。

    梯度通路：Δz 带梯度（与 RECC 相反，本正则训练 Expert 而非 Router）；
    未被 Top-K 选中的专家无任务梯度、仅受本正则 → 分化压力。
    aux 缺少 ``tfm_expert_delta_z_norm``（need_expert_pred=False）或 E<2 时
    返回零正则。
    """
    delta_z = aux_info.get('tfm_expert_delta_z_norm') \
        if isinstance(aux_info, dict) else None
    device = delta_z.device if torch.is_tensor(delta_z) else None
    if not torch.is_tensor(delta_z) or delta_z.ndim != 4 \
            or delta_z.shape[1] < 2:
        zero = torch.zeros((), device=device)
        return zero, {'expert_diversity': zero.detach(),
                      'expert_dz_cos': zero.detach()}
    b, e = delta_z.shape[0], delta_z.shape[1]
    v = delta_z.float().reshape(b, e, -1)                      # [B,E,F·D]
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    cos = torch.einsum('bid,bjd->bij', v, v)                   # [B,E,E]
    iu = torch.triu_indices(e, e, offset=1, device=cos.device)
    pair_cos = cos[:, iu[0], iu[1]]                            # [B,P]
    div = pair_cos.pow(2).mean()
    stats = {'expert_diversity': div.detach(),
             'expert_dz_cos': pair_cos.abs().mean().detach()}
    return div, stats


@torch.no_grad()
def compute_tfm_coupling_diagnostics(aux_info: Dict, target: torch.Tensor,
                                     actual_pred: Optional[torch.Tensor] = None,
                                     tau: float = 1.0, huber_delta: float = 1.0,
                                     eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Router–Expert 机制指标（方案 §63–§70），全部 no_grad、仅用于诊断。

    返回标量 tensor（batch 均值）：
      - ``router_top1_agreement``：P[argmax p = argmin ℓ]（§64，随机≈25%）
      - ``router_top2_recall``：|Router Top-2 ∩ Oracle Top-2| / 2（§65）
      - ``router_regret``：ℓ[argmax p] − min_i ℓ_i（§66，越低越好）
      - ``router_competence_spearman``：Spearman(p, −ℓ)（§67）
      - ``competence_entropy``：H(q)（§68）
      - ``oracle_best_expert_gap``：实际 MoE 预测损失 − 最好单专家损失
        （§69，需 actual_pred [B,F]）；gap 大 → Router 是瓶颈
      - ``oracle_best_pair_gap``：Router 实际 Top-2 pair 损失 − oracle 最优
        等权 pair 损失（§70，仅当 aux 含 ``tfm_pair_pred`` 时计算；
        gap 大 → 值得引入 Pair-RECC）
      - ``expert_loss_mean``：专家平均误差（绝对水平参照）
    """
    expert_pred = aux_info.get('tfm_expert_pred') if isinstance(aux_info, dict) else None
    probs = aux_info.get('tfm_router_probs') if isinstance(aux_info, dict) else None
    if not (torch.is_tensor(expert_pred) and torch.is_tensor(probs)):
        return {}
    ell = compute_expert_competence_losses(expert_pred, target, huber_delta,
                                           0.0, eps)             # [B,E]
    b, e = ell.shape
    stats: Dict[str, torch.Tensor] = {'expert_loss_mean': ell.mean()}

    router_top1 = probs.argmax(dim=-1)                            # [B]
    oracle_top1 = ell.argmin(dim=-1)                              # [B]
    stats['router_top1_agreement'] = (router_top1 == oracle_top1
                                      ).to(ell.dtype).mean()

    k2 = min(2, e)
    router_top2 = torch.topk(probs, k2, dim=-1).indices           # [B,2]
    oracle_top2 = torch.topk(-ell, k2, dim=-1).indices            # [B,2]
    # Recall_2 = |S_router ∩ S_oracle| / k2（逐样本交集大小，方案 §65）
    overlap = sum(
        len(set(router_top2[i].tolist()) & set(oracle_top2[i].tolist()))
        for i in range(b))
    stats['router_top2_recall'] = torch.tensor(
        overlap / float(b * k2), device=ell.device)

    regret = ell.gather(1, router_top1.unsqueeze(1)).squeeze(1) \
        - ell.min(dim=-1).values
    stats['router_regret'] = regret.mean()

    # 逐样本 Spearman(p, −ℓ)：秩相关（E=4 的诊断口径，§67）
    rank_p = probs.argsort(dim=-1).argsort(dim=-1).to(ell.dtype)
    rank_e = (-ell).argsort(dim=-1).argsort(dim=-1).to(ell.dtype)
    ra = rank_p - rank_p.mean(dim=-1, keepdim=True)
    rb = rank_e - rank_e.mean(dim=-1, keepdim=True)
    rho = (ra * rb).sum(dim=-1) / (
        (ra ** 2).sum(dim=-1).sqrt() * (rb ** 2).sum(dim=-1).sqrt() + eps)
    stats['router_competence_spearman'] = rho.mean()

    q, log_q = _competence_distribution(ell, tau, eps)
    stats['competence_entropy'] = -(q * log_q).sum(dim=-1).mean()

    if torch.is_tensor(actual_pred):
        actual2d = actual_pred.reshape(actual_pred.shape[0], -1)
        if actual2d.shape[-1] != expert_pred.shape[-1]:
            raise ValueError(
                f"actual_pred 宽度 {actual2d.shape[-1]} 与 expert_pred 的 ROI 数 "
                f"{expert_pred.shape[-1]} 不一致")
        e_abs = (actual2d - target.reshape(target.shape[0], -1)).abs()
        actual_loss = torch.where(
            e_abs <= huber_delta, 0.5 * e_abs ** 2,
            huber_delta * (e_abs - 0.5 * huber_delta)).mean(dim=-1)
        stats['oracle_best_expert_gap'] = (actual_loss - ell.min(dim=-1).values).mean()

    # Oracle Best-Pair Gap（§70）：Router 实际 Top-2 pair 的损失 − oracle 最优
    # 等权 pair 的损失。gap 大 → 存在 pair 级 mismatch，值得引入 Pair-RECC（§26）。
    # 仅在 aux 提供 pair candidate（recc_mode=pair）时可用。
    pair_pred = aux_info.get('tfm_pair_pred') if isinstance(aux_info, dict) else None
    if torch.is_tensor(pair_pred):
        ell_pair = compute_expert_competence_losses(pair_pred, target,
                                                    huber_delta, 0.0, eps)  # [B,P]
        pair_idx = build_pair_indices(e).to(ell_pair.device)          # [P,2]
        router_top2_sorted = torch.sort(router_top2, dim=-1).values    # [B,2]
        match = (pair_idx.unsqueeze(0) == router_top2_sorted.unsqueeze(1)
                 ).all(dim=-1)                                         # [B,P]
        if not bool(match.any(dim=1).all()):
            raise ValueError("Router Top-2 无法对应到 pair candidate 的索引表")
        router_pair_loss = ell_pair[match].view(b)
        stats['oracle_best_pair_gap'] = (
            router_pair_loss - ell_pair.min(dim=-1).values).mean()
    return stats
