# coding=utf-8
"""MoE 训练期调度与外部正则化：路由温度调度 + 负载均衡/Z-Loss/熵/多样性正则。"""
import math

import torch
import torch.nn.functional as F

from models.neurotwin import NeuroTwin


def _get(aux_info, key):
    """从 aux_info 中取 tensor 字段，缺失或非 tensor 时返回 None。"""
    if isinstance(aux_info, dict):
        v = aux_info.get(key, None)
        return v if torch.is_tensor(v) else None
    return None


def update_router_temperature(model: NeuroTwin, args, epoch: int,
                              expert_usage_stats: dict = None):
    """Hold-then-Decay 路由温度调度（负载均衡感知）。

    - Warmup 期（前 10% epochs）：从 moe_gate_temp_start 线性降到 moe_gate_temp_end；
    - 主体训练：余弦退火，在 [temp_end, temp_end+0.3] 间波动帮助探索；
    - 若专家使用变异系数 CV > 0.5（不均衡），临时升温最多 +0.5 重新探索。

    仅在 finetune 且存在 MoE 时生效，返回当前温度（float）或 None。
    （TFM 等无 MoE 模型返回 None，不参与调度。）
    """
    if args.mode != 'finetune' or getattr(model, 'moe', None) is None:
        return None

    temp_start = args.moe_gate_temp_start
    temp_end = args.moe_gate_temp_end
    total_epochs = args.train_epochs

    warmup_epochs = max(1, int(total_epochs * 0.1))
    if epoch < warmup_epochs:
        ratio = epoch / warmup_epochs
        temp = temp_start - (temp_start - temp_end) * ratio
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        temp = temp_end + 0.3 * cosine_decay

    if expert_usage_stats is not None and epoch > warmup_epochs:
        usage_values = list(expert_usage_stats.values())
        if len(usage_values) > 1:
            mean_usage = sum(usage_values) / len(usage_values)
            std_usage = (sum((x - mean_usage) ** 2 for x in usage_values) / len(usage_values)) ** 0.5
            cv = std_usage / (mean_usage + 1e-8)
            if cv > 0.5:
                temp_boost = min(0.5, (cv - 0.5))
                temp = temp + temp_boost

    model.set_moe_router_temperature(temp)
    return float(temp)


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


def compute_moe_regularization(aux_info, device,
                               load_balance_weight: float = 0.01,
                               entropy_weight: float = 1e-3,
                               z_loss_weight: float = 1e-3,
                               diversity_weight: float = 0.001):
    """MoE 外部正则化，返回 (total_reg, stats)。

    total_reg = 负载均衡 × load_balance_weight   # Switch 风格 importance×load（可微形式）
              + Z-Loss × z_loss_weight           # ST-MoE，惩罚路由 logit 幅度过大
              + (-路由熵) × entropy_weight        # 抑制路由过度塌缩
              + 多样性损失 × diversity_weight     # 惩罚 batch 内路由决策趋同（CV 反向奖励）

    所有字段从 aux_info 读取：soft_gates / gate_logits / importance / load /
    selection_frequency / load_balancing_term / gates / top_k_indices。
    """
    zero = torch.zeros(1, device=device).squeeze(0)

    soft_gates   = _get(aux_info, 'soft_gates')
    gate_logits  = _get(aux_info, 'gate_logits')
    importance   = _get(aux_info, 'importance')
    load         = _get(aux_info, 'load')
    sel_freq     = _get(aux_info, 'selection_frequency')
    lbt          = _get(aux_info, 'load_balancing_term')
    # 路由不确定性报告：eval_mc_samples>0 时由 MoE 写入，纯监控量
    prob_std     = _get(aux_info, 'moe_router_prob_std')

    zero_stats = {'moe_load_balance': zero.detach(),
                  'moe_entropy': zero.detach(), 'moe_z_loss': zero.detach()}
    if prob_std is not None:
        zero_stats['moe_router_prob_std'] = prob_std.detach()

    if soft_gates is None or soft_gates.ndim < 2:
        return zero, zero_stats

    # token 级路由（route_level='token'）形状为 [B, F, E]，展平到 token 维度 [N, E]
    if soft_gates.ndim > 2:
        soft_gates = soft_gates.reshape(-1, soft_gates.shape[-1])
    if gate_logits is not None and gate_logits.ndim > 2:
        gate_logits = gate_logits.reshape(-1, gate_logits.shape[-1])

    num_experts = soft_gates.shape[1]

    if importance is None or importance.ndim != 1:
        importance = soft_gates.mean(0)
    if load is None or load.ndim != 1:
        top_k_idx = _get(aux_info, 'top_k_indices')
        if top_k_idx is not None and top_k_idx.ndim >= 2:
            if top_k_idx.ndim > 2:
                top_k_idx = top_k_idx.reshape(-1, top_k_idx.shape[-1])
            oh = F.one_hot(top_k_idx, num_classes=num_experts).float()
            # 分母为 token 总数 × top_k（token 级路由时 token 数 = B × F）
            load = oh.sum(dim=(0, 1)) / float(
                max(1, top_k_idx.shape[0] * top_k_idx.shape[1]))
            if sel_freq is None:
                sel_freq = oh.amax(dim=1).float().mean(0)
        else:
            load = soft_gates.mean(0)
    if sel_freq is None:
        gates = _get(aux_info, 'gates')
        sel_freq = ((gates > 0).float().mean(0) if gates is not None
                    else torch.zeros_like(importance))

    # 1. 负载均衡损失（lbt 为 detach 的监控值，反传用 importance×load 可微形式）
    if lbt is not None and lbt.numel() == 1:
        lb_loss = num_experts * (load.detach() * importance).sum() * load_balance_weight
        lb_val  = lbt
    else:
        lb_loss = zero
        lb_val  = zero

    # 2. 路由 Z-Loss（ST-MoE）
    if gate_logits is not None and gate_logits.ndim == 2:
        z_loss = torch.logsumexp(gate_logits, dim=-1).pow(2).mean() * z_loss_weight
    else:
        z_loss = zero

    # 3. 熵正则化（最小化负熵 = 最大化路由分布熵）
    gate_entropy = -(soft_gates.clamp_min(1e-8) * soft_gates.clamp_min(1e-8).log()).sum(1).mean()
    entropy_reg  = (-gate_entropy) * entropy_weight

    # 4. 专家多样化损失：以专家使用比例的变异系数 CV 衡量均衡度，
    #    -1/(1+CV) 使 CV 越小（使用越均匀）奖励越大
    expert_usage = soft_gates.mean(dim=0)
    usage_std = expert_usage.std()
    usage_mean = expert_usage.mean()
    cv = usage_std / (usage_mean + 1e-8)
    diversity_loss = -diversity_weight / (1.0 + cv)

    total_reg = lb_loss + z_loss + entropy_reg + diversity_loss

    stats = {
        'moe_load_balance': lb_loss.detach(),
        'moe_lbt_raw':      lb_val.detach() if torch.is_tensor(lb_val) else zero.detach(),
        'moe_entropy':      gate_entropy.detach(),
        'moe_z_loss':       z_loss.detach(),
        'moe_diversity':    diversity_loss.detach(),
        'moe_usage_cv':     cv.detach(),
        'moe_gate_logits_mean': gate_logits.mean().detach() if gate_logits is not None else zero,
        'moe_gate_logits_std':  gate_logits.std(unbiased=False).detach() if gate_logits is not None else zero,
    }
    gates = _get(aux_info, 'gates')
    for idx in range(importance.numel()):
        stats[f'moe_importance_e{idx}'] = importance[idx].detach()
        stats[f'moe_load_e{idx}']       = load[idx].detach()
        stats[f'moe_select_freq_e{idx}'] = sel_freq[idx].detach()
        stats[f'moe_gate_mean_e{idx}'] = (gates[:, idx].mean().detach()
                                          if gates is not None else zero.detach())
    if prob_std is not None:
        stats['moe_router_prob_std'] = prob_std.detach()
    return total_reg, stats


def compute_expert_diversity(model, aux_info, device):
    """专家多样性/使用诊断，返回 {指标名: tensor}，**只报指标不返回损失**。

    - ``moe_expert_cos_mean``：路由专家参数展平后的平均成对余弦（越低越多样）
    - ``moe_expert_param_cv``：各专家参数范数的变异系数（越大容量分配越不均）
    - ``moe_expert_out_norm_e{i}`` / ``moe_router_window_consistency``：
      由 MoE 的 ``expert_stats_interval`` 采集后经 ``aux_info['expert_diag']`` 透传

    异构专家（``expert_kind='heterogeneous'``）的参数布局不同：余弦只在尺寸
    相同的专家对上统计，参数范数 CV 只在容量一致（同尺寸）时报告，避免把设计
    差异误读为"使用不均"。由 ``main.py`` 按 ``--moe_expert_stats_interval`` 采样
    调用；专家臂关闭（experts_mode='none'/'shared_only'）时只返回诊断透传项。
    """
    stats = {}
    experts = getattr(model, 'experts', None)
    if experts is not None and len(experts) > 1:
        with torch.no_grad():
            flats = []
            for module in experts.values():
                params = [p.detach().reshape(-1) for p in module.parameters()]
                flats.append(torch.cat(params) if params else torch.zeros(1))
            sizes = [int(f.numel()) for f in flats]
            normed = [f / f.norm().clamp_min(1e-8) for f in flats]
            sims = [torch.dot(normed[i], normed[j])
                    for i in range(len(normed)) for j in range(i + 1, len(normed))
                    if sizes[i] == sizes[j]]
            if sims:
                stats['moe_expert_cos_mean'] = torch.stack(sims).mean().to(device)
            if len(set(sizes)) == 1:
                norms = torch.stack([f.norm() for f in flats])
                stats['moe_expert_param_cv'] = (
                    norms.std(unbiased=False) / norms.mean().clamp_min(1e-8)).to(device)

    diag = aux_info.get('expert_diag', None) if isinstance(aux_info, dict) else None
    if isinstance(diag, dict):
        for k, v in diag.items():
            stats[f'moe_{k}'] = torch.as_tensor(float(v), device=device)
    return stats
