# coding=utf-8
"""消融变体注册表：文档 1.1-1.8 各结构模块的对照组清单。

变体字段说明：
- id:               实验唯一短 ID（registry 主键之一，用于目录名与检索）
- group:            实验组（同组变体在对比报告中一起呈现）
- doc_ref:          对应 NeuroTwin_当前问题与修复.md 的小节号
- priority:         P0（核心收益确认）/ P1（次要）/ P2（补充）
- description:      变体含义
- requires_pretrain: True 时结构性改动与预训练权重形状不兼容（load_backbone_weights
                    会静默 skip shape 不匹配的键），必须先自预训练再微调；
                    False 时复用 checkpoints/neurotwin_pretrain_pred1/base_best.pt
- overrides:        相对 BASE_ARGS 的参数覆盖
- eval_note:        评估时需要关注的附加指标

已知限制（文档中无现成开关，留待单独开发，不编造参数）：
- 1.7 BrainMDM 逐 branch removal（时间/空间/窗注意力分支独立移除）
- 1.3 GraphODE SDE 分支对照
- 1.4 异构专家（moe_expert_kind 有开关但文档要求异构设计后再对照）/ token 级路由
- 2.3 协变量条件化与 site split 接入
"""

BASELINE_ID = 'baseline'


def _v(exp_id, group, doc_ref, priority, description, overrides,
       requires_pretrain=False, eval_note=''):
    return dict(
        id=exp_id, group=group, doc_ref=doc_ref, priority=priority,
        description=description, requires_pretrain=requires_pretrain,
        overrides=overrides, eval_note=eval_note)


VARIANTS = [
    # ---- 基线（复跑当前最优配置，作为 ΔPCC/ΔMAE 的参照） ----
    _v(BASELINE_ID, 'G0_BASE', '0', 'P0',
       '基线：与 Finetune_MDD.sh 当前默认一致'
       '（2026-09-25 依消融结论调整：ode_steps=3 / delta_refiner_rounds=1 / routed_only；'
       '此前 pred1 基线记录见 registry 历史）',
       {}, requires_pretrain=False,
       eval_note='对比报告中作为参照行'),

    # ---- 1.1 病理条件化 ----
    _v('patho_zquad', 'G11_PATHO', '1.1', 'P1',
       'HAMD 归一化改为 zscore+二次特征',
       {'pathology_norm_mode': 'zscore_quadratic'},
       eval_note='对比 robust_z 基线的 HAMD 分层效应'),
    _v('patho_cdf', 'G11_PATHO', '1.1', 'P1',
       'HAMD 归一化改为经验 CDF',
       {'pathology_norm_mode': 'empirical_cdf'},
       eval_note='对比 robust_z 基线的 HAMD 分层效应'),
    _v('cond_residual', 'G11_PATHO', '1.1', 'P1',
       '条件化仅保留残差级（MoE）注入，关闭主干 AdaLN',
       {'patho_cond_layer': 'residual_only'},
       eval_note='HAMD 分层效应'),
    _v('cond_feature', 'G11_PATHO', '1.1', 'P1',
       '条件化仅保留主干特征级 AdaLN，关闭 MoE 残差条件',
       {'patho_cond_layer': 'feature_only'},
       eval_note='HAMD 分层效应；MoE 路由统计无意义（残差分支关闭）'),

    # ---- 1.2 SC 软先验 ----
    _v('sc_fixed', 'G12_SC', '1.2', 'P0',
       'SC 退化为固定结构先验（scaled+hard mask）',
       {'sc_prior_mode': 'scaled', 'sc_mask_mode': 'hard'},
       eval_note='图稀疏度/熵；FC 指标（edge-PCC/上三角 MAE）'),
    _v('sc_adaptive_only', 'G12_SC', '1.2', 'P0',
       'SC 改为纯自适应功能邻接（无二值掩码先验、不注入 refiner）',
       {'sc_prior_mode': 'adaptive_only', 'sc_mask_mode': 'none',
        'sc_refiner_inject': 'none'},
       eval_note='图稀疏度/熵；FC 指标'),
    _v('sc_no_inject', 'G12_SC', '1.2', 'P1',
       '关闭 SC 先验向 refiner 的注入（仅保留主干软先验）',
       {'sc_refiner_inject': 'none'},
       eval_note='FC 指标'),
    _v('sc_lambda_sample', 'G12_SC', '1.2', 'P1',
       'SC 融合系数 lambda 改为样本级动态',
       {'sc_lambda_mode': 'sample'},
       eval_note='lambda 统计量分布'),
    _v('sc_lambda_roi', 'G12_SC', '1.2', 'P1',
       'SC 融合系数 lambda 改为 ROI 级',
       {'sc_lambda_mode': 'roi'},
       eval_note='lambda 统计量分布'),
    _v('sc_no_delta_a', 'G12_SC', '1.2', 'P1',
       '关闭 SC 低秩增量 A_delta（仅静态先验）',
       {'sc_delta_a': False},
       eval_note='FC 指标'),

    # ---- 1.3 GraphODE ----
    _v('ode_s1', 'G13_ODE', '1.3', 'P0',
       'ODE 积分步数减为 1',
       {'ode_steps': 1}, eval_note=''),
    _v('ode_s3', 'G13_ODE', '1.3', 'P0',
       'ODE 积分步数减为 3',
       {'ode_steps': 3}, eval_note='注意：2026-09-25 起新基线已采用 steps=3，本变体与基线重合（历史记录仍有效）'),
    _v('ode_s12', 'G13_ODE', '1.3', 'P0',
       'ODE 积分步数增至 12',
       {'ode_steps': 12}, eval_note=''),
    _v('ode_rk4', 'G13_ODE', '1.3', 'P1',
       'ODE 求解器改为 rk4',
       {'ode_solver': 'rk4'}, eval_note='训练耗时对比'),

    # ---- 1.4 MoE ----
    _v('moe_none', 'G14_MOE', '1.4', 'P0',
       '关闭 MoE（无专家）',
       {'moe_experts_mode': 'none'},
       eval_note='gate 统计无意义'),
    _v('moe_shared_only', 'G14_MOE', '1.4', 'P0',
       '仅保留 shared expert（无路由专家）',
       {'moe_experts_mode': 'shared_only'},
       eval_note='gate 统计无意义'),
    _v('moe_routed_only', 'G14_MOE', '1.4', 'P0',
       '仅保留路由专家（无 shared expert）',
       {'moe_experts_mode': 'routed_only'},
       eval_note='注意：2026-09-25 起新基线已采用 routed_only，本变体与基线重合（历史记录仍有效）'),
    _v('moe_gate_difficulty', 'G14_MOE', '1.4', 'P1',
       '门控特征加入难度项（state_revin_difficulty）',
       {'moe_gate_features': 'state_revin_difficulty'},
       eval_note='gate 统计；top1 分配分布'),

    # ---- 1.4 路由坍缩对策（2026-09-26，P0 报告建议；基于 multinomial 修复后的新基线） ----
    _v('moe_exp2_t1', 'G14_MOE', '1.4', 'P0',
       '专家数 4→2、top_k 2→1（对齐实际有效容量 + 硬单选路由）',
       {'num_experts': 2, 'top_k': 1},
       eval_note='gate 统计；top1 分配分布；normalized entropy'),
    _v('moe_argmax', 'G14_MOE', '1.4', 'P0',
       '训练期硬确定性路由（use_argmax，容量感知 argmax 取代概率采样）',
       {'moe_use_argmax': True},
       eval_note='gate 统计；条件敏感性对照（multinomial 修复的消融）'),
    _v('moe_temp_low', 'G14_MOE', '1.4', 'P0',
       '路由温度下调（gate_temp 1.5→1.0 起、1.0→0.5 终），锐化概率分布配合 multinomial 采样',
       {'moe_gate_temp_start': 1.0, 'moe_gate_temp_end': 0.5},
       eval_note='gate 统计；路由熵；top1 集中度'),

    # ---- 1.5 预测头分支 ----
    _v('head_flatten', 'G15_HEAD', '1.5', 'P0',
       '形状分支退化为 flatten 展平（关闭 FutureQueryDecoder）',
       {'head_shape_mode': 'flatten'}, requires_pretrain=True,
       eval_note=''),
    # ---- 1.5 head_flatten 预训练混淆对照（2026-09-26，P0 报告建议） ----
    _v('head_shape_scratch', 'G15_HEAD', '1.5', 'P0',
       'baseline 结构但 shape 分支（future_query.*）强制随机初始化——'
       '隔离「shape 分支未预训练」混淆：与 baseline 差 = 预训练贡献，'
       '与 head_flatten 差 ≈ 0 则 flatten 损失主要是未预训练混淆',
       {'pretrained_skip_pattern': 'future_query.*'}, requires_pretrain=True,
       eval_note='核对日志中「受控跳过」键数与 missing 一致'),
    _v('head_no_history', 'G15_HEAD', '1.5', 'P1',
       '关闭 history_proj 分支',
       {'head_use_history_proj': False}, eval_note=''),
    _v('head_no_latent', 'G15_HEAD', '1.5', 'P1',
       '关闭 latent_proj 分支',
       {'head_use_latent_proj': False}, eval_note=''),
    _v('head_no_temporal', 'G15_HEAD', '1.5', 'P1',
       '关闭 temporal 分支',
       {'head_use_temporal': False}, eval_note=''),
    _v('head_no_cross_roi', 'G15_HEAD', '1.5', 'P1',
       '关闭 cross_roi 分支',
       {'head_use_cross_roi': False}, eval_note=''),
    _v('head_no_win_attn', 'G15_HEAD', '1.5', 'P1',
       '关闭因果跨窗注意力分支',
       {'head_use_win_attn': False}, eval_note=''),
    _v('ode_no_win_attn', 'G15_HEAD', '1.5', 'P1',
       '关闭 ODE 级窗口注意力',
       {'ode_window_attn': 'off'}, eval_note=''),

    # ---- 1.6 迭代细化 ----
    _v('refiner_1round', 'G16_REFINER', '1.6', 'P0',
       'base refiner 减为 1 轮',
       {'refiner_rounds': 1}, eval_note=''),
    _v('delta_1round', 'G16_REFINER', '1.6', 'P0',
       'MoE delta refiner 减为 1 轮',
       {'delta_refiner_rounds': 1},
       eval_note='注意：2026-09-25 起新基线已采用 1 轮，本变体与基线重合（历史记录仍有效）'),
    _v('refiner_none', 'G16_REFINER', '1.6', 'P0',
       '关闭 base refiner（refiner_rounds=0，依赖 neurotwin.py 的 0 轮支持）',
       {'refiner_rounds': 0}, eval_note=''),
    # TODO(1.6): delta_refiner_rounds=0（关闭 MoE 残差 refiner）需 models/neurotwin_moe.py
    # 同样支持 0 轮，当前未实现，不编造参数。

    # ---- 1.7 BrainMDM 多尺度 ----
    _v('mdm_s1', 'G17_MDM', '1.7', 'P1',
       'BrainMDM 尺度数减为 1',
       {'num_scales': 1}, eval_note=''),
    _v('mdm_s2', 'G17_MDM', '1.7', 'P1',
       'BrainMDM 尺度数减为 2',
       {'num_scales': 2}, eval_note=''),
    _v('mdm_gate_none', 'G17_MDM', '1.7', 'P1',
       '关闭 BrainMDM 尺度门控',
       {'mdm_scale_gate': 'none'}, eval_note=''),
    # TODO(1.7): BrainMDM 逐 branch removal（时间/空间/窗注意力分支独立移除）
    # 需在 BrainMDM 增加 per-branch 开关，当前无现成参数。

    # ---- 1.8 不确定性 / 反演 ----
    _v('head_point', 'G18_UNCERT', '1.8', 'P2',
       '点预测头（无 gaussian 不确定度）',
       {'pred_head': 'point'}, requires_pretrain=True,
       eval_note='无 PICP/CRPS 指标'),
    _v('head_quantile', 'G18_UNCERT', '1.8', 'P2',
       '分位数预测头（pinball 损失）',
       {'pred_head': 'quantile'}, requires_pretrain=True,
       eval_note='PICP 口径与 gaussian 不同，需注明'),
    _v('inversion', 'G18_UNCERT', '1.8', 'P2',
       '启用辅助反演头（inversion_weight=0.5）',
       {'inversion_weight': 0.5},
       eval_note='inversion_pred 与病理条件的相关性'),
    # TODO(2.3): 协变量条件化（site/sex/age）与 site split 接入：无现成参数。
]


def get_variant(exp_id):
    """按 ID 取变体；未注册时抛 KeyError。"""
    for v in VARIANTS:
        if v['id'] == exp_id:
            return v
    raise KeyError(f'未注册的实验 ID：{exp_id}（可选：{", ".join(v["id"] for v in VARIANTS)}）')


def filter_variants(group=None, only=None, priority=None):
    """按 组 / ID 列表 / 优先级 过滤变体，保持注册顺序。

    - group: 单个组名（如 'G14_MOE'）
    - only:  逗号分隔或列表形式的实验 ID
    - priority: 'P0'/'P1'/'P2'
    至少提供一个筛选条件时返回交集；全部为空返回完整注册表。
    """
    if only is not None and isinstance(only, str):
        only = [s.strip() for s in only.split(',') if s.strip()]
    selected = VARIANTS
    if only:
        id_set = set(only)
        unknown = id_set - {v['id'] for v in VARIANTS}
        if unknown:
            raise KeyError(f'未注册的实验 ID：{sorted(unknown)}')
        selected = [v for v in selected if v['id'] in id_set]
    if group:
        selected = [v for v in selected if v['group'] == group]
    if priority:
        selected = [v for v in selected if v['priority'] == priority]
    return selected
