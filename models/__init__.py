# coding=utf-8
"""NeuroTwin 模型包。

唯一主干为 NeuroTwinTFM（models/tfm.py，TimesFM-3 风格）。

ARCH_VERSION 用于检查点与当前代码的结构兼容性校验：
任何会改变参数集合或参数语义的结构改动都必须递增该版本号。

版本历史：
  1 — 初始架构（无版本元数据的旧检查点按此处理）
  2 — 病理条件化协议修订（AdaLN / 软 SC 先验 / 新预测头 / 检查点元数据）
  3 — 预测头查询式解码、GraphODE 求解器与可学习步长、MoE 门控与专家臂结构调整
      （v2 及更早的权重会被 load_backbone_weights 拦截，需重新预训练）
  4 — legacy 主干的概率输出头 / 辅助反演头（均已随 legacy 移除）；
      本版本同时覆盖 NeuroTwinTFM 主干的参数集合
      注：TFM 病理残差专家（tfm_patho_mode=moe，RECC 方案 §92）仅在 finetune
      阶段创建，HC 预训练结构与参数集合不变，故预训练 arch_version 仍为 4；
      finetune checkpoint 的结构由 config.args 快照重建（evaluate_variant）。
  5 — 条件注入 v1 revised（20261008）：新增逐字段独立编码器、E_demo/DeltaE 融合、
      Condition Dropout、双头 FiLM 的 λ 开关与主干条件 RMSNorm（AdaRMSNorm）。
      **HC 预训练也允许构建条件模块**，因此预训练 checkpoint 的参数集合可能
      包含条件模块（arm=none 时仍与 v4 完全一致）；条件模块的字段顺序、词表与
      统计量由 checkpoint 内的 layout_hash / schema_version 校验。
      v4 的无条件 checkpoint 需经 train.optim.migrate_pretrained_checkpoint 显式
      迁移（新增条件键记为 new_initialized / zero_expanded），不允许靠
      strict=False + shape 过滤静默忽略。
  6 — graph_ode 动态路线（20261012）：NeuroTwinTFM 新增 dynamics_mode /
      ode_steps / dynamics_dt / graph_source / personalized / profile_dim
      结构参数；graph_ode 分支构建 GraphODEDynamicsHead（含 VarianceHead）、
      personalized 时构建 ProfileEncoder/PersonalFiLM，discrete 分支结构与
      v5 完全一致。旧 checkpoint（缺新字段）按 discrete/personalized=False/
      legacy_mask 重建，不得将旧 one-step 头权重迁移到 ODE 头。
"""
ARCH_VERSION = 6

__all__ = ['ARCH_VERSION']
