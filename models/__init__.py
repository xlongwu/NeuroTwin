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
"""

ARCH_VERSION = 4

__all__ = ['ARCH_VERSION']
