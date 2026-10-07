# coding=utf-8
"""统一评估入口包。

历史沿革：本包曾是消融实验框架（变体注册表 -> 批量训练 -> 集中登记 -> 对比报告），
G30_TFM 消融完成后框架已归档并移除（归档提交 a673ca5），仅保留评估入口。

模块组成：
- evaluate_variant.py 从 checkpoint 重建模型并评估（next_timepoint 口径），
  供 main.py 训练后自动评估与手动复评共用。
"""
