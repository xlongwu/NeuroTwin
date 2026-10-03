# coding=utf-8
"""NeuroTwin 对比实验（消融）框架。

对应 docs/Version1_docs_0921/NeuroTwin_当前问题与修复.md 第 1 节（1.1-1.8）验收矩阵，
提供：变体注册表 -> 批量训练 -> 轻量评估 -> 集中登记 -> 对比报告 的最小闭环。

模块组成：
- base_config.py     基线配置（镜像 scripts/Finetune_MDD_next_point.sh，需同步维护）
- variants.py        变体注册表（文档 1.1-1.8 对照组）
- run_experiments.py 批量执行入口 CLI
- evaluate_variant.py 轻量评估（val/test + HAMD 分层 + shuffled 负对照 + FC 指标）
- registry.py        结果登记（results/ablation/registry.jsonl + registry.csv）
- compare.py         对比 Markdown 报告生成
"""
