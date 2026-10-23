# 损失函数实验归档

**任务 A 已结题：首轮研究完成，E2 保留。** L1–L3 不采用，P1 不定型。
原 L1–L3/P1 不再扩展。按用户新请求，仅补充 Temporal PCC 独立对照。

- 正式结论：`docs/EXPERIMENTS.md` §6.1。
- 完整报告：`results/loss/compare/RESULTS.md`。
- 工件索引：`results/loss/archive_manifest.json`。
- 最终产物：六组实际训练配置、指标 JSON、主对比表、配对统计及 Figure 1–5。

已清除本轮临时编排、测试和对比脚本，以及预测/训练参照数组、末轮 checkpoint、
重复评估、过程日志和 TensorBoard 文件。原有公共工具 `compare_phase1.py` 保留供历史实验使用。

训练与评估继续使用项目公共入口 `main.py`、`experiments/evaluate_variant.py` 和
`analysis/s4_probabilistic_eval.py`；实际实验参数保存在各臂 `training_config.json`
中。已结题实验在 `results/` 内的权重副本已清理；候选损失实现保留，正式配置不启用这些候选。

后续概率评分统一采用“逐锚点归一化 → 被试内平均 → 跨被试等权平均”，CSV 与 JSON 一致。
本轮最终概率对比已按此口径计算；历史导出路径仅作为来源记录，部分中间工件已清理。
正式基线配置、S2 权重与最终实验结论保持原样。

## Temporal PCC 补充实验

`temporal_pcc.py` 从已归档 L0 的实际命令读取训练配方。TP0 为 E2（新增项权重 0），
TP1 为 E2 + Temporal PCC（权重 0.05）。两臂从同一 HC 起点独立微调，分别使用两张 GPU，
模型、数据划分、随机种子、优化器和训练轮数保持一致。原有 E2 项与自动加权均保留。
新增项固定加权 CPM 与 ODE 的 H1–H8 未来轨迹 PCC 之和，分别沿用各自 mask，
不拼入真实历史帧；每条轨迹至少 3 个有效未来点，真实方差不超过 1e-8 时排除，
预测恒定时仍监督。比较 Temporal PCC、单步/多步 MAE、H1–H2/H4–H8 与幅度分层误差。

入口：`python -m experiments.loss.temporal_pcc`（训练及评估）；
`python -m experiments.loss.temporal_pcc --compare-only`（基于已有预测重新汇总）。
输出写 `results/temporal_pcc/`，正式基线不变，不覆盖旧归档。

Temporal PCC 补充实验已完成，结果见 `results/temporal_pcc/RESULTS.md`。
时间相关性略增，但 ODE/CPM 平均 MAE 略差；不纳入正式 E2。
