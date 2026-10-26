# 必要预测与参数输入

- `run_20261026/`：当前 HC 参照模型的 40 人 K64→H32 抽样预测、真值、图结构、重要性与 manifest。该目录虽然生成较早，仍是最新版图表和预测指标汇总实际使用的唯一源预测，不是重复旧版本。
- `profile_controls/`：4475 人、32 维 profile 的独立重提取与 extraction_diagnostic.json。它与原参数组间表存在已记录的复现差异，不应混作原显著性检验的输入。

已删除 `complete_20261026/` 的滚动逐时点缓存、分片副本和逐被试正负扰动缓存。
滚动 val/test 的必要聚合数据迁到 `../prediction_metrics_20261026/rolling_summary/`。
扰动的最终逐被试 early/late 传播矩阵、汇总与图表保留在 `../paper_figures/complete_20261026/`；
原扰动 manifest 保留在该目录的 `sources/perturbation_manifest.json`。
