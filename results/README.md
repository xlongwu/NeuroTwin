# 保留结果

本目录只保留当前研究需要的最终结果、必要图表输入与轻量溯源数据。
本次清理未训练、未重新推理、未改动指标数值，未修改原始数据或 checkpoint。

| 目录 | 保留内容 |
|---|---|
| [prediction_metrics_20261026/](prediction_metrics_20261026/README.md) | 预测性能汇总、逐被试指标、val/test 逐 ROI 滚动误差矩阵、微调历史 test 报告 |
| [fc_test_20261026/](fc_test_20261026/README.md) | 完整 test 的 FC 汇总与逐被试 FC 相关性 |
| normative_reference/ | 健康参照 `fit.npz` 与质量门槛报告 |
| [pathology_analysis/run1/](pathology_analysis/run1/README.md) | 偏离矩阵、ROI/网络组间检验、参数与临床关联结果 |
| [paper_figures/complete_20261026/](paper_figures/complete_20261026/README.md) | 唯一保留的完整图表版本：17 张图、PDF、图注、配套数据及必要历史源数据 |
| [paper_infer/](paper_infer/README.md) | 图表实际使用的 40 人预测、重要性结果和独立 profile 重提取记录 |
| [paper_stats/](paper_stats/README.md) | 唯一保留的图表统计、临床残差和 OOF 预测结果 |

已清除旧图版、旧 `final/` 目录、重复统计、日志、滚动分片与逐时点缓存、逐被试扰动计算缓存。
保留内容不是“最新结果自动覆盖旧结果”的机制；后续新计算仍使用独立目录。

迁移与删除范围见 [cleanup_summary.json](cleanup_summary.json)。原始 manifest 保留当时的来源路径，
实际保留位置通过清理记录映射；其中历史校验记录只说明生成当时的状态。
需要重新计算或重绘依赖逐时点缓存的图时，先按源代码重新生成缓存；当前保留的图像与配套数据可直接查看。
