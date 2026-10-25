# 最优 H32 结果

- 权重：`checkpoints/final/h32_best.pt`，内容与原H32 best逐字节相同，最佳第149轮。
- [manifest.json](manifest.json)：权重及正式结果文件的SHA256与来源。
- [metrics_val.json](metrics_val.json)、[metrics_test.json](metrics_test.json)：保留原评价的最优H32数值。
- [per_subject_val.csv](per_subject_val.csv)、[per_subject_test.csv](per_subject_test.csv)：每段225人的MAE。
- [source_run_config.json](source_run_config.json)：原run配置，旧输出路径只作来源记录。
- [experiments_summary.json](experiments_summary.json)：压缩后的总实验机器结论；不含训练指令。
- [h8_baseline_metrics_test.json](h8_baseline_metrics_test.json)：H8契约的历史结果，不与H32混用。
- [轨迹指标图](trajectory_metrics.png)、[固定规则轨迹示例](trajectory_examples.png)：保留的评价图。

本目录由已有工件提取/复制生成，没有重新推理、重新选优或重算预测。
淘汰模型、原运行目录、TensorBoard、日志及诊断数组已清理；原路径字符串仅作历史来源标识。
实验限制与负结果见 [EXPERIMENTS.md](../../docs/EXPERIMENTS.md)。
