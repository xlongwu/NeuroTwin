# 保留的完整图表

这是本次清理后唯一保留的图表版本，共 16 张主图与 1 张附图。
[图览](overview.png)、[报告与逐图说明](REPORT.md)、[中文图注](CAPTIONS_ZH.md)及每张图的 PDF/PNG/数据均保留。

- 图像、图注和统计数据未改动。`REPORT.md`、`CHECKS.json`、manifest 是生成当时的记录。
- `sources/h32/` 保留历史 H32 对照图实际需要的轻量指标与源记录。
- `sources/perturbation_manifest.json` 记录已清理扰动缓存的协议；最终传播数据仍为 `f14_propagation.npz`。
- `sources/previous_checks/` 保存前版交付的来源与校验记录，用于历史溯源，不代表旧缓存当前仍存在。

旧根目录图像与 `run_20261026/` 图版已经删除。
完整流水线重算需要重新生成滚动及扰动缓存；不能把历史检查记录当作清理后重新执行的检查。
