# 独立 test 划分的预测序列 FC 评价

模型固定为 `checkpoints/neurotwin_pretrain/base_best.pt`，
不训练、不调参，不覆盖旧结果。权重 SHA256、划分文件 SHA256 与评价口径
分别记录在 `hc/fc_summary_test.json` 和 `mdd/fc_summary_test.json`。
推理使用两张 NVIDIA RTX A6000：HC 在 `cuda:0`，MDD 在 `cuda:1`。

## 实测结果

| Test 组别 | 被试数 | FC Pearson 均值 ± 标准差 | FC Pearson 中位数 | FC MAE |
|---|---:|---:|---:|---:|
| HC | 222 | 0.997633 ± 0.001595 | 0.997912 | 0.013856 |
| MDD | 225 | 0.997384 ± 0.001840 | 0.997821 | 0.014302 |

全部 447 人均有有效 FC 相关，每人全部 6,670 条边有效。
HC 总计 25,264 个预测时间点，MDD 总计 25,366 个预测时间点；
每位被试 86–176 个连续预测点。只汇总本模型，不提供基线 FC 对照。
新增计算代码的轻量验证为 `8 passed`。

## 评价口径

- HC 使用 `data/subject_split_pretrain.json` 的 test（222 人）；
  MDD 使用 `data/subject_split_finetune.json` 的 test（225 人）。
  这是项目已有的被试留出测试集，不是新增的外部队列。
- 完整复用 `analysis.rolling_one_step`：Raw BOLD，最早 32 TR 校准，
  K64 真实可见历史，history_fc 图，训练期条件统计量 inherit，ODE lag-1 输出。
- 枚举每位被试全部连续合法锚点，预测与真值在相同目标时间点对齐。
  不把可见历史拼入预测序列，不跨被试拼接序列。
- 各自沿时间计算 116 ROI 的有符号 Pearson FC，取严格上三角 6,670 条边，
  再在每位被试内计算两组 FC 边向量的 Pearson 相关。
  总体均值与标准差采用被试等权；标准差为总体标准差。
- 恒定 ROI 关联边无定义并排除；无定义的相关性在 CSV 中记 NaN，
  汇总记录有效被试数。另给出模型 FC MAE、RMSE。
- 这里的 FC 相关是**单个被试内跨边相关**，不同于既有评价中的
  `edge_pcc_mean`（逐边跨任务相关再平均）。

**解释边界：每一步都输入真实历史，本评价衡量滚动单步预测序列的 FC 保真度。
不代表固定起点、不再观察真实未来的自由生成 FC 保真度。**

## 复现

在项目根目录运行。逐时点滚动缓存已清理，以下命令会在新目录重新生成；当前指标与逐被试 FC 表仍保留。

```bash
python -m analysis.rolling_one_step \
  --ckpt checkpoints/neurotwin_pretrain/base_best.pt \
  --group hc --split test --device cuda:0 \
  --out_dir results/recomputed_fc/rolling

python -m analysis.rolling_one_step \
  --ckpt checkpoints/neurotwin_pretrain/base_best.pt \
  --group mdd --split test --device cuda:1 \
  --out_dir results/recomputed_fc/rolling

python -m analysis.fc_sequence_eval \
  --cache-dir results/recomputed_fc/rolling/hc \
  --split-file data/subject_split_pretrain.json --split test \
  --out-dir results/recomputed_fc/hc

python -m analysis.fc_sequence_eval \
  --cache-dir results/recomputed_fc/rolling/mdd \
  --split-file data/subject_split_finetune.json --split test \
  --out-dir results/recomputed_fc/mdd

python -m pytest -q tests/test_fc_sequence_eval.py
```

## 产物

- `hc/fc_summary_test.json`、`mdd/fc_summary_test.json`：逐组汇总。
- `hc/per_subject_fc_test.csv`、`mdd/per_subject_fc_test.csv`：逐被试结果。
- 逐时点预测缓存已删除；原滚动评价的汇总、逐被试表、逐 ROI MAE 矩阵及 manifest 迁至
  `../prediction_metrics_20261026/rolling_summary/test/`。
