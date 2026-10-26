# v2 run1：HC 参照 + 病理异常定位（完整执行记录）

日期：2026-10-25。结论：[docs/V2_CLOSING_REPORT.md](../../../docs/V2_CLOSING_REPORT.md)。
主分析模型：`checkpoints/neurotwin_pretrain/base_best.pt`
（SHA256 `0e252be1af1750b35bd26a2112f66f6a0cb2ad6fa4249b7334d9cffdbb78d255`）。

## 复现命令（按序）

```bash
python -m utils.site_subset --emit-count-table && python -m utils.site_subset --match
# 滚动单步评估（hc val 先行过 gate，再全量）
python -m analysis.rolling_one_step --ckpt <CKPT> --group hc --split val  --device cuda:0
python -m analysis.normative_deviation gate --rolling-dir results/rolling_eval --split val
for g in hc mdd; do for s in train val test; do
  python -m analysis.rolling_one_step --ckpt <CKPT> --group $g --split $s --device cuda:0; done; done
python -m analysis.normative_deviation fit  --rolling-dir results/rolling_eval
python -m analysis.normative_deviation score --rolling-dir results/rolling_eval
python -m analysis.condition_usage --ckpt <CKPT> --split val --device cuda:0
python -m analysis.pathology_tests
python -m analysis.param_group_tests
python -m analysis.visualize_v2_results
```

种子与划分：模型 seed=2024 / sampling 20261009（checkpoint 快照）；置换与
bootstrap seed=2024；HC 划分 `subject_split_pretrain.json`（split_hash
`4aa135c007e75fe4`，与 checkpoint 一致）；MDD 划分 `subject_split_finetune.json`；
匹配子集 `data/splits/site_matched_subset.json`（36 站 1504 对，Age±5/Sex 同层）。

## 结论速览（详见 ../../docs/EXPERIMENTS.md「v2 主线结果」）

| 环节 | 结果 |
|---|---|
| B4 护栏 | 通过：HC val improvement 86.5%，配对置换 p=1e-4 |
| B5 条件使用 | 近零（Δpred/信号尺度中位 0.46%）→ 最终结论 |
| B7 模型 z | 0/116 ROI 过 FDR；网络/全脑不显著；负控通过；离散度无差异 |
| 六描述量基线 | 0/116 过 FDR（T2：模型无增量，已触发） |
| 静态 FC+岭回归 | 站内 CV OOF AUC=0.560 |
| T2 参数组间 | profile 32 维中 1 维 q=0.013（效应量极小） |
| 临床关联 | HAMD 无关联；LN↔HAMA r=−0.135（p=0.008，n=422，描述性） |

## 目录内容

- `z_maps/`：6 组 z 矩阵（hc train 为 10 折 cross-fitted）+ 逐被试 CSV
- `roi_fdr_table.csv` / `roi_fdr_table_descriptor.csv`：模型 / 描述量的逐 ROI 检验表
- `group_tests.json`：B7 主报告（含负控、离散度、FC 基线、T2 判定、临床关联）
- `param_group_tests.json` / `param_dim_table.csv`：T2 扩展（MINDy 式参数组间）
- `clinical_partial_corr.csv`、`v2_results_summary.png`
- 相关：`../../normative_reference/`（gate_report.json、fit.npz）、
  `../../rolling_eval/`（833 MB 逐被试缓存 + manifest）、`../../condition_usage.json`

## 已知限制

阴性结论限定于：lag-1 误差口径、K=64/首 64 TR 窗、单 seed、该 checkpoint、
静止态短窗；未匹配比较的正向偏移由站点混杂解释（负控 Δz≈0.04–0.18 同量级）。

## 当前保留范围

健康参照拟合、6 组偏离矩阵、组间/网络/临床检验、参数表与本目录图保留。
原逐时点滚动缓存已清理；val/test 的必要聚合表与逐 ROI MAE 矩阵位于
`results/prediction_metrics_20261026/rolling_summary/`。
上述复现命令中的完整训练/验证/测试缓存需要按需重新生成，不表示缓存目前仍全部存在。
