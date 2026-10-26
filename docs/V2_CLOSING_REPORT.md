# v2 主线收档报告：HC 参照与病理异常定位

收档日期：2026-10-25。Phase 0–2 与 T2 扩展全部执行完毕；站点解析规则已人工
确认（计数表与 2026-09-22 独立实现零差异）。主报告与数据见
[results/pathology_analysis/run1](../results/pathology_analysis/run1/README.md)。

## 一、主结论（阴性）

在预登记口径（HC 参照模型、lag-1 滚动误差、K=64、站点匹配 1504 对）下，
**MDD 相对 HC 不存在可定位的组间偏离效应**：

- 模型 z 偏离：0/116 ROI 过 FDR（max|Δz|=0.090，q≥0.34）；网络级与全脑
  均不显著（全脑 Δz=−0.023，p=0.11）；离散度无差异（方差比 0.98，p=0.27）。
- 六描述量基线与静态 FC+岭回归（OOF AUC=0.560）同样无定位效应——
  **队列可分性本身弱**，不是模型单方失败（护栏 improvement 86.5%）。
- 条件使用（Sex/Age/education FiLM）：近零（Δpred/信号尺度中位 0.46%，
  零化条件 MAE 反而略降），按预登记即为最终结论。

## 二、站点混杂的实证（本报告的核心方法论结果）

- 未匹配比较：全脑 Δz=+0.07，Top ROI 为楔后叶/扣带回/海马旁/眶额内侧等
  "MDD 经典脑区"——表面可用。
- 站点匹配后：正向偏移完全消失（−0.02），Top ROI 换成另一批分散小效应。
- 负控（HC val vs test，仅站点构成不同）：逐 ROI |Δz| 中位 0.051、最大
  0.176——**与未匹配 MDD 效应同量级**。
- 结论：未匹配的正向"效应"由站点构成混杂解释。REST-meta-MDD 的 HC/MDD
  站点分布不对称 + HC 划分未按站点分层，两层因素都会制造假组间差异；
  预登记的负控 + 匹配设计将其完整拦截。任何后续组间分析应默认沿用。

## 三、弱线索（不构成结论，仅供未来参考）

1. 个体 profile 32 维中 1 维 FDR 显著（Δ=+0.0056，q=0.013；范数差 −0.03%，
   p=0.039）——方向性存在但效应量极小，且 profile 99.4% 能量集中在单一
   共享主方向上。
2. LN 网络 z 与 HAMA 偏相关 r=−0.135（p=0.008，n=422 完整观测；未多重
   校正）。HAMA 有效率仅 25.3%、缺失机制未核实，深挖前须先完成来源确认。

## 四、阴性结论的边界

限定于：该 checkpoint（单 seed）、lag-1 误差口径、首 64 TR 描述/FC 窗、
静止态、被试级横断比较。不排除：更长时程误差、状态转移/动态 FC、
纵向或症状分层口径下存在效应。重启任一方向需新的预登记文档与门槛设计，
不在本报告范围内自动触发。

## 五、证据链（数值 → 产物）

| 环节 | 关键数值 | 产物 |
|---|---|---|
| 护栏 B4 | improvement 86.5%，p=1e-4 | `results/normative_reference/gate_report.json` |
| 全量评估 | 4475 人 / 480,320 锚点，improvement 86.6–86.9% | `results/rolling_eval/`（含 manifest） |
| 参照与 z | HC train 1778 拟合；hc train 为 10 折 cross-fitted | `fit.npz`、`run1/z_maps/` |
| 条件使用 | Δpred/信号尺度中位 0.46% | `results/pathology_analysis/condition_usage.json` |
| 组间检验 | 0/116 过 FDR；负控 0 显著 | `run1/group_tests.json`、`roi_fdr_table.csv` |
| 并排基线 | 描述量 0/116；FC AUC 0.560 | `roi_fdr_table_descriptor.csv`、group_tests.json |
| T2 参数分析 | 1/32 维 q=0.013 | `run1/param_group_tests.json` |
| 临床关联 | HAMD 无；LN↔HAMA r=−0.135 | `run1/clinical_partial_corr.csv` |
| 图 | 四联汇总 | `run1/v2_results_summary.png` |

复现命令链见 [run1/README.md](../results/pathology_analysis/run1/README.md)。
统计实现与测试：`analysis/pathology_tests.py`、`analysis/param_group_tests.py`
（tests/ 共 25 项新增单测覆盖置换、FDR、bootstrap、偏相关）。
