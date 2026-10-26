# NeuroTwin 论文级分析方案：模型性能 / 病理 / 可解释性 + 可视化

撰写日期：2026-10-25。定位：服务于当前已确认的方向 A（写作/复盘），把
v2 收档结论与 H32 历史结果组织成论文级分析与图表。**全部条目只读权重与
数据，不改 `models/`、`train/`、`configs/`、权重和既有结果目录**；新产物
一律写入新目录（默认 `results/paper_figures/` 与 `results/paper_stats/`）。

边界遵循 [PROJECT_STATUS.md](PROJECT_STATUS.md)：关闭清单不重启；任何产生
**新组间推断口径**的分析仍需预登记（门槛 + 站点匹配 + 负控三件套）并经
用户确认。本方案中凡属此类均标注 **[需预登记/需确认]**，其余均为已有结果
的重组织、可视化与描述性/结构性分析。

---

## 1. 文献调研：数字孪生脑相关研究做了哪些分析

本项目的定位是"数据驱动数字孪生脑"（个体化、从 BOLD 历史学习动力学并做
规范性病理偏离）。以下五类文献构成直接参照系。

### 1.1 Digital Twin Brain 平台（复旦 冯建峰组）

- **DTB 平台论文**（[arXiv 2308.01241]）：以 86B 神经元/47.8T 突触模拟为核心。分析包括：
  (a) 模拟性能——time-to-solution、CUDA vs CPU 数值一致性（spike 时刻一致、
  相对误差 <1e-4）、弱/强扩展曲线；
  (b) **数据同化（HMDA）**——Balloon-Windkessel 生成 BOLD 与实证 fMRI 拟合；
  情绪图片任务同化：输入区 BOLD 相关 >0.98，皮层体素平均 Pearson r=0.75；
  (c) **行为预测**——GLM 激活 + LASSO 预测被试评分，r=0.655（p=0.006）。
  可视化：放电率脑图、spike raster、任务 BOLD 实测 vs 同化时间序列叠加、
  体素级相关散点、预测 vs 实际评分散点、扩展性曲线。
- **Lu et al. 2024, NSR**（[Simulation and assimilation of the digital human brain]）：
  提出"仿真 + 同化"闭环框架，静息态活动复现、任务态模仿，FC 与实证对比是
  核心验证口径（全文 403 无法直接抓取，以上为检索摘要口径，引用前需核对原文）。
- **Takahashi et al. 2025, npj Digital Medicine**（[Digital twin brain simulator for
  real-time consciousness]）：与本项目的分析结构最接近的数据驱动孪生——
  V-RNN 孪生生成宽域 ECoG。分析包括：
  (a) **生成质量**——PSD 与 FC 矩阵和实测信号的相似度；
  (b) **潜状态结构**——三层潜变量（脑区/网络/全局）自组织出麻醉 vs 清醒
  聚类（k-NN 命中 0.78/0.68，轮廓宽度 p<2.2e-16）；
  (c) **虚拟干预**——"虚拟给药"（设定全局潜状态）后生成信号被判别器分类为
  麻醉/清醒 0.89/0.96；层级干预因果分析（z(2)→d(1)）；
  (d) **信息流**——传递熵量化网络对脑区的控制；
  (e) **诚实的负结果报告**——个体识别率 0.07–0.13 低于随机，明确可辨识性
  局限。可视化：波形叠加、潜状态时间序列、聚类散点、传递熵热图 + 脑图、
  KDE 密度图。
- **Xiong et al. 2023**（[DTB 愿景论文]）：定位性综述，DTB 作为生物与人工
  智能的桥梁。

### 1.2 The Virtual Brain / Virtual Epileptic Patient（Jirsa 组，机制型孪生）

- **Jirsa et al. 2023, Lancet Neurology**（[Personalised virtual brain models in
  epilepsy]）：个性化虚拟脑建模综述；TVB 是首个进入临床试验（EPINOV）的
  预测性脑建模技术。
- **Makhalova et al. 2022, eLife**（[Virtual epileptic patient brain modeling]）：
  VEP 估计致痫网络（EZN），与 sEEG 的 Epileptogenicity Index 对照验证，
  并做**虚拟切除**与手术结局预测。
- **Wang et al. 2025, Nat Commun Med**（[Virtual brain twins for stimulation in
  epilepsy]）：个性化全脑模型估计 EZN 并指导刺激。
- 分析范式：**以临床金标准（sEEG SOZ/术后结局）为外部锚点的生物标志物
  验证 + 虚拟干预（虚拟切除/刺激）**。该范式对本项目的启示：病理分析需要
  外部锚点与干预实验才有定位性说服力；本项目缺这两者，是阴性结论边界的一
  部分（见 §4 前言）。

### 1.3 MINDy：数据驱动个体化动力学模型（Singh 组）

- **Singh et al. 2020, NeuroImage**（[MINDy]）：逐被试拟合耦合动力学模型，
  分析个体化静息活动预测、模型参数作为个体表型的组间比较；扩展到 EEG
  意识障碍（[Schwamb et al. 2024]）。本项目 T2（MINDy 式参数组间，profile
  32 维）已执行：1 维 q=0.013、效应量极小——参数线结论与 MINDy 文献中
  "参数可作表型"的前提在本队列不成立（或效应被站点噪声淹没）。

### 1.4 Normative modeling（规范性建模，Marquand / PCNtoolkit 系）

- **Marquand et al. 2016**（[heterogeneity framework]）与 **Rutherford et al.
  2022/2023**（[PCNtoolkit]）：以健康参照拟合 μ(x)、σ(x)，输出逐区域逐被试
  z/centile 偏离分数；标准分析：区域级效应量（Cohen's d）、极端偏离统计、
  站点校准 QC（每站点 ≥30 匹配对照、重复校准稳定性分布）。
- **Practical guide 2025**（[arXiv 2509.07237]）：标准可视化 = 个体区域
  z 剖面条形图、效应量脑图、跨模型效应量散点（对角线一致性）、重复校准
  的效应量分布；明确"扫描仪/站点错配时效应量不可靠"。
- 本项目 v2 的 z 偏离（HC train 拟合 + 人口学校正 + conformal 分位）正是该
  范式；**站点混杂实证（§4 D2）本身即一篇方法学贡献**，与 Gaiser et al.
  2024（偏离分数对站点效应的独立性）同一议题、互补证据。

### 1.5 MDD 的全脑建模与可解释性

- **Escrichs et al. 2025, Mol Psychiatry**：全脑湍流动力学预测抑郁治疗反应。
- **An et al. 2022, NeuroImage**：个性化 TVB 指导 MDD 的 DBS 靶点。
- **Wang et al. 2022, HBM**：全脑模型驱动健康↔抑郁态的**状态转移与控制**。
- **Ramirez-Mahaluf et al. 2017, Cereb Cortex**：vACC/dlPFC 生物物理模型。
- 可解释性方法学：**BrainGNN**（ROI 感知注意力 + ROI 选择）、**IBGNN/BrainIB**
  （信息瓶颈子图）、[GNN 脑连接综述（Mohammadi 2024）]：注意力/边归因、
  子图发现、置换重要性三类是主流；本项目已有置换重要性管线
  （`evaluate --feature_importance`）与学习图（`models/graph_prior.py`）。

### 1.6 对本方案的三点直接启示

1. **验证口径**：孪生模型的"性能"不只是 MAE——DTB/Takahashi 均以
   **FC / PSD 相似度**与（若可得）行为/临床外部锚点作为核心证据；本方案
   P3 补齐这一层。
2. **病理分析的可信度来自设计**：normative modeling 文献把"站点校准 +
   负控"当作可信结论的前提；本项目已实现且实证了站点混杂（D2 应作为
   论文核心图之一）。
3. **可解释性的标准动作**：条件/输入使用率（本项目 B5）、学习图 vs 先验、
   置换重要性、时间归因、潜状态动态、虚拟干预（需标注"结构性质、非临床
   因果"——Takahashi 与本项目 T03 均采用此表述边界）。

---

## 2. 总体设计原则

1. **数据源分层**（按成本递增，每张图标注 tier）：
   - **T0 纯已有产物**：`results/final/`、`results/normative_reference/`、
     `results/pathology_analysis/run1/`（z 矩阵、FDR 表、network_summary、
     clinical_partial_corr、condition_usage、gate_report 等，字段已核对）。
   - **T1 无 GPU 统计重算**（分钟级，只读 z 矩阵/临床表，输出
     `results/paper_stats/`）：未匹配对照逐 ROI 统计、负控逐 ROI 分布、
     FC-ridge OOF 分数 ROC、临床残差散点。
   - **T2 轻量只读推理**（GPU 0.5–2h，权重只读，输出 `results/paper_infer/`）：
     少量被试（建议 HC/MDD 各 20）的 K64→H32 自然轨迹、FC/PSD 相似度、
     学习图导出、置换重要性。
   - **T3 重算滚动评估**（GPU 数小时，`analysis.rolling_one_step`，输出新
     rolling 目录）：仅当需要 ROI×被试级改善度脑图时执行（`results/rolling_eval`
     原缓存已清理，见成本表）。
2. **组间图三件套面板**：凡展示 MDD−HC 效应的图，必须同图含
   （匹配效应 / 未匹配对照 / HC 负控）三个面板，防止"未匹配脑图"被单独引用。
3. **产物命名**：`results/paper_figures/fig_<编号>_<slug>.png`（300 dpi）+
   同名 `.csv`/`.json` 数据；统计中间量写 `results/paper_stats/`。
4. **描述性 vs 推断性分离**：推断性结论只引用 run1 已有检验；本方案新增的
   一切数值仅作描述性呈现或方法学演示。
5. **图内标签英文、图注中文**（论文需要），网络配色与 ROI 命名复用
   `analysis/visualize_roi_importance.py` 的 `NETWORK_NAMES` 与
   `data/AAL116.xlsx`（含中文名/abbr/网络归属）。

---

## 3. Phase 1 模型性能分析（论文"结果-性能"部分）

| # | 分析 | 文献对齐 | 数据源(tier) | 图设计 |
|---|---|---|---|---|
| P1 | 滚动单步护栏：模型 lag-1 MAE 0.0814 vs persistence 0.6048，improvement 86.5%，置换 p=1e-4 | DTB 数值一致性验证；Takahashi 生成质量 | gate_report.json + condition_usage.csv（被试级 MAE 分布）(T0) | F1：左=组级条形+置换 null 直方图示意；右=被试级 MAE 分布（227 人 val） |
| P2 | 长程精度衰减：H1–8/9–16/17–32 MAE（val/test）vs 历史均值基线 0.922955；时间 PCC≈0.178 的诚实呈现 | 长程预测评测惯例 | metrics_val/test.json + per_subject_*.csv (T0) | F2：分段 MAE 柱状 + 被试级 box/violin + 基线虚线；附表：远期相对基线仅 +2.86%、时间 PCC |
| P3 | 生成/仿真质量（孪生模型的标准验证口径）：pred vs true 轨迹；FC(pred) vs FC(true) 矩阵与逐被试 r；PSD 对比 | DTB 任务同化（体素 r=0.75）；Takahashi（PSD/FC 相似度） | T2：`inference_twin.py` 只读推理 HC/MDD 各 20 人（K64→H32） | F3：三联——(a) 3–4 个代表 ROI 的轨迹叠加（近/远期分区着色）；(b) 单被试 FC 三联矩阵（true/pred/差异）+ 散点；(c) 逐被试 FC-r 与 PSD-KL 分布（HC vs MDD 配色） |
| P4 | 误差结构：逐 ROI 改善度脑图、误差 vs ROI 信号幅度、站点级 QC | normative guide 的站点 QC 规范 | T3：重算 rolling（至少 hc/mdd 的 val split）；`per_subject_roi_mae_*.npz` | F4：(a) ROI improvement 脑图（nilearn glass brain，AAL116）；(b) 网络|脑区条形；(c) 分站点 improvement 箱线（站点 QC） |
| P5 | 校准与概率路线结论（存档引用）：T04 公共系数 0.948788、远期仅改善 0.0228%；S4/T18 负结论 | — | EXPERIMENTS.md 数值 (T0) | 并入 P2 附表，不单独成图 |

说明：P3 是把本项目从"预测器评测"提升到"孪生验证"的关键补充，成本低
（每被试 CPU 数十秒），建议优先执行。FC 用 H32 预测段与对应真实段分别计算
（Pearson r 报告逐被试分布）；PSD 用 Welch 谱 + 一致性度量。所有推理种子与
checkpoint 快照一致（seed=2024/20261009），条件使用 hc3 字段。

---

## 4. Phase 2 病理分析（论文"结果-病理"部分，阴性结果 + 方法学贡献）

**前言（写进论文的边界声明）**：MDD 侧无外部锚点（sEEG/手术结局类金标准
不可得），因此本项目的病理分析只能回答"组间可分性与定位"，不能回答机制；
这与 VEP 有 sEEG 锚点的验证结构本质不同（§1.2）。

| # | 分析 | 文献对齐 | 数据源(tier) | 图设计 |
|---|---|---|---|---|
| D1 | 主结果：模型 z 逐 ROI/网络组间检验（0/116 过 FDR，max\|Δz\|=0.090，q≥0.34；离散度方差比 0.98） | normative 效应量图 | roi_fdr_table.csv、group_tests.json、network_summary.csv (T0) | F5：(a) Manhattan（−log10 p，FDR 线）；(b) Top-20 森林图（95% CI）；(c) 网络/全脑森林；(d) Δz 脑图 |
| D2 | **站点混杂方法学图（本文核心贡献）**：未匹配 Δz≈+0.07、Top ROI 为楔后叶/扣带/海马旁/眶额内侧"经典脑区" → 匹配后 −0.02 消失；负控同量级（max 0.176） | Gaiser 2024；normative guide 站点校准 QC | z 矩阵全量 (T1 重算未匹配/负控逐 ROI 统计，无 GPU) | F6：三联脑图/条形（未匹配 vs 匹配 vs 负控），叠加逐 ROI \|Δ\| 分布对比；附 Top-ROI 名单变化表 |
| D3 | 队列可分性上限（说明阴性非模型单方失败）：六描述量 0/116；静态 FC+岭回归 OOF AUC=0.560 | Marquand 框架的"case-control 效应量" | T1 重跑 FC-ridge 拿 OOF 分数（无 GPU，分钟级）；roi_fdr_table_descriptor.csv | F7：(a) ROC + 500 次 bootstrap CI 带；(b) 模型 z vs 描述量效应量散点（对角线，normative guide Fig.2C 风格） |
| D4 | 个体异质性（描述性）：被试级 mean\|z\| 分布（HC vs MDD 匹配子集）、个体 z 剖面示例（每网络取代表 ROI）、MDD 内部 z 离散度 | Marquand 2016 heterogeneity；guide Fig.2A 个体 z 剖面 | per_subject_z_*.csv (T0) | F8：(a) mean\|z\| 半小提琴+箱线；(b) 4–6 个个体（MDD 极端 + HC 中位）的 116-ROI z 剖面热条；(c) 个体内 z_std 分布 |
| D5 | 临床关联（描述性、透明报告）：HAMD 无关联（\|r\|≤0.043）；LN↔HAMA r=−0.135（p=0.008，n=422，未多重校正；HAMA 有效率仅 25.3%） | Escrichs 2025 的行为-脑关联 | clinical_partial_corr.csv；残差散点需 T1 重算（无 GPU） | F9：(a) 网络×评分偏相关热图（标注 n 与未校正）；(b) LN↔HAMA 残差散点（附缺失机制警示框） |
| D6 | T2 参数线弱线索（透明报告）：profile 32 维 1 维 q=0.013，99.4% 能量在单一共享方向 | MINDy 参数表型 | param_dim_table.csv、param_group_tests.json (T0) | 并入附图：profile 维度效应量条形 + FDR 线 |

D2 的 T1 重算 = 对已有 z 矩阵做未匹配/负控的逐 ROI 统计（复用
`analysis/pathology_tests.py` 的置换/FDR 函数，新增独立只读入口，输出
`results/paper_stats/`，不触碰 run1）。这是**方法学演示**，不产生新的病理
推断声明。

---

## 5. Phase 3 可解释性分析（论文"结果-机制/解释"部分）

| # | 分析 | 文献对齐 | 数据源(tier) | 图设计 |
|---|---|---|---|---|
| I1 | 条件使用（B5 终裁可视化）：Δpred/信号尺度中位 0.46%；零化条件 MAE 0.08143→0.08135；FiLM γ/β 统计 | 条件归因 | condition_usage.csv/json (T0) | F10：(a) delta_pred_relative 分布（对数轴，0.46% 中位线）；(b) mae_cond vs mae_zero_cond 配对散点（对角线）；(c) γ/β 分位数条形 |
| I2 | 学习图结构：`A_eff = λ·A_SC + (1−λ)·A_func(h)` 的样本级图 vs SC 先验与 history FC；λ 值与网络块结构 | GNN 边归因；MINDy 连接参数 | T2：读 checkpoint 图参数 + 20 被试前向导出 A_eff（与 P3 同批） | F11：(a) SC / A_func / A_eff 三联矩阵（网络分块排序）；(b) 网络块平均连接条形；(c) λ（全局/样本级）分布。图注必须写明"SC 为软先验、功能图来自 history FC 与学习参数，非独立实测解剖连接"（EXPERIMENTS 边界） |
| I3 | ROI 重要性（置换重要性 + FDR）：复用现有管线 | BrainGNN/置换归因 | T2：`python -m analysis.evaluate --feature_importance` + `analysis.visualize_roi_importance.py` | F12：ROI ΔMAE 脑图 + 网络条形（FDR 显着标注）；输出新目录 |
| I4 | 历史时间归因（可选）：T08 结论（打乱旧历史块远期 MAE +14.2%）可视化；如复算则做"历史块位置 vs 误差增量"衰减曲线（只读前向，与 P3 同批） | 时间注意力/消融 | EXPERIMENTS.md T08；复算 T2 | F13：块位置-误差增量曲线（近块>远块的结构性结论） |
| I5 | **[需预登记/需确认] 虚拟干预/损伤敏感性（探索性）**：单 ROI 输入扰动或 ODE 通路抑制的传播敏感性图 | Takahashi 虚拟给药（0.89/0.96 判别）；VEP 虚拟切除 | T2 只读推理（`inference_twin.py` 干预通路，**未校准**） | F14：lesion 敏感性脑图 + 传播热图。**硬性表述边界**：仅模型结构性质，不构成真实刺激响应或疾病因果证据（T03 结论沿用）；按 §5 规则此条须先小规模预登记再执行 |

I1–I3 无需任何新推断口径，均可直接执行；I4 建议复用 T08 既有结论先出文字
表，复算列为可选；I5 单列并默认**不执行**，待用户确认。

---

## 6. 可视化规范与图库

- **目录**：`results/paper_figures/`（图 + 数据副本）、`results/paper_stats/`
  （T1 统计中间量）、`results/paper_infer/`（T2 推理产物）。
- **样式**：新增 `analysis/paper_style.py`（共享 rcParams：Okabe-Ito 色盲
  安全色板、统一字号/线宽、网络配色表、AAL116 标签加载器）；不改动现有
  脚本的默认行为，新图脚本 `analysis/paper_figs/fig_*.py` 只 import 该样式。
- **脑图**：nilearn 0.13.1（已安装）`plot_glass_brain`/`view_img`；AAL116
  atlas 需一次性下载，若环境无外网则兜底为"网络分块矩阵 + 环形 chord 图"
  （networkx 已装）。ROI 数值 → 体素图用 AAL 索引映射。
- **组间图**：一律三件套面板（匹配/未匹配/负控）；HC=蓝、MDD=橙/红；
  效应量统一 Δz 与 95% bootstrap CI。
- **图清单**：F1–F14 如上；另附 2 张总览图——
  F0a 管线流程图（rolling → fit/score → tests → figures，数据流示意），
  F0b 被试构成 Sankey/表格（HC/MDD × split × 站点匹配前后）。

---

## 7. 执行顺序与成本

| 阶段 | 内容 | 计算成本 | 前置 |
|---|---|---|---|
| 步骤 1 | T0 图：F0b、F1、F2、F5、F8、F10 + 附表 | 分钟级（纯读已有 CSV/JSON） | 无 |
| 步骤 2 | T1 统计：D2 未匹配/负控重算、D3 ROC、D5 散点 → F6、F7、F9 | 无 GPU，分钟–十分钟级 | 新增只读统计入口（复用 pathology_tests 函数） |
| 步骤 3 | T2 推理：P3 轨迹/FC/PSD + I2 学习图 + I3 重要性（+可选 I4） → F3、F11、F12(、F13) | GPU 0.5–2h，只读权重 | 无（种子沿用快照） |
| 步骤 4 | （可选）T3 重算 rolling val（hc/mdd）→ F4 | GPU 数小时（222+MDD val 人） | 用户确认 |
| 步骤 5 | （默认不执行）I5 虚拟干预 → F14 | GPU <1h | **预登记 + 用户确认** |

说明：原 `results/rolling_eval/`（833 MB 缓存）已清理，故 F4 需重算；
若论文可接受"网络级改善度条形 + F1 被试级分布"，可跳过步骤 4。
所有新脚本入口遵循 `--out-dir` 显式参数，绝不写入 run1/final/normative_reference。

## 8. 与既有规则的关系（自检）

- 不触碰关闭清单（H32 优化、频域、条件敏感性、参数线、为解释而重训）。
- 本方案不产生新的组间推断声明；D2/D3 的重算是对既有结论的方法学演示。
- I5 是唯一的探索性干预分析，默认不执行、需预登记三件套。
- 全部产物进新目录；git 提交仍待用户授权（PROJECT_STATUS §6）。

---

## 9. 参考文献（调研来源）

- DTB 平台：[arXiv 2308.01241](https://arxiv.org/html/2308.01241v1)；
  [Xiong et al. 2023, Intelligent Computing](https://spj.science.org/doi/10.34133/icomputing.0055)；
  [Lu et al. 2024, NSR（检索摘要）](https://academic.oup.com/nsr/article/11/5/nwae080/7617698)
- 意识孪生：[Takahashi et al. 2025, npj Digital Medicine](https://www.nature.com/articles/s41746-025-01444-1)
- TVB/VEP：[Jirsa et al. 2023, Lancet Neurology](https://pubmed.ncbi.nlm.nih.gov/36972720)；
  [Makhalova et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9543509)；
  [Wang et al. 2025, Nat Commun Med](https://www.nature.com/articles/s43588-025-00841-6)；
  [EPINOV](https://ins-amu.fr/epinov)
- MINDy：[Singh et al. 2020, NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811920305322)；
  [Schwamb et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12133028)
- Normative modeling：[Marquand et al. 2016](https://www.biologicalpsychiatryjournal.com/article/S0006-3223(16)00002-0/fulltext)；
  [PCNtoolkit/Rutherford 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036120)；
  [Practical guide 2025](https://arxiv.org/html/2509.07237v1)；
  [Gaiser et al. 2024, HBM](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26565)；
  [Fraza et al. 2025 review](https://www.sciencedirect.com/science/article/pii/S0149763425002854)
- MDD 全脑建模：[Escrichs et al. 2025, Mol Psychiatry](https://www.nature.com/articles/s41380-024-02690-7)；
  [An et al. 2022, NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921011198)；
  [Wang et al. 2022, HBM](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26006)
- 可解释性：[Mohammadi et al. 2024, GNN 脑连接综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC11763835)；
  [IBGNN](https://etd.library.emory.edu/concern/etds/9880vs186)
