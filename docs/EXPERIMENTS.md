# NeuroTwin 已完成实验记录与最终结论

本文档是项目正式实验档案：记录已完成实验的目的、配置、最终指标、结论与是否纳入
正式基线的决策。历史各版本的阶段性报告已被本文吸收；如需过程细节，见
`docs/Version*_docs_*/`（按版本归档）与 git 历史。

**正式基线（2026-10-22 定型）：S2 History Memory + Graph-ODE + CPM 架构，E2 损失。**
配置契约见 `configs/baseline_s2_e2.json`，权重为
`checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt`。

---

## 0. 统一实验协议

- 任务口径：`next_timepoint`（连续 BOLD context → 下一 TR 全脑状态；多步由
  Graph-ODE 持续推进与 CPM 双头输出）。
- 数据：REST-meta-MDD，116 ROI，被试级 8:1:1 划分（split seed=42，
  版本化 manifest `data/subject_split_*.json`）。
- 随机种子：模型 seed=2024、采样 seed=2024、Condition Dropout seed=20261008
  （三者互斥校验）。
- 评估聚合：先被试内平均、再跨被试等权平均（`analysis/next_point_eval.py`）。
- 各阶段协议差异（不可跨口径直接比较指标）：
  - E/S 系列、S4：225 被试 / 3825 锚点 / H1–8。
  - D1/D2 刷新诊断（refresh_*）：225 被试 / 450 锚点 / H1–8。
  - S4 概率评估：K=64 集合预测复核口径。

---

## 1. 第一阶段：损失函数实验 E0 / E1 / E2（P0 损失扩展）

**目的**：在 graph_ode 路线上为监督目标增加（E1）一阶时间差分 Huber 项与
（E2）CPM←Graph-ODE 双头一致性项，检验其对点值与多步预测的影响。

**配置**：三臂同 seed（2024）、同划分、同架构（graph_ode + history_fc +
personalized + cat 条件）。差异仅在损失：E1 λ_td=0.05；E2 λ_td=0.05 + λ_ch=0.02。
权重：`checkpoints/phase1_E1、phase1_E2`（E0 = `archive_e0_graph_ode_*`）。

**最终指标（225 test 被试，`results/phase1/`）**：

| 臂 | t+1 MAE | t+1 PCC | ODE MAE(H1–8) | ODE MAE(H8) |
|---|---:|---:|---:|---:|
| E0 | 0.08085 | 0.99442 | 0.5384 | 0.8063 |
| E1 | 0.08051 | 0.99444 | 0.5382 | 0.8061 |
| E2 | 0.08056 | 0.99446 | **0.5376** | 0.8054 |

配对检验（`results/phase1/compare/phase1_paired_tests.csv`）：E2−E0 的 t+1 MAE
改善 2.9e-4（p=1.2e-11）、ODE MAE 改善 8.7e-4（p=9.4e-12）；E1−E0 t+1 改善
3.4e-4（p=9.2e-17）。

**结论**：新增损失无副作用，Delta MAE / 方向准确率 / 多步 MAE 有统计显著但量级
很小的改善；CPM–ODE Gap 相对 E0 下降约 5%。改善幅度不足以解决长时域动态退化。

**决策**：**E2 纳入正式基线**（λ_td=0.05、λ_ch=0.02）。E0/E1 仅保留消融证据，
不再是开发分支。后续损失优化见 `docs/RESEARCH_PLAN.md` 任务 A。

---

## 2. 幅度诊断（phase1.diagnostics）

**目的**：区分多步退化来自「读出幅度压缩」还是「动态传播退化」，检验全局缩放
与分层缩放能否修复。

**关键结果**（`results/phase1/` 各臂 diagnostics CSV 与图 4）：

- H1 幅度比（Delta Std Ratio）≈ 0.98：首步接近无压缩 ⇒ **H1 无明显幅度压缩**。
- H2–8 幅度比 ≈ 0.76：压缩随自由推进加深 ⇒ **多步传播性退化，非读出问题**。
- Q1 小变化 MAE ≈ 0.199 vs Q5 大变化 ≈ 0.515：Q1 过度响应、Q5 幅度低估（异方差）。
- persistence / trend / AR(1) 基线与全局缩放、分层 oracle scale 分析均表明
  **全局输出缩放不能修复**；单纯提高 λ_td 缺乏依据。

**决策**：属已完成的模型诊断。保留最终指标与图 4（幅度分层与 DSTD 诊断），
不再作为独立研发阶段。

---

## 3. 状态刷新实验 D1 / D2

**目的**：定位自由推进误差的来源——单步模型能力不足 vs 状态误差累积。

**配置**（`experiments/long_horizon/refresh_rollout.py`，450 锚点协议）：
Free = Graph-ODE 自由推进；D1 = 每步用真实观测刷新状态后继续推进；
D2 = 每步用模型自预测值重编码刷新。

**最终指标**（E2 臂，`results/phase1/refresh_E2/`；S 系列 `results/phase2/refresh_*`）：

| 方式 | H1 MAE | H8 MAE |
|---|---:|---:|
| Free rollout | 0.0817 | 0.8142 |
| D1（真值刷新） | ≈0.081 | ≈0.081（全程平稳） |
| D2（自预测刷新） | — | 1.0046（劣于 free） |

S2 臂：D1 temporal variance ratio = 1.0249（各臂最接近 1），D2 = 1.3193。

**结论**：模型局部（单步）预测能力良好；多步退化的主因是**状态误差累积**；
真实新观测可恢复后续单步能力；自预测重编码反而放大漂移；latent 方差大小
不能单独代表预测质量。**D1 使用未来真实观测，不能作为无观测自由预测的
性能基线。**

**决策**：D1/D2 定位为诊断工具（服务于长时域路线，任务 B），不是模型分支。
保留关键结果与 Figure 5/6（见 `results/phase1/refresh_E2/`、
`results/phase2/refresh_S2/`）。

---

## 4. 状态耦合实验 S0 / S1 / S2 / S3

**目的**：在 E2 基础上为 Graph-ODE 向量场引入状态耦合机制，改善多步推进。

**配置**（`results/phase2_train/run_phase2.sh`，同 seed、同 E2 损失口径）：
S0 = E2 复用；S1 = BOLD→Latent 反馈 G_φ(Z,q)；S2 = 历史记忆 FiLM（编码 token
的 patch 维均值，rollout 期间固定）；S3 = S1+S2 双开。

> **作废记录**：第一版 S1/S2/S3（2026-10-21 02:11–02:36，
> `*_initconfound_20261021`）因新模块构造推进了全局 RNG、导致消融被初始化
> 混杂污染而**整体作废**；修复为 `fork_rng` 内构建新模块后于同日重训全部臂。
> 受污染的 checkpoint 与中间结果已删除，不得与有效实验混合比较。

**最终指标（225 test 被试，`results/phase2/`）**：

| 臂 | t+1 MAE | ODE MAE(H1–8) | ODE MAE(H8) | DSTD(H2–8) |
|---|---:|---:|---:|---:|
| S0（=E2） | 0.08056 | 0.5376 | 0.8054 | 0.7591 |
| S1（反馈） | 0.08142 | 0.5364 | 0.8032 | 0.7692 |
| **S2（历史记忆）** | 0.08067 | **0.5329** | 0.8025 | 0.7651 |
| S3（双开） | 0.08112 | 0.5320 | 0.8017 | 0.7677 |

配对检验：S2−S0 单步 MAE 无显著差异（p=0.34，不劣化）；ODE MAE 改善 0.0047
（p=2.1e-26）、CPM MAE 改善 0.0031（p=8.5e-18）、幅度比 H2–8 +0.0059
（p=9.2e-29）。S1 与 S3 的单步 MAE 显著变差（S3 p=0.0033）。

**结论**：S2 历史记忆是多步唯一净收益项且不伤单步；S1 收益有限且伴随短期精度
退化；S3 中期最好但短期退化、结构复杂。S2 相对 S0 多步 MAE 约 −0.87%，
大多数测试被试改善，H4–H6 与 Q5 大变化区间改善。

**决策**：**S2 纳入正式基线**（工程开发基线）。S1 反馈与 S3 组合的实现与
实验入口已于 2026-10-22 移除（git 历史可追溯），其结果保留为本节证据。
**局限**：单训练 seed，不宣称多 seed 统计定论。

---

## 5. 概率预测实验 S4-0 / S4-1

**目的**：在 S2 确定性主干上检验概率化轨迹预测（输出端条件相关残差）。

**配置**：S4-0 = 直接评估 S2 已有逐 ROI 对角 Gaussian VarianceHead（不训练）；
S4-1 = 输出端低秩时空残差头（`models/tfm_probabilistic.py`），head 阶段训练、
主干冻结、单 seed、K=64 集合评估（`analysis/s4_probabilistic_eval.py`，
`checkpoints/phase4_S4_1`）。

**S4-0（对角高斯，`results/phase4/S4_0/`）**：cov80 = 91.7%（过宽）、宽度 2.78
（ROI×H 常数参照 1.71）；NLL 1.109 vs 全局常数 1.121（p=0.007）但 CRPS 0.415
劣于 0.395。对角方差捕捉部分幅度依赖但整体偏大、校准不足。

**S4-1 最终指标（`results/phase4/S4_1/metrics_s4_1.json`，225 被试/3825 锚点）**：

| 指标 | S4-1 | 关键对照 |
|---|---:|---|
| CRPS | 0.3740 | 常数方差高斯 0.3711 |
| fair CRPS | 0.3689 | — |
| cov80 | 0.7636（ceiling 比率 0.968） | — |
| Energy Score | 0.5176 | model_iid 0.5130 |
| Variogram Score | 2.062 | model_iid 1.670 |

相对 S4-0：CRPS −0.0502（−11.8%，p=9.8e-137）、Variogram 8.77→2.06，全面更优；
确定性输出（MAE=0.08067、PCC=0.99452）与 S2 逐位一致（主干未被扰动）。

**结论**：S4-1 显著改善边际与尺度，但**未超越输入无关的常数方差高斯基线**
（ΔCRPS=+0.0029，p=7.9e-12）；相关残差结构的联合评分（ES/Variogram）劣于相同
边际的 IID 对照——实测证明边际 CRPS 训练对联合时空依赖完全盲。逐 lag 相关
γ/真值 0.67–0.85，时间相关过强。

**决策**：**S4-0/S4-1 暂不合入正式主干**。保留最终概率指标、核心对照、
概率评估实现（`analysis/s4_probabilistic_eval.py`、`train/prob_losses.py`、
`models/tfm_probabilistic.py`）与 `checkpoints/phase4_S4_1` 权重，作为后续
概率化研究（任务 B）的历史对照。S4-2（潜状态随机创新）**尚未验证**，
不视为已有方案。K=64 集合预测数组（约 1.8G npz）已删除，可用上述命令从
checkpoint 重新生成。

---

## 6. 历史负结果存档（简）

- **MoE/RECC 病理专家线**（2026-10-07 结题）：`tfm_patho_mode='moe'` +
  Router–Expert Competence Coupling 为负结果，未进入 graph_ode 路线；
  结题记录见 `docs/Version3_docs_1003/NeuroTwin_TFM_RECC_详细应用方案.md` 与
  git 提交 162c90f。实现已随 2026-10-22 定型清理删除。
- **条件注入选型**：cat（hc3 / mdd8）为最终口径，study/diagnostics 链路已删除；
  结论已吸收进 README 与本档案（`docs/Version3_docs_1003/NeuroTwin_条件注入实验方案.md`）。

---

## 7. 最终基线与选择依据

| 维度 | 定型 |
|---|---|
| 架构 | S2：TFM backbone（patch=8, dim=256, 4 层）+ Graph-ODE 动力学（Heun RK2）+ 历史记忆 FiLM + CPM 多步头 + cat 条件注入 |
| 损失 | E2：one/CPM/dynamics/variance（uncertainty 加权）+ λ_td=0.05 + λ_ch=0.02 |
| 权重 | `checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt`（SHA256 见 `configs/baseline_s2_e2.json`） |
| 入口 | `python baseline.py`（构建/加载自检）；训练 `scripts/Pretrain_HC_tfm.sh` → `scripts/Finetune_MDD_tfm.sh`（默认即基线口径） |
| 性能 | t+1 MAE 0.08067 / ODE MAE(H1–8) 0.5329 / H8 0.8025（225 test 被试） |

**与历史冻结工件的关系**：`configs/final_model.json` 是 2026-10-07 冻结的
V4 discrete 时代历史契约（patch=4），其原始权重保存在
`checkpoints/Version3_weights_and_results_1003/`，与 S2 基线相互独立、
互不引用；2026-10-22 已将默认路径恢复为 SHA256 一致的原始冻结权重。

## 8. 关键图片索引

- E/S 系列对比图 1–4：`results/phase1/compare/`、`results/phase2/compare/`
  （fig1 精度、fig2 动态、fig3 多步双头、fig4 幅度诊断）。
- D1/D2 刷新诊断（Figure 5/6）：`results/phase1/refresh_E2/`、
  `results/phase2/refresh_S2/`。
- S4 概率评估：`results/phase4/S4_1/`（calibration_curve.png、
  horizon_probabilistic.png、q1_q5.png）。
- S2 基线评估：`results/phase2/eval_S2/`（同步副本 `results/baseline/`）。

## 9. 已知研究局限

1. S2 只有一个训练 seed，正式统计结论待多 seed。
2. 主模型 H ≤ 8（约 16 s）硬限；更长 horizon 未验证。
3. 长时域自由推进 H8 MAE 0.80，仍存在明显状态误差累积（任务 B 目标）。
4. S4 的边际 CRPS 训练不能约束联合时空依赖；概率化路线待任务 B 重启。
5. 已知训练配方问题：cosine eta_min=5e-5 高于 backbone 组初始峰值 2e-5，
   列为下次重训前独立修复项。
