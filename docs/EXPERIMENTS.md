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
- 数据：REST-meta-MDD，116 ROI，被试级 8:1:1 划分（manifest 记录
  split seed=2024，版本化 `data/subject_split_*.json`）。
- 随机种子：模型 seed=2024、采样 seed=20261009、Condition Dropout seed=20261008
  （三者互斥校验）。
- 训练配方：`configs/baseline_s2_e2.json` 是唯一来源；两份训练脚本通过
  `scripts/_baseline_defaults.sh` 读取，环境变量只用于显式消融覆盖。
- 评估聚合：先被试内平均、再跨被试等权平均（`analysis/next_point_eval.py`）。
- 后续概率评估统一采用：逐锚点在有效 ROI/horizon 上归一化，再被试内平均，最后跨被试
  等权平均；缺尾锚点仍与完整锚点等权，完全无有效目标的锚点不参与平均。
  `analysis/s4_probabilistic_eval.py` 的 JSON、总体 subject CSV 和被试×horizon CSV
  均采用该口径。本轮概率比较已从保存样本按该口径计算；历史元素加权 CSV 仅保留来源记录，
  不与新口径混用，也不为本次归档重新训练或重跑历史评估。
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
不再是开发分支。任务 A 的最终判定见 §6.1，首轮研究完成，E2 保留。

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


### 6.1 任务 A：Loss 优化正式结题（2026-10-22）

**状态：首轮研究完成，E2 保留。** L0–L3、P0/P1 全部训练与评估完成；
停止新的 Loss 消融，不实施 L4，不为 L1–L3 的微小收益追加多 seed，不重训已有实验。

**协议**：固定 S2 架构、HC 微调起点、数据划分与非 Loss 超参数，seed=2024，
225 名 test 被试、相同预测锚点、H=8。P0/P1 从同一 S2 起点独立初始化原 S4-1
残差头，主干全程冻结，test K=64。实际配方取自正式 checkpoint 快照；
沿用工作区已有的调度器修复并重跑 L0/P0，历史 S2/S4-1 数值未拼接为本轮对照。

| 实验 | t+1 MAE | ODE MAE H1–H8 | ODE H8 MAE | 核心判定 | 最终决策 |
|---|---:|---:|---:|---|---|
| L0：E2 | 0.08066735 | 0.53289542 | 0.80251911 | 配对正式对照 | 正式保留 E2 |
| L1：幅度自适应 | 0.08062067 | 0.53286162 | 0.80238032 | ODE MAE 仅改善 0.00634%；Q5 Delta MAE 改善 0.666%，Q1 恶化 1.078% | 不采用 |
| L2：Horizon-Aware | 0.08060921 | 0.53286233 | 0.80252928 | H4–H8 改善约 0.00198%，H8 略差；CPM 平均误差退化 | 不采用 |
| L3：Multi-Scale | 0.08067448 | 0.53292568 | 0.80253661 | 总体 MAE 与 lag=1/2/4 变化误差均退化 | 不采用 |

L1 的绝对 ODE MAE 改善仅 0.00003380，统计显著性不足以弥补效应过小及幅度区间
权衡。L2 的远期权重没有带来实用远期收益；L3 未显示多尺度监督的独立收益。
本轮没有具有实际价值且互补的确定性候选，故不组合 L4。

| 概率实验 | CRPS | Energy Score | 历史 Variogram | 80% Coverage | 80% Width | 决策 |
|---|---:|---:|---:|---:|---:|---|
| P0：CRPS | 0.37395749 | 0.51755812 | 2.06153293 | 0.76454095 | 1.44126546 | 保留概率对照记录 |
| P1：CRPS + TV | 0.37394655 | 0.51718534 | 2.02909431 | 0.77286310 | 1.47556209 | 不定型 |

P1 的训练前固定 lambda_v=0.1409571896，lag=1/2/4 等权、power=0.5。
历史 Variogram 改善 1.574%，新逐点 TV 改善 0.693%，Energy 改善 0.0720%；
CRPS 改善仅 0.00293%（配对 p=0.434）。覆盖率略好，80%/90% 区间均变宽约 2.38%。
P1 仍弱于自身 model_iid：Variogram 2.029094 对 1.649154，Energy 0.517185 对
0.512591。因此联合监督确实改善当前 S4-1 的部分评分，但尚未使其相关采样轨迹优于
独立边际采样；此结果不能解释成 Variogram Loss 本身无效，也不足以定型新概率方案。

**冻结的损失基线**：S2 + E2；One-Step/CPM 保留原 Huber + Spatial PCC，
Dynamics=0.5、Variance=0.05、Temporal Difference=0.05、Cross-Head Consistency=0.02，
原四任务 uncertainty 加权继续使用。L1–L3/P1 均不启用。
这表示在已完成候选、固定架构与训练协议下，没有值得替换 E2 的方案；
不宣称 E2 在所有可能损失中达到数学最优，也不宣称已验证跨 seed 稳定性。

**归档**：完整指标、配对检验与结论见 `results/loss/compare/RESULTS.md`；
保留 `compare/main_table.csv`、`probabilistic/compare/P0_P1_table.csv`、Figure 1–5、
各臂 `training_config.json`、最终指标 JSON 与汇总 CSV。
已结题实验在 `results/` 内的权重副本已清理；正式 S2 与历史 S4 权重保留于顶层 `checkpoints/`。
工件按原路径归档，索引及逐臂决策见 `results/loss/archive_manifest.json`；
本轮临时编排、测试和对比脚本已清理，公共训练/评估入口与损失实现保留；
预测采样数组、训练参照数组、末轮权重、重复评估、过程日志及中间 CSV 已清除。
L1–L3/P1 不再维护独立开发路线。
正式 `configs/baseline_s2_e2.json` 与 S2 checkpoint 保持原样。

### 6.2 Temporal PCC 独立补充（已完成，2026-10-23 记录）

按用户后续请求新增未来轨迹 Temporal PCC，TP0=E2，TP1 在 E2 上增加固定权重 0.05
的 CPM/ODE Temporal PCC 之和；同 HC 起点、seed、划分与原训练配方。
两臂分别在 43/31 轮早停，225 名 test 被试、H=8，训练和评估均已完成。

| 指标 | TP0 | TP1 |
|---|---:|---:|
| t+1 MAE | 0.08066734 | 0.08063608 |
| ODE 平均 MAE | 0.53289543 | 0.53294750 |
| CPM 平均 MAE | 0.53326342 | 0.53350778 |
| ODE H8 MAE | 0.80251914 | 0.80255400 |
| ODE Temporal PCC | 0.70562083 | 0.70604755 |
| CPM Temporal PCC | 0.70750744 | 0.70761189 |

**判定**：时间相关性有小幅提高，但 ODE/CPM 平均 MAE 分别恶化 0.00977%/0.04582%，
未得到整体预测收益，保持正式 E2，不启用该候选。完整分层指标和配对统计见
`results/temporal_pcc/RESULTS.md` 与 `compare/`。本轮为单 seed、单权重，
结论不外推为 Temporal PCC 在所有设置下均无效。

---

## 7. 最终基线与选择依据

| 维度 | 定型 |
|---|---|
| 架构 | S2：TFM backbone（patch=8, dim=256, 4 层）+ Graph-ODE 动力学（Heun RK2）+ 历史记忆 FiLM + CPM 多步头 + cat 条件注入 |
| 损失 | E2：one/CPM/dynamics/variance（uncertainty 加权）+ λ_td=0.05 + λ_ch=0.02 |
| 权重 | `checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt`（SHA256 见 `configs/baseline_s2_e2.json`） |
| 入口 | `python baseline.py`（构建/加载自检）；训练 `scripts/Pretrain_HC_tfm.sh` → `scripts/Finetune_MDD_tfm.sh`（共享配方从 `configs/baseline_s2_e2.json` 读取，默认即基线口径） |
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
- 任务 A 已结题：`results/loss/compare/figure1_overall.png`、`figure2_horizon.png`、
  `figure3_amplitude.png`、`figure4_fixed_trajectory.png`，以及
  `results/loss/probabilistic/compare/figure5_probabilistic.png` 与分 horizon 校准图。

## 9. 已知研究局限

1. S2 只有一个训练 seed，正式统计结论待多 seed。
2. CPM 直接预测长度为 H=8（约 16 s）；Graph-ODE `rollout_next_states()` 可自由推进
   超过 8 步，但 H>8 未经充分监督消融与验证，不能视为已具备可靠长程预测能力。
3. 长时域自由推进 H8 MAE 0.80，仍存在明显状态误差累积（任务 B 目标）。
4. S4 的边际 CRPS 训练不能约束联合时空依赖；概率化路线待任务 B 重启。
5. 训练配方（已修复，2026-10-22）：旧实现有两个问题——(a) 全局 `eta_min=5e-5`，
   若某参数组已在 scheduler 中且峰值低于 `eta_min`，余弦阶段会把该组学习率反向
   推高；(b) finetune 的 backbone 组在冻结期结束后才追加到 optimizer，旧 scheduler
   不会纳入该组，backbone 实际保持恒定 `2e-5`。现改为按组相对余弦，并在解冻追加
   参数组时同步 scheduler（backbone 从 2e-5 按同一进度衰减到 1e-5）。现有 S2 权重
   未重训（finetune 早停于 epoch 22/23）；修复从下一轮训练起生效，修复前后的训练
   结果不应直接混用。
6. 解冻有效性（已修复，2026-10-23）：在 PyTorch 2.6.0+cu118 下，完整 S2 模型
   （6,244,019 参数）使用真实 MDD-train batch（B=2、116 ROI、context=47、H=8）
   复现：先冻结再 `torch.compile(dynamic=True)`，完成冻结期反向后解冻，CPU
   `aot_eager` 与 CUDA `inductor` 均继续缺失主干梯度。虽然 `requires_grad=True`
   且参数已进入 optimizer，实际只有 37 个条件编码器/FiLM 参数张量获得梯度；
   eager 同一 batch 有 173/176 个参数张量获得有限非零梯度（其余 3 个为未使用的
   control 参数）。`set_finetune_stage` 现对编译包装器在解冻时调用
   `torch.compiler.reset()`，使后续前向重新生成含主干梯度的反向图，同时保留参数
   对象、optimizer 状态与 scheduler 进度。修复后 CPU `aot_eager` 与 CUDA `inductor`
   均恢复到 173/176 个参数张量获得有限非零梯度。现有权重与历史指标未重写；相关微调结果
   不能作为有效解冻全模型的证据，修复后的效果需后续重训评估。


## 10. 结果目录精简

`results/` 仅保留最终研究证据：指标 JSON、汇总/配对统计、关键图表及实际配置。
预测数组、逐样本明细、比较用被试长表、验证中间输出、过程日志、旧条件搜索计划，
以及本目录内已结题 Loss 实验的权重副本均已清理。历史记录中的原始导出路径用于
追溯来源，不保证相应中间文件仍存在。正式 S2 配置与顶层 `checkpoints/` 保持原样。
