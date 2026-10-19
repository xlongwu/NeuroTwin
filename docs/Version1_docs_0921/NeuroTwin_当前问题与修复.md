# NeuroTwin 当前存在问题及解决方案（未解决/未验证项）

> 更新日期：2026-09-26  
> 依据：代码库静态审计；2026-09-25 P0 消融实验（13 变体，seed=2024，原始产物 `results/ablation/` 已清理，结论以本文档与 2.8 修改记录为准）；2026-09-26 新代码重训复验（`checkpoints/neurotwin_pretrain_pred1`、`checkpoints/neurotwin_finetune_pred1`，multinomial 修复 + 取消融参数）及完整评估链路（`checkpoints/neurotwin_finetune_pred1/eval/`）。  
> 状态：按 2026-09-26 决定，已解决条目已从正文清除，本文仅保留**未解决**或**已实施但未验证**的问题。已解决历史见 2.8 修改记录与 git 历史。  
> 文档边界：本文只记录当前问题、修复方案、验证标准与实验协议。架构候选的完整文献池和研究组合见 `NeuroTwin_模型架构优化改进整合版.md`。

当前任务严格是 **SC 条件下的未来 ROI/BOLD 信号预测**：

$$\text{history ROI BOLD} + \text{SC} \rightarrow \text{future ROI BOLD}$$

MDD 微调在 HC backbone 预测上学习病理残差：

$$\hat{Y}_{\text{MDD}} = \hat{Y}_{\text{HC-base}} + \Delta Y_{\text{pathology}}$$

**2026-09-26 重训基线（seed=2024，新代码）：** test PCC 0.8336 / MAE 0.3443 / R² 0.6833（较 P0 baseline MAE 0.3933 改善 12.5%）；PICP_cal 0.9030（conformal 校准实战生效）；FC 重构显著优于 persistence/mean/linear 平凡基线。当前最关键的两个未解决问题是**病理条件注入失效（重验未通过）**与**MoE 路由坍缩持续**。

## 1. 模型结构问题

### 1.1 病理条件注入失效（最高优先级，重验未通过）

**问题确证。** 两轮独立证据：① 2026-09-25 P0 消融：全部 13 个变体（含 baseline）在 shuffled-HAMD 负对照下指标变化仅 ±1e-4；HAMD 分层效应微弱（corr_hamd_pcc=-0.07～-0.13）。② 2026-09-26 multinomial 修复 + 重训后复验：shuffled ΔPCC=1.06e-5，条件敏感性仍≈0。

**根因判断（2026-09-26 更新）。** 训练期采样机制缺陷已修复（`RouterCond` 恢复 multinomial 概率采样，合成验证条件敏感性 ≈0→22.3%），但真实数据复验未通过，说明根因不止采样机制：

- **路由器输入结构不含病理编码**：`moe_gate_features='state_revin'`，gate 输入=状态统计 + RevIN mean/std（`models/neurotwin_moe.py`），路由对 HAMD 天然不敏感（设计如此，需结构性改动）；
- **专家 FiLM 通路存在但未被利用**：`pathology_proj`→FiLM/path_gate 权重在 trained checkpoint 中存在，但 shuffled 对照证明其输出影响≈0——重建任务在给定脑状态输入下缺乏利用 HAMD 的激励（corr_hamd_pcc=-0.082）；
- 表达层面原始局限仍在：`pathology_input_dim=1` 单标量、`RicherPathologyProjection` 的 `x²/sin(πx)` 失衡、注入过浅（BrainMDM/GraphODEDDI 对病理无感）、加性假设过强。

**拟解决方案（按新证据重排优先级）。**

1. **gate 输入注入病理编码**：扩展 `moe_gate_features` 支持 `state_revin+pathology` 或 `state+pathology`，使路由条件真正依赖 HAMD；
2. **条件监督增强**：提高条件相关损失权重，或引入辅助反演头（latent→HAMD）作为多任务正则，强制特征携带病理信息；
3. 原 P1 方案保留：每 fold 拟合并冻结 robust z-score/quantile transform 对照组、移除 raw `sin(πx)` 默认路径；HAMD 条目/因子向量输入；BrainMDM/GraphODEDDI 内 FiLM/AdaLN 特征级调制（零初始化 residual gate）；双层条件化对照（additive only / feature modulation only / 结合）；
4. MDD 微调阶段探索最后 1--2 个 ODE block 的 LoRA 式 adapter（当前已用 LoRA，需对照其与特征级调制的组合效果）。

**验收。** shuffled-HAMD 负对照 ΔPCC 必须显著非零；HAMD 分层 PCC 单调性；低 PCC 被试（当前 10.2%，其 HAMD 显著更低，p=0.003）改善。若真实与打乱 HAMD 表现仍近似，不得提出 pathology-conditioned 结论。

**未验证项。** `cond_residual / cond_feature`（定位注入层级）已注册 P1 变体，未跑。

### 1.2 SC 约束过硬、重复注入且缺乏个体/状态适应（未解决）

**现状与问题。** `DFCAdapter` 的 `SC⊙learned_edge` 只能重标已有边，SC 为零时永远不能出现功能依赖；GraphODE attention 把 `sc_with_self>0` 当二值 hard mask 并施以固定 soft bias；DTI 假阴性边被永久切断，受试者共享先验、无方向性，静态 SC 无法表达动态 FC。SC 在 DFCAdapter、GraphODE、base refiner、MoE delta refiner 重复注入，可能压过功能证据并产生 over-smoothing。ForecastHead 的 `cross_roi_*` 是静态全连接 mixing，与上游"非 SC 边不可达"假设不一致。

**消融实证（2026-09-25，单 seed）。** `sc_fixed`（scaled+hard）与 `sc_adaptive_only` 相对 soft_prior 基线 ΔPCC 为 −0.0018 / −0.0007（噪声级，不可分辨）；FC 结构保真上两种简化略优。结论：**当前 soft_prior 组合的收益未被证明**，两种简化不劣。

**拟解决方案。**

1. SC hard mask → soft anatomical prior：$A_t = \lambda_t A_{\text{SC}} + (1-\lambda_t) A_{\text{func}}(h_t)$，低秩 `A_func=softmax(sym(UV^T))`（rank 8--16）；`λ` 依次尝试全局标量、sample-conditioned、ROI-specific。
2. 备选：straight-through/Gumbel 可学习概率掩码；低秩 subject-specific 邻接残差 `ΔA` + 稀疏/熵/时间一致性正则。
3. 对 Adapter、ODE、base refiner、delta refiner 的 SC 注入逐个消融，确定最小有效注入位置。

**未验证项。** `sc_no_inject / sc_lambda_sample / sc_lambda_roi / sc_no_delta_a` 已注册 P1 变体，未跑。

### 1.3 GraphODE 连续时间主张尚未成立（未解决）

**现状。** 未使用 `torchdiffeq/odeint`；手动 RK2/Heun 固定步数，`GraphODE.forward(t,...)` 丢弃 `t`（共享自治向量场）。stochastic depth 在 derivative 算完后才决定跳过，不能节省主成本。

**已确证（不再展开）。** 步数敏感性审计已完成：`ode_steps={1,3,6,12}` test PCC 差异 ≤0.002（单 seed 噪声级）；默认 `ode_steps` 已落地为 3。

**遗留问题。** 固定步长对可能刚性的 dynamics 缺少误差控制和稳定性证据；纯确定性 ODE 不能表达随机涨落或状态跳变；"已验证连续生理 ODE"的声明继续不成立。

**拟解决方案。**

1. 对照 RK4、可学习步长或自适应 solver（如 dopri5），明确训练稳定性与开销；不要仅因"ODE"命名引入复杂求解器。
2. Langevin noise/SDE 分支只作为 P2 探索，须给出不确定性校准和状态跳变收益；若单步与多步的 subject-level CI 重叠（需多 seed），应简化为 residual dynamics block。

### 1.4 MoE 路由坍缩持续（最高优先级，重验未通过）

**现状（2026-09-26 重训后）。** 评估期 top1 分配 675/675 集中于 E3（100%）；gate 均值 E0=E1=0，E2/E3 各约 0.26（top-2 份额）；训练 2 轮后 E0/E1 完全死亡（Load=0），4 专家仅约 2 个有效（normalized entropy≈0.50）。**multinomial 修复恢复了梯度信号（合成验证负载 0.24/0.24/0.24/0.27），但真实数据上路由仍坍缩**——与 1.1 同根因：gate 输入不含病理编码，且容量惩罚/负载均衡正则不足以对抗坍缩吸引子。

**已确证（不再展开）。** MoE 结构收益确证（`no-MoE` ΔPCC=−0.0308，四 HAMD 分层一致变差）；收益主体是路由专家，shared expert 增量≈0；默认 `moe_experts_mode=routed_only` 已落地。"自动发现临床亚型"声明不成立。

**未验证项（对策实验已注册，执行命令）。**

```bash
bash RunAblation.sh --group G14_MOE --only moe_exp2_t1,moe_argmax,moe_temp_low
```

- `moe_exp2_t1`：专家数 4→2、top_k 2→1（对齐实际有效容量 + 硬单选）；
- `moe_argmax`：训练期容量感知 argmax 硬路由（兼作 multinomial 修复的消融对照）；
- `moe_temp_low`：路由温度 1.5/1.0→1.0/0.5（锐化概率分布）。

评估关注：top1 分配分布、normalized entropy、expert-HAMD 分层、条件敏感性（与 1.1 联动）。

**拟解决方案。** state-aware routing（`moe_gate_difficulty`，P1 已注册未跑）；专家数缩减至实际有效容量；配合 1.1 的 gate 输入注入病理编码；稳定后再探索异构 experts 与 token-level routing。

### 1.5 ForecastHead 幅值重组与多分支冗余（部分未解决）

**已确证（不再展开）。** query decoder 贡献在幅值精度（`head_flatten` PCC +0.0008 噪声级但 MAE +0.0176 变差），暂保留 query decoder。

**未验证项（预训练混淆对照）。** `head_flatten` 的 shape 分支未预训练，其 MAE 损失可能混入"分支未预训练"代价。已注册 P0 对照 `head_shape_scratch`（baseline 结构 + finetune 强制 `future_query.*` 随机初始化，`--pretrained_skip_pattern` 受控跳过）。判读：head_shape_scratch ≈ head_flatten 则 1.5 节"保留 query decoder"结论需重新评估；可直接复用 baseline 预训练权重。

**未实施。**

1. 幅值重参数化：trend 同受 scale 调制或一致乘法表达；trend/scale consistency regularization；逐时间点 scale 与窗级 scale 对照；
2. 六路 branch leave-one-out（`head_no_*`，P1）与 GraphODE/head attention 去重消融；
3. `pred_window=1/2/3` 扩展性与长程误差比较（与 2.2 方案 3 联动）。

### 1.6 Refiner 自适应轮数（小项未解决）

**已确证（不再展开）。** `round_scales` 初始化修复已落地；base refiner 收益确证（`refiner_none` ΔPCC=−0.0045、ΔMAE=+0.024）；`delta_refiner_rounds=1` 已落地。

**未解决。** 轮数对所有样本固定、轮间无误差反馈。低风险改造：每轮按 residual magnitude 用轻量 gate 决定是否继续 + 轮间 residual supervision，比较 1--4 轮、固定轮数与早停轮数；须与 SC 注入位置消融联动（`base_refiner_rounds` P1 变体已注册未跑），避免"更多 refinement"加重过平滑。

### 1.7 多尺度、长期滚动与状态承接缺口（未解决）

BrainMDM 的 sequence/window 多尺度路径固定全开；`W=6` 时名义三种窗口尺度去重后约只有两种有效。应先做各 branch removal，再测试基于 pooled latent 的 sample-conditioned scale gate，避免无控制扩张。

当前逐窗并行编码、`pred_window=1`，没有将 ODE 终状态传给下一窗的自回归承接机制，不具备长程滚动外推的结构条件。P3 可测试"ODE terminal state→next-window initial state"递推设计 + scheduled sampling，评估 `pred_window∈{1,2,3}` 与误差—预测时程曲线。必须先完成单窗可信基线；不应与动态图、复杂 decoder 同时引入。

### 1.8 不确定性与 RevIN 幅度信息（PICP 校准已解决，其余未解决）

**已解决（2026-09-26 实战生效，不再列为问题）。** PICP 过覆盖已由评估侧 conformal 方差校准解决：重训模型 raw PICP 0.9619 → PICP_cal 0.9030（目标 0.90），MPIW 1.912→1.394（收窄 27%）。

**未解决（P2/P3 研究路线）。**

1. 异方差高斯/分位数预测头与 calibration curve 报告（当前 NLL 方差系统性上偏，仅靠事后校准缓解，训练侧改进未做）；
2. 辅助反演头 latent→HAMD 多任务正则（与 1.1 方案 2 联动）；
3. 病理条件扫掠的反事实模拟接口；
4. BrainRevIN mean/std 保留为 MoE 条件或预测头辅助特征，对照是否改善病理判别——MDD 整体幅度变化可能具病理信息，逐样本归一化可能抹除它。

上述能力须避免过度宣称：预测方差只是模型不确定性估计；反演头与 counterfactual 曲线不自动等同临床因果结论。

## 2. 实验设计、数据协议、评估与工程问题

### 2.1 统计严谨性剩余缺口（主体已解决，剩余待补）

**已解决（不再展开）。** 版本化固定 split manifest、`main.py` 训练后自动评 test、被试级配对 t/Wilcoxon 检验、被试级 (dataset, site) 8:1:1 切分——均已落地并经 2026-09-26 重训实战验证（train 1803 / val 225 / test 225 被试，HAMD 分布均衡）。多 seed 复验按 2026-09-26 决定暂不开展。

**未解决。**

1. **多重比较校正**：`--paired_tests` 报告注明未校正，正式结论前需补 Holm/BH 校正（工具就绪，待补实现）；
2. **历史口径警示**：既有 pred1/pred12/pred13 与 Version0 历史结果是 1,353 个 validation samples（`val_ratio=0.2`、旧切分），与当前独立 test（225 被试 manifest 锁定）不可横向比较；
3. 解冻 backbone 时 optimizer/scheduler 重建行为（参数集合变化 + LR 重启）未作为实验条件写入配置快照。

### 2.2 评价维度剩余缺口（平凡基线已解决，其余未解决）

**已解决（不再展开）。** FC 上三角 MAE/edge-PCC/网络级指标已扩展；无模型平凡基线（persistence/linear/mean）已落地并回答核心问题：模型 FC_upper_MAE 0.1494 / edge_PCC 0.8477，显著优于 persistence（0.2131/0.6594）、mean、linear——FC 重构收益成立。

**未解决。**

1. **dFC 评估**：滑窗 FC 矩阵相关、k-means state 驻留时间与转移矩阵 KL、PSD/fALFF——未实施，在此之前不得称 dFC prediction；
2. **VAR/岭回归基线**：仍走变体框架，未实施；
3. **`pred_window={1,2,3}` rolling evaluation** 与 error-horizon 曲线；`pred_window>1` 时确认 flatten 后 `torch.diff` 跨窗口物理相邻；
4. **效率报告规范**：所有结果同时报告 params、FLOPs/MACs、GPU memory、throughput/epoch time（当前仅 params 落盘）。

### 2.3 病理条件、协变量与多站点数据协议（未解决）

现有 dataloader 仅供应一维 HAMD，无条目/因子级症状、用药、站点、性别、年龄等协变量；限制 1.1 的向量条件建模，也可能使临床混杂被误归为病理机制。

**拟解决方案。**

1. 构建 HAMD 条目/因子向量（焦虑、躯体化、认知阻滞等），与 scalar HAMD 做独立和组合消融；
2. 纳入可用协变量（用药、站点、人口学），明确缺失值策略；每类协变量做贡献消融；
3. REST-meta-MDD 多中心数据：ComBat/neuroCombat 协调 ROI 信号，或至少 site one-hot 条件化，协调前后对照；
4. **站点分层划分待接入**：`scripts/site_stratified_split.py` 与 `data/splits/site_stratified_split_{MDD,HC}_seed2024.json` 已产出但未接入训练（当前 (dataset, site) 切分仅保证站点不泄漏，未做站点分层平衡）；接入后做 site-stratified 与 leave-one-site-out 测试，报告各站点方差。

### 2.4 独立队列外部验证（未解决）

HAMD 合并表 `Rest-meta-MDD-V1V2-Merged-MDD.xlsx` 汇集 V1、V2 两个**独立队列**（非纵向随访）。V2 尚未进入外部验证，跨队列泛化未知。

**拟解决方案。** 以 V1 训练、V2 整体作为锁定 external test，除主指标外报告 FC、校准、HAMD/站点分层及与 V1 内部 test 的差距。不得表述为 longitudinal validation。

### 2.5 论文声明、失败分析与解释性纪律（未解决）

- 在显式 FC 监督或完整派生 FC/dFC 评估前，不称 dFC prediction；
- 在 solver/step 证据前，不称当前 GraphODE 为已验证连续生理 ODE；
- 不从随机 routing、滑窗级统计或单次 seed 推出临床亚型/生物标志物；
- 可准确强调：SC prior、时空建模、HC→MDD 两阶段迁移、subject-level split、FC 重构优于平凡基线（2026-09-26 实证）；
- 预印本证据（BrainSymphony、BrainATCL、BrainWorld 等）应与同行评审工作明确区分。

**失败案例分析。** 初步产物已有（2026-09-26，`eval/low_pcc_analysis/`）：低 PCC 被试 23/225（10.2%），其**年龄显著更高**（44.6 vs 35.4，p=0.019）、**HAMD 显著更低**（16.3 vs 22.0，p=0.003），病程/性别不显著。待深入：按 HAMD low/medium/high、SC density、ROI network、forecast volatility、expert confidence/entropy 分层的系统 failure analysis；检验 adaptive graph 是否主要改善原 baseline 的低 PCC 被试；置换特征重要性 FDR 后无显著脑区（fast 模式，单 ROI 扰动影响 0.0055–0.0072 且网络间无差异）的解释与批量置换复验。

### 2.6 工程与可复现性（未解决）

待办：以 YAML 配置 + CLI override 收敛散落的 argparse/shell 超参；设备自动选择（当前硬编码 `cuda:1`），按需求支持 DDP；为 dataloader、loss、MoE routing 建立单元测试（当前仅 split manifest 有单测）；启动时写入完整配置、split、随机种子与 routing mode 快照。

### 2.7 统一优先级矩阵（未解决/未验证项）

| 优先级 | 项目 | 关键对照 / 产出 | 状态 |
|---|---|---|---|
| P0 | 病理条件注入失效修复（1.1） | gate 输入注入 HAMD、条件监督增强；shuffled ΔPCC 显著非零 | 机制修复已实施，真实数据复验未通过，结构性方案未实施 |
| P0 | MoE 路由坍缩对策实验（1.4） | moe_exp2_t1 / moe_argmax / moe_temp_low；top1 分布与熵 | 已注册未跑 |
| P0 | head_shape_scratch 预训练混淆对照（1.5） | vs baseline vs head_flatten | 已注册未跑 |
| P0 | 配对检验多重比较校正（2.1） | Holm/BH 校正纳入 paired_tests | 工具就绪待补 |
| P1 | 病理归一化对照与向量输入（1.1） | raw/robust_z/RBF；scalar/vector | 未实施 |
| P1 | 特征级病理调制（1.1） | cond_residual / cond_feature | 已注册未跑 |
| P1 | SC 注入位置与 λ 自适应消融（1.2） | sc_no_inject / sc_lambda_* / sc_no_delta_a | 已注册未跑 |
| P1 | ForecastHead 幅值重组与 branch LOO（1.5） | head_no_*；scale 重参数化 | 未实施 |
| P1 | Refiner 轮数消融与自适应 gate（1.6） | base_refiner_rounds；adaptive gate | 已注册/未实施 |
| P1 | VAR/岭回归基线与 dFC 状态分析（2.2） | VAR vs persistence vs model | 未实施 |
| P1 | `pred_window=1/2/3` rolling（2.2） | error-horizon 曲线 | 未实施 |
| P2 | 多站点协调与站点分层接入（2.3） | raw/ComBat/site-condition；site variance | 划分已产出未接入 |
| P2 | ODE solver/随机项（1.3） | RK4 / dopri5 / SDE | 未实施 |
| P2 | 效率报告规范（2.2） | FLOPs/MACs/显存/吞吐 | 未实施 |
| P3 | 窗口间状态递推、HAMD 反演、反事实（1.7/1.8） | scheduled sampling；反演误差 | 未实施 |
| P3 | V2 外部验证（2.4） | V1→V2 | 未实施 |
| P3 | RevIN 统计条件、YAML/DDP/单测（1.8/2.6） | with/without RevIN stats；unit tests | 未实施 |

**建议顺序。** 先执行已注册的 P0 对策实验（G14_MOE 组 + head_shape_scratch，零开发成本），同时实施 1.1 的 gate 输入注入病理编码与条件监督增强——这两项决定 MoE/条件化是否值得继续投入；随后按矩阵推进 SC 消融与幅值重组；最后多站点、ODE solver、状态递推与外验。每次只引入一个主要机制，保留参数量、训练成本与失败案例记录。

### 2.8 修改记录

| 日期 | 主要修改 |
|---|---|
| 2026-09-21 | 初版建立系统/结构问题与实施路线。 |
| 2026-09-22 | 明确 V1/V2 为独立队列；记录数据根目录和效率优化的已解决状态；补充站点分层划分待接入状态。 |
| 2026-09-22 | 合并既有"当前问题与修复"整合稿与本次补充内容；去除重复项，正文重构为"模型结构""实验设计、数据协议、评估与工程"两大块。 |
| 2026-09-22 | 将正文 4 处 `\[...\]` 公式转换为 Markdown 可渲染的 `$$...$$` 行间公式；移除公式内中文句号，多字母标识符统一用 `\text{}` 排版。 |
| 2026-09-25 | 补充 P0 消融实验（13 变体、seed=2024）实证结论至 1.1--1.6；1.1 条件失效被 shuffled 负对照确证、1.4 MoE 收益确证但路由坍缩加剧；已落地参数调整（`ode_steps` 6→3、`delta_refiner_rounds` 2→1、`moe_experts_mode` routed_shared→routed_only）；标注已完成代码项（refiner 初始化修复、`refiner_rounds=0` 支持、消融评估闭环）；修正 1.4 中 `moe_router_cond_only` 过时描述。 |
| 2026-09-25 | `main.py` CLI 默认参数与 `experiments/base_config.py` 对齐：`pred_window` 3→1、`ode_steps` 6→3、`delta_refiner_rounds` 2→1、`moe_experts_mode` routed_shared→routed_only，以及 `train_epochs/batch_size/lr_init/lr_final/warmup_epochs/patience/moe_load_balance_weight/moe_gate_temp_start/compile` 等训练规模项；`loss_lr_scale` 保持 1.0（随默认 pretrain 模式，finetune 由脚本显式传 0.5），属模式相关差异而非失同步。 |
| 2026-09-25 | `scripts/Pretrain_HC.sh` 同步取消融值：`ode_steps` 6→3、`delta_refiner_rounds` 2→1，与 `Finetune_MDD.sh`/`base_config.py` 对齐，两阶段 ODE 配置保持一致。 |
| 2026-09-25 | 评估分析脚本收敛为单一入口：`experiments/evaluate_variant.py` 吸收原 `analysis/run_comprehensive.py` 全部独有能力（expert-HAMD 分层分析、可视化、置换特征重要性 + FDR 显著脑区导出、预测数组保存），作为 `--visualize / --feature_importance / --save_arrays` 开关；`run_comprehensive.py` 已删除。`analyze_low_pcc_samples.py` 改读 `sample_metrics_<split>.csv`；README 与 Version1 文档引用同步更新。 |
| 2026-09-26 | **HAMD 条件注入失效根因修复**：`models/neurotwin_moe.py` 训练期路由选择由 Gumbel-argmax + 自适应高温恢复为 multinomial 概率采样（与设计一致），修复专家 FiLM 条件调制与路由器条件→专家映射同时失去梯度信号的问题；合成验证（真实正则配置）条件敏感性 ≈0→22.3%、训练期专家负载恢复近均衡、评估期 top1 集中度 1.000→0.688。 |
| 2026-09-26 | **注册 MoE 路由坍缩对策实验**（1.4 节，G14_MOE 组新增 3 个 P0 变体）：`moe_exp2_t1`、`moe_argmax`、`moe_temp_low`；变体总数 35→39（含同日 `head_shape_scratch`），全部通过 overrides 校验、dry-run 与 2 专家前向冒烟。 |
| 2026-09-26 | **PICP 过覆盖 conformal 方差校准**（1.8 节）：`analysis/metrics.py` 新增 `fit_conformal_scale`（val 拟合归一化残差经验高分位，finite-sample 修正），`evaluate_variant.py` 默认启用校准（`--no_calibrate` 关闭，val 自动补评拟合），产出 `PICP_cal/MPIW_cal`；registry/compare/run_experiments 同步 `test_picp_cal` 字段。 |
| 2026-09-26 | **注册 head_flatten 预训练混淆对照实验**（1.5 节）：`load_backbone_weights` 新增 `skip_pattern` 受控跳过（fnmatch 命中键保留随机初始化，单测验证），main.py 新增 `--pretrained_skip_pattern` CLI；注册 P0 变体 `head_shape_scratch`。 |
| 2026-09-26 | **完善评估/实验方法类任务（2.1/2.2 节）**：① 版本化固定 split manifest（单测 5/5）；② finetune 训练结束自动评 test（`--no_eval_after_train` 可关）；③ `compare.py --paired_tests` 被试级配对 t/Wilcoxon 检验；④ 新建 `analysis/fc_baselines.py` 无模型基线套件（persistence/linear/mean）。另修复 `main.py --help` 因 `--compile` help 字符串含未转义 `%` 导致的崩溃。 |
| 2026-09-26 | **新代码重训复验**（`neurotwin_pretrain_pred1`/`neurotwin_finetune_pred1`，seed=2024）：test PCC 0.8336 / MAE 0.3443（较 P0 baseline 改善 12.5%）；PICP_cal 0.9030 实战生效；FC 显著优于平凡基线；**条件注入失效与路由坍缩复验未通过**（shuffled ΔPCC=1.06e-5，top1 100% E3，E0/E1 死亡）——根因更新为 gate 输入不含病理编码 + 重建任务缺乏条件激励，见 1.1/1.4。完整评估链路产物（28 组可视化、特征重要性、FDR xlsx、预测数组、低 PCC 分析）落盘 `checkpoints/neurotwin_finetune_pred1/eval/`。 |
| 2026-09-26 | **按用户决定清除已解决条目**：正文重构为仅含未解决/未验证问题（已解决项：PICP 校准、评估闭环三项、平凡基线、ode_steps/refiner/routed_only 参数落地、MoE 收益与步数不敏感性确证、refiner 初始化修复、query decoder 保留结论等）；`results/ablation/` 原始产物已清理，P0 结论以本文档与修改记录为准。 |
