# NeuroTwin 当前存在问题及解决方案

> 更新日期：2026-09-22  
> 依据：对当前代码库（`models/`、`train/`、`utils/`、`analysis/`、`scripts/`）的静态审计，以及既有审计、架构研究与训练记录。  
> 状态：待实施；“已解决项”仅保留为工程历史，不再列入待办。  
> 文档边界：本文只记录当前问题、修复方案、验证标准与实验协议。架构候选的完整文献池和研究组合见 `NeuroTwin_模型架构优化改进整合版.md`。

当前任务严格是 **SC 条件下的未来 ROI/BOLD 信号预测**：

\[
\text{history ROI BOLD}+\text{SC}\rightarrow\text{future ROI BOLD}。
\]

MDD 微调在 HC backbone 预测上学习病理残差：

\[
\hat Y_{\mathrm{MDD}}=\hat Y_{\mathrm{HC-base}}+\Delta Y_{\mathrm{pathology}}。
\]

当前最关键的共同问题不是继续增加模块，而是：SC 以过硬且重复的方式注入、病理条件没有与脑状态共同作用、ForecastHead 过早丢失未来位置语义，以及评估闭环尚不完整。下文仅分为“模型结构”和“实验设计”两大块；每项同时给出问题、拟解决方案和必须验证的对照。

## 1. 模型结构问题与解决方案

### 1.1 当前结构、任务语义与病理条件化设计

当前典型配置为 AAL `F=116`、每窗 `S=30`、`total_windows=9`、`in_window=6`、`pred_window=1`：

```text
x:[B,116,6,30]，y:[B,116,1,30]，SC:[B,116,116]，MDD HAMD:[B,1]
x → BrainRevIN → DFCAdapter（SC 0/1/2-hop）→ BrainMDM
  → GraphODEDDI × 2（SC attention + temporal conv + window attention + FFN；RK2×6）
  → post-fusion → ForecastHead（anchor + trend + scale×shape + SC refinement）
  → base_pred
  → MDD MoE（HAMD 条件残差 + shared expert + SC delta refinement）→ prediction
```

当前 HC→MDD 的“基础通用动力学 + 病理残差”方向合理，但其病理条件化存在四层局限：

- **表达粗糙且尺度失衡**：`pathology_input_dim=1`，`RicherPathologyProjection` 使用 `[x,x²,sin(πx)]`。记录的原始 HAMD 约 1--58 且常为整数，`x²` 最大约 3364、`sin(πx)` 近零，会主导或失衡病理 embedding；同时单标量只能表达严重度，无法表达焦虑、躯体化、认知等异质症状维度。
- **注入过浅**：默认病理信息主要经输出端 MoE residual 生效，BrainMDM、GraphODEDDI 的特征形成与动力学本身对病理无感。
- **可加性假设过强**：`pred=base_pred+Δ(pathology)` 假定病理可由输出平移补偿；若病理改变有效连接、状态切换速率或振荡幅度，错误基线上叠加残差的上限有限。
- **微调覆盖不足**：MoE 仅在微调阶段出现，HC backbone 内已形成的病理动力学失配不能由末端专家充分修正。

**拟解决方案。**

1. 每个训练 fold 用 train subjects 拟合并冻结 robust z-score、经验 CDF 或 quantile HAMD transform；比较 `z-score linear`、`z-score+quadratic`、`z-score+RBF/spline`，移除 raw `sin(πx)` 默认路径。
2. 将病理条件升级为向量输入（HAMD 条目/因子与协变量由第 2 节数据协议提供）。
3. 在 BrainMDM/GraphODEDDI 内加入 FiLM 或 AdaLN 条件调制：`normalized pathology→γ,β,α`，对中间特征做 scale/shift，并让零初始化 residual gate 保持 HC 路径初始近似 identity。
4. 保留输出端 residual，与特征级调制形成“双层条件化”；必须对照 **additive residual only / feature modulation only / 两者结合**。
5. MDD 微调阶段探索最后 1--2 个 ODE block 的 LoRA 式低秩 adapter，先冻结 HC backbone 训练 adapter，再小学习率联合微调。

**验收。** 不能只比较整体 PCC；还需比较 HAMD 分层、低样本比例（25/50/75/100% subjects）、shuffled-HAMD negative control，以及 HC 表征保真与 MDD 泛化。若真实与打乱 HAMD 的表现近似，不应提出 pathology-conditioned 的强结论。

### 1.2 SC 约束过硬、重复注入且缺乏个体/状态适应

**现状与问题。** `DFCAdapter` 的 `SC⊙learned_edge` 只能重标已有边，SC 为零时永远不能出现功能依赖；GraphODE attention 又把 `sc_with_self>0` 当二值 hard mask，并施以固定 soft bias。这样 DTI 假阴性边会永久切断，受试者共享先验、无方向性，且静态 SC 无法表达动态 FC。与此同时，SC 在 DFCAdapter、GraphODE、base refiner、MoE delta refiner 中重复注入，可能使结构先验压过功能证据并产生 over-smoothing。

ForecastHead 的 `cross_roi_*` 实际是 ROI channel 上 `Conv1d(kernel_size=1)` 的静态全连接 ROI×ROI mixing，不是 graph attention；它允许任意 ROI 交互，与上游“非 SC 边不可达”的假设并不一致。

**拟解决方案。**

1. 从 SC hard mask 改为 soft anatomical prior：

   \[
   A_t=\lambda_tA_{\mathrm{SC}}+(1-\lambda_t)A_{\mathrm{func}}(h_t)。
   \]

   以 pooled ROI state 生成低秩 `A_func=softmax(sym(UV^T))`，先使用 rank 8--16；`λ` 依次尝试全局标量、sample-conditioned、ROI-specific。
2. 可学习概率掩码可作为备选：straight-through/Gumbel 在 SC 先验与数据证据间选择；同时加入低秩 subject-specific 邻接残差 `ΔA`、稀疏/熵/时间一致性正则。状态/时间调制和有向化仅在低风险版本验证后进入探索。
3. 对 Adapter、ODE、base refiner、delta refiner 的 SC 注入逐个消融，确定最小有效注入位置；预测端只保留一次轻量 refinement 作为候选。

**验收。** 比较 `no-SC / fixed-SC / adaptive-only / SC+static-adaptive / SC+dynamic-adaptive`，并报告固定/动态 `λ`。除 waveform 外必须报告 FC 指标、低 PCC 个体改善、图稀疏度/熵与计算成本。

### 1.3 GraphODE 的连续时间主张尚未成立

**现状。** 未使用 `torchdiffeq/odeint`；`ode_steps=6` 为固定步数手动 RK2/Heun 更新，`GraphODE.forward(t,...)` 实际丢弃 `t`，即共享自治向量场。`n_block=2, ode_steps=6` 约需 24 次完整四支路 derivative evaluation；stochastic depth 在 derivative 算完后才决定跳过，不能显著节省该主成本。

**问题。** 固定步长对可能刚性的 dynamics 缺少误差控制和稳定性证据；纯确定性 ODE 也不能表达随机涨落或状态跳变。当前更准确的名称是 shared-parameter graph-spatiotemporal RK2 residual block，而不是已验证的连续生理时间模型。

**拟解决方案。**

1. 首先做强制敏感性审计：`ode_steps={1,3,6,12}`，比较 PCC/MAE、FC edge-PCC/MAE、params、FLOPs/MACs、GPU memory、samples/s、epoch time、gradient clipping frequency、`||k1||`、`||k2||`、`||Δh||`。
2. 若多步有稳定收益，再对照 RK4、可学习步长或自适应 solver（如 dopri5），并明确其训练稳定性和开销；不要仅因“ODE”命名而引入复杂求解器。
3. Langevin noise/SDE 分支只作为 P2 探索，并须给出不确定性校准和状态跳变收益；若单步与六步的 subject-level CI 重叠，则简化成 residual dynamics block。

### 1.4 MoE：路由条件不足、专家同质且粒度过粗

**现状。** 默认 4 个同容量 experts、top-2、sample-level routing；`moe_router_cond_only=True` 使 router 主要只看 HAMD。前向中已经构造 `hist_mean/hist_std/latent_mean/latent_std/base_mean` 共 `5F` 脑状态统计量，却没有进入默认路由；同一患者在不同脑状态可得到近似路由。共享 pathology expert 始终激活，也可能吸收大部分 residual，使 routed experts 难分化。

现有验证 routing 还可能在 `eval()` 下用低温 multinomial 抽样；`pred13` top-1 expert 使用数为 `217/688/438/10`，第四专家近乎空转，已有 expert-HAMD Mann--Whitney 比较不显著。因此当前不能宣称自动发现临床亚型。

**拟解决方案。**

1. 立即放开 `moe_router_cond_only`，以 `normalized HAMD + 5F state summary + pooled latent/base difficulty` 作为 router 条件；validation、early stopping、test 强制 deterministic dense soft mixture 或 deterministic top-k。
2. 先验证四专家 dense differentiable Soft-MoE；随机 routing 只能作为 10--20 次 Monte-Carlo uncertainty 实验，报告 mean±std、温度和种子。
3. 比较 `no-MoE / shared-only / routed-only / hard top-k / deterministic dense soft`；增加 expert output diversity、pairwise cosine、expert ablation `ΔPCC`、跨窗 routing consistency，而不仅优化负载均衡。
4. 在稳定基线后探索异构 experts（不同感受野、频带、深度或功能路径）及 token-level ROI×window routing；专家数和 top-k 必须系统消融，不能先扩大规模。

### 1.5 ForecastHead 语义、幅值重组与多分支冗余

**现状。** 对 `W=6,S=30`，history/latent 很早压平为 180 维，六路 history/latent MLP、temporal、cross-ROI 与 window-attention 分支融合。输出采用：

\[
\hat y=anchor+trend+softplus(scale)\cdot standardized(shape)。
\]

**问题。**

- future ROI、future window、future time 没有显式 query 语义；当 `pred_window` 增加时，flatten readout 是信息压缩瓶颈；
- trend 与 `scale×shape` 可能重复承载幅值，独立回归会相互拉扯；scale 是每窗单一标量，不能表达窗内幅度演化；
- 多个分支均由相同 history/latent 派生，GraphODE 和 ForecastHead 的 window attention 也可能重复。

**拟解决方案。**

1. 首版保留 anchor+trend，只把 shape branch 替换为 Future ROI-Time Query Decoder：`ROI embedding + future-window embedding + future-time embedding → cross-attention(memory) → residual`。
2. 重参数化幅值结构，例如使 trend 同受 scale 调制，或使用一致乘法表达；增加 trend/scale consistency regularization，并对逐时间点 scale 与窗级 scale 做对照。
3. 做 current flatten / query decoder、六路 branch leave-one-out、GraphODE/head attention 去重的消融；`pred_window=1/2/3` 上比较扩展性和长程误差。

### 1.6 `IterativePredictionRefiner`：初始化错误与固定轮数

`round_scales` 使用 `0.5**i`，首轮目标实为 1，经 sigmoid 后约 0.9999，而非注释中的 0.5。修复为显式 `[0.50,0.33,0.20]`；比较 no-refiner、one-round、current three-round、corrected three-round。

此外 `n_rounds=3` 对所有样本固定、轮间没有误差反馈，无法按预测难度自适应。低风险改造是在每轮根据当前 residual magnitude 用轻量 gate 决定是否继续，加入轮间 residual supervision，并比较 1--4 轮、固定轮数与早停轮数。该机制必须和 SC 注入位置一起消融，避免“更多 refinement”仅加重过平滑。

### 1.7 多尺度、长期滚动与状态承接缺口

BrainMDM 的 sequence/window 多尺度路径固定全开；在当前 `W=6` 时，名义三种窗口尺度去重后约只有 `1/3` 两种有效尺度。应先做各 branch removal，再测试基于 pooled latent 的 sample-conditioned scale gate，避免无控制扩张。

当前逐窗并行编码、`pred_window=1`，没有将 ODE 终状态传给下一窗的自回归承接机制，尚不具备长程滚动外推或持续仿真的结构条件。后续 P3 可测试“ODE terminal state→next-window initial state”的递推设计，训练中 scheduled sampling，评估 `pred_window∈{1,2,3}` 和误差—预测时程曲线。必须先完成单窗可信基线；不应把这一扩展与动态图、复杂 decoder 同时引入。

### 1.8 不确定性、双向映射与 RevIN 幅度信息

**不确定性与孪生能力。** 当前是点估计；推理还需要 HAMD 输入。可作为 P2/P3 研究路线：

1. 异方差高斯（均值+方差、NLL）或分位数预测，报告 calibration curve；
2. 辅助反演头由 latent 预测 HAMD，形成脑状态→严重度的多任务正则；
3. 扫掠病理条件，生成“症状变化→动力学响应”的反事实模拟接口。

这些能力必须避免过度宣称：预测方差只是模型不确定性估计；反演头与 counterfactual 曲线不自动等同临床因果结论。

**BrainRevIN。** 它按 sample、ROI 对历史窗口统计并 detach mean/std，降低幅度尺度差异且在输出逆变换，是合理通用组件。但 MDD 的整体幅度变化可能具病理信息，逐样本归一化可能抹除它。保留 RevIN mean/std 作为 MoE 条件或预测头辅助特征，比较是否改善病理判别和预测；不应仅将它包装为核心创新。

## 2. 实验设计、数据协议、评估与工程问题

### 2.1 数据切分、训练/测试闭环与统计单位

当前 9 个窗在 `6→1` 设置下每被试通常产生 3 个样本；subject-level split 防止同一被试滑窗跨 train/val/test，是正确且不可退让的设计。MDD 已有 HAMD 分位分箱。SC 预处理为 NaN/Inf→0→对称化→负值截断→`log1p`→99th-percentile scaling→`[0,1]`；train 有 noise/scale/channel-drop/time-mask，val/test 不增强。

但评估闭环尚不完整：`main.py` 虽构造 train/val/test loader，主训练流程只使用 train/val，未自动加载最佳 checkpoint 执行最终 test；既有 pred1/pred12/pred13 是 1,353 个 validation samples 而非独立 test，且分析曾用 `val_ratio=0.2`、主 CLI 默认 `0.1`，不能作严格横向结论。

**拟解决方案。**

1. 版本化固定 subject-ID split（例如 `split_seed2024.json`）；训练、分析、所有 baseline/new architecture 共用，训练完成自动 load best checkpoint 并一次性输出 val/test 双列指标。
2. 最终结论采用至少 5 个 seeds×subject-level 切分；报告 mean±std、subject-level bootstrap 95% CI。架构筛选可先用 3 seeds。
3. 被试内聚合滑窗后再做 paired bootstrap/permutation；在 `analysis/metrics.py` 增加模型间配对置换/Bootstrap 与 ROI FDR。不能把滑窗、重复 batch 或随机 routing 当作独立临床重复。
4. 记录解冻 backbone 时重建 optimizer/scheduler 的行为：它会使参数集合变化和 LR schedule 重启同时发生，必须作为实验条件写入配置快照。

### 2.2 评价维度、基线套件与长期预测

当前 metrics 以信号级 PCC/MAE/RMSE/R²、ROI/window 指标为主，不能单独回答“功能连接动态组织是否复现”。主损失为 waveform PCC、MAE、diff、std 的 log-variance 加权；当 `pred_window>1` 时还需确认 flatten 后 `torch.diff` 跨窗口是否在物理时间上相邻。

**拟解决方案。**

1. 从预测 waveform 派生 Pearson FC，并报告 FC 上三角 MAE/RMSE、edge-PCC、network-wise、within-/between-network error；必要时低权重 Fisher-z FC loss。进一步的 dFC 评估可包括滑窗 FC 矩阵相关、k-means state 的驻留时间与转移矩阵 KL、PSD/fALFF。
2. 建立基线套件：persistence、VAR/岭回归、无 SC Transformer、no-MoE、no-ODE，以及各结构模块单独消融；形成清晰对照表。
3. 增加 `pred_window={1,2,3}` rolling evaluation，绘制 error-horizon 曲线；同步报告 refiner 轮数 `1--4` 的敏感性。
4. 所有结果同时报告 params、FLOPs/MACs、GPU memory、throughput/epoch time，避免把不可接受的代价隐藏在小幅 PCC 增益后。

### 2.3 病理条件、协变量与多站点数据协议

现有 dataloader 仅供应一维 HAMD，无条目/因子级症状、用药、站点、性别、年龄等协变量；这限制了第 1.1 节的向量条件建模，也可能使临床混杂被误归为病理机制。

**拟解决方案。**

1. 在数据侧构建 HAMD 条目/因子向量（焦虑、躯体化、认知阻滞等），并在模型侧与 scalar HAMD 做独立和组合消融。
2. 纳入可用协变量（用药、站点、人口学），明确缺失值策略；每类协变量均做贡献消融，避免将性能变化直接当成病理因果。
3. REST-meta-MDD 为多中心数据；使用 ComBat/neuroCombat 协调 ROI 信号，或至少将 site one-hot 作为条件，并对协调前后作对照。
4. 将已产出的 `scripts/site_stratified_split.py`、`data/splits/site_stratified_split_{MDD,HC}_seed2024.json` 与实体化数据划分真正接入训练，替换内部随机切分；再做 site-stratified 与 leave-one-site-out 测试，报告各站点方差。现有“脚本/划分已产出”是待接入状态，不应误写为训练已使用。

### 2.4 独立队列外部验证

HAMD 合并表 `Rest-meta-MDD-V1V2-Merged-MDD.xlsx` 汇集 V1、V2 两个**独立队列**；它们不是同一被试的纵向随访。V2 尚未进入外部验证，跨队列泛化未知。

**拟解决方案。** 以 V1 训练、V2 整体作为锁定 external test，除主指标外报告 FC、校准、HAMD/站点分层及与 V1 内部 test 的差距。不得把该设计表述为 longitudinal validation。

### 2.5 论文声明、失败分析与解释性纪律

- 在显式 FC 监督或完整派生 FC/dFC 评估前，不称 dFC prediction；
- 在 solver/step 证据前，不称当前 GraphODE 为已验证连续生理 ODE；
- 不从随机 routing、滑窗级统计或单次 seed 推出临床亚型/生物标志物；
- 可准确强调 SC prior、时空建模、HC→MDD 两阶段迁移、病理条件残差和 subject-level split；
- 预印本证据（如 BrainSymphony、BrainATCL、BrainWorld）应与同行评审工作明确区分。

必须做 failure-case analysis：HAMD low/medium/high、低 PCC subjects、SC density、ROI network、forecast volatility、expert confidence/entropy，以及有元数据时的 motion/noise。尤其要检验 adaptive graph 是否主要改善原 baseline 的低 PCC 被试；这往往比平均 PCC 的微小差异更有解释力。

### 2.6 工程与可复现性

待办包括：以 YAML 配置+CLI override 收敛散落的 argparse/shell 超参；设备自动选择，按需求支持 DDP；为 dataloader、loss、MoE routing 建立单元测试；启动时写入完整配置、split、随机种子与 routing mode 快照。

以下属于**已解决的工程历史**，不再占用待办优先级：数据根目录已统一至项目内 `data/` 并完成冒烟验证；效率侧已有 `torch.compile`（稳态约 35% 提速）、统计同步合并、EMA `_foreach_` 批量化、验证/测试较大 batch，以及按被试聚合的 mat→npz 缓存（`data/npz_cache/`，dataloader 自动优先读取）。仍应在不同硬件/数据规模下复核性能数字，且不把缓存文件纳入版本控制。

### 2.7 统一优先级、实施顺序与验收矩阵

| 优先级 | 项目 | 关键对照 / 产出 |
|---|---|---|
| P0 | 评估闭环：锁定 test、多 seeds、统计检验 | 自动 test、≥5 seeds、subject-level CI/paired test |
| P0 | HAMD 规范化与确定性 routing | raw/normalized HAMD；random/deterministic routing |
| P0 | FC/dFC 指标与基础 baseline | waveform+FC；persistence/VAR/no-SC/no-MoE/no-ODE |
| P0 | 特征级病理调制与 SC soft prior | residual/AdaLN/joint；fixed/adaptive/SC+adaptive |
| P1 | HAMD 因子+协变量+state-aware routing | scalar/vector/协变量；HAMD-only/brain-only/joint |
| P1 | ForecastHead 幅值重组与 future query | current/query；scale 重参数化；branch ablation |
| P1 | Refiner 初始化与自适应轮数、BrainMDM routing | no/1/current/corrected/adaptive；branch removal |
| P1 | 长程 `pred_window=1/2/3` 评估 | error-horizon、scheduled sampling（若做状态递推） |
| P2 | 多站点协调、site split 与 leave-one-site-out | raw/ComBat/site-condition；site variance |
| P2 | ODE solver/随机项、概率预测头 | steps/RK4/dopri5；calibration/NLL |
| P2 | MoE 异构专家与 token-level routing | homogeneous/heterogeneous；sample/token routing |
| P3 | 窗口间状态递推、HAMD 反演、反事实模拟 | 单窗/滚动；反演误差与反事实稳定性 |
| P3 | V2 外部验证、RevIN 统计条件、YAML/DDP/测试 | V1→V2；with/without RevIN stats；工程 smoke/unit tests |

**建议顺序。** 先完成 P0 评估闭环、HAMD 规范化、确定性 routing 与 FC/baseline，使后续每项结构收益可无偏归因；再独立实现并消融 SC soft prior 与 feature-level pathology modulation；随后推进 state-aware Soft-MoE、future query/幅值重组、refiner/multiscale；最后才投入多站点、ODE/SDE、状态递推、外部队列和反事实能力。每次只引入一个主要机制，保留参数量、训练成本与失败案例记录。

### 2.8 修改记录

| 日期 | 主要修改 |
|---|---|
| 2026-09-21 | 初版建立系统/结构问题与实施路线。 |
| 2026-09-22 | 明确 V1/V2 为独立队列；记录数据根目录和效率优化的已解决状态；补充站点分层划分待接入状态。 |
| 2026-09-22 | 合并既有“当前问题与修复”整合稿与本次补充内容；去除重复项，正文重构为“模型结构”“实验设计、数据协议、评估与工程”两大块。 |
