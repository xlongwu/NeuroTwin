# NeuroTwin 当前存在问题及解决方案

> 更新日期：2026-09-25  
> 依据：对当前代码库（`models/`、`train/`、`utils/`、`analysis/`、`scripts/`）的静态审计，以及既有审计、架构研究与训练记录；2026-09-25 起补充 P0 消融实验实证结论（`results/ablation/compare_all.md`）。  
> 状态：待实施；已由消融实证或代码改动落地的条目在各小节"消融实证/已完成"段中标注。"已解决项"仅保留为工程历史，不再列入待办。  
> 文档边界：本文只记录当前问题、修复方案、验证标准与实验协议。架构候选的完整文献池和研究组合见 `NeuroTwin_模型架构优化改进整合版.md`。

当前任务严格是 **SC 条件下的未来 ROI/BOLD 信号预测**：

$$\text{history ROI BOLD} + \text{SC} \rightarrow \text{future ROI BOLD}$$

MDD 微调在 HC backbone 预测上学习病理残差：

$$\hat{Y}_{\text{MDD}} = \hat{Y}_{\text{HC-base}} + \Delta Y_{\text{pathology}}$$

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

**消融实证（2026-09-25，P0，seed=2024，单 seed）。** 上述验收标准已被触发：全部 13 个消融实验（含 baseline 正式权重）在 shuffled-HAMD 负对照下指标变化仅 ±1e-4 量级（baseline ΔPCC=−1.29e-04），即模型输出对 HAMD 条件几乎零依赖；HAMD 分层效应同样微弱（corr_hamd_pcc=−0.07～−0.13，重度组 PCC 低约 0.018）。当前默认为 `pathology_norm_mode=robust_z`、无多项式扩展。本节问题确证成立且为当前最高优先级：拟解决方案 1--4（归一化对照、向量输入、FiLM/AdaLN 特征级调制、双层条件化）待实施；`cond_residual / cond_feature`（定位注入层级）已注册为 P1 消融变体。

### 1.2 SC 约束过硬、重复注入且缺乏个体/状态适应

**现状与问题。** `DFCAdapter` 的 `SC⊙learned_edge` 只能重标已有边，SC 为零时永远不能出现功能依赖；GraphODE attention 又把 `sc_with_self>0` 当二值 hard mask，并施以固定 soft bias。这样 DTI 假阴性边会永久切断，受试者共享先验、无方向性，且静态 SC 无法表达动态 FC。与此同时，SC 在 DFCAdapter、GraphODE、base refiner、MoE delta refiner 中重复注入，可能使结构先验压过功能证据并产生 over-smoothing。

ForecastHead 的 `cross_roi_*` 实际是 ROI channel 上 `Conv1d(kernel_size=1)` 的静态全连接 ROI×ROI mixing，不是 graph attention；它允许任意 ROI 交互，与上游“非 SC 边不可达”的假设并不一致。

**拟解决方案。**

1. 从 SC hard mask 改为 soft anatomical prior：

   $$A_t = \lambda_t A_{\text{SC}} + (1-\lambda_t) A_{\text{func}}(h_t)$$

   以 pooled ROI state 生成低秩 `A_func=softmax(sym(UV^T))`，先使用 rank 8--16；`λ` 依次尝试全局标量、sample-conditioned、ROI-specific。
2. 可学习概率掩码可作为备选：straight-through/Gumbel 在 SC 先验与数据证据间选择；同时加入低秩 subject-specific 邻接残差 `ΔA`、稀疏/熵/时间一致性正则。状态/时间调制和有向化仅在低风险版本验证后进入探索。
3. 对 Adapter、ODE、base refiner、delta refiner 的 SC 注入逐个消融，确定最小有效注入位置；预测端只保留一次轻量 refinement 作为候选。

**验收。** 比较 `no-SC / fixed-SC / adaptive-only / SC+static-adaptive / SC+dynamic-adaptive`，并报告固定/动态 `λ`。除 waveform 外必须报告 FC 指标、低 PCC 个体改善、图稀疏度/熵与计算成本。

**消融实证（2026-09-25，P0，单 seed）。** `sc_fixed`（scaled+hard）与 `sc_adaptive_only`（无二值先验、不注入 refiner）相对当前 soft_prior 基线 ΔPCC 分别为 −0.0018 / −0.0007（噪声级，单 seed 不可分辨）；FC 结构保真上两种简化略优（上三角 MAE 0.148 vs 0.153，edge-PCC 0.849 vs 0.842）。结论：当前 soft_prior 组合不劣于两种简化，但收益亦未被证明；soft prior 的 `λ` 自适应与注入位置消融（`sc_no_inject` / `sc_lambda_sample` / `sc_lambda_roi` / `sc_no_delta_a`，P1）仍待实施。

### 1.3 GraphODE 的连续时间主张尚未成立

**现状。** 未使用 `torchdiffeq/odeint`；`ode_steps=6` 为固定步数手动 RK2/Heun 更新，`GraphODE.forward(t,...)` 实际丢弃 `t`，即共享自治向量场。`n_block=2, ode_steps=6` 约需 24 次完整四支路 derivative evaluation；stochastic depth 在 derivative 算完后才决定跳过，不能显著节省该主成本。

**问题。** 固定步长对可能刚性的 dynamics 缺少误差控制和稳定性证据；纯确定性 ODE 也不能表达随机涨落或状态跳变。当前更准确的名称是 shared-parameter graph-spatiotemporal RK2 residual block，而不是已验证的连续生理时间模型。

**拟解决方案。**

1. 首先做强制敏感性审计：`ode_steps={1,3,6,12}`，比较 PCC/MAE、FC edge-PCC/MAE、params、FLOPs/MACs、GPU memory、samples/s、epoch time、gradient clipping frequency、`||k1||`、`||k2||`、`||Δh||`。
2. 若多步有稳定收益，再对照 RK4、可学习步长或自适应 solver（如 dopri5），并明确其训练稳定性和开销；不要仅因“ODE”命名而引入复杂求解器。
3. Langevin noise/SDE 分支只作为 P2 探索，并须给出不确定性校准和状态跳变收益；若单步与六步的 subject-level CI 重叠，则简化成 residual dynamics block。

**消融实证（2026-09-25，P0，单 seed）。** 步数敏感性审计已完成：`ode_steps={1,3,6,12}` 的 test PCC 依次为 0.8334 / 0.8347 / 0.8352 / 0.8350，差异 ≤0.002（单 seed 噪声级），MAE 亦然；即步数在 1--12 内不敏感，方案 1 的验收条件部分触发（"单步与六步 CI 重叠"的完整判定需多 seed CI）。**已落地调整：默认 `ode_steps` 6→3**（`scripts/Finetune_MDD.sh`、`experiments/base_config.py` 同步，约省 1/3 ODE 算力）；方案 2/3（RK4/自适应 solver/SDE）仍待实施，"已验证连续生理 ODE"的声明继续不成立。

### 1.4 MoE：路由条件不足、专家同质且粒度过粗

**现状。** 默认 4 个同容量 experts、top-2、sample-level routing；router 输入为 HAMD + 脑状态统计（`moe_router_cond_only=False`，文档初版所记 `cond_only=True` 已不是当前默认）。共享 pathology expert 始终激活，也可能吸收大部分 residual，使 routed experts 难分化。

现有验证 routing 还可能在 `eval()` 下用低温 multinomial 抽样；`pred13` top-1 expert 使用数为 `217/688/438/10`，第四专家近乎空转，已有 expert-HAMD Mann--Whitney 比较不显著。因此当前不能宣称自动发现临床亚型。

**拟解决方案。**

1. 立即放开 `moe_router_cond_only`，以 `normalized HAMD + 5F state summary + pooled latent/base difficulty` 作为 router 条件；validation、early stopping、test 强制 deterministic dense soft mixture 或 deterministic top-k。
2. 先验证四专家 dense differentiable Soft-MoE；随机 routing 只能作为 10--20 次 Monte-Carlo uncertainty 实验，报告 mean±std、温度和种子。
3. 比较 `no-MoE / shared-only / routed-only / hard top-k / deterministic dense soft`；增加 expert output diversity、pairwise cosine、expert ablation `ΔPCC`、跨窗 routing consistency，而不仅优化负载均衡。
4. 在稳定基线后探索异构 experts（不同感受野、频带、深度或功能路径）及 token-level ROI×window routing；专家数和 top-k 必须系统消融，不能先扩大规模。

**消融实证（2026-09-25，P0，单 seed）。** 对照矩阵已跑（deterministic dense_soft 评估，方案 1/3 部分完成）：`no-MoE` 相对基线 ΔPCC=−0.0308、ΔMAE=+0.106（四个 HAMD 分层一致变差），**MoE 结构收益确证**；`shared-only` ΔPCC=−0.0081，`routed-only` 与 `routed_shared` 差 −0.0016（噪声级），即收益主体是路由专家、shared expert 增量≈0。**已落地调整：默认 `moe_experts_mode` routed_shared→routed_only**（移除 shared expert）。路由坍缩在正式权重上依然存在：baseline top1 分配 674/675 集中于单一专家，4 专家中仅约 2 个有效（normalized entropy≈0.50）——比 pred13 时代更严重，方案 1/2 的 state-aware routing（`moe_gate_difficulty`，P1）与专家数缩减（4→2，实际有效容量）为下一步；"自动发现临床亚型"的声明继续不成立。

**条件注入失效根因修复（2026-09-26）。** 静态审计确认 HAMD 条件失效根因：`RouterCond._select_without_shared` 训练期实现为 Gumbel-argmax + 自适应高温（`adaptive_temp = temperature × (1 + 负载均值)`），与 docstring 声称的 multinomial 设计行为背离——高温 Gumbel 把专家选择近似为与条件无关的均匀随机，专家的 FiLM 条件调制与路由器的条件→专家映射同时失去梯度信号（专家看到的 (样本, HAMD) 配对统计独立，FiLM 最优解为忽略条件；路由 logits 收敛为条件无关常数，评估期 dense_soft 后坍缩单专家）。**修复：训练分支恢复为按容量惩罚后概率的 multinomial 采样**（`models/neurotwin_moe.py`，温度调度与容量软惩罚不变），恢复「概率→选择」保真通路。合成数据验证（真实正则配置 LB=0.05/熵=1e-3/Z=1e-3/多样性=0.001，400 步）：训练收敛、训练期专家负载近均衡（0.24/0.24/0.24/0.27）、条件敏感性（同输入 shuffled-HAMD 相对响应）从 ≈0 恢复至 22.3%、评估期 top1 集中度从 1.000 降至 0.688。限制：本机无 Version4 正式权重，真实数据收益需按新配置重训后用 `evaluate_variant --no_shuffled` 对照复验；路由集中度 0.688 仍未达均衡，方案 1/2 的 state-aware routing 与专家数缩减实验继续有效。

**路由坍缩对策实验注册（2026-09-26）。** 基于 multinomial 修复后的新基线，按 P0 报告建议在 `experiments/variants.py` 注册 3 个对策变体（G14_MOE 组，均为 P0）：`moe_exp2_t1`（专家数 4→2、top_k 2→1，对齐实际有效容量 + 硬单选）、`moe_argmax`（训练期容量感知 argmax 硬路由，兼作 multinomial 修复的消融对照）、`moe_temp_low`（路由温度 1.5/1.0→1.0/0.5，锐化概率分布）。全部变体通过 overrides 校验与 dry-run 验证，`num_experts=2/top_k=1` 路径通过前向/反传冒烟。执行：`bash RunAblation.sh --group G14_MOE --only moe_exp2_t1,moe_argmax,moe_temp_low`（需先按新代码重训 `neurotwin_pretrain_pred1`）。评估关注：top1 分配分布、normalized entropy、expert-HAMD 分层、条件敏感性（与 1.1 shuffled 负对照联动）。

### 1.5 ForecastHead 语义、幅值重组与多分支冗余

**现状。** 对 `W=6,S=30`，history/latent 很早压平为 180 维，六路 history/latent MLP、temporal、cross-ROI 与 window-attention 分支融合。输出采用：

$$\hat{y} = \text{anchor} + \text{trend} + \text{softplus}(\text{scale}) \cdot \text{standardized}(\text{shape})$$

**问题。**

- future ROI、future window、future time 没有显式 query 语义；当 `pred_window` 增加时，flatten readout 是信息压缩瓶颈；
- trend 与 `scale×shape` 可能重复承载幅值，独立回归会相互拉扯；scale 是每窗单一标量，不能表达窗内幅度演化；
- 多个分支均由相同 history/latent 派生，GraphODE 和 ForecastHead 的 window attention 也可能重复。

**拟解决方案。**

1. 首版保留 anchor+trend，只把 shape branch 替换为 Future ROI-Time Query Decoder：`ROI embedding + future-window embedding + future-time embedding → cross-attention(memory) → residual`。
2. 重参数化幅值结构，例如使 trend 同受 scale 调制，或使用一致乘法表达；增加 trend/scale consistency regularization，并对逐时间点 scale 与窗级 scale 做对照。
3. 做 current flatten / query decoder、六路 branch leave-one-out、GraphODE/head attention 去重的消融；`pred_window=1/2/3` 上比较扩展性和长程误差。

**消融实证（2026-09-25，P0，单 seed）。** 方案 1 的对照已完成：关闭 FutureQueryDecoder 退回 flatten（`head_flatten`）PCC +0.0008（噪声级）但 MAE +0.0176 明显变差，即 query decoder 的贡献在幅值精度而非整体相关——**保留 query decoder**。注意该对照经自预训练链路（`ablation_pre_head_flatten`）而非复用 pred1 权重，与其它变体不完全对等，结论强度有限。方案 2 的幅值重参数化与 branch leave-one-out（`head_no_*`，P1）待实施。

**预训练混淆对照实验注册（2026-09-26）。** `head_flatten` 变体中 shape 分支结构变化导致预训练权重无法加载（随机初始化），其 MAE 损失可能混入「shape 分支未预训练」的代价而非 flatten 结构本身的缺陷。为此新增受控跳过机制（`--pretrained_skip_pattern`，fnmatch 模式，[train/optim.py](../../train/optim.py) `load_backbone_weights` 加载时命中键保留随机初始化）并注册 P0 对照变体 `head_shape_scratch`（baseline query 结构 + finetune 时强制 `future_query.*` 随机初始化）。判读：head_shape_scratch vs baseline 差 = shape 分支预训练的独立贡献；若 head_shape_scratch ≈ head_flatten，则 head_flatten 的 MAE 损失主要是未预训练混淆而非结构缺陷，1.5 节「保留 query decoder」结论需重新评估。注意其自预训练产物与 baseline 完全同构同配置（skip_pattern 仅作用于 finetune 加载段），可直接复用 baseline 的预训练权重省一次训练。

### 1.6 `IterativePredictionRefiner`：初始化错误与固定轮数

`round_scales` 使用 `0.5**i`，首轮目标实为 1，经 sigmoid 后约 0.9999，而非注释中的 0.5。修复为显式 `[0.50,0.33,0.20]`；比较 no-refiner、one-round、current three-round、corrected three-round。

此外 `n_rounds=3` 对所有样本固定、轮间没有误差反馈，无法按预测难度自适应。低风险改造是在每轮根据当前 residual magnitude 用轻量 gate 决定是否继续，加入轮间 residual supervision，并比较 1--4 轮、固定轮数与早停轮数。该机制必须和 SC 注入位置一起消融，避免“更多 refinement”仅加重过平滑。

**已完成（代码）。** `round_scales` 初始化已修复为显式初值表（`models/common.py` `ROUND_INIT_SCALES`：3 轮为 `[0.50, 0.33, 0.20]`，经 logit 变换初始化）；`refiner_rounds=0` / `delta_refiner_rounds=0` 路径已打通（支撑消融）。`base_refiner_rounds=1` / `delta_refiner_rounds=1` 变体已注册为 P1 消融。

**消融实证（2026-09-25，P0，单 seed）。** 关闭 refiner（`refiner_none`）ΔPCC=−0.0045、ΔMAE=+0.024——**base refiner 收益确证，保留**；`refiner_rounds` 3→1 有轻微损失（−0.0023，保留 3 轮）；delta refiner 1 轮与 2 轮等价（ΔPCC=−0.0001）。**已落地调整：默认 `delta_refiner_rounds` 2→1**（省算力）。自适应轮数 gate（方案后半）仍待实施。

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

**PICP 过覆盖的 conformal 方差校准（2026-09-26）。** P0 消融实测名义 90% 区间实际覆盖 0.937–0.962。诊断确认区间构造与 NLL 实现均正确（`_gaussian_z` 分位数无误、`GaussianNLLLoss` 标准形式），过覆盖源于异方差 NLL 训练的方差系统性上偏（模型以大 σ 缓冲错误预测，回归 NLL 的固有现象），属评估侧校准问题而非训练 bug。**修复：评估侧 conformal 方差校准**——`analysis/metrics.py` 新增 `fit_conformal_scale`（val 上拟合归一化残差的经验高分位 q，含 finite-sample 修正 ⌈(N+1)(1-α)⌉/N），`evaluate_variant.py` 默认启用（`--no_calibrate` 关闭）：val 不在 `--splits` 时自动补评用于拟合，各 split 应用 `μ ± q·σ` 得 `PICP_cal/MPIW_cal`（写入 `metrics_<split>.json` 与 registry 新列 `test_picp_cal`）。合成验证：σ 上偏 1.6 倍时 PICP 0.990→0.904（对齐名义）、MPIW 收缩 35%，欠覆盖方向对照正确。校准不触碰模型权重与训练流程，点预测指标不变；原始 PICP 保留以便对照。

> 1.7 与 1.8 未进入 2026-09-25 的 P0 消融矩阵（BrainMDM 分支、MDM 自适应、概率预测头为 P1/P2 项），暂无新实证。

## 2. 实验设计、数据协议、评估与工程问题

### 2.1 数据切分、训练/测试闭环与统计单位

当前 9 个窗在 `6→1` 设置下每被试通常产生 3 个样本；subject-level split 防止同一被试滑窗跨 train/val/test，是正确且不可退让的设计。MDD 已有 HAMD 分位分箱。SC 预处理为 NaN/Inf→0→对称化→负值截断→`log1p`→99th-percentile scaling→`[0,1]`；train 有 noise/scale/channel-drop/time-mask，val/test 不增强。

但评估闭环尚不完整：`main.py` 虽构造 train/val/test loader，主训练流程只使用 train/val，未自动加载最佳 checkpoint 执行最终 test；既有 pred1/pred12/pred13 是 1,353 个 validation samples 而非独立 test，且分析曾用 `val_ratio=0.2`、主 CLI 默认 `0.1`，不能作严格横向结论。

**部分完成（2026-09-25）。** 消融评估框架（`experiments/`）已实现：被试级 (dataset, site) 8:1:1 切分、训练后自动加载 `finetuned_best.pt` 评估（默认仅 test，`--splits` 可显式加 val）并写入 `results/ablation/registry.csv`（本节方案 1 的框架级落地）。评估指标已扩展：FC 上三角 MAE/RMSE、edge-PCC、命名网络内/间边 MAE、被试级聚合 PCC/MAE 与低 PCC 被试诊断、参数量，并落盘被试级聚合 CSV（统计单位为被试）。多 seed 复验暂缓，当前固定 seed=2024（按 2026-09-26 决定暂不开展）。

**已完成（2026-09-26）：评估方法三项落地。**

1. **版本化固定 subject-ID split（方案 1 前半）**：`utils/dataloader.py` 的 `_split_subject_ids` 升级为 manifest 包装层，切分名单连同配置指纹（mode/seed/比例/分箱/被试数）落盘 `<data_root>/subject_split_<mode>.json`；同配置自动复用，配置不一致或被试集合变化直接报错，`--refresh_split_manifest` 显式重新生成（`main.py` / `evaluate_variant.py` / `fc_baselines.py` 三入口透传）。单测覆盖生成→复用→配置冲突→被试集合变化→refresh 五场景（5/5 通过）。训练、评估、基线自此共用同一锁定 test 集。
2. **`main.py` 训练结束自动评 test（方案 1 后半）**：finetune 收尾自动加载 `finetuned_best.pt` 调用 `evaluate_variant.evaluate`（含 conformal 校准自动补评 val 与 shuffled-HAMD 负对照），评估失败仅告警、不影响训练产物；`--no_eval_after_train` 可关闭。pretrain 模式不接入（预训练权重缺 MoE 分支，strict 加载必然报错）。
3. **被试级配对统计检验（方案 3 前半）**：`experiments/compare.py` 新增 `--paired_tests EXP_ID=EVAL_DIR ...`，对齐各评估目录 `subject_metrics_test.csv` 的逐被试 PCC 做配对 t 检验与 Wilcoxon 符号秩（共同被试 ≥8 才检验），报告落盘 `results/ablation/paired_tests.md`；多重比较校正仍待纳入，正式结论前需补。

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

**部分完成（2026-09-26）：无模型平凡基线落地。** 新建 `analysis/fc_baselines.py`，提供 persistence（末帧持续）、linear（每 ROI 窗口均值最小二乘线性外推）、mean（历史均值持续）三种基线；经 `--ckpt` 配置快照 + 版本化切分 manifest 构建与 `evaluate_variant` 完全一致的 test loader，输出 FC 上三角 MAE/edge-PCC 与点级 PCC/MAE 至 `<out_dir>/fc_baselines.json`，直接对照 `metrics_test.json` 同名 FC 指标，回答「模型 FC 重构是否优于平凡策略」（方案 2 前半）。VAR/岭回归与结构消融基线仍走 `experiments/variants.py` 变体框架（no-SC/no-MoE/no-ODE 已在册）。

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
| 2026-09-22 | 将正文 4 处 `\[...\]` 公式转换为 Markdown 可渲染的 `$$...$$` 行间公式；移除公式内中文句号，多字母标识符统一用 `\text{}` 排版。 |
| 2026-09-25 | 补充 P0 消融实验（13 变体、seed=2024）实证结论至 1.1--1.6；1.1 条件失效被 shuffled 负对照确证、1.4 MoE 收益确证但路由坍缩加剧；已落地参数调整（`ode_steps` 6→3、`delta_refiner_rounds` 2→1、`moe_experts_mode` routed_shared→routed_only）；标注已完成代码项（refiner 初始化修复、`refiner_rounds=0` 支持、消融评估闭环）；修正 1.4 中 `moe_router_cond_only` 过时描述。 |
| 2026-09-25 | `main.py` CLI 默认参数与 `experiments/base_config.py` 对齐：`pred_window` 3→1、`ode_steps` 6→3、`delta_refiner_rounds` 2→1、`moe_experts_mode` routed_shared→routed_only，以及 `train_epochs/batch_size/lr_init/lr_final/warmup_epochs/patience/moe_load_balance_weight/moe_gate_temp_start/compile` 等训练规模项；`loss_lr_scale` 保持 1.0（随默认 pretrain 模式，finetune 由脚本显式传 0.5），属模式相关差异而非失同步。 |
| 2026-09-25 | `scripts/Pretrain_HC.sh` 同步取消融值：`ode_steps` 6→3、`delta_refiner_rounds` 2→1，与 `Finetune_MDD.sh`/`base_config.py` 对齐，两阶段 ODE 配置保持一致。 |
| 2026-09-25 | 评估分析脚本收敛为单一入口：`experiments/evaluate_variant.py` 吸收原 `analysis/run_comprehensive.py` 全部独有能力（expert-HAMD 分层分析、可视化、置换特征重要性 + FDR 显著脑区导出、预测数组保存），作为 `--visualize / --feature_importance / --save_arrays` 开关；`run_comprehensive.py` 已删除（其 `build_model_and_loader` 部分参数重建对结构性变体会权重错配，评估功能与 evaluate_variant 完全重叠）。`analyze_low_pcc_samples.py` 改读 `sample_metrics_<split>.csv`；README 与 Version1 文档引用同步更新。 |
| 2026-09-26 | **HAMD 条件注入失效根因修复**：`models/neurotwin_moe.py` 训练期路由选择由 Gumbel-argmax + 自适应高温恢复为 multinomial 概率采样（与设计一致），修复专家 FiLM 条件调制与路由器条件→专家映射同时失去梯度信号的问题；合成验证（真实正则配置）条件敏感性 ≈0→22.3%、训练期专家负载恢复近均衡、评估期 top1 集中度 1.000→0.688；真实数据收益待重训复验。详见 1.4 节修复记录。 |
| 2026-09-26 | **注册 MoE 路由坍缩对策实验**（1.4 节，G14_MOE 组新增 3 个 P0 变体）：`moe_exp2_t1`（专家数 4→2/top_k 2→1）、`moe_argmax`（训练期硬确定性路由，兼作 multinomial 修复消融）、`moe_temp_low`（路由温度 1.5/1.0→1.0/0.5）；变体总数 35→38，全部通过 overrides 校验、dry-run 与 2 专家前向冒烟；执行命令见 1.4 节。 |
| 2026-09-26 | **PICP 过覆盖 conformal 方差校准**（1.8 节）：`analysis/metrics.py` 新增 `fit_conformal_scale`（val 拟合归一化残差经验高分位，finite-sample 修正），`evaluate_variant.py` 默认启用校准（`--no_calibrate` 关闭，val 自动补评拟合），产出 `PICP_cal/MPIW_cal`；registry/compare/run_experiments 同步 `test_picp_cal` 字段。合成验证 PICP 0.990→0.904、区间收窄 35%；训练零改动，对已有 checkpoint 立即可用。 |
| 2026-09-26 | **注册 head_flatten 预训练混淆对照实验**（1.5 节）：`load_backbone_weights` 新增 `skip_pattern` 受控跳过（fnmatch 命中键保留随机初始化，单测验证），main.py 新增 `--pretrained_skip_pattern` CLI，BASE_ARGS 同步；注册 P0 变体 `head_shape_scratch`（baseline 结构 + `future_query.*` 强制随机），变体总数 38→39，dry-run 参数注入验证通过；判读逻辑见 1.5 节。 |
| 2026-09-26 | **完善评估/实验方法类任务（2.1/2.2 节，多 seed 按用户决定暂不开展）**：① 版本化固定 split——`utils/dataloader.py` 切分 manifest 落盘/复用/配置冲突与被试集合变化报错/refresh（`--refresh_split_manifest` 于 main.py、evaluate_variant.py、fc_baselines.py 三入口透传；单测 5/5 通过）；② `main.py` finetune 训练结束自动评 test（复用 manifest，含 conformal 校准与 shuffled 负对照；`--no_eval_after_train` 可关；评估失败不影响训练产物）；③ `experiments/compare.py` 新增 `--paired_tests` 被试级配对 t/Wilcoxon 检验（报告落盘 `results/ablation/paired_tests.md`，多重比较校正待补）；④ 新建 `analysis/fc_baselines.py` 无模型基线套件（persistence/linear/mean，与 evaluate_variant 同 test 集对照，输出 `fc_baselines.json`）。另修复 `main.py --help` 因 `--compile` help 字符串含未转义 `%` 导致的崩溃。 |
