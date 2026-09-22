# NeuroTwin 顶会架构借鉴与优化设计

> 调研日期：2026-09-20  
> 目标：筛选能改变 NeuroTwin 信息流、且可在 AAL116、历史 6 窗、每窗 30 点、MDD 小样本条件下验证的架构。论文不要求同任务；只要求其机制能解决当前的图结构、融合、条件化、预测读出、迁移或路由问题。

## 1. 当前架构需要被解决的问题

```text
A. 固定 SC 硬先验无法表达功能状态依赖的新连接。
B. SC、ROI 信号、HAMD 的交互单向且多在末端发生。
C. 预测头过早展平历史，未来 ROI 与未来时间点没有明确语义。
D. HAMD-only、随机 top-k MoE 没有稳定的专家分工。
E. 波形预测良好并不能保证 FC/dFC 网络预测正确。
```

所有候选设计均需说明：新增模块输入、输出、放置位置、解决的问题和验证方式。

## 2. 同方向与时空图基础论文

| 论文 | 顶会/年份 | 架构创新点 | 借鉴理由与落点 |
|---|---|---|---|
| [STAGIN](https://proceedings.neurips.cc/paper_files/paper/2021/file/22785dd2577be2ce28ef79febe80db10-Paper.pdf), Kim, Ye, Kim；[代码](https://github.com/egyptdj/stagin) | NeurIPS 2021 | 动态脑图的空间 readout 与 Transformer 时间聚合。 | 明确将 FC 随窗口变化建模；可用于派生 FC 评估和 state token 设计。 |
| [DynDepNet](https://arxiv.org/abs/2209.13513), Farnell et al. | ICML Workshop 2023 | 从 fMRI 端到端学习时间变化依赖图。 | 直接否定“邻接固定且已知”的假设；用于 SC soft prior + `A_func(t)`。 |
| [Graph WaveNet](https://www.ijcai.org/Proceedings/2019/0264.pdf), Wu et al.；[代码](https://github.com/nnzhan/Graph-WaveNet) | IJCAI 2019 | 节点嵌入生成自适应邻接。 | 低秩功能邻接补足 SC 中没有的边；替换 GraphODE hard mask。 |
| [TimeXer](https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf), Zhang et al.；[代码](https://github.com/thuml/TimeXer) | NeurIPS 2024 | 内生 patch self-attention 与外生变量 cross-attention 并行。 | SC/HAMD 应当变为可被 ROI token 查询的外生信息，而非 mask/末端残差。 |
| [Crossformer](https://openreview.net/forum?id=vSVLM2j9eie), Zhang, Yan；[代码](https://github.com/Thinklab-SJTU/Crossformer) | ICLR 2023 | 时间 attention + 跨变量 router 聚合/广播。 | 用 4--8 routers 替换 current head 的 1×1 cross-ROI Conv，避免 ROI² attention。 |
| [CATS](https://proceedings.mlr.press/v235/lu24d.html), Lu et al. | ICML 2024 | 从原时序构造连续、稀疏、可变的辅助时间序列。 | 用 4--8 个动态辅助通道摘要 ROI 间关系，作为小样本下全图注意力的轻量替代。 |
| [Ada-MSHyper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a6935d11910d6f9142b0a1e36fc6753-Abstract-Conference.html), Shang et al.；[代码](https://github.com/shangzongjiang/Ada-MSHyper) | NeurIPS 2024 | 自适应超图与多尺度群组交互。 | 脑网络包含模块级共同作用；可用 6--10 个 hyperedge token 代替 pairwise `A_func`。 |
| [GraphSSM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e5ba3d6d93213db6b1d1931c6517fe1a-Abstract-Conference.html), Li et al. | NeurIPS 2024 | 图拉普拉斯约束状态空间模型。 | 是当前自治 RK2 ODE 的更严格图动力学 baseline；仅在 ODE step ablation 后测试。 |

## 3. 优先架构方案

### 方案 A：SC 软先验 + 动态功能邻接（P0）

```text
Before: ROI features → SC hard mask / diffusion → graph message passing
After:  history ROI tokens → A_func(t)
        SC → A_SC
        A(t)=λ(t)·A_SC+[1-λ(t)]·A_func(t) → graph message passing
```

- **输入**：window-pooled ROI latent `[B,F,d]` 与标准化 SC。
- **输出**：动态软邻接 `[B,F,F]`，输入 GraphODE 的图注意力和一次预测细化。
- **借鉴**：Graph WaveNet、DynDepNet。
- **迁移原因**：当前 `SC * learned_edge` 无法创造功能性新边；本方案保留 SC 生物学先验，同时允许状态依赖功能路径。
- **风险**：动态图高方差；限制 rank 为 8--16，对 `A_func` 加熵/稀疏正则。
- **ablation**：no-SC、fixed-SC、adaptive-only、SC+adaptive；固定/动态 λ。

### 方案 B：SC profile token 与 ROI token 双向交叉注意力（P1）

```text
ROI history tokens ─Query→ SC profile tokens (K/V)
ROI context        ←Query─ updated SC tokens
                         ↓
                    fused ROI latent
```

- **输入**：ROI history token 与每行 SC profile 编码得到的 token。
- **输出**：结构条件化 ROI latent，进入首个 GraphODE block。
- **借鉴**：TimeXer。
- **迁移原因**：当前 SC 只能决定“能否相互注意”，不能让信号决定“从结构 profile 的哪部分读取何种信息”。
- **风险**：不直接使用 116² edge token；从单层单向交叉注意力开始，再比较双向。

### 方案 C：Future ROI-Time Query Decoder（P1）

```text
encoder latent [ROI, history-window, time]
future query q(ROI_i, future-window_j, time_k)
      → cross-attention → prediction(i,j,k)
```

- **输入**：多层 encoder latent 与 `ROI embedding + future window embedding + future time embedding`。
- **输出**：每个未来 ROI/时刻的预测残差；与 anchor/trend 相加。
- **借鉴**：Perceiver IO。
- **迁移原因**：修复当前 ForecastHead 对 6×30 历史的压平，赋予输出位置显式语义；可自然扩展预测窗口数。
- **风险**：短序列下只做一层小维度 cross-attention；先只替换 shape head。

### 方案 D：确定性状态感知 Soft-MoE（P0/P2）

```text
latent → dynamic state tokens
[normalized HAMD, state, pooled latent, base difficulty]
       → deterministic soft router → residual experts
```

- **输入**：标准化 HAMD、history/latent/base summary、可选 state token。
- **输出**：专家软权重与 residual prediction。
- **借鉴**：Soft-MoE、ST-MoE、Slot Attention。
- **迁移原因**：当前 router 在 `cond_only=True` 时只看 HAMD，并且验证时随机采样；新路径将病理程度和脑动态状态共同用于路由。
- **风险**：先用四个专家的 dense soft mixture；在其稳定获益后才试稀疏/层级专家。

### 方案 E：HAMD AdaLN/FiLM 动力学调制（P1）

```text
normalized HAMD → condition encoder → γ_l, β_l, gate biases
history → AdaLN/FiLM(GraphODE_l) → pathology-conditioned dynamics
```

- **输入**：规范化 HAMD 编码。
- **输出**：每 GraphODE block 的小幅仿射调制与分支门控偏置。
- **借鉴**：FiLM。
- **迁移原因**：病理不应只在最后修残差，还应能改变特征如何演化。
- **风险**：小 MDD 数据可能破坏 HC backbone；先冻结 backbone 训练 adapter，再小学习率联合微调。

## 4. 跨领域顶会架构机制库

以下论文不以任务相似性为选择条件，而以“它解决的架构问题是否同构”为标准。

### 4.1 查询瓶颈、压缩与输出解码

| 论文 | 顶会/领域 | 架构创新点 | 为什么值得借鉴 | NeuroTwin 具体落点 |
|---|---|---|---|---|
| [Perceiver IO](https://arxiv.org/abs/2107.14795), Jaegle et al. | ICLR 2022；多模态 | latent array 经 cross-attention 读取任意输入，再以 query 解码任意结构输出。 | 当前历史被早期压平；其“压缩后按任务 query 读取”直接解决预测头信息瓶颈。 | Future ROI-time query decoder；也可将 SC/HAMD token 放入 encoder 输入。 |
| [BLIP-2](https://proceedings.mlr.press/v202/li23q.html), Li et al.；[代码](https://github.com/salesforce/LAVIS) | ICML 2023；视觉语言 | Q-Former 用少量 learnable queries 桥接冻结单模态编码器与下游模型。 | HC 预训练 backbone + 小样本 MDD 微调是同构场景；优于直接解冻大主干。 | HAMD-conditioned Q-Former 从 shallow/mid/deep latent 读出 MDD adapter tokens。 |
| [Flamingo](https://papers.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html), Alayrac et al. | NeurIPS 2022；视觉语言 | Perceiver Resampler 压缩可变输入；冻结主干层间插入 gated cross-attention。 | SC/HAMD 现在单向或末端融合；可用门控跨注意力低风险逐层注入而不破坏预训练能力。 | 每隔一层 GraphODE 加 `SC/HAMD → latent` gated adapter，门控零初始化。 |
| [Slot Attention](https://papers.nips.cc/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf), Locatello et al. | NeurIPS 2020；对象中心学习 | 竞争式迭代 attention 将集合输入压缩为可组合 task-dependent slots。 | ROI-window token 可看作脑状态/模块集合；slot 给 MoE 提供比单 HAMD 更有信息的路由变量。 | `ROI×window tokens → 4~6 state slots → state posterior → router`。 |

### 4.2 多尺度、跨层信息流与频域算子

| 论文 | 顶会/领域 | 架构创新点 | 为什么值得借鉴 | NeuroTwin 具体落点 |
|---|---|---|---|---|
| [Pathformer](https://openreview.net/forum?id=lJkOCMP2aW), Chen et al. | ICLR 2024；时序 | 多尺度 patch 后，由 adaptive pathways 按样本选择尺度路径。 | BrainMDM 的尺度分支目前全量开启；不同被试/脑状态未必需要相同尺度。 | 将 BrainMDM 多尺度池化/卷积分支改为 gated mixture，gate 来自 pooled latent。 |
| [DeepStack](https://proceedings.neurips.cc/paper_files/paper/2024/hash/29cd7f8331d13ede6dc6d6ef3dfacb70-Abstract-Conference.html), Meng et al. | NeurIPS 2024；多模态 | 不把所有外部 token 在首层注入，而将不同粒度 token 分配到不同深度。 | 当前 SC 在多个位置重复、同质注入；应改为分层且粒度不同的结构信息。 | 浅层局部 SC；中层模块/低秩 SC token；预测端仅一次轻量 refinement。 |
| [Dense Connector](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a10c46572628d58cb44fb705f25cbbf-Abstract-Conference.html), Yao et al.；[代码](https://github.com/HJYao00/DenseConnector) | NeurIPS 2024；多模态 | connector 聚合预训练编码器多层特征，而不只取最终层。 | 当前 MoE 只读 final latent/history/base；浅层 SC-adapted 与中层动力学信息未被个体化模块利用。 | 将 shallow/mid/deep latent 低维投影后，经门控供 Q-Former、MoE 或 decoder 读取。 |
| [Fourier Neural Operator](https://openreview.net/forum?id=c8P9NQVtmnO), Li et al. | ICLR 2021；科学机器学习 | 在频域学习函数到函数的全局 operator，有限 Fourier mode 可连接远程位置。 | 当前每窗只用小核局部卷积；若有低频 BOLD 节律，频域分支比加深局部 CNN 更有明确动机。 | 在 `S=30` 内时间轴并联低频 Fourier mixing，只保留 4--8 modes，再与 BrainMDM 融合。 |

### 4.3 动态路径、异构专家与参数高效迁移

| 论文 | 顶会/领域 | 架构创新点 | 为什么值得借鉴 | NeuroTwin 具体落点 |
|---|---|---|---|---|
| [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951), Puigcerver et al. | ICLR 2024；MoE | 连续 soft dispatch/combine 替代离散 token-to-expert 分配。 | 当前随机 top-k 正是它要规避的路由不稳定问题。 | 先把现有四个 experts 改为 dense soft mixture；再考虑 slot 式 Soft-MoE。 |
| [AVMoE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/009729d26288b9a8826023692a876107-Abstract-Conference.html), Cheng et al.；[代码](https://github.com/yingchengy/AVMOE) | NeurIPS 2024；多模态 | 单模态 adapter 与跨模态 adapter 是不同专家，由轻量 router 组合。 | 当前四个病理专家同构、可能功能重叠；专家应该按功能路径分工。 | 设为 shared/base、SC interaction、HAMD modulation、state adapter 四类小专家，不都预测完整 residual。 |
| [Wings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3852f6d247ba7deb46e4e4be9e702601-Abstract-Conference.html), Zhang et al. | NeurIPS 2024；持续多模态学习 | 主注意力旁并联低秩残差注意力，补偿新模态微调导致的原能力遗忘。 | HC→MDD 微调具有同样灾难性遗忘风险；目前冻结后解冻但没有显式保真路径。 | 在 GraphODE/ForecastHead 中添加 LoRA-style parallel adapter，仅更新 adapter 与小门控。 |
| [MoVA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bb0fea29f7aa6ede17e906ac6a225f34-Abstract-Conference.html), Zong et al. | NeurIPS 2024；多模态 | coarse-to-fine：先依上下文选专家族，再做细粒度 adapter 融合。 | 当前 top-2 HAMD routing 将病理大类选择与细粒度残差生成混为一谈。 | 一级以动态 state/病理程度选专家族，二级按 ROI/module token 软组合。 |
| [RouterDC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a641b8ec86162fc875fb9f6456a542f-Abstract-Conference.html), Chen et al.；[代码](https://github.com/shuhao02/RouterDC) | NeurIPS 2024；模型路由 | 以 sample-expert 与 sample-sample 双对比学习使路由器学习能力匹配。 | 负载均衡与熵只能让当前 router 均匀，不会让专家分配具有任务语义。 | 在确定性路由后增加基于专家误差/输出差异的 router 对比辅助项。 |

### 4.4 图关系、群组交互和消息路径

| 论文 | 顶会/领域 | 架构创新点 | 为什么值得借鉴 | NeuroTwin 具体落点 |
|---|---|---|---|---|
| [Ada-MSHyper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a6935d11910d6f9142b0a1e36fc6753-Abstract-Conference.html), Shang et al. | NeurIPS 2024；多变量时序 | 以 hyperedge 建模群组，并实现多尺度 group-wise interaction。 | 脑网络模块本身是群组关系；pairwise SC/FC 不足以表达共同调制。 | 使用 6--10 个可学习 hyperedge/module token；和 dynamic pairwise graph 二选一。 |
| [Towards Dynamic Message Passing on Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93b7e2780c4f6599837fdd3718c51fad-Abstract-Conference.html), Sun et al. | NeurIPS 2024；图学习 | pseudo nodes 动态形成低复杂度消息中介路径。 | 可把 Crossformer router 升级为可解释的动态脑网络中介节点。 | 在 GraphODE 图注意力旁加入 8 个 pseudo nodes：ROI→pseudo→ROI。 |
| [GraphSSM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e5ba3d6d93213db6b1d1931c6517fe1a-Abstract-Conference.html), Li et al. | NeurIPS 2024；时变图 | 图拉普拉斯显式约束 state-space evolution。 | 当前 ODE 未使用真实时间变量；GraphSSM 是更严谨的图动力学对照架构。 | 作为 GraphODEDDI 替代 baseline；不能和 ODE 叠加后宣称单模块贡献。 |

## 5. 机制选择规则

| 当前目标 | 首选借鉴 | 不应无控制地同时堆叠 |
|---|---|---|
| 保留 HC 基础能力 | BLIP-2 Q-Former、Flamingo gated adapter、Wings adapter | 全量解冻 + 大 MoE |
| 发现低维脑状态 | Slot Attention、CATS、pseudo nodes | slot + router + hypergraph 同时加入 |
| 加强 ROI 交互 | Crossformer router、pseudo node、hypergraph | full ROI attention + dynamic graph + hypergraph |
| 结构化未来预测 | Perceiver IO queries | 复杂 flatten MLP 继续堆分支 |
| 让病理参与动力学 | FiLM、gated cross-attention、Q-Former | 只在末端添加 HAMD residual |
| 稳定专家系统 | Soft-MoE、AVMoE、MoVA | 随机 HAMD-only top-k |
| 多尺度/频谱 | Pathformer gate、FNO | 无 ablation 地加深所有卷积分支 |

## 6. 推荐组合与优先级

### A. 稳健改进：Adaptive SC-FC Coupling

```text
SC prior + low-rank A_func → graph backbone → current forecast head
                                              └→ waveform + FC auxiliary loss
```

先建立 dynamic graph 的价值和 FC 评价，改动最小、可归因性最强。

### B. 主创新：Q-Former/Gated SC-HAMD Interaction + Future Queries

```text
frozen/partly-frozen HC backbone
SC/HAMD Q-Former or gated cross-attention → multi-layer latent
future ROI-time queries → cross-attention decoder → prediction
```

这同时借鉴 BLIP-2、Flamingo、Perceiver IO，真正改变“条件如何进入主干”和“未来如何从历史读取”。

### C. 高风险：State Slots + Heterogeneous Soft-MoE

```text
ROI-window tokens → state slots
[state, HAMD, latent] → coarse-to-fine router
→ {base, SC, pathology, state} adapter experts → residual
```

这借鉴 Slot Attention、AVMoE、MoVA。必须在确定性 baseline、专家有效样本数和 FC 指标已稳定后再实施。

## 7. 统一 ablation 和验收

| 问题 | 必须对照 |
|---|---|
| 图结构 | no-SC / fixed-SC / adaptive-only / SC+adaptive |
| 条件交互 | mask only / single cross-attention / gated multi-layer interaction |
| 预测头 | current flatten / future-query decoder |
| 多层信息 | final latent only / shallow+mid+deep gated connector |
| MoE | no-MoE / shared-only / dense soft / heterogeneous adapter experts |
| 病理 | no-HAMD / residual-only / AdaLN-FiLM / HAMD+state routing |
| 指标 | waveform metrics + FC metrics + 被试级 CI + 5 seeds |

## 8. 不建议优先做的事

1. 直接叠加 Mamba：窗口轴仅为 6、序列轴为 30，长上下文效率不是核心矛盾。
2. 直接扩展专家数量：E3 已近空转，增加 experts 会放大样本不足。
3. 在一版模型中同时加入 adaptive graph、full attention、hypergraph、router：它们均解决跨 ROI 交互，必须互斥比较。
4. 在修复 HAMD 编码和确定性评估前改大主干：任何结果变化均不可归因。

## 9. 实施顺序

```text
P0 可复现基线
  → FC 指标/低权重 FC loss
  → adaptive SC-FC graph
  → deterministic dense soft routing
  → Q-Former 或 gated SC/HAMD interaction
  → future-query decoder
  → state slots + heterogeneous Soft-MoE / hypergraph / GraphSSM
```

每个新增模块必须通过独立 ablation 回答“它是否有效、为何有效、代价是否合理”。
