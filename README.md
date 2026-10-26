# NeuroTwin

NeuroTwin 面向静息态 fMRI 的个体脑活动建模：根据已观测的 BOLD 历史、人口学信息与个体校准信息，预测 116 个 ROI 的后续信号，并以健康模型为参照开展病理偏离、脑网络和临床关联分析。

当前采用固定结构：**时空 Transformer 编码器 + 历史功能图先验 + 个体 profile + Graph-ODE 主预测头 + CPM 辅助预测头 + 方差读出**。当前研究重点是健康参照下的病理分析，暂不优化长程预测。

模型与训练配方见 [configs/neurotwin.json](configs/neurotwin.json)，模块索引见 [models/README.md](models/README.md)。本文描述当前代码；已有权重的实际运行参数以 checkpoint 内嵌的 `config.args`、条件统计量和数据划分记录为准。

## 1. 项目任务与输入输出

### 1.1 预测任务

给定截至时刻 `t` 的可见历史，预测下一 TR 的全脑 ROI 信号，并通过同一动力学系统连续推进多个 TR。

| 项目 | 当前定义 |
|---|---|
| 数据 | Raw BOLD，MAT 文件中的 `data` 为 `[TR, ROI]`，取前 116 个 ROI |
| 历史输入 | 训练时 `K=16~64` TR；常规评价固定 `K=64` |
| 个体校准 | 每次扫描最早 32 TR；预测目标必须晚于校准区间 |
| 条件输入 | HC：Sex、Age、education；MDD：再加入 episode、illness、medication、HAMD、HAMA |
| 主输出 | Graph-ODE 的下一 TR 预测；内部接口形状 `[B,116,1,1]` |
| 动力学轨迹 | Graph-ODE 连续推进得到 `[B,116,H]`，当前辅助监督长度 `H=8` |
| 辅助输出 | CPM 并行预测未来 8 TR，以及逐 ROI、逐预测时点的方差估计 |

`B` 为批大小，`F=116` 为 ROI 数，`K` 为历史长度，`H` 为预测长度，`D=256` 为隐状态宽度。公开推理接口使用 `[B,K,F]` 历史张量；模型内部输入布局为 `[B,F,1,K]`。

### 1.2 病理与脑网络任务

以 **HC 预训练模型**作为健康参照，比较其对 HC 与 MDD 的预测误差和偏离分布，再开展 ROI、网络及临床关联检验。MDD 微调模型用于疾病条件下的预测适配，与健康参照承担不同任务。

```text
Observed BOLD history
         |
         v
HC reference model -> rolling one-step predictions -> residual features
                                                           |
                                                           v
                     HC training reference fit -> deviation scores
                                                           |
                                                           v
                          ROI / network / clinical association tests
```

预测精度、疾病效应与因果解释分别评价。当前分析可以研究预测偏离和功能耦合；干预接口、学习图或高预测相关性本身不构成有效连接、治疗响应或临床诊断能力的证据。

## 2. 模型结构

### 2.1 总体结构（ASCII）

```text
 Raw BOLD [B,T,116]                         Demographic / clinical fields
         |                                                |
         v                                                v
 First-32-TR fixed calibration                  Field encoders + masks
         |                                                |
         +------------------------+                       v
         |                        |                   c [B,32]
         v                        v                       |
 Visible history [B,K,116]   First 32 TR                   |
         |                        |                       |
         v                        v                       |
 Context RevIN             Prefix RevIN                   |
         |                        |                       |
         +--> abs(history FC)     |                       |
         |         |              |                       |
         |         v              |                       |
         |    Soft graph prior    |                       |
         |         |              |                       |
         |     A_eff [B,F,F]      |                       |
         |         |              |                       |
         v         v              v                       |
 Patch embedding +       Shared encoder on prefix         |
 shared encoder x4       (own prefix graph)               |
         |                        |                       |
         v                        v                       |
 Tokens [B,F,P,256]       Pool + ProfileEncoder            |
         |                        |                       |
         +--> last token z0       p [B,32]                |
         +--> patch mean M -------+----------+            |
         |                                   |            |
         |     z0, last BOLD q0, A_eff -------+            |
         |                                   v            v
         |                            Graph-ODE + head FiLM
         |                            Heun: 2 substeps / TR
         |                                   |
         |                    +--------------+--------------+
         |                    |                             |
         |                    v                             v
         |              BOLD trajectory              Latent trajectory
         |              q1 ... qH                     (stop-gradient)
         |                    |                             |
         |                    v                             v
         |              Inverse RevIN                 VarianceHead
         |                    |                             |
         |              Main: next TR                 Per-ROI variance
         |              Also: ODE rollout
         |
         +--> CPM: horizon queries + cross-attention <----- c
                              |
                              v
                        Inverse RevIN
                              |
                        Auxiliary H=8 BOLD
```

图中 `P=ceil(K/8)` 为时间 patch 数，`K=64` 时 `P=8`。前缀分支与历史分支共享编码器参数，分别计算各自可见区间的归一化统计量与功能图。临床条件 `c` 注入 Graph-ODE 和 CPM 预测头；个体码 `p` 与历史记忆 `M` 注入 Graph-ODE。

单个共享编码块的内部结构如下，三个子层均采用预归一化与残差连接：

```text
Tokens [B,F,P,256]
    |
    v
RMSNorm -> Causal temporal attention -> Dropout -> Residual add
    |
    v
RMSNorm -> ROI attention + graph bias -> Dropout -> Residual add
    |
    v
RMSNorm -> FFN (256 -> 1024 -> 256) -> Dropout -> Residual add
    |
    v
Next block / final RMSNorm
```

时间注意力在每个 ROI 内沿 patch 轴计算，使用因果 mask；ROI 注意力在同一 patch 位置跨 ROI 计算，注意力分数加入 `beta * log(eps + A_eff)`，其中 `beta` 为逐头可学习参数。图由整段已观测历史构建，因此这里的因果 mask 指时间注意力的连接约束，整个模型的预测边界是不得读取目标区间。

### 2.2 各模块的作用

| 模块 | 实现 | 作用与设计 |
|---|---|---|
| 固定扫描校准 | `utils/dataloader.py` | 仅用最早 32 TR 计算每个 ROI 的均值、标准差，再变换整段扫描；避免用未来数据估计扫描尺度 |
| Context RevIN | `models/common.py` | 使用当前历史的统计量进行可逆归一化，预测后恢复到固定校准尺度；与扫描前缀校准属于两层不同处理 |
| 历史功能图 | `models/tfm.py` | 由当前历史的 ROI Pearson 相关取绝对值、去对角，提供非负耦合先验 |
| 软图先验 | `models/graph_prior.py` | 混合历史图与学习图，加入低秩残差，经对称 Sinkhorn 归一化得到 `A_eff`；同一历史的图供编码器和 ODE 使用 |
| Patch embedding | `models/tfm_backbone.py` | 每个 ROI 沿时间按 8 TR 分片，加入 ROI 与时间位置信息，压缩时间 token 数 |
| 时空编码器 | `models/tfm_backbone.py` | 4 个编码块交替进行因果时间注意力、图引导 ROI 注意力和前馈变换，提取局部时间变化与跨 ROI 关系 |
| 条件编码器 | `models/conditioning.py` | 对连续、类别字段分别编码，显式保留缺失 mask，输出 32 维条件码；MDD 以增量形式扩展 HC 字段 |
| 个体 profile | `models/tfm_personal.py` | 将最早 32 TR 的共享编码器 tokens 在 ROI 与 patch 维池化，经 MLP 得到 32 维个体码，描述当前扫描的个体特征 |
| 历史记忆 | `models/tfm.py` | 对历史 tokens 沿 patch 维平均，得到逐 ROI 记忆；通过 FiLM 持续调制 ODE 残差，补充最后 token 的信息 |
| Graph-ODE | `models/tfm_dynamics.py` | 融合图扩散、神经残差和可选控制项，连续演化隐状态，同时积分 BOLD 变化率；产生主预测 |
| CPM | `models/tfm.py` | 未来 horizon 查询通过两层 cross-attention 读取全部历史 tokens，一次并行预测 8 TR，提供辅助多步约束 |
| 方差头 | `models/tfm_dynamics.py` | 从停止梯度的 ODE 隐状态轨迹预测 log-variance；其 NLL 只训练方差头，不改变均值动力学 |

代码中的 `sc_*` 参数和 `SoftAnatomicalPrior` 是现有接口名称。当前 `graph_source=history_fc`，输入是**历史功能连接**，没有把独立实测的解剖 SC 输入模型。`A_eff` 是预测模型的非负耦合图，与保留正负号的信号 FC 矩阵也不是同一个对象。

### 2.3 条件设计与个体化

HC 的三个共享字段分别编码后，生成 `c_demo`。MDD 新增五个字段，经拼接增量 MLP 得到 `DeltaEncoder` 输出，并与共享条件码相加：

```text
 Sex / Age / education --------------------> E_demo -> c_demo ----+
                                                               |
 episode / illness / medication / HAMD / HAMA -> E_new            +--> c_MDD
                                                  |            |
                            [c_demo, E_new, masks] -> DeltaMLP ---+
```

增量 MLP 的末层零初始化，使新建微调模型初始继承 HC 条件表示，再学习新增临床字段的作用。连续字段使用训练集拟合的稳健统计量；类别字段使用固定词表，未知类别与合法的数值编码分开处理。微调继承 HC 三字段的统计量，新字段统计量仅在 MDD 训练集上拟合。

条件 FiLM 对预测头的隐藏表示做缩放与平移。个体 profile 则来自 BOLD 前缀，无需临床字段。两者分别表达外部已知信息和扫描内个体特征；当前主干注意力块不接受临床条件调制。

### 2.4 Graph-ODE 的计算含义

设隐状态为 $z$，RevIN 空间的 BOLD 状态为 $q$，图拉普拉斯为 $L_A$，条件码、个体码和历史记忆分别为 $c$、$p$、$M$。当前动力学可概括为：

$$
\begin{aligned}
\frac{\mathrm{d}z}{\mathrm{d}t}
&= -\kappa(c,p)L_Az + 0.1\,r_{\theta}(z,A,c,p,M)
   + g_{\theta}(z,c,p)\odot G_{\theta}u, \\
\frac{\mathrm{d}q}{\mathrm{d}t} &= W_yz+b_y.
\end{aligned}
$$

第一项通过图在 ROI 间传播状态；第二项学习超出线性扩散的变化，并接受条件、profile 和记忆的 FiLM 调制；第三项是可选控制输入 `u` 的门控作用。扩散率 `kappa` 经 softplus 保持为正。当前静息态训练没有干预标签，常规推理不传控制输入，此项为零。

初始 `z0` 为最后一个历史 token，`q0` 为最后一帧观测 BOLD。Heun 二阶积分每 TR 使用 2 个子步，隐状态与 BOLD 同步推进；ODE 求解使用 FP32。一次自由 rollout 内，图、条件、profile、记忆和 RevIN 统计量保持固定，中途不输入未来真值。CPM 是独立辅助解码头，不承担 ODE 的状态递推。

## 3. 训练策略

### 3.1 数据划分与采样

按**被试**划分训练、验证、测试集，比例约为 `8:1:1`，同一被试的不同窗口不跨集合。当前固定划分为：

| 人群 | 训练 | 验证 | 测试 | 合计 |
|---|---:|---:|---:|---:|
| HC | 1778 | 222 | 222 | 2222 |
| MDD | 1803 | 225 | 225 | 2253 |

划分读取优先级为：显式 `split_file` → 已有 `subject_split_<mode>.json` → 按种子生成。复现已有实验应使用其固定名单，而不只是相同划分比例。

训练视图初始化时按 `sampling_seed` 生成被试与历史长度的样本表；每次读取样本时随机抽取合法预测锚点，每个 epoch 重排长度分桶批次。`train_samples_per_subject=0` 使用自动采样，单被试样本数上限为 12。锚点同时满足历史长度、32 TR 校准和目标可用性约束。训练使用真实历史，不将整个测试序列用于统计量拟合。

训练历史输入还使用噪声、缩放、通道丢弃与时间遮蔽增强；验证/测试不做随机增强。当前加载器中的增强参数为 `noise_std=0.03`、`scale_std=0.05`、`channel_drop_prob=0.03`、`time_mask_prob=0.15`、`max_mask_ratio=0.10`，定义于 `utils/dataloader.py`，不在 JSON 配方中。

### 3.2 HC 预训练

仅使用 HC 训练集，从初始化参数开始联合学习历史编码器、图先验、profile、Graph-ODE、CPM、条件编码与方差头。使用 `hc3/demo` 条件，即 Sex、Age、education。

目标是建立健康人群的条件化短程预测模型，为后续 MDD 适配及健康参照分析提供起点。验证集用于选模型，测试集不参与训练、条件统计量拟合或早停。

### 3.3 MDD 微调

加载 HC 预训练权重及共享条件字段统计量，增加五个临床字段，使用 `mdd8/cat` 拼接增量条件编码。`cat` 指字段特征拼接方式，并不表示所有字段都按类别处理。

1. 前 10 个 epoch 冻结共享主干，训练条件编码与预测头 FiLM 等带条件适配器标记的模块。
2. 从第 11 个 epoch 解冻编码器、图先验、profile 和预测头等共享模块；共享参数学习率为条件参数的 `0.2` 倍。解冻时保留已训练参数的 Adam 状态，并将新增参数组接入同一调度进度。

微调改善疾病条件下的预测适配，不自动证明模型学习到了疾病机制。若将 HAMD/HAMA 作为条件输入，再研究输出与相同量表的关联，应区分条件直接贡献和独立关联；当前健康参照主分析使用 HC 模型。

### 3.4 优化、验证与保存

采用 AdamW、线性 warmup 后余弦衰减、梯度裁剪与 EMA。编码器及 CPM 可使用 AMP，ODE 积分保持 FP32。验证脚本使用固定 `K=64`、每被试 16 个锚点，并另设自由 rollout 任务。

**主要选模指标为验证集 Graph-ODE H1–H8 的被试等权 MAE**：先汇总每个被试的误差，再在被试间平均，避免长扫描或更多窗口获得额外权重。早停 patience 为 20；启用 EMA 时用 EMA 参数评价和保存最佳预测权重。

当前训练脚本已修复分组学习率调度，使共享参数组保持相对学习率比例。保留的 MDD 权重在该修复前训练，不能把当前调度描述成它的完整实际训练轨迹；本说明与整理工作没有重新训练已有权重。

## 4. 训练目标

信号预测损失在**固定前缀校准后的 BOLD 尺度**计算；方差 NLL 在模型内部的 RevIN 空间计算。当前目标包括：

| 损失项 | 监督对象 | 当前设置与作用 |
|---|---|---|
| 单步 `L_one` | ODE 的 H1 | Huber + `0.1 × (1 - 空间 PCC)`，兼顾幅值误差和同一时点的全脑空间形状 |
| CPM `L_cpm` | CPM 的 H1–H8 | 逐 horizon 的 Huber + 空间 PCC；`gamma=1`，各 horizon 均匀加权 |
| 动力学 `L_dynamics` | ODE 的 H2–H8 | Huber + 空间 PCC；H1 已由单步项监督，不在此重复 |
| 方差 `L_variance` | ODE 的 H1–H8 方差 | 对角高斯 NLL；均值、目标与输入隐状态停止梯度，仅拟合方差头 |
| 时间差分 `L_td` | CPM 与 ODE 轨迹 | 约束预测变化量；CPM 首步以真实历史末帧为参照，ODE 从 H2 起约束相邻预测差分 |
| 双头一致性 `L_cross` | CPM 对齐 ODE | L1 一致性，ODE 为停止梯度的教师；前 5 个 epoch 关闭，随后启用 |
| 图正则 `R_graph` | `A_eff` | 非对角 L1 与行熵正则，权重分别为 `1e-3`；图时间正则权重为 0 |

四个主要任务使用可学习的不确定性权重。令 $\mathcal{I}$ 表示单步、CPM、动力学、方差四项任务，$s_i$ 为各任务可学习的 log-variance 参数，则有效任务的总目标为：

$$
\mathcal{L}
= \sum_{i\in\mathcal{I}}\left[\exp(-s_i)L_i+s_i\right]
+0.05L_{\mathrm{td}}+0.02L_{\mathrm{cross}}+R_{\mathrm{graph}}.
$$

`s_i` 以 `-log(lambda_i)` 初始化，使初始有效权重等于下表的 `lambda_i`；训练后权重可变化。无有效监督的任务通过 mask 整项排除。时间差分、一致性和图正则保持固定权重，不进入上述自适应聚合。

| 参数 | 值 | 含义 |
|---|---:|---|
| `lambda_one` | 1.0 | 单步任务初始有效权重 |
| `lambda_cpm` | 1.0 | CPM 任务初始有效权重 |
| `lambda_dynamics` | 0.5 | ODE 多步任务初始有效权重 |
| `lambda_variance` | 0.05 | 方差任务初始有效权重 |
| `lambda_pcc` | 0.1 | 各预测任务内部空间 PCC 项权重 |
| `lambda_temporal_diff` | 0.05 | 时间差分固定权重 |
| `lambda_cross_head` / `cross_head_start_epoch` | 0.02 / 5 | 双头一致性权重与启用时机，epoch 使用零基索引 |
| `cpm_huber_delta` | 1.0 | Huber 二次区间阈值 |
| `cpm_horizon_weight` / `cpm_gamma` | `gamma` / 1.0 | CPM 当前使用均匀时程权重 |

当前配置未启用额外的未来轨迹时间 PCC 损失。训练中的空间 PCC、评价中的逐 ROI 时间 PCC、FC 矩阵之间的相关性属于不同指标。任务权重中的“不确定性”也不同于方差头输出的逐 ROI 预测方差。

## 5. 参数设计

以下是当前固定配方及其计算作用，并不表示这些值已经被证明为最优超参数。

### 5.1 网络与个体化参数

| 参数 | 当前值 | 设计作用 |
|---|---|---|
| `num_rois` | 116 | 固定 ROI 维度，统一预测、图和网络分析对象 |
| `context_min` / `context_max` | 16 / 64 | 变长历史训练；评价常用最长 64 TR |
| `calibration_length` | 32 | 固定扫描尺度和个体 profile 的可见前缀长度 |
| `tfm_patch_len` | 8 | 每个 token 覆盖 8 TR；不足整片时左侧补齐 |
| `tfm_dim` / `tfm_layers` / `tfm_heads` | 256 / 4 / 4 | 隐状态宽度、共享编码块数与注意力头数；每头宽度 64 |
| `tfm_ff_ratio` | 4 | 编码器前馈隐藏宽度为 1024 |
| `tfm_one_step_hidden_dim` | 256 | ODE 神经残差隐藏宽度 |
| `dropout` / `norm` | 0.1 / `true` | 编码与辅助解码正则，以及可逆输入归一化 |
| `cpm_horizon` / `cpm_layers` | 8 / 2 | 辅助未来时程与 cross-attention 解码层数 |
| `profile_dim` | 32 | 前缀个体码宽度；ProfileEncoder 隐藏宽度为 64 |
| `dynamics_history_memory` | `true` | 将历史 patch 平均记忆持续注入 ODE |
| `cond_dim` | 32 | 条件码宽度 |
| `cond_d_demo` / `cond_d_new` | 16 / 16 | 共享字段与新增字段的逐字段编码宽度 |
| `cond_h_delta` / `cond_delta_style` | 64 / `mlp` | MDD 条件增量网络隐藏宽度与形式 |
| `cond_inject` / `cond_dropout_p` | `heads` / 0.0 | 条件注入预测头；当前不做随机条件遮蔽 |

### 5.2 图与动力学参数

| 参数 | 当前值 | 设计作用 |
|---|---|---|
| `graph_source` | `history_fc` | 从当前可见历史计算功能图 |
| `sc_prior_mode` / `sc_lambda_mode` | `soft_prior` / `global` | 历史图与学习图采用全局可学习比例混合 |
| `sc_lambda_init` | 0.7 | 初始历史图占混合权重的 70%；训练后可变化 |
| `sc_prior_rank` | 12 | 学习功能图的低秩投影维度 |
| `sc_delta_a` / `sc_delta_rank` | `true` / `null` | 启用低秩图残差；`null` 按 `max(4,rank//2)` 得到秩 6，残差初始输出为零 |
| `sc_sinkhorn_iters` | 64 | 对称非负图的迭代归一化次数 |
| `dynamics_mode` | `graph_ode` | 当前唯一的动力学实现 |
| `ode_steps` / `dynamics_dt` | 2 / 1.0 | 每 1 TR 区间执行两次 Heun 子步；时间单位是 TR |
| ODE 残差系数 / 扩散率初值 | 0.1 / 0.05 | 控制神经残差幅度及初始图扩散强度，实现默认值 |
| 方差 log-range | `[-6,4]` | 约束预测 log-variance，实现默认值 |
| `cpm_intervention` | `false` | CPM 不接收控制嵌入；可选控制通过 ODE 处理 |

### 5.3 优化参数与随机性

| 参数 | 当前值 | 作用 |
|---|---|---|
| `batch_size` / `train_epochs` | 256 / 150 | 训练批大小与最大 epoch 数，允许提前停止 |
| 优化器 / betas | AdamW / `(0.9,0.95)` | 自适应优化，实现默认 betas |
| `lr_init` / `lr_peak` / `lr_final` | `5e-5` / `1e-4` / `5e-5` | 主学习率 warmup 起点、峰值与余弦终点 |
| `warmup_epochs` | 15 | 线性预热长度 |
| `weight_decay` / `grad_clip` | 0.01 / 1.0 | 权重衰减与梯度范数裁剪；可学习损失权重不做 weight decay |
| `freeze_backbone_epochs` / `backbone_lr_scale` | 10 / 0.2 | MDD 微调冻结期及解冻后共享参数学习率比例 |
| `use_ema` / `ema_decay` | `true` / 0.999 | 使用参数指数滑动平均进行评价与选模 |
| `patience` / `amp` | 20 / `true` | 早停耐心与混合精度开关 |
| `seed` / 固定划分 seed | 2024 / 2024 | 模型初始化等随机过程与已有数据划分 |
| `sampling_seed` | 20261009 | 训练样本表与长度分桶采样随机源；随机锚点使用 DataLoader 的随机流 |
| `cond_dropout_seed` | 20261008 | 条件遮蔽独立随机源；当前遮蔽概率为 0 |

当前保留的 HC 模型参数量为 **6,233,299**，MDD 模型为 **6,244,019**，均指完整模型参数，不含损失函数自身的可学习权重。两者主体结构一致，参数量差别来自条件字段扩展。

## 6. 权重与运行入口

### 6.1 当前权重

| 角色 | 路径 | 用途 |
|---|---|---|
| HC 预训练 | `checkpoints/neurotwin_pretrain/base_best.pt` | 健康参照、HC 预测及 MDD 偏离分析 |
| MDD 微调 | `checkpoints/neurotwin_finetune/finetuned_best.pt` | MDD 临床条件下的预测适配 |

权重 SHA256 登记于模型配置；`model.py` 根据内嵌参数重建模型、检查身份并严格加载权重。历史 H32 实验权重单独保存在 `archive/checkpoints/h32_best.pt`，不属于当前默认训练入口。数据和权重不纳入 Git。

```bash
# 在项目根目录运行：仅检查配置、哈希与严格加载，不训练
python model.py --mode pretrain --device cpu
python model.py --mode finetune --device cpu
```

### 6.2 训练

运行环境需要项目实际使用的 PyTorch、NumPy、SciPy、pandas、scikit-learn、openpyxl 和 tqdm 等依赖；绘图另需 Matplotlib。训练和批量推理使用可用 GPU。以下示例将新训练写入独立目录，并显式指定微调起点：

```bash
DEVICE=cuda:0 CHECKPOINT_DIR=checkpoints/new_training bash scripts/pretrain.sh

DEVICE=cuda:0 CHECKPOINT_DIR=checkpoints/new_training \
  HC_WEIGHT=checkpoints/new_training/neurotwin_pretrain/base_best.pt \
  bash scripts/finetune.sh
```

`PYTHON`、`DATA_ROOT`、`CHECKPOINT_DIR`、`HC_WEIGHT` 可覆盖运行环境与路径。输入数据位于 `data/Raw/HC/`、`data/Raw/MDD/`，临床表路径见配置。输出目录若已存在会另建带后缀的目录，微调时应指定本次预训练实际产生的文件。

`SAVE_TRAINING_STATE=True` 可额外保存完整训练状态；`RESUME_TRAINING_STATE` 指向对应状态文件进行恢复。最佳预测权重与包含优化器、调度器、AMP、EMA、随机源等信息的完整恢复状态用途不同。

### 6.3 推理与评价

```bash
# 将 SUBJECT 替换成数据与临床表中的实际 ID
python inference_twin.py \
  --checkpoint checkpoints/neurotwin_pretrain/base_best.pt \
  --raw data/Raw/MDD/ROISignals_SUBJECT.mat --subject_id SUBJECT \
  --condition_file Rest-meta-MDD-V1V2-Merged-MDD.xlsx \
  --horizon 8 --device cuda:0

# MDD 微调模型的固定 test 评价，使用独立输出目录
python -m analysis.evaluate \
  --ckpt checkpoints/neurotwin_finetune/finetuned_best.pt \
  --out_dir results/evaluation --splits test --device cuda:0

# HC 参照模型的小规模滚动单步检查；--limit 0 表示完整划分
python -m analysis.rolling_one_step \
  --ckpt checkpoints/neurotwin_pretrain/base_best.pt \
  --group hc --split val --limit 5 --device cuda:0 \
  --out_dir results/rolling_smoke
```

完整病理分析依赖 HC/MDD 各划分的全量滚动预测、健康训练参照拟合、偏离打分及混杂控制，不能用上面 5 人的检查缓存替代。步骤见[病理分析记录](results/pathology_analysis/run1/README.md)。独立临床结局读出位于 `models/twin_outcome.py`，需要另行训练和验证，不包含在当前两份预测权重的效果结论中。

## 7. 当前结果与解释边界

以下指标来自 **HC 参照权重**在完整独立 test 划分上的滚动单步预测，先逐被试计算，再等权平均：

| 指标 | HC（222 人） | MDD（225 人） |
|---|---:|---:|
| MAE ↓ | 0.078864 | 0.080961 |
| RMSE ↓ | 0.102733 | 0.105688 |
| R² ↑ | 0.992388 | 0.991629 |
| 逐 ROI 时间 PCC ↑ | 0.996208 | 0.995812 |
| 逐时点空间 PCC ↑ | 0.994882 | 0.994403 |
| FC 上三角 PCC ↑ | 0.997633 | 0.997384 |

MAE/RMSE 使用前 32 TR 固定校准后的尺度，不是原始扫描强度单位。时间 PCC 沿时间比较同一 ROI 的预测与真值；空间 PCC 在同一时点跨 ROI 比较；FC PCC 则先分别从预测序列与真实序列计算两张有符号 Pearson FC 矩阵，再比较 6,670 条非对角上三角边。

滚动单步的每个预测点都重新输入真实历史，不能代表连续自主生成能力。已有 `K64 -> H32` 自由外推图来自 HC/MDD 各 20 人的 val/test 抽样，时间 PCC 均值约 0.093/0.128、FC PCC 均值约 0.277/0.263，尚不支持长程轨迹重建良好的结论；该抽样也不是完整独立 test。当前按项目方向继续研究健康参照偏离，保留此能力边界。

已有站点、年龄、性别匹配的病理分析中，模型偏离的 116 个 ROI 均未通过 FDR 校正，网络与全脑组间差异也不显著。当前尚未建立可靠的疾病定位或诊断标志物；高单步精度不能替代疾病分析的独立验证。结论与限制见[收档报告](docs/V2_CLOSING_REPORT.md)。

详细指标见[预测性能汇总](results/prediction_metrics_20261026/README.md)和[完整 test FC 结果](results/fc_test_20261026/README.md)。[结果目录说明](results/README.md)列出当前保留产物；部分逐时点滚动与扰动中间缓存已清理，重新计算相关结果需先生成缓存。

## 8. 文件结构

```text
NeuroTwin/
|-- README.md                     Project task, architecture and training
|-- model.py                      Fixed model config and strict weight loading
|-- main.py                       Pretraining and finetuning loops
|-- inference_twin.py             Single-subject state prediction and update
|-- configs/
|   `-- neurotwin.json             Current recipe and checkpoint identities
|-- models/                       Encoder, graph, conditions, profile and heads
|-- train/                        Losses, optimizer and training state
|-- scripts/
|   |-- pretrain.sh               HC training entrypoint
|   |-- finetune.sh               MDD training entrypoint
|   `-- _model_defaults.sh         Shared recipe reader
|-- utils/                        Data, condition schema and subject splits
|-- analysis/                     Prediction, FC, deviation and pathology tests
|   `-- paper_figs/                Existing figure reproduction code
|-- tests/                        Implementation checks
|-- checkpoints/
|   |-- neurotwin_pretrain/        Current HC checkpoint
|   `-- neurotwin_finetune/        Current MDD checkpoint
|-- data/                         Raw data, metadata and fixed split manifests
|-- results/                      Retained metrics, analyses and figures
|-- docs/                         Project status and research records
|-- archive/checkpoints/          Historical H32 checkpoint
`-- experiments/long_horizon/     Paused long-horizon experiment tools
```

研究状态与后续约定见 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)；已有实验记录见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
