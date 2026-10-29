# NeuroTwin

NeuroTwin 面向静息态 fMRI 的个体脑活动建模：根据已观测的 BOLD 历史、人口学信息与个体校准信息，预测 116 个 ROI 的后续信号，并以健康模型为参照开展病理偏离、脑网络和临床关联分析。

原始基线采用 **时空 Transformer 编码器 + 历史功能图先验 + 个体 profile + Graph-ODE 主预测头 + CPM 辅助预测头 + 方差读出**。针对现有病理参照与自由生成结果，新增独立的模型优化和病理预测实验；原配方和权重继续保留，技术设计及命令见 [模型优化方案与训练交付](docs/模型优化方案与训练交付.md)。

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

仅使用 HC 训练集，从已登记的 HC 权重初始化继续学习历史编码器、图先验、profile、Graph-ODE、CPM、条件编码与方差头。使用 `hc3/demo` 条件，即 Sex、Age、education。

目标是建立健康人群的条件化短程预测模型，为后续 MDD 适配及健康参照分析提供起点。验证集用于选模型，测试集不参与训练、条件统计量拟合或早停。

### 3.3 MDD 微调

加载 HC 预训练权重及共享条件字段统计量，增加五个临床字段，使用 `mdd8/cat` 拼接增量条件编码。`cat` 指字段特征拼接方式，并不表示所有字段都按类别处理。

1. 前 10 个 epoch 冻结共享主干，训练条件编码与预测头 FiLM 等带条件适配器标记的模块。
2. 从第 11 个 epoch 解冻编码器、图先验、profile 和预测头等共享模块；共享参数学习率为条件参数的 `0.2` 倍。解冻时保留已训练参数的 Adam 状态，并将新增参数组接入同一调度进度。

微调改善疾病条件下的预测适配，不自动证明模型学习到了疾病机制。若将 HAMD/HAMA 作为条件输入，再研究输出与相同量表的关联，应区分条件直接贡献和独立关联；当前健康参照主分析使用 HC 模型。

### 3.4 优化、验证与保存

采用 AdamW、线性 warmup 后余弦衰减、梯度裁剪与 EMA。编码器及 CPM 可使用 AMP，ODE 积分保持 FP32。验证脚本使用固定 `K=64`、每被试 8 个锚点；不设自由 rollout 任务。

**主要选模指标为验证集的 H1–8 被试等权 MAE + FC MSE**：MAE 先在被试内汇总、再在被试间平均，避免长扫描或更多窗口获得额外权重；FC MSE 取完整 8 步未来轨迹的 Pearson FC 非对角边均方误差。两项同为 C 空间误差、越低越好，等权相加。早停 patience 为 12；启用 EMA 时用 EMA 参数评价和保存最佳预测权重。

当前 HC 与 MDD 权重均于 2026-10-28 按本配方重训，训练脚本已修复分组学习率调度，使共享参数组保持相对学习率比例；不存在调度修复前后权重混用的问题。

## 4. 训练目标

当前阶段只监督 Graph-ODE 的 **H1–8** 未来轨迹，损失在固定前缀校准后的 BOLD 尺度计算：

| 损失项 | 监督对象 | 计算方式 |
|---|---|---|
| 重建 `L_rec` | ODE H1–8 | 预测与真实未来轨迹的逐元素 L1 |
| FC `L_fc` | ODE H1–8 | 预测与真实序列的 Pearson FC 矩阵（6,670 条非对角边）MSE |
| 频域 `L_freq` | ODE H1–8 | 沿时间做正交 rFFT，比较幅度谱的 L1 |

三项使用同方差不确定性加权（Kendall et al., CVPR 2018）：

$$
\mathcal{L}=\sum_{i\in\{\mathrm{rec},\mathrm{fc},\mathrm{freq}\}}
\left[\exp(-s_i)L_i+s_i\right].
$$

`s_i` 为可学习 log 方差，以 `-log(lambda_i)` 初始化，使初始有效权重等于下表；训练后权重可自适应变化。某个 batch 没有完整的 H1–8 真值时，FC/频域项整项跳过，不引入常数项。

| 参数 | 值 | 含义 |
|---|---:|---|
| `lambda_rec` | 1.0 | 重建项初始有效权重 |
| `lambda_fc` | 1.0 | FC 项初始有效权重 |
| `lambda_freq` | 1.0 | 频域项初始有效权重 |

当前阶段不使用 CPM/方差辅助头监督、H9–32 监督、时间差分、双头一致性或图正则。单步 MAE、H1–8 MAE、FC PCC/MSE 与频域 L1 是训练/验证的四个观察指标；checkpoint 以 H1–8 被试级 MAE + FC MSE 选优。

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
| `cpm_horizon` / `cpm_layers` | 8 / 2 | 当前 ODE 监督/评估时程固定为 H1–8；CPM 辅助头保留但本阶段不监督 |
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
| HC 预训练 | `checkpoints/h1_8_v2/neurotwin_pretrain/base_best.pt` | 健康参照、HC 预测及 MDD 偏离分析 |
| MDD 微调 | `checkpoints/h1_8_v2/neurotwin_finetune/finetuned_best.pt` | MDD 临床条件下的预测适配 |

权重 SHA256 登记于模型配置；`model.py` 根据内嵌参数重建模型、检查身份并严格加载权重。数据和权重不纳入 Git。

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

`PYTHON`、`DATA_ROOT`、`CHECKPOINT_DIR`、`HC_WEIGHT` 可覆盖运行环境与路径；预训练另可用 `INIT_WEIGHT` 指定初始化权重（本轮 HC 重训从上一版 HC 权重 warm start）。输入数据位于 `data/Raw/HC/`、`data/Raw/MDD/`，临床表路径见配置。输出目录若已存在会另建带后缀的目录，微调时应指定本次预训练实际产生的文件。

`SAVE_TRAINING_STATE=True` 可额外保存完整训练状态；`RESUME_TRAINING_STATE` 指向对应状态文件进行恢复。最佳预测权重与包含优化器、调度器、AMP、EMA、随机源等信息的完整恢复状态用途不同。

### 6.3 推理与评价

```bash
# 将 SUBJECT 替换成数据与临床表中的实际 ID
python inference_twin.py \
  --checkpoint checkpoints/h1_8_v2/neurotwin_pretrain/base_best.pt \
  --raw data/Raw/MDD/ROISignals_SUBJECT.mat --subject_id SUBJECT \
  --condition_file Rest-meta-MDD-V1V2-Merged-MDD.xlsx \
  --horizon 8 --device cuda:0

# MDD 微调模型的固定 test 评价，使用独立输出目录
python -m analysis.evaluate \
  --ckpt checkpoints/h1_8_v2/neurotwin_finetune/finetuned_best.pt \
  --out_dir results/evaluation --splits test --device cuda:0

# HC 参照模型的小规模滚动单步检查；--limit 0 表示完整划分
python -m analysis.rolling_one_step \
  --ckpt checkpoints/h1_8_v2/neurotwin_pretrain/base_best.pt \
  --group hc --split val --limit 5 --device cuda:0 \
  --out_dir results/rolling_smoke
```

完整病理分析依赖 HC/MDD 各划分的全量滚动预测、健康训练参照拟合、偏离打分及混杂控制，不能用上面 5 人的检查缓存替代。历史结论见[实验结论与证据汇总](docs/实验结论与证据汇总.md)；当前仅保留完整 test 的偏离评分，重跑病理流水线需重新生成训练参照与缓存。独立临床结局读出位于 `models/twin_outcome.py`，需要另行训练和验证，不包含在当前两份预测权重的效果结论中。

`analysis.evaluation_report` 根据已有评估产物生成分析报告；`analysis.evaluate_generation_test` 为历史自主生成评估入口，旧版权重报告已随过期结果清理，当前结果见[结果目录](results/README.md)。`analysis.test_visualization` 是历史独立测试集可视化流程，具体步骤见[可视化说明](docs/独立测试集可视化说明.md)。`analysis.audit_cached_predictions` 和 `analysis.audit_cached_generation` 分析已有预测缓存，需要另外提供对应缓存和权重。

## 7. 当前结果与解释边界

### 7.1 先区分两个协议

- **滚动单步**（项目主协议）：每个 TR 都用真实观测历史预测下一点，再把逐点预测拼成完整序列；衡量“跟踪真实轨迹”的能力，**不是自主生成**。
- **固定锚点多步 H1–8**：在单个 K64 锚点一次前向输出未来 8 步，未来真值不反馈；衡量多步外推/长程能力。
- **单步（H1）** 是两个协议的交点，但锚点/样本不同；**单个时间点无法定义 FC**，FC 只在 8 步窗口（多步协议）或完整预测序列（滚动协议）上计算。

### 7.2 指标定义

- **空间 PCC**：同一时间点，预测与真实的 116 维 ROI 向量做 Pearson 相关，逐样本计算后被试等权。公式为
  $r=\sum_i(p_i-\bar p)(t_i-\bar t)/\sqrt{\sum_i(p_i-\bar p)^2\sum_i(t_i-\bar t)^2}$，代码见 [train/losses.py:30](train/losses.py#L30)。
- **时间 PCC**：每个 ROI 沿时间序列计算预测-真实相关，再对 ROI 和被试平均，衡量“时间形状”而非单时点空间 pattern。
- **FC 边 PCC**：分别计算预测和真实的 Pearson FC 矩阵，取 6,670 条非对角上三角边，比较两张边向量的 Pearson 相关。
- **组平均 FC 边 PCC**：先跨样本平均 FC 矩阵，再比较边。
- MAE/MSE/RMSE 使用前 32 TR 固定校准后的 BOLD 尺度。

### 7.3 滚动单步（完整 test）

| 指标 | HC | MDD 微调 |
|---|---:|---:|
| 下一步 MAE ↓ | 0.090406 | 0.092031 |
| 下一步 MSE ↓ | 0.014422 | 0.015494 |
| 下一步 RMSE ↓ | 0.116932 | 0.119408 |
| persistence MAE ↓ | 0.600366 | 0.596894 |
| 相对 persistence 的 MAE 降幅 ↑ | 84.9% | 84.7% |
| 整段时间 PCC ↑ | 0.995636 | 0.995224 |
| 完整序列 FC 边 PCC ↑ | 0.997472 | 0.997215 |
| 完整序列 FC MAE ↓ | 0.014002 | 0.014553 |

滚动协议每个锚点都重新输入真实历史，因此时间 PCC/FC 很高，但这是跟踪能力，不是自主生成。

### 7.4 固定锚点多步 H1–8（完整 test，8 锚点/被试）

| 指标 | HC | MDD 微调 |
|---|---:|---:|
| 单步 H1 MAE ↓ | 0.090648 | 0.091969 |
| 单步 H1 MSE ↓ | 0.013370 | 0.013793 |
| 单步 H1 空间 PCC ↑ | 0.993798 | 0.993521 |
| H1–8 MAE ↓ | 0.534367 | 0.532069 |
| H1–8 MSE ↓ | 0.451918 | 0.449535 |
| H1–8 空间 PCC ↑ | 0.737308 | 0.737501 |
| H1–8 R² ↑ | 0.387756 | 0.401180 |
| H1–8 方差比 ↑ | 0.659 | 0.659 |
| H1–8 方向准确率 ↑ | 0.824945 | 0.823830 |
| H2–8 等权平均 MAE ↓ | 0.597756 | 0.594940 |
| H2–8 等权平均空间 PCC ↑ | 0.700666 | 0.700927 |
| FC MSE（8 步）↓ | 0.244646 | 0.246774 |
| FC 边 PCC（8 步）↑ | 0.603448 | 0.601232 |
| 组平均 FC 边 PCC ↑ | 0.982368 | 0.982881 |
| 频域 L1 ↓ | 0.362376 | 0.358904 |

HC 的逐 horizon 退化：

| HC test | H1 | H2 | H4 | H6 | H8 |
|---|---:|---:|---:|---:|---:|
| MAE ↓ | 0.0906 | 0.2546 | 0.6025 | 0.6664 | 0.8110 |
| 空间 PCC ↑ | 0.9938 | 0.9507 | 0.7376 | 0.6754 | 0.4286 |
| 方向准确率 ↑ | 0.9729 | 0.9223 | 0.8832 | 0.7489 | 0.6524 |

MDD 的 H8 为 MAE 0.8018、空间 PCC 0.4359、方向准确率 0.6552，与 HC 接近。

### 7.5 解释边界

- **单步**：空间 PCC≈0.994、MAE≈0.09、方向≈0.97，可以认为较准确；相对 persistence 的 MAE 降幅约 85%。
- **H1–8 多步**：平均 MAE≈0.53、空间 PCC≈0.74、方差比≈0.66；H4 以后明显退化，H8 空间 PCC≈0.43、方向≈0.65。**不能把单步精度当成多步能力**。
- 滚动单步的完整序列时间 PCC≈0.995、FC≈0.997；固定锚点 H1–8 的 8 步时间 PCC 约 0.70。两者不能互相替代。
- 8 步 FC 相比上一版权重改善约 11%，但完整序列 FC 基本不变（0.9975 vs 0.9976）。
- 旧版权重上的站点匹配病理分析（0/116 ROI 通过 FDR）**尚未用当前 HC 参照重算**；在重新生成滚动预测、参照统计与偏离评分之前，不能把旧结论直接归到当前权重。
- 当前尚未建立可靠的疾病定位或诊断标志物；高单步精度不能替代疾病分析的独立验证。结论与限制见[实验结论与证据汇总](docs/实验结论与证据汇总.md)。

分协议汇总见 [单步与多步协议汇总](results/H1-8当前结果/单步与多步协议汇总.json)；原始 test 指标见 [HC](results/H1-8当前结果/健康对照/指标_测试集.json)、[MDD](results/H1-8当前结果/抑郁症/指标_测试集.json)；完整序列 FC 见 [HC](results/H1-8当前结果/功能连接_健康对照/功能连接汇总_测试集.json)、[MDD](results/H1-8当前结果/功能连接_抑郁症_微调/功能连接汇总_测试集.json)。

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
|   `-- h1_8_v2/
|       |-- neurotwin_pretrain/   Current HC checkpoint
|       `-- neurotwin_finetune/   Current MDD checkpoint
|-- data/                         Raw data, metadata and fixed split manifests
|-- results/                       Current H1-8 test results and summary
`-- docs/                         Project status and research records
```

研究状态与文档导航见 [项目状态与工作范围](docs/项目状态与工作范围.md)；已有结论与证据见 [实验结论与证据汇总](docs/实验结论与证据汇总.md)。
