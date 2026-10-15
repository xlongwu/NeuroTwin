# NeuroTwin 十模块借鉴与结构优化：Codex 实施方案

- 日期：2026-10-11；性质：结构实现规格，尚未修改模型或开展训练。
- 来源：[即插即用模块论文解读与适配评估](即插即用模块_论文解读与NeuroTwin适配评估.md)。模块编号沿用该文档。
- 代码基线：本日实际工作区，HEAD 为 `becb0d99dba4fdb59c7afbb25ddff37e035752ca`；条件注入等未提交代码确实存在，不能用 HEAD 的旧接口替代工作区接口。
- 本文独立保存，保留现有结构方案、未提交代码、定型配置和检查点。与旧方案不同的设计以本文所述新候选为准，不自动改写旧实验。

## 1. 目标与默认决策

目标是让模型同时表达：**个体当前动力学、相对健康参照的偏离、临床信息对预测的修正、外部驱动引起的未来状态变化**。不把病理分析压缩成一个分类分数，也不把健康/患者之间的差异直接当作刺激疗效。

推荐新候选 `tfm_twin_v1`：

**原有 patch 嵌入 → 4 层因果多尺度图混合器 → 共享临床乘性交互残差 → 原有 One-Step/CPM 双头；并联健康参照读出和图控制响应分支。**

从源文档选十个模块，但不同时叠加十个完整网络：七个模块的机制组成四个实际构件；其余三个给出可实施的替换规格，默认不创建其参数。选择依据是与上述目标的作用位置、所需数据和实现成本，而非论文指标排名。

| 优先序 | 来源模块 | 默认处置 | 直接服务的目标 |
|---|---|---|---|
| 1 | DDI，171 / P104 | 与 MDM 思想一起改造，替换时间/ROI attention 子路径 | 个体历史动力学与脑网络交互 |
| 2 | StarBlock，222 / P058 | 借鉴乘性交互，构建共享临床残差 | 同一病理条件对不同脑状态产生不同修正 |
| 3 | ContMix，228 / P048 | 借鉴上下文生成动态核，改为图上的控制响应核 | 个体化刺激传播与时间响应 |
| 4 | INP，210 / P044 | 借鉴正常原型重建；改为固定观测描述量上的 HC 参照 | ROI 级健康偏离与分量解释 |
| 5 | LSK，206 / P074 | 合并进时间混合器的尺度选择 | 不同 ROI/历史状态选择不同时间范围 |
| 6 | Moga，186 / P117 | 合并进时间混合器的局部多尺度门控 | 保留局部动态与慢变化的互补信息 |
| 7 | CGA，189 / P142 | 合并进临床残差的内容门控 | 让条件修正依赖当前 ROI 状态 |
| 8 | WConv，215 / P204 | 备选：替换时间混合器的局部分支 | 局部时间核的参数化先验 |
| 9 | DCTSA，201 / P107 | 备选：替换尺度选择器的描述量 | 历史频率信息辅助选择 |
| 10 | AgentA，89 / P080 | 备选：替换 CPM memory attention | 长上下文下压缩历史记忆 |

这十项是“最值得借鉴的机制”，不意味着十项均已具有脑影像迁移证据。源文档仅把 WConv/DDI 列为直接或改造适配候选；本方案扩大的是**针对新目标的结构借鉴**，没有将其余视觉模块重新宣布为可直接搬用。

## 2. 当前实现与需要补齐的部分

1. `models/tfm_backbone.py::TemporalPatchEmbed` 将 `[B,F,K]` 映射到 `[B,F,P,D]`。时间轴是 P，ROI 轴是 F；不能把 ROI 索引视作二维像素邻域。当前 patch 投影没有显式多尺度路径。
2. `TFMEncoderBlock` 顺序是因果时间注意力、SC 引导 ROI 注意力、FFN；工作区还接受三个 norm 的 `cond_mods`。新候选用一个时间/图混合器替换前两项，保留 FFN 和三个 RMSNorm 槽位。
3. `SoftAnatomicalPrior` 输出样本级 `A_eff`，供全层与 One-Step 共享。它是稠密软图，SC 只是先验，不能称为严格 SC 邻域，也不能将其注意力权重解释为因果有效连接。
4. `models/conditioning.py::ConditionEncoder` 已实现 HC 三字段、MDD 八字段、`c=c_demo+delta`；现有双头 FiLM 使用融合后条件。新候选复用字段编码、缺失 mask、train-only 统计和 dropout，不增加第二套临床数据入口。
5. One-Step 不接刺激；CPM 的 `intervention_embed` 只是未来 query 加法。新候选统一使用两个头共享的控制增量，避免“只有 CPM 才受到刺激”。
6. 现有输出主要是 BOLD 预测与图，缺少健康参照偏离读出。新方案补充 ROI × 描述分量和控制响应图，而不凭无标签数据新增疾病诊断监督。
7. `graph_prior.py` 中 `dp/dq` 两个低秩因子都零初始化，乘积对两者的初始梯度均为零；这是实际可学习性缺陷。新候选初始化时令 `dp.weight` Xavier、`dq.weight` 零，两个 bias 零，保持初始残差为零而允许 `dq` 首步学习。加载已有权重时不重新初始化，也不据此声称旧图残差曾训练有效。

默认参数：ROI=116、context=16–64 TR、patch=4、D=256、4 层、4 heads、FFN ratio=4、dropout=0.1、CPM H=8/2 层；SC rank=12、lambda_init=0.7、Sinkhorn=64。临床 C=32、适配器隐宽 R=64。显式写 `patch_len=4`，当前 CLI 的默认 8 不用于本候选。损失权重、优化器、种子与数据处理沿用当前训练入口，除非下面的响应数据路径确有新增目标。

## 3. 输入输出及总流程

符号：B 为 batch，F 为 ROI，K 为历史 TR 数，P=ceil(K/4)，D 为 token 宽度，H 为预测 TR 数，C 为条件宽度。

| 输入 | 形状 | 含义与来源 |
|---|---|---|
| `x` | `[B,F,1,K]` | 历史 BOLD；继续使用 context-only BrainRevIN |
| `sc_matrix` | `[F,F]` 或 `[B,F,F]` | 群体或个体 SC；来源身份保留在现有配置 |
| `condition` | 原有 `cond_z/cond_code/cond_mask` | Sex/Age/education；MDD 再加 episode/illness/medication/HAMD/HAMA |
| `future_control` | `[B,F,H]` | 未来每 ROI、每 TR 的已知空间驱动，不是临床评分 |

保留 `forward` 的 `(pred, aux_info)` 返回契约，One-Step 为 `[B,F,1,1]`，`aux_info['cpm_pred']` 为 `[B,F,H]`，均在原始 BOLD 空间。新增：

- `one_baseline`、`cpm_baseline`：无控制的完整状态预测。
- `control_delta_one`、`control_delta_cpm`：有/无控制预测之差，原始空间；CPM 的形状为 `[B,F,H]`。
- `scale_weights`：四层的 `[B,F,P,3]` 权重，仅在 `return_twin_info=True` 时收集。
- `clinical_delta_tokens`：`[B,F,P,D]`，临床 token 修正；默认诊断导出 detach，不在常规日志保存大张量。
- `twin_state`：最终临床修正 token 的最后状态 `[B,F,D]`；`sc_prior` 继续提供个体软图。
- `normal_residual`：`[B,F,6]`；`normal_distance`：`[B,F]`，由分析脚本加载参照后附加到其结果字典；模型forward不新增参照参数。

```text
x → 原有 RevIN → x_grid → 原有 SC 软先验 → A_eff
                     └→ patch_embed → 4×TwinMixerBlock → T

condition → 一次字段编码/dropout → c_demo、delta_c、c_full
T、delta_c → ClinicalStarAdapter（Star + CGA）→ T_clin
T_clin、c_demo → 原有双头人口学 FiLM → 无控制 baseline
T_clin_last、c_full、A_eff、future_control → GraphContMixResponse → 控制增量
baseline + 增量 → 使用同一历史 RevIN stats 反归一化 → 既有 pred/cpm_pred

原始 x 的固定描述量 → HC 原型参照 → ROI/分量偏离
```

条件只走一条临床路径：双头 FiLM 接 `c_demo`；新增临床信息接共享适配器。新候选不构建临床 AdaRMSNorm、旧完整条件 FiLM、旧 gated residual、MoE/Router/RECC，不删除这些旧模式代码。刺激分支的 `c_full` 只决定响应核，不再次改变无刺激 baseline。

## 4. 十模块的机制、改造和实施规格

本节所有 NeuroTwin 公式都是**本方案的实现定义**，不冒充原论文逐行复现。原机制说明只取与改造相关的部分。

### 4.1 DDI：时间递推与跨变量交互分开

原 AMD 的 MDM 负责多尺度分解，DDI 使用前一历史 patch 的残差递推与跨变量 MLP，AMS 负责预测器聚合。这里借鉴前两者，不移植 AMS/MoE。论文也明确讨论跨变量交互可能引入干扰。[原文 §4.1–4.3](https://arxiv.org/html/2406.03751v1#S4)

**Old → New**：`temporal_attention → roi_attention` 改成 `多尺度局部信息 → 有界历史递推 → A_eff 图传播`；输出形状不变。对 norm 后 v 定义尾随均值：

$$
\begin{aligned}
m_s(p)&=\frac{1}{\min(s,p+1)}\sum_{q=\max(0,p-s+1)}^p v_q,\\
b_1(p)&=v_p-m_2(p),\qquad b_2(p)=m_2(p)-m_4(p),\qquad b_3(p)=m_4(p).
\end{aligned}
$$

`s=2/4`；三分量之和为 v。用 `cumsum` 窗口相减实现，边界除实际 token 数，不做插值或下采样。时间状态递推使用逐通道可学习系数，初值 0.5：

$$
\begin{aligned}
\alpha&=\operatorname{sigmoid}(a),\\
q_{i,p}&=\alpha\odot q_{i,p-1}+(1-\alpha)\odot l_{i,p},\qquad q_{i,-1}=0,\\
h'&=x+\operatorname{Dropout}(W_t q),\\
h''&=h'+\operatorname{Dropout}\left(W_r\operatorname{GELU}\left(\sum_j A_{ij}\operatorname{RMSNorm}(h'_j)\right)\right),\\
h^{out}&=h''+\operatorname{Dropout}(\operatorname{FFN}(\operatorname{RMSNorm}(h''))).
\end{aligned}
$$

A 为现有 `A_eff`；按相同 p 混合 ROI。`W_t/W_r` 均为 `Linear(D,D)`。P 最大16，直接 Python 循环递推足够，不引入自定义 CUDA。图传播用 `einsum('bij,bjpd->bipd', A, n)`。`l` 由下述 Moga/LSK 联合局部分支产生。

计算为局部卷积与线性投影，加 `O(B P F² D)` 图传播；替换注意力不代表整体必然提速，CPM 仍可能占主导。该递推是新的有界混合设计，不是原 DDI 的无约束 MLP 递推。

### 4.2 LSK：逐 ROI/patch 选择时间范围

LSK 的关键是多种感受野加内容相关的选择；原模块使用级联大核/空洞卷积和空间选择。这里将几何轴改为历史 patch，并用三支并行因果核降低实现成本，属于结构改造。[原文 §3.2–3.3](https://arxiv.org/html/2303.09030v1#S3)

每个分量 `b_j` 走 `DWConv1d(D,D,k=3,dilation=1/2/4,groups=D,bias=False)`，分别对应三个尺度。统一左 padding=`2*dilation`；reshape 为 `[B*F,D,P]` 后计算，再恢复。每支输出 `u_j`。选择器与 Moga 共用：

$$
w_{i,p}=\operatorname{softmax}(W_s[b_1;b_2;b_3]_{i,p}+a_s).
$$

`W_s=Linear(3D,3)`，零初始化，初始均匀。局部核是3/5/9 patch的名义感受范围；与分解窗口叠加后覆盖更多历史，不能把它们简单称为固定 BOLD 生理频带。默认不让人口学或临床条件再进入选择器，避免重复注入。

### 4.3 Moga：多尺度值与门控相乘

Moga 原结构把不同局部范围的上下文聚合与 gate 分支相乘；不是新的预测器或专家路由。[原文 §4.2](https://arxiv.org/html/2211.03295v4#S4.SS2)

在上一节已有三个 `u_j` 上计算一次门控，不创建另一套尺度卷积：

$$
l= W_o\left(\operatorname{GELU}(W_g v)\odot 3\sum_{j=1}^3 w_j u_j\right).
$$

`W_g/W_o=Linear(D,D)`。系数3补偿均匀权重平均造成的幅度缩小；但卷积/gate本身仍非恒等，不声称整个 block 初始化等同旧 attention。`W_o` 正常 Xavier 初始化，新结构从头训练。LSK 的三支核、权重选择和 Moga 的 gate 都在一个 `CausalMultiScaleMixer` 中，避免重复局部算子。

### 4.4 StarBlock：脑状态 × 临床增量的交互

Star 的可借鉴点是两个线性分支逐元素乘法，在较小宽度中表达交互项；本方案把两分支分别指定为脑状态和临床条件。[原文 §3](https://arxiv.org/html/2403.19967v1#S3)

给 `ConditionEncoder.forward` 增加 `return_parts=False`，内部只执行一次 dropout/字段编码。默认返回原 Tensor；新候选请求字典 `{'c_demo','delta','c_full','new_mask'}`。HC/demo 的 delta 为零。临床适配器：

$$
\begin{aligned}
s_{i,p}&=\operatorname{GELU}(W_z\operatorname{RMSNorm}(T_{i,p}))\odot W_c\delta_c,\\
R_{i,p}&=W_{up}\left(g_{i,p}\odot s_{i,p}\right),\\
T^{clin}_{i,p}&=T_{i,p}+R_{i,p}.
\end{aligned}
$$

`W_z: D→64,bias=False`、`W_c: C→64,bias=False`、`W_up:64→D,bias=False`。`W_up` 零初始化，其余正常初始化；不同时把两个乘法因子都置零。MDD 新增字段全部缺失时，不等同健康：继续使用当前缺失编码，不强行令 delta 零。只有明确 demo 模式或显式“移除新增条件”的模型比较才用零 delta。

适配器共享于双头之前；两头不分别建立第二个临床网络。参数量约 `3DR+2CR+R+D`（含一处 D 维 RMSNorm 和下节 gate，按代码实际 bias 配置复核），不含原有字段编码器。以 D=256/C=32/R=64 为例约5.37万参数。

### 4.5 CGA：用当前内容决定临床修正强度

DEA-Net CGA 为不同通道产生内容相关空间权重，并用于不同特征融合。这里将“像素空间”改为 ROI/patch token，借鉴内容 gate，不复制二维卷积或去雾解码器。[原文 §III-B–C](https://arxiv.org/html/2301.04805v1#S3)

上一节 gate 定义为：

$$
g_{i,p}=\operatorname{sigmoid}(G_z\operatorname{RMSNorm}(T_{i,p})+G_c\delta_c+b_g).
$$

`G_z:D→R,bias=False`、`G_c:C→R,bias=False`；`b_g[R]` 初值零。gate 作用于 R 维乘性交互，不输出新的临床诊断分数。`delta_c=0` 时 `W_c delta_c=0`，因此严格零临床残差；该性质在 eval 下与 dropout 无关。

尺度权重、临床 gate 和修改量可以导出，但它们是模型内部使用量，不具备单凭数值大小证明致病机制的语义。

### 4.6 ContMix：由个体上下文生成图控制核

ContMix 原文由 token 与全局上下文中心的关系生成动态局部卷积核。本方案改成“历史状态与条件 → ROI 级图传播/保留/输入系数”，不在 ROI 编号上滑动卷积。[原文 §3.2](https://arxiv.org/html/2502.20087v2#S3.SS2)

新增 `GraphContMixResponse`；控制状态宽度 Q=32。先把 `T_clin_last` 与 `c_full` 转为每 ROI 的32维状态描述，拼接全 ROI 均值上下文，`Linear(64,64)→GELU→Linear(64,3)` 生成三个标量 `rho/eta/kappa`；输出层 weight/bias 零初始化，使初值分别为0.5/0.5/1：

$$
\begin{aligned}
h_i&=\operatorname{GELU}(V_z T^{clin}_{i,last}+V_c c_{full}),\\
(a_i,b_i,d_i)&=\operatorname{MLP}([h_i;\operatorname{mean}_j h_j]),\\
\rho_i&=\operatorname{sigmoid}(a_i),\quad\eta_i=\operatorname{sigmoid}(b_i),\quad\kappa_i=2\operatorname{sigmoid}(d_i),\\
r_{i,0}&=0,\\
r_{i,h}&=\rho_i\left((1-\eta_i)r_{i,h-1}+\eta_i\sum_j A_{ij}r_{j,h-1}\right)
       +\kappa_i\tanh(W_u U_{i,h}),\\
\Delta\hat x^{norm}_{i,h}&=W_y r_{i,h}.
\end{aligned}
$$

`V_z:D→32`、`V_c:C→32`；`W_u:1→32,bias=False`、`W_y:32→1,bias=False`。`W_y` 零初始化；其余 input/context 投影正常初始化，不零初始化 `W_u`。32维状态逐 TR 递推 H 次，不把 CPM 的整段未来 query 相互混合成控制传播。

图 A 在一个 horizon 内保持当前 `A_eff`，不额外重建图。响应允许正负，不宣称满足能量守恒或完整神经血管动力学。由于 tanh 输入及泄漏传播，单次驱动有界；同一时点的跨 ROI 扩散从下一步发生，这是明确的离散传播约定。

`U=future_control/control_scale`，`control_scale=1` 是默认无量纲模拟单位；真实实验须由调用方提供对应 ROI 的电场/驱动映射和采样间隔，不能把设备 mA、脉冲频率或药物类别直接当作 U。一次TR内的刺激聚合由数据层完成，模型不声称分辨TR以下神经响应。

双头集成：One-Step 的归一化完整预测加 `delta[:,:,0]`，CPM 完整预测加全部 delta。两头 baseline 可以不同，控制差值使用同一分支。先加完整状态再通过既有 RevIN denorm；原始空间增量用 `denorm(base+delta)-denorm(base)`，不可对 delta 单独加均值。

零 U 时响应严格为零，None 视为零控制；任何未来控制 `U_h` 仅影响 h 及以后。新候选不构建旧 `CPMHorizonHead.intervention_embed`。

**数据与训练集成的必要边界**：当前静息态窗口不给控制响应提供监督。默认静息态训练时冻结整个响应分支（不加入优化器及 AdamW weight decay），不伪造非零 U 并监督为原静息态 future。此时零初始化输出是接口的零响应，不能视作有用的干预预测。

响应样本使用独立的 `ControlWindowDataset` 接收调用方提供的 Tensor 文件：`x[N,F,1,K]`、`sc[N,F,F]`或公共`[F,F]`、条件列（沿用schema）、`future_control[N,F,H]`、`future[N,F,H]`、`future_mask[N,H]`、`subject_id[N]`。历史截止在第一未来控制前，future 对应实际受控观测；BOLD 预处理、ROI顺序和TR与候选一致。按被试划分，不用相邻窗口拆训练/验证。

`--twin_response_data` 指定此样本文件时，baseline网络及条件编码冻结，只有响应分支进入optimizer；仍使用现有 `TFMDualHeadLoss` 比较受控完整状态和真实future，one-step target=`future[:,:,0,None,None]`，不另引入响应差值标签。若只有治疗前后临床评分而没有逐TR响应，不能用此格式冒充短期BOLD监督；需另定义长期临床输出，超出本次默认结构。

响应文件不存在不阻碍代码交付；该分支可做形状/零控制检查，真实疗效识别仍需要实际刺激数据和对应时间尺度。

### 4.7 INP：ROI 级正常原型重建读出

原 INP 从单张图像自身动态提取正常原型并重建特征，依赖相应训练约束。不能从患者自己的全脑 token 自动推断哪些 ROI 正常。本方案借鉴“正常原型 → 重建 → 残差”，明确改用 **HC 训练集的固定观测描述量原型**；不是原 INP Extractor 的复现。[原文 §3.1–3.2](https://arxiv.org/html/2503.02424v1#S3)

新增 `HealthyReferenceReadout`，作为模型分析附件，不向主干回传梯度。对原始历史每ROI减时间均值、除以自身std，常量轨迹使用eps分母。算六维固定描述量 `d_i`：lag=1/2/4的Pearson相关、相邻差分均方、尾随2TR均值序列的方差比、尾随4TR均值序列的方差比。Pearson只在两个对应切片各自中心化后计算；常量分母加eps，不称为有意义相关。方差比的分母为标准化历史方差加eps。最低K=16，使用全可用历史，均值窗口左端除实际观测数。

`fit_reference` 只读取HC训练被试，每被试先平均其窗口描述量，避免长扫描被试占更多权重。逐ROI拟合6维均值/std，再对标准化被试描述量做4中心k-means（沿用 seed；k-means++初始化，最多20迭代；训练被试少于4则显式报错）。保存 `mean/std[F,6]` 和 `centers[F,4,6]` 为buffers，不需要新的可训练loss。默认参照context=64，推理参照读出同样要求K=64，其他K的BOLD预测仍可运行但不请求此读出，不机械建立多长度参照表。

对标准化描述量 v 与原型 p，温度 tau=1：

$$
\begin{aligned}
\omega_{i,m}&=\operatorname{softmax}_m\left(-\|v_i-p_{i,m}\|_2^2/\tau\right),\\
\hat v_i&=\sum_{m=1}^4\omega_{i,m}p_{i,m},\\
e_i&=v_i-\hat v_i,\qquad a_i=\frac{1}{6}\|e_i\|_2^2.
\end{aligned}
$$

返回六个分量偏离e和ROI距离a；HC描述标准化统计与原型持久化在单个参照文件中，同时保存ROI顺序、K、TR和预处理说明供调用方复用。固定观测描述量避免MDD更新编码器后HC token原型坐标漂移。参照拟合脚本使用现有dataloader的训练被试集合，不读取验证/测试参与拟合。

此读出与年龄/用药、扫描条件仍可能混杂，不叫疾病概率或病理因果归因。它不依赖把HAMD输入再作为预测标签，因此不会制造临床评分的标签泄漏。额外存储量为约24F个中心数加12F个统计数；对于116ROI约4176个数，计算很小。原始绝对BOLD幅度未保留在该描述量中，输出解释限于时间形态。

### 4.8 WConv：局部核的替换项，默认关闭

原方法固定密度乘到可学习卷积核上，“不增加参数”是相对已有卷积而言。[原文 §3.2](https://arxiv.org/html/2505.24558v1#S3.SS2)

若实施本替换，把默认 `CausalMultiScaleMixer` 的三支局部DWConv各自改为 `DensityWeightedCausalConv1d`，其他分解、LSK选择与Moga gate保持同一版本，不额外在patch之前增加滤波：

$$
u_{d,p}=\sum_{j=0}^2 \phi_j\theta_{d,j}\,b_{d,p-j\delta},\qquad \phi=(1,0.75,0.5).
$$

delta沿用1/2/4，phi为buffer，j=0表示当前。PyTorch核索引需反转为旧到新顺序，缺失位置只左补零。每支仍3D个权重，因此相对被替换DWConv参数量不变，新增核乘法可在eval预合并。单边密度是因果改造，不是原论文对称二维密度。非零phi可被无约束theta重新吸收，主要改变学习参数化而不扩大表达集合；故优先级低于条件/控制路径，不纳入默认实施。

### 4.9 DCTSA：频率描述替换项，默认关闭

原 FSTA-SNN 使用频率相关注意力，数据轴含spike/time/空间；这里只借鉴频率汇聚，不迁移LIF神经元或SNN训练。[原文方法](https://arxiv.org/html/2501.14744v1)

若实施，用prefix DCT描述替换 `Linear(3D,3)` 的尺度选择输入，三支值路径不变。在patch p只取v的前缀0..p，n=p+1；使用0..min(3,n-1)四个DCT-II系数，缺少的槽置零：

$$
c_k(p)=\sqrt{\frac{2-\mathbf{1}_{k=0}}{n}}\sum_{q=0}^{n-1}v_q\cos\left(\frac{\pi(q+1/2)k}{n}\right).
$$

用四个系数平方的 `log1p` 拼接成 `[B,F,P,4D]`，选择器 `Linear(4D,3)` 零初始化。P≤16时直接循环可执行，复杂度 `O(B F D P²)`；不对整段T做FFT后把结果广播到早期token，否则破坏局部时间因果性。频率随前缀长度变化，只解释为前缀模式，不能直接称为固定Hz频带。默认已有多尺度时域选择，无必要同时增加频域门控。

### 4.10 Agent Attention：CPM记忆压缩替换项，默认关闭

原Agent Attention用少量agent聚合K/V，再向query广播，是两次softmax注意力。[原文 §4.1](https://arxiv.org/html/2312.08874v3#S4.SS1)

若实施，仅替换CPM两层的 `_CrossAttention`：每层16个可学习agent `[M,D]`，用历史memory K/V聚合，再供ROI×horizon query访问：

$$
\begin{aligned}
V_A&=\operatorname{softmax}(Q_AK^\top/\sqrt{d_h})V,\\
O&=W_o\operatorname{softmax}(QK_A^\top/\sqrt{d_h})V_A.
\end{aligned}
$$

4 heads，`Q_A`与`K_A`分别为agent经过线性投影得到的query/key，广播key不依赖未来控制；memory始终只含历史。所有Q/K/V/输出投影尺寸沿用D，保留原CPMFFN/FiLM。显式标记这是agent式cross-attention改造，原论文还有agent bias/DWC，不将其省略后称完整原模块。跨ROI历史可以压缩，但不保证保留细微病理差异，默认先保留现有denseCPM。

注意力矩阵规模从 `F H × F P` 改为 `M × F P + F H × M`。F=116/P=16/H=8/M=16时每head约从172万元素变为4.45万，不含线性投影、激活或反向保存；这是计算估算，不是实测显存/速度收益。

## 5. 代码落点与兼容调整

默认路线只实现七个借鉴机制。WConv/DCTSA/AgentA保留本节规格，不为未实施替换项创建CLI、空类或兼容包装。

| 文件 | 必要改动 |
|---|---|
| 新增 `models/tfm_twin.py` | `CausalScaleDecomposition`、`CausalMultiScaleMixer`、`TwinMixerBlock`、`ClinicalStarAdapter`、`GraphContMixResponse`；每类职责对应本文公式 |
| 新增 `analysis/twin_reference.py` | 固定描述量、`HealthyReferenceReadout`、训练集参照拟合CLI；不新增可训练分类器 |
| 修改 `models/conditioning.py` | `return_parts`可选返回；一次dropout产生全部parts；保留默认Tensor接口 |
| 修改 `models/tfm.py` | 构造参数末尾添加 `structure_variant='tfm'`；`tfm_twin_v1`分支创建新blocks/共享适配器/控制响应；新增 `return_twin_info=False`；保留原输出键和布局 |
| 修改 `models/graph_prior.py` | 可选构造参数 `delta_init='legacy'`；新候选显式用 `one_factor` 初始化，旧模式保持旧权重重建行为 |
| 修改 `main.py` | `--structure_variant {tfm,tfm_twin_v1}`；新候选配置写入checkpoint；训练/评估调用传U；支持响应Tensor数据入口 |
| 修改 `utils/dataloader.py` | 仅新增上述 `ControlWindowDataset`；原静息态dataset不制造控制量或后治疗label |
| 修改 `train/optim.py` | 新适配器使用 `is_conditioning_adapter=True`；控制分支单独分组并依dataset模式冻结/启用；参照buffer不进optimizer |
| `train/losses.py` | 保留现有 `TFMDualHeadLoss`；控制样本仍比较完整预测，不引入另一个损失框架 |
| 新增 `analysis/twin_compare.py` | 同一输入缓存编码器/条件，复用baseline计算多个U；输出baseline、受控轨迹、差值、ROI/分量参照偏离；不输出“疗效概率” |
| README | 新候选入口及参照/响应数据说明；不重命名当前final model |

`_encode` 的现有 `(tokens,sc_info)`返回不变；收集scale信息通过仅新候选调用的helper返回额外列表，常规预测不收集大张量。新block提供 `forward(x,adj,cond_mods=None)`签名；新候选不传norm调制，旧block完全沿用。

结构参数固定在 `structure_variant` 分支内；仅暴露真正需要的控制数据/尺度/参照路径。默认参数由checkpoint保存：variant、patch/D/layers、sc delta_init、C/R/Q、三支dilation、control_scale、horizon。旧配置缺variant时按tfm重建。保存/恢复代码沿用现有config/state_dict机制，不扩展通用插件注册系统。

新结构不能strict加载旧attention权重。默认新HC随机初始化，MDD加载同结构新HC权重，继承已有共享条件统计/编码逻辑；响应训练加载该候选baseline再只训练控制分支。`configs/final_model.json`和定型 `inference.py`继续属于旧模型，不向其中添加新候选或复用旧指标。

新候选使用 `--tfm_patho_mode film` 以保留双头人口学FiLM；构造时固定 `cond_backbone_norm=False`、`cond_gate_residual=False`、`cpm_intervention=False`。用户显式提供冲突开关时直接说明冲突，而非静默叠加或自动fallback。对应flag实际拼写沿当前CLI，新增入口不要复制一套配置来源。

参照与比较脚本加载新候选checkpoint需复用main保存的结构/condition config建立 `NeuroTwinTFM`；采用普通严格state_dict恢复，不修改定型推理的final-only约束。U=None对静息态预测兼容；传非零U只允许新候选已构建响应分支。

## 6. 可以直接执行的实施顺序

1. **实现核心结构**：一次完成 `tfm_twin.py` 的时间/图block、临床适配器、控制分支；加入ConditionEncoder parts与单因子图初始化。四个构件按上面定义集成到 `NeuroTwinTFM`，不是十个独立研发阶段。
2. **接通入口和状态恢复**：main构造分支、checkpoint config、适配器/响应optimizer分组；旧模型保持默认variant；静息态训练冻结响应。新增control Tensor dataset并接通model的U参数。依赖第一步的接口，不依赖先训完模型。
3. **补齐分析读出**：固定描述量和HC原型拟合、同输入的U比较脚本、README。图和token导出复用现有aux，不建立新报告平台。
4. **做一次基础检查并交付代码**：下节所列小Tensor检查。训练、参照真实拟合、消融和临床结果都不是代码实施完成前置条件。

实施伪代码：

```python
x_grid, norm_stats = normalize_history(x)
parts = condition_encoder(..., return_parts=True)  # single dropout call
tokens, sc_info, scale_info = encode_twin(x_grid, sc_matrix, diagnostics)
c_demo, delta_c, c_full = parts['c_demo'], parts['delta'], parts['c_full']
t_clin = tokens + clinical_adapter(tokens, delta_c)
base_one = x_grid[:, :, -1] + one_step_head(t_clin[:, :, -1], A, c_demo)[0]
base_cpm = cpm_head(t_clin, roi_embed, c_demo, future_control=None)
delta = control_response(t_clin[:, :, -1], A, c_full, U)
pred = denorm_state(base_one + delta[:, :, 0], norm_stats)
cpm_pred = denorm_state(base_cpm + delta, norm_stats)
# baseline and controlled state use identical RevIN stats
# raw-space response = controlled prediction - baseline prediction
return existing_layout(pred), existing_aux_plus_twin_info(...)
```

HC condition为demo，因此delta_c为零；若condition_arm=none，创建零宽度条件投影会改变上述规格，新候选不支持该配置。本路线要求已有hc3/demo或mdd8/cat条件接口，避免额外设计无条件候选。

## 7. 最小必要检查与后续实验边界

代码实施后运行语法/导入检查以及一个CPU合成前向脚本即可，建议B=2/F=116/K=16、19、64/H=8。真实Tensor、实际模块，不用mock替代模型计算。

- shape：One-Step、CPM、控制差值、尺度权重与本文一致；K=19确认原有左补零仍工作。
- 运算正确性：三分量求和等于v；同一A固定时扰动未来patch不会影响混合器前缀；零delta_c导致临床残差为零；零U导致响应为零；更改U的第h步不会影响之前控制增量。
- 一次backward：随机有监督目标时，单因子图初始化的dq、临床W_up以及响应W_y有可用梯度；零初始化读出允许上游首步梯度为零，不要求每个参数首步非零。响应梯度检查使用显式合成非零U，只验证计算通路，不将其当作训练证据。
- 兼容：旧variant仍返回旧布局；新候选checkpoint保存/strict恢复一次；train模式确认响应冻结，response模式确认只有响应参数进入optimizer。
- 参照：合成HC描述量拟合四中心，返回 `[B,F,6]`；常量轨迹结果有限。该检查不拟合真实HC，也不证明临床区分能力。

上述“前缀因果”只指新混合器局部计算。现有RevIN统计和 `A_eff` 使用完整历史context，早期token在完整模型中并不具备原始输入逐前缀隔离性质。完整模型只保证不使用预测截止时点以后的真实BOLD；不为内部prefix测试重写整个归一化和图生成。

后续实验只需围绕三个问题展开，不占据实施主体：临床增量是否改善个体预测（可比较同人相同history的demo/full）；健康参照偏离是否与独立临床/随访标签关联；有实际控制数据时响应差值是否匹配真实刺激后变化。默认候选是新的组合假设，各模块独立贡献尚未确认，WConv/DCTSA/AgentA不能与未验证改动一起宣称最终最优模型。

完成代码后可称“具备个体状态、健康参照和控制响应接口的候选模型”。没有真实控制响应训练时，只能运行零响应初始化或明确标注的结构模拟，不能称为已验证个体疗效预测器；改变HAMD输入的差值也只是条件敏感性，不是治疗反事实。

## 8. 本文交付核对

本次只新增本文。已核对源文档十项编号、当前工作区入口与核心接口，并在线读取对应原论文方法；原论文视觉/一般时序结论不作为NeuroTwin实测结果。11个显示公式已通过KaTeX 0.16.22逐项解析，并在无界面Edge中生成预览，渲染错误数为0；已查看首屏公式排版。未声称整个Codex Markdown页面的布局都已人工查看。本文中的参数、公式和文件任务为待实施规格；没有修改模型、执行训练、拟合真实健康参照或运行模型前向测试。
