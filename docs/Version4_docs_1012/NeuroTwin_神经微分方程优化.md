# NeuroTwin 神经微分方程优化、病理分析与个体化孪生：合并实施方案

> 合并日期：2026-10-11。本文合并原“神经微分方程优化”与“病理分析／个体化孪生／干预优化”两份方案，作为这条研究主线唯一保留的实施文档。
> 交付性质：待实施的技术规格；下文新增模型、接口、CLI、训练与分析入口均未因本次文档合并而实现，不表示已有训练或疗效结果。
> 默认路线：保留现有 TFM 编码器与字段条件编码，以条件图 ODE 统一状态预测和控制响应，增加固定基线个体校准码、健康参照读出、观测更新及轻量概率/临床读出。
> 完成标准：下述代码与接口接通，基础计算检查通过即可交付；不以训练、消融、性能提升或真实疗效验证作为实施前置条件。

## 1. 当前问题、改进目标与明确取舍

以当前工作区为代码基线，而不是 HEAD 的旧单 HAMD 版本。本文覆盖上一轮分析中的病理参照、个体校准、受控状态推进、临床读出和不确定性；不把所有未来研究方向都变成新增模块。

| 当前实现事实 | 本次确定的实现 |
|---|---|
| `models/tfm.py` 的 One-Step 输出下一 BOLD；rollout 把预测回填历史后反复编码 | 历史初始化后持续推进同一 latent/BOLD 状态，采用固定 Heun RK2 |
| `future_control` 只进入 CPM query，训练没有控制输入 | 新增动态 `controls`，直接进入向量场；Dataset、训练、滚动与推理使用同一时序 |
| 八字段 `ConditionEncoder` 已存在，neurotwin_tfm_pretrain / neurotwin_tfm_finetune 是当前选定入口 | 复用该编码器，条件只准备一次；不重新比较融合方式，不新增临床字段 |
| 因人而异的输入不等于独立个体模型 | 从个人固定基线得到 profile，持续预测期间固定；新观测更新状态 |
| 缺少健康偏离的明确输出 | 对固定 BOLD 动态描述量建立 HC 人口学岭回归参照，输出逐 ROI、逐分量偏离 |
| 缺少症状变化读出 | 冻结动态模型，独立训练一个两输出临床变化读出，不向每个 TR 重复贴随访标签 |
| BOLD 预测主要是点值 | 增加逐 ROI/horizon 对角方差读出；不新增 SDE、采样器或贝叶斯过滤器 |
| `dp/dq` 同时零初始化，乘积残差无法启动学习 | dp Xavier、dq 零初始化，保留初始零残差及参数键 |
| loader 默认读取 `Mask_<id>.mat`；`data/utils/mask.py` 从 FC 做 Otsu 二值化，`correlation.py` 使用前 150 TR | 新路线采用 Raw 历史校准与可见历史 FC；该图称功能耦合图，不默认称解剖 SC |

上述最后一项是对此前“Mask 就是独立解剖 SC”的纠正：现有生成脚本不支持这个前提。本次不做全数据审计，不改原始数据，不覆盖旧缓存。

**默认不实施**：Loop 主干、视觉模块混合器、MoE/RECC、神经质量模型/血流动力学方程、跨日慢状态 ODE、刺激方案搜索。它们不是本路线的依赖。TR 级状态推进与随访天数由不同接口表达，不把数周间隔塞进 TR 级积分器。

保留原图动力学方案的组合 Laplacian、条件 FiLM、扩散率、BOLD 速率读出、同步 Heun RK2、初始化与梯度路径、持续状态和训练集成的详细方法，并在同一个模型中加入 profile、状态相关控制门、方差、健康参照和临床读出。不另建第二个 ODE 头或第二套状态接口；不叠加 [十模块方案](NeuroTwin_十模块借鉴与结构优化_Codex实施方案.md) 的替换主干和第二套控制核。

### 1.1 合并后统一的设计取舍

| 两份原方案的差异 | 本文唯一采用的定义 | 原因 |
|---|---|---|
| Raw 校准使用 context_min 或固定 32 TR | 固定 `calibration_length=32`，Raw 标准化与 profile 用同一前缀 | 个人基线需要独立于当前 context 长度；同步调整合法锚点 |
| 无 profile 或固定基线 profile | 新训练显式 `personalized=True`、32 维 profile；旧构造默认 False | 在共享动力学上表达个体基线，不引入 ID embedding 或逐人参数 |
| `G u` 或状态相关控制门 | 同一向量场内使用 `s(Z,c,p) * G u` | 统一控制与状态演化，不叠加 ContMix 或 CPM 控制嵌入 |
| 动力学 head 返回两项或三项 | 返回末状态、BOLD 轨迹、latent 轨迹三项 | 同一个 head 的 latent 轨迹服务方差读出，不另建积分器 |
| 共享一个 future mask 或分开监督 | dynamics_mask 与 cpm_mask 分开 | 无控制 CPM 不能被迫预测未知驱动的受控未来 |
| 图 ODE 起点目录或 twin 起点目录 | `hc_demo_twin` / `mdd_cat_twin` | 一个新架构家族，不混用历史或另一命名家族权重 |
| 只有状态预测或额外临床分析 | 保留健康参照、方差、独立随访读出 | 扩展目标读出，不改变 next-TR 样本的标签单位 |

本文不是两份方案的串接：动力学 head、数据分支、loss、CLI 与权重重建只实现一次。保留原 TFM 四层 attention 主干；Loop 不是本次依赖，其内部计算轮次不得充当未来 TR 或 RK2 子步。

## 2. 默认结构、数据尺度与参数

```text
Raw BOLD ──前 32 TR 校准──→ 数据尺度 X
    ├──固定个人基线 X_cal──→ 共享 TFM 编码器 → ProfileEncoder → p
    └──当前历史 X_ctx──────→ context RevIN → history FC → 原图先验
                              → 原 patch / TFM blocks → Z0、A
八字段条件 ──原 ConditionEncoder──→ c
Z0、当前 BOLD q0、A、c、p、ROI 控制 U
    → 条件图向量场 → 固定 RK2 同步推进 Z、q → 单步/动态轨迹
    ├──VarianceHead(Z.detach()) → BOLD 预测标准差
    └──原 CPM → 无控制辅助轨迹

固定 64 TR 历史 → 6 个动态描述量 → HC 岭回归参照 → ROI 偏离
p、c、描述量、偏离、早期模拟变化、控制摘要、随访天数
    → 独立 ClinicalOutcomeHead → HAMD/HAMA 变化及标准差
```

符号：B 为批大小，F 为 ROI 数，K 为当前历史长度，P 为 patch 数，D 为 latent 宽度，H 为预测间隔数，C 为临床条件宽度，J 为参照描述量数。

| 配置 | 默认值 | 落点 |
|---|---|---|
| F / K / patch / D | 116 / 16–64 / 4 / 256 | 现有模型参数，不改 attention 主干 |
| TFM layers / heads / FFN | 4 / 4 / 4 | 沿用当前脚本 |
| C / 条件注入 | 32 / heads | HC `hc3/demo`，MDD `mdd8/cat` |
| `dynamics_mode` | 新脚本 `graph_ode` | 构造默认仍为 `discrete`，保留旧调用 |
| `ode_steps` / `dynamics_dt` | 2 / 1.0 | 每个采样间隔 2 子步；dt 单位是 TR |
| H / 动力学 hidden | 8 / 256 | 复用 `cpm_horizon` / `tfm_one_step_hidden_dim`；动态 H 不另增 CLI，单步接口设 H=1 |
| `personalized` / `profile_dim` | 新脚本 True / 32 | 旧构造默认 False；profile MLP hidden=64 |
| `calibration_length` | 32 | 个人基线、Raw 均值/尺度使用同一前缀 |
| 残差强度 / 扩散初值 | 0.1 / 0.05 | 不新增可学习积分步长 |
| `lambda_dynamics` / `lambda_variance` | 0.5 / 0.05 | 多步监督与误差方差监督 |
| `graph_source` / `bold_source` | 新脚本 `history_fc` / `raw` | 旧默认 `legacy_mask` / `auto` |
| CPM control / 向量场 Dropout | False / 0 | 单一控制路径；编码器原 Dropout 保留 |
| 参照 K / J / ridge | 64 / 6 / 1.0 | 固定尺度分析，单个参照文件 |
| outcome MLP / outputs | 83→64→4 / 两个均值、两个 logvar | 单独保存读出权重和输入统计量 |

这些数值是确定的工程起点，不是实验最优结论。其余优化器、学习率、batch、seed 沿用当前配置。

### 2.1 状态定义、归一化与时间单位

模型内部 q 是 `BrainRevIN` 仿射归一化后的 BOLD；对外预测和 observation 使用 loader 的历史校准尺度 X。Z/p 是编码表征，不直接命名为兴奋性、病理负荷或生理参数。Raw 指现有 ROI 序列来源，不声称是未经预处理的 MRI 原始强度。

记当前历史编码器为 E、context-only RevIN 为 N；历史 X 已经过固定前缀的数据层校准。初始 latent 与 BOLD 状态为：

$$
Z_0=E(N(X))_{\mathrm{last}},\qquad q_0=N(X)_{\mathrm{last}}.
$$

直接取最后 patch token 作为 `[B,F,D]` 的 Z0，不新增状态投影。反归一化复用 [BrainRevIN](../models/common.py) 的统计量及可学习 affine 权重/偏置；不能每个预测子步重新估计统计量，也不能漏掉 affine。`norm=False` 时 q 使用数据输入尺度。

每个外部预测间隔默认 dt=1，代表一个采样间隔/TR；没有真实 TR 元信息时不自动解释为 1 秒或 2 秒。未来传入其他 dt 时必须使用与训练一致的采样步单位；真实秒值须由调用方按该扫描的 TR 换算。跨日随访通过独立 outcome 输入表达，不传入 TR 级积分器。

向量场内部不使用 Dropout、DropPath 或随机跳子步，编码器与 CPM 的现有 Dropout 保留。一个无观测预测区间内图、临床条件和 profile 固定；真观测更新时重编码当前历史、重建图，保留既有 profile。

## 3. 输入与输出契约

| 张量/数据 | 形状 | 明确语义 |
|---|---|---|
| `x` | `[B,F,1,K]` | 当前历史，沿用内部布局 |
| `calibration_bold` | `[B,F,1,32]` | 同一扫描最早 32 TR，不能取当前锚点之后的数据 |
| `condition` | 原 `cond_z/cond_code/cond_mask` 字典 | 复用现有字段、mask、train-only 统计量 |
| `profile` | `[B,32]` | 推理时可传已计算的个人基线码 |
| `controls` | `[B,H,F]` | 第 h 个区间内逐 ROI 有效驱动，控制区间是 `(t+h-1,t+h]` |
| `time_deltas` | `[H]` 或 `[B,H]` | 正的采样间隔长度；None 为全部 dt=1 |
| `future/future_mask` | `[B,F,H]` / `[B,H]` | 现有动态轨迹真值与可用 horizon |

保留 `forward -> (pred, aux_info)`，pred 为 `[B,F,1,1]`。新增 aux 键：

- `dynamics_pred`、`dynamics_std`：`[B,F,H]`，数据尺度。
- `dynamics_logvar_norm`、`dynamics_mean_norm`：`[B,F,H]`，仅供方差损失。
- `norm_stats`：当前历史的 RevIN 统计量；用于目标归一化，不写成模型参数。
- `personal_profile`：`[B,32]`；原 `cpm_pred/sc_prior/pathology_cond` 保留。

`normal_residual/normal_distance` 在分析层计算；临床读出是独立模块，不在普通 next-TR forward 强行生成临床结局。缺少临床读出文件时 API 不返回随机评分。

原 `future_control [B,F,H]` 继续只属于 legacy CPM；不偷偷转置成动态 controls。新脚本禁用 `cpm_intervention`，新的双方案模拟只使用受控动态轨迹。

## 4. 数据与历史图：直接修正来源，不增加审核流程

### 4.1 Raw 分支与锚点

在 `NeuroTwinNextPointDataset/NeuroTwinDataLoader` 增加 `num_rois`、`graph_source`、`calibration_length`、可选 `control_manifest`，并扩展 `bold_source='raw'`。

raw 分支读取 `Raw/HC|MDD/ROISignals_<id>.mat`，键复用 `ROISignals/Data`；按该来源的明确 `[T,F]` 布局取前 num_rois 列（完整时间轴，不截断），再转为 `[F,T]`。不使用 `max(shape)` 猜时间轴；raw 缺失直接报告路径，不回退 Normlize 或窗口缓存。

令 X_raw 为 Raw 中提取的 ROI 序列，K0=32；每个 ROI 的校准统计只使用最早 K0 个观测：

$$
\mu_i^0=\operatorname{mean}(X^{\mathrm{raw}}_{i,1:K_0}),\qquad s_i^0=\sqrt{\operatorname{var}(X^{\mathrm{raw}}_{i,1:K_0})+10^{-6}}.
$$

$$
X_{it}=(X^{\mathrm{raw}}_{it}-\mu_i^0)/s_i^0.
$$

0 基锚点下界改成 `max(context_len-1, calibration_length-1)`，上界仍沿用已有目标可用性计算。同步修改 `legal_anchor_range/legal_context_lengths/legal_task_count`，不能只在 make_sample 截取前缀。首个预测至少从第 33 个 TR 开始，未来样本不参与基线/profile 校准。

history_fc 分支从 Raw 文件枚举被试，不要求 Mask 存在；`batch['sc']` 保留单位图占位以兼容现有 batch 拆包，模型不会把占位当实际图。只有 legacy_mask 分支继续从 Mask 推断/读取现有外部图。`_probe_series_length` 从 Raw 的明确时间轴取长度，不能沿用 `max(shape)`；ROI 数由 `num_rois` 参数提供，不依赖单位图占位或 Mask 推断。`_build_index` 也应按校准长度和可用未来目标判断是否有合法锚点；`legal_context_lengths/legal_anchor_range/legal_task_count` 使用同一规则，避免保留无法采样的短序列。原始文件、标准化缓存和旧脚本均不删除。

### 4.2 可见历史 FC 与图残差

从当前可见 x_grid 沿 K 维去均值，得到 r。构造相关矩阵 C_hist 和非负外部图：

$$
C^{\mathrm{hist}}_{ij}=\frac{\sum_k r_{ik}r_{jk}}{\sqrt{\sum_k r_{ik}^2+10^{-6}}\sqrt{\sum_k r_{jk}^2+10^{-6}}},\qquad A^{\mathrm{hist}}_{ij}=|C^{\mathrm{hist}}_{ij}|\mathbf{1}[i\ne j].
$$

将 A_hist 作为已有 `SoftAnatomicalPrior` 的输入，复用其混合、self-loop 和 Sinkhorn；不额外阈值化，不新增图网络。后文 A 是其 A_eff 再对称化的结果。在预测区间固定 A，新观测更新时重新计算。

`graph_prior.py` 的 dp 权重 Xavier、dq 权重零、双方 bias 零；delta_scale 保留 raw=0 的初始化及当前 softplus 用法。不重置从 checkpoint 加载的权重。该修正保留原 state_dict 键，不创建旧权重修复工具。

### 4.3 控制文件

`control_manifest` 为 CSV，两列 `subject_id,controls_path`，相对路径相对 manifest 目录。每个 NPZ 只有必需数组 `u [T,F]`，单位为上游定义的有效驱动；`u[t]` 对应到达观测 t 的区间，因此锚点 t 的未来控制切片是 `u[t+1:t+H+1]`。缺尾区间随 future_mask 填零。

manifest 未提供时不产生伪刺激标签，模型收到 controls=None。提供 manifest 的被试条目必须有文件、匹配 T/F 且有限；错误直接报告，不用无刺激代替缺文件。控制幅值不可用 BOLD 的均值/方差标准化。

本次不内置 Hz/mA、电极位置或 TMS 电场到 u 的物理换算，也不把 medication 类别当瞬时驱动。外部生成的 u 可直接进入接口。没有响应数据时控制映射仍可执行，但输出只能称未校准的模拟响应。

## 5. 个体基线码：固定 profile，动态 state

新增 `models/tfm_personal.py::ProfileEncoder`。基线只走共享 TFM 编码器，不走当前临床 FiLM/AdaRMSNorm；临床影响留给向量场 c，避免把同一份当前评分再编码进 profile。

基线 tokens 为 T_cal `[B,F,P0,D]`。固定使用 ROI/patch 双均值池化和两层 MLP，不引入 ID embedding 或逐人训练参数：

$$
b=\frac{1}{FP_0}\sum_{i,p}T^{\mathrm{cal}}_{ip},\qquad p=\tanh(W_{p2}\operatorname{GELU}(W_{p1}\operatorname{LayerNorm}(b)+b_{p1})+b_{p2}).
$$

hidden=64、profile_dim=32；两层正常随机初始化。p 是当前基线的个体校准码，跨访视稳定性是待验证性质，不能直接称终生稳定特征。

训练 forward 在没有显式 profile 时用 calibration_bold 计算 p，并保持梯度；不得跨 optimizer step 缓存 detached tokens。当前历史与基线因此通常各编码一次。推理用 `calibrate_profile` 每个人计算一次，后续 rollout/observation update 复用 p。

`personalized=False` 时不创建该模块；新 graph_ode 脚本 True。缺 calibration_bold 且缺 profile 的 personalized 调用直接报必需输入缺失，不偷偷从包含未来的全序列计算。

## 6. 受控图动力学、同步 RK2 与概率读出

### 6.1 向量场

新增 `models/tfm_dynamics.py::GraphODEDynamicsHead`，在 graph_ode 模式替换 `one_step_head`，不包装旧头的 residual 输出冒充微分方程。旧 `OneStepDynamicsHead` 只在 discrete 分支实例化，新 head 只在 graph_ode 分支实例化，不构建未使用的第二套参数。

#### 6.1.1 图扩散结构

编码器使用 `SoftAnatomicalPrior` 返回的 A_eff；动态头取其 `(A_eff + A_eff.transpose(-1,-2))/2` 作为非负对称图 A。同一次历史初始化的图同时服务编码和动力学，动态头只做上述对称化，不另学一套图。定义逐 ROI 的度及组合 Laplacian：

$$
d_i=\sum_j A_{ij},\qquad L=\operatorname{diag}(d)-A.
$$

使用组合 Laplacian，不能因有限次 Sinkhorn 近似行归一就直接替换为 `I-A`。实现不显式构造 L，按 `degree.unsqueeze(-1) * z - torch.bmm(adj, z)` 计算 LZ。该项的图传播复杂度为每次向量场求值 `O(B F² D)`，F=116 时直接沿用稠密图运算，不新增图算子框架。A/L 是功能耦合约束，不称为已知解剖动力学。

初始化得到 Z0 为当前最后 token `[B,F,D]`，q0 为当前最后归一化 BOLD `[B,F]`。令 V=LayerNorm(Z)，d_i 为 A 的行和，L=diag(d)-A。图扩散使用组合 Laplacian：

$$
(LZ)_i=d_iZ_i-\sum_j A_{ij}Z_j.
$$

#### 6.1.2 条件化神经残差与扩散率

先对 Z 做共享 `LayerNorm(D)` 得到 V。`W1=Linear(2D,hidden)` 将 `[V,A@V]` 拼接映射到 hidden，默认 hidden=256；`W2=Linear(hidden,D)` 输出 latent 残差。临床向量 c 复用 `_prepare_condition`，一次字段编码/dropout 后由动态头与 CPM 使用，不重复读表。

$$
V=\operatorname{LayerNorm}(Z),\qquad a=W_1[V,AV]+b_1.
$$

由原 `ConditionFiLM(C,hidden)` 得到 gamma_c、beta_c；新增 `PersonalFiLM(32,hidden)` 得到 gamma_p、beta_p。FiLM 输出 `[B,hidden]`，沿 ROI 广播为 `[B,1,hidden]`：

$$
R=W_2\operatorname{GELU}((1+\gamma_c(c)+\gamma_p(p))\odot a+\beta_c(c)+\beta_p(p))+b_2.
$$

扩散率为逐样本标量，无条件项使用零增量：

$$
\kappa(c,p)=\operatorname{softplus}(\kappa_0+w_c^{\top}c+w_p^{\top}p).
$$

`kappa_raw` 是逐模型共享标量，初始化 `log(expm1(0.05))`。`kappa_cond=Linear(C,1,bias=False)` 与 `kappa_personal=Linear(profile_dim,1,bias=False)` 权重零初始化，分别实现上式 w_c 与 w_p，标记 `is_conditioning_adapter=True`。共享扩散基准已由 kappa_raw 表示，本合并版本不重复增加常数扩散 bias。

扩散率为逐样本标量 `[B,1]`，沿 ROI/latent 通道广播；不做逐 ROI 任意扩散矩阵，以保留简单对称图扩散项的耗散含义。条件通路未构建时不创建 `film/kappa_cond`，使用恒等 FiLM 与零条件扩散增量；`personalized=False` 时不创建 `PersonalFiLM/kappa_personal`。完整缺失字段由原条件编码器表达，不与未构建条件通路混为一谈。

#### 6.1.3 控制输入与连续 BOLD 读出

单区间 u 为 `[B,F,1]`，临床 c 与 profile p 在 ROI 维广播。令 s_i 是当前 ROI 状态、全局临床和个体码拼接后的 sigmoid 门，`W_s=Linear(D+C+profile_dim,1)`（未启用的条件/profile 维不构建）；G 为无 bias 的 `Linear(1,D)`，其投影权重跨 ROI 共享：

$$
s_i=\operatorname{sigmoid}(W_s[V_i,c,p]+b_s),\qquad f(Z,A,c,p,u)=-\kappa(c,p)LZ+0.1R+s\odot Gu.
$$

$$
\frac{dZ}{dt}=f(Z,A,c,p,u),\qquad \frac{dq}{dt}=v(Z)=W_yZ+b_y.
$$

**同步积分 Z 和 q，而不是积分 Z 后任意叠加一个解码值。** `decode_rate=Linear(D,1)` 给出 `[B,F]` BOLD 速率；每个区间的预测增量来自该速率的积分。控制改变 latent，并通过同一速率读出影响当前及后续观测；零 u 的附加项严格为零。门使相同驱动随当前状态/profile 变化，不另建专家路由或控制残差头，也不把 HAMD 或 medication 类别当作瞬时刺激。

#### 6.1.4 初始化、梯度路径与数值边界

初始化：W1 Xavier、W2 正态 std=0.01、各 bias 零；两类 FiLM 和 w_c/w_p 零初始化；kappa0=`log(expm1(0.05))`。G 正态 std=0.01、无 bias；W_s 正常随机、b_s=0。W_y/b_y 零初始化，初始 BOLD 为 persistence，首步可学习读出，后续才向动力学传播梯度。不要把 W2、G 和乘法门也一起清零。

初始化的计算含义：

- 零初始化 rate decoder 时 q 初始不变，预测为 persistence；decoder 输入 Z 非零，首步可以得到梯度，decoder 更新后才向动力学和编码器传播均值损失梯度。这是短暂梯度延迟，不是永久死分支。
- FiLM 和条件扩散增量零初始化，使临床/profile 调制初始不改基准动力学。残差 W2、G 和控制门不能全部一起清零。
- 图残差 dp 非零、dq 零时初始残差仍为零，dq 首步可学习；不重新初始化加载 checkpoint 的实际参数。
- 没有非零控制及对应受控观测时，数据不能识别 G 和控制门的响应映射。静息态损失不能被表述为已训练疗效模型；不伪造刺激标签。使用未校准控制参数的模拟必须标注该状态。

整个向量场不加 Dropout。c/p 固定于一次推进；不同条件分析重新从同一观测初始化，不将上一条件预测串入下一条件。图扩散项的耗散性不等于完整非线性向量场稳定；神经残差与学习后的扩散率仍可能导致固定步 RK2 发散。不加静默轨迹裁剪、自适应求解或假预测 fallback，稳定性由后续实际训练与必要的数值比较判断。

### 6.2 固定 Heun RK2

每个区间 dt>0，N=`ode_steps`（默认 2），h=dt/N；u 在该区间固定。对每个子步同步推进 Z/q；第二 stage 在预测 latent Z* 上重新计算向量场，状态相关控制门也重新计算，A/c/p/u 仍固定：

$$
\begin{aligned}
k_1&=f(Z,A,c,p,u), & v_1&=v(Z), \\
Z^*&=Z+h k_1, & k_2&=f(Z^*,A,c,p,u), \\
v_2&=v(Z^*), \\
Z_{\mathrm{new}}&=Z+\tfrac{h}{2}(k_1+k_2), & q_{\mathrm{new}}&=q+\tfrac{h}{2}(v_1+v_2).
\end{aligned}
$$

dt 扩展为 `[B]`；Z 更新广播 h `[B,1,1]`，q 更新广播 `[B,1]`。积分在 autocast 关闭的 float32 中执行，TFM/CPM 继续使用原 AMP。用普通循环与 Tensor，不安装 ODE 库、不加 solver registry 或自适应步长。

`head.forward(z0,q0,adj,cond=None,profile=None,controls=None,time_deltas=None,horizon=8)` 返回 `z_final [B,F,D]`、`q_trajectory [B,F,H]`、`z_trajectory [B,H,F,D]`。它逐间隔调用 `integrate_interval`，不包含 TFM 编码或 RevIN；controls/time_deltas 的布局见 §3。默认 H=8 的 latent 轨迹用于方差读出；预测期间不 detach Z/q，不启用额外 gradient checkpointing。

同步积分伪代码（非待调用的现有代码）：

```python
# 输入 z:[B,F,D], q:[B,F], dt:[B]；A/c/p/u 在本区间固定。
# integrate_interval 内关闭 autocast，输入和动态头运算使用 float32。
h = dt / ode_steps
hz = h[:, None, None]
hq = h[:, None]
for _ in range(ode_steps):
    k1 = field(z, adj, cond, profile, u)
    v1 = decode_rate(z).squeeze(-1)
    z_stage = z + hz * k1
    k2 = field(z_stage, adj, cond, profile, u)
    v2 = decode_rate(z_stage).squeeze(-1)
    z = z + 0.5 * hz * (k1 + k2)
    q = q + 0.5 * hq * (v1 + v2)
return z, q
```

重复 N 次得到一个预测间隔末状态，再持续进入下一间隔。两 stage 使用相同 h，不另设可学习 solver step scale 或 block residual gate。q 的导数仅依赖 Z，因此无需构造 q 的 predictor。H=8/N=2 时共求值向量场 32 次，这是运算次数估算，不是实测速度或显存结论。

将 Z/q/A/c/p/u 转 float32 后运算，编码器和 CPM 继续使用 AMP。非正时间间隔、N<1、已知形状错误直接报上下文，不引入 solver 库、registry、自适应步长、跳步或静默替代输出。

### 6.3 不重复编码的状态接口

状态是普通 dict：`z,q,adj,cond,profile,norm_stats,history`。history 是当前数据尺度 `[B,K,F]` 的 buffer；q 是 RevIN 空间 `[B,F]`。推进返回新 dict，保留反向图，不对 z/q detach 或原位覆盖；不保存 tensor 文件来代替模型权重，不建设状态框架。

| 方法 | 行为 |
|---|---|
| `calibrate_profile(calibration_bold)` | `[B,F,1,32]` → `[B,32]`，基线编码一次 |
| `initialize_state(history, sc_matrix=None, pathology_score=None, condition=None, profile=None, calibration_bold=None)` | 当前历史 `[B,K,F]` → state；history_fc 无需外部 SC；legacy_mask 必须提供外部图，字段与 legacy 条件不得同时传 |
| `advance_state(state, control=None, dt=None)` | 控制 `[B,F,1]`；积分一段，更新 q/z/history；图、p、RevIN stats 固定 |
| `readout_state(state)` | q 反归一化 → `[B,F]`，不再次编码 |
| `update_state(state, observation)` | 用对齐当前预测时点的真实 `[B,F]` 替换 history 末帧；重新估计当前 RevIN/图/Z/q；保留 p 和已经编码的 c |
| `simulate_intervention(history, controls, ...)` | 一次初始化，复制两份状态，分别用 U 和零 U 推进，输出两轨迹及差值 |

`update_state` 是观测重编码更新，不是 Bayesian 同化；首版只支持完整 ROI 观测。它调用内部“已准备 cond/profile 的初始化 helper”，不把 c 当原始字段重新编码，也不再次计算 profile。

核心 API 使用数据尺度 observation。新增 `inference_twin.py` 的 Raw 包装保存校准 mu0/s0；同一扫描的新原始观测先使用这些统计量换算，再传 update_state。换访视时重新校准新的扫描尺度、initialize_state，并显式沿用个人 p；不能把不同扫描尺度的 buffer 拼在一起。

#### 6.3.1 forward、单步与滚动的精确集成

`NeuroTwinTFM.forward` 保留现有位置参数和 `(pred,aux_info)` 返回，末尾增加可选 `controls/time_deltas/dynamics_horizon/calibration_bold/profile`。`dynamics_horizon` 是模型调用的可选覆盖，缺省复用 CPM H，不新增 horizon CLI。

- forward 先完成现有 RevIN、字段条件编码、当前历史 `_encode`，从 tokens/图/x_grid 直接创建内部初始状态；不能再调用 public initialize_state 重复编码当前历史。profile 基线编码另走一次共享 encoder，训练时保持梯度。
- graph_ode 分支调用新 `(z0,q0,adj,cond,profile,...)` head，反归一化 q_trajectory，取第一帧为 `[B,F,1,1]` 的主输出；不得再加一遍 x_last。旧 discrete 分支保留 `(delta_norm,one_aux)` 与 x_last 锚点相加，不把 ODE head 强塞入旧参数签名。
- `aux_info['cpm_pred']` 仍是 `[B,F,H]`；动态均值、方差及 profile 按 §3 提供，保留旧 sc_prior/pathology_cond 字段。
- `predict_next_state(...,control=None,dt=None,calibration_bold=None,profile=None)` 显式设动态 H=1；返回原 pred_next/pred_delta/pred/aux_info 的形状。只为首步积分，避免无用的 H=8 动态运算。
- `rollout_next_states(...,steps,controls=None,time_deltas=None,calibration_bold=None,profile=None)` 只初始化一次当前历史，按 `[B,steps,F]` 控制和 `[steps]` 或 `[B,steps]` 时间间隔推进，返回 `[B,steps,F]`。无真观测时固定图、c/p、RevIN stats；不要回填预测后再次编码。
- `forecast_horizon` 保持原 CPM 语义，动态轨迹通过 rollout 取得；`future_control [B,F,H]` 继续仅属于 legacy CPM，不能与 `controls [B,H,F]` 自动转置混用。新训练关闭 cpm_intervention。
- state API 仅供 graph_ode；向 discrete 传新控制或动态时间参数应直接说明不支持，不静默忽略，也不模拟假的持续 latent。graph_ode 与 MoE/RECC 不组合，显式要求时直接说明不支持。

不同条件或有/无控制的比较从相同观测、同一 profile、同一图和同一 RevIN stats 初始化，不串接前一个方案的预测。原始尺度控制差值通过 `denorm(q_control)-denorm(q_zero)` 取得，不能对增量单独加均值。

### 6.4 逐 ROI 方差

在新 head 内增加 `VarianceHead`：`LayerNorm(D)→Linear(D,1)`，末层 weight/bias 零。对每个未来 latent 计算 logvar，detach 输入，仅训练该小头：

$$
\ell_{ih}=\operatorname{clip}(W_\sigma\operatorname{LayerNorm}(Z_{ih}^{\mathrm{detach}})+b_\sigma,-6,4),\qquad \sigma^{\mathrm{norm}}_{ih}=\exp(\ell_{ih}/2).
$$

BrainRevIN 的 affine_weight 记 a_i，当前历史 stdev 记 s_i，则数据尺度标准差为：

$$
\sigma^{X}_{ih}=\sigma^{\mathrm{norm}}_{ih}\frac{s_i}{|a_i+\epsilon_{\mathrm{RevIN}}|}.
$$

复用实际 RevIN 分母 `a_i+eps`，不能错误写成 `abs(a_i)+eps`。无 RevIN 时直接返回 sigma_norm。不把该对角高斯自动称作校准置信区间，也不将两个方案的标准差独立相加作为干预效应区间；本次只输出预测标准差。

## 7. 健康参照：固定描述量上的人口学岭回归

新增 `analysis/twin_reference.py`；不往 TFM 增加健康分类 loss，不用随训练漂移的 latent 直接定义病理。

### 7.1 统一描述量

`describe_history(history)` 输入固定 K_ref=64 的 `[B,64,F]` 数据尺度 BOLD，输出 `[B,F,6]`。不同长度不混入同一个参照，调用方显式提供最近 64 个真实观测；不足 64 时该分析不执行，预测接口仍可运行。

沿时间去均值并除以当前历史 RMS，得到 r `[B,F,64]`，eps=1e-6。六分量按固定顺序：

1. acf1：lag=1 的中心化片段相关；两片段各自去均值、分母加 eps。
2. acf2：同样计算 lag=2。
3. low_power：rfft 功率在 `(0,0.125]` cycles/sample 的比例，分母为全部正频率功率。
4. spectral_centroid：正频率功率加权频率中心，单位 cycles/sample。
5. diff_rms：相邻 r 差分平方的均值开方。
6. signed_fc_mean：该 ROI 与其他 ROI 的 signed Pearson correlation 均值，对角不参与。

不把无 TR 信息的频率标成 Hz；不从归一化信号推断原始强度异常。rfft 的功率、分母和常量序列都按上述 eps 计算，不增加频谱网络。公开函数同时用于 HC 拟合、患者分析和 outcome 特征。

### 7.2 拟合与推理

每个 HC 训练被试只取最早 64 个观测的一份描述量，避免扫描窗口数决定权重。人口学输入沿用训练集 Age/education 统计量与 Sex 词表，不使用 HAMD/HAMA、诊断或 ID。

设计向量 a 包含：intercept、Age z、Age z 平方、education z、Sex 已登记类别 one-hot，以及三字段有效 mask；连续缺失使用 z=0。将 `[F,6]` 展平为 Y 的 6F 列，岭回归用 solve，不显式求逆：

$$
W=(A_{\mathrm{demo}}^{\top}A_{\mathrm{demo}}+\alpha R)^{-1}A_{\mathrm{demo}}^{\top}Y,\qquad \alpha=1,
$$

R 为对角矩阵，intercept 的惩罚为 0，其余为 1。类别/mask 列可相关，岭项保证对应方向可求解。保存训练残差逐 ROI/分量尺度：

$$
s_{ij}^{\mathrm{ref}}=\sqrt{\operatorname{mean}_n[(Y_{nij}-\hat Y_{nij})^2]+10^{-6}}.
$$

患者输出有符号分量偏离 r_ref 与 ROI 距离 d_ref：

$$
r^{\mathrm{ref}}_{ij}=\frac{D_{ij}-\hat D_{ij}(a)}{s_{ij}^{\mathrm{ref}}},\qquad d^{\mathrm{ref}}_i=\sqrt{\frac{1}{6}\sum_j(r^{\mathrm{ref}}_{ij})^2}.
$$

它们是人口学条件下的标准化动态偏离，不是疾病诊断概率、治疗目标或因果病灶。HC 训练残差尺度不是自动校准的正常人百分位；代码不设置诊断阈值。当前临床条件变化不改变同一真实 BOLD 的此参照输出。

接口：`fit_reference(dataset, train_subjects, stats)->payload`；`score_reference(history,demo,payload)->dict`。NPZ 保存 W、s_ref、字段布局、原人口学统计量、K_ref、六个分量名；不新增参照模型训练框架。命令行 `python -m analysis.twin_reference fit|score`，fit 从已有 split 读取 train 被试。

## 8. 临床结局：单独的随访读出，独立样本单位

新增 `models/twin_outcome.py::ClinicalOutcomeHead` 与 `train/twin_outcome.py`。冻结动态 checkpoint，使用患者/访视对训练，不将随访临床标签复制到数百个 next-TR 样本。

### 8.1 特征、目标与输出

一个访视对的特征 v 固定为 83 维，所有来源均在结局之前已知或由基线模拟生成：

| 分量 | 维数 | 定义 |
|---|---|---|
| profile p / 临床 c | 32+32 | 基线校准码与当前八字段条件编码 |
| 动态描述量摘要 | 6 | 基线 64 TR 描述量逐 ROI 平均 |
| 健康偏离摘要 | 6 | 基线参照残差逐 ROI 绝对值均值 |
| 模拟有/无控制差摘要 | 3 | 早期 H=8 差轨迹的有符号全均值、全 RMS、末帧绝对值均值 |
| 有效控制摘要 | 3 | mean(abs(U))、max(abs(U))、至少一次非零的 ROI 比例 |
| 随访间隔 | 1 | log1p(followup_days) |

Outcome head 不用真实随访 BOLD/随访临床条件作输入。模拟 8 TR 是早期响应特征，不是把患者向前积分数周。该读出从配对数据学习早期特征与较晚结局的映射。

输入的 83 列使用 outcome 训练对的均值/标准差标准化，尺度下限 1e-6；标准化统计量写入读出 checkpoint。MLP 固定 `83→64→4`、GELU、无 Dropout：前两列为 standardized delta 均值，后两列为 logvar 并 clamp 到 [-6,4]。

HAMD/HAMA 原始总分变化目标定义为 followup-baseline；分别使用训练对的变化量均值/标准差。mask_j 要求该分量基线和随访都存在，缺 HAMA 不丢掉 HAMD 标签：

$$
L_{\mathrm{outcome}}=\frac{\sum_{n,j}m_{nj}\left[\tfrac12\exp(-\ell_{nj})(\hat\delta_{nj}-\delta_{nj})^2+\tfrac12\ell_{nj}\right]}{\max(1,\sum_{n,j}m_{nj})}.
$$

输出返回原分数尺度的 `delta_mean [B,2]`、`delta_std [B,2]`，以及有基线值时的 `followup_mean=baseline+delta_mean`。不创建 responder 分类头、不人为设响应阈值；同日 HAMD 解释使用脑独立分析，不用已输入的 HAMD 回归自己。

### 8.2 配对数据与训练入口

配对 CSV 固定字段：`subject_id,baseline_visit_id,followup_visit_id,baseline_bold_path,control_path,followup_days,baseline_hamd,followup_hamd,baseline_hama,followup_hama,split`。baseline_bold 为 `[T,F]` NPZ，T>=64；control_path 是明确上游处理的 H=8 控制 NPZ `[8,F]`；no-treatment 样本显式保存全零控制，不将缺 control_path 猜成 sham。八字段基线条件另由现有 merged 表及该模型保存的统计量对齐 subject_id。

baseline NPZ 的键为 `bold`，控制 NPZ 的键为 `u`。特征固定取基线最早 64 TR，profile 用其中最早 32 TR；从第 64 TR 末状态开始模拟 H=8，参照也用同一基线 64 TR。真实随访影像不用于补足这段基线。首版 outcome 接口使用本文默认的 MDD 八字段 checkpoint、C=32、profile_dim=32，输入固定 83 维，不为其他宽度自动补零或截断。

一个 CSV 行就是一个训练样本；split 是按 subject_id 分组的 train/val/test 字段，不能让同人的不同访视对落在不同 split。代码只做这项直接防泄漏检查，不设计新划分/注册系统。若基线 merged 表不适用于该访视，应由输入准备提供对应基线条件表，不自动使用当前最新评分。

`python -m train.twin_outcome --pairs ... --dynamics_checkpoint ... --reference ... --out_dir ...`：冻结动态模型和参照，no_grad 提取 v，CPU/GPU 均可；默认 AdamW lr=1e-3、weight_decay=1e-4、batch=32、epochs=100、seed 沿用模型保存值。CLI 可改这些现有训练参数，本文不执行训练。

若某输出在训练集中完全没有有效标签，必需的统计量不存在，应直接说明该分量无训练标签；不要输出一个随机的 HAMA 模型。首版要求两种分量至少各有一条有效训练目标；缺当前配对数据不阻断结构/API 和合成前向交付，也不生成拟合权重。

## 9. 动态损失与现有训练集成

`TFMDualLoss.forward` 末尾增加可选 `dynamics_pred/dynamics_mean_norm/dynamics_logvar_norm/dynamics_target_norm/dynamics_mask`；保留原四个/五个位置参数及 stats 中 `loss_one/loss_cpm`，main 的现有进度条不需要重写。

L_existing 代表原 One-Step、CPM 和图正则。动态额外项只监督 h=2..H，因为第一帧已作为 One-Step 主输出；gamma/PCC 系数复用现有配置，默认 PCC=0.1：

$$
L_{\mathrm{dyn}}=\operatorname{maskedmean}_{b,h=2:H}\left[\operatorname{mean}_i\operatorname{Huber}(\hat X^{\mathrm{dyn}}_{bih},X_{bih})+0.1(1-\operatorname{PCC}_{\mathrm{ROI}}(\hat X^{\mathrm{dyn}}_{bh},X_{bh}))\right].
$$

方差项使用 h=1..H 的归一化误差，均值和目标 detach；VarianceHead 的 latent 输入也 detach，因而该项只训练方差头，不重复改变均值预测目标：

$$
L_{\mathrm{var}}=\operatorname{maskedmean}_{b,i,h}\left[\tfrac12\exp(-\ell_{bih})(\hat q^{\mathrm{detach}}_{bih}-q^{\mathrm{target}}_{bih})^2+\tfrac12\ell_{bih}\right].
$$

$$
L=L_{\mathrm{existing}}+0.5L_{\mathrm{dyn}}+0.05L_{\mathrm{var}}.
$$

main 用当前 aux.norm_stats 和 `BrainRevIN.normalize_with` 归一化 future 目标；不使用 future 估计统计量。动态/方差项使用独立 dynamics_mask=future_mask，广播到 ROI，分母按有效 horizon/ROI 元素数计算；h2..H 无有效元素时 L_dyn=0。legacy discrete 没有动态 aux 时损失不变。

CPM 不接新控制，因此真实受控样本不能强迫它预测未知未来驱动：有 controls 时令 `active_h=controls.abs().sum(-1)>0`，`cpm_mask=future_mask*(active_h.cumsum(-1)==0)`；即从首次非零未来驱动开始屏蔽 CPM 监督，但保留动态/方差监督。没有 controls 或全零 controls 时 CPM mask 不变。两类 mask 分开传，不能用 CPM 的屏蔽 mask 同时屏蔽受控动态学习。该处理避免相同历史、不同未来刺激给无控制辅助头提供相互矛盾的确定性目标。CPM 有效权重总和为零时，整个 CPM Huber/PCC 项都为零；现有 PCC 的 `1-corr` 空集合计算会留下常数 1，接入时必须处理这一直接边界，不仅把 Huber 置零。

HC 从头训练新结构；MDD 继承本轮同配置 HC，原人口学统计量及新增五字段拟合规则保留。图动力学条件 FiLM、PersonalFiLM 和条件扩散项标记 `is_conditioning_adapter=True`；不要给整个 head 加标记，否则冻结期会误训练共享动力学。

ProfileEncoder 属于共享 backbone：补到 `set_finetune_stage` 解冻模块列表。VarianceHead 在 one_step_head 下跟随共享参数解冻；临床读出由独立入口训练，不加入 next-TR optimizer。get_param_groups 复用已有按 adapter 标记分组机制，不建立新 optimizer 框架。

## 10. 配置、权重、复评与推理接入

保持 `model_arch='tfm'`，用明确构造参数选择 head，避免再建一个几乎重复的主模型类。新增构造参数 `dynamics_mode,ode_steps,dynamics_dt,graph_source,personalized,profile_dim`；CLI 再增加 loader 参数 `calibration_length/control_manifest` 和两个 loss 权重。全部沿用一个 args/config 来源，不创建候选注册器或新的配置框架。

| 新增或扩展 CLI | 本文推荐值 | 消费位置 |
|---|---|---|
| `--dynamics_mode {discrete,graph_ode}` | 新脚本 graph_ode，构造缺省 discrete | 模型 head 选择 |
| `--ode_steps` / `--dynamics_dt` | 2 / 1.0 | 固定 RK2 |
| `--graph_source {legacy_mask,history_fc}` | 新脚本 history_fc，原缺省 legacy_mask | 模型与 loader |
| `--bold_source` 扩展 raw | 新脚本 raw，原缺省 auto | loader；没有 Raw 时报告文件错误 |
| `--personalized` / `--profile_dim` | 新脚本 True / 32，旧构造 False | 个人校准模块与动态头 |
| `--calibration_length` | 32 | Raw 标准化、profile 输入及合法锚点 |
| `--control_manifest` | 未提供时 None | 真实逐区间控制序列 |
| `--lambda_dynamics` / `--lambda_variance` | 0.5 / 0.05 | TFMDualLoss，不作为 head 构造参数 |

num_rois 复用已有 `--num_rois`，动态 H 复用 `--cpm_horizon`，hidden 复用 `--tfm_one_step_hidden_dim`；loader 构造参数也必须传入所有训练、验证及复评路径。原脚本的优化器、学习率、batch 和 seed 保持原设置；不把 outcome 独立优化器参数套给 next-TR 训练。

`main.py`、`evaluate_variant.build_model_from_args`、`build_eval_loader`、`analysis.next_point_eval.build_next_point_loader/run_inference` 全部传递新输入。推理批次的 `calibration_bold` 和 controls 也必须搬到 device，不能只有训练支持 profile。现有 rollout 评估传相同 profile/控制切片，CPM 继续报告为无控制辅助预测，不用于评价受控动态响应。

ARCH_VERSION 由当前 5 增到 6，保存上述结构/数据参数。旧 checkpoint 缺参数时按 `discrete/personalized=False/legacy_mask` 构造；现有冻结 arch_version=4 推理入口和 `configs/final_model.json` 不改。本次不做旧 head 到 ODE 的迁移；新 HC/MDD 使用新架构同源权重，不用 strict=False 静默滤键。

修改原两个训练脚本的显式新参数，名称改为 `hc_demo_twin` / `mdd_cat_twin`；MDD 默认只匹配前者的 base_best.pt，不覆盖或混选原 hc_demo 产物。保留现有 HC_WEIGHT 指定起点入口及零/多个候选报错行为。

新增 `inference_twin.py`：调用现有 args/condition bundle 重建，strict 加载新动态 checkpoint，提供 Raw 校准、profile/state、受控轨迹和观察更新包装。`analyze_pathology` 需显式传 reference 文件；`predict_outcome` 需显式传 outcome 文件。把这些必需输入直接留在函数签名，不自动搜索“最新”参照或随机建头。

## 11. 文件级改动与实现顺序

| 文件 | 必需修改，不留给实施者重新设计 |
|---|---|
| 新增 `models/tfm_dynamics.py` | 图向量场、RK2、state rollout 所需 head、VarianceHead |
| 新增 `models/tfm_personal.py` | 基线 token 池化、ProfileEncoder、PersonalFiLM |
| `models/tfm.py` | 选择新头、共享编码 helper、基线/profile、state API、动态 aux、受控双轨迹 |
| `models/graph_prior.py` | 单零因子初始化修复 |
| `utils/dataloader.py` | Raw 前缀校准、合法锚点、calibration_bold、history_fc 队列、控制 manifest |
| `main.py`、`train/losses.py` | CLI、profile/control batch 流、动态与方差目标、原日志契约 |
| `train/optim.py` | profile_encoder 解冻及新 adapter 正确分组；沿用原 checkpoint 保存 |
| `models/__init__.py`、`experiments/evaluate_variant.py`、`analysis/next_point_eval.py` | 版本、完整构造/loader/推理重建 |
| 新增 `analysis/twin_reference.py` | 六分量描述函数、HC ridge 拟合与 ROI 分量评分 |
| 新增 `models/twin_outcome.py`、`train/twin_outcome.py` | 83 维特征、两结局概率读出、配对样本训练入口 |
| 新增 `inference_twin.py` | 新结构加载与用户推理包装；保留旧 inference.py |
| 两个 `*_tfm.sh`、`README.md` | 新默认显式配置、独立输出目录、新增三个调用示例 |

后续代码实施不删除原 discrete/MoE 代码、其他旧文档、定型配置或数据。新路线 graph_ode 不组合 MoE/RECC，显式要求时直接说明不支持；它不需要在旧路径模拟一套假的 state API。

按三项代码依赖实施，每项作为完整开发任务，不设置审核等待：

1. **动态模型及完整集成**：先实现新 head/profile 与图初始化；随后一次接通数据、state、loss、optimizer、checkpoint、训练脚本、复评和新推理。完成后可直接启动基线训练与控制模拟。
2. **健康参照**：实现固定描述量、ridge 和分析接口；它只依赖已有影像/人口学数据，不依赖动态模型训练完成。
3. **临床结局读出**：复用描述量和动态推理特征，接配对 CSV、MLP 和新推理函数。真实拟合需要用户的配对数据，代码集成用合成样本即可完成。

健康参照与动态核心可以并行编码和拟合，前提是先统一 ROI 顺序、Raw 校准尺度、人口学统计量以及 K_ref=64；参照不依赖动态模型训练完成。临床读出的特征提取依赖二者，真实训练还依赖冻结动态 checkpoint 和配对随访数据。不是先独立开发图 ODE，再开发第二个“孪生版 ODE”。

## 12. 最小必要检查与后续实验边界

实施后只运行直接相关检查：

- 对新增/修改 Python 文件做 compile/import；两个脚本做 `bash -n`，并解析其实际参数。
- B=2、F=4、K=32/64、D=32、H=3 的一次 CPU 前向/反向：核对 pred、动态均值/std、CPM、profile 和控制维度；VarianceHead 从 detached latent 获得梯度，方差 loss 不更新均值 head。
- 给 RK2 注入简单线性/常数 field，核对 stage、dt 广播与同步 q 积分；单步与同初始 state 的 rollout 第一帧一致，零控制分支相同。控制贯通检查用非零 rate decoder，避免初始 persistence 误判。
- 一个小型 Raw/control NPZ 样本核对前缀/锚点/区间索引，不读取 Mask 或全扫描缓存；一个新小模型 state_dict 保存后 strict 重建即可。
- 用合成 HC 描述量拟合 ridge 并输出 `[B,F,6]`/`[B,F]`；用合成配对特征执行 outcome `[B,2]` 前向与 masked loss。无需真实训练或完整回归。

不建立全量测试矩阵、门禁、风险登记、实验晋级或多轮审核。代码完成不要求用户先证明临床条件有效；控制和结局模块缺真实数据时保留已实现接口，明确没有拟合效果。

后续用户实验可分别回答：profile 是否改善个人未来预测；健康偏离是否稳定关联症状；持续 latent 是否改善长程动态；真实响应数据能否校准控制分支及临床读出。方差覆盖率和跨访视泛化由实际数据检验，不把点预测测试通过当作机制或疗效证据。

本文档交付检查只涉及本次写入内容、公式及文件引用，不运行上述尚未实施模块的检查，也不沿用上一轮测试数作为新方案效果。

## 13. 理论依据与解释限制

[Neural ODE 原始论文](https://arxiv.org/abs/1806.07366)提供参数化状态导数与可微积分思路；[UDE 原始论文](https://arxiv.org/abs/2001.04385)提供结构项与可学习残差组合的依据。本方案保留功能图扩散结构和神经残差，但不借原论文宣称已识别真实脑机制。

[SPM 官方 DCM 文档](https://www.fil.ion.ucl.ac.uk/spm/docs/manual/dcm/dcm/)区分神经状态与到 BOLD 的血流动力学观测层。本文采用 latent 到 BOLD 速率的学习读出，没有血流动力学方程，因此 Z、扩散率与临床调制不能直接解释为生理兴奋性、解剖因果连接或致病机制。

profile 是固定基线的编码表示，不是逐人独立训练的模型；跨访视稳定性与其是否提供额外信号由数据判断。比较 profile 的收益时保持 Raw 前缀、合法锚点和可见信息口径一致，避免把额外历史信息的收益全部归为个体化机制。

方差输出是条件预测误差的对角高斯读出，不自动等于校准置信区间、模型认知不确定性或刺激效应区间。健康参照输出是人口学条件下的动态偏离，不是病灶或诊断概率。

8 TR 的模拟响应与三个控制摘要不能完整代表数周治疗的频率、剂量和疗程；临床读出学习的是指定数据格式下的随访关联。没有真实受控观测或随访配对数据时，不宣称个体疗效已校准，也不根据随机控制映射搜索、推荐最优刺激处方。刺激优化器不在本次范围内。

历史单 HAMD arch_version=4、当前八字段 arch_version=5 与拟实施的新结构权重分别保存；历史预测指标和工程测试不能转作新结构效果。本次仅合并文档，没有改动代码、原始数据或模型权重，也没有执行新模型测试、训练、真实参照拟合或疗效验证。

本次文档检查：22 个显示公式的定界符、块结构和花括号检查通过，本地文档链接均存在；健康参照与临床结局两节的原有详细内容完整保留。22 个公式通过 KaTeX 0.16.22 严格解析，Markdown/KaTeX 本地预览经无界面 Edge 渲染，已查看向量场、同步 RK2 与临床损失的局部排版；未逐屏人工检查整个 Codex 页面，也没有把文档渲染检查表述为模型验证。
