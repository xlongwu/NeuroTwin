# NeuroTwin V2 重构方案：从 Next-State Predictor 到个体化脑动力学数字孪生

> 适用版本：NeuroTwin `main` 分支（以 2026-09-28 “修改任务为预测下一个时间点”后的版本为基线）  
> 核心目标：围绕 **HC 规范性脑动力学 → MDD 个体病理偏离 → 在线状态同步 → 虚拟干预** 重构任务与模型，避免继续堆叠与当前任务语义不匹配的模块。  
> 实施原则：每次只引入一个主要机制；所有新结构必须通过独立消融证明收益；优先修正任务定义和信息流，再增加模型复杂度。

---

## 0. 总体判断

当前 NeuroTwin 已经完成了一个重要方向调整：主任务从旧版“窗口级未来 BOLD / dFC”收敛为：

$$
p(x_{t+1}\mid x_{t-K+1:t},SC)
$$

MDD 阶段进一步学习：

$$
p(x_{t+1}\mid x_{t-K+1:t},SC,c_{\mathrm{MDD}})
$$

其中 $x_t\in\mathbb{R}^{F}$ 为第 $t$ 个 TR 的全脑 ROI 状态，当前 $F=116$，$c_{\mathrm{MDD}}$ 主要由 HAMD 病理条件构成。

这一任务定义比旧版明确，但当前模型结构仍保留大量面向 `[B,F,W,S]` 窗口预测任务的历史设计，而 next-timepoint 实际输入已经退化为：

$$
X\in\mathbb{R}^{B\times F\times 1\times K}
$$

即 $W=1$。因此当前最主要的问题不是模型能力不足，而是 **任务已经变化，backbone 的信息流却没有完全随任务重构**。

NeuroTwin V2 不建议继续通过增加 Transformer、更多专家、更多 ODE solver 等方式提高复杂度，而应将研究主线收敛为：

```text
Healthy Controls
      │
      ▼
Normative Brain Dynamics
      │
      ▼
Individual MDD Deviation
      │
      ▼
Patient-specific Latent Twin
      │
      ├── Observation Assimilation ──► 持续同步真实患者状态
      │
      └── Virtual Intervention ──────► 模拟候选干预后的状态传播
```

最终科学问题应从：

> 能否预测下一 TR 的 BOLD？

提升为：

> 能否学习结构约束的健康脑规范性状态转移动力学，并刻画 MDD 个体偏离，在新观测到达时持续同步患者状态，进一步支持患者特异的虚拟干预模拟？

---

# 1. 第一阶段：冻结当前基线，先修正任务和评估闭环

## 1.1 当前问题

当前代码刚完成 next-timepoint 重构，仍存在以下协议风险：

1. README、Version1 文档、实验注册和实际代码存在少量版本状态不一致。
2. 当前 BOLD 文件已经做过全序列逐 ROI z-score；对于严格 forecasting，这意味着未来时间点参与了该被试归一化统计量估计。
3. 当前主要 checkpoint 仍由单步 next-state MAE 选择，虽然已有 rollout 评估，但长程动力学尚未真正进入主训练目标。
4. shuffled-HAMD 等关键负对照尚未形成完整闭环。
5. 复杂模型若只和 persistence/trend 对比，仍不足以证明 Graph/SC/MoE 的独立价值。

---

## 1.2 具体修改

### 修改 1：锁定唯一任务定义

正式将主任务定义为：

$$
X_t=[x_{t-K+1},...,x_t]\rightarrow
\{x_{t+h}\mid h\in\mathcal H\}
$$

第一阶段建议：

$$
\mathcal H=\{1\}
$$

第二阶段扩展：

$$
\mathcal H=\{1,2,4,8\}
$$

所有旧文档中的：

- next-window prediction；
- future window reconstruction；
- dFC prediction；

必须与当前任务区分。

如果 dFC 继续保留，只能作为 **rollout 后的二级评估对象**，不能再称为直接监督目标。

---

### 修改 2：重新定义无泄漏归一化

推荐优先方案：

```text
raw / nuisance-regressed ROI BOLD
        │
        ▼
context-only RevIN
        │
        ├── 使用 context 估计 μ、σ
        │
        ├── context 按 μ、σ 归一化
        │
        └── target 也仅用同一个历史 μ、σ 转换
```

禁止：

```text
整段 T=150 BOLD
      ↓
计算整段 μ、σ
      ↓
再切 forecasting sample
```

因为这会让未来数据参与样本尺度估计。

如果短期无法重做原始数据，则必须：

- 将现有全序列 z-score 标为 legacy preprocessing；
- 在论文 limitation 中明确；
- 增加 context-only normalization 消融。

---

### 修改 3：建立强基线套件

至少固定以下基线：

1. Persistence  
   $$
   \hat x_{t+h}=x_t
   $$

2. Linear Trend

3. AR(1)

4. Ridge-VAR  
   必须限制参数并仅用训练被试拟合。

5. GRU / TCN  
   作为不使用 SC 的神经时序基线。

6. Graph temporal baseline  
   使用 SC + 简单 TCN/GNN，不使用 MoE/ODE。

7. NeuroTwin-NoSC

8. NeuroTwin-NoMoE

9. NeuroTwin-NoDynamicsBlock

复杂模型只有在这些基线下仍保持稳定收益，才说明结构设计有效。

---

## 1.3 本阶段必须舍弃

### 必须删除/停止作为主结果

- “当前模型直接预测 dFC”的表述。
- 仅凭 spatial PCC 宣称模型学到了脑动力学。
- 全序列 z-score 被视为严格无泄漏 forecasting preprocessing 的表述。
- 只报告单次 seed 最优结果。
- 只和 persistence 一个简单 baseline 比较。

### 降级为 legacy

- 旧窗口级可视化流程。
- 旧 `[B,F,W,S]` 单窗口任务代码。
- 与 next-timepoint 无关的旧指标。

---

## 1.4 验收条件

进入下一阶段前必须满足：

- subject-level split 完全锁定；
- test set 不参与结构调参；
- 至少 5 个随机种子；
- 报告 subject-level bootstrap 95% CI；
- persistence / AR / VAR / TCN 等基线完成；
- normalization protocol 明确；
- shuffled-HAMD 可以自动运行；
- next-state 与 rollout 指标可以由统一脚本复现。

---

# 2. 第二阶段：重构损失函数，解决当前监督重复问题

## 2.1 当前问题

当前默认预测形式：

$$
\hat x_{t+1}=x_t+\widehat{\Delta x}_{t+1}
$$

而当前损失同时包含：

$$
L_{\mathrm{abs}}
=
|\hat x_{t+1}-x_{t+1}|
$$

和：

$$
L_{\mathrm{delta}}
=
|(\hat x_{t+1}-x_t)-(x_{t+1}-x_t)|
$$

二者数学上完全相同：

$$
L_{\mathrm{abs}}=L_{\mathrm{delta}}
$$

因此当前：

```text
lambda_abs = 1
lambda_delta = 1
```

并不是两个互补目标，而相当于对同一个误差重复加权。

---

## 2.2 具体修改

### 推荐新主损失

$$
L =
\lambda_{state}L_{state}
+
\lambda_{pattern}L_{pattern}
+
\lambda_{roll}L_{roll}
+
\lambda_{dyn}L_{dyn}
$$

其中：

### 1. State loss

建议 Huber 或 L1：

$$
L_{state}
=
\operatorname{Huber}(\hat x_{t+h},x_{t+h})
$$

Huber 相比纯 MSE 对 fMRI 偶发异常值更稳健。

---

### 2. Spatial pattern loss

$$
L_{pattern}
=
1-\operatorname{Corr}_{ROI}(\hat x,x)
$$

保留当前 spatial PCC 思想，但只作为辅助约束。

建议初始：

```text
lambda_state   = 1.0
lambda_pattern = 0.1
```

---

### 3. Multi-horizon loss

开启：

$$
h\in\{1,2,4,8\}
$$

定义：

$$
L_{MTP}
=
\sum_h w_h L_h
$$

建议起始权重：

```text
h=1 : 1.0
h=2 : 0.7
h=4 : 0.5
h=8 : 0.3
```

具体权重必须通过 val 选择，不作为理论固定值。

---

### 4. Rollout loss

训练时随机抽取 2–4 个递归步：

```text
context
  ↓
predict t+1
  ↓
append prediction
  ↓
predict t+2
  ↓
...
```

约束：

$$
L_{roll}
=
\sum_{j=1}^{R}
\gamma_j
L(\hat x_{t+j}^{roll},x_{t+j})
$$

目的不是单纯提高 H1，而是抑制 free rollout 的快速漂移。

---

### 5. Dynamics regularization

若需要显式约束状态变化，可使用：

$$
v_t=x_t-x_{t-1}
$$

再定义：

$$
L_{velocity}
=
\|\hat v-v\|_1
$$

如果 multi-horizon 足够，还可定义：

$$
a_t=x_t-2x_{t-1}+x_{t-2}
$$

用于约束二阶变化，但只建议作为后续消融，不应一开始加入。

---

## 2.3 本阶段必须舍弃

### 必须删除

当前默认条件下重复的：

```text
L_abs + L_delta
```

只保留一个 state reconstruction error。

### 不建议继续

- 为了“看起来目标多”继续叠加数学等价损失。
- 使用过多 handcrafted loss 代替任务本身。
- H1 变好但 H8/H16 rollout 明显恶化的模型仍被判为更优。

---

## 2.4 验收

至少比较：

```text
A. State only
B. State + PCC
C. B + MTP
D. B + rollout
E. B + MTP + rollout
```

最终模型只有在：

- H1 不显著下降；
- H4/H8/H16 明显更稳定；
- variance ratio 不快速塌缩；

时才保留 MTP/rollout。

---

# 3. 第三阶段：彻底移除 W=1 下失去意义的旧窗口架构

## 3.1 当前问题

next-timepoint 输入本质为：

$$
X\in\mathbb R^{B\times F\times K}
$$

但当前模型为了兼容旧版仍表示成：

$$
[B,F,1,K]
$$

这导致多个模块语义退化。

### BrainMDM window-axis

当：

$$
W=1
$$

时不存在真正的跨窗口 multi-scale evolution。

### WindowTemporalAttention

只有一个 token 时不存在：

$$
t_1\leftrightarrow t_2
$$

的跨窗口 attention。

### GraphODE Window Attention

同样缺少实际 window sequence。

因此这些结构继续存在只会：

- 增加参数；
- 增加论文解释成本；
- 产生“模块很多但实际没有功能”的 reviewer 风险。

---

## 3.2 目标结构

统一输入：

$$
X\in\mathbb R^{B\times F\times K}
$$

整体结构：

```text
BOLD Context [B,F,K]
        │
        ▼
Causal Temporal Encoder
        │
        ▼
SC-guided Adaptive Graph
        │
        ▼
Temporal–Graph Dynamics Blocks × N
        │
        ▼
Latent Brain State z_t
        │
        ▼
ROI/Horizon Query Decoder
        │
        ▼
x_(t+1), x_(t+2), ...
```

---

## 3.3 Causal Temporal Encoder

K 最大约 64，并不属于超长序列。

因此不建议为了 novelty 直接加入 Mamba。

推荐优先：

```text
Depthwise Causal Conv
      +
Dilated Temporal Conv
      +
Gated residual
```

例如 dilation：

```text
1, 2, 4, 8
```

使 receptive field 覆盖不同时间尺度。

形式：

$$
H_{temp}
=
H+
Gate(H)\odot TCN(H)
$$

如果后续实验显示长依赖确实不足，再加入轻量 temporal attention。

---

## 3.4 Temporal–Graph Dynamics Block

推荐替代当前复杂的四分支旧 GraphODE block：

$$
H^{l+1}
=
H^l+
\alpha_l
\left(
F_{temp}(H^l)
+
F_{graph}(H^l,A_{eff})
+
F_{ffn}(H^l)
\right)
$$

其中：

- `F_temp`：causal temporal conv；
- `F_graph`：SC/adaptive graph message passing；
- `F_ffn`：feature transformation；
- $\alpha_l$：zero/small initialized residual scale。

这样仍保留“动力学演化”的核心思想，但与新任务维度完全一致。

---

## 3.5 Forecast Decoder

保留当前 Future Query 的思想，但重新定义 query：

$$
q_{i,h}
=
e_{ROI_i}
+
e_{horizon_h}
$$

即每个 query 同时表示：

- 哪个 ROI；
- 哪个未来时间偏移。

然后：

$$
q_{i,h}
\xrightarrow{CrossAttention(z_t)}
\hat x_{i,t+h}
$$

这比旧 flatten head 更适合：

$$
h\in\{1,2,4,8\}
$$

---

## 3.6 本阶段必须舍弃

### 必须从主模型删除

- `W=1` 条件下的 WindowTemporalAttention。
- BrainMDM 的 window-axis pathway。
- ODE window attention。
- 为旧 `pred_window × pred_seq_len` 设计的展平逻辑。
- 为兼容旧任务而长期维持 `[B,F,1,K]` 的核心内部张量语义。

可以在 dataloader 边界短期兼容，但模型内部必须最终切换为 `[B,F,K,D]` 或 `[B,F,K]`。

### 降级为消融

- flatten ForecastHead；
- window-based MDM；
- 旧 window attention。

---

## 3.7 验收

比较：

```text
Legacy compatible backbone
vs
New F×K backbone
```

必须同时报告：

- MAE；
- H1/H4/H8/H16 rollout；
- 参数量；
- FLOPs；
- GPU memory；
- epoch time。

若简化版性能相当，则无条件保留简化版。

---

# 4. 第四阶段：SC 只构造一次，避免重复注入

## 4.1 当前问题

当前项目已经实现较完善的 `SoftAnatomicalPrior`，这是应该保留的设计。

但 SC 信息仍可能在多个位置重复出现：

```text
DFCAdapter
Graph dynamics
Forecast refinement
MoE refinement
```

重复 SC 注入容易导致：

- anatomical prior 压过 functional evidence；
- 图传播过度平滑；
- 难以解释到底哪个 SC 注入位置有效；
- 模块消融高度耦合。

---

## 4.2 推荐统一图构造

整个 backbone 只生成一次：

$$
A_{eff}
=
\lambda A_{SC}
+
(1-\lambda)A_{func}
+
\Delta A_{subject}
$$

其中：

### Anatomical prior

$$
A_{SC}
$$

来自 DTI/Mask。

### Functional adaptive graph

从 context latent 构造：

$$
A_{func}
=
\operatorname{softmax}
(QK^\top/\sqrt d)
$$

### Subject-specific residual

$$
\Delta A
=
UV^\top
$$

使用低秩结构限制自由度。

---

## 4.3 推荐 λ

第一阶段只保留：

```text
global lambda
sample-conditioned lambda
```

暂时不要同时使用 ROI-wise + time-varying λ。

优先证明：

$$
A_{SC}+A_{func}
$$

确实优于：

- SC only；
- functional only；
- no graph。

---

## 4.4 图只作为 prior

必须允许：

```text
SC 很弱 / DTI 未检测到边
但 BOLD 强烈支持功能依赖
```

因此禁止把 SC=0 直接定义为“永远不能通信”。

SC 应作为：

> anatomical prior

而不是：

> hard causal topology。

---

## 4.5 本阶段必须舍弃

### 必须停止作为默认方案

- hard SC binary mask；
- 每个模块重新独立构造一套 SC constraint；
- SC=0 导致永久断边；
- SC 重复出现在多个 refinement stage。

### 保留

- SoftAnatomicalPrior；
- low-rank $\Delta A$；
- sample adaptive λ；
- graph regularization。

---

# 5. 第五阶段：GraphODE 做一次明确去留，而不是继续增加 solver

## 5.1 当前问题

当前代码已经支持：

- Euler；
- RK2；
- RK4；
- adaptive solver；
- SDE；
- learnable step。

但当前动力学函数仍基本属于 autonomous transition，并没有让真实扫描时间 $t$ 或 TR 充分进入模型。

更重要的是当前实验记录已经显示：

```text
ode_steps = 1 / 3 / 6 / 12
性能差异非常小
```

这意味着当前性能未必依赖“连续时间积分”本身。

---

## 5.2 推荐路线 A：主线简化

推荐把主模型重新命名为：

**SC-guided Graph Dynamics Block**

采用：

$$
H_{l+1}
=
H_l+
\alpha_l f_\theta(H_l,A_{eff})
$$

只使用 2–3 个 block。

论文中不再强调 Neural ODE。

这是当前最推荐方案。

---

## 5.3 路线 B：如果必须保留 ODE

只有当 ODE 本身准备成为核心贡献时，才继续：

$$
\frac{dz}{dt}
=
f(z(t),A,c,t)
$$

至少需要：

1. 输入真实 TR；
2. solver step 对应真实时间尺度；
3. irregular sampling 时能体现连续时间优势；
4. solver sensitivity analysis；
5. ODE 相对普通 residual block 有稳定优势。

否则没有必要继续投入 adaptive/SDE。

---

## 5.4 本阶段必须舍弃

### 必须舍弃

如果最终采用路线 A：

- “模型已学习真实连续神经生理时间动力学”的声明；
- 将 solver 数量作为主要创新；
- adaptive/SDE 作为主模型卖点。

### 降级为 appendix ablation

- Euler/RK2/RK4；
- ODE steps；
- SDE。

---

# 6. 第六阶段：将 HC 预训练升级为 Normative Dynamics Learning

## 6.1 当前问题

目前 HC 阶段主要目标仍是：

$$
X_t\rightarrow x_{t+1}
$$

这容易使模型成为局部 autoregressive predictor，而没有显式要求 latent space 形成稳定健康脑动力学结构。

---

## 6.2 目标

HC 阶段改成学习：

$$
p(z_{t+h},x_{t+h}\mid X_t,SC)
$$

使 HC backbone 成为：

> normative brain dynamics model

而不是简单 pretrained predictor。

---

## 6.3 推荐训练任务

按以下顺序逐个加入。

### Task A：Multi-Timepoint Prediction

$$
h\in\{1,2,4,8\}
$$

这是第一优先级。

---

### Task B：Masked Temporal Modeling

随机 mask：

- 10%–30% 时间点；
- 部分 ROI-time patch；

让模型根据时空上下文重建：

$$
X_\mathcal M
$$

参考 BrainLM 的 masked prediction 思想，但无需复制大模型规模。

---

### Task C：短程 rollout

2–4 steps 即可。

---

### Task D：Latent consistency

同一被试相邻 context：

$$
z_t,z_{t+\delta}
$$

应满足平滑但非塌缩的状态转移。

可用简单：

$$
L_{latent}
=
\|F(z_t)-sg(z_{t+1})\|
$$

不建议一开始引入复杂 contrastive negative mining。

---

## 6.4 推荐总损失

第一版：

$$
L_{HC}
=
L_{MTP}
+
\lambda_rL_{roll}
+
\lambda_mL_{mask}
$$

先验证三项即可。

不要一次把 CPC、contrastive、spectral、graph、masked、rollout 全部加入。

---

## 6.5 本阶段必须舍弃

- HC 预训练 = 单纯 H1 回归的长期设定。
- 为了增加“预训练任务数量”同时叠加多个未经消融的 SSL loss。
- 把 BrainLM 大规模 foundation setting 直接照搬到当前样本规模。

---

# 7. 第七阶段：病理条件从 HAMD scalar 升级为“状态 + 规范偏离 + 症状”

## 7.1 当前问题

HAMD total 是临床严重程度指标，但不能完整表示 MDD 神经生物学异质性。

可能出现：

```text
HAMD 相同
但脑连接异常完全不同
```

因此只用 HAMD 路由专家，容易使 MoE 学到：

> severity bins

而不是：

> pathological dynamics subtypes。

---

## 7.2 建立 Normative Deviation

利用 HC backbone 得到健康参考分布：

$$
z^{HC}\sim\mathcal N(\mu_{HC},\Sigma_{HC})
$$

MDD 被试：

$$
d_z
=
z^{MDD}-\mu_{HC}
$$

也可构建 standardized deviation：

$$
d_z^{std}
=
\frac{z^{MDD}-\mu_{HC}}
{\sigma_{HC}}
$$

如果年龄、性别、site 样本足够，进一步建模：

$$
\mu_{HC}
=
f(age,sex,site)
$$

然后：

$$
d_i=z_i-f(age_i,sex_i,site_i)
$$

---

## 7.3 FC deviation

在足够长 context 或整段 resting-state 数据上：

$$
D^{FC}_i
=
FC_i-FC^{norm}_i
$$

再编码：

$$
c_{dev}
=
Encoder(D^{FC}_i,d_z)
$$

注意 FC deviation 适合被试级 condition，不应从单 TR 计算。

---

## 7.4 病理 condition

最终：

$$
c_i=
[
c_{state},
c_{dev},
c_{symptom}
]
$$

其中：

### Current state

$$
c_{state}=Pool(z_t)
$$

### Normative deviation

$$
c_{dev}=Encoder(d_z,D^{FC})
$$

### Clinical phenotype

优先使用 HAMD item-level / factor-level：

$$
c_{symptom}
=
[HAMD_{core},
HAMD_{sleep},
HAMD_{anxiety},
...]
$$

若只有 total score，则先保留 total，但论文中必须承认表达能力有限。

---

# 8. 第八阶段：重构 MoE，让专家学习“病理动力学偏离”而不是重复 backbone

## 8.1 当前问题

MoE 如果只在末端产生 prediction residual：

$$
\hat x
=
\hat x_{HC}
+
\Delta x_{MoE}
$$

虽然工程上稳定，但其解释只能是：

> output correction

而不是：

> pathology changes the dynamics。

---

## 8.2 推荐结构

不建议专家复制整个 backbone。

使用低参数 residual dynamics adapter：

$$
z'_{t+1}
=
F_{HC}(z_t)
+
\sum_e p_e(c_i)\Delta F_e(z_t)
$$

其中：

$$
\Delta F_e
$$

可以是：

- LoRA；
- bottleneck adapter；
- low-rank graph residual；
- small MLP/TCN residual。

这样专家直接修正：

> 状态转移动力学

而不仅是最终输出。

---

## 8.3 Router

推荐：

$$
p_e
=
Router(c_{state},c_{dev},c_{symptom})
$$

而不是：

$$
Router(HAMD)
$$

评估阶段保持当前 deterministic dense-soft / deterministic top-k 思路。

---

## 8.4 专家数量

建议从：

```text
E = 4
```

开始。

不要在没有证据时增加到 8/16。

必须分析：

- expert usage；
- entropy；
- expert-output cosine similarity；
- symptom/deviation distribution；
- expert collapse。

---

## 8.5 必须舍弃

- stochastic routing 用于正式 test。
- 仅凭专家使用率不同就宣称发现 MDD 亚型。
- shared expert 如果当前消融已经证明无收益，则从主模型删除。
- 多个同构大专家完整复制 backbone。

---

# 9. 第九阶段：加入真正决定“数字孪生”性质的 Latent-State Assimilation

## 9.1 当前问题

当前模型主要是：

```text
历史观测
   ↓
预测未来
```

训练完成后参数固定，新的真实观测不会用于持续修正“患者当前数字状态”。

这更像：

> personalized predictor

而不是严格意义上的 adaptive digital twin。

---

## 9.2 新结构

### Step 1：Initial State

$$
z_t^+=E(X_{t-K+1:t})
$$

### Step 2：Prior Transition

$$
z_{t+1}^-
=
F_\theta(z_t^+,A,c)
$$

### Step 3：Observation Prediction

$$
\hat x_{t+1}
=
D(z_{t+1}^-)
$$

### Step 4：真实观测到达

$$
e_{t+1}
=
x_{t+1}-\hat x_{t+1}
$$

### Step 5：Correction

构建 correction encoder：

$$
r_{t+1}=C_\phi(e_{t+1},z_{t+1}^-)
$$

再构建 gate：

$$
K_{t+1}
=
\sigma(
MLP[
z_{t+1}^-,
e_{t+1}
])
$$

更新：

$$
z_{t+1}^+
=
z_{t+1}^-
+
K_{t+1}\odot r_{t+1}
$$

这里：

- $z^-$：prediction prior；
- $z^+$：观测同步后的 posterior twin state。

---

## 9.3 训练方式

随机从连续序列选择：

```text
context
  ↓
predict
  ↓
observe real point
  ↓
assimilate
  ↓
predict
  ↓
observe
  ↓
assimilate
```

训练 2–8 个循环。

重点：

**不在线更新整个模型参数。**

只更新 latent state。

这样既具有 twin synchronization，又避免每个病人进行在线 SGD 造成过拟合。

---

## 9.4 对照

必须比较：

```text
Free rollout
vs
Re-encoding full context
vs
Latent assimilation
```

重点观察：

- H8/H16/H32 error；
- posterior correction magnitude；
- state tracking 稳定性；
- computational cost。

---

## 9.5 必须舍弃

如果加入 assimilation：

- “模型一次编码后无限 rollout 就代表患者数字孪生”的叙事。
- 为每个新 TR 在线 finetune 全网络参数。
- 使用 target 直接进入未来 prediction、产生隐式泄漏。

---

# 10. 第十阶段：虚拟干预从 latent hack 升级为显式 Control Input

## 10.1 当前问题

现有：

```text
excitatory
inhibitory
variance_boost
variance_suppress
```

更多属于 latent perturbation。

它可以做机制探索，但不能天然解释为：

> 某个脑区受到 TMS 后的真实生理干预。

---

## 10.2 推荐控制动力学

将状态转移改为：

$$
z_{t+1}
=
F(z_t,A,c)
+
B_\theta u_t
$$

其中：

$$
u_t\in\mathbb R^F
$$

为 ROI-level intervention vector。

例如刺激 ROI $r$：

$$
u_{t,r}=a
$$

其余位置为 0。

---

## 10.3 干预传播

对于候选 ROI $r$：

$$
u^{(r)}
\rightarrow
z_{t+1}^{(r)}
\rightarrow
z_{t+2}^{(r)}
\rightarrow ...
$$

分析：

$$
\Delta x_j(h|r)
=
x_j^{intervention}(t+h)
-
x_j^{baseline}(t+h)
$$

得到：

- propagation strength；
- propagation delay；
- affected network；
- structural-path consistency。

---

## 10.4 Normative Restoration Score

可以构造研究性指标：

$$
R(r)
=
D_{norm}(baseline)
-
D_{norm}(intervention_r)
$$

其中：

$$
D_{norm}
$$

表示当前患者状态与 HC normative manifold 的距离。

即：

> 哪个干预能够让模拟后的脑状态更接近健康规范动力学？

注意：

这个指标只能称为：

**virtual normative-restoration score**

不能在没有真实干预数据验证的情况下称为：

**治疗效果预测**。

---

## 10.5 TMS 扩展

未来如果针对 DLPFC-rTMS，可进一步利用：

- DLPFC–sgACC functional relationship；
- SC polysynaptic pathways；
- 个体 normative deviation；

构建：

```text
candidate DLPFC target
      │
      ▼
SC-constrained propagation
      │
      ▼
predicted whole-brain response
      │
      ▼
distance to HC normative manifold
      │
      ▼
virtual target score
```

---

## 10.6 必须舍弃

- 直接改 latent 某几个维度后称为“TMS 模拟”。
- 未经真实刺激数据验证就使用“causal treatment effect”表述。
- 仅依据最终 BOLD 值变化选择刺激靶点。
- 不考虑 SC pathway 的任意全脑传播。

---

# 11. 第十一阶段：建立与“数字孪生”相匹配的评估体系

## 11.1 Level 1：单步预测

报告：

- MAE；
- RMSE；
- spatial PCC；
- $R^2$；
- delta direction accuracy。

---

## 11.2 Level 2：Multi-horizon

分别报告：

```text
t+1
t+2
t+4
t+8
```

禁止只平均成一个指标。

---

## 11.3 Level 3：Free Rollout

建议：

```text
H = 1,2,4,8,16,32
```

报告：

- temporal PCC；
- MAE；
- variance ratio；
- drift；
- PSD similarity；
- low-frequency power。

---

## 11.4 Level 4：Network Dynamics

rollout 足够长后计算：

- FC edge correlation；
- within-network FC；
- between-network FC；
- network modularity；
- dynamic FC state occupancy（后续）。

不要从单 TR 计算 FC。

---

## 11.5 Level 5：Personalization

按以下变量分层：

- HAMD severity；
- normative deviation magnitude；
- SC density；
- site；
- prediction volatility；
- baseline low-PCC subjects。

重点分析：

> 哪一类患者真正从 pathology conditioning 中获益？

---

## 11.6 Level 6：Twin Synchronization

专门评价 assimilation：

$$
Error_{free}(h)
$$

对比：

$$
Error_{assim}(h)
$$

以及：

$$
\Delta Error(h)
=
Error_{free}(h)-Error_{assim}(h)
$$

---

## 11.7 Level 7：Virtual Intervention

在没有真实刺激数据时，只报告：

- reproducibility；
- stability；
- SC consistency；
- intervention locality；
- perturbation-response matrix；
- sensitivity to intervention strength。

如果有 pre/post TMS：

再比较真实和模拟的：

- regional response；
- network response；
- treatment-associated change。

---

# 12. 必做消融矩阵

## 12.1 Task Ablation

```text
H1 only
H1 + PCC
MTP
rollout
MTP + rollout
```

---

## 12.2 Graph Ablation

```text
No graph
SC only
Functional graph only
SC + functional
SC + functional + subject residual
```

---

## 12.3 Pathology Ablation

```text
No condition
HAMD only
Brain-state only
Normative-deviation only
HAMD + state
HAMD + state + deviation
Shuffled HAMD
```

---

## 12.4 MoE Ablation

```text
No MoE
single adapter
4-expert MoE
HAMD router
state router
joint router
```

---

## 12.5 Dynamics Ablation

```text
Plain residual
Graph dynamics
RK2 GraphODE
RK4 GraphODE
```

如果 residual 与 ODE 无显著差异，主论文使用 residual。

---

## 12.6 Assimilation Ablation

```text
Free rollout
Full-context re-encode
Latent assimilation
```

---

# 13. 最终必须删除/舍弃清单

本节是实施过程中最重要的“负向约束”。

## A. 必须从主模型删除

### 1. W=1 后失去作用的 window 模块

包括：

- WindowTemporalAttention；
- BrainMDM window-axis multi-scale；
- ODE window attention；
- 旧 future-window flatten logic。

---

### 2. 重复监督

删除默认：

$$
L_{abs}+L_{delta}
$$

数学等价的重复加权。

---

### 3. SC 多位置重复强约束

SC 只用于生成一次统一 $A_{eff}$。

不要在：

```text
adapter
ODE
forecast refiner
MoE refiner
```

反复独立重新注入。

---

### 4. 随机 test routing

正式 test：

- dense soft；
- deterministic top-k；

二选一。

不允许 multinomial sampling 影响 checkpoint 比较。

---

### 5. Shared expert

如果当前已有消融确认无收益，则从主模型彻底移除，只保留历史实验记录。

---

### 6. 旧 dFC 主任务叙事

当前不是直接 dFC prediction。

dFC 只能是 rollout 后的派生评估。

---

## B. 必须舍弃的论文声明

除非额外完成相应验证，否则不能写：

- “GraphODE accurately models continuous physiological time”；
- “virtual stimulation identifies optimal clinical treatment target”；
- “MoE discovers biological MDD subtypes”；
- “perturbation proves causal connectivity”；
- “single-step PCC demonstrates digital-twin fidelity”。

---

## C. 必须降级为 appendix / ablation

- Euler / RK2 / RK4 solver comparison；
- SDE；
- flatten head；
- legacy hard SC；
- stochastic routing；
- excessive refiner rounds；
- legacy window pipeline。

这些可以证明设计过程，但不应占主模型叙事。

---

# 14. 推荐最终 NeuroTwin V2 架构

```text
                         Subject SC
                             │
                             ▼
                  Soft Anatomical Prior
                             │
                ┌────────────┴────────────┐
                │                         │
         BOLD Context                Functional Graph
           [B,F,K]                       │
                │                         │
                ▼                         │
        Causal Temporal Encoder          │
                │                         │
                └──────────┬──────────────┘
                           ▼
                 Graph Dynamics × N
                           │
                           ▼
                   Latent State z_t
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
     Brain State      Normative          Symptoms
                       Deviation
          │                │                 │
          └────────────────┼─────────────────┘
                           ▼
                Pathology Conditioner
                           │
                           ▼
               Residual Dynamics Experts
                           │
                           ▼
                  z_(t+h)^prior
                           │
                           ▼
                  ROI/Horizon Queries
                           │
                           ▼
                   x_(t+1...t+h)
                           │
                      real x_(t+1)
                           │
                           ▼
                  prediction error
                           │
                           ▼
                  Assimilation Module
                           │
                           ▼
                z_(t+1)^posterior

Virtual intervention:
u_t ─────► Control Projection B ─────► Dynamics Transition
```

---

# 15. 推荐实施顺序

## Phase 0：协议冻结

完成：

- normalization 修复；
- baseline；
- multi-seed；
- shuffled HAMD；
- test lock。

**不改模型大结构。**

---

## Phase 1：任务修复

完成：

- 删除 duplicate delta loss；
- MTP；
- rollout training；
- 完整 horizon evaluation。

进入下一阶段条件：

> NeuroTwin 在 H4/H8/H16 稳定优于 persistence / AR / VAR。

---

## Phase 2：Backbone 简化

完成：

- `[B,F,K]`；
- causal temporal encoder；
- graph dynamics；
- ROI/horizon query。

删除所有 W=1 退化模块。

进入下一阶段条件：

> 简化模型在性能不下降的同时显著降低参数/计算量。

---

## Phase 3：Normative HC Dynamics

完成：

- MTP；
- masked modeling；
- short rollout；
- HC latent reference distribution。

进入下一阶段条件：

> HC latent 能稳定跨 context 表示个体状态，并提升 MDD transfer。

---

## Phase 4：Pathological Deviation

完成：

- latent deviation；
- FC deviation；
- HAMD/state/deviation joint conditioning；
- pathology adapter/MoE。

进入下一阶段条件：

> true clinical/deviation condition 明显优于 shuffled condition。

---

## Phase 5：Twin Assimilation

完成：

- prior state；
- observation error；
- correction network；
- posterior state。

进入下一阶段条件：

> assimilation 在长 horizon 显著优于 free rollout。

此时项目才可以更有底气地使用：

> adaptive patient-specific digital twin

这一表述。

---

## Phase 6：Virtual Intervention

完成：

- explicit $u_t$；
- perturbation propagation；
- normative restoration；
- SC consistency。

没有真实刺激数据时：

> 只做 mechanistic virtual intervention。

有 pre/post TMS 或刺激数据后：

> 再做 intervention validation。

---

# 16. 推荐论文最终故事线

不要写成：

```text
我们设计了：
RevIN
+ DFCAdapter
+ BrainMDM
+ GraphODE
+ ForecastHead
+ MoE
```

这种模块堆叠叙事。

建议写成三个连续科学问题：

## Problem 1：健康脑动力学应该如何学习？

解决：

**SC-guided normative dynamics**

---

## Problem 2：MDD 个体如何偏离健康动力学？

解决：

**normative deviation-conditioned pathological dynamics**

---

## Problem 3：患者状态不断变化，静态预测器如何成为数字孪生？

解决：

**latent-state assimilation**

---

最终再扩展：

## Application：如果对某脑区实施虚拟干预，会发生什么？

解决：

**SC-constrained virtual perturbation**

形成完整逻辑：

$$
\boxed{
Normative\ Dynamics
\rightarrow
Pathological\ Deviation
\rightarrow
Twin\ Assimilation
\rightarrow
Virtual\ Intervention
}
$$

---

# 17. 推荐优先阅读文献

1. Wang HE, Triebkorn P, Breyton M, et al. **Virtual brain twins: from basic neuroscience to clinical use.** National Science Review. 2024;11(5):nwae079.  
   DOI: 10.1093/nsr/nwae079  
   用途：确定 virtual brain twin 的 personalized、generative、adaptive 定义，以及个体连接、参数反演和干预之间的关系。

2. Takahashi Y, Idei H, Komatsu M, et al. **Digital twin brain simulator for real-time consciousness monitoring and virtual intervention using primate electrocorticogram data.** npj Digital Medicine. 2025;8:80.  
   DOI: 10.1038/s41746-025-01444-1  
   用途：重点参考 data assimilation、hierarchical latent state、实时同步和 virtual intervention。

3. Luo Z, Peng K, Liang Z, et al. **Mapping effective connectivity by virtually perturbing a surrogate brain.** Nature Methods. 2025;22:1376–1385.  
   DOI: 10.1038/s41592-025-02654-x  
   用途：参考“先训练神经动力学 surrogate，再系统 perturb 各脑区并分析传播”的方法学。

4. Chen C, Lin L, Liu Y, et al. **Stable depression subtypes identified using functional connectome normative deviation models and their response to rTMS.** Molecular Psychiatry. 2026.  
   DOI: 10.1038/s41380-026-03634-z  
   用途：直接支持 HC normative model → MDD individual deviation → treatment-related heterogeneity。

5. Seguin C, Mansour L S, Betzel RF, et al. **White matter pathways mediating dorsolateral prefrontal TMS therapy for depression.** Nature Neuroscience. 2026;29:1048–1053.  
   DOI: 10.1038/s41593-026-02248-6  
   用途：支持未来将结构连接路径引入 DLPFC–sgACC TMS 传播模拟。

6. Ortega Caro J, Fonseca AHO, Rizvi SA, et al. **BrainLM: A foundation model for brain activity recordings.** ICLR 2024.  
   用途：支持 fMRI masked prediction、自监督脑状态表征和 future brain-state forecasting。

---

# 18. 最终决策原则

后续任何新模块必须回答至少一个明确问题：

1. 它解决了哪一个已经被实验确认的瓶颈？
2. 简单 baseline 为什么不能解决？
3. 它提高的是 H1，还是长期 dynamics？
4. 它是否增强 personalization？
5. 它是否增强 twin synchronization？
6. 它是否增强 intervention interpretability？
7. 去掉它后，性能或科学解释是否显著下降？

如果不能回答以上问题，则不应进入主模型。

NeuroTwin V2 的目标不是构建一个“模块最多”的模型，而应构建一个能够清晰回答以下问题的最小充分系统：

> 健康脑通常如何演化？  
> 当前患者偏离在哪里？  
> 患者的新观测如何更新他的数字状态？  
> 改变某个脑区后，个体化动力学会如何传播和变化？

这四个问题构成后续方法设计、实验设计和论文叙事的统一主线。
