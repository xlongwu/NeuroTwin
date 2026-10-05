# NeuroTwin 借鉴 TimesFM-3 的模型改进方案

> 目标：在保留 NeuroTwin “HC 规范性脑动力学 → MDD 个体化病理修正 → 数字孪生/虚拟干预”主线的前提下，借鉴 TimesFM-3 在多变量时序预测上的关键设计，重构 NeuroTwin 的时空 Backbone、未来轨迹预测方式、条件变量接口和概率预测目标。  
> 核心原则：**借鉴 TimesFM-3 的建模思想，而不是直接迁移其 330M 级模型、checkpoint 或默认超参数。**

---

# 1. 为什么 TimesFM-3 对 NeuroTwin 有参考价值

NeuroTwin 当前任务已经重构为：

$$
p(x_{t+1}\mid x_{t-K+1:t},SC)
$$

MDD 阶段进一步加入病理条件：

$$
p(x_{t+1}\mid x_{t-K+1:t},SC,c_{\mathrm{MDD}})
$$

其中：

- $x_t\in\mathbb R^{F}$：第 $t$ 个 TR 的全脑 ROI BOLD 状态；
- $F=116$：AAL116 ROI；
- $K\in[16,64]$：历史 context 长度；
- $SC\in\mathbb R^{F\times F}$：个体结构连接；
- $c_{\mathrm{MDD}}$：HAMD、病理偏离、个体状态等条件。

这本质上是一个典型的 **multivariate temporal forecasting** 问题：

$$
\text{ROI}\times\text{Time}
$$

TimesFM-3 的核心变化正是从单变量 forecasting 升级为原生多变量 forecasting，并显式拆分：

1. 单变量内部的时间依赖；
2. 不同变量之间的同步关系；
3. 整个 future horizon 的非自回归预测；
4. 历史条件和未来已知条件；
5. 概率预测。

这些机制与 NeuroTwin 的 “ROI 时间演化 + ROI 间网络交互 + 多步脑状态预测 + 病理/干预条件”具有高度对应关系。

---

# 2. TimesFM-3 的核心设计

## 2.1 Patch-based 时间序列表示

TimesFM-3 不逐时间点直接构造 Transformer token，而是将连续时间点切成 patch。

官方配置：

```text
input patch length  = 32
output patch length = 64
```

假设某变量序列为：

$$
x_1,x_2,\ldots,x_T
$$

则将其切成：

$$
P_1=[x_1,\ldots,x_{32}],\quad
P_2=[x_{33},\ldots,x_{64}],\ldots
$$

每个 patch 被编码成一个 token。

这样做的目的主要是：

- 缩短 token 序列；
- 降低 temporal attention 计算量；
- 让一个 token 表示局部时间模式，而不是单个观测值。

---

## 2.2 多变量二维 Token Grid

TimesFM-3 将多变量时序表示为：

$$
H\in
\mathbb R^{
N_{var}\times N_{patch}\times D
}
$$

可以理解为二维网格：

```text
                     Time / Patch →
               p1      p2      p3      p4

Variate 1      ● ───── ● ───── ● ───── ●
               ↕       ↕       ↕       ↕
Variate 2      ● ───── ● ───── ● ───── ●
               ↕       ↕       ↕       ↕
Variate 3      ● ───── ● ───── ● ───── ●
```

模型不把所有 token 混在一起做一次标准 full attention，而是显式区分两个轴。

---

## 2.3 Causal Temporal Attention

第一类 attention 只沿 **时间轴** 建模。

对于变量 $i$：

$$
H_{i,1:P}
\rightarrow
\operatorname{TemporalAttention}
$$

并严格使用 causal mask：

$$
p_k\leftarrow p_{\le k}
$$

即当前 patch 只能访问：

- 当前变量；
- 当前及过去时间。

不能看到未来 patch。

其作用是建模：

> “一个变量自身如何随时间演化”。

---

## 2.4 Full Variate Attention

第二类 attention 沿 **变量轴** 建模。

对于固定时间 patch $p$：

$$
H_{1:N,p}
\rightarrow
\operatorname{VariateAttention}
$$

不同变量之间可以相互看到。

因此模型可以学习：

$$
x_i(t)\leftrightarrow x_j(t)
$$

的跨变量依赖。

TimesFM-3 将 temporal attention 和 variate attention 交替堆叠，从而分别解决：

$$
\text{within-series temporal dependency}
$$

和：

$$
\text{cross-series dependency}
$$

---

## 2.5 Contiguous Patch Masking：一次预测完整未来区间

传统 autoregressive forecasting：

```text
History
  ↓
预测 t+1
  ↓
把预测值放回输入
  ↓
预测 t+2
  ↓
...
```

存在：

- inference latency；
- exposure bias；
- error accumulation。

TimesFM-3 采用 Contiguous Patch Masking（CPM）。

给定：

```text
History                  Future
████████████████ | MASK MASK MASK MASK
```

未来 target patch 全部使用 mask token。

模型一次 forward 同时输出整个 future horizon：

$$
\hat X_{t+1:t+H}
=
F(X_{\le t},M_{future})
$$

而不是逐步递归产生。

这使：

$$
\epsilon_{t+1}
$$

不会直接作为输入继续污染：

$$
\hat x_{t+2}
$$

---

## 2.6 Past-only Covariates

TimesFM-3 支持只在历史阶段已知的额外变量，例如：

```text
Target:
sales

Past-only covariate:
historical foot traffic
```

预测时：

$$
C_{\le t}
$$

可进入模型，但未来：

$$
C_{>t}
$$

未知，因此 future 位置同样 masked。

---

## 2.7 Past-Future Covariates 与 Lookahead

TimesFM-3 还支持预测时已经知道未来值的变量，例如：

- 节假日；
- 已计划的促销；
- 天气预报；
- 调度信息。

这些变量满足：

$$
C_{1:t+H}
$$

预测时已知。

TimesFM-3 对其采用 lookahead token construction，使未来条件可以直接影响对应未来位置的预测。

本质是在建模：

$$
p(
Y_{future}
\mid
Y_{history},
C_{history},
C_{future-known}
)
$$

---

## 2.8 概率预测

TimesFM-3 不只输出 point forecast。

官方公开模型输出：

$$
q\in
\{
0.1,0.2,\ldots,0.9
\}
$$

共 9 个 quantile。

因此模型预测的不只是：

$$
\hat y
$$

而是近似：

$$
p(y_{future}\mid context)
$$

从而显式表达预测不确定性。

---

## 2.9 TimesFM-3 官方规模

官方公开配置包括：

```text
Transformer layers : 20
model dimension    : 1280
attention heads    : 16
input patch        : 32
output patch       : 64
max variates       : 32
quantiles          : 0.1 ... 0.9
```

这套规模不能直接迁移到 NeuroTwin。

---

# 3. TimesFM-3 与 NeuroTwin 的变量对应关系

TimesFM-3：

```text
Variate 1
Variate 2
...
Variate N
```

在 NeuroTwin 中天然对应：

```text
ROI 1
ROI 2
...
ROI 116
```

因此：

$$
\boxed{
TimesFM\ Variate
\Longleftrightarrow
Brain\ ROI
}
$$

同时：

$$
\boxed{
Temporal\ Attention
\Longleftrightarrow
ROI\ 内时间动力学
}
$$

$$
\boxed{
Variate\ Attention
\Longleftrightarrow
ROI\ 间脑网络交互
}
$$

这是 TimesFM-3 最值得 NeuroTwin 借鉴的地方。

---

# 4. 改进一：用 Temporal–ROI Alternating Block 重构 Backbone

## 4.1 当前 NeuroTwin 的问题

当前 NeuroTwin 主干仍继承旧 `[B,F,W,S]` window 任务设计，包括：

- BrainMDM；
- window-axis mixing；
- GraphODE；
- WindowTemporalAttention；
- 多个 graph / refinement 路径。

但 next-timepoint 实际输入已经近似：

$$
[B,F,1,K]
$$

即：

$$
W=1
$$

因此大量 window-axis 结构已经失去原始语义。

当前真正需要建模的只有两个维度：

$$
ROI\times Time
$$

---

## 4.2 推荐输入表示

去除人为 window 轴。

统一为：

$$
X\in
\mathbb R^{B\times F\times K}
$$

经过 patch embedding：

$$
H_0
\in
\mathbb R^{
B\times F\times P\times D
}
$$

其中：

- $F=116$；
- $P=K/p$；
- $p$：patch length；
- $D$：hidden dimension。

---

## 4.3 NeuroTwin 不应使用 TimesFM 的 patch=32

当前：

$$
K\le64
$$

如果：

$$
p=32
$$

则只有：

$$
64/32=2
$$

个 temporal tokens。

这几乎失去了 temporal attention 的意义。

建议：

$$
p\in\{4,8\}
$$

例如：

$$
K=64,p=4
\Rightarrow16\ temporal\ tokens
$$

或：

$$
K=64,p=8
\Rightarrow8\ temporal\ tokens
$$

推荐第一阶段直接消融：

```text
patch = 1
patch = 4
patch = 8
```

不应预设 patch 一定有效。

---

# 5. 改进二：Causal Temporal Attention

对于每个 ROI $i$：

$$
H_{i,:,:}
\in
\mathbb R^{P\times D}
$$

独立执行：

$$
H'_{i}
=
H_i+
\operatorname{TemporalAttn}
(
Norm(H_i)
)
$$

使用 causal mask：

$$
M_{jk}
=
\begin{cases}
0,& k\le j\\
-\infty,&k>j
\end{cases}
$$

从而保证：

$$
t_j
\not\leftarrow
t_{>j}
$$

### 为什么适合 BOLD

一个 ROI 的当前活动不仅由最后一个 TR 决定，还受到不同时间尺度历史状态影响。

相比固定卷积：

- attention 能自适应选择历史位置；
- causal mask 与 forecasting 语义一致；
- patch 可以降低短序列噪声敏感性。

但由于 $K\le64$，不需要超大型 long-context architecture。

---

# 6. 改进三：把 Full Variate Attention 改造成 SC-guided ROI Attention

## 6.1 为什么不能直接复制 TimesFM

TimesFM 不具备：

$$
SC_{ij}
$$

这种明确的物理解剖先验。

因此可以自由使用 full variate attention。

脑网络不同：

$$
ROI_i\leftrightarrow ROI_j
$$

受到结构连接显著约束。

但 SC 又不能作为 hard mask，因为：

- DTI 存在 false negative；
- 功能耦合可以通过 polysynaptic pathway 产生；
- 脑区间功能依赖不等于直接白质连接。

因此应该：

> 用 SC bias attention，而不是用 SC 禁止 attention。

---

## 6.2 推荐公式

普通 ROI attention：

$$
S_{ij}
=
\frac{q_i^\top k_j}{\sqrt d}
$$

加入结构先验：

$$
S^{brain}_{ij}
=
\frac{q_i^\top k_j}{\sqrt d}
+
\beta
\log(
\epsilon+A_{eff,ij}
)
$$

然后：

$$
A^{attn}_{ij}
=
Softmax_j(
S^{brain}_{ij}
)
$$

其中：

$$
A_{eff}
=
\lambda A_{SC}
+
(1-\lambda)A_{func}
+
\Delta A_{subject}
$$

---

## 6.3 三部分图信息

### Structural Prior

$$
A_{SC}
$$

来源于个体 DTI。

### Functional Adaptive Graph

由当前 BOLD latent 计算：

$$
A_{func}
=
Softmax(
Q_gK_g^\top/\sqrt d
)
$$

### Subject-specific Residual

低秩：

$$
\Delta A
=
UV^\top
$$

限制自由度，避免在 MDD 小样本中过拟合。

---

## 6.4 推荐 Block

最终一个 Block：

```text
Input H
  │
  ├── RMSNorm
  │
  ▼
Causal Temporal Attention
  │
Residual
  │
  ├── RMSNorm
  │
  ▼
SC-guided ROI Attention
  │
Residual
  │
  ├── RMSNorm
  │
  ▼
FFN
  │
Residual
```

记作：

**SC-guided Temporal–Variate Block**

堆叠：

$$
N=4\sim6
$$

即可。

---

# 7. 改进四：取消多处重复 SC Injection

TimesFM 的一个重要启示是：

> 跨变量关系应该成为主干的一种明确 interaction axis，而不是在网络多个位置零散加入。

当前 NeuroTwin 的 SC 先验已经出现在多个模块。

建议统一：

```text
SC
 │
 ▼
Soft Anatomical Prior
 │
 ▼
A_eff
 │
 └────────────────────────┐
                          ▼
               所有 ROI Attention
```

整个 backbone 共享：

$$
A_{eff}
$$

可以在层内加入轻量：

$$
\Delta A_l
$$

但不再每层重新构建完整 SC pipeline。

### 建议舍弃

- DFCAdapter 单独再做 SC diffusion；
- GraphODE 内再构建独立 SC mask；
- ForecastHead 再做独立 SC refiner；
- MoE 输出后再做另一套 SC refinement。

这些设计容易形成：

$$
SC\rightarrow SC\rightarrow SC\rightarrow SC
$$

而不是一个清晰的结构先验。

---

# 8. 改进五：加入 CPM-style Full-Horizon Prediction

## 8.1 当前问题

NeuroTwin 当前核心仍偏向：

$$
X_{\le t}
\rightarrow
x_{t+1}
$$

虽然支持 MTP 和 rollout，但：

- 单步任务容易学习 persistence；
- recursive rollout 会产生误差累积；
- 稀疏的 $\{1,2,4,8\}$ 目标不等同于学习完整 future trajectory。

---

## 8.2 新任务：Contiguous Future Masking

给定：

$$
X_{t-K+1:t}
$$

构造 future placeholder：

$$
M_{t+1:t+H}
$$

输入：

```text
Observed History             Future
x x x x x x x x | MASK MASK MASK MASK MASK
```

模型直接预测：

$$
\hat X_{t+1:t+H}
$$

即：

$$
p(
X_{t+1:t+H}
\mid
X_{t-K+1:t},
SC
)
$$

推荐：

$$
H\in\{4,8,16\}
$$

第一阶段优先：

$$
H=8
$$

---

# 9. 不能让 CPM 完全替代 One-Step Transition

这是 NeuroTwin 与普通 forecasting model 的关键区别。

如果只做 CPM：

$$
X_{history}
\rightarrow
X_{future}
$$

模型可以预测未来轨迹，但缺乏清晰的：

$$
z_t\rightarrow z_{t+1}
$$

状态转移接口。

而数字孪生未来需要：

```text
current twin state
      ↓
transition
      ↓
new observation
      ↓
assimilation
      ↓
updated twin state
```

也需要：

```text
state
 ↓
intervention
 ↓
next state
```

因此推荐：

# Dual Dynamics Objective

保留两条 head：

```text
                     Shared Encoder
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       One-Step Head              CPM Horizon Head
              │                         │
       z_t → z_(t+1)             t+1 ... t+H
              │
              ▼
   Assimilation / Intervention
```

---

# 10. One-Step Dynamics Head

首先得到当前状态：

$$
z_t
=
Pool_{time}(H)
$$

或者使用最后 causal token：

$$
z_t=H[:,:,P,:]
$$

状态转移：

$$
z_{t+1}
=
F_\theta(z_t,A_{eff},c)
$$

再：

$$
\hat x_{t+1}
=
D(z_{t+1})
$$

这条路径继续承担：

- next-state prediction；
- latent-state assimilation；
- virtual intervention；
- sequential simulation。

---

# 11. CPM Horizon Head

未来构造：

$$
H
$$

个 future positions。

每个未来位置 query：

$$
q_{i,h}
=
e^{ROI}_i
+
e^{horizon}_h
$$

其中：

- $i$：ROI；
- $h$：未来时间点。

利用 cross-attention：

$$
q_{i,h}
\rightarrow
H_{context}
$$

一次输出：

$$
\hat x_{i,t+h}
$$

形成：

$$
\hat X
\in
\mathbb R^{B\times F\times H}
$$

---

# 12. 改进六：将 TimesFM Covariate 设计映射到 NeuroTwin

TimesFM 的 covariate 分类非常适合 NeuroTwin。

---

## 12.1 Target Variables

目标：

$$
X^{BOLD}
$$

即 116 ROI BOLD。

---

## 12.2 Static Conditions

不随当前扫描时间改变的变量：

```text
SC
age
sex
site
diagnosis
HAMD
normative deviation
```

这些不应该直接当普通时间序列 token。

建议：

$$
c_{static}
=
ConditionEncoder(...)
$$

再用于：

- AdaLN / FiLM；
- MoE Router；
- graph mixing coefficient；
- decoder conditioning。

---

## 12.3 Past-only Covariates

只在历史阶段观测：

例如：

- framewise displacement；
- physiological measurements；
- arousal proxy；
- 已观测 medication state；
- 已观测行为/临床状态。

记：

$$
C^{past}_{\le t}
$$

与 BOLD context 同步编码，但未来位置 masked。

---

## 12.4 Future-known Covariates

这是 TimesFM 对 NeuroTwin 最有潜力的迁移之一。

未来如果进行刺激模拟：

$$
U_{t+1:t+H}
$$

在预测之前是人为设定的。

例如：

- stimulation target；
- stimulation amplitude；
- pulse train；
- frequency；
- timing。

这正对应 TimesFM 的：

> past-future covariate。

最终模型可以定义：

$$
p(
X_{future}
\mid
X_{history},
SC,
c_{patient},
U_{future}
)
$$

形成：

**intervention-conditioned forecasting**。

---

# 13. Intervention Lookahead 设计

假设：

$$
U
\in
\mathbb R^{F\times H\times D_u}
$$

对 future token 增加：

$$
q_{i,h}
=
e^{ROI}_i
+
e^{time}_h
+
E_u(U_{i,h})
$$

因此：

```text
BOLD history:
████████████ | MASK MASK MASK MASK

Intervention:
0 0 0 0 0 0 |  0    1    1    0
                        ↑
                future-known input
```

分别执行：

### Baseline Simulation

$$
U=0
$$

得到：

$$
X^{base}_{future}
$$

### Intervention Simulation

$$
U=U^{stim}
$$

得到：

$$
X^{stim}_{future}
$$

再分析：

$$
\Delta X
=
X^{stim}
-
X^{base}
$$

---

## 13.1 重要限制

如果训练数据中没有真实刺激 / intervention response：

**不能仅靠 observational rs-fMRI 学到真实 TMS treatment effect。**

因此第一阶段：

- 只预留 intervention interface；
- 可做 model sensitivity / mechanistic perturbation；
- 不宣称治疗效果预测。

必须有真实 pre/post stimulation 或 perturbation 数据后，才能训练或验证 intervention-conditioned branch。

---

# 14. 改进七：借鉴 Quantile Forecasting

## 14.1 为什么 NeuroTwin 需要概率预测

BOLD 的未来状态存在明显不确定性：

$$
p(x_{t+1}\mid history)
$$

不应被假设为单点确定值。

尤其：

- 长 horizon；
- MDD 高异质性；
- 高波动 ROI；
- virtual intervention；

都需要报告 uncertainty。

---

## 14.2 推荐第一版

不需要像 TimesFM 一样输出 9 个 quantiles。

先输出：

$$
q\in
\{0.1,0.5,0.9\}
$$

其中：

$$
q_{0.5}
$$

作为 point forecast。

---

## 14.3 Quantile Loss

对于 quantile $q$：

$$
L_q(y,\hat y_q)
=
\begin{cases}
q(y-\hat y_q),&y\ge\hat y_q\\
(1-q)(\hat y_q-y),&y<\hat y_q
\end{cases}
$$

总损失：

$$
L_{quantile}
=
\sum_q
L_q
$$

---

## 14.4 评估

增加：

- interval coverage；
- interval width；
- calibration error；
- horizon-wise uncertainty；
- ROI-wise uncertainty。

对于 virtual intervention：

不仅比较：

$$
E[\Delta x]
$$

还要比较：

$$
CI(\Delta x)
$$

防止把高不确定性 perturbation 误认为可靠结果。

---

# 15. 改进八：借鉴 CPM RevIN，但必须适配 forecasting boundary

TimesFM-3 官方模型使用 CPM Iterative RevIN。

NeuroTwin 当前也已经有 BrainRevIN，因此不需要重新发明 normalization。

真正要改的是：

> normalization statistics 从哪里来。

推荐：

$$
\mu_t,\sigma_t
=
Stats(
X_{t-K+1:t}
)
$$

只用历史 context。

然后：

$$
\tilde X_{history}
=
\frac{X_{history}-\mu_t}{\sigma_t}
$$

future target 也使用同一组历史统计量：

$$
\tilde X_{future}
=
\frac{X_{future}-\mu_t}{\sigma_t}
$$

禁止 target/future 参与：

$$
\mu,\sigma
$$

估计。

这不仅避免 leakage，也与未来 Twin Assimilation 相容。

---

# 16. 修改后的 NeuroTwin 整体架构

推荐最终主模型：

```text
                          Subject SC
                              │
                              ▼
                    Soft Anatomical Prior
                              │
                  Functional Adaptive Graph
                              │
                    Subject Graph Residual
                              │
                              ▼
                           A_eff
                              │
                              │
BOLD Context                  │
[B,116,K]                     │
     │                        │
     ▼                        │
Context-only RevIN            │
     │                        │
     ▼                        │
Temporal Patching             │
p = 4 / 8                     │
     │                        │
     ▼                        │
[B,116,P,D]                   │
     │                        │
     ▼                        │
┌─────────────────────────────────────────────┐
│ SC-guided Temporal–Variate Block × N       │
│                                             │
│  1. Causal Temporal Attention               │
│              ↓                              │
│  2. SC-guided ROI / Variate Attention ◄──── A_eff
│              ↓                              │
│  3. FFN                                     │
└─────────────────────────────────────────────┘
     │
     ▼
Latent Brain Representation H
     │
     ├──────────────────────────────────────┐
     │                                      │
     ▼                                      ▼
Current Latent State z_t              Future Mask Tokens
     │                                      │
     ▼                                      ▼
Pathology-conditioned              ROI × Horizon Query Decoder
Dynamics Adapter / MoE                   CPM Head
     │                                      │
     ▼                                      ▼
One-Step Transition                 Full Future Trajectory
z_t → z_(t+1)                       t+1 ... t+H
     │                                      │
     ▼                                      ▼
One-Step Prediction                Quantile Forecast
     │
     ├────► Latent Assimilation
     │
     └────► Sequential Simulation

Patient Conditions
HAMD + Brain State + Normative Deviation
             │
             └────► FiLM / Adapter / MoE Router

Future-known Intervention
u_(t+1:t+H)
             │
             └────► Future Query / Covariate Lookahead
```

---

# 17. 推荐模型规模

不要复制 TimesFM-3：

```text
20 layers
D=1280
16 heads
```

NeuroTwin 数据规模较小，建议：

```text
hidden dim D       = 128 / 192 / 256
layers N           = 4 / 6
temporal heads     = 4 / 8
ROI heads          = 4 / 8
patch size         = 4 / 8
dropout            = 0.1–0.2
```

首选起点：

```text
D = 192 or 256
N = 4
patch = 4
heads = 4
```

再通过验证集调优。

---

# 18. 新的任务设计

整个 NeuroTwin 建议不再只围绕一个 next-timepoint task。

---

## Task A：One-Step Brain-State Transition

输入：

$$
X_{t-K+1:t}
$$

目标：

$$
x_{t+1}
$$

用途：

- 学习局部状态转移；
- Twin Assimilation；
- Virtual Intervention；
- sequential rollout。

---

## Task B：CPM Full-Horizon Forecasting

输入：

$$
X_{t-K+1:t}
$$

未来：

$$
MASK_{t+1:t+H}
$$

目标：

$$
X_{t+1:t+H}
$$

推荐：

$$
H=8
$$

随后测试：

$$
H=4,8,16
$$

用途：

- 学习 future trajectory；
- 抑制 autoregressive error accumulation；
- 学习跨 horizon 依赖。

---

## Task C：Multi-Offset Prediction

继续保留：

$$
h\in\{1,2,4,8\}
$$

但其角色改为：

> 稀疏 horizon auxiliary supervision / evaluation。

不再承担全部 multi-horizon 学习。

---

## Task D：Masked Historical Modeling

HC pretraining 中随机 mask 历史 BOLD patch：

$$
X_{\mathcal M}
$$

要求模型重建。

目的：

- 学习稳健 HC representation；
- 降低单纯 persistence shortcut；
- 增强 normative representation。

---

## Task E：MDD Pathological Dynamics Adaptation

HC backbone 学：

$$
F_{HC}(z_t)
$$

MDD 学：

$$
F_{MDD}(z_t)
=
F_{HC}(z_t)
+
\Delta F(
z_t,
c_{state},
c_{dev},
c_{symptom}
)
$$

病理条件：

$$
c=
[
state,
normative\ deviation,
HAMD
]
$$

---

## Task F：Twin Assimilation

真实：

$$
x_{t+1}
$$

到达后：

$$
e_{t+1}
=
x_{t+1}-\hat x_{t+1}
$$

更新：

$$
z_{t+1}^{+}
=
z_{t+1}^{-}
+
K_\phi(
z^-,
e
)
$$

用途：

> 让数字孪生持续与患者真实状态同步。

---

## Task G：Intervention-conditioned Forecasting

仅在未来存在真实干预数据时作为监督任务：

$$
p(
X_{future}
\mid
X_{history},
SC,
c_{patient},
U_{future}
)
$$

无真实干预数据阶段只保留 interface，不作为治疗预测任务训练。

---

# 19. 新的训练目标

建议明确区分 HC 和 MDD 两阶段。

---

# 20. HC 阶段：Normative Dynamics Pretraining

总体：

$$
L_{HC}
=
\lambda_1L_{one}
+
\lambda_HL_{CPM}
+
\lambda_ML_{mask}
+
\lambda_QL_{quantile}
+
\lambda_GL_{graph}
$$

---

## 20.1 One-Step Loss

建议：

$$
L_{one}
=
L_{Huber}
+
\lambda_{pcc}
(
1-PCC_{ROI}
)
$$

不再同时使用数学等价的：

$$
L_{abs}+L_{delta}
$$

---

## 20.2 CPM Trajectory Loss

$$
L_{CPM}
=
\frac{
\sum_{h=1}^{H}
w_h
\operatorname{Huber}
(
\hat X_{t+h},X_{t+h}
)
}{
\sum_h w_h
}
$$

可以：

$$
w_h=\gamma^{h-1}
$$

例如：

$$
\gamma=0.9
$$

也可以先全部设为 1，通过实验确定。

---

## 20.3 Trajectory Pattern Loss

可加：

$$
L_{traj-pattern}
=
1-
PCC(
\hat X_{future},
X_{future}
)
$$

但需要分别报告：

- spatial PCC；
- temporal PCC；

避免语义混淆。

---

## 20.4 Quantile Loss

$$
L_Q
=
\sum_{q\in\{0.1,0.5,0.9\}}
L_q
$$

第一阶段可以先关闭：

$$
\lambda_Q=0
$$

待 point forecast 基线稳定后再加入。

---

## 20.5 Historical Mask Loss

$$
L_{mask}
=
\|
\hat X_{\mathcal M}
-
X_{\mathcal M}
\|
$$

建议 mask ratio：

```text
10%
20%
30%
```

消融。

---

## 20.6 Graph Regularization

对：

$$
A_{func},\Delta A
$$

加入轻量：

- sparsity；
- entropy；
- temporal consistency。

不应强迫：

$$
A_{func}\approx A_{SC}
$$

因为功能网络本身并不等于结构网络。

---

# 21. MDD 阶段：Pathology-conditioned Adaptation

MDD 总损失：

$$
L_{MDD}
=
L_{forecast}
+
\lambda_{moe}L_{MoE}
+
\lambda_{dev}L_{dev}
$$

其中：

$$
L_{forecast}
=
L_{one}
+
\lambda_HL_{CPM}
+
\lambda_QL_Q
$$

---

## 21.1 MoE regularization

继续保留当前合理部分：

- load balancing；
- router entropy；
- Z-loss。

但正式 evaluation 使用：

```text
dense-soft
或 deterministic top-k
```

不能使用 stochastic multinomial test routing。

---

## 21.2 Pathology Condition

推荐：

$$
c=
[
z_{state},
d_{norm},
HAMD_{factor}
]
$$

而不是只输入：

$$
HAMD_{total}
$$

---

# 22. Twin Assimilation 训练目标

Assimulation 阶段：

### Prior

$$
z^-_{t+1}
=
F(z_t^+)
$$

### Observation

$$
\hat x_{t+1}
=
D(z^-_{t+1})
$$

### Innovation

$$
e=
x_{t+1}-\hat x_{t+1}
$$

### Posterior

$$
z^+_{t+1}
=
z^-_{t+1}
+
C_\phi(e,z^-)
$$

训练时继续预测：

$$
\hat x_{t+2}
=
D(F(z^+_{t+1}))
$$

Assimilation loss：

$$
L_{assim}
=
L(
\hat x_{t+2:t+R},
x_{t+2:t+R}
)
$$

核心比较：

$$
Free\ Rollout
\quad vs\quad
Assimilation
$$

---

# 23. 推荐最终总目标

成熟版本可以写成：

$$
L
=
\lambda_{one}L_{one}
+
\lambda_{CPM}L_{CPM}
+
\lambda_{mask}L_{mask}
+
\lambda_{quant}L_{quant}
+
\lambda_{assim}L_{assim}
+
\lambda_{moe}L_{moe-reg}
+
\lambda_{graph}L_{graph-reg}
$$

但实现时：

**禁止第一版全部打开。**

推荐逐步加入。

---

# 24. 推荐训练顺序

## Stage 0：Baseline

只训练：

$$
L_{one}
$$

确定新 backbone 自身有效。

---

## Stage 1：Temporal–Variate Backbone

比较：

```text
TCN / old backbone
vs
Temporal Attention
vs
Temporal + ordinary ROI Attention
vs
Temporal + SC-guided ROI Attention
```

只有 SC-guided attention 确认有效后才进入主模型。

---

## Stage 2：CPM

增加：

$$
L_{CPM}
$$

观察：

- H1；
- H4；
- H8；
- H16；
- free rollout。

如果 H1 基本不损失而中长期明显改善，则保留。

---

## Stage 3：HC Masked Pretraining

增加：

$$
L_{mask}
$$

评估：

- HC forecasting；
- MDD transfer；
- low-data MDD fine-tuning。

---

## Stage 4：Pathology Adaptation

增加：

- normative deviation；
- HAMD；
- brain-state；
- MoE/adapters。

必须加入：

```text
true HAMD
shuffled HAMD
no HAMD
```

---

## Stage 5：Probability

加入：

$$
q_{0.1},q_{0.5},q_{0.9}
$$

完成 calibration。

---

## Stage 6：Assimilation

建立：

```text
prior → observation → correction → posterior
```

验证数字孪生同步能力。

---

## Stage 7：Intervention

只有拥有真实刺激数据后，才把：

$$
U_{future}
$$

升级为正式监督条件。

---

# 25. 必须做的消融实验

## 25.1 Patching

```text
No patch
patch=4
patch=8
patch=16
```

不要使用 TimesFM 默认 32 作为首选。

---

## 25.2 ROI Interaction

```text
No ROI mixing
Full ROI Attention
Hard-SC Attention
SC-biased Attention
SC + Functional Adaptive Attention
```

预期最值得保留的是：

$$
SC\text{-biased adaptive attention}
$$

---

## 25.3 Prediction Mode

```text
One-step only
MTP only
CPM only
One-step + CPM
One-step + CPM + rollout
```

重点观察：

- one-step accuracy；
- long-horizon stability；
- computational cost。

---

## 25.4 Condition

```text
No condition
HAMD
state
normative deviation
HAMD + state
HAMD + state + deviation
```

---

## 25.5 Uncertainty

```text
Point
Gaussian
Quantile 0.1/0.5/0.9
```

如果 quantile calibration 很差，则不要为了模仿 TimesFM 强行保留。

---

# 26. TimesFM-3 哪些设计不能直接照搬

## 26.1 不直接迁移 20-layer / D=1280 Transformer

原因：

- NeuroTwin 数据规模远小于 TimesFM pretraining corpus；
- 116 ROI × 医学数据更容易过拟合；
- 当前问题并没有 foundation model 级参数规模需求。

---

## 26.2 不直接使用 patch=32

原因：

$$
K\le64
$$

会造成 temporal token 数过少。

---

## 26.3 不直接使用纯 Full Variate Attention

原因：

NeuroTwin 有额外的：

$$
SC
$$

结构先验。

需要变成：

**SC-guided adaptive ROI attention**。

---

## 26.4 不把 CPM 完全替代 sequential transition

原因：

数字孪生需要明确：

$$
z_t\rightarrow z_{t+1}
$$

进行：

- assimilation；
- intervention；
- online synchronization。

---

## 26.5 不直接把 TimesFM checkpoint 当 NeuroTwin backbone

官方 TimesFM-3 配置：

```text
max_variates = 32
```

而 NeuroTwin：

```text
ROI = 116
```

同时训练域高度不同。

TimesFM 的一般跨变量先验并不天然等价于脑网络神经动力学。

可以把 TimesFM-3 当 forecasting baseline 或架构参考，但不建议作为核心模型初始化。

---

## 26.6 注意 TimesFM-3 权重许可证

当前官方 TimesFM-3 pretrained weights 使用单独的：

**TimesFM Non-Commercial License v1.0**

因此若后续涉及非研究用途，不能默认把官方权重作为可自由商用组件。

NeuroTwin 当前建议只是借鉴架构思想，不依赖其权重。

---

# 27. 与当前 NeuroTwin 模块的映射

| 当前模块 | 改进后 |
|---|---|
| BrainRevIN | 保留，改成 context-only statistics |
| DFCAdapter | 功能合并进统一 Soft Anatomical Prior / ROI attention |
| BrainMDM temporal pathway | 由 Temporal Attention / patch mixer 替代 |
| BrainMDM window pathway | 删除 |
| GraphODE temporal branch | 由 Temporal Attention 替代或降级消融 |
| GraphODE graph branch | 融入 SC-guided ROI Attention |
| ODE window attention | 删除 |
| FutureQueryDecoder | 保留思想，升级成 ROI × Horizon query |
| Iterative SC Refiner | 默认删除；必要时做消融 |
| SoftAnatomicalPrior | 保留并作为唯一 graph construction |
| NeuroTwinMoE | 保留，但改为 dynamics adapter / pathology residual |
| virtual_intervention | 改造成 future-known control interface |
| NextTimepointLoss | 重构成 one-step + CPM + optional quantile |

---

# 28. 最终推荐研究主线

借鉴 TimesFM-3 后，NeuroTwin 不应该变成：

> TimesFM for fMRI。

更合理的定位是：

> 一个面向脑数字孪生的、结构连接约束的多变量时空预测框架。

其核心区别在于：

TimesFM：

$$
Temporal\ Forecasting
+
Cross\text{-}Variate\ Dependency
$$

NeuroTwin：

$$
Temporal\ Brain\ Dynamics
+
SC\text{-}guided\ ROI\ Interaction
+
Pathological\ Deviation
+
Twin\ Assimilation
+
Virtual\ Intervention
$$

因此最终主线建议收敛为：

$$
\boxed{
Temporal\ Patching
\rightarrow
Causal\ Temporal\ Attention
\rightarrow
SC\text{-}guided\ Variate\ Attention
\rightarrow
Normative/Pathological\ Dynamics
\rightarrow
Dual\ Forecasting
\rightarrow
Twin\ Assimilation
}
$$

其中 Dual Forecasting 指：

$$
\boxed{
One\text{-}Step\ Transition
+
CPM\ Full\text{-}Horizon\ Forecast
}
$$

前者承担：

- 状态转移；
- assimilation；
- intervention。

后者承担：

- 长期 trajectory；
- non-autoregressive forecasting；
- 减少 error accumulation。

---

# 29. 推荐最终论文结构表达

模型部分不要写成多个独立模块堆叠。

可以收敛成四个核心机制。

## 29.1 Structure-aware Multivariate Dynamics

通过：

$$
Temporal\ Attention
+
SC\text{-}guided\ ROI\ Attention
$$

分别建模时间动力学与跨脑区交互。

---

## 29.2 Normative-to-Pathological Adaptation

先学习 HC：

$$
F_{HC}
$$

然后学习：

$$
F_{MDD}
=
F_{HC}
+
\Delta F_{pathology}
$$

---

## 29.3 Dual-Horizon Forecasting

联合：

$$
One\text{-}Step
+
CPM
$$

同时保证：

- 局部 transition 可解释；
- 长程 trajectory 稳定。

---

## 29.4 Adaptive Digital Twin

通过：

$$
Prediction
\rightarrow
Observation
\rightarrow
Assimilation
\rightarrow
Updated\ State
$$

实现持续个体同步。

未来再接：

$$
Intervention\ Lookahead
$$

完成反事实干预。

---

# 30. 优先实施版本

如果当前只允许做一轮结构升级，我建议不要一次实现所有内容。

## NeuroTwin-TFM V1

只做四项：

1. `[B,F,K]` + temporal patch；
2. Causal Temporal Attention；
3. SC-guided ROI Attention；
4. One-Step + CPM 双预测头。

暂时不加入：

- quantile；
- intervention tokens；
- assimilation；
- 新 MoE；
- SDE；
- 更复杂 graph learning。

先验证：

$$
\text{TimesFM-style inductive bias}
$$

是否真的适合 BOLD。

如果这个版本已经：

- H1 不下降；
- H4/H8/H16 明显改善；
- 参数量低于当前复杂 backbone；
- no-SC ablation 显著下降；

那么再将其作为 NeuroTwin V2 backbone。

---

# 31. 参考资料

1. Google Research. **TimesFM-3: A zero-shot foundation model for multivariate forecasting.** 2026-08-31.  
   https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/

2. Google Research. **TimesFM GitHub repository.**  
   https://github.com/google-research/timesfm

3. Google. **TimesFM 3.0 PyTorch Model Card.**  
   https://huggingface.co/google/timesfm-3.0-pytorch

4. Google. **TimesFM 3.0 configuration.**  
   https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/config.json

5. Das A, Kong W, Sen R, Zhou Y. **A Decoder-only Foundation Model for Time-series Forecasting.** ICML 2024 / arXiv:2310.10688.

---

# 32. 最终结论

TimesFM-3 最值得 NeuroTwin 借鉴的不是模型规模，而是以下四个 inductive biases：

$$
\boxed{
1.\ Temporal / Variate\ Axes\ Explicitly\ Separated
}
$$

$$
\boxed{
2.\ Non\text{-}autoregressive\ Full\text{-}Horizon\ Forecasting
}
$$

$$
\boxed{
3.\ Known\text{-}Future\ Covariate\ Conditioning
}
$$

$$
\boxed{
4.\ Probabilistic\ Forecasting
}
$$

针对 NeuroTwin，应分别转化成：

$$
\boxed{
Causal\ Temporal\ Attention
}
$$

$$
\boxed{
SC\text{-}guided\ ROI\ Attention
}
$$

$$
\boxed{
CPM\ Brain\ Trajectory\ Prediction
}
$$

$$
\boxed{
Intervention\ Lookahead
}
$$

$$
\boxed{
Quantile\ Brain\ State\ Forecasting
}
$$

最终不应构建一个“脑信号版 TimesFM”，而应该构建一个：

> **以 TimesFM-3 的多变量预测思想为基础、以结构连接为脑网络先验、以 HC→MDD 病理偏离为个体化机制，并能够进一步支持状态同步与虚拟干预的 NeuroTwin。**
