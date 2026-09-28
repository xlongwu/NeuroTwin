你是一名资深深度学习工程师、时序建模研究员和科研代码重构专家。请直接基于当前 NeuroTwin 项目完成一次系统性任务重构。

本次修改的核心目标是：

将当前基于滑动窗口的：

历史多个 BOLD window → 预测下一个 BOLD window

重构为：

连续 BOLD 历史序列 → 预测下一个时间点的全脑 BOLD 状态

即建立类似 autoregressive next-token prediction 的：

Next Brain-State Prediction / Next-Timepoint Prediction

训练范式。

注意：

这里的“下一个时间点”不是预测单个 ROI 的一个标量，而是预测下一个 TR 时刻所有 ROI 组成的完整全脑状态向量。

如果采用 AAL116，则：

x_t ∈ R^116

模型任务为：

p(x_{t+1} | x_1, ..., x_t, SC)

MDD 阶段进一步变成：

p(x_{t+1} | x_1, ..., x_t, SC, HAMD)

本次修改重点是：

1. 数据构造方式
2. Dataset / DataLoader
3. 模型输入输出接口
4. causal autoregressive training
5. next-timepoint loss
6. residual / delta prediction
7. evaluation
8. autoregressive free rollout
9. baseline
10. 为后续 Multi-Timepoint Prediction（MTP）预留接口

不要首先大规模重构整个模型 backbone。

━━━━━━━━━━━━━━━━━━━━
一、首先完整审计当前代码
━━━━━━━━━━━━━━━━━━━━

在修改任何代码之前，请完整阅读当前项目。

至少检查：

- README.md
- AGENTS.md
- main.py
- models/
- train/
- utils/
- scripts/
- analysis/
- 数据集定义
- 数据预处理逻辑
- HC pretraining
- MDD finetuning
- evaluation
- inference
- checkpoint
- config / argparse

不要只根据 README 推断实现。

重点回答：

1. 当前原始 BOLD 从哪里读取？
2. 原始 BOLD 的真实 shape 是什么？
3. 当前 window 是在哪一步构造的？
4. 当前 x=[F,6,S]、y=[F,1,S] 在哪里生成？
5. window size、in_window、pred_window 是否存在硬编码？
6. BrainRevIN 当前在哪一维进行 normalization？
7. DFCAdapter 输入的实际 shape 是什么？
8. BrainMDM 是否依赖固定 window 数量？
9. GraphODE / GraphODEDDI 当前处理的是 ROI、window 还是 hidden state？
10. ForecastHead 是否将 window 维 flatten？
11. 当前 HC 和 MDD Dataset 是否共用数据格式？
12. 当前 subject-level split 如何实现？
13. 当前 validation / test 是否会随机抽样？
14. 当前 loss 在哪些维度计算？
15. 当前模型是否可以直接处理 [B,T,F] 或 [B,F,T]？

先输出一份简短审计结果，然后再开始修改。

━━━━━━━━━━━━━━━━━━━━
二、重新定义核心任务
━━━━━━━━━━━━━━━━━━━━

废弃“下一个窗口预测”作为新任务的核心定义。

旧任务：

X_{t-L+1:t}
→
X_{t+1}

其中每个 X 是一个包含多个 TR 的 window。

新任务：

x_{1:t}
→
x_{t+1}

其中：

x_t ∈ R^F

是某一个真实 TR 时间点所有 ROI 的 BOLD 状态。

如果：

F = 116

则：

x_t ∈ R^116

完整 BOLD：

X ∈ R^(T×F)

或者项目内部保持：

X ∈ R^(F×T)

均可。

但是进入 temporal autoregressive backbone 后，推荐统一内部表示：

[B,T,F]

即：

batch
×
time
×
ROI

因为 autoregressive sequence modeling 时，T 是 sequence dimension。

━━━━━━━━━━━━━━━━━━━━
三、不要再把 window 作为 Dataset 的基本样本
━━━━━━━━━━━━━━━━━━━━

当前如果 Dataset 中存在：

x:
[F,6,30]

y:
[F,1,30]

请新增新的任务模式：

task_mode = next_timepoint

在该模式下：

Dataset 基本单位必须是完整 subject / session。

例如：

{
    "bold": [T,F],
    "sc": [F,F],
    "hamd": [1],
    "subject_id": ...,
    "tr": ...,
    "valid_mask": [T]
}

HC 可以没有 HAMD。

MDD 保留 HAMD。

不要先把一个 subject 拆成数百个独立：

history → next point

样本保存到磁盘。

一个 subject trajectory 必须保持完整。

━━━━━━━━━━━━━━━━━━━━
四、Subject-level split 必须保持
━━━━━━━━━━━━━━━━━━━━

train / validation / test 必须首先在 subject 级别划分。

严格保证：

train_subjects ∩ val_subjects = ∅

train_subjects ∩ test_subjects = ∅

val_subjects ∩ test_subjects = ∅

禁止：

先把完整 BOLD 切成时间片段，然后随机划分片段。

训练过程中可以从训练 subject 中随机抽 temporal context。

但是统计独立单位仍然是 subject。

请增加显式 sanity check。

━━━━━━━━━━━━━━━━━━━━
五、原始 BOLD 数据处理
━━━━━━━━━━━━━━━━━━━━

优先直接读取已经完成标准 rs-fMRI preprocessing 和 ROI extraction 的连续 BOLD。

基础形式：

bold:
[T,F]

例如：

[T,116]

需要检查：

1. NaN
2. Inf
3. 极端异常值
4. ROI 数是否与 SC 一致
5. ROI 顺序是否与 SC atlas 顺序一致
6. 每个 subject 的 TR
7. 序列长度 T 是否一致
8. scrubbing 后是否存在无效时间点

禁止为了 next-timepoint prediction 再进行 window averaging。

每个真实 TR 都应该保留下来。

━━━━━━━━━━━━━━━━━━━━
六、避免 normalization leakage
━━━━━━━━━━━━━━━━━━━━

重点检查 normalization。

不能使用：

整个 subject 的完整时间序列 mean/std

来归一化一个 forecasting task，因为这样统计量包含未来时间点信息。

如果使用 BrainRevIN：

优先修改成：

对于当前 context：

x_{t-K+1:t}

只根据历史 context 计算 mean/std。

target：

x_{t+1}

不能参与 normalization statistics。

预测结束后：

inverse normalization

也必须使用对应 history statistics。

如果当前 RevIN 逻辑无法满足这一点，请修复。

必须增加测试确认：

future target 没有参与 mean/std 计算。

━━━━━━━━━━━━━━━━━━━━
七、训练时使用 Context，而不是旧 Window
━━━━━━━━━━━━━━━━━━━━

虽然预测目标只有：

x_{t+1}

但是模型不能只输入：

x_t

建议使用一段历史：

x_{t-K+1:t}

预测：

x_{t+1}

其中 K 称为：

context length

而不是 prediction window。

第一版支持：

context_min = 16
context_max = 64

训练时可以随机采样：

K ∈ {16,...,64}

或者使用配置中的离散集合：

context_lengths = [16,32,64]

需要配置化。

不能把 32 或 64 写死。

例如：

输入：

[B,32,116]

target：

[B,116]

即：

过去 32 TR
→
下一 TR 的全脑状态。

━━━━━━━━━━━━━━━━━━━━
八、训练样本动态采样
━━━━━━━━━━━━━━━━━━━━

对于：

X = [x_1,...,x_T]

训练时动态采样：

context length K

以及预测位置 t。

满足：

t-K+1 >= 0

t+1 < T

生成：

history =
x_(t-K+1:t)

target =
x_(t+1)

例如：

K=32

history:
[x_51,...,x_82]

target:
x_83

同一个 subject 在不同 epoch 应该可以采样不同：

K
t

但：

validation / test 必须 deterministic。

━━━━━━━━━━━━━━━━━━━━
九、优先支持全序列 causal training
━━━━━━━━━━━━━━━━━━━━

如果当前 backbone 能够高效支持 causal sequence modeling，则优先实现一种类似 GPT teacher-forcing 的训练方式。

给定：

X =
[x_1,x_2,...,x_T]

构造：

input:

[x_1,...,x_(T-1)]

target:

[x_2,...,x_T]

模型通过 causal mask 保证：

位置 t 的隐藏状态只能使用：

x_1,...,x_t

不能访问：

x_(t+1:)

一次 forward 输出：

[x_hat_2,...,x_hat_T]

从而对多个时间位置同时计算 next-state loss。

如果由于现有 NeuroTwin 架构无法直接支持完整 causal sequence：

允许第一版采用：

random context → next point

方式训练。

优先级：

A. 全序列 causal parallel training

如果改动合理，则实现。

B. random context autoregressive training

如果 A 需要大规模破坏 backbone，则优先实现 B。

不要为了模仿 LLM 强行重写整个模型。

━━━━━━━━━━━━━━━━━━━━
十、Causal mask
━━━━━━━━━━━━━━━━━━━━

如果引入 Transformer / attention temporal modeling：

必须使用严格 causal mask。

对于预测：

x_(t+1)

隐藏状态 h_t 只能访问：

x_≤t

不能访问未来 BOLD。

增加测试验证：

changing x_(t+1:) must not change h_t

即未来时间点变化不能影响过去位置输出。

任何双向 attention 如果用于 prediction backbone，都必须进行检查。

如果 BrainMDM 当前是双向 temporal mixing，需要判断：

它是否会看到 future context。

如果只是处理已经截取好的 history context，则没有问题。

如果全序列一次训练，则必须保证 causal。

━━━━━━━━━━━━━━━━━━━━
十一、模型输入接口
━━━━━━━━━━━━━━━━━━━━

新的标准输入建议：

bold_history:
[B,K,F]

SC:
[B,F,F]

MDD:

HAMD:
[B,1]

target:

[B,F]

如果项目内部空间模块更适合：

[B,F,K]

可以在模型入口统一 transpose。

但是必须统一规范，禁止各模块任意交换维度。

建议定义清晰接口：

forward(
    bold_history,
    sc,
    hamd=None,
    ...
)

输出：

{
    "pred_next": [B,F],
    "pred_delta": [B,F],
    "latent": ...,
    ...
}

━━━━━━━━━━━━━━━━━━━━
十二、重点排查固定窗口相关代码
━━━━━━━━━━━━━━━━━━━━

搜索整个项目：

in_window
pred_window
window_size
num_windows
6
30

重点判断这些数字是否与：

6 historical windows
30 TR per window

相关。

特别检查：

reshape
view
flatten
Linear
Conv
positional embedding
ForecastHead

任何依赖：

6 × hidden_dim

或者：

30 × hidden_dim

的固定 shape 都必须处理。

但是注意：

不要机械替换所有数字 6/30。

只有与旧任务 shape 有关的硬编码才修改。

━━━━━━━━━━━━━━━━━━━━
十三、推荐采用 residual / delta prediction
━━━━━━━━━━━━━━━━━━━━

BOLD 在相邻 TR 之间高度相关。

因此直接预测：

x_hat_(t+1)

很容易退化成：

x_hat_(t+1) ≈ x_t

为了强化 dynamics learning，建议 ForecastHead 优先预测：

Δx_(t+1)
=
x_(t+1)-x_t

模型输出：

Δx_hat_(t+1)

最终：

x_hat_(t+1)
=
x_t + Δx_hat_(t+1)

实现配置：

prediction_target = absolute

或者：

prediction_target = delta

建议默认：

delta

但保留 absolute 作为 ablation。

━━━━━━━━━━━━━━━━━━━━
十四、Next-Timepoint Prediction Loss
━━━━━━━━━━━━━━━━━━━━

第一版不要立即加入过多复杂 loss。

保留当前项目已有且合理的：

MAE
PCC

以及可能已有的：

difference loss
std loss

重新定义为 next-timepoint objective。

基础：

L_next =
L_MAE(x_hat_(t+1), x_(t+1))
+
lambda_pcc * L_PCC(...)

但是注意：

对于单个时间点：

x_(t+1) ∈ R^116

PCC 如果沿 ROI 维计算，其含义变成：

predicted spatial pattern
vs
true spatial pattern

请明确当前 PCC 的计算维度。

不要直接复用原来针对：

[F,S]

时序 waveform 的 PCC 而不检查语义。

需要分别定义：

1. spatial PCC：
在 ROI 维比较一个时间点的空间 pattern

2. temporal PCC：
在多个预测时间点聚合后沿时间维比较 ROI trajectory

训练阶段如果 single-step：

优先：

MAE / MSE / Huber

作为稳定主损失。

PCC 可作为辅助。

━━━━━━━━━━━━━━━━━━━━
十五、增加 Delta Loss
━━━━━━━━━━━━━━━━━━━━

因为采用 residual prediction，增加：

真实变化：

delta_true =
x_(t+1) - x_t

预测变化：

delta_pred

定义：

L_delta =
MAE(delta_pred, delta_true)

建议：

L =
L_next
+
lambda_delta * L_delta

如果 prediction_target=delta：

L_delta 可以直接作为主要 loss。

配置：

lambda_delta

默认建议从：

1.0

开始。

不要锁死数值。

━━━━━━━━━━━━━━━━━━━━
十六、增加 Persistence Baseline
━━━━━━━━━━━━━━━━━━━━

必须实现最重要的 baseline：

Persistence：

x_hat_(t+1) = x_t

这是 next-timepoint BOLD prediction 必须比较的基线。

因为 BOLD 高度自相关。

如果 NeuroTwin 连 persistence 都不能稳定超过，则 next-state task 没有证明复杂模型的价值。

Evaluation 中必须同时报告：

NeuroTwin
Persistence

━━━━━━━━━━━━━━━━━━━━
十七、增加 Linear Trend Baseline
━━━━━━━━━━━━━━━━━━━━

第二个 baseline：

x_hat_(t+1)
=
x_t + (x_t - x_(t-1))

即简单线性趋势外推。

同样计算所有 next-state metrics。

━━━━━━━━━━━━━━━━━━━━
十八、增加 AR Baseline
━━━━━━━━━━━━━━━━━━━━

如果实现成本合理：

加入：

AR(1)

至少支持：

每个 ROI 独立 AR(1)

例如：

x_(t+1)^i
=
a_i x_t^i + b_i

所有参数必须只使用 training subjects 拟合。

禁止 test leakage。

如果方便，也可以后续增加 VAR。

优先级：

Persistence
>
Linear trend
>
AR(1)
>
VAR

━━━━━━━━━━━━━━━━━━━━
十九、Free Rollout Evaluation
━━━━━━━━━━━━━━━━━━━━

虽然训练目标主要是：

next-state prediction

但是数字孪生不能只评估：

t → t+1

必须增加 free autoregressive rollout。

给定真实 context：

x_(t-K+1:t)

预测：

x_hat_(t+1)

然后把：

x_hat_(t+1)

作为模型下一步输入的一部分，再预测：

x_hat_(t+2)

继续：

x_hat_(t+3)
...
x_hat_(t+H)

中间禁止重新使用 ground truth。

默认评估：

rollout_horizons =
[1,2,4,8,16]

如果序列长度允许，可以增加：

32。

这些都需要配置化。

━━━━━━━━━━━━━━━━━━━━
二十、Rollout Metrics
━━━━━━━━━━━━━━━━━━━━

分别报告：

H=1
H=2
H=4
H=8
H=16

至少：

MAE
RMSE

如果合理：

spatial PCC
temporal PCC
R2

同时输出：

performance degradation

例如：

MAE_H1
MAE_H2
MAE_H4
MAE_H8
MAE_H16

以及：

相对于 Persistence baseline 的 improvement。

不要只给 overall average。

━━━━━━━━━━━━━━━━━━━━
二十一、增加 trajectory-level evaluation
━━━━━━━━━━━━━━━━━━━━

对于 rollout：

Y_true:
[H,F]

Y_pred:
[H,F]

除了逐点 MAE，还增加：

trajectory MAE

ROI-wise temporal correlation

global temporal correlation

variance ratio

即：

Var(pred) / Var(true)

因为 autoregressive model 很容易产生：

variance collapse

最终趋向平滑均值。

如果预测随着 rollout 越来越平：

必须能够通过 metric 被发现。

━━━━━━━━━━━━━━━━━━━━
二十二、FC evaluation
━━━━━━━━━━━━━━━━━━━━

next-timepoint 本身不能直接计算有意义的 FC。

因此 FC evaluation 必须建立在一段 rollout trajectory 上。

例如：

rollout 32 或 64 TR：

Y_pred:
[H,F]

Y_true:
[H,F]

计算：

FC_pred =
Corr(Y_pred)

FC_true =
Corr(Y_true)

比较：

upper triangular off-diagonal edges

至少输出：

FC edge PCC
FC MAE

如果 H 太短：

例如 H=4

不要计算 FC。

设置：

fc_min_length

例如：

32

只有 rollout 长度 >= fc_min_length 才计算。

━━━━━━━━━━━━━━━━━━━━
二十三、Spectral Evaluation
━━━━━━━━━━━━━━━━━━━━

如果实现成本不高：

对较长 rollout：

Y_pred
Y_true

增加：

PSD / frequency-domain evaluation

例如：

Welch PSD

比较主要低频段。

第一阶段只作为 metric。

不必立即加入训练 loss。

优先级低于：

next-state
rollout
baseline
FC。

━━━━━━━━━━━━━━━━━━━━
二十四、HC Training
━━━━━━━━━━━━━━━━━━━━

HC 阶段任务变成：

p(
x_(t+1)
|
x_(t-K+1:t),
SC
)

即：

SC-conditioned normal brain dynamics learning。

不要再使用：

6 windows → 1 window

作为默认主任务。

保留旧任务用于 ablation 即可。

━━━━━━━━━━━━━━━━━━━━
二十五、MDD Finetuning
━━━━━━━━━━━━━━━━━━━━

保留现有：

HC pretraining
→
MDD finetuning

逻辑。

MDD 阶段：

p(
x_(t+1)
|
history,
SC,
HAMD
)

保留当前 pathology MoE / residual mechanism。

但是需要重新检查：

原来的 pathology residual 是否输出：

future window

现在应该改成：

next-state residual

或者：

next-state delta correction。

更合理的形式为：

delta_pred =
delta_HC
+
delta_MDD

其中：

delta_HC

来自通用 HC dynamics。

delta_MDD

来自 pathology-conditioned MoE。

这比直接对整个 future window 做 residual 更符合新的任务定义。

━━━━━━━━━━━━━━━━━━━━
二十六、MoE Router
━━━━━━━━━━━━━━━━━━━━

第一版尽量保持当前 router 行为。

如果当前：

router_condition = HAMD

则先兼容。

同时为后续预留：

router_condition = HAMD + current latent state

接口。

如果实现成本很低，可以加入开关：

--moe_router_use_state

但本次 P0 不要求强制启用。

━━━━━━━━━━━━━━━━━━━━
二十七、重新定义 GraphODE 的职责
━━━━━━━━━━━━━━━━━━━━

不要为了本次任务强行重写 GraphODE。

但是需要分析：

next-timepoint task 下，

GraphODE 当前是否能够自然解释成：

z_t
→
z_(t+1)

如果 TR 已知：

Δt = TR

那么未来可以使：

GraphODE(z_t, SC, Δt)

真正对应一个实际时间间隔的 transition。

本次至少：

1. 检查 GraphODE 是否使用真实 t / Δt
2. 检查是否只是重复 autonomous block
3. 在代码报告中说明

如果当前没有真实 continuous-time semantics：

不要伪装成已经实现。

先保留现有模块。

后续再独立重构。

━━━━━━━━━━━━━━━━━━━━
二十八、为 MTP 预留结构
━━━━━━━━━━━━━━━━━━━━

本次主要任务是：

Next-Timepoint Prediction。

但是模型接口要为后续：

Multi-Timepoint Prediction

预留扩展能力。

未来希望支持：

x_≤t
→
x_(t+1)
x_(t+2)
x_(t+4)
x_(t+8)

因此 ForecastHead 不要设计成只能永久输出一个固定 scalar/vector。

建议设计：

forecast_offsets = [1]

默认：

[1]

未来可以直接改成：

[1,2,4,8]

对应：

Multi-Timepoint Prediction / Multi-Scale Temporal Prediction。

本次不要求完全实现复杂 sequential MTP。

但是：

数据接口
loss interface
evaluation interface

尽量不要阻止未来扩展。

━━━━━━━━━━━━━━━━━━━━
二十九、可选实现 Parallel MTP
━━━━━━━━━━━━━━━━━━━━

如果完成 next-timepoint 主任务以后代码稳定，并且实现成本较低，可以额外增加：

--enable_mtp

默认：

false

开启后：

forecast_offsets = [1,2,4,8]

模型从同一个 causal hidden state：

h_t

分别预测：

x_(t+1)
x_(t+2)
x_(t+4)
x_(t+8)

第一版采用：

Parallel MTP

不要一开始实现复杂 sequential MTP。

定义：

L_MTP =
sum_k w_k L(x_hat_(t+delta_k), x_(t+delta_k))

例如默认：

offsets:
1 2 4 8

weights:
1.0 0.7 0.5 0.3

必须配置化。

如果此次修改会显著增加风险：

优先完成 NTP。

MTP 可以只留下接口和 TODO。

━━━━━━━━━━━━━━━━━━━━
三十、Validation / Test 必须 deterministic
━━━━━━━━━━━━━━━━━━━━

训练时：

可以随机：

subject
context length
prediction location

Validation/Test：

必须固定。

建议：

对于每个 subject：

选择固定 context length。

例如：

eval_context_length = 64

然后枚举所有合法预测点：

t

或者均匀选择固定若干 anchor points。

每次 evaluation 的：

subject
context
target

必须完全一致。

禁止：

每次 validation 随机位置不同。

━━━━━━━━━━━━━━━━━━━━
三十一、Subject-level metric aggregation
━━━━━━━━━━━━━━━━━━━━

一个 subject 内有很多预测位置。

这些位置不是独立被试。

所以：

先计算一个 subject 内所有时间点的指标：

subject MAE
subject RMSE
subject temporal PCC
subject rollout metrics

然后：

在 subjects 之间求：

mean
std

论文统计必须以 subject 为基本独立单位。

同时可以保存：

timepoint-level metrics

用于调试，但不能把它们直接当作独立样本进行统计显著性分析。

━━━━━━━━━━━━━━━━━━━━
三十二、推荐的训练 Loss 第一版
━━━━━━━━━━━━━━━━━━━━

第一版建议保持简单：

如果使用 delta prediction：

L_total =
lambda_abs * L_abs
+
lambda_delta * L_delta
+
lambda_pcc * L_spatial_pcc

其中：

L_abs =
MAE(x_hat_(t+1), x_(t+1))

L_delta =
MAE(
delta_hat,
x_(t+1)-x_t
)

L_spatial_pcc：

比较预测和真实下一个时间点的 ROI spatial pattern。

默认参数可以：

lambda_abs = 1.0
lambda_delta = 1.0
lambda_pcc = 0.1

但必须配置化。

不要把这些值写死。

当前已有 uncertainty weighting 如果与新 loss 兼容，可以保留。

如果不兼容：

第一版先不用复杂 weighting。

━━━━━━━━━━━━━━━━━━━━
三十三、第二阶段可增加 rollout loss
━━━━━━━━━━━━━━━━━━━━

不要一开始强制进行长 autoregressive backpropagation。

先把：

next-state training

跑稳定。

之后增加可选：

--enable_rollout_loss

随机选择一个 context。

模型：

history
→ x_hat_(t+1)

然后：

history + x_hat_(t+1)
→ x_hat_(t+2)

最多先：

rollout_train_steps = 2 或 4

定义：

L_rollout =
sum_k D(x_hat_(t+k), x_(t+k))

最终：

L =
L_next
+
lambda_rollout L_rollout

默认：

关闭或低权重。

避免第一版训练不稳定。

━━━━━━━━━━━━━━━━━━━━
三十四、配置参数
━━━━━━━━━━━━━━━━━━━━

至少新增或统一：

task_mode

context_min
context_max
context_lengths

prediction_target
    absolute / delta

causal_training

forecast_offsets

enable_mtp

mtp_weights

enable_rollout_loss
rollout_train_steps
lambda_rollout

lambda_abs
lambda_delta
lambda_pcc

eval_context_length
eval_rollout_horizons

eval_fc
fc_min_length

eval_spectral

sampling_seed

所有参数进入统一 config / argparse。

不要散落硬编码。

━━━━━━━━━━━━━━━━━━━━
三十五、兼容旧任务
━━━━━━━━━━━━━━━━━━━━

尽量保留：

task_mode = window_forecast

或者当前旧模式名称。

新增：

task_mode = next_timepoint

使后续可以公平比较：

Old:

6 windows → 1 window

New:

continuous history → next TR

进一步：

New + MTP

不要删除旧任务所有代码。

不要导致以前实验完全无法复现。

━━━━━━━━━━━━━━━━━━━━
三十六、Checkpoint Compatibility
━━━━━━━━━━━━━━━━━━━━

因为 ForecastHead shape 很可能改变：

旧 checkpoint 可能不能完整加载。

必须明确处理。

不要：

strict=False

然后静默忽略大量参数。

如果 backbone 可以复用：

允许：

load backbone
reinitialize forecast head

但是必须：

明确打印：

loaded parameters
missing parameters
reinitialized modules

最好增加：

--load_backbone_only

模式。

━━━━━━━━━━━━━━━━━━━━
三十七、训练日志
━━━━━━━━━━━━━━━━━━━━

至少输出：

train/loss_total

train/loss_abs

train/loss_delta

train/loss_pcc

如果启用 MTP：

train/loss_t+1
train/loss_t+2
train/loss_t+4
train/loss_t+8

如果启用 rollout：

train/loss_rollout

Validation：

val/next_MAE
val/next_RMSE
val/next_PCC

val/persistence_MAE
val/trend_MAE

如果进行 rollout：

val/rollout_MAE_H1
val/rollout_MAE_H2
val/rollout_MAE_H4
val/rollout_MAE_H8
val/rollout_MAE_H16

━━━━━━━━━━━━━━━━━━━━
三十八、Checkpoint selection
━━━━━━━━━━━━━━━━━━━━

第一版不要使用单纯 PCC 选最佳模型。

推荐：

validation next-state MAE

或者：

标准化后的：

next-state loss + short rollout loss

如果 rollout 尚未加入：

使用：

val next-state MAE

作为主 checkpoint criterion。

同时记录其他指标。

最终说明采用了什么标准。

━━━━━━━━━━━━━━━━━━━━
三十九、需要实现的 Baseline Evaluation
━━━━━━━━━━━━━━━━━━━━

至少：

A. Persistence

x_hat_(t+1) = x_t

B. Linear Trend

x_hat_(t+1)
=
x_t+(x_t-x_(t-1))

C. NeuroTwin

如果实现：

D. AR(1)

对于 rollout：

所有 baseline 都应该递归生成。

例如 persistence rollout：

未来所有时刻均保持最后真实状态。

Trend rollout：

使用自身预测继续外推。

这样比较才公平。

━━━━━━━━━━━━━━━━━━━━
四十、核心 sanity checks
━━━━━━━━━━━━━━━━━━━━

必须增加至少以下测试：

1. train/val/test subjects 无交叉

2. target 时间点始终严格晚于 context

3. context 不含 target

4. future 不参与 normalization statistics

5. SC ROI 数和 BOLD ROI 数一致

6. ROI 顺序检查

7. 输入 shape 正确

8. target shape 正确

9. batch size > 1 正常

10. variable context 正常

11. causal mask 无未来泄漏

12. delta reconstruction 正确：

x_t + delta_true == x_(t+1)

13. model forward 正常

14. loss 为有限值

15. backward 正常

16. optimizer step 正常

17. validation deterministic

18. rollout 不读取未来 ground truth

19. HC 模式不强制要求 HAMD

20. MDD 模式正确使用 HAMD

21. checkpoint save/load 正常

22. persistence baseline 正常

23. old task_mode 仍能启动

━━━━━━━━━━━━━━━━━━━━
四十一、必须完成的 Smoke Tests
━━━━━━━━━━━━━━━━━━━━

修改完成后实际运行：

Test 1：

加载一个 HC subject 原始 BOLD。

打印：

BOLD shape
SC shape
TR
subject_id

Test 2：

生成一个：

K=32

的 context：

history:
[B,32,116]

target:
[B,116]

Test 3：

执行：

model.forward()

输出：

pred_next:
[B,116]

Test 4：

计算：

loss

确保 finite。

Test 5：

backward。

Test 6：

optimizer.step()。

Test 7：

运行至少一个小型 validation loop。

Test 8：

运行：

rollout H=8

检查：

[B,8,116]

Test 9：

Persistence baseline。

Test 10：

如果启用 MTP：

检查：

forecast_offsets=[1,2,4,8]

各 prediction shape 正确。

如果没有实际 GPU，也必须至少在 CPU 上运行最小 smoke test。

━━━━━━━━━━━━━━━━━━━━
四十二、推荐实现优先级
━━━━━━━━━━━━━━━━━━━━

P0：

- 读取连续 BOLD
- subject-level Dataset
- subject-level split
- context → next point
- variable context
- normalization leakage 修复
- model next-state output
- delta prediction
- next-state loss
- training

P1：

- deterministic validation
- persistence baseline
- trend baseline
- subject-level metrics
- free rollout
- horizon metrics

P2：

- FC rollout evaluation
- spectral evaluation
- AR baseline
- rollout training loss

P3：

- Parallel MTP
- state-aware MoE router
- GraphODE real Δt
- Sequential MTP
- probabilistic next-state prediction

必须优先完成 P0/P1。

不要因为 P2/P3 阻塞核心任务。

━━━━━━━━━━━━━━━━━━━━
四十三、最终新的数据流
━━━━━━━━━━━━━━━━━━━━

最终希望代码中的核心数据流变成：

Raw subject BOLD

[T,F]

↓

Subject-level Dataset

↓

Random / causal history

[K,F]

↓

SC-conditioned NeuroTwin backbone

↓

Current latent brain state

h_t / z_t

↓

Next-state dynamics prediction

delta_hat_(t+1)

↓

Residual reconstruction

x_hat_(t+1)
=
x_t + delta_hat_(t+1)

↓

Next-state supervision

x_(t+1)

训练阶段：

x_(≤t)
→
x_(t+1)

测试阶段：

x_(≤t)
→
x_hat_(t+1)
→
x_hat_(t+2)
→
...
→
x_hat_(t+H)

━━━━━━━━━━━━━━━━━━━━
四十四、最终研究任务定义
━━━━━━━━━━━━━━━━━━━━

修改以后，NeuroTwin 的核心任务不再表述为：

“根据若干历史 BOLD windows 预测下一个 BOLD window。”

而应该变成：

“Given a subject's historical whole-brain BOLD states and structural connectivity, NeuroTwin learns the conditional transition dynamics of brain activity by autoregressively predicting the next whole-brain state.”

HC 阶段：

学习一般脑状态转移动力学：

p(
x_(t+1)
|
x_(≤t),
SC
)

MDD 阶段：

学习病理条件下的个体化动力学偏移：

p(
x_(t+1)
|
x_(≤t),
SC,
HAMD
)

训练采用：

next-state autoregressive objective

评估重点不只包括：

next-step accuracy

还必须包括：

multi-step free rollout stability。

━━━━━━━━━━━━━━━━━━━━
四十五、MTP 后续扩展目标
━━━━━━━━━━━━━━━━━━━━

代码设计需要允许后续自然扩展为：

forecast_offsets =
[1,2,4,8]

即：

x_≤t
→
{
x_(t+1),
x_(t+2),
x_(t+4),
x_(t+8)
}

其目的不是简单模仿语言模型的 multi-token prediction，而是通过：

multi-timepoint prediction

减少模型只利用相邻 BOLD 平滑性的 shortcut。

本次如果只完成：

forecast_offsets=[1]

完全可以接受。

但是不要把接口设计死。

━━━━━━━━━━━━━━━━━━━━
四十六、最终需要提交给我的实施报告
━━━━━━━━━━━━━━━━━━━━

代码修改完成后，请给出详细报告。

不要只回复：

“修改完成”。

报告必须包括：

1. 原任务分析

说明原来的：

window 构造
x/y
模型输入
loss
evaluation

2. 修改文件列表

逐文件说明。

3. 新 Dataset

说明：

原始 BOLD shape

Dataset 返回内容

subject split

context sampling。

4. 新任务定义

明确：

history shape
target shape
prediction shape。

5. Normalization

说明如何保证未来 target 不泄漏。

6. 模型修改

说明：

哪些 backbone 保留

哪些模块修改

ForecastHead 如何修改

delta prediction 如何实现。

7. Loss

给出准确数学含义：

absolute loss

delta loss

PCC loss。

8. Evaluation

说明：

next-state
rollout
Persistence
Trend
FC
subject aggregation。

9. MTP

说明：

本次是否实现

如果没有：

哪些接口已经预留。

10. Compatibility

说明：

旧任务能否运行

旧 checkpoint 如何处理。

11. Smoke Test

列出每个测试结果。

12. Remaining Issues

必须诚实列出：

仍然存在的问题

未完成事项

潜在风险。

━━━━━━━━━━━━━━━━━━━━
四十七、最后的执行要求
━━━━━━━━━━━━━━━━━━━━

请现在直接执行：

第一步：
完整审计当前 NeuroTwin 代码。

第二步：
明确找出旧 window-based forecasting 的所有相关代码。

第三步：
设计最小侵入式 next-timepoint task 改造方案。

第四步：
修改 Dataset。

第五步：
修改模型输入输出。

第六步：
修改训练 loss。

第七步：
修改 validation / test。

第八步：
增加 persistence / trend baseline。

第九步：
增加 free rollout evaluation。

第十步：
完成所有 smoke tests。

第十一步：
更新 README 或新增：

docs/next_timepoint_forecasting.md

第十二步：
输出完整实施报告。

不要只提供设计建议。

不要停下来询问是否继续。

如果发现某个 P2/P3 功能实现风险过高：

先跳过该增强功能，

保证 P0/P1 完整可运行，

并在最终报告明确说明。

最终最重要的目标是：

让 NeuroTwin 真正从：

window-to-window regression

转变为：

continuous BOLD autoregressive brain-state dynamics learning。