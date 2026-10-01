# NeuroTwin Next-Timepoint Autoregressive Task 重构方案

# 0. 实现状态总览（2026-09-28 核对，以当前代码为准）

> 本节为事后核对补充；正文各节保留原始设计表述（Version1，0921）供溯源。现行任务协议与参数以 [docs/next_timepoint_forecasting.md](../next_timepoint_forecasting.md) 为准。
>
> 标记含义：**【已实现】**代码已落地 ｜ **【部分实现】**接口或子集已落地、其余未做 ｜ **【未实现】**按计划暂缓（P3 等）｜ **【已删除】**随 2026-09-27 任务口径统一被整体移除或被推翻。

| 章节 | 状态 | 与当前代码的对应 / 差异 |
|---|---|---|
| 1 文档目的 | 【已实现】 | 重构已落地，`next_timepoint` 为唯一任务口径（`main.py` 默认值） |
| 2 当前任务定义（旧 6→1 滑窗） | 【已删除】 | 旧任务连同代码、评估模块、脚本与文档已于 2026-09-27 整体移除 |
| 3 旧任务问题分析 | 【已实现】 | 动机结论仍有效，作为设计依据保留 |
| 4 新任务定义 | 【已实现】 | 数据流与 shape 见协议文档 §1；`NextPointTrainView` / `NextPointEvalView` |
| 5 数据组织 | 【已实现】 | Subject 级 Dataset + 动态 context 采样，不再离线生成滑窗样本 |
| 6 Subject-Level Split | 【已实现】 | 版本化 `subject_split_<mode>.json` manifest，配置不一致直接报错 |
| 7 Historical Context | 【已实现】 | `K∈[--context_min,--context_max]`（默认 16–64）随机采样或 `--context_lengths` 离散集合；按 K 分桶组 batch |
| 8 两种训练实现方式 | 【部分实现】 | 仅「随机 context→下一时间点」落地；full-sequence causal 未实现（`--causal_training full_sequence` 显式报错）；本节方案 A/B 命名与协议文档 §9.1 相反 |
| 9 Causal Constraint | 【部分实现】 | 因果性由数据构造保证（未来不进前向）；独立 causal leakage test 脚本未实现 |
| 10 Normalization | 【部分实现】 | BrainRevIN 仅用当前 context 统计量（含反归一化）；`data/Normlize` 逐 ROI 全序列 z-score 属预处理固定仿射，模型侧无法撤销（协议文档 §2.1） |
| 11 模型标准输入输出 | 【已实现】 | 入口一次 transpose 至 `[B,F,1,K]`；W=1、S=K 前缀切片；单点输出保护 |
| 12 Residual / Delta Prediction | 【已实现】 | `--prediction_target delta`（默认，x_t 锚点）/ `absolute`（零锚点消融） |
| 13 新训练目标 | 【已实现】 | `NextTimepointLoss`：L_abs+L_delta+L_pcc（λ 默认 1/1/0.1）；delta 锚点下 L_delta 与 L_abs 数值重合（协议文档 §4）；NLL 默认关闭 |
| 14 旧 Loss 的处理 | 【已删除】 | 随旧任务删除 |
| 15 HC Pretraining | 【已实现】 | `scripts/Pretrain_HC_next_point.sh` |
| 16 MDD Finetuning | 【已实现】 | `scripts/Finetune_MDD_next_point.sh`；病理残差 = 下一状态残差 delta_HC + delta_MDD |
| 17 Pathology MoE | 【部分实现】 | HAMD-conditioned routing 保留；state-aware router（HAMD+z_t）未实现（P3） |
| 18 GraphODE 重新定位 | 【部分实现】 | 审计结论成立（`t` 被丢弃、自治向量场）；真实 Δt 未实现，未伪装连续时间语义（协议文档 §9.2） |
| 19–20 评估与 Next-State 指标 | 【已实现】 | `analysis/next_point_eval.py`：MAE/RMSE/spatial PCC/R² + delta_direction，被试级聚合 |
| 21–23 Baselines | 【已实现】 | persistence / linear trend / AR(1)（仅 train subjects 拟合，n_pairs 入报告）；VAR 未实现 |
| 24–25 Free Rollout | 【已实现】 | horizons 1/2/4/8/16；MAE/RMSE/spatial PCC/temporal PCC/R²/variance ratio + degradation + vs persistence improvement；baselines 递归生成 |
| 26 FC 评估 | 【已实现】 | rollout ≥ `--fc_min_length`(32) 才计算；上三角边 fc_mae/fc_rmse/edge_pcc + within/between-network（AAL116「对应网络」列，缺失自动跳过） |
| 27 Spectral 评估 | 【已实现】 | `--eval_spectral` 可选（Welch PSD） |
| 28 Val/Test Protocol | 【已实现】 | 固定 K_eval + 确定性 anchor（`--eval_anchors_per_subject`，0=枚举全部） |
| 29 Subject-Level 聚合 | 【已实现】 | `per_subject_metrics_<split>.csv` 主表、per_task 供调试 |
| 30–31 MTP | 【部分实现】 | Parallel MTP 已实现（`--enable_mtp` + `--mtp_weights`，同一 hidden 并行输出多偏移）；Sequential MTP 未实现（P3） |
| 32 Short Rollout Training | 【已实现】 | `--enable_rollout_loss`（默认关闭；steps=2、λ_rollout=0.2，成本约 (1+steps) 倍） |
| 33 新配置参数 | 【已实现】 | CLI 参数名与设计一致（`main.py --help` / 协议文档 §6） |
| 34 兼容旧任务 | 【已删除】 | 被后续决策推翻：旧任务（window_forecast）完全删除，`--task_mode` 仅支持 `next_timepoint`；Experiment A 不再可复现 |
| 35 推荐消融实验 | 【部分实现】 | `experiments/variants.py` 已注册 `G20_NEXTPOINT`（Experiment B–F 全组合）；Experiment A 因旧任务删除不可运行 |
| 36 工程重构优先级 | 【部分实现】 | P0/P1 全部完成；P2 中 AR(1)、FC rollout、频谱、短 rollout loss 已完成，latent transition supervision 未做；P3 均未实现 |
| 37 Sanity Checks | 【部分实现】 | 多数约束由实现与数据构造保证；独立 causal leakage test 无脚本 |
| 38 Smoke Tests | 【已删除】 | 旧冒烟脚本随任务统一删除；`smoke_test_next_point.py` 尚未补齐（已知缺口） |
| 39 Checkpoint Compatibility | 【已实现】 | `--load_backbone_only`：显式 loaded / missing / incompatible / reinitialized 四类清单；strict=False 静默过滤已移除 |
| 40 推荐日志 | 【已实现】 | `train/loss_abs|delta|pcc`、`val next_*`、`Val/Rollout_MAE_H*`；MTP 分偏移日志 |
| 41 Checkpoint Selection | 【已实现】 | 判据 = val next-state MAE，并打印 vs persistence 的 ΔMAE |
| 42 新任务整体数据流 | 【已实现】 | 与实现一致（协议文档 §1） |
| 43–44 科学问题 / 与数字孪生关系 | — | 概念性内容，无实现对应 |
| 45 本轮重构的最终边界 | 【已实现】 | 「本轮必须完成」清单全部落地；「暂缓」清单仍按计划暂缓 |
| 46 最终目标 | 【已实现】 | 默认任务已切换为 next_timepoint（连续 BOLD → 下一 TR 全脑状态） |

---


## 1. 文档目的

本文档定义 NeuroTwin 下一阶段的任务重构方案。

当前 NeuroTwin 主要采用基于固定滑动窗口的未来窗口预测任务，即使用若干历史 BOLD window 预测下一个 BOLD window。该任务能够用于短期 BOLD 预测，但存在窗口高度重叠、样本独立性不足、预测目标过于局部、容易利用短期平滑性以及难以评估长期脑动力学等问题。

本轮重构将核心任务从：

```text
Historical BOLD Windows → Next BOLD Window
```

调整为：

```text
Historical Continuous BOLD States → Next Whole-Brain BOLD State
```

即直接基于连续原始 BOLD 序列进行自回归脑状态建模：

$$p(x_{t+1}\mid x_{\leq t}, SC)$$

对于 MDD 阶段进一步建模：

$$
p(x_{t+1}\mid x_{\leq t}, SC, HAMD)
$$

其中：

- $x_t\in\mathbb{R}^{F}$ 表示第 $t$ 个 TR 时刻所有 ROI 的联合 BOLD 状态；
- $F$ 为 ROI 数，例如 AAL116 时 $F=116$；
- $SC\in\mathbb{R}^{F\times F}$ 为个体结构连接；
- HAMD 为 MDD 个体病理条件。

本轮重构的目标不是立即重写整个 NeuroTwin 模型，而是首先建立更加合理、统一且可扩展的任务基础，为后续 Multi-Timepoint Prediction、长程 rollout、个体化病理动力学和干预模拟提供统一接口。

---

# 2. 当前任务定义

当前典型训练输入为：

$$
x\in\mathbb{R}^{F\times 6\times 30}
$$

目标为：

$$
y\in\mathbb{R}^{F\times 1\times 30}
$$

即：

```text
6 historical windows → 1 future window
```

典型数据构造方式为：

```text
X1 ... X6 → X7
X2 ... X7 → X8
X3 ... X8 → X9
```

其中多个训练样本之间存在大量时间重叠。

当前模型主要包含：

```text
BOLD
  ↓
BrainRevIN
  ↓
DFCAdapter
  ↓
BrainMDM
  ↓
GraphODE / GraphODEDDI
  ↓
ForecastHead
  ↓
Base Prediction
  ↓
MDD: Pathology MoE Residual
```

当前训练目标主要包括：

- PCC loss；
- MAE；
- first-difference loss；
- std loss；
- uncertainty weighting。

当前 evaluation 主要集中在短期 waveform regression。

---

# 3. 当前任务设计的主要问题

## 3.1 滑动窗口样本高度相关

例如：

```text
X1...X6 → X7
X2...X7 → X8
```

两个样本历史信息高度重叠。

这类样本可以作为同一 subject trajectory 内的训练位置，但不应被视为统计意义上的独立被试样本。

---

## 3.2 固定窗口引入人为动力学时间尺度

当前一个 window 包含固定数量 TR，例如 30 TR。

因此模型实际学习的是：

```text
30-TR local segment → next 30-TR segment
```

而不是原始 BOLD 中最基本的：

```text
brain state at t → brain state at t+1
```

窗口长度会人为决定模型动力学时间分辨率。

---

## 3.3 单窗口预测容易依赖局部平滑性

BOLD 信号具有明显的低频性和时间自相关。

因此模型可以通过：

- last-window copy；
- local trend；
- smooth extrapolation；

获得较好的短期 PCC。

高 next-window PCC 并不能充分证明模型学习到了真实脑动力学。

---

## 3.4 任务与数字孪生长期推演目标不完全一致

数字孪生脑最终需要具备：

- 状态转移建模；
- 多步轨迹生成；
- rollout；
- 个体化状态修正；
- 干预模拟；
- 反事实模拟。

单独优化：

```text
6 windows → next window
```

无法充分评价这些能力。

---

# 4. 新任务：Next Brain-State Prediction

## 4.1 基本定义

将每一个真实 TR 时刻的全脑 ROI vector 视为一个 brain-state token：

$$
x_t=[x_t^1,\ldots,x_t^F]
$$

其中：

$$
x_t\in\mathbb{R}^{F}
$$

对于完整 BOLD：

$$
X=[x_1,x_2,\ldots,x_T]
$$

新的核心任务定义为：

$$
x_{\leq t}\rightarrow x_{t+1}
$$

即：

$$
p(x_{t+1}\mid x_{\leq t},SC)
$$

HC 阶段学习一般脑动力学；

MDD 阶段进一步学习：

$$
p(x_{t+1}\mid x_{\leq t},SC,HAMD)
$$

---

## 4.2 推荐术语

论文和代码中建议使用：

- Next-Timepoint Prediction；
- Next Brain-State Prediction；
- Autoregressive Brain Dynamics Learning。

不建议正式称为“next-token prediction”，因为 BOLD state 是连续向量而非离散 token。

可以在动机中表述为：

> Inspired by causal autoregressive next-token learning, NeuroTwin models continuous whole-brain state transitions through next-timepoint prediction.

---

# 5. 数据组织方式重构

## 5.1 Dataset 基本单位改为 Subject

新的 Dataset 基本单位不再是 window sample，而是完整被试或完整 session。

推荐数据结构：

```text
SubjectSample
├── bold        [T, F]
├── sc          [F, F]
├── hamd        [1]          # MDD only
├── subject_id
├── tr
├── site
└── valid_mask  [T]
```

其中推荐内部统一：

```text
BOLD: [T, F]
```

便于 temporal autoregressive modeling。

---

## 5.2 不再离线生成固定 6→1 window 样本

旧流程：

```text
Raw BOLD
  ↓
Sliding Windows
  ↓
6-window history / 1-window target
  ↓
Dataset
```

新流程：

```text
Raw Continuous BOLD
  ↓
Subject-level Dataset
  ↓
Dynamic Historical Context Sampling
  ↓
Next-Timepoint Target
```

即不再提前保存：

```text
x=[F,6,30]
y=[F,1,30]
```

---

# 6. Subject-Level Split

必须严格按照 subject 划分：

```text
Subjects
  ├── Train Subjects
  ├── Validation Subjects
  └── Test Subjects
```

要求：

$$
Train\cap Val=\varnothing
$$

$$
Train\cap Test=\varnothing
$$

$$
Val\cap Test=\varnothing
$$

禁止采用：

```text
先切时间片 → 再随机划分时间片
```

同一个 subject 的任何时间位置均不得跨 train/val/test。

---

# 7. Historical Context 构造

虽然预测目标是下一个 TR：

$$
x_{t+1}
$$

但输入不应只包含：

$$
x_t
$$

而应使用一定长度的历史上下文：

$$
x_{t-K+1:t}
$$

任务变成：

$$
x_{t-K+1:t}\rightarrow x_{t+1}
$$

其中 $K$ 是 context length。

---

## 7.1 推荐第一版设置

建议支持：

```text
context_min = 16
context_max = 64
```

或者：

```text
context_lengths = [16, 32, 64]
```

训练期间动态采样。

例如：

```text
History:
[x51, x52, ..., x82]

Target:
x83
```

输入 shape：

$$
[B,K,F]
$$

目标：

$$
[B,F]
$$

---

## 7.2 Variable Context

不建议再次固定：

```text
K = 30
```

而建议训练时随机选择不同历史长度。

目标是避免模型只适应单一时间尺度，并提升其对不同 context 长度的鲁棒性。

---

# 8. 两种训练实现方式
> **实现状态（2026-09-28）：【部分实现】** 仅「随机 context → 下一时间点」落地（本节 8.1 思路）；full-sequence causal training 未实现，`--causal_training full_sequence` 显式报错（协议文档 §9.1）。注意本节方案 A/B 命名与协议文档 §9.1 相反（协议文档 A=全序列 causal，B=随机 context）。

## 8.1 方案 A：Random Context → Next State

第一版最稳妥。

每次训练：

1. 随机选择 subject；
2. 随机选择 context length $K$；
3. 随机选择预测位置 $t$；
4. 输入 $x_{t-K+1:t}$；
5. 预测 $x_{t+1}$。

优点：

- 对当前 NeuroTwin backbone 改动较小；
- 容易复用 BrainMDM、GraphODE 等模块；
- 容易保证因果性；
- 工程风险较低。

建议作为 P0 版本。

---

## 8.2 方案 B：Full-Sequence Causal Training

如果后续 backbone 能自然支持 causal sequence modeling，可进一步改为：

输入：

$$
[x_1,x_2,\ldots,x_{T-1}]
$$

目标右移：

$$
[x_2,x_3,\ldots,x_T]
$$

通过 causal mask，一次 forward 同时计算多个位置的 next-state loss。

形式与 autoregressive language modeling 类似。

优点：

- 每个 forward 可以利用大量时间位置；
- supervision density 更高；
- 更接近标准 causal autoregressive learning。

但要求所有 temporal module 严格无未来泄漏。

因此第一阶段不强制实现。

---

# 9. Causal Constraint

新任务的基本原则：

$$
\hat x_{t+1}
$$

只能使用：

$$
x_{\leq t}
$$

禁止模型看到：

$$
x_{t+1:T}
$$

如果采用完整序列训练，则需要严格 causal mask。

如果采用截取好的 history context：

```text
history = x[t-K+1:t]
```

则可以继续使用 context 内部双向特征提取，但必须确保输入中不存在未来 target。

需要增加 causal leakage test：

```text
Changing future observations must not change prediction at t.
```

---

# 10. Normalization 重构

必须重点检查 normalization leakage。

禁止使用完整 subject：

$$
x_{1:T}
$$

计算 mean/std 后，再预测其中的未来时间点。

因为这会将未来信息泄漏到输入。

推荐：

```text
history context
   ↓
compute mean/std
   ↓
normalize history
   ↓
model
   ↓
predict normalized next state
   ↓
inverse transform using history statistics
```

即 normalization statistics 必须只来自：

$$
x_{t-K+1:t}
$$

目标：

$$
x_{t+1}
$$

不得参与。

现有 BrainRevIN 应优先复用，但需要重新检查其统计维度和 inverse normalization 行为。

---

# 11. 模型标准输入输出

建议统一模型接口：

```python
outputs = model(
    bold_history,   # [B, K, F]
    sc,             # [B, F, F]
    hamd=None,
)
```

输出：

```text
pred_next    [B, F]
pred_delta   [B, F]
latent_state [...]
```

如果空间模块内部更适合：

```text
[B,F,K]
```

只允许在模型入口进行一次标准 transpose。

不要让不同模块不断隐式交换维度。

---

# 12. Residual / Delta Prediction

这是本次重构的重点之一。

由于：

$$
x_{t+1}\approx x_t
$$

直接预测 absolute BOLD 容易产生 identity shortcut。

因此建议默认预测：

$$
\Delta x_t=x_{t+1}-x_t
$$

模型输出：

$$
\Delta \hat{x}_t
$$

最终：

$$
\hat{x}_{t+1}
=
x_t+\Delta\hat{x}_t
$$

即：

```text
Historical Context
      ↓
NeuroTwin
      ↓
Predicted State Change Δx
      ↓
Last True State + Δx
      ↓
Predicted Next Brain State
```

配置支持：

```text
prediction_target = delta
```

以及 ablation：

```text
prediction_target = absolute
```

---

# 13. 新训练目标

## 13.1 Absolute State Loss

预测：

$$
\hat{x}_{t+1}
$$

与真实：

$$
x_{t+1}
$$

计算：

$$
L_{abs}
=
\|\hat{x}_{t+1}-x_{t+1}\|_1
$$

默认优先 MAE / Huber。

---

## 13.2 Delta Loss

真实变化：

$$
\Delta x_t
=
x_{t+1}-x_t
$$

预测变化：

$$
\Delta\hat{x}_t
$$

定义：

$$
L_{\Delta}
=
\|\Delta\hat{x}_t-\Delta x_t\|_1
$$

该 loss 用于强化真正的状态变化建模。

---

## 13.3 Spatial Pattern Loss

可以保留 PCC，但必须重新解释。

对于单个时间点：

$$
x_{t+1}\in\mathbb{R}^{F}
$$

PCC 应沿 ROI 维计算：

$$
PCC(\hat{x}_{t+1},x_{t+1})
$$

此时衡量的是：

> 下一时间点全脑空间激活 pattern 是否一致。

这不同于旧任务中的时间 waveform PCC。

建议：

$$
L_{spatial}
=
1-PCC(\hat{x}_{t+1},x_{t+1})
$$

作为辅助，而非唯一主 loss。

---

## 13.4 第一版推荐总损失

$$
L_{total}
=
\lambda_{abs}L_{abs}
+
\lambda_{\Delta}L_{\Delta}
+
\lambda_{pcc}L_{spatial}
$$

推荐初始值：

```text
lambda_abs   = 1.0
lambda_delta = 1.0
lambda_pcc   = 0.1
```

所有权重必须配置化。

---

# 14. 当前旧 Loss 的处理

现有：

- MAE：保留；
- PCC：重新定义计算维度；
- first-difference loss：与新的 delta objective 合并或重新解释；
- std loss：可保留为可选辅助项；
- uncertainty weighting：第一阶段若兼容可保留，否则先简化。

第一阶段优先保证任务语义正确，而不是保留所有旧 loss。

---

# 15. HC Pretraining 新任务

HC 阶段定义：

$$
p(x_{t+1}\mid x_{t-K+1:t},SC)
$$

目标：

> 学习一般脑状态转移动力学。

可表示为：

$$
z_t
=
E(x_{t-K+1:t},SC)
$$

$$
\Delta \hat{x}_{t}
=
F_{HC}(z_t,SC)
$$

$$
\hat{x}_{t+1}
=
x_t+\Delta\hat{x}_{t}
$$

HC 阶段不需要 HAMD。

---

# 16. MDD Finetuning 新任务

MDD 阶段定义：

$$
p(x_{t+1}\mid x_{t-K+1:t},SC,HAMD)
$$

建议保持：

```text
HC Backbone
    +
Pathology-Conditioned Residual
```

新的动态解释为：

$$
\Delta\hat{x}_t
=
\Delta\hat{x}^{HC}_t
+
\Delta\hat{x}^{MDD}_t
$$

其中：

$$
\Delta\hat{x}^{HC}_t
$$

表示一般脑动力学；

$$
\Delta\hat{x}^{MDD}_t
$$

表示病理状态对下一步状态变化的修正。

这比原来直接修正整个 future window 更符合状态转移建模。

---

# 17. Pathology MoE

第一阶段可以继续使用现有 HAMD-conditioned routing。

但建议预留：

```text
router_condition = HAMD + current latent brain state
```

未来：

$$
Router(HAMD,z_t)
$$

比仅：

$$
Router(HAMD)
$$

更符合个体化动力学。

第一阶段该功能不是 P0 必需项。

---

# 18. GraphODE 的重新定位
> **实现状态（2026-09-28）：【部分实现】** 审计结论已确认（`GraphODE.forward(t,…)` 直接丢弃 `t`，自治向量场）；真实 Δt 未实现，按本节要求在协议文档 §9.2 如实说明，未伪装连续时间语义。

新任务下，GraphODE 更容易被解释为：

$$
z_t\rightarrow z_{t+1}
$$

如果 TR 已知：

$$
\Delta t=TR
$$

则未来可进一步实现：

$$
z(t+\Delta t)
=
ODESolve(f_\theta,z(t),\Delta t)
$$

因此后续 GraphODE 应显式接收真实：

```text
delta_t = TR
```

但是本轮不强制重写。

首先需要审计：

- 当前 GraphODE 是否实际使用 t；
- 是否使用真实 Δt；
- 是否只是重复 autonomous vector field。

如果没有真实时间语义，应在文档中如实说明。

---

# 19. Evaluation 必须重新设计

新的评价不能只看：

```text
next-step PCC
```

必须同时覆盖：

1. next-state accuracy；
2. simple baseline；
3. multi-step rollout；
4. trajectory stability；
5. network-level consistency。

---

# 20. Next-State Metrics

单步预测至少计算：

- MAE；
- RMSE；
- spatial PCC；
- R²。

同时可以统计：

- delta MAE；
- delta direction consistency。

---

# 21. Persistence Baseline

最重要基线：

$$
\hat{x}_{t+1}=x_t
$$

因为 BOLD 高度自相关。

NeuroTwin 必须证明其性能优于简单 persistence。

---

# 22. Linear Trend Baseline

定义：

$$
\hat{x}_{t+1}
=
x_t+(x_t-x_{t-1})
$$

用于判断模型是否只是学习简单局部趋势。

---

# 23. AR(1) Baseline

如果实现成本合理，增加每 ROI 的 AR(1)：

$$
x_{t+1}^{(i)}
=
a_i x_t^{(i)}+b_i
$$

参数只能使用训练集 subject 拟合。

优先级：

```text
Persistence
> Linear Trend
> AR(1)
> VAR
```

---

# 24. Free Autoregressive Rollout

虽然训练目标是 next-state prediction，但数字孪生评价必须包含 free rollout。

给定真实历史：

$$
x_{t-K+1:t}
$$

预测：

$$
\hat{x}_{t+1}
$$

随后：

$$
[x_{t-K+2:t},\hat{x}_{t+1}]
\rightarrow
\hat{x}_{t+2}
$$

继续：

$$
\hat{x}_{t+3},\ldots,\hat{x}_{t+H}
$$

从第一步预测开始，中间不得读取未来 ground truth。

推荐：

```text
rollout_horizons = [1, 2, 4, 8, 16]
```

条件允许可增加 32。

---

# 25. Rollout Metrics

每个 horizon 分别输出：

```text
MAE_H1
MAE_H2
MAE_H4
MAE_H8
MAE_H16
```

以及：

- RMSE；
- spatial PCC；
- ROI-wise temporal PCC；
- trajectory MAE；
- variance ratio。

重点观察：

$$
H\uparrow
\Rightarrow
Error\uparrow
$$

以及模型是否发生：

- error accumulation；
- variance collapse；
- signal flattening；
- dynamical drift。

---

# 26. Functional Connectivity Evaluation

单个 timepoint 无法计算 FC。

因此 FC evaluation 只在较长 rollout trajectory 上执行。

例如：

$$
Y_{pred}\in\mathbb{R}^{H\times F}
$$

$$
Y_{true}\in\mathbb{R}^{H\times F}
$$

当：

```text
H >= fc_min_length
```

例如 32 或 64 TR 时：

$$
FC_{pred}
=
Corr(Y_{pred})
$$

$$
FC_{true}
=
Corr(Y_{true})
$$

比较：

- upper-triangle edge PCC；
- FC MAE；
- within-network FC；
- between-network FC。

短 rollout 不计算 FC。

---

# 27. Spectral Evaluation

对于较长 rollout，可以增加：

- PSD；
- frequency-band power；
- predicted / true PSD distance。

目的：

判断长程生成是否保持 BOLD 低频动力学。

第一阶段只作为 metric，不进入训练 loss。

---

# 28. Validation / Test Protocol

训练可以随机采样 context 和预测位置。

Validation/Test 必须 deterministic。

推荐：

```text
eval_context_length = 64
```

然后：

- 枚举所有合法预测位置；
- 或固定均匀采样若干 anchor points。

每次 validation/test 必须使用完全相同的位置。

---

# 29. Subject-Level Aggregation

一个 subject 内可能产生大量预测时间点。

这些时间点不能作为独立统计样本。

正确统计流程：

```text
Timepoint Predictions
      ↓
Aggregate Within Subject
      ↓
Subject-Level Metric
      ↓
Across-Subject Statistics
```

论文显著性分析必须以 subject 为统计独立单位。

---

# 30. MTP 后续扩展

完成 Next-Timepoint Prediction 后，可以进一步加入：

## Multi-Timepoint Prediction

未来 offsets：

$$
\mathcal{H}=\{1,2,4,8\}
$$

即：

$$
x_{\leq t}
\rightarrow
\{
x_{t+1},
x_{t+2},
x_{t+4},
x_{t+8}
\}
$$

建议名称：

- Multi-Timepoint Prediction；
- Multi-Horizon State Prediction；
- Multi-Scale Temporal Prediction。

不建议论文正式称为 Multi-Token Prediction。

---

# 31. MTP 第一版结构
> **实现状态（2026-09-28）：【部分实现】** Parallel MTP 已实现：`--enable_mtp` + `--mtp_weights`，同一 hidden state 并行输出多偏移，预测头无需改结构；Sequential MTP 未实现（P3）。

优先采用 Parallel MTP：

```text
                h_t
          ┌──────┼──────┬──────┐
          ↓      ↓      ↓      ↓
        Head1  Head2  Head4  Head8
          ↓      ↓      ↓      ↓
        t+1    t+2    t+4    t+8
```

主 next-state 仍然为：

$$
t+1
$$

其他 horizon 作为辅助预测。

损失：

$$
L_{MTP}
=
\sum_k
w_k
L(\hat{x}_{t+\delta_k},x_{t+\delta_k})
$$

推荐初始：

```text
offsets = [1, 2, 4, 8]
weights = [1.0, 0.7, 0.5, 0.3]
```

本轮重构仅需要预留接口，不要求 P0 阶段立即启用。

---

# 32. Short Rollout Training
> **实现状态（2026-09-28）：【已实现】** `--enable_rollout_loss`（默认关闭），`--rollout_train_steps` 默认 2、`--lambda_rollout` 默认 0.2，成本约为单步的 (1+steps) 倍；第一版与设计一致保持关闭。

Next-state training 使用真实历史，会产生 teacher-forcing / free-rollout mismatch。

后续可以加入短 rollout loss。

例如训练：

$$
x_{\leq t}
\rightarrow
\hat{x}_{t+1}
$$

然后使用预测：

$$
\hat{x}_{t+1}
$$

继续：

$$
\hat{x}_{t+2}
$$

定义：

$$
L_{roll}
=
\sum_{k=1}^{K_r}
D(\hat{x}_{t+k},x_{t+k})
$$

第一版建议：

```text
rollout_train_steps = 2 or 4
```

该功能应作为 P2，而不是一开始强制启用。

---

# 33. 新配置参数

建议新增：

```yaml
task_mode: next_timepoint

context_min: 16
context_max: 64
context_lengths: [16, 32, 64]

prediction_target: delta

causal_training: false

lambda_abs: 1.0
lambda_delta: 1.0
lambda_pcc: 0.1

eval_context_length: 64
eval_rollout_horizons: [1, 2, 4, 8, 16]

eval_fc: true
fc_min_length: 32

eval_spectral: false

enable_rollout_loss: false
rollout_train_steps: 2
lambda_rollout: 0.1

enable_mtp: false
forecast_offsets: [1]
mtp_weights: [1.0]

sampling_seed: 42
```

后续开启 MTP：

```yaml
enable_mtp: true
forecast_offsets: [1, 2, 4, 8]
mtp_weights: [1.0, 0.7, 0.5, 0.3]
```

---

# 34. 兼容旧任务
> **实现状态（2026-09-28）：【已删除】** 本节设计已被后续决策推翻：2026-09-27 任务口径统一时，旧任务（滑窗 6→1）连同兼容别名与旧 CLI 参数完全删除，`--task_mode` 仅支持 `next_timepoint`；Experiment A（window forecast）不再可复现，消融基线由 Experiment B（无 delta）承担。

必须尽量保留旧模式：

```text
task_mode = window_forecast
```

新增：

```text
task_mode = next_timepoint
```

这样可以进行公平 ablation：

### Experiment A

旧任务：

```text
6 windows → 1 window
```

### Experiment B

Next-Timepoint：

```text
history → t+1
```

### Experiment C

Next-Timepoint + Delta Prediction

### Experiment D

Next-Timepoint + Short Rollout Loss

### Experiment E

Next-Timepoint + MTP

---

# 35. 推荐消融实验
> **实现状态（2026-09-28）：【部分实现】** `experiments/variants.py` 已注册 `G20_NEXTPOINT`（Experiment B–F：next_timepoint / delta / rollout loss / MTP 全组合）；Experiment A 因旧任务删除不可运行。

至少设计：

| Experiment | Task | Delta | Rollout Loss | MTP |
|---|---|---:|---:|---:|
| A | Window Forecast | No | No | No |
| B | Next-Timepoint | No | No | No |
| C | Next-Timepoint | Yes | No | No |
| D | Next-Timepoint | Yes | Yes | No |
| E | Next-Timepoint | Yes | No | Yes |
| F | Next-Timepoint | Yes | Yes | Yes |

目的不是只比较最终最复杂模型，而是明确：

- 任务改变是否有效；
- delta prediction 是否有效；
- rollout supervision 是否有效；
- MTP 是否有效。

---

# 36. 工程重构优先级
> **实现状态（2026-09-28）：【部分实现】** P0、P1 全部完成；P2 中 AR(1) baseline、FC rollout 评估、频谱评估、短 rollout training loss 已完成，latent transition supervision 未实现；P3（Sequential MTP、state-aware MoE router、真实 Δt GraphODE、概率预测、干预模拟）均未实现。

## P0：必须完成

- 原始连续 BOLD 读取；
- subject-level Dataset；
- subject-level split；
- dynamic context sampling；
- next-timepoint target；
- normalization leakage 修复；
- model 输入 `[B,K,F]`；
- next-state output `[B,F]`；
- delta prediction；
- next-state loss；
- HC training；
- MDD finetuning。

---

## P1：必须完成

- deterministic validation；
- persistence baseline；
- linear trend baseline；
- subject-level metrics；
- free rollout；
- H1/H2/H4/H8/H16 metrics；
- checkpoint save/load；
- 文档更新。

---

## P2：推荐完成

- AR(1) baseline；
- FC rollout evaluation；
- spectral evaluation；
- short rollout training loss；
- latent transition supervision。

---

## P3：后续研究扩展

- Parallel MTP；
- Sequential MTP；
- state-aware MoE router；
- real-$\Delta t$ GraphODE；
- probabilistic next-state prediction；
- intervention/counterfactual simulation。

---

# 37. 必须增加的 Sanity Checks

至少检查：

1. Train/Val/Test subject 无交叉；
2. Context 不包含 target；
3. Target 时间严格晚于 context；
4. Future 不参与 normalization statistics；
5. BOLD ROI 数与 SC 一致；
6. BOLD ROI 顺序与 SC 对齐；
7. `history.shape == [B,K,F]`；
8. `target.shape == [B,F]`；
9. variable context 正常；
10. batch size > 1 正常；
11. delta reconstruction 正确；
12. loss 不出现 NaN/Inf；
13. backward 正常；
14. optimizer step 正常；
15. validation deterministic；
16. rollout 不读取 future ground truth；
17. HC 不强制要求 HAMD；
18. MDD 正确读取 HAMD；
19. checkpoint 正常保存和恢复；
20. persistence baseline 输出正确；
21. old task mode 仍可启动。

---

# 38. Smoke Tests

代码修改完成后至少运行：

## Test 1：Data Loading

加载一个真实 subject：

```text
BOLD: [T,F]
SC: [F,F]
TR
subject_id
```

---

## Test 2：Context Sampling

生成：

```text
history: [B,32,116]
target:  [B,116]
```

---

## Test 3：Forward

执行：

```text
model(history, sc)
```

检查：

```text
pred_next: [B,116]
pred_delta: [B,116]
```

---

## Test 4：Loss

确认：

```text
loss_abs
loss_delta
loss_total
```

均为有限值。

---

## Test 5：Backward

执行完整：

```text
zero_grad
forward
loss.backward
optimizer.step
```

---

## Test 6：Validation

运行最小 validation loop，并确认重复运行结果一致。

---

## Test 7：Rollout

执行：

```text
H = 8
```

输出：

```text
[B,8,116]
```

确认第 2 步开始使用预测结果而非真实未来。

---

## Test 8：Baseline

运行：

- Persistence；
- Linear Trend。

---

## Test 9：MDD

确认：

```text
BOLD + SC + HAMD
```

能够正常完成 forward/backward。

---

# 39. Checkpoint Compatibility
> **实现状态（2026-09-28）：【已实现】** `--load_backbone_only True` 只加载主干键，预测头/MoE/条件模块重新初始化，显式打印 loaded / missing / incompatible / reinitialized 四类清单；同口径（next_timepoint → next_timepoint）预训练→微调可完整加载，跨口径由 `check_pretrain_config_compat` 报错。

ForecastHead 改变后，旧 checkpoint 很可能不能完整加载。

禁止简单：

```python
strict=False
```

然后静默忽略大量权重。

应明确支持：

```text
load_backbone_only
```

并打印：

- successfully loaded parameters；
- missing parameters；
- reinitialized modules；
- incompatible parameters。

旧 ForecastHead 不兼容时，应显式重新初始化。

---

# 40. 推荐日志

训练：

```text
train/loss_total
train/loss_abs
train/loss_delta
train/loss_pcc
```

Validation：

```text
val/next_MAE
val/next_RMSE
val/next_spatial_PCC

val/persistence_MAE
val/trend_MAE
```

Rollout：

```text
val/rollout_MAE_H1
val/rollout_MAE_H2
val/rollout_MAE_H4
val/rollout_MAE_H8
val/rollout_MAE_H16
```

启用 MTP 后：

```text
train/loss_t+1
train/loss_t+2
train/loss_t+4
train/loss_t+8
```

---

# 41. Checkpoint Selection
> **实现状态（2026-09-28）：【已实现】** checkpoint 判据 = val next-state MAE（日志同时打印与 persistence 的 ΔMAE），未使用单纯 PCC。

第一阶段建议使用：

```text
Validation Next-State MAE
```

选择最佳 checkpoint。

不建议只根据 PCC 选择。

后续加入 rollout training 后，可以考虑：

```text
Next-State Loss
+
Short-Rollout Loss
```

组成综合选择指标。

---

# 42. 新任务整体数据流

最终数据流应调整为：

```text
Raw ROI-Level Continuous BOLD
          │
          ▼
    Subject-Level Dataset
          │
          ▼
Historical Context Sampling
    [B, K, F]
          │
          ├──────── SC [B,F,F]
          │
          └──────── HAMD [B,1]  (MDD)
          │
          ▼
     NeuroTwin Backbone
          │
          ▼
 Current Latent Brain State
          │
          ▼
 Next-State Transition Model
          │
          ▼
 Predicted State Change Δx
          │
          ▼
 x_t + Δx
          │
          ▼
 Predicted Next Brain State
       [B,F]
```

训练：

```text
history → next true state
```

测试：

```text
history
  ↓
pred t+1
  ↓
pred t+2
  ↓
pred t+4
  ↓
...
long free rollout
```

---

# 43. 新的科学问题

旧任务回答：

> 给定若干历史 BOLD window，能否预测下一个 BOLD window？

新任务希望回答：

> 给定个体历史全脑活动状态与结构连接，模型能否学习脑活动的条件状态转移规律，并由此连续推演未来脑动态？

数学上：

$$
p(x_{t+1}\mid x_{\leq t},SC)
$$

进一步：

$$
p(x_{t+1}\mid x_{\leq t},SC,HAMD)
$$

最终希望扩展：

$$
p(x_{t+1:t+H}\mid x_{\leq t},SC,HAMD)
$$

---

# 44. 与数字孪生脑目标的关系

本次改造使 NeuroTwin 从：

```text
Window Regression Model
```

逐渐转向：

```text
Personalized Brain State Transition Model
```

后续才能进一步构建：

```text
Current Individual Brain State
          ↓
Learned Transition Dynamics
          ↓
Future State Rollout
          ↓
Pathological Dynamics Correction
          ↓
Perturbation / Intervention
          ↓
Counterfactual Brain Trajectory
```

因此 next-timepoint prediction 不是最终目标，而是构建可 rollout 数字孪生动力学模型的基础学习任务。

---

# 45. 本轮重构的最终边界
> **实现状态（2026-09-28）：【已实现】** 「本轮必须完成」清单已全部落地；「暂缓」清单（Sequential MTP、复杂概率预测、连续时间 GraphODE 重写、干预模拟、临床反事实）仍按计划暂缓。

本轮必须完成：

```text
Continuous BOLD
→ Subject-Level Dataset
→ Variable Historical Context
→ Next-Timepoint Prediction
→ Delta Prediction
→ Persistence/Trend Baselines
→ Free Rollout Evaluation
```

本轮可以暂缓：

```text
Sequential MTP
Complex Probabilistic Forecasting
Full Continuous-Time GraphODE Rewrite
Intervention Simulation
Clinical Counterfactual Prediction
```

这些功能应建立在稳定的 next-state training pipeline 之后。

---

# 46. 最终目标

本轮重构完成后，NeuroTwin 的默认任务应从：

```text
6 historical BOLD windows
→
1 future BOLD window
```

变为：

```text
Historical continuous whole-brain states
→
next whole-brain state
```

即：

$$
\boxed{
x_{t-K+1:t},SC
\rightarrow
x_{t+1}
}
$$

MDD：

$$
\boxed{
x_{t-K+1:t},SC,HAMD
\rightarrow
x_{t+1}
}
$$

推荐默认采用：

$$
\boxed{
\hat{x}_{t+1}
=
x_t+\Delta\hat{x}_t
}
$$

训练阶段首先学习稳定的 next-state transition；

评估阶段使用 free autoregressive rollout 检查长期动力学；

后续通过 MTP、rollout loss、真实 $\Delta t$ GraphODE 和 intervention simulation 逐步扩展为更加完整的个体化数字孪生脑模型。
