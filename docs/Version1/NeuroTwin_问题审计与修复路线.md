# NeuroTwin 问题审计与修复路线

> 审计日期：2026-09-20  
> 依据：`utils/dataloader.py`、`models/*.py`、`train/*.py`、`main.py`、分析脚本及已保存指标。以代码实际行为为准。

## 1. 任务边界与数据流

当前项目是两阶段的 ROI/BOLD 信号预测：

```text
HC 预训练：历史 ROI 信号窗 + SC → 未来 ROI 信号窗
MDD 微调：预训练预测 + HAMD 条件残差 → 个体化未来 ROI 信号窗
```

典型输入为 `x:[B,116,6,30]`，目标为 `y:[B,116,1,30]`，SC 为 `[B,116,116]`，MDD 额外输入 `HAMD:[B,1]`。样本按被试切分，同一被试滑窗不会跨 train/val/test。

**严格说，当前任务是 SC 条件的未来 ROI 信号预测，不是直接预测 dFC 矩阵。** FC/dFC 只能从预测信号派生，当前没有作为直接监督或主要评价指标。

```text
x → BrainRevIN → DFCAdapter(SC 0/1/2-hop)
  → BrainMDM(序列轴/窗口轴多尺度混合)
  → GraphODEDDI × N(SC attention + temporal conv + window attention + FFN + RK2)
  → post_fusion → ForecastHead(anchor/trend/shape/scale + SC refinement)
  → base_pred
  → [MDD] Pathology MoE(HAMD 条件专家残差 + shared expert + SC refinement)
  → prediction
```

## 2. 已确认的 P0 问题

### P0-1：HAMD 投影尺度不一致

**证据**：数据集将临床表中的原始 HAMD 直接传入；`RicherPathologyProjection` 却构造 `[x,x²,sin(πx)]`。已保存分析中 HAMD 约为 1--58，且大量为整数。

**影响**：整数 HAMD 使 `sin(πx)` 近乎恒零，`x²` 最大达 3364，病理嵌入会被尺度主导；当前专家分型不可信。

**措施**：在每个训练折内拟合 robust z-score、经验 CDF 或 quantile transform，并将同一变换固定用于验证/测试。比较 `z-score linear`、`z-score + quadratic`、`z-score + RBF/spline`；不要沿用原始 `sin(πx)`。

### P0-2：验证与推理路由并不确定

**证据**：默认 `moe_use_argmax=False`；`RouterCond` 在 `eval()` 时仍通过低温 `torch.multinomial` 采样专家，且随 batch 内容和随机状态变化。

**影响**：同一 checkpoint 的 PCC、专家使用率、ROI 重要性和早停结果均可能变化。

**措施**：训练可以探索；验证早停和最终测试必须使用确定性 hard top-k 或 dense soft mixture。若保留随机推理，单列 Monte-Carlo 实验，报告至少 10 次均值±标准差，并记录 routing mode、温度和种子。

### P0-3：IterativePredictionRefiner 首轮初始化错误

`round_scales` 使用 `0.5**i`。`i=0` 的目标是 1，经 logit 后为约 `sigmoid(9.21)=0.9999`，而不是注释宣称的 0.5。

**措施**：显式初始化目标尺度为 `[0.50,0.33,0.20]`，并比较：无 refinement、单轮、当前初始化、修正初始化。

### P0-4：GraphODE 不是已验证的连续时间模型

`GraphODE.forward(t,...)` 直接丢弃 `t`；每个 block 只是将同一自治向量场重复 RK2 更新。当前没有不规则时间、真实时间间隔、导数收敛或步长敏感性证据。

**措施**：比较 `ode_steps={1,3,6,12}` 的性能、耗时、每步导数范数和稳定性。若单步不弱于六步，应视为共享参数图残差块并简化论文表述。

### P0-5：波形损失没有保证 FC/dFC 正确

主损失只有逐 ROI 波形 PCC、MAE、一阶差分和标准差。高波形 PCC 不等价于 ROI×ROI 功能连接拓扑正确。

**措施**：按预测窗计算 Pearson FC，报告 FC 上三角 MAE/RMSE、edge-PCC、网络内/网络间误差；增加低权重 Fisher-z FC loss。30 点短窗 FC 方差较大，因此 FC loss 只能是辅助项。

### P0-6：统计单位和可解释性结论不充分

滑窗样本不是独立被试，且当前分析受随机路由影响。ROI 置换中的 batch 重复不等价于独立临床重复。

**措施**：先在被试内聚合，再以被试为单位做 paired bootstrap/permutation；ROI 多重比较做 FDR；将“预测敏感性”与“临床生物标志物”严格区分。

## 3. 架构性瓶颈

| 问题 | 位置 | 影响 |
|---|---|---|
| SC 是硬掩码 | `GraphODE._prepare_sc` | 零 SC 边永远无法成为条件性功能依赖。 |
| SC 重复注入 | Adapter、GraphODE、两个 refiner | 可能过平滑，使结构先验压过功能证据。 |
| 自适应边仍受 SC 限制 | `DFCAdapter` | `SC * learned_edge` 只能缩放既有边。 |
| Router 不看脑状态 | `moe_router_cond_only=True` | 构建的 `5F` history/latent/base 统计量不参与路由。 |
| shared expert 未量化 | `SharedPathologyExpert` always-on | 共享分支可能吸收主要残差，专家增益未知。 |
| 六路融合可能冗余 | `ForecastHead` | 多个分支均来自 history，未必互补。 |
| future readout 过早展平 | `ForecastHead` | 未来 ROI/时间点没有独立语义 query。 |

## 4. 已有结果与正确边界

已保存的综合分析均为 1,353 个样本的验证分析，并非独立测试结果：

| 目录 | PCC | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| pred1 | 0.8505 | 0.3257 | 0.5529 | 0.7086 |
| pred12 | 0.8485 | 0.3281 | 0.5560 | 0.7052 |
| pred13 | 0.8497 | 0.3269 | 0.5545 | 0.7069 |

分析脚本默认 `val_ratio=0.2`，与主训练 CLI 默认值不同，后续必须锁定 split。`pred13` top-1 专家样本数 E0/E1/E2/E3 为 `217/688/438/10`；专家 HAMD 的 Mann--Whitney 检验均未显著，E3 近乎空转。因此不能据此宣称专家发现临床亚型。

## 5. 修复和重构顺序

### 阶段 0：可复现基线

1. 修复 HAMD 编码；
2. 验证/测试使用确定性 routing；
3. 修正 refiner 初始化；
4. 固定 split 与评估参数；
5. 增加 FC 指标和低权重 FC loss；
6. 做 ODE steps 敏感性分析。

### 阶段 1：低风险结构改造

```text
SC hard mask → SC soft prior + low-rank adaptive functional adjacency
waveform-only loss → waveform + low-weight FC loss
random HAMD-only top-k → deterministic soft routing
```

### 阶段 2：中风险信息流重构

```text
ROI history queries ↔ SC profile tokens
latent → future ROI-time queries → future signal
HAMD → AdaLN/FiLM → GraphODE gates / forecast decoder
```

### 阶段 3：高风险高收益

```text
latent → dynamic-state prototypes
[state, normalized HAMD, latent summary] → hierarchical Soft-MoE → residual
```

## 6. 最小实验矩阵

| 目的 | 必须比较的组 |
|---|---|
| 图先验 | no-SC / fixed-SC / adaptive-only / SC+adaptive |
| MoE | no-MoE / shared-only / hard top-k / dense soft mixture |
| 条件信息 | no-HAMD / normalized HAMD / HAMD+brain-state |
| 预测头 | current flatten / ROI router / future-query decoder |
| 监督 | waveform-only / waveform+FC loss |
| 稳定性 | 至少 5 seeds；被试级 CI；固定独立 test |

## 7. 论文表述约束

- 不应直接称为 dFC prediction，除非有显式 FC 监督或完整派生 FC 评估。
- 不应把当前 GraphODE 表述为已验证的连续时间生理模型。
- 不应从随机 routing、滑窗级统计推出临床亚型或生物标志物。
- 可以强调 SC 先验、时空建模、HC→MDD 两阶段迁移、病理条件残差和 subject-level 切分。
