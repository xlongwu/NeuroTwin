# NeuroTwin 后续研究路线

本文件是后续开发入口。所有新研究必须归入以下两条**互相独立**的路线之一；
实验前请先读 `docs/EXPERIMENTS.md` 了解已完成实验与统一协议。

**共同起点（正式基线）**：S2 History Memory + Graph-ODE + CPM 架构，E2 损失。
基线契约：`configs/baseline_s2_e2.json`；构建/加载自检：`python baseline.py`；
对照指标：`results/baseline/S2_metrics_test.json`。

两条路线完成各自实验后分别得到最佳损失 L* 与最佳架构 M*，再研究组合。
建议使用独立 Git 分支/worktree 隔离开发，互不合并中间状态。

---

## 任务 A：损失函数优化

**目标**：完善 BOLD 神经生理信号预测的监督目标（不解决长时域退化本身）。

**基线**：E2 Loss（λ_td=0.05、λ_ch=0.02，其余 one/CPM/dynamics/variance 项不变）。

**允许修改**：

- 损失函数数学形式与各项组合/权重（`train/losses.py`）。
- 损失实验配置与消融脚本（`experiments/loss/`，新结果写 `results/loss/`）。
- 对应损失的评估与消融入口（复用 `experiments/evaluate_variant.py` 与
  `analysis/next_point_eval.py`，不另建评估系统）。

**禁止**：修改模型主干、重构 Graph-ODE、增加状态转移模块、将架构优化混入
同一次 Loss 实验。每个 Loss 变体必须在同一 S2 架构、同 seed、同划分下与
E2 做配对消融（协议见 `docs/EXPERIMENTS.md` §0）。

**候选方向**（均未实施，待各自独立实验验证）：

1. 幅度自适应 Temporal Loss（Q1 过度响应 / Q5 低估的异方差加权）。
2. Horizon-Aware Loss（逐 horizon 加权策略，替代当前 gamma）。
3. Multi-Scale Temporal Loss（多尺度时间差分）。
4. CRPS 与时空联合概率损失（与任务 B 的概率模型区分：这里只动监督目标）。

**对照实验**：每臂 vs E2（S0 口径）配对检验；报告 t+1 MAE/PCC、Delta MAE、
Direction Accuracy、ODE H1–H8 MAE、Q1–Q5 分层、Delta Std Ratio。

**入口**：`experiments/loss/compare_phase1.py` 是 E 系列配对对比工具，
新实验沿用其产物结构（`results/loss/<arm>/` + `results/loss/compare/`）。

---

## 任务 B：长时域预测能力优化

**目标**：改善 Graph-ODE 自由推进的状态误差累积（Free H8 MAE ≈ 0.81 vs
D1 ≈ 0.081 的差距即改进空间），并探索概率化脑动力学预测。

**基线**：S2 架构 + E2 损失（E2 为固定确定性监督，架构消融时不同时调 Loss）。

**允许修改**：

- Graph-ODE 状态转移与向量场（`models/tfm_dynamics.py`）。
- 长时域状态表示与历史记忆机制（`_history_memory` / `memory_film`）。
- 长时域预测头（`models/tfm.py` CPMHorizonHead）。
- 随机潜在状态转移（S4-2 方向，`LowRankLatentInnovation`）。
- 与架构直接相关的训练实现（`main.py` 相应分支）。
- 新实验脚本放 `experiments/long_horizon/`，结果写 `results/long_horizon/`。

**约束**：

- 以 S2 为初始架构；架构消融固定 E2 Loss，不混入 Loss 调优。
- 概率模型若引入新的分布训练目标，使用独立固定配置，与任务 A 的损失消融
  区分（S4-1 教训：边际 CRPS 对联合依赖盲，见 `docs/EXPERIMENTS.md` §5）。
- D1/D2 刷新诊断（`experiments/long_horizon/refresh_rollout.py`）继续作为
  状态误差累积的标准诊断；注意 D1 使用未来真值，不是无观测基线。

**候选方向**（均未实施）：

1. 历史条件长 horizon 校正（利用 S2 记忆机制向远期延伸）。
2. 输出端概率轨迹建模（S4-1 之上的联合依赖目标）。
3. 随机潜在状态转移（S4-2，尚未验证）。

**对照实验**：每臂 vs S2 配对检验（协议同 §4）；确定性指标（MAE/PCC/
H1–H8/Q1–Q5/DSTD）+ 若概率化则 CRPS/ES/Variogram/coverage/PIT 全套
（`analysis/s4_probabilistic_eval.py`）。

**入口**：从 `checkpoints/phase2_S2` 与 `configs/baseline_s2_e2.json` 出发；
S4-1 历史对照权重在 `checkpoints/phase4_S4_1`。

---

## 与对方的隔离关系

| | 任务 A | 任务 B |
|---|---|---|
| 共用 | S2 架构、E2 起点、统一评估协议、`experiments/evaluate_variant.py` | 同左 |
| 各自产出 | L*（最佳损失） | M*（最佳架构） |
| 结果目录 | `results/loss/` | `results/long_horizon/` |
| 修改禁区 | `models/`（除评估所需只读） | `train/losses.py` 的确定性监督项 |

两路线各自完成独立实验、形成结论后，再开组合实验检验 L*+M*。
