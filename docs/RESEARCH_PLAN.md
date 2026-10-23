# NeuroTwin 后续研究路线

本文件是后续开发入口。所有新研究必须归入以下两条**互相独立**的路线之一；
实验前请先读 `docs/EXPERIMENTS.md` 了解已完成实验与统一协议。

**共同起点（正式基线）**：S2 History Memory + Graph-ODE + CPM 架构，E2 损失。
基线契约：`configs/baseline_s2_e2.json`（训练脚本的共享配方也从此读取）；
构建/加载自检：`python baseline.py`；对照指标：`results/baseline/S2_metrics_test.json`。

任务 A 首轮研究已结题，正式保留 E2。后续架构研究固定该损失基线，
任务 A 不再扩展新的 Loss 消融或组合。

---

## 任务 A：损失函数优化

**状态：首轮研究完成，E2 保留（已结题）。** 最终实验档案见 `docs/EXPERIMENTS.md` §6.1。
L1–L3 不采用，P0 保留对照记录，P1 不定型；不实施 L4，不追加多 seed 或历史重训。

**目标**：完善 BOLD 神经生理信号预测的监督目标（不解决长时域退化本身）。

**基线**：S2 架构 + E2 Loss（λ_td=0.05、λ_ch=0.02，其余 one/CPM/dynamics/variance
项不变）。

**允许修改**：

- 损失函数数学形式与各项组合/权重（`train/losses.py`）。
- 损失实验配置与消融脚本（`experiments/loss/`，新结果写 `results/loss/`）。
- 对应损失的评估与消融入口（复用 `experiments/evaluate_variant.py` 与
  `analysis/next_point_eval.py`，不另建评估系统）。

**禁止**：修改模型主干、重构 Graph-ODE、增加状态转移模块、将架构优化混入
同一次 Loss 实验。每个 Loss 变体必须在同一 S2 架构、同 seed、同划分下与
**S2 + E2 基线**做配对消融（协议见 `docs/EXPERIMENTS.md` §0）。S0/E2 是历史
架构对照口径，不是当前 Loss 实验的基线。

**已完成首轮消融的方向**（完整结果见 `results/loss/compare/RESULTS.md`）：

1. 幅度自适应 Temporal Loss（Q1 过度响应 / Q5 低估的异方差加权）。
2. Horizon-Aware Loss（逐 horizon 加权策略，替代当前 gamma）。
3. Multi-Scale Temporal Loss（多尺度时间差分）。
4. CRPS 与时空联合概率损失（与任务 B 的概率模型区分：这里只动监督目标）。

**对照实验**：每臂 vs **S2 + E2** 配对检验；报告 t+1 MAE/PCC、Delta MAE、
Direction Accuracy、ODE H1–H8 MAE、Q1–Q5 分层、Delta Std Ratio。

**入口**：`experiments/loss/compare_phase1.py` 是 E 系列配对对比工具，
新实验沿用其产物结构（`results/loss/<arm>/` + `results/loss/compare/`）。

### 任务 A 实施状态（2026-10-22）

已在原损失类接入 L1 幅度自适应、L2 CPM/ODE 同时 horizon 加权、L3 多尺度差分；
原 E2 默认行为保留。P1 复用概率损失函数，新增 lag=1/2/4、p=0.5 的逐点 Variogram
训练模式，保留历史 Variogram 指标和 CRPS checkpoint 判据。未改模型架构。

归档说明见 `experiments/loss/README.md`，实际参数见各臂 `training_config.json`。
L0–L3 从同一 HC 起点独立微调，
P0/P1 从同一 S2 起点初始化残差头，主干全程冻结。所有臂使用实际 checkpoint 快照中的
原优化超参数；工作区已存在的调度器修复统一作用于本轮，L0/P0 已按相同配方重跑。
历史 S2 与 S4-1 数值不作为本轮配对结果。

L0–L3 和 P0/P1 的训练、test 评估及配对比较均已完成，225 名 test 被试、
同一批预测锚点，概率测试 K=64。DiffNext 环境使用 PyTorch 2.6.0+cu118 与两张 RTX A6000。
L0/L1/L2/L3 在 43/41/41/43 轮早停，P0/P1 完成 150 轮。首轮数值如下：

| 实验 | t+1 MAE | ODE H1–H8 MAE | H8 MAE | Delta PCC | Q1 点值 MAE | Q5 点值 MAE |
| --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.08066735 | 0.53289542 | 0.80251911 | 0.81518652 | 0.45090215 | 0.63155561 |
| L1 | 0.08062067 | 0.53286162 | 0.80238032 | 0.81523462 | 0.45125628 | 0.63077656 |
| L2 | 0.08060921 | 0.53286233 | 0.80252928 | 0.81522297 | 0.45081855 | 0.63162929 |
| L3 | 0.08067448 | 0.53292568 | 0.80253661 | 0.81515896 | 0.45091441 | 0.63161925 |

**确定性结论**：L1 的 ODE 平均 MAE 仅改善 0.00634%，Q5 Delta MAE 改善 0.666%，
但 Q1 Delta MAE 恶化 1.078%，CPM 差分也退化。L2 的 H4–H8 平均改善只有 0.00198%，
H8 略差，CPM 平均 MAE 退化。L3 没有多尺度独立收益，lag=1/2/4 误差均略差。
没有选出具有实用、互补收益的替代方案，不开展 L4 或失败候选的额外多 seed 训练。

**概率结论**：P1 固定 lambda_v=0.1409571896。历史 Variogram 下降 1.574%，
p=0.5 TV 下降 0.693%，Energy 下降 0.0720%；CRPS 仅下降 0.00293%，p=0.434。
覆盖率更接近有限 K 参照，但区间宽度增加约 2.38%。P1 仍弱于 model_iid 和同 K
常数高斯的联合评分，只得到单 seed 方向性证据，不定型为正式替代目标。

**最终选择**：保持 E2 原权重与 uncertainty 机制，不覆盖正式基线或 checkpoint。
全部对比表、被试改善比例、配对检验、Figure 1–5、负结果和概率汇总口径说明见
`results/loss/compare/RESULTS.md`。概率对比从保存的样本按“先锚点、再被试”重算，
与原 JSON 记录一致；旧 S4 subject CSV 的元素加权口径不用于本轮因果比较。
本轮为单训练 seed；现有证据足以作出“不替换 E2”的项目决策，不追加多 seed。
该结题决策不等同于已验证跨 seed 稳定性。最终工件按原路径归档，见 `results/loss/archive_manifest.json`；临时测试/对比脚本及中间产物已清理。

基础验证：50 项定向测试通过，默认 E2 在 8 种组合下与修改前实现逐位一致；
六臂命令通过解析，实际训练/评估均完成。所有概率边际汇总已与原 JSON 逐项核对。
未修改模型文件，未开展状态诊断、S4-2 或额外性能基准。

### 用户授权的独立补充：Temporal PCC（已完成）

原 L1–L3/P1 判定保持。根据后续明确请求，仅测试未来轨迹 Temporal PCC，
不把它视为已做过的候选。TP0=原 E2，TP1=E2 + 0.05×两头 Temporal PCC 之和；
同一 HC 起点、同一训练配方，两臂独立微调。每条轨迹至少 3 个有效未来点，
排除真实近常数轨迹，预测常数仍计入。实现入口 `experiments/loss/temporal_pcc.py`，
结果目录 `results/temporal_pcc/`。TP0/TP1 已分别在 43/31 轮早停并完成评估。
ODE Temporal PCC 从 0.70562083 升至 0.70604755，但 ODE/CPM 平均 MAE
分别恶化 0.00977%/0.04582%，没有净预测收益，不纳入正式 E2。
完整结果见 `results/temporal_pcc/RESULTS.md`；本结论限于本轮固定权重与单 seed。

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

任务 A 已结题并固定 E2；后续任务 B 使用该损失基线，不自动重启 Loss 组合实验。
