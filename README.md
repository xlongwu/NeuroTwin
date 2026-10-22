# NeuroTwin

基于个体化脑数字孪生的 BOLD 神经动力学预测模型：以连续 BOLD context 预测下一
TR 全脑状态（116 ROI），支持多步自由推进、病理条件注入与概率轨迹预测研究。

- 项目背景与设计溯源：`docs/Version3_docs_1003/`、`docs/Version4_docs_1012/`。
- 正式实验记录与结论：`docs/EXPERIMENTS.md`。
- 后续研究路线（损失优化 / 长时域优化）：`docs/RESEARCH_PLAN.md`。

## 正式基线（2026-10-22 定型）

**架构 = S2 History Memory + Graph-ODE + CPM；损失 = E2。**

- TFM 主干（`models/tfm_backbone.py`、`models/tfm.py`）：patch=8、dim=256、
  4 层 SC-guided encoder。
- 受控 Graph-ODE 动力学头（`models/tfm_dynamics.py`）：Laplacian 扩散 +
  FiLM 残差 + 控制门，Heun RK2 积分，附 S2 历史记忆 FiLM。
- CPM 全 horizon 头：ROI × Horizon 查询一次输出 H=8。
- 条件注入：cat 臂（HC: hc3 / MDD: mdd8），双头 FiLM。
- 损失：one/CPM/dynamics/variance（uncertainty 加权）+ Temporal Diff 0.05 +
  Cross-Head Consistency 0.02（`train/losses.py`）。

基线契约（结构/损失/种子/划分/权重 SHA256）唯一事实来源：
`configs/baseline_s2_e2.json`。自检与加载入口：

```bash
python baseline.py                 # 构建 + strict 加载 + SHA256 校验
python baseline.py --device cuda:0
```

## 环境与数据

- Python 环境：mamba（`/home/user/anaconda3/envs/mamba`），PyTorch。
- 数据：REST-meta-MDD（`data/`，不入版本库），116 ROI BOLD + 临床表
  `Rest-meta-MDD-V1V2-Merged-MDD.xlsx`；被试级 8:1:1 划分（seed=42，
  版本化 manifest `data/subject_split_*.json`）。
- 随机种子：模型/采样 2024、划分 42、Condition Dropout 20261008（互斥校验）。

## 训练与推理

```bash
# 正式基线口径（两脚本默认即 E2 损失 + S2 历史记忆；消融用环境变量覆盖）
bash scripts/Pretrain_HC_tfm.sh        # HC 预训练
bash scripts/Finetune_MDD_tfm.sh       # MDD 微调（起点为本目录 HC 权重）

# 完整编排（S2 臂复现入口，训练后自动评估+刷新诊断）
bash results/phase2_train/run_phase2.sh
```

评估与诊断：

```bash
# 统一评估入口（ckpt → 重建模型 → 指标/图落盘）
python -m experiments.evaluate_variant --ckpt <ckpt> --out_dir <dir> --splits test

# D1/D2 状态刷新诊断（长时域路线标准工具）
python -m experiments.long_horizon.refresh_rollout --ckpt <ckpt> --split test --steps 8

# S4 概率评估（K 集合 CRPS/Energy/Variogram/coverage/PIT）
python -m analysis.s4_probabilistic_eval ...
```

历史冻结推理入口（V4 discrete 时代契约，独立于 S2 基线）：
`python inference.py`（配置 `configs/final_model.json`，原始权重在
`checkpoints/Version3_weights_and_results_1003/`）；当前 graph_ode 推理：
`python inference_twin.py`。

## Checkpoint

| 位置 | 内容 |
|---|---|
| `checkpoints/phase2_S2/` | **正式基线**（finetuned_best + HC pretrain，SHA256 见基线契约） |
| `checkpoints/phase1_E2/` | E2 损失 / S0 对照臂 |
| `checkpoints/phase4_S4_1/` | S4-1 概率头（历史对照） |
| `checkpoints/Version*_*/` | 各版本冻结历史工件（勿动） |
| `checkpoints/archive_e0_graph_ode_*` | E0 消融臂归档 |

## 主要结果（225 test 被试；完整记录见 `docs/EXPERIMENTS.md`）

| 指标 | S2 基线 |
|---|---:|
| t+1 MAE / PCC | 0.08067 / 0.9945 |
| ODE MAE（H1–8 / H8） | 0.5329 / 0.8025 |
| Free rollout H8 MAE | 0.8106 |

关键结论：S2 相对 S0 多步 MAE −0.87%（p=2.1e-26）且单步不退化；E2 损失小幅
稳定改善；多步退化主因是状态误差累积（D1 真值刷新后 H8 ≈ 0.081 平稳）；
S4-1 改善概率边际但未超常数方差高斯基线（未合入主干）。

## 目录结构

```
models/        正式模型（tfm 主干、graph_ode 动力学、概率头、条件机制）
train/         损失与优化（losses、prob_losses、optim）
utils/         数据加载与通用工具
analysis/      统一分析评估（next_point_eval、s4_probabilistic_eval、metrics）
configs/       baseline_s2_e2.json（正式基线）+ final_model.json（历史冻结契约）
scripts/       正式训练入口（Pretrain_HC / Finetune_MDD）
experiments/   evaluate_variant（统一评估）+ loss/ + long_horizon/（两条研究路线）
results/       baseline/ + phase1-4（历史证据）+ loss/ + long_horizon/
checkpoints/   正式权重与历史冻结工件
tests/         当前有效实现的测试
docs/          EXPERIMENTS.md、RESEARCH_PLAN.md、版本归档
```

## 后续开发

两条独立研究路线（互不合并中间状态，建议独立分支/worktree）：

- **任务 A 损失函数优化**：从 `experiments/loss/` 与 `train/losses.py` 开始。
- **任务 B 长时域预测优化**：从 `experiments/long_horizon/` 与
  `models/tfm_dynamics.py` 开始。

两者共用 S2 + E2 基线（`python baseline.py` 验证）与统一评估协议；范围边界与
候选方案见 `docs/RESEARCH_PLAN.md`。
