# Next-Timepoint Prediction（连续 BOLD → 下一 TR 全脑状态）

> 任务口径：`--task_mode next_timepoint`（Phase 9）
> 研究表述：*Given a subject's historical whole-brain BOLD states and structural
> connectivity, NeuroTwin learns the conditional transition dynamics of brain activity
> by autoregressively predicting the next whole-brain state.*
>
> HC 阶段学 `p(x_{t+1} | x_{≤t}, SC)`；MDD 阶段学 `p(x_{t+1} | x_{≤t}, SC, HAMD)`
> 的个体化偏移（delta_HC + delta_MDD）。

本文档描述该任务口径的完整协议：数据与 shape、模型适配、损失、评估、参数、
运行命令与已知限制。

> **唯一任务口径**：`main.py --task_mode` 仅支持 `next_timepoint`（默认值）。

---

## 1. 任务定义与数据流

```
Raw continuous BOLD  [F, T]        （data/Normlize/<HC|MDD>/ROISignals_<id>.mat）
    ↓  slice context
context = series[:, t-K+1 : t+1]   [F, 1, K]     （W=1 单窗口，S=K；K ≤ context_max）
    ↓  SC-conditioned backbone（BrainRevIN → DFCAdapter → BrainMDM → GraphODE×N）
latent h_t
    ↓  ForecastHead（anchor = x_t）
Δx̂_(t+δ)                            [F, H]        （H = len(forecast_offsets)，默认 1）
    ↓  x̂ = x_t + Δx̂（RevIN 反归一化后为原始 BOLD 空间）
next-state supervision x_(t+δ)
```

- **每个真实 TR 都保留**：不再做 window averaging，也不再使用 50% 重叠滑窗；
- **subject 是统计独立单位**：train/val/test 先在被试级划分（沿用版本化
  `subject_split_<mode>.json` manifest，配置不一致直接报错）；
- **训练随机、评估确定**：训练动态采样 (K, t)，评估固定 `K_eval` 与等间隔 anchor。

## 2. 数据协议

| 项 | 取值 |
| --- | --- |
| 数据源 | `data/Normlize/<grp>/ROISignals_<id>.mat`（`[T=150, F=116]`，已逐 ROI 全序列 z-score）；缺失时回退用偶数序号滑窗重构 |
| context 长度 | `K ∈ [--context_min, --context_max]`（默认 16…64）随机采样，或用 `--context_lengths 16,32,64` 指定离散集合 |
| 预测偏移 | `--forecast_offsets`（默认 `1`，即下一个 TR）；`--enable_mtp True` → `[1,2,4,8]` |
| 样本张量 | `x [F,1,K]`（batch 后 `[B,F,1,K]`；prompt 标准布局 `[B,K,F]` 可由一次 transpose 得到）、`y [F,H]`、`x_last [F]`（= x_t）、`target_mask [H]` |
| 训练视图 | `NextPointTrainView`：每 item 绑定 (subject, K)（构造期抽定，供按 K 分桶），t 在 `[K-1, T-1-max_offset]` 内由 worker RNG 采样 |
| 评估视图 | `NextPointEvalView`：固定 K，等间隔确定性 anchor（`--eval_anchors_per_subject`，0=枚举全部）；另标 rollout anchor（`--eval_rollout_tasks_per_subject`） |
| batch 对齐 | `LengthBucketBatchSampler` 按 K 分桶（同一 batch 内 K 一致，无 padding 污染） |

### 2.1 归一化与无泄漏

- `BrainRevIN` 的 mean/std 只由**当前 context** 估计（`dim=(2,3)` 即 (W,S)），
  目标 `x_(t+δ)` 从不进入统计量；反归一化也使用同一组历史统计量；
- **已知数据层限制**：`data/Normlize` 的 z-score 使用逐 ROI 全序列统计量，属预处理
  阶段固定的仿射变换；模型侧无法撤销。若要求绝对严格的逐样本无泄漏，需要在数据
  预处理阶段改为 context 内归一化（见 §9 未完成事项）。

## 3. 模型适配（最小侵入）

主干（`BrainRevIN / DFCAdapter / BrainMDM / GraphODEDDI / SoftAnatomicalPrior`）
**未重写**，只补了「S 轴变长」支持：

| 维度 | 取值 |
| --- | --- |
| W（窗口轴） | **1**（整段 context 作单窗口） |
| S（时间轴） | **K ∈ [16, context_max]（变长，前缀切片）** |

- **S 轴前缀切片**：所有按 `S_max = context_max` 建模的 LayerNorm / Linear 在
  `K < context_max` 时取权重前缀（输入侧列前缀、输出侧行前缀），第 i 个 TR 恒对应
  权重第 i 个位置；`K == context_max` 时与定长实现逐位等价。
  涉及 `BrainMDM._pool_along_s / scale_gate_s`、`GraphODE.input_norm / q,k,v / out_proj
  / ffn / window_norm`、`SoftAnatomicalPrior.*`、`LowRankDelta`、预测头 `win_proj`。
- **预测头锚点**（`--head_anchor_mode`）：`last_timestep`（delta 模式，锚点 = x_t，
  模型只学 Δx̂，抑制“复制 x_t”退化）/ `zero`（absolute 模式，零锚点直接预测状态）。
  单点输出（`out_dim == 1`）时跳过 `standardize_future`（否则会把输出整体置零）。
- **GraphODE 跨窗注意力自动关闭**（`--ode_window_attn off`）：W=1 时该分支退化为
  逐 token 线性映射，关闭后既省算力又解除 `S % window_heads == 0` 约束；
- **MoE / 条件模块**保持不变，仅修两处单点退化：`base_norm = LayerNorm(1)` 与
  `standardize_future` 在 `out_dim == 1` 时会把输入置零，现按 `_scalar_out` 跳过；
  病理残差语义变为「下一状态残差」`delta_pred = delta_HC + delta_MDD`；
- **MTP 接口**：`pred_window = len(forecast_offsets)`、`pred_seq_len = 1`，同一个
  hidden state 并行输出多个偏移（Parallel MTP），预测头无需改结构。

## 4. 损失（`NextTimepointLoss`）

```
L_abs   = Σ w_bh·|x̂_(t+δ) − x_(t+δ)| / Σ w_bh                 （原始空间）
L_delta = Σ w_bh·|Δx̂_(t+δ) − Δx_(t+δ)| / Σ w_bh               （原始空间；Δx = x − x_t）
L_pcc   = 1 − Σ w_bh·corr_ROI(x̂_(t+δ), x_(t+δ)) / Σ w_bh       （spatial PCC）
L_total = λ_abs·L_abs + λ_delta·L_delta + λ_pcc·L_pcc          （+ λ_nll·NLL，默认关闭）
```

- **权重 `w_bh`** = 偏移权重（`--mtp_weights`，单步恒为 1）× 目标有效掩码
  （MTP 下序列末尾缺真值的偏移自动不参与）；
- **PCC 口径**：单时间点目标 `x_(t+δ) ∈ R^F` 只有一个时间点，因此训练/评估的 PCC
  一律是**沿 ROI 维的空间 PCC**（prompt §十四）；时序口径只在 rollout 指标中计算；
- **重要说明（L_delta）**：在 `prediction_target=delta`（x_t 锚点）下，
  `L_delta` 与 `L_abs` 数值重合（`x̂ = x_t + Δx̂` 是仿射关系），此时 λ_delta 等价于
  给同一目标再加权重；只有在 `prediction_target=absolute`（零锚点）下两项才提供
  不同梯度。两项都会如实记录到日志便于核对；
- **设计取舍（实测依据）**：最初把 L_delta 定义在 RevIN 归一化空间
  （`Δx_raw / stdev_local`）时，本数据存在近乎常数的时间窗（实测单窗口 stdev 最小
  0.003），会被极少数近常数 ROI 主导（实测把该项放大到 L_abs 的 ~9 倍），故改为
  原始空间定义；
- **概率项默认关闭**（`--lambda_nll 0`）：单点预测下 logvar 头的方差换算在近常数
  context 上不稳定（`exp(−logvar)` 可达 1e3 量级，实测 NLL≈5e3 完全主导损失），
  需要概率输出时显式开启并自行检查数值；
- **可选 rollout 损失**（`--enable_rollout_loss`，默认关闭）：模型自由滚动
  `--rollout_train_steps`（默认 2）步，`L_rollout = Σ mask·|x̂ − x| / Σ mask`，
  总损失加 `λ_rollout`（默认 0.2）；成本约为单步的 `1 + steps` 倍，第一版默认关闭。

## 5. 评估协议（`analysis/next_point_eval.py`）

| 模块 | 内容 |
| --- | --- |
| next-state | 逐 (task, offset) 的 MAE / RMSE / **spatial PCC** / R²，被试级 mean/std/median；另报告 **delta_direction**（逐 ROI 变化方向 `sign(x̂−x_t)` 与 `sign(x_true−x_t)` 的一致比例） |
| baselines | persistence `x_t`、linear trend `x_t + δ·(x_t−x_(t−1))`、**AR(1)**（逐 ROI 最小二乘，**只用 train subjects 拟合**，`n_pairs` 与 a 的范围写入报告）；三者都有 `delta_MAE_model_minus_*` 对照 |
| free rollout | `H ∈ --eval_rollout_horizons`（默认 1,2,4,8,16）：MAE/RMSE/spatial PCC/**temporal PCC**/R²/**variance_ratio**（Var(pred)/Var(true)，用于发现方差塌缩）+ horizon degradation + improvement vs persistence；baselines 同样**递归**生成 |
| FC | 仅在 rollout 长度 ≥ `--fc_min_length`（默认 32）时计算：FC(前 32 步) 上三角非对角边的 `fc_mae`/`fc_rmse`/`edge_pcc_mean`，模型与三条 baseline 同口径比较；另按 `data/AAL116.xlsx`「对应网络」列把边分为 **within-network / between-network** 两集合分别报告（标签缺失/ROI 数不匹配时自动跳过该块，不做标签猜测） |
| 频谱（可选） | `--eval_spectral`：Welch PSD 低频段相对功率 MAE/PCC 与 log-PSD 形状相关 |
| 聚合 | 被试内先平均、跨被试再统计（论文口径）；`per_subject_metrics_<split>.csv` 为主表，`per_task_metrics_<split>.csv` 供调试 |
| 产物 | `metrics_<split>.json`、两个 csv、（可选）`predictions_<split>.npz` |

确定性：同一 checkpoint、同一 split 两次评估的 (subject, K, t) 与全部指标逐位一致。

## 6. 参数

新增（Phase 9，`python main.py --help` 可见）：`--context_min`、`--context_max`、
`--context_lengths`、`--prediction_target {delta,absolute}`、`--causal_training
{random_context,full_sequence}`、`--forecast_offsets`、`--enable_mtp`、
`--mtp_weights`、`--lambda_abs/--lambda_delta/--lambda_pcc/--lambda_nll`、
`--enable_rollout_loss`、`--rollout_train_steps`、`--lambda_rollout`、
`--eval_context_length`、`--eval_anchors_per_subject`、
`--eval_rollout_tasks_per_subject`、`--eval_rollout_horizons`、`--eval_rollout_steps`、
`--fc_min_length`、`--load_backbone_only`。

自动生效（有提示打印，并写回 checkpoint 快照）：`--ode_window_attn off`、
`--head_scale_granularity timestep`、`--head_anchor_mode`（由 `--prediction_target`
推导）、`pred_seq_len = 1`、`pred_window = len(forecast_offsets)`、
`in_window = 1`、`in_seq_len = context_max`、跳过 `torch.compile`（变长 K 会导致按
shape 反复重编译）、跳过轮间监督（单点目标上 PCC 项无定义）。

**checkpoint 判据**：`val next-state MAE`（prompt §三十八，不使用单纯 PCC），
日志会打印该判据与「模型 vs persistence 的 ΔMAE」。

**训练期日志（prompt §四十）**：训练侧打印 `train/loss_abs|delta|pcc`（MTP 下另有
`t+δ` 分偏移 loss/PCC）；验证侧打印 `val next_MAE / next_RMSE / next_spatial_PCC`、
`persistence_MAE / trend_MAE`，并对验证集的 rollout 锚点做 free rollout，按 horizon
打印 `val rollout_MAE_H1/H2/H4/H8/H16`（TensorBoard 标签 `Val/Rollout_MAE_H*`），
数值按有效任务数加权。

## 7. 运行

```bash
# 阶段一：HC 预训练（连续 BOLD → 下一 TR，SC 条件）
bash scripts/Pretrain_HC_next_point.sh

# 阶段二：MDD 个体化微调（HAMD 条件 + MoE 病理残差）
bash scripts/Finetune_MDD_next_point.sh

# 直接 CLI 示例（MTP 消融：并行预测 t+1/t+2/t+4/t+8）
python main.py --mode pretrain --task_mode next_timepoint \
    --data_root ./data --name nt_mtp --enable_mtp \
    --mtp_weights 1.0,0.7,0.5,0.3

# 评估（统一入口，自动按 checkpoint 的 task_mode 分派）
python -m experiments.evaluate_variant \
    --ckpt checkpoints/neurotwin_nextpoint_finetune/finetuned_best.pt \
    --out_dir ./results/eval/neurotwin_nextpoint --splits test
```

## 8. 兼容性

- `--task_mode` 仅支持 `next_timepoint`，`resolve_task_dims` 对其它取值直接报错；
- 权重加载闸门 `check_pretrain_config_compat` 校验任务口径与结构签名，不一致时默认报错
  （不再静默按 shape 过滤）；
- 跨口径复用主干用 `--load_backbone_only True`：只加载主干键，预测头/MoE/条件模块
  重新初始化，并显式打印 `loaded / missing / incompatible / reinitialized` 四类清单
  （`incompatible` = checkpoint 中存在但形状不匹配的键，例如旧 ForecastHead，会被显式
  重新初始化而非静默忽略）；
- 同口径（next_timepoint → next_timepoint）预训练→微调可完整加载。

## 9. 未完成事项与已知限制

1. **方案 A（全序列 causal parallel training）未实现**：`--causal_training full_sequence`
   显式报错并说明原因（主干时间轴算子是双向卷积/池化，需要重写才能做位置级 causal
   mask）；当前实现为方案 B（随机 context → 下一时间点），未来信息从设计上不进入
   前向，因果性由数据构造保证；
2. **GraphODE 的真实 Δt 语义未实现**：`t` 仍被丢弃，ODE 仍是自治系统重复积分；按
   prompt §二十七 保留现状，未伪装成连续时间语义；
3. **数据层归一化泄漏**：`data/Normlize` 的全序列 z-score 属于预处理固定仿射变换，
   模型侧无法撤销（见 §2.1）；
4. **概率/不确定性**：next_timepoint 的 NLL 默认关闭，conformal 校准与 PICP 未接入
   （P3）；`--pred_head point` 为脚本默认；
5. **shuffled-HAMD 负对照未接入**（评估报告里明确标注 `available: False`）；
6. **`--visualize / --feature_importance` 不支持**（旧可视化基于 `[B,F,W,S]` 单窗口径）；
7. **AR(1) 只有一阶**：VAR（prompt §十八的可选增强）未实现；
8. **Sequential MTP / state-aware MoE router / 概率 next-state** 未实现（P3）。
