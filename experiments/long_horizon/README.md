# experiments/long_horizon/ — 长时域预测能力优化路线（任务 B）

后续所有长时域 / 状态演化 / 概率化实验放本目录，结果写 `results/long_horizon/`。

- `refresh_rollout.py`：D1（真值刷新）/ D2（自预测刷新）/ Free 三种推进方式的
  状态误差累积诊断，是长时域改进的标准诊断工具。注意 D1 使用未来真实观测，
  不是无观测自由预测的性能基线。
- 约束与候选方向见 `docs/RESEARCH_PLAN.md` 任务 B：以 S2 为初始架构、
  E2 为固定损失；概率模型使用独立配置与任务 A 区分。
- 统一评估入口：`experiments/evaluate_variant.py`；概率评估：
  `analysis/s4_probabilistic_eval.py`。

## H32 直接监督对照（2026-10-23）

实验配置：`configs/long_horizon_h32.json`。复用正式 S2 + E2 配方和
`scripts/Finetune_MDD_tfm.sh`，不改变 Graph-ODE、记忆、CPM 或损失数学形式。
这是新的同条件 H8/H32 对照，不能把历史 S2 权重当作本轮重训的 H8 对照。

- 两臂从同一 S2 HC H8 `base_best.pt` 微调；模型 seed=2024、采样 seed=20261009、
  条件 Dropout seed=20261008，共用 `data/subject_split_finetune.json`。
- H8 训练监督 H1–8；H32 的 CPM、ODE、方差、时间差分和双头一致性监督全部覆盖
  H1–32。E2 的原权重、均匀 horizon 权重和第 5 个零基 epoch 一致性门控沿用基线。
- 两臂训练锚点均要求完整 32 步未来；训练仍随机采样历史长度和锚点。
  短序列不再采到没有合法 H32 起点的 context 长度。
- 两臂 `freeze_backbone_epochs=0`，首轮训练前走现有解冻路径，并沿用已修复的
  optimizer/scheduler/编译缓存处理。其余优化设置保持基线；现有历史 S2 使用旧调度，
  不应把这次重训差异全部归因于 horizon。
- 两臂统一按验证集 **Graph-ODE H1–32 被试等权 MAE** 选优及早停，指标键为
  `metric_mae_dynamics_h1_32_subject`，越低越好。H8 仅在验证时自由推进到 32 步，
  验证损失仍截取前 8 步，不向训练额外注入远期真值。
- 评估固定历史长度 64，每被试等距选取至多 16 个完整 H32 锚点。
  验证选优关闭额外 rollout 锚点，保证与最终分段评估使用同一网格。
  两臂的
  `(subject, cutoff, context_len)`、future mask 和真值必须完全一致。
  报告 H1–8、H9–16、H17–32 与总体 H1–32 的 ODE MAE，以及 persistence 参照。
  H17–32 是重点远期观察区间；总体 H1–32 是预先固定的主指标。

H8→H32 采用显式 `--pretrained_horizon_policy repeat_last`：
`cpm_head.horizon_embed` 原 H1–8 逐位复制，H9–32 重复原 H8 行；其他已有参数
完整加载，正常 HC→MDD 新条件模块沿用现有初始化。新增行随后独立接受各自步数的监督。
任何其他形状冲突、源参数无法加载或非条件参数缺失均报错，迁移报告随 checkpoint 保存。
默认策略仍为 `require_match`，不会自动迁移正式 H8 基线。

在目标 Python 环境中运行。以下默认只打印命令；加 `--execute` 才训练：

```bash
python -m experiments.long_horizon.train_supervision --arm h8
python -m experiments.long_horizon.train_supervision --arm h32
python -m experiments.long_horizon.train_supervision --arm h8 --device cuda:0 --execute
python -m experiments.long_horizon.train_supervision --arm h32 --device cuda:0 --execute
```

输出分别在 `checkpoints/long_horizon/h32_supervision/{h8,h32}/`，保留脚本已有的
目录防覆盖逻辑（重复运行会追加编号）。两臂均关闭训练结束后的自动 test 评估；
训练完成后显式选择本轮实际生成的 best 文件，运行共同锚点评估：

```bash
python -m experiments.long_horizon.evaluate_supervision \
  --h8 checkpoints/long_horizon/h32_supervision/h8/neurotwin_tfm_finetune/finetuned_best.pt \
  --h32 checkpoints/long_horizon/h32_supervision/h32/neurotwin_tfm_finetune/finetuned_best.pt \
  --device cuda:0
```

默认写入 `results/long_horizon/h32_supervision/metrics_test.json` 和
`per_subject_test.csv`，记录种子、划分哈希、锚点哈希、样本数及 H32−H8 配对差值
（负值表示 H32 较好）。`--split val` 可复核选优口径；`--out_dir`、`--data_root`
和 `--num_workers` 可覆盖评估位置及吞吐设置。标准 `experiments.evaluate_variant`
仍可生成各臂原预测头长度的完整诊断；跨 H8/H32 的共同长程比较使用本入口。

该配置不重训 HC 预训练，仅研究共同 HC H8 起点上的 MDD H32 微调监督。
实现与小模型检查不证明实际误差改善；收敛、显存需求和收益由后续正式实验验证。

GPU0 的 RTX A6000 实际尝试在 H32、batch=256 的首批训练中发生 OOM，
未生成可用 best checkpoint。实验快照可通过 `shared.batch_size` 和
`shared.eval_batch_size` 统一设置两臂的训练及验证 batch；未设置时仍沿用
训练 256、验证 512。减小训练 batch 会改变每 epoch 的 optimizer 更新次数，
需要单独记录，且 H8/H32 必须同时修改以保持对照一致。
