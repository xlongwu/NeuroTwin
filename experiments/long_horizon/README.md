# H32 训练与评价入口（存档线）

> **v2 重定向（2026-10-25）**：H32 优化支线已收档，本目录只保留可复用入口与
> 预算内配对实验工具；`prediction_summary.py` 降级为 teacher-forced 口径的
> **描述性存档输出**。当前研究主线（HC 参照 + 病理异常定位）的护栏指标与
> 偏离信号入口是 `analysis/rolling_one_step.py` 与
> `analysis/normative_deviation.py`，站点匹配子集见 `utils/site_subset.py`
> （[v2 收档报告](../../docs/V2_CLOSING_REPORT.md)）。

本目录只保留可复用入口，已结束的一次性消融/归因脚本移除。

- `train_supervision.py`：读取 `configs/long_horizon_h32.json`，复用正式训练脚本。
- `evaluate_supervision.py`：同协议H8/H32配对评价工具。
- `refresh_rollout.py`：可选状态刷新诊断；真值刷新使用未来观测，不是自由预测基线。
- `prediction_summary.py`：H32 指标汇总（**存档·描述性**，teacher-forced 口径；
  v2 护栏指标族由 `analysis/rolling_one_step.py` 产生，两者不可混比）。

最优权重为 `archive/checkpoints/h32_best.pt`，已有结果在
[保留的历史 H32 源数据](../../results/paper_figures/complete_20261026/sources/h32/README.md)。直接评估单模型使用
`python -m analysis.evaluate`。

```bash
# 默认仅打印命令，显式 --execute 才启动训练
python -m experiments.long_horizon.train_supervision --arm h32 --device cuda:0
```

H32配置保留所选run的真实科学参数，包括BS64/eval128；新训练输出到
`checkpoints/h32_runs/`、`results/h32_runs/`，不覆盖final。
不因项目清理自动启动训练。结论与限制见 [实验记录](../../docs/EXPERIMENTS.md)。

## 冻结特征的频域监督配对实验

实现依据为已收档的研究计划（结论并入 [EXPERIMENTS.md](../../docs/EXPERIMENTS.md)）。以下命令默认只打印安排，
不会读取真实数据、创建结果目录或更新参数；明确加入 `--execute` 才执行。
缓存及实验输出必须使用新的目录，不覆盖 `results/paper_figures/complete_20261026/sources/h32` 或原 checkpoint。

```bash
# 一次缓存：train/val 各64人，每人4个完整 K64/H32 锚点
python -m experiments.long_horizon.frozen_features \
  --out-dir results/frequency_cache --device cuda:0

# 两臂短程适配：末次权重评价，不按 val 选择训练步数
python -m experiments.long_horizon.frequency_adaptation \
  --cache results/frequency_cache --out-dir results/frequency_pair \
  --device cuda:0 --updates 300 --batch-size 64 --microbatch 8
```

默认 checkpoint 为 `archive/checkpoints/h32_best.pt`。数据迁移后可显式传
`--data-root`、`--split-file`，必须使用现存划分与 checkpoint 中的条件统计量。
缓存按 checkpoint 的 sampling seed 固定选择被试，采用现有确定性锚点规则；
人数限制为0表示该 split 全部被试。分片保存完整历史 tokens、初态、条件、图、
profile、历史归一化统计、目标及原两头预测，不保存32步 latent 轨迹。
提取前打印预计 token 容量，提取时核对缓存重放与原 CPM 输出。

只有 CPM 参数参与适配。原目标复用现有 Huber、空间 PCC、时间差分、
双头一致性和已启用的时间 PCC；任务权重固定为 checkpoint 元数据中的值。
元数据缺失时报告明确标记固定替代目标，不声称恢复原学习权重。
实验臂额外使用完整 H32 的正交归一化 rFFT 复数残差模长均值，排除 DC，
不使用未来方差定标、不做功率谱单独匹配、不换算未经核实的 Hz。
首个 train 有效 batch 的 CPM 梯度范数仅校准一次，使频域项约占原梯度的10%。

两臂共用初始化、seed、样本顺序和 microbatch，显式保留 CPM 的训练期 dropout。
新建 AdamW，betas 为原实现的 `(0.9, 0.95)`，固定学习率使用 checkpoint 的
`lr_init`，weight decay 和梯度裁剪沿用原配置；不恢复优化器、不用 EMA 或调度器。
有效 batch 默认64，microbatch 默认8，最后不足一个有效 batch 时按实际样本数
加权梯度；每臂最多300次更新。运行前应在实际设备上用较短更新预算确认资源需求，
当前实现不承诺未经测量的耗时或显存。

只有初筛出现信号后，才考虑扩大全 val。追加提取通过 checkpoint SHA、划分和
被试/锚点标识复用已有分片；新的锚点选择替换 active 列表，旧记录仍保留，
避免将4锚点与16锚点混合后误作正式评价。以下示例仅追加全 val，保留 train 缓存：

```bash
python -m experiments.long_horizon.frozen_features \
  --out-dir results/frequency_cache --extend --splits val \
  --subjects 0 --anchors 16 --device cuda:0

# 仅评价已生成的两臂权重；不继续适配
python -m experiments.long_horizon.frequency_adaptation \
  --cache results/frequency_cache --pair results/frequency_pair/pair.pt \
  --out-dir results/frequency_full_val --device cuda:0
```

`pair.pt` 保存两臂 CPM 权重、一次校准信息、优化器设置、逐更新损失、训练锚点与
最初验证锚点；`report.json` 比较原 ODE、原 CPM、控制臂、频域臂。
报告覆盖 H1、H1–8、H17–32、H1–32 MAE、逐步方向与变化误差，
多步片段另报时间 PCC、有效比例和标准差比，以及被试级配对差异、收益比例与
1000次被试 bootstrap 的描述性95%区间。常数预测在有效目标上记零 PCC，
不因删除常数预测抬高均值；目标常数轨迹排除并记录有效比例。
已反复探索的 val 区间不代表独立确认。

仅全225人、原16锚点规则的 val 才检查文档中的正式推进门槛。
数值门槛通过仍须检查被试收益分布；小样本只标记 `screening_only`。
不会自动运行 ODE 适配、扩大训练、扫系数或读取 test。
本轮基础验证为合成 CPU 小模型测试，真实数据提取、显存及改善幅度留待实际实验。
