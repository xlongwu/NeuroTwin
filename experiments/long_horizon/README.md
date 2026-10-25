# H32 训练与评价入口

本目录只保留可复用入口，已结束的一次性消融/归因脚本移除。

- `train_supervision.py`：读取 `configs/long_horizon_h32.json`，复用正式训练脚本。
- `evaluate_supervision.py`：同协议H8/H32配对评价工具。
- `refresh_rollout.py`：可选状态刷新诊断；真值刷新使用未来观测，不是自由预测基线。

最优权重为 `checkpoints/final/h32_best.pt`，已有结果在
[results/final](../../results/final/README.md)。直接评估单模型使用
`python -m experiments.evaluate_variant`。

```bash
# 默认仅打印命令，显式 --execute 才启动训练
python -m experiments.long_horizon.train_supervision --arm h32 --device cuda:0
```

H32配置保留所选run的真实科学参数，包括BS64/eval128；新训练输出到
`checkpoints/h32_runs/`、`results/h32_runs/`，不覆盖final。
不因项目清理自动启动训练。结论与限制见 [实验记录](../../docs/EXPERIMENTS.md)。
