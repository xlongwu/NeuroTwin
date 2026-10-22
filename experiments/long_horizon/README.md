# experiments/long_horizon/ — 长时域预测能力优化路线（任务 B）

后续所有长时域 / 状态演化 / 概率化实验放本目录，结果写 `results/long_horizon/`。

- `refresh_rollout.py`：D1（真值刷新）/ D2（自预测刷新）/ Free 三种推进方式的
  状态误差累积诊断，是长时域改进的标准诊断工具。注意 D1 使用未来真实观测，
  不是无观测自由预测的性能基线。
- 约束与候选方向见 `docs/RESEARCH_PLAN.md` 任务 B：以 S2 为初始架构、
  E2 为固定损失；概率模型使用独立配置与任务 A 区分。
- 统一评估入口：`experiments/evaluate_variant.py`；概率评估：
  `analysis/s4_probabilistic_eval.py`。
