# experiments/loss/ — 损失函数优化路线（任务 A）

后续所有损失函数实验放本目录，结果写 `results/loss/`。

- `compare_phase1.py`：E0/E1/E2（及后续臂）的配对对比与作图工具，
  产物结构为各臂 `results/<phase>/<arm>/` + `results/<phase>/compare/`。
- 约束与候选方向见 `docs/RESEARCH_PLAN.md` 任务 A：**只动监督目标，
  不动模型结构**；每臂与 E2 同架构、同 seed、同划分做配对消融。
- 统一评估入口：`experiments/evaluate_variant.py`（勿另建评估系统）。
