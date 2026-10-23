# TASK_HANDOFF（2026-10-22 基线定型轮）

本轮完成实验收敛、基线定型与项目清理。历史交接细节（条件注入轮 §0–§8、
S4 判读口径 §9）已吸收进 `docs/EXPERIMENTS.md`；重写前的完整原文备份在
`docs/archive/TASK_HANDOFF_pre_baseline_20261022.md`。

## 当前状态

- **正式基线 = S2 History Memory + Graph-ODE + CPM + E2 Loss**。
  契约：`configs/baseline_s2_e2.json`；自检：`python baseline.py`；
  权重：`checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt`。
- 实验记录：`docs/EXPERIMENTS.md`；后续路线：`docs/RESEARCH_PLAN.md`。
- 测试：`python -m pytest tests/` 全部通过（200 项，含基线契约、划分清单、
  脚本参数契约与调度器测试）。

## 本轮关键决定（要点）

1. S1（BOLD→Latent 反馈）与 S3（组合）实现已从代码中移除（git 历史可追溯）；
   结果保留于 `results/phase2/`。
2. MoE/RECC 病理专家线（`models/tfm_experts.py`、`train/tfm_coupling.py`、
   `train/condition_activation.py`、`analysis/analyze_tfm_experts.py` 及其测试）
   已删除；负结果结题记录在 `docs/Version3_docs_1003/`。
3. `scripts/Pretrain_HC_tfm.sh` / `Finetune_MDD_tfm.sh` 默认值改为基线口径
   （λ_td=0.05、λ_ch=0.02、history_memory=True）。
4. `experiments/` 重组为 `loss/`、`long_horizon/` 两路线 + 共享
   `evaluate_variant.py`；`results/` 同步建立 `baseline/`、`loss/`、
   `long_horizon/`。
5. 清理：删除 initconfound 作废实验（第一版 S1–S3 初始化混杂）、phase1/phase4
   训练日志与 smoke 产物、S4 巨量集合 npz（可从 checkpoint 重生成）、
   淘汰臂 checkpoint（S1/S3/E1）、`.runtime_tmp` 残留、pycache、`.trae` 草稿。
6. **冻结契约路径恢复**：`checkpoints/neurotwin_tfm_finetune/finetuned_best.pt`
   曾于 2026-10-20 被后续训练覆盖（sha 不符）；本轮将 E0 臂权重改名归档
   （`checkpoints/archive_e0_graph_ode_*`），并从
   `Version3_weights_and_results_1003/` 复制回原始冻结权重，默认路径 SHA256
   重新与 `configs/final_model.json` 一致（未改任何记录值）。
7. E0 臂原权重归档于 `checkpoints/archive_e0_graph_ode_*`（graph_ode、λ=0）。

## 2026-10-22 复核修正

- 基线契约更正：采样 seed=20261009、Cond Dropout seed=20261008、HC 预训练为
  `hc3/demo`；README/EXPERIMENTS 同步。
- 训练脚本改为从 `configs/baseline_s2_e2.json` 读取共享配方（helper：
  `scripts/_baseline_defaults.sh`），环境变量覆盖接口保留。
- `baseline.py` 对 checkpoint `config.args` 做全量契约校验；新增脚本参数与
  checkpoint 快照的一致性测试。
- 学习率调度改为按参数组相对衰减，并在解冻追加参数组时同步 scheduler
  （修复 backbone 组未进入余弦调度的问题）；现有 S2 权重不重训，修复从下一轮
  训练起生效。

## 遗留事项 / 下一步

- 后续开发按 `docs/RESEARCH_PLAN.md` 两条路线进行；不提前实现 L*/M*。
- S2 为单 seed 工程基线；正式统计定论需多 seed。
- 下次重训将首次使用修复后的学习率调度；与现有 S2 指标对比时需注明配方差异。
