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

## 2026-10-23 新任务交接：H32 长程监督与选优

- 用户确认上一独立对话已解决完整模型解冻与核心参数梯度问题；接手时阅读现有修复，不重复按旧审查结论修改。本对话未重新验证该修复。
- 本次目标：在 S2 + E2 上实现直接覆盖 H32 的训练监督和对应 checkpoint 选优，建立长程预测对照所需的运行入口与评估支持。
- 已明确研究取舍：优先降低 16–32 TR 及更远期 BOLD 预测误差；个体化、HC–MDD 分析与模型内数值扰动是后续目标。真实刺激响应不属于当前任务。
- 当前缺口：正式基线的训练未来长度和选优范围均为 H1–H8，推理/评估虽能延伸，不能替代长程监督。
- 实现边界：保留现有 Graph-ODE、S2 记忆、CPM 和 E2 数学形式，仅接通必要的预测长度、采样、监督、配置、选优及评估；不同时引入新状态结构、删除 CPM 或追加损失。
- 复用现有入口，新增独立长程实验配置/输出位置；保持正式 H8 基线契约、已有 checkpoint、数据和结果不变。
- 对照要求：H8/H32 使用有效解冻路径、相同划分与随机种子；在相同且具有完整 H32 真值的锚点上比较，分别报告 H1–8、H9–16、H17–32，明确主选优指标。
- 如 H32 带来参数形状或权重迁移问题，采用明确方案并记录，不静默丢弃不兼容权重；具体实现由接手对话核对代码后决定。
- 本对话仅检查了工作区状态与交接所需文档，未实施 H32 功能、运行测试或启动训练。接手后进行必要的轻量检查，不默认启动正式训练或全量评估。
- 用户此前指定后续任务使用 gpt-6.1-sol、high；创建工具支持模型字段，但不支持单独设置推理强度。
