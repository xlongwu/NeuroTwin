# NeuroTwin

当前模型使用 **TFM 编码器 + 历史功能耦合图 + 多字段临床条件 + 个体基线 profile + Graph-ODE 动力学**，另有 CPM 直接多步辅助头。

统一分析入口：`python -m analysis.run_study --config configs/analysis/study.json --stage all`。该入口保存独立的受试者级分析结果；图形重绘入口为 `python -m analysis.run_visualizations --help`。现有分析配置位于 `configs/analysis/`。

## 模型结构

历史 BOLD 经可逆归一化、时间 patch 和时间/脑区注意力编码。图输入从当前可见历史的 Pearson 相关绝对值计算，经 `SoftAnatomicalPrior` 融合与归一化后供编码器和动力学共享。这是功能耦合图，不能直接解释为白质结构连接或因果连接。

Graph-ODE 头结合图扩散、神经残差、临床条件、个体 profile 与可选外部控制，通过固定 Heun RK2 同步推进隐状态和 BOLD。CPM 用历史 tokens 一次预测多个未来时刻；它与持续状态积分是两个不同输出。方差头使用 detached 隐状态估计逐脑区方差。

| 文件 | 当前职责 |
| --- | --- |
| `models/tfm.py` | 模型组装、临床 FiLM、CPM、预测与持续状态接口 |
| `models/tfm_backbone.py` | 时间 patch、因果时间注意力、图引导脑区注意力 |
| `models/graph_prior.py` | 输入功能图与特征图融合，生成共享有效图 |
| `models/conditioning.py` | 独立字段编码、人口学编码、cat 增量融合、Condition Dropout |
| `models/tfm_dynamics.py` | 受控 Graph-ODE 向量场、RK2 积分、方差头 |
| `models/tfm_personal.py` | 基线 profile 编码与个体 FiLM |
| `models/twin_outcome.py` | 独立临床结局读出及标准化 |
| `models/common.py` | RevIN、图预处理及公共张量工具 |
| `models/__init__.py` | 架构版本标记 |

训练参数集中在 `train/cli.py`；验证选优与恢复状态位于 `train/training_state.py`。`analysis/checkpoint.py` 统一读取 checkpoint 配置、重建模型并加载权重，`utils/batches.py` 提供训练和评估共用的批次处理。

HC 使用 `hc3/demo`：Sex、Age、education。MDD 使用 `mdd8/cat`，在共享人口学编码上融合 episode、illness、medication、HAMD、HAMA。每个字段独立处理有效性与缺失；统计量只由训练集拟合，MDD 从 HC 起点继承共享字段统计量。`cat` 是选定方案，不能称为实测最优。

## 数据与训练

输入目录：

```text
data/
  Raw/HC/ROISignals_<id>.mat
  Raw/MDD/ROISignals_<id>.mat
  Rest-meta-MDD-V1V2-Merged-HC.xlsx
  Rest-meta-MDD-V1V2-Merged-MDD.xlsx
```

Raw BOLD 明确采用 `[T,F]` 布局，最多读取前 150 TR。默认以前 32 TR 计算固定校准统计量与个人 profile；历史和预测窗口不得让未来信息进入校准。图从历史信号计算，无需外部 Mask 或滑窗缓存。

```bash
bash scripts/pretrain_hc_tfm.sh
bash scripts/finetune_mdd_tfm.sh
```

脚本支持 `PYTHON`、`DATA_ROOT`、`CHECKPOINT_DIR`、`DEVICE` 和 `PRINT_CMD`。微调支持 `HC_WEIGHT` 显式指定 HC 起点；自动查找存在多个候选时要求明确指定。直接运行 `main.py --mode finetune` 必须提供 `--pretrained_weight`。

两脚本的默认配置为 116 ROI、context 16–64、评估 context 64、patch 4、主干宽度 256、4 层/4 头、CPM horizon 8、profile 宽度 32、每区间 2 个 ODE 子步。微调先训练条件模块，再解冻主干。具体参数以脚本和 checkpoint 的 `config.args` 为准。

训练保留单步、CPM、动态多步与方差损失。默认使用同方差不确定性任务加权，图正则单独控制。checkpoint 按验证集完整 H1–H8 轨迹的受试者等权 MAE 选优，单步指标继续记录。未指定控制序列表示零外部控制，不代表接受了真实刺激监督。

## 推理与分析

```bash
python inference_twin.py --help
python -m experiments.evaluate_variant --help
python -m analysis.run_study --help
python -m analysis.run_visualizations --help
python -m analysis.test_eval_report --help
python -m analysis.plot_model_analysis --help
python -m analysis.analyze_roi_disease --help
```

绘图源码位于 `analysis/` 和 `scripts/draw_architecture.py`；默认图片输出仍位于 `figures/model_analysis/` 和 `figures/architecture/`。分析输出与工具缓存已加入忽略规则。

`inference_twin.py` 提供基线校准、持续状态推进、观测更新和控制双轨迹模拟。模型接口包括 `initialize_state`、`advance_state`、`readout_state`、`update_state`、`predict_next_state`、`rollout_next_states`、`forecast_horizon` 与 `simulate_intervention`。

通过 `experiments.evaluate_variant` 从参数快照和条件统计量重建模型，再严格加载权重。当前仅支持 Graph-ODE、history_fc 和头部条件注入的 checkpoint；保留架构版本 6，因为现有有效权重的参数名称与形状未改变。旧离散头、单 HAMD 模型、MoE/RECC、主干条件 RMSNorm、门控残差、条件低秩及其专用工具链均已清除。

`models/twin_outcome.py` 与 `train/twin_outcome.py` 单独处理随访 HAMD/HAMA 变化读出。它使用基线信息和基线模拟摘要；需要真实随访标签训练，未拟合的读出不得报告疗效。控制模拟、条件敏感性和个人编码不自动构成真实刺激响应或跨访视稳定性的证据。

本次清理保留 `checkpoints/`、`data/`、已有结果和独立分析实现；不执行训练、不覆盖权重。Python `__pycache__` 是自动生成缓存。
