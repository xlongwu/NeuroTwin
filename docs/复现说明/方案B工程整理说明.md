# 方案 B 工程整理说明

日期：2026-11-06。本次以本地已验证 B 的代码路径、配置及 best 权重为依据，收敛正式实现；没有重新设计模型或启动正式重训。

## 正式模型及保留范围

预测流程为：Raw BOLD 最早 32 TR 逐 ROI 固定校准 → 最近 64 TR 的 RevIN → 历史 FC 与可学习功能图融合 → 时间 patch → 四层时空 Transformer → 字段条件 CPM → 反 RevIN → 完整 H1–32。

- 历史维度为 `[B,116,1,64]`，patch 长度 8，隐藏维度 256，四头注意力，前馈倍率 4。
- 每个编码块依次执行因果时间注意力、有效图引导 ROI 注意力及前馈计算。图使用历史绝对 FC、学习功能图、低秩残差与对称 Sinkhorn；模块原名 `SoftAnatomicalPrior` 不表示输入包含独立实测的解剖连接。
- CPM 采用共享 ROI 嵌入与 32 个时距嵌入作为查询，两层交叉注意力读取历史 token，经过 FiLM 与线性读出，直接输出 `[B,116,32]`。
- HC 条件为 `hc3/demo`：Sex、Age、education。MDD 为 `mdd8/cat`，继承上述字段及 HC 统计量，增加 episode、illness、medication、HAMD、HAMA。
- B 没有执行趋势季节拆分、频域记忆或 ODE。FFT 仅用于既有预测损失和频域评价。本次不新增这些计算。
- 辅助输出保留 `sc_prior`、`norm_stats`、`condition_embedding`；请求 `return_tokens=True` 时增加历史 token，用于个体表征分析。

## 损失与训练机制

六项损失和聚合顺序保持原 B 实现：重建 L1 权重 1；FC MSE 权重 0.05；频谱幅值 L1 权重 0.05；相邻时间差分 Huber 权重 0.5；完整时间 PCC 权重 0.1；四个连续 8 TR 窗口的局部时间 PCC 均值权重 0.05。Huber 阈值为 1。

保留目标有效 mask、FC／FFT 完整轨迹筛选、有效元素归一化和时间差分 H1 使用真实历史末帧的规则。没有引入分阶段重建加权、不确定性权重、跨头一致性或新增辅助目标。

| 项目 | 已验证 B 配方 |
|---|---|
| 优化器 | AdamW，betas 为 0.9／0.95，weight decay 0.01 |
| 学习率 | 初始 2e-5，峰值 5e-5，末值 1e-5；3 轮 warmup，按参数组比例余弦衰减 |
| 编码器学习率 | 主学习率的 0.2；CPM 与条件参数比例为 1 |
| Batch／轮数／早停 | 64；每阶段最多 150 轮；patience 12，冻结期不触发早停 |
| 冻结／解冻 | 前10轮冻结；解冻保留已有 Adam 状态，并接入同一调度进度 |
| HC 冻结期 | 完整 CPM 及条件编码器训练 |
| MDD 冻结期 | 仅条件编码器和 CPM FiLM 训练 |
| 梯度／精度 | 梯度裁剪 1；CUDA AMP；EMA 0.999；默认不 compile |
| 随机种子 | 模型 2024，采样 20261009，条件 dropout 20261008 |
| 训练采样 | 每被试8样本，固定 K64、随机合法锚点，保留数据加载器既有增强 |
| 验证采样 | 每被试8个确定性锚点，要求完整 H32 真值 |
| 选模 | 被试等权完整 H32 MAE，加 0.05 乘以“1 减完整时间 PCC”；越低越好 |

图、编码器、CPM、条件编码和损失公式没有因重构重新调参。Raw 加载、扫描校准、增强、条件统计拟合及被试划分使用原共享数据实现；训练入口仅构造 B 所需的任务视图。

## 代码组织与删除内容

```text
configs/neurotwin.json          唯一正式配方与权重登记
model.py                       模型构建、严格加载、配置／SHA256核对
models/
  tfm.py                       单一路径及公开预测接口
  tfm_backbone.py               时间与ROI注意力、patch、归一化
  graph_prior.py               历史功能图融合
  conditioning.py              HC／MDD字段条件
  common.py                    共享运算
main.py                        HC／MDD唯一训练循环
train/
  losses.py                    六项预测损失
  optim.py                     冻结、参数组、AdamW、调度、EMA、权重加载
  training_state.py            完整训练状态保存与恢复
  pathology_fingerprint.py     独立下游病理读出
  pathology_readout.py          训练集站点交叉验证读出
scripts/
  train.sh                     统一训练包装
  pretrain.sh / finetune.sh     阶段入口
  migrate_b_checkpoint.py      明确的一次性旧B权重迁移
analysis/
  evaluate.py / next_point_eval.py     H32统一评价
  rolling_one_step.py                  真实历史刷新单步
  condition_usage.py                   条件贡献及FiLM读数
  h18_probes.py / h18_importance.py     历史片段及ROI归因
  h18_graph.py / fc_matrix_plot.py     图与功能连接
  individual_baseline.py               历史token个体表征
  shared_mechanism.py / information_*  共享模型敏感度及表征读出
  normative_deviation.py / pathology_tests.py  HC参照及临床／残差分析
  evaluation_report.py / separated_metrics_report.py  正式结果汇总
```

删除内容包括：`tfm_dynamics.py`、`tfm_personal.py`、`individual_dynamics.py`、`twin_outcome.py`，对应独立训练模块和脚本，旧频域／分阶段／个体化配置，旧双头及频域实验测试，以及依赖已移除 ODE／profile 的专属分析、旧结果格式的时距图和旧批量制图执行器。

同时删除单步头、方差读出、profile、初态 FiLM、频域编码模块、A／C／联合切换、动态辅助损失、跨头一致性、淘汰参数组和旧选模分支。严格加载已集中到 `model.py`，删除重复的 `analysis/checkpoint.py` 包装。通用下游统计代码与所依赖的共享工具保留。

现有 H1–8 归因代码使用 B 的 H32 输出前8步，属于同一正式模型的分析子集。原 profile 专属分析已经移除；新个体表征来自实际使用的历史编码 token，不能将旧 profile 统计结果直接作为该表征的结果。

## 权重与历史产物

| 角色 | 当前权重 | 原始来源 |
|---|---|---|
| HC H8 初始化 | `checkpoints/方案B/初始化/hc_h8_to_h32.pt` | `checkpoints/baseline_1031/neurotwin_pretrain/base_best.pt` |
| HC H32 best | `checkpoints/方案B/neurotwin_pretrain/base_best.pt` | 原 B HC 第145轮 |
| MDD H32 best | `checkpoints/方案B/neurotwin_finetune/finetuned_best.pt` | 原 B MDD 第94轮 |

原 B best 所在目录为 `checkpoints/全时距双分支_20261104_231429/B/`。原 A／B／C 权重、日志、训练状态和结果不变；精简权重另存，不覆盖源文件。SHA256 同时登记在唯一配置及各权重目录的 `权重迁移记录.json`。

迁移使用明确的参数键白名单：保留 RevIN、图、字段条件、patch、编码块、最终归一化和 CPM 参数，保留键名不变；明确排除停用模块。仅 HC H8 初始化将 `cpm_head.horizon_embed` 以原训练使用的 `repeat_last` 扩展至 H32。HC／MDD best 没有扩展或重算参数。

精简结构为 v7，全部参数使用严格加载。迁移器只接受登记的 v6 来源和来源 SHA256，并拒绝覆盖已存在目标；迁移已经完成，日常预测无需再次执行。旧训练状态原样保留，新代码续训使用 v7 工作流生成的 `training_state.pt`，不把旧架构状态或 best 权重当作完整续训状态。

## 训练命令

以下命令从仓库根目录运行。默认数据根目录为 `data/`，算法默认值只从唯一 JSON 读取；命令行可指定设备、数据与输出路径。

```bash
# HC：保持实际 B 的 H8热启动→H32 配方
DEVICE=cuda:0 bash scripts/pretrain.sh

# MDD：默认以当前正式 HC best 为起点
DEVICE=cuda:0 bash scripts/finetune.sh

# 使用刚训练出的 HC best；以训练输出的真实路径为准
HC_WEIGHT=checkpoints/方案B/训练/neurotwin_pretrain/base_best.pt \
  DEVICE=cuda:0 bash scripts/finetune.sh

# 新工作流断点恢复
RESUME_TRAINING_STATE=checkpoints/方案B/训练/neurotwin_pretrain/training_state.pt \
  DEVICE=cuda:0 bash scripts/pretrain.sh
```

默认输出 `checkpoints/方案B/训练/neurotwin_pretrain/` 与 `neurotwin_finetune/`。已有目录会生成带序号的新目录。脚本默认保存完整训练状态，包括优化器、调度器、scaler、EMA、随机状态、采样生成器及早停进度。HC 预训练在已验证 B 中使用既有 H8 初始化；本次没有把这一机制改成随机初始化训练。

## 推理及评价

```python
import torch
from model import build_model, load_model_checkpoint

model, config = build_model(device='cpu', mode='finetune')
# 或加载本次新训练的v7权重：
# model, args, bundle = load_model_checkpoint(checkpoint_path, device='cpu')

# history为扫描固定校准后的[B,64,116]，condition必须用该权重的训练统计量编码。
with torch.no_grad():
    future = model.forecast_horizon(history, condition=condition)  # [B,32,116]
```

原始扫描及被试条件编码可使用 session：

```python
from inference_twin import TwinSession, load_raw_series

session = TwinSession('checkpoints/方案B/neurotwin_finetune/finetuned_best.pt', device='cpu')
session.subject_id = subject_id
session.calibrate_scan(load_raw_series(raw_path, session.num_rois))
future = session.forecast(cutoff=63)  # [32,116]，零起始cutoff，校准尺度
```

```bash
python -m analysis.evaluate --mode pretrain --device cuda:0 \
  --out_dir results/方案B/健康评估 --save_arrays
python -m analysis.evaluate --mode finetune --device cuda:0 \
  --out_dir results/方案B/抑郁症评估 --save_arrays
python -m analysis.evaluation_report --eval_dir results/方案B/抑郁症评估 \
  --out_dir results/方案B/抑郁症汇总
```

`--ckpt` 可指定新训练权重；`--feature_importance` 可计算 ROI 历史时间置换的重要性。评价输出统一为 `指标_测试集.json`、`表格/` 下的逐时距／阶段／逐被试表，以及可选预测数组。逐步空间 PCC 与沿阶段时间轴的时间 PCC 分开记录；保留 FC、频域、差分、幅度及各阶段指标。Persistence、历史均值、趋势和训练集拟合的 AR(1) 仅作为评价基线。

HC 参照用于 MDD 时显式指定 MDD 条件表，继承 HC 训练统计量。ROI 描述性可视化入口 `visualize_roi_importance` 和 `visualize_sig_region` 已改读正式评价 JSON 的 `mae_increase`，不再把未计算的 FDR 结果标为显著性结论。

下游残差分析流程继续使用 `rolling_one_step` → `normative_deviation` → `pathology_tests`；新产物写独立目录，避免混入旧缓存。H32 固定锚点预测与真实历史刷新的滚动单步是不同协议。

## 一致性与必要检查

重构前保存原 B 代码快照、确定性输入、真实字段编码、损失、随机种子和冻结／解冻一次参数更新结果；重构后以相同权重和输入核对。

| 参照 | 最大预测绝对差 | 总损失绝对差 | 冻结期参数更新最大差 | 解冻期参数更新最大差 |
|---|---:|---:|---:|---:|
| HC best | 0 | 0 | 0 | 0 |
| MDD best | 0 | 0 | 0 | 0 |

检查使用 CPU、原始完整模型尺寸、两条确定性历史与对应条件；另外对 HC／MDD 各一个真实测试被试窗口，比较完整、部分及空 mask 下的全部六项损失，逐项差异为零。训练更新复用相同种子、损失、梯度裁剪和 AdamW／调度器。该检查确认所测输入及一步更新等价，不代替完整多轮训练或跨硬件数值保证。

已实际完成的其他检查：

- 保留的测试集125项通过；Python编译、Shell语法及正式模块静态检查通过。
- 临时小模型、合成 Raw 与条件表执行 HC 两轮、MDD 两轮，覆盖冻结转解冻、EMA、训练状态保存、自动 H32 评价；HC 从状态恢复至第三轮通过。
- 两阶段分别验证 session 推理、ROI 置换、固定锚点 FC、滚动单步 CLI、条件贡献 CLI，当前报告与分协议汇总通过。
- 正式 HC／MDD 精简 best 的 SHA256、配置与全部参数键严格加载通过。
- 原始三个来源权重的 SHA256 不变；没有删除训练记录、数据或历史实验结果。

核验记录位于 `checkpoints/方案B/工程整理/一致性检查.json`，该目录同时保存本次检查日志。

没有执行正式全量重训、全队列下游重算或新的预测性能实验。已有 B 的测试性能来自原训练结果；本次只验证工程等价，不宣称性能提升。环境中的 pandas 可选加速依赖版本警告不影响上述检查；CPU 冒烟不替代新 GPU 环境的正式运行验证。
