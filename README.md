# NeuroTwin：病理条件驱动的脑动态数字孪生框架

> **NeuroTwin: Pathology-Conditioned Mixture-of-Denoising-Experts Digital Twin for Brain Dynamics**
>
> 面向脑动态功能连接（dFC）预测的两阶段数字孪生框架：先在健康对照（HC）数据上预训练结构连接（SC）引导的神经动力学主干，再在抑郁症（MDD）数据上通过病理感知 MoE（NeuroTwinMoE）学习个体化病理残差，实现"通用动力学预测 + 病理残差修正"的个体化建模。

## 目录

- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [模型架构概览](#模型架构概览)
- [两阶段训练策略](#两阶段训练策略)
- [损失函数与 MoE 正则](#损失函数与-moe-正则)
- [评估与可解释性分析](#评估与可解释性分析)
- [关键超参数](#关键超参数)
- [文档索引](#文档索引)
- [变更记录](#变更记录)

## 核心特性

1. **SC 先验显式注入 + 可学习图动力学积分**：`DFCAdapter` 多阶图扩散适配结构连接先验，`GraphODEDDI` 以 Heun/RK2 多步积分建模连续动力学演化。
2. **双轴多尺度混合**：`BrainMDM` 同时在序列轴与窗口轴做多尺度卷积与池化，兼顾局部纹理与跨窗口演化。
3. **跨窗口因果注意力预测头**：六路特征融合（含 `WindowTemporalAttention`）+ 迭代 SC 约束细化，增强未来窗口形态一致性。
4. **病理条件驱动 MoE 个体化残差**（MoDE 风格）：HAMD 评分嵌入驱动专家路由与 FiLM 调制，训练期 multinomial 高熵探索、推理期幂次温度缩放路由，抑制专家坍塌。
5. **Next-Timepoint 预测目标**：连续 BOLD context → 下一 TR 全脑状态，以 `x_t` 为锚点做 delta 预测；损失 = 绝对状态 L1 + delta L1 + 空间 PCC（`NextTimepointLoss`）。
6. **严格 subject-level 数据划分**：同一被试样本不跨 train/val/test，支持按 HAMD 分数分层切分，防泄漏。
7. **完整评估链路**：next-state 主指标与被试级聚合、3 条 trivial baseline 对照、free rollout 逐 horizon 退化分析、FC/频谱层面指标、ROI 置换重要性（含 BH-FDR 校正）与玻璃脑可视化，以及模型内置虚拟干预接口。

## 项目结构

```
NeuroTwin/
├── main.py                     # 训练入口：main 训练/验证循环、evaluate_next_point 验证与 CLI 定义
├── models/
│   ├── neurotwin.py            # 主模型 NeuroTwin（编码器 + 预测头 + MoE 挂载）
│   ├── neurotwin_moe.py        # NeuroTwinMoE：病理感知 MoDE 专家模块
│   └── common.py               # BrainRevIN / DFCAdapter / BrainMDM / GraphODE 等公共模块
├── train/
│   ├── losses.py               # NextTimepointLoss 下一时间点预测损失 + rollout 损失
│   ├── optim.py                # AdamW 参数组 / Warmup+Cosine 调度 / ModelEMA / 分阶段微调
│   └── moe.py                  # 路由温度调度（Hold-then-Decay）+ 负载均衡/Z-Loss/熵/多样性正则
├── utils/
│   ├── dataloader.py           # NeuroTwinNextPointDataset / NeuroTwinDataLoader（连续 context 样本、subject-level 划分）
│   ├── augmentation.py         # BrainSignalAugmentation 训练集增强
│   └── common.py               # set_seed / 任务维度推导 / 通用工具
├── analysis/
│   ├── analyzer.py / metrics.py / checkpoint.py / visualizer.py
│   ├── next_point_eval.py            # next-state 指标 + 被试级聚合 + persistence/trend/AR(1) baseline + rollout + FC
│   ├── analyze_low_pcc_samples.py    # 低 PCC 被试临床特征关联分析（消费 per_subject/per_task CSV）
│   ├── visualize_roi_importance.py   # 结合 AAL116 图谱的 ROI 重要性可视化（消费 feature_importance_<split>.json）
│   └── visualize_sig_region.py       # 按脑网络分组的显著脑区玻璃脑图（消费 feature_importance_<split>.json）
├── scripts/
│   └── Pretrain_HC_next_point.sh / Finetune_MDD_next_point.sh # 阶段一 / 阶段二训练脚本
├── docs/                       # 模型架构、机理与可解释性实验设计文档（见文末索引）
└── checkpoints/                # 权重与 TensorBoard 日志（runs/）输出目录
```

## 环境依赖

核心依赖（当前开发环境版本）：

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| Python | 3.11 | 运行环境 |
| PyTorch | 2.5.1 (CUDA 12.4) | 训练与推理 |
| NumPy | 1.26 | 数值计算 |
| pandas | 3.0 | 临床评分表读取 |
| SciPy | 1.17 | SC 预处理与统计检验 |
| matplotlib | 3.10 | 分析可视化 |
| openpyxl | 3.1 | 读取 `.xlsx` 临床表 |
| tqdm | — | 进度条 |
| nilearn | 0.13 | 玻璃脑可视化（仅 `visualize_sig_region.py` 需要） |
| tensorboard | 可选 | 训练曲线记录（缺失时自动降级为空实现） |

安装示例：

```bash
pip install torch numpy pandas scipy matplotlib openpyxl tqdm nilearn tensorboard
```

## 数据准备

数据根目录（`--data_root`）按以下结构组织，预训练与微调分别使用 HC / MDD 两套分组：

```
data_root/
├── Normlize/
│   ├── HC/    # ROISignals_{subj_id}.mat，[T, F] 连续 BOLD 序列（已逐 ROI 全序列 z-score）
│   └── MDD/
├── Mask/
│   ├── HC/    # Mask_{subj_id}.mat（结构连接矩阵 SC）
│   └── MDD/
└── Rest-meta-MDD-V1V2-Merged-MDD.xlsx              # 临床评分表（微调必需，含 ID 与 HAMD 列）
```

`next_timepoint` 的样本由连续序列动态切出（`NeuroTwinNextPointDataset`），不落盘预切窗口：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `x` | `[F, 1, K]` | context：连续 BOLD 的历史 K 个 TR（W 轴恒为 1，K ∈ [context_min, context_max] 可变） |
| `y` | `[F, H]` | 下一时间点监督信号（H = len(forecast_offsets)，默认 1） |
| `x_last` | `[F]` | 锚点 `x_t`（delta 预测的基准） |
| `sc` | `[F, F]` | 结构连接矩阵（清洗 NaN/Inf → 对称化 → 截负 → log1p → 99 分位缩放 → 裁剪至 [0,1]） |
| `pathology_score` | `[1]` | HAMD 评分（仅 finetune 模式） |

默认 `F=116`（AAL ROI 数）、`T=150`（连续 TR 数）、`context_min/context_max=16/64`。
当 `data_root/Normlize` 缺失时可按 `--bold_source` 回退用偶数序号滑窗重构连续序列。

## 快速开始

### 阶段一：HC 预训练

```bash
bash scripts/Pretrain_HC_next_point.sh
# 或直接调用：
python main.py --mode pretrain \
    --data_root ./data --checkpoint_dir ./checkpoints \
    --name neurotwin_nextpoint_pretrain \
    --task_mode next_timepoint --context_min 16 --context_max 64
```

- 输出权重：`checkpoints/<name>/base_best.pt`（按验证 next-state MAE 早停选优）与 `base_last.pt`
- TensorBoard 日志：`checkpoints/runs/<name>_<时间戳>/`

### 阶段二：MDD 个体化微调（NeuroTwinMoE）

```bash
bash scripts/Finetune_MDD_next_point.sh
# 或直接调用：
python main.py --mode finetune \
    --pretrained_weight ./checkpoints/neurotwin_nextpoint_pretrain/base_best.pt \
    --data_root ./data --checkpoint_dir ./checkpoints \
    --name neurotwin_nextpoint_finetune \
    --task_mode next_timepoint --context_min 16 --context_max 64
```

- 自动加载预训练主干（校验任务口径与结构签名），前 `--freeze_backbone_epochs` 个 epoch 冻结主干仅训练 MoE，随后解冻联合优化
- 输出权重：`finetuned_best.pt` 与 `finetuned_last.pt`

> 注意：`main.py` 中设备选择硬编码为 `cuda:1`（`torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')`），多卡机器请按需调整或通过 `CUDA_VISIBLE_DEVICES` 控制。

### 完整参数说明

运行 `python main.py --help` 查看全部 CLI 参数（数据、模型结构、优化器、损失权重、MoE 超参、next_timepoint 任务与评估开关等），脚本中的典型配置见 [scripts/Pretrain_HC_next_point.sh](scripts/Pretrain_HC_next_point.sh) 与 [scripts/Finetune_MDD_next_point.sh](scripts/Finetune_MDD_next_point.sh)。

## 模型架构概览

`NeuroTwin` 前向流程：

```
x [B,F,W,S] ──► BrainRevIN（可选 ROI 级可逆标准化）
            ──► DFCAdapter（SC 引导多阶图扩散）
            ──► BrainMDM（双轴多尺度混合）
            ──► GraphODEDDI × n_block（图神经 ODE 积分 + 零初始化块级残差缩放）
            ──► post_fusion + GroupNorm
            ──► NeuroTwinForecastHead（六路融合 + 跨窗口因果注意力 + 迭代 SC 细化）──► base_pred
            ──► [仅 finetune] NeuroTwinMoE（条件路由 → 病理残差专家 → 残差细化）──► delta_pred
            ──► 输出 = base_pred + delta_pred ──► BrainRevIN 反归一化
```

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| `BrainRevIN` | models/common.py | ROI 级可逆实例归一化，缓解个体尺度差异 |
| `DFCAdapter` | models/common.py | SC 引导 0/1/2-hop 图扩散 + SE 门控，注入结构先验 |
| `BrainMDM` | models/common.py | 序列轴/窗口轴双轴 8 路分支混合 |
| `GraphODEDDI` | models/common.py | 4 分支增量动力学（SC 图注意力/时间卷积/跨窗注意力/FFN）+ Heun/RK2 积分 |
| `NeuroTwinForecastHead` | models/neurotwin.py | 六路特征融合预测头：`pred = anchor + trend + softplus(scale)·shape + refinement` |
| `NeuroTwinMoE` | models/neurotwin_moe.py | MoDE 风格病理残差专家（FiLM 调制、共享专家、条件路由） |

## 两阶段训练策略

| | 阶段一 Pretrain（HC） | 阶段二 Finetune（MDD） |
| --- | --- | --- |
| 数据 | HC 组（`Normlize/HC` + `Mask/HC`） | MDD 组 + HAMD 临床评分 |
| 模型形态 | 无 MoE 分支（`pretrain_mode=True`） | 加载预训练主干 + NeuroTwinMoE |
| 参数冻结 | 全参数训练 | 前 N epoch 仅训练 MoE，之后解冻主干（主干学习率 × `backbone_lr_scale`） |
| 优化器 | AdamW（统一学习率） | AdamW（backbone / MoE / 损失权重分组学习率） |
| 调度器 | Warmup + Cosine 退火 | Warmup + Cosine 退火 |
| 路由温度 | — | Hold-then-Decay：冻结期高温探索，解冻后衰减；专家使用 CV>0.5 时自动升温 |
| 输出权重 | `base_best.pt` / `base_last.pt` | `finetuned_best.pt` / `finetuned_last.pt` |

通用稳定性组件：AMP 混合精度、EMA 权重滑动平均、梯度裁剪、按验证 next-state MAE 的早停（`patience`）、全局固定随机种子（`set_seed`）。

## 损失函数与 MoE 正则

主损失 `NextTimepointLoss`（`--task_mode next_timepoint`，以 `x_t` 为锚点的 delta 预测）：

```text
L_abs   = Σ w_bh·|x̂_(t+δ) − x_(t+δ)| / Σ w_bh              # 绝对状态误差
L_delta = Σ w_bh·|Δx̂_(t+δ) − Δx_(t+δ)| / Σ w_bh            # delta 误差（Δx = x − x_t）
L_pcc   = 1 − Σ w_bh·corr_ROI(x̂_(t+δ), x_(t+δ)) / Σ w_bh    # 沿 ROI 维的空间 PCC
L_total = λ_abs·L_abs + λ_delta·L_delta + λ_pcc·L_pcc      # 默认 λ = 1 / 1 / 0.1
```

概率项（`--lambda_nll`）与短程 rollout 训练损失（`--enable_rollout_loss`）默认关闭。

MoE 训练正则（`compute_moe_regularization`，仅在 finetune 生效）：

| 正则项 | 权重参数（默认） | 作用 |
| --- | --- | --- |
| 负载均衡（Switch 风格 importance×load） | `--moe_load_balance_weight` (0.01) | 鼓励专家负载均衡 |
| Z-Loss（ST-MoE） | `--moe_z_loss_weight` (1e-3) | 惩罚路由 logit 幅度过大 |
| 熵正则 | `--moe_entropy_weight` (1e-3) | 抑制路由过度集中 |
| 多样性损失 | `--moe_diversity_weight` (0.001) | 惩罚 batch 内路由决策趋同 |

## 评估与可解释性分析

统一评估入口为 [experiments/evaluate_variant.py](experiments/evaluate_variant.py)。
模型结构与数据划分参数直接取自 checkpoint 内的训练配置快照（含 `task_mode` 闸门），
无需手工同步超参。默认在被试级 8:1:1 留出的独立 test 集评估（训练全程未参与选权重），
需要 val 时显式 `--splits test,val`：

```bash
python -m experiments.evaluate_variant \
    --ckpt checkpoints/neurotwin_nextpoint_finetune/finetuned_best.pt \
    --out_dir ./results/eval/neurotwin_nextpoint_finetune
```

基础产出（`<split>` 为 `test` 或 `val`）：

| 文件 | 内容 |
| --- | --- |
| `metrics_<split>.json` | next-state 主指标（MAE/RMSE/spatial PCC/R²/delta_direction）、被试级聚合、3 条 baseline（persistence/trend/AR(1)）、free rollout 逐 horizon 指标与相对 persistence 的 improvement、FC（边级 + within/between 网络）与可选频谱、协议元信息 |
| `per_subject_metrics_<split>.csv` | 被试级明细（论文统计主口径：先被试内平均再跨被试统计，含 `hamd` 列） |
| `per_task_metrics_<split>.csv` | 任务级明细（`context_len` / `cutoff` 逐任务核对） |
| `predictions_<split>.npz` | 预测数组（仅 `--save_arrays`） |
| `feature_importance_<split>.json` | ROI 置换重要性（ΔMAE、p 值、BH-FDR q 值、排名、显著 ROI 列表；仅 `--feature_importance`） |

权重加载不完整时直接报错退出（strict 加载）。可选开关：

```bash
python -m experiments.evaluate_variant \
    --ckpt checkpoints/neurotwin_nextpoint_finetune/finetuned_best.pt \
    --out_dir ./results/eval/neurotwin_nextpoint_finetune \
    --splits test --save_arrays --spectral
```

- `--rollout` / `--no_rollout`：强制开启 / 跳过 free rollout（默认沿用 checkpoint 配置）
- `--fc` / `--no_fc`：强制开启 / 跳过 FC 层面指标（默认沿用 checkpoint 配置）
- `--spectral`、`--spectral_tr`：额外输出 Welch PSD 低频段一致性与所用 TR
- `--save_arrays`：保存预测数组 `predictions_<split>.npz`
- `--feature_importance`：计算 ROI 置换重要性（逐个 ROI 沿样本维置换 context 中该 ROI 的取值，ΔMAE 作为重要性），产出 `feature_importance_<split>.json`
- `--feature_importance_max_tasks`（默认 256）：限定参与重要性计算的任务数（冻结前 N 个任务，保证结果可复现）
- `--feature_importance_permutations`（默认 0）：>0 时用经验置换零分布估计 p 值，代价随置换次数线性增加；默认 0 用单侧 t 检验近似
- `--feature_importance_fdr_alpha`（默认 0.05）：BH-FDR 显著判定阈值
- `--refresh_split_manifest`：重新生成被试级切分 manifest（默认复用）
- `--no_shuffled`：跳过 shuffled-HAMD 负对照

指标口径：next-state 用 MAE / RMSE / **spatial PCC（沿 ROI 维，主口径）** / R² /
`delta_direction`（逐 ROI 变化方向一致比例）；rollout 追加 temporal PCC 与
variance ratio（诊断方差塌缩）；FC 仅在 rollout 长度 ≥ `--fc_min_length` 时估计。
模型内置 `virtual_intervention` 虚拟干预接口（excitatory / inhibitory /
variance_boost / variance_suppress），支持机制层面的可干预性分析。

### 分析与可解释性工具

以下工具均消费上面的 next_timepoint 评估产物（`<eval_dir>` 即 `--out_dir`）：

```bash
# ROI 重要性条形图 / 网络汇总 / 热图（结合 AAL116 名称与网络标签，标注 BH-FDR 显著项）
python analysis/visualize_roi_importance.py \
    --input_json <eval_dir>/feature_importance_test.json

# 按脑网络分组的显著脑区玻璃脑图（优先取 FDR 显著 ROI，无显著项回落重要性 Top N）
python analysis/visualize_sig_region.py \
    --input_json <eval_dir>/feature_importance_test.json

# 低 spatial PCC 被试的临床特征关联（被试内 PCC 分布 + 10% 分位阈值 + HAMD 子项比较）
python analysis/analyze_low_pcc_samples.py \
    --eval_dir <eval_dir> --split test
```

- `analyze_low_pcc_samples.py` 读取 `per_task_metrics_<split>.csv`（`t+<δ>_PCC_spatial` 逐任务 PCC）
  与 `per_subject_metrics_<split>.csv`（被试级 `pathology_score`），支持 `--offset` 指定偏移。
- `visualize_roi_importance.py` / `visualize_sig_region.py` 读取 `feature_importance_<split>.json`
  （由 `--feature_importance` 生成），后者渲染玻璃脑图需本地 nilearn 与 AAL atlas。

**已知限制**：`--visualize` 仍基于已删除的单窗口 `[B,F,W,S]` 口径，next_timepoint 评估传入即报错，
请改用上述 `analysis/visualize_roi_importance.py` 与 `analysis/visualize_sig_region.py`。

## 关键超参数

以下为脚本中的典型配置：

| 类别 | 参数 |
| --- | --- |
| 数据 | `num_rois=116`、`context_min=16`、`context_max=64`、`eval_context_length=64`、`prediction_target=delta` |
| 优化 | `train_epochs=150`、`warmup_epochs=15`、`batch_size=32`、`patience=20`、`lr_init=5e-5 / lr_peak=1e-4 / lr_final=5e-5`、`weight_decay=1e-2`、`grad_clip=1.0` |
| 模型 | `n_block=2`、`ode_steps=3`、`ode_hidden_dim=256`、`num_scales=3`、`dropout=0.2`、`alpha=0.5` |
| MoE | `num_experts=4`、`top_k=2`、`moe_expert_hidden_dim=256`、`moe_gate_temp_start=1.5 → end=1.0`、`moe_inference_temperature=0.3`、共享专家开启、路由仅条件化 |
| 微调专属 | `freeze_backbone_epochs=10`、`backbone_lr_scale=0.2`、`loss_lr_scale=0.5` |
| 损失权重 | `lambda_abs=1.0`、`lambda_delta=1.0`、`lambda_pcc=0.1` |

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/next_timepoint_forecasting.md](docs/next_timepoint_forecasting.md) | 当前任务（Next-Timepoint Prediction）的完整协议：数据、损失、评估、参数与命令 |
| [docs/Version0_docs_0910/模型架构详细文档.md](docs/Version0_docs_0910/模型架构详细文档.md) | 各模块结构与张量流的完整说明 |
| [docs/Version0_docs_0910/模型机理分析文档.md](docs/Version0_docs_0910/模型机理分析文档.md) | 机理层面的设计动机分析 |
| [docs/Version0_docs_0910/模型结构与训练策略总结.md](docs/Version0_docs_0910/模型结构与训练策略总结.md) | 结构 + 数据 + 训练策略论文参考底稿 |
| [docs/Version0_docs_0910/全面模型分析指南.md](docs/Version0_docs_0910/全面模型分析指南.md) | 分析工具链使用指南 |
| [docs/Version0_docs_0910/可解释性实验设计文档.md](docs/Version0_docs_0910/可解释性实验设计文档.md) | 可解释性实验（置换重要性、虚拟干预等）设计 |

## 任务口径：Next-Timepoint Prediction（连续 BOLD → 下一 TR 全脑状态）

`--task_mode next_timepoint`（唯一支持的任务口径，也是 `main.py` 默认值）提供
「Next Brain-State Prediction」协议：

```
Raw ROI BOLD [F, T=150]（data/Normlize，已逐 ROI z-score）
    context  [B, F, 1, K]（K ∈ [context_min, context_max] 可变，按 K 分桶组 batch）
    target   [B, F, H]   （H = len(forecast_offsets)，默认 [1] → 预测下一个 TR）
```

- **预测头以 x_t 为锚点做 delta 预测**（`x̂ = x_t + Δx̂`，`--prediction_target delta`），
  抑制退化成"复制最近 TR"；`absolute`（零锚点）作为消融对照；
- **损失 = 绝对状态 L1 + delta L1 + spatial PCC**（`--lambda_abs/--lambda_delta/--lambda_pcc`，
  默认 1/1/0.1），可选概率项与短程 rollout 损失默认关闭；
- **baselines**：persistence / linear trend / **AR(1)**（只用 train subjects 拟合），
  free rollout 阶段三者递归生成；
- **free rollout 评估**：horizon 1/2/4/8/16，报告 MAE/RMSE/spatial PCC/temporal PCC/
  R²/variance ratio 与 horizon degradation、相对 persistence 的 improvement；
- **FC** 仅在 rollout 长度 ≥ `--fc_min_length`（默认 32）时计算（prompt 要求），
  并与三条 baseline 同口径对比；另按 `data/AAL116.xlsx`「对应网络」列分解为
  **within-network / between-network FC**（标签不可用时自动跳过）；
  频谱评估可选（`--eval_spectral`）；
- **被试级聚合 + 确定性评估协议**（固定 K_eval 与等间隔 anchor，禁止随机位置）；
- **checkpoint 判据 = val next-state MAE**（不再用纯 PCC）；选主干预训练权重时用
  `--load_backbone_only`（显式打印 loaded / missing / incompatible / reinitialized
  四类键清单，形状不匹配的键不被静默忽略）。

完整协议（shape、变长 S 轴前缀切片、损失数学定义与设计取舍、评估产物、参数表、
运行命令、已知限制）见 [docs/next_timepoint_forecasting.md](docs/next_timepoint_forecasting.md)。

## 变更记录

| 变更日期 | 文件名 | 主要修改 |
| --- | --- | --- |
| 2026-09-21 | README.md | 新建：补全项目简介、目录结构、环境依赖、数据准备、训练/评估流程、架构概览、超参数与文档索引 |
| 2026-09-27 | utils/dataloader.py | 新增 `NeuroTwinNextPointDataset` / `NextPointTrainView` / `NextPointEvalView` 与 `next_timepoint` 数据路径（连续 BOLD context → 下一 TR，按 K 分桶；确定性评估 anchor 与 rollout 子集） |
| 2026-09-27 | models/{common,graph_prior,pathology,neurotwin,neurotwin_moe}.py | S 轴（context TR 轴）变长前缀/行前缀支持（`prefix_linear_out`）、预测头 `head_anchor_mode`（last_timestep/zero）与单点输出保护、`predict_next_state` / `rollout_next_states` 标准接口、MoE 单点退化保护 |
| 2026-09-27 | train/losses.py | 新增 `NextTimepointLoss`（abs / delta / spatial PCC + MTP 权重 + mask）与 `compute_rollout_loss`、`spatial_pcc` |
| 2026-09-27 | main.py | 新增 next_timepoint CLI（context/offsets/lambda/eval/load_backbone_only 等）与训练/验证循环、日志、TensorBoard、checkpoint 判据（val next-state MAE） |
| 2026-09-27 | train/optim.py | `pretrain_config_signature` 支持 next_timepoint 口径；新增 `load_backbone_only`（显式汇报 loaded/missing/incompatible/reinitialized 四类键清单） |
| 2026-09-27 | analysis/next_point_eval.py | 新建：next-state 指标 + 被试级聚合 + persistence/trend/AR(1) baseline + free rollout + FC + 频谱评估 |
| 2026-09-27 | experiments/evaluate_variant.py | 按 `task_mode` 分派 next_timepoint（模型维度对齐、`evaluate_next_point` 入口） |
| 2026-09-27 | scripts/{Pretrain_HC_next_point,Finetune_MDD_next_point}.sh、docs/next_timepoint_forecasting.md | 新建：两阶段训练脚本与任务协议文档 |
| 2026-09-27 | analysis/next_point_eval.py | §26：FC 增加 within-network / between-network 分解（读 `data/AAL116.xlsx`「对应网络」列，缺失时跳过），模型与三条 baseline 同口径；§20：next-state 增加 `delta_direction`（逐 ROI 变化方向一致比例） |
| 2026-09-27 | main.py | §40：验证循环补充 free rollout 日志 `rollout_mae_H{h}`（tqdm 与 TensorBoard `Val/Rollout_MAE_H*`），数值口径按有效任务数加权 |
| 2026-09-27 | experiments/variants.py | §35：注册 `G20_NEXTPOINT` 组（Experiment B–F：next_timepoint / delta / rollout loss / MTP 全组合） |
| 2026-09-27 | utils/、models/、train/、main.py、experiments/、analysis/、scripts/、docs/、README.md | 统一任务口径：删除全部旧任务（滑窗 6→1 与 chunk 级被试轨迹）及其兼容别名与旧 CLI 参数，`--task_mode` 仅保留 `next_timepoint`；同步删除对应评估模块、冒烟脚本、协议文档与任务口径归一化等旧接口，并清理训练脚本与文档中的旧参数 |
| 2026-09-27 | analysis/fc_baselines.py、scripts/{Pretrain_HC,Finetune_MDD}.sh | 删除：`fc_baselines.py` 按已删除的 `[B,F,W,S]` 窗口口径编写（运行即形状不匹配报错），其 FC 基线能力已内建于 `next_point_eval` 的评估报告；旧训练脚本职责由 `scripts/*_next_point.sh` 完全覆盖。同步更新 base_config/variants/optim 与 README 中的脚本引用 |
| 2026-09-27 | analysis/next_point_eval.py、experiments/evaluate_variant.py | 新增 ROI 置换重要性（逐个 ROI 沿样本维置换 context 值，ΔMAE 作为重要性 + 单侧 t 检验 / 经验置换零分布 + BH-FDR），落盘 `feature_importance_<split>.json`；`evaluate_variant` 新增 `--feature_importance` 系列开关 |
| 2026-09-27 | analysis/{analyze_low_pcc_samples,visualize_roi_importance,visualize_sig_region}.py | 接入 next_timepoint 产物：前者改读 `per_subject/per_task_metrics_<split>.csv`（新增 `--offset`），后两者消费 `feature_importance_<split>.json` 并标注 FDR 显著项；`visualize_sig_region.py` 由硬编码脑区改为 JSON + AAL 网络标签驱动 |
