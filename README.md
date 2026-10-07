# NeuroTwin-TFM：病理条件驱动的脑动态基础模型

> **NeuroTwin-TFM: Pathology-Conditioned TimesFM-style Foundation Model for Brain Dynamics**
>
> 面向脑动态功能连接（dFC）预测的两阶段框架：先在健康对照（HC）数据上预训练结构连接（SC）引导的 TimesFM-3 风格时序主干，再在抑郁症（MDD）数据上通过病理 FiLM 条件（HAMD 评分）学习个体化病理调制，实现"通用动力学预测 + 病理条件调制"的个体化建模。

## 目录

- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [模型架构概览](#模型架构概览)
- [两阶段训练策略](#两阶段训练策略)
- [损失函数与图正则](#损失函数与图正则)
- [评估与可解释性分析](#评估与可解释性分析)
- [关键超参数](#关键超参数)
- [文档索引](#文档索引)
- [变更记录](#变更记录)

## 核心特性

1. **TimesFM-3 风格时序主干**：temporal patching（`--tfm_patch_len`）+ 逐 ROI Causal Temporal Attention，长 context 用 `[B,F,K]` 单窗口口径建模。
2. **SC 软解剖先验全层共享**：`SoftAnatomicalPrior` 一次计算 `A_eff = λ·A_SC + (1-λ)·A_func + ΔA`，以加性偏置注入全部 ROI attention，并由 One-Step 头做一次图混合。
3. **One-Step + CPM 双预测头**：One-Step 头以 x_t 为锚点建模下一 TR（零初始化残差，初始 = persistence）；CPM 头用 ROI × Horizon 查询一次非自回归输出 t+1..t+H 全轨迹。
4. **病理条件 FiLM 注入**：HAMD 评分 → `PathologyNormalizer`（仅用训练集拟合，随 checkpoint 保存）→ 零初始化 FiLM 调制两个预测头；冻结主干阶段条件适配器仍参与训练。
5. **双目标损失**：`TFMDualLoss` = one-step Huber + spatial PCC + γ 加权逐 horizon CPM Huber，不再使用数学等价的 abs+delta 双计项。
6. **严格 subject-level 数据划分**：同一被试样本不跨 train/val/test，支持按 HAMD 分数分层切分，防泄漏。
7. **完整评估链路**：next-state 主指标与被试级聚合、3 条 trivial baseline 对照、free rollout 逐 horizon 退化分析、CPM 逐 horizon 对照、FC/频谱层面指标、ROI 置换重要性（含 BH-FDR 校正）。

## 项目结构

```
NeuroTwin/
├── main.py                     # 训练入口：main 训练/验证循环与 CLI 定义
├── models/
│   ├── tfm.py                  # NeuroTwinTFM：唯一主干（Context-only RevIN + SoftAnatomicalPrior + One-Step/CPM 双头 + 病理 FiLM）
│   ├── tfm_backbone.py         # TFM 构件：TemporalPatchEmbed / CausalTemporalAttention / SCGuidedROIAttention / TFMEncoderBlock
│   ├── graph_prior.py          # SoftAnatomicalPrior：SC 软解剖先验（A_eff）
│   ├── pathology.py            # PathologyNormalizer：临床评分（HAMD）→ 条件向量
│   └── common.py               # BrainRevIN / prepare_sc_matrix / 变长前缀切片工具
├── train/
│   ├── losses.py               # TFMDualLoss（One-Step+CPM 双目标）+ SC 图正则
│   └── optim.py                # AdamW 参数组 / Warmup+Cosine 调度 / ModelEMA / 分阶段微调 / 权重加载
├── utils/
│   ├── dataloader.py           # NeuroTwinNextPointDataset / NeuroTwinDataLoader（连续 context 样本、subject-level 划分；支持 CPM future 目标）
│   ├── augmentation.py         # BrainSignalAugmentation 训练集增强
│   └── common.py               # set_seed / 任务维度推导 / 通用工具
├── tests/                      # 单元测试（pytest）：TFM backbone / 完整模型 / 双目标损失
├── analysis/
│   ├── metrics.py / checkpoint.py
│   ├── next_point_eval.py            # next-state 指标 + 被试级聚合 + baselines + free rollout + FC + CPM 逐 horizon
│   ├── analyze_low_pcc_samples.py    # 低 PCC 被试临床特征关联分析（消费 per_subject/per_task CSV）
│   ├── visualize_roi_importance.py   # 结合 AAL116 图谱的 ROI 重要性可视化（消费 feature_importance_<split>.json）
│   └── visualize_sig_region.py       # 按脑网络分组的显著脑区玻璃脑图（消费 feature_importance_<split>.json）
├── experiments/                # evaluate_variant：从 checkpoint 重建 NeuroTwinTFM 并统一评估
├── scripts/
│   ├── Pretrain_HC_tfm.sh      # 阶段一：HC 预训练
│   ├── Finetune_MDD_tfm.sh     # 阶段二：MDD 病理 FiLM 微调
│   └── Phase0_test_eval.sh     # 独立 test 集锁定评估
├── docs/                       # 模型架构、机理与可解释性文档（见文末索引）
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
| `y` | `[F, H]` | 下一时间点监督信号（H = 1，one-step 主目标） |
| `future` | `[F, R]` | t+1..t+R 真值（CPM 头监督与 free rollout 评估用） |
| `x_last` | `[F]` | 锚点 `x_t`（one-step 残差预测的基准） |
| `sc` | `[F, F]` | 结构连接矩阵（清洗 NaN/Inf → 对称化 → 截负 → log1p → 99 分位缩放 → 裁剪至 [0,1]） |
| `pathology_score` | `[1]` | HAMD 评分（仅 finetune 模式） |

默认 `F=116`（AAL ROI 数）、`T=150`（连续 TR 数）、`context_min/context_max=16/64`。
当 `data_root/Normlize` 缺失时可按 `--bold_source` 回退用偶数序号滑窗重构连续序列。

## 快速开始

### 阶段一：HC 预训练

```bash
bash scripts/Pretrain_HC_tfm.sh
# 或直接调用：
python main.py --mode pretrain \
    --data_root ./data --checkpoint_dir ./checkpoints \
    --name neurotwin_tfm_pretrain \
    --task_mode next_timepoint --context_min 16 --context_max 64
```

- 输出权重：`checkpoints/<name>/base_best.pt`（按验证 next-state MAE 早停选优）与 `base_last.pt`
- TensorBoard 日志：`checkpoints/runs/<name>_<时间戳>/`

### 阶段二：MDD 病理条件微调（NeuroTwinTFM FiLM）

```bash
bash scripts/Finetune_MDD_tfm.sh
# 或直接调用：
python main.py --mode finetune \
    --pretrained_weight ./checkpoints/neurotwin_tfm_pretrain/base_best.pt \
    --data_root ./data --checkpoint_dir ./checkpoints \
    --name neurotwin_tfm_finetune \
    --task_mode next_timepoint --context_min 16 --context_max 64
```

- 自动加载预训练主干（校验任务口径与结构签名），前 `--freeze_backbone_epochs` 个 epoch 冻结主干仅训练病理 FiLM 条件适配器，随后解冻联合优化
- 微调模式用**仅 train subjects** 的 HAMD 评分拟合 `PathologyNormalizer`（统计量随 checkpoint 保存，`--reuse_pretrained_norm_stats` 可复用已加载统计量）
- 输出权重：`finetuned_best.pt` 与 `finetuned_last.pt`

> 注意：`main.py` 中设备选择硬编码为 `cuda:1`（`torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')`），多卡机器请按需调整或通过 `CUDA_VISIBLE_DEVICES` 控制。

### 完整参数说明

运行 `python main.py --help` 查看全部 CLI 参数（数据、TFM 结构、优化器、损失权重、病理归一化、SC 软先验、next_timepoint 任务与评估开关等），脚本中的典型配置见 [scripts/Pretrain_HC_tfm.sh](scripts/Pretrain_HC_tfm.sh) 与 [scripts/Finetune_MDD_tfm.sh](scripts/Finetune_MDD_tfm.sh)。

## 模型架构概览

`NeuroTwinTFM` 前向流程：

```
BOLD context [B,F,K]
  → Context-only RevIN（μ/σ 只由历史 context 估计，--norm True）
  → A_eff = SoftAnatomicalPrior（λ·A_SC + (1-λ)·A_func + ΔA，一次计算全层共享）
  → Temporal Patching（--tfm_patch_len，K 非 p 整数倍时开头补零）
  → [B,F,P,D] token grid
  → SC-guided Temporal–Variate Block × N（--tfm_layers）
      RMSNorm → Causal Temporal Attention（逐 ROI 沿 patch 轴，严格 causal mask）
      → RMSNorm → SC-guided ROI Attention（加性偏置 β·log(ε+A_eff)，β 逐头可学习）
      → RMSNorm → FFN
  ├─ One-Step 头：z_t = 最后 causal token → z_(t+1)=F_θ(z_t, A_eff, c)（A_eff 图混合 + FiLM）
  │              → x̂_(t+1) = x_t + Δ̂（零初始化残差，初始=persistence）
  └─ CPM 头：q_{i,h} = e^ROI_i + e^horizon_h → 交叉注意力到全部 context token
             → 一次非自回归输出 x̂_(t+1:t+H)（--cpm_horizon，默认 8）
患者条件 c = [HAMD]（finetune）→ PathologyNormalizer → FiLM（零初始化，冻结期仍训练）
```

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| `BrainRevIN` | models/common.py | ROI 级可逆实例归一化，缓解个体尺度差异 |
| `SoftAnatomicalPrior` | models/graph_prior.py | SC 软先验：λ·A_SC + (1-λ)·低秩功能图 A_func + 低秩 ΔA，Sinkhorn 对称归一 |
| `TemporalPatchEmbed` / `CausalTemporalAttention` / `SCGuidedROIAttention` / `TFMEncoderBlock` | models/tfm_backbone.py | temporal patching + 因果时间注意力 + SC 偏置 ROI 注意力的编码块 |
| `OneStepDynamicsHead` / `CPMHorizonHead` | models/tfm.py | One-Step 状态转移头（§10）与 CPM 全 horizon 查询头（§11） |
| `PathologyNormalizer` | models/pathology.py | HAMD 评分归一化（robust_z 等模式），条件向量生成与反演 |
| `ConditionFiLM` | models/tfm.py | 零初始化 FiLM 调制（identity-at-init，不破坏预训练行为） |

## 两阶段训练策略

| | 阶段一 Pretrain（HC） | 阶段二 Finetune（MDD） |
| --- | --- | --- |
| 数据 | HC 组（`Normlize/HC` + `Mask/HC`） | MDD 组 + HAMD 临床评分 |
| 模型形态 | 无条件模块（`pretrain_mode=True`） | 加载预训练主干 + PathologyNormalizer + FiLM |
| 参数冻结 | 全参数训练 | 前 N epoch 仅训练 FiLM 条件适配器，之后解冻主干（主干学习率 × `backbone_lr_scale`） |
| 优化器 | AdamW（统一学习率） | AdamW（backbone / adapter / 损失权重分组学习率） |
| 调度器 | Warmup + Cosine 退火 | Warmup + Cosine 退火 |
| 输出权重 | `base_best.pt` / `base_last.pt` | `finetuned_best.pt` / `finetuned_last.pt` |

通用稳定性组件：AMP 混合精度、EMA 权重滑动平均、梯度裁剪、按验证 next-state MAE 的早停（`patience`）、全局固定随机种子（`set_seed`）。

## 损失函数与图正则

主损失 `TFMDualLoss`（`--task_mode next_timepoint`，one-step 主输出 + CPM 全 horizon 辅助输出）：

```text
L_one = Huber(x̂_(t+1), x_(t+1)) + λ_pcc·(1 − corr_ROI(x̂_(t+1), x_(t+1)))
L_cpm = Σ_h w_h·Huber(x̂_(t+h), x_(t+h)) / Σ_h w_h,   w_h = γ^(h−1)·mask_h
L     = λ_one·L_one + λ_cpm·L_cpm                    # 默认 λ_one = λ_cpm = 1
```

`--lambda_cpm 0` 可退化为纯 one-step 的 Stage 0 基线；`--cpm_gamma < 1` 对远期 horizon 降权。

SC 软先验图正则（`compute_graph_regularization`，train/losses.py）：

| 正则项 | 权重参数（默认） | 作用 |
| --- | --- | --- |
| 稀疏（A_eff 非对角 L1） | `--sc_sparsity_weight` (1e-3) | 抑制模糊全连接图 |
| 行熵 | `--sc_entropy_weight` (1e-3) | 鼓励图结构锐化 |
| 时间一致性（A_seq 一阶差分） | `--sc_temporal_weight` (0.0) | 逐窗功能图平滑（>0 才生成 A_seq） |

## 评估与可解释性分析

统一评估入口为 [experiments/evaluate_variant.py](experiments/evaluate_variant.py)。
模型结构与数据划分参数直接取自 checkpoint 内的训练配置快照（含 `task_mode` 闸门），
无需手工同步超参。默认在被试级 8:1:1 留出的独立 test 集评估（训练全程未参与选权重），
需要 val 时显式 `--splits test,val`：

```bash
python -m experiments.evaluate_variant \
    --ckpt checkpoints/neurotwin_tfm_finetune/finetuned_best.pt \
    --out_dir ./results/eval/neurotwin_tfm_finetune
```

基础产出（`<split>` 为 `test` 或 `val`）：

| 文件 | 内容 |
| --- | --- |
| `metrics_<split>.json` | next-state 主指标（MAE/RMSE/spatial PCC/R²/delta_direction）、被试级聚合、3 条 baseline（persistence/trend/AR(1)）、free rollout 逐 horizon 指标与相对 persistence 的 improvement、CPM 逐 horizon 段（模型 vs persistence）、FC（边级 + within/between 网络）与可选频谱、协议元信息 |
| `per_subject_metrics_<split>.csv` | 被试级明细（论文统计主口径：先被试内平均再跨被试统计，含 `hamd` 列） |
| `per_task_metrics_<split>.csv` | 任务级明细（`context_len` / `cutoff` 逐任务核对） |
| `predictions_<split>.npz` | 预测数组（仅 `--save_arrays`） |
| `feature_importance_<split>.json` | ROI 置换重要性（ΔMAE、p 值、BH-FDR q 值、排名、显著 ROI 列表；仅 `--feature_importance`） |

权重加载不完整时直接报错退出（strict 加载）。可选开关：

```bash
python -m experiments.evaluate_variant \
    --ckpt checkpoints/neurotwin_tfm_finetune/finetuned_best.pt \
    --out_dir ./results/eval/neurotwin_tfm_finetune \
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

指标口径：next-state 用 MAE / RMSE / **spatial PCC（沿 ROI 维，主口径）** / R² /
`delta_direction`（逐 ROI 变化方向一致比例）；CPM 段报告 t+1..t+H 逐 horizon 的
MAE/RMSE/spatial PCC 与 improvement；rollout 追加 temporal PCC 与
variance ratio（诊断方差塌缩）；FC 仅在 rollout 长度 ≥ `--fc_min_length` 时估计。

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

## 关键超参数

以下为脚本中的典型配置：

| 类别 | 参数 |
| --- | --- |
| 数据 | `num_rois=116`、`context_min=16`、`context_max=64`、`eval_context_length=64` |
| 优化 | `train_epochs=150`、`warmup_epochs=15`、`batch_size=32`、`patience=20`、`lr_init=5e-5 / lr_peak=1e-4 / lr_final=5e-5`、`weight_decay=1e-2`、`grad_clip=1.0` |
| TFM 结构 | `tfm_dim=256`、`tfm_layers=4`、`tfm_heads=4`、`tfm_patch_len=8`（G30_TFM 消融最优）、`tfm_ff_ratio=4`、`dropout=0.2` |
| CPM | `cpm_horizon=8`、`cpm_layers=2`、`cpm_gamma=1.0`、`cpm_huber_delta=1.0` |
| 微调专属 | `freeze_backbone_epochs=10`、`backbone_lr_scale=0.2`、`loss_lr_scale=0.5`、`adapter_lr_scale=1.0` |
| 损失权重 | `lambda_one=1.0`、`lambda_cpm=1.0`、`lambda_pcc=0.1` |

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/NeuroTwinTFM_模型结构流程图.png](docs/NeuroTwinTFM_模型结构流程图.png) | TFM 主干结构流程图 |
| [docs/Version3_docs_1003/NeuroTwin_TimesFM3_改进方案.md](docs/Version3_docs_1003/NeuroTwin_TimesFM3_改进方案.md) | TFM 设计方案（§30 V1 四项改动 + 双头 + 条件化协议） |
| docs/Version0-2 版本文档目录 | 历史（legacy 主干时代）架构与机理档案，仅供追溯；所述模块已随 legacy 移除 |

## 任务口径：Next-Timepoint Prediction（连续 BOLD → 下一 TR 全脑状态）

`--task_mode next_timepoint`（唯一支持的任务口径，也是 `main.py` 默认值）提供
「Next Brain-State Prediction」协议：

```
Raw ROI BOLD [F, T=150]（data/Normlize，已逐 ROI z-score）
    context  [B, F, 1, K]（K ∈ [context_min, context_max] 可变，按 K 分桶组 batch）
    target   [B, F, 1]   （预测下一个 TR；t+1..t+H 轨迹由 CPM 头与 free rollout 覆盖）
```

- **one-step 头以 x_t 为锚点做残差预测**（`x̂ = x_t + Δx̂`，零初始化），
  抑制退化成"复制最近 TR"；`--forecast_offsets` 固定为 [1]（多步预测由 `--cpm_horizon` 覆盖）；
- **损失 = one-step Huber + spatial PCC + CPM 逐 horizon Huber**
  （`--lambda_one/--lambda_cpm/--lambda_pcc/--cpm_gamma`）；
- **baselines**：persistence / linear trend / **AR(1)**（只用 train subjects 拟合），
  free rollout 阶段三者递归生成；
- **free rollout 评估**：horizon 1/2/4/8/16，报告 MAE/RMSE/spatial PCC/temporal PCC/
  R²/variance ratio 与 horizon degradation、相对 persistence 的 improvement；
- **FC** 仅在 rollout 长度 ≥ `--fc_min_length`（默认 32）时计算，
  并与三条 baseline 同口径对比；另按 `data/AAL116.xlsx`「对应网络」列分解为
  **within-network / between-network FC**（标签不可用时自动跳过）；
  频谱评估可选（`--eval_spectral`）；
- **被试级聚合 + 确定性评估协议**（固定 K_eval 与等间隔 anchor，禁止随机位置）；
- **checkpoint 判据 = val next-state MAE**；选主干预训练权重时用
  `--load_backbone_only`（显式打印 loaded / missing / incompatible / reinitialized
  四类键清单，形状不匹配的键不被静默忽略）。

## 变更记录

| 变更日期 | 文件名 | 主要修改 |
| --- | --- | --- |
| 2026-10-05 | models/{neurotwin,neurotwin_moe,decoder,common,pathology}.py、train/{moe,losses,optim,__init__}.py、main.py、utils/common.py、experiments/evaluate_variant.py、analysis/{analyzer,visualizer}.py、scripts/、README.md | 移除 legacy NeuroTwin 主干（NeuroTwin/MoE/DFCAdapter/BrainMDM/GraphODEDDI/预测头/MoE 正则/legacy CLI），唯一主干为 NeuroTwinTFM：main.py 展平双架构分支并删除 ~60 个 legacy 参数与 `--model_arch` 开关（内部固定 tfm，checkpoint 快照兼容）；`compute_graph_regularization` 移入 train/losses.py；evaluate_variant 仅重建 TFM（legacy 快照显式报错）；TFM checkpoint 与两阶段脚本行为不变 |
| 2026-10-05 | main.py、models/tfm.py、train/optim.py、analysis/next_point_eval.py、experiments/、tests/、README.md | G30_TFM 消融完成后的精简：`--tfm_patch_len` 默认 4→8（消融最优）；移除消融专用开关 `--tfm_use_patho_cond` 及配套分支（模型构造/冻结阶段/评估别名）；移除 shuffled-HAMD 负对照评估；移除消融框架（variants/run_experiments/registry/compare/base_config 与 test_ablation.py，代码归档于提交 `a673ca5`）；`experiments/` 仅保留统一评估入口 `evaluate_variant` |
| 2026-10-03 | models/tfm.py、main.py、train/optim.py、experiments/evaluate_variant.py | 新增 `--tfm_use_patho_cond`（§25.4 条件消融开关：不构建 Normalizer/FiLM、前向忽略评分）；`set_finetune_stage` 对无条件适配器的 TFM 自动跳过冻结阶段（避免优化器空参数组）；`_KWARG_ALIAS_TFM` 补 `use_patho_cond` 别名（修复快照重建 strict 加载失败） |
| 2026-10-03 | analysis/next_point_eval.py、experiments/evaluate_variant.py | shuffled-HAMD 负对照内建：`run_inference(pathology_override)` 按被试置换评分重评，`metrics_<split>.json` 新增 `shuffled_hamd` 段（TFM 附 CPM 逐 horizon 对照），summary 键 `*_shuffled_hamd_mae / *_shuffled_delta_mae`；顺带清除 run_inference 的历史重复定义（死代码） |
| 2026-10-03 | experiments/variants.py、experiments/run_experiments.py、tests/test_ablation.py | 注册 `G30_TFM` 消融组（§25 矩阵 9 变体：patch 1/8、no-SC/functional-only/无 ΔA、no-CPM、gamma=0.9、no-HAMD）；变体支持 `pretrained_from` 共享生产预训练；新增 6 项测试（条件开关/评分重映射/变体合法性/重建往返） |
| 2026-10-02 | models/tfm_backbone.py、models/tfm.py | 新建：TimesFM-3 风格 TFM 主干（TemporalPatchEmbed / CausalTemporalAttention / SCGuidedROIAttention / TFMEncoderBlock，方案 §4–§7）与 NeuroTwinTFM 完整模型（Context-only RevIN + SoftAnatomicalPrior 共享 A_eff + One-Step/CPM 双头 + 病理 FiLM 条件 + intervention 接口预留，方案 §8–§16/§30 V1） |
| 2026-10-02 | train/losses.py | 新增 `TFMDualLoss`：L_one = Huber + λ_pcc·(1−spatial PCC)（不再 abs+delta 双计）+ L_CPM = γ 加权逐 horizon Huber（方案 §20），含逐 horizon 明细与配置校验 |
| 2026-10-02 | utils/dataloader.py | `NextPointTrainView` 新增 `anchor_max_offset`（训练锚点保证 t+H 全 horizon 真值）、`NextPointEvalView` 新增 `all_future_steps`（所有 anchor 返回 future，宽度与 rollout_steps 取大者保证 collate 一致）；`NeuroTwinDataLoader` 透传 |
| 2026-10-02 | main.py、train/optim.py、utils/common.py | 新增 TFM 全套 CLI（tfm_dim/layers/heads/patch_len/ff_ratio/one_step_hidden_dim、cpm_horizon/layers/intervention、lambda_one/lambda_cpm/cpm_gamma/cpm_huber_delta）与 `evaluate_tfm` 验证循环（one-step 主指标 + CPM 逐 horizon MAE + persistence 对照）；`pretrain_config_signature` 纳入 model_arch/patch_len/cpm_horizon；`resolve_task_dims` 支持 TFM 校验 |
| 2026-10-02 | analysis/next_point_eval.py、experiments/evaluate_variant.py | 评估链路支持 TFM：`run_inference(collect_cpm)` 收集 CPM 全 horizon 预测，报告新增 `cpm_horizon` 段（模型 vs persistence 逐 horizon + improvement），summary 键 `*_cpm_mae_H*`；`build_model_from_args` 按 model_arch 分派（TFM 签名自省重建），评估 loader 传入 `eval_all_future_steps` |
| 2026-10-02 | scripts/{Pretrain_HC_tfm,Finetune_MDD_tfm}.sh、tests/{test_tfm_backbone,test_tfm,test_tfm_loss}.py、README.md | 新建 TFM 两阶段训练脚本与三套单元测试（21→29 用例：因果性、SC 偏置语义、双头契约、初始化不变量、损失闭式核对）；README 新增 TFM 章节 |
| 2026-09-27 | utils/dataloader.py、models/、train/、main.py、analysis/、experiments/、scripts/ | next_timepoint 数据路径与任务协议落地：`NeuroTwinNextPointDataset`、`NextTimepointLoss`、next-state 指标 + baseline + free rollout + FC + 频谱评估、ROI 置换重要性、两阶段脚本；统一删除旧任务口径及兼容别名 |
