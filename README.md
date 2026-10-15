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
4. **字段级临床条件注入**：merged 表 8 个字段逐字段独立编码 → `cat` 融合 $\Phi$ → **零初始化**
   $\Delta E$（起点与 HC 侧 $c_{\text{demo}}$ 逐位相同）→ 双头独立 FiLM 调制 One-Step 与 CPM，
   支持 Condition Dropout；统计量与类别词表由 HC→MDD **按位继承**（仅 train subjects 拟合）。
   冻结主干阶段只有条件模块参与训练。另有 legacy 单标量 HAMD FiLM 通路
   （`PathologyNormalizer` + `--pathology_input_dim`），声明条件臂时由模型显式互斥。
5. **双目标损失**：`TFMDualLoss` = one-step Huber + spatial PCC + γ 加权逐 horizon CPM Huber，不再使用数学等价的 abs+delta 双计项。
6. **严格 subject-level 数据划分**：同一被试样本不跨 train/val/test，支持按 HAMD 分数分层切分，防泄漏。
7. **完整评估链路**：next-state 主指标与被试级聚合、3 条 trivial baseline 对照、free rollout 逐 horizon 退化分析、CPM 逐 horizon 对照、FC/频谱层面指标、ROI 置换重要性（含 BH-FDR 校正）。

## 项目结构

```
NeuroTwin/
├── main.py                     # 训练入口：main 训练/验证循环与 CLI 定义
├── inference.py                # 最终定型主模型加载与推理入口（predict_next_state / forecast_horizon）
├── configs/
│   └── final_model.json        # 最终定型配置唯一事实来源（结构签名、参数量、权重哈希、训练配方）
├── models/
│   ├── tfm.py                  # NeuroTwinTFM：唯一主干（Context-only RevIN + SoftAnatomicalPrior + One-Step/CPM 双头 + 病理 FiLM）
│   ├── tfm_backbone.py         # TFM 构件：TemporalPatchEmbed / CausalTemporalAttention / SCGuidedROIAttention / TFMEncoderBlock
│   ├── graph_prior.py          # SoftAnatomicalPrior：SC 软解剖先验（A_eff）
│   ├── pathology.py            # PathologyNormalizer：临床评分（HAMD）→ 条件向量
│   ├── conditioning.py         # 字段级条件注入：独立编码器 / cat 融合 / ΔE / Condition Dropout / AdaRMSNorm
│   └── common.py               # BrainRevIN / prepare_sc_matrix / 变长前缀切片工具
├── train/
│   ├── losses.py               # TFMDualLoss（One-Step+CPM 双目标）+ SC 图正则
│   └── optim.py                # AdamW 参数组 / Warmup+Cosine 调度 / ModelEMA / 分阶段微调 / 权重加载
├── utils/
│   ├── dataloader.py           # NeuroTwinNextPointDataset / NeuroTwinDataLoader（连续 context 样本、subject-level 划分；支持 CPM future 目标）
│   ├── condition_schema.py     # 8 字段白名单与清洗/码值协议、layout/schema 哈希、随机来源常量
│   ├── augmentation.py         # BrainSignalAugmentation 训练集增强
│   └── common.py               # set_seed / 任务维度推导 / 通用工具
├── tests/                      # 单元测试（pytest）：TFM backbone / 完整模型 / 双目标损失 / 病理专家 MoE / RECC
├── analysis/
│   ├── metrics.py / checkpoint.py
│   ├── next_point_eval.py            # next-state 指标 + 被试级聚合 + baselines + free rollout + FC + CPM 逐 horizon
│   ├── analyze_low_pcc_samples.py    # 低 PCC 被试临床特征关联分析（消费 per_subject/per_task CSV）
│   ├── analyze_tfm_experts.py        # TFM 病理专家机制分析：Router–Expert competence 诊断（§94–§95）
│   ├── visualize_roi_importance.py   # 结合 AAL116 图谱的 ROI 重要性可视化（消费 feature_importance_<split>.json）
│   └── visualize_sig_region.py       # 按脑网络分组的显著脑区玻璃脑图（消费 feature_importance_<split>.json）
├── experiments/                # evaluate_variant：从 checkpoint 重建 NeuroTwinTFM 并统一评估
│                                 # （2026-10-10：condition_study / condition_diagnostics 已删除，见 TASK_HANDOFF §0.27）
├── scripts/
│   ├── Pretrain_HC_tfm.sh      # 阶段一：HC 预训练（条件臂 hc3/demo，随机初始化）
│   └── Finetune_MDD_tfm.sh     # 阶段二：MDD 字段级条件微调（条件臂 mdd8/cat，起点=阶段一产物）
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

> **2026-10-10 第三次口径变更（TASK_HANDOFF §0.27）**：`experiments/condition_study.py` 与
> `main.py` 的 fresh 身份闸门已删除，脚本不再需要 `STUDY_ID`/`STUDY_ROOT`/`LEGACY`。
> 条件注入仍按选定方案固定为 **HC 侧 `hc3/demo` + MDD 侧 `mdd8/cat`**
> （`cat` 是**选定方案，不是实测最优**，见 §0.26）。
> ⚠️ 「HC 从随机初始化训练」「MDD 起点是本轮 HC」现在**不再由代码校验**，只由脚本不传
> `--pretrained_weight` 以及两脚本按顺序串接这两件事维持。

两个脚本共用这些环境变量：`PYTHON`（装了 torch 的 ≥3.10 解释器）、`CHECKPOINT_DIR`
（默认 `$ROOT/checkpoints`；想隔离本轮产物就指到一个空目录）、`DEVICE`、`DATA_ROOT`、
`PRINT_CMD`（=`1` 时打印完整命令行；默认只打印 mode/name/条件臂/epochs/bs/输出目录摘要，
生效参数的完整快照始终写在 checkpoint 的 `config.args` 里）。

### 阶段一：HC 预训练（条件臂 demo，随机初始化）

```bash
PYTHON="$HOME/anaconda3/envs/mamba/bin/python" \
  CHECKPOINT_DIR=./runs/run1010 DEVICE=cuda:1 \
  bash scripts/Pretrain_HC_tfm.sh
# 或直接调用（等价显式命令；宽度用 main.py 的锚点默认值 d_demo=16/C=32/H=64）：
python main.py --mode pretrain \
    --data_root ./data --checkpoint_dir ./runs/run1010 --name neurotwin_tfm_pretrain \
    --condition_group hc3 --condition_arm demo \
    --seed 2024 --sampling_seed 20261009 --cond_dropout_seed 20261008 \
    --task_mode next_timepoint --context_min 16 --context_max 64
```

- 输出权重：`$CHECKPOINT_DIR/<name>/base_best.pt`（按验证 next-state MAE 早停选优）
  与 `base_last.pt`；TensorBoard 日志在 `$CHECKPOINT_DIR/runs/<name>_<时间戳>/`
- 三个 seed 必须两两不同，由 `main.py:414 check_seed_separation()` 无条件校验
  （登记值在 `utils/condition_schema.py`）；同时沿用两个历史默认值（`--seed` 与
  `--sampling_seed` 都是 2024）会被拒

### 阶段二：MDD 条件微调（条件臂 cat，起点=阶段一的产物）

```bash
PYTHON="$HOME/anaconda3/envs/mamba/bin/python" \
  CHECKPOINT_DIR=./runs/run1010 DEVICE=cuda:1 \
  bash scripts/Finetune_MDD_tfm.sh
# 或直接调用：起点与继承统计量都用同一个 HC 权重文件
python main.py --mode finetune \
    --pretrained_weight ./runs/run1010/neurotwin_tfm_pretrain/base_best.pt \
    --condition_inherit_stats ./runs/run1010/neurotwin_tfm_pretrain/base_best.pt \
    --data_root ./data --checkpoint_dir ./runs/run1010 --name neurotwin_tfm_finetune \
    --condition_group mdd8 --condition_arm cat \
    --seed 2024 --sampling_seed 20261009 --cond_dropout_seed 20261008 \
    --task_mode next_timepoint --context_min 16 --context_max 64
```

- 微调脚本在 `$CHECKPOINT_DIR/neurotwin_tfm_pretrain*/base_best.pt` 里找起点：找到 0 个报错（阶段一还没跑），
  找到多个**拒绝替你猜**并要求 `HC_WEIGHT=<完整路径>` 显式指定

- 自动加载预训练主干（校验任务口径与结构签名），前 `--freeze_backbone_epochs` 个 epoch 冻结主干
  仅训练条件模块（独立字段编码器、$\Delta E$、双头 FiLM；`train/optim.py:556-576`），
  随后解冻联合优化（主干组学习率 × `--backbone_lr_scale`）
- 微调模式用**仅 train subjects** 的 HAMD 评分拟合 `PathologyNormalizer`（统计量随 checkpoint 保存，`--reuse_pretrained_norm_stats` 可复用已加载统计量）
- 输出权重：`finetuned_best.pt` 与 `finetuned_last.pt`
- ⚠️ 微调结束后 `main.py:1149-1159` 会**自动评一次 test** 并写
  `<save_dir>/eval/metrics_test.json`；要把 test 留作人工闸门，加
  `--no_eval_after_train`（TASK_HANDOFF §7.1 第 3 条）

> **重跑注意**：`checkpoint_dir/<name>` 已存在时 `main.py:801-808` 会把新目录自增成
> `neurotwin_tfm_pretrain2`、`neurotwin_tfm_finetune2`（不覆盖旧权重）。此时微调脚本会因为命中多个
> `neurotwin_tfm_pretrain*/base_best.pt` 而拒绝启动，需要显式 `HC_WEIGHT=<完整路径>` 指定起点。

> 注意：不给 `--device` 时 `main.py` 沿用历史默认（`torch.device('cuda:1' if cuda.is_available() else 'cpu')`，
> `main.py:472-473`）。脚本通过 `DEVICE=` 环境变量显式传 `--device`，不依赖该默认值；
> 多卡机器也可用 `CUDA_VISIBLE_DEVICES` 控制可见设备。

### 阶段二变体：病理残差专家 MoE 与 RECC（TFM 专属）

One-Step 头病理条件模式 `--tfm_patho_mode ∈ {none, film, moe}`（方案
`docs/Version3_docs_1003/NeuroTwin_TFM_RECC_详细应用方案.md`）：`film` 为基线
行为，`moe` 把共享 HC 动力学与病理残差专家解耦（`z_(t+1) = z_t + Δz_shared +
Δz_path`），病理模块仅在 MDD 微调阶段创建，HC 预训练权重完整复用。

> ⚠ 本节原来的 `Finetune_MDD_tfm_moe.sh` / `Finetune_MDD_tfm_recc.sh` 四条命令对应的脚本
> 已不在仓库里（早前的变体收敛中删除，`git status` 里是 `D`）。MoE/RECC 的模型能力仍在
> `models/` 内，需要跑该变体时请直接给 `--tfm_patho_mode moe` 等参数调用 `main.py`，
> 命令口径见方案 §56/§96–§100。

- 实验分层与验收顺序（A→B→C→D→E）见方案 §56/§96–§100；checkpoint 判据仍为
  val next-state MAE，RECC/路由正则不进入该判据
- 训练日志输出 Router–Expert 机制指标（top1 agreement / top2 recall / regret /
  spearman / competence entropy / oracle gap，方案 §63–§70）
- 专家机制分析（competence matrix、逐被试路由一致性、逐专家 ROI/网络修正画像）：

```bash
python -m analysis.analyze_tfm_experts --ckpt <finetuned_best.pt> \
    --out_dir results/tfm_experts --split test
```

### 最终定型主模型与推理入口（2026-10-07 定型，零训练）

最终主模型 = **NeuroTwinTFM + SC 软先验 + 单路 FiLM + One-Step/CPM 双头（patch=4）**，
主权重为现存 FiLM baseline `finetuned_best.pt`；不使用 MoE/Router/RECC/Pair-RECC。
定型决定与会话记录见 [TASK_HANDOFF.md](TASK_HANDOFF.md)（2026-10-07 第四次会话）；
最终配置的唯一事实来源为 [configs/final_model.json](configs/final_model.json)
（结构签名、参数量 6,029,728 / FiLM 2,048、权重 sha256、损失与历史训练配方）。

```bash
# 加载自检：哈希校验 → arch_version 校验 → FiLM 闸门 → strict 加载 → 摘要
python inference.py                # 加 --device cpu 可强制 CPU；--ckpt 可指定其它 FiLM 权重
```

```python
from inference import (load_final_model, predict_next_state,
                       forecast_horizon, normalize_raw_sc)

model = load_final_model()                       # eval 模式；HAMD normalizer buffers 随权重加载
out = predict_next_state(model, hist, sc, hamd)  # 下一状态：out['pred_next'] [B,F,1]
traj = forecast_horizon(model, hist, sc, hamd, steps=8)   # 2–8 步整段预测 [B,8,F]
```

- 输入口径沿用训练版本：`hist` 为上游逐 ROI 全序列 z-score 后的 BOLD `[B,K,F]`
  （K ≤ 64，训练分布 K∈[16,64]，评估协议固定 K=64）；`sc` 为数据侧
  `NeuroTwinNextPointDataset._normalize_sc` 清洗后的 SC（原始 Mask 矩阵先过
  `normalize_raw_sc`）；`hamd` 为原始 HAMD 评分（必要输入，模型内按 checkpoint
  保存的 robust_z 统计量归一化，禁止在 val/test 重新拟合）。
- One-Step 保持 residual → 加 x_last_norm → 同 context 统计量 denorm；CPM 输出
  完整状态预测，不加 One-Step 锚点、不用 One-Step 替换 CPM 第一帧。
- **主模型范围 H≤8**：`forecast_horizon(steps>8)` 明确报错，不静默切换权重或
  rollout 协议。H16 长程参考是 B（MoE）best 的固定 CPM 分块滚动
  （`checkpoints/neurotwin_tfm_finetune_moe/finetuned_best.pt`，独立归档配置，
  不得与主权重混用）。
- 已知限制：历史 cosine 调度 `eta_min=5e-5` 高于 MDD backbone 组初始峰值
  （`1e-4×0.2=2e-5`），不能把旧训练描述为 backbone 全程以 adapter 0.2 倍学习率
  退火——已列为下一次重训前的独立修复项（记录于 configs/final_model.json
  的 `training_recipe.known_issue_lr_schedule`），本轮不改该行为。
- 本轮零训练：未训练、未重新划分数据、未覆盖任何 checkpoint、未重跑全量评估；
  MoE/RECC 归档权重（B/C/D/E1/E2）经 `experiments.evaluate_variant` 按各自
  快照加载的路径不受影响。

### Graph-ODE 动态路线：twin 家族（2026-10-12 实施框架）

按 [docs/Version4_docs_1012/NeuroTwin_神经微分方程优化.md](docs/Version4_docs_1012/NeuroTwin_神经微分方程优化.md)
实施的受控条件图 ODE 路线。**本次交付的是代码框架**：未经真实受控数据训练的控制分支
只能输出未校准模拟；预测 std 不自动称校准置信区间；健康偏离不称诊断概率或因果病灶；
不得把旧 discrete checkpoint 指标当作本路线结果。

- 结构：graph_ode 动态头（组合 Laplacian + ConditionFiLM/PersonalFiLM + 状态相关控制门）
  + 同步 Heun RK2（`ode_steps=2, dynamics_dt=1.0`）持续状态推进 + VarianceHead 对角方差
  + ProfileEncoder（`personalized=True`，基线前 32 TR → 32 维固定 profile）
- 数据：`--bold_source raw`（`Raw/<group>/ROISignals_<id>.mat`，[T,F] 明确布局取前 150 TR，
  每 ROI 用最早 32 TR 计算固定 mu0/s0）+ `--graph_source history_fc`（当前可见历史 FC →
  SoftAnatomicalPrior，不要求 Mask）+ 可选 `--control_manifest`（CSV 两列
  `subject_id,controls_path`；NPZ 键 `u [T,F]`，锚点 t 的未来切片为 `u[t+1:t+H+1]`，
  缺尾按 future_mask 填零；无 manifest 时 controls=None，不伪造刺激标签）
- checkpoint：`ARCH_VERSION=6`；旧 discrete 快照按 discrete/personalized=False/legacy_mask
  重建并 strict 加载，不把旧 One-Step 权重迁移到 ODE 头
- 训练脚本输出目录家族：`hc_demo_twin` / `mdd_cat_twin`（不与旧
  `neurotwin_tfm_pretrain/finetune` 产物混用）

#### 新动态训练与复评

```bash
PYTHON="$HOME/anaconda3/envs/mamba/bin/python" \
  CHECKPOINT_DIR=./runs/twin DEVICE=cuda:1 bash scripts/Pretrain_HC_tfm.sh
PYTHON="$HOME/anaconda3/envs/mamba/bin/python" \
  CHECKPOINT_DIR=./runs/twin DEVICE=cuda:1 bash scripts/Finetune_MDD_tfm.sh
```

twin 脚本显式传入的新 CLI（旧构造默认保持 discrete / personalized=False /
legacy_mask / bold_source=auto，不受影响）：

| 参数 | twin 脚本值 | 说明 |
| --- | --- | --- |
| `--dynamics_mode` | `graph_ode` | `discrete` 保留原 One-Step 残差逻辑 |
| `--graph_source` | `history_fc` | `legacy_mask` 继续走外部 Mask 图 |
| `--bold_source` | `raw` | Raw 前缀校准，首个预测至少第 33 个 TR |
| `--personalized` / `--profile_dim` | `True` / `32` | profile 基线独立于当前 context |
| `--calibration_length` | `32` | Raw 标准化与 profile 用同一前缀 |
| `--ode_steps` / `--dynamics_dt` | `2` / `1.0` | 每区间 2 个 Heun 子步 |
| `--lambda_dynamics` / `--lambda_variance` | `0.5` / `0.05` | 动态多步(Huber+PCC)与方差 NLL 权重 |

复评沿用统一入口（评估 loader 自动传递 raw/history_fc/calibration/manifest 参数，
rollout 使用同一 profile 与对齐控制切片）：

```bash
python -m experiments.evaluate_variant \
    --ckpt ./runs/twin/mdd_cat_twin/finetuned_best.pt \
    --out_dir ./results/eval/mdd_cat_twin --save_arrays
```

#### 健康人口学参照（HC train 被试拟合）

```bash
# fit：每被试最早 64 观测 → 6 个描述量（acf1/acf2/low_power/spectral_centroid/
#      diff_rms/signed_fc_mean，频率单位 cycles/sample），人口学 ridge(alpha=1)
python -m analysis.twin_reference fit \
    --data_root ./data --mode pretrain --condition_group hc3 \
    --bold_source raw --calibration_length 32 \
    --out ./results/twin_reference/hc_demo_twin_ref.npz

# score：单被试历史（不足 64 个真实观测时不执行参照分析）
python -m analysis.twin_reference score \
    --reference ./results/twin_reference/hc_demo_twin_ref.npz \
    --history <被试 NPZ（键 bold，[T,F]，T>=64）> \
    --age 26 --education 16 --sex 1 \
    --out ./results/twin_reference/score_xxx.npz
```

#### 独立 HAMD/HAMA 随访变化读出（冻结动态模型与参照）

配对 CSV 表头字段严格为
`subject_id,baseline_visit_id,followup_visit_id,baseline_bold_path,control_path,followup_days,baseline_hamd,followup_hamd,baseline_hama,followup_hama,split`；
基线 NPZ 键 `bold [T,F]`（T≥64，取最早 64 TR），控制 NPZ 键 `u [8,F]`
（无治疗样本必须显式提供全零控制，不把缺路径当 sham）；同一 subject_id 不得跨 split。

```bash
python -m train.twin_outcome \
    --pairs <配对 CSV> \
    --dynamics_checkpoint ./runs/twin/mdd_cat_twin/finetuned_best.pt \
    --reference ./results/twin_reference/hc_demo_twin_ref.npz \
    --out_dir ./runs/twin_outcome
```

输出 `outcome_best.pt`（含 standardizer 与特征配置）；某分量完全没有有效训练标签时
直接报错，不输出随机模型。未拟合 outcome 前不产生任何疗效评分。

#### 孪生推理入口

```bash
python inference_twin.py \
    --checkpoint ./runs/twin/mdd_cat_twin/finetuned_best.pt \
    --raw data/Raw/MDD/ROISignals_<id>.mat \
    --subject_id <id> \
    --horizon 8 \
    --controls <控制 NPZ（键 u [T,F]，与序列 T 匹配）> \
    --observe 4 \
    --reference ./results/twin_reference/hc_demo_twin_ref.npz \
    --outcome ./runs/twin_outcome/outcome_best.pt \
    --followup_days 84 --baseline_hamd 22 --baseline_hama 14 \
    --out_npz ./results/twin_infer_<id>.npz --output ./results/twin_infer_<id>.json
```

- `--checkpoint` 必须显式指定，不自动搜索"最新"权重；`--reference` / `--outcome`
  分别为 `analyze_pathology` / `predict_outcome` 的必需显式文件
- Raw 前缀校准（固定 mu0/s0）→ profile 一次计算 → 持续状态 rollout → 控制双轨迹
  （`--controls` 或显式 `--zero_controls`）→ 真实观测更新（`--observe N`，新观测用
  同一扫描固定 mu0/s0 转数据尺度）
- Python API：`TwinSession`（`calibrate_scan / compute_profile / init_state / advance /
  observe / rollout / simulate / new_visit`，新访视 `new_visit(..., reuse_profile=True)`
  显式沿用已有 p）、`analyze_pathology(history, demo, reference_path)`、
  `predict_outcome(..., dynamics_checkpoint, outcome_path)`

### 完整参数说明

运行 `python main.py --help` 查看全部 CLI 参数（数据、TFM 结构、优化器、损失权重、病理归一化、SC 软先验、TFM 病理专家 MoE 与 RECC、next_timepoint 任务与评估开关等），脚本中的典型配置见 [scripts/Pretrain_HC_tfm.sh](scripts/Pretrain_HC_tfm.sh) 与 [scripts/Finetune_MDD_tfm.sh](scripts/Finetune_MDD_tfm.sh)。

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
| TFM 结构 | `tfm_dim=256`、`tfm_layers=4`、`tfm_heads=4`、`tfm_ff_ratio=4`、`dropout=0.1`；`tfm_patch_len` CLI 默认 8（G30_TFM 消融最优，另一研究候选），**最终定型主模型为 4**（checkpoint 快照固化，见 configs/final_model.json） |
| CPM | `cpm_horizon=8`、`cpm_layers=2`、`cpm_gamma=1.0`、`cpm_huber_delta=1.0` |
| 微调专属 | `freeze_backbone_epochs=10`、`backbone_lr_scale=0.2`、`loss_lr_scale=1.0`（CLI 默认）、`adapter_lr_scale=1.0` |
| 损失权重 | `lambda_one=1.0`、`lambda_cpm=1.0`、`lambda_pcc=0.1` |

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/NeuroTwinTFM_模型结构流程图.png](docs/NeuroTwinTFM_模型结构流程图.png) | TFM 主干结构流程图 |
| [docs/Version3_docs_1003/NeuroTwin_TimesFM3_改进方案.md](docs/Version3_docs_1003/NeuroTwin_TimesFM3_改进方案.md) | TFM 设计方案（§30 V1 四项改动 + 双头 + 条件化协议） |
| [docs/Version3_docs_1003/NeuroTwin_TFM_RECC_详细应用方案.md](docs/Version3_docs_1003/NeuroTwin_TFM_RECC_详细应用方案.md) | TFM 病理专家 MoE / RECC 方案（§1–§100，已结题的扩展研究线） |
| [docs/Version3_docs_1003/G30_TFM消融实验总结_20261005.md](docs/Version3_docs_1003/G30_TFM消融实验总结_20261005.md) | G30_TFM 9 变体消融结论（patch_len 最优、HAMD 条件等） |
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
| 2026-10-12 | models/{tfm_personal,tfm_dynamics,twin_outcome}.py（新增）、models/{tfm,graph_prior,__init__}.py、utils/dataloader.py、train/{losses,optim}.py、train/twin_outcome.py（新增）、main.py、analysis/{twin_reference,next_point_eval}.py（twin_reference 新增）、experiments/evaluate_variant.py、inference_twin.py（新增）、scripts/{Pretrain_HC_tfm,Finetune_MDD_tfm}.sh、README.md | 实施 graph_ode 动态路线（docs/Version4_docs_1012/NeuroTwin_神经微分方程优化.md，代码框架）：受控条件图 ODE（组合 Laplacian + ConditionFiLM/PersonalFiLM + 状态相关控制门）+ 同步 Heun RK2 持续状态推进 + VarianceHead + ProfileEncoder（新增 tfm_personal/tfm_dynamics）；graph_prior dp Xavier/dq 零初始化；dataloader 新增 bold_source=raw（前缀校准）/graph_source=history_fc/控制 manifest/锚点下界 max(K-1,calib-1)；TFMDualLoss 扩展动态多步与方差 NLL、dynamics_mask 与 cpm_mask 分离；main.py/optim 接通 CLI、loader、profile 解冻；ARCH_VERSION 5→6（旧 discrete 快照按默认重建、strict 加载、不迁移 ODE）；复评与 next_point_eval 传递新参数；新增健康人口学参照（twin_reference：6 描述量 + ridge）、独立 HAMD/HAMA 随访读出（twin_outcome：83 维特征 + masked Gaussian NLL）与孪生推理入口（inference_twin.py：Raw 前缀校准、profile/state、rollout、控制双轨迹、观测更新）；训练脚本切换 twin 家族 hc_demo_twin/mdd_cat_twin。**代码框架交付，未经真实数据训练** |
| 2026-10-07 | configs/final_model.json（新增）、inference.py（新增）、tests/test_final_model.py（新增）、README.md、TASK_HANDOFF.md | 最终基线模型定型与推理配置固化（零训练）：configs/final_model.json 固化最终结构/损失/训练配方/结构签名/参数量/权重 sha256（唯一事实来源）；inference.py 提供主权重 strict 加载入口（哈希校验 → arch_version 校验 → FiLM 闸门拒绝 MoE/HC 权重 → normalizer buffers 校验 → 参数量核对）与 predict_next_state / forecast_horizon 接口（H≤8，超限明确报错不切换协议；SC 预处理复用数据侧口径）；新增 18 项定型契约测试（95 用例全通过）；MoE/RECC 归档加载路径不受影响 |
| 2026-10-06 | models/tfm_experts.py（新增）、models/tfm.py、models/__init__.py、train/tfm_coupling.py（新增）、train/{optim,__init__}.py、main.py、analysis/analyze_tfm_experts.py（新增）、scripts/{Finetune_MDD_tfm_moe,Finetune_MDD_tfm_recc}.sh（新增）、tests/{test_tfm_experts,test_tfm_recc}.py（新增）、README.md | 实施 RECC 方案（docs/Version3_docs_1003/NeuroTwin_TFM_RECC_详细应用方案.md）：新增 TFM 病理残差专家模块（TFMPathologyRouter/Expert/MoE，`--tfm_patho_mode moe` 时 One-Step 头 z=z_t+Δz_shared+Δz_path，One-Step 关闭原 FiLM，CPM 不变）；`train/tfm_coupling.py` 实现 Expert competence→Router 校准（KL(sg(q)‖p_cpl) 梯度只更新 Router）、Pair-RECC、Router 正则与 §63–§70 机制指标；冻结阶段 MoE 可训练（is_conditioning_adapter）+ 独立参数组 tfm_moe；main.py 新增 §48–§51 CLI/fail-fast、dense warmup 与 λ_recc ramp 调度；HC 预训练结构与 arch_version 不变（§52/§92），finetune checkpoint 由 config.args 快照重建（evaluate_variant 自省兼容，已验证）；测试 29→57 用例 |
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
