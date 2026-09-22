<<<<<<< HEAD
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
5. **多目标不确定性加权损失**：PCC / MAE / 一阶差分 / 窗口内标准差四项，log-variance 可学习权重自动平衡。
6. **严格 subject-level 数据划分**：同一被试样本不跨 train/val/test，支持按 HAMD 分数分层切分，防泄漏。
7. **完整可解释性分析链路**：回归指标、ROI 置换重要性 + FDR 校正、专家-HAMD 分层分析、玻璃脑可视化、虚拟干预接口。

## 项目结构

```
NeuroTwin/
├── main.py                     # 训练入口：evaluate、main 训练循环与 CLI 定义
├── models/
│   ├── neurotwin.py            # 主模型 NeuroTwin（编码器 + 预测头 + MoE 挂载）
│   ├── neurotwin_moe.py        # NeuroTwinMoE：病理感知 MoDE 专家模块
│   └── common.py               # BrainRevIN / DFCAdapter / BrainMDM / GraphODE 等公共模块
├── train/
│   ├── losses.py               # UncertaintyWeightedHybridLoss 不确定性加权混合损失
│   ├── optim.py                # AdamW 参数组 / Warmup+Cosine 调度 / ModelEMA / 分阶段微调
│   └── moe.py                  # 路由温度调度（Hold-then-Decay）+ 负载均衡/Z-Loss/熵/多样性正则
├── utils/
│   ├── dataloader.py           # NeuroTwinDataset / NeuroTwinDataLoader（滑窗样本、subject-level 划分）
│   ├── augmentation.py         # BrainSignalAugmentation 训练集增强
│   └── common.py               # set_seed / str2bool 通用工具
├── analysis/
│   ├── run_comprehensive.py    # 综合分析 CLI 入口
│   ├── analyzer.py / metrics.py / checkpoint.py / visualizer.py
│   ├── analyze_low_pcc_samples.py    # 低 PCC 样本临床特征关联分析
│   ├── visualize_roi_importance.py   # 结合 AAL116 图谱的 ROI 重要性可视化
│   └── visualize_sig_region.py       # 按脑网络分组的显著脑区玻璃脑图
├── scripts/
│   ├── Pretrain_HC.sh          # 阶段一：HC 预训练脚本
│   └── Finetune_MDD.sh         # 阶段二：MDD 个体化微调脚本
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
├── ROISignals_window/
│   ├── HC/   # ROISignals_{subj_id}-{i}.mat，i = 1..total_windows（滑动窗口 ROI 信号）
│   └── MDD/
├── Mask/
│   ├── HC/   # Mask_{subj_id}.mat（结构连接矩阵 SC）
│   └── MDD/
└── Rest-meta-MDD-HAMD-V1-V2-Merge-Normalize.xlsx   # 临床评分表（微调必需，含 ID 与 HAMD 列）
```

单个样本的张量构成（由 `NeuroTwinDataset` 滑窗生成，最大起点为 `total_windows - in_window - pred_window + 1`）：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `x` | `[F, in_window, S]` | 历史 ROI 信号输入 |
| `y` | `[F, pred_window, S]` | 未来窗口监督信号 |
| `sc` | `[F, F]` | 结构连接矩阵（清洗 NaN/Inf → 对称化 → 截负 → log1p → 99 分位缩放 → 裁剪至 [0,1]） |
| `pathology_score` | `[1]` | HAMD 评分（仅 finetune 模式） |

默认 `F=116`（AAL ROI 数）、`S=30`（每窗序列长度）、`total_windows=9`。

## 快速开始

### 阶段一：HC 预训练

```bash
bash scripts/Pretrain_HC.sh
# 或直接调用：
python main.py --mode pretrain --pred_window 1 \
    --data_root ./data --checkpoint_dir ./checkpoints \
    --name neurotwin_pretrain_pred1
```

- 输出权重：`checkpoints/<name>/base_best.pt`（按验证 `loss_pcc` 早停选优）与 `base_last.pt`
- TensorBoard 日志：`checkpoints/runs/<name>_<时间戳>/`

### 阶段二：MDD 个体化微调（NeuroTwinMoE）

```bash
bash scripts/Finetune_MDD.sh
# 或直接调用：
python main.py --mode finetune --pred_window 1 \
    --pretrained_weight ./checkpoints/neurotwin_pretrain_pred1/base_best.pt \
    --data_root ./data --checkpoint_dir ./checkpoints \
    --name neurotwin_finetune_pred1
```

- 自动加载预训练主干（按 shape 兼容过滤），前 `--freeze_backbone_epochs` 个 epoch 冻结主干仅训练 MoE，随后解冻联合优化
- 输出权重：`finetuned_best.pt` 与 `finetuned_last.pt`

> 注意：`main.py` 中设备选择硬编码为 `cuda:1`（`torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')`），多卡机器请按需调整或通过 `CUDA_VISIBLE_DEVICES` 控制。

### 完整参数说明

运行 `python main.py --help` 查看全部 CLI 参数（数据、模型结构、优化器、损失权重、MoE 超参等约 60 项），脚本中的典型配置见 [scripts/Pretrain_HC.sh](scripts/Pretrain_HC.sh) 与 [scripts/Finetune_MDD.sh](scripts/Finetune_MDD.sh)。

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
| 数据 | HC 组（`ROISignals_window/HC` + `Mask/HC`） | MDD 组 + HAMD 临床评分 |
| 模型形态 | 无 MoE 分支（`pretrain_mode=True`） | 加载预训练主干 + NeuroTwinMoE |
| 参数冻结 | 全参数训练 | 前 N epoch 仅训练 MoE，之后解冻主干（主干学习率 × `backbone_lr_scale`） |
| 优化器 | AdamW（统一学习率） | AdamW（backbone / MoE / 损失权重分组学习率） |
| 调度器 | Warmup + Cosine 退火 | Warmup + Cosine 退火 |
| 路由温度 | — | Hold-then-Decay：冻结期高温探索，解冻后衰减；专家使用 CV>0.5 时自动升温 |
| 输出权重 | `base_best.pt` / `base_last.pt` | `finetuned_best.pt` / `finetuned_last.pt` |

通用稳定性组件：AMP 混合精度、EMA 权重滑动平均、梯度裁剪、按验证 `loss_pcc` 的早停（`patience`）、全局固定随机种子（`set_seed`）。

## 损失函数与 MoE 正则

主损失 `UncertaintyWeightedHybridLoss`（同方差不确定性加权）同时优化四项：

```text
L_pcc  = 1 - Pearson(pred, target)        # 形态相关性
L_mae  = L1(pred, target)                 # 幅值误差
L_diff = L1(diff(pred), diff(target))     # 一阶差分一致性
L_std  = L1(std(pred), std(target))       # 波动幅度一致性
L_total = Σ_i [exp(-log_var_i)·L_i + log_var_i]   # log_var_i 可学习，clamp 至 [-6, 6]
```

MoE 训练正则（`compute_moe_regularization`，仅在 finetune 生效）：

| 正则项 | 权重参数（默认） | 作用 |
| --- | --- | --- |
| 负载均衡（Switch 风格 importance×load） | `--moe_load_balance_weight` (0.01) | 鼓励专家负载均衡 |
| Z-Loss（ST-MoE） | `--moe_z_loss_weight` (1e-3) | 惩罚路由 logit 幅度过大 |
| 熵正则 | `--moe_entropy_weight` (1e-3) | 抑制路由过度集中 |
| 多样性损失 | `--moe_diversity_weight` (0.001) | 惩罚 batch 内路由决策趋同 |

## 评估与可解释性分析

综合分析（回归指标、专家-HAMD 分层、ROI 置换重要性、FDR 显著脑区、玻璃脑图等）：

```bash
python analysis/run_comprehensive.py \
    --finetuned_weight checkpoints/neurotwin_finetune_pred1/finetuned_best.pt \
    --data_root /data3/Digital_Brain/AMD/data \
    --output_dir ./checkpoints/analysis_results
```

配套脚本：

| 脚本 | 用途 |
| --- | --- |
| [analysis/analyze_low_pcc_samples.py](analysis/analyze_low_pcc_samples.py) | 低 PCC 样本与临床特征（HAMD 等）的关联分析 |
| [analysis/visualize_roi_importance.py](analysis/visualize_roi_importance.py) | 结合 AAL116 图谱的 ROI 置换重要性可视化 |
| [analysis/visualize_sig_region.py](analysis/visualize_sig_region.py) | 按脑网络分组渲染显著脑区玻璃脑图（依赖 nilearn） |

回归指标覆盖 MAE / MSE / RMSE / PCC / R² / SMAPE / MASE / 误差分位数；模型内置 `virtual_intervention` 虚拟干预接口（excitatory / inhibitory / variance_boost / variance_suppress），支持机制层面的可干预性分析。

## 关键超参数

以下为脚本（`pred_window=1`）中的典型配置：

| 类别 | 参数 |
| --- | --- |
| 数据 | `num_rois=116`、`seq_len=30`、`total_windows=9`、`in_window=6`、`pred_window=1` |
| 优化 | `train_epochs=150`、`warmup_epochs=15`、`batch_size=32`、`patience=20`、`lr_init=5e-5 / lr_peak=1e-4 / lr_final=5e-5`、`weight_decay=1e-2`、`grad_clip=1.0` |
| 模型 | `n_block=2`、`ode_steps=6`、`ode_hidden_dim=256`、`num_scales=3`、`dropout=0.2`、`alpha=0.5` |
| MoE | `num_experts=4`、`top_k=2`、`moe_expert_hidden_dim=256`、`moe_gate_temp_start=1.5 → end=1.0`、`moe_inference_temperature=0.3`、共享专家开启、路由仅条件化 |
| 微调专属 | `freeze_backbone_epochs=10`、`backbone_lr_scale=0.2`、`loss_lr_scale=0.5` |
| 损失权重初值 | `init_log_var_pcc=0.0`、`init_log_var_mae=-1.5`、`init_log_var_diff=-2.0`、`init_log_var_std=-2.0` |

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/模型架构详细文档.md](docs/模型架构详细文档.md) | 各模块结构与张量流的完整说明 |
| [docs/模型机理分析文档.md](docs/模型机理分析文档.md) | 机理层面的设计动机分析 |
| [docs/模型结构与训练策略总结.md](docs/模型结构与训练策略总结.md) | 结构 + 数据 + 训练策略论文参考底稿 |
| [docs/全面模型分析指南.md](docs/全面模型分析指南.md) | 分析工具链使用指南 |
| [docs/可解释性实验设计文档.md](docs/可解释性实验设计文档.md) | 可解释性实验（置换重要性、虚拟干预等）设计 |

## 变更记录

| 变更日期 | 文件名 | 主要修改 |
| --- | --- | --- |
| 2026-09-21 | README.md | 新建：补全项目简介、目录结构、环境依赖、数据准备、训练/评估流程、架构概览、超参数与文档索引 |
