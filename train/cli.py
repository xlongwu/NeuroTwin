"""训练命令行参数与默认配置。"""

import argparse

from train.training_state import DYNAMICS_CKPT_HORIZON
from utils.common import str2bool
from utils.condition_schema import DROPOUT_SEED


def parse_args():
    """解析训练参数，保留 checkpoint 配置字段及默认值。"""
    parser = argparse.ArgumentParser(
        description="NeuroTwin-TFM: Pathology-Conditioned TimesFM-style Foundation Model for Brain Dynamics"
    )
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument(
        "--device",
        default="",
        help="训练设备（如 cuda:0）。空 = 历史默认（有 GPU 时用 cuda:1）。用于多卡并行跑消融：不同执行器指定不同 --device。",
    )
    parser.add_argument(
        "--pretrained_weight",
        default="",
        help="起点权重路径。finetune 必填；pretrain 一律留空。必须显式指定 HC 预训练权重。",
    )
    parser.add_argument(
        "--save_training_state",
        action="store_true",
        help="额外保存 optimizer/scheduler/scaler/EMA 和随机状态，供中断后恢复。",
    )
    parser.add_argument("--pathology_input_dim", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--name", default="neurotwin")
    parser.add_argument("--clinical_file", default="Rest-meta-MDD-V1V2-Merged-MDD.xlsx")
    parser.add_argument("--num_rois", type=int, default=116)
    parser.add_argument("--norm", type=str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_init", type=float, default=5e-05)
    parser.add_argument("--lr_peak", type=float, default=0.0001)
    parser.add_argument("--lr_final", type=float, default=5e-05)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--loss_lr_scale", type=float, default=1.0)
    parser.add_argument(
        "--pretrained_arch_policy",
        default="require_match",
        choices=["require_match"],
        help="Require the current checkpoint architecture and task dimensions.",
    )
    parser.add_argument(
        "--refresh_split_manifest",
        action="store_true",
        help="重新生成 subject_split_<mode>.json 切分 manifest（默认存在且配置一致时复用，配置不一致时报错）",
    )
    parser.add_argument(
        "--no_eval_after_train",
        dest="eval_after_train",
        action="store_false",
        help="finetune 训练结束后不自动评 test（默认自动评估并落盘到 <save_dir>/eval/）",
    )
    parser.add_argument(
        "--pathology_fields",
        default=None,
        help="逗号分隔的病理字段列表，默认 None 时使用 HAMD 标量。",
    )
    parser.add_argument(
        "--pathology_missing",
        default="drop",
        choices=["drop", "zero", "mean"],
        help="病理字段缺失处理策略。",
    )
    parser.add_argument(
        "--adapter_lr_scale",
        type=float,
        default=1.0,
        help="条件适配器（adapter）参数组的学习率缩放（历史键；未设 --condition_lr_scale 时作为其回退值）。",
    )
    parser.add_argument(
        "--condition_arm",
        default=None,
        choices=["none", "demo", "cat"],
        help="Field condition encoding; HC demo, MDD cat.",
    )
    parser.add_argument(
        "--condition_group",
        default="",
        choices=["", "hc3", "mdd8"],
        help="条件字段分组；留空按 --mode 推导（pretrain→hc3，finetune→mdd8）。",
    )
    parser.add_argument(
        "--condition_file",
        default="",
        help="条件 merged 文件名（默认按 mode 取 Rest-meta-MDD-V1V2-Merged-{HC,MDD}.xlsx）。条件数据只允许来自该文件，禁止回读单独 V1/V2 源表。",
    )
    parser.add_argument(
        "--split_file",
        default="",
        help="外部被试级划分 JSON（train/val/test 名单）；提供并校验通过后优先于 data/subject_split_<mode>.json，所有对照臂共用同一份。",
    )
    parser.add_argument(
        "--stratify_key",
        default="",
        help="划分分层键（MDD 默认 HAMD）。不随条件字段顺序 fields[0] 变化；键缺失者进入预注册的缺失桶确定性分配，不删人。",
    )
    parser.add_argument(
        "--condition_inherit_stats",
        default="",
        help="HC checkpoint / JSON 路径：MDD 实验从中继承 Age/education 的 mu/scale 与类别协议，只在新增五字段上重新拟合。",
    )
    parser.add_argument(
        "--condition_extra_codes",
        default="",
        help="显式预注册的有效类别码（Sex:0,3;medication:2 形式），用于merged 中确实存在但未落在 train 的词表；运行时禁止静默扩表。",
    )
    parser.add_argument(
        "--cond_d_demo",
        type=int,
        default=16,
        help="共享三字段（HC）独立编码宽度 d_demo。",
    )
    parser.add_argument(
        "--cond_d_new", type=int, default=16, help="MDD 新增五字段独立编码宽度 d_new。"
    )
    parser.add_argument(
        "--cond_dim", type=int, default=32, help="融合后条件向量宽度 C。"
    )
    parser.add_argument(
        "--cond_h_delta", type=int, default=64, help="增量 DeltaE 的隐藏宽度 H_delta。"
    )
    parser.add_argument(
        "--cond_delta_style",
        default="mlp",
        choices=["mlp"],
        help="MLP increment for concatenated MDD fields.",
    )
    parser.add_argument(
        "--cond_dropout_p",
        type=float,
        default=0.0,
        help="Condition Dropout 的遮蔽概率 parser（受试者级、每 epoch 固定）。",
    )
    parser.add_argument(
        "--cond_dropout_seed",
        type=int,
        default=DROPOUT_SEED,
        help="Condition Dropout 种子（与 --seed 模型种子、划分 seed 分离）。",
    )
    parser.add_argument(
        "--condition_lr_scale",
        type=float,
        default=None,
        help="条件参数组学习率相对 baseline 的倍率（未设则回退 --adapter_lr_scale）。",
    )
    parser.add_argument(
        "--sc_prior_mode",
        default="soft_prior",
        choices=["scaled", "soft_prior", "adaptive_only", "functional_only"],
        help="SC 注入形态：scaled=旧的静态 SC 度归一化（复现 Phase 1）；soft_prior=λ·A_SC+(1-λ)·A_func（默认）；adaptive_only=纯数据驱动功能图（同归一化口径）；functional_only=功能图原始 row_softmax（不做 Sinkhorn）。",
    )
    parser.add_argument(
        "--sc_lambda_mode",
        default="global",
        choices=["global", "sample", "roi"],
        help="λ 混合系数的粒度：global 全局标量 / sample 样本级 / roi 逐 ROI。",
    )
    parser.add_argument(
        "--sc_lambda_init",
        type=float,
        default=0.7,
        help="λ 初值（越大越信任 SC 结构先验）。",
    )
    parser.add_argument(
        "--sc_prior_rank", type=int, default=12, help="功能图 A_func 的低秩分解维度。"
    )
    parser.add_argument(
        "--sc_delta_a",
        type=str2bool,
        default=True,
        help="是否启用低秩 subject-specific 邻接残差 ΔA（零初始化）。",
    )
    parser.add_argument(
        "--sc_delta_rank",
        type=int,
        default=None,
        help="ΔA 的秩，默认 max(4, sc_prior_rank//2)。",
    )
    parser.add_argument(
        "--sc_sinkhorn_iters",
        type=int,
        default=64,
        help="对称 Sinkhorn 迭代次数（使 A_eff 同时行归一且对称；靠近均匀的图收敛慢）。",
    )
    parser.add_argument(
        "--sc_sparsity_weight",
        type=float,
        default=0.001,
        help="SC 软先验稀疏正则权重（A_eff 非对角 L1），<=0 关闭。",
    )
    parser.add_argument(
        "--sc_entropy_weight",
        type=float,
        default=0.001,
        help="SC 软先验行熵正则权重（抑制模糊均匀图），<=0 关闭。",
    )
    parser.add_argument(
        "--sc_temporal_weight",
        type=float,
        default=0.0,
        help="逐窗功能图时间一致性正则权重（>0 时才生成 A_seq），默认关闭。",
    )
    parser.add_argument("--freeze_backbone_epochs", type=int, default=10)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--stratify_bins", type=int, default=5)
    parser.add_argument("--cache_in_memory", type=str2bool, default=False)
    parser.add_argument("--use_ema", type=str2bool, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--persistent_workers", type=str2bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="验证/测试 DataLoader 的 batch 大小，默认为 batch_size×4",
    )
    parser.add_argument(
        "--tf32",
        type=str2bool,
        default=True,
        help="启用 cudnn.benchmark 与 TF32（对 autocast 未覆盖的 fp32 计算加速）",
    )
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=True,
        help="torch.compile 编译模型前向（稳态约 35%% 提速；首次编译有额外耗时）",
    )
    parser.add_argument(
        "--tfm_dim",
        type=int,
        default=256,
        help="TFM 隐藏维度 D。",
    )
    parser.add_argument(
        "--tfm_layers",
        type=int,
        default=4,
        help="TFM Encoder Block 层数 N。",
    )
    parser.add_argument(
        "--tfm_heads", type=int, default=4, help="TFM temporal/ROI attention 的头数。"
    )
    parser.add_argument(
        "--tfm_patch_len",
        type=int,
        default=8,
        help="TFM temporal patch 长度 parser（G30_TFM 消融：parser=8 最优）。K 不是 parser 的整数倍时开头补零对齐。",
    )
    parser.add_argument(
        "--tfm_ff_ratio",
        type=int,
        default=4,
        help="TFM FFN 隐层扩张倍数（ff_dim = ff_ratio × dim）。",
    )
    parser.add_argument(
        "--tfm_one_step_hidden_dim",
        type=int,
        default=256,
        help="TFM One-Step 状态转移头的 MLP 隐层宽度。",
    )
    parser.add_argument(
        "--cpm_horizon",
        type=int,
        default=8,
        help="CPM 全 horizon 预测步数 H。",
    )
    parser.add_argument(
        "--cpm_layers",
        type=int,
        default=2,
        help="CPM ROI×Horizon 查询解码器的交叉注意力层数。",
    )
    parser.add_argument(
        "--lambda_one",
        type=float,
        default=1.0,
        help="TFM 损失：one-step Huber 项权重。",
    )
    parser.add_argument(
        "--lambda_cpm",
        type=float,
        default=1.0,
        help="TFM 损失：CPM 轨迹项权重（0 可退化为纯 one-step 的 Stage 0 基线，方案 §24）。",
    )
    parser.add_argument(
        "--lambda_pcc",
        type=float,
        default=0.1,
        help="损失：spatial PCC 惩罚项权重（1 - 逐 ROI 空间相关）。",
    )
    parser.add_argument(
        "--cpm_gamma",
        type=float,
        default=1.0,
        help="CPM 逐 horizon 损失权重衰减 γ（w_h=γ^(h-1)；1=均匀，方案 §20.2）。须在 (0, 1] 内。",
    )
    parser.add_argument(
        "--cpm_huber_delta",
        type=float,
        default=1.0,
        help="TFM Huber 损失的过渡阈值 δ（原始空间 BOLD 量纲，z-score 后单位方差取 1.0）。",
    )
    parser.add_argument(
        "--loss_weighting",
        default="uncertainty",
        choices=["uncertainty", "fixed"],
        help="任务损失聚合方式：uncertainty=同方差不确定性自动加权（Σ exp(-log_s_i)·L_i + log_s_i，log_s_i 为可学习参数，初始有效权重等于对应 --lambda_*，Kendall et al. CVPR 2018）；fixed=固定权重（历史行为）。仅作用于 TFMDualLoss 的 one/CPM/dynamics/variance 四个任务项；图正则仍由自身权重控制。",
    )
    parser.add_argument(
        "--task_mode",
        default="next_timepoint",
        choices=["next_timepoint"],
        help="任务口径：next_timepoint=连续 BOLD context → 下一 TR 全脑状态（Next Brain-State Prediction，context_min..context_max 个 TR → 未来 TR，默认仅 +1）。",
    )
    parser.add_argument(
        "--random_context",
        type=str2bool,
        default=True,
        help="训练是否随机采样 context 长度 K（False=固定为可用最大 K）。",
    )
    parser.add_argument(
        "--random_cutoff",
        type=str2bool,
        default=True,
        help="训练是否随机采样 cutoff t（False=固定最晚 cutoff）。",
    )
    parser.add_argument(
        "--bold_source",
        default="raw",
        choices=["raw"],
        help="Raw BOLD [T,F], calibrated on the baseline prefix.",
    )
    parser.add_argument(
        "--train_samples_per_subject",
        type=int,
        default=0,
        help="训练每个被试每 epoch 的样本预算；0=auto（min(合法组合数, 12)）。",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=2024,
        help="采样/分桶随机种子（训练采样可复现；评估协议本身与随机性无关）。",
    )
    parser.add_argument(
        "--subject_cache_size",
        type=int,
        default=256,
        help="数据集每个 worker 的被试级缓存上限（0=不缓存）。",
    )
    parser.add_argument(
        "--eval_fc", type=str2bool, default=True, help="评估是否输出 FC 层面指标。"
    )
    parser.add_argument(
        "--eval_spectral",
        type=str2bool,
        default=False,
        help="评估是否输出 Welch PSD 低频段一致性指标（可选）。",
    )
    parser.add_argument(
        "--eval_spectral_tr",
        type=float,
        default=2.0,
        help="频谱评估使用的 TR（秒），默认 2.0 s（REST-meta-MDD 常用值）。",
    )
    parser.add_argument(
        "--context_min",
        type=int,
        default=16,
        help="next_timepoint 训练 context 长度下界 K_min（TR 数）。",
    )
    parser.add_argument(
        "--context_max",
        type=int,
        default=64,
        help="next_timepoint context 长度上界 K_max（= 模型 S 轴建模宽度，决定参数量）。训练随机采样 K ∈ context_lengths 或 [context_min, context_max]。",
    )
    parser.add_argument(
        "--context_lengths",
        default="",
        help="可选：离散 context 长度集合（逗号分隔，如 16,32,64）；空=在 [context_min, context_max] 区间内均匀随机采样。",
    )
    parser.add_argument(
        "--causal_training",
        default="random_context",
        choices=["random_context", "full_sequence"],
        help="训练范式：random_context=随机 context → 下一时间点（方案 B，已实现，未来信息不进入前向）；full_sequence=GPT 式全序列 teacher-forcing（方案 A，需要重写主干时间轴算子，当前显式报错未实现）。",
    )
    parser.add_argument(
        "--forecast_offsets",
        default="",
        help="预测偏移列表（逗号分隔，相对当前 TR 的 +Δ，需严格递增）；TFM 的 one-step 头只建模 +1 偏移，留空即 [1]，多步预测用 --cpm_horizon 控制 CPM 头。",
    )
    parser.add_argument(
        "--eval_context_length",
        type=int,
        default=0,
        help="评估固定 context 长度 K_eval；0=自动取 --context_max。评估协议对每个被试使用同一 K_eval 与确定性 anchor，禁止随机位置。",
    )
    parser.add_argument(
        "--eval_anchors_per_subject",
        type=int,
        default=16,
        help="每个被试评估的 next-state 预测点数量（等间隔确定性选取；0=枚举全部）。",
    )
    parser.add_argument(
        "--eval_rollout_tasks_per_subject",
        type=int,
        default=2,
        help="每个被试参与 free rollout / FC / 频谱评估的 anchor 数（0=不做 rollout）。",
    )
    parser.add_argument(
        "--eval_rollout_horizons",
        default="1,2,4,8,16",
        help="free rollout 的评估 horizon 列表（逗号分隔）。",
    )
    parser.add_argument(
        "--eval_rollout_steps",
        type=int,
        default=0,
        help="单次 rollout 的滚动步数 R；0=自动取 max(horizons) 并在启用 FC/频谱时抬到 --fc_min_length（FC 需要更长的轨迹）。",
    )
    parser.add_argument(
        "--fc_min_length",
        type=int,
        default=32,
        help="FC/频谱评估所需的最小 rollout 长度（prompt §二十二；短于该长度不计算 FC）。",
    )
    parser.add_argument(
        "--dynamics_mode",
        default="graph_ode",
        choices=["graph_ode"],
        help="Continuous Graph-ODE dynamics.",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=2,
        help="graph_ode 每个区间的固定 Heun RK2 步数（>=1；默认 2）。",
    )
    parser.add_argument(
        "--dynamics_dt",
        type=float,
        default=1.0,
        help="graph_ode 每个区间的积分步长 dt（>0，单位=TR；默认 1.0）。",
    )
    parser.add_argument(
        "--graph_source",
        default="history_fc",
        choices=["history_fc"],
        help="Coupling graph from visible history.",
    )
    parser.add_argument(
        "--personalized",
        type=str2bool,
        default=True,
        help="是否启用固定基线个体 profile（graph_ode 专用；需 --bold_source raw 并提供 calibration_bold）。False 时不创建 profile 模块。",
    )
    parser.add_argument(
        "--profile_dim",
        type=int,
        default=32,
        help="个体 profile 维度（personalized=True 时生效；默认 32）。",
    )
    parser.add_argument(
        "--calibration_length",
        type=int,
        default=32,
        help="基线校准前缀长度（TR 数；bold_source=raw 时必须 >=1，同时决定合法锚点下界与 calibration_bold 输出）。",
    )
    parser.add_argument(
        "--control_manifest",
        default="",
        help="控制序列 manifest CSV（表头 subject_id,controls_path；相对路径相对 manifest 所在目录；NPZ 键 u，[T,F] 逐区间控制）。空=无控制（controls=None，不伪造刺激标签）。",
    )
    parser.add_argument(
        "--lambda_dynamics",
        type=float,
        default=0.5,
        help="graph_ode 动态多步损失权重（h=2..H masked Huber+PCC；默认 0.5）。",
    )
    parser.add_argument(
        "--lambda_variance",
        type=float,
        default=0.05,
        help="方差读出高斯 NLL 损失权重（只更新 VarianceHead；默认 0.05）。",
    )
    parser.add_argument(
        "--dynamics_metric_horizon",
        type=int,
        default=DYNAMICS_CKPT_HORIZON,
        help="graph_ode 选优指标覆盖的预测步数 N（H1..N，默认 8）；指标键名随 N 变化（metric_mae_dynamics_h1_N_subject），且只纳入具备完整 N 步真值的锚点。必须 <= --cpm_horizon。",
    )
    args = parser.parse_args()
    if args.condition_arm is None:
        args.condition_arm = "demo" if args.mode == "pretrain" else "cat"
    if args.mode == "finetune" and (not args.pretrained_weight):
        parser.error("finetune requires --pretrained_weight")
    return args
