#!/usr/bin/env bash
# ============================================================================
# NeuroTwin-TFM 微调（MDD 队列，graph_ode 动态路线 + 临床条件注入）
#
# 用法（起点必须是 Pretrain_HC_tfm.sh 刚产出的 hc_demo_twin 权重）
#   PYTHON="$HOME/anaconda3/envs/mamba/bin/python" bash scripts/Finetune_MDD_tfm.sh
#   # 前置：先跑 scripts/Pretrain_HC_tfm.sh（两者用同一 CHECKPOINT_DIR）
#   # 可选环境变量：
#   #   CHECKPOINT_DIR=<目录>   默认 $ROOT/checkpoints（HC 起点与本轮产物都在这个目录下）
#   #   HC_WEIGHT=<权重路径>    覆盖自动查找，显式指定起点
#   #   DEVICE=cuda:1           不给则沿用 main.py 的设备默认
#   #   DATA_ROOT=<数据根目录>   默认 $ROOT/data
#   #   PRINT_CMD=1             打印完整命令行（默认只打印关键参数摘要）
#
# 条件注入（2026-10-10 选定方案，见 TASK_HANDOFF §0.26）：
#   ``--condition_group mdd8 --condition_arm cat`` —— 八个字段各自独立编码，
#   新增五字段按 ``d_new`` 拼接成 Φ 后过**零初始化** DeltaE，因此训练起点的条件向量
#   与 HC 侧的 c_demo 逐位相同；条件经双头独立 FiLM 注入 One-Step 与 CPM 两头。
#
# graph_ode 动态路线（方案 docs/Version4_docs_1012）：twin 家族 mdd_cat_twin，
#   与预训练同口径（raw + history_fc + personalized + graph_ode），动态 H 复用
#   cpm_horizon=8；起点权重默认只在 hc_demo_twin 家族内匹配，不混选旧
#   neurotwin_tfm_pretrain 的 discrete 产物。
#
# 统计量继承（方案 §四）：``--condition_inherit_stats`` 指向**同一个起点权重文件**，
#   共享三字段的 median/IQR 与类别词表从里面逐位读出、不重新拟合。
#   ⚠️ 2026-10-10 起不再有 study/fresh 身份闸门（`experiments/condition_study.py` 已删除，
#   §0.27）：脚本只在**本目录里**找 HC 起点并在有多个候选时拒绝替你猜，但不会核对权重内嵌
#   来源、也不校验 SHA256 —— 「起点确实是本轮 HC」这件事现在靠串接顺序，不靠代码保证。
#
# 两阶段逻辑：冻结 TFM 主干（仅训练条件模块）→ 解冻联合优化
#   （backbone 学习率 × backbone_lr_scale；profile_encoder 随共享 backbone 解冻，
#   VarianceHead 随动力学头解冻）。
#
# ⚠️ 必须与预训练使用同一 graph_ode/raw/history_fc/personalized 口径
#   （ARCH_VERSION=6 结构校验会拦截不匹配的组合；旧 discrete 权重不迁移）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# DataLoader 共享内存：固定到项目内目录，避免 Qoder 会话切换清理 TMPDIR 后
# multiprocessing resource_sharer socket 消失（长训练中断的根因）。
export TMPDIR="$ROOT/.runtime_tmp"
mkdir -p "$TMPDIR"
export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"

PYTHON_BIN="${PYTHON:-python}"
data_root="${DATA_ROOT:-$ROOT/data}"
checkpoint_dir="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
DEVICE="${DEVICE:-}"
HC_WEIGHT="${HC_WEIGHT:-}"
# 概率化实验一般从 S2 微调权重继续，而不是同目录的 HC base_best.pt。该覆盖必须发生在
# 下面的 HC 起点候选检查之前，否则会被"找不到 HC 起点"提前拦停。
if [[ "${S4_VARIANT:-off}" != "off" && -n "${S4_INIT_WEIGHT:-}" ]]; then
  HC_WEIGHT="$S4_INIT_WEIGHT"
fi
name="neurotwin_tfm_finetune"
hc_name="neurotwin_tfm_pretrain"   # 与 Pretrain_HC_tfm.sh 的 name 同步（twin 家族）

# 起点权重：hc_demo_twin 试次目录下的 base_best.pt。找到 0 个→还没训；
# 找到多个→必须显式指定，绝不静默挑一个（一个融合家族只能共享一个起点）。
# 只匹配 twin 家族：旧 neurotwin_tfm_pretrain 的 discrete 产物不会被混选。
if [[ -z "$HC_WEIGHT" ]]; then
  mapfile -t hc_cands < <(compgen -G "$checkpoint_dir/$hc_name*/base_best.pt" || true)
  [[ ${#hc_cands[@]} -gt 0 ]] || {
    echo "[错误] 找不到 HC 起点（$checkpoint_dir/$hc_name*/base_best.pt 不存在）；" \
         "先用同一 CHECKPOINT_DIR 跑 scripts/Pretrain_HC_tfm.sh，" \
         "或用 HC_WEIGHT=<完整路径> 显式指定" >&2
    exit 1
  }
  [[ ${#hc_cands[@]} -eq 1 ]] || {
    echo "[错误] 找到 ${#hc_cands[@]} 个候选 HC 起点，无法判断该继承哪一个：${hc_cands[*]}"
    echo "        请用 HC_WEIGHT=<完整路径> 显式指定（不替你猜）" >&2
    exit 1
  }
  HC_WEIGHT="${hc_cands[0]}"
fi
pretrained_weight="$HC_WEIGHT"

# ---- 实验规模（与预训练保持一致的任务口径） ----
# 随机来源只在 utils/condition_schema.py 登记一次（与 Pretrain 脚本同源，不抄字面量）
read -r seed sampling_seed cond_dropout_seed < <(
  PYTHONPATH=. "$PYTHON_BIN" -c 'from utils.condition_schema import (DROPOUT_SEED, MODEL_SEED, SAMPLING_SEED); print(MODEL_SEED, SAMPLING_SEED, DROPOUT_SEED)') \
  || { echo "[错误] $PYTHON_BIN 读取 utils.condition_schema 的随机来源常量失败" >&2; exit 1; }
if [[ "$seed" == "$sampling_seed" || "$seed" == "$cond_dropout_seed" \
      || "$sampling_seed" == "$cond_dropout_seed" ]]; then
  echo "[错误] 随机来源未分离（seed=$seed sampling=$sampling_seed dropout=$cond_dropout_seed）" >&2
  exit 1
fi
num_rois=116
context_min=16
context_max=64
eval_context_length=64
eval_anchors_per_subject=16
eval_rollout_tasks_per_subject=2
eval_rollout_horizons="1,2,4,8,16"
tfm_dim=256
tfm_layers=4
tfm_heads=4
tfm_patch_len=8
cpm_horizon=8

# ---- P0 损失扩展（默认=E2 正式基线；必须与预训练同口径）----
# 微调启用时间差分/一致性时，HC 预训练也应使用相同开关，
# 否则微调起点缺少对应监督。消融覆盖方式同 Pretrain_HC_tfm.sh：
#   E0: LAMBDA_TEMPORAL_DIFF=0 LAMBDA_CROSS_HEAD=0 bash scripts/Finetune_MDD_tfm.sh
#   E1: LAMBDA_TEMPORAL_DIFF=0.05 bash scripts/Finetune_MDD_tfm.sh
lambda_temporal_diff="${LAMBDA_TEMPORAL_DIFF:-0.05}"
lambda_cross_head="${LAMBDA_CROSS_HEAD:-0.02}"
cross_head_start_epoch="${CROSS_HEAD_START_EPOCH:-5}"
cpm_horizon_weight="${CPM_HORIZON_WEIGHT:-gamma}"
cpm_horizon_alpha="${CPM_HORIZON_ALPHA:-0.5}"
# ---- S2 历史记忆（正式基线默认开启；必须与预训练同口径）----
dynamics_history_memory="${DYNAMICS_HISTORY_MEMORY:-True}"

# ---- S4 概率轨迹（默认全部不追加，脚本行为与 S2 逐位一致）----
#   S4-1 输出端相关残差：S4_VARIANT=s4_1 HC_WEIGHT=<S2 finetuned_best.pt> \
#          DYNAMICS_HISTORY_MEMORY=True bash scripts/Finetune_MDD_tfm.sh
#   S4-2 潜状态随机创新：S4_VARIANT=s4_2 ...
# head 阶段要求 freeze_backbone_epochs>=1（main.py 会校验），且必须与 S2 同口径
# 打开 DYNAMICS_HISTORY_MEMORY，否则重建的架构与起点权重不一致。
s4_variant="${S4_VARIANT:-off}"
s4_stage="${S4_STAGE:-head}"
s4_crps_weight="${S4_CRPS_WEIGHT:-1.0}"
s4_lr_scale="${S4_LR_SCALE:-1.0}"
s4_train_samples="${S4_TRAIN_SAMPLES:-32}"
s4_residual_rank="${S4_RESIDUAL_RANK:-8}"
s4_residual_hidden="${S4_RESIDUAL_HIDDEN:-128}"
s4_residual_init_scale="${S4_RESIDUAL_INIT_SCALE:-0.1}"
s4_noise_rank="${S4_NOISE_RANK:-8}"
s4_noise_scale="${S4_NOISE_SCALE:-1e-2}"
s4_ckpt_metric="${S4_CKPT_METRIC:-point}"

train_epochs="${TRAIN_EPOCHS:-150}"
warmup_epochs=15
# 数据加载并发（只影响吞吐，不影响训练结果）
num_workers="${NUM_WORKERS:-6}"
pin_memory="${PIN_MEMORY:-False}"
batch_size=256
patience=20
lr_init=5e-5
lr_peak=1e-4
lr_final=5e-5
freeze_backbone_epochs="${FREEZE_BACKBONE_EPOCHS:-10}"
backbone_lr_scale=0.2

# ---- 预检查 ----
# graph_ode 路线用 raw + history_fc：只要求 Raw 目录与临床表，不要求 Mask，
# 也不回退 Normlize/窗口缓存（方案 §4.1）。
[ -d "$data_root/Raw/MDD" ] || {
  echo "[错误] 缺少 MDD Raw 目录（$data_root/Raw/MDD/ROISignals_<id>.mat，[T,F] 布局）"; exit 1; }
[ -f "$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx" ] || {
  echo "[错误] 缺少临床评分表：$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx"; exit 1; }
[ -f "$pretrained_weight" ] || {
  echo "[错误] 起点权重文件不存在：$pretrained_weight"
  echo "        它应当是 scripts/Pretrain_HC_tfm.sh 在同一个 CHECKPOINT_DIR 下产出的" \
       "base_best.pt；请先跑预训练，或用 HC_WEIGHT=<完整路径> 指定真实存在的路径" >&2; exit 1; }

ARGS=(
  # ---- 基础与路径 ----
  --mode finetune
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --pretrained_weight "$pretrained_weight"
  --num_rois "$num_rois"
  # ---- Next-Timepoint 任务口径（与预训练严格一致）----
  --task_mode next_timepoint
  --context_min "$context_min"
  --context_max "$context_max"
  --forecast_offsets 1
  --eval_context_length "$eval_context_length"
  --eval_anchors_per_subject "$eval_anchors_per_subject"
  --eval_rollout_tasks_per_subject "$eval_rollout_tasks_per_subject"
  --eval_rollout_horizons "$eval_rollout_horizons"
  --fc_min_length 32
  --eval_fc True
  --eval_spectral False
  # ---- TFM 模型规模（与预训练严格一致）----
  --tfm_dim "$tfm_dim"
  --tfm_layers "$tfm_layers"
  --tfm_heads "$tfm_heads"
  --tfm_patch_len "$tfm_patch_len"
  --tfm_ff_ratio 4
  --tfm_one_step_hidden_dim 256
  --cpm_horizon "$cpm_horizon"
  --cpm_layers 2
  --cpm_intervention False
  # ---- SC 软先验（与预训练一致）----
  --sc_prior_mode soft_prior
  --sc_lambda_mode global
  --sc_lambda_init 0.7
  --sc_prior_rank 12
  --sc_delta_a True
  --sc_sinkhorn_iters 64
  # ---- TFM 损失权重 ----
  --lambda_one 1.0
  --lambda_cpm 1.0
  --lambda_pcc 0.1
  --cpm_gamma 1.0
  --cpm_huber_delta 1.0
  # ---- P0 损失扩展（默认关；见文件顶部环境变量说明）----
  --lambda_temporal_diff "$lambda_temporal_diff"
  --lambda_cross_head "$lambda_cross_head"
  --cross_head_start_epoch "$cross_head_start_epoch"
  --cpm_horizon_weight "$cpm_horizon_weight"
  --cpm_horizon_alpha "$cpm_horizon_alpha"
  --norm True
  --dropout 0.1
  # ---- 两阶段冻结（前期只训条件模块，后期解冻主干）----
  --freeze_backbone_epochs "$freeze_backbone_epochs"
  --backbone_lr_scale "$backbone_lr_scale"
  # ---- 优化与训练规模 ----
  --train_epochs "$train_epochs"
  --warmup_epochs "$warmup_epochs"
  --batch_size "$batch_size"
  --patience "$patience"
  --lr_init "$lr_init"
  --lr_peak "$lr_peak"
  --lr_final "$lr_final"
  --weight_decay 0.01
  --grad_clip 1.0
  --use_ema True
  --amp True
  --num_workers "$num_workers"       # 见文件顶部 PIN_MEMORY/NUM_WORKERS 说明
  --pin_memory "$pin_memory"
  --eval_batch_size "${EVAL_BATCH_SIZE:-512}"  # 仅影响验证吞吐；降低 Val 峰值内存
  --subject_cache_size 4096         # 覆盖全部被试，消除每 epoch 对全宽 Raw mat 的重复 loadmat
  --train_samples_per_subject 0
  --sampling_seed "$sampling_seed"
)

# ---- 本轮条件注入（选定方案：mdd8 + cat）----
ARGS+=(
  --condition_group mdd8 --condition_arm cat   # 选定方案：逐字段独立编码 + cat 融合
  --cond_dropout_seed "$cond_dropout_seed"
  # 统计量快照直接取自同一个 HC 起点权重文件：共享三字段的 median/IQR/类别词表逐位继承，
  # 不重新拟合（方案 §四）。
  --condition_inherit_stats "$pretrained_weight"
)
# ---- graph_ode 动态路线（与预训练严格同口径；twin 家族固定参数）----
ARGS+=(
  --dynamics_mode graph_ode
  --graph_source history_fc        # 功能耦合图来自当前可见历史，不要求 Mask
  --bold_source raw                # Raw/MDD/ROISignals_<id>.mat，前 32 TR 校准
  --personalized True
  --profile_dim 32
  --calibration_length 32
  --ode_steps 2
  --dynamics_dt 1.0
  --lambda_dynamics 0.5
  --lambda_variance 0.05
  --dynamics_history_memory "$dynamics_history_memory"
)
# ---- S4 概率轨迹参数（仅 S4_VARIANT!=off 时追加）----
if [[ "$s4_variant" != "off" ]]; then
  ARGS+=(
    --s4_variant "$s4_variant"
    --s4_stage "$s4_stage"
    --s4_crps_weight "$s4_crps_weight"
    --s4_lr_scale "$s4_lr_scale"
    --s4_train_samples "$s4_train_samples"
    --s4_residual_rank "$s4_residual_rank"
    --s4_residual_hidden "$s4_residual_hidden"
    --s4_residual_init_scale "$s4_residual_init_scale"
    --s4_noise_rank "$s4_noise_rank"
    --s4_noise_scale "$s4_noise_scale"
    --s4_ckpt_metric "$s4_ckpt_metric"
  )
fi
if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi

# 生效参数已随 checkpoint 的 config.args 快照保存，完整命令行只在 PRINT_CMD=1 时打印。
arg_value() {   # 从 ARGS 取开关取值，避免把同一个字面量抄进第二处
  local key="--$1" i
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    [[ ${ARGS[i]} == "$key" ]] && { echo "${ARGS[i + 1]}"; return 0; }
  done
}

if [[ "${PRINT_CMD:-0}" == "1" ]]; then
  echo "[CMD] $PYTHON_BIN -u main.py ${ARGS[*]}"
else
  echo "[CMD] main.py --mode $(arg_value mode) --name $(arg_value name)" \
       "| condition=$(arg_value condition_group)/$(arg_value condition_arm)" \
       "| epochs=$(arg_value train_epochs) bs=$(arg_value batch_size)" \
       "| ckpt=$(arg_value checkpoint_dir)" \
       "| ${#ARGS[@]} 个参数（PRINT_CMD=1 看完整命令行）"
fi
PYTHONPATH=. "$PYTHON_BIN" -u main.py "${ARGS[@]}"
