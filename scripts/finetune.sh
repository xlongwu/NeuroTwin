#!/usr/bin/env bash
# NeuroTwin MDD 微调：继承 HC 权重与共享字段统计量，加入临床条件。
# 配方来自 configs/neurotwin.json；PYTHON、DEVICE、DATA_ROOT、CHECKPOINT_DIR 可覆盖。
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
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-}"
SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-False}"
name="neurotwin_finetune"
if [[ -z "$HC_WEIGHT" && -z "$RESUME_TRAINING_STATE" ]]; then
  HC_WEIGHT="$checkpoint_dir/neurotwin_pretrain/base_best.pt"
fi
pretrained_weight="$HC_WEIGHT"

# ---- 模型配方（configs/neurotwin.json）----
# 模型/损失/优化/条件共享值由 helper 导出；随机种子在 helper 内与
# utils/condition_schema.py 的登记值逐项核对；阶段特有的 MDD 条件臂也在 JSON。
source "$ROOT/scripts/_model_defaults.sh"
if [[ "$seed" == "$sampling_seed" || "$seed" == "$cond_dropout_seed" \
      || "$sampling_seed" == "$cond_dropout_seed" ]]; then
  echo "[错误] 随机来源未分离（seed=$seed sampling=$sampling_seed dropout=$cond_dropout_seed）" >&2
  exit 1
fi

# ---- 评估协议（不属于训练配方，保持脚本内单点定义）----
eval_context_length=64
eval_anchors_per_subject=16
eval_rollout_tasks_per_subject="${EVAL_ROLLOUT_TASKS_PER_SUBJECT:-2}"
eval_rollout_horizons="1,2,4,8,16"

# ---- 环境变量覆盖（消融/排程用）----
# 微调启用时间差分/一致性时，HC 预训练也应使用相同开关，
# 否则微调起点缺少对应监督。消融覆盖方式同 pretrain.sh：
lambda_temporal_diff="${LAMBDA_TEMPORAL_DIFF:-$lambda_temporal_diff}"
lambda_cross_head="${LAMBDA_CROSS_HEAD:-$lambda_cross_head}"
cross_head_start_epoch="${CROSS_HEAD_START_EPOCH:-$cross_head_start_epoch}"
cpm_horizon_weight="${CPM_HORIZON_WEIGHT:-$cpm_horizon_weight}"
cpm_horizon_alpha="${CPM_HORIZON_ALPHA:-$cpm_horizon_alpha}"
train_supervision_horizon="${TRAIN_SUPERVISION_HORIZON:-0}"
dynamics_history_memory="${DYNAMICS_HISTORY_MEMORY:-$dynamics_history_memory}"

train_epochs="${TRAIN_EPOCHS:-$train_epochs}"
patience="${PATIENCE:-$patience}"
# 数据加载并发（只影响吞吐，不影响训练结果）
num_workers="${NUM_WORKERS:-6}"
pin_memory="${PIN_MEMORY:-False}"
freeze_backbone_epochs="${FREEZE_BACKBONE_EPOCHS:-$freeze_backbone_epochs}"
cpm_horizon="${CPM_HORIZON:-$cpm_horizon}"
batch_size="${BATCH_SIZE:-$batch_size}"

# ---- 预检查 ----
# graph_ode 路线用 raw + history_fc：只要求 Raw 目录与临床表，不要求 Mask，
[ -d "$data_root/Raw/MDD" ] || {
  echo "[错误] 缺少 MDD Raw 目录（$data_root/Raw/MDD/ROISignals_<id>.mat，[T,F] 布局）"; exit 1; }
[ -f "$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx" ] || {
  echo "[错误] 缺少临床评分表：$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx"; exit 1; }
if [[ -z "$RESUME_TRAINING_STATE" ]]; then
  [ -f "$pretrained_weight" ] || {
    echo "[错误] 起点权重文件不存在：$pretrained_weight"
    echo "        它应当是 scripts/pretrain.sh 在同一个 CHECKPOINT_DIR 下产出的" \
         "base_best.pt；请先跑预训练，或用 HC_WEIGHT=<完整路径> 指定真实存在的路径" >&2; exit 1; }
fi

ARGS=(
  # ---- 基础与路径 ----
  --mode finetune
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --num_rois "$num_rois"
  # ---- Next-Timepoint 任务口径（与预训练严格一致）----
  --task_mode next_timepoint
  --context_min "$context_min"
  --context_max "$context_max"
  --forecast_offsets "$forecast_offsets"
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
  --tfm_ff_ratio "$tfm_ff_ratio"
  --tfm_one_step_hidden_dim "$tfm_one_step_hidden_dim"
  --tfm_patho_mode "$tfm_patho_mode"
  --cpm_horizon "$cpm_horizon"
  --dynamics_metric_horizon "${DYNAMICS_METRIC_HORIZON:-8}"
  --train_anchor_max_offset "${TRAIN_ANCHOR_MAX_OFFSET:-0}"
  --eval_anchor_min_future_steps "${EVAL_ANCHOR_MIN_FUTURE_STEPS:-0}"
  --pretrained_horizon_policy "${PRETRAINED_HORIZON_POLICY:-require_match}"
  --split_file "${SPLIT_FILE:-}"
  --cpm_layers "$cpm_layers"
  --cpm_intervention "$cpm_intervention"
  # ---- SC 软先验（与预训练一致）----
  --sc_prior_mode "$sc_prior_mode"
  --sc_lambda_mode "$sc_lambda_mode"
  --sc_lambda_init "$sc_lambda_init"
  --sc_prior_rank "$sc_prior_rank"
  --sc_delta_a "$sc_delta_a"
  --sc_sinkhorn_iters "$sc_sinkhorn_iters"
  # ---- TFM 损失权重 ----
  --lambda_one "$lambda_one"
  --lambda_cpm "$lambda_cpm"
  --lambda_pcc "$lambda_pcc"
  --loss_weighting "$loss_weighting"
  --cpm_gamma "$cpm_gamma"
  --cpm_huber_delta "$cpm_huber_delta"
  --train_supervision_horizon "$train_supervision_horizon"
  --lambda_temporal_diff "$lambda_temporal_diff"
  --lambda_cross_head "$lambda_cross_head"
  --cross_head_start_epoch "$cross_head_start_epoch"
  --cpm_horizon_weight "$cpm_horizon_weight"
  --cpm_horizon_alpha "$cpm_horizon_alpha"
  --norm "$norm"
  --dropout "$dropout"
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
  --weight_decay "$weight_decay"
  --grad_clip "$grad_clip"
  --use_ema "$use_ema"
  --ema_decay "$ema_decay"
  --amp "$amp"
  --num_workers "$num_workers"       # 见文件顶部 PIN_MEMORY/NUM_WORKERS 说明
  --pin_memory "$pin_memory"
  --eval_batch_size "${EVAL_BATCH_SIZE:-512}"  # 仅影响验证吞吐；降低 Val 峰值内存
  --subject_cache_size 4096         # 覆盖全部被试，消除每 epoch 对全宽 Raw mat 的重复 loadmat
  --train_samples_per_subject 0
  --sampling_seed "$sampling_seed"
  --val_ratio "$val_ratio"
  --test_ratio "$test_ratio"
)
if [[ -z "$RESUME_TRAINING_STATE" ]]; then
  ARGS+=(--pretrained_weight "$pretrained_weight")
fi

# ---- 本轮条件注入（选定方案：mdd8 + cat）----
ARGS+=(
  --condition_group "$cond_group_finetune" --condition_arm "$cond_arm_finetune"
  --cond_inject "$cond_inject" --cond_dim "$cond_dim"
  --cond_d_demo "$cond_d_demo" --cond_d_new "$cond_d_new"
  --cond_h_delta "$cond_h_delta" --cond_delta_style "$cond_delta_style"
  --cond_dropout_p "$cond_dropout_p"
  --cond_dropout_seed "$cond_dropout_seed"
)
if [[ -z "$RESUME_TRAINING_STATE" ]]; then
  # 统计量快照直接取自同一个 HC 起点权重文件：共享三字段的 median/IQR/类别词表逐位继承。
  ARGS+=(--condition_inherit_stats "$pretrained_weight")
fi
ARGS+=(
  --dynamics_mode "$dynamics_mode"
  --graph_source "$graph_source"    # 功能耦合图来自当前可见历史，不要求 Mask
  --bold_source "$bold_source"      # Raw/MDD/ROISignals_<id>.mat，前 32 TR 校准
  --personalized "$personalized"
  --profile_dim "$profile_dim"
  --calibration_length "$calibration_length"
  --ode_steps "$ode_steps"
  --dynamics_dt "$dynamics_dt"
  --lambda_dynamics "$lambda_dynamics"
  --lambda_variance "$lambda_variance"
  --dynamics_history_memory "$dynamics_history_memory"
)

if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ "$SAVE_TRAINING_STATE" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ \
      || -n "$RESUME_TRAINING_STATE" ]]; then
  ARGS+=(--save_training_state)
fi
if [[ -n "$RESUME_TRAINING_STATE" ]]; then
  ARGS+=(--resume_training_state "$RESUME_TRAINING_STATE")
fi
if [[ "${EVAL_AFTER_TRAIN:-True}" == "False" ]]; then
  ARGS+=(--no_eval_after_train)
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
