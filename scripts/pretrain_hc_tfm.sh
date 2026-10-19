#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
data_root="${DATA_ROOT:-$ROOT/data}"
checkpoint_dir="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
DEVICE="${DEVICE:-}"
name="neurotwin_tfm_pretrain"

read -r seed sampling_seed cond_dropout_seed < <(
  PYTHONPATH=. "$PYTHON_BIN" -c 'from utils.condition_schema import (DROPOUT_SEED, MODEL_SEED, SAMPLING_SEED); print(MODEL_SEED, SAMPLING_SEED, DROPOUT_SEED)') \
  || { echo "[错误] $PYTHON_BIN 读取 utils.condition_schema 的随机来源常量失败（需 PYTHON=<≥3.10 且带 torch 的解释器>）" >&2; exit 1; }
if [[ "$seed" == "$sampling_seed" || "$seed" == "$cond_dropout_seed" \
      || "$sampling_seed" == "$cond_dropout_seed" ]]; then
  echo "[错误] 登记处自身的随机来源就不同值（seed=$seed sampling=$sampling_seed dropout=$cond_dropout_seed），违反方案 §四" >&2
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
tfm_patch_len=4
cpm_horizon=8

train_epochs=150
warmup_epochs=15
batch_size=32
patience=20
lr_init=5e-5
lr_peak=1e-4
lr_final=5e-5

[ -d "$data_root/Raw/HC" ] || {
  echo "[错误] 缺少 HC Raw 目录（$data_root/Raw/HC/ROISignals_<id>.mat，[T,F] 布局）"; exit 1; }

ARGS=(
  --mode pretrain
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --num_rois "$num_rois"
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
  --tfm_dim "$tfm_dim"
  --tfm_layers "$tfm_layers"
  --tfm_heads "$tfm_heads"
  --tfm_patch_len "$tfm_patch_len"
  --tfm_ff_ratio 4
  --tfm_one_step_hidden_dim 256
  --cpm_horizon "$cpm_horizon"
  --cpm_layers 2
  --sc_prior_mode soft_prior
  --sc_lambda_mode global
  --sc_lambda_init 0.7
  --sc_prior_rank 12
  --sc_delta_a True
  --sc_sinkhorn_iters 64
  --lambda_one 1.0
  --lambda_cpm 1.0
  --lambda_pcc 0.1
  --cpm_gamma 1.0
  --cpm_huber_delta 1.0
  --norm True
  --dropout 0.1
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
  --num_workers 4
  --train_samples_per_subject 0
  --sampling_seed "$sampling_seed"
)

ARGS+=(
  --condition_group hc3 --condition_arm demo
  --cond_dropout_seed "$cond_dropout_seed"
)
ARGS+=(
  --dynamics_mode graph_ode
  --graph_source history_fc
  --bold_source raw
  --personalized True
  --profile_dim 32
  --calibration_length 32
  --ode_steps 2
  --dynamics_dt 1.0
  --lambda_dynamics 0.5
  --lambda_variance 0.05
)
if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi

arg_value() {
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
