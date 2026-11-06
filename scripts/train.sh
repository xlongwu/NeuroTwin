#!/usr/bin/env bash
# 方案 B 共用入口；算法参数统一由 configs/neurotwin.json 提供。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mode="${1:?用法: bash scripts/train.sh pretrain|finetune [main.py 参数]}"
shift
case "$mode" in pretrain|finetune) ;; *) echo "未知训练阶段: $mode" >&2; exit 1 ;; esac
export TMPDIR="$ROOT/.runtime_tmp"
mkdir -p "$TMPDIR"
export TORCH_SHARING_STRATEGY=file_system
command=("${PYTHON:-python}" -u main.py --mode "$mode")
[[ -z "${MODEL_CONFIG:-}" ]] || command+=(--model_config "$MODEL_CONFIG")
[[ -z "${DATA_ROOT:-}" ]] || command+=(--data_root "$DATA_ROOT")
[[ -z "${DEVICE:-}" ]] || command+=(--device "$DEVICE")
[[ -z "${CHECKPOINT_DIR:-}" ]] || command+=(--checkpoint_dir "$CHECKPOINT_DIR")
[[ -z "${RESUME_TRAINING_STATE:-}" ]] || command+=(--resume_training_state "$RESUME_TRAINING_STATE")
if [[ "${SAVE_TRAINING_STATE:-True}" != False ]]; then command+=(--save_training_state); fi
if [[ "$mode" == pretrain && -n "${INIT_WEIGHT:-}" ]]; then command+=(--init_weight "$INIT_WEIGHT"); fi
if [[ "$mode" == finetune && -n "${HC_WEIGHT:-}" ]]; then command+=(--pretrained_weight "$HC_WEIGHT"); fi
if [[ "${EVAL_AFTER_TRAIN:-True}" == False ]]; then command+=(--no_eval_after_train); fi
exec "${command[@]}" "$@"
