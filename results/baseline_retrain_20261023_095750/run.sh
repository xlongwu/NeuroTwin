#!/usr/bin/env bash
# S2+E2：修复编译解冻及学习率调度后的完整复训，保留历史产物。
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$RUN_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export DEVICE="${DEVICE:-cuda:1}"
export CHECKPOINT_DIR="$ROOT/checkpoints/$(basename "$RUN_DIR")"
export PYTHONIOENCODING=utf-8
export NUM_WORKERS=6 PIN_MEMORY=False EVAL_BATCH_SIZE=512
export TORCH_SHARING_STRATEGY=file_system
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRINT_CMD=1
stage=starting
trap 'code=$?; printf "%s %s exit=%s\n" "$(date -Is)" "$stage" "$code" > "$RUN_DIR/status.txt"' EXIT
# 本脚本只允许新建本轮 checkpoint 目录；不会选中旧 run 的权重。
if [[ -e "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory already exists: $CHECKPOINT_DIR" >&2
  exit 1
fi
mkdir -p "$CHECKPOINT_DIR"
cp configs/baseline_s2_e2.json "$RUN_DIR/recipe.json"
stage=pretrain
printf '%s %s\n' "$(date -Is)" "$stage" > "$RUN_DIR/status.txt"
bash scripts/Pretrain_HC_tfm.sh > "$RUN_DIR/pretrain.log" 2>&1
stage=finetune
printf '%s %s\n' "$(date -Is)" "$stage" > "$RUN_DIR/status.txt"
export HC_WEIGHT="$CHECKPOINT_DIR/neurotwin_tfm_pretrain/base_best.pt"
bash scripts/Finetune_MDD_tfm.sh > "$RUN_DIR/finetune.log" 2>&1
stage=evaluate
printf '%s %s\n' "$(date -Is)" "$stage" > "$RUN_DIR/status.txt"
ckpt="$CHECKPOINT_DIR/neurotwin_tfm_finetune/finetuned_best.pt"
eval_dir="$CHECKPOINT_DIR/neurotwin_tfm_finetune/eval"
# main.py 已自动评 test；若该步骤失败，显式复评并保留非零退出状态。
if [[ ! -s "$eval_dir/metrics_test.json" ]]; then
  "$PYTHON" -u -m experiments.evaluate_variant --ckpt "$ckpt" \
    --out_dir "$eval_dir" --splits test --device "$DEVICE" \
    > "$RUN_DIR/eval.log" 2>&1
fi
printf '%s\n' "$eval_dir/metrics_test.json" > "$RUN_DIR/metrics_path.txt"
stage=complete
