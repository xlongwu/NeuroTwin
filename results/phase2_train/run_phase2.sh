#!/usr/bin/env bash
# S2 正式基线复训脚本：训练结束后自动跑 next-state 评估与 D1/D2 状态刷新诊断。
# 历史 S0（=E2）/S1/S3 的配置与结果见 docs/EXPERIMENTS.md §4；当前只复现 S2。
# 训练配方由 configs/baseline_s2_e2.json 提供；PYTHON 可指向项目 mamba 解释器。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8
"$PYTHON" -c 'import torch' 2>/dev/null || {
  echo "[错误] $PYTHON 无法导入 torch；请用 PYTHON=<项目环境解释器> 指定" >&2
  exit 1
}
# 数据加载：关闭 pin_memory、降低 worker 数，规避长任务中 pin-memory/resource_sharer
# 线程在临时 socket 消失后中断训练（只影响吞吐，不影响结果）。
export NUM_WORKERS="${NUM_WORKERS:-6}"
export PIN_MEMORY="${PIN_MEMORY:-False}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# TMPDIR 必须稳定：Qoder/GUI 会话的 TMPDIR 指向 worker 专属目录，会在会话切换时
# 被清理，导致 DataLoader 的 multiprocessing resource_sharer socket 消失。
# 固定到项目内目录，长任务不再因临时目录回收而中断。
export TMPDIR="$ROOT/.runtime_tmp"
mkdir -p "$TMPDIR"
LOG="$ROOT/results/phase2_train"

eval_arm () {  # $1=arm, $2=device
  local arm="$1" dev="$2"
  local ckpt="$ROOT/checkpoints/phase2_$arm/neurotwin_tfm_finetune/finetuned_best.pt"
  "$PYTHON" -u -m experiments.evaluate_variant --ckpt "$ckpt" \
      --out_dir "$ROOT/results/phase2/eval_$arm" --splits test --save_arrays \
      --device "$dev" > "$LOG/${arm}_eval.log" 2>&1 \
      || echo "[phase2] $arm eval FAILED"
  "$PYTHON" -u -m experiments.long_horizon.refresh_rollout --ckpt "$ckpt" \
      --out_dir "$ROOT/results/phase2/refresh_$arm" --split test --steps 8 \
      --device "$dev" > "$LOG/${arm}_refresh.log" 2>&1 \
      || echo "[phase2] $arm refresh FAILED"
}

run_arm () {  # $1=arm, $2=device（训练配方来自 baseline JSON；S1 反馈已移除）
  local arm="$1" dev="$2"
  export CHECKPOINT_DIR="$ROOT/checkpoints/phase2_$arm"
  export DEVICE="$dev"
  mkdir -p "$CHECKPOINT_DIR" "$LOG" "$ROOT/results/phase2"
  echo "[phase2] $arm start $(date -Is) device=$dev workers=$NUM_WORKERS pin=$PIN_MEMORY"
  bash scripts/Pretrain_HC_tfm.sh > "$LOG/${arm}_pretrain.log" 2>&1 \
    && bash scripts/Finetune_MDD_tfm.sh > "$LOG/${arm}_finetune.log" 2>&1 \
    || { echo "[phase2] $arm TRAIN FAILED $(date -Is)"; return 1; }
  echo "[phase2] $arm train done $(date -Is); eval..."
  eval_arm "$arm" "$dev"
  echo "[phase2] $arm done $(date -Is)"
}

run_arm S2 cuda:1 & P2=$!
if ! wait "$P2"; then
  echo "[phase2] S2 FAILED $(date -Is)" >&2
  exit 1
fi
echo "[phase2] all done $(date -Is)"
