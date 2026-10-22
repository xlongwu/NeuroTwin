#!/usr/bin/env bash
# S2 正式基线复训脚本（S0 = E2 对照；S1/S3 路线已随 2026-10-22 定型淘汰，如需
# 复现见 git 历史）。每臂训练结束后自动跑 next-state 评估与 D1/D2 状态刷新诊断。
# 与 E2 同口径：LAMBDA_TEMPORAL_DIFF=0.05、LAMBDA_CROSS_HEAD=0.02、相同 seed。
set -u
ROOT=/data3/Digital_Brain/NeuroTwin
cd "$ROOT"
export PYTHON=/home/user/anaconda3/envs/mamba/bin/python
export PYTHONIOENCODING=utf-8
export LAMBDA_TEMPORAL_DIFF=0.05
export LAMBDA_CROSS_HEAD=0.02
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

run_arm () {  # $1=arm, $2=device, $3=history_memory（S1 反馈已随 2026-10-22 定型移除）
  local arm="$1" dev="$2" hmem="$3"
  export CHECKPOINT_DIR="$ROOT/checkpoints/phase2_$arm"
  export DEVICE="$dev" DYNAMICS_HISTORY_MEMORY="$hmem"
  mkdir -p "$CHECKPOINT_DIR" "$LOG" "$ROOT/results/phase2"
  echo "[phase2] $arm start $(date -Is) device=$dev hmem=$hmem workers=$NUM_WORKERS pin=$PIN_MEMORY"
  bash scripts/Pretrain_HC_tfm.sh > "$LOG/${arm}_pretrain.log" 2>&1 \
    && bash scripts/Finetune_MDD_tfm.sh > "$LOG/${arm}_finetune.log" 2>&1 \
    || { echo "[phase2] $arm TRAIN FAILED $(date -Is)"; return 1; }
  echo "[phase2] $arm train done $(date -Is); eval..."
  eval_arm "$arm" "$dev"
  echo "[phase2] $arm done $(date -Is)"
}

run_arm S2 cuda:1 True & P2=$!
wait "$P2"
# 等前两臂的 CUDA 上下文与共享内存完全释放后再启动 S3，避免残留显存导致 OOM
sleep 120
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 3000 ] && break
  echo "[phase2] waiting cuda:0 release: ${used} MiB"
  sleep 20
done
echo "[phase2] all done $(date -Is)"