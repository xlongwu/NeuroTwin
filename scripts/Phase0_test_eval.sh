#!/usr/bin/env bash
# ============================================================================
# Phase 0（协议冻结）统一评估脚本 —— 只加载**锁定的独立 test 集**做评估
#
# 用法：
#   bash scripts/Phase0_test_eval.sh                      # 评默认两个 finetune 权重
#   bash scripts/Phase0_test_eval.sh <ckpt.pt> [...]      # 评指定权重
#   BOOTSTRAP_REPS=0 bash scripts/Phase0_test_eval.sh     # 关闭 bootstrap（快，但无 CI）
#
# 产出（每个 checkpoint 一个目录，默认 analysis_results/phase0_eval/<名称>/）：
#   metrics_test.json                    含 split_lock / confidence_intervals /
#                                        paired_tests / shuffled_hamd 四个 Phase 0 块
#   per_subject_metrics_test.csv         被试级指标（含 HAMD）
#   per_task_metrics_test.csv            逐任务 t+1 指标
#   per_task_rollout_metrics_test.csv    逐任务逐 horizon 指标（长程配对检验的前提）
#   shuffled_hamd/metrics_test.json      shuffled-HAMD 负对照那一次评估的完整报告
#
# 说明：
#   - 本脚本**不训练、不改结构、不改损失**，只做评估；输出目录与
#     checkpoints/<name>/eval/ 分离，不会覆盖既有实验结果。
#   - AR(1) 基线只用 train subjects 拟合；shuffled-HAMD 只做被试↔评分置换，
#     取值多重集不变。
#   - split manifest 缺失或与配置不一致时 dataloader 会直接报错（这是 Phase 0
#     要求的"切分完全锁定"）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
out_root="${OUT_ROOT:-$ROOT/analysis_results/phase0_eval}"
bootstrap_reps="${BOOTSTRAP_REPS:-2000}"
seed="${SEED:-2024}"

# ---- 默认评估对象（可用位置参数覆盖） ----
if [ "$#" -gt 0 ]; then
  ckpts=("$@")
else
  ckpts=(
    "$checkpoint_dir/neurotwin_nextpoint_finetune/finetuned_best.pt"
    "$checkpoint_dir/neurotwin_nextpoint_finetune_rollamp/finetuned_best.pt"
  )
fi

# ---- 预检查 ----
for ckpt in "${ckpts[@]}"; do
  [ -f "$ckpt" ] || { echo "[错误] 权重不存在：$ckpt"; exit 1; }
done
[ -d "$data_root/Normlize/MDD" ] || {
  echo "[错误] 缺少 MDD 连续 BOLD 目录：$data_root/Normlize/MDD"; exit 1; }
[ -f "$data_root/subject_split_finetune.json" ] || {
  echo "[错误] 缺少版本化切分 manifest：$data_root/subject_split_finetune.json"
  echo "        Phase 0 要求 subject-level split 完全锁定；请先用训练/评估生成该文件。"
  exit 1; }

mkdir -p "$out_root"
echo "[Phase 0] test 集评估 | 权重数=${#ckpts[@]} | bootstrap_reps=$bootstrap_reps" \
     "| seed=$seed | 输出=$out_root"

for ckpt in "${ckpts[@]}"; do
  name="$(basename "$(dirname "$ckpt")")"
  out_dir="$out_root/$name"
  mkdir -p "$out_dir"
  echo "------------------------------------------------------------"
  echo "[Phase 0] $name ← $ckpt"
  "$PYTHON" -u -m experiments.evaluate_variant \
    --ckpt "$ckpt" \
    --out_dir "$out_dir" \
    --splits test \
    --seed "$seed" \
    --bootstrap_reps "$bootstrap_reps" \
    2>&1 | tee "$out_dir/eval.log"
done

echo "============================================================"
echo "[Phase 0] 完成。核对要点："
echo "  1) metrics_test.json -> split_lock：manifest_sha256 是否与各次运行一致；"
echo "     n_evaluated_subjects 是否等于 n_manifest_subjects_in_split"
echo "  2) confidence_intervals：主指标是否带 95% CI（被试级 cluster bootstrap）"
echo "  3) paired_tests.model_vs.*：模型是否**显著**优于 persistence/trend/ar1"
echo "  4) paired_tests.rollout_model_vs.persistence：H4/H8/H16 是否仍占优"
echo "  5) shuffled_hamd.delta_true_minus_shuffled：HAMD 条件化是否有独立贡献"
echo "  6) per_task_rollout_metrics_test.csv：长程指标可做任何两个权重的配对检验"
