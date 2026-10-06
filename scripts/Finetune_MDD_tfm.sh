#!/usr/bin/env bash
# ============================================================================
# NeuroTwin-TFM 微调（MDD 队列，病理 FiLM 条件化，方案 §12.2 / Task E）
#
# 用法：bash scripts/Finetune_MDD_tfm.sh
#
# 先在 HC 上按 scripts/Pretrain_HC_tfm.sh 预训练，得到
#   checkpoints/neurotwin_tfm_pretrain/base_best.pt，再运行本脚本。
# 两阶段逻辑：冻结 TFM 主干（仅训练病理 FiLM 条件适配器）→ 解冻联合优化
#   （backbone 学习率 × backbone_lr_scale）。
#
# ⚠️ 必须与预训练使用同一 model_arch/patch_len/cpm_horizon 任务口径
#   （检查点签名校验会拦截不匹配的组合）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
pretrained_weight="$checkpoint_dir/neurotwin_tfm_pretrain/base_best.pt"
name="neurotwin_tfm_finetune"

# ---- 实验规模（与预训练保持一致的任务口径） ----
seed=2024
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
freeze_backbone_epochs=10
backbone_lr_scale=0.2

# ---- 预检查 ----
[ -d "$data_root/Normlize/MDD" ] || {
  echo "[错误] 缺少 MDD 连续 BOLD 目录：$data_root/Normlize/MDD"; exit 1; }
[ -d "$data_root/Mask/MDD" ] || {
  echo "[错误] 缺少 MDD 结构连接目录：$data_root/Mask/MDD"; exit 1; }
[ -f "$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx" ] || {
  echo "[错误] 缺少临床评分表：$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx"; exit 1; }
[ -f "$pretrained_weight" ] || {
  echo "[错误] 缺少预训练权重：$pretrained_weight"
  echo "        请先运行 bash scripts/Pretrain_HC_tfm.sh"; exit 1; }

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
  --norm True
  --dropout 0.1
  # ---- 病理条件（HAMD → FiLM；冻结期 FiLM 仍参与训练）----
  --pathology_input_dim 1
  --pathology_norm_mode robust_z
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
  --num_workers 4
  --train_samples_per_subject 0
  --sampling_seed "$seed"
)

echo "[CMD] python -u main.py ${ARGS[*]}"
python -u main.py "${ARGS[@]}"
