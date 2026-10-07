#!/usr/bin/env bash
# ============================================================================
# NeuroTwin-TFM 微调（MDD 队列）— C/D 变体：TFM-MoE + RECC（方案 §53/§55/§56-C/D）
#
# 用法：
#   bash scripts/Finetune_MDD_tfm_recc.sh diag   # C. MoE + Diagnose（只统计 mismatch）
#   bash scripts/Finetune_MDD_tfm_recc.sh        # D. MoE + RECC R2E（默认）
#   bash scripts/Finetune_MDD_tfm_recc.sh pair   # E. 叠加 Top-2 Pair-RECC
#
# 与 scripts/Finetune_MDD_tfm_moe.sh（B. TFM-MoE）的结构完全一致，仅新增
# Router–Expert competence coupling：真实下一时刻 BOLD → Expert competence →
# KL(sg(q) ‖ p_cpl) 校准 Router（方案 §16–§24）。梯度只更新 Router（§43）。
#
# 前置验收（§96–§98）：先确认 B（TFM-MoE）不明显差于 FiLM 基线再跑本脚本；
#   C（diagnose）用于确认 Router–Expert mismatch 存在（top1_agreement 不高、
#   oracle gap 明显），否则 RECC 必要性弱。
# ============================================================================
set -euo pipefail

RECC_VARIANT="${1:-r2e}"
case "$RECC_VARIANT" in
  diag)  TFM_RECC_MODE=diagnose ;;
  pair)  TFM_RECC_MODE=pair ;;
  r2e)   TFM_RECC_MODE=r2e ;;
  *) echo "[错误] 未知变体 '$RECC_VARIANT'（可选：diag / r2e / pair）"; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
pretrained_weight="$checkpoint_dir/neurotwin_tfm_pretrain/base_best.pt"
name="neurotwin_tfm_finetune_recc_${TFM_RECC_MODE}"

# ---- 实验规模（与预训练/基线微调保持一致的任务口径） ----
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

# ---- 病理残差专家 MoE（与 TFM-MoE 变体一致） ----
tfm_num_experts=4
tfm_top_k=2
tfm_expert_hidden_dim=256
tfm_router_hidden_dim=128
tfm_router_temperature=1.0
tfm_moe_dense_warmup_epochs=5
tfm_router_balance_weight=0.01
tfm_router_z_weight=0.001
tfm_router_entropy_weight=0.0005

# ---- RECC（方案 §55 推荐默认；第二批再搜索 0.01/0.10 与 τ 0.5/2.0，§59） ----
tfm_recc_weight=0.05
tfm_pair_recc_weight=0.05
tfm_competence_temperature=1.0
tfm_competence_pcc_weight=0.0
tfm_recc_start_epoch=5
tfm_recc_ramp_epochs=10

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
  # ---- 病理残差专家 MoE（与 TFM-MoE 变体一致）----
  --tfm_patho_mode moe
  --tfm_num_experts "$tfm_num_experts"
  --tfm_top_k "$tfm_top_k"
  --tfm_expert_hidden_dim "$tfm_expert_hidden_dim"
  --tfm_router_hidden_dim "$tfm_router_hidden_dim"
  --tfm_router_temperature "$tfm_router_temperature"
  --tfm_moe_dense_warmup_epochs "$tfm_moe_dense_warmup_epochs"
  --tfm_router_balance_weight "$tfm_router_balance_weight"
  --tfm_router_z_weight "$tfm_router_z_weight"
  --tfm_router_entropy_weight "$tfm_router_entropy_weight"
  # ---- RECC（本变体的核心开关）----
  --tfm_recc_mode "$TFM_RECC_MODE"
  --tfm_recc_weight "$tfm_recc_weight"
  --tfm_pair_recc_weight "$tfm_pair_recc_weight"
  --tfm_competence_temperature "$tfm_competence_temperature"
  --tfm_competence_pcc_weight "$tfm_competence_pcc_weight"
  --tfm_recc_start_epoch "$tfm_recc_start_epoch"
  --tfm_recc_ramp_epochs "$tfm_recc_ramp_epochs"
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
  # ---- 病理条件（HAMD → MoE Router/Expert-FiLM + CPM FiLM）----
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
