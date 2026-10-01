#!/usr/bin/env bash
# ============================================================================
# NeuroTwin Next-Timepoint 微调（MDD 队列，p(x_{t+1} | x_{t-K+1..t}, SC, HAMD)）
#
# 用法：bash scripts/Finetune_MDD_next_point.sh
#
# 先在 HC 上按 scripts/Pretrain_HC_next_point.sh 预训练，得到
#   checkpoints/neurotwin_nextpoint_pretrain/base_best.pt，再运行本脚本。
# 保留两阶段逻辑：冻结主干（仅训练 MoE / 条件模块）→ 解冻联合优化；
# 病理残差仍作用在**下一状态**上（delta_HC + delta_MDD），不再针对整段未来窗口。
#
# 若要用旧任务权重复用主干（预测头重新初始化），加 --load_backbone_only True
#   （脚本会打印 loaded / missing / reinitialized 三类清单）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
pretrained_weight="$checkpoint_dir/neurotwin_nextpoint_pretrain/base_best.pt"
name="neurotwin_nextpoint_finetune"

# ---- 实验规模（与预训练保持一致的任务口径） ----
seed=2024
num_rois=116
context_min=16
context_max=64
eval_context_length=64

train_epochs=150
warmup_epochs=15
batch_size=32
patience=20
lr_init=5e-5
lr_peak=1e-4
lr_final=5e-5

# ---- 预检查 ----
[ -d "$data_root/Normlize/MDD" ] || {
  echo "[错误] 缺少 MDD 连续 BOLD 目录：$data_root/Normlize/MDD"; exit 1; }
[ -d "$data_root/Mask/MDD" ] || {
  echo "[错误] 缺少 MDD 结构连接目录：$data_root/Mask/MDD"; exit 1; }
[ -f "$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx" ] || {
  echo "[错误] 缺少临床评分表：$data_root/Rest-meta-MDD-V1V2-Merged-MDD.xlsx"; exit 1; }
[ -f "$pretrained_weight" ] || {
  echo "[错误] 缺少预训练权重：$pretrained_weight"
  echo "        请先运行 bash scripts/Pretrain_HC_next_point.sh"; exit 1; }

ARGS=(
  # ---- 基础与路径 ----
  --mode finetune
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --pretrained_weight "$pretrained_weight"
  --num_rois "$num_rois"
  --norm False            # 2026-09-27：数据已逐 ROI z-score，关闭 BrainRevIN（与 Pretrain 脚本同步）
  # ---- Next-Timepoint 任务口径（Phase 9）----
  --task_mode next_timepoint
  --context_min "$context_min"
  --context_max "$context_max"
  --prediction_target delta
  --forecast_offsets 1
  --lambda_abs 1.0
  --lambda_delta 1.0
  --lambda_pcc 0.1
  --eval_context_length "$eval_context_length"
  --eval_anchors_per_subject 16
  --eval_rollout_tasks_per_subject 2
  --eval_rollout_horizons "1,2,4,8,16"
  --fc_min_length 32
  --eval_fc True
  --eval_spectral False
  # ---- 分阶段微调 ----
  --freeze_backbone_epochs 10
  --backbone_lr_scale 0.2
  --load_backbone_only False
  --pretrained_arch_policy require_match
  # ---- 病理条件化与 MoE（保留既有机制）----
  --pathology_fields HAMD
  --pathology_missing drop
  --pathology_norm_mode robust_z
  --patho_cond_layer joint
  --patho_adaln_targets both
  --lora_enable True
  --lora_rank 8
  --lora_n_blocks 2
  --num_experts 4
  --top_k 2
  --moe_experts_mode routed_only
  --moe_expert_kind homogeneous
  --moe_route_level sample
  --moe_router_cond_only False
  --moe_use_argmax False
  --moe_inference_temperature 0.3
  --moe_eval_mode dense_soft
  --moe_gate_features state_revin
  --moe_gate_temp_start 1.5
  --moe_gate_temp_end 1.0
  --moe_load_balance_weight 0.05
  --moe_entropy_weight 0.001
  --moe_z_loss_weight 0.001
  --moe_diversity_weight 0.001
  --moe_expert_hidden_dim 256
  --delta_refiner_rounds 1
  --refiner_rounds 3
  --pred_head point
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
  --compile True
  --num_workers 4
  --train_samples_per_subject 0
  --sampling_seed "$seed"
  # --refresh_split_manifest 是 store_true 开关（不接受取值），默认 False 不刷新
  # 训练结束默认自动在 test 集评估（evaluate_variant 的 next_timepoint 分支）；
  # 需要关闭时加 --no_eval_after_train（该开关不接受取值）
)

echo "[CMD] python -u main.py ${ARGS[*]}"
python -u main.py "${ARGS[@]}"
