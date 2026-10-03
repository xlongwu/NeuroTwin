#!/usr/bin/env bash
# ============================================================================
# NeuroTwin Next-Timepoint 预训练（HC 队列，p(x_{t+1} | x_{t-K+1..t}, SC)）
#
# 用法：bash scripts/Pretrain_HC_next_point.sh
#
# 任务口径（Phase 9）：连续 BOLD context（K∈[context_min, context_max] 个 TR）
#   → 下一 TR 全脑状态；delta 预测（x̂ = x_t + Δx̂）+ spatial PCC。
#   当前唯一任务口径，权重与任何旧口径产物不可混用（任务口径闸门会拦截）。
#
# ⚠️ 数据：直接读取已经完成 preprocessing 与 ROI extraction 的连续序列
#   data/Normlize/<HC|MDD>/ROISignals_<id>.mat（T=150 TR，无 NaN/Inf）；
#   该数据已做逐 ROI 全序列 z-score，训练侧不再做第二次标准化（仅做 context 内 RevIN）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
name="neurotwin_nextpoint_pretrain"

# ---- 实验规模 ----
seed=2024
num_rois=116
context_min=16
context_max=64                 # = 模型 S 轴建模宽度；训练采样 K∈[context_min, context_max]
eval_context_length=64         # 评估固定 K（确定性 protocol）
eval_anchors_per_subject=16
eval_rollout_tasks_per_subject=2
eval_rollout_horizons="1,2,4,8,16"
n_block=2
ode_steps=3
ode_hidden_dim=256
num_scales=3

train_epochs=150
warmup_epochs=15
batch_size=32
patience=20
lr_init=5e-5
lr_peak=1e-4
lr_final=5e-5

# ---- 预检查 ----
[ -d "$data_root/Normlize/HC" ] || {
  echo "[错误] 缺少 HC 连续 BOLD 目录：$data_root/Normlize/HC"; exit 1; }
[ -d "$data_root/Mask/HC" ] || {
  echo "[错误] 缺少 HC 结构连接目录：$data_root/Mask/HC"; exit 1; }

ARGS=(
  # ---- 基础与路径 ----
  --mode pretrain
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --num_rois "$num_rois"
  # ---- Next-Timepoint 任务口径（Phase 9）----
  --task_mode next_timepoint
  --context_min "$context_min"
  --context_max "$context_max"
  --prediction_target delta          # 预测头以 x_t 为锚点：x̂ = x_t + Δx̂
  --causal_training random_context   # 方案 B（未来信息不进入前向）
  --forecast_offsets 1               # 单步；MTP 消融改用 --enable_mtp True
  --lambda_abs 1.0
  --lambda_delta 1.0
  --lambda_pcc 0.1
  --eval_context_length "$eval_context_length"
  --eval_anchors_per_subject "$eval_anchors_per_subject"
  --eval_rollout_tasks_per_subject "$eval_rollout_tasks_per_subject"
  --eval_rollout_horizons "$eval_rollout_horizons"
  --fc_min_length 32
  --eval_fc True
  --eval_spectral False
  # ---- 模型规模 ----
  --n_block "$n_block"
  --alpha 0.5
  --dropout 0.2
  --norm False
  --ode_steps "$ode_steps"
  --ode_hidden_dim "$ode_hidden_dim"
  --num_scales "$num_scales"
  --stochastic_depth_rate 0.1
  --sc_prior_mode soft_prior
  --sc_lambda_mode global
  --sc_lambda_init 0.7
  --sc_prior_rank 12
  --sc_delta_a True
  --sc_mask_mode soft
  --sc_refiner_inject both
  --head_cross_roi conv1
  --head_shape_mode query
  --future_query_mode roi
  --head_amp_mode scale_mod_trend
  --head_amp_consistency_weight 0.01
  --head_use_history_proj True
  --head_use_latent_proj True
  --head_use_temporal True
  --head_use_cross_roi True
  --head_use_win_attn True
  --head_use_revin_stats True
  --ode_solver rk2
  --ode_step_mode learnable
  --mdm_scale_scheme divisor
  --mdm_scale_gate sample
  --pred_head point                  # 单点预测默认不启用概率头（logvar 换算在近常数 context 上不稳定）
  --refiner_rounds 3
  --refiner_inter_sup_weight 0.05
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
  --train_samples_per_subject 0    # 0=auto（每被试每 epoch min(合法组合数, 12)）
  --sampling_seed "$seed"
  # --refresh_split_manifest 是 store_true 开关（不接受取值），默认 False 不刷新
)

echo "[CMD] python -u main.py ${ARGS[*]}"
python -u main.py "${ARGS[@]}"
