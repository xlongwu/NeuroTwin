#!/usr/bin/env bash
# ============================================================================
# NeuroTwin-TFM 预训练（HC 队列，TimesFM-3 风格主干，方案 §30 V1）
#
# 用法：bash scripts/Pretrain_HC_tfm.sh
#
# 任务口径（方案 §16/§30）：连续 BOLD context [B,F,K]
#   → temporal patching（p=4）→ SC-guided Temporal–Variate Block × N
#   → One-Step 头（x̂_(t+1)，z_t→z_(t+1) 状态转移）
#   + CPM 头（非自回归一次输出 x̂_(t+1:t+H)，H=8）
# 损失（方案 §20）：L = λ_one·(Huber + λ_pcc·(1−PCC_sp)) + λ_cpm·(γ 加权逐 horizon Huber)
#
# ⚠️ 与 legacy（scripts/Pretrain_HC_next_point.sh）的权重互不兼容：
#   检查点按 model_arch + patch_len + cpm_horizon 做任务口径签名校验。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
name="neurotwin_tfm_pretrain"

# ---- 实验规模（方案 §17 首选起点：D=256, N=4, patch=4, heads=4）----
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
tfm_patch_len=4                    # 方案 §4.3：{1,4,8} 消融，不用 TimesFM 的 32
cpm_horizon=8                      # 方案 §8：首选 8，随后测 {4,8,16}

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
  --model_arch tfm
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --num_rois "$num_rois"
  # ---- Next-Timepoint 任务口径 ----
  --task_mode next_timepoint
  --context_min "$context_min"
  --context_max "$context_max"
  --forecast_offsets 1               # TFM 的 one-step 头只建模 +1 偏移（方案 §10）
  --eval_context_length "$eval_context_length"
  --eval_anchors_per_subject "$eval_anchors_per_subject"
  --eval_rollout_tasks_per_subject "$eval_rollout_tasks_per_subject"
  --eval_rollout_horizons "$eval_rollout_horizons"
  --fc_min_length 32
  --eval_fc True
  --eval_spectral False
  # ---- TFM 模型规模 ----
  --tfm_dim "$tfm_dim"
  --tfm_layers "$tfm_layers"
  --tfm_heads "$tfm_heads"
  --tfm_patch_len "$tfm_patch_len"
  --tfm_ff_ratio 4
  --tfm_one_step_hidden_dim 256
  --cpm_horizon "$cpm_horizon"
  --cpm_layers 2
  --cpm_intervention False           # V1 仅预留接口，不做干预预测（方案 §13.1/§30）
  # ---- SC 软先验（A_eff 一次计算全层共享，方案 §7）----
  --sc_prior_mode soft_prior
  --sc_lambda_mode global
  --sc_lambda_init 0.7
  --sc_prior_rank 12
  --sc_delta_a True
  --sc_sinkhorn_iters 64
  # ---- TFM 损失权重（方案 §20；λ_cpm=0 可退化 Stage 0 基线）----
  --lambda_one 1.0
  --lambda_cpm 1.0
  --lambda_pcc 0.1
  --cpm_gamma 1.0                    # 1=逐 horizon 均匀权重
  --cpm_huber_delta 1.0
  --norm True                        # Context-only RevIN（方案 §15）
  --dropout 0.1
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
  --train_samples_per_subject 0    # 0=auto（每被试每 epoch min(合法组合数, 12)）
  --sampling_seed "$seed"
  # --refresh_split_manifest 是 store_true 开关（不接受取值），默认 False 不刷新
)

echo "[CMD] python -u main.py ${ARGS[*]}"
python -u main.py "${ARGS[@]}"
