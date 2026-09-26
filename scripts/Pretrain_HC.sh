#!/bin/bash
# ============================================================================
# NeuroTwin 预训练（HC 队列，不含病理条件）
#
# 用法：bash scripts/Pretrain_HC.sh        # 在任意工作目录下均可执行
#
# ⚠️ 架构版本：当前代码 ARCH_VERSION=4（预测头概率分支、辅助反演头、Phase 0~7 结构调整）。
#    v3 及更早的权重（含 checkpoints/Version0_weights_and_results_0921/）会被
#    load_backbone_weights 拦截，必须用本脚本重新预训练，再用 scripts/Finetune_MDD.sh
#    微调；两个脚本必须由同一份代码产生，否则微调会因 arch_version 不匹配而报错。
#
# 参数分为「实验规模」与「固定开关」两部分：固定开关显式写出当前代码的默认值，
# 目的是让每次实验的配置在命令层面可复现；做消融时直接改这里的值，不要依赖默认值。
# ============================================================================
set -euo pipefail

# ---- 项目根目录（脚本位于 <root>/scripts/，路径均基于项目内，迁移无需修改） ----
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data_root="$ROOT/data"
checkpoint_dir="$ROOT/checkpoints"
model_id_name="neurotwin_pretrain"

# ---- 实验规模（改动需记录到实验说明） ----
seed=2024
num_rois=116
seq_len=30
total_windows=9
in_window=6
pred_windows=(1)              # 未来窗长度；可写 (1 2 3)，每个值是一次完整训练
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

# ---- 预检查（缺数据时立即失败，避免跑出空数据集） ----
[ -d "$data_root/ROISignals_window/HC" ] || {
  echo "[错误] 缺少 HC 滑窗数据目录：$data_root/ROISignals_window/HC"; exit 1; }
[ -d "$data_root/Mask/HC" ] || {
  echo "[错误] 缺少 HC 结构连接目录：$data_root/Mask/HC"; exit 1; }
if [ ! -d "$data_root/npz_cache/HC" ]; then
  echo "[提示] 未找到 $data_root/npz_cache/HC，将回退读取原始 .mat（I/O 更慢）"
fi

for pred_window in "${pred_windows[@]}"; do
  name="${model_id_name}_pred${pred_window}"

  echo "=================================================="
  echo "[Pretrain/HC] ${name} | pred_window=${pred_window}"
  echo "=================================================="

  ARGS=(
    # ---- 基础与路径 ----
    --mode pretrain
    --seed "$seed"
    --data_root "$data_root"
    --checkpoint_dir "$checkpoint_dir"
    --name "$name"
    --num_rois "$num_rois"
    --seq_len "$seq_len"
    --total_windows "$total_windows"
    --in_window "$in_window"
    --pred_window "$pred_window"
    # ---- 模型规模 ----
    --n_block "$n_block"
    --alpha 0.5
    --norm True
    --dropout 0.2
    --ode_steps "$ode_steps"
    --ode_hidden_dim "$ode_hidden_dim"
    --num_scales "$num_scales"
    --stochastic_depth_rate 0.10
    # ---- 优化器与训练规模 ----
    --train_epochs "$train_epochs"
    --batch_size "$batch_size"
    --lr_init "$lr_init"
    --lr_peak "$lr_peak"
    --lr_final "$lr_final"
    --warmup_epochs "$warmup_epochs"
    --weight_decay 1e-2
    --grad_clip 1.0
    --patience "$patience"
    --amp True
    --compile True
    --use_ema True
    --ema_decay 0.999
    --num_workers 4
    --num_threads 4
    --pin_memory True
    --cache_in_memory False
    --tf32 True
    # ---- 不确定度加权损失初值（沿用既有设置） ----
    --init_log_var_pcc 0.0
    --init_log_var_mae -1.5
    --init_log_var_diff -2.0
    --init_log_var_std -2.0
    --loss_lr_scale 1.0
    --clamp_log_vars True
    --log_var_min -6.0
    --log_var_max 6.0
    # ---- Phase 0：协议基础（检查点版本化 / 划分 / Refiner / 轮间监督） ----
    --pretrained_arch_policy require_match
    --val_ratio 0.10                 # 被试级 8:1:1 划分
    --test_ratio 0.10
    --stratify_bins 5
    --refiner_rounds 3
    --delta_refiner_rounds 1
    --refiner_adaptive False
    --refiner_inter_sup_weight 0.05
    --refiner_inter_sup_decay 0.5
    # ---- Phase 2：SC 软先验 ----
    --sc_prior_mode soft_prior
    --sc_lambda_mode global
    --sc_lambda_init 0.7
    --sc_mask_mode soft
    --sc_delta_a True
    --sc_prob_mask False             # 概率化 SC 掩码为 opt-in，默认关闭
    --sc_refiner_inject both
    --sc_sparsity_weight 1e-3
    --sc_entropy_weight 1e-3
    --sc_temporal_weight 0.0
    # ---- Phase 3：预测头 future query / 幅值重参数化 / 多尺度 ----
    --head_shape_mode query
    --future_query_mode roi
    --future_query_dim 32
    --future_query_layers 1
    --future_query_heads 4
    --head_amp_mode scale_mod_trend
    --head_scale_granularity window
    --head_amp_consistency_weight 0.01
    --head_cross_roi conv1
    --head_use_history_proj True
    --head_use_latent_proj True
    --head_use_temporal True
    --head_use_cross_roi True
    --head_use_win_attn True
    --head_use_revin_stats True
    --ode_window_attn on
    --mdm_scale_scheme divisor
    --mdm_scale_gate sample
    # ---- Phase 4：GraphODE 连续时间 ----
    --ode_solver rk2
    --ode_step_mode learnable
    --ode_step_scale 0.1
    # ---- Phase 5：MoE（预训练阶段不构建 MoE，参数仅为与微调命令对照） ----
    --moe_load_balance_weight 0.05
    --moe_entropy_weight 1e-3
    --moe_z_loss_weight 1e-3
    --moe_diversity_weight 0.001
    --moe_gate_temp_start 1.5
    --moe_gate_temp_end 1.0
    --moe_expert_hidden_dim 256
    --moe_use_shared_expert True
    --moe_router_cond_only True
    --moe_use_argmax False
    --moe_inference_temperature 0.3
    # ---- Phase 6：长程与状态递推（高风险项默认关闭，需先通过门控验证） ----
    --variable_cutoff False          # 保持既有样本口径；开启后样本数增加但需掩码配合
    --loss_diff_mode per_window      # 避免跨窗边界伪差分
    --recur_mode none
    --scheduled_sampling_start 0.0
    --scheduled_sampling_end 0.0
    # ---- Phase 7：不确定性 / 辅助反演 ----
    --pred_head gaussian             # 预训练主干即含概率头，微调/分析需保持一致
    --pred_quantiles 0.1,0.5,0.9
    --init_log_var_nll 2.0           # 初始权重 exp(-2)≈0.135，避免概率项早期主导
    --inversion_weight 0.0           # opt-in，默认不构建反演头
    --inversion_hidden_dim 128
    --intervention_mode latent       # 仅分析侧消费，此处固定口径
  )

  # 完整命令留痕，便于复现与写入实验说明
  echo "[CMD] python -u main.py ${ARGS[*]}"
  python -u main.py "${ARGS[@]}"
done

echo "所有预训练实验执行完毕！"