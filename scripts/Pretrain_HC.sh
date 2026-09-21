#!/bin/bash
data_root="/data3/Digital_Brain/AMD/data"
checkpoint_dir="/data3/Digital_Brain/NeuroTwin/checkpoints"
model_id_name="neurotwin_pretrain"

seed=2024
num_rois=116
seq_len=30
total_windows=9
in_window=6

lr_init=5e-5
lr_peak=1e-4
lr_final=5e-5

train_epochs=150
warmup_epochs=15
batch_size=32
patience=20

n_block=2
ode_steps=6
ode_hidden_dim=256
num_scales=3

for pred_window in 1
do
  echo "=================================================="
  echo "[Pretrain/HC] Starting pretrain | Pred windows: ${pred_window}"
  echo "=================================================="

  python -u main.py \
    --mode pretrain \
    --seed $seed \
    --data_root $data_root \
    --checkpoint_dir $checkpoint_dir \
    --name ${model_id_name}_pred${pred_window} \
    --num_rois $num_rois \
    --seq_len $seq_len \
    --total_windows $total_windows \
    --in_window $in_window \
    --pred_window $pred_window \
    --n_block $n_block \
    --alpha 0.5 \
    --norm True \
    --dropout 0.2 \
    --ode_steps $ode_steps \
    --ode_hidden_dim $ode_hidden_dim \
    --num_scales $num_scales \
    --train_epochs $train_epochs \
    --batch_size $batch_size \
    --lr_init $lr_init \
    --lr_peak $lr_peak \
    --lr_final $lr_final \
    --warmup_epochs $warmup_epochs \
    --weight_decay 1e-2 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --num_threads 4 \
    --pin_memory True \
    --patience $patience \
    --amp True \
    --use_ema True \
    --ema_decay 0.999 \
    --init_log_var_pcc 0.0 \
    --init_log_var_mae -1.5 \
    --init_log_var_diff -2.0 \
    --init_log_var_std -2.0 \
    --loss_lr_scale 1.0 \
    --clamp_log_vars True \
    --log_var_min -6.0 \
    --log_var_max 6.0 \
    --moe_load_balance_weight 0.05 \
    --moe_entropy_weight 1e-3 \
    --moe_z_loss_weight 1e-3 \
    --moe_diversity_weight 0.001 \
    --moe_gate_temp_start 1.5 \
    --moe_gate_temp_end 1.0 \
    --moe_expert_hidden_dim 256 \
    --moe_use_shared_expert True \
    --moe_router_cond_only True \
    --moe_use_argmax False \
    --moe_inference_temperature 0.3
done

echo "所有预训练实验执行完毕！"
