#!/usr/bin/env bash
# ============================================================================
# 从 configs/neurotwin.json 读取正式训练配方。
#
# 调用方先设置 ROOT 与 PYTHON_BIN，再 source 本文件。导出的变量供
# pretrain.sh / finetune.sh 拼装 main.py 参数；阶段特有的
# 条件分组/臂分别导出为 cond_group_* / cond_arm_*。环境变量覆盖由调用方
# 以 ${VAR:-$var} 的形式应用。
#
# 随机种子在导出前会与 utils/condition_schema.py 的登记值逐项核对：JSON 是
# 模型配方的唯一来源，登记处是代码侧守卫，两者不一致时直接失败。
# ============================================================================

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[错误] 本文件只供训练脚本 source，不能直接执行" >&2
  exit 1
fi
if [[ -z "${ROOT:-}" || -z "${PYTHON_BIN:-}" ]]; then
  echo "[错误] source 本文件前必须先设置 ROOT 与 PYTHON_BIN" >&2
  return 1
fi

model_json="$ROOT/configs/neurotwin.json"
[ -f "$model_json" ] || {
  echo "[错误] 找不到模型配置：$model_json" >&2
  return 1
}

_model_env="$(
  PYTHONPATH="$ROOT" "$PYTHON_BIN" - "$model_json" <<'PY'
import json
import shlex
import sys

from utils.condition_schema import DROPOUT_SEED, MODEL_SEED, SAMPLING_SEED

with open(sys.argv[1], encoding='utf-8') as f:
    cfg = json.load(f)
arch = cfg['architecture']
loss = cfg['loss']
train = cfg['training']
cond = cfg['data_and_split']['conditioning']

seeds = (int(train['seed']), int(train['sampling_seed']),
         int(train['cond_dropout_seed']))
registry = (MODEL_SEED, SAMPLING_SEED, DROPOUT_SEED)
if seeds != registry:
    raise SystemExit(
        '模型配置的随机种子与 utils/condition_schema.py 登记值不一致：'
        f'json={seeds} registry={registry}')

values = {
    'seed': train['seed'],
    'sampling_seed': train['sampling_seed'],
    'cond_dropout_seed': train['cond_dropout_seed'],
    'num_rois': arch['num_rois'],
    'context_min': arch['context_min'],
    'context_max': arch['context_max'],
    'forecast_offsets': arch['forecast_offsets'],
    'calibration_length': arch['calibration_length'],
    'tfm_dim': arch['tfm_dim'],
    'tfm_layers': arch['tfm_layers'],
    'tfm_heads': arch['tfm_heads'],
    'tfm_ff_ratio': arch['tfm_ff_ratio'],
    'tfm_patch_len': arch['tfm_patch_len'],
    'tfm_one_step_hidden_dim': arch['tfm_one_step_hidden_dim'],
    'tfm_patho_mode': arch['tfm_patho_mode'],
    'cpm_horizon': arch['cpm_horizon'],
    'cpm_layers': arch['cpm_layers'],
    'cpm_intervention': arch['cpm_intervention'],
    'norm': arch['norm'],
    'dropout': arch['dropout'],
    'sc_prior_mode': arch['sc_prior_mode'],
    'sc_lambda_mode': arch['sc_lambda_mode'],
    'sc_lambda_init': arch['sc_lambda_init'],
    'sc_prior_rank': arch['sc_prior_rank'],
    'sc_delta_a': arch['sc_delta_a'],
    'sc_sinkhorn_iters': arch['sc_sinkhorn_iters'],
    'dynamics_mode': arch['dynamics_mode'],
    'ode_steps': arch['ode_steps'],
    'dynamics_dt': arch['dynamics_dt'],
    'graph_source': arch['graph_source'],
    'personalized': arch['personalized'],
    'profile_dim': arch['profile_dim'],
    'dynamics_history_memory': arch['dynamics_history_memory'],
    'bold_source': cfg['data_and_split']['bold_source'],
    'val_ratio': cfg['data_and_split']['val_ratio'],
    'test_ratio': cfg['data_and_split']['test_ratio'],
    'cond_group_pretrain': cond['pretrain']['condition_group'],
    'cond_arm_pretrain': cond['pretrain']['condition_arm'],
    'cond_group_finetune': cond['finetune']['condition_group'],
    'cond_arm_finetune': cond['finetune']['condition_arm'],
    'cond_inject': cond['cond_inject'],
    'cond_dim': cond['cond_dim'],
    'cond_d_demo': cond['cond_d_demo'],
    'cond_d_new': cond['cond_d_new'],
    'cond_h_delta': cond['cond_h_delta'],
    'cond_delta_style': cond['cond_delta_style'],
    'cond_dropout_p': cond['cond_dropout_p'],
    'lambda_one': loss['lambda_one'],
    'lambda_cpm': loss['lambda_cpm'],
    'lambda_pcc': loss['lambda_pcc'],
    'lambda_dynamics': loss['lambda_dynamics'],
    'lambda_variance': loss['lambda_variance'],
    'lambda_temporal_diff': loss['lambda_temporal_diff'],
    'lambda_cross_head': loss['lambda_cross_head'],
    'cross_head_start_epoch': loss['cross_head_start_epoch'],
    'loss_weighting': loss['loss_weighting'],
    'cpm_gamma': loss['cpm_gamma'],
    'cpm_huber_delta': loss['cpm_huber_delta'],
    'cpm_horizon_weight': loss['cpm_horizon_weight'],
    'cpm_horizon_alpha': loss['cpm_horizon_alpha'],
    'train_epochs': train['train_epochs'],
    'batch_size': train['batch_size'],
    'lr_init': train['lr_init'],
    'lr_peak': train['lr_peak'],
    'lr_final': train['lr_final'],
    'warmup_epochs': train['warmup_epochs'],
    'weight_decay': train['weight_decay'],
    'grad_clip': train['grad_clip'],
    'patience': train['patience'],
    'freeze_backbone_epochs': train['freeze_backbone_epochs'],
    'backbone_lr_scale': train['backbone_lr_scale'],
    'use_ema': train['use_ema'],
    'ema_decay': train['ema_decay'],
    'amp': train['amp'],
}
for key, value in values.items():
    if isinstance(value, bool):
        value = 'True' if value else 'False'
    elif value is None:
        continue
    print(f'{key}={shlex.quote(str(value))}')
PY
)" || {
  echo "[错误] 读取模型配置失败：$model_json" >&2
  return 1
}
eval "$_model_env"
unset _model_env model_json
