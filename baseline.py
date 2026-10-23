# coding=utf-8
"""正式基线（S2 History Memory + E2 Loss）的构建与加载入口。

唯一事实来源为 ``configs/baseline_s2_e2.json``。本入口复用
``experiments/evaluate_variant`` 的快照重建逻辑（checkpoint 内嵌完整
config.args，避免散落在脚本里的硬编码），校验权重 SHA256，并逐项比对
快照与基线 JSON，保证「配置 → 权重 → 模型」三者可追溯。

用法::

    python baseline.py                       # 校验 + 构建 + 加载（CPU 冒烟）
    python baseline.py --device cuda:0       # 指定设备
    python baseline.py --skip-hash           # 跳过 SHA256（不推荐）
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BASELINE_CONFIG = ROOT / 'configs' / 'baseline_s2_e2.json'

#: JSON 路径 → checkpoint ``config.args`` 键。只列两份工件都有的字段；
#: 划分名单等不在 checkpoint 快照里的信息由数据清单单独负责。
CONTRACT_FIELDS = (
    (('architecture', 'model_arch'), 'model_arch'),
    (('architecture', 'task_mode'), 'task_mode'),
    (('architecture', 'num_rois'), 'num_rois'),
    (('architecture', 'context_min'), 'context_min'),
    (('architecture', 'context_max'), 'context_max'),
    (('architecture', 'forecast_offsets'), 'forecast_offsets'),
    (('architecture', 'calibration_length'), 'calibration_length'),
    (('architecture', 'tfm_dim'), 'tfm_dim'),
    (('architecture', 'tfm_layers'), 'tfm_layers'),
    (('architecture', 'tfm_heads'), 'tfm_heads'),
    (('architecture', 'tfm_ff_ratio'), 'tfm_ff_ratio'),
    (('architecture', 'tfm_patch_len'), 'tfm_patch_len'),
    (('architecture', 'tfm_one_step_hidden_dim'), 'tfm_one_step_hidden_dim'),
    (('architecture', 'tfm_patho_mode'), 'tfm_patho_mode'),
    (('architecture', 'cpm_horizon'), 'cpm_horizon'),
    (('architecture', 'cpm_layers'), 'cpm_layers'),
    (('architecture', 'cpm_intervention'), 'cpm_intervention'),
    (('architecture', 'norm'), 'norm'),
    (('architecture', 'dropout'), 'dropout'),
    (('architecture', 'sc_prior_mode'), 'sc_prior_mode'),
    (('architecture', 'sc_lambda_mode'), 'sc_lambda_mode'),
    (('architecture', 'sc_lambda_init'), 'sc_lambda_init'),
    (('architecture', 'sc_prior_rank'), 'sc_prior_rank'),
    (('architecture', 'sc_delta_a'), 'sc_delta_a'),
    (('architecture', 'sc_delta_rank'), 'sc_delta_rank'),
    (('architecture', 'sc_sinkhorn_iters'), 'sc_sinkhorn_iters'),
    (('architecture', 'dynamics_mode'), 'dynamics_mode'),
    (('architecture', 'ode_steps'), 'ode_steps'),
    (('architecture', 'dynamics_dt'), 'dynamics_dt'),
    (('architecture', 'graph_source'), 'graph_source'),
    (('architecture', 'personalized'), 'personalized'),
    (('architecture', 'profile_dim'), 'profile_dim'),
    (('architecture', 'dynamics_history_memory'), 'dynamics_history_memory'),
    (('architecture', 's4_variant'), 's4_variant'),
    (('loss', 'lambda_one'), 'lambda_one'),
    (('loss', 'lambda_cpm'), 'lambda_cpm'),
    (('loss', 'lambda_pcc'), 'lambda_pcc'),
    (('loss', 'lambda_dynamics'), 'lambda_dynamics'),
    (('loss', 'lambda_variance'), 'lambda_variance'),
    (('loss', 'lambda_temporal_diff'), 'lambda_temporal_diff'),
    (('loss', 'lambda_cross_head'), 'lambda_cross_head'),
    (('loss', 'cross_head_start_epoch'), 'cross_head_start_epoch'),
    (('loss', 'loss_weighting'), 'loss_weighting'),
    (('loss', 'cpm_gamma'), 'cpm_gamma'),
    (('loss', 'cpm_huber_delta'), 'cpm_huber_delta'),
    (('loss', 'cpm_horizon_weight'), 'cpm_horizon_weight'),
    (('loss', 'cpm_horizon_alpha'), 'cpm_horizon_alpha'),
    (('training', 'seed'), 'seed'),
    (('training', 'sampling_seed'), 'sampling_seed'),
    (('training', 'cond_dropout_seed'), 'cond_dropout_seed'),
    (('training', 'batch_size'), 'batch_size'),
    (('training', 'train_epochs'), 'train_epochs'),
    (('training', 'lr_init'), 'lr_init'),
    (('training', 'lr_peak'), 'lr_peak'),
    (('training', 'lr_final'), 'lr_final'),
    (('training', 'warmup_epochs'), 'warmup_epochs'),
    (('training', 'weight_decay'), 'weight_decay'),
    (('training', 'grad_clip'), 'grad_clip'),
    (('training', 'patience'), 'patience'),
    (('training', 'freeze_backbone_epochs'), 'freeze_backbone_epochs'),
    (('training', 'backbone_lr_scale'), 'backbone_lr_scale'),
    (('training', 'use_ema'), 'use_ema'),
    (('training', 'ema_decay'), 'ema_decay'),
    (('training', 'amp'), 'amp'),
    (('data_and_split', 'bold_source'), 'bold_source'),
    (('data_and_split', 'val_ratio'), 'val_ratio'),
    (('data_and_split', 'test_ratio'), 'test_ratio'),
    (('data_and_split', 'conditioning', 'finetune', 'condition_group'),
     'condition_group'),
    (('data_and_split', 'conditioning', 'finetune', 'condition_arm'),
     'condition_arm'),
    (('data_and_split', 'conditioning', 'cond_inject'), 'cond_inject'),
    (('data_and_split', 'conditioning', 'cond_dim'), 'cond_dim'),
    (('data_and_split', 'conditioning', 'cond_d_demo'), 'cond_d_demo'),
    (('data_and_split', 'conditioning', 'cond_d_new'), 'cond_d_new'),
    (('data_and_split', 'conditioning', 'cond_h_delta'), 'cond_h_delta'),
    (('data_and_split', 'conditioning', 'cond_delta_style'), 'cond_delta_style'),
    (('data_and_split', 'conditioning', 'cond_dropout_p'), 'cond_dropout_p'),
)

#: 旧 checkpoint 早于某个参数时允许按该默认值解释；新参数缺失仍报错。
_MISSING_ARG_DEFAULTS = {'s4_variant': 'off'}
_MISSING = object()


def load_baseline_config(path=BASELINE_CONFIG) -> dict:
    """读取正式基线配置（configs/baseline_s2_e2.json）。"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def config_value(cfg: dict, path):
    """按嵌套路径取基线配置值（测试与契约校验共用）。"""
    value = cfg
    for key in path:
        value = value[key]
    return value


def verify_checkpoint_contract(args, cfg: dict) -> None:
    """逐项核对 checkpoint ``config.args`` 与基线 JSON。

    只比两份工件都声明的字段；checkpoint 中缺失的旧字段按
    :data:`_MISSING_ARG_DEFAULTS` 解释，其余缺失视为契约不完整。
    """
    mismatches = []
    for path, arg_key in CONTRACT_FIELDS:
        expected = config_value(cfg, path)
        actual = getattr(args, arg_key, _MISSING)
        if actual is _MISSING:
            actual = _MISSING_ARG_DEFAULTS.get(arg_key, _MISSING)
        if actual is _MISSING:
            mismatches.append(
                f'{arg_key}: checkpoint 缺少该字段（基线={expected!r}）')
        elif actual != expected:
            mismatches.append(
                f'{arg_key}: checkpoint={actual!r} 基线={expected!r}')
    if mismatches:
        raise RuntimeError(
            'checkpoint config.args 与基线配置不一致：\n  - '
            + '\n  - '.join(mismatches)
            + '\n请确认权重来自正式 S2 基线，或更新 configs/baseline_s2_e2.json。')


def verify_checkpoint_hash(path, expected_sha256) -> None:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h != expected_sha256:
        raise RuntimeError(
            f'基线权重 SHA256 不匹配：{path}\n'
            f'  期望 {expected_sha256}\n  实际 {h}\n'
            '该权重可能被覆盖或损坏；请勿修改 configs/baseline_s2_e2.json 的'
            '记录值，而应重新确认权重来源。')


def build_baseline_model(device='cpu', verify_hash=True):
    """按基线契约重建 S2 模型并 strict 加载正式 checkpoint。

    Returns:
        (model, config)：model 为 eval 模式；config 为基线 JSON dict。
    """
    from experiments.evaluate_variant import (build_model_from_args,
                                              load_condition_bundle,
                                              load_train_args)
    cfg = load_baseline_config()
    ckpt_path = ROOT / cfg['weights']['main']['path']
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'基线权重不存在: {ckpt_path}（configs/baseline_s2_e2.json）')
    if verify_hash:
        verify_checkpoint_hash(ckpt_path, cfg['weights']['main']['sha256'])

    args = load_train_args(ckpt_path)
    # 契约一致性：checkpoint 快照必须与基线声明逐项一致，防止错拿其他臂的权重
    verify_checkpoint_contract(args, cfg)
    bundle = load_condition_bundle(ckpt_path)
    model = build_model_from_args(args, torch.device(device), bundle)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description='正式基线（S2+E2）构建/加载自检')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--skip-hash', action='store_true')
    ns = ap.parse_args()

    model, cfg = build_baseline_model(ns.device, verify_hash=not ns.skip_hash)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[baseline] {cfg["_meta"]["name"]}')
    print(f'[baseline] checkpoint: {cfg["weights"]["main"]["path"]} '
          f'(sha256 校验通过)')
    print(f'[baseline] dynamics_mode=graph_ode | history_memory='
          f'{model.dynamics_history_memory} | s4_variant={model.s4_variant}')
    print(f'[baseline] 模型构建并 strict 加载成功 | 参数量={n_params}')
    print(f'[baseline] λ_temporal_diff={cfg["loss"]["lambda_temporal_diff"]} | '
          f'λ_cross_head={cfg["loss"]["lambda_cross_head"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
