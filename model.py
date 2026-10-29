# coding=utf-8
"""NeuroTwin 固定结构的模型配置、权重身份校验与严格加载。

配置来源为 configs/neurotwin.json，实际结构从权重的 config.args 重建。
用法：python model.py --mode pretrain 或 --mode finetune。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = ROOT / 'configs' / 'neurotwin.json'

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
    (('loss', 'lambda_rec'), 'lambda_rec'),
    (('loss', 'lambda_fc'), 'lambda_fc'),
    (('loss', 'lambda_freq'), 'lambda_freq'),
    (('loss', 'loss_weighting'), 'loss_weighting'),
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

_MISSING = object()


def load_model_config(path=MODEL_CONFIG) -> dict:
    """读取正式模型配置（configs/neurotwin.json）。"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def config_value(cfg: dict, path):
    """按嵌套路径取模型配置值（测试与契约校验共用）。"""
    value = cfg
    for key in path:
        value = value[key]
    return value


def verify_checkpoint_contract(args, cfg: dict) -> None:
    """逐项核对 checkpoint ``config.args`` 与模型 JSON。

    检查模型角色对应的结构、训练配方和条件编码字段；缺失字段直接报错。
    """
    mismatches = []
    for path, arg_key in CONTRACT_FIELDS:
        mode = str(getattr(args, 'mode', 'finetune'))
        if path[:3] == ('data_and_split', 'conditioning', 'finetune'):
            path = (*path[:2], mode, *path[3:])
        expected = config_value(cfg, path)
        if path == ('architecture', 'tfm_patho_mode') and mode == 'pretrain':
            expected = 'none'
        actual = getattr(args, arg_key, _MISSING)
        if actual is _MISSING:
            mismatches.append(
                f'{arg_key}: checkpoint 缺少该字段（模型={expected!r}）')
        elif actual != expected:
            mismatches.append(
                f'{arg_key}: checkpoint={actual!r} 模型={expected!r}')
    if mismatches:
        raise RuntimeError(
            'checkpoint config.args 与模型配置不一致：\n  - '
            + '\n  - '.join(mismatches)
            + '\n请核对模型角色、权重与 configs/neurotwin.json。')


def verify_checkpoint_hash(path, expected_sha256) -> None:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h != expected_sha256:
        raise RuntimeError(
            f'模型权重 SHA256 不匹配：{path}\n'
            f'  期望 {expected_sha256}\n  实际 {h}\n'
            '该权重可能被覆盖或损坏；请勿修改 configs/neurotwin.json 的'
            '记录值，而应重新确认权重来源。')


def build_model(device='cpu', verify_hash=True, mode='finetune'):
    """按模型角色重建并 strict 加载已登记权重。

    Returns:
        (model, config)：model 为 eval 模式；config 为模型 JSON dict。
    """
    from analysis.evaluate import (build_model_from_args,
                                              load_condition_bundle,
                                              load_train_args)
    cfg = load_model_config()
    weight = cfg['weights'][mode]
    ckpt_path = ROOT / weight['path']
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'模型权重不存在: {ckpt_path}（configs/neurotwin.json）')
    if verify_hash:
        verify_checkpoint_hash(ckpt_path, weight['sha256'])

    args = load_train_args(ckpt_path)
    # 契约一致性：checkpoint 快照必须与模型声明逐项一致，防止加载错误角色的权重
    verify_checkpoint_contract(args, cfg)
    bundle = load_condition_bundle(ckpt_path)
    model = build_model_from_args(args, torch.device(device), bundle)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description='NeuroTwin 模型构建与权重加载自检')
    ap.add_argument('--mode', choices=['pretrain', 'finetune'], default='pretrain')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--skip-hash', action='store_true')
    ns = ap.parse_args()

    model, cfg = build_model(ns.device, verify_hash=not ns.skip_hash, mode=ns.mode)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] {cfg["_meta"]["name"]}')
    print(f'[model] checkpoint: {cfg["weights"][ns.mode]["path"]} '
          f'(sha256 校验通过)')
    print(f'[model] dynamics_mode=graph_ode | history_memory='
          f'{model.dynamics_history_memory}')
    print(f'[model] 模型构建并 strict 加载成功 | 参数量={n_params}')
    print(f'[model] λ_rec={cfg["loss"]["lambda_rec"]} | '
          f'λ_fc={cfg["loss"]["lambda_fc"]} | '
          f'λ_freq={cfg["loss"]["lambda_freq"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
