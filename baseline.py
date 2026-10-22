# coding=utf-8
"""正式基线（S2 History Memory + E2 Loss）的构建与加载入口。

唯一事实来源为 ``configs/baseline_s2_e2.json``。本入口复用
``experiments/evaluate_variant`` 的快照重建逻辑（checkpoint 内嵌完整
config.args，避免散落在脚本里的硬编码），并校验权重 SHA256，保证
「配置 → 权重 → 模型」三者可追溯。

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


def load_baseline_config(path=BASELINE_CONFIG) -> dict:
    """读取正式基线配置（configs/baseline_s2_e2.json）。"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


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
    # 契约一致性：checkpoint 快照必须与基线声明一致，防止错拿其他臂的权重
    for key, expected in (('dynamics_history_memory', True),
                          ('dynamics_mode', 'graph_ode'),
                          ('lambda_temporal_diff',
                           cfg['loss']['lambda_temporal_diff']),
                          ('lambda_cross_head', cfg['loss']['lambda_cross_head']),
                          ('condition_arm', 'cat')):
        actual = getattr(args, key, None)
        if actual != expected:
            raise RuntimeError(
                f'checkpoint 快照 {key}={actual!r} 与基线声明 {expected!r} 不一致；'
                '请确认权重为 checkpoints/phase2_S2 的正式 S2 产物。')
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
