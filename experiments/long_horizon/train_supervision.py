"""H8/H32 同起点对照；默认仅打印命令，--execute 才启动单臂训练。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/long_horizon_h32.json'


def training_environment(config: dict, arm: str, python: str = sys.executable) -> dict:
    """复用正式脚本，锁定实验相关覆盖值，避免继承其他消融的环境变量。"""
    baseline = json.loads((ROOT / config['baseline_config']).read_text())
    shared, variant = config['shared'], config['arms'][arm]
    loss, training = baseline['loss'], baseline['training']
    env = {
        'PYTHON': str(python),
        'HC_WEIGHT': str(ROOT / config['init_weight']),
        'SPLIT_FILE': str(ROOT / config['split_file']),
        'CHECKPOINT_DIR': str(ROOT / config['checkpoint_root'] / arm),
        'CPM_HORIZON': str(variant['cpm_horizon']),
        'PRETRAINED_HORIZON_POLICY': variant['pretrained_horizon_policy'],
        'FREEZE_BACKBONE_EPOCHS': str(shared['freeze_backbone_epochs']),
        'TRAIN_ANCHOR_MAX_OFFSET': str(shared['train_anchor_max_offset']),
        'EVAL_ANCHOR_MIN_FUTURE_STEPS': str(shared['eval_anchor_min_future_steps']),
        'EVAL_ROLLOUT_TASKS_PER_SUBJECT': str(shared['eval_rollout_tasks_per_subject']),
        'DYNAMICS_METRIC_HORIZON': str(shared['dynamics_metric_horizon']),
        'TRAIN_EPOCHS': str(training['train_epochs']),
        'BATCH_SIZE': str(shared.get('batch_size', training['batch_size'])),
        'EVAL_BATCH_SIZE': str(shared.get('eval_batch_size', 512)),
        'DYNAMICS_HISTORY_MEMORY': 'True', 'S4_VARIANT': 'off',
        'EVAL_AFTER_TRAIN': 'False',
        'LAMBDA_TEMPORAL_DIFF': str(loss['lambda_temporal_diff']),
        'LAMBDA_CROSS_HEAD': str(loss['lambda_cross_head']),
        'CROSS_HEAD_START_EPOCH': str(loss['cross_head_start_epoch']),
        'CPM_HORIZON_WEIGHT': loss['cpm_horizon_weight'],
        'CPM_HORIZON_ALPHA': str(loss['cpm_horizon_alpha']),
    }
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--arm', choices=['h8', 'h32'], required=True)
    parser.add_argument('--device', default='')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    env = training_environment(config, args.arm)
    if args.device:
        env['DEVICE'] = args.device
    cmd = ['bash', 'scripts/Finetune_MDD_tfm.sh']
    print(shlex.join(['env', *(f'{k}={v}' for k, v in env.items()), *cmd]), flush=True)
    if args.execute:
        subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=True)


if __name__ == '__main__':
    main()
