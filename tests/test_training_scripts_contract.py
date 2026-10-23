# coding=utf-8
"""训练脚本 ↔ 基线 JSON / checkpoint 快照的参数契约测试。

通过一个只捕获 `main.py` argv 的假 Python 入口运行两份训练脚本，不启动训练。
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from baseline import CONTRACT_FIELDS, config_value, load_baseline_config

ROOT = Path(__file__).resolve().parents[1]
REAL_CKPT = ROOT / 'checkpoints' / 'phase2_S2' / 'neurotwin_tfm_finetune' \
    / 'finetuned_best.pt'
PRETRAIN_CKPT = ROOT / 'checkpoints' / 'phase2_S2' / 'neurotwin_tfm_pretrain' \
    / 'base_best.pt'

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='训练入口是 bash 脚本')


def _fake_python(tmp_path):
    path = tmp_path / 'fake_python.sh'
    path.write_text(
        '#!/usr/bin/env bash\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == "main.py" ]]; then\n'
        '    printf "%s\\n" "$@" > "${CAPTURE:?}"\n'
        '    exit 0\n'
        '  fi\n'
        'done\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding='utf-8')
    path.chmod(0o755)
    return path


def _prepare(tmp_path):
    data = tmp_path / 'data'
    (data / 'Raw' / 'HC').mkdir(parents=True)
    (data / 'Raw' / 'MDD').mkdir(parents=True)
    (data / 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx').write_text('', encoding='utf-8')
    checkpoints = tmp_path / 'checkpoints'
    hc = checkpoints / 'neurotwin_tfm_pretrain'
    hc.mkdir(parents=True)
    (hc / 'base_best.pt').write_text('', encoding='utf-8')
    return data, checkpoints


def _run_script(tmp_path, script, capture):
    data, checkpoints = _prepare(tmp_path)
    env = os.environ.copy()
    env.update(
        PYTHON=str(_fake_python(tmp_path)),
        CAPTURE=str(capture),
        DATA_ROOT=str(data),
        CHECKPOINT_DIR=str(checkpoints),
        TMPDIR=str(tmp_path / 'runtime'),
    )
    proc = subprocess.run(
        ['bash', str(ROOT / 'scripts' / script)],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return _parse_argv(capture)


def _parse_argv(path):
    tokens = Path(path).read_text(encoding='utf-8').splitlines()
    out = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('--'):
            key = tokens[i][2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                out[key] = tokens[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1
    return out


def _same(actual, expected):
    if isinstance(expected, bool):
        return str(actual) == str(expected)
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _expected_from_json(stage):
    cfg = load_baseline_config()
    skip = {'model_arch', 's4_variant', 'sc_delta_rank'}
    if stage == 'pretrain':
        # HC 预训练由 main.py 强制 tfm_patho_mode=none，且不进入冻结/解冻阶段。
        skip |= {'tfm_patho_mode', 'freeze_backbone_epochs', 'backbone_lr_scale'}
    expected = {
        arg_key: config_value(cfg, path)
        for path, arg_key in CONTRACT_FIELDS if arg_key not in skip}
    cond = cfg['data_and_split']['conditioning'][stage]
    expected['condition_group'] = cond['condition_group']
    expected['condition_arm'] = cond['condition_arm']
    return expected


@pytest.mark.parametrize('stage,script', [
    ('pretrain', 'Pretrain_HC_tfm.sh'),
    ('finetune', 'Finetune_MDD_tfm.sh'),
])
def test_scripts_follow_baseline_json(tmp_path, stage, script):
    got = _run_script(tmp_path, script, tmp_path / f'{stage}_args.txt')
    expected = _expected_from_json(stage)
    missing = sorted(set(expected) - set(got))
    mismatched = {
        key: (got[key], value) for key, value in expected.items()
        if key in got and not _same(got[key], value)}
    assert not missing, f'{script} 未传基线字段：{missing}'
    assert not mismatched, f'{script} 与基线不一致：{mismatched}'


@pytest.mark.skipif(not REAL_CKPT.exists() or not PRETRAIN_CKPT.exists(),
                    reason='正式 S2 权重不在本机')
@pytest.mark.parametrize('stage,script,ckpt', [
    ('pretrain', 'Pretrain_HC_tfm.sh', PRETRAIN_CKPT),
    ('finetune', 'Finetune_MDD_tfm.sh', REAL_CKPT),
])
def test_scripts_match_checkpoint_snapshot(tmp_path, stage, script, ckpt):
    import torch

    got = _run_script(tmp_path, script, tmp_path / f'{stage}_args.txt')
    snapshot = torch.load(ckpt, map_location='cpu', weights_only=True)
    snapshot = snapshot['config']['args']
    path_keys = {'mode', 'data_root', 'checkpoint_dir', 'name',
                 'pretrained_weight', 'condition_inherit_stats', 'device'}
    compared = {
        key: value for key, value in got.items()
        if key not in path_keys and key in snapshot}
    mismatched = {
        key: (actual, snapshot[key]) for key, actual in compared.items()
        if not _same(actual, snapshot[key])}
    assert compared, '未捕获到可比的训练参数'
    assert not mismatched, f'{script} 与 checkpoint 快照不一致：{mismatched}'


@pytest.mark.parametrize('arm,horizon,policy', [('h8', 8, 'require_match'),
                                              ('h32', 32, 'repeat_last')])
def test_h32_experiment_reuses_script_with_explicit_overrides(tmp_path, arm, horizon, policy):
    import json
    from experiments.long_horizon.train_supervision import training_environment, DEFAULT_CONFIG

    config = json.loads(DEFAULT_CONFIG.read_text())
    config['shared'].update(batch_size=64, eval_batch_size=128)
    overrides = training_environment(config, arm)
    data, checkpoints = _prepare(tmp_path)
    capture = tmp_path / 'args.txt'
    env = {**os.environ, **overrides, 'PYTHON': str(_fake_python(tmp_path)),
           'CAPTURE': str(capture), 'DATA_ROOT': str(data),
           'CHECKPOINT_DIR': str(checkpoints),
           'HC_WEIGHT': str(checkpoints / 'neurotwin_tfm_pretrain/base_best.pt')}
    subprocess.run(['bash', str(ROOT / 'scripts/Finetune_MDD_tfm.sh')],
                   cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    got = _parse_argv(capture)
    expected = _expected_from_json('finetune')
    expected.update(cpm_horizon=horizon, dynamics_metric_horizon=32,
                    batch_size=64, eval_batch_size=128,
                    pretrained_horizon_policy=policy, train_anchor_max_offset=32,
                    eval_rollout_tasks_per_subject=0,
                    eval_anchor_min_future_steps=32, freeze_backbone_epochs=0)
    assert all(_same(got[key], value) for key, value in expected.items())
    assert got['no_eval_after_train'] is True
    assert got['split_file'] == overrides['SPLIT_FILE']
