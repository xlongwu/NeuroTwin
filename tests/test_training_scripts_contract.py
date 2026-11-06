"""训练脚本只传递阶段和运行路径，算法参数由唯一配置解析。"""
import os
from pathlib import Path
import subprocess
from main import parse_args
from model import load_model_config

ROOT=Path(__file__).resolve().parents[1]


def test_pretrain_and_finetune_scripts_share_the_formal_config(tmp_path):
    capture=tmp_path/'argv.txt'
    fake=tmp_path/'python.sh'
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${CAPTURE:?}"\n')
    fake.chmod(0o755)
    for mode in ('pretrain','finetune'):
        env=os.environ.copy();env.update(PYTHON=str(fake),CAPTURE=str(capture),
            DEVICE='cpu',DATA_ROOT=str(tmp_path/'data'),CHECKPOINT_DIR=str(tmp_path/'weights'))
        result=subprocess.run(['bash',str(ROOT/'scripts'/f'{mode}.sh')],cwd=ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        tokens=capture.read_text().splitlines();index=tokens.index('main.py')
        args=parse_args(tokens[index+1:])
        cfg=load_model_config()
        assert args.mode==mode and args.batch_size==cfg['training']['batch_size']
        assert args.cpm_horizon==32 and args.lambda_delta==.5
        assert args.save_training_state and args.data_root==str(tmp_path/'data')
        assert args.condition_arm==('demo' if mode=='pretrain' else 'cat')


def test_finetune_defaults_to_registered_hc_weight():
    args=parse_args(['--mode','finetune'])
    assert args.pretrained_weight==load_model_config()['weights']['pretrain']['path']
