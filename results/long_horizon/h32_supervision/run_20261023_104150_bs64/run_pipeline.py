"""Sequential GPU0 H32/H8 training, followed by paired validation and test evaluation."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path.cwd()
RUN_DIR = Path(__file__).resolve().parent
CONFIG = RUN_DIR / 'run_config.json'
CFG = json.loads(CONFIG.read_text())
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '4',
       'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
STATE = {'pid': os.getpid(), 'device': 'physical GPU0 / cuda:0',
         'config': str(CONFIG.relative_to(ROOT)), 'completed': []}

def status(phase, **extra):
    STATE.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
    temp = RUN_DIR / 'status.tmp'
    temp.write_text(json.dumps(STATE, indent=2, ensure_ascii=False) + '\n')
    temp.replace(RUN_DIR / 'status.json')
    print(phase, extra, flush=True)

def run(phase, command):
    status(phase, command=command)
    with (RUN_DIR / (phase + '.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log,
                                   stderr=subprocess.STDOUT)
        status(phase, child_pid=process.pid)
        result = process.wait()
    if result:
        raise RuntimeError(f'{phase} exited with code {result}; see {phase}.log')
    STATE['completed'].append(phase)

try:
    for arm in ('h32', 'h8'):
        destination = ROOT / CFG['checkpoint_root'] / arm
        if destination.exists():
            raise FileExistsError(f'Refusing to reuse run checkpoint path: {destination}')
        run('train_' + arm, [sys.executable, '-u', '-m',
            'experiments.long_horizon.train_supervision', '--config', str(CONFIG),
            '--arm', arm, '--device', 'cuda:0', '--execute'])
        checkpoint = destination / 'neurotwin_tfm_finetune/finetuned_best.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        STATE[arm + '_checkpoint'] = str(checkpoint.relative_to(ROOT))
    for split in ('val', 'test'):
        run('eval_' + split, [sys.executable, '-u', '-m',
            'experiments.long_horizon.evaluate_supervision',
            '--h8', STATE['h8_checkpoint'], '--h32', STATE['h32_checkpoint'],
            '--config', str(CONFIG), '--device', 'cuda:0', '--split', split,
            '--num_workers', '0', '--out_dir', str(RUN_DIR / 'evaluation')])
    status('complete', child_pid=None)
except BaseException as error:
    status('failed', error=str(error), child_pid=None)
    traceback.print_exc()
    raise
