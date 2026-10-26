"""CPU partitions of the unchanged rolling evaluator; all subjects/all anchors.

Partitions write independent directories. The figure builder combines only
compatible manifests, then uses the original summarize_group implementation.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import torch

from analysis import rolling_one_step as rolling
from analysis.param_group_tests import _build
from analysis.evaluate import load_condition_bundle, load_train_args
from inference_twin import TwinSession
from analysis.pathology_tests import HC_MERGED_FILE
from utils.common import set_seed
from .style import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['hc', 'mdd'], required=True)
    parser.add_argument('--partition', type=int, required=True)
    parser.add_argument('--partitions', type=int, default=4)
    parser.add_argument('--out-dir', type=Path, required=True)
    ns = parser.parse_args()
    if not 0 <= ns.partition < ns.partitions:
        raise ValueError('Invalid partition index')
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO)
    checkpoint = ROOT/'checkpoints/neurotwin_pretrain/base_best.pt'
    args = load_train_args(str(checkpoint))
    bundle = load_condition_bundle(str(checkpoint))
    set_seed(int(args.seed))
    session = TwinSession(str(checkpoint), device='cpu', condition_file=HC_MERGED_FILE)
    loader = _build(args, ns.group, ROOT/'data', bundle)
    subjects = rolling.get_split_subjects(loader, 'val')
    todo = subjects[ns.partition::ns.partitions]
    view = rolling.build_dense_view(loader, todo, 64, 1)
    manifest = dict(checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=rolling.sha256(checkpoint), group=ns.group,
        mode='pretrain' if ns.group=='hc' else 'finetune', context_length=64,
        calibration_length=32, bold_source=str(args.bold_source),
        graph_source=str(args.graph_source), anchor_rule='All legal K64 anchors; stride=1',
        dtype='float32', deadzone=rolling.DEFAULT_DEADZONE,
        min_points=rolling.DEFAULT_MIN_POINTS, condition_group='hc3',
        lag_output="aux['dynamics_pred'][:, :, 0] (ODE head lag-1)")
    output = ns.out_dir/f'part_{ns.partition}'
    print(f'{ns.group} partition {ns.partition}/{ns.partitions}: {len(todo)} subjects, {len(view)} anchors', flush=True)
    rolling.forward_subjects(session.model, view, todo, torch.device('cpu'),64,0,
                             output,ns.group,manifest)
    summary = rolling.summarize_group(output,ns.group,'val',rolling.DEFAULT_DEADZONE,
                                      rolling.DEFAULT_MIN_POINTS,subjects_filter=todo)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
