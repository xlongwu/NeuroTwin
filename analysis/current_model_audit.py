"""Read-only checkpoint/data audit for the current analysis (no training)."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from experiments.evaluate_variant import build_model_from_args, load_condition_bundle, load_train_args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_digest(payload, mode):
    obj = {'mode': mode, **{k: sorted(payload[k]) for k in ('train', 'val', 'test')}}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def audit(root, out):
    records = []
    for p in sorted((root / 'checkpoints').glob('*/*.pt')):
        raw = torch.load(p, map_location='cpu', weights_only=False)
        args = load_train_args(p)
        bundle = load_condition_bundle(p)
        model = build_model_from_args(args, torch.device('cpu'), bundle)
        model.load_state_dict(raw['state_dict'], strict=True)
        mode = args.mode
        split_path = root / 'data' / ('subject_split_pretrain.json' if mode == 'pretrain' else 'subject_split_finetune.json')
        split = json.loads(split_path.read_text())
        h = split_digest(split, mode)
        groups = [set(split[k]) for k in ('train', 'val', 'test')]
        overlap = len(groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2])
        cohort = 'HC' if mode == 'pretrain' else 'MDD'
        missing = [s for g in groups for s in g if not (root / 'data' / 'Raw' / cohort / f'ROISignals_{s}.mat').exists()]
        rec = dict(checkpoint=str(p.relative_to(root)), sha256=digest(p), arch_version=raw.get('arch_version'),
                   strict_load=True, config=raw['config'], split_file=str(split_path.relative_to(root)),
                   split_hash=h, saved_split_hash=bundle['split_hash'] if bundle else None,
                   split_hash_matches=h == bundle['split_hash'] if bundle else False,
                   counts={k:len(split[k]) for k in ('train','val','test')}, overlap=overlap,
                   missing_raw=missing)
        records.append(rec)
        print(p.name, mode, getattr(args,'dynamics_mode',None), 'strict=OK', 'split=', rec['split_hash_matches'], flush=True)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'checkpoint_audit.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.root.resolve(), a.out)
