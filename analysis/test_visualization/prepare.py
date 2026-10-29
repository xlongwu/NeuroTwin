"""Create a fresh test-only run and record frozen sources before computation."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import (ROOT,CHECKPOINT,DATA,TABLES,FIGURES,SPLITS,test_ids,write_json,digest)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=DATA);ap.add_argument('--table-dir',type=Path,default=TABLES)
    ap.add_argument('--figure-dir',type=Path,default=FIGURES)
    ns=ap.parse_args()
    for path in [ns.data_dir,ns.table_dir,ns.figure_dir]:
        if path.exists():raise FileExistsError(f'Choose a fresh run directory: {path}')
    for path in [ns.data_dir,ns.table_dir,ns.figure_dir]:path.mkdir(parents=True)
    sources=[]
    roots=['results/数据/评估来源.json','results/数据/健康参照模型','results/数据/抑郁症微调模型',
           'results/表格/健康参照模型','results/表格/抑郁症微调模型','checkpoints','configs']
    for name in roots:
        root=ROOT/name
        for path in ([root] if root.is_file() else sorted(root.rglob('*'))):
            if path.is_file():sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
    protocol=dict(scope='Full internal held-out test; no train/val analysis subjects',
        checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),
        subjects={g:test_ids(g) for g in ['hc','mdd']},
        split_hashes={g:digest(ROOT/'data'/SPLITS[g]) for g in ['hc','mdd']},
        K=64,H=32,calibration_length=32,seed=2024,
        probe=dict(amplitude=.1,last_tr=8,signs=[1,-1],history_block=8,repeats=3,sham_threshold=1e-6),
        inference='eval/no_grad/frozen HC reference; H32 extrapolation',
        statistics='Test-only descriptive comparisons. No archived mixed-split p/q reused.',
        ridge_baseline=dict(alpha=1.,fit='original train only',evaluation='full fixed test only',selection='none; fixed before test scores'),
        matched_controls='Same-site, same sex, age difference<=5; site minimum10/group; HC test site-stratified split-half negative control',
        output=dict(pdf=str(ns.figure_dir.resolve().relative_to(ROOT)),
                    csv=str(ns.table_dir.resolve().relative_to(ROOT)),
                    arrays_metadata=str(ns.data_dir.resolve().relative_to(ROOT))))
    write_json(ns.data_dir/'protocol.json',protocol);write_json(ns.data_dir/'source_hashes.json',sources)
    print('Prepared full test:',{g:len(test_ids(g)) for g in ['hc','mdd']})


if __name__=='__main__':main()
