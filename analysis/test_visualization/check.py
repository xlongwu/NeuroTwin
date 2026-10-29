"""Verify test membership, frozen sources, numerical gates and PDF-only delivery."""
from __future__ import annotations
import argparse
import ast
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.metrics import gaussian_crps
from sklearn.metrics import roc_auc_score
from .common import (ROOT,DATA,TABLES,FIGURES,CHECKPOINT,SPLITS,test_ids,check_ids,
                     sample_path,read_json,write_json,digest)


def pdf_pages(path: Path) -> int:
    output=subprocess.run(['pdfinfo',str(path)],check=True,capture_output=True,text=True).stdout
    return int(next(line.split(':',1)[1] for line in output.splitlines() if line.startswith('Pages:')).strip())


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=DATA);ap.add_argument('--table-dir',type=Path,default=TABLES)
    ap.add_argument('--figure-dir',type=Path,default=FIGURES)
    ns=ap.parse_args();protocol=read_json(ns.data_dir/'protocol.json')
    if digest(CHECKPOINT)!=protocol['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    for group,name in SPLITS.items():
        if digest(ROOT/'data'/name)!=protocol['split_hashes'][group]:raise ValueError('Split changed')
        check_ids(protocol['subjects'][group],group)
    for record in read_json(ns.data_dir/'source_hashes.json'):
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError(f'Source changed: {record["path"]}')
    table_counts={}
    for path in ns.table_dir.glob('*.csv'):
        df=pd.read_csv(path)
        if {'group','subject'}<=set(df):
            for group,sub in df.groupby('group'):
                ids=sub.subject.astype(str).unique().tolist()
                if not set(ids)<=set(test_ids(group)):raise ValueError(f'Non-test rows: {path.name}')
            table_counts[path.name]=len(df)
        elif 'subject' in df:
            if not set(df.subject.astype(str))<=set(test_ids('mdd')):
                raise ValueError(f'Non-MDD-test clinical rows: {path.name}')
            table_counts[path.name]=len(df)
    pairs=pd.read_csv(ns.table_dir/'test_pairs.csv')
    check_ids(pairs.hc,'hc',complete=False);check_ids(pairs.mdd,'mdd',complete=False)
    ridge=read_json(ns.data_dir/'fc_ridge_protocol.json')
    if ridge['fit']!='train only' or ridge['hyperparameter_selection']!='None; alpha fixed before test scoring':
        raise ValueError('FC baseline fitting protocol changed')
    for g in ['hc','mdd']:
        split=read_json(ROOT/'data'/SPLITS[g])
        if ridge['train_counts'][g]!=len(split['train']) or ridge['split_hashes'][g]!=protocol['split_hashes'][g]:
            raise ValueError('FC baseline training cohort changed')
    ridge_scores=pd.read_csv(ns.table_dir/'fc_ridge_test_scores.csv')
    for g,s in ridge_scores.groupby('group'):check_ids(s.subject,g)
    np.testing.assert_allclose(roc_auc_score(ridge_scores.label,ridge_scores.fc_ridge_score),ridge['test_auc'],atol=1e-12)
    diagnostics=[];max_crps_difference=0.
    cached=pd.read_csv(ROOT/'results/表格/健康参照模型/自主生成32步/per_subject.csv').set_index(['group','subject'])
    for group in ['hc','mdd']:
        ids=test_ids(group)
        paths=list((ns.data_dir/'自主生成'/group).glob('*.npz'))
        check_ids([p.stem for p in paths],group)
        check_ids([p.stem for p in (ns.data_dir/'扰动'/group).glob('*.npz')],group)
        for sid in ids:
            with np.load(sample_path(group,sid,ns.data_dir)) as d:
                if d['pred'].shape!=(32,116) or d['std'].shape!=(32,116) or d['profile'].shape!=(32,):
                    raise ValueError(f'Base shape changed: {sid}')
                if not all(np.isfinite(d[k]).all() for k in d.files):raise ValueError(f'Nonfinite base: {sid}')
                crps=gaussian_crps(d['pred'],d['std'],d['true'])
                actual=np.array([crps[0].mean(),crps.mean(),crps[16:].mean()])
                reference=cached.loc[(group,sid),['crps_h1','crps_h1_32','crps_h17_32']].to_numpy(float)
                difference=float(np.max(np.abs(actual-reference)));max_crps_difference=max(max_crps_difference,difference)
                np.testing.assert_allclose(actual,reference,atol=2e-5,rtol=1e-5)
            with np.load(ns.data_dir/f'扰动/{group}/{sid}.npz') as d:
                if d['response_early'].shape!=(116,116) or d['response_late'].shape!=(116,116) or d['history_mae'].shape!=(8,3,3):
                    raise ValueError('Probe dimensions changed')
                if not all(np.isfinite(d[k]).all() for k in d.files) or d['sham_max']>1e-6:
                    raise ValueError(f'Invalid probe: {sid}')
                diagnostics.append(dict(group=group,subject=sid,sham_max=float(d['sham_max']),
                    batched_source_difference=float(d['saved_prediction_max_difference'])))
        files=list((ns.data_dir/'重要性'/group).glob('roi_*.npz'))
        if len(files)!=116:raise ValueError('Incomplete ROI importance')
        for path in files:
            with np.load(path) as d:
                check_ids(d['subjects'],group)
                if not np.isfinite(d['delta_mae']).all():raise ValueError('Invalid importance values')
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    registry=read_json(ns.data_dir/'figure_registry.json')
    if len(registry)!=18:raise ValueError('Expected 16 primary + 2 supplemental figures')
    files=list(ns.figure_dir.iterdir())
    if any(not p.is_file() or p.suffix!='.pdf' for p in files):raise ValueError('Non-PDF file in visualization directory')
    expected={r['figure']+'.pdf' for r in registry}|{'全部图表.pdf'}
    if {p.name for p in files}!=expected:raise ValueError('Incomplete PDF set')
    for record in registry:
        if pdf_pages(ns.figure_dir/(record['figure']+'.pdf'))!=record['pages']:raise ValueError('PDF page count differs')
        for source in record['sources']:
            if not (ns.table_dir/source).exists() and not (ns.data_dir/source).exists():raise FileNotFoundError(source)
    total=sum(r['pages'] for r in registry)
    if pdf_pages(ns.figure_dir/'全部图表.pdf')!=total:raise ValueError('Combined PDF page count differs')
    diag=pd.DataFrame(diagnostics)
    summary=dict(cohorts={g:len(test_ids(g)) for g in ['hc','mdd']},
        membership='Every analysis subject belongs to fixed test; no train/val rows',
        split='Internal subject-held-out test; not an external cohort',
        source_preservation='All pre-existing source files/checkpoints unchanged',
        statistics='No archived mixed-split p/q or OOF scores transferred',
        fc_baseline='Fixed alpha=1 fit on original train only; complete independent test scoring; no selection',
        tables=table_counts,n_test_matched_pairs=len(pairs),sham_max=float(diag.sham_max.max()),
        max_batched_source_difference=float(diag.batched_source_difference.max()),
        max_crps_difference_from_existing_test=max_crps_difference,
        figures=dict(pdf_files=len(files),individual_figures=len(registry),combined_pages=total,
                     formats='PDF only; CSV and NPZ/JSON stored separately'),
        syntax='All new Python modules parse',
        scope='NeuroTwin frozen/eval/no_grad; train-only static FC ridge baseline; no NeuroTwin optimization/weight/config changes or git commit')
    write_json(ns.data_dir/'checks.json',summary)
    write_json(ns.data_dir/'artifacts.json',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p))
        for directory in [ns.figure_dir,ns.table_dir] for p in sorted(directory.iterdir()) if p.is_file()])
    print(summary,flush=True)


if __name__=='__main__':main()
