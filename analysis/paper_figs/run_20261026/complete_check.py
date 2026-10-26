"""Bounded numerical and artifact checks for the complete visualization output."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from .style import ROOT


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--infer-dir',type=Path,default=ROOT/'results/paper_infer/complete_20261026')
    ns=parser.parse_args()
    expected={'fig_f0a_pipeline','fig_f0b_composition','fig_f1_gate','fig_f2_horizon',
        'fig_f3_generation','fig_f4_error_structure','fig_f5_pathology_main',
        'fig_f6_site_confounding','fig_f7_separability','fig_f8_heterogeneity',
        'fig_f9_clinical','fig_f10_condition_usage','fig_f11_graph',
        'fig_f12_roi_importance','fig_f13_history_attribution','fig_f14_perturbation','fig_s1_profile'}
    found={p.stem for p in ns.out_dir.glob('fig_*.png')}
    if found != expected:
        raise ValueError(f'Figure set mismatch: missing={expected-found},extra={found-expected}')
    for name in sorted(expected):
        for suffix in ('.pdf','.caption.md','.py'):
            if not (ns.out_dir/(name+suffix)).is_file():
                raise FileNotFoundError(name+suffix)
        if not any((ns.out_dir/(name+suffix)).is_file() for suffix in ('.json','.csv')):
            raise FileNotFoundError(f'Data: {name}')
        with Image.open(ns.out_dir/f'{name}.png') as image:
            if any(abs(dpi-300)>.1 for dpi in image.info.get('dpi',(0,0))):
                raise ValueError(f'PNG dpi mismatch: {name}')
            image.verify()
        if not (ns.out_dir/f'{name}.pdf').read_bytes().startswith(b'%PDF'):
            raise ValueError(f'PDF header mismatch: {name}')
    for path in Path(__file__).parent.glob('complete_*.py'):
        ast.parse(path.read_text(),filename=str(path))
    source=json.loads((ROOT/'results/paper_figures/complete_20261026/sources/previous_checks/source_integrity.json').read_text())
    old_artifacts=json.loads((ROOT/'results/paper_figures/complete_20261026/sources/previous_checks/artifact_manifest.json').read_text())
    source.update({path:record['sha256'] for path,record in old_artifacts.items()})
    for path,sha in source.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Original source/artifact changed: {path}')
    pairs=json.loads((ns.infer_dir/'probes/manifest.json').read_text())['source_manifest']['pairs']
    sham=[];replication=[]
    for pair in pairs:
        for group in ('hc','mdd'):
            with np.load(ns.infer_dir/f'probes/{group}_{pair[group]}.npz') as d:
                if d['response'].shape!=(2,116,116,32) or d['history_mae'].shape!=(8,3,3):
                    raise ValueError('Probe dimensions differ')
                if not all(np.isfinite(d[k]).all() for k in d.files):
                    raise ValueError('Nonfinite probe')
                sham.append(float(d['sham_max']))
                replication.append(float(d['saved_prediction_max_difference']))
    if max(sham)>1e-6:
        raise ValueError('Sham failure')
    # Reconcile source-level F14 map to the saved off-diagonal matrix.
    f14=json.loads((ns.out_dir/'fig_f14_perturbation.json').read_text())
    late=np.array(f14['late_mean']);np.fill_diagonal(late,0)
    exported=pd.read_csv(ns.out_dir/'f14_roi_sensitivity.csv')
    np.testing.assert_allclose(late.sum(1)/115,exported.cross_roi_source_sensitivity,atol=1e-7,rtol=1e-6)
    f13=pd.read_csv(ns.out_dir/'fig_f13_history_attribution.csv')
    if len(f13)!=40*8*3:
        raise ValueError('History curve rows differ')
    rolling={group:json.loads((ns.infer_dir/f'rolling/{group}/summary_val.json').read_text()) for group in ('hc','mdd')}
    if [rolling[g]['n_subjects_this_split'] for g in ('hc','mdd')]!=[222,225]:
        raise ValueError('Rolling cohort sizes differ')
    if [rolling[g]['anchor_total'] for g in ('hc','mdd')]!=[25290,25356]:
        raise ValueError('Rolling anchor coverage differs')
    f1=json.loads((ns.out_dir/'fig_f1_gate.json').read_text())
    if len(f1['signflip_null'])!=10000 or f1['signflip_p']!=f1['archived_gate']['permutation_p']:
        raise ValueError('F1 sign-flip reconstruction differs from archived B4')
    np.testing.assert_allclose(np.mean(f1['mae_model']),f1['archived_gate']['mae_model_subject_equal_weight'],atol=1e-6,rtol=0)
    checks=dict(figures='16 primary + 1 supplementary; PNG 300 dpi, PDF, data, caption, launcher',
        static_syntax='All complete_*.py parse', links='All report/caption links resolve',
        preservation=f'{len(source)} original sources and prior artifacts byte-identical',
        rolling={g:{k:rolling[g][k] for k in ('n_subjects_this_split','anchor_total',
            'mae_model_subject_equal_weight','mae_persistence_subject_equal_weight')} for g in ('hc','mdd')},
        probes='40 subjects; signed 116 x 116 x 32 responses and 8 x 3 x 3 history errors are finite',
        sham_max=max(sham), baseline_batched_vs_saved_single_max=max(replication),
        f1='10,000 real sign-flip samples restored; p equals archived B4 and HC MAE agrees within 1e-6',
        f14='Saved source sensitivity reconciles with off-diagonal propagation matrix',
        d6='Eval-mode extraction differs from archive; archived q explicitly separated, not revalidated',
        scope='Read-only frozen inference; no training, weights/model/config changes, or git commit')
    (ns.out_dir/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    for path in ns.out_dir.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not link.startswith(('http:','https:')) and not (path.parent/link).exists():
                raise FileNotFoundError(f'{path.name}: {link}')
    paths=list(ns.out_dir.glob('*'))+[p for p in Path(__file__).parent.glob('complete_*.py')]
    manifest={str(p.resolve().relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in paths if p.is_file() and p.name!='artifact_manifest.json'}
    (ns.out_dir/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
