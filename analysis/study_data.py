"""Frozen subject manifests, native data loading, and explicitly synthetic fixtures."""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_splits(manifest):
    groups = []
    for name in ('train', 'val', 'test'):
        ids = [str(s) for s in manifest[name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate subject IDs in {name}')
        groups.append(set(ids))
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError('Subject leakage across splits')
    return groups


@dataclass
class Subject:
    subject_id: str
    diagnosis: str
    split: str
    bold: np.ndarray
    roi_labels: tuple
    metadata: dict = field(default_factory=dict)
    dataset: object = None
    native_id: str = ''
    synthetic: bool = False


def select_ids(ids, limit, seed):
    ids = sorted(ids)
    if limit and len(ids) > limit:
        ids = sorted(np.random.default_rng(seed).choice(ids, limit, replace=False).tolist())
    return ids


def synthetic_subjects(config):
    rng = np.random.default_rng(config['seed'])
    f, t = 4, 150
    subjects = []
    for group in ('HC', 'MDD'):
        for split, n in [('train', 12), ('val', 4), ('test', 8)]:
            for i in range(n):
                x = rng.normal(size=(t, f))
                for k in range(1, t):
                    x[k] += (.6 if group == 'HC' else .4)*x[k-1]
                x = (x-x[:32].mean(0))/np.sqrt(x[:32].var(0)+1e-6)
                subjects.append(Subject(f'synthetic-{group}-{split}-{i}', group, split, x.astype('float32'),
                                        tuple(f'ROI_{j+1:03d}' for j in range(f)),
                                        {'Age': float(20+i), 'Sex': float(i%2+1), 'education': 12.,
                                         'HAMD': float(12+i) if group == 'MDD' else None,
                                         'HAMA': None, 'site': None, 'FD': None, 'medication': None},
                                        synthetic=True))
    return subjects, {'source': 'synthetic', 'split_validation': 'passed', 'normalization': 'first 32 TR only'}


def native_subjects(config, adapter):
    from utils.condition_schema import condition_stats_from_bundle
    from utils.dataloader import NeuroTwinNextPointDataset, load_condition_table

    class ManifestDataset(NeuroTwinNextPointDataset):
        """Reuse native sample/loading logic, index only the frozen study subset."""
        def __init__(self, selected_ids, **kwargs):
            self.selected_ids = selected_ids
            super().__init__(**kwargs)

        def _build_index(self):
            self._min_series_len = max(self.context_min,self.calibration_length)+self.max_offset
            for sid in sorted(self.selected_ids):
                length = self._probe_series_length(sid)
                if length is None:
                    raise FileNotFoundError(f'MISSING_DATA: selected subject {sid}')
                if length < self._min_series_len:
                    raise ValueError(f'Selected subject {sid} has insufficient TRs')
                self._subject_pos[sid] = len(self.subject_ids)
                self.subject_ids.append(sid)
                self.subject_series_len[sid] = self.subject_n_tr[sid] = length
            self._load_condition_columns()
    root = Path(config['data_root'])
    labels = tuple(f'ROI_{i+1:03d}' for i in range(adapter.args.num_rois))
    atlas = root/'AAL116.xlsx'
    if atlas.exists():
        import pandas as pd
        df = pd.read_excel(atlas)
        if len(df) != len(labels) or 'abbr' not in df or df['abbr'].isna().any() or df['abbr'].duplicated().any():
            raise ValueError('AAL label count/order contract failed')
        # File row order is the repository's AAL116 data convention, not independently verified anatomy.
        labels = tuple(df['abbr'].astype(str))
    subjects, provenance = [], dict(source='native', cohorts={}, roi_order='AAL116.xlsx row order' if atlas.exists() else 'numeric indices')
    manifests = {}
    for group in ('HC', 'MDD'):
        path = root/('subject_split_pretrain.json' if group == 'HC' else 'subject_split_finetune.json')
        manifests[group] = json.loads(path.read_text(encoding='utf-8'))
        validate_splits(manifests[group])
    all_hc = set().union(*validate_splits(manifests['HC']))
    all_mdd = set().union(*validate_splits(manifests['MDD']))
    if all_hc & all_mdd:
        raise ValueError('Ambiguous subject identity across HC/MDD manifests')
    # Enforce the exact manifest attached to the checkpoint; no split regeneration.
    from analysis.current_model_audit import split_digest
    model_group = 'HC' if adapter.args.mode == 'pretrain' else 'MDD'
    expected = (adapter.bundle or {}).get('split_hash')
    if expected and split_digest(manifests[model_group], adapter.args.mode) != expected:
        raise ValueError('Checkpoint split hash differs from frozen study manifest')
    if adapter.args.mode == 'finetune' and 'HC' in config['cohorts']:
        raise ValueError('Finetuned model HC pretraining exposure is not verified; use MDD-only config')
    for group in config['cohorts']:
        mode = 'pretrain' if group == 'HC' else 'finetune'
        a = adapter.args
        source = getattr(a, 'bold_source', 'auto')
        if source != 'raw':
            raise ValueError('Prospective study requires raw prefix calibration; whole-scan normalized sources leak future information')
        file = f'Rest-meta-MDD-V1V2-Merged-{group}.xlsx'
        chosen = {split:select_ids(manifests[group][split],config['max_subjects'][split],config['seed'])
                  for split in ('train','val','test')}
        ds = ManifestDataset(selected_ids=set().union(*map(set,chosen.values())),data_root=str(root),mode=mode, context_min=a.context_min,
                context_max=a.context_max, offsets=[1], bold_source=source,
                num_rois=a.num_rois, graph_source=a.graph_source,
                calibration_length=a.calibration_length,
                condition_group=a.condition_group, condition_file=file,
                clinical_file=file, subject_cache_size=16)
        ds.attach_condition_stats(condition_stats_from_bundle(adapter.bundle, a.condition_group))
        clinical = load_condition_table(root, file, 'hc3' if group == 'HC' else 'mdd8')
        m = manifests[group]
        provenance['cohorts'][group] = dict(split_sha256=sha256(root/('subject_split_pretrain.json' if group == 'HC' else 'subject_split_finetune.json')),
            merged_sha256=clinical.merged_sha256, counts={k:len(m[k]) for k in ('train','val','test')},
            selected={}, missing_fields=['FD', 'verified_site'], bold_sha256={})
        for split in ('train', 'val', 'test'):
            ids = chosen[split]
            provenance['cohorts'][group]['selected'][split] = ids
            missing = set(ids)-set(ds.subject_ids)
            if missing:
                raise FileNotFoundError(f'{len(missing)} selected {group}/{split} subjects unavailable')
            values, mask = clinical.align(ids)
            for i, sid in enumerate(ids):
                meta = {s.name: float(values[i,j]) if mask[i,j] else None for j,s in enumerate(clinical.specs)}
                row = clinical.row_of_id.get(sid)
                for key in ('site', 'Dataset'):
                    v = clinical.aux.get(key)
                    meta[key] = str(v[row]) if v is not None and row is not None and str(v[row]) not in ('nan','None','') else None
                meta['FD'] = None
                for key in ('HAMD', 'HAMA', 'medication'):
                    meta.setdefault(key, None)
                x = ds.get_series(sid).T.copy()
                if not np.isfinite(x).all():
                    raise ValueError(f'Nonfinite BOLD: {group}/{sid}')
                provenance['cohorts'][group]['bold_sha256'][sid] = sha256(root/'Raw'/group/f'ROISignals_{sid}.mat')
                subjects.append(Subject(sid, group, split, x, labels, meta, ds, sid))
    provenance['split_validation'] = 'passed'
    return subjects, provenance


def window_anchors(subject, context, horizon, calibration, count):
    lo = max(context, calibration)-1
    hi = len(subject.bold)-horizon-1
    if hi < lo:
        return []
    return np.unique(np.linspace(lo, hi, min(count, hi-lo+1), dtype=int)).tolist()
