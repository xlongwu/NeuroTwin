"""Six-descriptor HC-reference baseline for the prespecified matched cohort."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.pathology_tests import collect_descriptors, site_blocked_paired_permutation, site_blocked_unpaired_permutation, bh_fdr
from analysis.h18_group_tests import bootstrap
from utils.site_subset import parse_site

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/H1-8分析_20261028/02_病理分析'


def main():
    if not json.loads((OUT/'原始_健康负控_20261028.json').read_text())['passed_roi_negative_control']:
        raise RuntimeError('HC 负控未通过，停止疾病基线')
    pairs=json.loads((ROOT/'data/splits/site_matched_subset.json').read_text())['pairs']
    split=json.loads((ROOT/'data/subject_split_pretrain.json').read_text())
    hc=sorted({p['hc'] for p in pairs}|set(split['val'])|set(split['test']))
    mdd=sorted({p['mdd'] for p in pairs})
    scores,meta=collect_descriptors(ROOT/'data',ROOT/'checkpoints/neurotwin_pretrain/base_best.pt',hc,mdd)
    all_ids=hc+mdd
    np.savez_compressed(OUT/'描述量基线_固定首锚点_逐被试.npz',subjects=np.array(all_ids),
                        scores=np.stack([scores[s] for s in all_ids]))
    diff=np.stack([scores[p['mdd']]-scores[p['hc']] for p in pairs])
    p=site_blocked_paired_permutation(diff,[p['site'] for p in pairs],n_perm=10000,seed=2024)
    q=bh_fdr(p);lo,hi=bootstrap(diff)
    pd.DataFrame({'roi':np.arange(1,117),'delta_descriptor_distance':diff.mean(0),'ci_low':lo,'ci_high':hi,
                  'p':p,'q':q,'n_pairs':len(pairs)}).to_csv(OUT/'表22_描述量基线_固定首锚点_匹配ROI.csv',index=False)
    va=np.stack([scores[s] for s in split['val']]);te=np.stack([scores[s] for s in split['test']])
    neg=site_blocked_unpaired_permutation(va,te,[parse_site(s) for s in split['val']],
                                            [parse_site(s) for s in split['test']],n_perm=10000,seed=2024)
    nq=bh_fdr(neg)
    pd.DataFrame({'roi':np.arange(1,117),'p':neg,'q':nq}).to_csv(OUT/'表23_描述量负控_固定首锚点_ROI.csv',index=False)
    model=json.loads((OUT/'原始_组间偏离_20261028.json').read_text())
    increment=bool(model['roi_significant']>int((q<.05).sum()) and model['max_abs_delta_z']>float(np.abs(diff.mean(0)).max()))
    report={'n_pairs':len(pairs),'fit_hc_train':meta['n_train'],'n_perm':10000,'n_boot':10000,'seed':2024,
            'roi_significant':int((q<.05).sum()),'negative_control_roi_significant':int((nq<.05).sum()),
            'max_abs_delta':float(np.abs(diff.mean(0)).max()),'model_t2_increment':increment,
            'baseline': 'six history descriptors; first 64 TR; HC-train reference',
            'limitation':'HC train subjects are scored in-sample for descriptor reference; model HC train z is cross-fitted'}
    (OUT/'原始_描述量基线_20261028.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
