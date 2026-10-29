"""Describe current learned graphs at the first K64 test anchor per subject."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from inference_twin import TwinSession, load_raw_series
from analysis.h18_data import calibrate_raw
from models.common import prepare_sc_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/H1-8分析_20261028'
NETWORKS = ['VN', 'SMN', 'DAN', 'SN', 'FN', 'DMN', 'LN']


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--limit', type=int, default=0, help='仅供单独临时目录冒烟，不是完整 test')
    ns = ap.parse_args()
    table_dir, figure_dir = ns.out_dir / '03_可解释性', ns.out_dir / '04_图件'
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_excel(ROOT/'data/AAL116.xlsx')
    networks = atlas['对应网络'].to_numpy()
    if len(atlas) != 116 or set(networks) != set(NETWORKS):
        raise ValueError('AAL116 网络表不匹配')
    order = np.concatenate([np.flatnonzero(networks == n) for n in NETWORKS])
    groups = [('HC','hc','checkpoints/neurotwin_pretrain/base_best.pt','subject_split_pretrain.json','Rest-meta-MDD-V1V2-Merged-HC.xlsx'),
              ('MDD','mdd','checkpoints/neurotwin_finetune/finetuned_best.pt','subject_split_finetune.json','Rest-meta-MDD-V1V2-Merged-MDD.xlsx')]
    graphs, subjects, blocks, subject_blocks, metadata = {}, [], [], [], {}
    network_pairs=[]
    for row in NETWORKS:
        for col in NETWORKS:
            ix,iy=np.flatnonzero(networks==row),np.flatnonzero(networks==col)
            rr,cc=np.meshgrid(ix,iy,indexing='ij'); keep=rr!=cc
            network_pairs.append((row,col,rr[keep],cc[keep]))
    for label, group, checkpoint, split, condition_file in groups:
        ids = json.loads((ROOT/'data'/split).read_text())['test']
        ids = ids[:ns.limit] if ns.limit else ids
        session = TwinSession(str(ROOT/checkpoint), device=ns.device, condition_file=condition_file)
        sums = {k: np.zeros((116, 116), dtype=np.float64) for k in ('prior','A_func','A_eff')}
        per_blocks = {k:[] for k in sums}
        for sid in ids:
            raw = load_raw_series(ROOT/f'data/Raw/{group.upper()}/ROISignals_{sid}.mat', 116)
            series = calibrate_raw(raw, session.calibration_length)
            x = torch.from_numpy(series[:64].T.copy())[None,:,None,:].to(session.device)
            cal = x[..., :32]
            _, aux = session.model(x, sc_matrix=None, condition=session.condition_for(sid),
                                   calibration_bold=cal, dynamics_horizon=8)
            info = aux['sc_prior']
            graph_x = session.model.rev_norm(x, 'norm')[0] if session.model.rev_norm is not None else x
            _, prior = prepare_sc_matrix(session.model._history_fc_adj(graph_x[:,:,0,:]), graph_x)
            matrices = {'prior': prior[0], 'A_func': info['A_func'][0], 'A_eff': info['A_eff'][0]}
            for key, value in matrices.items():
                array = value.detach().cpu().numpy()
                if array.shape != (116,116) or not np.isfinite(array).all():
                    raise ValueError(f'{label} {sid} {key} invalid')
                sums[key] += array
                block_vector=[float(array[rr,cc].mean()) for _,_,rr,cc in network_pairs]
                per_blocks[key].append(block_vector)
                subject_blocks.extend({'group':label,'subject':sid,'graph':key,'network_row':row,
                                       'network_col':col,'mean':value}
                                      for (row,col,_,_),value in zip(network_pairs,block_vector))
            lam = info['lambda'].detach().cpu().numpy().reshape(-1)
            subjects.append({'group':label, 'subject':sid, 'lambda_mean':float(lam.mean())})
        graphs[label] = {key:value/len(ids) for key,value in sums.items()}
        metadata[label] = {'n_subjects':len(ids), 'n_samples':len(ids), 'anchor':'first K64; predict H1–8',
                           'checkpoint_sha256':hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()}
        for key, matrix in graphs[label].items():
            variability=np.asarray(per_blocks[key]).std(axis=0)
            for i,(row,col,rr,cc) in enumerate(network_pairs):
                blocks.append({'group':label, 'graph':key, 'network_row':row,'network_col':col,
                               'n_edges':len(rr), 'n_subjects':len(ids), 'n_samples':len(ids),
                               'mean':float(matrix[rr,cc].mean()),'sd':float(variability[i])})
        print(f'{label}: {len(ids)} subjects',flush=True)
    pd.DataFrame(subjects).to_csv(table_dir/'表13_学习图_固定首锚点_逐被试.csv',index=False)
    pd.DataFrame(blocks).to_csv(table_dir/'表14_学习图_固定首锚点_网络块.csv',index=False)
    pd.DataFrame(subject_blocks).to_csv(table_dir/'表30_学习图_固定首锚点_逐被试网络块.csv',index=False)
    np.savez_compressed(table_dir/'学习图_固定首锚点_组均值.npz',
                        **{f'{g}_{k}':v for g,d in graphs.items() for k,v in d.items()})
    (table_dir/'原始_学习图_20261028.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    fig, axes = plt.subplots(2,3,figsize=(9,6),constrained_layout=True)
    for j, key in enumerate(('prior','A_func','A_eff')):
        vmax = max(float(graphs[g][key].max()) for g in ('HC','MDD'))
        for i,g in enumerate(('HC','MDD')):
            ax=axes[i,j]
            im=ax.imshow(graphs[g][key][np.ix_(order,order)],vmin=0,vmax=vmax,cmap='viridis')
            ax.set(title=f'{g}: {key}',xticks=[],yticks=[])
            fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle('First K64 anchor; subject-mean graphs (not anatomical SC)')
    for ext in ('png','pdf'):
        fig.savefig(figure_dir/f'图04_学习图_固定首锚点_被试等权.{ext}',dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
