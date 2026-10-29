"""Exploratory H1–8 FC identity, ACF and within-scan profile stability."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from analysis.h18_probes import GROUPS, ROOT, OUT
from analysis.metrics import autocorrelation
from inference_twin import TwinSession


def edges(series):
    x=np.asarray(series,dtype=np.float64);x=x-x.mean(axis=0)
    norm=np.sqrt((x*x).sum(0));den=norm[:,None]*norm[None,:]
    fc=np.full((116,116),np.nan);np.divide(x.T@x,den,out=fc,where=den>0)
    return fc[np.triu_indices(116,1)]


def similarity(left,right):
    left,right=np.stack(left),np.stack(right)
    if np.isfinite(left).all() and np.isfinite(right).all():
        a=left-left.mean(axis=1,keepdims=True);b=right-right.mean(axis=1,keepdims=True)
        denom=np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:]
        result=np.full(denom.shape,np.nan)
        np.divide(a@b.T,denom,out=result,where=denom>0)
        return result
    result=np.full((len(left),len(right)),np.nan)
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            good=np.isfinite(a)&np.isfinite(b)
            if good.sum()<3:continue
            x,y=a[good]-a[good].mean(),b[good]-b[good].mean()
            den=np.linalg.norm(x)*np.linalg.norm(y)
            if den>0:result[i,j]=x@y/den
    return result


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch-size',type=int,default=32)
    ns=ap.parse_args();out=ROOT/'results/H1-8分析_20261028';table=out/'03_可解释性';figures=out/'04_图件'
    rows=[];profiles=[];meta={};pcs={}
    for group in ('hc','mdd'):
        ckpt,split,condition=GROUPS[group];ids=json.loads((ROOT/'data'/split).read_text())['test']
        history=[];pred=[];true=[];first=[]
        for sid in ids:
            with np.load(OUT/group/f'{sid}.npz') as d:
                history.append(d['history'].copy());pred.append(d['baseline'].T.copy());true.append(d['true'].T.copy());first.append(d['profile'].copy())
        history=np.stack(history);pred=np.stack(pred);true=np.stack(true);first=np.stack(first)
        if first.shape!=(len(ids),32) or pred.shape!=(len(ids),8,116):raise ValueError('profile 或 H1–8 维度错误')
        session=TwinSession(str(ROOT/ckpt),device=ns.device,condition_file=condition);second=[]
        for start in range(0,len(ids),ns.batch_size):
            x=torch.from_numpy(history[start:start+ns.batch_size,:,None,32:64]).to(session.device)
            second.append(session.model.calibrate_profile(x).cpu().numpy())
        second=np.concatenate(second)
        pred_edges=np.stack([edges(a) for a in pred])
        true_edges=np.stack([edges(a) for a in true])
        observed_edges=np.stack([edges(a.T[-8:]) for a in history])
        gen=similarity(pred_edges,true_edges)
        observed=similarity(observed_edges,true_edges)
        distances=np.linalg.norm(first[:,None,:]-second[None,:,:],axis=2)
        pca=PCA(n_components=2);pc=pca.fit_transform(first);pcs[group]=pc
        for i,sid in enumerate(ids):
            delta=np.abs(autocorrelation(pred[i])-autocorrelation(true[i]));valid=np.isfinite(delta)
            own=float(distances[i,i]);other=float(np.delete(distances[i],i).mean())
            def identify(matrix):
                valid_row=np.isfinite(matrix[i]);return int(np.nanargmax(matrix[i])==i) if valid_row.any() else None
            rows.append({'group':group,'subject':sid,'acf_mae_lag1_5':float(delta[valid].mean()) if valid.any() else None,
                         'acf_valid_count':int(valid.sum()),'acf_total_count':int(valid.size),
                         'predicted_fc_identified':identify(gen),'observed_fc_identified':identify(observed),
                         'predicted_fc_invalid_edges':int((~np.isfinite(pred_edges[i])).sum()),
                         'observed_fc_invalid_edges':int((~np.isfinite(observed_edges[i])).sum()),
                         'true_fc_invalid_edges':int((~np.isfinite(true_edges[i])).sum()),
                         'predicted_fc_valid_candidates':int(np.isfinite(gen[i]).sum()),
                         'observed_fc_valid_candidates':int(np.isfinite(observed[i]).sum()),
                         'profile_half_identified':int(np.argmin(distances[i])==i),
                         'profile_half_own_distance':own,'profile_half_other_distance':other})
            profiles.append({'group':group,'subject':sid,'pc1':float(pc[i,0]),'pc2':float(pc[i,1]),
                             **{f'profile_{j+1:02d}':float(value) for j,value in enumerate(first[i])}})
        meta[group]={'n_subjects':len(ids),'chance_identification':1/len(ids),
                     'pca_variance_ratio':pca.explained_variance_ratio_.tolist(),
                     'identity':'within group, first-anchor predicted H1–8 FC vs observed H1–8 FC; 6670 edges',
                     'profile':'first 32 vs second 32 TR within same scan; not longitudinal repeatability'}
    pd.DataFrame(rows).to_csv(table/'表26_短序列一致性_H1-8_逐被试.csv',index=False)
    pd.DataFrame(profiles).to_csv(table/'表27_个体profile_固定首锚点_逐被试.csv',index=False)
    (table/'原始_短序列一致性_20261028.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),constrained_layout=True)
    for ax,(group,color) in zip(axes,[('hc','#df5252'),('mdd','#643c33')]):
        ax.scatter(pcs[group][:,0],pcs[group][:,1],s=12,alpha=.45,color=color)
        ax.set(xlabel='PC1',ylabel='PC2',title=group.upper()+' test; separate PCA')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Individual 32D profile: within-group descriptive PCA')
    for ext in ('png','pdf'):fig.savefig(figures/f'图12_个体profile_固定首锚点_PCA.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
