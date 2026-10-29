"""Summarize complete fixed-donor lag-1 ROI importance results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/H1-8分析_20261028'
CACHE=OUT/'03_可解释性/重要性缓存'
NETWORKS=['VN','SMN','DAN','SN','FN','DMN','LN']


def main():
    atlas=pd.read_excel(ROOT/'data/AAL116.xlsx')
    all_rows, summary=[],[]
    for group,split in [('hc','subject_split_pretrain.json'),('mdd','subject_split_finetune.json')]:
        ids=json.loads((ROOT/'data'/split).read_text())['test']
        protocol=json.loads((CACHE/group/'数据清单.json').read_text())
        if protocol['subjects']!=ids:raise ValueError(f'{group}: 不完整测试集')
        saved_donor=None
        for roi in range(116):
            with np.load(CACHE/group/f'roi_{roi+1:03d}.npz') as d:
                subjects=d['subjects'].astype(str);donor=d['donor_index'];values=d['delta_mae']; batches=d['batch_means']
                if list(subjects)!=ids or len(values)!=len(ids) or not np.isfinite(values).all():raise ValueError('ROI 缓存不完整')
                if np.any(donor==np.arange(len(ids))):raise ValueError('存在自身供体')
                if saved_donor is None:saved_donor=donor.copy()
                elif not np.array_equal(donor,saved_donor):raise ValueError('ROI 间供体未固定')
                all_rows.extend({'group':group,'subject':sid,'roi':roi+1,'delta_mae':float(value)} for sid,value in zip(ids,values))
                summary.append({'group':group,'roi':roi+1,'label':atlas.iloc[roi]['micro命名'],
                                'network':atlas.iloc[roi]['对应网络'],'n_subjects':len(ids),'n_batches':len(batches),
                                'delta_mae_mean':float(values.mean()),'subject_sd':float(values.std()),
                                'batch_mean_sd':float(batches.std(ddof=1)),'batch_mean_se':float(batches.std(ddof=1)/np.sqrt(len(batches)))})
    table=OUT/'03_可解释性';figures=OUT/'04_图件'
    data=pd.DataFrame(summary)
    data.loc[data.roi==25,'label']='Frontal_Med_Orb_L';data.loc[data.roi==26,'label']='Frontal_Med_Orb_R'
    data.to_csv(table/'表20_ROI重要性_固定首锚点_被试等权.csv',index=False)
    pd.DataFrame(all_rows).to_csv(table/'表21_ROI重要性_固定首锚点_逐被试.csv',index=False)
    ordered=pd.concat([data[data.network==n] for n in NETWORKS],ignore_index=True)
    values=np.stack([ordered[ordered.group==g].delta_mae_mean.to_numpy() for g in ('hc','mdd')])
    # Both groups use the same atlas ordering within each network.
    maximum=float(np.abs(values).max())
    fig,ax=plt.subplots(figsize=(11,2.6),constrained_layout=True)
    im=ax.imshow(values,aspect='auto',cmap='coolwarm',vmin=-maximum,vmax=maximum)
    mids=[];end=0
    for n in NETWORKS:
        count=int((atlas['对应网络']==n).sum());mids.append(end+(count-1)/2)
        if end:ax.axvline(end-.5,color='white',lw=.7)
        end+=count
    ax.set(xticks=mids,xticklabels=NETWORKS,yticks=[0,1],yticklabels=['HC','MDD'],
           title='Fixed-donor K64 replacement: subject-mean lag-1 MAE change')
    fig.colorbar(im,ax=ax,label='Δ MAE',fraction=.03,pad=.02)
    for ext in ('png','pdf'):fig.savefig(figures/f'图10_ROI重要性_固定首锚点_被试等权.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
