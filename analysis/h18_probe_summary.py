"""Aggregate complete H1–8 history and perturbation caches subject equally."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/H1-8分析_20261028'
CACHE=OUT/'03_可解释性/扰动缓存'
COLORS={'hc':'#df5252','mdd':'#643c33'}


def main():
    history, source, curves, curves_subject, matrices, gates = [], [], [], [], {}, []
    for group, split in [('hc','subject_split_pretrain.json'),('mdd','subject_split_finetune.json')]:
        ids=json.loads((ROOT/'data'/split).read_text())['test']
        protocol=json.loads((CACHE/group/'数据清单.json').read_text())
        if ids != protocol['subjects'] or protocol['horizon'] != 8 or protocol['amplitude'] != .1:
            raise ValueError(f'{group}: 扰动清单不完整或协议不一致')
        matrix_sum=np.zeros((116,116)); time_sum=np.zeros(8)
        for sid in ids:
            with np.load(CACHE/group/f'{sid}.npz') as d:
                delta=d['history_mae']-d['baseline_mae'][None,None,:]
                response=d['response']
                sham=float(d['sham_max'])
                if response.shape!=(2,116,116,8) or not np.isfinite(response).all() or sham > 1e-6:
                    raise ValueError(f'{group}/{sid}: 扰动或 sham 检查失败')
                gates.append({'group':group,'subject':sid,'sham_max':sham})
                for block in range(8):
                    history.append({'group':group,'subject':sid,'block':block+1,
                                    'delta_mae_h1_8':float(delta[block,:,0].mean()),
                                    'delta_mae_h8':float(delta[block,:,1].mean())})
                abs_response=np.abs(response).mean(axis=0)
                matrix_sum += abs_response.mean(axis=-1)
                subject_time=abs_response.mean(axis=(0,1))
                time_sum += subject_time
                curves_subject.extend({'group':group,'subject':sid,'horizon':i+1,'mean_abs_change':float(v)}
                                      for i,v in enumerate(subject_time))
                strength=abs_response.mean(axis=(1,2))
                source.extend({'group':group,'subject':sid,'roi':i+1,'mean_abs_change':float(v)} for i,v in enumerate(strength))
        matrices[group]=matrix_sum/len(ids)
        subject_values=pd.DataFrame(curves_subject)
        for i,v in enumerate(time_sum/len(ids)):
            x=subject_values[(subject_values.group==group)&(subject_values.horizon==i+1)].mean_abs_change.to_numpy()
            curves.append({'group':group,'horizon':i+1,'mean_abs_change':float(v), 'sd':float(x.std()), 'n_subjects':len(ids)})
    table_dir=OUT/'03_可解释性';fig_dir=OUT/'04_图件'
    hist=pd.DataFrame(history);src=pd.DataFrame(source);time=pd.DataFrame(curves)
    hist.to_csv(table_dir/'表15_历史归因_H1-8_逐被试.csv',index=False)
    src.to_csv(table_dir/'表16_输入扰动_H1-8_逐被试ROI.csv',index=False)
    time.to_csv(table_dir/'表17_输入扰动_H1-8_逐时距.csv',index=False)
    pd.DataFrame(curves_subject).to_csv(table_dir/'表29_输入扰动_H1-8_逐被试时距.csv',index=False)
    pd.DataFrame(gates).to_csv(table_dir/'表18_输入扰动_H1-8_sham.csv',index=False)
    np.savez_compressed(table_dir/'扰动响应_H1-8_组均值.npz',**matrices)
    rows=[]
    for (group,block),d in hist.groupby(['group','block']):
        for metric in ('delta_mae_h1_8','delta_mae_h8'):
            v=d[metric].to_numpy()
            rows.append({'group':group,'block':block,'metric':metric,'n_subjects':len(v),
                         'mean':v.mean(),'sd':v.std(),'q25':np.quantile(v,.25),'q75':np.quantile(v,.75)})
    summary=pd.DataFrame(rows);summary.to_csv(table_dir/'表19_历史归因_H1-8_被试等权.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,('delta_mae_h1_8','delta_mae_h8'),('H1–8 mean error','H8 error')):
        for group in ('hc','mdd'):
            d=summary[(summary.group==group)&(summary.metric==metric)].sort_values('block')
            x=d.block.to_numpy();ax.plot(x,d['mean'],marker='o',color=COLORS[group],label=group.upper())
            ax.fill_between(x,d.q25.to_numpy(),d.q75.to_numpy(),color=COLORS[group],alpha=.15)
        ax.axhline(0,color='.6',lw=.8);ax.set(xticks=range(1,9),xlabel='History block (old → recent)',ylabel='MAE change',title=title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False);fig.suptitle('First anchor: within-block shuffle; subject mean and IQR')
    for ext in ('png','pdf'):fig.savefig(fig_dir/f'图07_历史归因_H1-8_被试等权.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,5),constrained_layout=True)
    heatmaps={g:hist[hist.group==g].pivot(index='subject',columns='block',values='delta_mae_h1_8').to_numpy()
              for g in ('hc','mdd')}
    maximum=max(float(np.abs(value).max()) for value in heatmaps.values())
    for ax,group in zip(axes,('hc','mdd')):
        im=ax.imshow(heatmaps[group],aspect='auto',cmap='coolwarm',vmin=-maximum,vmax=maximum,interpolation='nearest')
        ax.set(xticks=np.arange(8),xticklabels=np.arange(1,9),xlabel='History block (old → recent)',ylabel='Subject (sorted by ID)',title=group.upper())
    fig.colorbar(im,ax=axes,label='H1–8 MAE change',fraction=.035,pad=.02)
    fig.suptitle('Within-block shuffle: each subject averaged over 3 permutations')
    for ext in ('png','pdf'):fig.savefig(fig_dir/f'图09_历史归因_H1-8_逐被试.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
    shown={g:v.copy() for g,v in matrices.items()}
    for value in shown.values():np.fill_diagonal(value,np.nan)
    vmax=max(float(np.nanmax(v)) for v in shown.values())
    for ax,group in zip(axes,('hc','mdd')):
        im=ax.imshow(shown[group],vmin=0,vmax=vmax,cmap='magma',aspect='equal')
        ticks=[0,19,39,59,79,99,115];labels=[1,20,40,60,80,100,116]
        ax.set(title=group.upper(),xlabel='Target ROI',ylabel='Source ROI',xticks=ticks,xticklabels=labels,yticks=ticks,yticklabels=labels)
    fig.colorbar(im,ax=axes[:2],label='Mean |Δ prediction|',fraction=.04)
    for group in ('hc','mdd'):
        d=time[time.group==group];axes[2].plot(d.horizon,d.mean_abs_change,marker='o',label=group.upper(),color=COLORS[group])
    axes[2].set(xticks=range(1,9),xlabel='Horizon (TR)',ylabel='Mean |Δ prediction|',title='All ROI pairs (incl. self)')
    axes[2].legend(frameon=False);axes[2].spines[['top','right']].set_visible(False)
    fig.suptitle('±0.1 over last 8 TR; first anchor; matrices exclude self response')
    for ext in ('png','pdf'):fig.savefig(fig_dir/f'图08_输入扰动_H1-8_被试等权.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
