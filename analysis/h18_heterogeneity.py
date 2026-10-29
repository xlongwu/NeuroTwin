"""Descriptive test-set heterogeneity of HC-reference ROI deviation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/H1-8分析_20261028'


def main():
    rows=[];arrays={}
    for group in ('hc','mdd'):
        with np.load(OUT/f'02_病理分析/评分缓存/z_matrix_{group}_test.npz') as d:
            ids=d['subjects'].astype(str);z=d['z'].astype(float)
        if z.shape!=(len(ids),116) or not np.isfinite(z).all():raise ValueError('无效测试集 z')
        mag=np.abs(z).mean(axis=1);order=np.argsort(mag,kind='stable')
        arrays[group]=z[order]
        rows.extend({'group':group,'subject':sid,'mean_abs_z':float(v)} for sid,v in zip(ids,mag))
    data=pd.DataFrame(rows); data.to_csv(OUT/'02_病理分析/表25_异质性_滚动单步_逐被试.csv',index=False)
    atlas=pd.read_excel(ROOT/'data/AAL116.xlsx');net=atlas['对应网络'].to_numpy()
    networks=['VN','SMN','DAN','SN','FN','DMN','LN']; order=np.concatenate([np.flatnonzero(net==n) for n in networks])
    fig=plt.figure(figsize=(10,5.5),constrained_layout=True); gs=fig.add_gridspec(1,3,width_ratios=[1,1.4,1.4])
    ax=fig.add_subplot(gs[0,0]);rng=np.random.RandomState(2024)
    for i,(group,color) in enumerate([('hc','#df5252'),('mdd','#643c33')]):
        vals=data[data.group==group].mean_abs_z.to_numpy()
        ax.boxplot(vals,positions=[i],widths=.35,showfliers=False,patch_artist=True,
                   boxprops={'facecolor':color,'alpha':.2},medianprops={'color':color},whiskerprops={'color':color},capprops={'color':color})
        ax.scatter(i+rng.uniform(-.15,.15,len(vals)),vals,s=4,alpha=.25,color=color)
    ax.set(xticks=[0,1],xticklabels=['HC test','MDD test'],ylabel='Subject mean |z|',title='Distribution')
    ax.spines[['top','right']].set_visible(False)
    vmax=max(float(np.abs(v).max()) for v in arrays.values())
    axes=[]
    for j,group in enumerate(('hc','mdd'),start=1):
        axis=fig.add_subplot(gs[0,j]);axes.append(axis)
        im=axis.imshow(arrays[group][:,order],aspect='auto',cmap='coolwarm',vmin=-vmax,vmax=vmax,interpolation='nearest')
        mids=[];pos=0
        for network in networks:
            count=int((net==network).sum());mids.append(pos+(count-1)/2)
            if pos:axis.axvline(pos-.5,color='white',lw=.5)
            pos+=count
        axis.set(xticks=mids,xticklabels=networks,yticks=[],xlabel='ROI grouped by network',title=group.upper()+' test')
        axis.tick_params(axis='x',labelrotation=45,labelsize=7)
    fig.colorbar(im,ax=axes,label='z',fraction=.03,pad=.02)
    fig.suptitle('Rolling lag-1 deviation: subjects sorted by mean |z|')
    for ext in ('png','pdf'):fig.savefig(OUT/f'04_图件/图11_异质性_滚动单步_测试集.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
