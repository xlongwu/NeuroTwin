"""Shared static paper styling adapted from the inspected gallery references."""
from pathlib import Path
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.paper_figs.common import half_violin, HC, HC_PALE, MDD_PALE, DIV_CMAP
from analysis.visualize_roi_importance import NETWORK_COLORS, NETWORK_NAMES

ROOT=Path(__file__).resolve().parents[3]
RUN1=ROOT/'results/pathology_analysis/run1'
MDD='#D55E00'  # plan's MDD orange; gallery's pale coral used for density fill
NEG='#8e8cc0'
INK='#252533'
NETWORK_ORDER=['VN','SMN','DAN','SN','FN','DMN','LN']


def apply_style() -> None:
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.titlesize':10,'axes.labelsize':9,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.7,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':300})


def json_read(path: Path) -> dict:
    return json.loads(path.read_text())


def labels() -> pd.DataFrame:
    table=pd.read_excel(ROOT/'data/AAL116.xlsx')
    # The supplied spreadsheet duplicates Frontal_Mid_Orb at indices 25/26;
    # cached AAL SPM12 and run1 use the canonical Frontal_Med_Orb labels.
    for i,side in [(24,'L'),(25,'R')]:
        table.loc[i,'micro命名']='Frontal_Med_Orb_'+side
    return table


def partition() -> tuple:
    table=labels()
    net=table['对应网络'].to_numpy()
    order=np.concatenate([np.flatnonzero(net==n) for n in NETWORK_ORDER])
    return net,order


def strip(ax, values, label: str, limit: float, colorbar=False):
    net,order=partition()
    im=ax.imshow(np.asarray(values)[order][None,:],aspect='auto',cmap=DIV_CMAP,vmin=-limit,vmax=limit)
    mids=[]; offset=0
    for n in NETWORK_ORDER:
        count=(net==n).sum(); mids.append(offset+(count-1)/2)
        if offset: ax.axvline(offset-.5,color='white',lw=1)
        offset+=count
    ax.set_xticks(mids,NETWORK_ORDER,fontsize=7,rotation=45,ha='right')
    ax.set_yticks([]); ax.set_title(label,fontsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    if colorbar: plt.colorbar(im,ax=ax,orientation='horizontal',fraction=.35,pad=.45,label='Raw mean difference in z')
    return im


def rain(ax, values, names, colors=None, ylabel=''):
    colors=colors or [HC,MDD,NEG]
    rng=np.random.RandomState(2024)
    for j,(v,c) in enumerate(zip(values,colors)):
        v=np.asarray(v);v=v[np.isfinite(v)]
        ax.scatter(j-rng.uniform(.04,.22,len(v)),v,s=4,color=c,alpha=.18,rasterized=True,lw=0)
        half_violin(ax,v,j,c,pale=c,width=.3)
        ax.scatter([j+.02],[np.median(v)],s=35,color='white',edgecolor=INK,zorder=6)
    ax.set_xticks(range(len(names)),names);ax.set_ylabel(ylabel)


def save(fig, out: Path, name: str, data, caption: str) -> None:
    if any((out/f'{name}.{ext}').exists() for ext in ('png','pdf','json','csv')):
        raise FileExistsError(name)
    fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)
    if isinstance(data,pd.DataFrame): data.to_csv(out/f'{name}.csv',index=False)
    else: (out/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    (out/f'{name}.caption.md').write_text(caption+'\n',encoding='utf-8')
    # Self-contained launcher: same-name source invokes the single requested figure.
    (out/f'{name}.py').write_text('"""Run from the repository root; choose a fresh output directory."""\nfrom analysis.paper_figs.run_20261026.build import main\nif __name__ == "__main__":\n    main(default_figure='+repr(name)+')\n')
    print(f'Saved {name}',flush=True)
