"""Render only PDF files; chart-ready tables and arrays live in other directories."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch

from analysis.paper_figs.run_20261026.style import (apply_style,rain,labels,partition,
    HC,MDD,NEG,INK,DIV_CMAP,NETWORK_ORDER,NETWORK_COLORS)
from analysis.paper_figs.run_20261026.build import atlas_image
from analysis.normative_deviation import paired_signflip_p
from utils.site_subset import parse_site
from .common import ROOT,DATA,TABLES,FIGURES,test_ids,sample_path,write_json,read_json,check_ids

COLORS={'hc':HC,'mdd':MDD}


class Plotter:
    def __init__(self,data: Path,tables: Path,out: Path,book: PdfPages):
        self.data,self.tables,self.out,self.book=data,tables,out,book
        self.registry=[]
        self.atlas=Path.home()/'nilearn_data/aal_SPM12/aal'

    def table(self,name: str) -> pd.DataFrame:
        return pd.read_csv(self.tables/f'{name}.csv')

    def sample(self,group: str,sid: str) -> dict:
        with np.load(sample_path(group,sid,self.data)) as d:return {k:d[k].copy() for k in d.files}

    def emit(self,name: str,figures,caption: str,sources: list[str]) -> None:
        if not isinstance(figures,list):figures=[figures]
        with PdfPages(self.out/f'{name}.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig,bbox_inches='tight');self.book.savefig(fig,bbox_inches='tight');plt.close(fig)
        self.registry.append(dict(figure=name,pages=len(figures),caption_zh=caption,sources=sources,
                                 scope='Fixed test subjects only; no train/val analysis rows'))
        print(f'PDF {name}: {len(figures)} page(s)',flush=True)

    def brain(self,fig,ax,values,title,vmin=None,vmax=None,cmap=DIV_CMAP):
        from nilearn.plotting import plot_glass_brain
        if vmax is None:vmax=max(float(np.abs(values).max()),1e-8)
        if vmin is None:vmin=-vmax
        plot_glass_brain(atlas_image(values,self.atlas),axes=ax,figure=fig,display_mode='lyrz',
                         plot_abs=False,threshold=None,cmap=cmap,vmin=vmin,vmax=vmax,colorbar=False)
        ax.set_title(title,fontsize=10)

    def contrasts(self,fig,axes,values,caption='Raw mean difference'):
        net,order=partition();limit=max(float(np.abs(v).max()) for v in values) or 1
        mids=[];offset=0
        for n in NETWORK_ORDER:
            count=sum(net==n);mids.append(offset+(count-1)/2);offset+=count
        for ax,v,title in zip(axes,values,['Matched test MDD−HC','All test MDD−HC','HC test split-half control']):
            im=ax.imshow(np.asarray(v)[order][None],aspect='auto',cmap=DIV_CMAP,vmin=-limit,vmax=limit)
            ax.set_xticks(mids,NETWORK_ORDER,rotation=45,fontsize=7);ax.set_yticks([]);ax.set_title(title,fontsize=9)
        fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.1,pad=.35,label=caption)

    def f0a(self):
        fig,ax=plt.subplots(figsize=(12,3.6));ax.axis('off')
        boxes=[('Fixed test subjects','HC 222 / MDD 225\nDisjoint from own train and val\nNo test-based model selection'),
          ('Frozen reference','HC checkpoint / hc3 statistics\nFirst 32 TR calibration\nObserved K64 history'),
          ('Evaluation protocols','Rolling lag-1 / existing z\nOne-origin H32 extrapolation\nCondition / ROI / history probes'),
          ('Separate artifacts','CSV tables / NPZ and metadata\nPDF visualization only\nNo mixed-split statistics reused')]
        for j,(title,body) in enumerate(boxes):
            x=.01+j*.25
            ax.add_patch(FancyBboxPatch((x,.25),.22,.55,boxstyle='round,pad=.01',fc='#f3f6f8',ec=HC,transform=ax.transAxes))
            ax.text(x+.11,.70,title,ha='center',weight='bold',transform=ax.transAxes)
            ax.text(x+.11,.48,body,ha='center',va='center',fontsize=8,linespacing=1.8,transform=ax.transAxes)
            if j<3:ax.annotate('',xy=(x+.244,.53),xytext=(x+.222,.53),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
        ax.set_title('Independent internal test evaluation and visualization',weight='bold')
        self.emit('fig_f0a_pipeline',fig,'F0a｜全部分析对象为固定内部 test：HC 222 人、MDD 225 人；各自与 train/val 无交集。沿用冻结 HC 权重和训练条件统计。H32 为当前 H8 辅助训练时程之外的外推；不使用历史 H32 模型、val 或混合子集结果。CSV、数组/元数据与 PDF 分目录保存。',['protocol.json'])

    def f0b(self):
        pairs=self.table('test_pairs');rows=[]
        sites=sorted({parse_site(s) for g in ['hc','mdd'] for s in test_ids(g)})
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),gridspec_kw={'width_ratios':[1,2]})
        for j,g in enumerate(['hc','mdd']):
            n=len(test_ids(g));m=len(pairs)
            axes[0].bar(j,m,color=COLORS[g]);axes[0].bar(j,n-m,bottom=m,color=COLORS[g],alpha=.25)
            axes[0].text(j,n+4,f'{n}',ha='center')
            rows.append(dict(group=g,n_test=n,n_test_matched=m))
            counts=[sum(parse_site(s)==site for s in test_ids(g)) for site in sites]
            axes[1].bar(np.arange(len(sites))+(j-.5)*.38,counts,width=.36,color=COLORS[g],label=g.upper())
        axes[0].set_xticks([0,1],['HC','MDD']);axes[0].set_ylim(0,260);axes[0].set_ylabel('Test subjects')
        axes[0].set_title(f'(a) Dark: test-only matched n={len(pairs)}/group')
        axes[1].set_xticks(range(len(sites)),sites,rotation=90,fontsize=7);axes[1].set_ylabel('Subjects');axes[1].legend();axes[1].set_title('(b) Fixed test subjects by site')
        fig.tight_layout()
        pd.DataFrame(rows).to_csv(self.tables/'cohort_counts.csv',index=False)
        self.emit('fig_f0b_composition',fig,'F0b｜仅展示 HC/MDD 全 test 与重新建立的 test 内匹配子集。匹配沿用同站点、同性别、年龄差不超过 5 岁及每站点两组至少各 10 人的规则；未匹配 test 被试仍保留在完整队列描述和模型评估中。没有画入训练或验证被试。',['cohort_counts.csv','test_pairs.csv'])

    def f1(self):
        fig,axes=plt.subplots(2,3,figsize=(13,8));nullrows=[];summary=[]
        for row,g in enumerate(['hc','mdd']):
            with np.load(ROOT/f'results/数据/健康参照模型/滚动单步/{g}/roi_mae.npz') as d:
                check_ids(d['subjects'],g);mm=d['mae_model'].astype(float).mean(1);pp=d['mae_persistence'].astype(float).mean(1)
            diff=pp-mm;rng=np.random.RandomState(2024)
            null=(rng.choice([-1.,1.],size=(10000,len(mm)))*diff).mean(1)
            p=paired_signflip_p(diff);gain=1-mm.mean()/pp.mean()
            summary.append(dict(group=g,n=len(mm),mae_model=mm.mean(),mae_persistence=pp.mean(),improvement=gain,p=p))
            nullrows.extend(dict(group=g,draw=j,mean_difference=v) for j,v in enumerate(null))
            ax=axes[row,0];ax.bar([0,1],[mm.mean(),pp.mean()],color=[COLORS[g],'#aaa'])
            ax.set_xticks([0,1],['Model','Persistence']);ax.set_ylabel('Subject-equal MAE');ax.set_ylim(0,.8)
            ax.text(.04,.90,f'Gain {gain:.2%}',transform=ax.transAxes);ax.set_title(f'{g.upper()} test n={len(mm)}')
            ax=axes[row,1];ax.hist(null,bins=40,color=COLORS[g],alpha=.7);ax.axvline(diff.mean(),color=INK)
            ax.set_xlabel('Persistence MAE − model MAE');ax.set_ylabel('Sign-flip draws');ax.set_title(f'10,000 sign flips; p={p:.5f}')
            rain(axes[row,2],[mm,pp],['Model','Persistence'],colors=[COLORS[g],'#888'],ylabel='Per-subject MAE')
        fig.suptitle('Rolling lag-1 prediction: full HC and MDD test',weight='bold');fig.tight_layout(rect=(0,0,1,.96))
        pd.DataFrame(summary).to_csv(self.tables/'rolling_guardrail.csv',index=False)
        pd.DataFrame(nullrows).to_csv(self.tables/'rolling_signflip_null.csv',index=False)
        self.emit('fig_f1_gate',fig,'F1｜HC/MDD 分别使用全部 test 的逐 ROI 滚动误差，对 ROI、被试等权汇总；p 值按既有单侧配对符号翻转方法在各 test 组内重算，seed=2024，10000 次。这里检验模型与 persistence 的预测性能差，不检验疾病组间效应。每个锚点重新输入真实历史，不能等同自主生成。',['rolling_guardrail.csv','rolling_signflip_null.csv'])

    def f2(self):
        d=self.table('horizon_by_subject');step=self.table('horizon_by_step')
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for j,g in enumerate(['hc','mdd']):
            sub=d[d.group==g]
            for k,seg in enumerate(['H1_8','H9_16','H17_32','H1_32']):
                v=sub.loc[sub.segment==seg,'mae'].to_numpy();base=sub.loc[sub.segment==seg,'mean_baseline_mae'].mean()
                vp=axes[j,0].violinplot(v,positions=[k],showextrema=False)
                for body in vp['bodies']:body.set_color(COLORS[g]);body.set_alpha(.3)
                axes[j,0].scatter(k,v.mean(),color=COLORS[g],zorder=3)
                axes[j,0].hlines(base,k-.3,k+.3,color=INK,ls='--')
            axes[j,0].set_xticks(range(4),['H1–8','H9–16','H17–32','H1–32']);axes[j,0].set_ylabel('Autonomous ODE MAE')
            axes[j,0].set_title(f'{g.upper()} test n={len(test_ids(g))}; dashed: history-mean baseline')
            s=step[step.group==g].groupby('horizon').mae
            x=np.arange(1,33);axes[j,1].plot(x,s.mean(),color=COLORS[g]);axes[j,1].fill_between(x,s.quantile(.25),s.quantile(.75),color=COLORS[g],alpha=.2)
            axes[j,1].set_xlabel('Forecast horizon (TR)');axes[j,1].set_ylabel('MAE');axes[j,1].set_title('Mean error by horizon; shade: subject IQR')
        fig.suptitle('HC reference autonomous H32 extrapolation on full test',weight='bold');fig.tight_layout(rect=(0,0,1,.96))
        self.emit('fig_f2_horizon',fig,'F2｜统一使用当前 HC 参照权重在 HC/MDD 全 test 上最初 K64→H32 的自主生成，未反馈未来真值。虚线来自同一被试 K64 历史均值基线，按相同时段计算。小提琴为被试分布，点为均值，逐步曲线色带为被试四分位范围。没有引用 val 或存档 H32 权重指标。',['horizon_by_subject.csv','horizon_by_step.csv'])

    def f3(self):
        metrics=self.table('generation_metrics');examples={};pages=[];selected=[]
        fig,axes=plt.subplots(4,2,figsize=(13,10))
        for col,g in enumerate(['hc','mdd']):
            d=metrics[metrics.group==g];sid=d.iloc[np.argmin(np.abs(d.fc_r-d.fc_r.median()))].subject
            s=self.sample(g,sid);examples[g]=s;selected.append(dict(group=g,subject=sid,rule='nearest within-group median FC similarity'))
            for j,roi in enumerate([0,28,30,60]):
                ax=axes[j,col];ax.plot(np.arange(-15,1),s['history'][-16:,roi],color='#aaa',label='Observed history')
                ax.plot(np.arange(1,33),s['true'][:,roi],color=HC,label='Observed future')
                ax.plot(np.arange(1,33),s['pred'][:,roi],color=MDD,ls='--',label='Generated')
                ax.axvspan(.5,8.5,color='#eee');ax.set_title(f'{g.upper()}: {labels()["micro命名"].iloc[roi]}',fontsize=9)
                ax.set_xlabel('TR relative to forecast origin');ax.set_ylabel('Calibrated BOLD')
                if j==0:ax.legend(fontsize=7,ncol=3)
        fig.suptitle('Full-test trajectory examples: deterministic median-similarity selection',weight='bold');fig.tight_layout(rect=(0,0,1,.96));pages.append(fig)
        fig,axes=plt.subplots(2,3,figsize=(12,8));_,order=partition()
        for row,g in enumerate(['hc','mdd']):
            s=examples[g]
            for col,(mat,title,lim) in enumerate([(s['fc_true'],'Observed',1),(s['fc_pred'],'Generated',1),(s['fc_pred']-s['fc_true'],'Generated − observed',2)]):
                ax=axes[row,col];im=ax.imshow(mat[np.ix_(order,order)],cmap=DIV_CMAP,vmin=-lim,vmax=lim)
                ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{g.upper()} {title} FC');fig.colorbar(im,ax=ax,fraction=.045)
        fig.tight_layout();pages.append(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,key,title in zip(axes[:2],['fc_r','psd_kl'],['FC pattern Pearson r','Spectral KL(true || generated)']):
            rain(ax,[metrics.loc[metrics.group==g,key] for g in ['hc','mdd']],['HC test','MDD test'],ylabel=title)
        for g in ['hc','mdd']:
            arrays=[]
            for sid in test_ids(g):
                with np.load(sample_path(g,sid,self.data)) as s:arrays.append((s['psd_true'].mean(1),s['psd_pred'].mean(1),s['frequency']))
            for j,ls,label in [(0,'-','observed'),(1,'--','generated')]:
                axes[2].plot(arrays[0][2],np.stack([a[j] for a in arrays]).mean(0),color=COLORS[g],ls=ls,label=f'{g.upper()} {label}')
        axes[2].set_yscale('log');axes[2].set_xlabel('Frequency (cycles/TR)');axes[2].set_ylabel('Normalized power');axes[2].legend(fontsize=7)
        fig.tight_layout();pages.append(fig)
        pd.DataFrame(selected).to_csv(self.tables/'trajectory_examples.csv',index=False)
        self.emit('fig_f3_generation',pages,'F3｜HC/MDD 各完整 test 队列的自主生成质量。两组分别选 FC 相似度最接近本组中位数的一个示例，非最好个体；固定展示 ROI 1/29/31/61。FC 取未来 32 TR 的上三角相关结构。Welch 使用 nperseg=32，去 DC、逐 ROI 归一化，KL 越小越接近；频率为 cycles/TR，不假定跨站点 TR 秒数相同。仅 32 个未来点，不能据此推断低频临床机制。',['generation_metrics.csv','trajectory_examples.csv'])

    def f4(self):
        roi=self.table('rolling_roi');network=self.table('rolling_network')
        fig=plt.figure(figsize=(14,10));grid=fig.add_gridspec(3,2,height_ratios=[1.1,1,1.15],hspace=.4,wspace=.28)
        axes=np.empty((3,2),dtype=object);subjects=[]
        for col,g in enumerate(['hc','mdd']):
            subgrid=grid[0,col].subgridspec(1,2,width_ratios=[20,1],wspace=.1)
            axes[0,col]=fig.add_subplot(subgrid[0]);cax=fig.add_subplot(subgrid[1])
            axes[1,col]=fig.add_subplot(grid[1,col]);axes[2,col]=fig.add_subplot(grid[2,col])
            d=roi[roi.group==g];self.brain(fig,axes[0,col],d.improvement.to_numpy(),f'{g.upper()} test ROI improvement',vmin=-100,vmax=100)
            fig.colorbar(ScalarMappable(norm=Normalize(-100,100),cmap=DIV_CMAP),cax=cax,label='Improvement (%)')
            n=network[network.group==g];axes[1,col].bar(range(7),[n[n.network==k].improvement.iloc[0] for k in NETWORK_ORDER],color=COLORS[g])
            axes[1,col].set_xticks(range(7),NETWORK_ORDER);axes[1,col].set_ylim(0,100);axes[1,col].set_ylabel('Mean ROI improvement (%)')
            sub=pd.read_csv(ROOT/f'results/表格/健康参照模型/滚动单步/{g}/rolling_metrics.csv');check_ids(sub.subject,g)
            sub['site']=sub.subject.map(parse_site);sub['group']=g;subjects.append(sub)
        sites=sorted(set(pd.concat(subjects).site));limit=[pd.concat(subjects).improvement.min()*100-2,pd.concat(subjects).improvement.max()*100+2]
        for col,(g,sub) in enumerate(zip(['hc','mdd'],subjects)):
            ax=axes[2,col];rng=np.random.RandomState(2024)
            for j,site in enumerate(sites):
                v=sub.loc[sub.site==site,'improvement'].to_numpy()*100
                if not len(v):continue
                ax.scatter(j+rng.uniform(-.15,.15,len(v)),v,color=COLORS[g],s=5,alpha=.4)
                ax.boxplot([v],positions=[j],widths=.5,showfliers=False,medianprops={'color':INK})
            ax.set_xticks(range(len(sites)),sites,rotation=90,fontsize=7);ax.set_ylim(*limit);ax.set_xlim(-1,len(sites));ax.set_ylabel('Subject improvement (%)');ax.set_title(f'{g.upper()} test site QC')
        fig.suptitle('Rolling one-step spatial and site QC: full test only',weight='bold')
        pd.concat(subjects).to_csv(self.tables/'rolling_subjects.csv',index=False)
        self.emit('fig_f4_error_structure',fig,'F4｜复用完整 test 滚动单步的逐 ROI MAE，逐人计算相对 persistence 改善度后对被试等权汇总；HC/MDD 分别成图，共用脑图和站点轴尺度。网络分组仅为 AAL116 表中的展示分组。站点图为预测 QC，不是疾病组间检验。',['rolling_roi.csv','rolling_network.csv','rolling_subjects.csv'])

    def f5(self):
        d=self.table('roi_descriptive_contrasts');pages=[]
        fig,axes=plt.subplots(1,2,figsize=(12,6));net,order=partition()
        axes[0].scatter(range(116),d.matched.to_numpy()[order],c=[NETWORK_COLORS[n] for n in net[order]],s=16)
        sizes=np.array([sum(net==n) for n in NETWORK_ORDER]);axes[0].set_xticks(sizes.cumsum()-sizes/2-.5,NETWORK_ORDER)
        axes[0].axhline(0,color=INK,lw=.7);axes[0].set_xlabel('ROI ordered by AAL display network');axes[0].set_ylabel('Matched test mean Δz');axes[0].set_title('Descriptive ROI differences; no p/q copied')
        top=d.iloc[np.argsort(-np.abs(d.matched))[:20]].iloc[::-1];y=np.arange(20)
        axes[1].hlines(y,top.ci_low,top.ci_high,color=HC);axes[1].scatter(top.matched,y,color=HC,s=18);axes[1].axvline(0,color=INK,lw=.7)
        axes[1].set_yticks(y,top.label,fontsize=8);axes[1].set_xlabel('Raw mean Δz, descriptive 95% pair bootstrap CI');axes[1].set_title(f'Top-20 |Δz|; test pairs={int(d.n_pairs.iloc[0])}')
        fig.tight_layout();pages.append(fig)
        fig=plt.figure(figsize=(15,4.2));grid=fig.add_gridspec(2,3,height_ratios=[1,.08],hspace=.3)
        axes=[fig.add_subplot(grid[0,j]) for j in range(3)]
        values=[d.matched,d.unmatched,d.hc_negative];limit=max(np.abs(v).max() for v in values)
        for ax,v,title in zip(axes,values,['Matched test MDD−HC','All test MDD−HC','HC test split-half control']):self.brain(fig,ax,v.to_numpy(),title,vmax=limit)
        fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap=DIV_CMAP),cax=fig.add_subplot(grid[1,:]),orientation='horizontal',label='Raw mean Δz');pages.append(fig)
        self.emit('fig_f5_pathology_main',pages,'F5｜仅使用已有 HC/MDD test z。健康参照来自原 HC train 拟合，但图内没有 train/val 被试。匹配在 test 内重新按既定规则建立；点和森林图区间仅描述这些 test 对的原始均值差，不新增显著性结论。Top-20 按绝对差排序，区间没有选择后校正。HC 负控为 test 内站点分层随机分半，绝非 val−test。旧 0/116 FDR 结果不移用于此。',['roi_descriptive_contrasts.csv','test_pairs.csv'])

    def f6(self):
        d=self.table('roi_descriptive_contrasts');fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(2,3,height_ratios=[2,1],wspace=.3,hspace=.5)
        ix=np.argsort(-np.abs(d.unmatched))[:15][::-1];lim=max(abs(d[['matched','unmatched','hc_negative']]).max())*1.1
        for j,(col,title,c) in enumerate([('matched','Matched test',HC),('unmatched','All test',MDD),('hc_negative','HC test split control',NEG)]):
            ax=fig.add_subplot(gs[0,j]);ax.barh(range(15),d[col].iloc[ix],color=c);ax.axvline(0,color=INK,lw=.7);ax.set_xlim(-lim,lim)
            ax.set_yticks(range(15),d.label.iloc[ix] if j==0 else ['']*15,fontsize=8);ax.set_xlabel('Raw mean Δz');ax.set_title(title)
        ax=fig.add_subplot(gs[1,:2])
        for col,c,title in [('matched',HC,'Matched'),('unmatched',MDD,'All test'),('hc_negative',NEG,'HC control')]:ax.plot(np.sort(np.abs(d[col])),np.arange(1,117)/116,color=c,label=title)
        ax.set_xlabel('|Raw mean Δz|');ax.set_ylabel('ROI cumulative fraction');ax.legend()
        ax=fig.add_subplot(gs[1,2]);ax.axis('off');ax.text(0,.9,f'All test mean: {d.unmatched.mean():+.4f}\nMatched mean: {d.matched.mean():+.4f}\nMatched pairs: {int(d.n_pairs.iloc[0])}',va='top',linespacing=1.8)
        self.emit('fig_f6_site_confounding',fig,'F6｜同一组全 test 未匹配 Top-15 ROI 的匹配、未匹配和 HC test 分半负控比较，共用轴。这里比较均值差与分布，不沿用旧混合划分的 p 值或站点混杂数值。匹配子集变小且有站点筛选，变化不能单独证明病理或站点因果。',['roi_descriptive_contrasts.csv'])

    def f7(self):
        roc=self.table('roc');summ=self.table('roc_summary');z=self.table('deviation_subjects');d=self.table('roi_descriptive_contrasts')
        fc=self.table('fc_ridge_roc');fcsummary=self.table('fc_ridge_roc_summary');fcinterval=self.table('fc_ridge_roc_ci')
        fig=plt.figure(figsize=(12,6));gs=fig.add_gridspec(2,3,height_ratios=[3,1],hspace=.7,wspace=.4)
        grid=gs[0,:].subgridspec(1,2,wspace=.3)
        for j,label in enumerate(['All test','Matched test']):
            ax=fig.add_subplot(grid[j])
            band=fcinterval[fcinterval.comparison==label]
            ax.fill_between(band.fpr,band.tpr_low,band.tpr_high,color=HC,alpha=.15)
            for frame,summary,c,name in [(fc,fcsummary,HC,'Static FC K64 (train fit)'),(roc,summ,MDD,'Fixed rolling mean|z|')]:
                r=frame[frame.comparison==label];auc=summary[summary.comparison==label].auc.iloc[0]
                ax.plot(r.fpr,r.tpr,color=c,label=f'{name}: AUC={auc:.3f}')
            ax.plot([0,1],[0,1],color=INK,ls='--');ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.legend(fontsize=7);ax.set_title(label)
        self.contrasts(fig,[fig.add_subplot(gs[1,j]) for j in range(3)],[d.matched,d.unmatched,d.hc_negative])
        self.emit('fig_f7_separability',fig,'F7｜静态 FC 基线仅在原 HC/MDD train 上拟合固定 RidgeClassifier(alpha=1)，不使用 val/test 选参、定向或选阈值，全部 ROC 分数来自完整独立 test。蓝色带为固定分类器 test 分数的 500 次 bootstrap 逐点 95% 区间：全样本按组与站点分层，匹配样本按站点内匹配对重采样；不重拟合分类器，不包含训练样本不确定性。另与固定 mean|z|（高分预设为 MDD）并排；FC 用最初 K64，z 用完整滚动误差，时间窗口不同，不能由 AUC 差量化模型增益。未移用旧 OOF，也不称为队列可分性上限或外部验证。',['fc_ridge_test_scores.csv','fc_ridge_roc.csv','fc_ridge_roc_ci.csv','fc_ridge_roc_summary.csv','fc_ridge_protocol.json','roc.csv','roc_summary.csv'])

    def f8(self):
        d=self.table('deviation_subjects');fig=plt.figure(figsize=(12,7));gs=fig.add_gridspec(2,3,wspace=.35,hspace=.45)
        for col,(sub,title) in enumerate([(d,'All test'),(d[d.matched],'Matched test')]):
            ax=fig.add_subplot(gs[0,col]);rain(ax,[sub.loc[sub.group==g,'mean_abs_z'] for g in ['hc','mdd']],['HC','MDD'],ylabel='Mean |z|');ax.set_title(title)
        ax=fig.add_subplot(gs[0,2]);rain(ax,[d.loc[d.group==g,'z_std'] for g in ['hc','mdd']],['HC','MDD'],ylabel='Within-subject z SD');ax.set_title('Full test heterogeneity')
        net,order=partition();selected=[];profiles=[]
        for g in ['hc','mdd']:
            sub=d[d.group==g].sort_values('mean_abs_z');take=[sub.iloc[-1],sub.iloc[-2],sub.iloc[len(sub)//2]]
            z=pd.read_csv(ROOT/f'results/表格/健康参照模型/参照偏离/{g}/roi_z.csv').set_index('subject')
            for r in take:selected.append(f'{g.upper()} {r.subject}');profiles.append(z.loc[r.subject].to_numpy())
        profiles=np.array(profiles);ax=fig.add_subplot(gs[1,:]);lim=np.abs(profiles).max()
        im=ax.imshow(profiles[:,order],aspect='auto',cmap=DIV_CMAP,vmin=-lim,vmax=lim);ax.set_yticks(range(6),selected,fontsize=8);ax.set_xlabel('116 ROI in AAL network order');fig.colorbar(im,ax=ax,label='Normative z')
        self.emit('fig_f8_heterogeneity',fig,'F8｜完整 test 及 test 匹配子集的个体 mean|z| 分布，以及完整 test 的个体内 z 离散度。示例是每组 mean|z| 最大、次大、中位个体，确定性选择且均为 test；不能作为典型患者或稳定病灶。未复制旧队列方差检验。',['deviation_subjects.csv'])

    def f9(self):
        d=self.table('clinical_correlations');r=self.table('clinical_residuals');fig,axes=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1.2,1.2]})
        mat=d.pivot(index='network',columns='score',values='r').loc[NETWORK_ORDER,['HAMD','HAMA']]
        im=axes[0].imshow(mat,cmap=DIV_CMAP,vmin=-.4,vmax=.4,aspect='auto');axes[0].set_xticks([0,1],['HAMD','HAMA']);axes[0].set_yticks(range(7),NETWORK_ORDER)
        for i,n in enumerate(NETWORK_ORDER):
            for j,score in enumerate(['HAMD','HAMA']):
                row=d[(d.network==n)&(d.score==score)].iloc[0];axes[0].text(j,i,f'{row.r:+.3f}\nn={int(row.n)}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=axes[0],fraction=.035,label='Partial r')
        for ax,score in zip(axes[1:],['HAMD','HAMA']):
            sub=r[(r.network=='LN')&(r.score==score)];ax.scatter(sub.resid_score,sub.resid_z,c=MDD,s=15,alpha=.45)
            if len(sub)>=3:
                beta=np.polyfit(sub.resid_score,sub.resid_z,1);x=np.array([sub.resid_score.min(),sub.resid_score.max()]);ax.plot(x,np.polyval(beta,x),color=INK)
            ax.set_xlabel(f'{score} residual');ax.set_ylabel('LN mean z residual');ax.set_title(f'MDD test LN vs {score}; n={len(sub)}')
        fig.suptitle('Clinical association: MDD test complete cases only',weight='bold');fig.tight_layout(rect=(0,0,1,.95))
        self.emit('fig_f9_clinical',fig,'F9｜MDD test 内完整观测病例的临床描述性偏相关，控制年龄、性别与站点。采用 AAL116.xlsx 的完整展示网络，与旧稀疏网络统计不同，不搬用旧 r/p；不新增显著性标记。HAMA 缺失率及控制矩阵秩在表中记录。HC 无对应疾病症状评分，未伪造 HC 临床关联面板。',['clinical_correlations.csv','clinical_residuals.csv'])

    def f10(self):
        d=self.table('condition_usage');fig,axes=plt.subplots(2,3,figsize=(12,7))
        for row,g in enumerate(['hc','mdd']):
            s=d[d.group==g];rel=s.delta_pred_relative*100
            axes[row,0].hist(rel,bins=24,color=COLORS[g],alpha=.7);axes[row,0].set_xscale('log');axes[row,0].axvline(rel.median(),color=INK,ls='--')
            axes[row,0].set_xlabel('Δprediction / true one-step change (%)');axes[row,0].set_title(f'{g.upper()} test median {rel.median():.3f}%')
            ax=axes[row,1];lo=min(s.mae_cond.min(),s.mae_zero.min());hi=max(s.mae_cond.max(),s.mae_zero.max())
            ax.scatter(s.mae_cond,s.mae_zero,c=COLORS[g],s=8,alpha=.4);ax.plot([lo,hi],[lo,hi],color=INK,ls='--');ax.set_xlabel('Conditional MAE');ax.set_ylabel('All-missing condition MAE')
            q=s[['gamma','beta']].quantile([.25,.5,.75]);axes[row,2].errorbar([0,1],q.loc[.5],yerr=[q.loc[.5]-q.loc[.25],q.loc[.75]-q.loc[.5]],fmt='o',color=COLORS[g],capsize=4)
            axes[row,2].set_xticks([0,1],['FiLM gamma','FiLM beta']);axes[row,2].set_ylabel('L2 norm; median / IQR')
        fig.tight_layout()
        self.emit('fig_f10_condition_usage',fig,'F10｜全部 HC/MDD test 在最初 K64 固定起点的条件使用检查，分母为真实下一 TR 与最后观测的平均绝对变化；与旧全部滚动锚点 B5 口径不同。全缺失条件按已有 zero_condition 编码实现，校准与 profile 固定。参数范数不等于输出贡献；未启用的 AdaRMS 不造分布。',['condition_usage.csv'])

    def f11(self):
        pages=[];means={};keys=['prior_norm','A_func','A_eff'];_,order=partition();net,_=partition();block=[]
        for g in ['hc','mdd']:
            with np.load(self.data/f'graph_mean_{g}.npz') as d:means[g]={k:d[k].copy() for k in d.files}
        vmax=max(v.max() for d in means.values() for v in d.values())
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        for row,g in enumerate(['hc','mdd']):
            for col,k in enumerate(keys):
                im=axes[row,col].imshow(means[g][k][np.ix_(order,order)],cmap='viridis',vmin=0,vmax=vmax)
                axes[row,col].set_title(f'{g.upper()} test {k}');axes[row,col].set_xticks([]);axes[row,col].set_yticks([]);fig.colorbar(im,ax=axes[row,col],fraction=.04)
        fig.tight_layout();pages.append(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,g in zip(axes,['hc','mdd']):
            for j,k in enumerate(keys):
                values=[]
                for n in NETWORK_ORDER:
                    ix=np.flatnonzero(net==n);m=means[g][k][np.ix_(ix,ix)];v=m[~np.eye(len(ix),dtype=bool)].mean();values.append(v)
                    block.append(dict(group=g,network=n,graph=k,mean_weight=v))
                ax.bar(np.arange(7)+(j-1)*.24,values,width=.22,color=[HC,NEG,MDD][j],label=k)
            ax.set_xticks(range(7),NETWORK_ORDER);ax.set_ylabel('Within-network weight, no diagonal');ax.set_title(f'{g.upper()} test mean graph');ax.legend(fontsize=7)
        fig.tight_layout();pages.append(fig);pd.DataFrame(block).to_csv(self.tables/'graph_network_blocks.csv',index=False)
        self.emit('fig_f11_graph',pages,'F11｜完整 HC/MDD test 的历史功能连接先验、学习功能图、有效图均值，共享矩阵色阶；网络均值排除对角。输入是 history-FC，不是实测解剖 SC；最终图包含残差、自环及归一化。共享 λ 从 generation_metrics.csv 可查，不能解释成独立解剖连接比例。',['generation_metrics.csv','graph_network_blocks.csv','graph_mean_hc.npz','graph_mean_mdd.npz'])

    def f12(self):
        d=self.table('roi_importance');fig=plt.figure(figsize=(13,9));grid=fig.add_gridspec(2,2,wspace=.65,hspace=.35)
        axes=np.empty((2,2),dtype=object);color_axes=[]
        for row in range(2):
            subgrid=grid[row,0].subgridspec(1,2,width_ratios=[20,1],wspace=.1)
            axes[row,0]=fig.add_subplot(subgrid[0]);color_axes.append(fig.add_subplot(subgrid[1]))
            axes[row,1]=fig.add_subplot(grid[row,1])
        limit=d.delta_mae.abs().max()
        for row,g in enumerate(['hc','mdd']):
            s=d[d.group==g];self.brain(fig,axes[row,0],s.delta_mae.to_numpy(),f'{g.upper()} test lag-1 ROI importance',vmax=limit)
            fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap=DIV_CMAP),cax=color_axes[row],label='ΔODE lag-1 MAE')
            top=s.nlargest(15,'delta_mae').iloc[::-1];y=np.arange(15);ax=axes[row,1]
            ax.hlines(y,top.ci_low,top.ci_high,color=COLORS[g]);ax.scatter(top.delta_mae,y,c=COLORS[g],s=18);ax.axvline(0,color=INK,lw=.6)
            ax.set_yticks(y,top.label,fontsize=8);ax.set_xlabel('ΔODE lag-1 MAE, subject bootstrap 95% CI');ax.set_title(f'{g.upper()} top-15 (descriptive)')
        self.emit('fig_f12_roi_importance',fig,'F12｜在每个完整 test 队列内预先固定供体排列，跨被试替换一个 ROI 的 K64 历史；条件、校准与 profile 保留目标被试值，功能图随扰动输入重算。比较相同批形状的未扰动和扰动 ODE lag-1 MAE，避免跨批数值差。区间为逐被试 bootstrap 描述性区间，不沿用旧 4 批近似 FDR，也不新增星号。不是疾病定位或因果归因。',['roi_importance.csv','importance_subjects.csv'])

    def f13(self):
        d=self.table('history_sensitivity');fig,axes=plt.subplots(2,2,figsize=(12,8))
        for col,g in enumerate(['hc','mdd']):
            sub=d[d.group==g]
            for segment,c,marker in [('H1_8',HC,'o'),('H9_16',NEG,'s'),('H17_32',MDD,'^')]:
                m=sub[sub.segment==segment].pivot(index='subject',columns='block',values='relative_change').to_numpy()
                axes[0,col].plot(range(1,9),m.mean(0),color=c,marker=marker,label=segment)
                lo,hi=np.quantile(m,[.25,.75],axis=0);axes[0,col].fill_between(range(1,9),lo,hi,color=c,alpha=.15)
            axes[0,col].axhline(0,color=INK,lw=.7);axes[0,col].set_title(f'{g.upper()} full test n={len(test_ids(g))}');axes[0,col].set_ylabel('Relative MAE change (%)');axes[0,col].set_xlabel('History block (oldest → newest)');axes[0,col].legend(fontsize=7)
        matrices=[d[(d.group==g)&(d.segment=='H17_32')].pivot(index='subject',columns='block',values='relative_change').to_numpy() for g in ['hc','mdd']]
        limit=max(np.abs(m).max() for m in matrices)
        for ax,m,g in zip(axes[1],matrices,['hc','mdd']):
            im=ax.imshow(m,aspect='auto',cmap=DIV_CMAP,vmin=-limit,vmax=limit);ax.set_xticks(range(8),range(1,9));ax.set_xlabel('Shuffled history block');ax.set_ylabel('Test subject');ax.set_title(f'{g.upper()} H17–32 change')
            fig.colorbar(im,ax=ax,fraction=.035,label='Relative change (%)')
        fig.tight_layout()
        self.emit('fig_f13_history_attribution',fig,'F13｜完整 HC/MDD test，K64 分为 8 个 8 TR 块，每次打乱一个块内时间顺序，固定种子重复 3 次；每人先平均重复，不将重复当独立被试。曲线色带为被试四分位范围；两热图共享色阶。最后一块置换会改变最后观测值，因此不能将短期误差增量全部解释成纯时间顺序理解。H32 属于 HC 参照外推，不复制旧 T08 数值。',['history_sensitivity.csv'])

    def f14(self):
        means={};net,order=partition();mask=~np.eye(116,dtype=bool);rows=[];pages=[]
        for g in ['hc','mdd']:
            with np.load(self.data/f'propagation_{g}.npz') as d:means[g]={k:d[k].mean(0) for k in ['early','late']}
        limit=max(m[mask].max() for group in means.values() for m in group.values());cmap=plt.get_cmap('magma').copy();cmap.set_bad('#ddd')
        fig,axes=plt.subplots(2,2,figsize=(10,9))
        for row,g in enumerate(['hc','mdd']):
            for col,key in enumerate(['early','late']):
                m=np.ma.array(means[g][key],mask=~mask);im=axes[row,col].imshow(m[np.ix_(order,order)],cmap=cmap,vmin=0,vmax=limit)
                sizes=np.array([sum(net==n) for n in NETWORK_ORDER]);ticks=sizes.cumsum()-sizes/2-.5
                axes[row,col].set_xticks(ticks,NETWORK_ORDER,rotation=45,fontsize=7);axes[row,col].set_yticks(ticks,NETWORK_ORDER,fontsize=7)
                axes[row,col].set_xlabel('Target ROI');axes[row,col].set_ylabel('Perturbed source ROI');axes[row,col].set_title(f'{g.upper()} test {"H1–8" if key=="early" else "H17–32"}')
        fig.subplots_adjust(right=.86,hspace=.45,wspace=.3);fig.colorbar(im,cax=fig.add_axes([.89,.16,.025,.68]),label='Mean |Δprediction| / 0.1; gray: diagonal omitted');pages.append(fig)
        fig=plt.figure(figsize=(13,7));grid=fig.add_gridspec(2,3,width_ratios=[1,1,.03],hspace=.4,wspace=.3)
        axes=np.array([[fig.add_subplot(grid[i,j]) for j in range(2)] for i in range(2)]);source={}
        for g in ['hc','mdd']:
            m=means[g]['late'].copy();np.fill_diagonal(m,0);source[g]=m.sum(1)/115
            rows.extend(dict(group=g,roi=j+1,label=labels()['micro命名'].iloc[j],source_sensitivity=source[g][j]) for j in range(116))
        maximum=max(v.max() for v in source.values());time=self.table('perturbation_timecourse')
        for col,g in enumerate(['hc','mdd']):
            self.brain(fig,axes[0,col],source[g],f'{g.upper()} test cross-ROI source sensitivity',vmin=0,vmax=maximum,cmap='magma')
            values=time[time.group==g].pivot(index='subject',columns='horizon',values='response').to_numpy();lo,hi=np.quantile(values,[.25,.75],axis=0)
            axes[1,col].plot(range(1,33),values.mean(0),color=COLORS[g]);axes[1,col].fill_between(range(1,33),lo,hi,color=COLORS[g],alpha=.2)
            axes[1,col].axhline(0,color=INK,ls='--',label='Zero-input sham');axes[1,col].set_xlabel('Forecast horizon (TR)');axes[1,col].set_ylabel('Cross-ROI |Δprediction| / 0.1');axes[1,col].legend()
        fig.colorbar(ScalarMappable(norm=Normalize(0,maximum),cmap='magma'),cax=fig.add_subplot(grid[0,2]),label='Source sensitivity');pages.append(fig)
        pd.DataFrame(rows).to_csv(self.tables/'perturbation_roi_summary.csv',index=False)
        self.emit('fig_f14_perturbation',pages,'F14｜每个 HC/MDD test 被试、每个源 ROI、正负两个方向完整执行最后 8 TR 的 ±0.1 输入扰动。响应为同形状批次相对未扰动预测的平均绝对差除以 0.1；矩阵及脑图排除直接自响应，两组共享对应色阶。每人零扰动 sham 必须通过 1e-6 门槛。条件、校准和 profile 固定，功能图随历史变化；仅模型结构敏感性，不是实测刺激、损伤、治疗或疾病因果。',['perturbation_roi_summary.csv','perturbation_timecourse.csv','probe_gates.csv','propagation_hc.npz','propagation_mdd.npz'])

    def s1(self):
        d=self.table('profiles');pairs=self.table('test_pairs');neg=read_json(self.data/'hc_test_negative_control.json');values={}
        for g in ['hc','mdd']:values[g]=d[d.group==g].pivot(index='subject',columns='dim',values='value')
        matched=values['mdd'].loc[pairs.mdd].to_numpy().mean(0)-values['hc'].loc[pairs.hc].to_numpy().mean(0)
        unmatched=values['mdd'].mean(0).to_numpy()-values['hc'].mean(0).to_numpy()
        negative=values['hc'].loc[neg['A']].mean(0).to_numpy()-values['hc'].loc[neg['B']].mean(0).to_numpy()
        fig,axes=plt.subplots(2,3,figsize=(13,7));limit=max(np.abs(v).max() for v in [matched,unmatched,negative])*1.1
        for ax,v,title,c in zip(axes[0],[matched,unmatched,negative],['Matched test MDD−HC','All test MDD−HC','HC test split control'],[HC,MDD,NEG]):
            ax.bar(range(32),v,color=c);ax.axhline(0,color=INK,lw=.7);ax.set_ylim(-limit,limit);ax.set_xlabel('Profile dimension (zero-based)');ax.set_title(title)
        full=pd.concat([values['hc'],values['mdd']]).to_numpy();centered=full-full.mean(0)
        _,s,v=np.linalg.svd(centered,full_matrices=False);scores=centered@v[:2].T;offset=0
        for g in ['hc','mdd']:
            n=len(values[g]);axes[1,0].scatter(scores[offset:offset+n,0],scores[offset:offset+n,1],s=7,c=COLORS[g],alpha=.4,label=g.upper());offset+=n
        axes[1,0].set_xlabel('Descriptive pooled-test PC1');axes[1,0].set_ylabel('PC2');axes[1,0].legend()
        axes[1,1].bar(range(1,9),100*s[:8]**2/(s**2).sum(),color=HC);axes[1,1].set_xlabel('Centered PCA direction');axes[1,1].set_ylabel('Test sample variance (%)')
        axes[1,2].axis('off');axes[1,2].text(0,.9,'All profiles: model.eval()\nNo archived q-values reused\nPCA is descriptive, not a classifier\nNo disease mechanism inference',va='top',linespacing=1.8)
        fig.tight_layout();pd.DataFrame(dict(dim=range(32),matched=matched,unmatched=unmatched,hc_negative=negative)).to_csv(self.tables/'profile_contrasts.csv',index=False)
        self.emit('fig_s1_profile',fig,'S1｜全部 test 的 eval 模式 profile，匹配、未匹配和 HC test 分半负控采用共同纵轴；不复用旧脚本受 dropout 影响的 q 值。PCA 在所画 test 样本上中心化拟合，仅为描述性嵌入，不是独立训练分类器或外部泛化成绩；其方差比例不能与旧未中心化 SVD 能量占比混为一谈。',['profiles.csv','profile_contrasts.csv'])

    def s2(self):
        primary=pd.read_csv(ROOT/'results/表格/健康参照模型/自主生成32步/per_subject.csv');fine=pd.read_csv(ROOT/'results/表格/抑郁症微调模型/自主生成32步/per_subject.csv')
        sets=[(primary[primary.group=='hc'],'HC ref → HC test',HC),(primary[primary.group=='mdd'],'HC ref → MDD test',MDD),(fine,'MDD fine → MDD test',NEG)]
        pages=[];fig,axes=plt.subplots(1,3,figsize=(14,4.5))
        for ax,key,title in zip(axes,['acf_mae_lags_1_5','profile_half_own_distance','profile_half_other_distance'],['ACF error, lags 1–5','Profile own-half distance','Profile other-person distance']):
            rain(ax,[s[key] for s,_,_ in sets],[t.replace(' → ','\n') for _,t,_ in sets],colors=[c for _,_,c in sets],ylabel=title)
        fig.tight_layout();pages.append(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,4.8));keys=['generated_identified','observed_identified','split_half_identified','profile_half_identified']
        for j,(sub,title,c) in enumerate(sets):
            axes[0].bar(np.arange(4)+(j-1)*.24,[100*sub[k].mean() for k in keys],width=.22,color=c,label=title)
        axes[0].set_xticks(range(4),['Generated FC','History → future FC','Observed FC halves','Profile halves'],rotation=20,ha='right');axes[0].set_ylabel('Within-scan identification (%)');axes[0].legend(fontsize=7)
        axes[0].set_ylim(0,100)
        axes[0].axhline(100/222,color=INK,ls='--',lw=.6,label='Random ≈0.45%');axes[0].legend(fontsize=7)
        for j,(sub,title,c) in enumerate(sets):
            axes[1].plot(range(3),[sub[k].mean() for k in ['crps_h1','crps_h1_32','crps_h17_32']],marker=['o','s','^'][j],color=c,label=title)
        axes[1].set_xticks(range(3),['H1','H1–32','H17–32']);axes[1].set_ylabel('Gaussian marginal CRPS (lower is better)');axes[1].set_ylim(bottom=0);axes[1].legend(fontsize=7)
        fig.tight_layout();pages.append(fig)
        summary=[]
        for sub,title,c in sets:
            g='hc' if 'HC test' in title else 'mdd';check_ids(sub.subject,g)
            summary.append(dict(evaluation=title,n=len(sub),**{k:sub[k].mean() for k in sub.columns if k not in ['subject','group']}))
        pd.DataFrame(summary).to_csv(self.tables/'additional_test_metrics.csv',index=False)
        self.emit('fig_s2_dynamics_identification',pages,'S2｜复用已验证完整 test 的 ACF、识别、profile 扫描内分半与高斯边际 CRPS。HC 参照→HC/MDD 全 test，另示 MDD 微调→同一 MDD test，三个组合独立标识；微调使用 mdd8 条件，参照使用 hc3，不能把差异归因于单一结构因素。身份识别候选为同组同一扫描相邻时段，不是跨扫描重测或外部队列。CRPS 没有配套概率基线，不能由绝对值宣称校准良好。',['additional_test_metrics.csv'])


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=DATA);ap.add_argument('--table-dir',type=Path,default=TABLES)
    ap.add_argument('--out-dir',type=Path,default=FIGURES);ap.add_argument('--figure',default='all')
    ns=ap.parse_args();ns.out_dir.mkdir(parents=True,exist_ok=True)
    if any(ns.out_dir.iterdir()):raise FileExistsError('Use a fresh PDF-only output directory')
    apply_style()
    with PdfPages(ns.out_dir/'全部图表.pdf') as book:
        plot=Plotter(ns.data_dir,ns.table_dir,ns.out_dir,book)
        mapping={f'f{i}':getattr(plot,f'f{i}') for i in range(1,15)}
        mapping={'f0a':plot.f0a,'f0b':plot.f0b,**mapping,'s1':plot.s1,'s2':plot.s2}
        for name in (mapping if ns.figure=='all' else [ns.figure]):mapping[name]()
    write_json(ns.data_dir/'figure_registry.json',plot.registry)
    pd.DataFrame([dict(figure=r['figure'],pdf=r['figure']+'.pdf',pages=r['pages'],caption_zh=r['caption_zh']) for r in plot.registry]).to_csv(ns.table_dir/'figure_captions.csv',index=False)
    print(f'Completed {len(plot.registry)} figure PDFs; PDF-only directory',flush=True)


if __name__=='__main__':main()
