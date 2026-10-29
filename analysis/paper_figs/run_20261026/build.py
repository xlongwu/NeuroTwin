"""Build static paper figures from reviewed artifacts, with data and Chinese captions.
Run: python -m analysis.paper_figs.run_20261026.build --out-dir NEW_DIR
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from analysis.pathology_tests import load_z
from analysis.paper_figs.common import matched_pairs
from .style import (ROOT,RUN1,HC,MDD,NEG,INK,NETWORK_ORDER,NETWORK_COLORS,
                    DIV_CMAP,apply_style,json_read,labels,partition,strip,rain,save)

STATS=ROOT/'results/paper_stats/run_20261026'
INFER=ROOT/'results/paper_infer/run_20261026'
CAPTIONS={}


def emit(fig,out,name,data,caption):
    save(fig,out,name,data,caption)
    CAPTIONS[name]=caption


def three_data():
    return pd.read_csv(STATS/'roi_three_comparisons.csv')


def controls(fig,gs,table):
    limit=max(table.delta_z.abs().max(),table.delta_unmatched.abs().max(),table.delta_negative.abs().max())
    fig.text(.5,.02,f'Shared color scale: -{limit:.3f} to +{limit:.3f} raw mean z difference',ha='center',fontsize=8)
    for j,(col,title) in enumerate([('delta_z','Matched MDD - HC'),('delta_unmatched','Unmatched MDD - HC'),('delta_negative','HC val - HC test')]):
        strip(fig.add_subplot(gs[j]),table[col],title,limit)


def atlas_image(values, atlas_dir):
    import nibabel as nib
    image=nib.load(str(atlas_dir/'ROI_MNI_V4.nii'))
    atlas=np.asarray(image.dataobj)
    mapping=pd.read_csv(atlas_dir/'ROI_MNI_V4.txt',sep='\t',header=None,names=['abbr','name','value'])
    expected=labels()['micro命名'].tolist()
    if expected != mapping.name.tolist():
        raise ValueError('Atlas/AAL116 order mismatch')
    vox=np.zeros(atlas.shape,dtype=np.float32)
    for code,v in zip(mapping.value,values): vox[atlas==code]=v
    return nib.Nifti1Image(vox,image.affine)


def f0a(out,atlas):
    fig,ax=plt.subplots(figsize=(11,3.5));ax.axis('off')
    boxes=[('Read-only sources','HC base checkpoint\nRaw prefix calibration (32 TR)\nK64 histories / fixed hc3'),('Archived rolling / z','Rolling lag-1 errors\nHC train reference fit\nScore all cohort splits'),('Controls / tests','1504 same-site pairs\nUnmatched descriptive contrast\nHC val-test negative control'),('Paper artifacts','T0: reuse archived outputs\nT1: residuals / ROC intervals\nT2: H32 / graph / importance')]
    for i,(title,text) in enumerate(boxes):
        x=.015+i*.25
        ax.add_patch(FancyBboxPatch((x,.28),.22,.52,boxstyle='round,pad=.01',facecolor='#f3f6f8',edgecolor=HC,lw=1,transform=ax.transAxes))
        ax.text(x+.11,.70,title,ha='center',weight='bold',fontsize=10,transform=ax.transAxes)
        ax.text(x+.11,.51,text,ha='center',va='center',linespacing=1.7,fontsize=8,transform=ax.transAxes)
        if i<3:ax.annotate('',xy=(x+.245,.54),xytext=(x+.223,.54),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':INK})
    ax.text(.5,.1,'T3 rolling recalculation and I5 interventions excluded',ha='center',fontsize=10,transform=ax.transAxes)
    ax.set_title('Read-only analysis pipeline',pad=12)
    emit(fig,out,'fig_f0a_pipeline',{'stages':boxes,'scope':'T0/T1/T2; no training'},'F0a｜分析流程。滚动评估、规范性参照和组间检验均复用既有产物；本次只补充统计呈现和冻结权重前向。HC 参照模型与历史 H32 模型的结果分别标识；不执行 T3 滚动重算及 I5 虚拟干预。流程框为数据流示意，不代表本次重新执行全部上游。')


def f0b(out,atlas):
    matched=set(s for p in matched_pairs() for s in (p['hc'],p['mdd']))
    rows=[]
    for g in ('hc','mdd'):
        for sp in ('train','val','test'):
            ids,z=load_z(RUN1,g,sp)
            rows.append(dict(group=g.upper(),split=sp,total=len(ids),matched=sum(s in matched for s in ids)))
    df=pd.DataFrame(rows);df['other']=df.total-df.matched
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={'width_ratios':[1.2,1]})
    for j,r in df.iterrows():
        c=HC if r.group=='HC' else MDD
        axes[0].barh(j,r.matched,color=c,label='Matched' if j==0 else None)
        axes[0].barh(j,r.other,left=r.matched,color=c,alpha=.23,label='Other' if j==0 else None)
        axes[0].text(r.total+20,j,str(r.total),va='center',fontsize=8)
    axes[0].set_yticks(range(6),df.group+' '+df.split);axes[0].invert_yaxis();axes[0].set_xlim(0,2050)
    axes[0].set_xlabel('Subjects');axes[0].set_title('(a) Cohort and split');axes[0].legend()
    axes[1].axis('off'); t=axes[1].table(cellText=df[['group','split','total','matched']].values,colLabels=['Group','Split','All','Matched'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.7)
    axes[1].set_title('(b) 1504 pairs across 36 sites')
    fig.tight_layout()
    emit(fig,out,'fig_f0b_composition',df,'F0b｜被试构成。HC 2222 人、MDD 2253 人，按原 train/val/test 划分展示；深色为已登记站点匹配子集，浅色为其余被试。匹配共 1504 对、36 站点。此处为 z 矩阵的分析队列，不能与 F2 的 MDD 验证/测试各 225 人或 F1 的 HC 验证 222 人混用。')


def f1(out,atlas):
    gate=json_read(ROOT/'results/normative_reference/gate_report.json');cu=pd.read_csv(ROOT/'results/pathology_analysis/condition_usage.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    vals=[gate['mae_model_subject_equal_weight'],gate['mae_persistence_subject_equal_weight']]
    axes[0].bar([0,1],vals,color=[HC,'#aaa'],width=.55)
    for i,v in enumerate(vals):axes[0].text(i,v+.018,f'{v:.4f}',ha='center')
    axes[0].set_xticks([0,1],['HC reference','Persistence']);axes[0].set_ylim(0,.74);axes[0].set_ylabel('Subject-equal rolling lag-1 MAE')
    axes[0].set_title('(a) HC val: n=222')
    axes[0].text(.04,.90,f'Improvement {gate["improvement"]:.1%}\nArchived permutation p = 0.00010',transform=axes[0].transAxes,fontsize=9)
    rain(axes[1],[cu.mae_cond],['Model'],ylabel='Per-subject rolling lag-1 MAE');axes[1].set_title('(b) Subject distribution')
    axes[1].text(.04,.9,f'Median = {cu.mae_cond.median():.4f}',transform=axes[1].transAxes)
    fig.tight_layout()
    emit(fig,out,'fig_f1_gate',{'gate':gate,'subjects':cu[['subject','mae_cond']].to_dict('records')},'F1｜滚动单步护栏（T0）。HC 验证集 222 人，模型与 persistence 被试等权 MAE 分别为 0.08143 和 0.60484，改善 86.54%；p 来自原 10000 次配对符号翻转检验。右图展示真实逐被试分布与中位数。原置换零分布未保留，因此不绘制模拟 null 直方图。预测优势不等于病理定位能力。')


def f2(out,atlas):
    frames=[]; fig,axes=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.4,1]})
    segments=['H1_8','H9_16','H17_32','H1_32']
    for split,dx,c in [('val',-.16,HC),('test',.16,MDD)]:
        d=pd.read_csv(ROOT/f'results/paper_figures/complete_20261026/sources/h32/per_subject_{split}.csv');d['split']=split;frames.append(d)
        metrics=json_read(ROOT/f'results/paper_figures/complete_20261026/sources/h32/metrics_{split}.json')
        for i,seg in enumerate(segments):
            v=d.loc[d.segment==seg,'mae'];np.testing.assert_allclose(v.mean(),metrics['ode_mae'][seg],atol=1e-12)
            vp=axes[0].violinplot(v,positions=[i+dx],widths=.28,showextrema=False)
            for body in vp['bodies']:body.set_facecolor(c);body.set_alpha(.25)
            axes[0].scatter(i+dx,v.mean(),color=c,s=25)
    axes[0].set_xticks(range(4),['H1–8','H9–16','H17–32','H1–32']);axes[0].set_ylabel('ODE MAE');axes[0].set_title('(a) Historical selected H32 model')
    axes[0].hlines(.922955,1.65,2.35,color=INK,ls='--',label='Test far-horizon mean baseline')
    axes[0].legend(handles=[plt.Line2D([],[],color=HC,marker='o',ls='',label='Val n=225'),plt.Line2D([],[],color=MDD,marker='o',ls='',label='Test n=225')],loc='upper left')
    axes[1].axis('off');axes[1].text(0,.92,'Far-horizon limits',weight='bold',fontsize=11)
    test=json_read(ROOT/'results/paper_figures/complete_20261026/sources/h32/metrics_test.json')['ode_mae']['H17_32'];gain=1-test/.922955
    axes[1].text(0,.73,f'Test H17–32 MAE: {test:.6f}\nHistorical mean: 0.922955\nRelative gain: {gain:.2%}\nTemporal PCC: ≈ 0.178 (archive)',linespacing=1.8,va='top')
    axes[1].text(0,.26,'T04 common coefficient: 0.948788\nFar-horizon gain: 0.0228%\nCalibration / S4 / T18: negative',fontsize=9,linespacing=1.7,va='top')
    fig.tight_layout()
    emit(fig,out,'fig_f2_horizon',pd.concat(frames),'F2｜历史 H32 最优模型的长程精度（T0），来源为 final 存档，权重 SHA256 为 264507…d78b；它与其余主分析使用的 HC 参照权重不同。两组点表示 val/test 各 225 人的均值，小提琴表示逐被试分布。虚线仅对应 test H17–32 历史均值基线 0.922955，不能外推到其他分段或 val。test 远期 MAE 为 0.896604，相对基线改善 2.86%；PCC 约 0.178 是历史实验摘要值。T04/S4/T18 仅引用存档阴性结论，未重跑或部署校准。')


def f5(out,atlas):
    from nilearn.plotting import plot_glass_brain
    d=three_data();net,order=partition(); gt=json_read(RUN1/'group_tests.json')
    fig=plt.figure(figsize=(14,9));gs=fig.add_gridspec(2,3,height_ratios=[2.2,1],hspace=.5,wspace=.6)
    ax=fig.add_subplot(gs[0,0]);q=d.q.to_numpy();x=np.arange(116)
    ax.scatter(x,-np.log10(d.p.to_numpy()[order]),c=[NETWORK_COLORS[n] for n in net[order]],s=17)
    # BH has no single accepted threshold when no p passes; label Bonferroni honestly.
    ax.axhline(-np.log10(.05/116),color=INK,ls='--',lw=.8)
    ax.set_title('(a) ROI tests: 0/116 BH-FDR q<0.05');ax.set_ylabel('-log10(p)');ax.set_xlabel('ROI order grouped by AAL networks')
    ax.text(.03,.89,'Dashed: Bonferroni 0.05/116',transform=ax.transAxes,va='top',fontsize=8)
    ax=fig.add_subplot(gs[0,1]);top=d.iloc[np.argsort(-d.delta_z.abs())[:20]].iloc[::-1];y=np.arange(20)
    ax.hlines(y,top.ci_lo,top.ci_hi,color=HC);ax.scatter(top.delta_z,y,color=HC,s=18);ax.axvline(0,color=INK,lw=.8)
    ax.set_yticks(y,top.label,fontsize=7);ax.set_xlabel('Matched delta z, 95% paired bootstrap CI');ax.set_title('(b) Largest absolute raw effects')
    ax=fig.add_subplot(gs[0,2]);nd=pd.DataFrame(gt['network_and_whole_brain']);y=np.arange(len(nd))
    ax.hlines(y,nd.ci_lo,nd.ci_hi,color=HC);ax.scatter(nd.delta,y,color=HC);ax.axvline(0,color=INK,lw=.8)
    ax.set_yticks(y,[f'{n} ({r})' for n,r in zip(nd.name,nd.n_rois)]);ax.set_xlabel('Delta z, original 95% CI');ax.set_title('(c) Original sparse network grouping')
    limit=max(d.delta_z.abs().max(),d.delta_unmatched.abs().max(),d.delta_negative.abs().max())
    for j,(col,name) in enumerate([('delta_z','Matched'),('delta_unmatched','Unmatched'),('delta_negative','HC negative control')]):
        sub=gs[1,j].subgridspec(1,2,width_ratios=[18,1],wspace=.35)
        ax=fig.add_subplot(sub[0]);display=plot_glass_brain(atlas_image(d[col],atlas),axes=ax,figure=fig,display_mode='lyrz',plot_abs=False,cmap=DIV_CMAP,vmin=-limit,vmax=limit,threshold=None,colorbar=False,title=None)
        if j==2:
            cax=fig.add_subplot(sub[1]);fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap=DIV_CMAP),cax=cax,label='Raw delta z')
        ax.set_title(name,fontsize=10)
    fig.suptitle('Normative deviations: archived matched inference and required controls',weight='bold',y=.98)
    emit(fig,out,'fig_f5_pathology_main',d,'F5｜病理主结果（T0，三组图为 T1 复核值）。匹配 1504 对，原站点块置换加 BH-FDR 检验为 0/116 ROI 显著，最小 q=0.336；森林图为原 95% 配对 bootstrap 区间，按绝对效应选择 Top-20，未校正区间不作为显著性依据。网络/全脑面板严格保留原稀疏分组：仅 14 ROI 被具名，102 ROI 属于 Other，不能解释为完整 AAL 网络检验。底部脑图共享色阶，依次为匹配 MDD−HC、未匹配 MDD−HC、HC val−test 原始均值差；后两者仅提供混杂与噪声对照。展示排序依据 AAL116 完整网络标签，与原推断分组不同。')


def f6(out,atlas):
    d=three_data();fig=plt.figure(figsize=(12,7));gs=fig.add_gridspec(2,3,height_ratios=[2,1],hspace=.5,wspace=.25)
    ix=np.argsort(-d.delta_unmatched.abs())[:15][::-1];vmax=.2
    for j,(col,title,c) in enumerate([('delta_z','(a) Matched',HC),('delta_unmatched','(b) Unmatched',MDD),('delta_negative','(c) HC negative control',NEG)]):
        ax=fig.add_subplot(gs[0,j]);ax.barh(range(len(ix)),d[col].iloc[ix],color=c,alpha=.8);ax.axvline(0,color=INK,lw=.8);ax.set_xlim(-vmax,vmax)
        ax.set_yticks(range(len(ix)),d.label.iloc[ix] if j==0 else ['']*len(ix),fontsize=8);ax.set_xlabel('Raw mean delta z');ax.set_title(title)
    ax=fig.add_subplot(gs[1,:2])
    for col,c,label in [('delta_z',HC,'Matched'),('delta_unmatched',MDD,'Unmatched'),('delta_negative',NEG,'HC control')]:
        vals=np.sort(np.abs(d[col]));ax.plot(vals,np.arange(1,117)/116,color=c,label=label)
    ax.set_xlabel('|Raw mean delta z|');ax.set_ylabel('ROI cumulative fraction');ax.legend(loc='lower right')
    ax=fig.add_subplot(gs[1,2]);ax.axis('off');ax.text(0,.95,f'Whole-brain raw delta\nUnmatched: {d.delta_unmatched.mean():+.4f}\nMatched: {d.delta_z.mean():+.4f}\n\nMax |delta|, HC control\n{d.delta_negative.abs().max():.4f}',va='top',linespacing=1.5)
    fig.suptitle('Site matching changes apparent group differences',weight='bold',y=.99)
    emit(fig,out,'fig_f6_site_confounding',d,'F6｜站点混杂方法学演示（T1）。前三面板用同一组未匹配 Top-15 ROI、相同横轴，对照匹配、未匹配和 HC 负控的原始均值差；下方比较全部 116 ROI 的绝对差异分布。未匹配全脑原始差为 +0.06880，匹配后为 −0.02281，HC 负控最大绝对差 0.17622。排名表另存 top_roi_changes.csv。既有未匹配/负控 p 值检验的是站点加权差，与本图原始差的估计量不同，故不在此图标注或解释为其显著性。变化与站点构成混杂一致，不能据此指定病理脑区。')


def f7(out,atlas):
    d=three_data();desc=pd.read_csv(RUN1/'roi_fdr_table_descriptor.csv');roc=pd.read_csv(STATS/'roc_observed.csv');ci=pd.read_csv(STATS/'roc_bootstrap.csv');summ=json_read(STATS/'summary.json')
    fig=plt.figure(figsize=(11,6));gs=fig.add_gridspec(2,3,height_ratios=[3,1],wspace=.4,hspace=.5)
    ax=fig.add_subplot(gs[0,0]);ax.fill_between(ci.fpr,ci.tpr_lo,ci.tpr_hi,color=HC,alpha=.2);ax.plot(roc.fpr,roc.tpr,color=HC);ax.plot([0,1],[0,1],color=INK,ls='--',lw=.8)
    ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.set_title('(a) Static FC + ridge OOF ROC')
    ax.text(.97,.05,f'AUC {summ["auc"]:.3f}\n95% CI {summ["auc_ci95"][0]:.3f}–{summ["auc_ci95"][1]:.3f}',transform=ax.transAxes,ha='right')
    ax=fig.add_subplot(gs[0,1:]);ax.scatter(desc.delta_dref,d.delta_z,color=HC,s=18,alpha=.7);ax.plot([-.12,.12],[-.12,.12],color=INK,ls='--');ax.axhline(0,color='#bbb',lw=.6);ax.axvline(0,color='#bbb',lw=.6)
    ax.set_xlim(-.12,.12);ax.set_ylim(-.12,.12);ax.set_xlabel('Descriptor-reference raw difference');ax.set_ylabel('Model-z raw difference');ax.set_title('(b) Matched ROI contrasts; distinct score scales')
    controls(fig,[gs[1,j] for j in range(3)],d)
    fig.tight_layout()
    emit(fig,out,'fig_f7_separability',{'roc':roc.to_dict('records'),'bootstrap':ci.to_dict('records'),'summary':summ,'roi_scatter':pd.DataFrame({'roi_index':d.roi_index,'descriptor_delta':desc.delta_dref,'model_delta':d.delta_z}).to_dict('records')},'F7｜队列可分性与描述量基线（T1）。复用原站内轮转 5 折岭分类器的固定 OOF 分数，AUC=0.55980；500 次站内匹配对 bootstrap 的 AUC 95% 区间为 0.54026–0.57822，阴影为逐点 ROC 区间，未重新拟合分类器，也不表示跨站点或外部验证。右图比较原模型 z 与六描述量规范性距离的匹配 ROI 均值差；两者尺度不同，对角线仅作数值参考，不能据此量化模型增益。二者原检验均 0/116 过 FDR。下方保留模型 z 的匹配、未匹配和 HC 负控三组对照；描述量非匹配/负控逐 ROI 数组未保留，不新增相应推断。')


def f8(out,atlas):
    pairs=matched_pairs();zmap={};by={};byids={}
    for g in ('hc','mdd'):
        for sp in ('train','val','test'):
            ids,z=load_z(RUN1,g,sp);zmap.update(zip(ids,z));by[g+'_'+sp]=z;byids[g+'_'+sp]=ids
    h=np.stack([zmap[p['hc']] for p in pairs]);m=np.stack([zmap[p['mdd']] for p in pairs])
    fig=plt.figure(figsize=(12,7));gs=fig.add_gridspec(2,3,height_ratios=[1.3,1],hspace=.65,wspace=.45)
    sets=[(h,m),(np.concatenate([by['hc_'+s] for s in ('train','val','test')]),np.concatenate([by['mdd_'+s] for s in ('train','val','test')])),(by['hc_val'],by['hc_test'])]
    identities=[([p['hc'] for p in pairs],[p['mdd'] for p in pairs]),([sid for sp in ('train','val','test') for sid in byids['hc_'+sp]],[sid for sp in ('train','val','test') for sid in byids['mdd_'+sp]]),(byids['hc_val'],byids['hc_test'])]
    ymax=max(np.abs(arr).mean(1).max() for pair in sets for arr in pair)*1.05
    records=[]
    for j,((a,b),title,names) in enumerate(zip(sets,['(a) Matched','(b) Unmatched','(c) HC negative control'],[['HC','MDD'],['HC','MDD'],['HC val','HC test']])):
        ax=fig.add_subplot(gs[0,j]);rain(ax,[np.abs(a).mean(1),np.abs(b).mean(1)],names,colors=[HC,MDD] if j<2 else [HC,NEG],ylabel='Subject mean |z|');ax.set_ylim(0,ymax);ax.set_title(title)
        for arr,name,ids in zip((a,b),names,identities[j]):
            records.extend(dict(subject=sid,comparison=title[4:],group=name,mean_abs_z=float(np.abs(r).mean()),within_subject_z_std=float(r.std())) for sid,r in zip(ids,arr))
    _,order=partition();profiles=[];names=[];profile_meta=[]
    for a,g in [(m,'MDD'),(h,'HC')]:
        ix=np.argsort(np.abs(a).mean(1))
        for rank,label in [(-1,'max'),(-2,'second'),(len(a)//2,'median')]:
            i=ix[rank];profiles.append(a[i]);names.append(g+' '+label);profile_meta.append({'group':g,'label':label,'subject':pairs[i][g.lower()]})
    ax=fig.add_subplot(gs[1,:2]);mat=np.array(profiles)[:,order];limit=np.abs(mat).max();im=ax.imshow(mat,aspect='auto',cmap=DIV_CMAP,vmin=-limit,vmax=limit);ax.set_yticks(range(6),names);ax.set_xlabel('116 ROIs in AAL network order');ax.set_title('(d) Deterministic illustrative individuals');fig.colorbar(im,ax=ax,label='z',fraction=.025,pad=.02)
    ax=fig.add_subplot(gs[1,2]);rain(ax,[h.std(1),m.std(1)],['HC','MDD'],ylabel='Within-subject z SD');ax.set_title('(e) Matched ROI dispersion')
    emit(fig,out,'fig_f8_heterogeneity',{'subject_statistics':records,'profiles':np.array(profiles).tolist(),'profile_metadata':profile_meta},'F8｜个体异质性（T0 描述性）。上排同尺度展示站点匹配、未匹配与 HC val/test 负控的逐被试 mean|z| 分布；点为被试，白点为中位数。下排按 mean|z| 确定性选取各组最大、次大及中位个体的完整 116 ROI 剖面，共享对称色阶；这些个体为明确选择的示例，不代表典型患病定位。右侧为匹配组个体内 z 标准差。原匹配组 mean|z| 组内方差比为 0.977、p=0.274；未对本图新增统计检验。')


def f9(out,atlas):
    c=pd.read_csv(RUN1/'clinical_partial_corr.csv');r=pd.read_csv(STATS/'clinical_residuals.csv');summary=json_read(STATS/'summary.json')
    networks=sorted(c.network.unique());scores=['HAMD','HAMA'];mat=c.pivot(index='network',columns='score',values='partial_r').loc[networks,scores]
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),gridspec_kw={'width_ratios':[1,1.4]})
    im=axes[0].imshow(mat,cmap=DIV_CMAP,vmin=-.15,vmax=.15,aspect='auto');axes[0].set_xticks([0,1],scores);axes[0].set_yticks(range(len(networks)),networks);axes[0].set_title('(a) Archived partial correlations')
    for i,n in enumerate(networks):
        for j,s in enumerate(scores):
            row=c[(c.network==n)&(c.score==s)].iloc[0];axes[0].text(j,i,f'{row.partial_r:+.3f}\nn={int(row.n)}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes[0],label='Partial r',fraction=.035,pad=.03)
    rr=r[(r.network=='LN')&(r.score=='HAMA')];axes[1].scatter(rr.resid_score,rr.resid_net_z,color=MDD,s=14,alpha=.35,lw=0)
    beta=np.polyfit(rr.resid_score,rr.resid_net_z,1);xx=np.array([rr.resid_score.min(),rr.resid_score.max()]);axes[1].plot(xx,np.polyval(beta,xx),color=INK,lw=1)
    axes[1].set_xlabel('HAMA residual (Age / Sex / site adjusted)');axes[1].set_ylabel('LN mean z residual');axes[1].set_title('(b) LN–HAMA: r=-0.135; p=0.0081')
    fig.text(.5,.01,'Uncorrected descriptive association; HAMA complete cases: 422/1504 matched, 569/2253 all MDD',ha='center',fontsize=8,color=MDD)
    fig.tight_layout(rect=(0,.06,1,1))
    emit(fig,out,'fig_f9_clinical',{'correlations':c.to_dict('records'),'LN_HAMA_residuals':rr.to_dict('records'),'availability':summary},'F9｜临床关联（T0/T1 描述性）。偏相关沿用原稀疏网络分组，控制年龄、性别和站点哑变量。每格标注原 r 与 n；HAMD 均无明显关联。右图为同一完整观测集的 LN−HAMA 残差散点，r=−0.13489、未校正 p=0.00812、n=422，全部残差 r/n 已与原表逐项核对。HAMA 全 MDD 可用率 569/2253=25.3%，匹配组为 422/1504=28.1%；未检验缺失机制，不能把这条弱关联当作病理或临床预测证据。未进行插补或新增组间推断。')


def f10(out,atlas):
    cu=pd.read_csv(ROOT/'results/pathology_analysis/condition_usage.csv');fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    rel=cu.delta_pred_relative*100;axes[0].hist(rel,bins=24,color=HC,alpha=.6);axes[0].set_xscale('log');axes[0].axvline(rel.median(),color=INK,ls='--');axes[0].set_xlabel('Prediction change / signal scale (%)');axes[0].set_ylabel('Subjects');axes[0].set_title(f'(a) Median {rel.median():.3f}%')
    x,y=cu.mae_cond,cu.mae_zero_cond;lims=[min(x.min(),y.min())*.95,max(x.max(),y.max())*1.05];axes[1].scatter(x,y,color=HC,s=9,alpha=.4);axes[1].plot(lims,lims,color=INK,ls='--');axes[1].set_xlabel('MAE with conditions');axes[1].set_ylabel('MAE with conditions zeroed');axes[1].set_title('(b) Paired subject MAE')
    qs=cu[['ode_film_gamma','ode_film_beta']].quantile([.25,.5,.75])
    axes[2].errorbar([0,1],qs.loc[.5],yerr=np.array([qs.loc[.5]-qs.loc[.25],qs.loc[.75]-qs.loc[.5]]),fmt='o',color=HC,capsize=4);axes[2].set_xticks([0,1],['ODE gamma','ODE beta']);axes[2].set_ylabel('L2 norm (median and IQR)');axes[2].set_title('(c) Conditioning magnitudes')
    axes[2].text(.5,.04,'Encoder AdaRMS: not enabled',transform=axes[2].transAxes,ha='center',fontsize=8)
    fig.tight_layout()
    emit(fig,out,'fig_f10_condition_usage',cu,'F10｜条件使用终裁（T0）。HC val 222 人，条件引起的预测差相对真实单步变化尺度的中位数为 0.458%；注意分母为 mean|true−x_last|，不是 BOLD 振幅。条件零化按原全缺失编码定义，平均 MAE 从 0.081428 变为 0.081354。右图展示 ODE FiLM 的 L2 范数中位数与四分位区间，参数大小不等同输出贡献。编码器 AdaRMS 未启用，原数据为缺失而非零，不绘制虚构分布。此图复用 B5 终裁，不重启敏感性分支。')


def infer_samples():
    df=pd.read_csv(INFER/'generation_metrics.csv')
    arrays=[]
    for _,r in df.iterrows():
        with np.load(INFER/f'{r.group}_{r.subject}.npz') as d:
            arrays.append({k:d[k].copy() for k in d.files})
    return df,arrays


def f3(out,atlas):
    df,arrays=infer_samples();hc=np.flatnonzero(df.group=='hc');idx=int(hc[np.argmin(np.abs(df.loc[hc,'fc_r']-df.loc[hc,'fc_r'].median()))]);s=arrays[idx]
    roi=[0,28,30,60];names=labels()['micro命名'];fig=plt.figure(figsize=(14,10));outer=fig.add_gridspec(3,1,height_ratios=[1.5,1.2,1.1],hspace=.55)
    gs=outer[0].subgridspec(2,2,hspace=1.0,wspace=.22)
    for j,k in enumerate(roi):
        ax=fig.add_subplot(gs[j//2,j%2]);ax.plot(np.arange(-15,1),s['history'][-16:,k],color='#aaa',lw=.8,label='History');ax.plot(np.arange(1,33),s['true'][:,k],color=HC,lw=1.2,label='Observed');ax.plot(np.arange(1,33),s['pred'][:,k],color=MDD,lw=1.2,ls='--',label='Predicted');ax.axvspan(.5,8.5,color='#ddd',alpha=.35);ax.set_xlabel('TR relative to forecast origin');ax.set_ylabel('Calibrated BOLD');ax.set_title(f'(a{j+1}) {names.iloc[k]}',fontsize=9)
        if j==0:ax.legend(ncol=3,fontsize=7,loc='upper left')
    gs=outer[1].subgridspec(1,4,wspace=.4);_,order=partition()
    for j,(key,title) in enumerate([('fc_true','(b1) Observed FC'),('fc_pred','(b2) Predicted FC'),('diff','(b3) Predicted - observed')]):
        ax=fig.add_subplot(gs[j]);mat=s[key] if key!='diff' else s['fc_pred']-s['fc_true'];lim=1 if key!='diff' else 2
        im=ax.imshow(mat[np.ix_(order,order)],cmap=DIV_CMAP,vmin=-lim,vmax=lim);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=9);fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
    ax=fig.add_subplot(gs[3]);iu=np.triu_indices(116,1);ax.scatter(s['fc_true'][iu],s['fc_pred'][iu],s=1,color=HC,alpha=.15,rasterized=True);ax.plot([-1,1],[-1,1],color=INK,ls='--',lw=.8);ax.set_xlabel('Observed FC');ax.set_ylabel('Predicted FC');ax.set_title(f'(b4) Edge r={s["fc_r"]:.3f}',fontsize=9)
    gs=outer[2].subgridspec(1,3,wspace=.35)
    for j,(key,title) in enumerate([('fc_r','(c1) FC edge correlation'),('psd_kl','(c2) Welch spectral-shape KL')]):
        ax=fig.add_subplot(gs[j]);rain(ax,[df.loc[df.group==g,key] for g in ('hc','mdd')],['HC (20)','MDD (20)'],ylabel='FC edge r' if key=='fc_r' else 'Mean ROI KL(true || predicted)');ax.set_title(title,fontsize=9)
    ax=fig.add_subplot(gs[2]);freq=s['frequency']
    for key,c,label,ls in [('psd_true',HC,'Observed','-'),('psd_pred',MDD,'Predicted','--')]:
        values=np.stack([a[key].mean(1) for a in arrays]);ax.plot(freq,values.mean(0),color=c,label=label,ls=ls)
    ax.set_yscale('log');ax.set_xlabel('Frequency (cycles/TR)');ax.set_ylabel('Normalized power (DC removed)');ax.legend();ax.set_title('(c3) Average ROI spectral shape',fontsize=9)
    fig.suptitle(f'HC reference model: descriptive K64 to H32 extrapolation | example {df.iloc[idx].subject}',fontsize=12,weight='bold',y=.99)
    data={'metrics':df.to_dict('records'),'example_subject':df.iloc[idx].subject,'example_selection':'HC subject nearest median FC-r; ROIs fixed a priori [1,29,31,61]','roi_indices_1based':(np.array(roi)+1).tolist(),'pred':s['pred'].tolist(),'true':s['true'].tolist(),'fc_pred':s['fc_pred'].tolist(),'fc_true':s['fc_true'].tolist(),'frequency':freq.tolist(),'psd_true_mean':np.stack([a['psd_true'].mean(1) for a in arrays]).mean(0).tolist(),'psd_pred_mean':np.stack([a['psd_pred'].mean(1) for a in arrays]).mean(0).tolist()}
    emit(fig,out,'fig_f3_generation',data,'F3｜HC 参照模型的生成质量补充（T2 描述性），不是 F2 的历史 H32 权重。按固定种子在原站点匹配对中抽取双方均位于 val/test 的 20 对，每人最早 K64 预测未来 H32，使用前 32 TR 固定校准、继承 hc3 条件统计。此权重快照 forecast_offsets=1、cpm_horizon=8，H32 超出其辅助训练时程，为外推。轨迹示例选 HC 组 FC-r 最接近中位数者，ROI 固定为 1/29/31/61，灰带标记 H1–8；FC 在真实和预测未来 32 TR 分别计算，上三角边 Pearson r 描述相似性，差矩阵色阶为 ±2、FC 为 ±1。下排展示 40 人的 FC-r、ROI 平均 KL(true||pred) 和平均谱形；Welch nperseg=32、去 DC、逐 ROI 谱归一化、频率单位 cycles/TR，不假定所有站点 TR 秒数相同。32 点频谱和 FC 估计不稳定，且无低频临床解释；HC/MDD 分布不做新的组间检验。')


def f11(out,atlas):
    df,arrays=infer_samples();_,order=partition();net,_=partition();fig=plt.figure(figsize=(12,7));gs=fig.add_gridspec(2,3,height_ratios=[1.6,1],hspace=.45,wspace=.4)
    keys=['prior_norm','A_func','A_eff'];titles=['(a) Normalized history-|FC| prior','(b) Learned functional graph','(c) Effective graph']
    means={key:np.stack([s[key] for s in arrays]).mean(0) for key in keys};vmax=max(m.max() for m in means.values())
    for j,(key,title) in enumerate(zip(keys,titles)):
        ax=fig.add_subplot(gs[0,j]);im=ax.imshow(means[key][np.ix_(order,order)],cmap='viridis',vmin=0,vmax=vmax);ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    ax=fig.add_subplot(gs[1,:2]);block=[]
    for n in NETWORK_ORDER:
        ix=np.flatnonzero(net==n);mask=~np.eye(len(ix),dtype=bool);record={'network':n}
        for key in keys:record[key]=float(means[key][np.ix_(ix,ix)][mask].mean())
        block.append(record)
    blockdf=pd.DataFrame(block)
    for j,(key,c) in enumerate(zip(keys,[HC,NEG,MDD])):ax.bar(np.arange(7)+(j-1)*.23,blockdf[key],width=.23,label=key,color=c)
    ax.set_xticks(range(7),NETWORK_ORDER);ax.set_ylabel('Within-network mean edge weight');ax.set_title('(d) AAL display networks, diagonal excluded');ax.legend(ncol=3,fontsize=7)
    ax=fig.add_subplot(gs[1,2]);lv=df.lambda_mean.to_numpy();ax.scatter(np.arange(1,len(lv)+1),lv,color=HC,s=12);ax.set_xlabel('Subject index');ax.set_ylabel('Lambda');ax.set_ylim(max(0,lv.min()-.05),min(1,lv.max()+.05));ax.set_title(f'(e) Mean lambda {lv.mean():.4f}',fontsize=9)
    fig.suptitle('Learned graph structure: functional input, not measured anatomical SC',weight='bold',y=.99)
    emit(fig,out,'fig_f11_graph',{'subject_metrics':df.to_dict('records'),'mean_matrices':{k:v.tolist() for k,v in means.items()},'block_means':block,'network_order':NETWORK_ORDER},'F11｜学习图结构（T2 描述性）。展示同一批 40 人平均的历史绝对 FC 度归一化先验、学习功能图 A_func 及有效图 A_eff，按 AAL 完整网络标签排序，三矩阵共享色阶。当前 checkpoint 的 graph_source=history_fc，所谓 SC 位置实际输入可见历史功能耦合图，不是独立实测解剖连接。有效图还包含可学习低秩残差、自环及 Sinkhorn 归一化，不能只用 λ 混合公式概括最终输出。下排展示剔除对角后的网络块均值和 λ；λ 若为全局参数，其跨人恒定不代表个体结构相同。图仅说明模型结构性质，不构成生理连接或因果机制证据。')


def f12(out,atlas):
    from nilearn.plotting import plot_glass_brain
    imp=json_read(INFER/'feature_importance_heldout.json');table=labels();v=np.array(imp['roi_importance']);q=np.array(imp['roi_pvalues_fdr']);se=np.array(imp['roi_std_errors']);net=table['对应网络'].to_numpy()
    d=pd.DataFrame({'roi_index':np.arange(1,117),'label':table['micro命名'],'network':net,'delta_mae':v,'se_batch':se,'p':imp['roi_pvalues'],'q':q});fig=plt.figure(figsize=(13,6));gs=fig.add_gridspec(2,2,width_ratios=[1,1.2],hspace=.45,wspace=.55)
    sub=gs[0,0].subgridspec(1,2,width_ratios=[18,1],wspace=.4);ax=fig.add_subplot(sub[0]);lim=float(np.abs(v).max());plot_glass_brain(atlas_image(v,atlas),axes=ax,figure=fig,plot_abs=False,cmap=DIV_CMAP,vmin=-lim,vmax=lim,threshold=None,colorbar=False,display_mode='lyrz');ax.set_title('(a) Lag-1 input ROI sensitivity',fontsize=10)
    cax=fig.add_subplot(sub[1]);fig.colorbar(ScalarMappable(norm=Normalize(-lim,lim),cmap=DIV_CMAP),cax=cax,label='Delta MAE')
    ax=fig.add_subplot(gs[1,0]);means=[v[net==n].mean() for n in NETWORK_ORDER];ax.bar(range(7),means,color=[NETWORK_COLORS[n] for n in NETWORK_ORDER]);ax.set_xticks(range(7),NETWORK_ORDER);ax.set_ylabel('Mean ROI delta MAE');ax.set_title('(b) AAL display-network mean')
    ax=fig.add_subplot(gs[:,1]);top=np.argsort(v)[-20:];y=np.arange(20);ax.barh(y,v[top],color=[NETWORK_COLORS[n] for n in net[top]],alpha=.8);ax.errorbar(v[top],y,xerr=se[top],fmt='none',ecolor=INK,capsize=2,lw=.8);ax.set_yticks(y,[table['micro命名'].iloc[i]+(' *' if q[i]<.05 else '') for i in top],fontsize=8);ax.set_xlabel('Delta lag-1 MAE (batch SEM)');ax.set_title('(c) Top-20; * approximate BH q<0.05')
    fig.suptitle(f'Input importance: {imp["n_tasks"]} held-out subjects, {imp["n_batches"]} batches; structural description',weight='bold',y=.99)
    emit(fig,out,'fig_f12_roi_importance',d,'F12｜ROI 置换重要性（T2，小样本结构性描述）。复用 next_point_eval 的跨样本 ROI 历史置换与逐批单侧 t 近似口径，在同批 40 人、4 个各 10 人批次内分别置换一个 ROI 的 K64 历史；批次不跨 HC/MDD。保留目标被试原 hc3 条件、前缀校准及 profile，图按扰动后历史重新计算。ΔMAE 为主输出 lag-1 的误差增量，不是 H32 或疾病分类重要性。星号仅指 116 ROI 的近似 BH-FDR q<0.05，误差线为批间 SEM，不是 95% CI；4 批且每 ROI 单次随机置换的稳定性有限。脑图、网络条形和 Top-20 使用同一向量，不作为病理定位或临床因果证据。')


FIGURES={'fig_f0a_pipeline':f0a,'fig_f0b_composition':f0b,'fig_f1_gate':f1,'fig_f2_horizon':f2,'fig_f3_generation':f3,'fig_f5_pathology_main':f5,'fig_f6_site_confounding':f6,'fig_f7_separability':f7,'fig_f8_heterogeneity':f8,'fig_f9_clinical':f9,'fig_f10_condition_usage':f10,'fig_f11_graph':f11,'fig_f12_roi_importance':f12}


def report(out:Path) -> None:
    t1=json_read(STATS/'summary.json');gen=pd.read_csv(INFER/'generation_metrics.csv');imp=json_read(INFER/'feature_importance_heldout.json');manifest=json_read(INFER/'manifest.json')
    pd.read_csv(RUN1/'param_dim_table.csv').to_csv(out/'table_d6_profile.csv',index=False)
    param=json_read(RUN1/'param_group_tests.json')
    (out/'table_d6_profile_summary.json').write_text(json.dumps(param,ensure_ascii=False,indent=2))
    pd.DataFrame([{'analysis':'I4/T08','quantity':'far-horizon ODE MAE increase under old-history-block shuffle','value_percent':14.2,'source':'docs/实验结论与证据汇总.md T08','status':'archival summary only; no blockwise arrays retained; no new shuffle inference'}]).to_csv(out/'table_i4_t08.csv',index=False)
    pd.DataFrame([{'analysis':'T04','coefficient':.948788,'test_far_mae_gain_percent':.0228,'decision':'not deployed'}, {'analysis':'S4/T18','coefficient':None,'test_far_mae_gain_percent':None,'decision':'archived negative; no rerun'}]).to_csv(out/'table_p5_calibration.csv',index=False)
    methods='''本次执行默认 T0/T1/T2，不重训、不调整模型/训练/配置、不写入权重、不执行 git 提交。所有新增产物位于带 run_20261026 的独立目录。既有 6 张图作为版式和数据入口参考，因标签及口径需要纠正，本目录重新渲染，原图不覆盖。

主分析 checkpoint 为 HC base，SHA256：`'''+manifest['checkpoint_sha256']+'''`。F2 单独引用 final 历史 H32 权重及原正式评测，不重新选优。T2：seed=2024，sampling_seed=20261009，在站点匹配对中选择双方均为 val/test 的 20 对，首次 K64→H32；CPU 两线程，不使用训练模式。严格 test/test 匹配仅 11 对，因此使用双方均未进入训练集的 val/test 联合候选；具体被试和站点见推理 manifest。

T1 复用已完成的 D2/D3 产物，独立核验原始均值差、AUC、临床 r/n；ROC 500 次站内匹配对 bootstrap 仅重采样固定 OOF 分数，不重训分类器。D2 的 raw delta 与既有站点加权置换检验不是相同估计量，图中不将两者配对推断。HC val/test 负控人数小于主比较，不应将其效应大小当作等精度检测阈值。

推断性结论仅引用原 run1；本次新数值均为描述性。完整 AAL116 网络分组只用于显示和图结构汇总；原病理网络检验及临床关联保留其稀疏 ROI 分组，102/116 ROI 属于 Other。未修正原分组，也未据新分组重跑疾病推断。
'''
    metric_table=gen.groupby('group')[['fc_r','psd_kl','mae','lambda_mean']].agg(['mean','median','min','max'])
    metric_table.to_csv(out/'table_t2_summary.csv')
    paragraphs=['# 论文分析交付（2026-10-26）','',methods,'','## 主要结果','',f'- 原 HC val 滚动单步改善 86.54%，p=0.00010；匹配病理检验仍为 0/116 ROI 通过 FDR。',f'- 未匹配全脑 raw Δz=+0.06880，匹配后 −0.02281；未据此新增病理推断。',f'- 静态 FC OOF AUC={t1["auc"]:.5f}，固定 OOF bootstrap 95% CI [{t1["auc_ci95"][0]:.5f}, {t1["auc_ci95"][1]:.5f}]。',f'- LN–HAMA 原 r=−0.13489，p=0.00812，n=422，未多重校正。缺失机制未判定。',f'- T2 ROI 置换近似 FDR 标记 {len(imp["significant_roi_indices"])}/116；基于 4 批、每 ROI 单次随机置换，不能称稳定生物标志物。','']
    for g in ('hc','mdd'):
        d=gen[gen.group==g];paragraphs.append(f'- {g.upper()} 20 人：FC-r 中位 {d.fc_r.median():.4f}，谱形 KL 中位 {d.psd_kl.median():.4f}，H32 MAE 中位 {d.mae.median():.4f}。这是 HC 参照权重（快照 cpm_horizon=8）的长程外推，不能与 F2 跨权重数值直接比较。')
    iu=np.triu_indices(116,1)
    _,a=infer_samples()
    fc_pred_med=float(np.median([np.median(s['fc_pred'][iu]) for s in a]))
    fc_true_med=float(np.median([np.median(s['fc_true'][iu]) for s in a]))
    paragraphs.append(f'- 本批 H32 外推呈共同漂移：先求每人的非对角 FC 中位数，再跨人取中位数，预测为 {fc_pred_med:.4f}，真实为 {fc_true_med:.4f}。此现象限制生成质量，不能用单步精度替代长程仿真真实性。')
    paragraphs+=['','## 图表与中文图注','']
    for name,caption in CAPTIONS.items():
        paragraphs += [f'### {name}','',f'![{name}]({name}.png)','',caption,'',f'[矢量 PDF]({name}.pdf) · [数据]({name}.csv)' if (out/f'{name}.csv').exists() else f'[矢量 PDF]({name}.pdf) · [数据]({name}.json)','']
    paragraphs+=['## 附表与明确缺项','',
        '- D6：table_d6_profile.csv/summary.json 原表；32 维中 dim 20（零基）q=0.01280，Δ=0.005635，原未中心化 SVD 首方向能量占 99.36%。该能量占比不能等同中心化个体差异解释率。因未保留参数线的未匹配与负控逐维数据，且支线已关闭，不补跑、不单绘缺少同类对照的参数组间图。',
        '- I4：table_i4_t08.csv 引用原 T08 +14.2%，未复算可选历史块曲线，不绘造分块数据的 F13。',
        '- F4/T3 与 F14/I5：按要求不执行。',
        '- F1 置换 null 样本未保留，不用模拟直方图替代。编码器 AdaRMS 未启用，F10 明确显示不可用。',
        '- F11 没有独立实测解剖 SC；按实际 history-FC 先验展示，不制造 SC 面板。',
        '- AAL116.xlsx 的 25/26 行 micro 名与缓存 SPM12 atlas 不同（Mid/Med）；显示标签按 atlas 与 run1 的 Frontal_Med_Orb_L/R 对齐，原表未修改；全部 116 行经名称顺序核验。',
        '- 原 inference_twin.TwinSession.init_state 向模型传入 [B,F,K]，模型接口要求 [B,K,F]，本批前向已复现尺寸错误；本次未改该既有入口，使用其校准/条件/profile 辅助函数并调用训练同路 model.forward。未验证持续状态 rollout。',
        '- Welch 仅 32 未来 TR，不推断低频临床谱效应；频率用 cycles/TR，未假设跨站点 TR 秒数相同。',
        '- 本次未核查计划中的外部文献正文，未把其中未核实的引文写作新增论据；论文引用仍需原文核对。','',
        '## 复现命令','',
        '从仓库根目录运行，输出路径必须为新的独立目录；下列为本次实际命令（环境变量仅控制缓存和 CPU 线程）：','',
        '```bash',
        'MPLCONFIGDIR=/tmp/paper-mpl OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m analysis.paper_figs.run_20261026.t1_stats --out-dir results/paper_stats/run_20261026',
        'MPLCONFIGDIR=/tmp/paper-mpl OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m analysis.paper_figs.run_20261026.t2_infer --out-dir results/paper_infer/run_20261026 --resume',
        'MPLCONFIGDIR=/tmp/paper-mpl OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m analysis.paper_figs.run_20261026.build --out-dir results/paper_figures/run_20261026',
        '```','',
        'build 支持 --stats-dir、--infer-dir、--atlas-dir 指向同契约输入，支持 --figure 单图重渲染至新目录。单图同名 .py 为仓库内调用启动脚本，不是脱离仓库的独立数据包。','',
        '## 样例借鉴','',
        '实际查看图库 raincloud-median-badges、oa-tech-pone-0258309-f05、oa-bio-plos-pcbi-1011915-f05。首选借鉴雨云的原始点+半小提琴+中位点，森林图借鉴克制点区间，FC/图结构借鉴等尺寸矩阵与统一色阶。HC 用样例蓝色 #2997c9；MDD 按计划改为色盲友好橙色 #D55E00；网络色复用原 visualize_roi_importance。备选及与成品的对照图见 reference_comparison.png。图库样例含其自身示例数据，不是本项目结果。本次未向图库收图。','']
    if (out/'CHECKS.json').exists():
        paragraphs += ['## 基础检查','', '[检查明细](CHECKS.json)。13 张 PNG 均为 300 dpi，PDF/数据/图注/启动脚本配套完整；报告链接可解析。40 人数组的维度、有限值、FC/MAE 和归一化谱已核验；116 ROI 重要性与逐批差值、FDR 一致；抽取 1 名 HC 重新前向与保存结果在 1e-6 内一致。原 92 个相关文件哈希一致。静态语法检查通过。未运行训练、完整实验或全量回归。', '', '[全部图览](overview.png) · [图库并排对照](reference_comparison.png)', '']
    (out/'REPORT.md').write_text('\n'.join(paragraphs))
    (out/'CAPTIONS_ZH.md').write_text('\n\n'.join(CAPTIONS.values())+'\n')
    sources=[ROOT/'docs/实验结论与证据汇总.md',ROOT/'docs/项目状态与工作范围.md',ROOT/'data/AAL116.xlsx',RUN1/'group_tests.json',RUN1/'roi_fdr_table.csv',RUN1/'clinical_partial_corr.csv',ROOT/'results/paper_figures/complete_20261026/sources/h32/manifest.json',ROOT/'results/paper_stats/run_20261026/fc_ridge_oof.npz']
    sources+=list(Path(__file__).parent.glob('*.py'))
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out/'provenance.json').write_text(json.dumps({'sources':provenance,'infer_manifest':manifest,'figure_ids':list(CAPTIONS),'tier_map':{'T0':['F0a','F0b','F1','F2','F5','F8','F10','D6 table','I4 table'],'T1':['F6','F7','F9'],'T2':['F3','F11','F12']},'excluded':['T3/F4','I5/F14'],'output_dpi':300},ensure_ascii=False,indent=2))


def main(default_figure=None) -> None:
    global STATS,INFER
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--stats-dir',type=Path,default=STATS)
    ap.add_argument('--infer-dir',type=Path,default=INFER)
    ap.add_argument('--atlas-dir',type=Path,default=Path.home()/'nilearn_data/aal_SPM12/aal')
    ap.add_argument('--figure',choices=list(FIGURES),default=default_figure)
    ap.add_argument('--resume',action='store_true',help='Reuse only complete figures already produced in this run')
    ns=ap.parse_args();STATS,INFER=ns.stats_dir,ns.infer_dir;ns.out_dir.mkdir(parents=True,exist_ok=True);apply_style()
    for name,fn in FIGURES.items():
        if ns.figure is None or name==ns.figure:
            complete=all((ns.out_dir/f'{name}.{ext}').exists() for ext in ('png','pdf','caption.md','py')) and any((ns.out_dir/f'{name}.{ext}').exists() for ext in ('csv','json'))
            if ns.resume and complete:
                CAPTIONS[name]=(ns.out_dir/f'{name}.caption.md').read_text().strip()
            else:
                fn(ns.out_dir,ns.atlas_dir)
    if ns.figure is None:report(ns.out_dir)

if __name__=='__main__':
    main()
