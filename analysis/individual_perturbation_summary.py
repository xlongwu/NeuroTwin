"""Subject-weighted ROI permutation summaries and figures for completed caches."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.h18_graph import NETWORKS, ROOT


def bootstrap_mean_ci(values: np.ndarray, seed=2024, repeats=10000):
    rng = np.random.default_rng(seed)
    n = len(values)
    result = []
    for start in range(0, repeats, 250):
        indices = rng.integers(0, n, size=(min(250, repeats-start), n))
        result.append(values[indices].mean(axis=1))
    return np.quantile(np.concatenate(result), [0.025, 0.975], axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=('hc', 'mdd'), required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.cache_dir / args.group
    m = json.loads((root/'数据清单.json').read_text())
    if m['n_roi'] != 116 or m['repeats'] != 3:
        raise ValueError('Complete 116 ROI, three-repeat cache required')
    missing = [s for s in m['subjects'] if not (root/f'{s}.npz').is_file()]
    if missing:
        raise ValueError(f'{len(missing)} subjects missing; refusing partial summary')
    atlas = pd.read_excel(ROOT/'data/AAL116.xlsx')
    networks = atlas['对应网络'].to_numpy()
    if len(atlas) != 116 or set(networks) != set(NETWORKS):
        raise ValueError('AAL network mapping mismatch')
    dest = args.out_dir / args.group
    dest.mkdir(parents=True, exist_ok=True)
    subject_effect, subject_early, subject_late = [], [], []
    subject_self, subject_other, subject_response, subject_abs_response = [], [], [], []
    base_mae, coverage, max_cpu_diff = [], [], []
    for sid in m['subjects']:
        with np.load(root/f'{sid}.npz') as d:
            base, true, pert = d['baseline'], d['true'], d['perturbed']
            anchors = d['cutoff']
            indices = d['permutation_indices']
            if indices.shape != (len(anchors),116,3,64) or not (indices[...,63]==63).all():
                raise ValueError(f'{sid}: invalid permutation indices')
            for block in range(8):
                lo = 8*block
                if not np.all(np.sort(indices[...,lo:lo+8],axis=-1)==np.arange(lo,lo+8)):
                    raise ValueError(f'{sid}: permutation crosses block boundary')
            if pert.shape != (len(anchors), 116, 3, 116) or not all(np.isfinite(x).all() for x in (base, true, pert)):
                raise ValueError(f'{sid}: invalid cached predictions')
            error = np.abs(pert - true[:, None, None, :])
            reference = np.abs(base-true)
            delta = error.mean(axis=-1) - reference.mean(axis=-1)[:, None, None]
            subject_effect.append(delta.mean(axis=(0, 2)))
            middle = len(anchors)//2
            subject_early.append(delta[:middle].mean(axis=(0, 2)))
            subject_late.append(delta[middle:].mean(axis=(0, 2)))
            diag = np.arange(116)
            own_values = error[:, diag, :, diag].transpose(1, 0, 2)
            own_error = own_values - reference[:, :, None]
            subject_self.append(own_error.mean(axis=(0, 2)))
            subject_other.append(((error.sum(axis=-1)-own_values)/115
                                  - (reference.sum(axis=-1)[:, None, None]-reference[:, :, None])/115).mean(axis=(0, 2)))
            response = pert - base[:, None, None, :]
            subject_response.append(response.mean(axis=(0, 2)))
            subject_abs_response.append(np.abs(response).mean(axis=(0, 2)))
            base_mae.append(reference.mean())
            overlap = sum(abs(int(a)-int(b)) < 64 for a in anchors[:middle] for b in anchors[middle:])
            coverage.append({'subject':sid, 'n_anchors':len(anchors), 'early_anchors':middle,
                             'late_anchors':len(anchors)-middle, 'cross_period_pairs':middle*(len(anchors)-middle),
                             'overlapping_history_pairs':overlap})
            max_cpu_diff.append(float(d['cpu_baseline_max_abs']))
    values = np.stack(subject_effect).astype(np.float64)
    early, late = np.stack(subject_early), np.stack(subject_late)
    own, other = np.stack(subject_self), np.stack(subject_other)
    signed, absolute = np.stack(subject_response), np.stack(subject_abs_response)
    base_mae = np.array(base_mae, dtype=float)
    ci = bootstrap_mean_ci(values)
    subject_ratio = np.divide(values, base_mae[:, None], out=np.full_like(values, np.nan), where=base_mae[:, None]>0)
    roi = pd.DataFrame({'roi':np.arange(1,117), 'name':atlas['abbr'], 'network':networks,
                        'n_subjects':len(values), 'baseline_mae_mean':base_mae.mean(),
                        'perturbed_mae_mean':base_mae.mean()+values.mean(axis=0),
                        'delta_mae_mean':values.mean(axis=0), 'delta_mae_median':np.median(values,axis=0),
                        'ci_low':ci[0], 'ci_high':ci[1], 'positive_fraction':(values>0).mean(axis=0),
                        'relative_delta_subject_mean':np.nanmean(subject_ratio,axis=0),
                        'self_delta_mean':own.mean(axis=0), 'other_delta_mean':other.mean(axis=0)})
    roi = roi.sort_values('delta_mae_mean', ascending=False).reset_index(drop=True)
    roi.insert(0,'rank',np.arange(1,117))
    roi.to_csv(dest/'ROI汇总.csv',index=False)
    rows = []
    for i, sid in enumerate(m['subjects']):
        for j in range(116):
            rows.append({'subject':sid,'roi':j+1,'network':networks[j], 'baseline_mae':base_mae[i],
                         'delta_mae':values[i,j], 'relative_delta':subject_ratio[i,j],
                         'self_delta':own[i,j], 'other_delta':other[i,j],
                         'early_delta':early[i,j], 'late_delta':late[i,j]})
    pd.DataFrame(rows).to_csv(dest/'逐被试_ROI.csv',index=False)
    nv = np.stack([values[:, networks == net].mean(axis=1) for net in NETWORKS], axis=1)
    nci = bootstrap_mean_ci(nv)
    net = pd.DataFrame({'network':NETWORKS, 'n_roi':[int((networks==n).sum()) for n in NETWORKS],
                        'n_subjects':len(values), 'mean':nv.mean(axis=0), 'median':np.median(nv,axis=0),
                        'ci_low':nci[0], 'ci_high':nci[1], 'positive_fraction':(nv>0).mean(axis=0),
                        'relative_delta_subject_mean':np.nanmean(np.divide(nv,base_mae[:,None],
                            out=np.full_like(nv,np.nan),where=base_mae[:,None]>0),axis=0)})
    net.to_csv(dest/'网络汇总.csv', index=False)
    pd.DataFrame(nv,columns=NETWORKS,index=m['subjects']).rename_axis('subject').to_csv(dest/'逐被试_网络.csv')
    stability = pd.DataFrame(coverage)
    stability['rank_spearman'] = [spearmanr(a,b).statistic if np.std(a)>0 and np.std(b)>0 else np.nan for a,b in zip(early,late)]
    stability['direction_agreement'] = (np.sign(early)==np.sign(late)).mean(axis=1)
    stability['baseline_mae'] = base_mae
    stability['cpu_gpu_baseline_max_abs'] = max_cpu_diff
    stability.to_csv(dest/'逐被试_早晚稳定性.csv',index=False)
    early_network=np.stack([early[:,networks==net].mean(axis=1) for net in NETWORKS],axis=1)
    late_network=np.stack([late[:,networks==net].mean(axis=1) for net in NETWORKS],axis=1)
    network_stability=[]
    for i,sid in enumerate(m['subjects']):
        for j,net_name in enumerate(NETWORKS):
            network_stability.append({'subject':sid,'network':net_name,'early_delta':early_network[i,j],
                                      'late_delta':late_network[i,j],
                                      'same_direction':bool(np.sign(early_network[i,j])==np.sign(late_network[i,j]))})
    pd.DataFrame(network_stability).to_csv(dest/'逐被试_网络早晚.csv',index=False)
    response_rows = []
    for i, source in enumerate(NETWORKS):
        for j, target in enumerate(NETWORKS):
            ii,jj = np.meshgrid(np.flatnonzero(networks==source),np.flatnonzero(networks==target),indexing='ij')
            ok=ii!=jj
            a=signed[:,ii[ok],jj[ok]].mean(axis=1)
            b=absolute[:,ii[ok],jj[ok]].mean(axis=1)
            response_rows.append({'source_network':source,'target_network':target,
                                  'n_nonself_edges':int(ok.sum()),'n_subjects':len(values),
                                  'signed_mean':a.mean(),'absolute_mean':b.mean()})
    response_df = pd.DataFrame(response_rows)
    response_df.to_csv(dest/'网络响应.csv',index=False)
    diagonal = np.arange(116)
    pd.DataFrame({'subject':m['subjects'], 'self_abs_response':absolute[:,diagonal,diagonal].mean(axis=1),
                  'other_abs_response':(absolute.sum(axis=(1,2))-absolute[:,diagonal,diagonal].sum(axis=1))/(116*115)}).to_csv(dest/'逐被试_自身与跨ROI响应.csv',index=False)
    plt.rcParams.update({'font.size':8, 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,4,figsize=(14,14),sharex=True,constrained_layout=True)
    for col, ax in enumerate(axes):
        subset=roi.iloc[col*29:(col+1)*29].iloc[::-1]; y=np.arange(len(subset))
        ax.hlines(y,subset.ci_low,subset.ci_high,color='#87ceeb',lw=.9)
        ax.scatter(subset.delta_mae_mean,y,color='#2b438d',s=9,zorder=3)
        ax.axvline(0,color='#999999',lw=.7); ax.set_yticks(y,subset['name'],fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_xlabel('ΔMAE (perturbed − baseline)')
    fig.suptitle(f'{args.group.upper()}: all source ROIs, subject mean and 95% bootstrap CI; n={len(values)}')
    for ext in ('png','pdf'): fig.savefig(dest/f'ROI排序.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    rng=np.random.default_rng(2024)
    ax.boxplot([nv[:,i] for i in range(7)],patch_artist=True,showfliers=False,
               boxprops={'facecolor':'#87ceeb','edgecolor':'#2b438d'},medianprops={'color':'#2b438d'})
    for j in range(7): ax.scatter(j+1+rng.uniform(-.15,.15,len(values)),nv[:,j],s=5,color='#2b438d',alpha=.35)
    ax.axhline(0,color='#999999',lw=.7);ax.set_xticks(range(1,8),NETWORKS);ax.set_ylabel('Within-subject network mean ΔMAE')
    ax.set_title(f'{args.group.upper()}: source-network distributions; n={len(values)}')
    for ext in ('png','pdf'): fig.savefig(dest/f'网络分布.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),constrained_layout=True)
    for ax,key,title in zip(axes,('signed_mean','absolute_mean'),('Signed response','Absolute response')):
        grid=response_df.pivot(index='source_network',columns='target_network',values=key).reindex(index=NETWORKS,columns=NETWORKS).to_numpy()
        if key=='signed_mean':
            limit=np.abs(grid).max(); im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit)
        else: im=ax.imshow(grid,cmap='Blues',vmin=0)
        ax.set_xticks(range(7),NETWORKS,rotation=90);ax.set_yticks(range(7),NETWORKS)
        ax.set_xlabel('Target network');ax.set_ylabel('Source network');ax.set_title(title)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle(f'{args.group.upper()}: mean prediction response, self ROI excluded; n={len(values)}')
    for ext in ('png','pdf'): fig.savefig(dest/f'网络响应.{ext}',dpi=300)
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(7,3.6),constrained_layout=True)
    rng=np.random.default_rng(2024)
    for ax,column,label in zip(axes,('rank_spearman','direction_agreement'),('ROI rank Spearman correlation','Fraction of matching ROI signs')):
        vals=stability[column].dropna().to_numpy()
        if len(vals):
            ax.boxplot([vals],positions=[1],widths=.36,patch_artist=True,showfliers=False,
                       boxprops={'facecolor':'#87ceeb','edgecolor':'#2b438d'},medianprops={'color':'#2b438d'})
            ax.scatter(1+rng.uniform(-.12,.12,len(vals)),vals,s=5,c='#2b438d',alpha=.4)
        ax.set_xticks([1],[args.group.upper()]);ax.set_ylabel(label)
    fig.suptitle(f'{args.group.upper()}: early vs late, within scan; n={len(values)}')
    for ext in ('png','pdf'):fig.savefig(dest/f'早晚稳定性.{ext}',dpi=300)
    plt.close(fig)
    subject_mean = values.mean(axis=1)
    subject_order = np.argsort(subject_mean, kind='stable')
    quantiles = np.array([.25,.5,.75])
    chosen = subject_order[np.rint(quantiles*(len(values)-1)).astype(int)]
    cases = pd.DataFrame({'quantile':quantiles,'subject':np.array(m['subjects'])[chosen],
                          'subject_mean_delta':subject_mean[chosen]})
    cases.to_csv(dest/'案例选择.csv',index=False)
    network_order=np.concatenate([np.flatnonzero(networks==net_name) for net_name in NETWORKS])
    case_values=values[chosen][:,network_order]
    fig, ax=plt.subplots(figsize=(10,2.8),constrained_layout=True)
    limit=np.abs(case_values).max()
    image=ax.imshow(case_values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.set_yticks(range(3),['25th percentile','Median','75th percentile'])
    ax.set_xlabel('Source ROIs grouped by network')
    bounds=np.cumsum([int((networks==net).sum()) for net in NETWORKS])
    starts=np.r_[0,bounds[:-1]]
    ax.set_xticks((starts+bounds-1)/2,NETWORKS)
    for boundary in bounds[:-1]:ax.axvline(boundary-.5,color='white',lw=.8)
    fig.colorbar(image,ax=ax,label='Within-subject ΔMAE')
    ax.set_title(f'{args.group.upper()}: illustrative subjects selected by overall mean ΔMAE')
    for ext in ('png','pdf'):fig.savefig(dest/f'个体案例.{ext}',dpi=300)
    plt.close(fig)
    report = [f'# {args.group.upper()} 个体内历史置换主分析','',
              f'测试被试 {len(values)} 人；每人 {len(anchors)} 锚点、116 个源 ROI、每 ROI 3 次置换。',
              '每次只置换单个源 ROI 的 8 TR 块内顺序，末帧固定；图由扰动后的历史重算。ΔMAE 是全脑误差增量；先对每人的锚点和重复求均值，再对被试等权汇总。相对变化为每人 ΔMAE 除以该人的基线 MAE，再跨被试求均值。区间为被试 bootstrap 95% 区间（10000 次，seed=2024），没有进行显著性检验。',
              f'基线 MAE 被试均值 {base_mae.mean():.6f}；全部源 ROI 的 ΔMAE 均值 {values.mean():+.6f}。',
              f'按源 ROI 自身目标与其余 115 个目标拆分，误差增量分别为 {own.mean():+.6f} 与 {other.mean():+.6f}；两者分别以 1 和 115 个目标加权构成全脑均值。',
              f'早晚 ROI 排名 Spearman 相关的被试中位数 {stability.rank_spearman.median():.4f}；方向一致比例的被试中位数 {stability.direction_agreement.median():.4f}。',
              f'跨早晚历史窗对中存在重叠的被试 {int((stability.overlapping_history_pairs>0).sum())}/{len(values)}；时间切分不是独立扫描复现。',
              '案例以每人 116 ROI 的平均 ΔMAE 排序，按 25%、50%、75% 的位置选取，具体被试列于案例选择表。',
              f'GPU 与保存的 CPU 基线最大绝对差 {max(max_cpu_diff):.3g}。',
              '网络响应矩阵排除源 ROI 对自身响应，逐被试自身与跨 ROI 响应另表。该置换同时改变时序和历史相关图，结果是模型敏感性，不是生物因果或真实连接。HC/MDD 使用不同模型和条件，不作疾病效应比较。',
              f'环境与执行：DiffNext、{m["torch"]}、{m["device"]}、float32、批大小 {m["batch_size"]}；命令为 `python -m analysis.individual_perturbation --group {args.group} --device {m["device"]} --batch-size {m["batch_size"]} --baseline-dir results/个体化可解释性_20261030/基线缓存 --out-dir results/个体化可解释性_20261030/扰动缓存`。','']
    (dest/'主分析报告.md').write_text('\n'.join(report))


if __name__=='__main__': main()
