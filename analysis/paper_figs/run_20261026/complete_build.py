"""Complete F0--F14 plus D6, preserving the earlier delivery byte for byte."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

from utils.site_subset import parse_site
from .style import (ROOT, RUN1, HC, MDD, INK, NETWORK_ORDER, NETWORK_COLORS,
                    DIV_CMAP, apply_style, labels, partition, save, json_read, rain)
from .build import atlas_image

PREVIOUS = ROOT/'results/paper_figures/complete_20261026'


def combine_rolling(infer: Path) -> None:
    """Combine disjoint completed partitions, requiring the original full val set."""
    from analysis import rolling_one_step as rolling
    for group in ('hc', 'mdd'):
        destination = infer/'rolling'/group
        if (destination/'summary_val.json').exists():
            continue
        partitions = sorted((infer/'rolling_parts'/group).glob(f'part_*/{group}'))
        if len(partitions) != 4:
            raise ValueError(f'Expected four rolling partitions: {group}')
        expected = set(pd.read_csv(RUN1/f'z_maps/per_subject_z_{group}_val.csv').subject)
        found, files, manifests = set(), [], []
        for directory in partitions:
            if not (directory/'summary_val.json').exists():
                raise ValueError(f'Rolling partition incomplete: {directory}')
            manifest = json_read(directory/'manifest.json')
            manifests.append(manifest)
            for sid in manifest['subjects_done']:
                if sid in found:
                    raise ValueError(f'Overlapping rolling partitions: {sid}')
                found.add(sid);files.append(directory/f'{sid}.npz')
        if expected != found:
            raise ValueError(f'Rolling cohort mismatch: {group}, missing={expected-found}, extra={found-expected}')
        protocols = [rolling._protocol_block(m) for m in manifests]
        if any(p != protocols[0] for p in protocols):
            raise ValueError('Rolling partition protocols differ')
        destination.mkdir(parents=True,exist_ok=True)
        for source in files:
            if (destination/source.name).exists():
                raise FileExistsError(destination/source.name)
            shutil.copy2(source,destination/source.name)
        manifest=manifests[0].copy();manifest['subjects_done']=sorted(found)
        (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        rolling.summarize_group(infer/'rolling',group,'val',rolling.DEFAULT_DEADZONE,
                                rolling.DEFAULT_MIN_POINTS,subjects_filter=sorted(expected))


def emit(fig, out: Path, name: str, data, caption: str) -> None:
    save(fig, out, name, data, caption)
    (out/f'{name}.py').write_text(
        '"""Run from the repository root; pass a fresh --out-dir."""\n'
        'from analysis.paper_figs.run_20261026.complete_build import main\n'
        'if __name__ == "__main__":\n'
        f'    main(default_figure={name!r})\n')


def f0a(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis('off')
    stages = [('Frozen sources', 'HC base + historical H32\nFixed splits / hc3 statistics\nAAL116 labels and atlas'),
              ('T0 / T1', 'Archived inference and tests\nMatched / unmatched / HC controls\nClinical residuals and OOF ROC'),
              ('T2 / T3', '40 matched held-out subjects\n447 val subjects: all rolling anchors\nHistory shuffles / signed ROI probes'),
              ('Full visualization', 'F0a / F0b / F1–F14 + D6\n300 dpi PNG / vector PDF / data\nChinese captions / visual checks')]
    for index, (title, body) in enumerate(stages):
        x = .01 + .25*index
        ax.add_patch(FancyBboxPatch((x, .28), .22, .53, boxstyle='round,pad=.01',
                                   fc='#f3f6f8', ec=HC, transform=ax.transAxes))
        ax.text(x+.11, .71, title, ha='center', weight='bold', transform=ax.transAxes)
        ax.text(x+.11, .51, body, ha='center', va='center', fontsize=8,
                linespacing=1.8, transform=ax.transAxes)
        if index < 3:
            ax.annotate('', xy=(x+.244, .54), xytext=(x+.223, .54), xycoords='axes fraction',
                        arrowprops={'arrowstyle': '->', 'color': INK})
    ax.text(.5, .12, 'Frozen weights; descriptive perturbation responses; no new disease inference',
            ha='center', transform=ax.transAxes)
    ax.set_title('Complete paper analysis and visualization', pad=12)
    emit(fig, out, 'fig_f0a_pipeline', {'stages': stages, 'scope': 'T0/T1/T2/T3; user-authorized completion'},
         'F0a｜完整可视化流程。保留前批 T0/T1/T2 结果，补充 HC/MDD 全 val 滚动误差、固定匹配样本的历史顺序和单 ROI 输入扰动。模型、训练及权重不变；新结果仅作描述性结构分析，不新增疾病判据。')


def f4(out: Path, infer: Path, atlas: Path) -> None:
    from nilearn.plotting import plot_glass_brain
    net, order = partition()
    lab = labels()['micro命名'].tolist()
    roi_rows, subject_rows, arrays = [], [], {}
    for group in ('hc', 'mdd'):
        directory = infer/'rolling'/group
        with np.load(directory/'per_subject_roi_mae_val.npz') as data:
            ids, model, persistence = data['subjects'], data['mae_model'], data['mae_persistence']
        if model.shape != (len(ids), 116) or np.any(persistence <= 0):
            raise ValueError(f'Invalid rolling ROI arrays: {group}')
        improvement = 100*(1-model/persistence)
        amplitude = []
        for sid, mm, pp in zip(ids, model, persistence):
            with np.load(directory/f'{sid}.npz') as data:
                true, pred, last = data['true'], data['pred'], data['x_last']
                np.testing.assert_allclose(np.abs(pred-true).mean(0), mm, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(np.abs(last-true).mean(0), pp, rtol=1e-5, atol=1e-7)
                amplitude.append(np.sqrt(np.mean(true.astype(np.float64)**2, axis=0)))
                n = len(true)
            subject_rows.append(dict(subject=str(sid), group=group, site=parse_site(str(sid)),
                                     n_anchors=n, improvement=100*(1-mm.mean()/pp.mean()),
                                     mae_model=float(mm.mean()), mae_persistence=float(pp.mean())))
        amp = np.stack(amplitude)
        arrays[group] = (improvement, model, persistence, amp)
        for roi in range(116):
            roi_rows.append(dict(group=group, roi=roi+1, label=lab[roi], network=net[roi],
                                 improvement=float(improvement[:, roi].mean()),
                                 mae_model=float(model[:, roi].mean()),
                                 mae_persistence=float(persistence[:, roi].mean()),
                                 signal_rms=float(amp[:, roi].mean())))
    roi_table, subjects = pd.DataFrame(roi_rows), pd.DataFrame(subject_rows)
    roi_table.to_csv(out/'f4_roi_metrics.csv', index=False)
    subjects.to_csv(out/'f4_subject_metrics.csv', index=False)
    all_imp = np.concatenate([a[0].mean(0) for a in arrays.values()])
    limit = max(100, float(np.ceil(np.max(np.abs(all_imp)))))
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.1, 1.2], wspace=.27, hspace=.68)
    for column, (group, color) in enumerate([('hc', HC), ('mdd', MDD)]):
        d = roi_table[roi_table.group == group]
        sub = grid[0, column].subgridspec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(sub[0])
        plot_glass_brain(atlas_image(d.improvement.to_numpy(), atlas), axes=ax, figure=fig,
                         display_mode='lyrz', plot_abs=False, threshold=None,
                         cmap=DIV_CMAP, vmin=-limit, vmax=limit, colorbar=False)
        ax.set_title(f'({chr(97+column)}) {group.upper()} val, n={len(arrays[group][0])}')
        fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap=DIV_CMAP),
                     cax=fig.add_subplot(sub[1]), label='Improvement (%)')
    ax = fig.add_subplot(grid[1, 0])
    for group, color, dx in [('hc', HC, -.17), ('mdd', MDD, .17)]:
        imp = arrays[group][0]
        ax.bar(np.arange(7)+dx, [imp[:, net==n].mean() for n in NETWORK_ORDER],
               width=.32, color=color, label=group.upper())
    ax.axhline(0, color=INK, lw=.7)
    ax.set_xticks(range(7), NETWORK_ORDER)
    ax.set_ylabel('Mean ROI improvement (%)')
    ax.set_ylim(0,105)
    ax.set_title('(c) Prediction QC by display network')
    ax.legend(ncol=2)
    ax = fig.add_subplot(grid[1, 1])
    for group, color, marker in [('hc', HC, 'o'), ('mdd', MDD, '^')]:
        d = roi_table[roi_table.group==group]
        ax.scatter(d.signal_rms, d.mae_model, c=color, marker=marker, s=17,
                   alpha=.7, label=group.upper())
    ax.set_xlabel('ROI mean signal RMS (fixed calibration units)')
    ax.set_ylabel('ROI mean lag-1 MAE')
    ax.set_title('(d) Error and signal amplitude, 116 ROIs/group')
    ax.legend()
    sites = sorted(subjects.site.unique())
    # Two QC panels with the same axes, showing every site's actual subjects.
    ylo, yhi = subjects.improvement.min()-2, subjects.improvement.max()+2
    for col, (group, color) in enumerate([('hc', HC), ('mdd', MDD)]):
        ax = fig.add_subplot(grid[2, col])
        subset = subjects[subjects.group==group]
        rng = np.random.RandomState(2024)
        for position, site in enumerate(sites):
            values = subset.loc[subset.site==site, 'improvement'].to_numpy()
            if not len(values):
                continue
            ax.scatter(position+rng.uniform(-.17, .17, len(values)), values,
                       s=6, c=color, alpha=.4, lw=0, rasterized=True)
            ax.boxplot([values], positions=[position], widths=.5, showfliers=False,
                       medianprops={'color': INK}, boxprops={'color': color},
                       whiskerprops={'color': color}, capprops={'color': color})
        ax.set_xticks(range(len(sites)), sites, rotation=90, fontsize=7)
        ax.set_xlim(-1, len(sites)); ax.set_ylim(ylo, yhi)
        ax.set_ylabel('Subject improvement (%)')
        ax.set_title(f'({chr(101+col)}) {group.upper()} site QC; points are subjects')
    fig.suptitle('Rolling lag-1 error structure: all validation anchors', y=.99, weight='bold')
    emit(fig, out, 'fig_f4_error_structure', roi_table,
         'F4｜滚动误差结构（T3）。冻结 HC base、K64、全部合法锚点，HC val 222 人/MDD val 225 人；逐被试逐 ROI 改善度先计算 1−MAE_model/MAE_persistence，再对被试等权平均。两脑图共享对称色阶；网络条形先按 ROI 汇总，不作组间显著性推断。散点每点为一个 ROI 的被试均值，幅度为真实下一步信号的 RMS，单位沿用每人前 32 TR 固定校准；不等同去均值后时间标准差。站点箱线保留全部实际被试点，两个面板共享横纵轴。展示预测 QC，不代表疾病效应。逐被试/ROI 原数值另存 f4_subject_metrics.csv、f4_roi_metrics.csv。')


def f1(out: Path, infer: Path) -> None:
    from analysis.normative_deviation import paired_signflip_p
    gate=json_read(ROOT/'results/normative_reference/gate_report.json')
    with np.load(infer/'rolling/hc/per_subject_roi_mae_val.npz') as data:
        subjects=data['subjects'].tolist()
        model=data['mae_model'].astype(np.float64).mean(1)
        persistence=data['mae_persistence'].astype(np.float64).mean(1)
    settings=gate['criteria'];difference=persistence-model
    rng=np.random.RandomState(settings['seed'])
    null=(rng.choice((-1.,1.),size=(settings['n_perm'],len(subjects)))*difference).mean(1)
    p=paired_signflip_p(difference,n_perm=settings['n_perm'],seed=settings['seed'])
    np.testing.assert_allclose(p,gate['permutation_p'],atol=0,rtol=0)
    fig,axes=plt.subplots(1,3,figsize=(13,4.1))
    values=[model.mean(),persistence.mean()]
    axes[0].bar([0,1],values,color=[HC,'#aaa'],width=.55)
    for j,v in enumerate(values):axes[0].text(j,v+.02,f'{v:.4f}',ha='center')
    axes[0].set_xticks([0,1],['Model','Persistence']);axes[0].set_ylim(0,.74)
    axes[0].set_ylabel('Subject-equal rolling MAE');axes[0].set_title('(a) HC val: n=222')
    axes[0].text(.03,.9,f'Improvement: {1-values[0]/values[1]:.2%}',transform=axes[0].transAxes)
    axes[1].hist(null,bins=40,color=HC,alpha=.75)
    axes[1].axvline(difference.mean(),color=MDD,lw=1.5,label='Observed paired mean')
    axes[1].set_xlabel('Mean MAE(persistence) − MAE(model)')
    axes[1].set_ylabel('Sign-flip draws');axes[1].set_title('(b) 10,000 paired sign flips')
    axes[1].text(.55,.86,f'p = {p:.5f}',ha='center',transform=axes[1].transAxes)
    axes[1].legend(loc='upper left',fontsize=7)
    rain(axes[2],[model,persistence],['Model','Persistence'],colors=[HC,'#888'],ylabel='Per-subject MAE')
    axes[2].set_title('(c) True subject-level errors')
    fig.tight_layout()
    emit(fig,out,'fig_f1_gate',dict(archived_gate=gate,subjects=subjects,
        mae_model=model.tolist(),mae_persistence=persistence.tolist(),
        observed_paired_mean=float(difference.mean()),signflip_null=null.tolist(),
        signflip_p=p,seed=settings['seed']),
        'F1｜滚动单步护栏及实际零分布。原 null 样本未保留，本次利用恢复的完整 HC val 222 人滚动 MAE，按原 B4 方法及 seed=2024 重算 10000 次被试级配对符号翻转，单侧 p 与存档一致，仍为约 0.00010；直方图为真实重算的置换分布，不是模拟示意数据。柱状图与雨云图来自本次全部锚点重算，先对每人 ROI 等权再对被试等权。原 gate_report 保留在数据副本中，不覆盖正式护栏结果。')


def probe_arrays(infer: Path) -> tuple:
    manifest = json_read(infer/'probes/manifest.json')
    rows, results = [], []
    for pair_id, pair in enumerate(manifest['source_manifest']['pairs']):
        for group in ('hc', 'mdd'):
            with np.load(infer/f'probes/{group}_{pair[group]}.npz') as data:
                result = {key: data[key].copy() for key in data.files}
            if result['response'].shape != (2, 116, 116, 32):
                raise ValueError('Response dimension mismatch')
            if not all(np.isfinite(v).all() for v in result.values()):
                raise ValueError('Nonfinite probe output')
            if result['sham_max'] > manifest['sham_max_threshold']:
                raise ValueError('Sham gate failed')
            rows.append(dict(pair=pair_id, group=group, subject=pair[group], site=pair['site']))
            results.append(result)
    return pd.DataFrame(rows), results, manifest


def f13(out: Path, rows: pd.DataFrame, results: list) -> None:
    # Repeat is averaged within subject, then both members within their matched pair.
    change = np.stack([100*(d['history_mae'].mean(1)/d['baseline_mae'][None, :]-1) for d in results])
    pair_change = change.reshape(-1, 2, 8, 3).mean(1)
    rows_out = [{**row, 'block': block+1, 'tr_start': block*8+1, 'tr_end': block*8+8,
                 'segment': segment, 'relative_mae_change_percent': float(change[i, block, j]),
                 'baseline_mae': float(results[i]['baseline_mae'][j])}
                for i, row in enumerate(rows.to_dict('records')) for block in range(8)
                for j, segment in enumerate(['H1–8', 'H9–16', 'H17–32'])]
    d = pd.DataFrame(rows_out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.1, 1]})
    for j, (color, marker, segment) in enumerate([(HC, 'o', 'H1–8'), ('#8e8cc0', 's', 'H9–16'), (MDD, '^', 'H17–32')]):
        values = pair_change[:, :, j]
        center = values.mean(0)
        lower, upper = np.quantile(values, [.25, .75], axis=0)
        axes[0].fill_between(np.arange(8), lower, upper, color=color, alpha=.14)
        axes[0].plot(center, color=color, marker=marker, label=segment, lw=1.3)
    axes[0].axhline(0, color=INK, lw=.8)
    axes[0].set_xticks(range(8), [f'{8*k+1}–{8*k+8}' for k in range(8)], rotation=45)
    axes[0].set_xlabel('Shuffled history TR block (oldest → newest)')
    axes[0].set_ylabel('Relative MAE change (%)')
    axes[0].set_title('(a) Mean over 20 pairs; shade: pair IQR')
    axes[0].legend()
    matrix = pair_change[:, :, 2]
    limit = float(np.abs(matrix).max()) or 1
    image = axes[1].imshow(matrix, aspect='auto', cmap=DIV_CMAP, vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(8), [f'{8*k+1}–{8*k+8}' for k in range(8)], rotation=45)
    axes[1].set_xlabel('Shuffled history TR block')
    axes[1].set_ylabel('Fixed matched pair')
    axes[1].set_yticks([0,4,9,14,19],[1,5,10,15,20])
    axes[1].set_title('(b) H17–32 change by pair')
    fig.colorbar(image, ax=axes[1], label='Relative MAE change (%)')
    fig.suptitle('History order sensitivity: HC reference, H32 extrapolation', weight='bold')
    fig.text(.5, .015, 'Archived T08 (+14.2%) used a different H32 checkpoint and probe; no direct replication claim.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .95))
    emit(fig, out, 'fig_f13_history_attribution', d,
         'F13｜历史时间顺序敏感性。固定的 20 对同站点留出被试，每人 K64 分为 8 个连续 8 TR 块，每次只对一块内部时间顺序进行 3 次固定种子置换，所有 ROI 同步置换；校准、条件与 profile 固定，功能图从改变后的输入重算。每人先平均 3 次，再对每对 HC/MDD 等权平均；折线为 20 对均值，色带为匹配对之间 25–75 分位范围（不是置信区间）。热图展示 H17–32 的逐对百分比增量，负值如实保留。使用 HC base 的 H32 外推，与历史 T08 的 H32 权重、扰动口径不同，原 +14.2% 仅为存档参照。不能预设近历史更重要，不作疾病或临床因果解释。')


def f14(out: Path, rows: pd.DataFrame, results: list, atlas: Path) -> None:
    from nilearn.plotting import plot_glass_brain
    net, order = partition()
    # Equal weighting over sign, subject, matched pair, time: source x target.
    early = np.stack([np.abs(d['response'][:, :, :, :8]).mean((0, 3)) for d in results])
    late = np.stack([np.abs(d['response'][:, :, :, 16:]).mean((0, 3)) for d in results])
    early_mean, late_mean = early.mean(0), late.mean(0)
    mask = ~np.eye(116, dtype=bool)
    cross = late_mean.copy(); np.fill_diagonal(cross, 0)
    source = cross.sum(1)/115
    target = cross.sum(0)/115
    lab = labels()['micro命名'].tolist()
    d = pd.DataFrame(dict(roi=np.arange(1,117), label=lab, network=net,
                          cross_roi_source_sensitivity=source,
                          cross_roi_target_sensitivity=target,
                          own_roi_sensitivity=np.diag(late_mean)))
    d.to_csv(out/'f14_roi_sensitivity.csv', index=False)
    np.savez_compressed(out/'f14_propagation.npz', early=early, late=late,
                        subjects=rows.subject.to_numpy(), source_order=order)
    fig = plt.figure(figsize=(14, 8.8))
    grid = fig.add_gridspec(2, 3, hspace=.55, wspace=.75, height_ratios=[1.2, 1])
    vmax = float(max(early_mean[mask].max(), late_mean[mask].max()))
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#e8e8e8')
    matrix_axes=[]
    for column, (matrix, title) in enumerate([(early_mean, '(a) H1–8 propagation'), (late_mean, '(b) H17–32 propagation')]):
        ax = fig.add_subplot(grid[0,column])
        masked=np.ma.array(matrix,mask=~mask)
        im = ax.imshow(masked[np.ix_(order,order)], cmap=cmap, vmin=0, vmax=vmax)
        matrix_axes.append(ax)
        sizes = np.array([(net==n).sum() for n in NETWORK_ORDER])
        mids = sizes.cumsum()-sizes/2-.5
        for boundary in sizes.cumsum()[:-1]:
            ax.axhline(boundary-.5, color='white', lw=.5); ax.axvline(boundary-.5, color='white', lw=.5)
        ax.set_xticks(mids, NETWORK_ORDER, rotation=45, fontsize=7)
        ax.set_yticks(mids, NETWORK_ORDER, fontsize=7)
        ax.set_xlabel('Target ROI'); ax.set_ylabel('Perturbed source ROI'); ax.set_title(title)
    fig.colorbar(im,ax=matrix_axes,orientation='horizontal',fraction=.05,pad=.18,
                 label='Cross-ROI mean |Δprediction| / 0.1 (gray: self-response excluded)')
    ax = fig.add_subplot(grid[0,2])
    idx = np.argsort(source)[-15:]
    ax.barh(range(15), source[idx], color=[NETWORK_COLORS[n] for n in net[idx]])
    ax.set_yticks(range(15), [lab[i] for i in idx], fontsize=7)
    ax.set_xlabel('Off-diagonal mean response / 0.1')
    ax.set_title('(c) Top sources, H17–32')
    ax = fig.add_subplot(grid[1,:2])
    plot_glass_brain(atlas_image(source, atlas), axes=ax, figure=fig, display_mode='lyrz',
                     plot_abs=False, threshold=None, cmap='magma', vmin=0,
                     vmax=float(source.max()), colorbar=True)
    ax.set_title('(d) Source sensitivity: self-response excluded')
    ax = fig.add_subplot(grid[1,2])
    timecourse = np.stack([np.abs(d['response']).mean(0)[mask].mean(0) for d in results])
    pair_time = timecourse.reshape(-1,2,32).mean(1)
    lo, hi = np.quantile(pair_time,[.25,.75],axis=0)
    ax.fill_between(np.arange(1,33),lo,hi,color=HC,alpha=.2)
    ax.plot(np.arange(1,33),pair_time.mean(0),color=HC,label='Signed input probes')
    sham = max(float(d['sham_max']) for d in results)
    ax.axhline(sham/.1,color=INK,ls='--',label=f'Sham max / 0.1: {sham/.1:.1g}')
    ax.set_xlabel('Forecast horizon (TR)'); ax.set_ylabel('Cross-ROI response / 0.1')
    ax.set_title('(e) Time course; shade: pair IQR'); ax.legend(fontsize=7)
    fig.suptitle('Input perturbation sensitivity: structural model response only', weight='bold', y=.99)
    fig.text(.5,.015,'±0.1 input units for the last 8 history TR; 20 fixed same-site pairs; no clinical calibration or disease inference.',ha='center',fontsize=8)
    emit(fig,out,'fig_f14_perturbation',dict(roi_summary=d.to_dict('records'),
         sham_max=sham, early_mean=early_mean.tolist(), late_mean=late_mean.tolist(),
         pair_timecourse=pair_time.tolist(), n_pairs=20, amplitude=.1),
         'F14｜输入扰动传播敏感性（预登记后描述性前向）。固定 20 对同站点留出被试，对每个源 ROI 的最后 8 TR 分别加/减 0.1 个固定校准单位，其他输入不变；条件、32 TR 校准、profile 固定，功能图随输入重算。响应为相对同批次未扰动预测的绝对变化除以 0.1，先平均正负方向、再对 40 人等权汇总。两个矩阵共享色阶，行为源、列为目标；矩阵灰色对角线、脑图、Top-15 及时间曲线均排除源 ROI 对自身的直接响应，数据仍保留完整矩阵与直接响应。时间曲线色带为匹配对间四分位范围。每人零扰动 sham 最大差均须不超过 1e-6。这里只描述模型输入敏感性，未校准为真实刺激、损伤或治疗响应，不构成疾病因果证据；H32 对 HC base 属于外推。完整逐人早/远期传播矩阵另存 f14_propagation.npz。')


def d6(out: Path, infer: Path) -> None:
    original = pd.read_csv(RUN1/'param_dim_table.csv')
    with np.load(infer/'profile_controls/profiles.npz') as data:
        ids, group, split, profile = data['subjects'], data['groups'], data['splits'], data['profiles']
    lookup = {(g, sid): profile[i] for i,(g,sid) in enumerate(zip(group,ids))}
    pairs = json_read(ROOT/'data/splits/site_matched_subset.json')['pairs']
    mh = np.stack([lookup['hc',p['hc']] for p in pairs])
    mm = np.stack([lookup['mdd',p['mdd']] for p in pairs])
    delta = (mm-mh).mean(0)
    # The archived extractor never called model.eval(); its dropout path differs
    # from this deterministic extraction. Keep the archived statistic separately,
    # rather than pairing its q-values with the newly extracted effect vector.
    archived_delta = original.delta.to_numpy().copy()
    original['archived_delta'] = archived_delta
    original['delta'] = delta
    extraction_difference = float(np.max(np.abs(delta-archived_delta)))
    diagnostic=json_read(infer/'profile_controls/extraction_diagnostic.json')
    (out/'d6_extraction_comparison.json').write_text(json.dumps({
        'max_abs_difference_from_archived_delta': extraction_difference,
        'archived_extractor': 'analysis/param_group_tests.py: main does not call model.eval()',
        'current_extractor': 'TwinSession explicitly uses model.eval(); float64 prefix calibration',
        'interpretation': 'Current descriptive means and archived q-values are distinct analyses; no new tests.',
        'one_subject_diagnostic': diagnostic
    }, indent=2)+'\n')
    control_unmatched = profile[group=='mdd'].mean(0)-profile[group=='hc'].mean(0)
    control_hc = profile[(group=='hc')&(split=='val')].mean(0)-profile[(group=='hc')&(split=='test')].mean(0)
    original['delta_unmatched'] = control_unmatched
    original['delta_hc_control'] = control_hc
    fig = plt.figure(figsize=(13,7))
    gs = fig.add_gridspec(2,3,height_ratios=[1.2,1],hspace=.55,wspace=.3)
    limit = max(np.abs(original.delta).max(),abs(control_unmatched).max(),abs(control_hc).max())*1.1
    for j,(col,title,color) in enumerate([('delta','(a) Matched: 1504 pairs',HC),
        ('delta_unmatched','(b) Unmatched: all subjects',MDD),('delta_hc_control','(c) HC val - test control','#8e8cc0')]):
        ax = fig.add_subplot(gs[0,j]);ax.bar(original.dim,original[col],color=color,width=.8)
        ax.axhline(0,color=INK,lw=.7);ax.set_ylim(-limit,limit)
        ax.set_xlabel('Profile dimension (zero-based)');ax.set_title(title)
        if j==0: ax.set_ylabel('Raw mean profile difference')
    ax = fig.add_subplot(gs[1,:2]);ax.bar(original.dim,-np.log10(original.q),color=HC)
    ax.axhline(-np.log10(.05),color=INK,ls='--',label='Archived BH-FDR q=0.05')
    ax.set_xlabel('Profile dimension (zero-based)');ax.set_ylabel('-log10(archived q)')
    ax.set_title('(d) Archived q: different extraction; not tests of (a)');ax.legend()
    summary = json_read(RUN1/'param_group_tests.json')['summary_stats']
    ax=fig.add_subplot(gs[1,2]);ratio=summary['svd_top_explained_ratio']
    ax.barh([0],[ratio*100],color=HC);ax.barh([0],[(1-ratio)*100],left=ratio*100,color='#ddd')
    ax.set_xlim(0,100);ax.set_yticks([]);ax.set_xlabel('Uncentered SVD energy (%)')
    ax.set_title(f'(e) First direction: {ratio:.2%}')
    ax.text(0,-.7,'Not centered between-subject variance\nNot a disease effect size',fontsize=8)
    fig.suptitle('Archived profile results with descriptive comparison controls',weight='bold',y=.99)
    emit(fig,out,'fig_s1_profile',original,
         'S1/D6｜参数线完整可视化。上排以统一纵轴显示本次 eval 模式提取 profile 的 1504 对匹配差、全队列未匹配差与 HC val−test 负控，不新增 p 值或病理结论。本次匹配均值与存档值存在差异，量化见 d6_extraction_comparison.json；旧参数提取脚本未调用 model.eval()，模型可能启用 dropout，前缀浮点计算路径也不同，因此不把原 q 值配给上排新效应。下排仅存档展示 run1 原 q 值，dim=20（零基）q≈0.0128 的可靠性尚待单独复核，不能视为已复现的参数标志物。能量占比引用原未中心化 SVD 首方向 99.36%，不等同中心化个体差异解释率。不重训、不调参数、不重跑参数线推断。')


def sheet(paths: list, destination: Path, columns: int = 4, width: int = 650) -> None:
    cell_h, header, padding = 480, 35, 18
    canvas = Image.new('RGB',(columns*width,(int(np.ceil(len(paths)/columns)))*(cell_h+header+padding)),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for i,path in enumerate(paths):
        with Image.open(path) as image:
            image=image.convert('RGB');image.thumbnail((width-20,cell_h))
            x=(i%columns)*width;y=(i//columns)*(cell_h+header+padding)
            draw.text((x+10,y+8),path.stem,fill=INK,font=font)
            canvas.paste(image,(x+(width-image.width)//2,y+header+(cell_h-image.height)//2))
    canvas.save(destination)


def main(default_figure: str | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--infer-dir',type=Path,default=ROOT/'results/paper_infer/complete_20261026')
    parser.add_argument('--atlas-dir',type=Path,default=Path.home()/'nilearn_data/aal_SPM12/aal')
    parser.add_argument('--figure',default=default_figure)
    ns=parser.parse_args();ns.out_dir.mkdir(parents=True,exist_ok=True)
    if any(ns.out_dir.iterdir()):
        raise FileExistsError('Use a fresh --out-dir')
    apply_style()
    if not ns.figure or ns.figure=='fig_f4_error_structure':
        combine_rolling(ns.infer_dir)
    if not ns.figure:
        for pattern in ('fig_*.*','table_*'):
            for path in PREVIOUS.glob(pattern):
                if not path.name.startswith(('fig_f0a_','fig_f1_')):
                    shutil.copy2(path,ns.out_dir/path.name)
    builders={'fig_f0a_pipeline':lambda:f0a(ns.out_dir),
              'fig_f1_gate':lambda:f1(ns.out_dir,ns.infer_dir),
              'fig_f4_error_structure':lambda:f4(ns.out_dir,ns.infer_dir,ns.atlas_dir),
              'fig_s1_profile':lambda:d6(ns.out_dir,ns.infer_dir)}
    if not ns.figure or ns.figure in ('fig_f13_history_attribution','fig_f14_perturbation'):
        rows,results,manifest=probe_arrays(ns.infer_dir)
        builders.update({'fig_f13_history_attribution':lambda:f13(ns.out_dir,rows,results),
                         'fig_f14_perturbation':lambda:f14(ns.out_dir,rows,results,ns.atlas_dir)})
    names=[ns.figure] if ns.figure else list(builders)
    for name in names:
        builders[name]()
    if ns.figure:
        return
    figures=sorted(ns.out_dir.glob('fig_*.png'),key=lambda p:(p.stem.startswith('fig_s'),int(p.stem.split('_')[1][1:].rstrip('ab')),p.stem))
    sheet(figures,ns.out_dir/'overview.png')
    gallery=Path.home()/'.agents/skills/plot-is-all-you-need/gallery'
    references=[gallery/'raincloud-median-badges.png',ns.out_dir/'fig_f4_error_structure.png',
        gallery/'oa-tech-pone-0258309-f05.png',ns.out_dir/'fig_f13_history_attribution.png',
        gallery/'oa-bio-plos-pcbi-1011915-f05.png',ns.out_dir/'fig_f14_perturbation.png']
    sheet(references,ns.out_dir/'reference_comparison.png',columns=2,width=1000)
    captions=['# 完整图注','']
    report=['# 完整可视化交付','',
            '在上一批 13 张图的基础上补齐 F4、F13、F14 与 D6 附图，共 16 张主图及 1 张附图；保留全部真实输入、统计数据和中文图注。旧目录不覆盖。','',
            '新增计算复用 HC base、固定种子和原条件统计；HC/MDD 全 val 滚动评估使用全部锚点。历史块和 ROI 扰动仅描述模型结构性质，不作新的组间推断。F2 保持历史 H32 权重口径，其余 HC base H32 结果属于外推。','',
            '[全部图览](overview.png) · [新增图与图库示例对照](reference_comparison.png) · [检查记录](CHECKS.json)','',
            '原方案中的“默认不执行”项在用户追加“执行完整可视化”后授权执行；设置见 [执行登记](../../../docs/VISUALIZATION_COMPLETION_PREREG.md)。','']
    rolling_summary={g:json_read(ns.infer_dir/f'rolling/{g}/summary_val.json') for g in ('hc','mdd')}
    history=pd.read_csv(ns.out_dir/'fig_f13_history_attribution.csv')
    last=history[history.block==8].groupby('segment').relative_mae_change_percent.mean()
    report.extend(['## 新增结果与边界','',
        f'- 全 val 滚动重算：HC {rolling_summary["hc"]["n_subjects_this_split"]} 人、{rolling_summary["hc"]["anchor_total"]} 锚点，模型 MAE {rolling_summary["hc"]["mae_model_subject_equal_weight"]:.6f}；MDD {rolling_summary["mdd"]["n_subjects_this_split"]} 人、{rolling_summary["mdd"]["anchor_total"]} 锚点，模型 MAE {rolling_summary["mdd"]["mae_model_subject_equal_weight"]:.6f}。',
        f'- F13：打乱最近 8 TR，H1–8 相对误差变化 {last["H1–8"]:+.2f}%，H17–32 为 {last["H17–32"]:+.2f}%；这是 HC base 外推，不是历史 T08 的复现。',
        '- F14：40 人、116 个源 ROI、双符号扰动全部完成，零扰动负控最大差为 0；传播仅解释模型结构性质。',
        '- D6：本次 eval 提取与原参数表未完全复现；下排原 q 值单独保留存档，不能为本次描述性效应提供显著性。',''])
    report.extend(['## 图库借鉴','',
        '实际查看 plot-is-all-you-need 的 raincloud-median-badges、oa-tech-pone-0258309-f05、oa-bio-plos-pcbi-1011915-f05 原图与卡片。借鉴真实点分布、克制的点线与基准线、等尺寸矩阵和共同色阶；HC 蓝/MDD 橙沿用上一批，网络色沿用项目约定。F14 将直接自响应从传播矩阵的显示色阶中排除，完整数据保留。',
        'reference_comparison.png 左列为图库参考，右列为本项目新图；参考图使用其自身数据，不是本项目结果。本次未向共享图库收图。',''])
    for path in figures:
        name=path.stem;caption=(ns.out_dir/f'{name}.caption.md').read_text().strip()
        captions.extend([f'## {name}','',caption,''])
        report.extend([f'## {name}','',f'![{name}]({name}.png)','',caption,'',f'[PDF]({name}.pdf) · [脚本]({name}.py)',''])
    report.extend(['## 复现','', '从仓库根目录运行，使用新的输出路径：','', '```bash',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m analysis.paper_figs.run_20261026.complete_infer --out-dir NEW_INFER/probes',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m analysis.paper_figs.run_20261026.complete_profiles --out-dir NEW_INFER/profile_controls',
        '# For each cohort hc/mdd and partition 0/1/2/3 (eight disjoint jobs):',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m analysis.paper_figs.run_20261026.complete_rolling --group COHORT --partition PART --out-dir NEW_INFER/rolling_parts/COHORT',
        'MPLCONFIGDIR=/tmp/paper-mpl python -m analysis.paper_figs.run_20261026.complete_build --infer-dir NEW_INFER --out-dir NEW_FIGURES',
        'MPLCONFIGDIR=/tmp/paper-mpl python -m analysis.paper_figs.run_20261026.complete_check --infer-dir NEW_INFER --out-dir NEW_FIGURES',
        '```','',
        'F4 的 complete_rolling 复用 analysis.rolling_one_step 的视图、前向与汇总函数，使用 torch.set_num_threads(2)、batch_size=64、context_length=64、anchor_stride=1；四个分片无重叠，合并时核对完整 val 被试集合与协议。F13/F14 本次用 --pair-start/--pair-stop 分为六份同协议 CPU 作业并行，所有样本及随机排列不变。','',
        '沿用的限制：F1 原 null 未保留，本次恢复 HC 全 val 滚动数据后按原口径重算实际零分布；F10 未启用的 AdaRMS 不绘制伪分布；F11 实际 history-FC 先验不称为实测解剖 SC；HAMA 缺失机制未知；不新增临床因果结论。',
        '本批补齐 D6 未匹配和 HC 负控的描述性面板，不重跑疾病检验。原参数提取脚本没有切换 eval；一个固定 HC 前缀的重复诊断显示 eval 差为 0、训练模式差约 0.1324，而两种前缀浮点路径差约 3.5e-7。因此旧参数线 q≈0.0128 只作存档，不称为本次已复现的弱线索。详情见 [提取对照](d6_extraction_comparison.json)。',
        '原 13 图的说明详见 [上一批报告](../run_20261026/REPORT.md)。',''])
    (ns.out_dir/'CAPTIONS_ZH.md').write_text('\n'.join(captions))
    (ns.out_dir/'REPORT.md').write_text('\n'.join(report))
    manifest={'figure_ids':[p.stem for p in figures], 'primary_figure_count':16,
              'supplementary_figure_count':1, 'excluded':[],
              'new_infer_dir':str(ns.infer_dir.relative_to(ROOT)) if ns.infer_dir.is_relative_to(ROOT) else str(ns.infer_dir)}
    (ns.out_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Complete: {len(figures)} figures',flush=True)


if __name__ == '__main__':
    main()
