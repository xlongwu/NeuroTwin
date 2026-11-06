"""Summarize completed eight-anchor descriptive probes; subject is the unit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.h18_graph import NETWORKS, ROOT
from utils.site_subset import parse_site


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--groups', nargs='+', choices=('hc', 'mdd'), default=['hc', 'mdd'])
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_excel(ROOT / 'data/AAL116.xlsx')
    networks = atlas['对应网络'].to_numpy()
    if len(networks) != 116 or set(networks) != set(NETWORKS):
        raise ValueError('AAL116 network mapping mismatch')
    groups = {}
    for group in args.groups:
        root = args.cache_dir / group
        manifest = json.loads((root / '数据清单.json').read_text())
        missing = [sid for sid in manifest['subjects'] if not (root / f'{sid}.npz').is_file()]
        if missing:
            raise RuntimeError(f'{group}: {len(missing)} subjects missing; refuse partial summary')
        records, blocks, representations, graph_records = [], [], [], []
        for sid in manifest['subjects']:
            with np.load(root / f'{sid}.npz') as z:
                true, pred, altered, anchors = z['true'], z['prediction'], z['missing_prediction'], z['cutoff']
                p_abs = np.abs(pred).mean()
                baseline_mae = np.abs(pred-true).mean()
                missing_mae = np.abs(altered-true).mean()
                graphs = {k: z[k].copy() for k in ('history_fc_abs', 'A_func', 'A_eff')}
                per_graph = np.empty((3, 7, 7))
                for k, values in enumerate(graphs.values()):
                    for i, left in enumerate(NETWORKS):
                        for j, right in enumerate(NETWORKS):
                            ii, jj = np.meshgrid(np.flatnonzero(networks == left), np.flatnonzero(networks == right), indexing='ij')
                            keep = ii != jj
                            per_graph[k, i, j] = values[:, ii[keep], jj[keep]].mean()
                edge = np.triu_indices(116, 1)
                for graph_name, graph_values in graphs.items():
                    edges = graph_values[:, edge[0], edge[1]].reshape(-1)
                    graph_records.append({'subject':sid,'graph':graph_name,'n_anchors':len(anchors),
                                          'edge_mean':edges.mean(),'edge_median':np.median(edges),
                                          'edge_q05':np.quantile(edges,.05),'edge_q95':np.quantile(edges,.95),
                                          'lambda_mean':float(np.mean(z['lambda']))})
                matrix = graphs['A_eff']
                similarities, gaps = [], []
                for i in range(len(anchors)):
                    for j in range(i+1, len(anchors)):
                        gap = int(anchors[j] - anchors[i])
                        if gap < 64:
                            continue
                        x, y = matrix[i][edge], matrix[j][edge]
                        if x.std() > 0 and y.std() > 0:
                            similarities.append(float(np.corrcoef(x, y)[0, 1])); gaps.append(gap)
                records.append({'subject': sid, 'site': parse_site(sid), 'n_anchors': len(anchors),
                                'baseline_mae': baseline_mae, 'missing_mae': missing_mae,
                                'missing_minus_baseline_mae': missing_mae-baseline_mae,
                                'mean_abs_output_difference': np.abs(altered-pred).mean(),
                                'relative_output_difference': np.abs(altered-pred).mean()/p_abs if p_abs > 0 else np.nan,
                                'nonoverlap_pairs': len(similarities),
                                'graph_similarity_mean': np.mean(similarities) if similarities else np.nan,
                                'gap_min': min(gaps) if gaps else np.nan, 'gap_max': max(gaps) if gaps else np.nan})
                blocks.append(per_graph)
                representations.append(z['representation'].copy())
        groups[group] = {'records': pd.DataFrame(records), 'blocks': np.stack(blocks),
                         'representations': np.stack(representations)}
        groups[group]['records'].to_csv(args.out_dir / f'{group}_逐被试.csv', index=False)
        pd.DataFrame(graph_records).to_csv(args.out_dir / f'{group}_图边分布.csv', index=False)
        pd.DataFrame(groups[group]['representations'], index=manifest['subjects']).rename_axis('subject').to_csv(args.out_dir / f'{group}_个体表征.csv')
        rows = []
        for k, key in enumerate(('history_fc_abs', 'A_func', 'A_eff')):
            for i, left in enumerate(NETWORKS):
                for j, right in enumerate(NETWORKS):
                    vals = groups[group]['blocks'][:, k, i, j]
                    rows.append({'graph': key, 'source_network': left, 'target_network': right,
                                 'n_subjects': len(vals), 'source_roi_count': int((networks == left).sum()),
                                 'target_roi_count': int((networks == right).sum()),
                                 'n_nonself_edges': int((networks == left).sum() * (networks == right).sum() - ((networks == left).sum() if left == right else 0)),
                                 'mean': vals.mean(), 'median': np.median(vals)})
        pd.DataFrame(rows).to_csv(args.out_dir / f'{group}_网络块.csv', index=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    report = ['# 八锚点基线与描述性探针', '', '每人先对锚点平均，再对被试等权汇总。以下是滚动单步锚点抽样，不是密集滚动，也不是 ROI 时间置换主分析。', '网络块先在每个锚点排除 ROI 自环，再对锚点和被试分别等权平均；三张矩阵使用独立色阶。条件对照将所有条件字段设为缺失，不表示各字段贡献。', '有效图稳定性以同一被试不重叠 K64 窗的上三角非对角边 Pearson 相关为基础，每人先对可用窗对取均值；这不是跨扫描重测。', '']
    for group, data in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
        for i, (ax, title) in enumerate(zip(axes, ('History |FC|', 'Learned graph', 'Effective graph'))):
            m = data['blocks'][:, i].mean(axis=0)
            im = ax.imshow(m, cmap='Blues', aspect='equal', vmin=0)
            ax.set_xticks(range(7), NETWORKS, rotation=90); ax.set_yticks(range(7), NETWORKS)
            ax.set_title(title); fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        fig.suptitle(f'{group.upper()}: subject-mean network blocks; n={len(data["records"])}; diagonal excluded')
        for ext in ('png', 'pdf'): fig.savefig(args.out_dir / f'{group}_graph_blocks.{ext}', dpi=300)
        plt.close(fig)
        df = data['records']
        valid = df['graph_similarity_mean'].dropna()
        site_counts = ', '.join(f'{site}: {count}' for site, count in df['site'].value_counts().sort_index().items())
        report.extend([f'## {group.upper()}', '', f'被试数 {len(df)}；锚点数范围 {df.n_anchors.min()}–{df.n_anchors.max()}；站点人数：{site_counts}。', f'原始 MAE 被试均值 {df.baseline_mae.mean():.6f}；全缺失条件减原始 MAE 的被试均值 {df.missing_minus_baseline_mae.mean():+.6f}；输出相对差的被试中位数 {df.relative_output_difference.median():.6f}。', f'存在非重叠图窗对的被试 {len(valid)}/{len(df)}；其窗对平均相关的被试中位数 {valid.median():.4f}。', ''])
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
        rng = np.random.default_rng(2024)
        for ax, (column, label) in zip(axes, [('missing_minus_baseline_mae', 'Missing − baseline MAE'),
                                              ('relative_output_difference', 'Mean |output difference| / mean |baseline|')]):
            vals = df[column].dropna().to_numpy()
            ax.boxplot([vals], positions=[1], widths=.36, patch_artist=True,
                       boxprops={'facecolor':'#87ceeb', 'edgecolor':'#2b438d'},
                       medianprops={'color':'#2b438d'}, showfliers=False)
            ax.scatter(1+rng.uniform(-.12,.12,len(vals)), vals, s=5, c='#2b438d', alpha=.45)
            ax.axhline(0, color='#bbbbbb', linewidth=.7); ax.set_xticks([1], [group.upper()]); ax.set_ylabel(label)
        fig.suptitle(f'Condition probe: full vs missing; {group.upper()} n={len(df)}')
        for ext in ('png', 'pdf'): fig.savefig(args.out_dir / f'{group}_condition.{ext}', dpi=300)
        plt.close(fig)
        features = data['representations']
        representation_mae_r = [np.corrcoef(features[:,j], df['baseline_mae'])[0,1]
                         if features[:,j].std() > 1e-8 and df['baseline_mae'].std() > 0 else np.nan
                         for j in range(features.shape[1])]
        pd.DataFrame({'dimension': np.arange(1, features.shape[1]+1),
                      'mean': features.mean(axis=0), 'sd': features.std(axis=0, ddof=1),
                      'median': np.median(features, axis=0), 'baseline_mae_pearson_r':representation_mae_r}).to_csv(args.out_dir / f'{group}_表征汇总.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        x = np.arange(1, features.shape[1]+1)
        axes[0].errorbar(x, features.mean(axis=0), yerr=features.std(axis=0, ddof=1),
                         fmt='o', ms=2.8, lw=.8, color='#2b438d', ecolor='#87ceeb', capsize=1)
        axes[0].set(xlabel='TFM dimension', ylabel='Mean ± between-subject SD', title='Historical TFM representation')
        vals = df['graph_similarity_mean'].dropna().to_numpy()
        if len(vals):
            axes[1].boxplot([vals], positions=[1], widths=.36, patch_artist=True,
                            boxprops={'facecolor':'#87ceeb', 'edgecolor':'#2b438d'},
                            medianprops={'color':'#2b438d'}, showfliers=False)
            axes[1].scatter(1+rng.uniform(-.12,.12,len(vals)), vals, s=5, c='#2b438d', alpha=.45)
        axes[1].set(xticks=[1], xticklabels=[group.upper()], ylabel='Within-subject mean Pearson r',
                    title=f'Nonoverlapping effective-graph windows (n={len(vals)})')
        fig.suptitle(f'{group.upper()}: representation and within-scan graph stability; n={len(df)}')
        for ext in ('png', 'pdf'): fig.savefig(args.out_dir / f'{group}_表征稳定性.{ext}', dpi=300)
        plt.close(fig)
    (args.out_dir / ('_'.join(args.groups) + '_基线描述报告.md')).write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
