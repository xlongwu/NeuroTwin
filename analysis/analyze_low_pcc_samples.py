#!/usr/bin/env python
"""
低PCC样本临床特征分析
分析预测PCC较低的样本与临床特征的关联
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    p = argparse.ArgumentParser(description='低 PCC 样本临床特征分析')
    p.add_argument('--analysis_dir', type=str, default='/data3/Digital_Brain/NeuroTwin/checkpoints/neurotwin_finetune_pred13/interpretability/comprehensive_analysis_20260422_230827',
                   help='run_comprehensive 输出目录（含 sample_metrics.json）')
    p.add_argument('--output_dir', type=str, default=None,
                   help='结果输出目录，默认 <analysis_dir>/low_pcc_analysis')
    p.add_argument('--v1_file', type=str, default='/data3/Digital_Brain/AMD/data/Rest-meta-MDD-V1-MDD.xlsx')
    p.add_argument('--v2_file', type=str, default='/data3/Digital_Brain/AMD/data/Rest-meta-MDD-V2-MDD.xlsx')
    return p.parse_args()


def load_data(args):
    """加载所有数据"""
    # 加载sample metrics
    with open(Path(args.analysis_dir) / 'sample_metrics.json', 'r') as f:
        sample_metrics = json.load(f)

    # 加载V1和V2 MDD数据
    v1_df = pd.read_excel(args.v1_file)
    v2_df = pd.read_excel(args.v2_file)

    return sample_metrics, v1_df, v2_df

def process_sample_metrics(sample_metrics):
    """处理样本metrics，提取PCC信息"""
    df = pd.DataFrame(sample_metrics)
    
    # 按样本统计平均PCC
    subj_stats = df.groupby('subj_id').agg({
        'sample_pcc': ['mean', 'std', 'min', 'max', 'count'],
        'pathology_score': 'first'
    }).reset_index()
    
    subj_stats.columns = ['subj_id', 'mean_pcc', 'std_pcc', 'min_pcc', 'max_pcc', 'n_samples', 'pathology_score']
    
    # 定义低PCC阈值（使用10%分位数作为阈值，约0.81）
    pcc_threshold = subj_stats['mean_pcc'].quantile(0.10)
    subj_stats['is_low_pcc'] = subj_stats['mean_pcc'] < pcc_threshold
    
    return df, subj_stats, pcc_threshold

def merge_with_clinical_data(subj_stats, v1_df, v2_df):
    """合并临床数据"""
    
    # 标准化ID格式
    def normalize_id(subj_id):
        """标准化ID格式"""
        # IS001-1-0003 -> IS001-1-0003
        # S1-1-0001 -> S1-1-0001
        return str(subj_id).strip()
    
    subj_stats = subj_stats.copy()
    v1_df = v1_df.copy()
    v2_df = v2_df.copy()
    
    subj_stats['subj_id_norm'] = subj_stats['subj_id'].apply(normalize_id)
    v1_df['ID_norm'] = v1_df['ID'].apply(normalize_id)
    v2_df['ID_norm'] = v2_df['ID'].apply(normalize_id)
    
    # 合并V2数据
    merged = subj_stats.merge(v2_df, left_on='subj_id_norm', right_on='ID_norm', how='left', suffixes=('', '_v2'))
    
    # 未匹配的尝试V1数据
    unmatched = merged[merged['ID'].isna()].copy()
    if len(unmatched) > 0:
        # 复制需要合并的列，包括subj_id_norm
        unmatched_cols = ['subj_id', 'subj_id_norm', 'mean_pcc', 'std_pcc', 'min_pcc', 'max_pcc', 'pathology_score', 'is_low_pcc']
        unmatched_subset = unmatched[unmatched_cols].copy()
        
        matched_v1 = unmatched_subset.merge(
            v1_df, left_on='subj_id_norm', right_on='ID_norm', how='left'
        )
        
        # 更新已匹配的行
        for idx in matched_v1.index:
            if pd.notna(matched_v1.loc[idx, 'ID']):
                for col in v1_df.columns:
                    if col in merged.columns:
                        merged.loc[idx, col] = matched_v1.loc[idx, col]
    
    # 标记数据来源
    merged['data_source'] = merged['ID'].apply(lambda x: 'V2' if pd.notna(x) and str(x).startswith('IS') else ('V1' if pd.notna(x) else 'Unknown'))
    
    return merged

def analyze_low_pcc_characteristics(merged_df):
    """分析低PCC样本的临床特征"""
    
    low_pcc = merged_df[merged_df['is_low_pcc'] == True]
    normal_pcc = merged_df[merged_df['is_low_pcc'] == False]
    
    print(f"\n{'='*60}")
    print("低PCC样本分析")
    print(f"{'='*60}")
    print(f"总样本数: {len(merged_df)}")
    print(f"低PCC样本数: {len(low_pcc)} ({len(low_pcc)/len(merged_df)*100:.1f}%)")
    print(f"正常PCC样本数: {len(normal_pcc)} ({len(normal_pcc)/len(merged_df)*100:.1f}%)")
    
    results = {
        'total_samples': len(merged_df),
        'low_pcc_samples': len(low_pcc),
        'normal_pcc_samples': len(normal_pcc),
        'low_pcc_percentage': len(low_pcc)/len(merged_df)*100,
        'comparisons': {}
    }
    
    # 比较临床特征
    clinical_vars = [
        'Age', 'HAMD', 'HAMA', 'IllnessDuration', 
        'Sex', 'Education', 'FirstEpisode', 'OnMedication'
    ]
    
    print(f"\n{'='*60}")
    print("临床特征比较 (低PCC vs 正常PCC)")
    print(f"{'='*60}")
    
    for var in clinical_vars:
        if var in merged_df.columns:
            low_vals = low_pcc[var].dropna()
            normal_vals = normal_pcc[var].dropna()
            
            if len(low_vals) > 0 and len(normal_vals) > 0:
                # 确保数据是数值型
                try:
                    low_vals_numeric = pd.to_numeric(low_vals, errors='coerce').dropna()
                    normal_vals_numeric = pd.to_numeric(normal_vals, errors='coerce').dropna()
                except:
                    continue
                
                if len(low_vals_numeric) == 0 or len(normal_vals_numeric) == 0:
                    continue
                
                # 统计检验
                if var in ['Sex', 'FirstEpisode', 'OnMedication']:
                    # 分类变量
                    low_prop = low_vals_numeric.mean()
                    normal_prop = normal_vals_numeric.mean()
                    print(f"\n{var}:")
                    print(f"  低PCC: {low_prop:.2%} (n={len(low_vals_numeric)})")
                    print(f"  正常PCC: {normal_prop:.2%} (n={len(normal_vals_numeric)})")
                    
                    results['comparisons'][var] = {
                        'low_pcc_mean': float(low_prop),
                        'normal_pcc_mean': float(normal_prop),
                        'low_pcc_n': len(low_vals_numeric),
                        'normal_pcc_n': len(normal_vals_numeric)
                    }
                else:
                    # 连续变量用t检验
                    stat, pval = stats.ttest_ind(low_vals_numeric, normal_vals_numeric)
                    print(f"\n{var}:")
                    print(f"  低PCC: {low_vals_numeric.mean():.2f} ± {low_vals_numeric.std():.2f} (n={len(low_vals_numeric)})")
                    print(f"  正常PCC: {normal_vals_numeric.mean():.2f} ± {normal_vals_numeric.std():.2f} (n={len(normal_vals_numeric)})")
                    print(f"  p-value: {pval:.4f} {'*' if pval < 0.05 else ''}")
                    
                    results['comparisons'][var] = {
                        'low_pcc_mean': float(low_vals_numeric.mean()),
                        'low_pcc_std': float(low_vals_numeric.std()),
                        'normal_pcc_mean': float(normal_vals_numeric.mean()),
                        'normal_pcc_std': float(normal_vals_numeric.std()),
                        'p_value': float(pval),
                        'significant': pval < 0.05
                    }
    
    return results, low_pcc, normal_pcc

def plot_pcc_distribution(subj_stats, threshold, output_path):
    """绘制PCC分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax = axes[0]
    ax.hist(subj_stats['mean_pcc'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Low PCC threshold ({threshold})')
    ax.axvline(subj_stats['mean_pcc'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({subj_stats["mean_pcc"].mean():.3f})')
    ax.set_xlabel('Mean Sample PCC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Mean Sample PCC', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 箱线图
    ax = axes[1]
    bp = ax.boxplot([subj_stats[subj_stats['is_low_pcc']]['mean_pcc'], 
                     subj_stats[~subj_stats['is_low_pcc']]['mean_pcc']],
                    labels=['Low PCC\n(< 0.6)', 'Normal PCC\n(>= 0.6)'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Mean Sample PCC', fontsize=12)
    ax.set_title('PCC Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

def plot_clinical_comparison(low_pcc, normal_pcc, output_path):
    """绘制临床特征比较图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    continuous_vars = ['Age', 'HAMD', 'IllnessDuration', 'Education']
    
    for i, var in enumerate(continuous_vars):
        if var in low_pcc.columns and var in normal_pcc.columns:
            ax = axes[i]
            
            # 转换为数值并去除NaN
            low_vals = pd.to_numeric(low_pcc[var], errors='coerce').dropna()
            normal_vals = pd.to_numeric(normal_pcc[var], errors='coerce').dropna()
            
            if len(low_vals) >= 3 and len(normal_vals) >= 3:
                try:
                    # 小提琴图
                    parts = ax.violinplot([low_vals.values, normal_vals.values], positions=[1, 2], showmeans=True)
                    parts['bodies'][0].set_facecolor('#C55A11')  # 低PCC用赭石色
                    parts['bodies'][1].set_facecolor('#5B9BD5')  # 正常PCC用钢蓝色
                    
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(['Low PCC', 'Normal PCC'])
                    ax.set_ylabel(var, fontsize=11)
                    ax.set_title(f'{var} Distribution', fontsize=12, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # 添加统计信息
                    ax.text(0.5, 0.95, f'Low: {low_vals.mean():.1f}±{low_vals.std():.1f}\nNormal: {normal_vals.mean():.1f}±{normal_vals.std():.1f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
                except:
                    ax.text(0.5, 0.5, f'{var}\nInsufficient data', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{var}', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'{var}\nInsufficient data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{var}', fontsize=12, fontweight='bold')
    
    # 隐藏多余的子图
    for j in range(len(continuous_vars), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_hamd_factors_comparison(low_pcc, normal_pcc, output_path):
    """绘制HAMD因子比较"""
    # HAMD子项
    hamd_items = [f'HAMD{i}' for i in range(1, 25)]
    available_items = [item for item in hamd_items if item in low_pcc.columns and item in normal_pcc.columns]
    
    if len(available_items) < 5:
        print("Not enough HAMD items available")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    low_means = []
    normal_means = []
    p_values = []
    
    for item in available_items:
        low_vals = low_pcc[item].dropna()
        normal_vals = normal_pcc[item].dropna()
        
        if len(low_vals) > 0 and len(normal_vals) > 0:
            low_means.append(low_vals.mean())
            normal_means.append(normal_vals.mean())
            
            # 统计检验
            _, pval = stats.ttest_ind(low_vals, normal_vals)
            p_values.append(pval)
        else:
            low_means.append(0)
            normal_means.append(0)
            p_values.append(1)
    
    x = np.arange(len(available_items))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, low_means, width, label='Low PCC', color='#C55A11', alpha=0.8)
    bars2 = ax.bar(x + width/2, normal_means, width, label='Normal PCC', color='#5B9BD5', alpha=0.8)
    
    # 标记显著差异
    for i, pval in enumerate(p_values):
        if pval < 0.05:
            ax.text(i, max(low_means[i], normal_means[i]) + 0.1, '*', 
                   ha='center', fontsize=14, color='red')
    
    ax.set_xlabel('HAMD Items', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('HAMD Item Comparison (Low PCC vs Normal PCC)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_items, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_pcc_vs_pathology(subj_stats, output_path):
    """绘制PCC与病理评分的相关性"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散点图
    colors = ['#C55A11' if is_low else '#5B9BD5' for is_low in subj_stats['is_low_pcc']]
    ax.scatter(subj_stats['pathology_score'], subj_stats['mean_pcc'], 
              alpha=0.5, c=colors, s=30, edgecolors='black', linewidth=0.5)
    
    # 计算相关性
    valid_idx = subj_stats[['pathology_score', 'mean_pcc']].notna().all(axis=1)
    if valid_idx.sum() > 10:
        corr, pval = stats.pearsonr(subj_stats.loc[valid_idx, 'pathology_score'], 
                                    subj_stats.loc[valid_idx, 'mean_pcc'])
        
        # 拟合线
        z = np.polyfit(subj_stats.loc[valid_idx, 'pathology_score'], 
                       subj_stats.loc[valid_idx, 'mean_pcc'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subj_stats['pathology_score'].min(), 
                             subj_stats['pathology_score'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
               label=f'r={corr:.3f}, p={pval:.4f}')
    
    ax.set_xlabel('Pathology Score (Normalized HAMD)', fontsize=12)
    ax.set_ylabel('Mean Sample PCC', fontsize=12)
    ax.set_title('PCC vs Pathology Score', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#C55A11', edgecolor='black', label='Low PCC (< 0.6)'),
                      Patch(facecolor='#5B9BD5', edgecolor='black', label='Normal PCC (>= 0.6)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def identify_worst_cases(subj_stats, merged_df, top_n=20):
    """识别PCC最低的样本"""
    worst_cases = subj_stats.nsmallest(top_n, 'mean_pcc')
    
    print(f"\n{'='*60}")
    print(f"PCC最低的 {top_n} 个样本")
    print(f"{'='*60}")
    
    # 选择存在的列
    available_cols = ['subj_id']
    for col in ['Age', 'HAMD', 'HAMA', 'Sex', 'IllnessDuration', 'FirstEpisode', 'OnMedication']:
        if col in merged_df.columns:
            available_cols.append(col)
    
    worst_cases_detailed = worst_cases.merge(
        merged_df[available_cols].drop_duplicates('subj_id'),
        on='subj_id', how='left'
    )
    
    # 选择要显示的列
    display_cols = ['subj_id', 'mean_pcc']
    for col in ['Age', 'HAMD', 'HAMA', 'IllnessDuration']:
        if col in worst_cases_detailed.columns:
            display_cols.append(col)
    
    print(worst_cases_detailed[display_cols].to_string(index=False))
    
    return worst_cases_detailed

def main(args):
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.analysis_dir) / 'low_pcc_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    print("="*60)
    print("低PCC样本临床特征分析")
    print("="*60)
    
    # 加载数据
    print("\n1. 加载数据...")
    sample_metrics, v1_df, v2_df = load_data(args)
    
    # 处理样本metrics
    print("2. 处理样本metrics...")
    sample_df, subj_stats, threshold = process_sample_metrics(sample_metrics)
    
    # 合并临床数据
    print("3. 合并临床数据...")
    merged_df = merge_with_clinical_data(subj_stats, v1_df, v2_df)
    
    # 分析低PCC样本
    print("4. 分析低PCC样本特征...")
    results, low_pcc, normal_pcc = analyze_low_pcc_characteristics(merged_df)
    
    # 识别最差样本
    worst_cases = identify_worst_cases(subj_stats, merged_df)
    results['worst_cases'] = worst_cases.to_dict('records')
    
    # 生成可视化
    print("\n5. 生成可视化...")
    
    print("   - PCC分布图...")
    plot_pcc_distribution(subj_stats, threshold, out_dir /'pcc_distribution.png')
    
    print("   - 临床特征比较...")
    plot_clinical_comparison(low_pcc, normal_pcc, out_dir /'clinical_comparison.png')
    
    print("   - HAMD因子比较...")
    plot_hamd_factors_comparison(low_pcc, normal_pcc, out_dir /'hamd_factors_comparison.png')
    
    print("   - PCC与病理评分相关性...")
    plot_pcc_vs_pathology(subj_stats, out_dir /'pcc_vs_pathology.png')
    
    # 保存结果
    print("\n6. 保存分析结果...")
    
    # 处理numpy类型
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 递归转换
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(item) for item in d]
        else:
            return convert_numpy(d)
    
    results_converted = convert_dict(results)
    
    with open(out_dir /'low_pcc_analysis_results.json', 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    # 保存低PCC样本列表
    save_cols = ['subj_id', 'mean_pcc']
    for col in ['Age', 'HAMD', 'HAMA', 'Sex']:
        if col in merged_df.columns:
            save_cols.append(col)
    low_pcc_list = merged_df[merged_df['is_low_pcc']][save_cols].copy()
    low_pcc_list.to_excel(out_dir /'low_pcc_samples.xlsx', index=False)
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")
    print(f"\n结果保存至: {out_dir}")
    print(f"  - low_pcc_analysis_results.json (详细统计)")
    print(f"  - low_pcc_samples.xlsx (低PCC样本列表)")
    print(f"  - *.png (可视化图表)")

if __name__ == '__main__':
    main(parse_args())
