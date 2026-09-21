#!/usr/bin/env python
"""
ROI重要性可视化脚本
结合AAL116脑区图谱信息进行可视化
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# AAL116 网络缩写映射
NETWORK_NAMES = {
    'VN': 'Visual Network (视觉网络)',
    'SMN': 'Sensorimotor Network (感觉运动网络)',
    'DAN': 'Dorsal Attention Network (背侧注意网络)',
    'VAN': 'Ventral Attention Network (腹侧注意网络)',
    'LN': 'Limbic Network (边缘网络)',
    'FN': 'Frontoparietal Network (额顶网络)',
    'DMN': 'Default Mode Network (默认模式网络)',
    'SN': 'Salience Network (显著性网络)',
    'SC': 'Subcortical (皮层下)',
    'CB': 'Cerebellum (小脑)'
}

NETWORK_COLORS = {
    'VN': '#8B7355',    # 棕褐色 - 沉稳
    'SMN': '#5B9BD5',   # 钢蓝色 - 专业
    'DAN': '#70AD47',   # 橄榄绿 - 自然
    'VAN': '#C55A11',   # 赭石色 - 温暖
    'LN': '#9E480E',    # 深棕色 - 重要
    'FN': '#636363',    # 灰色 - 中性
    'DMN': '#4472C4',   # 深蓝色 - 稳重
    'SN': '#FFC000',    # 金黄色 - 突出但不过艳
    'SC': '#264478',    # 深藏青 - 深沉
    'CB': '#A5A5A5'     # 浅灰 - 辅助
}

def parse_args():
    p = argparse.ArgumentParser(description='ROI 重要性可视化（结合 AAL116 图谱）')
    p.add_argument('--input_json', type=str, default='/data3/Digital_Brain/NeuroTwin/checkpoints/neurotwin_finetune_pred13/interpretability/comprehensive_analysis_20260422_230827/feature_importance.json',
                   help='run_comprehensive 输出的 feature_importance.json')
    p.add_argument('--aal_file', type=str, default='/data3/Digital_Brain/AMD/data/AAL116.xlsx',
                   help='AAL116 脑区图谱 xlsx')
    p.add_argument('--output_dir', type=str, default=None,
                   help='图片输出目录，默认 <input_json 所在目录>/visualizations')
    return p.parse_args()


def load_data(args):
    """加载ROI重要性和AAL116映射"""
    # 加载feature importance
    with open(args.input_json, 'r') as f:
        importance_data = json.load(f)

    # 加载AAL116映射
    aal_df = pd.read_excel(args.aal_file)

    return importance_data, aal_df

def prepare_data(importance_data, aal_df):
    """准备数据用于可视化"""
    roi_importance = importance_data['roi_importance']
    
    # AAL116.xlsx 列名映射（中文列名）
    # 'micro编号', '中文名称', 'micro命名', 'abbr', '对应网络'
    name_col = 'micro命名' if 'micro命名' in aal_df.columns else 'AAL_Name'
    network_col = '对应网络' if '对应网络' in aal_df.columns else 'Network'
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ROI_Index': range(len(roi_importance)),
        'Importance': roi_importance,
        'AAL_Name': aal_df[name_col].values if name_col in aal_df.columns else [f'ROI_{i}' for i in range(len(roi_importance))],
        'Network': aal_df[network_col].values if network_col in aal_df.columns else ['Unknown'] * len(roi_importance)
    })
    
    # 按重要性排序
    df_sorted = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    return df, df_sorted

def plot_roi_importance_bar(df_sorted, output_path, top_n=30):
    """绘制ROI重要性条形图（Top N）"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    top_df = df_sorted.head(top_n)
    
    # 为每个网络分配颜色
    colors = [NETWORK_COLORS.get(net, '#999999') for net in top_df['Network']]
    
    bars = ax.barh(range(top_n), top_df['Importance'].values, color=colors, edgecolor='black', linewidth=0.5)
    
    # 设置y轴标签
    y_labels = [f"{row['AAL_Name'][:30]}" for _, row in top_df.iterrows()]
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(top_df.iterrows()):
        ax.text(row['Importance'] + 0.0001, i, f'{row["Importance"]:.4f}', 
                va='center', fontsize=8, color='black')
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} ROI Importance (AAL116)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=NETWORK_COLORS.get(net, '#999999'), 
                            edgecolor='black', label=NETWORK_NAMES.get(net, net)) 
                      for net in top_df['Network'].unique() if net in NETWORK_COLORS]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_network_importance(df, output_path):
    """按脑网络分组统计重要性"""
    network_stats = df.groupby('Network').agg({
        'Importance': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    network_stats.columns = ['Network', 'Mean_Importance', 'Std_Importance', 'Min_Importance', 'Max_Importance', 'Count']
    network_stats = network_stats.sort_values('Mean_Importance', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：平均重要性
    colors = [NETWORK_COLORS.get(net, '#999999') for net in network_stats['Network']]
    bars1 = ax1.bar(range(len(network_stats)), network_stats['Mean_Importance'], 
                    color=colors, edgecolor='black', linewidth=0.5, yerr=network_stats['Std_Importance'],
                    capsize=5, alpha=0.8)
    
    ax1.set_xticks(range(len(network_stats)))
    ax1.set_xticklabels([f"{net}\n({NETWORK_NAMES.get(net, net).split('(')[0][:15]}...)" 
                         for net in network_stats['Network']], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Importance', fontsize=12)
    ax1.set_title('Mean ROI Importance by Brain Network', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(network_stats.iterrows()):
        ax1.text(i, row['Mean_Importance'] + row['Std_Importance'] + 0.0001, 
                f'{row["Mean_Importance"]:.4f}', ha='center', fontsize=9)
    
    # 右图：ROI数量分布
    bars2 = ax2.bar(range(len(network_stats)), network_stats['Count'], 
                    color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax2.set_xticks(range(len(network_stats)))
    ax2.set_xticklabels(network_stats['Network'], rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Number of ROIs', fontsize=12)
    ax2.set_title('ROI Count by Brain Network', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(network_stats.iterrows()):
        ax2.text(i, row['Count'] + 0.5, str(int(row['Count'])), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return network_stats

def plot_importance_heatmap(df, output_path):
    """绘制ROI重要性热力图（按网络排序）"""
    # 按网络分组排序
    df_sorted_by_network = df.sort_values(['Network', 'Importance'], ascending=[True, False])
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # 创建热力图数据
    importance_values = df_sorted_by_network['Importance'].values.reshape(-1, 1)
    
    im = ax.imshow(importance_values, cmap='YlOrRd', aspect='auto')
    
    # 设置y轴标签
    y_labels = [f"{row['AAL_Name'][:25]} ({row['Network']})" 
                for _, row in df_sorted_by_network.iterrows()]
    ax.set_yticks(range(len(df_sorted_by_network)))
    ax.set_yticklabels(y_labels, fontsize=7)
    
    # 添加网格线按网络分组
    current_network = None
    for i, (_, row) in enumerate(df_sorted_by_network.iterrows()):
        if current_network != row['Network']:
            if current_network is not None:
                ax.axhline(y=i-0.5, color='black', linewidth=1.5)
            current_network = row['Network']
    
    ax.set_xticks([])
    ax.set_title('ROI Importance Heatmap (Grouped by Network)', fontsize=14, fontweight='bold', pad=20)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02, aspect=50)
    cbar.set_label('Importance Score', fontsize=11)
    
    # 添加网络标签
    network_positions = df_sorted_by_network.groupby('Network').apply(lambda x: x.index[0] + len(x)/2 - 0.5)
    for network, pos in network_positions.items():
        ax.text(1.02, pos, network, transform=ax.get_yaxis_transform(),
                fontsize=9, va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=NETWORK_COLORS.get(network, '#999999'), alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_importance_distribution(df, output_path):
    """绘制重要性分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 整体分布直方图
    ax1 = axes[0, 0]
    ax1.hist(df['Importance'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['Importance'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Importance"].mean():.4f}')
    ax1.axvline(df['Importance'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Importance"].median():.4f}')
    ax1.set_xlabel('Importance Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of ROI Importance', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. 箱线图（按网络）
    ax2 = axes[0, 1]
    networks = df['Network'].unique()
    network_data = [df[df['Network'] == net]['Importance'].values for net in networks if pd.notna(net)]
    network_labels = [net for net in networks if pd.notna(net)]
    
    bp = ax2.boxplot(network_data, labels=network_labels, patch_artist=True)
    for patch, net in zip(bp['boxes'], network_labels):
        patch.set_facecolor(NETWORK_COLORS.get(net, '#999999'))
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Brain Network', fontsize=11)
    ax2.set_ylabel('Importance Score', fontsize=11)
    ax2.set_title('Importance Distribution by Network', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 小提琴图
    ax3 = axes[1, 0]
    parts = ax3.violinplot(network_data, positions=range(len(network_labels)), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        net = network_labels[i]
        pc.set_facecolor(NETWORK_COLORS.get(net, '#999999'))
        pc.set_alpha(0.7)
    
    ax3.set_xticks(range(len(network_labels)))
    ax3.set_xticklabels(network_labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Importance Score', fontsize=11)
    ax3.set_title('Importance Violin Plot by Network', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 累积分布
    ax4 = axes[1, 1]
    sorted_importance = np.sort(df['Importance'])
    cumulative = np.arange(1, len(sorted_importance) + 1) / len(sorted_importance)
    ax4.plot(sorted_importance, cumulative, linewidth=2, color='darkblue')
    ax4.fill_between(sorted_importance, cumulative, alpha=0.3)
    
    # 标记重要分位数
    for q in [0.25, 0.5, 0.75, 0.9]:
        val = np.quantile(df['Importance'], q)
        ax4.axvline(val, color='red', linestyle='--', alpha=0.5)
        ax4.text(val, q, f'{q*100:.0f}%\n{val:.4f}', ha='center', fontsize=8)
    
    ax4.set_xlabel('Importance Score', fontsize=11)
    ax4.set_ylabel('Cumulative Proportion', fontsize=11)
    ax4.set_title('Cumulative Distribution of Importance', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_summary_table(df_sorted, network_stats, output_path):
    """生成汇总统计表并保存"""
    # Top 20 ROI
    top20 = df_sorted.head(20)[['Rank', 'AAL_Name', 'Network', 'Importance']]
    
    summary = {
        'top_20_roi': top20.to_dict('records'),
        'network_summary': network_stats.to_dict('records'),
        'overall_stats': {
            'mean_importance': float(df_sorted['Importance'].mean()),
            'std_importance': float(df_sorted['Importance'].std()),
            'min_importance': float(df_sorted['Importance'].min()),
            'max_importance': float(df_sorted['Importance'].max()),
            'median_importance': float(df_sorted['Importance'].median())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved: {output_path}")
    return summary

def main():
    print("="*60)
    print("ROI重要性可视化")
    print("="*60)
    
    # 加载数据
    print("\n1. 加载数据...")
    importance_data, aal_df = load_data()
    
    # 准备数据
    print("2. 准备数据...")
    df, df_sorted = prepare_data(importance_data, aal_df)
    
    # 输出路径
    output_dir = '/data3/Digital_Brain/NeuroTwin/checkpoints/neurotwin_finetune_pred13/interpretability/comprehensive_analysis_20260422_230827/visualizations'
    
    # 生成可视化
    print("\n3. 生成可视化...")
    
    print("   - Top 30 ROI重要性条形图...")
    plot_roi_importance_bar(df_sorted, f'{output_dir}/roi_importance_top30.png', top_n=30)
    
    print("   - 脑网络重要性统计...")
    network_stats = plot_network_importance(df, f'{output_dir}/roi_importance_by_network.png')
    
    print("   - ROI重要性热力图...")
    plot_importance_heatmap(df, f'{output_dir}/roi_importance_heatmap.png')
    
    print("   - 重要性分布分析...")
    plot_importance_distribution(df, f'{output_dir}/roi_importance_distribution.png')
    
    print("\n4. 生成汇总统计...")
    summary = generate_summary_table(df_sorted, network_stats, f'{output_dir}/roi_importance_summary.json')
    
    # 打印关键发现
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    
    print(f"\n总体统计:")
    print(f"  - 平均重要性: {summary['overall_stats']['mean_importance']:.4f}")
    print(f"  - 重要性标准差: {summary['overall_stats']['std_importance']:.4f}")
    print(f"  - 重要性范围: [{summary['overall_stats']['min_importance']:.4f}, {summary['overall_stats']['max_importance']:.4f}]")
    
    print(f"\nTop 5 最重要ROI:")
    for i, roi in enumerate(summary['top_20_roi'][:5], 1):
        print(f"  {i}. {roi['AAL_Name']} ({roi['Network']}): {roi['Importance']:.4f}")
    
    print(f"\n脑网络重要性排名:")
    for net in summary['network_summary']:
        print(f"  - {net['Network']}: {net['Mean_Importance']:.4f} (±{net['Std_Importance']:.4f}, n={int(net['Count'])})")
    
    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)

if __name__ == '__main__':
    main(parse_args())
