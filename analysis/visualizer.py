# coding=utf-8
"""VisualizationGenerator：分析结果可视化图表生成（散点/误差/热图/玻璃脑等）。"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from analysis.metrics import (NATURE_COLORS, NETWORK_GROUPS, ensure_dir,
                              fdr_correction, save_json)

log = logging.getLogger(__name__)

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}
class VisualizationGenerator:
    """Nature-style visualization generator"""

    # Unified low-saturation palette (NMI pastel family)
    PALETTE = {
        "primary": "#4A6FA5",      # steel blue
        "secondary": "#C45C3E",    # ochre / brick
        "tertiary": "#5A8F7B",     # sage green
        "quaternary": "#8E6B8A",   # muted purple
        "neutral": "#7D7D7D",      # gray
        "light": "#B0B0B0",        # light gray
        "bg": "#F5F5F5",           # near-white
    }

    def __init__(self, output_dir: Path):
        self.output_dir = ensure_dir(output_dir)
        self.fig_count = 0

        # Try importing matplotlib and apply Nature style
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update(NATURE_RC)
            self.plt = plt
            self.matplotlib_available = True
        except ImportError:
            log.warning("matplotlib not available, visualization will be skipped")
            self.matplotlib_available = False

        try:
            import seaborn as sns
            self.sns = sns
            self.seaborn_available = True
        except ImportError:
            self.seaborn_available = False

    def _save_pub_fig(self, fig, name: str):
        """Save publication-ready figure in SVG, PDF, and TIFF."""
        if not self.matplotlib_available:
            return
        self.fig_count += 1
        base = self.output_dir / f"{self.fig_count:03d}_{name}"
        fig.savefig(f"{base}.svg", bbox_inches="tight")
        fig.savefig(f"{base}.pdf", bbox_inches="tight")
        fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
        self.plt.close(fig)
        log.info(f"Saved: {base}.{{svg,pdf,tiff}}")

    def _nature_scatter(self, ax, x, y, color=None, alpha=0.4, s=4, **kwargs):
        """Nature-style scatter with subtle edge."""
        c = color or self.PALETTE["primary"]
        ax.scatter(x, y, c=c, alpha=alpha, s=s, edgecolors="none", **kwargs)

    def _nature_bar(self, ax, x, heights, color=None, width=0.6, **kwargs):
        """Nature-style bar with no edge."""
        c = color or self.PALETTE["primary"]
        ax.bar(x, heights, color=c, width=width, edgecolor="none", **kwargs)
    
    def plot_prediction_scatter(self, pred: np.ndarray, target: np.ndarray,
                                 max_points: int = 5000):
        """Prediction vs true scatter (Nature style)."""
        if not self.matplotlib_available:
            return

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        if len(pred_flat) > max_points:
            idx = np.random.choice(len(pred_flat), max_points, replace=False)
            pred_flat = pred_flat[idx]
            target_flat = target_flat[idx]

        fig, axes = self.plt.subplots(1, 2, figsize=(7.2, 3.0))

        # Scatter
        ax = axes[0]
        self._nature_scatter(ax, target_flat, pred_flat, alpha=0.35, s=3)
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color=self.PALETTE["secondary"], ls="--", lw=1.0,
                label="Perfect prediction")
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.legend(loc="upper left")

        # Residuals
        ax = axes[1]
        residuals = pred_flat - target_flat
        self._nature_scatter(ax, target_flat, residuals,
                             color=self.PALETTE["neutral"], alpha=0.35, s=3)
        ax.axhline(y=0, color=self.PALETTE["secondary"], ls="--", lw=1.0)
        ax.set_xlabel("True values")
        ax.set_ylabel("Residuals")

        self._save_pub_fig(fig, 'prediction_scatter')
    
    def plot_error_distribution(self, pred: np.ndarray, target: np.ndarray):
        """Error distribution (Nature style)."""
        if not self.matplotlib_available:
            return

        error = (pred - target).reshape(-1)
        abs_error = np.abs(error)

        fig, axes = self.plt.subplots(1, 3, figsize=(7.2, 2.4))

        # Error histogram
        ax = axes[0]
        ax.hist(error, bins=80, color=self.PALETTE["primary"],
                edgecolor="none", alpha=0.85)
        ax.axvline(x=0, color=self.PALETTE["secondary"], ls="--", lw=1.0)
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")

        # Absolute error histogram
        ax = axes[1]
        ax.hist(abs_error, bins=80, color=self.PALETTE["tertiary"],
                edgecolor="none", alpha=0.85)
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Frequency")

        # Q-Q plot
        ax = axes[2]
        from scipy import stats
        stats.probplot(error, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(self.PALETTE["primary"])
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_alpha(0.5)
        ax.get_lines()[1].set_color(self.PALETTE["secondary"])
        ax.get_lines()[1].set_linewidth(1.0)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered values")

        self._save_pub_fig(fig, 'error_distribution')
    
    def plot_temporal_prediction(self, pred: np.ndarray, target: np.ndarray,
                                  sample_idx: int = 0, roi_idx: int = 0):
        """Temporal prediction trace (Nature style)."""
        if not self.matplotlib_available:
            return

        pred_sample = pred[sample_idx, roi_idx]
        target_sample = target[sample_idx, roi_idx]

        fig, axes = self.plt.subplots(2, 1, figsize=(7.2, 4.0))

        # Prediction vs true
        ax = axes[0]
        for w in range(pred_sample.shape[0]):
            offset = w * pred_sample.shape[1]
            x = np.arange(offset, offset + pred_sample.shape[1])
            if w == 0:
                ax.plot(x, target_sample[w], color=self.PALETTE["primary"],
                        lw=1.0, label="True")
                ax.plot(x, pred_sample[w], color=self.PALETTE["secondary"],
                        lw=1.0, ls="--", label="Predicted")
            else:
                ax.plot(x, target_sample[w], color=self.PALETTE["primary"], lw=1.0)
                ax.plot(x, pred_sample[w], color=self.PALETTE["secondary"],
                        lw=1.0, ls="--")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Signal value")
        ax.legend(loc="upper right")

        # Error
        ax = axes[1]
        error_sample = pred_sample - target_sample
        for w in range(error_sample.shape[0]):
            offset = w * error_sample.shape[1]
            x = np.arange(offset, offset + error_sample.shape[1])
            ax.plot(x, error_sample[w], color=self.PALETTE["neutral"], lw=0.8)
        ax.axhline(y=0, color=self.PALETTE["secondary"], ls="--", lw=1.0)
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Prediction error")

        self._save_pub_fig(fig, f'temporal_prediction_s{sample_idx}_r{roi_idx}')
    
    def plot_roi_heatmap(self, roi_metrics: Dict[str, np.ndarray]):
        """ROI-level heatmaps (Nature style)."""
        if not self.matplotlib_available:
            return

        fig, axes = self.plt.subplots(1, 3, figsize=(7.2, 2.8))

        # MAE heatmap
        ax = axes[0]
        roi_mae = roi_metrics['roi_mae']
        im = ax.imshow(roi_mae.reshape(-1, 1), aspect='auto', cmap='YlOrBr')
        ax.set_xlabel("MAE")
        ax.set_ylabel("ROI index")
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5)

        # PCC heatmap
        ax = axes[1]
        roi_pcc = roi_metrics['roi_pcc']
        im = ax.imshow(roi_pcc.reshape(-1, 1), aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xlabel("PCC")
        ax.set_ylabel("ROI index")
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5)

        # Variance heatmap
        ax = axes[2]
        roi_var = roi_metrics['roi_variance']
        im = ax.imshow(roi_var.reshape(-1, 1), aspect='auto', cmap='Blues')
        ax.set_xlabel("Variance")
        ax.set_ylabel("ROI index")
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5)

        self._save_pub_fig(fig, 'roi_heatmap')
    
    def plot_sample_pcc_distribution(self, sample_pcc: np.ndarray):
        """Sample PCC distribution (Nature style)."""
        if not self.matplotlib_available:
            return

        fig, axes = self.plt.subplots(1, 2, figsize=(7.2, 2.6))

        # Histogram
        ax = axes[0]
        ax.hist(sample_pcc, bins=50, color=self.PALETTE["primary"],
                edgecolor="none", alpha=0.85)
        ax.axvline(x=sample_pcc.mean(), color=self.PALETTE["secondary"],
                   ls="--", lw=1.0, label=f"Mean={sample_pcc.mean():.3f}")
        ax.axvline(x=np.median(sample_pcc), color=self.PALETTE["tertiary"],
                   ls="--", lw=1.0, label=f"Median={np.median(sample_pcc):.3f}")
        ax.set_xlabel("Sample PCC")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper left")

        # CDF
        ax = axes[1]
        sorted_pcc = np.sort(sample_pcc)
        cdf = np.arange(1, len(sorted_pcc) + 1) / len(sorted_pcc)
        ax.plot(sorted_pcc, cdf, color=self.PALETTE["primary"], lw=1.2)
        ax.axvline(x=0, color=self.PALETTE["secondary"], ls="--", lw=1.0)
        ax.set_xlabel("Sample PCC")
        ax.set_ylabel("CDF")

        self._save_pub_fig(fig, 'sample_pcc_distribution')
    
    def plot_moe_gate_analysis(self, gate_weights: Optional[np.ndarray], metrics: Dict):
        """MoE gate analysis (Nature style)."""
        if not self.matplotlib_available or gate_weights is None:
            return

        n_experts = gate_weights.shape[1]
        expert_colors = [self.PALETTE["primary"], self.PALETTE["secondary"],
                         self.PALETTE["tertiary"], self.PALETTE["quaternary"]]
        expert_colors = (expert_colors * ((n_experts // 4) + 1))[:n_experts]

        fig, axes = self.plt.subplots(2, 2, figsize=(7.2, 5.5))

        # Mean gate weights
        ax = axes[0, 0]
        gate_mean = gate_weights.mean(axis=0)
        self._nature_bar(ax, range(n_experts), gate_mean, color=expert_colors)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Average gate weight")
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels([f"E{i}" for i in range(n_experts)])

        # Gate weight distributions
        ax = axes[0, 1]
        for i in range(n_experts):
            ax.hist(gate_weights[:, i], bins=30, alpha=0.6,
                    color=expert_colors[i], label=f"E{i}", edgecolor="none")
        ax.set_xlabel("Gate weight")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")

        # Gate weights heatmap (first N samples)
        ax = axes[1, 0]
        n_samples_to_plot = min(100, gate_weights.shape[0])
        im = ax.imshow(gate_weights[:n_samples_to_plot], aspect="auto", cmap="YlOrBr")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Sample index")
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5)

        # Expert load (top-1)
        ax = axes[1, 1]
        top_expert_per_sample = np.argmax(gate_weights, axis=1)
        expert_counts = np.bincount(top_expert_per_sample, minlength=n_experts)
        self._nature_bar(ax, range(n_experts), expert_counts, color=expert_colors)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Count (top-1)")
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels([f"E{i}" for i in range(n_experts)])

        self._save_pub_fig(fig, 'moe_gate_analysis')
    
    def plot_pathology_correlation(self, pathology_scores: np.ndarray, sample_pcc: np.ndarray):
        """Pathology score vs performance (Nature style)."""
        if not self.matplotlib_available:
            return

        valid_mask = ~np.isnan(pathology_scores)
        if valid_mask.sum() < 10:
            return

        pathology_valid = pathology_scores[valid_mask]
        pcc_valid = sample_pcc[valid_mask]

        fig, axes = self.plt.subplots(1, 2, figsize=(7.2, 3.0))

        # Scatter + fit
        ax = axes[0]
        self._nature_scatter(ax, pathology_valid, pcc_valid, alpha=0.45, s=5)
        z = np.polyfit(pathology_valid, pcc_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(pathology_valid.min(), pathology_valid.max(), 100)
        ax.plot(x_line, p(x_line), color=self.PALETTE["secondary"],
                ls="--", lw=1.0, label=f"Linear fit (slope={z[0]:.3f})")
        corr = np.corrcoef(pathology_valid, pcc_valid)[0, 1]
        ax.set_xlabel("Pathology score")
        ax.set_ylabel("Sample PCC")
        ax.legend(loc="upper right")

        # Binned boxplot
        ax = axes[1]
        n_bins = 5
        bins = np.percentile(pathology_valid, np.linspace(0, 100, n_bins + 1))
        bin_labels = [f"Q{i+1}" for i in range(n_bins)]
        binned_pcc = []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (pathology_valid >= bins[i]) & (pathology_valid < bins[i + 1])
            else:
                mask = (pathology_valid >= bins[i]) & (pathology_valid <= bins[i + 1])
            binned_pcc.append(pcc_valid[mask])

        bp = ax.boxplot(binned_pcc, labels=bin_labels, patch_artist=True,
                        medianprops=dict(color=self.PALETTE["secondary"], lw=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor(self.PALETTE["primary"])
            patch.set_alpha(0.6)
        ax.set_xlabel("Pathology score quartile")
        ax.set_ylabel("Sample PCC")

        self._save_pub_fig(fig, 'pathology_correlation')
    
    def plot_expert_hamd_distribution(self, expert_hamd_analysis: Dict[str, Any]):
        """
        绘制各专家处理样本的HAMD分数分布可视化
        
        Args:
            expert_hamd_analysis: analyze_expert_hamd_distribution 的返回结果
        """
        if not self.matplotlib_available or not expert_hamd_analysis:
            return
        
        expert_stats = expert_hamd_analysis.get('expert_stats', {})
        n_experts = expert_hamd_analysis.get('n_experts', 0)
        
        if n_experts == 0:
            return
        
        # 使用原始HAMD分数进行可视化
        fig = self.plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 颜色方案
        colors = self.plt.cm.Set2(np.linspace(0, 1, n_experts))
        expert_keys = [f'E{i}' for i in range(n_experts)]
        
        # ═════════════════════════════════════════════════════════════════
        # 第1行: 箱线图和小提琴图
        # ═════════════════════════════════════════════════════════════════
        
        # 收集有效数据
        valid_experts = []
        valid_data = []
        valid_labels = []
        valid_colors = []
        
        for i, key in enumerate(expert_keys):
            if key in expert_stats and expert_stats[key].get('matched_count', 0) > 0:
                values = expert_stats[key]['original_hamd'].get('values', [])
                if len(values) > 0:
                    valid_experts.append(key)
                    valid_data.append(values)
                    valid_labels.append(f'{key}\n(n={len(values)})')
                    valid_colors.append(colors[i])
        
        if len(valid_data) == 0:
            log.warning("No valid HAMD data for visualization")
            return
        
        # 1.1 箱线图
        ax1 = fig.add_subplot(gs[0, 0])
        bp = ax1.boxplot(valid_data, labels=[e.split('\n')[0] for e in valid_labels], 
                         patch_artist=True, showmeans=True, 
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_xlabel('Expert')
        ax1.set_ylabel('HAMD Score (Original)')
        ax1.set_title('HAMD Distribution by Expert (Boxplot)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加统计注释
        if self.seaborn_available:
            ax1 = fig.add_subplot(gs[0, 1])
            import pandas as pd
            df_data = []
            for key, values in zip(valid_experts, valid_data):
                for v in values:
                    df_data.append({'Expert': key, 'HAMD': v})
            df = pd.DataFrame(df_data)
            self.sns.violinplot(data=df, x='Expert', y='HAMD', ax=ax1, palette=valid_colors)
            ax1.set_title('HAMD Distribution by Expert (Violin)')
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            # 使用matplotlib的hist作为替代
            ax1 = fig.add_subplot(gs[0, 1])
            for i, (key, values) in enumerate(zip(valid_experts, valid_data)):
                ax1.hist(values, bins=15, alpha=0.5, label=key, color=valid_colors[i])
            ax1.set_xlabel('HAMD Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('HAMD Distribution by Expert (Histogram)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 1.3 均值和置信区间
        ax1 = fig.add_subplot(gs[0, 2])
        means = []
        stds = []
        counts = []
        for key in valid_experts:
            stats = expert_stats[key]['original_hamd']
            means.append(stats['mean'])
            stds.append(stats['std'])
            counts.append(stats.get('matched_count', expert_stats[key].get('matched_count', 0)))
        
        x_pos = np.arange(len(valid_experts))
        # 使用标准误计算误差条
        sems = [s / np.sqrt(c) if c > 0 else 0 for s, c in zip(stds, counts)]
        bars = ax1.bar(x_pos, means, yerr=sems, capsize=5, color=valid_colors, 
                       edgecolor='black', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(valid_experts)
        ax1.set_xlabel('Expert')
        ax1.set_ylabel('Mean HAMD Score')
        ax1.set_title('Mean HAMD by Expert (±SEM)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, mean, sem in zip(bars, means, sems):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.3,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        # ═════════════════════════════════════════════════════════════════
        # 第2行: 统计摘要和分布对比
        # ═════════════════════════════════════════════════════════════════
        
        # 2.1 统计摘要表
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        
        summary_text = "Expert HAMD Statistics (Original)\n" + "="*50 + "\n\n"
        summary_text += f"{'Expert':<10} {'Count':<8} {'Mean':<8} {'Std':<8} {'Median':<8}\n"
        summary_text += "-"*50 + "\n"
        
        for key in valid_experts:
            stats = expert_stats[key]['original_hamd']
            summary_text += f"{key:<10} {stats.get('matched_count', expert_stats[key].get('matched_count', 0)):<8} "
            summary_text += f"{stats['mean']:<8.2f} {stats['std']:<8.2f} {stats['median']:<8.2f}\n"
        
        # 添加全样本统计
        all_values = [v for vals in valid_data for v in vals]
        if len(all_values) > 0:
            summary_text += "-"*50 + "\n"
            summary_text += f"{'All':<10} {len(all_values):<8} "
            summary_text += f"{np.mean(all_values):<8.2f} {np.std(all_values):<8.2f} {np.median(all_values):<8.2f}\n"
        
        ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', family='monospace')
        
        # 2.2 累积分布函数 (CDF)
        ax2 = fig.add_subplot(gs[1, 1])
        for key, values, color in zip(valid_experts, valid_data, valid_colors):
            sorted_data = np.sort(values)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax2.plot(sorted_data, cdf, label=key, color=color, lw=2)
        ax2.set_xlabel('HAMD Score')
        ax2.set_ylabel('CDF')
        ax2.set_title('HAMD Cumulative Distribution by Expert')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2.3 分位数对比
        ax2 = fig.add_subplot(gs[1, 2])
        percentiles = [25, 50, 75]
        x = np.arange(len(percentiles))
        width = 0.8 / len(valid_experts)
        
        for i, (key, values, color) in enumerate(zip(valid_experts, valid_data, valid_colors)):
            quants = [np.percentile(values, p) for p in percentiles]
            offset = (i - len(valid_experts)/2 + 0.5) * width
            ax2.bar(x + offset, quants, width, label=key, color=color, alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Q1 (25%)', 'Median (50%)', 'Q3 (75%)'])
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('HAMD Score')
        ax2.set_title('HAMD Percentiles by Expert')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ═════════════════════════════════════════════════════════════════
        # 第3行: 配对检验和专家负载
        # ═════════════════════════════════════════════════════════════════
        
        # 3.1 配对统计检验
        ax3 = fig.add_subplot(gs[2, 0])
        pairwise_tests = expert_hamd_analysis.get('pairwise_tests', {})
        
        if pairwise_tests:
            test_labels = []
            pvalues = []
            significant = []
            
            for test_name, test_result in pairwise_tests.items():
                test_labels.append(test_name.replace('_vs_', '\nvs\n'))
                pvalues.append(test_result['pvalue'])
                significant.append(test_result.get('significant', False))
            
            colors_sig = ['red' if s else 'gray' for s in significant]
            bars = ax3.barh(range(len(test_labels)), pvalues, color=colors_sig, alpha=0.7)
            ax3.axvline(x=0.05, color='red', linestyle='--', lw=2, label='p=0.05')
            ax3.set_yticks(range(len(test_labels)))
            ax3.set_yticklabels(test_labels, fontsize=8)
            ax3.set_xlabel('p-value')
            ax3.set_title('Pairwise Mann-Whitney U Tests')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='x')
            
            # 添加p值标签
            for bar, pval in zip(bars, pvalues):
                ax3.text(pval + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{pval:.4f}', va='center', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No pairwise tests available\n(insufficient data)',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Pairwise Statistical Tests')
        
        # 3.2 专家样本负载分布
        ax3 = fig.add_subplot(gs[2, 1])
        sample_counts = [expert_stats[key].get('sample_count', 0) for key in valid_experts]
        matched_counts = [expert_stats[key].get('matched_count', 0) for key in valid_experts]
        
        x_pos = np.arange(len(valid_experts))
        width = 0.35
        bars1 = ax3.bar(x_pos - width/2, sample_counts, width, label='Total Samples',
                       color=valid_colors, alpha=0.6, edgecolor='black')
        bars2 = ax3.bar(x_pos + width/2, matched_counts, width, label='Matched HAMD',
                       color=valid_colors, alpha=1.0, edgecolor='black')
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(valid_experts)
        ax3.set_xlabel('Expert')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Expert Sample Load')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 3.3 HAMD范围热力图
        ax3 = fig.add_subplot(gs[2, 2])
        
        # 创建HAMD范围统计
        hamd_ranges = ['0-7\n(Normal)', '8-16\n(Mild)', '17-23\n(Moderate)', '24-52\n(Severe)']
        range_matrix = np.zeros((len(valid_experts), 4))
        
        for i, (key, values) in enumerate(zip(valid_experts, valid_data)):
            values_arr = np.array(values)
            range_matrix[i, 0] = np.sum((values_arr >= 0) & (values_arr <= 7))
            range_matrix[i, 1] = np.sum((values_arr >= 8) & (values_arr <= 16))
            range_matrix[i, 2] = np.sum((values_arr >= 17) & (values_arr <= 23))
            range_matrix[i, 3] = np.sum((values_arr >= 24) & (values_arr <= 52))
        
        # 归一化为百分比
        range_matrix_pct = range_matrix / range_matrix.sum(axis=1, keepdims=True) * 100
        
        im = ax3.imshow(range_matrix_pct, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(hamd_ranges, fontsize=8)
        ax3.set_yticks(range(len(valid_experts)))
        ax3.set_yticklabels(valid_experts)
        ax3.set_xlabel('HAMD Severity Range')
        ax3.set_ylabel('Expert')
        ax3.set_title('HAMD Severity Distribution (%)')
        
        # 添加数值标注
        for i in range(len(valid_experts)):
            for j in range(4):
                text = ax3.text(j, i, f'{range_matrix_pct[i, j]:.1f}%',
                               ha="center", va="center", color="black" if range_matrix_pct[i, j] < 50 else "white",
                               fontsize=8)
        
        self.plt.colorbar(im, ax=ax3, label='Percentage')
        
        # 保存图形
        self._save_pub_fig(fig, 'expert_hamd_distribution')

    def plot_feature_importance(self, importance: Dict[str, np.ndarray]):
        """Feature importance (Nature style)."""
        if not self.matplotlib_available:
            return

        roi_importance = importance['roi_importance']
        ranked_indices = importance['roi_importance_rank'][:20]

        fig, axes = self.plt.subplots(1, 2, figsize=(7.2, 3.2))

        # All ROI importance
        ax = axes[0]
        colors = [self.PALETTE["secondary"] if i in ranked_indices[:10]
                  else self.PALETTE["light"] for i in range(len(roi_importance))]
        self._nature_bar(ax, range(len(roi_importance)), roi_importance, color=colors)
        ax.set_xlabel("ROI index")
        ax.set_ylabel("Importance (MAE increase)")

        # Top 20
        ax = axes[1]
        top_importance = roi_importance[ranked_indices]
        ax.barh(range(len(ranked_indices)), top_importance,
                color=self.PALETTE["primary"], height=0.6)
        ax.set_yticks(range(len(ranked_indices)))
        ax.set_yticklabels([f"ROI {i}" for i in ranked_indices])
        ax.set_xlabel("Importance (MAE increase)")
        ax.invert_yaxis()

        self._save_pub_fig(fig, 'feature_importance')

    def plot_significant_brain_regions(self, importance: Dict[str, np.ndarray], 
                                       fdr_alpha: float = 0.001):
        """
        可视化 FDR 校正后的显著脑区（玻璃脑图，按脑网络分组，Nature 风格）
        
        Args:
            importance: 特征重要性字典，包含 roi_importance 和 roi_pvalues
            fdr_alpha: FDR 校正后的显著性阈值（默认 0.001）
        """
        if not self.matplotlib_available:
            log.warning("matplotlib not available, skipping brain region visualization")
            return
        
        try:
            from nilearn import datasets, plotting, image
            from matplotlib.colors import LinearSegmentedColormap, to_rgb
            from matplotlib.patches import Patch
        except ImportError:
            log.warning("nilearn not available, skipping glass brain visualization")
            return
        
        roi_importance = importance['roi_importance']
        roi_pvalues = importance.get('roi_pvalues', None)
        
        if roi_pvalues is None:
            log.warning("No p-values available, cannot perform FDR correction")
            return
        
        # FDR 校正
        rejected, corrected_pvalues = fdr_correction(roi_pvalues, alpha=fdr_alpha)
        
        # 筛选显著脑区（FDR 校正后 p < fdr_alpha）
        significant_mask = corrected_pvalues < fdr_alpha
        significant_indices = np.where(significant_mask)[0]
        
        if len(significant_indices) == 0:
            log.warning(f"No significant regions found after FDR correction (p < {fdr_alpha})")
            return
        
        log.info(f"Found {len(significant_indices)} significant regions after FDR correction (p < {fdr_alpha})")
        
        # 按脑网络分组筛选显著脑区
        significant_network_groups = {}
        for net_name, region_indexes in NETWORK_GROUPS.items():
            significant_in_net = [idx for idx in region_indexes if idx in significant_indices]
            if significant_in_net:
                significant_network_groups[net_name] = significant_in_net
        
        if not significant_network_groups:
            log.warning("No significant regions belong to predefined networks")
            return
        
        # 加载 AAL116 atlas
        try:
            atlas = datasets.fetch_atlas_aal(version="SPM12")
            atlas_img = atlas.maps
            labels = list(atlas.labels)
            indices = list(atlas.indices)
            
            # 构建区域信息
            regions = []
            for label_name, label_value in zip(labels, indices):
                value = int(label_value)
                if value == 0 or "background" in label_name.lower():
                    continue
                regions.append({"name": label_name, "value": value})
            
            atlas_nii = image.load_img(atlas_img)
            atlas_data = atlas_nii.get_fdata()
            
        except Exception as e:
            log.warning(f"Failed to load AAL atlas: {e}")
            return
        
        # 为每个脑网络构建颜色映射
        def _make_cmap(hex_color, alpha=0.88):
            rgb = to_rgb(hex_color)
            return LinearSegmentedColormap.from_list(
                f"region_{hex_color.lstrip('#')}",
                [(1, 1, 1, 0), (*rgb, 0.18), (*rgb, alpha)],
                N=256,
            )
        
        # 构建区域颜色映射
        all_significant_indices = sorted({
            idx 
            for region_list in significant_network_groups.values() 
            for idx in region_list
        })
        
        if len(all_significant_indices) > len(NATURE_COLORS):
            log.warning(f"Need {len(all_significant_indices)} colors but only have {len(NATURE_COLORS)}")
        
        region_colors = {
            idx: NATURE_COLORS[i] 
            for i, idx in enumerate(all_significant_indices)
        }
        
        # 为每个网络绘制玻璃脑图
        for net_name, region_indexes in significant_network_groups.items():
            log.info(f"Plotting network: {net_name} (regions: {region_indexes})")
            
            masks = []
            names = []
            colors = []
            
            for idx in region_indexes:
                if idx < 1 or idx > len(regions):
                    log.warning(f"Region index {idx} out of range")
                    continue
                    
                region = regions[idx - 1]
                mask_data = (atlas_data == region["value"]).astype(np.float64)
                voxel_count = int(mask_data.sum())
                
                if voxel_count == 0:
                    log.warning(f"AAL{idx} ({region['name']}) has no voxels")
                    continue
                
                mask_img = image.new_img_like(atlas_nii, mask_data, copy_header=True)
                masks.append(mask_img)
                names.append(f"AAL{idx} {region['name']}")
                colors.append(region_colors[idx])
                
                log.info(f"  AAL{idx:3d} value={region['value']:4d} voxels={voxel_count:5d} {region['name']}")
            
            if not masks:
                continue
            
            n_regions = len(masks)
            
            # 绘制第一个区域作为基础
            first_cmap = _make_cmap(colors[0])
            display = plotting.plot_glass_brain(
                masks[0],
                title=f"{net_name} (n={n_regions}, p<{fdr_alpha})",
                display_mode="lyrz",
                threshold=0.5,
                cmap=first_cmap,
                vmin=0.0,
                vmax=1.0,
                colorbar=False,
                plot_abs=False,
                black_bg=False,
            )
            
            # 叠加其他区域
            for i in range(1, n_regions):
                cmap = _make_cmap(colors[i])
                display.add_overlay(
                    masks[i],
                    threshold=0.5,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    colorbar=False,
                )
            
            # 添加图例
            legend_patches = [
                Patch(facecolor=colors[i], 
                      edgecolor="#4D4D4D", linewidth=0.5,
                      label=f"{names[i]} (p={corrected_pvalues[region_indexes[i]]:.2e})")
                for i in range(n_regions)
            ]
            display.frame_axes.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=min(3, n_regions),
                fontsize=7,
                frameon=False,
                bbox_to_anchor=(0.5, -0.08),
            )
            
            # 保存图像
            output_dir = self.output_dir / 'brain_regions'
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = output_dir / f'glass_brain_{net_name}_significant.png'
            display.savefig(str(fname), dpi=300, bbox_inches="tight")
            log.info(f"Saved: {fname}")
            
            self.plt.close(display)
        
        # 保存统计结果
        stats_results = {
            'fdr_alpha': fdr_alpha,
            'n_significant': int(len(significant_indices)),
            'significant_indices': significant_indices.tolist(),
            'corrected_pvalues': corrected_pvalues[significant_indices].tolist(),
            'roi_importance': roi_importance[significant_indices].tolist(),
            'network_groups': {
                k: v for k, v in significant_network_groups.items()
            },
        }
        
        save_json(self.output_dir / 'significant_brain_regions.json', stats_results)
        log.info(f"Saved statistical results to significant_brain_regions.json")

    def plot_temporal_metrics(self, temporal_metrics: Dict[str, np.ndarray]):
        """Temporal metrics (Nature style)."""
        if not self.matplotlib_available:
            return

        fig, axes = self.plt.subplots(2, 2, figsize=(7.2, 5.5))

        # Window-level MAE
        ax = axes[0, 0]
        window_mae = temporal_metrics['window_mae']
        ax.plot(range(len(window_mae)), window_mae, "o-",
                color=self.PALETTE["primary"], lw=1.0, markersize=3)
        ax.set_xlabel("Window index")
        ax.set_ylabel("MAE")

        # Sequence-level MAE
        ax = axes[0, 1]
        seq_mae = temporal_metrics['seq_mae']
        ax.plot(range(len(seq_mae)), seq_mae, "s-",
                color=self.PALETTE["tertiary"], lw=1.0, markersize=3)
        ax.set_xlabel("Sequence index")
        ax.set_ylabel("MAE")

        # Temporal variance comparison
        ax = axes[1, 0]
        pred_var = temporal_metrics['pred_temporal_var']
        target_var = temporal_metrics['target_temporal_var']
        x = np.arange(len(pred_var))
        width = 0.35
        self._nature_bar(ax, x - width / 2, target_var,
                         color=self.PALETTE["primary"], width=width, label="Target")
        self._nature_bar(ax, x + width / 2, pred_var,
                         color=self.PALETTE["secondary"], width=width, label="Predicted")
        ax.set_xlabel("Window index")
        ax.set_ylabel("Temporal variance")
        ax.legend(loc="upper right")

        # Variance ratio
        ax = axes[1, 1]
        var_ratio = temporal_metrics['temporal_var_ratio']
        ax.plot(range(len(var_ratio)), var_ratio, "D-",
                color=self.PALETTE["quaternary"], lw=1.0, markersize=3)
        ax.axhline(y=1.0, color=self.PALETTE["secondary"], ls="--", lw=1.0)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Variance ratio (pred/target)")

        self._save_pub_fig(fig, 'temporal_metrics')

    def plot_comprehensive_dashboard(self, metrics: Dict, artifacts: Dict):
        """Comprehensive dashboard (Nature style)."""
        if not self.matplotlib_available:
            return

        fig = self.plt.figure(figsize=(7.2, 5.5))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        # 1. Key metrics text
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")
        key_metrics = ["MAE", "RMSE", "PCC", "R2"]
        text = "Key metrics\n" + "=" * 18 + "\n"
        for k in key_metrics:
            if k in metrics:
                text += f"{k}: {metrics[k]:.4f}\n"
        ax.text(0.1, 0.5, text, fontsize=6, family="monospace", verticalalignment="center")

        # 2. Sample PCC distribution
        ax = fig.add_subplot(gs[0, 1])
        sample_pcc = artifacts['sample_pcc']
        ax.hist(sample_pcc, bins=30, color=self.PALETTE["primary"],
                edgecolor="none", alpha=0.85)
        ax.axvline(sample_pcc.mean(), color=self.PALETTE["secondary"],
                   ls="--", lw=1.0, label=f"Mean={sample_pcc.mean():.3f}")
        ax.set_xlabel("Sample PCC")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left")

        # 3. ROI MAE
        ax = fig.add_subplot(gs[0, 2])
        roi_mae = artifacts['roi_metrics']['roi_mae']
        self._nature_bar(ax, range(len(roi_mae)), roi_mae,
                         color=self.PALETTE["tertiary"])
        ax.set_xlabel("ROI index")
        ax.set_ylabel("MAE")

        # 4. Error stats text
        ax = fig.add_subplot(gs[1, 0])
        ax.axis("off")
        error_stats = ["MAE_50th", "MAE_90th", "MAE_95th", "Error_Skewness"]
        text = "Error stats\n" + "=" * 18 + "\n"
        for k in error_stats:
            if k in metrics:
                text += f"{k}: {metrics[k]:.4f}\n"
        ax.text(0.1, 0.5, text, fontsize=6, family="monospace", verticalalignment="center")

        # 5. Prediction vs true scatter
        ax = fig.add_subplot(gs[1, 1:])
        pred = artifacts['pred'].reshape(-1)
        target = artifacts['target'].reshape(-1)
        idx = np.random.choice(len(pred), min(5000, len(pred)), replace=False)
        self._nature_scatter(ax, target[idx], pred[idx], alpha=0.35, s=2)
        min_val, max_val = target.min(), target.max()
        ax.plot([min_val, max_val], [min_val, max_val],
                color=self.PALETTE["secondary"], ls="--", lw=1.0)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

        # 6. Temporal example
        ax = fig.add_subplot(gs[2, :2])
        pred_sample = artifacts['pred'][0, 0]
        target_sample = artifacts['target'][0, 0]
        for w in range(pred_sample.shape[0]):
            offset = w * pred_sample.shape[1]
            x = np.arange(offset, offset + pred_sample.shape[1])
            if w == 0:
                ax.plot(x, target_sample[w], color=self.PALETTE["primary"],
                        lw=0.8, label="True")
                ax.plot(x, pred_sample[w], color=self.PALETTE["secondary"],
                        lw=0.8, ls="--", label="Pred")
            else:
                ax.plot(x, target_sample[w], color=self.PALETTE["primary"], lw=0.8)
                ax.plot(x, pred_sample[w], color=self.PALETTE["secondary"],
                        lw=0.8, ls="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")

        # 7. Window-level MAE
        ax = fig.add_subplot(gs[2, 2])
        window_mae = artifacts['temporal_metrics']['window_mae']
        ax.plot(range(len(window_mae)), window_mae, "o-",
                color=self.PALETTE["quaternary"], lw=1.0, markersize=3)
        ax.set_xlabel("Window")
        ax.set_ylabel("MAE")

        self._save_pub_fig(fig, 'comprehensive_dashboard')


# ═══════════════════════════════════════════════════════════════════════════════
