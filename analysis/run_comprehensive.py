# coding=utf-8
"""NeuroTwin 综合模型分析 CLI 入口。

用法：
    python analysis/run_comprehensive.py --finetuned_weight <ckpt> [options]
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # noqa: E402 - 需在项目导入前完成 bootstrap

from utils.common import set_seed, str2bool  # noqa: E402
from train.losses import UncertaintyWeightedHybridLoss  # noqa: E402
from analysis.metrics import (ensure_dir, export_significance_to_xlsx,  # noqa: E402
                              load_hamd_data, resolve_device, save_json)
from analysis.checkpoint import build_model_and_loader  # noqa: E402
from analysis.analyzer import ModelAnalyzer  # noqa: E402
from analysis.visualizer import VisualizationGenerator  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Comprehensive Model Analysis')
    
    # 基础参数
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='auto')
    
    # 路径参数
    parser.add_argument('--data_root', type=str, default='/data3/Digital_Brain/AMD/data')
    parser.add_argument('--finetuned_weight', type=str,
                        default='/data3/Digital_Brain/NeuroTwin/checkpoints/neurotwin_finetune_pred13/finetuned_best.pt')
    parser.add_argument('--output_dir', type=str,
                        default='/data3/Digital_Brain/NeuroTwin/checkpoints/neurotwin_finetune_pred13/analysis_results')
    parser.add_argument('--run_name', type=str,
                        default='comprehensive_analysis')
    parser.add_argument('--clinical_file', type=str,
                        default='Rest-meta-MDD-HAMD-V1-V2-Merge-Normalize.xlsx')
    
    # HAMD分析参数
    parser.add_argument('--hamd_normalized_file', type=str,
                       default='/data3/Digital_Brain/AMD/data/Rest-meta-MDD-HAMD-V1-V2-Merge-Normalize.xlsx',
                       help='归一化HAMD文件路径（用于匹配样本）')
    parser.add_argument('--hamd_original_file', type=str,
                       default='/data3/Digital_Brain/AMD/data/Rest-meta-MDD-HAMD-V1-V2-Merge.xlsx',
                       help='原始HAMD文件路径（用于可视化）')
    parser.add_argument('--expert_assignment_method', type=str, default='top1',
                       choices=['top1', 'threshold'],
                       help='样本分配给专家的方法：top1表示分配给权重最高的专家')
    
    # 数据参数
    parser.add_argument('--num_rois', type=int, default=116)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--in_window', type=int, default=6)
    parser.add_argument('--pred_window', type=int, default=1)
    parser.add_argument('--total_windows', type=int, default=9)
    parser.add_argument('--pathology_input_dim', type=int, default=1)
    parser.add_argument('--pathology_dim', type=int, default=32)
    
    # 模型参数
    parser.add_argument('--n_block', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=3)
    parser.add_argument('--ode_steps', type=int, default=6)
    parser.add_argument('--ode_hidden_dim', type=int, default=256)
    parser.add_argument('--stochastic_depth_rate', type=float, default=0.10)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=256)
    parser.add_argument('--moe_use_shared_expert', type=str2bool, default=True)
    parser.add_argument('--moe_router_cond_only', type=str2bool, default=True)
    parser.add_argument('--moe_use_argmax', type=str2bool, default=False)
    parser.add_argument('--moe_inference_temperature', type=float, default=0.3)
    
    # 加载参数
    parser.add_argument('--strict_load', type=str2bool, default=False)
    
    # DataLoader 参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=str2bool, default=True)
    parser.add_argument('--persistent_workers', type=str2bool, default=True)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--cache_in_memory', type=str2bool, default=False)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--stratify_bins', type=int, default=5)
    
    # 损失权重
    parser.add_argument('--moe_load_balance_weight', type=float, default=0.01)
    parser.add_argument('--moe_entropy_weight', type=float, default=1e-3)
    parser.add_argument('--moe_z_loss_weight', type=float, default=1e-3)
    parser.add_argument('--moe_gate_temp_end', type=float, default=1.0)
    
    # 特征重要性参数
    parser.add_argument('--compute_feature_importance', type=str2bool, default=True)
    parser.add_argument('--feature_importance_samples', type=int, default=50)
    parser.add_argument('--feature_importance_method', type=str, default='balanced',
                       choices=['fast', 'balanced', 'precise', 'permutation'],
                       help='特征重要性计算方法: fast(~1秒), balanced(~5分钟,推荐), precise(~25分钟)')
    parser.add_argument('--feature_importance_permutations', type=int, default=200)
    parser.add_argument('--generate_visualizations', type=str2bool, default=True)
    parser.add_argument('--save_arrays', type=str2bool, default=True)
    
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  §6 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    
    log.info(f'Device: {device}')
    log.info(f'Checkpoint: {args.finetuned_weight}')
    
    # 构建模型和数据加载器
    model, val_data, compat_info = build_model_and_loader(args, device)
    criterion = UncertaintyWeightedHybridLoss().to(device)
    
    # 创建输出目录
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = ensure_dir(Path(args.output_dir) / f"{args.run_name}_{run_stamp}")
    viz_dir = ensure_dir(save_dir / 'visualizations')
    
    log.info(f'Output directory: {save_dir}')
    
    # 初始化分析器
    analyzer = ModelAnalyzer(model, device)
    visualizer = VisualizationGenerator(viz_dir)
    
    # ═══════════════════════════════════════════════════════════════════════
    #  1. 核心评估
    # ═══════════════════════════════════════════════════════════════════════
    
    metrics, artifacts = analyzer.evaluate(
        val_data, criterion,
        moe_load_balance_weight=args.moe_load_balance_weight,
        moe_entropy_weight=args.moe_entropy_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
    )
    
    # 添加额外信息
    metrics['ckpt_version'] = 'standard'
    metrics['ckpt_missing_count'] = len(compat_info['missing'])
    metrics['ckpt_unexpected_count'] = len(compat_info['unexpected'])
    
    # 保存指标
    save_json(save_dir / 'metrics.json', metrics)
    
    # 保存每样本指标
    sample_rows = [
        {
            'sample_index': i,
            'subj_id': str(artifacts['subj_ids'][i]),
            'pathology_score': float(artifacts['pathology_scores'][i]),
            'sample_pcc': float(artifacts['sample_pcc'][i]),
        }
        for i in range(len(artifacts['sample_pcc']))
    ]
    save_json(save_dir / 'sample_metrics.json', sample_rows)
    
    # ═══════════════════════════════════════════════════════════════════════
    #  2. 专家-HAMD分布分析
    # ═══════════════════════════════════════════════════════════════════════
    
    # 加载HAMD数据
    hamd_data = load_hamd_data(
        args.hamd_normalized_file,
        args.hamd_original_file
    )
    
    expert_hamd_analysis = {}
    if hamd_data and artifacts['gate_weights'] is not None:
        expert_hamd_analysis = analyzer.analyze_expert_hamd_distribution(
            subj_ids=artifacts['subj_ids'],
            gate_weights=artifacts['gate_weights'],
            hamd_data=hamd_data,
            assignment_method=args.expert_assignment_method
        )
        
        # 保存分析结果
        if expert_hamd_analysis:
            save_json(save_dir / 'expert_hamd_analysis.json', expert_hamd_analysis)
    else:
        log.warning("Skipping expert-HAMD analysis: no HAMD data or gate weights available")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  3. 可视化生成
    # ═══════════════════════════════════════════════════════════════════════
    if args.generate_visualizations:
        
        # 基础可视化
        visualizer.plot_prediction_scatter(artifacts['pred'], artifacts['target'])
        visualizer.plot_error_distribution(artifacts['pred'], artifacts['target'])
        visualizer.plot_sample_pcc_distribution(artifacts['sample_pcc'])
        visualizer.plot_roi_heatmap(artifacts['roi_metrics'])
        visualizer.plot_temporal_metrics(artifacts['temporal_metrics'])
        visualizer.plot_moe_gate_analysis(artifacts['gate_weights'], metrics)
        
        # 专家-HAMD分布可视化
        if expert_hamd_analysis:
            visualizer.plot_expert_hamd_distribution(expert_hamd_analysis)
        
        # 时间序列预测示例
        for i in range(min(3, artifacts['pred'].shape[0])):
            for r in range(0, artifacts['pred'].shape[1], 20):
                visualizer.plot_temporal_prediction(artifacts['pred'], artifacts['target'], i, r)
        
        # 病理分数相关性
        if not np.all(np.isnan(artifacts['pathology_scores'])):
            visualizer.plot_pathology_correlation(
                artifacts['pathology_scores'], 
                artifacts['sample_pcc']
            )
        
        # 综合仪表板
        visualizer.plot_comprehensive_dashboard(metrics, artifacts)
        
        log.info(f'Generated {visualizer.fig_count} visualizations')
    
    # ═══════════════════════════════════════════════════════════════════════
    #  4. 特征重要性分析
    # ═══════════════════════════════════════════════════════════════════════
    if args.compute_feature_importance:
        importance = analyzer.compute_feature_importance(
            val_data, 
            n_samples=args.feature_importance_samples,
            n_permutations=args.feature_importance_permutations,
            method=args.feature_importance_method
        )
        
        # 保存完整结果（包括新字段）
        save_result = {
            'roi_importance': importance['roi_importance'].tolist(),
            'roi_pvalues': importance.get('roi_pvalues', []).tolist(),
            'roi_std_errors': importance.get('roi_std_errors', []).tolist(),
            'roi_importance_rank': importance['roi_importance_rank'].tolist(),
            'baseline_mae': importance['baseline_mae'],
            'method': importance.get('method', args.feature_importance_method),
        }
        save_json(save_dir / 'feature_importance.json', save_result)
        
        if args.generate_visualizations:
            # 原始特征重要性可视化
            visualizer.plot_feature_importance(importance)
            
            # FDR 校正后的显著脑区玻璃脑图可视化
            log.info('Generating FDR-corrected significant brain region visualizations...')
            visualizer.plot_significant_brain_regions(importance, fdr_alpha=0.001)
            
            export_significance_to_xlsx(
                importance,
                output_path=save_dir / 'brain_regions' / 'region_significance.xlsx',
                fdr_alpha=0.05,
            )
    
    
    # ═══════════════════════════════════════════════════════════════════════
    #  5. 保存预测数组
    # ═══════════════════════════════════════════════════════════════════════
    if args.save_arrays:
        log.info('Saving prediction arrays...')
        
        np.savez_compressed(
            save_dir / 'predictions.npz',
            pred=artifacts['pred'],
            target=artifacts['target'],
            sample_pcc=artifacts['sample_pcc'],
            gate_weights=artifacts['gate_weights'] if artifacts['gate_weights'] is not None else [],
            subj_ids=artifacts['subj_ids'],
            pathology_scores=artifacts['pathology_scores'],
            roi_mae=artifacts['roi_metrics']['roi_mae'],
            roi_pcc=artifacts['roi_metrics']['roi_pcc'],
        )


if __name__ == '__main__':
    main()
