# coding=utf-8
"""ModelAnalyzer：模型评估、特征重要性与专家-HAMD 分层分析核心。"""
import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm

from analysis.metrics import (compute_advanced_metrics,
                              compute_roi_level_metrics,
                              compute_temporal_metrics,
                              pearson_per_sample)
from models.neurotwin import NeuroTwin
from train.moe import compute_moe_regularization

log = logging.getLogger(__name__)
class ModelAnalyzer:
    """模型分析器类"""
    
    def __init__(self, model: NeuroTwin, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def evaluate(self, data_loader, criterion, 
                 moe_load_balance_weight: float = 0.01,
                 moe_entropy_weight: float = 1e-3,
                 moe_z_loss_weight: float = 1e-3) -> Tuple[Dict, Dict]:
        """执行完整评估"""
        meters = defaultdict(float)
        all_preds, all_targets, all_sample_pccs = [], [], []
        all_gate_weights = []
        all_subj_ids = []
        all_pathology_scores = []
        all_batch_data = []  # 存储原始数据用于后续分析
        
        pbar = tqdm(data_loader, total=len(data_loader), desc='[Evaluating]')
        
        for step, batch in enumerate(pbar, start=1):
            x = batch['x'].to(self.device, non_blocking=True)
            y = batch['y'].to(self.device, non_blocking=True)
            sc = batch['sc'].to(self.device, non_blocking=True)
            pathology = batch.get('pathology_score', None)
            if pathology is not None:
                pathology = pathology.to(self.device, non_blocking=True)
            
            # 前向传播
            out, aux = self.model(x, sc, pathology)
            
            # 计算损失
            total_loss, loss_stats = criterion(out, y)
            moe_reg, _ = compute_moe_regularization(
                aux_info=aux, device=self.device,
                load_balance_weight=moe_load_balance_weight,
                entropy_weight=moe_entropy_weight,
                z_loss_weight=moe_z_loss_weight,
            )
            total_loss = total_loss + moe_reg
            mae = torch.mean(torch.abs(out - y))
            
            # 更新 meters
            meters['loss_total'] += float(total_loss.item())
            meters['loss_pcc'] += float(loss_stats['loss_pcc'].item())
            meters['loss_mae'] += float(loss_stats['loss_mae'].item())
            meters['loss_diff'] += float(loss_stats['loss_diff'].item())
            meters['loss_std'] += float(loss_stats['loss_std'].item())
            meters['metric_mae'] += float(mae.item())
            
            # 样本级 PCC
            batch_pcc = pearson_per_sample(out, y).detach().cpu().numpy()
            all_sample_pccs.append(batch_pcc)
            
            # 存储预测和目标
            pred_np = out.detach().cpu().numpy()
            target_np = y.detach().cpu().numpy()
            all_preds.append(pred_np)
            all_targets.append(target_np)
            
            # 存储 batch 数据
            all_batch_data.append({
                'x': x.detach().cpu().numpy(),
                'y': y.detach().cpu().numpy(),
                'sc': sc.detach().cpu().numpy(),
                'pred': pred_np,
                'target': target_np,
                'sample_pcc': batch_pcc,
            })
            
            # 门控权重
            if isinstance(aux, dict) and 'gates' in aux and torch.is_tensor(aux['gates']):
                all_gate_weights.append(aux['gates'].detach().cpu().numpy())
            
            # 元数据
            batch_subj = batch['subj_id']
            if isinstance(batch_subj, (list, tuple)):
                all_subj_ids.extend([str(s) for s in batch_subj])
            else:
                all_subj_ids.extend([str(batch_subj)])
            
            if pathology is not None:
                all_pathology_scores.extend(pathology.detach().cpu().reshape(-1).tolist())
            else:
                all_pathology_scores.extend([float('nan')] * len(batch_pcc))
            
            pbar.set_postfix({
                'MAE': f"{meters['metric_mae']/step:.4f}",
                'PCC': f"{np.mean(np.concatenate(all_sample_pccs)):.4f}",
            })
        
        # 合并所有数据
        n_step = max(1, len(data_loader))
        pred_np = np.concatenate(all_preds, axis=0)
        target_np = np.concatenate(all_targets, axis=0)
        sample_pcc_np = np.concatenate(all_sample_pccs, axis=0)
        gate_np = np.concatenate(all_gate_weights, axis=0) if all_gate_weights else None
        
        # 计算指标
        metrics = {k: v / n_step for k, v in meters.items()}
        metrics.update(compute_advanced_metrics(pred_np, target_np))
        metrics['sample_pcc_mean'] = float(np.mean(sample_pcc_np))
        metrics['sample_pcc_std'] = float(np.std(sample_pcc_np))
        metrics['sample_pcc_median'] = float(np.median(sample_pcc_np))
        metrics['num_samples'] = int(pred_np.shape[0])
        
        # ROI 级别指标
        roi_metrics = compute_roi_level_metrics(pred_np, target_np)
        metrics['roi_mae_mean'] = float(roi_metrics['roi_mae'].mean())
        metrics['roi_mae_std'] = float(roi_metrics['roi_mae'].std())
        metrics['roi_pcc_mean'] = float(roi_metrics['roi_pcc'].mean())
        
        # 时间维度指标
        temporal_metrics = compute_temporal_metrics(pred_np, target_np)
        
        # MoE 门控统计
        gate_stats = {}
        if gate_np is not None:
            gate_mean = gate_np.mean(axis=0)
            gate_std = gate_np.std(axis=0)
            for i, (gm, gs) in enumerate(zip(gate_mean.tolist(), gate_std.tolist())):
                metrics[f'gate_mean_E{i}'] = float(gm)
                metrics[f'gate_std_E{i}'] = float(gs)
                gate_stats[f'E{i}'] = {'mean': float(gm), 'std': float(gs)}
        
        artifacts = {
            'pred': pred_np,
            'target': target_np,
            'sample_pcc': sample_pcc_np,
            'gate_weights': gate_np,
            'subj_ids': np.array(all_subj_ids, dtype=object),
            'pathology_scores': np.array(all_pathology_scores, dtype=np.float32),
            'roi_metrics': roi_metrics,
            'temporal_metrics': temporal_metrics,
            'batch_data': all_batch_data,
        }
        
        return metrics, artifacts
    
    @torch.no_grad()
    def compute_feature_importance(self, data_loader, n_samples: int = 100, 
                                   n_permutations: int = 200,
                                   method: str = 'balanced') -> Dict[str, np.ndarray]:
        """
        计算特征重要性（优化版本）
        
        Args:
            data_loader: 数据加载器
            n_samples: 使用的样本数
            n_permutations: 置换次数（method='permutation'时生效）
            method: 计算方法
                - 'fast': 使用 t-test 近似（最快，~1秒）
                - 'balanced': 200次置换 + t-test 验证（推荐，~5分钟）
                - 'precise': 1000次置换（最准确，~25分钟）
                - 'permutation': 纯置换检验，使用指定的 n_permutations
        """
        self.model.eval()
        
        # 收集一些样本
        samples = []
        for batch in data_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            sc = batch['sc'].to(self.device)
            pathology = batch.get('pathology_score', None)
            if pathology is not None:
                pathology = pathology.to(self.device)
            
            samples.append((x, y, sc, pathology))
            if len(samples) * x.shape[0] >= n_samples:
                break
        
        # 计算基准性能
        baseline_maes = []
        baseline_outputs = []  # 保存输出用于后续分析
        
        for x, y, sc, pathology in samples:
            out, _ = self.model(x, sc, pathology)
            mae = torch.abs(out - y).mean().item()
            baseline_maes.append(mae)
            baseline_outputs.append(out.detach())
        
        baseline_mae = np.mean(baseline_maes)
        baseline_outputs_cat = torch.cat(baseline_outputs, dim=0)
        
        # ROI 级别重要性 [F]
        n_rois = samples[0][0].shape[1]
        roi_importance = np.zeros(n_rois)
        roi_pvalues = np.ones(n_rois)
        roi_std_errors = np.zeros(n_rois)
        
        if method == 'fast':
            log.info("Using FAST mode: t-test approximation (no permutation)")
            
            # 快速模式：只进行少量观测 + t-test
            for roi_idx in tqdm(range(n_rois), desc='Computing ROI importance (fast)'):
                permuted_maes = []
                
                for x, y, sc, pathology in samples:
                    x_perm = x.clone()
                    perm_idx = torch.randperm(x.shape[0])
                    x_perm[:, roi_idx, :, :] = x_perm[perm_idx, roi_idx, :, :]
                    
                    out, _ = self.model(x_perm, sc, pathology)
                    mae = torch.abs(out - y).mean().item()
                    permuted_maes.append(mae)
                
                observed_importance = np.mean(permuted_maes) - baseline_mae
                roi_importance[roi_idx] = observed_importance
                
                # 使用单样本 t-test 计算 p 值
                if len(permuted_maes) > 1:
                    std_error = np.std(permuted_maes, ddof=1) / np.sqrt(len(permuted_maes))
                    t_statistic = observed_importance / (std_error + 1e-12)
                    
                    # 单侧 t-test p 值
                    df = len(permuted_maes) - 1
                    p_value = 1.0 - scipy_stats.t.cdf(t_statistic, df)
                    roi_pvalues[roi_idx] = p_value
                    roi_std_errors[roi_idx] = std_error
                    
        elif method in ['balanced', 'precise', 'permutation']:
            # 根据模式设置置换次数
            actual_n_perm = {
                'balanced': 200,
                'precise': 1000,
                'permutation': n_permutations
            }.get(method, n_permutations)
            
            log.info(f"Using {method.upper()} mode: {actual_n_perm} permutations per ROI")
            
            # 优化：批量处理所有样本的置换
            for roi_idx in tqdm(range(n_rois), desc=f'Computing ROI importance ({method})'):
                # 观测值
                observed_maes = []
                for x, y, sc, pathology in samples:
                    x_perm = x.clone()
                    perm_idx = torch.randperm(x.shape[0])
                    x_perm[:, roi_idx, :, :] = x_perm[perm_idx, roi_idx, :, :]
                    
                    out, _ = self.model(x_perm, sc, pathology)
                    mae = torch.abs(out - y).mean().item()
                    observed_maes.append(mae)
                
                observed_importance = np.mean(observed_maes) - baseline_mae
                roi_importance[roi_idx] = observed_importance
                
                # 置换检验（减少次数）
                null_importances = []
                for _ in range(actual_n_perm):
                    perm_maes = []
                    for x, y, sc, pathology in samples:
                        x_null = x.clone()
                        null_perm_idx = torch.randperm(x.shape[0])
                        x_null[:, roi_idx, :, :] = x_null[null_perm_idx, roi_idx, :, :]
                        
                        out_null, _ = self.model(x_null, sc, pathology)
                        mae_null = torch.abs(out_null - y).mean().item()
                        perm_maes.append(mae_null)
                    
                    null_importance = np.mean(perm_maes) - baseline_mae
                    null_importances.append(null_importance)
                
                # 计算 p 值
                null_importances = np.array(null_importances)
                p_value = np.mean(null_importances >= observed_importance)
                roi_pvalues[roi_idx] = max(p_value, 1.0 / (actual_n_perm + 1))  # 避免 p=0
                
                # 同时计算标准误用于参考
                roi_std_errors[roi_idx] = np.std(null_importances, ddof=1)
                
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'fast', 'balanced', 'precise', 'permutation'")
        
        return {
            'roi_importance': roi_importance,
            'roi_pvalues': roi_pvalues,
            'roi_std_errors': roi_std_errors,
            'roi_importance_rank': np.argsort(roi_importance)[::-1],
            'baseline_mae': baseline_mae,
            'method': method,
        }
    
    @torch.no_grad()
    def extract_attention_patterns(self, data_loader, n_samples: int = 50) -> Dict[str, Any]:
        """提取注意力模式"""
        self.model.eval()
        
        attention_patterns = {
            'graph_attn_weights': [],
            'window_attn_weights': [],
            'mta_attn_weights': [],
        }
        
        count = 0
        for batch in data_loader:
            if count >= n_samples:
                break
                
            x = batch['x'].to(self.device)
            sc = batch['sc'].to(self.device)
            pathology = batch.get('pathology_score', None)
            if pathology is not None:
                pathology = pathology.to(self.device)
            
            # 通过 hook 提取注意力
            attn_hooks = []
            
            def make_hook(name):
                def hook(module, input, output):
                    if hasattr(module, 'last_attn_weights'):
                        attention_patterns[name].append(module.last_attn_weights.detach().cpu().numpy())
                return hook
            
            # 注册 hooks
            handles = []
            if hasattr(self.model, 'ode_blocks'):
                for i, block in enumerate(self.model.ode_blocks):
                    # 注意：这里需要根据实际模型结构调整
                    pass
            
            _ = self.model(x, sc, pathology)
            
            for h in handles:
                h.remove()
            
            count += x.shape[0]
        
        return attention_patterns
    
    def compute_gradcam(self, x: torch.Tensor, sc: torch.Tensor, pathology: Optional[torch.Tensor],
                       target_roi: int = 0) -> np.ndarray:
        """计算 Grad-CAM 热力图"""
        self.model.eval()
        x.requires_grad = True
        
        # 前向传播
        out, _ = self.model(x, sc, pathology)
        
        # 选择目标 ROI 的预测输出
        target_output = out[:, target_roi, :, :].mean()
        
        # 反向传播
        self.model.zero_grad()
        target_output.backward()
        
        # 获取梯度
        gradients = x.grad.detach().cpu().numpy()
        
        # 计算 Grad-CAM (简化版)
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * x.detach().cpu().numpy(), axis=1)
        
        # 归一化
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-12)
        
        return cam
    
    def analyze_expert_hamd_distribution(
        self,
        subj_ids: np.ndarray,
        gate_weights: np.ndarray,
        hamd_data: Dict[str, Dict[str, float]],
        assignment_method: str = 'top1'
    ) -> Dict[str, Any]:
        """
        分析各专家处理样本的HAMD分数分布
        
        Args:
            subj_ids: 样本ID数组 [N]
            gate_weights: 门控权重数组 [N, num_experts]
            hamd_data: HAMD数据字典 {sample_id: {'normalized': float, 'original': float}}
            assignment_method: 样本分配方法，'top1' 表示分配给权重最高的专家
        
        Returns:
            包含各专家HAMD分布统计的字典
        """
        if gate_weights is None or len(hamd_data) == 0:
            log.warning("No gate weights or HAMD data available for analysis")
            return {}
        
        n_experts = gate_weights.shape[1]
        n_samples = len(subj_ids)
        
        # 为每个样本确定主要处理的专家
        if assignment_method == 'top1':
            primary_experts = np.argmax(gate_weights, axis=1)  # [N]
        elif assignment_method == 'threshold':
            # 分配给权重超过阈值的专家（一个样本可能属于多个专家）
            threshold = 1.0 / n_experts + 0.1  # 略高于均匀分布
            primary_experts = np.argmax(gate_weights, axis=1)  # 默认用top1
        else:
            primary_experts = np.argmax(gate_weights, axis=1)
        
        # 收集各专家处理样本的HAMD分数
        expert_hamd_data = {f'E{i}': {'original': [], 'normalized': []} for i in range(n_experts)}
        expert_sample_counts = {f'E{i}': 0 for i in range(n_experts)}
        
        for i, subj_id in enumerate(subj_ids):
            subj_id_str = str(subj_id).strip()
            expert_idx = primary_experts[i]
            expert_key = f'E{expert_idx}'
            
            # 查找该样本的HAMD数据
            # 尝试直接匹配
            hamd_info = hamd_data.get(subj_id_str)
            
            # 如果没有找到，尝试去除可能的后缀（如窗口索引）
            if hamd_info is None:
                # 尝试匹配基础ID（如 S1-1-0001_0 -> S1-1-0001）
                base_id = subj_id_str.split('_')[0]
                hamd_info = hamd_data.get(base_id)
            
            if hamd_info is not None:
                expert_hamd_data[expert_key]['original'].append(hamd_info['original'])
                expert_hamd_data[expert_key]['normalized'].append(hamd_info['normalized'])
            
            expert_sample_counts[expert_key] += 1
        
        # 计算统计信息
        expert_stats = {}
        for i in range(n_experts):
            expert_key = f'E{i}'
            orig_scores = expert_hamd_data[expert_key]['original']
            norm_scores = expert_hamd_data[expert_key]['normalized']
            
            if len(orig_scores) > 0:
                orig_scores_arr = np.array(orig_scores)
                norm_scores_arr = np.array(norm_scores)
                
                expert_stats[expert_key] = {
                    'sample_count': int(expert_sample_counts[expert_key]),
                    'matched_count': len(orig_scores),
                    'original_hamd': {
                        'mean': float(np.mean(orig_scores_arr)),
                        'std': float(np.std(orig_scores_arr)),
                        'median': float(np.median(orig_scores_arr)),
                        'min': float(np.min(orig_scores_arr)),
                        'max': float(np.max(orig_scores_arr)),
                        'q25': float(np.percentile(orig_scores_arr, 25)),
                        'q75': float(np.percentile(orig_scores_arr, 75)),
                        'values': orig_scores_arr.tolist(),
                    },
                    'normalized_hamd': {
                        'mean': float(np.mean(norm_scores_arr)),
                        'std': float(np.std(norm_scores_arr)),
                        'median': float(np.median(norm_scores_arr)),
                        'min': float(np.min(norm_scores_arr)),
                        'max': float(np.max(norm_scores_arr)),
                        'q25': float(np.percentile(norm_scores_arr, 25)),
                        'q75': float(np.percentile(norm_scores_arr, 75)),
                    }
                }
            else:
                expert_stats[expert_key] = {
                    'sample_count': int(expert_sample_counts[expert_key]),
                    'matched_count': 0,
                    'original_hamd': {},
                    'normalized_hamd': {},
                }
        
        # 计算各专家间的统计检验
        from scipy import stats
        pairwise_tests = {}
        expert_keys = [f'E{i}' for i in range(n_experts)]
        
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                key_i, key_j = expert_keys[i], expert_keys[j]
                values_i = expert_hamd_data[key_i]['original']
                values_j = expert_hamd_data[key_j]['original']
                
                if len(values_i) > 5 and len(values_j) > 5:
                    # Mann-Whitney U 检验（非参数）
                    statistic, pvalue = stats.mannwhitneyu(
                        values_i, values_j, alternative='two-sided'
                    )
                    pairwise_tests[f'{key_i}_vs_{key_j}'] = {
                        'statistic': float(statistic),
                        'pvalue': float(pvalue),
                        'significant': pvalue < 0.05,
                    }
        
        return {
            'expert_stats': expert_stats,
            'expert_hamd_data': expert_hamd_data,
            'pairwise_tests': pairwise_tests,
            'n_experts': n_experts,
            'assignment_method': assignment_method,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  §3 可视化模块
# ═══════════════════════════════════════════════════════════════════════════════

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

