# coding=utf-8
"""评估指标计算与 AAL116 脑区标签/显著性导出工具（analysis 包叶子模块）。"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_device(device_text: str) -> torch.device:
    text = str(device_text).strip().lower()
    if text != 'auto':
        return torch.device(device_text)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 数据类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_hamd_data(normalized_path: str, original_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载HAMD数据，返回样本ID到HAMD分数的映射
    
    Args:
        normalized_path: 归一化HAMD文件路径（用于匹配样本）
        original_path: 原始HAMD文件路径（用于可视化）
    
    Returns:
        字典: {样本ID: {'normalized': 归一化分数, 'original': 原始分数}}
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available, cannot load HAMD data")
        return {}
    
    try:
        # 读取归一化文件
        df_norm = pd.read_excel(normalized_path)
        # 读取原始文件
        df_orig = pd.read_excel(original_path)
        
        # 创建映射字典
        hamd_data = {}
        
        # 归一化数据
        norm_map = {}
        for _, row in df_norm.iterrows():
            sample_id = str(row['ID']).strip()
            norm_map[sample_id] = float(row['HAMD'])
        
        # 原始数据
        orig_map = {}
        for _, row in df_orig.iterrows():
            sample_id = str(row['ID']).strip()
            orig_map[sample_id] = float(row['HAMD'])
        
        # 合并数据（以归一化文件的ID为准）
        for sample_id, norm_score in norm_map.items():
            hamd_data[sample_id] = {
                'normalized': norm_score,
                'original': orig_map.get(sample_id, float('nan'))
            }
        
        log.info(f"Loaded HAMD data for {len(hamd_data)} samples")
        return hamd_data
    
    except Exception as e:
        log.warning(f"Failed to load HAMD data: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  §1 评估指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def pearson_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算每个样本的皮尔逊相关系数"""
    b = pred.shape[0]
    pf = pred.reshape(b, -1)
    tf = target.reshape(b, -1)
    pc = pf - pf.mean(1, keepdim=True)
    tc = tf - tf.mean(1, keepdim=True)
    return (pc * tc).sum(1) / ((pc**2).sum(1).sqrt() * (tc**2).sum(1).sqrt() + eps)


def pearson_global(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """计算全局皮尔逊相关系数"""
    x = pred.reshape(-1).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float(np.sum(x * y) / (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)) + eps))


def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """计算基础回归指标"""
    error = pred - target
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    pcc = pearson_global(pred, target)
    y = target.reshape(-1).astype(np.float64)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((pred.reshape(-1).astype(np.float64) - y) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'PCC': pcc, 'R2': float(r2)}


def compute_advanced_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """计算高级评估指标"""
    error = pred - target
    metrics = {}
    
    # 基础指标
    metrics['MAE'] = float(np.mean(np.abs(error)))
    metrics['MSE'] = float(np.mean(error ** 2))
    metrics['RMSE'] = float(np.sqrt(metrics['MSE']))
    metrics['PCC'] = pearson_global(pred, target)
    
    # R² 和调整后 R²
    y = target.reshape(-1).astype(np.float64)
    n = len(y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((pred.reshape(-1).astype(np.float64) - y) ** 2))
    metrics['R2'] = 1.0 - ss_res / (ss_tot + 1e-12)
    
    # 百分位误差
    metrics['MAE_50th'] = float(np.percentile(np.abs(error), 50))
    metrics['MAE_90th'] = float(np.percentile(np.abs(error), 90))
    metrics['MAE_95th'] = float(np.percentile(np.abs(error), 95))
    metrics['MAE_99th'] = float(np.percentile(np.abs(error), 99))
    
    # 误差偏度和峰度
    error_flat = error.reshape(-1)
    metrics['Error_Skewness'] = float(np.mean((error_flat - error_flat.mean())**3) / (error_flat.std()**3 + 1e-12))
    metrics['Error_Kurtosis'] = float(np.mean((error_flat - error_flat.mean())**4) / (error_flat.std()**4 + 1e-12))
    
    # 最大误差和最小误差
    abs_error = np.abs(error)
    metrics['Max_Error'] = float(np.max(abs_error))
    metrics['Min_Error'] = float(np.min(abs_error))
    
    # 对称平均绝对百分比误差 (SMAPE)
    denom = (np.abs(pred) + np.abs(target)) / 2 + 1e-12
    metrics['SMAPE'] = float(np.mean(np.abs(error) / denom))
    
    # 平均绝对比例误差 (MASE) - 使用朴素预测作为基准
    naive_error = np.abs(target[:, 1:] - target[:, :-1]).mean() if target.shape[1] > 1 else metrics['MAE']
    metrics['MASE'] = metrics['MAE'] / (naive_error + 1e-12)
    
    return metrics


def compute_roi_level_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    """计算 ROI 级别的指标 [B, F, W, S]"""
    # pred/target shape: [B, F, W, S]
    error = pred - target
    abs_error = np.abs(error)
    
    # ROI 级别 MAE [F]
    roi_mae = abs_error.mean(axis=(0, 2, 3))  # 平均 batch, window, seq
    
    # ROI 级别 MSE [F]
    roi_mse = (error ** 2).mean(axis=(0, 2, 3))
    
    # ROI 级别 PCC [F]
    roi_pcc = []
    for f in range(pred.shape[1]):
        p = pred[:, f, :, :].reshape(-1)
        t = target[:, f, :, :].reshape(-1)
        roi_pcc.append(pearson_global(p, t))
    roi_pcc = np.array(roi_pcc)
    
    return {
        'roi_mae': roi_mae,
        'roi_mse': roi_mse,
        'roi_pcc': roi_pcc,
        'roi_variance': target.var(axis=(0, 2, 3)),
    }


def compute_temporal_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    """计算时间维度上的指标 [B, F, W, S]"""
    error = pred - target
    abs_error = np.abs(error)
    
    # 窗口级别 MAE [W]
    window_mae = abs_error.mean(axis=(0, 1, 3))
    
    # 序列级别 MAE [S]
    seq_mae = abs_error.mean(axis=(0, 1, 2))
    
    # 时间自相关
    pred_temporal_var = pred.var(axis=3).mean(axis=(0, 1))  # [W]
    target_temporal_var = target.var(axis=3).mean(axis=(0, 1))
    
    return {
        'window_mae': window_mae,
        'seq_mae': seq_mae,
        'pred_temporal_var': pred_temporal_var,
        'target_temporal_var': target_temporal_var,
        'temporal_var_ratio': pred_temporal_var / (target_temporal_var + 1e-12),
    }


def fdr_correction(pvalues: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR 校正
    
    Args:
        pvalues: 原始 p 值数组
        alpha: 显著性水平
    
    Returns:
        rejected: 是否拒绝原假设的布尔数组
        corrected_pvalues: 校正后的 p 值
    """
    n = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    
    # 计算 BH 临界值
    critical_values = alpha * np.arange(1, n + 1) / n
    
    # 从最大的 p 值开始比较
    corrected_pvalues = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    
    # 计算校正后的 p 值
    cumulative_min = 1.0
    for i in range(n - 1, -1, -1):
        corrected_pvalue = sorted_pvalues[i] * n / (i + 1)
        cumulative_min = min(corrected_pvalue, cumulative_min)
        corrected_pvalues[sorted_indices[i]] = cumulative_min
    
    # 确定哪些被拒绝
    for i in range(n):
        if sorted_pvalues[i] <= critical_values[i]:
            rejected[sorted_indices[i]] = True
        else:
            break  # 一旦不满足，更大的 p 值也不满足
    
    return rejected, corrected_pvalues


# 脑网络分组定义（基于 AAL116 atlas）
NETWORK_GROUPS = {
    "DMN": [5, 24, 26, 27, 68, 86],   # Default Mode Network
    "FPN": [8, 61],                     # Frontoparietal Network
    "VAN": [12, 14, 29],                # Ventral Attention Network
    "LN": [10, 83],                     # Limbic Network
    "VN": [50],                         # Visual Network
}

# Nature 风格配色方案（色盲友好）
NATURE_COLORS = [
    "#3B6EA8", "#D05A50", "#4B9B82", "#A2678A",
    "#E3A246", "#5E9FBA", "#8C7A4F", "#7A7A7A",
    "#BC6C25", "#6D87C3", "#75A87B", "#C56E90",
    "#9A8F4F", "#4F8C8B", "#B06C49", "#6F6F9E",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  §1.1 AAL116 脑区标签 & xlsx 导出
# ═══════════════════════════════════════════════════════════════════════════════

_aal116_labels_cache = None


def get_aal116_labels() -> List[str]:
    """获取 AAL116 脑区名称列表（索引0对应AAL1）"""
    global _aal116_labels_cache
    if _aal116_labels_cache is not None:
        return _aal116_labels_cache

    labels = []
    try:
        from nilearn import datasets
        atlas = datasets.fetch_atlas_aal(version="SPM12")
        for label_name, label_value in zip(atlas.labels, atlas.indices):
            value = int(label_value)
            if value == 0 or "background" in label_name.lower():
                continue
            labels.append(label_name)
    except Exception:
        labels = [f"AAL{i}" for i in range(1, 117)]

    _aal116_labels_cache = labels
    return _aal116_labels_cache


def _get_network_for_region(aal_index: int) -> str:
    """获取脑区所属网络名称"""
    for net_name, region_list in NETWORK_GROUPS.items():
        if aal_index in region_list:
            return net_name
    return "Other"


def export_significance_to_xlsx(importance: Dict[str, np.ndarray],
                                output_path: Path,
                                fdr_alpha: float = 0.05) -> None:
    """
    将所有 ROI 的显著性分析结果导出为 xlsx 文件

    Args:
        importance: 特征重要性字典
        output_path: 输出 xlsx 路径
        fdr_alpha: FDR 校正的显著性水平
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        log.warning("openpyxl not available, falling back to pandas for xlsx export")
        _export_significance_via_pandas(importance, output_path, fdr_alpha)
        return

    roi_importance = np.asarray(importance['roi_importance'])
    roi_pvalues = np.asarray(importance.get('roi_pvalues', np.ones_like(roi_importance)))

    # FDR 校正
    _, corrected_pvalues = fdr_correction(roi_pvalues, alpha=fdr_alpha)
    ranks = np.argsort(np.argsort(roi_importance)[::-1]) + 1

    # 获取脑区名称
    aal_labels = get_aal116_labels()
    n_rois = len(roi_importance)

    # 创建工作簿
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "脑区显著性分析"

    # ── 样式定义 ──
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    sig_fill_001 = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # 绿
    sig_fill_01 = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")    # 黄
    sig_fill_05 = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")    # 粉

    thin_border = Border(
        left=Side(style="thin", color="D0D0D0"),
        right=Side(style="thin", color="D0D0D0"),
        top=Side(style="thin", color="D0D0D0"),
        bottom=Side(style="thin", color="D0D0D0"),
    )

    # ── 表头 ──
    headers = [
        "AAL_ID", "脑区名称", "所属网络", "重要性得分",
        "排名", "原始p值", "FDR校正p值",
        "显著(P<0.05)", "显著(P<0.01)", "显著(P<0.001)"
    ]

    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    ws.row_dimensions[1].height = 28

    # ── 冻结首行 ──
    ws.freeze_panes = "A2"

    # ── 写入数据 ──
    for i in range(n_rois):
        row = i + 2
        aal_id = i + 1
        name = aal_labels[i] if i < len(aal_labels) else f"AAL{aal_id}"
        network = _get_network_for_region(aal_id)
        importance_val = float(roi_importance[i])
        rank = int(ranks[i])
        raw_p = float(roi_pvalues[i])
        corrected_p = float(corrected_pvalues[i])

        sig_05 = corrected_p < 0.05
        sig_01 = corrected_p < 0.01
        sig_001 = corrected_p < 0.001

        row_data = [
            aal_id, name, network, round(importance_val, 6),
            rank, f"{raw_p:.4e}", f"{corrected_p:.4e}",
            "Yes" if sig_05 else "No",
            "Yes" if sig_01 else "No",
            "Yes" if sig_001 else "No",
        ]

        # 行底色
        if sig_001:
            row_fill = sig_fill_001
        elif sig_01:
            row_fill = sig_fill_01
        elif sig_05:
            row_fill = sig_fill_05
        else:
            row_fill = None

        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row, column=col_idx, value=value)
            cell.font = Font(name="Arial", size=9)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border
            if row_fill:
                cell.fill = row_fill

    # ── 列宽 ──
    col_widths = [8, 28, 10, 14, 8, 14, 14, 14, 14, 14]
    for col_idx, width in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    # ── 自动筛选 ──
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{n_rois + 1}"

    # ── 摘要 sheet ──
    ws2 = wb.create_sheet("摘要")
    summary_data = [
        ["项目", "值"],
        ["总脑区数", n_rois],
        ["FDR显著性水平", fdr_alpha],
        ["显著脑区数 (P<0.05)", int(np.sum(corrected_pvalues < 0.05))],
        ["显著脑区数 (P<0.01)", int(np.sum(corrected_pvalues < 0.01))],
        ["显著脑区数 (P<0.001)", int(np.sum(corrected_pvalues < 0.001))],
        ["计算方法", importance.get("method", "N/A")],
        ["基准MAE", float(importance.get("baseline_mae", 0))],
    ]

    for row_idx, (label, value) in enumerate(summary_data, 1):
        c1 = ws2.cell(row=row_idx, column=1, value=label)
        c2 = ws2.cell(row=row_idx, column=2, value=value)
        c1.font = Font(name="Arial", size=10, bold=True)
        c2.font = Font(name="Arial", size=10)
        c1.alignment = Alignment(horizontal="right")
        c2.alignment = Alignment(horizontal="left")

    ws2.column_dimensions["A"].width = 24
    ws2.column_dimensions["B"].width = 18

    # ── 保存 ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
    log.info(f"Significance xlsx saved: {output_path}")


def _export_significance_via_pandas(importance: Dict[str, np.ndarray],
                                    output_path: Path,
                                    fdr_alpha: float) -> None:
    """pandas 回退方案"""
    import pandas as pd

    roi_importance = np.asarray(importance['roi_importance'])
    roi_pvalues = np.asarray(importance.get('roi_pvalues', np.ones_like(roi_importance)))
    _, corrected_pvalues = fdr_correction(roi_pvalues, alpha=fdr_alpha)
    ranks = np.argsort(np.argsort(roi_importance)[::-1]) + 1

    aal_labels = get_aal116_labels()
    n_rois = len(roi_importance)

    records = []
    for i in range(n_rois):
        aal_id = i + 1
        name = aal_labels[i] if i < len(aal_labels) else f"AAL{aal_id}"
        network = _get_network_for_region(aal_id)
        records.append({
            "AAL_ID": aal_id,
            "脑区名称": name,
            "所属网络": network,
            "重要性得分": round(float(roi_importance[i]), 6),
            "排名": int(ranks[i]),
            "原始p值": float(roi_pvalues[i]),
            "FDR校正p值": float(corrected_pvalues[i]),
            "显著(P<0.05)": "Yes" if corrected_pvalues[i] < 0.05 else "No",
            "显著(P<0.01)": "Yes" if corrected_pvalues[i] < 0.01 else "No",
            "显著(P<0.001)": "Yes" if corrected_pvalues[i] < 0.001 else "No",
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, sheet_name="脑区显著性分析", index=False, engine="openpyxl")
    log.info(f"Significance xlsx saved (pandas): {output_path}")

