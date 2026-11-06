"""共享 AAL116 标签及既有脑网络分组；预测指标由正式评价入口计算。"""
from typing import List

# 脑网络分组定义（基于 AAL116 atlas）
NETWORK_GROUPS = {
    "DMN": [5, 24, 26, 27, 68, 86],   # Default Mode Network
    "FPN": [8, 61],                     # Frontoparietal Network
    "VAN": [12, 14, 29],                # Ventral Attention Network
    "LN": [10, 83],                     # Limbic Network
    "VN": [50],                         # Visual Network
}

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
