# coding=utf-8
"""论文图共享样式与数据装载。

样式与配色取样自 plot-is-all-you-need 图库卡片（/tmp 克隆件 gallery/）：
- HC/MDD 主色取 raincloud-median-badges 卡片（深蓝 #2997c9 / 亮粉 #e61e71，
  浅色 #b1e5f2 / #f5999d）；
- 负控与参照灰取 alluvial-stages-outlined 卡片（灰紫 #8e8cc0）；
- 发散色带端点取 pred-vs-obs-model-grid 卡片（藏青 #222d5a / 陶红 #cd7f6d）；
- 版式规范取其样例脚本：Nimbus Roman（Times 替代）、pdf.fonttype=42、
  300 dpi、无顶右刺、粗体标题。

所有图脚本通过 ``savefig(fig, name)`` 落盘三件套（PNG/PDF/PY）到
``results/paper_figures/``；PY 为脚本自身副本，可独立重跑。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ── 路径（脚本可能被复制到 results/paper_figures/ 独立运行，向上找仓库根）──
_here = Path(__file__).resolve()
_CAND = [_here] + list(_here.parents)
ROOT = next(p for p in _CAND if (p / 'AGENTS.md').exists())
FIG_DIR = ROOT / 'results' / 'paper_figures'
STATS_DIR = ROOT / 'results' / 'paper_stats' / 'run_20261026'
RUN1 = ROOT / 'results' / 'pathology_analysis' / 'run1'

# ── 色板（图库取样，见模块 docstring）──
HC = '#2997c9'
HC_PALE = '#b1e5f2'
MDD = '#e61e71'
MDD_PALE = '#f5999d'
NEGC = '#8e8cc0'          # 负控/参照灰紫
INK = '#252533'           # 主墨色（alluvial 卡片）
ACCENT_RED = '#b0461e'    # 提醒红（样例脚本 Simulated 标注同源）
GOLD = '#f5c287'          # 浅橙（alluvial 卡片）
DIV_CMAP = LinearSegmentedColormap.from_list(
    'pian_div', ['#222d5a', '#98c1dc', '#f4f4f4', '#e0a294', '#8c3a2b'])

# 网络配色（ muted NPG 系，色盲安全顺序；与 run1 network_summary 一致的分组）
NETWORK_ORDER = ['VN', 'SMN', 'DAN', 'VAN', 'DMN', 'FPN', 'LN', 'Other']
NETWORK_COLORS = {
    'VN': '#3C5488', 'SMN': '#4DBBD5', 'DAN': '#00A087', 'VAN': '#91D1C5',
    'DMN': '#E64B35', 'FPN': '#F39B7F', 'LN': '#8491B4', 'Other': '#B8B8B8'}
NETWORK_LABELS = {
    'VN': 'Visual', 'SMN': 'Sensorimotor', 'DAN': 'Dorsal attention',
    'VAN': 'Ventral attention', 'DMN': 'Default mode', 'FPN': 'Frontoparietal',
    'LN': 'Limbic', 'Other': 'Other/uncategorized'}

_split_style = {
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.frameon': False,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def apply_style():
    plt.rcParams.update(_split_style)


def savefig(fig, name: str):
    """落盘三件套：PNG（300 dpi）+ PDF + 脚本自身副本。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{name}.png'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    src = Path(sys.modules['__main__'].__file__)
    if src.stem == name or src.stem.startswith('<'):
        shutil.copy(src, FIG_DIR / f'{name}.py')
    plt.close(fig)
    print(f'saved {png}')


# ── 数据装载 ─────────────────────────────────────────────────────────────
def roi_labels() -> List[str]:
    sys.path.insert(0, str(ROOT))
    from analysis.metrics import get_aal116_labels
    return get_aal116_labels()


def roi_networks() -> np.ndarray:
    """116 ROI 的网络名数组（与 run1 的 network_columns 口径一致）。"""
    sys.path.insert(0, str(ROOT))
    from analysis.pathology_tests import network_columns
    net = np.array(['Other'] * 116, dtype=object)
    for name, cols in network_columns(116).items():
        net[np.asarray(cols, dtype=int)] = name
    return net


def gate_report() -> dict:
    return json.loads((ROOT / 'results' / 'normative_reference' /
                       'gate_report.json').read_text())


def group_tests() -> dict:
    return json.loads((RUN1 / 'group_tests.json').read_text())


def roi_fdr(model: bool = True) -> pd.DataFrame:
    f = RUN1 / ('roi_fdr_table.csv' if model else 'roi_fdr_table_descriptor.csv')
    return pd.read_csv(f)


def condition_usage() -> pd.DataFrame:
    return pd.read_csv(ROOT / 'results' / 'pathology_analysis' /
                       'condition_usage.csv')


def z_matrix(group: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (subjects, Z[subject×ROI])。"""
    d = np.load(RUN1 / 'z_maps' / f'z_matrix_{group}_{split}.npz', allow_pickle=True)
    return d['subjects'], d['z']


def matched_pairs() -> List[dict]:
    sub = json.loads((ROOT / 'data' / 'splits' / 'site_matched_subset.json')
                     .read_text())
    return sub['pairs']


def final_metrics() -> Tuple[dict, dict]:
    f = ROOT / 'results/paper_figures/complete_20261026/sources/h32'
    return (json.loads((f / 'metrics_val.json').read_text()),
            json.loads((f / 'metrics_test.json').read_text()))


def per_subject_segment(split: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / 'results/paper_figures/complete_20261026/sources/h32' / f'per_subject_{split}.csv')


def network_summary() -> pd.DataFrame:
    return pd.read_csv(RUN1 / 'z_maps' / 'network_summary.csv')


def half_violin(ax, data, pos, color, pale=None, side='right', width=0.9,
                inner='box'):
    """半小提琴（借鉴 raincloud 卡片：点云+半小提琴+中位圆标）。"""
    from scipy.stats import gaussian_kde
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    k = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), 80)
    dens = k(grid)
    dens = dens / dens.max() * width if dens.max() > 0 else dens
    fill = pale or color
    sign = 1 if side == 'right' else -1
    ax.fill_betweenx(grid, pos, pos + sign * dens, color=fill, alpha=0.85, lw=0)
    for q in (25, 50, 75):
        v = np.percentile(x, q)
        d = k(np.array([v]))[0] / (k(grid).max() + 1e-12) * width
        lw = 2.0 if q == 50 else 0.9
        col = INK if q == 50 else 'white'
        ax.plot([pos + 0.04, pos + sign * d], [v, v], color=col, lw=lw, zorder=5)
