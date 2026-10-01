# -*- coding: utf-8 -*-
"""按脑网络分组渲染 AAL116 显著脑区玻璃脑图（next_timepoint 口径）。

输入为 experiments/evaluate_variant.py --feature_importance 落盘的
``feature_importance_<split>.json``：优先取 ``significant_roi_indices``（ROI 置换重要性
经 BH-FDR 校正后的显著 ROI，1 基 AAL 索引），无显著 ROI 时回落到重要性 Top N，并按
``data/AAL116.xlsx`` 的「对应网络」列把脑区分组，逐网络绘制一张玻璃脑图（每个脑区
独立配色，Nature 风格）。

用法:
    python analysis/visualize_sig_region.py \
        --input_json results/eval/<run>/feature_importance_test.json
"""
import argparse
import json
import os
from collections import OrderedDict

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

from nilearn import datasets, plotting, image

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
})

# 项目根目录（脚本位于 <root>/analysis/，路径均基于项目内）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AAL_FILE = os.path.join(PROJECT_ROOT, "data", "AAL116.xlsx")

# Muted, color-blind-conscious palette suitable for Nature-style figures.
# Colors are assigned by AAL index, so the same brain region keeps the same color.
NATURE_COLORS = [
    "#3B6EA8", "#D05A50", "#4B9B82", "#A2678A",
    "#E3A246", "#5E9FBA", "#8C7A4F", "#7A7A7A",
    "#BC6C25", "#6D87C3", "#75A87B", "#C56E90",
    "#9A8F4F", "#4F8C8B", "#B06C49", "#6F6F9E",
]

SAVE_FIG = True
MASK_THRESHOLD = 0.5
REGION_ALPHA = 0.88
DISPLAY_MODE = "lyrz"


def parse_args():
    p = argparse.ArgumentParser(description="显著脑区玻璃脑图（按脑网络分组）")
    p.add_argument("--input_json", required=True,
                   help="evaluate_variant.py --feature_importance 输出的 "
                        "feature_importance_<split>.json")
    p.add_argument("--aal_file", default=DEFAULT_AAL_FILE,
                   help="AAL116 脑区图谱 xlsx（提供脑区名称与「对应网络」列）")
    p.add_argument("--output_dir", default=None,
                   help="图片输出目录，默认 <input_json 所在目录>/brain_regions")
    p.add_argument("--top_n", type=int, default=12,
                   help="无 FDR 显著 ROI 时回落到重要性 Top N（默认 12）")
    p.add_argument("--max_regions_per_figure", type=int, default=8,
                   help="单张网络图的脑区数上限（超出按重要性截断并打印提示）")
    p.add_argument("--fdr_alpha", type=float, default=None,
                   help="显著判定阈值；默认取 json 中的 fdr_alpha")
    return p.parse_args()


def load_significant_regions(args):
    """读取重要性 JSON → (显著 1 基 ROI 索引列表, 判据说明字符串)。"""
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    alpha = (float(args.fdr_alpha) if args.fdr_alpha is not None
             else float(data.get("fdr_alpha", 0.05)))
    qvals = data.get("roi_pvalues_fdr")
    sig = data.get("significant_roi_indices")

    if sig:
        return sorted(int(r) for r in sig), f"FDR q<{alpha:g} 显著"

    importance = np.asarray(data["roi_importance"], dtype=float)
    if qvals is not None:
        # 产物中 q 值存在但与索引不一致时以 q 值为准，避免静默使用过期字段
        q = np.asarray(qvals, dtype=float)
        idx = np.flatnonzero(q < alpha) + 1
        if idx.size:
            return sorted(int(r) for r in idx), f"FDR q<{alpha:g} 显著（按 q 值重算）"
    if importance.size == 0:
        raise ValueError(f"{args.input_json} 中 roi_importance 为空，无法确定显著脑区。")
    top = np.argsort(importance)[::-1][: max(1, int(args.top_n))]
    return sorted(int(r) + 1 for r in top), f"无 FDR 显著 ROI，回落重要性 Top {len(top)}"


def load_network_labels(aal_file, n_rois):
    """读取 AAL116 的脑区名与网络标签（长度不足或列缺失时显式报错）。"""
    import pandas as pd
    df = pd.read_excel(aal_file)
    if len(df) != int(n_rois):
        raise ValueError(
            f"AAL 图谱行数({len(df)})与重要性长度({n_rois})不一致，无法按索引对齐。")
    name_col = "micro命名" if "micro命名" in df.columns else None
    names = ([str(v) for v in df[name_col].tolist()] if name_col
             else [f"AAL{df.index[i] + 1}" for i in range(len(df))])
    if "对应网络" not in df.columns:
        raise ValueError(f"{aal_file} 缺少「对应网络」列，无法按网络分组。")
    networks = [str(v).strip() for v in df["对应网络"].tolist()]
    return names, networks


def group_by_network(region_indexes, importance, networks,
                     max_regions_per_figure):
    """显著 ROI → OrderedDict{网络: [ROI 索引...]}，组内按重要性降序、组间按峰值降序。"""
    groups = {}
    for idx in region_indexes:
        net = networks[idx - 1]
        groups.setdefault(net, []).append(int(idx))
    for net in groups:
        groups[net] = sorted(groups[net], key=lambda i: -float(importance[i - 1]))
        if len(groups[net]) > max_regions_per_figure:
            print(f"  [提示] {net} 有 {len(groups[net])} 个显著脑区，"
                  f"仅绘制重要性最高的 {max_regions_per_figure} 个")
            groups[net] = groups[net][:max_regions_per_figure]
    ordered = sorted(groups.items(),
                     key=lambda kv: -max(float(importance[i - 1]) for i in kv[1]))
    return OrderedDict(ordered)


def _make_cmap(hex_color):
    rgb = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(
        f"region_{hex_color.lstrip('#')}",
        [(1, 1, 1, 0), (*rgb, 0.18), (*rgb, REGION_ALPHA)],
        N=256,
    )


def build_region_color_map(region_indexes):
    region_indexes = sorted(set(region_indexes))
    if len(region_indexes) > len(NATURE_COLORS):
        raise ValueError(
            f"需要 {len(region_indexes)} 种脑区颜色，但当前仅提供 "
            f"{len(NATURE_COLORS)} 种。请扩展 NATURE_COLORS 或调小 "
            f"--max_regions_per_figure。"
        )
    return {region_idx: NATURE_COLORS[i]
            for i, region_idx in enumerate(region_indexes)}


def load_aal_atlas():
    atlas = datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = atlas.maps
    labels = list(atlas.labels)
    indices = list(atlas.indices)

    regions = []
    for label_name, label_value in zip(labels, indices):
        value = int(label_value)
        if value == 0:
            continue
        if "background" in label_name.lower():
            continue
        regions.append({"name": label_name, "value": value})

    print(f"AAL116 有效脑区数量: {len(regions)}")
    return atlas_img, regions


def build_region_masks(atlas_nii, atlas_data, region_indexes, regions):
    masks = []
    names = []
    for idx in region_indexes:
        if idx < 1 or idx > len(regions):
            raise ValueError(f"脑区索引 {idx} 超出范围 1~{len(regions)}")
        region = regions[idx - 1]
        mask_data = (atlas_data == region["value"]).astype(np.float64)
        voxel_count = int(mask_data.sum())
        if voxel_count == 0:
            raise ValueError(
                f"AAL{idx} ({region['name']}, value={region['value']}) "
                "在当前 atlas 中没有体素，无法绘制。"
            )
        mask_img = image.new_img_like(atlas_nii, mask_data, copy_header=True)
        masks.append(mask_img)
        names.append(region["name"])
        print(f"  AAL{idx:3d}  value={region['value']:4d}  "
              f"voxels={voxel_count:5d}  {region['name']}")
    return masks, names


def plot_network(net_name, region_indexes, output_dir, region_colors,
                 atlas_nii, atlas_data, regions, aal_names, importance):
    """绘制单个网络的玻璃脑图（每脑区独立配色 + 图例标注重要性）。"""
    masks, names = build_region_masks(atlas_nii, atlas_data, region_indexes, regions)
    colors = [region_colors[idx] for idx in region_indexes]

    display = plotting.plot_glass_brain(
        masks[0],
        title=f"{net_name}  (n={len(masks)})",
        display_mode=DISPLAY_MODE,
        threshold=MASK_THRESHOLD,
        cmap=_make_cmap(colors[0]),
        vmin=0.0,
        vmax=1.0,
        colorbar=False,
        plot_abs=False,
        black_bg=False,
    )
    for i in range(1, len(masks)):
        display.add_overlay(
            masks[i],
            threshold=MASK_THRESHOLD,
            cmap=_make_cmap(colors[i]),
            vmin=0.0,
            vmax=1.0,
            colorbar=False,
        )

    legend_patches = [
        Patch(facecolor=colors[i], edgecolor="#4D4D4D", linewidth=0.5,
              label=f"AAL{region_indexes[i]} {names[i]} "
                    f"(AAL名:{aal_names[region_indexes[i] - 1]}, "
                    f"ΔMAE={importance[region_indexes[i] - 1]:.4f})")
        for i in range(len(masks))
    ]
    display.frame_axes.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=1,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.32),
    )

    if SAVE_FIG:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"glass_brain_{net_name}.png")
        display.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"  已保存: {fname}")


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.input_json)), "brain_regions")

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    importance = np.asarray(payload["roi_importance"], dtype=float)
    region_indexes, criterion = load_significant_regions(args)
    aal_names, networks = load_network_labels(args.aal_file, importance.size)

    print("=" * 60)
    print(f"重要性文件: {args.input_json}")
    print(f"判据: {criterion} | 脑区数: {len(region_indexes)}")
    print(f"输出目录: {output_dir}")

    groups = group_by_network(region_indexes, importance, networks,
                              int(args.max_regions_per_figure))
    print(f"网络分组: " + ", ".join(f"{k}({len(v)})" for k, v in groups.items()))

    atlas_img, regions = load_aal_atlas()
    atlas_nii = image.load_img(atlas_img)
    atlas_data = atlas_nii.get_fdata()
    region_colors = build_region_color_map(
        [idx for idxs in groups.values() for idx in idxs])

    for net_name, idxs in groups.items():
        print(f"\n{'=' * 60}")
        print(f"处理网络: {net_name}  (脑区: {idxs})")
        plot_network(net_name, idxs, output_dir, region_colors,
                     atlas_nii, atlas_data, regions, aal_names, importance)

    print(f"\n{'=' * 60}")
    print("全部网络玻璃脑图生成完毕。")
    plotting.show()


if __name__ == "__main__":
    main()