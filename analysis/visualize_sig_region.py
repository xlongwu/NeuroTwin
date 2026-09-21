# -*- coding: utf-8 -*-
"""
按脑网络分组可视化 AAL116 显著脑区（玻璃脑图，各脑区不同颜色，Nature风格）。

网络分组（来自 region_significance.csv p<0.001 的脑区）：
  DMN: 5, 24, 26, 27, 68, 86
  FPN: 8, 61
  VAN: 12, 14, 29
  LN:  10, 83
  VN:  50
"""

import os

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
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


'''
DMN：65、6、66
SMN：17、20、70
VN：47、55、45
Subcortical：76
'''
NETWORK_GROUPS = {
    "DMN": [6,65,66],
    "SMN": [17,20,70],
    "VN":  [47,55,45],
    "Subcortical": [76],
}

# Muted, color-blind-conscious palette suitable for Nature-style figures.
# Colors are assigned by AAL index, so the same brain region keeps the same color.
NATURE_COLORS = [
    "#3B6EA8", "#D05A50", "#4B9B82", "#A2678A",
    "#E3A246", "#5E9FBA", "#8C7A4F", "#7A7A7A",
    "#BC6C25", "#6D87C3", "#75A87B", "#C56E90",
    "#9A8F4F", "#4F8C8B", "#B06C49", "#6F6F9E",
]

SAVE_FIG = True
OUTPUT_DIR = "results/explainability"
MASK_THRESHOLD = 0.5
REGION_ALPHA = 0.88
DISPLAY_MODE = "lyrz"


def _make_cmap(hex_color):
    rgb = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(
        f"region_{hex_color.lstrip('#')}",
        [(1, 1, 1, 0), (*rgb, 0.18), (*rgb, REGION_ALPHA)],
        N=256,
    )


def build_region_color_map(network_groups):
    region_indexes = sorted({
        region_idx
        for region_list in network_groups.values()
        for region_idx in region_list
    })
    if len(region_indexes) > len(NATURE_COLORS):
        raise ValueError(
            f"需要 {len(region_indexes)} 种脑区颜色，但当前仅提供 "
            f"{len(NATURE_COLORS)} 种。请扩展 NATURE_COLORS。"
        )
    return {
        region_idx: NATURE_COLORS[i]
        for i, region_idx in enumerate(region_indexes)
    }


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
    voxel_counts = []
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
        voxel_counts.append(voxel_count)
        print(
            f"  AAL{idx:3d}  value={region['value']:4d}  "
            f"voxels={voxel_count:5d}  {region['name']}"
        )
    return masks, names, voxel_counts


def main():
    print("=" * 60)
    print("加载 AAL116 模板...")
    atlas_img, regions = load_aal_atlas()

    atlas_nii = image.load_img(atlas_img)
    atlas_data = atlas_nii.get_fdata()
    region_colors = build_region_color_map(NETWORK_GROUPS)

    for net_name, region_indexes in NETWORK_GROUPS.items():
        print(f"\n{'=' * 60}")
        print(f"处理网络: {net_name}  (脑区: {region_indexes})")

        masks, names, _ = build_region_masks(
            atlas_nii, atlas_data, region_indexes, regions
        )

        n_regions = len(masks)
        colors = [region_colors[idx] for idx in region_indexes]

        first_cmap = _make_cmap(colors[0])
        display = plotting.plot_glass_brain(
            masks[0],
            title=f"{net_name}",
            display_mode=DISPLAY_MODE,
            threshold=MASK_THRESHOLD,
            cmap=first_cmap,
            vmin=0.0,
            vmax=1.0,
            colorbar=False,
            plot_abs=False,
            black_bg=False,
        )

        for i in range(1, n_regions):
            cmap = _make_cmap(colors[i])
            display.add_overlay(
                masks[i],
                threshold=MASK_THRESHOLD,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                colorbar=False,
            )

        legend_patches = [
            Patch(facecolor=colors[i],
                  edgecolor="#4D4D4D", linewidth=0.5,
                  label=f"AAL{region_indexes[i]} {names[i]}")
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

        if SAVE_FIG:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fname = f"{OUTPUT_DIR}/glass_brain_{net_name}.png"
            display.savefig(fname, dpi=300, bbox_inches="tight")
            print(f"  已保存: {fname}")

    print(f"\n{'=' * 60}")
    print("全部网络玻璃脑图生成完毕。")
    plotting.show()


if __name__ == "__main__":
    main()
