# coding=utf-8
"""F0b 被试构成冲积图（cohort → split → 站点匹配 B7 子集）。

借鉴图库样例 alluvial-stages-outlined（描边竖条 + 半透明 S 形飘带 + 横向虚线
刻度；深色竖条一档、飘带浅一档）；数据全部来自 v2 已落盘产物。
运行：python -m analysis.paper_figs.fig_f0b_composition
"""
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from analysis.paper_figs.common import (HC, HC_PALE, INK, MDD, MDD_PALE,
                                        NEGC, apply_style, matched_pairs,
                                        savefig)

DARK = {'HC': HC, 'MDD': MDD, 'Matched': '#c99a3f'}
PALE = {'HC': HC_PALE, 'MDD': MDD_PALE, 'Matched': '#f5e3c0'}


def build_flows():
    splits = {}
    for g in ('hc', 'mdd'):
        for s in ('train', 'val', 'test'):
            import csv
            with open(f'results/pathology_analysis/run1/z_maps/'
                      f'per_subject_z_{g}_{s}.csv') as f:
                splits.update({r['subject']: (g.upper(), s)
                               for r in csv.DictReader(f)})
    pairs = matched_pairs()
    matched = set()
    for p in pairs:
        matched.add(p['hc']); matched.add(p['mdd'])
    n = Counter(splits.values())
    m = Counter(splits[subj] for subj in matched)
    return n, m


def main():
    apply_style()
    n, m = build_flows()
    # 阶段 A：组；B：split；C：匹配与否
    stageA = [('HC', n[('HC', 'train')] + n[('HC', 'val')] + n[('HC', 'test')]),
              ('MDD', n[('MDD', 'train')] + n[('MDD', 'val')] + n[('MDD', 'test')])]
    stageB = [(f'{g}\n{s}', n[(g, s)])
              for g in ('HC', 'MDD') for s in ('train', 'val', 'test')]
    stageC = [('HC\nmatched', m[('HC', 'train')] + m[('HC', 'val')] + m[('HC', 'test')]),
              ('HC\nother', sum(v for (g, _), v in n.items()
                                if g == 'HC') - sum(v for (g, _), v in m.items() if g == 'HC')),
              ('MDD\nmatched', m[('MDD', 'train')] + m[('MDD', 'val')] + m[('MDD', 'test')]),
              ('MDD\nother', sum(v for (g, _), v in n.items()
                                 if g == 'MDD') - sum(v for (g, _), v in m.items() if g == 'MDD'))]

    stages = [stageA, stageB, stageC]
    n_stage = len(stages)
    GAP = 260.0
    widths = []
    for st in stages:
        widths.append(sum(w for _, w in st) + GAP * (len(st) - 1))
    total = max(widths)
    tops = []
    for st in stages:
        # 每阶段整体居中
        start = (total - (sum(w for _, w in st) + GAP * (len(st) - 1))) / 2
        t, cur = [], start
        for _, w in st:
            t.append((cur, w)); cur += w + GAP
        tops.append(t)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.93, bottom=0.14)
    u = np.linspace(0, 1, 60)
    ease = 3 * u ** 2 - 2 * u ** 3

    def node_color(stage_idx, label):
        lab = label.split('\n')[0]
        if stage_idx == 2:
            return DARK['Matched'] if 'matched' in label else NEGC
        return DARK[lab]

    def flow_colors(si, i, sj, j):
        gi_a = stageA_labels(stages, si, i)
        gi_b = stageA_labels(stages, sj, j)
        # 飘带着色：用上游组的浅色；匹配流向用金色浅档
        if sj == 2 and 'matched' in stages[sj][j][0]:
            return PALE['Matched']
        return PALE[gi_a or gi_b]

    for si in range(n_stage - 1):
        sj = si + 1
        a_use = [0.0] * len(stages[si])
        b_use = [0.0] * len(stages[sj])
        for i, (_, wa) in enumerate(stages[si]):
            for j, (_, wb) in enumerate(stages[sj]):
                flow = flows_between(stages, si, i, sj, j, n, m)
                if flow <= 0:
                    continue
                ya = tops[si][i][0] + wa - a_use[i]
                yb = tops[sj][j][0] + wb - b_use[j]
                upper = ya + (yb - ya) * ease
                col = flow_colors(si, i, sj, j)
                ax.fill_between(si + 0.05 + 0.9 * u, upper - flow, upper,
                                color=col, alpha=0.75, lw=0, zorder=2)
                a_use[i] += flow
                b_use[j] += flow
    for si, st in enumerate(stages):
        for k, (lab, w) in enumerate(st):
            t, _ = tops[si][k]
            ax.add_patch(Rectangle((si - 0.05, t), 0.1, w,
                                   fc=node_color(si, lab), ec='black',
                                   lw=1.2, zorder=4))
            ax.text(si, t + w + 90, f'{w:,}', ha='center', va='bottom',
                    fontsize=8.5)
    for frac in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(total * frac, color='black', lw=0.6, ls=(0, (4, 3)),
                   zorder=1)
    ax.set_xlim(-0.3, n_stage - 0.7)
    ax.set_ylim(-total * 0.02, total * 1.04)
    ax.set_xticks(range(n_stage),
                  ['Cohort', 'Split (z-score sets)', 'Site-matched B7 subset'],
                  fontsize=10)
    ax.set_yticks([])
    ax.tick_params(bottom=False, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('Subject composition: cohorts, splits, and the '
                 'site-matched analysis subset', pad=8)
    for si, st in enumerate(stages):
        for k, (lab, w) in enumerate(st):
            t, _ = tops[si][k]
            side = 'left' if si == 0 else ('right' if si == n_stage - 1
                                           else 'center')
            if si == 0:
                ax.text(-0.12, t + w / 2, lab, ha='right', va='center',
                        fontsize=9.5, fontweight='bold')
            elif si == n_stage - 1:
                ax.text(n_stage - 1 + 0.12, t + w / 2, lab.replace('\n', ' '),
                        ha='left', va='center', fontsize=8.5)
    fig.legend(handles=[Patch(fc=HC, ec='black', lw=0.8, label='HC'),
                        Patch(fc=MDD, ec='black', lw=0.8, label='MDD'),
                        Patch(fc=DARK['Matched'], ec='black', lw=0.8,
                              label='in matched pairs (B7)'),
                        Patch(fc=NEGC, ec='black', lw=0.8,
                              label='not in matched pairs')],
               loc='lower center', ncol=4, fontsize=8.5, handlelength=1.3,
               bbox_to_anchor=(0.5, 0.005))
    savefig(fig, 'fig_f0b_composition')


def stageA_labels(stages, si, i):
    lab = stages[si][i][0].split('\n')[0]
    return 'HC' if lab == 'HC' else ('MDD' if lab == 'MDD' else '')


def flows_between(stages, si, i, sj, j, n, m):
    """相邻阶段节点间的被试数。"""
    if si == 0:
        g = stages[0][i][0]
        return sum(v for (gg, s), v in n.items() if gg == g
                   and stages[1][j][0].split('\n') == [gg, s])
    # split → matched/other
    lab = stages[1][i][0].split('\n')
    g, s = lab[0], lab[1]
    matched_flag = 'matched' in stages[2][j][0]
    key = (g, s)
    tot = n[key]
    mt = m.get(key, 0)
    return mt if matched_flag else tot - mt


if __name__ == '__main__':
    main()
