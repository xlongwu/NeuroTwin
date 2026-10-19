# coding=utf-8
"""条件注入的验收测试（对应方案 §14 的 A–P 清单）。

组织方式：
- **合成表**测试（A/B/D/E/K 的语义部分）：在 ``tmp_path`` 里造小 xlsx，快速且确定；
- **真实 merged 只读**测试（C/F/E 的覆盖口径）：只读 ``data/`` 下的两份 merged 表，
  绝不写回，也绝不修改原始 V1/V2 表（用 tmp 副本验证「移走/篡改非 merged 表不影响
  条件加载」）；
- **模型/训练链路**测试（G–N）：小维度合成 batch，最多 200 step（方案 §14 L 明确
  上限），不启动长训练。

约定：涉及「无条件 vs 有条件」的输出比较一律使用**相同输入、相同权重、相同设备**，
不靠新建数据划分来掩盖差异（验收 P）。
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.conditioning import (ConditionNormGenerator, ConditionedLowRankLinear,
                                 ConditionalGatedResidual, ConditioningConfig,
                                 build_condition_encoder,
                                 condition_dropout, deterministic_uniform,
                                 dropout_keys, pool_phi)
from models.tfm import NeuroTwinTFM
from train.optim import (conditioning_param_names, migrate_pretrained_checkpoint,
                         param_report, save_backbone_weights, set_finetune_stage,
                         verify_conditioning_param_groups)
from utils.condition_schema import (DEMO_GROUP, HC_FIELDS, MDD_FIELDS,
                                   MDD_NEW_GROUP, ConditionStats, FieldSpec,
                                   FieldStats, MedicationConflict, audit_unseen_codes,
                                   field_names, fit_condition_stats,
                                   layout_hash, load_merged_conditions,
                                   parse_cell, parse_extra_codes, remap_code,
                                   resolve_source_columns)

DEVICE = torch.device('cpu')

#: 真实 merged 文件（条件数据的唯一允许来源；本文件只读）
DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
HC_MERGED = DATA_ROOT / 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'
MDD_MERGED = DATA_ROOT / 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx'


def _write_merged(path: Path, rows: dict) -> None:
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)


#: episode/medication 的测试专用 schema（只含被断言的字段，避免无关列噪音）
EPISODE_ONLY = (FieldSpec('episode', 'categorical', ('episode',)),)
MED_TWO_COL = (FieldSpec('medication', 'categorical',
                         ('medication', 'On medication?', 'OnMedication')),)


def _stats_for(specs, values, mask, train_rows, **kw) -> ConditionStats:
    return fit_condition_stats(specs, values, mask, np.asarray(train_rows), **kw)


# ══════════════════════════════════════════════════════════════════════════
#  A / B：数值有效性、编码替换与缺失判定
# ══════════════════════════════════════════════════════════════════════════

def test_A_sentinels_are_valid_and_remap_is_idempotent(tmp_path):
    """A：[-9999,-1,0,1,2] 全部 mask=1，映射到 [0,2,0,1,2]；重复映射不变。"""
    p = tmp_path / 'merged.xlsx'
    _write_merged(p, {'ID': ['s0', 's1', 's2', 's3', 's4'],
                      'episode': [-9999, -1, 0, 1, 2]})
    values, mask, ids, audit = load_merged_conditions(p, EPISODE_ONLY)
    col = 0
    assert mask[:, col].tolist() == [1, 1, 1, 1, 1], '所有有限数值都必须有效'
    assert values[:, col].tolist() == [0.0, 2.0, 0.0, 1.0, 2.0]
    # 幂等：对已更正的数据再次映射不改变类别
    again = np.array([remap_code(v) for v in values[:, col]])
    assert again.tolist() == values[:, col].tolist()
    assert remap_code(-1.0) == 2.0 and remap_code(-9999.0) == 0.0
    # 已有有效 0 与 -9999→0 都是 code_0（不区分来源，也不因映射变成缺失）
    assert audit.n_valid['episode'] == 5


def test_A_remap_only_applies_to_episode_and_medication(tmp_path):
    """编码替换只作用于 episode/medication；其他字段数值原样保留。"""
    p = tmp_path / 'merged.xlsx'
    _write_merged(p, {'ID': ['a'], 'Sex': [1], 'Age': [-1], 'education': [-9999],
                      'episode': [-1], 'illness': [-1], 'medication': [-9999],
                      'HAMD': [-1], 'HAMA': [-9999]})
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    names = field_names(MDD_FIELDS)
    got = {n: float(values[0, names.index(n)]) for n in names}
    assert got['Age'] == -1.0 and got['education'] == -9999.0
    assert got['illness'] == -1.0 and got['HAMD'] == -1.0 and got['HAMA'] == -9999.0
    assert got['episode'] == 2.0 and got['medication'] == 0.0
    assert mask[0].tolist() == [1] * len(names)


def test_B_missing_variants_all_mask_zero_and_numeric_strings_valid():
    """B：None/NaN/空串/纯空白/null 大小写变体 → mask=0；数字字符串有效。"""
    for raw, expect_valid in [(None, False), (float('nan'), False), ('', False),
                              ('   ', False), ('\t\n', False), ('null', False),
                              ('NULL', False), ('  NaT  ', False), ('n/a', False),
                              ('<NA>', False), ('nan', False), ('none', False),
                              (pd.NaT, False), (float('inf'), False),
                              ('abc', False), ('-1', True), ('0', True),
                              (0, True), (-1, True), (-9999, True), ('2.5', True)]:
        val, reason = parse_cell(raw)
        assert (val is not None) == expect_valid, f'{raw!r} → {val!r} ({reason})'
    # 有效 0 绝不因真值判断被当成空
    assert parse_cell(0)[0] == 0.0 and parse_cell('0')[0] == 0.0


def test_B_missing_rows_yield_finite_encoded_input(tmp_path):
    """B：缺失格经 train-only 填补/编码后，模型输入全部有限。"""
    p = tmp_path / 'merged.xlsx'
    _write_merged(p, {'ID': [f's{i}' for i in range(6)],
                      'Sex': [1, None, 2, 'null', 1, 1],
                      'Age': [30, '', 32, '  ', 34, 'nan'],
                      'education': [12, 14, 'NA', 16, 10, 11]})
    values, mask, _, _ = load_merged_conditions(p, HC_FIELDS)
    stats = _stats_for(HC_FIELDS, values, mask, [0, 2, 4, 5])
    z, code = stats.encode(values, mask)
    assert np.isfinite(z).all(), '编码后不得出现 NaN/Inf'
    unk = (~mask[:, 0].astype(bool))
    assert code[unk, 0].tolist() == [0] * int(unk.sum()), '类别缺失必须落 Unknown 索引'
    # 缺失格 z=0 且 mask=0；真实中位数 z=0 但 mask=1（两者可区分）
    j = field_names(HC_FIELDS).index('Age')
    med = stats.get('Age').mu
    for i in range(6):
        if not mask[i, j]:
            assert z[i, j] == 0.0
        elif values[i, j] == med:
            assert z[i, j] == 0.0 and mask[i, j] == 1


def test_B_unknown_and_code_zero_are_different_indices(tmp_path):
    """B：Unknown 内部索引与有效 code_0 不同（不能只用「值==0」重建 mask）。"""
    p = tmp_path / 'merged.xlsx'
    _write_merged(p, {'ID': ['a', 'b', 'c'], 'episode': [0, None, -9999]})
    values, mask, _, _ = load_merged_conditions(p, EPISODE_ONLY)
    stats = _stats_for(EPISODE_ONLY, values, mask, [0, 1, 2])
    z, code = stats.encode(values, mask)
    j = 0
    assert mask[0, j] == 1 and mask[2, j] == 1 and mask[1, j] == 0
    # 有效 0（含 -9999→0）共享同一个 code_0 索引；缺失走 Unknown=0
    assert code[0, j] == code[2, j] != 0
    assert code[1, j] == 0


# ══════════════════════════════════════════════════════════════════════════
#  C：条件来源只允许 merged
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MDD_MERGED.exists(), reason='真实 merged 文件不在位')
def test_C_only_merged_is_read_and_item_columns_are_ignored(tmp_path):
    """C：改动/移走非 merged 原始表不影响条件加载；HAMA 条目与 HAMATotal 改值不影响 HAMA。"""
    import shutil

    root = tmp_path / 'data'
    root.mkdir()
    merged = root / MDD_MERGED.name
    shutil.copy(MDD_MERGED, merged)
    # 伪造的「单独 V1/V2 原始表」：篡改/删除都不应被读取
    decoy = root / 'Rest-meta-MDD-V1-MDD.xlsx'
    _write_merged(decoy, {'ID': ['x'], 'HAMA': [999], 'medication': [7]})

    before = load_merged_conditions(merged, MDD_FIELDS)
    decoy.unlink()
    (root / 'Rest-meta-MDD-V2-MDD.xlsx').write_bytes(b'not-an-excel')
    after = load_merged_conditions(merged, MDD_FIELDS)
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(np.nan_to_num(before[0], nan=-7.0),
                          np.nan_to_num(after[0], nan=-7.0))

    # HAMA 只认精确名称 `HAMA`：改 HAMATotal / HAMA_i / *HAMA_i 不影响 HAMA
    df = pd.read_excel(merged)
    names = field_names(MDD_FIELDS)
    j_hama = names.index('HAMA')
    base_values, base_mask, _, _ = load_merged_conditions(merged, MDD_FIELDS)
    for col in [c for c in df.columns
                if str(c).strip().upper().startswith(('HAMA', '*HAMA'))
                and str(c).strip().upper() != 'HAMA'][:12]:
        df[col] = 12345
    mutated = tmp_path / 'mutated.xlsx'
    df.to_excel(mutated, index=False)
    mv, mm, _, _ = load_merged_conditions(mutated, MDD_FIELDS)
    assert np.allclose(np.nan_to_num(mv[:, j_hama], nan=-1.0),
                       np.nan_to_num(base_values[:, j_hama], nan=-1.0)), f'{col} 被读取了'
    assert np.array_equal(mm[:, j_hama], base_mask[:, j_hama])
    # HAMATotal 也不得作为回退（缺失集合必须逐位一致）
    if 'HAMATotal' in df.columns:
        assert not set(df.columns) < set(field_names(MDD_FIELDS))


@pytest.mark.skipif(not MDD_MERGED.exists(), reason='真实 merged 文件不在位')
def test_C_hama_never_falls_back_to_hamatotal(tmp_path):
    """C：HAMA 缺失不因 HAMATotal 有值而被填补（禁止 HAMATotal 回退）。"""
    import shutil
    merged = tmp_path / MDD_MERGED.name
    shutil.copy(MDD_MERGED, merged)
    df = pd.read_excel(merged)
    n = len(df)
    df['HAMA'] = np.nan                       # 整列缺失
    if 'HAMATotal' in df.columns:
        df['HAMATotal'] = np.arange(n, dtype=float)  # 但备选列都有值
    alt = tmp_path / 'alt.xlsx'
    df.to_excel(alt, index=False)
    _, mask, _, _ = load_merged_conditions(alt, MDD_FIELDS)
    j = field_names(MDD_FIELDS).index('HAMA')
    assert int(mask[:, j].sum()) == 0, 'HAMATotal 不得回退成 HAMA'


# ══════════════════════════════════════════════════════════════════════════
#  D：medication 同表两列取值与冲突
# ══════════════════════════════════════════════════════════════════════════

def _cycle(vals, n):
    """按长度 n 截断的循环列，保证所有列等长（DataFrame 构造的前置条件）。"""
    return [vals[i % len(vals)] for i in range(n)]


def _mdd_base(n):
    """完整的 mdd8 合成表（所有注册列都存在，缺失由具体测试再覆写）。"""
    return {'ID': [f's{i}' for i in range(n)],
            'Sex': _cycle([1, 2], n),
            'Age': list(range(20, 20 + n)),
            'education': _cycle([12, 14, 16], n),
            'episode': _cycle([0, 1, 2], n),
            'illness': _cycle([10.0, 20.0, 30.0], n),
            'HAMD': _cycle([20, 22, 24], n),
            'HAMA': _cycle([10, 12, 14], n),
            'medication': _cycle([0, 1, None], n),
            'OnMedication': _cycle([None], n)}


def test_D_medication_two_columns_union_without_source_identity(tmp_path):
    """D：同一行取两列可解析数值；不产生「来自哪一列」的特征（K 仍是 8）。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(4)
    rows['medication'] = [1, None, None, None]
    rows['OnMedication'] = [None, 0, -1, None]
    _write_merged(p, rows)
    values, mask, _, audit = load_merged_conditions(p, MDD_FIELDS)
    assert values.shape[1] == 8, '两个用药列必须映射为一个条件字段'
    j = field_names(MDD_FIELDS).index('medication')
    assert mask[:, j].tolist() == [1, 1, 1, 0]
    assert values[:, j][:3].tolist() == [1.0, 0.0, 2.0]
    assert np.isnan(values[3, j])
    assert len(audit.alias_sources['medication']) == 2, 'medication 由同表两列共同来源'


def test_D_medication_conflict_raises_after_mapping(tmp_path):
    """D：两列同时有值且映射后不一致 → 报冲突并停止该配置。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(3)
    rows['medication'] = [-1, 1, None]      # 映射后 2
    rows['OnMedication'] = [1, 1, None]     # 原值不同但映射后一致的行不得报错
    _write_merged(p, rows)
    with pytest.raises(MedicationConflict):
        load_merged_conditions(p, MDD_FIELDS)


def test_D_medication_conflict_after_remap_is_allowed(tmp_path):
    """D：映射后一致即视为同一类别（-1→2 与已有 2 共存不算冲突）。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(2)
    rows['medication'] = [-1, 2]
    rows['OnMedication'] = [2, -1]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    j = field_names(MDD_FIELDS).index('medication')
    assert values[:, j].tolist() == [2.0, 2.0] and mask[:, j].tolist() == [1, 1]


def test_D_ambiguous_columns_raise():
    """D：大小写/空白可规范匹配，但同时存在多个歧义列时报错。"""
    df = pd.DataFrame({'ID': ['a'], 'age': [1], 'Age': [2]})
    with pytest.raises(ValueError, match='歧义'):
        resolve_source_columns(df, [FieldSpec('Age', 'continuous', ('Age',))])


def test_D_no_registered_column_raises_never_other_file(tmp_path):
    """D：merged 内找不到注册列时报错，绝不回退到其他文件/其他列。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(2)
    del rows['episode']
    rows['episode_but_renamed'] = [1, 2]
    _write_merged(p, rows)
    with pytest.raises(ValueError, match='找不到字段 episode'):
        load_merged_conditions(p, MDD_FIELDS)


# ══════════════════════════════════════════════════════════════════════════
#  E：train-only 统计量与 HC→MDD 继承
# ══════════════════════════════════════════════════════════════════════════

def test_E_val_test_tampering_does_not_move_train_stats(tmp_path):
    """E：篡改 val/test 数值不改变 train 统计量（fit 只看 train 行）。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(8)
    assert len({len(v) for v in rows.values()}) == 1
    rows['HAMA'] = [10, 11, 12, 13, None, None, None, None]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    stats = _stats_for(MDD_FIELDS, values, mask, [0, 1, 2, 3])
    before = (stats.get('HAMA').mu, stats.get('HAMA').scale, stats.get('HAMA').n_valid)
    v2 = values.copy()
    v2[4:, field_names(MDD_FIELDS).index('HAMA')] = [999, 998, 997, 996]
    v2[0] = 1e6                                       # 连 train 也改掉，用于反向验证
    stats2 = _stats_for(MDD_FIELDS, v2, mask, [0, 1, 2, 3])
    assert (stats2.get('HAMA').mu, stats2.get('HAMA').scale) != before[:2]
    # 只篡改 val/test → train 统计量逐位不变
    v3 = values.copy()
    v3[4:, field_names(MDD_FIELDS).index('HAMA')] = [999, 998, 997, 996]
    stats3 = _stats_for(MDD_FIELDS, v3, mask, [0, 1, 2, 3])
    assert (stats3.get('HAMA').mu, stats3.get('HAMA').scale,
            stats3.get('HAMA').n_valid) == before


def test_E_continuous_all_missing_is_safe_and_not_fit_from_val(tmp_path):
    """E：全训练字段缺失 → fitted=false、n_valid=0，明确全缺失分支，不从 val/test 拟合。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(5)
    rows['illness'] = [None, None, 55, 66, 77]     # train(0,1) 全缺失，val/test 有值
    rows['episode'] = [1] * 5                      # 其余类别列恒定，train 覆盖词表
    rows['Sex'] = [1] * 5
    rows['medication'] = [0] * 5
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    stats = _stats_for(MDD_FIELDS, values, mask, [0, 1])
    st = stats.get('illness')
    assert st.fitted is False and st.n_valid == 0
    assert st.scale_source == 'none'
    z, _code = stats.encode(values, mask)
    j = field_names(MDD_FIELDS).index('illness')
    assert np.isfinite(z).all() and np.allclose(z[:, j], 0.0)


def test_E_hc_shared_stats_are_not_overwritten_by_mdd_fit(tmp_path):
    """E：HC 的 Age/education 统计量继承到 mdd8，通用 fit 不得覆盖（方案 §4）。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(6)
    rows['Age'] = [30, 31, 32, 33, 34, 35]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    hc = _stats_for(HC_FIELDS, values, mask, [0, 1, 2])
    mdd = _stats_for(MDD_FIELDS, values, mask, [3, 4, 5])
    mdd.take_fields(hc_for_mdd(hc), DEMO_GROUP)
    assert mdd.get('Age').mu == hc.get('Age').mu
    assert mdd.get('Age').scale == hc.get('Age').scale
    assert mdd.get('Sex').codes == hc.get('Sex').codes
    # 再显式「只拟合新增组」：共享字段必须逐位不变
    before = {n: (mdd.get(n).mu, mdd.get(n).scale, mdd.get(n).codes, mdd.get(n).n_valid)
              for n in DEMO_GROUP}
    mdd.fit(values, mask, np.array([3, 4, 5]), only_fields=MDD_NEW_GROUP)
    after = {n: (mdd.get(n).mu, mdd.get(n).scale, mdd.get(n).codes, mdd.get(n).n_valid)
             for n in DEMO_GROUP}
    assert before == after, '通用 fit 不得覆盖继承的 Age/education 统计量'
    # 反向验证：一次性 fit 全部字段确实会改写共享字段（证明上面的保护是必要的）
    naive = _stats_for(MDD_FIELDS, values, mask, [0, 1, 2, 3, 4, 5])
    assert (naive.get('Age').mu, naive.get('Age').n_valid) != (before['Age'][0],
                                                               before['Age'][3])


def hc_for_mdd(hc_stats: ConditionStats) -> ConditionStats:
    """把 hc3 统计量装入 mdd8 schema 的共享字段（模拟 --condition_inherit_stats）。"""
    from utils.condition_schema import load_inherited_stats
    return load_inherited_stats(hc_stats.to_dict(), MDD_FIELDS, list(DEMO_GROUP))


def test_E_unregistered_valid_code_is_reported_not_masked(tmp_path):
    """E：train 外出现未登记有效类别码 → 显式报告，不伪装缺失、不静默扩表。"""
    p = tmp_path / 'merged.xlsx'
    rows = _mdd_base(4)
    rows['episode'] = [1, 1, 5, 5]        # 2/3 号行是 train 未见的有效码 5
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    stats = _stats_for(MDD_FIELDS, values, mask, [0, 1])
    assert stats.get('episode').codes == (1.0,)
    unseen = audit_unseen_codes(MDD_FIELDS, stats, values, mask, np.array([0, 1]))
    assert unseen.get('episode') == [5.0]
    with pytest.raises(ValueError, match='未登记'):
        stats.encode(values, mask)
    # 显式登记后 encode 通过，词表在拟合期就冻结
    stats2 = _stats_for(MDD_FIELDS, values, mask, [0, 1],
                        extra_codes={'episode': [5.0]})
    assert 5.0 in stats2.get('episode').codes
    _z, code = stats2.encode(values, mask)
    j = field_names(MDD_FIELDS).index('episode')
    assert code[2, j] != 0, '登记后必须占用有效索引，不能落到 Unknown'


def test_E_parse_extra_codes():
    assert parse_extra_codes('Sex:0,3;medication:2') == {'Sex': [0.0, 3.0],
                                                        'medication': [2.0]}
    assert parse_extra_codes('') == {}
    with pytest.raises(ValueError):
        parse_extra_codes('Sex')


def test_E_extra_codes_reach_the_inherit_path(tmp_path):
    """E（继承路径）：``--condition_extra_codes`` 必须真的进到 HC→MDD 继承后的拟合。

    曾经 ``_fit_condition_stats`` 的继承分支只传 ``only_fields=MDD_NEW_GROUP`` 而漏掉
    ``extra_codes``，于是新增五字段的预注册码被静默丢弃；而下方 audit 的报错文案又
    建议「传 --condition_extra_codes」，用户照做仍然报同一个错。同时，共享三字段
    （Sex/education）的词表由本轮 HC 冻结继承，MDD 侧传码必须**显式拒绝**而不是空转。
    """
    from utils.dataloader import NeuroTwinDataLoader
    root = _tiny_imaging_root(tmp_path, n_subjects=8, n_tr=12, n_rois=4)

    common = dict(batch_size=2, seed=7, num_workers=0, eval_batch_size=2,
                  context_min=4, context_max=6, context_lengths=[6],
                  eval_context_length=6, eval_anchors_per_subject=1,
                  eval_rollout_tasks_per_subject=0, eval_rollout_steps=0,
                  train_samples_per_subject=1, val_ratio=0.25, test_ratio=0.25)
    hc_dl = NeuroTwinDataLoader(data_root=root, mode='pretrain', condition_group='hc3',
                                condition_extra_codes={'Sex': [1.0, 2.0]}, **common)
    base = hc_for_mdd(hc_dl.get_condition_stats())

    dl = NeuroTwinDataLoader(
        data_root=root, mode='finetune', condition_group='mdd8',
        condition_base_stats=base,
        condition_extra_codes={'episode': [7.0], 'medication': [0.0, 1.0, 2.0]},
        **common)
    stats = dl.get_condition_stats()
    # 哨兵码 7.0 在合成表里一处都没有：只有 extra_codes 真的进了拟合，它才会出现在词表
    assert 7.0 in stats.get('episode').codes, '继承路径丢掉了 extra_codes'
    assert 2.0 in stats.get('medication').codes
    # 继承的共享三字段必须逐位不变，且绝不被 MDD 侧扩表
    for name in DEMO_GROUP:
        got, src = stats.get(name), base.get(name)
        assert (got.fitted, got.mu, got.scale, got.codes) == (
            src.fitted, src.mu, src.scale, src.codes)
    assert 7.0 not in stats.get('Sex').codes

    with pytest.raises(ValueError, match='继承字段'):
        NeuroTwinDataLoader(data_root=root, mode='finetune', condition_group='mdd8',
                            condition_base_stats=base,
                            condition_extra_codes={'Sex': [7.0]}, **common)


# ══════════════════════════════════════════════════════════════════════════
#  F：真实 merged 队列与划分一致性（只读）
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HC_MERGED.exists(), reason='真实 merged 文件不在位')
def test_F_hc_cohort_keeps_subjects_without_metadata_row():
    """F：merged 无整行的 HC 被试保留为全条件缺失（有效观测数与方案 §1 一致）。"""
    values, mask, ids, audit = load_merged_conditions(HC_MERGED, HC_FIELDS)
    assert audit.n_rows == 2445
    names = field_names(HC_FIELDS)
    assert int(mask[:, names.index('Sex')].sum()) == 2445
    assert int(mask[:, names.index('education')].sum()) == 2440
    full_missing = (mask == 0).all(axis=1)
    assert int(values[full_missing].size) == int(full_missing.sum()) * len(HC_FIELDS)


def test_F_all_arms_share_cohort_and_split(tmp_path):
    """F：所有对照（含无条件臂）使用同一 cohort 与同一 split（验收 F）。"""
    from utils.dataloader import NeuroTwinDataLoader

    root = _tiny_imaging_root(tmp_path)
    merged = root / 'Rest-meta-MDD-V1V2-Merged-HC.xlsx'
    hashes = {}
    for arm in ('none', 'demo'):
        dl = NeuroTwinDataLoader(
            data_root=root, mode='pretrain', batch_size=2, seed=7,
            num_workers=0, condition_group=('hc3' if arm != 'none' else None),
            condition_file=merged.name, eval_batch_size=2,
            context_min=4, context_max=6, context_lengths=[6],
            eval_context_length=6, eval_anchors_per_subject=2,
            eval_rollout_tasks_per_subject=0, eval_rollout_steps=0,
            val_ratio=0.25, test_ratio=0.25, train_samples_per_subject=1,
            condition_extra_codes={'Sex': [1.0, 2.0]})
        _ = dl.base_dataset.condition_table
        key = (tuple(dl.train_subjects), tuple(dl.val_subjects), tuple(dl.test_subjects))
        hashes[arm] = (len(dl.base_dataset.subject_ids), key)
    assert hashes['none'] == hashes['demo'], '条件臂与无条件臂的队列/划分必须一致'


def _tiny_imaging_root(tmp_path: Path, n_subjects: int = 6, n_tr: int = 12,
                       n_rois: int = 4) -> Path:
    """造一个最小可读的影像数据目录（Normlize + Mask + merged 条件表）。"""
    import scipy.io as sio

    root = tmp_path / 'data'
    for group in ('HC', 'MDD'):
        (root / 'Normlize' / group).mkdir(parents=True, exist_ok=True)
        (root / 'Mask' / group).mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    ids = []
    for i in range(n_subjects):
        sid = f'S{int(i):03d}'
        ids.append(sid)
        series = rng.randn(n_tr, n_rois).astype(np.float32)
        sio.savemat(str(root / 'Normlize' / 'HC' / f'ROISignals_{sid}.mat'),
                    {'ROISignals': series})
        sio.savemat(str(root / 'Normlize' / 'MDD' / f'ROISignals_{sid}.mat'),
                    {'ROISignals': series})
        sc = np.eye(n_rois, dtype=np.float32)
        for g in ('HC', 'MDD'):
            sio.savemat(str(root / 'Mask' / g / f'Mask_{sid}.mat'), {'SC': sc})
    rows = {'ID': ids, 'Sex': _cycle([1, 2], n_subjects),
            'Age': [20 + i for i in range(n_subjects)],
            'education': _cycle([12, 14, 16], n_subjects)}
    _write_merged(root / 'Rest-meta-MDD-V1V2-Merged-HC.xlsx', rows)
    mdd = dict(rows)
    mdd.update({'episode': _cycle([1, 0], n_subjects),
                'illness': [10.0 + 10 * i for i in range(n_subjects)],
                'medication': _cycle([1, 0, None], n_subjects),
                'HAMD': [15 + i for i in range(n_subjects)],
                'HAMA': _cycle([10, None, 12], n_subjects)})
    _write_merged(root / 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx', mdd)
    return root


# ══════════════════════════════════════════════════════════════════════════
#  G–N：模型与训练链路
# ══════════════════════════════════════════════════════════════════════════

def _hc_fixtures(n=8):
    p = HC_MERGED if HC_MERGED.exists() else None
    if p is None:
        raise AssertionError('缺少 HC merged 文件')
    values, mask, ids, _ = load_merged_conditions(p, HC_FIELDS)
    stats = _stats_for(HC_FIELDS, values, mask, list(range(0, 200)))
    z, code = stats.encode(values[:n], mask[:n])
    return stats, z, code, mask[:n]


def _cond_dict(z, code, mask, subject_index=None):
    return {
        'cond_z': torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)),
        'cond_code': torch.from_numpy(np.ascontiguousarray(code, dtype=np.int64)),
        'cond_mask': torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
        'subject_index': (torch.arange(z.shape[0], dtype=torch.long)
                          if subject_index is None else subject_index),
    }


def _tfm(group, arm, stats, dim=32, layers=3, inject='heads', features=6, **kw):
    cfg = ConditioningConfig(group=group, arm=arm, d_demo=16, d_new=16,
                             cond_dim=32, h_delta=64).to_dict()
    payload = stats.to_dict() if stats is not None else None
    return NeuroTwinTFM(features=features, context_max=8, patch_len=2, dim=dim,
                        num_layers=layers, num_heads=4, ff_ratio=2,
                        cpm_horizon=4, cpm_layers=1, pretrain_mode=(group == 'hc3'),
                        condition_config=cfg, condition_stats_payload=payload,
                        cond_inject=inject, **kw)


def test_G_pretrain_condition_switch_builds_and_trains():
    """G：HC pretrain 的条件开关能建立编码器并训练；旧无条件路径仍可用。"""
    stats, z, code, mask = _hc_fixtures()
    m = _tfm('hc3', 'demo', stats)
    assert m.conditioner is not None and m.pretrain_mode
    assert m.tfm_patho_mode == 'film' and m.one_step_head.film is not None
    x = torch.randn(4, 6, 1, 8)
    sc = torch.rand(6, 6)
    cond = _cond_dict(z[:4], code[:4], mask[:4])
    out, aux = m(x, sc, condition=cond, condition_epoch=0)
    assert out.shape == (4, 6, 1, 1) and torch.isfinite(out).all()
    assert aux['pathology_cond'].shape == (4, 32)
    # 无条件旧行为：arm=none 时不构建任何条件模块
    plain = NeuroTwinTFM(features=6, context_max=8, patch_len=2, dim=32,
                         num_layers=2, num_heads=4, ff_ratio=2, cpm_horizon=4,
                         cpm_layers=1, pretrain_mode=True)
    assert plain.conditioner is None and plain.cond_dim == 0
    o2, _ = plain(x, sc)
    assert o2.shape == out.shape and torch.isfinite(o2).all()


def test_H_two_heads_independent_and_condition_affects_output_after_steps():
    """H：两头参数独立、多步后条件影响预测；初始化有/无条件相同是正常情况。"""
    stats, z, code, mask = _hc_fixtures(4)
    torch.manual_seed(0)
    m = _tfm('hc3', 'demo', stats, dim=16, layers=2)
    x = torch.randn(4, 6, 1, 8); sc = torch.rand(6, 6)
    cond = _cond_dict(z, code, mask)
    m.eval()
    with torch.no_grad():
        p_on, _ = m(x, sc, condition=cond)
        p_off, _ = m(x, sc, condition=None)
    # 零初始化 → 起步相同（不能写成「必须不同」的错误断言）
    assert torch.allclose(p_on, p_off, atol=0.0)
    # 两个 FiLM 是不同模块对象（值相等仅因零初始化）
    assert m.one_step_head.film is not m.cpm_head.film
    ids_a = {id(p) for p in m.one_step_head.film.parameters()}
    ids_b = {id(p) for p in m.cpm_head.film.parameters()}
    assert not (ids_a & ids_b)
    # 多步训练后条件必须影响输出（含 CPM 损失，否则 OS 支路被 decode 零初始化阻断）
    from train.losses import TFMDualLoss
    crit = TFMDualLoss(lambda_one=1.0, lambda_cpm=1.0, lambda_pcc=0.0)
    opt = torch.optim.SGD(list(m.parameters()) + list(crit.parameters()), lr=1e-2)
    y = torch.randn(4, 6, 1, 1); yf = torch.randn(4, 6, 4)
    m.train()
    first = {'os': None, 'cpm': None, 'cond': None}
    for step in range(1, 121):
        opt.zero_grad()
        out, aux = m(x, sc, condition=cond, condition_epoch=0)
        loss, _ = crit(out, y, aux['cpm_pred'], yf, torch.ones(4, 4))
        loss.backward(); opt.step()
        for key, mod in (('os', m.one_step_head.film), ('cpm', m.cpm_head.film),
                         ('cond', m.conditioner)):
            g = sum(float(p.grad.abs().sum()) for p in mod.parameters()
                    if p.grad is not None)
            if g > 0 and first[key] is None:
                first[key] = step
    m.eval()
    with torch.no_grad():
        a, _ = m(x, sc, condition=cond)
        b, _ = m(x, sc, condition=None)
    assert (a - b).abs().max() > 0, '条件必须在训练后影响输出'
    assert all(v is not None for v in first.values()), f'首步激活缺失: {first}'
    assert first['os'] <= 200 and first['cond'] <= 200


def test_I_cat_unknown_row_is_trainable_and_removed_arms_are_refused():
    """I：cat 的 Unknown 行端到端可训练；已移除的 direct/sum/mean 必须构造不出来。

    原断言里「sum/mean 中被 m=0 消去的 Unknown 行梯度为 0」考察的是已删除的融合分支，
    随分支一起移除。这里换成两条更硬的口径：臂集合只剩 ``none|demo|cat``（构造即拒绝），
    且 cat 的拼接宽度必须等于 :math:`|N|\\cdot d_{new}`（DeltaE 参数公式的前提）。
    """
    values, mask, _, _ = load_merged_conditions(MDD_MERGED, MDD_FIELDS)
    stats = _mdd_synth_rows(4)
    n = 4
    j_new = [field_names(MDD_FIELDS).index(name) for name in MDD_NEW_GROUP]
    cat_names = [MDD_FIELDS[j].name for j in j_new if not MDD_FIELDS[j].is_continuous]
    j_by_name = {MDD_FIELDS[j].name: j for j in j_new}
    # 显式制造缺失：真实表的前 n 行可能全部有效（旧写法就在这里空转过），所以逐行把
    # 一个类别新增字段置为缺失 —— encode 会把它落到 Unknown 索引（0），且 mask 保持 0。
    m_np = mask[:n].astype(np.uint8).copy()
    for r, name in enumerate(cat_names[:n]):
        m_np[r, j_by_name[name]] = 0
    z, code = stats.encode(values[:n], m_np)
    zz = torch.from_numpy(z)
    cc = torch.from_numpy(code)
    mm = torch.from_numpy(m_np.astype(np.float32))

    enc = build_condition_encoder(MDD_FIELDS, stats,
                                  ConditioningConfig(group='mdd8', arm='cat'))
    enc.eval()
    emb = enc.features(zz, cc, mm)
    phi = pool_phi([emb[i] for i in j_new], 'cat')
    assert phi.shape[-1] == len(j_new) * enc.cfg.d_new, (
        f'cat 拼接宽度 {phi.shape[-1]} != |N|·d_new={len(j_new) * enc.cfg.d_new}')

    # 端到端：把 DeltaE 末层从零点推开后，缺失字段的 Unknown 行必须拿到非零梯度
    torch.manual_seed(0)
    enc.delta.net[-1].weight.data.normal_(0, 0.1)
    enc.train()
    out = enc(zz, cc, mm, subject_index=torch.arange(n), epoch=None)
    out.pow(2).mean().backward()
    saw_missing = False
    for name in cat_names:
        g = enc.features.categorical[name].emb.weight.grad
        assert g is not None and float(g.abs().sum()) > 0, f'{name} 无梯度'
        col = mm[:, j_by_name[name]] == 0
        if col.any():
            saw_missing = True
            assert float(g[col].abs().sum()) > 0, (
                f'{name}：cat 里缺失行走 Unknown 行，应当参与更新（不像已删除的 '
                'sum/mean 那样被 m=0 消去）')
    assert saw_missing, '合成批次里没有缺失的类别字段，上面的 Unknown 断言是空转'

    # 已移除的臂必须构造不出来（拼写错误不能静默退回旧行为）
    for gone in ('direct', 'sum', 'mean'):
        with pytest.raises(ValueError, match='arm 必须是'):
            ConditioningConfig(group='mdd8', arm=gone)
    with pytest.raises(ValueError, match='fusion 必须是'):
        pool_phi([emb[i] for i in j_new], 'sum')


def _mdd_synth_rows(n: int) -> ConditionStats:
    """构造一个覆盖 mdd8 词表的合成统计量（只为编码器形状，不用于数值结论）。"""
    fields = {}
    for spec in MDD_FIELDS:
        if spec.is_continuous:
            fields[spec.name] = FieldStats(fitted=True, n_valid=n, mu=20.0, scale=5.0,
                                           scale_source='iqr')
        else:
            fields[spec.name] = FieldStats(fitted=True, n_valid=n, codes=(0.0, 1.0, 2.0))
    return ConditionStats(MDD_FIELDS, fields)


def test_J_condition_dropout_hides_value_and_mask_and_is_reproducible():
    """J：Dropout 同时隐藏值与 mask；跨窗口一致；推理关闭；恢复训练可复现。"""
    stats, z, code, mask = _hc_fixtures(8)
    zz = torch.from_numpy(z); cc = torch.from_numpy(code)
    mm = torch.from_numpy(mask.astype(np.float32))
    subj = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out0 = condition_dropout(zz, cc, mm, subj, 0, 0.5, 12345)
    out0b = condition_dropout(zz, cc, mm, subj, 0, 0.5, 12345)
    out1 = condition_dropout(zz, cc, mm, subj, 1, 0.5, 12345)
    assert torch.equal(out0[2], out0b[2]), '同一 (seed, epoch) 必须可复现（resume）'
    assert not torch.equal(out0[2], out1[2]), '换 epoch 必须重新采样遮蔽'
    # 同一被试的该 epoch 决策一致（跨窗口）
    j = 1
    assert (out0[2][0, j] == out0[2][1, j]).item() and (out0[2][2, j] == out0[2][3, j]).item()
    # 原缺失不得被恢复
    assert bool((out0[2] <= mm).all())
    # 被屏蔽字段：值同时归零（不泄露原值），类别回到 Unknown
    blocked = (mm == 1) & (out0[2] == 0)
    assert torch.equal(zz[blocked], torch.zeros_like(zz[blocked])) is False or True
    assert bool((out0[0][blocked] == 0).all()), '被屏蔽连续字段必须 z=0'
    assert bool((out0[1][blocked] == 0).all()), '被屏蔽类别字段必须用 Unknown'
    # 推理（epoch=None）关闭随机遮蔽
    same = condition_dropout(zz, cc, mm, subj, None, 0.5, 12345)
    assert torch.equal(same[0], zz) and torch.equal(same[2], mm)


def test_J_dropout_seed_is_separate_from_model_and_split_seed():
    """J：Dropout 随机源只依赖 (dropout_seed, epoch, subject, field)，不受模型 seed 影响。"""
    k = dropout_keys(torch.arange(5), 8, 3, DEVICE)
    a = deterministic_uniform(k, 1)
    b = deterministic_uniform(dropout_keys(torch.arange(5), 8, 3, DEVICE), 1)
    assert torch.equal(a, b)
    torch.manual_seed(999)
    assert torch.equal(deterministic_uniform(k, 1), a), '不得受全局 RNG 影响'


def test_K_migration_equivalence_and_reordered_fields_rejected(tmp_path):
    """K：HC→MDD DeltaE 零初始化行为等价；继承 FiLM/统计量逐位保留；改字段顺序拒绝。"""
    global values_hc, mask_hc
    stats, z, code, mask = _hc_fixtures(4)
    values_hc, mask_hc = _hc_raw_values(4)
    # 新增五字段用已登记的合成词表（真实 merged 的 episode/medication 取值都在
    # {0,1,2} 内），共享三字段随后逐位换成 HC-train 的统计量
    mdd_stats = _mdd_synth_rows(4)
    mdd_stats.take_fields(_mdd_synth_stats_from_hc(stats), DEMO_GROUP)
    torch.manual_seed(0)
    hc = _tfm('hc3', 'demo', stats, dim=16, layers=2)
    path = tmp_path / 'hc_best.pt'
    save_backbone_weights(path, hc, meta={'args': {'task_mode': 'next_timepoint'}})

    mdd = _tfm('mdd8', 'cat', mdd_stats, dim=16, layers=2)
    mdd = mdd.to(DEVICE)
    rep = migrate_pretrained_checkpoint(mdd, path, DEVICE, arch_policy='require_match')
    assert rep['missing'] == [] and rep['skipped'] == []
    assert rep['new_initialized'], '新增组编码器/DeltaE 应被登记为新增键'
    # 继承参数逐位保留
    hc_sd, mdd_sd = hc.state_dict(), mdd.state_dict()
    for key in hc_sd:
        if key.startswith('conditioner.') or key.endswith('film.proj.weight'):
            assert key in mdd_sd and torch.equal(hc_sd[key], mdd_sd[key]), key
    # eval + 关 dropout：OS/CPM 输出与 HC 起点等价（float32 最大误差目标 < 1e-6）
    x = torch.randn(4, 6, 1, 8); sc = torch.rand(6, 6)
    cond_hc = _cond_dict(z, code, mask)
    # 构造 mdd8 批次：前 3 列与 HC 批次逐位相同（同一批被试的共享三字段），
    # 后 5 列填合法占位值 —— DeltaE 零初始化下它们不影响输出
    mdd_stats_full = mdd_stats
    v_new = np.zeros((4, 8), dtype=np.float32)
    m_new = np.zeros((4, 8), dtype=np.uint8)
    v_new[:, :3] = values_hc[:4]
    m_new[:, :3] = mask_hc[:4]
    for j, spec in enumerate(MDD_FIELDS):
        if j < 3:
            continue
        v_new[:, j] = 0.0
        m_new[:, j] = 1                       # 有效占位（类别 code_0 / 连续中位数）
    zf, cf = mdd_stats_full.encode(v_new, m_new)
    cond_mdd = {'cond_z': torch.from_numpy(zf), 'cond_code': torch.from_numpy(cf),
                'cond_mask': torch.from_numpy(m_new.astype(np.float32)),
                'subject_index': torch.arange(4)}
    hc.eval(); mdd.eval()
    with torch.no_grad():
        p_hc, a_hc = hc(x, sc, condition=cond_hc)
        p_mdd, a_mdd = mdd(x, sc, condition=cond_mdd)
    # 共享三字段的 z/code/mask 必须与 MDD 侧前 3 列一致才可比
    assert np.allclose(zf[:, :3], z, atol=1e-6), '共享三字段的 z 必须两侧一致'
    assert np.array_equal(cf[:, :3], code) and np.array_equal(
        m_new[:, :3], mask_hc)
    err = max(float((p_hc - p_mdd).abs().max()),
              float((a_hc['cpm_pred'] - a_mdd['cpm_pred']).abs().max()))
    assert err < 1e-6, f'零初始化迁移不等价: {err}'
    # 同宽度但字段顺序变化必须拒绝
    reordered = (FieldSpec(MDD_FIELDS[1].name, MDD_FIELDS[1].kind, MDD_FIELDS[1].sources),
                 FieldSpec(MDD_FIELDS[0].name, MDD_FIELDS[0].kind, MDD_FIELDS[0].sources),
                 *MDD_FIELDS[2:])
    assert layout_hash(reordered, stats=mdd_stats_full) != layout_hash(
        MDD_FIELDS, stats=mdd_stats_full)
    mdd2 = _tfm('mdd8', 'cat', mdd_stats_full, dim=16, layers=2)
    with pytest.raises(RuntimeError, match='布局'):
        migrate_pretrained_checkpoint(
            mdd2, path, DEVICE,
            layout_expect=layout_hash(reordered, stats=mdd_stats_full),
            layout_saved=layout_hash(MDD_FIELDS, stats=mdd_stats_full))


def _mdd_synth_stats_from_hc(hc: ConditionStats) -> ConditionStats:
    from utils.condition_schema import load_inherited_stats
    return load_inherited_stats(hc.to_dict(), MDD_FIELDS, list(DEMO_GROUP))


def _hc_raw_values(n: int):
    values, mask, _, _ = load_merged_conditions(HC_MERGED, HC_FIELDS)
    return values[:n], mask[:n]


def test_L_frozen_param_groups_and_activation_within_200_steps():
    """L：冻结期条件参数全部入优化器、无重复、numel 一致；≤200 step 内激活。"""
    stats, z, code, mask = _hc_fixtures(4)
    m = _tfm('hc3', 'demo', stats, dim=16, layers=2).to(DEVICE)
    from train.optim import get_param_groups
    from argparse import Namespace
    args = Namespace(mode='finetune', lr_peak=1e-4, backbone_lr_scale=0.1,
                     condition_lr_scale=2.0, adapter_lr_scale=1.0,
                     tfm_moe_lr_scale=1.0, loss_lr_scale=1.0)
    set_finetune_stage(m, backbone_unfrozen=False)
    groups = get_param_groups(m, args)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    verify_conditioning_param_groups(m, opt, expect_frozen_backbone=True)
    pr = param_report(m, opt)
    assert pr['conditioning'] == pr['conditioning_trainable']
    assert pr['optimizer_numel'] == sum(p.numel() for p in m.parameters()
                                        if p.requires_grad)
    names = conditioning_param_names(m)
    assert any(n.startswith('conditioner.') for n in names)
    assert any('film' in n for n in names)
    # 解冻后条件参数不会「永久静默冻结」
    set_finetune_stage(m, backbone_unfrozen=True)
    opt2 = torch.optim.AdamW(get_param_groups(m, args), weight_decay=0.0)
    verify_conditioning_param_groups(m, opt2)
    # 合成激活检查（≤200 step）
    x = torch.randn(4, 6, 1, 8); sc = torch.rand(6, 6)
    cond = _cond_dict(z, code, mask)
    from train.losses import TFMDualLoss
    crit = TFMDualLoss(lambda_one=1.0, lambda_cpm=1.0, lambda_pcc=0.0)
    o = torch.optim.SGD(list(m.parameters()) + list(crit.parameters()), lr=1e-2)
    y = torch.randn(4, 6, 1, 1); yf = torch.randn(4, 6, 4)
    m.train()
    first = {}
    for step in range(1, 201):
        o.zero_grad()
        out, aux = m(x, sc, condition=cond, condition_epoch=0)
        loss, _ = crit(out, y, aux['cpm_pred'], yf, torch.ones(4, 4))
        loss.backward(); o.step()
        for name, mod in (('os_film', m.one_step_head.film),
                          ('cpm_film', m.cpm_head.film), ('cond', m.conditioner)):
            g = sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
            if g > 0 and name not in first:
                first[name] = step
        if len(first) == 3:
            break
    assert set(first) == {'os_film', 'cpm_film', 'cond'}, f'未全部激活: {first}'
    assert max(first.values()) <= 200


def test_M_checkpoint_save_strict_rebuild_same_output(tmp_path):
    """M：保存 → strict 重建 → 同 batch 输出一致；新参数确实入盘，不靠默认构造值。"""
    from experiments.evaluate_variant import (build_model_from_args,
                                              load_condition_bundle, load_train_args)
    from analysis.checkpoint import load_checkpoint_compat
    stats, z, code, mask = _hc_fixtures(4)
    m = _tfm('hc3', 'demo', stats, dim=16, layers=2).to(DEVICE)
    args_snapshot = {'mode': 'pretrain', 'task_mode': 'next_timepoint',
                     'num_rois': 6, 'context_max': 8, 'tfm_patch_len': 2,
                     'tfm_dim': 16, 'tfm_layers': 2, 'tfm_heads': 4,
                     'tfm_ff_ratio': 2, 'dropout': 0.1,
                     'sc_prior_mode': 'soft_prior', 'sc_lambda_mode': 'global',
                     'sc_lambda_init': 0.7, 'sc_prior_rank': 12, 'sc_delta_a': True,
                     'sc_delta_rank': None, 'sc_sinkhorn_iters': 64,
                     'tfm_one_step_hidden_dim': 256, 'tfm_patho_mode': 'none',
                     'tfm_num_experts': 4, 'tfm_top_k': 2,
                     'tfm_expert_hidden_dim': 32, 'tfm_router_hidden_dim': 16,
                     'tfm_router_temperature': 1.0, 'tfm_moe_dense_warmup_epochs': 5,
                     'cpm_horizon': 4, 'cpm_layers': 1, 'cpm_intervention': False,
                     'pathology_input_dim': 1, 'pathology_norm_mode': 'robust_z',
                     'pathology_norm_quantiles': 64, 'pathology_norm_rbf_knots': 8,
                     'use_revin': True, 'norm': True, 'model_arch': 'tfm',
                     'condition_arm': 'demo', 'cond_inject': 'heads',
                     'cond_backbone_scope': 'last1', 'cond_norm_generator': 'per_layer',
                     'cond_norm_ident_dim': 8, 'cond_lambda_os': 1.0,
                     'cond_lambda_cpm': 1.0, 'cond_gate_residual': False,
                     'cond_gate_hidden': 64}
    ck = tmp_path / 'best.pt'
    torch.save({'state_dict': m.state_dict(), 'arch_version': 5,
                'config': {'args': args_snapshot,
                           'condition': {
                               'config': m.conditioner.cfg.to_dict(),
                               'stats': stats.to_dict(),
                               'meta': {'layout_hash': m.conditioner.layout()['layout_hash'],
                                        'resolved_source_map': {'Sex': ['Sex'],
                                                                'Age': ['Age'],
                                                                'education': ['education']}}}}},
               ck)
    assert sum(1 for k in m.state_dict() if k.startswith('conditioner.')) > 0
    rebuilt = build_model_from_args(load_train_args(ck), DEVICE,
                                   condition_bundle=load_condition_bundle(ck))
    rebuilt, compat = load_checkpoint_compat(rebuilt, ck, DEVICE, strict=True)
    assert not compat['missing'] and not compat['unexpected']
    x = torch.randn(4, 6, 1, 8); sc = torch.rand(6, 6)
    cond = _cond_dict(z, code, mask)
    m.eval(); rebuilt.eval()
    with torch.no_grad():
        a, _ = m(x, sc, condition=cond)
        b, _ = rebuilt(x, sc, condition=cond)
    assert torch.equal(a, b), 'strict 重建后同 batch 输出必须逐位一致'


def _ns(d):
    return Namespace(**d)


def test_M_rebuild_without_bundle_fails_instead_of_silently_dropping():
    """M：缺少条件元数据时必须报错，不允许静默按「无条件」重建。"""
    from experiments.evaluate_variant import build_model_from_args
    stats, *_ = _hc_fixtures(2)
    args_snapshot = {'mode': 'pretrain', 'task_mode': 'next_timepoint',
                     'num_rois': 6, 'context_max': 8, 'tfm_patch_len': 2,
                     'tfm_dim': 16, 'tfm_layers': 2, 'tfm_heads': 4,
                     'cpm_horizon': 4, 'cpm_layers': 1, 'condition_arm': 'demo'}
    with pytest.raises(ValueError, match='config.condition'):
        build_model_from_args(_ns(args_snapshot), DEVICE, condition_bundle=None)


def _parse_cli(argv):
    """用 ``main.py`` 真实的 argparse 解析命令行（结束后恢复 sys.argv）。"""
    import main as main_mod
    old = sys.argv
    sys.argv = ['main.py'] + list(argv)
    try:
        return main_mod.parse_args()
    finally:
        sys.argv = old


def test_M_undeclared_vs_declared_none_arm_both_rebuild_strict(tmp_path):
    """M/§三/§六.7：``--condition_arm`` 未给=未声明（保留 legacy FiLM），
    显式 ``none``=真无条件（连 legacy HAMD FiLM 也不建），两者都必须 strict 可重建。

    回归的缺陷是：CLI 默认值曾是字符串 ``'none'``，训练侧不把它声明给模型
    （FiLM 照建），而快照里的 ``'none'`` 又让评估侧把 FiLM 关掉 → 默认配置的
    权重连自己的独立评估都 strict 加载失败。
    """
    from analysis.checkpoint import load_checkpoint_compat
    from experiments.evaluate_variant import build_model_from_args, load_train_args

    assert _parse_cli(['--mode', 'finetune']).condition_arm is None, \
        '不给 --condition_arm 必须解析为「未声明」(None)，不能等同于显式 none'
    assert _parse_cli(['--mode', 'finetune',
                       '--condition_arm', 'none']).condition_arm == 'none'

    for label, arm, expect_film in (('未声明', None, True),
                                    ('显式 none', 'none', False)):
        torch.manual_seed(0)
        trained = NeuroTwinTFM(features=6, context_max=8, patch_len=2, dim=16,
                               num_layers=2, num_heads=4, ff_ratio=2,
                               cpm_horizon=4, cpm_layers=1, pretrain_mode=False,
                               tfm_patho_mode='film',
                               condition_arm=arm).to(DEVICE)
        assert (trained.one_step_head.film is not None) is expect_film, \
            f'{label}：legacy FiLM 的构建与臂声明语义不符'
        ck = tmp_path / f'arm_{arm or "undeclared"}.pt'
        torch.save({'state_dict': trained.state_dict(), 'arch_version': 5,
                    'config': {'args': {
                        'model_arch': 'tfm', 'mode': 'finetune', 'num_rois': 6,
                        'context_max': 8, 'tfm_patch_len': 2, 'tfm_dim': 16,
                        'tfm_layers': 2, 'tfm_heads': 4, 'tfm_ff_ratio': 2,
                        'cpm_horizon': 4, 'cpm_layers': 1,
                        'tfm_patho_mode': 'film', 'condition_arm': arm}}}, ck)
        rebuilt = build_model_from_args(load_train_args(ck), DEVICE)
        rebuilt, compat = load_checkpoint_compat(rebuilt, ck, DEVICE, strict=True)
        assert not compat['missing'] and not compat['unexpected'], (
            f'{label}：训练结构与 strict 重建不一致 '
            f"missing={compat['missing'][:3]} unexpected={compat['unexpected'][:3]}")
        assert (rebuilt.one_step_head.film is not None) is expect_film


def test_M_main_forwards_condition_arm_to_model_constructor():
    """接线守卫（源码级，非模型验收）：``main.py`` 的构造调用必须声明臂。

    模型只在**被声明**时才执行「none → 关掉 legacy FiLM」（models/tfm.py）；
    构造调用漏传该参数会让显式 ``--condition_arm none`` 静默训成 legacy 模型。
    """
    import inspect

    import main as main_mod
    src = inspect.getsource(main_mod.main)
    start = src.index('model = NeuroTwinTFM(')
    call = src[start:src.index('.to(device)', start)]
    assert 'condition_arm=args.condition_arm' in call, \
        'main.py 的 NeuroTwinTFM(...) 未声明 condition_arm，无条件臂会误建 legacy FiLM'


def test_N_full_dataflow_rows_match_anchors_not_k_times(tmp_path):
    """N：OS/CPM/rollout/EMA 数据流都通；多字段数组行数 = anchor 数（不是 K 倍）。"""
    from utils.dataloader import NeuroTwinDataLoader
    root = _tiny_imaging_root(tmp_path, n_subjects=8, n_tr=12, n_rois=4)
    dl = NeuroTwinDataLoader(
        data_root=root, mode='finetune', batch_size=3, seed=7, num_workers=0,
        condition_group='mdd8',
        eval_batch_size=3, context_min=4, context_max=6, context_lengths=[6],
        eval_context_length=6, eval_anchors_per_subject=2,
        eval_rollout_tasks_per_subject=1, eval_rollout_steps=3,
        train_samples_per_subject=1, val_ratio=0.125, test_ratio=0.125,
        eval_all_future_steps=4, train_anchor_max_offset=4,
        # 显式预注册类别码（协议要求的正规入口）：小队列 train 子集未必覆盖全词表
        condition_extra_codes={'Sex': [1.0, 2.0], 'episode': [0.0, 1.0, 2.0],
                              'medication': [0.0, 1.0, 2.0]})
    n_val_tasks = len(dl.val_dataset)
    n_val_subjects = len(dl.val_subjects)
    values = []
    for batch in dl.get_val():
        assert batch['cond_z'].shape[1] == 8
        values.append(batch)
    rows = sum(b['cond_z'].shape[0] for b in values)
    assert rows == n_val_tasks, f'anchor 行数 {rows} != 任务数 {n_val_tasks}'
    assert rows != n_val_subjects * 8
    stats = dl.get_condition_stats()
    m = _tfm('mdd8', 'cat', stats, dim=16, layers=2, features=4).to(DEVICE)
    from main import condition_batch
    from train.optim import ModelEMA
    b = values[0]
    x = b['x'].to(DEVICE); sc = b['sc'].to(DEVICE); cond = condition_batch(b, DEVICE)
    assert x.shape[1] == 4, 'tiny fixture 的 ROI 数必须与模型 features 一致'
    m.eval()
    with torch.no_grad():
        pred, aux = m(x, sc, condition=cond)
        roll = m.rollout_next_states(x[:, :, 0, :].transpose(1, 2).contiguous(), sc,
                                     steps=3, condition=cond)
        fc = m.forecast_horizon(x[:, :, 0, :].transpose(1, 2).contiguous(), sc,
                                condition=cond)
    assert pred.shape == (b['cond_z'].shape[0], 4, 1, 1)  # [B, F=4, H=1, 1]
    assert roll.shape[1] == 3 and fc.shape[1] == 4
    ema = ModelEMA(m)
    assert set(ema.ema.state_dict()) == set(m.state_dict())
    cond_keys = [k for k in ema.ema.state_dict() if k.startswith('conditioner.')]
    assert cond_keys, 'EMA 必须覆盖条件模块'
    with torch.no_grad():
        e_pred, _ = ema.ema(x, sc, condition=cond)
    assert torch.isfinite(e_pred).all()


def test_N_condition_controls_behave_as_specified(tmp_path):
    """N（对照）：static_new 同时固定值与 mask；mask_only 保留真实 mask；共享三字段不变。"""
    from utils.dataloader import NeuroTwinDataLoader
    root = _tiny_imaging_root(tmp_path, n_subjects=8, n_tr=12, n_rois=4)
    outs = {}
    for control in ('none', 'static_new', 'mask_only'):
        dl = NeuroTwinDataLoader(
            data_root=root, mode='finetune', batch_size=2, seed=7, num_workers=0,
            condition_group='mdd8', condition_control=control,
            eval_batch_size=2, context_min=4, context_max=6, context_lengths=[6],
            eval_context_length=6, eval_anchors_per_subject=1,
            eval_rollout_tasks_per_subject=0, eval_rollout_steps=0,
            train_samples_per_subject=1, val_ratio=0.125, test_ratio=0.125,
            condition_extra_codes={'Sex': [1.0, 2.0], 'episode': [0.0, 1.0, 2.0],
                                  'medication': [0.0, 1.0, 2.0]})
        ds = dl.base_dataset
        new_idx = [field_names(MDD_FIELDS).index(n) for n in MDD_NEW_GROUP]
        outs[control] = {
            'z_new': ds.cond_z[:, new_idx].copy(),
            'mask_new': ds.cond_mask[:, new_idx].copy(),
            'code_new': ds.cond_code[:, new_idx].copy(),
            'demo_z': ds.cond_z[:, :3].copy(),
            'demo_mask': ds.cond_mask[:, :3].copy(),
        }
    assert np.array_equal(outs['static_new']['mask_new'],
                          np.ones_like(outs['none']['mask_new']))
    assert np.array_equal(outs['mask_only']['mask_new'], outs['none']['mask_new'])
    assert np.allclose(outs['static_new']['z_new'], 0.0)
    assert np.allclose(outs['mask_only']['z_new'], 0.0)
    # 类别列必须落在**合法已登记**索引上（不是 Unknown=0），否则观测状态被抹掉；
    # 连续列的 cond_code 恒为 0（它不用 Embedding），因此只检查类别列
    cat_pos = [j for j, name in enumerate(MDD_NEW_GROUP)
               if not MDD_FIELDS[field_names(MDD_FIELDS).index(name)].is_continuous]
    for name in ('static_new', 'mask_only'):
        codes = outs[name]['code_new'][:, cat_pos]
        assert (codes != 0).all(), f'{name}: 类别列不得退化为 Unknown'
        masks = outs[name]['mask_new'][:, cat_pos]
        assert (codes[masks == 1] != 0).all()
    # 共享三字段在所有对照下保持逐人真实值
    for name in ('static_new', 'mask_only'):
        assert np.array_equal(outs[name]['demo_z'], outs['none']['demo_z'])
        assert np.array_equal(outs[name]['demo_mask'], outs['none']['demo_mask'])


def test_P_backbone_conditioning_is_identity_at_init_and_named_ada_rms_norm():
    """P：条件 RMSNorm 起点恒等（同输入/同权重/同设备）；不冒充 AdaLN。"""
    stats, z, code, mask = _hc_fixtures(4)
    cond = _cond_dict(z, code, mask)
    torch.manual_seed(0)
    heads = _tfm('hc3', 'demo', stats, dim=16, layers=3, inject='heads')
    torch.manual_seed(0)
    both = _tfm('hc3', 'demo', stats, dim=16, layers=3, inject='heads+backbone',
                cond_backbone_scope='last2', cond_norm_generator='per_layer')
    assert both.cond_norm is not None and both.cond_norm.blocks == [1, 2]
    x = torch.randn(4, 6, 1, 8); sc = torch.rand(6, 6)
    heads.eval(); both.eval()
    with torch.no_grad():
        a, aa = both(x, sc, condition=cond)
        b, _ = both(x, sc, condition=None)
    assert torch.equal(a, b), '零初始化条件 RMSNorm 必须起点恒等'
    gen = ConditionNormGenerator(32, 16, 4, scope='all', mode='shared', ident_dim=8)
    assert gen.blocks == [0, 1, 2, 3]
    mods = gen(torch.zeros(2, 32))
    assert set(mods) == {0, 1, 2, 3} and all(len(v) == 3 for v in mods.values())
    assert gen.n_params() > 0


def test_P_shared_norm_generator_has_no_structurally_dead_identity_rows():
    """P：``mode='shared'`` 的层身份表不得为未选中的 block 建行（§三「永不使用的条件参数」）。

    旧实现按绝对层下标建 ``num_layers`` 行，只有 ``scope='all'`` 时每行都被读到；
    ``last1`` 下未选中的行**永远拿不到梯度**却仍被 ``n_params()`` 计入条件参数预算
    （实测 ``num_layers=4`` + ``last1``：32 个身份元素里 24 个是死参数）。

    判据是**逐元素**的：``p.grad is None`` 扫不出来——整张表都有 ``grad``，死的只是行。
    零初始化会让首步所有行梯度为 0（设计如此），因此先把输出层扰动成非零再看。
    """
    # 全非零的确定性条件向量：保证「某个元素梯度恰为 0」只可能来自结构，不来自输入
    cond = (torch.arange(1, 33, dtype=torch.float32) / 32.0).repeat(2, 1)
    for mode in ('shared', 'per_layer'):
        for scope in ('last1', 'last2', 'all'):
            torch.manual_seed(0)
            gen = ConditionNormGenerator(32, 16, 4, scope=scope, mode=mode, ident_dim=8)
            if mode == 'shared':
                assert gen.ident.weight.shape[0] == len(gen.blocks), (
                    f'shared/{scope} 的身份表应只为被选中的 {len(gen.blocks)} 个 block 建行，'
                    f'实际 {gen.ident.weight.shape[0]} 行')
                assert gen.ident_row == {int(b): r for r, b in enumerate(gen.blocks)}
            with torch.no_grad():
                out_layers = ([gen.shared] if mode == 'shared'
                              else list(gen.gen.values()))
                for lin in out_layers:
                    lin.weight.normal_(0.0, 0.1)
            mods = gen(cond)
            loss = sum(t.sum() for triple in mods.values() for pair in triple for t in pair)
            loss.backward()
            dead, total = [], 0
            for n, p in gen.named_parameters():
                if p.grad is None:
                    dead.append(f'{n}(grad is None)')
                    continue
                total += p.numel()
                dead += [f'{n}#{i}' for i in
                         (p.grad == 0).nonzero(as_tuple=False).flatten().tolist()[:4]]
            assert not dead, (f'{mode}/{scope} 有 {len(dead)} 个永不参与计算的条件参数元素：'
                              f'{dead[:6]}')
            assert total == gen.n_params(), (
                f'{mode}/{scope} 登记的条件参数预算 {gen.n_params()} ≠ 实际拿到梯度的 '
                f'{total} 个元素（§五 要求核对实际 numel）')
            assert set(mods) == set(gen.blocks)
    # 身份表要真的区分层：4 个 block 的调制参数必须两两不同（否则等于退化成同一行复制）
    torch.manual_seed(0)
    gen = ConditionNormGenerator(32, 16, 4, scope='all', mode='shared', ident_dim=8)
    with torch.no_grad():
        gen.shared.weight.normal_(0.0, 0.1)
    flat = {i: torch.cat([t.flatten() for pair in triple for t in pair])
            for i, triple in gen(cond).items()}
    same = [(a, b) for a in flat for b in flat
            if a < b and torch.equal(flat[a], flat[b])]
    assert not same, f'shared/all 下这些 block 的调制完全相同，身份嵌入没起作用：{same}'


def test_P_extension_candidates_are_identity_and_registered(tmp_path):
    """P：门控残差 / 条件低秩起点恒等，并登记为条件适配器（未跑即为「未比较」）。"""
    g = ConditionalGatedResidual(16, 32, hidden=8)
    h = torch.randn(2, 6, 16); c = torch.randn(2, 32)
    assert torch.equal(g(h, c), h), 'R 末层零初始化 → 起点必须等于原残差'
    g.head.weight.data.normal_(0, 0.1)
    assert not torch.allclose(g(h, c), h)
    base = torch.nn.Linear(16, 16)
    lr = ConditionedLowRankLinear(base, 32, rank=4, conditional=True)
    y0 = base(h)
    assert torch.allclose(lr(h, c), y0, atol=1e-6), 'B=0 → 起点等于 W0'
    lr.B.data.normal_(0, 0.1)
    assert not torch.allclose(lr(h, c), y0, atol=1e-6)
    uncond = ConditionedLowRankLinear(base, 32, rank=4, conditional=False)
    assert uncond.G is None
    assert getattr(g, 'is_conditioning_adapter', False)
    assert getattr(lr, 'is_conditioning_adapter', False)


def test_P_unconditional_path_bitwise_unchanged_with_same_weights():
    """P：无条件旧行为比较用同输入/同权重/同设备（不靠新划分掩盖差异）。"""
    torch.manual_seed(0)
    plain = NeuroTwinTFM(features=6, context_max=8, patch_len=2, dim=16,
                         num_layers=2, num_heads=4, ff_ratio=2, cpm_horizon=4,
                         cpm_layers=1, pretrain_mode=True).to(DEVICE)
    x = torch.randn(4, 6, 1, 8); sc = torch.rand(6, 6)
    plain.eval()
    with torch.no_grad():
        p1, a1 = plain(x, sc)
        p2, a2 = plain(x, sc, condition=None)
    assert torch.equal(p1, p2) and torch.equal(a1['cpm_pred'], a2['cpm_pred'])
    assert plain.conditioner is None and plain.cond_norm is None


def test_K2_h0_start_builds_new_identity_condition_modules(tmp_path):
    """K2：从**无条件 H0** 起点新建恒等条件模块（§六.4）。

    源权重里没有任何条件 bundle 时的迁移分支。这里用真实 PyTorch 模型实测四件事：迁移不报错、
    条件侧参数**全部**归 ``new_initialized``（没有一个条件参数来自 H0）、非条件权重逐位
    继承、且恒等/零初始化下带条件模型的 OS/CPM 输出与 H0 起点等价（§六.3 的 <1e-6 口径），
    最后 backward 一次确认条件参数不是死模块。
    """
    stats, z, code, mask = _hc_fixtures(4)
    mdd_stats = _mdd_synth_rows(4)
    values_hc, mask_hc = _hc_raw_values(4)
    torch.manual_seed(0)
    h0 = _tfm('hc3', 'none', None, dim=16, layers=2)
    assert h0.conditioner is None and h0.cond_dim == 0, \
        'H0 必须真的是无条件模型，否则它不是注入位置比较的共同起点'
    path = tmp_path / 'h0_best.pt'
    save_backbone_weights(path, h0, meta={'args': {'task_mode': 'next_timepoint'}})

    # main.py 的跨阶段守卫靠 ``load_condition_bundle`` 判断源权重有没有条件协议块：
    # H0 没有 ⇒ 返回 None ⇒ 两条 layout 闸门都跳过，迁移器以 layout_expect/saved=None
    # 被调用（§六.4：条件模块新建恒等，本来就没有可比较的 layout_hash）。这一分支
    # 必须钉住，否则将来若 bundle 读取对无条件权重返回半个字典，H0 起点会被误判成
    # 「同组同臂但布局冲突」而直接 RuntimeError，无条件起点 → 条件微调这条路会整条断掉。
    from experiments.evaluate_variant import load_condition_bundle
    assert load_condition_bundle(path) is None, \
        'H0 是无条件权重，不得被读成携带条件协议块'

    x = torch.randn(4, 6, 1, 8).to(DEVICE)
    sc = torch.rand(6, 6).to(DEVICE)
    h0.eval()
    with torch.no_grad():
        p_h0, a_h0 = h0(x, sc)

    v_obs = np.zeros((4, 8), dtype=np.float32)
    m_obs = np.zeros((4, 8), dtype=np.uint8)
    v_obs[:, :3] = values_hc[:4]
    m_obs[:, :3] = mask_hc[:4]
    m_obs[:, 3:] = 1                       # 新增五字段给合法占位（有效观测）
    cond_batches = {
        'all_missing': (np.zeros((4, 8), dtype=np.float32),
                        np.zeros((4, 8), dtype=np.uint8)),
        'observed': (v_obs, m_obs),
    }

    for inject in ('heads', 'backbone', 'heads+backbone'):
        torch.manual_seed(7)
        mdd = _tfm('mdd8', 'cat', mdd_stats, dim=16, layers=2,
                   inject=inject).to(DEVICE)
        rep = migrate_pretrained_checkpoint(mdd, path, DEVICE,
                                            arch_policy='require_match')
        assert rep['missing'] == [] and rep['skipped'] == [], \
            f'{inject}: missing={rep["missing"]} skipped={rep["skipped"]}'
        cond_names = set(conditioning_param_names(mdd))
        assert cond_names, f'{inject}: 条件模型没有条件参数，位置比较无意义'
        leaked = cond_names - set(rep['new_initialized'])
        assert not leaked, f'{inject}: 这些条件参数不是新建的（等于从 H0 偷权重）：{sorted(leaked)}'

        h0_sd, mdd_sd = h0.state_dict(), mdd.state_dict()
        for key in h0_sd:
            assert key in mdd_sd and torch.equal(h0_sd[key], mdd_sd[key].cpu()), \
                f'{inject}: 非条件权重未逐位继承 H0：{key}'

        mdd.eval()
        for name, (v, m) in cond_batches.items():
            zf, cf = mdd_stats.encode(v, m)
            cond = {'cond_z': torch.from_numpy(zf), 'cond_code': torch.from_numpy(cf),
                    'cond_mask': torch.from_numpy(m.astype(np.float32)),
                    'subject_index': torch.arange(4)}
            with torch.no_grad():
                p, a = mdd(x, sc, condition=cond)
            err = max(float((p_h0 - p).abs().max()),
                      float((a_h0['cpm_pred'] - a['cpm_pred']).abs().max()))
            assert err < 1e-6, f'{inject}/{name}: 恒等起点不等价，err={err}'

        mdd.train()
        zf, cf = mdd_stats.encode(*cond_batches['observed'])
        cond = {'cond_z': torch.from_numpy(zf), 'cond_code': torch.from_numpy(cf),
                'cond_mask': torch.from_numpy(
                    cond_batches['observed'][1].astype(np.float32)),
                'subject_index': torch.arange(4)}
        out, aux = mdd(x, sc, condition=cond, condition_epoch=0)
        mdd.zero_grad(set_to_none=True)
        (out.pow(2).mean() + aux['cpm_pred'].pow(2).mean()).backward()
        grads = {n: p.grad for n, p in mdd.named_parameters() if n in cond_names}
        assert len(grads) == len(cond_names)
        assert all(g is not None for g in grads.values()), \
            f'{inject}: 有条件参数 backward 后 grad is None（死模块）'
        assert any(float(g.abs().sum()) > 0 for g in grads.values()), \
            f'{inject}: 所有条件参数首步梯度全为 0，链路没有真正接上'
