"""Field parsing, missingness, train-only statistics and Raw BOLD cohort contracts."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from utils.condition_schema import (
    DEMO_GROUP,
    HC_FIELDS,
    MDD_FIELDS,
    MDD_NEW_GROUP,
    ConditionStats,
    FieldSpec,
    MedicationConflict,
    audit_unseen_codes,
    field_names,
    fit_condition_stats,
    load_merged_conditions,
    parse_cell,
    parse_extra_codes,
    remap_code,
    resolve_source_columns,
)

DEVICE = torch.device("cpu")
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
HC_MERGED = DATA_ROOT / "Rest-meta-MDD-V1V2-Merged-HC.xlsx"
MDD_MERGED = DATA_ROOT / "Rest-meta-MDD-V1V2-Merged-MDD.xlsx"


def _write_merged(path: Path, rows: dict) -> None:
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)


EPISODE_ONLY = (FieldSpec("episode", "categorical", ("episode",)),)
MED_TWO_COL = (
    FieldSpec(
        "medication", "categorical", ("medication", "On medication?", "OnMedication")
    ),
)


def _stats_for(specs, values, mask, train_rows, **kw) -> ConditionStats:
    return fit_condition_stats(specs, values, mask, np.asarray(train_rows), **kw)


def test_A_sentinels_are_valid_and_remap_is_idempotent(tmp_path):
    """A：[-9999,-1,0,1,2] 全部 mask=1，映射到 [0,2,0,1,2]；重复映射不变。"""
    p = tmp_path / "merged.xlsx"
    _write_merged(
        p, {"ID": ["s0", "s1", "s2", "s3", "s4"], "episode": [-9999, -1, 0, 1, 2]}
    )
    values, mask, ids, audit = load_merged_conditions(p, EPISODE_ONLY)
    col = 0
    assert mask[:, col].tolist() == [1, 1, 1, 1, 1], "所有有限数值都必须有效"
    assert values[:, col].tolist() == [0.0, 2.0, 0.0, 1.0, 2.0]
    again = np.array([remap_code(v) for v in values[:, col]])
    assert again.tolist() == values[:, col].tolist()
    assert remap_code(-1.0) == 2.0 and remap_code(-9999.0) == 0.0
    assert audit.n_valid["episode"] == 5


def test_A_remap_only_applies_to_episode_and_medication(tmp_path):
    """编码替换只作用于 episode/medication；其他字段数值原样保留。"""
    p = tmp_path / "merged.xlsx"
    _write_merged(
        p,
        {
            "ID": ["a"],
            "Sex": [1],
            "Age": [-1],
            "education": [-9999],
            "episode": [-1],
            "illness": [-1],
            "medication": [-9999],
            "HAMD": [-1],
            "HAMA": [-9999],
        },
    )
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    names = field_names(MDD_FIELDS)
    got = {n: float(values[0, names.index(n)]) for n in names}
    assert got["Age"] == -1.0 and got["education"] == -9999.0
    assert got["illness"] == -1.0 and got["HAMD"] == -1.0 and (got["HAMA"] == -9999.0)
    assert got["episode"] == 2.0 and got["medication"] == 0.0
    assert mask[0].tolist() == [1] * len(names)


def test_B_missing_variants_all_mask_zero_and_numeric_strings_valid():
    """B：None/NaN/空串/纯空白/null 大小写变体 → mask=0；数字字符串有效。"""
    for raw, expect_valid in [
        (None, False),
        (float("nan"), False),
        ("", False),
        ("   ", False),
        ("\t\n", False),
        ("null", False),
        ("NULL", False),
        ("  NaT  ", False),
        ("n/a", False),
        ("<NA>", False),
        ("nan", False),
        ("none", False),
        (pd.NaT, False),
        (float("inf"), False),
        ("abc", False),
        ("-1", True),
        ("0", True),
        (0, True),
        (-1, True),
        (-9999, True),
        ("2.5", True),
    ]:
        val, reason = parse_cell(raw)
        assert (val is not None) == expect_valid, f"{raw!r} → {val!r} ({reason})"
    assert parse_cell(0)[0] == 0.0 and parse_cell("0")[0] == 0.0


def test_B_missing_rows_yield_finite_encoded_input(tmp_path):
    """B：缺失格经 train-only 填补/编码后，模型输入全部有限。"""
    p = tmp_path / "merged.xlsx"
    _write_merged(
        p,
        {
            "ID": [f"s{i}" for i in range(6)],
            "Sex": [1, None, 2, "null", 1, 1],
            "Age": [30, "", 32, "  ", 34, "nan"],
            "education": [12, 14, "NA", 16, 10, 11],
        },
    )
    values, mask, _, _ = load_merged_conditions(p, HC_FIELDS)
    stats = _stats_for(HC_FIELDS, values, mask, [0, 2, 4, 5])
    z, code = stats.encode(values, mask)
    assert np.isfinite(z).all(), "编码后不得出现 NaN/Inf"
    unk = ~mask[:, 0].astype(bool)
    assert code[unk, 0].tolist() == [0] * int(unk.sum()), "类别缺失必须落 Unknown 索引"
    j = field_names(HC_FIELDS).index("Age")
    med = stats.get("Age").mu
    for i in range(6):
        if not mask[i, j]:
            assert z[i, j] == 0.0
        elif values[i, j] == med:
            assert z[i, j] == 0.0 and mask[i, j] == 1


def test_B_unknown_and_code_zero_are_different_indices(tmp_path):
    """B：Unknown 内部索引与有效 code_0 不同（不能只用「值==0」重建 mask）。"""
    p = tmp_path / "merged.xlsx"
    _write_merged(p, {"ID": ["a", "b", "c"], "episode": [0, None, -9999]})
    values, mask, _, _ = load_merged_conditions(p, EPISODE_ONLY)
    stats = _stats_for(EPISODE_ONLY, values, mask, [0, 1, 2])
    z, code = stats.encode(values, mask)
    j = 0
    assert mask[0, j] == 1 and mask[2, j] == 1 and (mask[1, j] == 0)
    assert code[0, j] == code[2, j] != 0
    assert code[1, j] == 0


@pytest.mark.skipif(not MDD_MERGED.exists(), reason="真实 merged 文件不在位")
def test_C_only_merged_is_read_and_item_columns_are_ignored(tmp_path):
    """C：改动/移走非 merged 原始表不影响条件加载；HAMA 条目与 HAMATotal 改值不影响 HAMA。"""
    import shutil

    root = tmp_path / "data"
    root.mkdir()
    merged = root / MDD_MERGED.name
    shutil.copy(MDD_MERGED, merged)
    decoy = root / "Rest-meta-MDD-V1-MDD.xlsx"
    _write_merged(decoy, {"ID": ["x"], "HAMA": [999], "medication": [7]})
    before = load_merged_conditions(merged, MDD_FIELDS)
    decoy.unlink()
    (root / "Rest-meta-MDD-V2-MDD.xlsx").write_bytes(b"not-an-excel")
    after = load_merged_conditions(merged, MDD_FIELDS)
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(
        np.nan_to_num(before[0], nan=-7.0), np.nan_to_num(after[0], nan=-7.0)
    )
    df = pd.read_excel(merged)
    names = field_names(MDD_FIELDS)
    j_hama = names.index("HAMA")
    base_values, base_mask, _, _ = load_merged_conditions(merged, MDD_FIELDS)
    for col in [
        c
        for c in df.columns
        if str(c).strip().upper().startswith(("HAMA", "*HAMA"))
        and str(c).strip().upper() != "HAMA"
    ][:12]:
        df[col] = 12345
    mutated = tmp_path / "mutated.xlsx"
    df.to_excel(mutated, index=False)
    mv, mm, _, _ = load_merged_conditions(mutated, MDD_FIELDS)
    assert np.allclose(
        np.nan_to_num(mv[:, j_hama], nan=-1.0),
        np.nan_to_num(base_values[:, j_hama], nan=-1.0),
    ), f"{col} 被读取了"
    assert np.array_equal(mm[:, j_hama], base_mask[:, j_hama])
    if "HAMATotal" in df.columns:
        assert not set(df.columns) < set(field_names(MDD_FIELDS))


@pytest.mark.skipif(not MDD_MERGED.exists(), reason="真实 merged 文件不在位")
def test_C_hama_never_falls_back_to_hamatotal(tmp_path):
    """C：HAMA 缺失不因 HAMATotal 有值而被填补（禁止 HAMATotal 回退）。"""
    import shutil

    merged = tmp_path / MDD_MERGED.name
    shutil.copy(MDD_MERGED, merged)
    df = pd.read_excel(merged)
    n = len(df)
    df["HAMA"] = np.nan
    if "HAMATotal" in df.columns:
        df["HAMATotal"] = np.arange(n, dtype=float)
    alt = tmp_path / "alt.xlsx"
    df.to_excel(alt, index=False)
    _, mask, _, _ = load_merged_conditions(alt, MDD_FIELDS)
    j = field_names(MDD_FIELDS).index("HAMA")
    assert int(mask[:, j].sum()) == 0, "HAMATotal 不得回退成 HAMA"


def _cycle(vals, n):
    """按长度 n 截断的循环列，保证所有列等长（DataFrame 构造的前置条件）。"""
    return [vals[i % len(vals)] for i in range(n)]


def _mdd_base(n):
    """完整的 mdd8 合成表（所有注册列都存在，缺失由具体测试再覆写）。"""
    return {
        "ID": [f"s{i}" for i in range(n)],
        "Sex": _cycle([1, 2], n),
        "Age": list(range(20, 20 + n)),
        "education": _cycle([12, 14, 16], n),
        "episode": _cycle([0, 1, 2], n),
        "illness": _cycle([10.0, 20.0, 30.0], n),
        "HAMD": _cycle([20, 22, 24], n),
        "HAMA": _cycle([10, 12, 14], n),
        "medication": _cycle([0, 1, None], n),
        "OnMedication": _cycle([None], n),
    }


def test_D_medication_two_columns_union_without_source_identity(tmp_path):
    """D：同一行取两列可解析数值；不产生「来自哪一列」的特征（K 仍是 8）。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(4)
    rows["medication"] = [1, None, None, None]
    rows["OnMedication"] = [None, 0, -1, None]
    _write_merged(p, rows)
    values, mask, _, audit = load_merged_conditions(p, MDD_FIELDS)
    assert values.shape[1] == 8, "两个用药列必须映射为一个条件字段"
    j = field_names(MDD_FIELDS).index("medication")
    assert mask[:, j].tolist() == [1, 1, 1, 0]
    assert values[:, j][:3].tolist() == [1.0, 0.0, 2.0]
    assert np.isnan(values[3, j])
    assert len(audit.alias_sources["medication"]) == 2, "medication 由同表两列共同来源"


def test_D_medication_conflict_raises_after_mapping(tmp_path):
    """D：两列同时有值且映射后不一致 → 报冲突并停止该配置。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(3)
    rows["medication"] = [-1, 1, None]
    rows["OnMedication"] = [1, 1, None]
    _write_merged(p, rows)
    with pytest.raises(MedicationConflict):
        load_merged_conditions(p, MDD_FIELDS)


def test_D_medication_conflict_after_remap_is_allowed(tmp_path):
    """D：映射后一致即视为同一类别（-1→2 与已有 2 共存不算冲突）。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(2)
    rows["medication"] = [-1, 2]
    rows["OnMedication"] = [2, -1]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    j = field_names(MDD_FIELDS).index("medication")
    assert values[:, j].tolist() == [2.0, 2.0] and mask[:, j].tolist() == [1, 1]


def test_D_ambiguous_columns_raise():
    """D：大小写/空白可规范匹配，但同时存在多个歧义列时报错。"""
    df = pd.DataFrame({"ID": ["a"], "age": [1], "Age": [2]})
    with pytest.raises(ValueError, match="歧义"):
        resolve_source_columns(df, [FieldSpec("Age", "continuous", ("Age",))])


def test_D_no_registered_column_raises_never_other_file(tmp_path):
    """D：merged 内找不到注册列时报错，绝不回退到其他文件/其他列。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(2)
    del rows["episode"]
    rows["episode_but_renamed"] = [1, 2]
    _write_merged(p, rows)
    with pytest.raises(ValueError, match="找不到字段 episode"):
        load_merged_conditions(p, MDD_FIELDS)


def test_E_val_test_tampering_does_not_move_train_stats(tmp_path):
    """E：篡改 val/test 数值不改变 train 统计量（fit 只看 train 行）。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(8)
    assert len({len(v) for v in rows.values()}) == 1
    rows["HAMA"] = [10, 11, 12, 13, None, None, None, None]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    stats = _stats_for(MDD_FIELDS, values, mask, [0, 1, 2, 3])
    before = (stats.get("HAMA").mu, stats.get("HAMA").scale, stats.get("HAMA").n_valid)
    v2 = values.copy()
    v2[4:, field_names(MDD_FIELDS).index("HAMA")] = [999, 998, 997, 996]
    v2[0] = 1000000.0
    stats2 = _stats_for(MDD_FIELDS, v2, mask, [0, 1, 2, 3])
    assert (stats2.get("HAMA").mu, stats2.get("HAMA").scale) != before[:2]
    v3 = values.copy()
    v3[4:, field_names(MDD_FIELDS).index("HAMA")] = [999, 998, 997, 996]
    stats3 = _stats_for(MDD_FIELDS, v3, mask, [0, 1, 2, 3])
    assert (
        stats3.get("HAMA").mu,
        stats3.get("HAMA").scale,
        stats3.get("HAMA").n_valid,
    ) == before


def test_E_continuous_all_missing_is_safe_and_not_fit_from_val(tmp_path):
    """E：全训练字段缺失 → fitted=false、n_valid=0，明确全缺失分支，不从 val/test 拟合。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(5)
    rows["illness"] = [None, None, 55, 66, 77]
    rows["episode"] = [1] * 5
    rows["Sex"] = [1] * 5
    rows["medication"] = [0] * 5
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    stats = _stats_for(MDD_FIELDS, values, mask, [0, 1])
    st = stats.get("illness")
    assert st.fitted is False and st.n_valid == 0
    assert st.scale_source == "none"
    z, _code = stats.encode(values, mask)
    j = field_names(MDD_FIELDS).index("illness")
    assert np.isfinite(z).all() and np.allclose(z[:, j], 0.0)


def test_E_hc_shared_stats_are_not_overwritten_by_mdd_fit(tmp_path):
    """E：HC 的 Age/education 统计量继承到 mdd8，通用 fit 不得覆盖（方案 §4）。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(6)
    rows["Age"] = [30, 31, 32, 33, 34, 35]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    hc = _stats_for(HC_FIELDS, values, mask, [0, 1, 2])
    mdd = _stats_for(MDD_FIELDS, values, mask, [3, 4, 5])
    mdd.take_fields(hc_for_mdd(hc), DEMO_GROUP)
    assert mdd.get("Age").mu == hc.get("Age").mu
    assert mdd.get("Age").scale == hc.get("Age").scale
    assert mdd.get("Sex").codes == hc.get("Sex").codes
    before = {
        n: (mdd.get(n).mu, mdd.get(n).scale, mdd.get(n).codes, mdd.get(n).n_valid)
        for n in DEMO_GROUP
    }
    mdd.fit(values, mask, np.array([3, 4, 5]), only_fields=MDD_NEW_GROUP)
    after = {
        n: (mdd.get(n).mu, mdd.get(n).scale, mdd.get(n).codes, mdd.get(n).n_valid)
        for n in DEMO_GROUP
    }
    assert before == after, "通用 fit 不得覆盖继承的 Age/education 统计量"
    naive = _stats_for(MDD_FIELDS, values, mask, [0, 1, 2, 3, 4, 5])
    assert (naive.get("Age").mu, naive.get("Age").n_valid) != (
        before["Age"][0],
        before["Age"][3],
    )


def hc_for_mdd(hc_stats: ConditionStats) -> ConditionStats:
    """把 hc3 统计量装入 mdd8 schema 的共享字段（模拟 --condition_inherit_stats）。"""
    from utils.condition_schema import load_inherited_stats

    return load_inherited_stats(hc_stats.to_dict(), MDD_FIELDS, list(DEMO_GROUP))


def test_E_unregistered_valid_code_is_reported_not_masked(tmp_path):
    """E：train 外出现未登记有效类别码 → 显式报告，不伪装缺失、不静默扩表。"""
    p = tmp_path / "merged.xlsx"
    rows = _mdd_base(4)
    rows["episode"] = [1, 1, 5, 5]
    _write_merged(p, rows)
    values, mask, _, _ = load_merged_conditions(p, MDD_FIELDS)
    stats = _stats_for(MDD_FIELDS, values, mask, [0, 1])
    assert stats.get("episode").codes == (1.0,)
    unseen = audit_unseen_codes(MDD_FIELDS, stats, values, mask, np.array([0, 1]))
    assert unseen.get("episode") == [5.0]
    with pytest.raises(ValueError, match="未登记"):
        stats.encode(values, mask)
    stats2 = _stats_for(
        MDD_FIELDS, values, mask, [0, 1], extra_codes={"episode": [5.0]}
    )
    assert 5.0 in stats2.get("episode").codes
    _z, code = stats2.encode(values, mask)
    j = field_names(MDD_FIELDS).index("episode")
    assert code[2, j] != 0, "登记后必须占用有效索引，不能落到 Unknown"


def test_E_parse_extra_codes():
    assert parse_extra_codes("Sex:0,3;medication:2") == {
        "Sex": [0.0, 3.0],
        "medication": [2.0],
    }
    assert parse_extra_codes("") == {}
    with pytest.raises(ValueError):
        parse_extra_codes("Sex")


def test_E_extra_codes_reach_the_inherit_path(tmp_path):
    """E（继承路径）：``--condition_extra_codes`` 必须真的进到 HC→MDD 继承后的拟合。

    曾经 ``_fit_condition_stats`` 的继承分支只传 ``only_fields=MDD_NEW_GROUP`` 而漏掉
    ``extra_codes``，于是新增五字段的预注册码被静默丢弃；而下方 audit 的报错文案又
    建议「传 --condition_extra_codes」，用户照做仍然报同一个错。同时，共享三字段
    （Sex/education）的词表由本轮 HC 冻结继承，MDD 侧传码必须**显式拒绝**而不是空转。
    """
    from utils.dataloader import NeuroTwinDataLoader

    root = _tiny_imaging_root(tmp_path, n_subjects=8, n_tr=12, n_rois=4)
    common = dict(
        batch_size=2,
        seed=7,
        num_workers=0,
        eval_batch_size=2,
        context_min=4,
        context_max=6,
        context_lengths=[6],
        num_rois=4,
        calibration_length=4,
        eval_context_length=6,
        eval_anchors_per_subject=1,
        eval_rollout_tasks_per_subject=0,
        eval_rollout_steps=0,
        train_samples_per_subject=1,
        val_ratio=0.25,
        test_ratio=0.25,
    )
    hc_dl = NeuroTwinDataLoader(
        data_root=root,
        mode="pretrain",
        condition_group="hc3",
        condition_extra_codes={"Sex": [1.0, 2.0]},
        **common,
    )
    base = hc_for_mdd(hc_dl.get_condition_stats())
    dl = NeuroTwinDataLoader(
        data_root=root,
        mode="finetune",
        condition_group="mdd8",
        condition_base_stats=base,
        condition_extra_codes={"episode": [7.0], "medication": [0.0, 1.0, 2.0]},
        **common,
    )
    stats = dl.get_condition_stats()
    assert 7.0 in stats.get("episode").codes, "继承路径丢掉了 extra_codes"
    assert 2.0 in stats.get("medication").codes
    for name in DEMO_GROUP:
        got, src = (stats.get(name), base.get(name))
        assert (got.fitted, got.mu, got.scale, got.codes) == (
            src.fitted,
            src.mu,
            src.scale,
            src.codes,
        )
    assert 7.0 not in stats.get("Sex").codes
    with pytest.raises(ValueError, match="继承字段"):
        NeuroTwinDataLoader(
            data_root=root,
            mode="finetune",
            condition_group="mdd8",
            condition_base_stats=base,
            condition_extra_codes={"Sex": [7.0]},
            **common,
        )


@pytest.mark.skipif(not HC_MERGED.exists(), reason="真实 merged 文件不在位")
def test_F_hc_cohort_keeps_subjects_without_metadata_row():
    """F：merged 无整行的 HC 被试保留为全条件缺失（有效观测数与方案 §1 一致）。"""
    values, mask, ids, audit = load_merged_conditions(HC_MERGED, HC_FIELDS)
    assert audit.n_rows == 2445
    names = field_names(HC_FIELDS)
    assert int(mask[:, names.index("Sex")].sum()) == 2445
    assert int(mask[:, names.index("education")].sum()) == 2440
    full_missing = (mask == 0).all(axis=1)
    assert int(values[full_missing].size) == int(full_missing.sum()) * len(HC_FIELDS)


def test_F_all_arms_share_cohort_and_split(tmp_path):
    """F：所有对照（含无条件臂）使用同一 cohort 与同一 split（验收 F）。"""
    from utils.dataloader import NeuroTwinDataLoader

    root = _tiny_imaging_root(tmp_path)
    merged = root / "Rest-meta-MDD-V1V2-Merged-HC.xlsx"
    hashes = {}
    for arm in ("none", "demo"):
        dl = NeuroTwinDataLoader(
            data_root=root,
            mode="pretrain",
            batch_size=2,
            seed=7,
            num_workers=0,
            condition_group="hc3" if arm != "none" else None,
            condition_file=merged.name,
            eval_batch_size=2,
            context_min=4,
            context_max=6,
            context_lengths=[6],
            num_rois=4,
            calibration_length=4,
            eval_context_length=6,
            eval_anchors_per_subject=2,
            eval_rollout_tasks_per_subject=0,
            eval_rollout_steps=0,
            val_ratio=0.25,
            test_ratio=0.25,
            train_samples_per_subject=1,
            condition_extra_codes={"Sex": [1.0, 2.0]},
        )
        _ = dl.base_dataset.condition_table
        key = (
            tuple(dl.train_subjects),
            tuple(dl.val_subjects),
            tuple(dl.test_subjects),
        )
        hashes[arm] = (len(dl.base_dataset.subject_ids), key)
    assert hashes["none"] == hashes["demo"], "条件臂与无条件臂的队列/划分必须一致"


def _tiny_imaging_root(
    tmp_path: Path, n_subjects: int = 6, n_tr: int = 12, n_rois: int = 4
) -> Path:
    """造一个最小可读的影像数据目录（Raw + merged 条件表）。"""
    import scipy.io as sio

    root = tmp_path / "data"
    for group in ("HC", "MDD"):
        (root / "Raw" / group).mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    ids = []
    for i in range(n_subjects):
        sid = f"S{int(i):03d}"
        ids.append(sid)
        series = rng.randn(n_tr, n_rois).astype(np.float32)
        sio.savemat(
            str(root / "Raw" / "HC" / f"ROISignals_{sid}.mat"), {"ROISignals": series}
        )
        sio.savemat(
            str(root / "Raw" / "MDD" / f"ROISignals_{sid}.mat"), {"ROISignals": series}
        )
    rows = {
        "ID": ids,
        "Sex": _cycle([1, 2], n_subjects),
        "Age": [20 + i for i in range(n_subjects)],
        "education": _cycle([12, 14, 16], n_subjects),
    }
    _write_merged(root / "Rest-meta-MDD-V1V2-Merged-HC.xlsx", rows)
    mdd = dict(rows)
    mdd.update(
        {
            "episode": _cycle([1, 0], n_subjects),
            "illness": [10.0 + 10 * i for i in range(n_subjects)],
            "medication": _cycle([1, 0, None], n_subjects),
            "HAMD": [15 + i for i in range(n_subjects)],
            "HAMA": _cycle([10, None, 12], n_subjects),
        }
    )
    _write_merged(root / "Rest-meta-MDD-V1V2-Merged-MDD.xlsx", mdd)
    return root
