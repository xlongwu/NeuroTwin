# coding=utf-8
"""graph_ode 多步 checkpoint 选优指标的轻量验证（小样本、可手算、不依赖数据/GPU）。

覆盖：完整 H 步锚点筛选、被试等权聚合、batch 划分不变性、无有效目标报错、
以及「单步更优但多步更差」时的选优与 early stopping 行为。
"""
from types import SimpleNamespace

import pytest
import torch

from main import (DYNAMICS_CKPT_HORIZON, CheckpointSelector,
                  aggregate_dynamics_subject_metric, dynamics_anchor_mae,
                  dynamics_metric_key, dynamics_metric_name,
                  resolve_dynamics_metric_horizon)

H = DYNAMICS_CKPT_HORIZON


# ────────────────────────────── 键名与 horizon 解析 ──────────────────────────────

def test_metric_key_follows_horizon():
    assert dynamics_metric_key(H) == 'metric_mae_dynamics_h1_8_subject'
    assert dynamics_metric_key(4) == 'metric_mae_dynamics_h1_4_subject'
    with pytest.raises(ValueError):
        dynamics_metric_key(0)


def test_resolve_horizon_prefers_cli_then_model_then_default():
    assert resolve_dynamics_metric_horizon(
        SimpleNamespace(dynamics_metric_horizon=6, cpm_horizon=8)) == 6
    # CLI 未给出（0/缺失）时回落到模型自身的预测范围，而不是直接写死 8
    assert resolve_dynamics_metric_horizon(
        SimpleNamespace(dynamics_metric_horizon=0, cpm_horizon=4)) == 4
    assert resolve_dynamics_metric_horizon(SimpleNamespace()) == H


# ────────────────────────────── 锚点筛选（可手算）──────────────────────────────

def test_only_full_horizon_anchors_are_selected():
    """3 个锚点分别有 8/7/8 步真值：第 2 个必须以不同预测范围被剔除。"""
    pred = torch.zeros(3, 2, H)
    future = torch.zeros(3, 2, H)
    future[0, :, :] = 1.0                      # 逐元素误差 1 → 锚点 MAE = 1
    future[1, :, :H - 1] = 2.0                 # 末步无真值（mask=0，补零）
    future[2, :, :] = 0.5                      # 锚点 MAE = 0.5
    mask = torch.tensor([[1.0] * H, [1.0] * (H - 1) + [0.0], [1.0] * H])

    anchor_mae, full = dynamics_anchor_mae(pred, future, mask, horizon=H)

    assert full.tolist() == [True, False, True]
    assert torch.allclose(anchor_mae[full], torch.tensor([1.0, 0.5]))
    # 不完整锚点被置零，不会误入累积
    assert anchor_mae[1].item() == 0.0


def test_anchor_mae_averages_all_steps_and_all_rois():
    """ROI 之间误差量级不同：锚点值 = 8 步 × 2 ROI 的等权平均。"""
    pred = torch.zeros(1, 2, H)
    future = torch.zeros(1, 2, H)
    future[0, 0, :] = 1.0                      # ROI0 误差 1
    future[0, 1, :] = 3.0                      # ROI1 误差 3
    anchor_mae, full = dynamics_anchor_mae(pred, future, torch.ones(1, H), horizon=H)
    assert bool(full[0])
    assert abs(anchor_mae[0].item() - 2.0) < 1e-6   # (1+3)/2


def test_anchor_mae_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='cpm_horizon|只有'):
        dynamics_anchor_mae(torch.zeros(2, 2, 4), torch.zeros(2, 2, H),
                            torch.ones(2, H), horizon=H)
    with pytest.raises(ValueError, match='future'):
        dynamics_anchor_mae(torch.zeros(2, 2, H), torch.zeros(2, 3, H),
                            torch.ones(2, H), horizon=H)
    with pytest.raises(ValueError, match='future_mask'):
        dynamics_anchor_mae(torch.zeros(2, 2, H), torch.zeros(2, 2, H),
                            torch.ones(2, H - 1), horizon=H)


# ────────────────────────────── 被试等权聚合 ──────────────────────────────

def test_subject_equal_weight_not_anchor_weighted():
    """S1 有 3 个锚点(1,1,1)，S2 有 1 个锚点(5)：等权 → (1+5)/2=3，锚点权 → 2.0。"""
    subj_acc = {'S1': [3.0, 3], 'S2': [5.0, 1]}
    value, n_subj, n_anchor = aggregate_dynamics_subject_metric(subj_acc, 4)
    assert value == pytest.approx(3.0)
    assert n_subj == 2
    assert n_anchor == 4


def test_aggregation_invariant_to_batch_partition():
    """同一批锚点按不同 batch/被试顺序喂入，主指标一致。"""
    n_subjects, anchors_per_subject, n_rois = 4, 6, 5
    pred = torch.zeros(n_subjects * anchors_per_subject, n_rois, H)
    future = torch.tensor(
        [float(1 + i % n_subjects) for i in range(n_subjects * anchors_per_subject)]
    ).view(-1, 1, 1).expand(-1, n_rois, H).contiguous()
    mask = torch.ones(n_subjects * anchors_per_subject, H)
    anchor_mae, full = dynamics_anchor_mae(pred, future, mask, horizon=H)
    subjects = [f'S{i % n_subjects}' for i in range(n_subjects * anchors_per_subject)]

    def aggregate(order):
        acc = {}
        for i in order:
            if not bool(full[i]):
                continue
            rec = acc.setdefault(subjects[i], [0.0, 0])
            rec[0] += float(anchor_mae[i])
            rec[1] += 1
        return aggregate_dynamics_subject_metric(acc, len(order))

    whole = aggregate(list(range(len(subjects))))
    # 逐样本一个 batch（batch=1）与打乱顺序后分批，结果都应相同
    assert whole[0] == pytest.approx(aggregate(list(reversed(range(len(subjects)))))[0])
    assert whole[1] == n_subjects
    assert whole[2] == n_subjects * anchors_per_subject
    assert whole[0] == pytest.approx(sum(range(1, n_subjects + 1)) / n_subjects)


def test_aggregation_raises_when_no_full_horizon_target():
    with pytest.raises(RuntimeError, match='没有具备完整'):
        aggregate_dynamics_subject_metric({}, 12, horizon=H)
    # 被试存在但全部锚点不完整（累积里没有条目）同样报错，不回退
    with pytest.raises(RuntimeError):
        aggregate_dynamics_subject_metric({'S1': [0.0, 0]}, 3, horizon=H)


# ────────────────────────────── 选优与 early stopping ──────────────────────────────

DYN_KEY = dynamics_metric_key(H)


def test_selection_metric_combines_mae_and_fc_mse():
    """联合判据 = H1–N MAE + FC MSE；分项同时返回供日志核对。"""
    from main import selection_metric, selection_metric_key
    metrics = {DYN_KEY: 0.5, 'metric_fc_mse': 0.25}
    score, mae, fc = selection_metric(metrics, H)
    assert selection_metric_key(H) == f'metric_selection_h1_{H}_mae_plus_fc_mse'
    assert score == pytest.approx(0.75)
    assert mae == pytest.approx(0.5)
    assert fc == pytest.approx(0.25)


def test_selection_uses_multi_step_metric_over_single_step():
    """候选 A 单步更优、候选 B 多步更优 → 选 B，且单步指标不影响判定。"""
    selector = CheckpointSelector(DYN_KEY, dynamics_metric_name(H), patience=3)
    candidate_a = {'metric_mae_next': 0.10, DYN_KEY: 0.90}
    candidate_b = {'metric_mae_next': 0.20, DYN_KEY: 0.50}

    assert selector.step(candidate_a)[0] is True      # 首个 epoch 即为 best
    is_best_b, cur_b = selector.step(candidate_b)
    assert is_best_b is True
    assert cur_b == pytest.approx(0.50)
    assert selector.best == pytest.approx(0.50)
    assert selector.counter == 0
    # A 的单步 MAE 更低，但选优只看多步指标
    assert selector.step(candidate_a)[0] is False


def test_early_stopping_consumes_same_metric_as_best():
    selector = CheckpointSelector(DYN_KEY, dynamics_metric_name(H), patience=2)
    selector.step({'metric_mae_next': 0.9, DYN_KEY: 0.4})
    assert selector.step({'metric_mae_next': 0.1, DYN_KEY: 0.6})[0] is False
    assert not selector.should_stop
    selector.step({'metric_mae_next': 0.1, DYN_KEY: 0.7})
    assert selector.should_stop
    assert selector.best == pytest.approx(0.4)


def test_single_step_improvement_does_not_reset_patience():
    """多步持续变差时，单步改善不得重置 patience（两套口径不混用）。"""
    selector = CheckpointSelector(DYN_KEY, dynamics_metric_name(H), patience=3)
    selector.step({'metric_mae_next': 0.9, DYN_KEY: 0.4})
    for _ in range(2):
        selector.step({'metric_mae_next': 0.1, DYN_KEY: 0.5})
    assert selector.counter == 2
    assert not selector.should_stop
    selector.step({'metric_mae_next': 0.05, DYN_KEY: 0.5})
    assert selector.should_stop


def test_selector_requires_its_metric_and_finite_value():
    selector = CheckpointSelector(DYN_KEY, patience=2)
    with pytest.raises(KeyError, match=DYN_KEY):
        selector.step({'metric_mae_next': 0.1})
    with pytest.raises(ValueError, match='非有限值'):
        selector.step({DYN_KEY: float('nan')})


# ────────────────────── evaluate_tfm 端到端接线（stub 模型）──────────────────────

STUB_H = 4          # stub 模型的预测范围（与 --dynamics_metric_horizon 一致）
STUB_F = 2          # ROI 数


class _StubModel(torch.nn.Module):
    """只产出 Graph-ODE 轨迹所需 aux_info 的最小模型（全零预测）。"""

    cpm_horizon = STUB_H

    def forward(self, bx, bs, bp, **kwargs):
        b, f = bx.shape[0], bx.shape[1]
        outputs = torch.zeros(b, f, 1, 1)
        aux = {'dynamics_pred': torch.zeros(b, f, STUB_H)}
        return outputs, aux


class _StubCriterion:
    def __call__(self, pred, target, mask):
        zero = torch.zeros(())
        return zero, {'loss_rec': zero, 'loss_fc': zero, 'loss_freq': zero}


def _stub_args():
    return SimpleNamespace(cpm_horizon=STUB_H)


def _make_batches(specs, partitions):
    """specs: [(subj_id, 逐元素误差, 有效步数)]；partitions: 每个 batch 的行数。

    预测恒为 0，因此锚点 MAE == 给定误差值，可手算。
    """
    rows = []
    for subj, value, valid in specs:
        future = torch.zeros(1, STUB_F, STUB_H)
        future[:, :, :valid] = value
        mask = torch.zeros(1, STUB_H)
        mask[:, :valid] = 1.0
        rows.append({'subj_id': [subj], 'x': torch.zeros(1, STUB_F, 1, 3),
                     'y': torch.zeros(1, STUB_F, 1), 'x_last': torch.zeros(1, STUB_F),
                     'target_mask': torch.ones(1, 1), 'sc': torch.zeros(1, STUB_F, STUB_F),
                     'future': future, 'future_mask': mask})
    batches, start = [], 0
    for size in partitions:
        chunk = rows[start:start + size]
        start += size
        batches.append({key: torch.cat([r[key] for r in chunk])
                        if key != 'subj_id' else [r['subj_id'][0] for r in chunk]
                        for key in chunk[0]})
    return batches


def _run(specs, partitions):
    from main import evaluate_tfm
    return evaluate_tfm(_StubModel(), _make_batches(specs, partitions),
                        _StubCriterion(), 'cpu', _stub_args())


#: S1 三个完整锚点 (1,3,5) → 被试内均值 3；S2 一个完整锚点 (7) → 均值 7；S3 锚点只有 3 步
_SPECS = [('S1', 1.0, STUB_H), ('S1', 3.0, STUB_H), ('S1', 5.0, STUB_H),
          ('S2', 7.0, STUB_H), ('S3', 9.0, STUB_H - 1)]


def test_evaluate_emits_subject_equal_metric():
    key = dynamics_metric_key(STUB_H)
    metrics = _run(_SPECS, [5])
    # 被试等权 (3+7)/2 = 5；锚点等权会是 (1+3+5+7)/4 = 4，两者必须可区分
    assert metrics[key] == pytest.approx(5.0)
    assert metrics['metric_mae_h1_4_n_subjects'] == 2.0
    # 单步指标仍然产出（观察短期精度）
    assert 'metric_mae_next' in metrics


@pytest.mark.parametrize('partitions', [[5], [1, 1, 1, 1, 1], [3, 2], [2, 1, 2]])
def test_metric_invariant_to_batch_partition(partitions):
    key = dynamics_metric_key(STUB_H)
    assert _run(_SPECS, partitions)[key] == pytest.approx(5.0)


def test_evaluate_reports_missing_subject_id():
    """无 subj_id 时不得退化成 batch 均值（被试等权是主判据定义的一部分）。"""
    from main import evaluate_tfm
    batches = _make_batches(_SPECS, [5])
    for b in batches:
        b.pop('subj_id')
    with pytest.raises(KeyError, match='subj_id'):
        evaluate_tfm(_StubModel(), batches, _StubCriterion(), 'cpu', _stub_args())
