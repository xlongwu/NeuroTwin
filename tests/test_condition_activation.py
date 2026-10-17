# coding=utf-8
"""``train/condition_activation.py`` 的验收测试（全部真实 torch 模块，CPU 合成张量）。

覆盖协议对「条件激活记录器」的要求：
  1. 真实 MDD 条件臂（``inject='heads'``）下 os/cpm/encoder 三桶都找到且在 200 步内
     激活，且 ``assert_activated`` 不会去要求本次配置根本没建立的 backbone 桶；
  2. 真实 :class:`~models.conditioning.DeltaEncoder`（末层零初始化，``mlp``）首步梯度
     恰为 0，但桶在后续某步（≤200）被判定激活 —— 记录器不得把零初始化模块一棒打死；
  3. 故意接不上的适配器（带标记但从不在 forward 里调用的 ``nn.Linear``）必须报
     ``no_grad_at_all``/``never_activated``，且 ``assert_activated`` 必须报错；
  4. 步数预算守卫：``max_steps=201`` 抛 :class:`ValueError`；
  5. 归桶歧义抛 :class:`ValueError` 并点名模块。

第 6 节把方案 §五 要求的「合成激活证据」逐项落到实测数字上（真实训练步，非 mock）：
  6. ``arm='demo'``/``heads`` 与 ``arm='cat'``/``heads``：os/cpm 的首步梯度号与首参更新号
     都被记录，且增量末层早于其输入侧激活（逐参证据，不写死步号）；
  7. ``arm='none'`` 绕过：没有适配器可观察时直接报错，绝不产出「全部未激活」的假指标，
     期望桶不匹配时报错而非静默通过；
  8. ``inject='backbone'``/``'heads+backbone'``：backbone 桶建立并激活；
  9. output-change witness：确定性适配器必须在 ≤ 上限的某步被见证，含 ``nn.Dropout``
     的适配器必须整条关闭 witness，功能版 Condition Dropout 的确定性来自 ``epoch=None``；
 10. 同 seed 两次检查结论逐位一致；``max_steps=1`` 给出 ``not activated within 1 steps``
     这一与 ``activated at step k`` 不同的结论。

不启动真实训练、不写 ``data/``、不使用 GPU；最长链路 200 步小模型合成梯度。
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_condition_injection as tci  # noqa: E402  复用已验证的 fixture/建模 helper
from models.tfm import ConditionFiLM  # noqa: E402
from train.condition_activation import (  # noqa: E402
    ACTIVATION_STEP_CAP, BUCKET_BACKBONE, BUCKET_CPM, BUCKET_ENCODER, BUCKET_OS,
    STATUSES, STATUS_ACTIVATED, STATUS_NEVER, STATUS_NO_GRAD, STATUS_NOT_BUILT,
    ActivationProbe, format_summary, normalize_bucket_name, run_activation_check)
from train.optim import (  # noqa: E402
    get_param_groups, verify_conditioning_param_groups)

DEVICE = torch.device('cpu')
N_ROIS = 6
CONTEXT_MAX = 8
BATCH = 4
HORIZON = 4


# ══════════════════════════════════════════════════════════════════════════
# 夹具
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def mdd_stats():
    """真实 merged MDD 表的 train-only 统计量（只读；本文件绝不写 data/）。"""
    values, mask, _, _ = tci.load_merged_conditions(tci.MDD_MERGED, tci.MDD_FIELDS)
    stats = tci._stats_for(tci.MDD_FIELDS, values, mask, list(range(12)))
    return stats, values, mask


def _build(arm: str, stats, inject: str = 'heads', seed: int = 2024) -> nn.Module:
    torch.manual_seed(seed)
    return tci._tfm('mdd8', arm, stats, dim=16, layers=2, inject=inject).to(DEVICE)


def _make_batch_fn(stats, values, mask, seed: int = 7):
    """固定合成批次（影像/SC/条件/双头目标）；``step`` 为 1 基但批次不随步变化。"""
    gen = torch.Generator().manual_seed(seed)
    z, code = stats.encode(values[:BATCH], mask[:BATCH])
    cond = tci._cond_dict(z, code, mask[:BATCH])
    x = torch.randn(BATCH, N_ROIS, 1, CONTEXT_MAX, generator=gen)
    sc = torch.rand(BATCH, N_ROIS, N_ROIS, generator=gen)
    y = torch.randn(BATCH, N_ROIS, 1, 1, generator=gen)
    yf = torch.randn(BATCH, N_ROIS, HORIZON, generator=gen)
    cmask = torch.ones(BATCH, HORIZON)

    def make_batch(step: int) -> dict:
        assert int(step) >= 1, f'make_batch 收到非 1 基步号 {step}'
        return {'x': x, 'sc_matrix': sc, 'condition': cond,
                'y': y, 'cpm_target': yf, 'cpm_mask': cmask}
    return make_batch


def _optimizer(model: nn.Module):
    """按现有 finetune 分组口径建优化器（backbone / condition 两组，学习率不同）。"""
    args = Namespace(mode='finetune', lr_peak=1e-2, backbone_lr_scale=0.1,
                     condition_lr_scale=2.0, adapter_lr_scale=1.0,
                     tfm_moe_lr_scale=1.0, loss_lr_scale=1.0)
    return torch.optim.SGD(get_param_groups(model, args), weight_decay=0.0)


# ══════════════════════════════════════════════════════════════════════════
# 1. 真实 MDD 条件臂 + inject='heads'
# ══════════════════════════════════════════════════════════════════════════

def test_real_heads_arm_activates_os_cpm_encoder_and_skips_backbone(mdd_stats):
    """heads 档：os/cpm/encoder 三桶全部激活；backbone 未建立，要求它属于假失败。"""
    stats, values, mask = mdd_stats
    model = _build('cat', stats)
    assert model.cond_inject == 'heads' and model.cond_norm is None

    probe = ActivationProbe(model)
    rows = probe.summary()['buckets']
    assert probe.summary()['built_buckets'] == [BUCKET_OS, BUCKET_CPM, BUCKET_ENCODER]
    assert rows[BUCKET_OS]['modules'] == ['one_step_head.film']
    assert rows[BUCKET_CPM]['modules'] == ['cpm_head.film']
    assert set(rows[BUCKET_ENCODER]['modules']) == {
        'conditioner', 'conditioner.demo', 'conditioner.delta'}
    # 每个条件参数只归一个桶：嵌套的 demo/delta 不会被 conditioner 重复计数
    assert rows[BUCKET_OS]['n_params'] == 2 and rows[BUCKET_CPM]['n_params'] == 2
    assert rows[BUCKET_BACKBONE]['status'] == STATUS_NOT_BUILT
    total = sum(r['n_params'] for r in rows.values())
    assert total == len([n for n, _ in model.named_parameters()
                         if n.startswith(('conditioner.', 'one_step_head.film.',
                                          'cpm_head.film.'))])

    optimizer = _optimizer(model)
    verify_conditioning_param_groups(model, optimizer)
    out = run_activation_check(
        model, _make_batch_fn(stats, values, mask), optimizer,
        max_steps=ACTIVATION_STEP_CAP, probe=probe,
        forward_kwargs={'condition_epoch': 0},
        expected_buckets=(BUCKET_OS, BUCKET_CPM, BUCKET_ENCODER), device=DEVICE)
    rows = out['buckets']
    for bucket in (BUCKET_OS, BUCKET_CPM, BUCKET_ENCODER):
        row = rows[bucket]
        assert row['status'] == STATUS_ACTIVATED, format_summary(out)
        assert 1 <= row['first_nonzero_grad_step'] <= ACTIVATION_STEP_CAP
        assert row['first_grad_max'] > 0.0
        assert row['first_param_update_step'] is not None
        assert row['in_optimizer'] is True
        assert row['trainable'] == row['n_params']
        assert row['params_without_grad_object'] == []
    assert out['all_activated'] is True
    assert rows[BUCKET_BACKBONE]['status'] == STATUS_NOT_BUILT
    probe.assert_activated(ACTIVATION_STEP_CAP,
                           (BUCKET_OS, BUCKET_CPM, 'shared/encoder'))
    with pytest.raises(AssertionError, match=BUCKET_BACKBONE):
        probe.assert_activated(ACTIVATION_STEP_CAP, ('backbone', BUCKET_OS))


def test_heads_plus_backbone_arm_builds_and_activates_four_buckets(mdd_stats):
    """heads+backbone 档：四桶全建且全激活（条件 RMSNorm 零初始化起点也不判死）。"""
    stats, values, mask = mdd_stats
    model = _build('cat', stats, inject='heads+backbone')
    probe = ActivationProbe(model)
    assert probe.summary()['built_buckets'] == [BUCKET_OS, BUCKET_CPM,
                                                BUCKET_BACKBONE, BUCKET_ENCODER]
    assert probe.summary()['buckets'][BUCKET_BACKBONE]['modules'] == ['cond_norm']
    optimizer = _optimizer(model)
    out = run_activation_check(
        model, _make_batch_fn(stats, values, mask), optimizer,
        max_steps=ACTIVATION_STEP_CAP, probe=probe,
        forward_kwargs={'condition_epoch': 0},
        expected_buckets=('os', 'cpm', 'shared/encoder', 'ada_rms_norm'),
        device=DEVICE)
    assert out['required_buckets'] == sorted([BUCKET_OS, BUCKET_CPM,
                                              BUCKET_BACKBONE, BUCKET_ENCODER])
    assert out['all_activated'] is True, format_summary(out)


# ══════════════════════════════════════════════════════════════════════════
# 2. 零初始化 DeltaEncoder：首步梯度恰为 0，但不能判死
# ══════════════════════════════════════════════════════════════════════════

class _DeltaChain(nn.Module):
    """隔离的 ``DeltaEncoder -> 零初始化 FiLM`` 链路（复刻真实模型的起步结构）。

    FiLM 权重为 0 ⇒ 首步 :math:`\\partial L/\\partial c \\equiv 0` ⇒ 整个 DeltaEncoder
    （含末层）首步梯度恰为 0；末层先被更新，输入侧才拿得到梯度。只看首步就把模块判死，
    会把这条完全正常的两级起步链路误报成「条件模块没接上」。
    """

    def __init__(self, in_dim: int, cond_dim: int, hidden: int, mod_dim: int):
        super().__init__()
        from models.conditioning import DeltaEncoder
        from models.tfm import ConditionFiLM
        self.delta = DeltaEncoder(in_dim, cond_dim, hidden, 'mlp')
        self.film = ConditionFiLM(cond_dim, mod_dim)

    def forward(self, c_demo: torch.Tensor, phi: torch.Tensor, mask: torch.Tensor):
        return self.film(c_demo + self.delta(c_demo, phi, mask))


def _delta_chain(seed: int = 0):
    torch.manual_seed(seed)
    in_dim, cond_dim, hidden, mod_dim, n_new = 12, 4, 8, 3, 3
    chain = _DeltaChain(in_dim, cond_dim, hidden, mod_dim).to(DEVICE)
    gen = torch.Generator().manual_seed(11)
    fixed = {'c_demo': torch.randn(BATCH, cond_dim, generator=gen),
             'phi': torch.randn(BATCH, in_dim - cond_dim - n_new, generator=gen),
             'mask': torch.ones(BATCH, n_new)}
    target = torch.randn(BATCH, mod_dim, generator=gen)

    def make_batch(step: int) -> dict:
        return {**fixed, 'target': target}

    def loss_fn(outputs, batch):
        scale, shift = outputs
        return nn.functional.mse_loss(1.0 + scale + shift, batch['target'])
    return chain, make_batch, loss_fn, fixed


def test_zero_init_delta_encoder_is_not_judged_dead_from_first_step():
    """末层零初始化 ⇒ 首步梯度恰为 0（但梯度对象存在），桶仍在后续步被判激活。"""
    chain, make_batch, loss_fn, fixed = _delta_chain()
    delta = chain.delta
    with torch.no_grad():
        increment = delta(fixed['c_demo'], fixed['phi'], fixed['mask'])
        assert float(increment.abs().max()) == 0.0, '起点增量必须恰为 0（c_MDD=c_demo）'

    opt = torch.optim.SGD(list(chain.parameters()), lr=1e-2)
    opt.zero_grad()
    loss_fn(chain(**fixed), {'target': make_batch(1)['target']}).backward()
    assert delta.out.weight.grad is not None and delta.out.bias.grad is not None
    assert float(delta.out.weight.grad.abs().max()) == 0.0
    assert float(delta.out.bias.grad.abs().max()) == 0.0
    assert float(delta.net[0].weight.grad.abs().max()) == 0.0
    assert float(chain.film.proj.weight.grad.abs().max()) > 0.0

    opt.zero_grad(set_to_none=True)
    probe = ActivationProbe(
        modules={'conditioner.delta': delta, 'one_step_head.film': chain.film},
        optimizer=opt,
        probe_inputs={'conditioner.delta':
                      ((fixed['c_demo'], fixed['phi'], fixed['mask']), {})})
    out = run_activation_check(chain, make_batch, opt,
                               max_steps=ACTIVATION_STEP_CAP, probe=probe,
                               loss_fn=loss_fn, device=DEVICE,
                               expected_buckets=(BUCKET_ENCODER, BUCKET_OS),
                               stop_when_activated=False)
    enc = out['buckets'][BUCKET_ENCODER]
    assert enc['status'] == STATUS_ACTIVATED, format_summary(out)
    assert enc['first_nonzero_grad_step'] is not None
    assert enc['first_nonzero_grad_step'] > 1, '首步梯度应恰为 0（零初始化起步）'
    assert enc['first_nonzero_grad_step'] <= ACTIVATION_STEP_CAP
    # 梯度落地为参数更新，并且增量输出真的动起来（比梯度更强的功能性证据）
    assert enc['first_param_update_step'] == enc['first_nonzero_grad_step']
    assert enc['first_output_change_step'] == enc['first_nonzero_grad_step']
    assert enc['output_change_witness'] == 'witnessed'
    assert enc['n_params'] == len(list(delta.parameters()))
    # 逐参证据：末层先被更新，输入侧（net.0）随后一步才拿到梯度
    per_param = enc['param_first_nonzero_grad_step']
    final_key = 'conditioner.delta.' + [k for k, v in delta.named_parameters()
                                        if v is delta.out.weight][0]
    input_key = 'conditioner.delta.' + [k for k, v in delta.named_parameters()
                                        if v is delta.net[0].weight][0]
    assert per_param[final_key] == enc['first_nonzero_grad_step']
    assert per_param[input_key] > per_param[final_key]
    os_row = out['buckets'][BUCKET_OS]
    assert os_row['first_nonzero_grad_step'] == 1
    assert os_row['status'] == STATUS_ACTIVATED
    probe.assert_activated(ACTIVATION_STEP_CAP, (BUCKET_ENCODER, BUCKET_OS))


def test_probe_does_not_consume_global_rng(mdd_stats):
    """构造与观察都不得推进全局随机流：否则诊断本身会改变训练轨迹（不可复现）。"""
    stats, _values, _mask = mdd_stats
    model = _build('cat', stats)
    probe = ActivationProbe(model)
    torch.manual_seed(777)
    cpu_before = torch.get_rng_state().clone()
    ActivationProbe(model)                       # 起点 witness 评估
    probe.record_step(1)                         # 观察步（含 witness 重算）
    assert torch.equal(torch.get_rng_state(), cpu_before), '探针推进了全局 CPU RNG'


def test_update_detection_falls_back_without_step_boundary():
    """忘调 note_step_boundary 仍能检出更新，但第一次观察没有基线 —— 这正是快照的意义。

    退回「上一次观察值」时第 1 步看不到变化（还没有更早的观察），第 2 步才看到；用
    步前快照则第 1 步的优化器更新立即可见。两种口径都不该静默漏检，所以这里把差异
    显式断言出来，而不是假装它们等价。
    """
    batch_keys = ('c_demo', 'phi', 'mask')

    def run(boundary: bool):
        chain, make_batch, loss_fn, _ = _delta_chain()
        probe = ActivationProbe(
            modules={'conditioner.delta': chain.delta,
                     'one_step_head.film': chain.film}, enable_witness=False)
        opt = torch.optim.SGD(list(chain.parameters()), lr=1e-2)
        batch = make_batch(1)
        for step in (1, 2, 3):
            opt.zero_grad(set_to_none=True)
            if boundary:
                probe.note_step_boundary(chain)
            loss_fn(chain(**{k: batch[k] for k in batch_keys}), batch).backward()
            opt.step()
            probe.record_step(step)
        return probe.summary()['buckets']

    fallback = run(boundary=False)
    assert fallback[BUCKET_OS]['first_nonzero_grad_step'] == 1
    assert fallback[BUCKET_OS]['first_param_update_step'] == 2, '首步无基线可比'
    assert fallback[BUCKET_ENCODER]['first_nonzero_grad_step'] > 1
    assert (fallback[BUCKET_ENCODER]['first_param_update_step']
            == fallback[BUCKET_ENCODER]['first_nonzero_grad_step'])

    boundary = run(boundary=True)
    assert boundary[BUCKET_OS]['first_param_update_step'] == 1
    assert (boundary[BUCKET_ENCODER]['first_param_update_step']
            == boundary[BUCKET_ENCODER]['first_nonzero_grad_step'] > 1)


def test_probe_marks_witness_unavailable_for_modules_it_cannot_call():
    """无法独立调用的模块：witness 记 unavailable，而不是伪造一个「已见证」。"""
    from models.conditioning import DeltaEncoder
    delta = DeltaEncoder(9, 4, 8, 'linear')
    probe = ActivationProbe(modules={'conditioner.delta': delta})
    row = probe.summary()['buckets'][BUCKET_ENCODER]
    assert row['witness_available'] == []
    assert row['witness_unavailable'] == ['conditioner.delta']
    assert row['output_change_witness'] == 'unavailable'
    assert row['first_output_change_step'] is None


# ══════════════════════════════════════════════════════════════════════════
# 3. 故意接不上的适配器
# ══════════════════════════════════════════════════════════════════════════

class _OsHeadStub(nn.Module):
    """给死适配器一个可判定的头名（``one_step_head.*``）；``dead`` 从不被调用。"""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Linear(dim, 1)
        self.dead = nn.Linear(dim, dim)
        self.dead.is_conditioning_adapter = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)


class _DeadAdapterNet(nn.Module):
    def __init__(self, features: int = 6, dim: int = 4):
        super().__init__()
        self.embed = nn.Linear(features, dim)
        self.one_step_head = _OsHeadStub(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.one_step_head(self.embed(x))


def _dead_net(seed: int = 0):
    torch.manual_seed(seed)
    model = _DeadAdapterNet().to(DEVICE)
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(BATCH, 6, generator=gen)
    y = torch.randn(BATCH, 1, generator=gen)

    def make_batch(step: int) -> dict:
        return {'x': x, 'y': y}

    def loss_fn(outputs, batch):
        return nn.functional.mse_loss(outputs, batch['y'])
    return model, make_batch, loss_fn


def test_dead_adapter_is_reported_as_never_activated():
    """带标记却从未进入前反向的适配器：no_grad_at_all + assert_activated 必须报错。"""
    model, make_batch, loss_fn = _dead_net()
    probe = ActivationProbe(model)
    assert probe.summary()['built_buckets'] == [BUCKET_OS]
    opt = torch.optim.SGD(list(model.parameters()), lr=1e-2)
    out = run_activation_check(model, make_batch, opt,
                               max_steps=ACTIVATION_STEP_CAP, probe=probe,
                               loss_fn=loss_fn, device=DEVICE,
                               expected_buckets=(BUCKET_OS,))
    row = out['buckets'][BUCKET_OS]
    assert row['status'] in (STATUS_NO_GRAD, STATUS_NEVER), format_summary(out)
    assert row['first_nonzero_grad_step'] is None
    assert row['first_param_update_step'] is None
    assert row['first_output_change_step'] is None
    assert row['outcome'] == f'not activated within {ACTIVATION_STEP_CAP} steps'
    assert row['in_optimizer'] is True, '死适配器即使在优化器里也不等于已激活'
    assert sorted(row['params_without_grad_object']) == [
        'one_step_head.dead.bias', 'one_step_head.dead.weight']
    assert row['output_change_witness'] == 'unavailable'
    assert out['all_activated'] is False and out['activated_within_budget'] is False
    assert out['steps_observed'] == ACTIVATION_STEP_CAP
    with pytest.raises(AssertionError) as exc:
        probe.assert_activated(ACTIVATION_STEP_CAP, (BUCKET_OS,))
    assert 'not activated within' in str(exc.value)
    assert BUCKET_OS in str(exc.value)


# ══════════════════════════════════════════════════════════════════════════
# 4. 步数预算守卫
# ══════════════════════════════════════════════════════════════════════════

def test_step_budget_guard_rejects_over_cap_and_non_finite():
    """协议硬上限 200：201/inf/nan/0/负数/bool/字符串一律 ValueError。"""
    assert ACTIVATION_STEP_CAP == 200
    model, make_batch, loss_fn = _dead_net()
    opt = torch.optim.SGD(list(model.parameters()), lr=1e-2)
    for bad in (201, 10_000, float('inf'), float('nan'), 0, -1, True, '200'):
        with pytest.raises(ValueError):
            run_activation_check(model, make_batch, opt, max_steps=bad,
                                 loss_fn=loss_fn, device=DEVICE)
    probe = ActivationProbe(model)
    with pytest.raises(ValueError, match='max_steps'):
        probe.assert_activated(ACTIVATION_STEP_CAP + 1, (BUCKET_OS,))
    # 上限内合法：预算是「最多看多少步」，1 步也允许（结论会标注只观察了 1 步）
    summary = run_activation_check(model, make_batch, opt, max_steps=1,
                                   loss_fn=loss_fn, device=DEVICE)
    assert summary['steps_observed'] == 1
    assert summary['buckets'][BUCKET_OS]['outcome'] == 'not activated within 1 steps'


# ══════════════════════════════════════════════════════════════════════════
# 5. 归桶歧义
# ══════════════════════════════════════════════════════════════════════════

class _AmbiguousNet(nn.Module):
    def __init__(self):
        super().__init__()
        from models.tfm import ConditionFiLM
        self.mystery_adapter = nn.ModuleDict({'film': ConditionFiLM(4, 3)})


def test_ambiguous_module_name_raises_with_the_offending_name():
    """判定不了所属头必须报错并点名，绝不静默归桶（否则 per-head 统计失真）。"""
    from models.tfm import ConditionFiLM
    with pytest.raises(ValueError, match='mystery_adapter.film'):
        ActivationProbe(modules={'mystery_adapter.film': ConditionFiLM(4, 3)})
    with pytest.raises(ValueError, match='mystery_adapter.film'):
        ActivationProbe(_AmbiguousNet())
    with pytest.raises(ValueError, match='is_conditioning_adapter'):
        ActivationProbe(modules={'one_step_head.plain': nn.Linear(2, 2)})
    with pytest.raises(ValueError, match='is_conditioning_adapter'):
        ActivationProbe(nn.Linear(2, 2))
    with pytest.raises(ValueError, match='model 或 modules'):
        ActivationProbe()


def test_bucket_alias_normalisation_and_unknown_bucket():
    """``shared/encoder`` 与 ``encoder`` 同义；未知桶名报错（不按子串猜桶）。"""
    assert normalize_bucket_name('shared/encoder') == BUCKET_ENCODER
    assert normalize_bucket_name('ada_rms_norm') == BUCKET_BACKBONE
    assert normalize_bucket_name(BUCKET_OS) == BUCKET_OS
    with pytest.raises(ValueError, match='未知条件桶'):
        normalize_bucket_name('lora')
    chain, _mb, _lf, _ = _delta_chain()
    probe = ActivationProbe(modules={'conditioner.delta': chain.delta})
    with pytest.raises(ValueError, match='未知条件桶'):
        probe.require(['lora'])
    with pytest.raises(ValueError, match='为空'):
        probe.require([])


def test_format_summary_reports_required_buckets_and_outcome():
    """日志摘要必须同时给出「被要求的桶」与逐步结论（激活步号 / 未激活步数上限）。"""
    model, _make_batch, _loss_fn = _dead_net()
    probe = ActivationProbe(model)
    probe.require((BUCKET_OS,))
    text = format_summary(probe.summary())
    assert 'required=os' in text
    assert 'not activated within 0 steps' in text
    assert 'not built by this config' in text        # cpm/backbone/encoder 未建立
    assert probe.last_step is None
    with pytest.raises(TypeError):
        format_summary({'no_buckets': 1})


def test_record_step_requires_monotonic_steps():
    """步号必须严格递增，否则「首次非零梯度」失去定义。"""
    model, _make_batch, _loss_fn = _dead_net()
    probe = ActivationProbe(model)
    probe.note_step_boundary(model)
    probe.record_step(5)
    with pytest.raises(ValueError, match='递增'):
        probe.record_step(5)
    with pytest.raises(ValueError, match='递增'):
        probe.record_step(3)


# ══════════════════════════════════════════════════════════════════════════
# 6. 协议 §五「合成激活」的实测证据（真实 forward/backward/optimizer.step）
#
# 本节每条结论都来自实际训练步，不 mock、不做 AST 检查。为把运行时间压在秒级，
# 预算取 ``EVIDENCE_BUDGET = 60`` « 协议上限 200：实测最晚的首次非零梯度出现在
# **第 3 步**（arm='cat' 的增量输入侧 ``delta.net.0``），因此 60 步已有 20 倍余量；
# 「超过上限就必须改写成 not activated within N steps」由 §7 的上限语义测试单独覆盖。
#
# 实测口径（CPU、mdd8、dim=16/layers=2、lr_peak=1e-2、SGD、同一合成批次）：
#   heads            ：cpm=1、os=2、encoder=2（cat 的 delta.net.0=3）
#   backbone         ：backbone=1、encoder=2（os/cpm 未建立）
#   heads+backbone   ：cpm=1、backbone=1、os=2、encoder=2
# ══════════════════════════════════════════════════════════════════════════

EVIDENCE_BUDGET = 60


def _heads_check(mdd_stats, arm: str, inject: str = 'heads', *,
                 max_steps: int = EVIDENCE_BUDGET, stop_when_activated: bool = False,
                 expected_buckets=('os', 'cpm', 'encoder'), with_condition: bool = True):
    """建小模型 + 真跑 ``max_steps`` 步，返回 ``(summary, probe, model)``。"""
    stats, values, mask = mdd_stats
    model = _build(arm, stats, inject=inject)
    make_batch = _make_batch_fn(stats, values, mask)
    if not with_condition:
        def make_batch(step, _base=make_batch):   # 故意不把条件张量喂给 forward
            b = _base(step)
            b.pop('condition')
            return b
    probe = ActivationProbe(model)
    kwargs = {'condition_epoch': 0} if with_condition else None
    out = run_activation_check(
        model, make_batch, _optimizer(model), max_steps=max_steps, probe=probe,
        forward_kwargs=kwargs, expected_buckets=expected_buckets, device=DEVICE,
        stop_when_activated=stop_when_activated)
    return out, probe, model


def _delta_param_names(model) -> set:
    """增量分支在全模型参数表里的名字（不写死 ``net.0``/``net.2`` 这类下标）。"""
    return {n for n, _ in model.named_parameters() if n.startswith('conditioner.delta.')}


# --------------------------------------------------------------- 1. demo/heads
def test_demo_heads_records_os_and_cpm_first_grad_and_first_update_steps(mdd_stats):
    """``arm='demo'`` + ``inject='heads'``：os/cpm 两桶都在上限内激活并记下首步号。

    协议关心的是**区间**而非某个写死的数字：os FiLM 因双零初始化（``ConditionFiLM``
    权重为 0 + OS 头 ``decode`` 为 0）首步梯度恰为 0，第 2 步才出现非零梯度，所以
    断言 ``1 <= first_gradient_step <= max_steps``，并断言首步号确实被记录下来了。
    """
    out, probe, _model = _heads_check(mdd_stats, 'demo')
    rows = out['buckets']
    assert out['steps_observed'] == EVIDENCE_BUDGET
    assert out['all_activated'] is True, format_summary(out)
    assert out['activated_within_budget'] is True
    assert rows[BUCKET_BACKBONE]['status'] == STATUS_NOT_BUILT

    for key, module_name in ((BUCKET_OS, 'one_step_head.film'),
                             (BUCKET_CPM, 'cpm_head.film')):
        row = rows[key]
        assert row['status'] == STATUS_ACTIVATED, format_summary(out)
        assert row['modules'] == [module_name]
        grad_step = row['first_nonzero_grad_step']
        update_step = row['first_param_update_step']
        assert grad_step is not None and 1 <= grad_step <= EVIDENCE_BUDGET
        assert update_step is not None and 1 <= update_step <= EVIDENCE_BUDGET
        # 参数更新不可能早于首次非零梯度（同一步内可以：backward 后立即 step）
        assert update_step >= grad_step
        assert row['first_grad_max'] > 0.0
        assert row['first_param_update_max'] > 0.0
        assert row['outcome'].startswith('activated at step ')
        assert row['in_optimizer'] is True
    # 零初始化起步的正面证据：os 桶确实有「梯度对象存在但恰为 0」的步，仍未被判死
    assert rows[BUCKET_OS]['zero_grad_steps'] >= 1
    assert rows[BUCKET_OS]['params_without_grad_object'] == []
    probe.assert_activated(EVIDENCE_BUDGET, (BUCKET_OS, BUCKET_CPM))


# ----------------------------------------------------------- 2. cat 的增量链路
def test_real_cat_delta_final_layer_activates_before_its_input_side(mdd_stats):
    """真实 ``arm='cat'`` 模型：增量**末层**先激活，输入侧晚一步（模块 docstring 的顺序）。

    只看桶级首步会把这条链路压成一个数字，所以这里断言逐参证据的**先后关系**：
    输入侧的梯度被自身零初始化末层挡住，必须等末层先被改写。
    """
    out, probe, model = _heads_check(mdd_stats, 'cat')
    enc = out['buckets'][BUCKET_ENCODER]
    assert enc['status'] == STATUS_ACTIVATED, format_summary(out)
    assert set(enc['modules']) == {'conditioner', 'conditioner.demo', 'conditioner.delta'}
    assert enc['params_without_grad_object'] == [], '真实 cat 臂不该有从不参与计算的条件参数'

    per = enc['param_first_nonzero_grad_step']
    delta_names = _delta_param_names(model)
    assert delta_names <= set(per), f'增量参数未逐参记录：{sorted(delta_names - set(per))}'
    delta = model.conditioner.delta
    out_ids = {id(p) for p in delta.out.parameters()}
    params = dict(model.named_parameters())
    final_steps = {n: s for n, s in per.items()
                   if n in delta_names and id(params[n]) in out_ids}
    input_steps = {n: s for n, s in per.items()
                   if n in delta_names and id(params[n]) not in out_ids}
    assert final_steps and input_steps, (
        f'增量末层/输入侧参数缺失：final={sorted(final_steps)} input={sorted(input_steps)}')
    assert min(final_steps.values()) < max(input_steps.values()), (
        f'零初始化起步顺序被破坏：末层={final_steps} 输入侧={input_steps}')
    for n, s in list(final_steps.items()) + list(input_steps.items()):
        assert 1 <= s <= EVIDENCE_BUDGET, f'{n} 首步梯度 {s} 越出预算'
        assert per[n] == s
    assert enc['first_nonzero_grad_step'] == min(per[n] for n in delta_names)
    assert 1 <= enc['first_param_update_step'] <= EVIDENCE_BUDGET
    probe.assert_activated(EVIDENCE_BUDGET, ('shared/encoder',))


# --------------------------------------------------------------- 3. arm='none'
def test_bypassed_arm_none_never_fabricates_activations(mdd_stats):
    """``arm='none'`` 绕过条件通路：没有适配器可观察时**报错**，不产出「全部未激活」假指标。"""
    stats, values, mask = mdd_stats
    model = _build('none', stats)
    assert model.conditioner is None and model.cond_norm is None
    assert model.one_step_head.film is None, 'arm=none 仍建了 head FiLM'
    opt = _optimizer(model)
    make_batch = _make_batch_fn(stats, values, mask)

    with pytest.raises(ValueError, match='未发现任何条件模块'):
        run_activation_check(model, make_batch, opt, max_steps=20, device=DEVICE)
    with pytest.raises(ValueError, match='未发现任何条件模块'):
        ActivationProbe(model)
    with pytest.raises(ValueError, match='未发现任何条件模块'):
        ActivationProbe(modules={})       # 空映射也不能换出一条「全部未激活」的报表

    # 归桶口径不被静默放宽：把本次配置根本没建立的 backbone 列进期望必须报错
    heads_out, heads_probe, _ = _heads_check(mdd_stats, 'demo',
                                             expected_buckets=('os', 'cpm', 'backbone'))
    assert heads_out['all_activated'] is False
    assert heads_out['buckets'][BUCKET_BACKBONE]['status'] == STATUS_NOT_BUILT
    with pytest.raises(AssertionError, match=BUCKET_BACKBONE):
        heads_probe.assert_activated(EVIDENCE_BUDGET, ('os', 'cpm', 'backbone'))
    with pytest.raises(ValueError, match='未知条件桶'):
        heads_probe.require(('decoder',))
    with pytest.raises(ValueError, match='为空'):
        heads_probe.require(())


# ------------------------------------------------------------- 4. backbone 档
def test_backbone_injection_builds_and_activates_backbone_bucket(mdd_stats):
    """主干注入在 tiny 测试模型上**可用**（无需 skip）：``cond_norm`` 桶在上限内激活。"""
    stats, values, mask = mdd_stats
    model = _build('cat', stats, inject='backbone')
    assert model.cond_norm is not None, 'cond_inject=backbone 未建立 ConditionNormGenerator'
    probe = ActivationProbe(model)
    built = probe.summary()['built_buckets']
    assert built == [BUCKET_BACKBONE, BUCKET_ENCODER], f'实建桶={built}'
    out = run_activation_check(
        model, _make_batch_fn(stats, values, mask), _optimizer(model),
        max_steps=EVIDENCE_BUDGET, probe=probe,
        forward_kwargs={'condition_epoch': 0},
        expected_buckets=('ada_rms_norm', 'shared/encoder'), device=DEVICE,
        stop_when_activated=False)
    assert out['required_buckets'] == [BUCKET_BACKBONE, BUCKET_ENCODER]
    bb = out['buckets'][BUCKET_BACKBONE]
    assert bb['status'] == STATUS_ACTIVATED, format_summary(out)
    assert bb['modules'] == ['cond_norm']
    assert 1 <= bb['first_nonzero_grad_step'] <= EVIDENCE_BUDGET
    assert bb['first_param_update_step'] == bb['first_nonzero_grad_step']
    assert bb['output_change_witness'] == 'witnessed'
    assert out['all_activated'] is True


def test_heads_plus_backbone_activates_all_four_buckets(mdd_stats):
    """``heads+backbone`` 档：四桶全建全激活，且 ``None`` 期望=按已建成的桶要求。"""
    out, probe, _model = _heads_check(
        mdd_stats, 'cat', inject='heads+backbone',
        expected_buckets=('os', 'cpm', 'backbone', 'shared/encoder'))
    assert out['required_buckets'] == list(sorted(['os', 'cpm', 'backbone', 'encoder']))
    assert out['all_activated'] is True, format_summary(out)
    for key in ('os', 'cpm', 'backbone', 'encoder'):
        row = out['buckets'][key]
        assert row['status'] == STATUS_ACTIVATED
        assert 1 <= row['first_nonzero_grad_step'] <= EVIDENCE_BUDGET
        assert 1 <= row['first_param_update_step'] <= EVIDENCE_BUDGET
    probe.assert_activated(EVIDENCE_BUDGET, None)      # None ⇒ 按本次真正建立的桶


# ------------------------------------------------------------------ 5. witness
def test_output_change_witness_fires_within_cap_for_deterministic_adapters(mdd_stats):
    """无随机性的适配器：固定探针输入下的输出变化必须在 ≤ 上限的某一步被见证。"""
    out, _probe, _model = _heads_check(mdd_stats, 'cat', inject='heads+backbone',
                                       expected_buckets=('os', 'cpm', 'backbone', 'encoder'))
    for key in ('os', 'cpm', 'backbone', 'encoder'):
        row = out['buckets'][key]
        assert row['witness_unavailable'] == [], f'{key} 的 witness 未登记：{row}'
        assert row['output_change_witness'] == 'witnessed', format_summary(out)
        step = row['first_output_change_step']
        assert step is not None and 1 <= step <= EVIDENCE_BUDGET
        # 输出变化只能发生在参数被改写之后（witness 在 optimizer.step 之后评估），
        # 早于首参更新步就意味着 witness 比的是噪声而不是训练结果
        assert step >= row['first_param_update_step']


class _StochasticFiLM(ConditionFiLM):
    """探针**能**独立调用的适配器 + 真 ``nn.Dropout``：用于验证 witness 的关闭口径。"""

    def __init__(self, cond_dim: int, dim: int, p: float = 0.5):
        super().__init__(cond_dim, dim)
        self.drop = nn.Dropout(p)

    def forward(self, c: torch.Tensor):
        scale, shift = super().forward(c)
        return self.drop(scale), shift


def _train_three_steps(module, fixed: torch.Tensor) -> ActivationProbe:
    probe = ActivationProbe(modules={'one_step_head.film': module})
    opt = torch.optim.SGD(list(module.parameters()), lr=1e-1)
    module.train()
    for step in range(1, 4):
        probe.note_step_boundary()
        opt.zero_grad(set_to_none=True)
        scale, shift = module(fixed)
        (scale.sum() + shift.sum()).backward()
        opt.step()
        probe.record_step(step)
    return probe


def test_witness_is_not_used_for_adapters_that_contain_dropout():
    """含 ``nn.Dropout`` 的适配器：witness 整条关闭，即使参数真的被优化器改写了。

    固定输入下 Dropout 仍逐次改变输出，witness 会把随机噪声解释成「训练改变了输出」，
    产生协议明确要避免的假阳性激活。对照组用同名的真 :class:`ConditionFiLM`（无
    Dropout）证明差异只来自 Dropout，而不是来自「探针调不动这个模块」。
    """
    fixed = torch.randn(BATCH, 4, generator=torch.Generator().manual_seed(3))

    torch.manual_seed(0)
    probe = _train_three_steps(_StochasticFiLM(4, 3).to(DEVICE), fixed)
    row = probe.summary()['buckets'][BUCKET_OS]
    assert row['witness_available'] == []
    assert row['witness_unavailable'] == ['one_step_head.film']
    assert row['output_change_witness'] == 'unavailable'
    assert row['first_output_change_step'] is None
    assert row['first_param_update_step'] is not None, '对照：参数确实动了，仍不记 witness'

    torch.manual_seed(0)
    clean_probe = _train_three_steps(ConditionFiLM(4, 3).to(DEVICE), fixed)
    clean = clean_probe.summary()['buckets'][BUCKET_OS]
    assert clean['witness_available'] == ['one_step_head.film']
    assert clean['output_change_witness'] == 'witnessed'
    assert 1 <= clean['first_output_change_step'] <= 3


def test_functional_condition_dropout_keeps_witness_via_epoch_none(mdd_stats):
    """功能版 Condition Dropout 不是 ``nn.Dropout``：witness 仍登记，确定性来自不传 epoch。

    与 docstring 的字面口径不同，实测 ``dropout_p > 0`` 的真实编码器**没有**被当成随机
    模块（遮蔽是函数式的，模块树里没有 Dropout 实例），它落在 witness 清单里；这里把
    「固定输入 ⇒ 逐位同输出」与「遮蔽本身确实随机」两面都断言下来，防止将来把
    ``epoch`` 传进探针后 witness 变成噪声源却无人察觉。
    """
    from models.conditioning import ConditioningConfig, build_condition_encoder
    stats, values, _mask = mdd_stats
    cfg = ConditioningConfig(group='mdd8', arm='cat', d_demo=16, d_new=16,
                             cond_dim=32, h_delta=64, dropout_p=0.2)
    enc = build_condition_encoder(tci.MDD_FIELDS, stats, cfg).to(DEVICE).train()
    assert float(enc.cfg.dropout_p) > 0.0

    probe = ActivationProbe(modules={'conditioner': enc})
    row = probe.summary()['buckets'][BUCKET_ENCODER]
    assert row['witness_available'] == ['conditioner']
    assert row['witness_unavailable'] == []

    module, args, kwargs, baseline = probe._witness['conditioner']
    assert module is enc, 'witness 登记的必须是被探测模块本身'
    with torch.no_grad():
        a, b = enc(*args, **kwargs), enc(*args, **kwargs)
    assert torch.equal(a, b), '探针输入下编码器输出必须逐位可复现，否则 witness 无效'
    assert torch.equal(a, baseline), '起点快照应与当前输出一致（尚未训练）'

    # 遮蔽本身确实生效（给了 epoch 就随机）⇒ 确定性来自 epoch=None 契约，不是没建 Dropout
    si = torch.arange(BATCH, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        e0, e1 = enc(*args, subject_index=si, epoch=0), enc(*args, subject_index=si, epoch=1)
    assert not torch.equal(e0, e1), 'Condition Dropout 未生效，本测试的前提不成立'


# --------------------------------------------------------------- 6. 确定性
def test_same_seed_gives_identical_activation_evidence(mdd_stats):
    """同 seed 两次检查：状态、结论串与全部首步号逐位一致（报告数字必须可复现）。"""
    stats, values, mask = mdd_stats

    def once():
        model = _build('cat', stats)
        probe = ActivationProbe(model)
        out = run_activation_check(
            model, _make_batch_fn(stats, values, mask), _optimizer(model),
            max_steps=EVIDENCE_BUDGET, probe=probe,
            forward_kwargs={'condition_epoch': 0},
            expected_buckets=('os', 'cpm', 'encoder'), device=DEVICE,
            stop_when_activated=False)
        return {k: (r['status'], r['outcome'], r['first_nonzero_grad_step'],
                    r['first_param_update_step'], r['first_output_change_step'],
                    r['param_first_nonzero_grad_step'], r['grad_max_seen'] > 0.0)
                for k, r in out['buckets'].items()}

    first, second = once(), once()
    assert first == second, f'同 seed 两次结论不一致：\n{first}\n{second}'
    assert first['cpm'][2] == 1, 'cpm FiLM 首步梯度是本协议证据的一部分'
    assert first['os'][2] >= 2, 'os 首步为 0 初始化起步，激活晚于第 1 步'
    assert first['encoder'][2] >= 2


# ----------------------------------------------------------- 7. 上限语义
def test_one_step_budget_is_a_different_conclusion_from_activated(mdd_stats):
    """``max_steps=1`` 必须给出 ``not activated within 1 steps``，不能与「已激活」合并。"""
    out, probe, _model = _heads_check(mdd_stats, 'cat', max_steps=1)
    rows = out['buckets']
    assert out['steps_observed'] == 1
    assert out['all_activated'] is False and out['activated_within_budget'] is False
    assert rows[BUCKET_CPM]['status'] == STATUS_ACTIVATED, 'cpm 首步就有非零梯度'
    assert rows[BUCKET_CPM]['outcome'] == 'activated at step 1'

    for key in (BUCKET_OS, BUCKET_ENCODER):
        row = rows[key]
        # 在图中、但梯度恰为 0 ⇒ never_activated（不是 no_grad_at_all，也不是 activated）
        assert row['status'] == STATUS_NEVER, format_summary(out)
        assert row['outcome'] == 'not activated within 1 steps'
        assert row['outcome'] != rows[BUCKET_CPM]['outcome']
        assert row['first_nonzero_grad_step'] is None
        assert row['first_param_update_step'] is None
        assert row['first_output_change_step'] is None
        assert row['output_change_witness'] == 'no_change'
        assert row['params_without_grad_object'] == []
        assert row['zero_grad_steps'] == row['n_params'], '本步每个条件参数都拿到恰为 0 的梯度'
        assert row['trainable'] == row['n_params'] and row['in_optimizer'] is True

    with pytest.raises(AssertionError) as exc:
        probe.assert_activated(1, ('os', 'cpm', 'encoder'))
    text = str(exc.value)
    assert 'not activated within 1 steps' in text
    assert '实际只观察' not in text, (
        f'跑满 1 步预算就是终局结论，不该被写成观察不足：{text.splitlines()[:2]}')
    assert BUCKET_OS in text and BUCKET_ENCODER in text
    problem_lines = [ln.strip().lstrip('- ').strip()
                     for ln in text.splitlines() if ln.strip().startswith('- ')]
    problem_buckets = [ln.split(':', 1)[0] for ln in problem_lines]
    assert sorted(problem_buckets) == sorted([BUCKET_OS, BUCKET_ENCODER]), (
        f'未激活的桶应被逐条点名，且已激活的 cpm 不得出现在名单里：{problem_lines}')

    # 「跑满了要求的步数」与「看得比要求的少」必须是两种文案：同一份证据按 200 步要求
    # 重述时，要明说只观察了 1 步，而不是假装 200 步都没激活
    with pytest.raises(AssertionError) as under:
        probe.assert_activated(ACTIVATION_STEP_CAP, ('os', 'cpm', 'encoder'))
    under_text = str(under.value)
    assert 'not activated within 1 steps' in under_text
    assert '实际只观察 1 步' in under_text, under_text.splitlines()[:2]

    # 同一批证据在协议上限内重跑就是「已激活」：两种结论必须可区分
    wide, probe2, _ = _heads_check(mdd_stats, 'cat')
    probe2.assert_activated(ACTIVATION_STEP_CAP, ('os', 'cpm', 'encoder'))
    assert wide['all_activated'] is True


def test_full_budget_negative_conclusion_is_not_framed_as_under_observation():
    """跑满协议上限的死适配器：结论是「200 步都没激活」，不能被写成「只观察了 200 步」。

    ``not activated within 200 steps``（接线问题）与「观察步数不足」（再多跑几步看看）
    的修复动作完全不同；若两条共用同一句文案，报告本身就会把前者说成后者。
    """
    model, make_batch, loss_fn = _dead_net()
    probe = ActivationProbe(model)
    opt = torch.optim.SGD(list(model.parameters()), lr=1e-2)
    out = run_activation_check(model, make_batch, opt, max_steps=ACTIVATION_STEP_CAP,
                               loss_fn=loss_fn, probe=probe, device=DEVICE,
                               expected_buckets=(BUCKET_OS,))
    assert out['steps_observed'] == ACTIVATION_STEP_CAP
    assert out['buckets'][BUCKET_OS]['status'] == STATUS_NO_GRAD
    with pytest.raises(AssertionError) as exc:
        probe.assert_activated(ACTIVATION_STEP_CAP, (BUCKET_OS,))
    text = str(exc.value)
    assert f'not activated within {ACTIVATION_STEP_CAP} steps' in text
    assert '实际只观察' not in text, text.splitlines()[:2]


def test_zero_grad_and_no_grad_object_are_reported_as_two_statuses(mdd_stats):
    """「在图中但梯度恰为 0」与「从未进入反向图」必须是两种状态（协议 §五 (b)）。

    同一套真实模型、只换批次里有没有条件张量，就能把两条口径同时跑出来：前者是
    零初始化起步（等几步就会激活），后者是根本没接上（等多久都不该激活）。
    """
    zero, _p1, _m1 = _heads_check(mdd_stats, 'cat', max_steps=1)
    dead, _p2, model_off = _heads_check(mdd_stats, 'cat', max_steps=5,
                                        with_condition=False,
                                        expected_buckets=('os', 'cpm', 'encoder'))

    def off_graph_names(row):
        return sorted(n for n, _ in model_off.named_parameters()
                      if any(n.startswith(m + '.') for m in row['modules']))

    # os/encoder：在图中、首步梯度恰为 0 ⇒ never_activated（不是 no_grad_at_all）
    for key in (BUCKET_OS, BUCKET_ENCODER):
        in_graph = zero['buckets'][key]
        off_graph = dead['buckets'][key]
        assert in_graph['status'] == STATUS_NEVER, format_summary(zero)
        assert off_graph['status'] == STATUS_NO_GRAD, format_summary(dead)
        assert in_graph['params_without_grad_object'] == []
        assert in_graph['zero_grad_steps'] == in_graph['n_params'], '恰为 0 的梯度要被记下来'
        assert off_graph['zero_grad_steps'] == 0
        assert sorted(off_graph['params_without_grad_object']) == off_graph_names(off_graph), (
            f'{key}: 未进图的参数必须被点名：{off_graph}')
        assert off_graph['grad_max_seen'] == 0.0 and in_graph['grad_max_seen'] == 0.0
        assert in_graph['status'] != off_graph['status']
        assert in_graph['outcome'].startswith('not activated within')
        assert off_graph['outcome'].startswith('not activated within')

    # cpm FiLM 首步就有非零梯度：同一个桶在「没喂条件」的模型里必须判成没接上，
    # 而不是「接上了但没学起来」——两条结论的修复动作完全不同
    cpm_in, cpm_off = zero['buckets'][BUCKET_CPM], dead['buckets'][BUCKET_CPM]
    assert cpm_in['status'] == STATUS_ACTIVATED and cpm_in['first_nonzero_grad_step'] == 1
    assert cpm_off['status'] == STATUS_NO_GRAD
    assert cpm_off['params_without_grad_object'] == off_graph_names(cpm_off)
    assert model_off.one_step_head.film is not None, '对照组模型必须真的建了适配器'
    assert STATUS_NEVER != STATUS_NO_GRAD
    assert set(STATUSES) == {STATUS_ACTIVATED, STATUS_NEVER, STATUS_NO_GRAD,
                             STATUS_NOT_BUILT}
