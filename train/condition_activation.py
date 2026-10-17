# coding=utf-8
"""条件模块「真实激活」诊断记录器（方案 §14 验收 L 的合成激活检查）。

为什么需要这个模块（约束来源）
------------------------------
条件注入的全部收益都取决于「条件通路是否真的参与了这次训练」。只看参数量是否登记、
或只看**单一步**的梯度，都会得出错误结论：本仓库的条件构件大量使用零初始化末层
（:class:`~models.conditioning.DeltaEncoder`、:class:`ConditionNormGenerator`、
:class:`~models.tfm.ConditionFiLM`），再加上 One-Step 头零初始化的 ``decode``，构成一条
真实的多级起步链路。实测 ``mdd8/cat`` + ``inject='heads'`` 的小模型：

- ``step 1``：双头 FiLM 权重为 0 ⇒ :math:`\\partial L/\\partial c = 0` ⇒ 整个条件编码器
  （含 ``DeltaEncoder`` 末层）梯度恰为 0；OS 头的 ``decode`` 也是 0 初始化 ⇒ OS FiLM
  首步梯度同为 0。只有 CPM FiLM 首步拿到非零梯度。
- ``step 2``：FiLM 已被更新 ⇒ 编码器与 OS FiLM 首次出现非零梯度；``delta.net.0``
  （增量输入侧）要到 ``step 3``，因为它的梯度被**自身**零初始化末层挡住，需末层先更新。

因此判定「模块是否激活」必须满足三条：(a) 观察窗口有硬上限（200 步；超过即记为
``not activated within N steps``，与 ``activated at step k`` 是两种不同结论，不能靠多等
几步来模糊）；(b) 区分「梯度对象不存在」（模块从未进入反向图 = 真的没接上）与「梯度
恰为 0」（在图中，但被零初始化挡住）；(c) 参数值是否**真的被优化器改写**，并以固定
探针输入下的**输出变化**（output-change witness）作为比参数梯度更强的功能性证据。

设计与边界
----------
- 只读观察：不重建模型、不改优化器状态；witness 在 ``torch.no_grad()`` 下评估，且只对
  **不含 Dropout** 的模块启用（含 Dropout 时固定输入的输出本身随机，会把噪声误报成
  激活）。
- 「参数更新」按张量**值**比较，不比张量对象身份：``foreach`` 类优化器路径会替换
  ``p.data``，用身份判断会把真实更新误判为未更新。
- 归桶不做静默兜底：判定不了所属头的条件模块直接报错并点名，否则「OS 头与 CPM 头
  分开记录」这一协议要求会被悄悄破坏。
"""
from __future__ import annotations

import operator
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from models.conditioning import (ConditionEncoder, ConditionNormGenerator,
                                 ConditionedLowRankLinear,
                                 ConditionalGatedResidual, DeltaEncoder,
                                 DemoEncoder)
from models.tfm import CPMHorizonHead, ConditionFiLM, OneStepDynamicsHead
from train.optim import conditioning_module_prefixes

#: 协议对合成激活检查的硬上限（超过即判「未激活」，不是「再等几步看看」）
ACTIVATION_STEP_CAP = 200

#: 分桶键（协议要求 OS 头与 CPM 头**分开**记录）
BUCKET_OS = 'os'
BUCKET_CPM = 'cpm'
BUCKET_BACKBONE = 'backbone'
BUCKET_ENCODER = 'encoder'
BUCKETS: Tuple[str, ...] = (BUCKET_OS, BUCKET_CPM, BUCKET_BACKBONE, BUCKET_ENCODER)
#: 协议文本别名 → 桶键（``shared/encoder`` 与 ``encoder`` 同义）
BUCKET_ALIASES: Dict[str, str] = {
    'shared/encoder': BUCKET_ENCODER,
    'shared_encoder': BUCKET_ENCODER,
    'condition_encoder': BUCKET_ENCODER,
    'one_step': BUCKET_OS,
    'cpm_query': BUCKET_CPM,
    'ada_rms_norm': BUCKET_BACKBONE,
}

STATUS_ACTIVATED = 'activated'
STATUS_NEVER = 'never_activated'
STATUS_NO_GRAD = 'no_grad_at_all'
STATUS_NOT_BUILT = 'not_built'
STATUSES: Tuple[str, ...] = (STATUS_ACTIVATED, STATUS_NEVER, STATUS_NO_GRAD,
                             STATUS_NOT_BUILT)

#: ``NeuroTwinTFM`` 的真实顶层属性名（见 models/tfm.py）；显式 modules 映射按此判头
OS_HEAD_ATTRS = ('one_step_head',)
CPM_HEAD_ATTRS = ('cpm_head',)
BACKBONE_ATTRS = ('cond_norm',)
ENCODER_ATTRS = ('conditioner', 'pathology_normalizer')

#: 探针输入的 batch 维（条件张量 batch 为 0 会让广播与统计退化）
DEFAULT_PROBE_BATCH = 4
#: 与模型 seed / split seed / Condition Dropout seed 分离的探针输入种子
DEFAULT_PROBE_SEED = 20261008

#: 默认 ``loss_fn`` 从批次读取的目标键；其余键全部作为模型 forward 的 kwargs
TARGET_KEYS = frozenset({
    'y', 'target', 'cpm_target', 'cpm_mask', 'future', 'future_mask', 'target_mask',
})


def normalize_bucket_name(name: str) -> str:
    """把协议别名归一到 :data:`BUCKETS`；未知桶名直接报错（不按子串猜桶）。"""
    key = str(name)
    if key in BUCKETS:
        return key
    alias = BUCKET_ALIASES.get(key)
    if alias is not None:
        return alias
    raise ValueError(
        f'未知条件桶 {key!r}；可用桶={BUCKETS}，可用别名={sorted(BUCKET_ALIASES)}')


def _classify_adapter(name: str, module: nn.Module,
                      root_child: Optional[nn.Module]) -> str:
    """按**类型优先、再按所属顶层构件**判定条件模块的桶。

    类型规则先于头规则：``DemoEncoder``/``DeltaEncoder`` 无论挂在哪一层都属于条件编码
    语义，若先按名字归桶会把同类构件拆进不同桶，「编码器是否激活」就失去可比口径。
    """
    if isinstance(module, ConditionNormGenerator):
        return BUCKET_BACKBONE
    if isinstance(module, (ConditionEncoder, DemoEncoder, DeltaEncoder)):
        return BUCKET_ENCODER
    if isinstance(root_child, OneStepDynamicsHead):
        return BUCKET_OS
    if isinstance(root_child, CPMHorizonHead):
        return BUCKET_CPM
    head = name.split('.', 1)[0]
    if head in OS_HEAD_ATTRS:
        return BUCKET_OS
    if head in CPM_HEAD_ATTRS:
        return BUCKET_CPM
    if head in BACKBONE_ATTRS:
        return BUCKET_BACKBONE
    if head in ENCODER_ATTRS:
        return BUCKET_ENCODER
    raise ValueError(
        f'无法判定条件模块 {name!r}（{type(module).__name__}）属于哪个头/桶：它既不是'
        f'已知条件类型，也不挂在 {list(OS_HEAD_ATTRS + CPM_HEAD_ATTRS + BACKBONE_ATTRS)}'
        f' 之下。请修正模型属性名，或用 ActivationProbe(modules=...) 显式命名；'
        f'静默归桶会让 per-head 激活统计凭空多算或少算。')


def _param_device(module: nn.Module) -> torch.device:
    """模块参数所在设备（无参数时退回 cpu）；不硬编码任何 CUDA 下标。"""
    for p in module.parameters(recurse=True):
        return p.device
    return torch.device('cpu')


def _has_stochastic_submodule(module: nn.Module) -> bool:
    """模块内是否含 Dropout：含则 output-change witness 不可用。

    固定输入下 Dropout 仍逐次改变输出，witness 会把随机噪声报成「训练改变了输出」，
    产生**假阳性激活** —— 这正是本模块要避免的那类错误结论。
    """
    return any(isinstance(sub, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))
               for sub in module.modules())


def _probe_input(module: nn.Module, *, parent: Optional[nn.Module], batch: int,
                 seed: int) -> Optional[Tuple[tuple, dict]]:
    """为可独立调用的条件构件构造**固定**探针输入；无法独立调用则返回 ``None``。

    宽度全部取自模块自己登记的维度属性（``cond_dim``/``dim``/``in_dim``），不从全局
    配置或 CLI 猜；需要模块自身无法确定的输入（MoE 的 ``z_g``/epoch 语义等）时返回
    ``None``，由调用方记为 witness ``unavailable``。
    """
    device = _param_device(module)

    def randn(*shape: int, key: int = 0) -> torch.Tensor:
        gen = torch.Generator(device='cpu').manual_seed(int(seed) + int(key))
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device)

    if isinstance(module, (ConditionFiLM, ConditionNormGenerator)):
        return ((randn(batch, module.cond_dim, key=1),), {})
    if isinstance(module, ConditionalGatedResidual):
        # ROI 数取 batch 只是占位：该模块对 F 轴无长度约束
        return ((randn(batch, batch, module.dim, key=2),
                 randn(batch, module.cond_dim, key=3)), {})
    if isinstance(module, ConditionedLowRankLinear):
        return ((randn(batch, batch, module.base.in_features, key=4),
                 randn(batch, module.cond_dim, key=5)), {})
    if isinstance(module, ConditionEncoder):
        return _encoder_probe(module, randn, batch, device)
    if isinstance(module, (DemoEncoder, DeltaEncoder)):
        return _leaf_encoder_probe(module, parent, randn, batch, device)
    return None


def _encoder_probe(enc: ConditionEncoder, randn, batch: int,
                   device: torch.device) -> Optional[Tuple[tuple, dict]]:
    """``ConditionEncoder(z, codes, mask)``：按自身 schema/词表造合法输入。

    类别索引必须落在该字段独立 Embedding 的行范围 ``[0, vocab_size]``（含专属 Unknown
    行 0），越界会被 :class:`~models.conditioning.CategoricalFieldEncoder` 判为协议
    错误 —— 探针不该触发它。``epoch`` 留 ``None`` 使 Condition Dropout 恒等返回，固定
    输入才有固定输出。
    """
    if not enc.enabled:
        return None
    k = len(enc.specs)
    z = randn(batch, k, key=11)
    codes = torch.zeros(batch, k, dtype=torch.long, device=device)
    for j, spec in enumerate(enc.specs):
        if spec.is_continuous:
            continue
        n_rows = int(enc.stats.get(spec.name).vocab_size()) + 1
        gen = torch.Generator(device='cpu').manual_seed(1000 + int(j))
        codes[:, j] = torch.randint(0, n_rows, (batch,), generator=gen).to(device)
    mask = torch.ones(batch, k, dtype=torch.float32, device=device)
    return ((z, codes, mask), {})


def _leaf_encoder_probe(module: nn.Module, parent: Optional[nn.Module], randn,
                        batch: int, device: torch.device):
    """``DemoEncoder``/``DeltaEncoder`` 的输入列数只登记在父编码器里，需要父上下文。

    单看这两个类无法从 ``in_dim`` 反推「几路特征、每路多宽」（``cat/sum/mean`` 的
    :math:`\\Phi_f` 宽度不同），因此必须由父 :class:`ConditionEncoder` 给出分组；
    父不可用时返回 ``None``（记 unavailable），不猜列数。
    """
    if not isinstance(parent, ConditionEncoder):
        return None
    c_dim = int(parent.cfg.cond_dim)
    if isinstance(module, DemoEncoder):
        if parent.demo is not module:
            return None
        feats = [randn(batch, parent.features.dim_of(n), key=20 + i)
                 for i, n in enumerate(parent.demo_group)]
        mask = torch.ones(batch, len(parent.demo_group), dtype=torch.float32,
                          device=device)
        return ((feats, mask), {})
    if isinstance(module, DeltaEncoder):
        if parent.delta is not module:
            return None
        n_new = len(parent.new_idx)
        phi_w = int(module.in_dim) - c_dim - n_new
        if phi_w < 1:
            raise ValueError(
                f'{type(module).__name__}: in_dim={module.in_dim} 与父编码器登记的 '
                f'C={c_dim}、新增组列数={n_new} 不自洽（推出 Phi 宽度 {phi_w}）；'
                f'说明融合算子或字段组与探针假设不符')
        mask = torch.ones(batch, n_new, dtype=torch.float32, device=device)
        return ((randn(batch, c_dim, key=31), randn(batch, phi_w, key=32), mask), {})
    return None


def _foreach_max_abs_diff(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]
                          ) -> List[float]:
    """逐张量 ``max|cur - base|``，按 (dtype, device) 分组做 foreach 融合。

    用 ``torch._foreach_*`` 而不是逐参数 Python 循环：这个比较每步都要覆盖全部条件
    参数，融合成少量 kernel 才能保证 200 步预算内诊断开销可忽略（否则记录器自身会
    拖慢检查，诱导调用方减小 max_steps）。
    """
    out: List[float] = [0.0] * len(pairs)
    groups: Dict[Tuple[torch.dtype, str], List[int]] = {}
    for i, (cur, base) in enumerate(pairs):
        if not cur.is_floating_point():
            raise ValueError(
                f'条件参数必须是浮点张量才能判定「值是否被优化器改变」，第 {i} 个'
                f'张量 dtype={cur.dtype} shape={tuple(cur.shape)}')
        groups.setdefault((cur.dtype, str(cur.device)), []).append(i)
    for idxs in groups.values():
        sub = torch._foreach_sub([pairs[i][0] for i in idxs],
                                 [pairs[i][1] for i in idxs])
        for i, d in zip(idxs, torch._foreach_abs(sub)):
            out[i] = float(d.max()) if d.numel() else 0.0
    return out


class _Bucket:
    """单个桶的累计观察状态（协议要记录的是「首次」，不是「某一步」）。"""

    def __init__(self, key: str) -> None:
        self.key = key
        self.modules: List[str] = []
        self.param_names: List[str] = []
        self.trainable_names: Set[str] = set()
        #: 每个参数是否出现过非 None 的 grad 对象（区分「不在图中」与「梯度恰为 0」）
        self.grad_seen: Dict[str, bool] = {}
        self.param_first_grad: Dict[str, int] = {}
        self.first_grad_step: Optional[int] = None
        self.first_grad_max: Optional[float] = None
        self.grad_max_seen: float = 0.0
        self.first_update_step: Optional[int] = None
        self.first_update_max: Optional[float] = None
        self.first_witness_step: Optional[int] = None
        self.witness_available: List[str] = []
        self.witness_unavailable: List[str] = []
        self.zero_grad_steps: int = 0

    @property
    def built(self) -> bool:
        return bool(self.param_names) or bool(self.modules)

    def note_grad(self, name: str, step: int, magnitude: float) -> None:
        """记录一次梯度观察；``first_grad_max`` 只取**首步当时**的量级。

        不再用后续步的更大量级覆盖它：摘要里 first_grad_step 与 first_grad_max 必须
        描述同一件事，否则「首步梯度多大」会被后来的步数解释成别的东西。
        """
        self.grad_seen[name] = True
        self.grad_max_seen = max(self.grad_max_seen, magnitude)
        if magnitude <= 0.0:
            self.zero_grad_steps += 1
            return
        self.param_first_grad.setdefault(name, step)
        if self.first_grad_step is None:
            self.first_grad_step = step
            self.first_grad_max = magnitude
        elif step == self.first_grad_step:
            self.first_grad_max = max(self.first_grad_max or 0.0, magnitude)

    def note_update(self, step: int, magnitude: float) -> None:
        if magnitude <= 0.0:
            return
        if self.first_update_step is None:
            self.first_update_step = step
            self.first_update_max = magnitude
        elif step == self.first_update_step:
            self.first_update_max = max(self.first_update_max or 0.0, magnitude)

    def to_dict(self, steps_observed: int, in_optimizer: Mapping[str, bool]) -> dict:
        """渲染协议规定字段，并把「未激活」写成带步数上限的独立结论。

        ``no_grad_at_all`` 与 ``never_activated`` 必须分开：前者是「从未被反向图触及」
        （模块根本没接上），后者是「梯度对象存在但恒为 0」（在图中，却被零初始化或
        冻结挡住）；两者的修复动作完全不同，合并成一个状态就会误导结论。
        """
        if not self.built:
            status = STATUS_NOT_BUILT
        elif (self.first_grad_step is not None or self.first_update_step is not None
              or self.first_witness_step is not None):
            status = STATUS_ACTIVATED
        elif any(self.grad_seen.values()):
            status = STATUS_NEVER
        else:
            status = STATUS_NO_GRAD
        if status == STATUS_ACTIVATED:
            firsts = [s for s in (self.first_grad_step, self.first_update_step,
                                  self.first_witness_step) if s is not None]
            outcome = f'activated at step {min(firsts)}'
        elif status == STATUS_NOT_BUILT:
            outcome = 'not built by this config'
        else:
            outcome = f'not activated within {steps_observed} steps'
        return {
            'status': status,
            'outcome': outcome,
            'first_nonzero_grad_step': self.first_grad_step,
            'first_grad_max': self.first_grad_max,
            'grad_max_seen': self.grad_max_seen,
            'first_param_update_step': self.first_update_step,
            'first_param_update_max': self.first_update_max,
            'first_output_change_step': self.first_witness_step,
            'output_change_witness': ('unavailable' if not self.witness_available
                                      else ('witnessed'
                                            if self.first_witness_step is not None
                                            else 'no_change')),
            'n_params': len(self.param_names),
            'trainable': len(self.trainable_names),
            'in_optimizer': (bool(self.param_names)
                             and all(in_optimizer.get(n, False)
                                     for n in self.param_names)),
            'modules': list(self.modules),
            'witness_available': list(self.witness_available),
            'witness_unavailable': list(self.witness_unavailable),
            'zero_grad_steps': self.zero_grad_steps,
            'params_without_grad_object': sorted(
                n for n, seen in self.grad_seen.items() if not seen),
            # 逐参证据：零初始化链路里「输入侧晚于末层」只有逐参才看得见
            'param_first_nonzero_grad_step': dict(sorted(self.param_first_grad.items())),
        }


class ActivationProbe:
    """按 OS / CPM / backbone / encoder 四桶记录条件模块的真实激活。

    构造方式二选一：

    - ``ActivationProbe(model)``：从 ``model.named_modules()`` 按
      ``is_conditioning_adapter`` 发现条件模块（并与
      :func:`train.optim.conditioning_module_prefixes` 显式核对口径一致），桶归属由
      **顶层构件的类型**决定：``OneStepDynamicsHead`` 下的适配器 → ``os``，
      ``CPMHorizonHead`` 下的 → ``cpm``，:class:`ConditionNormGenerator` →
      ``backbone``，:class:`ConditionEncoder` 及其 ``demo``/``delta`` → ``encoder``。
    - ``ActivationProbe(modules={'one_step_head.film': film, ...})``：显式名字→模块
      映射，用于隔离测试或自定义容器；名字必须能判定所属头，否则报错。

    参数按**最长适配器前缀**唯一归属到一个桶，因此 ``conditioner`` 与其嵌套的
    ``conditioner.demo`` / ``conditioner.delta`` 不会重复计数（三者同属 encoder 桶，
    但每个参数只统计一次）。
    """

    def __init__(self, model: Optional[nn.Module] = None, *,
                 modules: Optional[Mapping[str, nn.Module]] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 probe_inputs: Optional[Mapping[str, Tuple[tuple, dict]]] = None,
                 enable_witness: bool = True,
                 probe_batch: int = DEFAULT_PROBE_BATCH,
                 probe_seed: int = DEFAULT_PROBE_SEED) -> None:
        if (model is None) == (modules is None):
            raise ValueError(
                'ActivationProbe 需要且只需要一个来源：model 或 modules；收到 '
                f'model={type(model).__name__}, modules={type(modules).__name__}')
        if modules is not None and not isinstance(modules, Mapping):
            raise TypeError(
                f'modules 必须是 模块名→nn.Module 的映射，收到 {type(modules).__name__}')
        if int(probe_batch) < 1:
            raise ValueError(f'probe_batch 必须 >= 1，收到 {probe_batch}')
        self.model = model
        self.optimizer = optimizer
        self.steps_observed = 0
        self.last_step: Optional[int] = None
        self.required_buckets: List[str] = []
        self._snapshot: Dict[str, torch.Tensor] = {}
        self._last_seen: Dict[str, torch.Tensor] = {}
        self._params: Dict[str, nn.Parameter] = {}
        self._buckets: Dict[str, _Bucket] = {b: _Bucket(b) for b in BUCKETS}
        self._module_bucket: Dict[str, _Bucket] = {}
        self._witness: Dict[str, Tuple[nn.Module, tuple, dict, object]] = {}

        adapters = self._discover(model, modules)
        self._assign_params(adapters, model)
        if enable_witness:
            self._build_witness(adapters, dict(probe_inputs or {}),
                                batch=int(probe_batch), seed=int(probe_seed))

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _discover(self, model: Optional[nn.Module],
                  modules: Optional[Mapping[str, nn.Module]]) -> Dict[str, nn.Module]:
        """按 ``is_conditioning_adapter`` 发现条件模块并归桶（不做静默归桶）。"""
        out: Dict[str, nn.Module] = {}
        if model is not None:
            children = dict(model.named_children())
            for name, module in model.named_modules():
                if not name or not getattr(module, 'is_conditioning_adapter', False):
                    continue
                out[name] = module
                self._register(name, module, children.get(name.split('.', 1)[0]))
        else:
            unflagged = [str(n) for n, m in modules.items()
                         if not getattr(m, 'is_conditioning_adapter', False)]
            if unflagged:
                raise ValueError(
                    f'modules 中这些对象没有 is_conditioning_adapter 标记：'
                    f'{unflagged[:5]}（共 {len(unflagged)} 个）；显式映射也必须走同一'
                    f'发现口径，否则 per-head 统计与 train/optim 的冻结/参数分组会分叉。')
            for name, module in modules.items():
                out[str(name)] = module
                self._register(str(name), module, None)
        if not out:
            raise ValueError(
                '未发现任何条件模块（is_conditioning_adapter）。若本次配置确实是 '
                "arm='none'/条件关闭，请显式构造带 modules= 的探针并按 "
                f'{STATUS_NOT_BUILT} 解读，不要让空探针产出一份「全部未激活」的假指标。')
        return out

    def _register(self, name: str, module: nn.Module,
                  root_child: Optional[nn.Module]) -> None:
        bucket = self._buckets[_classify_adapter(name, module, root_child)]
        bucket.modules.append(name)
        self._module_bucket[name] = bucket

    def _assign_params(self, adapters: Dict[str, nn.Module],
                       model: Optional[nn.Module]) -> None:
        """参数 → 桶：取最长适配器前缀，保证每个参数只属于一个桶。"""
        prefixes = sorted(((n, n + '.') for n in adapters), key=lambda kv: -len(kv[1]))
        pairs: List[Tuple[str, nn.Parameter]] = []
        if model is not None:
            pairs = list(model.named_parameters())
            expected = set(conditioning_module_prefixes(model))
            mine = {pre for _, pre in prefixes}
            if mine != expected:
                raise ValueError(
                    '条件模块前缀与 train/optim 的登记口径不一致：探针多出 '
                    f'{sorted(mine - expected)}、缺少 {sorted(expected - mine)}；'
                    f'口径分叉会让「是否入优化器」的判断失真。')
        else:
            for name, module in adapters.items():
                for sub, p in module.named_parameters():
                    pairs.append((f'{name}.{sub}', p))
        for pname, p in pairs:
            owner = next((n for n, pre in prefixes if pname.startswith(pre)), None)
            if owner is None:
                continue                      # 非条件参数（主干与头部的无条件部件）
            if pname in self._params:
                raise ValueError(
                    f'参数 {pname} 被多个条件模块同时持有，无法唯一归桶；共享适配器需要'
                    f'显式声明所属头（否则 per-head 计数会重复）。')
            bucket = self._module_bucket[owner]
            self._params[pname] = p
            bucket.param_names.append(pname)
            bucket.grad_seen[pname] = False
            if p.requires_grad:
                bucket.trainable_names.add(pname)
        for bucket in self._buckets.values():
            bucket.param_names.sort()

    # ------------------------------------------------------------------
    # output-change witness
    # ------------------------------------------------------------------
    def _build_witness(self, adapters: Dict[str, nn.Module],
                       explicit: Dict[str, Tuple[tuple, dict]], *, batch: int,
                       seed: int) -> None:
        """登记固定探针输入并快照起点（step 0）输出。"""
        extra = sorted(set(explicit) - set(adapters))
        if extra:
            raise ValueError(f'probe_inputs 含未发现的模块名：{extra}')
        for i, (name, module) in enumerate(sorted(adapters.items())):
            bucket = self._module_bucket[name]
            spec = explicit.get(name)
            if spec is None:
                if _has_stochastic_submodule(module):
                    bucket.witness_unavailable.append(name)
                    continue
                parent_name = name.rsplit('.', 1)[0] if '.' in name else None
                spec = _probe_input(
                    module,
                    parent=adapters.get(parent_name) if parent_name else None,
                    batch=batch, seed=seed + 7919 * i)
            if spec is None:
                bucket.witness_unavailable.append(name)
                continue
            args, kwargs = spec
            with torch.no_grad():
                raw = module(*args, **kwargs)
            if raw is None:
                # 例如 arm='none' 的编码器：绕过条件通路时本就没有输出可比
                bucket.witness_unavailable.append(name)
                continue
            self._witness[name] = (module, args, kwargs,
                                   _clone_outputs(raw, name))
            bucket.witness_available.append(name)

    # ------------------------------------------------------------------
    # 观察
    # ------------------------------------------------------------------
    def attach_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        """绑定优化器，使 ``in_optimizer`` 可判定（未绑定时如实报 False）。"""
        self.optimizer = optimizer

    def note_step_boundary(self, model: Optional[nn.Module] = None) -> None:
        """在进入下一次 train step **之前**快照条件参数值。

        只有与「步前值」比较才能把「优化器真实改写了参数」与「调用方手动重新初始化/
        载入权重」区分开；只看 ``p.grad`` 非零不能证明更新落地（梯度裁剪、缩放跳过
        step、LR=0 都会让梯度非零而参数不变）。不调用本方法时 :meth:`record_step`
        退回「与上一次观察比较」，代价是第一个观察步没有基线（更新晚一步才被看到）。
        """
        if model is not None and self.model is not None and model is not self.model:
            raise ValueError(
                'note_step_boundary 收到的 model 与探针绑定的 model 不是同一对象；'
                '跨模型比较参数值会把「未更新」误报为「已更新」。')
        if model is not None and self.model is None:
            ids = {id(p) for p in model.parameters()}
            foreign = [n for n, p in self._params.items() if id(p) not in ids]
            if foreign:
                raise ValueError(
                    f'{len(foreign)} 个被探测参数不属于传入的 model'
                    f'（例如 {foreign[:3]}）；快照必须与参数对象同源。')
        self._snapshot = {n: p.detach().clone() for n, p in self._params.items()}

    def record_step(self, step: int) -> None:
        """记录本步的梯度事实、参数变化与输出见证（在 backward + step 之后调用）。

        「参数变化」的基线是**最近一次** :meth:`note_step_boundary` 的快照，因此多次
        记录之间没有新边界时，比较的是「自上次边界以来是否动过」——这是保守且可解释
        的口径，不会把梯度非零误当成参数已更新。
        """
        step = operator.index(step)
        if step < 0:
            raise ValueError(f'step 必须 >= 0（1 基训练步号），收到 {step}')
        if self.last_step is not None and step <= self.last_step:
            raise ValueError(
                f'step 必须严格递增（本次 {step} <= 上次 {self.last_step}）；'
                f'重复或回退会让「首次非零梯度」失去定义。')
        self.last_step = step
        self.steps_observed = max(self.steps_observed, step)
        for bucket in self._buckets.values():
            if not bucket.param_names:
                continue
            snap: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for name in bucket.param_names:
                p = self._params[name]
                if p.grad is not None:
                    mag = float(p.grad.detach().abs().max()) if p.grad.numel() else 0.0
                    bucket.note_grad(name, step, mag)
                # 基线优先取「步前快照」（note_step_boundary），没有则退回上一次观察：
                # 退回口径下第一个观察步没有基线（更新要晚一步才被看到），不静默漏检
                # 但精度更低，所以训练循环应当每步先调用 note_step_boundary。
                base = self._snapshot.get(name, self._last_seen.get(name))
                if base is not None:
                    snap.append((p.detach(), base))
                self._last_seen[name] = p.detach().clone()
            if snap:
                for d in _foreach_max_abs_diff(snap):
                    bucket.note_update(step, d)
        for name, (module, args, kwargs, baseline) in self._witness.items():
            with torch.no_grad():
                now = module(*args, **kwargs)
            if _diff_norm(now, baseline, name) > 0.0:
                bucket = self._module_bucket[name]
                if bucket.first_witness_step is None:
                    bucket.first_witness_step = step

    def param_bucket(self, param_name: str) -> str:
        """参数所属桶（供外部按桶核对/报告）。"""
        for bucket in self._buckets.values():
            if param_name in bucket.param_names:
                return bucket.key
        raise KeyError(
            f'参数 {param_name!r} 不属于任何条件桶（共 {len(self._params)} 个被探测参数）')

    def all_activated(self, buckets: Optional[Sequence[str]] = None) -> bool:
        """当前为止，指定桶（默认被要求的桶）是否都已出现激活证据。"""
        rows = self.summary()['buckets']
        keys = list(buckets) if buckets is not None else self.required_buckets
        return all(rows[k]['status'] == STATUS_ACTIVATED for k in keys)

    # ------------------------------------------------------------------
    # 汇报
    # ------------------------------------------------------------------
    def require(self, expected_buckets: Optional[Iterable[str]] = None) -> List[str]:
        """登记「本次配置真正建立了哪些桶」；``None`` = 按已建成的桶要求。

        ``inject='heads'`` 的运行没有 backbone 桶：把它列进期望就是**假失败**；反过来
        在 ``heads+backbone`` 下漏列又会掩盖真失败，所以这个集合必须由调用方给出。
        """
        if expected_buckets is None:
            self.required_buckets = [b for b in BUCKETS if self._buckets[b].built]
        else:
            names = [normalize_bucket_name(b) for b in expected_buckets]
            self.required_buckets = sorted(set(names))
        if not self.required_buckets:
            raise ValueError(
                'expected_buckets 归一后为空：没有被要求的桶就无法判定激活。请显式列出'
                '本次配置真正建立的条件桶（或传 None 用已建成的桶）。')
        return list(self.required_buckets)

    def summary(self) -> dict:
        """逐桶首次非零梯度 / 首次参数更新 / 输出变化见证 + 结构登记。"""
        opt_ids = {id(p)
                   for g in (self.optimizer.param_groups if self.optimizer else ())
                   for p in g['params']}
        in_opt = {n: (id(p) in opt_ids) for n, p in self._params.items()}
        buckets = {b: self._buckets[b].to_dict(self.steps_observed, in_opt)
                   for b in BUCKETS}
        required = list(self.required_buckets) or [b for b in BUCKETS
                                                   if self._buckets[b].built]
        return {
            'buckets': buckets,
            'required_buckets': required,
            'built_buckets': [b for b in BUCKETS if self._buckets[b].built],
            'steps_observed': self.steps_observed,
            'step_cap': ACTIVATION_STEP_CAP,
            'optimizer_attached': self.optimizer is not None,
            'all_activated': all(buckets[b]['status'] == STATUS_ACTIVATED
                                for b in required),
            'n_probed_params': len(self._params),
        }

    def assert_activated(self, max_steps: int = ACTIVATION_STEP_CAP,
                         expected_buckets: Optional[Iterable[str]] = None) -> None:
        """要求被指定的桶在步数上限内出现激活证据，否则带完整摘要报错。

        「未激活」与「观察步数不足」必须可区分：报错文案会明说实际看了多少步，避免
        把「只跑了 3 步」说成「跑了 200 步都没激活」。
        """
        budget = _validate_step_budget(max_steps)
        required = self.require(expected_buckets) if expected_buckets is not None \
            else (list(self.required_buckets) or
                  [b for b in BUCKETS if self._buckets[b].built])
        summary = self.summary()
        seen = summary['steps_observed']
        problems: List[str] = []
        for b in required:
            row = summary['buckets'][b]
            if row['status'] == STATUS_ACTIVATED:
                continue
            if row['status'] == STATUS_NOT_BUILT:
                problems.append(f'{b}: 本次配置未建立该桶（{STATUS_NOT_BUILT}）')
            else:
                # 只有「看得比要求的少」才需要追加提醒：跑满预算的 not activated within
                # N steps 本身就是协议的终局结论，把它写成「实际只观察 N 步」会让
                # 「跑了 200 步都没激活」和「只跑了 3 步」这两种结论看起来一模一样。
                problems.append(
                    f"{b}: {row['outcome']}"
                    + ('' if seen >= budget else
                       f'（要求观察 >{budget} 步，实际只观察 {seen} 步）'))
        if problems:
            raise AssertionError(
                f'条件模块在 {budget} 步上限内未激活：\n  - '
                + '\n  - '.join(problems) + '\n' + format_summary(summary))


def _clone_outputs(out, name: str):
    """把 witness 起点输出克隆成同结构；不支持的结构报错（不静默跳过）。"""
    if isinstance(out, torch.Tensor):
        return out.detach().clone()
    if isinstance(out, (list, tuple)):
        return [_clone_outputs(o, name) for o in out]
    if isinstance(out, dict):
        return {k: _clone_outputs(v, name) for k, v in out.items()}
    raise TypeError(
        f'模块 {name!r} 的 forward 返回了 witness 无法比较的对象'
        f'（{type(out).__name__}）；这类模块应按 unavailable 记录，请显式传 '
        f'probe_inputs 或 enable_witness=False。')


def _diff_norm(now, base, name: str) -> float:
    """递归计算 witness 输出与起点的最大绝对差（结构必须同形）。"""
    if isinstance(now, torch.Tensor) and isinstance(base, torch.Tensor):
        if tuple(now.shape) != tuple(base.shape):
            raise ValueError(
                f'witness {name!r} 输出形状漂移：{tuple(base.shape)} → '
                f'{tuple(now.shape)}；固定输入下这只可能是结构被改动。')
        return float((now.detach() - base).abs().max()) if now.numel() else 0.0
    if isinstance(now, (list, tuple)) and isinstance(base, (list, tuple)):
        if len(now) != len(base):
            raise ValueError(
                f'witness {name!r} 输出长度漂移：{len(base)} → {len(now)}')
        return max((_diff_norm(a, b, name) for a, b in zip(now, base)), default=0.0)
    if isinstance(now, dict) and isinstance(base, dict):
        if set(now) != set(base):
            raise ValueError(
                f'witness {name!r} 输出键漂移：缺失={sorted(set(base) - set(now))} '
                f'新增={sorted(set(now) - set(base))}')
        return max((_diff_norm(now[k], base[k], name) for k in base), default=0.0)
    raise TypeError(
        f'witness {name!r} 无法比较的输出类型：now={type(now).__name__} '
        f'base={type(base).__name__}')


def _validate_step_budget(max_steps: int) -> int:
    """步数预算必须是**有限**整数且 <= 协议上限（200）。

    ``inf`` / ``nan`` / 浮点 / 字符串都会让「未激活」永远无法被证明，因此一律拒绝，
    而不是放宽成「多跑一会儿再看看」。
    """
    if isinstance(max_steps, bool):
        raise ValueError(f'max_steps 不能是 bool，收到 {max_steps!r}')
    try:
        budget = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError(
            f'max_steps 必须是有限整数（协议上限 {ACTIVATION_STEP_CAP}），收到 '
            f'{type(max_steps).__name__}={max_steps!r}') from exc
    if budget < 1 or budget > ACTIVATION_STEP_CAP:
        raise ValueError(
            f'max_steps={budget} 越界：合成激活检查要求 1 <= max_steps <= '
            f'{ACTIVATION_STEP_CAP}（协议 §14 L 的硬上限）。超过上限的结论必须是 '
            f'"not activated within {ACTIVATION_STEP_CAP} steps"，不允许无限等待。')
    return budget


_DUAL_LOSS_SINGLETON = None


def _default_criterion():
    """默认双头损失的惰性单例（:class:`TFMDualLoss` 无参数，可安全复用）。"""
    global _DUAL_LOSS_SINGLETON
    if _DUAL_LOSS_SINGLETON is None:
        from train.losses import TFMDualLoss
        _DUAL_LOSS_SINGLETON = TFMDualLoss()
    return _DUAL_LOSS_SINGLETON


def dual_loss_from_batch(outputs, batch: Mapping[str, object]) -> torch.Tensor:
    """默认损失：复刻 ``main.py`` 的 ``criterion(outputs, by, aux['cpm_pred'], ...)``。

    只包含 :class:`train.losses.TFMDualLoss` 的双头主干项（λ_one=1.0、λ_cpm=1.0、
    λ_pcc=0.1、cpm_gamma=1.0、huber_delta=1.0，即 CLI 默认值）。SC 图正则、Router 正则
    与 RECC 项**不**包含：它们各由独立 λ 控制，混进默认损失会让不同配置的激活检查
    不可比。需要这些项的调用方必须显式传 ``loss_fn``。

    批次键约定：``y``/``target`` 为 one-step 真值 ``[B,F,H,1]``（``[B,F,H]`` 自动升维），
    ``cpm_target``/``future`` 为 ``[B,F,H]``，``cpm_mask``/``future_mask`` 为 ``[B,H]``。
    """
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise TypeError(
            f'默认 loss_fn 期望模型返回 (pred, aux_info)（NeuroTwinTFM.forward 契约），'
            f'收到 {type(outputs).__name__}')
    pred, aux = outputs
    if not isinstance(aux, dict) or 'cpm_pred' not in aux:
        raise KeyError(
            f'aux_info 缺少 cpm_pred，无法计算 CPM 头损失；收到键='
            f'{sorted(aux) if isinstance(aux, dict) else type(aux).__name__}')
    y = batch.get('y', batch.get('target', None))
    if y is None:
        raise KeyError(f'批次缺少 one-step 目标键 y/target，收到键={sorted(batch)}')
    cpm_target = batch.get('cpm_target', batch.get('future', None))
    if cpm_target is None:
        raise KeyError(f'批次缺少 CPM 目标键 cpm_target/future，收到键={sorted(batch)}')
    cpm_mask = batch.get('cpm_mask', batch.get('future_mask',
                                               batch.get('target_mask', None)))
    if y.dim() == 3:
        y = y.unsqueeze(-1)
    total, _stats = _default_criterion()(pred, y, aux['cpm_pred'], cpm_target, cpm_mask)
    return total


def run_activation_check(model: nn.Module, make_batch, optimizer, *,
                         max_steps: int = ACTIVATION_STEP_CAP,
                         loss_fn=None, forward_kwargs: Optional[Mapping] = None,
                         expected_buckets: Optional[Iterable[str]] = None,
                         device: Optional[torch.device] = None,
                         probe: Optional[ActivationProbe] = None,
                         stop_when_activated: bool = True,
                         probe_inputs: Optional[Mapping[str, Tuple[tuple, dict]]] = None,
                         ) -> dict:
    """在合成批次数上跑 ``<= max_steps`` 个真实 train step，返回逐桶激活摘要。

    ``make_batch(step)``（``step`` 为 1 基整数）返回一个 dict：除 :data:`TARGET_KEYS`
    之外的键全部作为模型 forward 的 kwargs；``loss_fn(outputs, batch)`` 必须返回标量
    张量。本函数**不重建任何模型/数据链路**，因此每一步都与调用方自己的训练循环可比。

    默认 ``loss_fn`` 是 :func:`dual_loss_from_batch`（与 ``main.py`` 同式的 OS+CPM 双头
    损失，不含图正则/RECC）；要与自己主损失完全一致的调用方应显式传 ``loss_fn``。

    需要随后断言时把同一个探针传进来：``run_activation_check(..., probe=p)`` 后调用
    ``p.assert_activated(max_steps, expected_buckets)``。
    """
    budget = _validate_step_budget(max_steps)
    if not callable(make_batch):
        raise TypeError(f'make_batch 必须可调用（make_batch(step) -> dict），'
                        f'收到 {type(make_batch).__name__}')
    if not isinstance(model, nn.Module):
        raise TypeError(f'model 必须是 nn.Module，收到 {type(model).__name__}')
    if not hasattr(optimizer, 'step') or not callable(optimizer.step):
        raise TypeError('optimizer 必须提供可调用的 .step()')
    criterion = loss_fn if loss_fn is not None else dual_loss_from_batch
    if not callable(criterion):
        raise TypeError(f'loss_fn 必须可调用，收到 {type(criterion).__name__}')
    reserved = dict(forward_kwargs or {})
    device = torch.device(device) if device is not None else _param_device(model)
    model.to(device)
    if probe is None:
        probe = ActivationProbe(model, optimizer=optimizer, probe_inputs=probe_inputs)
    else:
        probe.attach_optimizer(optimizer)
    required = probe.require(expected_buckets)

    was_training = model.training
    try:
        model.train()
        for step in range(1, budget + 1):
            batch = _move_batch(make_batch(step), device)
            kwargs = {k: v for k, v in batch.items() if k not in TARGET_KEYS}
            clash = sorted(set(reserved) & set(kwargs))
            if clash:
                raise ValueError(
                    f'forward_kwargs 与 make_batch 的键冲突：{clash}；静默覆盖会让前向'
                    f'到底用谁的输入不可追溯。')
            kwargs.update(reserved)
            # set_to_none=True 是记录口径的前提：只有把梯度置为 None，才能区分
            # 「模块从未进入反向图」与「梯度恰为 0（零初始化起步）」
            optimizer.zero_grad(set_to_none=True)
            probe.note_step_boundary(model)
            outputs = model(**kwargs)
            loss = criterion(outputs, batch)
            if not torch.is_tensor(loss):
                raise TypeError(f'loss_fn 必须返回标量张量，收到 {type(loss).__name__}')
            if loss.ndim != 0:
                raise ValueError(
                    f'loss 必须是 0 维标量，收到 shape={tuple(loss.shape)}；非标量'
                    f'backward 不出梯度。')
            loss.backward()
            optimizer.step()
            probe.record_step(step)
            if stop_when_activated and probe.all_activated(required):
                break
    finally:
        model.train(was_training)
    summary = probe.summary()
    summary['required_buckets'] = required
    summary['activated_within_budget'] = summary['all_activated']
    return summary


def _move_batch(batch: Mapping[str, object], device: torch.device) -> dict:
    """把批次里的张量全部搬到 device；非张量值原样保留。"""
    if not isinstance(batch, Mapping):
        raise TypeError(
            f'make_batch 必须返回 dict/Mapping（forward kwargs + 目标键），收到 '
            f'{type(batch).__name__}')
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def format_summary(summary: Mapping) -> str:
    """把逐桶摘要渲染成可直接贴进日志/报告的紧凑文本（含被要求的桶清单）。"""
    if not isinstance(summary, Mapping) or 'buckets' not in summary:
        raise TypeError(
            'format_summary 需要 ActivationProbe.summary() 的返回值（含 buckets 键），'
            f'收到 {type(summary).__name__}')
    steps = int(summary.get('steps_observed', 0))
    cap = int(summary.get('step_cap', ACTIVATION_STEP_CAP))
    required = list(summary.get('required_buckets') or [])
    lines = [
        f'[condition-activation] steps={steps}/{cap} '
        f'required={",".join(required) or "-"} '
        f'all_activated={bool(summary.get("all_activated"))} '
        f'probed_params={summary.get("n_probed_params", 0)} '
        f'optimizer_attached={bool(summary.get("optimizer_attached"))}',
    ]
    for key in BUCKETS:
        row = summary['buckets'].get(key)
        if row is None:
            continue
        if row['status'] == STATUS_NOT_BUILT and key not in required:
            lines.append(f'  {key:<9} not built by this config')
            continue
        lines.append(
            f"  {key:<9} {row['status']:<15} {row['outcome']}"
            f" | firstGrad={row['first_nonzero_grad_step']}"
            f" max|g|={_fmt(row['first_grad_max'])}"
            f" firstUpdate={row['first_param_update_step']}"
            f" firstOutputChange={row['first_output_change_step']}"
            f" witness={row['output_change_witness']}"
            f" | n={row['n_params']} trainable={row['trainable']}"
            f" in_optimizer={row['in_optimizer']}")
        if row['modules']:
            lines.append(
                f"  {'':<9} modules={','.join(row['modules'])} "
                f"witnessUnavailable={','.join(row['witness_unavailable']) or '-'}")
        if row['params_without_grad_object']:
            lines.append(
                f"  {'':<9} noGradObject="
                f"{row['params_without_grad_object'][:3]} "
                f"(n={len(row['params_without_grad_object'])})")
    return '\n'.join(lines)


def _fmt(v: Optional[float]) -> str:
    return '-' if v is None else f'{float(v):.3g}'


__all__ = [
    'ACTIVATION_STEP_CAP', 'BUCKETS', 'BUCKET_ALIASES', 'BUCKET_BACKBONE',
    'BUCKET_CPM', 'BUCKET_ENCODER', 'BUCKET_OS', 'STATUSES', 'STATUS_ACTIVATED',
    'STATUS_NEVER', 'STATUS_NO_GRAD', 'STATUS_NOT_BUILT', 'TARGET_KEYS',
    'ActivationProbe', 'dual_loss_from_batch', 'format_summary',
    'normalize_bucket_name', 'run_activation_check',
]
