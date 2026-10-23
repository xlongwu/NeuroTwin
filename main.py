# coding=utf-8
"""NeuroTwin-TFM 训练入口。

训练基础设施（损失/图正则/优化器/EMA）位于 train/ 包，
通用工具（set_seed/str2bool）位于 utils/common.py；
本文件保留 main 训练循环与 CLI 定义。
"""
import argparse
import contextlib
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.log_dir = kwargs.get('log_dir', None)
        def add_scalar(self, *args, **kwargs): return None
        def close(self): return None

from utils.dataloader import (CONDITION_FILE_BY_MODE, CONDITION_GROUP_BY_MODE,
                              NeuroTwinDataLoader)
from utils.common import (parse_int_list, resolve_forecast_offsets,
                          resolve_task_dims, set_seed, str2bool)
from models.conditioning import ConditioningConfig
from models.tfm import NeuroTwinTFM
from utils.condition_schema import DROPOUT_SEED, parse_extra_codes
from train.losses import (TFMDualLoss, compute_graph_regularization, spatial_pcc,
                          compute_cpm_mask)
from train.prob_losses import (ensemble_crps, ensemble_energy_score,
                               temporal_variogram_score)
from train.optim import (ModelEMA, build_optimizer, build_scheduler,
                         check_pretrain_config_compat, count_trainable_params,
                         load_backbone_only, load_backbone_weights,
                         migrate_pretrained_checkpoint, param_report,
                         save_backbone_weights, set_finetune_stage,
                         unfreeze_while_preserving_state)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# ──────────────────────────────────────────────────────────────────────────
#  单行自适应进度条
# ──────────────────────────────────────────────────────────────────────────

#: {bar} 至少保留的列数；预算不足时裁剪指标而不是让整行折行。
PBAR_MIN_BAR = 12
#: 固定字段（百分比、计数分隔、耗时）的保守留白，按 'h:mm:ss' 形态的耗时取值。
PBAR_OVERHEAD = 30


def pbar_ncols() -> int:
    """进度条整行的目标列宽。

    重定向到文件/管道时 `shutil.get_terminal_size()` 走 `COLUMNS` → 80 的回退，
    因此日志行宽与终端里一致，不会因为宽度未知而每次刷新另起一行。
    """
    return max(PBAR_MIN_BAR + PBAR_OVERHEAD,
               min(int(shutil.get_terminal_size().columns), 120))


def make_pbar(iterable, total: int, desc: str) -> tqdm:
    """宽度自适应的单行进度条。

    `tqdm` 的 `dynamic_ncols` 只收缩 `{bar}`：desc 与 postfix 一旦超过终端宽度，
    整行仍然折行，表现为每个 iteration 打印一行。这里固定 `bar_format`（去掉变长的
    remaining/rate 字段）、把 desc 截进宽度预算，并配合 :func:`pbar_metrics` 裁剪
    指标，使整行始终占一行。
    """
    cols = pbar_ncols()
    return tqdm(iterable, total=total, leave=False, ncols=cols,
                dynamic_ncols=sys.stderr.isatty(), mininterval=0.5,
                desc=desc[:max(8, cols - PBAR_OVERHEAD)],
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}')


def pbar_metrics(pbar: tqdm, **metrics) -> None:
    """把指标拼成 `k=v` 串，并按剩余列宽从右往左丢弃放不下的项。

    `tqdm` 会在字符串 postfix 前自动加 `', '`，因此预算里扣掉这 2 列。
    """
    cols = pbar.ncols if isinstance(pbar.ncols, int) else pbar_ncols()
    count_w = max(len(str(pbar.total)), len(str(pbar.n)))
    budget = (cols - PBAR_MIN_BAR - PBAR_OVERHEAD - count_w - 2
              - len(str(pbar.desc or '')))
    shown, used = [], 0
    for name, value in metrics.items():
        part = f'{name}={value}'
        need = len(part) + (1 if shown else 0)
        if used + need > budget:
            break
        shown.append(part)
        used += need
    pbar.set_postfix_str(' '.join(shown), refresh=False)


# ──────────────────────────────────────────────────────────────────────────
#  next_timepoint（Next Brain-State Prediction）验证循环
# ──────────────────────────────────────────────────────────────────────────

def next_point_batch(batch, device):
    """取出 next_timepoint batch 的字段。

    y 由 [B,F,H] 升维为 [B,F,H,1]（与模型输出 [B,F,H,1] 对齐）；x_last 为
    context 末位 x_t，target_mask [B,H] 标记各预测偏移是否有真值。

    返回的 ``bp`` 是 **legacy 数值条件**（pathology_score）；字段级条件注入使用
    :func:`condition_batch`，两者由模型显式互斥校验。
    """
    bx = batch['x'].to(device, non_blocking=True)                # [B,F,1,K]
    by = batch['y'].to(device, non_blocking=True).unsqueeze(-1)  # [B,F,H,1]
    bl = batch['x_last'].to(device, non_blocking=True)           # [B,F]
    bm = batch.get('target_mask', None)
    if bm is not None:
        bm = bm.to(device, non_blocking=True)                    # [B,H]
    bs = batch['sc'].to(device, non_blocking=True)
    bp = batch.get('pathology_score', None)
    if bp is not None:
        bp = bp.to(device, non_blocking=True)
    return bx, by, bl, bm, bs, bp


#: 字段级条件批次的键（Dataset 在 condition_group 开启时产出）
CONDITION_BATCH_KEYS = ('cond_z', 'cond_code', 'cond_mask', 'subject_index')


def condition_batch(batch, device):
    """把字段级条件张量搬到 device，返回模型 ``condition=`` 所需的 dict 或 None。

    ``cond_raw`` 只服务诊断/置换对照，不进模型，因此这里刻意不透传。
    三个必需键缺一即报错：静默降级成「无条件」会让对照臂悄悄变成不同的模型。
    """
    if 'cond_z' not in batch:
        return None
    missing = [k for k in ('cond_z', 'cond_code', 'cond_mask') if k not in batch]
    if missing:
        raise KeyError(f'条件批次缺少必需键 {missing}')
    out = {
        'cond_z': batch['cond_z'].to(device, non_blocking=True),
        'cond_code': batch['cond_code'].to(device, non_blocking=True),
        'cond_mask': batch['cond_mask'].to(device, non_blocking=True),
    }
    if 'subject_index' in batch:
        out['subject_index'] = batch['subject_index'].to(device, non_blocking=True)
    return out


def _index_condition(cond, index):
    """按布尔/整数索引切片条件批次（rollout 只取被选中的样本）。"""
    if cond is None:
        return None
    return {k: (v[index] if torch.is_tensor(v) else v) for k, v in cond.items()}


def dynamics_extras(batch, device):
    """搬运 graph_ode 动态输入：calibration_bold [B,F,1,L] 与 controls [B,H,F]。

    旧数据路径（discrete / legacy_mask / 非 raw）不产出这两个键，返回
    ``(None, None)``，forward 与 loss 的行为与旧版完全一致。
    """
    cal = batch.get('calibration_bold', None)
    if cal is not None:
        cal = cal.to(device, non_blocking=True)
    ctl = batch.get('controls', None)
    if ctl is not None:
        ctl = ctl.to(device, non_blocking=True)
    return cal, ctl


def _normalized_future(model, fut: torch.Tensor, norm_stats):
    """用当前历史的 RevIN 统计量归一化未来真值（方差损失目标，方案 §9）。

    fut: [B,F,H]（数据尺度）→ [B,F,H]（归一化空间）。norm_stats 为 None
    （--norm 关闭 RevIN）时直接返回原值，不再另行估计统计量。
    """
    if norm_stats is None:
        return fut
    revin = getattr(model, 'rev_norm', None)
    if revin is None:
        return fut
    return revin.normalize_with(fut.unsqueeze(-1), norm_stats).squeeze(-1)


def condition_loader_kwargs(args, bundle=None):
    """从 checkpoint 快照推导评估/推理 DataLoader 的条件参数（与训练同口径）。

    ``bundle`` 提供训练期的 train-only 统计量 → 使用 ``inherit`` 模式**逐位复用**，
    不在评估侧重拟合；缺少 bundle 时退回 ``fit``（用同一份划分与 seed 重新拟合，
    结果应与训练一致，但会在日志中显式区分）。
    """
    arm = str(getattr(args, 'condition_arm', 'none') or 'none')
    if arm == 'none':
        return {}
    group = (str(getattr(args, 'condition_group', '') or '')
             or CONDITION_GROUP_BY_MODE.get(str(args.mode)))
    if not group:
        raise ValueError(f'mode={args.mode} 无法推导 condition_group，请显式指定')
    base = None
    if bundle is not None:
        from experiments.evaluate_variant import condition_stats_from_bundle
        base = condition_stats_from_bundle(bundle, group)
    return {
        'condition_group': group,
        'condition_file': (getattr(args, 'condition_file', '') or None),
        'split_file': (getattr(args, 'split_file', '') or None),
        'stratify_key': (getattr(args, 'stratify_key', '') or None),
        'condition_base_stats': base,
        'condition_stats_mode': 'inherit' if base is not None else 'fit',
        'condition_control': str(getattr(args, 'cond_control', 'none') or 'none'),
    }


def split_hash(dl, args) -> str:
    """被试划分的规范哈希：所有对照臂必须共享同一 cohort/split（验收 F）。

    只依赖 train/val/test 的被试名单（排序后），因此不同 seed 的重复运行、以及
    同一划分下的不同条件臂得到相同哈希；名单变化立即暴露。
    """
    import hashlib
    import json
    payload = {'mode': str(args.mode),
               'train': sorted(dl.train_subjects),
               'val': sorted(dl.val_subjects),
               'test': sorted(dl.test_subjects)}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def _report_conditioning(model, optimizer, args, dl,
                         allow_frozen_conditioning=False):
    """登记实际参数量并校验条件参数全部进入优化器（方案 §11 / 验收 L）。"""
    from train.optim import param_report, verify_conditioning_param_groups
    rep = param_report(model, optimizer)
    print(f"[params] total={rep['total']} | conditioning={rep['conditioning']} "
          f"| trainable={rep['trainable']} | rho_cond="
          f"{rep['rho_conditioning']:.4f} | optimizer_numel={rep['optimizer_numel']} "
          f"| distinct_tensors={rep['optimizer_distinct_tensors']}")
    if getattr(model, 'conditioner', None) is not None:
        print(f"[params] 条件编码器内部: {model.conditioner.param_report()} "
              f"| layout={model.conditioner.layout()['layout_hash']}")
        verify_conditioning_param_groups(
            model, optimizer,
            allow_frozen_conditioning=allow_frozen_conditioning)
        print('[params] 条件参数组校验通过（全部入优化器、无重复、numel 一致）')
    return rep


def load_condition_stats_payload(path):
    """从 checkpoint 读取 train-only 条件统计量快照（HC→MDD 继承用）。"""
    import json
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'--condition_inherit_stats 指定的文件不存在: {path}')
    if p.suffix.lower() == '.json':
        payload = json.loads(p.read_text(encoding='utf-8'))
        # 允许传整个 checkpoint config 的导出件：既支持裸 stats，也支持含
        # config.condition.stats 的结构
        if isinstance(payload, dict) and 'fields' not in payload:
            payload = ((payload.get('config') or {}).get('condition') or {}).get('stats')
    else:
        raw = torch.load(p, map_location='cpu', weights_only=False)
        cfg = raw.get('config') if isinstance(raw, dict) else None
        payload = (cfg or {}).get('condition', {}).get('stats') if isinstance(cfg, dict) else None
    if not isinstance(payload, dict) or 'fields' not in payload:
        raise KeyError(
            f'{path} 中没有可用的条件统计量快照（需要 config.condition.stats 或裸 '
            f'stats JSON）：无法继承 HC 的 Age/education 统计量与类别协议。')
    return payload


def _masked_l1_weighted(pred, target, weight, eps=1e-8):
    """按 [B,H] 权重（广播到 [B,F,H,1]）加权的逐元素 L1。"""
    w = weight.to(pred.dtype).view(weight.shape[0], 1, weight.shape[1], 1)
    w = w.expand_as(pred)
    return (torch.abs(pred - target) * w).sum() / w.sum().clamp_min(eps)


def parse_pathology_fields(fields):
    """把逗号分隔的字段字符串解析为列表；空/None 返回 None（由 Dataset 使用默认）。"""
    if fields is None:
        return None
    if isinstance(fields, (list, tuple)):
        parsed = [str(f).strip() for f in fields if str(f).strip()]
        return parsed or None
    parsed = [f.strip() for f in str(fields).split(',') if f.strip()]
    return parsed or None


#: graph_ode 路线的 checkpoint 主判据默认口径：H1..H8 全 ROI、先被试内再跨被试等权平均
#: 实际步数由 ``--dynamics_metric_horizon`` 决定，键名由 :func:`dynamics_metric_key` 派生
DYNAMICS_CKPT_HORIZON = 8


def dynamics_metric_key(horizon: int) -> str:
    """主判据键名，如 ``metric_mae_dynamics_h1_8_subject``（h1..hN 逐被试等权）。"""
    if int(horizon) < 1:
        raise ValueError(f'horizon 必须 >= 1，收到 {horizon}')
    return f'metric_mae_dynamics_h1_{int(horizon)}_subject'


def dynamics_metric_name(horizon: int) -> str:
    """主判据的日志显示名。"""
    if int(horizon) < 1:
        raise ValueError(f'horizon 必须 >= 1，收到 {horizon}')
    return f'val graph-ODE H1-{int(horizon)} subject MAE'


def resolve_dynamics_metric_horizon(args) -> int:
    """本实验的选优 horizon：CLI 覆盖 > 模型的 cpm_horizon（graph_ode 头的范围）> 8。"""
    configured = int(getattr(args, 'dynamics_metric_horizon', 0) or 0)
    if configured > 0:
        return configured
    return int(getattr(args, 'cpm_horizon', 0) or 0) or DYNAMICS_CKPT_HORIZON


def dynamics_anchor_mae(dynamics_pred, future, future_mask,
                        horizon: int = DYNAMICS_CKPT_HORIZON):
    """Graph-ODE 轨迹的逐锚点 H1..H``horizon`` 全 ROI 平均绝对误差。

    Args:
        dynamics_pred: ``[B,F,H]`` Graph-ODE 持续推进得到的预测（数据尺度，
            与动态损失 ``dynamics_target`` 同尺度）
        future: ``[B,F,R]`` 未来真值（数据尺度，序列末尾截断处补零）
        future_mask: ``[B,R]`` 逐步真值有效性（动态口径用 future_mask，
            不是 CPM 的 cpm_mask）
    Returns:
        ``(anchor_mae [B], full_mask [B] bool)``。``full_mask`` 标记具备完整
        ``horizon`` 步真值的锚点；``anchor_mae`` 只在被标记处有意义，其余位置
        置零，避免短未来锚点以不同预测范围参与主指标。
    """
    if int(horizon) < 1:
        raise ValueError(f'horizon 必须 >= 1，收到 {horizon}')
    if dynamics_pred.ndim != 3:
        raise ValueError(
            f'dynamics_pred 期望 [B,F,H]，收到 {tuple(dynamics_pred.shape)}')
    b, f, h = dynamics_pred.shape
    if h < horizon:
        raise ValueError(
            f'dynamics_pred 只有 {h} 步，无法计算 H1..{horizon} 指标；'
            f'请确认模型的 dynamics_horizon/cpm_horizon >= {horizon}。')
    if future.ndim != 3 or tuple(future.shape[:2]) != (b, f):
        raise ValueError(
            f'future 期望 [B,F,R] 且 (B,F)==({b},{f})，收到 {tuple(future.shape)}')
    if future.shape[2] < horizon:
        raise ValueError(
            f'future 只有 {future.shape[2]} 步，少于指标要求的 {horizon} 步；'
            '请确认验证视图的 future_steps 覆盖完整预测范围。')
    if tuple(future_mask.shape) != (b, future.shape[2]):
        raise ValueError(
            f'future_mask 期望 [B,{future.shape[2]}]，收到 {tuple(future_mask.shape)}')
    err = (dynamics_pred[:, :, :horizon]
           - future[:, :, :horizon].to(dynamics_pred.dtype)).abs()
    anchor_mae = err.mean(dim=(1, 2))                                   # [B]
    full_mask = (future_mask[:, :horizon] > 0).all(dim=1)                # [B]
    return anchor_mae * full_mask.to(anchor_mae.dtype), full_mask


def aggregate_dynamics_subject_metric(subj_acc, n_total_anchors: int,
                                      horizon: int = DYNAMICS_CKPT_HORIZON):
    """``{subj: [Σ锚点MAE, n_完整锚点]}`` → (主指标, 被试数, 锚点数)。

    聚合顺序为「先被试内均值、再跨被试等权均值」，因此结果与 batch 划分和
    每名被试的锚点数无关。没有任何完整 ``horizon`` 步目标时直接报错，
    不回退到单步 MAE 等其他口径。
    """
    subj_means = [total / n for total, n in subj_acc.values() if n > 0]
    if not subj_means:
        raise RuntimeError(
            f'验证集没有具备完整 {horizon} 步真实目标的锚点'
            f'（总锚点={n_total_anchors}，涉及被试={len(subj_acc)}），'
            f'无法计算 {dynamics_metric_key(horizon)}。请检查验证视图的 '
            f'future_steps / future_mask 是否覆盖 H1..{horizon}；'
            '本指标作为 checkpoint 选优判据时不允许回退到其他口径。')
    n_anchors = int(sum(n for _, n in subj_acc.values()))
    return sum(subj_means) / len(subj_means), len(subj_means), n_anchors


class CheckpointSelector:
    """按单一验证指标跟踪 best checkpoint 与 patience（指标越低越好）。

    指标键在构造时固定：单步与多步 MAE 量纲不同、数值不可比较，因此同一 run
    内不允许中途换判据，也不继承其他判据的历史 best 值。
    """

    def __init__(self, metric_key: str, metric_name: str = '', patience: int = 1):
        if not metric_key:
            raise ValueError('metric_key 不能为空')
        if int(patience) < 1:
            raise ValueError(f'patience 必须 >= 1，收到 {patience}')
        self.metric_key = metric_key
        self.metric_name = metric_name or metric_key
        self.patience = int(patience)
        self.best = float('inf')
        self.counter = 0

    def step(self, metrics) -> tuple:
        """消费一个 epoch 的验证指标，返回 ``(is_best, current)`` 并更新 patience。"""
        if self.metric_key not in metrics:
            raise KeyError(
                f'验证结果缺少选优指标 {self.metric_key}；'
                f'可用键示例: {sorted(k for k in metrics if k.startswith("metric_"))[:8]}')
        current = float(metrics[self.metric_key])
        if not math.isfinite(current):
            raise ValueError(
                f'{self.metric_key} 为非有限值 {current}，无法据此选优；'
                '请先排查该 epoch 的动态预测是否发散，而不是让 NaN 参与比较。')
        is_best = current < self.best
        if is_best:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
        return is_best, current

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def _format_loss_weight(name, info):
    """[LossW] 日志项：固定权重项没有 log_s，不能按浮点格式打印。"""
    log_s = info.get('log_s')
    log_s_text = 'fixed' if log_s is None else f'{log_s:+.4f}'
    return f"{name}: w={info['weight']:.4f} (log_s={log_s_text})"


@contextlib.contextmanager
def paired_val_rng(active: bool, seed: int):
    """验证期共同随机数：固定采样轨迹的随机流，使各 epoch 的集合指标（CRPS/ES/
    Variogram）比较的是模型权重而不是采样运气；退出时恢复原随机流，训练侧
    dropout/数据采样序列逐位不变。``active=False`` 时是空操作。

    S4 的 ``dynamics_samples`` 在 forward 内用 ``torch.randn`` 生成，若沿用训练推进后
    的全局 RNG，checkpoint 选优可能选中「运气好」的那一次采样（方案 §S4 验收）。
    """
    if not active:
        yield
        return
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def evaluate_tfm(model, data_loader, criterion, device, args,
                 rollout_steps=0, rollout_horizons=None,
                 pathology_epoch=None, loss_epoch=None):
    """TFM 的验证/测试循环：one-step 指标 + CPM 逐 horizon + Graph-ODE 多步判据。

    指标键（``metric_mae_next`` 等）供日志与 checkpoint 判据复用，另加：
      - ``cpm_mae_H{h}``：CPM 头在 t+h 处的逐元素 MAE（元素数归一）；
      - ``cpm_persistence_mae_H{h}``：persistence 基线在同一 horizon 的 MAE；
      - ``metric_cpm_mae``：CPM 全 horizon 平均 MAE；
      - ``metric_mae_dynamics_h1_N_subject``（N=--dynamics_metric_horizon）：
        Graph-ODE 持续推进（``aux_info['dynamics_pred']``，不用 CPM 输出替代）
        在 H1..N、全部 ROI 上的逐锚点 MAE，**只纳入 future_mask 显示具备完整
        N 步真值的锚点**，先被试内平均、再跨被试等权平均。graph_ode 路线的
        checkpoint 判据即此指标（越低越好）；配套计数键
        ``..._n_subjects`` / ``..._n_anchors`` / ``..._n_subjects_without_full_anchor``
        / ``..._n_anchors_total``。整轮验证集没有有效锚点时直接报错，不回退。

    聚合口径：单步 ``metric_mae_next`` 与多步主判据均采用
    **先被试内平均、再跨被试平均**，与最终评估 analysis/next_point_eval.py 的
    subject 口径一致，避免 batch 均值权重随末位小 batch / task 数变化影响
    微小收益下的选权重；batch 均值另存 ``metric_mae_next_batchmean`` 供追溯。
    其余指标仍为 batch 均值（仅用于趋势观察）。
    """
    model.eval()
    meters = defaultdict(list)
    subj_mae_acc = defaultdict(lambda: [0.0, 0])     # subj -> [Σ逐样本MAE, n]
    # graph_ode 多步主判据的累积：只统计具备完整 H 步真值的锚点
    dyn_metric_horizon = resolve_dynamics_metric_horizon(args)
    dyn_subj_acc = defaultdict(lambda: [0.0, 0])     # subj -> [Σ锚点MAE, n_完整锚点]
    dyn_subjects_seen = set()                         # 验证集中出现过的被试
    dyn_anchor_total = 0                              # 遍历过的锚点总数（含不完整）
    rollout_acc = defaultdict(lambda: [0.0, 0, 0])
    cpm_acc = defaultdict(lambda: [0.0, 0, 0])       # h -> [Σ|err|, Σw, n_elem]
    rollout_hs = [int(h) for h in (rollout_horizons or []) if int(h) <= int(rollout_steps)]
    horizon = int(getattr(model, 'cpm_horizon', 0) or args.cpm_horizon)
    dyn_horizon = (max(horizon, dyn_metric_horizon)
                   if getattr(model, 'dynamics_mode', 'discrete') == 'graph_ode' else horizon)
    pbar = make_pbar(data_loader, len(data_loader), '[Val-TFM]')

    for batch in pbar:
        bx, by, bl, bm, bs, bp = next_point_batch(batch, device)
        cond = condition_batch(batch, device)
        bcal, bctl = dynamics_extras(batch, device)
        fut = batch['future'].to(device, non_blocking=True)          # [B,F,R]
        fmask = batch['future_mask'].to(device, non_blocking=True)   # [B,R]
        outputs, aux_info = model(bx, bs, bp,
                                  pathology_epoch=pathology_epoch,
                                  condition=cond,
                                  condition_epoch=None,              # 验证：关闭随机 Dropout
                                  controls=(bctl[:, :dyn_horizon] if bctl is not None else None),
                                  calibration_bold=bcal,
                                  **({'dynamics_horizon': dyn_horizon}
                                     if dyn_horizon > horizon else {}))
        cpm_pred = aux_info['cpm_pred']                              # [B,F,H]
        cpm_target = fut[:, :, :horizon]                             # [B,F,H]
        if bctl is not None:
            # graph_ode：CPM 是无控制辅助预测，从首个非零未来控制起屏蔽监督
            cpm_mask = compute_cpm_mask(bctl[:, :horizon], fmask[:, :horizon])
        else:
            cpm_mask = fmask[:, :horizon]                            # [B,H]
        dyn_kwargs = {}
        if aux_info.get('dynamics_pred', None) is not None:
            norm_stats = aux_info.get('norm_stats', None)
            dyn_kwargs = dict(
                dynamics_pred=aux_info['dynamics_pred'][:, :, :horizon],
                dynamics_target=cpm_target,                          # 数据尺度 [B,F,H]
                dynamics_mask=fmask[:, :horizon],                    # 完整 future_mask
                dynamics_target_norm=_normalized_future(model, cpm_target, norm_stats),
            )
            for key in ('dynamics_mean_norm', 'dynamics_logvar_norm'):
                value = aux_info.get(key)
                dyn_kwargs[key] = value[:, :, :horizon] if value is not None else None
        total_loss, loss_stats = criterion(outputs, by, cpm_pred,
                                           cpm_target, cpm_mask,
                                           context_last=bl, epoch=loss_epoch,
                                           **dyn_kwargs)
        # S4：验证集集合 CRPS（仅监控/可选选优；不参与反传）
        prob_samples = aux_info.get('dynamics_samples', None)
        if torch.is_tensor(prob_samples) and prob_samples.shape[0] > 0:
            hp = min(int(prob_samples.shape[3]), int(horizon), int(fut.shape[2]))
            if hp > 0:
                ps = prob_samples[:, :, :, :hp]
                ft = fut[:, :, :hp]
                fm = fmask[:, :hp]
                meters['metric_crps_dynamics'].append(
                    ensemble_crps(ps, ft, fm).detach())
                meters['metric_energy_dynamics'].append(
                    ensemble_energy_score(ps, ft, fm).detach())
                if hp >= 2:
                    meters['metric_variogram_dynamics'].append(
                        temporal_variogram_score(ps, ft, fm).detach())
                    tv = temporal_variogram_score(ps, ft, fm, lags=(1, 2, 4), power=0.5)
                    meters['metric_temporal_variogram_p05'].append(tv.detach())
                    meters['metric_probability_joint'].append((
                        args.s4_crps_weight * ensemble_crps(ps, ft, fm)
                        + getattr(args, 's4_variogram_weight', 0.0) * tv).detach())
        graph_reg, graph_stats = compute_graph_regularization(
            aux_info, device,
            sparsity_weight=args.sc_sparsity_weight,
            entropy_weight=args.sc_entropy_weight,
            temporal_weight=args.sc_temporal_weight,
        )
        total_loss = total_loss + graph_reg

        b, f, h, _ = outputs.shape
        if bm is None:
            bm = torch.ones(b, h, device=outputs.device, dtype=outputs.dtype)
        w4 = bm.view(b, 1, h, 1)
        mae = _masked_l1_weighted(outputs, by, bm)
        # 主判据的被试内聚合：逐样本 masked MAE → 按被试累积（§5.3 统一口径）
        per_sample_mae = ((outputs - by).abs() * w4.expand_as(outputs)
                          ).sum(dim=(1, 2, 3)) / \
            w4.expand_as(outputs).sum(dim=(1, 2, 3)).clamp_min(1e-8)   # [B]
        subj_ids = batch.get('subj_id', None)
        if subj_ids is not None:
            for i, sid in enumerate(subj_ids):
                acc = subj_mae_acc[str(sid)]
                acc[0] += float(per_sample_mae[i].item())
                acc[1] += 1
        # ---- 多步主判据累积：本次前向的 Graph-ODE 轨迹（不额外推理、不用 CPM 替代）----
        if aux_info.get('dynamics_pred', None) is not None:
            if subj_ids is None:
                raise KeyError(
                    'batch 缺少 subj_id：graph_ode 多步判据必须先按被试平均再跨被试等权'
                    '平均，不能用 batch 均值替代。')
            dyn_pred = aux_info['dynamics_pred']                       # [B,F,H] 数据尺度
            dyn_anchor_mae, dyn_full = dynamics_anchor_mae(
                dyn_pred, fut, fmask, horizon=dyn_metric_horizon)
            dyn_anchor_total += int(dyn_pred.shape[0])
            for i, sid in enumerate(subj_ids):
                sid = str(sid)
                dyn_subjects_seen.add(sid)
                if bool(dyn_full[i]):
                    acc = dyn_subj_acc[sid]
                    acc[0] += float(dyn_anchor_mae[i].item())
                    acc[1] += 1
        rmse = (((outputs - by) ** 2) * w4.expand_as(outputs)).sum() / \
            (w4.expand_as(outputs).sum().clamp_min(1e-8))
        rmse = rmse.clamp_min(0).sqrt()
        pcc_spatial = ((spatial_pcc(outputs, by) * bm).sum()
                       / bm.sum().clamp_min(1e-8))
        x_t = bx[:, :, 0, -1:].unsqueeze(-1)                         # [B,F,1,1]
        x_prev = bx[:, :, 0, -2:-1].unsqueeze(-1)
        pers = x_t.expand(b, f, h, 1)
        trend = (x_t + (x_t - x_prev)).expand(b, f, h, 1)
        pers_mae = _masked_l1_weighted(pers, by, bm)
        trend_mae = _masked_l1_weighted(trend, by, bm)

        # CPM 逐 horizon MAE（元素数归一，与 next-state MAE 同口径）
        cpm_mae_sum = 0.0
        cpm_mae_weight = 0.0
        for hi in range(horizon):
            mh = cpm_mask[:, hi].unsqueeze(1)                        # [B,1] 广播到 ROI
            n = float(mh.sum().item())
            if n <= 0:
                continue
            err = (cpm_pred[:, :, hi] - cpm_target[:, :, hi]).abs()  # [B,F]
            wsum = float((err * mh).sum().item())
            cpm_acc[hi + 1][0] += wsum
            cpm_acc[hi + 1][1] += n
            cpm_acc[hi + 1][2] += n * f
            cpm_mae_sum += wsum
            cpm_mae_weight += n * f
            # persistence 基线在同一 horizon 的 MAE（x̂=x_t）
            pers_err = (x_t[:, :, 0, 0] - cpm_target[:, :, hi]).abs()
            cpm_acc.setdefault(f'pers_{hi + 1}', [0.0, 0, 0])
            cpm_acc[f'pers_{hi + 1}'][0] += float((pers_err * mh).sum().item())
            cpm_acc[f'pers_{hi + 1}'][1] += n
            cpm_acc[f'pers_{hi + 1}'][2] += n * f
        metric_cpm_mae = (cpm_mae_sum / cpm_mae_weight) if cpm_mae_weight > 0 else 0.0

        for k, v in {**loss_stats, **graph_stats,
                     'metric_total': total_loss.detach(),
                     'metric_mae_next': mae.detach(),
                     'metric_rmse_next': rmse.detach(),
                     'metric_pcc_next': pcc_spatial.detach(),
                     'metric_persistence_mae': pers_mae.detach(),
                     'metric_trend_mae': trend_mae.detach(),
                     'metric_cpm_mae': torch.tensor(metric_cpm_mae)}.items():
            meters[k].append(v.detach())
        pbar_metrics(pbar, MAE=f"{float(mae):.4f}",
                     PCC=f"{float(pcc_spatial):.4f}",
                     CPM=f"{metric_cpm_mae:.4f}")

        # ---- free rollout（one-step 头自回归；中间不使用真值）----
        if rollout_hs and 'rollout_flag' in batch:
            rf = batch['rollout_flag'].bool()
            rf_gpu = rf.to(device)
            if rf_gpu.any():
                hist = bx[rf_gpu][:, :, 0, :].transpose(1, 2).contiguous()   # [n,K,F]
                roll = model.rollout_next_states(
                    hist, bs[rf_gpu], (bp[rf_gpu] if bp is not None else None),
                    steps=int(rollout_steps),
                    pathology_epoch=pathology_epoch,
                    condition=_index_condition(cond, rf_gpu),
                    controls=(bctl[rf_gpu][:, :int(rollout_steps)]
                              if bctl is not None else None),
                    calibration_bold=(bcal[rf_gpu] if bcal is not None else None))
                fut_r = batch['future'][rf].transpose(1, 2).contiguous().to(
                    device).float()                                  # [n,R,F]
                fmask_r = batch['future_mask'][rf].to(device).float()  # [n,R]
                for rh in rollout_hs:
                    ok = fmask_r[:, rh - 1] > 0
                    k = int(ok.sum())
                    if k:
                        rollout_acc[rh][0] += float(
                            (roll[ok, rh - 1, :] - fut_r[ok, rh - 1, :]).abs().sum())
                        rollout_acc[rh][1] += k
                        rollout_acc[rh][2] += k * int(roll.shape[-1])

    for hi, (err_sum, _n_tasks, n_elem) in cpm_acc.items():
        if isinstance(hi, int) and n_elem:
            meters[f'cpm_mae_H{hi}'].append(torch.tensor(err_sum / n_elem))
    for key, (err_sum, _n_tasks, n_elem) in cpm_acc.items():
        if isinstance(key, str) and key.startswith('pers_') and n_elem:
            meters[f'cpm_persistence_mae_H{key.split("_")[1]}'].append(
                torch.tensor(err_sum / n_elem))
    for rh, (err_sum, _n_tasks, n_elem) in rollout_acc.items():
        if n_elem:
            meters[f'rollout_mae_H{rh}'].append(torch.tensor(err_sum / n_elem))
    result = {k: torch.stack(vs).mean().item() for k, vs in meters.items()}
    if subj_mae_acc:
        # 主判据改为先被试内、再跨被试平均（与最终评估口径一致，§5.3）
        subj_means = [s / n for s, n in subj_mae_acc.values() if n > 0]
        result['metric_mae_next_batchmean'] = result['metric_mae_next']
        result['metric_mae_next'] = sum(subj_means) / len(subj_means)
        result['metric_mae_next_n_subjects'] = float(len(subj_means))
    if dyn_anchor_total:
        # 多步主判据：无完整 H 步锚点时由聚合函数直接报错，不静默缺键
        dyn_key = dynamics_metric_key(dyn_metric_horizon)
        metric_value, n_subj, n_anchors = aggregate_dynamics_subject_metric(
            dyn_subj_acc, dyn_anchor_total, horizon=dyn_metric_horizon)
        result[dyn_key] = metric_value
        result[f'{dyn_key}_n_subjects'] = float(n_subj)
        result[f'{dyn_key}_n_anchors'] = float(n_anchors)
        result[f'{dyn_key}_n_subjects_without_full_anchor'] = float(
            len(dyn_subjects_seen) - n_subj)
        result[f'{dyn_key}_n_anchors_total'] = float(dyn_anchor_total)
    return result


def fit_pathology_normalizer(model, dl, args):
    """用训练集病理评分拟合模型内归一化模块的统计量。

    仅在 finetune 模式且模型持有 pathology_normalizer 时生效。
    当 --reuse_pretrained_norm_stats 为真且模块已带拟合标记时，跳过重新拟合。
    """
    norm = getattr(model, 'pathology_normalizer', None)
    if norm is None:
        return
    if getattr(args, 'reuse_pretrained_norm_stats', False) and norm.is_fitted():
        print(f'[HAMD norm] 复用已加载的归一化统计量 (mode={norm.mode})')
        return
    scores = dl.get_train_pathology_scores()
    if scores is None or len(scores) == 0:
        raise RuntimeError('无法获取训练集病理（HAMD）评分，pathology normalizer 拟合失败。')
    norm.fit(scores)
    print(f'[HAMD norm] mode={norm.mode} | 拟合样本数={len(scores)} | '
          f'center={norm.center.tolist()} | scale={norm.scale.tolist()} | '
          f'out_dim={norm.out_dim}')


def check_seed_separation(args):
    """方案 §四：模型 seed / 采样 seed / Condition Dropout seed 必须是三条独立随机流。

    任意两者同值等于把两条流并为一条，「分离」名存实亡（``--seed`` 与 ``--sampling_seed``
    的历史默认值都是 2024，正好撞车）。这里拒绝，而不是静默改默认值；登记值见
    ``utils/condition_schema.py`` 的 MODEL_SEED / SAMPLING_SEED / DROPOUT_SEED。
    """
    flags = ('seed', 'sampling_seed', 'cond_dropout_seed')
    absent = [f for f in flags if not hasattr(args, f)]
    if absent:
        raise ValueError(
            f'缺少随机来源参数 {absent}：请经 main.py 的 argparse 入口调用，'
            '不要构造缺参的 Namespace（方案 §四 的分离要求无法在没有取值的条件下核验）。')
    seen = {}
    for flag in flags:
        value = int(getattr(args, flag))
        if value in seen:
            raise ValueError(
                f'随机来源必须两两分离（方案 §四）：--{flag}={value} 与 '
                f'--{seen[value]} 同值。')
        seen[value] = flag


def main(args):
    # checkpoint config.args 快照保留 model_arch 键：evaluate_variant 重建与
    # pretrain 签名校验依赖它（TFM 已是唯一主干，CLI 不再暴露该开关）
    args.model_arch = 'tfm'
    # DataLoader 共享内存策略：默认的 file_descriptor 依赖 TMPDIR 下的
    # resource_sharer socket；Qoder/GUI 会话切换会清理该临时目录，导致长训练在
    # worker 交接时抛 FileNotFoundError。file_system 走 /dev/shm 文件描述符，
    # 不依赖该 socket（可用 TORCH_SHARING_STRATEGY 显式覆盖）。
    _sharing = os.environ.get('TORCH_SHARING_STRATEGY', 'file_system')
    _setter = getattr(torch.multiprocessing, 'set_sharing_strategy', None)
    if _setter is not None and _sharing:
        try:
            _setter(_sharing)
            print(f'[dataloader] torch 共享内存策略: {_sharing}')
        except (RuntimeError, ValueError) as exc:
            print(f'[dataloader] 共享内存策略 {_sharing} 设置失败，沿用默认：{exc}')
    check_seed_separation(args)
    # ---- 病理条件模式约束 ----
    if args.mode == 'pretrain':
        # HC 预训练不构建病理条件模块（避免把病理效应学成健康亚型）；条件注入
        # 模块允许在 HC 上建立/拟合/前反向（v1 revised §1.3），由 condition_arm
        # 决定，One-Step 的 film 模式在模型内部随条件开关设置。
        if getattr(args, 'tfm_patho_mode', 'none') != 'none':
            print(f"[TFM] HC 预训练不构建病理条件模块：tfm_patho_mode="
                  f"'{args.tfm_patho_mode}' 自动置为 none。")
        args.tfm_patho_mode = 'none'
    if getattr(args, 'device', ''):
        device = torch.device(args.device)
    else:
        # 缺省维持历史行为（多卡机器上训练集中到 cuda:1）；可用 --device 覆盖
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    amp_enabled = (device.type == 'cuda' and args.amp)
    torch.set_num_threads(args.num_threads)
    if device.type == 'cuda' and args.tf32:
        # 输入形状固定，启用 cudnn 自动调优；TF32 加速 autocast 覆盖不到的 fp32 matmul
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    # 任务维度解析：next_timepoint（in_window=1, pred_window=1, seq_len=context_max）。
    # 模型/数据/损失共用同一口径。
    dims = resolve_task_dims(args)
    forecast_offsets = resolve_forecast_offsets(args)
    eval_rollout_horizons = (parse_int_list(args.eval_rollout_horizons,
                                            'eval_rollout_horizons')
                             or [1, 2, 4, 8, 16])
    needs_long_rollout = bool(args.eval_fc or args.eval_spectral)
    auto_steps = max(eval_rollout_horizons)
    if needs_long_rollout:
        auto_steps = max(auto_steps, int(args.fc_min_length))
    eval_rollout_steps = int(args.eval_rollout_steps) or auto_steps

    print('=' * 50)
    if args.mode == 'pretrain':
        print(f'[Pretrain] 正在启动预训练 | task_mode={dims["task_mode"]} | '
              f'model_arch=tfm | CPM horizon: {dims["cpm_horizon"]}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 1] Healthy Control (HC) physics backbone pretraining')
    else:
        print(f'[Finetune] Starting finetuning | task_mode={dims["task_mode"]} | '
              f'model_arch=tfm | CPM horizon: {dims["cpm_horizon"]}')
        print('=' * 50 + '\n' + '=' * 50)
        print('启动模式: [Stage 2] MDD individualized pathology finetuning '
              '[NeuroTwinTFM FiLM]')
    print('=' * 50)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.checkpoint_dir, 'runs', f"{args.name}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志已开启，保存路径: {log_dir}")

    print(f"[next_timepoint] 模型维度映射: in_window←1（整段 context 作为单窗口） | "
          f"pred_window←len(forecast_offsets)={dims['pred_window']} | "
          f"seq_len←context_max={dims['seq_len']} | "
          f"forecast_offsets={forecast_offsets} | "
          f"causal_training={args.causal_training}")
    print(f"[next_timepoint] 训练 context: K∈[--context_min, --context_max]="
          f"[{args.context_min}, {args.context_max}]"
          + (f"（离散 {parse_int_list(args.context_lengths, 'context_lengths')}）"
             if args.context_lengths else "（连续区间随机采样）")
          + f" | 评估固定 K_eval={int(args.eval_context_length) or args.context_max} | "
          f"anchors/被试={args.eval_anchors_per_subject} | "
          f"rollout: R={eval_rollout_steps}, horizons={eval_rollout_horizons}, "
          f"锚点/被试={args.eval_rollout_tasks_per_subject}")

    cond_enabled = bool(getattr(args, 'condition_arm', None) not in (None, '', 'none'))
    if args.pretrained_horizon_policy != 'require_match' and (
            args.mode != 'finetune' or not cond_enabled or args.load_backbone_only
            or args.s4_variant != 'off' or args.pretrained_arch_policy != 'require_match'):
        raise ValueError('CPM horizon 扩展仅支持确定性条件微调的严格显式迁移路径')
    condition_group = (str(args.condition_group) if getattr(args, 'condition_group', None)
                       else CONDITION_GROUP_BY_MODE.get(args.mode)) if cond_enabled else None
    if cond_enabled:
        print(f'[condition] 阶段={args.mode} 分组={condition_group} 臂={args.condition_arm} '
              f'| merged 文件={args.condition_file or CONDITION_FILE_BY_MODE.get(args.mode)} '
              f'（条件数据**只**来自该 merged 表）')
    cond_base_stats = None
    if cond_enabled and getattr(args, 'condition_inherit_stats', ''):
        from utils.condition_schema import (DEMO_GROUP, load_inherited_stats,
                                            schema_for)
        payload = load_condition_stats_payload(args.condition_inherit_stats)
        target_specs = schema_for(condition_group or 'mdd8')
        src_names = [f['name'] for f in payload.get('fields', [])]
        share = [n for n in DEMO_GROUP if n in src_names]
        cond_base_stats = load_inherited_stats(payload, target_specs, share)
        print(f'[condition] 已载入继承统计量快照：{args.condition_inherit_stats} | '
              f'来源字段={src_names} → 继承共享字段={share}；'
              f'其余字段将在 MDD-train 上拟合（不会覆盖继承值）')

    dl = NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode, batch_size=args.batch_size,
        condition_group=condition_group,
        condition_file=args.condition_file or None,
        split_file=args.split_file or None,
        stratify_key=(args.stratify_key or None),
        condition_base_stats=cond_base_stats,
        condition_extra_codes=parse_extra_codes(args.condition_extra_codes)
        if args.condition_extra_codes else None,
        condition_control=args.cond_control,
        pathology_input_dim=args.pathology_input_dim,
        pathology_fields=parse_pathology_fields(args.pathology_fields),
        pathology_missing=args.pathology_missing,
        clinical_file=args.clinical_file,
        total_windows=args.total_windows, seq_len=args.seq_len,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        eval_batch_size=args.eval_batch_size,
        refresh_split_manifest=getattr(args, 'refresh_split_manifest', False),
        task_mode=dims['task_mode'],
        bold_source=args.bold_source,
        random_context=args.random_context, random_cutoff=args.random_cutoff,
        train_samples_per_subject=args.train_samples_per_subject,
        subject_cache_size=args.subject_cache_size,
        sampling_seed=args.sampling_seed,
        # next_timepoint 的数据协议
        context_min=args.context_min, context_max=args.context_max,
        context_lengths=parse_int_list(args.context_lengths, 'context_lengths'),
        forecast_offsets=forecast_offsets,
        eval_context_length=(int(args.eval_context_length) or args.context_max),
        eval_anchors_per_subject=args.eval_anchors_per_subject,
        eval_rollout_tasks_per_subject=args.eval_rollout_tasks_per_subject,
        eval_rollout_steps=eval_rollout_steps,
        train_rollout_steps=dims['cpm_horizon'],
        train_anchor_max_offset=(args.train_anchor_max_offset or dims['cpm_horizon']),
        eval_all_future_steps=max(dims['cpm_horizon'], resolve_dynamics_metric_horizon(args)),
        eval_anchor_min_future_steps=args.eval_anchor_min_future_steps,
        # graph_ode 动态路线的数据参数（方案 §4）：raw 源、可见历史 FC、
        # 校准前缀与控制 manifest；旧默认值下这些参数不改变任何行为
        num_rois=args.num_rois,
        graph_source=args.graph_source,
        calibration_length=args.calibration_length,
        control_manifest=(args.control_manifest or None),
    )
    train_data, val_data = dl.get_train(), dl.get_val()

    condition_cfg_dict = None
    condition_stats_payload = None
    if cond_enabled:
        condition_cfg_dict = ConditioningConfig(
            group=condition_group, arm=args.condition_arm,
            d_demo=args.cond_d_demo, d_new=args.cond_d_new,
            cond_dim=args.cond_dim, h_delta=args.cond_h_delta,
            delta_style=args.cond_delta_style,
            dropout_p=args.cond_dropout_p,
            dropout_seed=args.cond_dropout_seed).to_dict()
        condition_stats_payload = dl.get_condition_stats().to_dict()
        print(f'[condition] 编码器配置 {condition_cfg_dict}')

    model = NeuroTwinTFM(
        features=args.num_rois, context_max=dims['seq_len'],
        patch_len=args.tfm_patch_len, dim=args.tfm_dim,
        num_layers=args.tfm_layers, num_heads=args.tfm_heads,
        ff_ratio=args.tfm_ff_ratio, dropout=args.dropout,
        sc_prior_mode=args.sc_prior_mode,
        sc_lambda_mode=args.sc_lambda_mode,
        sc_lambda_init=args.sc_lambda_init,
        sc_prior_rank=args.sc_prior_rank,
        sc_delta_a=args.sc_delta_a, sc_delta_rank=args.sc_delta_rank,
        sc_sinkhorn_iters=args.sc_sinkhorn_iters,
        one_step_hidden_dim=args.tfm_one_step_hidden_dim,
        tfm_patho_mode=args.tfm_patho_mode,
        cpm_horizon=args.cpm_horizon, cpm_layers=args.cpm_layers,
        cpm_intervention=args.cpm_intervention,
        pretrain_mode=(args.mode == 'pretrain'),
        condition_arm=args.condition_arm,
        condition_config=condition_cfg_dict,
        condition_stats_payload=condition_stats_payload,
        cond_inject=args.cond_inject,
        cond_backbone_scope=args.cond_backbone_scope,
        cond_norm_generator=args.cond_norm_generator,
        cond_norm_ident_dim=args.cond_norm_ident_dim,
        cond_lambda_os=args.cond_lambda_os,
        cond_lambda_cpm=args.cond_lambda_cpm,
        cond_gate_residual=args.cond_gate_residual,
        cond_gate_hidden=args.cond_gate_hidden,
        pathology_input_dim=args.pathology_input_dim,
        pathology_norm_mode=args.pathology_norm_mode,
        pathology_norm_quantiles=args.pathology_norm_quantiles,
        pathology_norm_rbf_knots=args.pathology_norm_rbf_knots,
        use_revin=args.norm,
        # graph_ode 动态路线（方案 §6）：旧默认 discrete / personalized=False /
        # legacy_mask 不改变任何已有行为
        dynamics_mode=args.dynamics_mode,
        ode_steps=args.ode_steps,
        dynamics_dt=args.dynamics_dt,
        graph_source=args.graph_source,
        personalized=args.personalized,
        profile_dim=args.profile_dim,
        dynamics_history_memory=args.dynamics_history_memory,
        s4_variant=args.s4_variant,
        s4_residual_rank=args.s4_residual_rank,
        s4_residual_hidden=args.s4_residual_hidden,
        s4_residual_init_scale=args.s4_residual_init_scale,
        s4_noise_rank=args.s4_noise_rank,
        s4_noise_scale=args.s4_noise_scale,
        s4_train_samples=args.s4_train_samples,
    ).to(device)
    s4_active = args.s4_variant != 'off'
    migration_report = None

    if args.mode == 'finetune':
        if not os.path.exists(args.pretrained_weight):
            raise FileNotFoundError(f"未找到预训练权重: {args.pretrained_weight}")
        # 跨阶段迁移守卫：HC（hc3）与 MDD（mdd8）的完整 layout_hash 天然不同
        # （K=3 vs K=8），但**共享前三字段必须逐位一致**，否则 E_demo / 特征编码器 /
        # FiLM 会挂到语义不同的列上（方案 §8：同宽度但字段顺序变化也必须拒绝）。
        saved_layout = None
        same_group_arm = False
        if cond_enabled:
            from experiments.evaluate_variant import load_condition_bundle
            bundle = load_condition_bundle(args.pretrained_weight)
            if bundle is not None:
                saved_layout = (bundle.get('meta') or {}).get('layout_hash')
            if saved_layout and getattr(model, 'conditioner', None) is not None:
                now = model.conditioner.layout()
                same_group_arm = (bundle['config'].get('group') == condition_group
                                  and bundle['config'].get('arm') == now['arm'])
                if same_group_arm and saved_layout != now['layout_hash']:
                    raise RuntimeError(
                        f'条件布局不一致：权重={saved_layout}，当前={now["layout_hash"]}。'
                        '字段顺序/来源列/类别词表任一变化都禁止复用该权重。')
                if not same_group_arm and saved_layout != now.get('demo_layout_hash'):
                    raise RuntimeError(
                        f'HC→MDD 共享三字段布局不一致：HC 权重 layout_hash={saved_layout}，'
                        f'当前 mdd8 的 demo 子集哈希={now.get("demo_layout_hash")}。'
                        'Sex/Age/education 的顺序、类型或词表必须逐位一致才能继承。')
        if args.load_backbone_only:
            # 显式模式：只加载主干、头部/条件模块重新初始化（prompt §三十六）。
            # 任务口径闸门降级为 warn：口径差异只提示，主干按 shape 逐键加载并汇报。
            check_pretrain_config_compat(args.pretrained_weight, dims,
                                         arch_policy='warn')
            load_backbone_only(model, args.pretrained_weight, device)
        elif cond_enabled:
            # 条件注入路径必须走**显式迁移器**：逐键归类并汇报，未分类键报错，
            # 不允许 strict=False + shape 过滤的静默忽略（方案 §8）
            check_pretrain_config_compat(args.pretrained_weight, dims,
                                         arch_policy=args.pretrained_arch_policy,
                                         horizon_policy=args.pretrained_horizon_policy)
            migration_report = migrate_pretrained_checkpoint(
                model, args.pretrained_weight, device,
                arch_policy=args.pretrained_arch_policy,
                horizon_policy=args.pretrained_horizon_policy,
                # 同组同臂才比较完整 layout_hash；跨组（hc3→mdd8）的完整哈希本就
                # 不同，上面已用 demo 子集哈希做过共享前缀校验
                layout_expect=(model.conditioner.layout()['layout_hash']
                               if (getattr(model, 'conditioner', None) and same_group_arm)
                               else None),
                layout_saved=saved_layout if same_group_arm else None)
        else:
            # 任务口径闸门：不同任务的目标/输入 shape 语义不同，
            # 不允许静默按 shape 过滤加载（形状对不上时会静默跳过大量键）
            check_pretrain_config_compat(args.pretrained_weight, dims,
                                         arch_policy=args.pretrained_arch_policy)
            load_backbone_weights(model, args.pretrained_weight, device,
                                  arch_policy=args.pretrained_arch_policy,
                                  skip_pattern=getattr(args, 'pretrained_skip_pattern', ''))
        set_finetune_stage(model, backbone_unfrozen=False)
        stage1_trainables = []
        if getattr(model.one_step_head, 'film', None) is not None:
            stage1_trainables.append('One-Step FiLM')
        if getattr(model.cpm_head, 'film', None) is not None:
            stage1_trainables.append('CPM FiLM')
        if s4_active and args.s4_stage == 'head':
            # S4 两阶段 stage 1：严格只训练新增概率模块，连条件适配器也冻结，
            # 避免 S2 点预测损失（或常数项）污染概率模块的梯度解释。
            for p in model.parameters():
                p.requires_grad = False
            s4_modules = model.s4_modules()
            if not s4_modules:
                raise RuntimeError('S4 已启用但没有构建任何 S4 模块，请检查模型构造参数。')
            for module in s4_modules:
                for p in module.parameters():
                    p.requires_grad = True
            print(f'[S4] stage=head：前 {args.freeze_backbone_epochs} epoch 仅训练 '
                  f'{args.s4_variant} 新模块（主干/条件/双头全部冻结）；'
                  '之后解冻并联合微调。')
        elif s4_active and args.s4_stage == 'joint':
            set_finetune_stage(model, backbone_unfrozen=True)
            print(f'[S4] stage=joint：从 epoch 0 起联合训练 {args.s4_variant} 与 S2 全部模块。')
        else:
            print(f'已进入分阶段微调: 前期仅训练条件适配器'
                  f'（{ "+ ".join(stage1_trainables) }），后期逐步解冻主干。')

    if args.mode == 'finetune':
        fit_pathology_normalizer(model, dl, args)

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Parameters: {count_trainable_params(model)}")
    print(f"[TFM] dim={args.tfm_dim} layers={args.tfm_layers} heads={args.tfm_heads} "
          f"patch_len={args.tfm_patch_len} ff_ratio={args.tfm_ff_ratio} | "
          f"one_step_hidden={args.tfm_one_step_hidden_dim} | "
          f"cpm_horizon={args.cpm_horizon} cpm_layers={args.cpm_layers} "
          f"cpm_intervention={args.cpm_intervention} | revin={args.norm}")
    print(f"[SC prior] mode={args.sc_prior_mode} "
          f"| lambda_mode={args.sc_lambda_mode}(init={args.sc_lambda_init}) "
          f"| rank={args.sc_prior_rank} | delta_a={args.sc_delta_a} "
          f"| sinkhorn_iters={args.sc_sinkhorn_iters}")

    # TFM 双目标损失（方案 §20）：L_one = Huber + λ_pcc(1-PCC)；
    # L_CPM = γ 加权逐 horizon Huber。不再使用数学等价的 abs+delta 双计。
    if args.s4_variogram_weight == -1:
        # A small, train-only pilot fixes lambda before any optimizer update.
        # Separate zero-worker loader plus RNG restoration preserves the P0/P1
        # initialization and subsequent training sampling sequence.
        import random
        import numpy as np
        import itertools
        np_state, py_state = np.random.get_state(), random.getstate()
        was_training = model.training
        pilot = torch.utils.data.DataLoader(
            train_data.dataset, batch_sampler=train_data.batch_sampler,
            collate_fn=train_data.collate_fn, num_workers=0)
        crps_sum, tv_sum, n_pilot = 0.0, 0.0, 0
        try:
            with torch.random.fork_rng(), torch.no_grad():
                model.eval()
                for batch in itertools.islice(pilot, 4):
                    bx, _, _, _, bs, bp = next_point_batch(batch, device)
                    bcal, bctl = dynamics_extras(batch, device)
                    _, aux = model(bx, bs, bp, condition=condition_batch(batch, device),
                                   controls=bctl, calibration_bold=bcal)
                    ps = aux['dynamics_samples']
                    target = batch['future'].to(device)
                    mask = batch['future_mask'].to(device)
                    crps_sum += float(ensemble_crps(ps, target, mask))
                    tv_sum += float(temporal_variogram_score(
                        ps, target, mask, lags=(1, 2, 4), power=0.5))
                    n_pilot += 1
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            model.train(was_training)
        if not n_pilot or not math.isfinite(tv_sum) or tv_sum <= 0:
            raise ValueError('P1 pilot has no finite positive temporal variogram')
        args.s4_variogram_weight = 0.05 * crps_sum / tv_sum
        args.s4_variogram_calibration = dict(
            batches=n_pilot, crps=crps_sum/n_pilot, variogram=tv_sum/n_pilot,
            target_loss_ratio=0.05, source='train-only initialization; no optimizer updates')
        print(f'[P1] fixed train-only lambda_v={args.s4_variogram_weight:.8g}; '
              f'pilot={args.s4_variogram_calibration}')

    if args.temporal_mode == 'adaptive' and args.adaptive_delta_scale <= 0:
        # Read only MDD-train subjects, in the calibrated data space used by loss.
        # This does not iterate the random training sampler or consume its RNG.
        import numpy as np
        delta_sum, delta_count = 0.0, 0
        for subject_id in dl.get_train_subjects():
            series = dl.base_dataset.get_series(subject_id)
            differences = np.abs(np.diff(series, axis=-1))
            delta_sum += float(differences.sum(dtype=np.float64))
            delta_count += differences.size
        if not delta_count or delta_sum <= 0:
            raise ValueError('train-only delta scale is empty or zero')
        args.adaptive_delta_scale = delta_sum / delta_count
        print(f'[loss] train-only mean absolute delta scale={args.adaptive_delta_scale:.8g}')

    criterion = TFMDualLoss(
        lambda_one=args.lambda_one, lambda_cpm=args.lambda_cpm,
        lambda_pcc=args.lambda_pcc, cpm_gamma=args.cpm_gamma,
        huber_delta=args.cpm_huber_delta,
        lambda_dynamics=args.lambda_dynamics,
        lambda_variance=args.lambda_variance,
        weighting=getattr(args, 'loss_weighting', 'fixed'),
        lambda_temporal_diff=args.lambda_temporal_diff,
        lambda_temporal_pcc=getattr(args, 'lambda_temporal_pcc', 0.0),
        lambda_cross_head=args.lambda_cross_head,
        cpm_horizon_weight=args.cpm_horizon_weight,
        cpm_horizon_alpha=args.cpm_horizon_alpha,
        cross_head_start_epoch=args.cross_head_start_epoch,
        temporal_mode=args.temporal_mode,
        temporal_lags=tuple(map(int, args.temporal_lags.split(','))),
        temporal_scale_weights=tuple(map(float, args.temporal_scale_weights.split(','))),
        adaptive_delta_alpha=args.adaptive_delta_alpha,
        adaptive_delta_clip=args.adaptive_delta_clip,
        adaptive_delta_scale=args.adaptive_delta_scale,
        dynamics_horizon_weight=args.dynamics_horizon_weight,
        horizon_weight_power=args.horizon_weight_power,
    ).to(device)
    print(f"[TFM] 损失: lambda_one={args.lambda_one} "
          f"lambda_cpm={args.lambda_cpm} lambda_pcc={args.lambda_pcc} "
          f"cpm_gamma={args.cpm_gamma} huber_delta={args.cpm_huber_delta} | "
          f"lambda_dynamics={args.lambda_dynamics} "
          f"lambda_variance={args.lambda_variance} | "
          f"lambda_temporal_diff={args.lambda_temporal_diff} "
          f"lambda_temporal_pcc={args.lambda_temporal_pcc} "
          f"lambda_cross_head={args.lambda_cross_head} "
          f"cpm_horizon_weight={args.cpm_horizon_weight}（α={args.cpm_horizon_alpha}）| "
          f"cpm_horizon={args.cpm_horizon}（One-Step + CPM 双头，方案 §9/§20）")
    if criterion.weighting == 'uncertainty':
        n_log_s = len(list(criterion.log_s.parameters()))
        print(f"[TFM] 损失权重: 同方差不确定性加权（Kendall CVPR 2018，"
              f"Σ exp(-log_s_i)·L_i + log_s_i）| 可学习 log 方差参数 {n_log_s} 个"
              f"，初始有效权重=对应 --lambda_*（warm start）| "
              f"优化器组 loss_weight: lr=lr_peak×{args.loss_lr_scale}, "
              f"weight_decay=0 | 启用任务项: "
              + ', '.join(k for k, v in criterion.get_loss_weights().items()
                          if isinstance(v, dict)))

    print(f"[TFM] One-Step 条件模式: 模型生效 tfm_patho_mode={model.tfm_patho_mode}"
          f"（args 值 {args.tfm_patho_mode}）"
          f" | OS FiLM: {'on' if getattr(model.one_step_head, 'film', None) is not None else 'off'}"
          f" | CPM FiLM: {'on' if getattr(model.cpm_head, 'film', None) is not None else 'off'}"
          f" | 主干条件 RMSNorm: {'on' if getattr(model, 'cond_norm', None) is not None else 'off'}")

    optimizer = build_optimizer(model, criterion, args)
    scheduler = build_scheduler(optimizer, args)
    _report_conditioning(
        model, optimizer, args, dl,
        allow_frozen_conditioning=(s4_active and args.s4_stage == 'head'))
    scaler    = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    ema       = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if args.compile and device.type == 'cuda' and hasattr(torch, 'compile'):
        # 变长 context（K 逐 batch 变化）按静态 shape 编译会每个 K 重编译一份计算图，
        # 曾因此整体跳过编译；改用 dynamic=True 将 shape 维符号化，一次编译覆盖不同 K。
        # ODE 头内部的 bool((dt<=0).any()) 等检查会造成少量 graph break，不影响正确性。
        # EMA（上方 deepcopy）与 save_backbone_weights 均按 _orig_mod 解包，
        # checkpoint 键名不受编译包装影响（train/optim.py）。
        model = torch.compile(model, dynamic=True)
        print("[next_timepoint] torch.compile(dynamic=True) 已启用（变长 K 符号化，"
              "首个 epoch 含一次性编译耗时属预期）。")

    # 主 checkpoint 判据（prompt §三十八：不使用单纯 PCC 选模型）：
    #   graph_ode → 验证集 H1..N 被试级多步 MAE（本阶段改动，越低越好）
    #   discrete  → 无 Graph-ODE 轨迹可评估，保持原单步 next-state MAE 判据
    # 两个口径数值量纲不同、不可互相比较；同一 run 内判据固定，不继承另一口径的历史 best。
    dyn_metric_horizon = resolve_dynamics_metric_horizon(args)
    dyn_metric_key = dynamics_metric_key(dyn_metric_horizon)
    if s4_active and args.s4_ckpt_metric == 'crps':
        ckpt_metric_key = 'metric_crps_dynamics'
        ckpt_metric_name = 'val dynamics CRPS'
    elif args.dynamics_mode == 'graph_ode':
        if s4_active and dyn_metric_horizon > int(dims['cpm_horizon']):
            raise ValueError(
                f'--dynamics_metric_horizon={dyn_metric_horizon} 超出模型的 '
                f'--cpm_horizon={dims["cpm_horizon"]}：多步判据无法覆盖到第 '
                f'{dyn_metric_horizon} 步，请调小 horizon 或增大预测范围。')
        ckpt_metric_key = dyn_metric_key
        ckpt_metric_name = dynamics_metric_name(dyn_metric_horizon)
    else:
        ckpt_metric_key = 'metric_mae_next'
        ckpt_metric_name = 'val next-state MAE'
    selector = CheckpointSelector(ckpt_metric_key, ckpt_metric_name,
                                  patience=args.patience)
    backbone_unfrozen = ((args.mode == 'pretrain')
                         or (s4_active and args.s4_stage == 'joint'))
    print(f"[next_timepoint] checkpoint 选择判据: {ckpt_metric_name}"
          f"（dynamics_mode={args.dynamics_mode}, metric_key={ckpt_metric_key}）| "
          f"单步 MAE 继续记录为观察指标（metric_mae_next）| "
          f"最终说明见训练日志/评估报告")

    save_dir = os.path.join(args.checkpoint_dir, args.name)
    if os.path.exists(save_dir):
        import glob, re
        p = Path(save_dir)
        dirs = glob.glob(f"{p}*")
        ms   = [re.search(rf"%s(\d+)" % p.stem, d) for d in dirs]
        idx  = [int(m.groups()[0]) for m in ms if m]
        save_dir = f"{p}{max(idx)+1 if idx else 2}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.train_epochs):
        if (args.mode == 'finetune'
                and not backbone_unfrozen
                and epoch >= args.freeze_backbone_epochs):
            set_finetune_stage(model, backbone_unfrozen=True)
            backbone_unfrozen = True
            # 解冻只追加新可训练参数：已训练条件参数的 Adam 状态与 LR 进度必须保留
            added, names = unfreeze_while_preserving_state(model, optimizer, args)
            scheduler.sync_param_groups()
            print(f"Epoch {epoch+1}: 已解冻主干，新增可训练参数 numel={added} "
                  f"groups={names}，可训练参数量 = {count_trainable_params(model)}"
                  f"（沿用现有 optimizer 状态与 scheduler 进度）")

        model.train()
        # 轨迹任务：每 epoch 重新打乱分桶 sampler（同 batch 内 L 一致的约束保持不变）
        batch_sampler = getattr(train_data, 'batch_sampler', None)
        if hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)
        train_meters = defaultdict(list)    # tensor 累积，epoch 末一次性同步
        live = defaultdict(float)           # 进度条所需的少数关键指标
        pbar = make_pbar(
            train_data, len(train_data),
            f"Epoch {epoch + 1:03d}/{args.train_epochs:03d} [Train]")

        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            # next_timepoint：x [B,F,1,K] / y [B,F,H,1] / x_last [B,F] / mask [B,H]
            bx, by, bl, bm, bs, bp = next_point_batch(batch, device)
            cond = condition_batch(batch, device)
            bcal, bctl = dynamics_extras(batch, device)
            by_roll = batch.get('future', None)
            bm_roll = batch.get('future_mask', None)
            if by_roll is not None: by_roll = by_roll.to(device, non_blocking=True)
            if bm_roll is not None: bm_roll = bm_roll.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs, aux_info = model(bx, bs, bp,
                                          pathology_epoch=epoch,
                                          condition=cond,
                                          condition_epoch=epoch,
                                          controls=bctl,
                                          calibration_bold=bcal)
                # CPM 目标：训练视图 future_steps=cpm_horizon → by_roll [B,F,H]。
                # graph_ode 有控制序列时，CPM 监督从首个非零未来控制起屏蔽
                # （受控区间的 CPM 辅助预测没有真值对应，方案 §9）。
                cpm_mask_train = bm_roll
                if bctl is not None and bm_roll is not None:
                    cpm_mask_train = compute_cpm_mask(bctl, bm_roll)
                dyn_kwargs = {}
                if aux_info.get('dynamics_pred', None) is not None and by_roll is not None:
                    dyn_kwargs = dict(
                        dynamics_pred=aux_info['dynamics_pred'],
                        dynamics_target=by_roll,                     # 数据尺度 [B,F,H]
                        dynamics_mask=bm_roll,                       # 完整 future_mask
                        dynamics_mean_norm=aux_info.get('dynamics_mean_norm', None),
                        dynamics_logvar_norm=aux_info.get('dynamics_logvar_norm', None),
                        dynamics_target_norm=_normalized_future(
                            model, by_roll, aux_info.get('norm_stats', None)),
                    )
                prob_loss = None
                variogram_loss = outputs.new_zeros(())
                if s4_active:
                    prob_samples = aux_info.get('dynamics_samples', None)
                    if not torch.is_tensor(prob_samples):
                        raise RuntimeError(
                            'S4 已启用但模型未返回 dynamics_samples；请检查 s4_variant '
                            '与 graph_ode 路径。')
                    if by_roll is None or bm_roll is None:
                        raise RuntimeError(
                            'S4 CRPS 训练需要未来真值 future/future_mask，但当前 batch '
                            '没有（请确认 train_rollout_steps > 0）。')
                    hp = min(int(prob_samples.shape[3]), int(by_roll.shape[2]))
                    prob_loss = ensemble_crps(
                        prob_samples[:, :, :, :hp], by_roll[:, :, :hp],
                        bm_roll[:, :hp])
                    if args.s4_variogram_weight > 0:
                        variogram_loss = temporal_variogram_score(
                            prob_samples[:, :, :, :hp], by_roll[:, :, :hp],
                            bm_roll[:, :hp], lags=(1, 2, 4), power=0.5)
                if s4_active and args.s4_stage == 'head' and not backbone_unfrozen:
                    # stage 1：只回传 CRPS，不计算/不反传 S2 点预测损失与损失权重参数
                    zero = outputs.new_zeros(())
                    total_loss = args.s4_crps_weight * prob_loss
                    loss_stats = {'loss_one': zero, 'loss_cpm': zero,
                                  'loss_crps': prob_loss.detach()}
                    graph_reg = zero
                    graph_stats = {}
                else:
                    total_loss, loss_stats = criterion(
                        outputs, by, aux_info['cpm_pred'], by_roll,
                        cpm_mask_train, context_last=bl, epoch=epoch, **dyn_kwargs)
                    graph_reg, graph_stats = compute_graph_regularization(
                        aux_info, device,
                        sparsity_weight=args.sc_sparsity_weight,
                        entropy_weight=args.sc_entropy_weight,
                        temporal_weight=args.sc_temporal_weight,
                    )
                    total_loss = total_loss + graph_reg
                    if prob_loss is not None:
                        total_loss = total_loss + args.s4_crps_weight * prob_loss
                        loss_stats['loss_crps'] = prob_loss.detach()

                if prob_loss is not None:
                    total_loss = total_loss + args.s4_variogram_weight * variogram_loss
                    loss_stats['loss_temporal_variogram'] = variogram_loss.detach()
                    loss_stats['loss_probability_joint'] = (
                        args.s4_crps_weight * prob_loss
                        + args.s4_variogram_weight * variogram_loss).detach()

            scaler.scale(total_loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad]
                + [p for p in criterion.parameters() if p.requires_grad],
                max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if ema is not None: ema.update(model)

            for k, v in {**loss_stats, **graph_stats,
                         'metric_total': total_loss.detach()}.items():
                train_meters[k].append(v.detach())
            live['metric_total'] += float(total_loss.item())
            live['loss_one'] += float(loss_stats['loss_one'].item())
            live['loss_cpm'] += float(loss_stats['loss_cpm'].item())
            postfix = {
                'Loss': f"{live['metric_total']/step:.4f}",
                'One': f"{live['loss_one']/step:.4f}",
                'CPM': f"{live['loss_cpm']/step:.4f}",
            }
            if 'loss_dynamics' in loss_stats:
                live['_dyn'] = live.get('_dyn', 0.0) + float(loss_stats['loss_dynamics'].item())
                postfix['Dyn'] = f"{live['_dyn']/step:.4f}"
            if 'loss_variance' in loss_stats:
                live['_var'] = live.get('_var', 0.0) + float(loss_stats['loss_variance'].item())
                postfix['Var'] = f"{live['_var']/step:.4f}"
            if 'loss_crps' in loss_stats:
                live['_crps'] = live.get('_crps', 0.0) + float(loss_stats['loss_crps'].item())
                postfix['CRPS'] = f"{live['_crps']/step:.4f}"
            pbar_metrics(pbar, **postfix)

        eval_model  = ema.ema if ema is not None else model
        with paired_val_rng(s4_active, args.val_sampling_seed):
            val_metrics = evaluate_tfm(eval_model, val_data, criterion, device,
                                       args, rollout_steps=eval_rollout_steps,
                                       rollout_horizons=eval_rollout_horizons,
                                       pathology_epoch=epoch,
                                       loss_epoch=epoch)
        scheduler.step()

        train_metrics = {k: torch.stack(vs).mean().item() for k, vs in train_meters.items()}
        group_lrs    = {g.get('name', f'g{i}'): g['lr']
                        for i, g in enumerate(optimizer.param_groups)}
        lr_text = ' | '.join(f"{k}: {v:.2e}" for k, v in group_lrs.items())

        # TFM 口径：one-step 主指标 + CPM 全 horizon 指标 + trivial baseline
        tqdm.write(
            f"Epoch {epoch+1:03d} | {lr_text} | "
            f"backbone={'unfrozen' if backbone_unfrozen else 'frozen'} | "
            f"Train Total: {train_metrics['metric_total']:.4f} | "
            f"one: {train_metrics['loss_one']:.4f} | "
            f"cpm: {train_metrics['loss_cpm']:.4f} | "
            f"Val Total: {val_metrics['metric_total']:.4f} | "
            f"Val next MAE: {val_metrics['metric_mae_next']:.4f} | "
            f"Val next PCC(sp): {val_metrics['metric_pcc_next']:.4f} | "
            f"Val CPM MAE: {val_metrics['metric_cpm_mae']:.4f} | "
            f"Val persistence MAE: {val_metrics['metric_persistence_mae']:.4f} | "
            f"Val trend MAE: {val_metrics['metric_trend_mae']:.4f}"
            + (f" | Val CRPS: {val_metrics['metric_crps_dynamics']:.4f}"
               if 'metric_crps_dynamics' in val_metrics else "")
        )
        cpm_h_log = [f"H{h}: {val_metrics[f'cpm_mae_H{h}']:.4f}"
                     for h in range(1, int(args.cpm_horizon) + 1)
                     if f'cpm_mae_H{h}' in val_metrics]
        if cpm_h_log:
            tqdm.write("[TFM/CPM] val MAE per horizon | " + ' | '.join(cpm_h_log))
        roll_log = [f"H{h}: {val_metrics[f'rollout_mae_H{h}']:.4f}"
                    for h in (eval_rollout_horizons or [])
                    if f'rollout_mae_H{h}' in val_metrics]
        if roll_log:
            tqdm.write(f"[NextPoint] val rollout MAE (free, R={eval_rollout_steps}) | "
                       + ' | '.join(roll_log))
        is_best, cur_metric = selector.step(val_metrics)

        if dyn_metric_key in val_metrics:
            dyn_text = (
                f"{val_metrics[dyn_metric_key]:.4f}"
                f"(被试={val_metrics[f'{dyn_metric_key}_n_subjects']:.0f}, "
                f"完整{dyn_metric_horizon}步锚点={val_metrics[f'{dyn_metric_key}_n_anchors']:.0f}, "
                f"无完整锚点被试="
                f"{val_metrics[f'{dyn_metric_key}_n_subjects_without_full_anchor']:.0f})")
        else:
            dyn_text = 'N/A（本配置无 Graph-ODE dynamics 轨迹）'
        tqdm.write(
            f"[NextPoint] 单步 MAE={val_metrics['metric_mae_next']:.4f} | "
            f"多步 MAE={dyn_text} | 选优指标 {ckpt_metric_name}={cur_metric:.4f} | "
            f"model vs persistence ΔMAE="
            f"{val_metrics['metric_persistence_mae'] - val_metrics['metric_mae_next']:+.4f}")
        if criterion.weighting == 'uncertainty':
            # 学习到的任务权重演化（收敛监控）：weight=exp(-log_s)
            tqdm.write("[LossW] " + ' | '.join(
                _format_loss_weight(k, v)
                for k, v in criterion.get_loss_weights().items()
                if isinstance(v, dict)))

        def _ckpt_meta():
            # 权重本身不携带 epoch 曾导致 best/last 无法追溯（S0 缺口清单）；
            # 自本版本起 checkpoint 记录训练进度与阶段状态；条件注入进一步记录
            # schema/统计量/mask 语义/布局哈希/merged 文件哈希/split 哈希，
            # 使 strict 重建与跨实验对齐可验证（方案 §4/§8）。
            meta = {'args': vars(args), 'epoch': int(epoch),
                    'epoch_1based': int(epoch + 1),
                    'val_metric': float(cur_metric),
                    'ckpt_metric_key': selector.metric_key,
                    'ckpt_metric_name': selector.metric_name,
                    'backbone_unfrozen': bool(backbone_unfrozen)}
            if migration_report is not None:
                meta['pretrained_migration'] = migration_report
            meta['condition'] = {
                'config': condition_cfg_dict,
                'stats': (dl.get_condition_stats().to_dict()
                          if dl.base_dataset.condition_enabled else None),
                'meta': dl.get_condition_meta(),
                'split_hash': split_hash(dl, args),
            }
            meta['params'] = param_report(model, optimizer)
            # 损失加权快照：记录加权模式与当时的可学习 log 方差 / 有效权重，
            # 便于事后审计任务平衡的演化（fixed 模式下 log_s 为 None）。
            lw = criterion.get_loss_weights()
            meta['loss_weighting'] = {
                'mode': criterion.weighting,
                'log_s': {k: v['log_s'] for k, v in lw.items()
                          if isinstance(v, dict)},
                'weights': {k: v['weight'] for k, v in lw.items()
                            if isinstance(v, dict)},
            }
            return meta
        if is_best:
            best_name = 'base_best.pt' if args.mode == 'pretrain' else 'finetuned_best.pt'
            save_backbone_weights(os.path.join(save_dir, best_name), eval_model,
                                  meta=_ckpt_meta())
            tqdm.write(f"New Best Model Saved! (Best {ckpt_metric_name}: "
                       f"{selector.best:.4f})")
        elif selector.should_stop:
            tqdm.write(f"\nEarly Stopping! 连续 {selector.patience} epoch 未改善。"
                       f"\n最佳 {ckpt_metric_name}: {selector.best:.4f}")
            last_name = 'base_last.pt' if args.mode == 'pretrain' else 'finetuned_last.pt'
            save_backbone_weights(os.path.join(save_dir, last_name),
                                  ema.ema if ema else model,
                                  meta=_ckpt_meta())
            break

        if epoch == args.train_epochs - 1:
            last_name = 'base_last.pt' if args.mode == 'pretrain' else 'finetuned_last.pt'
            save_backbone_weights(os.path.join(save_dir, last_name),
                                  ema.ema if ema else model,
                                  meta=_ckpt_meta())

        # TensorBoard
        tb_scalars = [
            ('Train/Total', train_metrics['metric_total']),
            ('Train/One', train_metrics['loss_one']),
            ('Train/CPM', train_metrics['loss_cpm']),
            ('Val/Total', val_metrics['metric_total']),
            ('Val/Next_MAE', val_metrics['metric_mae_next']),
            ('Val/Next_RMSE', val_metrics['metric_rmse_next']),
            ('Val/Next_PCC', val_metrics['metric_pcc_next']),
            ('Val/CPM_MAE', val_metrics['metric_cpm_mae']),
            ('Val/Persistence_MAE', val_metrics['metric_persistence_mae']),
            ('Val/Trend_MAE', val_metrics['metric_trend_mae']),
        ]
        if dyn_metric_key in val_metrics:
            tb_scalars.append(
                (f'Val/Dynamics_MAE_H1_{dyn_metric_horizon}_Subject',
                 val_metrics[dyn_metric_key]))
        if 'metric_crps_dynamics' in val_metrics:
            tb_scalars.append(('Val/Dynamics_CRPS',
                               val_metrics['metric_crps_dynamics']))
        if 'metric_energy_dynamics' in val_metrics:
            tb_scalars.append(('Val/Dynamics_EnergyScore',
                               val_metrics['metric_energy_dynamics']))
        if 'metric_variogram_dynamics' in val_metrics:
            tb_scalars.append(('Val/Dynamics_VariogramScore',
                               val_metrics['metric_variogram_dynamics']))
        for key in ('metric_temporal_variogram_p05', 'metric_probability_joint'):
            if key in val_metrics:
                tb_scalars.append((f'Val/{key}', val_metrics[key]))
        if 'loss_temporal_pcc' in train_metrics:
            writer.add_scalar('Train/Temporal_PCC', train_metrics['loss_temporal_pcc'], epoch)
        for key in ('loss_temporal_variogram', 'loss_probability_joint'):
            if key in train_metrics:
                writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
        for h, key in sorted((h, f'rollout_mae_H{h}')
                             for h in (eval_rollout_horizons or [])):
            if key in val_metrics:
                tb_scalars.append((f'Val/Rollout_MAE_H{h}', val_metrics[key]))
        for tag, val in tb_scalars:
            writer.add_scalar(tag, val, epoch)
        for name, lr in group_lrs.items():
            writer.add_scalar(f'LR/{name}', lr, epoch)
        if 'loss_crps' in train_metrics:
            writer.add_scalar('Train/CRPS', train_metrics['loss_crps'], epoch)
        for tag, key in [('Train/Graph_Sparsity', 'graph_sparsity'),
                          ('Train/Graph_Entropy', 'graph_entropy'),
                          ('Train/Graph_Temporal', 'graph_temporal'),
                          ('Train/RECC_Loss', 'recc_loss'),
                          ('Train/Pair_RECC_Loss', 'pair_recc_loss'),
                          ('Train/Router_Balance', 'router_balance'),
                          ('Train/Router_ZLoss', 'router_zloss'),
                          ('Train/Router_Entropy', 'router_entropy'),
                          ('Train/Expert_Diversity', 'expert_diversity'),
                          ('Train/Expert_DzCos', 'expert_dz_cos')]:
            if key in train_metrics:
                writer.add_scalar(tag, train_metrics[key], epoch)
        for tag, key in [('Val/Router_Top1_Agreement', 'router_top1_agreement'),
                         ('Val/Router_Top2_Recall', 'router_top2_recall'),
                         ('Val/Router_Regret', 'router_regret'),
                         ('Val/Router_Competence_Spearman', 'router_competence_spearman'),
                         ('Val/Competence_Entropy', 'competence_entropy'),
                         ('Val/Oracle_Best_Expert_Gap', 'oracle_best_expert_gap'),
                         ('Val/Oracle_Best_Pair_Gap', 'oracle_best_pair_gap')]:
            if key in val_metrics:
                writer.add_scalar(tag, val_metrics[key], epoch)
        if criterion.weighting == 'uncertainty':
            # 可学习任务权重的收敛曲线
            for name, v in criterion.get_loss_weights().items():
                if isinstance(v, dict):
                    writer.add_scalar(f'LossW/{name}', v['weight'], epoch)

    writer.close()
    best_n = 'base_best.pt' if args.mode == 'pretrain' else 'finetuned_best.pt'
    best_path = os.path.join(save_dir, best_n)
    print(f"\n训练完成！最佳权重已保存至: {best_path}")

    if args.mode == 'finetune' and getattr(args, 'eval_after_train', True) \
            and os.path.exists(best_path):
        # 训练结束自动评 test（闭环）：复用版本化切分 manifest。评估失败不影响训练产物。
        from experiments.evaluate_variant import evaluate as _eval_variant
        eval_dir = os.path.join(save_dir, 'eval')
        try:
            summary = _eval_variant(best_path, eval_dir, eval_splits=('test',),
                                    seed=args.seed)
            print(f"[post-train eval] test_PCC={summary.get('test_pcc')} "
                  f"test_MAE={summary.get('test_mae')}")
            print(f"[post-train eval] 详细指标见: {eval_dir}/metrics_test.json")
        except Exception as e:
            print(f"[post-train eval] 自动评估失败（训练产物不受影响）: {type(e).__name__}: {e}\n"
                  f"可手动复评: python -m experiments.evaluate_variant --ckpt {best_path} --out_dir {eval_dir}")


#: 历史入口的默认预训练权重：**只在 finetune 省略 --pretrained_weight 时回填**
#: （保持旧脚本行为不变）。正式 runbook（两个 *_tfm.sh 脚本）始终显式传入起点权重，
#: 不依赖这个默认值。
LEGACY_PRETRAINED_WEIGHT = './checkpoints/neurotwin_pretrain/base_best.pt'


def parse_args():
    p = argparse.ArgumentParser(
        description='NeuroTwin-TFM: Pathology-Conditioned TimesFM-style Foundation '
                    'Model for Brain Dynamics')
    p.add_argument('--mode', default='pretrain', choices=['pretrain', 'finetune'])
    p.add_argument('--device', default='',
                   help='训练设备（如 cuda:0）。空 = 历史默认（有 GPU 时用 cuda:1）。'
                        '用于多卡并行跑消融：不同执行器指定不同 --device。')
    p.add_argument('--pretrained_weight', default='',
                   help='起点权重路径。finetune 必填；pretrain 一律留空。'
                        '省略时按历史入口回填 '
                        f'{LEGACY_PRETRAINED_WEIGHT}。')
    p.add_argument('--save_training_state', action='store_true',
                   help='工程恢复接口：额外落盘 optimizer/scheduler/scaler/EMA/随机状态。'
                        '本轮正式从头实验不使用。')
    p.add_argument('--pathology_input_dim', type=int, default=1)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--name', default='neurotwin')
    p.add_argument('--clinical_file', default='Rest-meta-MDD-V1V2-Merged-MDD.xlsx')
    p.add_argument('--num_rois', type=int, default=116)
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--total_windows', type=int, default=9)
    p.add_argument('--norm', type=str2bool, default=False)  # 数据已 z-score，默认关闭 BrainRevIN
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--train_epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_init', type=float, default=5e-5)
    p.add_argument('--lr_peak', type=float, default=1e-4)
    p.add_argument('--lr_final', type=float, default=5e-5)
    p.add_argument('--warmup_epochs', type=int, default=15)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--loss_lr_scale', type=float, default=1.0)

    # ---------- 协议基础（检查点版本化 / 病理归一化） ----------
    p.add_argument('--pretrained_arch_policy', default='require_match',
                   choices=['require_match', 'warn', 'ignore'],
                   help='预训练权重架构版本不匹配时的策略：require_match 直接报错；'
                        'warn 打印并跳过不兼容键；ignore 静默跳过。')
    p.add_argument('--pretrained_skip_pattern', default='',
                   help='fnmatch 模式，命中的预训练键不加载（保留随机初始化），'
                         '用于受控消融对照，如 one_step_head.*')
    p.add_argument('--pretrained_horizon_policy', default='require_match',
                   choices=['require_match', 'repeat_last'],
                   help='显式条件迁移的 CPM horizon 策略；repeat_last 复制已有行，'
                        '新增行重复最后一行，其余形状冲突仍报错。')
    p.add_argument('--refresh_split_manifest', action='store_true',
                   help='重新生成 subject_split_<mode>.json 切分 manifest'
                        '（默认存在且配置一致时复用，配置不一致时报错）')
    p.add_argument('--no_eval_after_train', dest='eval_after_train',
                   action='store_false',
                   help='finetune 训练结束后不自动评 test（默认自动评估并落盘到'
                        ' <save_dir>/eval/）')
    p.add_argument('--pathology_norm_mode', default='robust_z',
                   choices=['identity', 'robust_z', 'zscore_quadratic',
                            'empirical_cdf', 'quantile', 'zscore_rbf'],
                   help='模型内病理评分归一化模式，决定条件输入宽度。')
    p.add_argument('--pathology_norm_quantiles', type=int, default=64,
                   help='empirical_cdf/quantile 模式的分位数网格点数。')
    p.add_argument('--pathology_norm_rbf_knots', type=int, default=8,
                   help='zscore_rbf 模式的 RBF 结点数。')
    p.add_argument('--pathology_fields', default=None,
                   help='逗号分隔的病理字段列表，默认 None 时使用 HAMD 标量。')
    p.add_argument('--pathology_missing', default='drop',
                   choices=['drop', 'zero', 'mean'],
                   help='病理字段缺失处理策略。')
    p.add_argument('--reuse_pretrained_norm_stats', type=str2bool, default=True,
                   help='加载的检查点已带归一化统计量时是否复用（否则用 train subjects 重新拟合）。')
    p.add_argument('--adapter_lr_scale', type=float, default=1.0,
                   help='条件适配器（adapter）参数组的学习率缩放（历史键；'
                        '未设 --condition_lr_scale 时作为其回退值）。')

    # ---------- 条件注入（docs/Version3_docs_1003/NeuroTwin_条件注入实验方案.md）----------
    # default=None 是「未声明」哨兵，与显式 ``--condition_arm none`` 语义不同：
    # 未声明保持 legacy 行为（tfm_patho_mode 决定要不要建 HAMD FiLM），显式 none
    # 是「结构性地不建条件通路」，必须连 legacy FiLM 一起关掉（§三）。
    # 两者都要写进快照，独立评估才能重建出与训练同一个模型（§六.7）。
    # 臂空间已按 2026-10-10 的决定收敛为 none|demo|cat：HC 侧只能用 demo（hc3 分组没有
    # 新增字段），MDD 侧选定 cat；direct/sum/mean 随对比工具链一起移除。
    p.add_argument('--condition_arm', default=None,
                   choices=['none', 'demo', 'cat'],
                   help='条件注入臂。不给该参数=未声明（保留 legacy 数值通路或'
                        '无条件）；none=显式不建条件通路，legacy HAMD FiLM 也不构建；'
                        'demo=仅共享三字段（HC 预训练侧唯一可行值）；'
                        'cat=逐字段独立编码 + 拼接后过零初始化 DeltaE（MDD 微调选定值）。')
    p.add_argument('--condition_group', default='',
                   choices=['', 'hc3', 'mdd8'],
                   help='条件字段分组；留空按 --mode 推导（pretrain→hc3，finetune→mdd8）。')
    p.add_argument('--condition_file', default='',
                   help='条件 merged 文件名（默认按 mode 取 '
                        'Rest-meta-MDD-V1V2-Merged-{HC,MDD}.xlsx）。条件数据只允许'
                        '来自该文件，禁止回读单独 V1/V2 源表。')
    p.add_argument('--split_file', default='',
                   help='外部被试级划分 JSON（train/val/test 名单）；提供并校验通过后'
                        '优先于 data/subject_split_<mode>.json，所有对照臂共用同一份。')
    p.add_argument('--stratify_key', default='',
                   help='划分分层键（MDD 默认 HAMD）。不随条件字段顺序 fields[0] 变化；'
                        '键缺失者进入预注册的缺失桶确定性分配，不删人。')
    p.add_argument('--condition_inherit_stats', default='',
                   help='HC checkpoint / JSON 路径：MDD 实验从中继承 Age/education 的 '
                        'mu/scale 与类别协议，只在新增五字段上重新拟合。')
    p.add_argument('--condition_extra_codes', default='',
                   help='显式预注册的有效类别码（Sex:0,3;medication:2 形式），用于'
                        'merged 中确实存在但未落在 train 的词表；运行时禁止静默扩表。')
    p.add_argument('--cond_d_demo', type=int, default=16,
                   help='共享三字段（HC）独立编码宽度 d_demo（搜索起点，非已知最优）。')
    p.add_argument('--cond_d_new', type=int, default=16,
                   help='MDD 新增五字段独立编码宽度 d_new。')
    p.add_argument('--cond_dim', type=int, default=32,
                   help='融合后条件向量宽度 C。')
    p.add_argument('--cond_h_delta', type=int, default=64,
                   help='增量 DeltaE 的隐藏宽度 H_delta。')
    p.add_argument('--cond_delta_style', default='mlp', choices=['linear', 'mlp'],
                   help='DeltaE 结构；mlp 末层零初始化使 c_MDD 起点等于 c_demo。')
    p.add_argument('--cond_dropout_p', type=float, default=0.0,
                   help='Condition Dropout 的遮蔽概率 p（受试者级、每 epoch 固定）。')
    p.add_argument('--cond_dropout_seed', type=int, default=DROPOUT_SEED,
                   help='Condition Dropout 种子（与 --seed 模型种子、划分 seed 分离）。')
    p.add_argument('--cond_inject', default='heads',
                   choices=['heads', 'backbone', 'heads+backbone'],
                   help='注入位置：双头 FiLM / 主干条件 RMSNorm / 两者。')
    p.add_argument('--cond_backbone_scope', default='last1',
                   choices=['last1', 'last2', 'all'],
                   help='主干调制范围（先只调制最后一个 block 的三个 norm 位置）。')
    p.add_argument('--cond_norm_generator', default='per_layer',
                   choices=['per_layer', 'shared'],
                   help='条件 norm 生成器：逐层独立 或 共享生成器＋层身份嵌入。')
    p.add_argument('--cond_norm_ident_dim', type=int, default=8,
                   help='shared 生成器的层身份嵌入宽度。')
    p.add_argument('--cond_lambda_os', type=float, default=1.0,
                   help='OS 头注入强度 λ（固定值，可被可学习 W 吸收，不是特征贡献比）。')
    p.add_argument('--cond_lambda_cpm', type=float, default=1.0,
                   help='CPM 头注入强度 λ（同上）。')
    p.add_argument('--cond_gate_residual', type=str2bool, default=False,
                   help='扩展候选：条件门控残差 h + g(h,c)⊙R(h,c)（R 末层零初始化）。')
    p.add_argument('--cond_gate_hidden', type=int, default=64,
                   help='条件门控残差分支 R 的隐藏宽度。')
    p.add_argument('--cond_control', default='none',
                   choices=['none', 'static_new', 'mask_only'],
                   help='信息/容量对照：static_new=新增五字段取训练常量且新增 mask '
                        '固定为全 1（同时取消值与缺失模式的个体差异）；mask_only=值固定'
                        '但保留真实 mask（检验缺失模式是否解释收益）。共享三字段不变。')
    p.add_argument('--condition_lr_scale', type=float, default=None,
                   help='条件参数组学习率相对 baseline 的倍率（未设则回退 '
                        '--adapter_lr_scale）。')

    # ---------- SC 软先验（soft anatomical prior / 图正则） ----------
    p.add_argument('--sc_prior_mode', default='soft_prior',
                   choices=['scaled', 'soft_prior', 'adaptive_only', 'functional_only'],
                   help='SC 注入形态：scaled=旧的静态 SC 度归一化（复现 Phase 1）；'
                        'soft_prior=λ·A_SC+(1-λ)·A_func（默认）；'
                        'adaptive_only=纯数据驱动功能图（同归一化口径）；'
                        'functional_only=功能图原始 row_softmax（不做 Sinkhorn）。')
    p.add_argument('--sc_lambda_mode', default='global',
                   choices=['global', 'sample', 'roi'],
                   help='λ 混合系数的粒度：global 全局标量 / sample 样本级 / roi 逐 ROI。')
    p.add_argument('--sc_lambda_init', type=float, default=0.7,
                   help='λ 初值（越大越信任 SC 结构先验）。')
    p.add_argument('--sc_prior_rank', type=int, default=12,
                   help='功能图 A_func 的低秩分解维度。')
    p.add_argument('--sc_delta_a', type=str2bool, default=True,
                   help='是否启用低秩 subject-specific 邻接残差 ΔA（零初始化）。')
    p.add_argument('--sc_delta_rank', type=int, default=None,
                   help='ΔA 的秩，默认 max(4, sc_prior_rank//2)。')
    p.add_argument('--sc_sinkhorn_iters', type=int, default=64,
                   help='对称 Sinkhorn 迭代次数（使 A_eff 同时行归一且对称；靠近均匀的图收敛慢）。')
    p.add_argument('--sc_sparsity_weight', type=float, default=1e-3,
                   help='SC 软先验稀疏正则权重（A_eff 非对角 L1），<=0 关闭。')
    p.add_argument('--sc_entropy_weight', type=float, default=1e-3,
                   help='SC 软先验行熵正则权重（抑制模糊均匀图），<=0 关闭。')
    p.add_argument('--sc_temporal_weight', type=float, default=0.0,
                   help='逐窗功能图时间一致性正则权重（>0 时才生成 A_seq），默认关闭。')

    p.add_argument('--freeze_backbone_epochs', type=int, default=10)
    p.add_argument('--backbone_lr_scale', type=float, default=0.20)
    p.add_argument('--val_ratio', type=float, default=0.10)
    p.add_argument('--test_ratio', type=float, default=0.10)
    p.add_argument('--stratify_bins', type=int, default=5)
    p.add_argument('--cache_in_memory', type=str2bool, default=False)
    p.add_argument('--use_ema', type=str2bool, default=True)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--amp', type=str2bool, default=True)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_threads', type=int, default=4)
    p.add_argument('--pin_memory', type=str2bool, default=True)
    p.add_argument('--persistent_workers', type=str2bool, default=True)
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--eval_batch_size', type=int, default=None,
                   help='验证/测试 DataLoader 的 batch 大小，默认为 batch_size×4')
    p.add_argument('--tf32', type=str2bool, default=True,
                   help='启用 cudnn.benchmark 与 TF32（对 autocast 未覆盖的 fp32 计算加速）')
    p.add_argument('--compile', type=str2bool, default=True,
                   help='torch.compile 编译模型前向（稳态约 35%% 提速；首次编译有额外耗时）')

    # ---------- NeuroTwin-TFM（TimesFM-3 风格主干，方案 §30 V1） ----------
    p.add_argument('--tfm_dim', type=int, default=256,
                   help='TFM 隐藏维度 D（方案 §17 推荐 128/192/256）。')
    p.add_argument('--tfm_layers', type=int, default=4,
                   help='TFM Encoder Block 层数 N（方案 §17 推荐 4/6）。')
    p.add_argument('--tfm_heads', type=int, default=4,
                   help='TFM temporal/ROI attention 的头数。')
    p.add_argument('--tfm_patch_len', type=int, default=8,
                   help='TFM temporal patch 长度 p（G30_TFM 消融：p=8 最优）。'
                        'K 不是 p 的整数倍时开头补零对齐。')
    p.add_argument('--tfm_ff_ratio', type=int, default=4,
                   help='TFM FFN 隐层扩张倍数（ff_dim = ff_ratio × dim）。')
    p.add_argument('--tfm_one_step_hidden_dim', type=int, default=256,
                   help='TFM One-Step 状态转移头的 MLP 隐层宽度。')
    p.add_argument('--cpm_horizon', type=int, default=8,
                   help='CPM 全 horizon 预测步数 H（方案 §8：推荐 {4,8,16}，'
                        '首选 8）。')
    p.add_argument('--cpm_layers', type=int, default=2,
                   help='CPM ROI×Horizon 查询解码器的交叉注意力层数。')
    p.add_argument('--cpm_intervention', type=str2bool, default=False,
                   help='构建 future-known covariate（intervention lookahead）嵌入'
                        '接口（方案 §12.4/§13）。V1 默认关闭：无真实干预数据，'
                        '仅保留接口，不作为治疗预测任务训练（方案 §13.1）。')
    p.add_argument('--lambda_one', type=float, default=1.0,
                   help='TFM 损失：one-step Huber 项权重（方案 §20.1）。')
    p.add_argument('--lambda_cpm', type=float, default=1.0,
                   help='TFM 损失：CPM 轨迹项权重（0 可退化为纯 one-step 的 '
                        'Stage 0 基线，方案 §24）。')
    p.add_argument('--lambda_pcc', type=float, default=0.1,
                   help='损失：spatial PCC 惩罚项权重（1 - 逐 ROI 空间相关）。')
    p.add_argument('--cpm_gamma', type=float, default=1.0,
                   help='CPM 逐 horizon 损失权重衰减 γ（w_h=γ^(h-1)；1=均匀，'
                        '方案 §20.2）。须在 (0, 1] 内。')
    p.add_argument('--cpm_huber_delta', type=float, default=1.0,
                   help='TFM Huber 损失的过渡阈值 δ（原始空间 BOLD 量纲，'
                        'z-score 后单位方差取 1.0）。')
    p.add_argument('--loss_weighting', default='uncertainty',
                   choices=['uncertainty', 'fixed'],
                   help='任务损失聚合方式：uncertainty=同方差不确定性自动加权'
                        '（Σ exp(-log_s_i)·L_i + log_s_i，log_s_i 为可学习参数，'
                        '初始有效权重等于对应 --lambda_*，Kendall et al. '
                        'CVPR 2018）；fixed=固定权重（历史行为）。仅作用于 '
                        'TFMDualLoss 的 one/CPM/dynamics/variance 四个任务项；'
                        'SC 图正则等其余项仍由各自 λ 控制。')

    # ---------- One-Step 头病理条件模式 ----------
    p.add_argument('--tfm_patho_mode', default='film',
                   choices=['none', 'film'],
                   help='One-Step 头病理条件模式：none=无条件（HC 预训练自动置为 '
                        'none）；film=FiLM 调制转移隐层（当前 baseline，legacy '
                        'discrete 路线）。CPM 恒保留自身 FiLM。')

    # ---------- 任务口径与 next_timepoint 数据协议 ----------
    p.add_argument('--task_mode', default='next_timepoint',
                   choices=['next_timepoint'],
                   help='任务口径：next_timepoint=连续 BOLD context → 下一 TR 全脑状态'
                        '（Next Brain-State Prediction，context_min..context_max 个 TR → '
                        '未来 TR，默认仅 +1）。')
    p.add_argument('--random_context', type=str2bool, default=True,
                   help='训练是否随机采样 context 长度 K（False=固定为可用最大 K）。')
    p.add_argument('--random_cutoff', type=str2bool, default=True,
                   help='训练是否随机采样 cutoff t（False=固定最晚 cutoff）。')
    p.add_argument('--bold_source', default='auto',
                   choices=['auto', 'normlize', 'windows', 'raw'],
                   help='连续 BOLD 数据源：auto=优先 data/Normlize/<grp>/ROISignals_<id>.mat，'
                        '缺失时回退用 50%% 重叠滑窗重构；normlize=仅用连续序列；windows=仅用滑窗重构；'
                        'raw=仅用 data/Raw/<grp>/ROISignals_<id>.mat（[T,F] 完整时间轴、前 num_rois 列，'
                        '每 ROI 用最早 calibration_length 个 TR 固定 mu0/s0 标准化，缺失直接报错不回退）。')
    p.add_argument('--train_samples_per_subject', type=int, default=0,
                   help='训练每个被试每 epoch 的样本预算；0=auto（min(合法组合数, 12)）。')
    p.add_argument('--sampling_seed', type=int, default=2024,
                   help='采样/分桶随机种子（训练采样可复现；评估协议本身与随机性无关）。')
    p.add_argument('--subject_cache_size', type=int, default=256,
                   help='数据集每个 worker 的被试级缓存上限（0=不缓存）。')
    p.add_argument('--eval_fc', type=str2bool, default=True,
                   help='评估是否输出 FC 层面指标。')
    p.add_argument('--eval_spectral', type=str2bool, default=False,
                   help='评估是否输出 Welch PSD 低频段一致性指标（可选）。')
    p.add_argument('--eval_spectral_tr', type=float, default=2.0,
                   help='频谱评估使用的 TR（秒），默认 2.0 s（REST-meta-MDD 常用值）。')

    # ---------- Next-Timepoint Prediction（连续 BOLD → 下一 TR 全脑状态） ----------
    p.add_argument('--context_min', type=int, default=16,
                   help='next_timepoint 训练 context 长度下界 K_min（TR 数）。')
    p.add_argument('--context_max', type=int, default=64,
                   help='next_timepoint context 长度上界 K_max（= 模型 S 轴建模宽度，决定参数量）。'
                        '训练随机采样 K ∈ context_lengths 或 [context_min, context_max]。')
    p.add_argument('--context_lengths', default='',
                   help='可选：离散 context 长度集合（逗号分隔，如 16,32,64）；'
                        '空=在 [context_min, context_max] 区间内均匀随机采样。')
    p.add_argument('--causal_training', default='random_context',
                   choices=['random_context', 'full_sequence'],
                   help='训练范式：random_context=随机 context → 下一时间点（方案 B，已实现，'
                        '未来信息不进入前向）；full_sequence=GPT 式全序列 teacher-forcing'
                        '（方案 A，需要重写主干时间轴算子，当前显式报错未实现）。')
    p.add_argument('--forecast_offsets', default='',
                   help='预测偏移列表（逗号分隔，相对当前 TR 的 +Δ，需严格递增）；'
                        'TFM 的 one-step 头只建模 +1 偏移，留空即 [1]，多步预测用 '
                        '--cpm_horizon 控制 CPM 头。')
    p.add_argument('--eval_context_length', type=int, default=0,
                   help='评估固定 context 长度 K_eval；0=自动取 --context_max。'
                        '评估协议对每个被试使用同一 K_eval 与确定性 anchor，禁止随机位置。')
    p.add_argument('--eval_anchors_per_subject', type=int, default=16,
                   help='每个被试评估的 next-state 预测点数量（等间隔确定性选取；0=枚举全部）。')
    p.add_argument('--eval_anchor_min_future_steps', type=int, default=0,
                   help='评估锚点必须拥有的完整未来步数；0 保持原 H8 协议。')
    p.add_argument('--train_anchor_max_offset', type=int, default=0,
                   help='训练锚点所需未来长度；0 使用 cpm_horizon，对照可统一为 32。')
    p.add_argument('--eval_rollout_tasks_per_subject', type=int, default=2,
                   help='每个被试参与 free rollout / FC / 频谱评估的 anchor 数（0=不做 rollout）。')
    p.add_argument('--eval_rollout_horizons', default='1,2,4,8,16',
                   help='free rollout 的评估 horizon 列表（逗号分隔）。')
    p.add_argument('--eval_rollout_steps', type=int, default=0,
                   help='单次 rollout 的滚动步数 R；0=自动取 max(horizons) 并在启用 FC/频谱时'
                        '抬到 --fc_min_length（FC 需要更长的轨迹）。')
    p.add_argument('--fc_min_length', type=int, default=32,
                   help='FC/频谱评估所需的最小 rollout 长度（prompt §二十二；短于该长度不计算 FC）。')
    p.add_argument('--load_backbone_only', type=str2bool, default=False,
                   help='finetune 时只加载主干权重（预测头 / 条件模块保持随机初始化），'
                        '并显式打印 loaded/missing/reinitialized 清单。')

    # ---------- graph_ode 动态路线（神经微分方程优化，方案 §6/§9/§10） ----------
    p.add_argument('--dynamics_mode', default='discrete',
                   choices=['discrete', 'graph_ode'],
                   help='动态分支：discrete=旧 OneStepDynamicsHead（delta_norm+x_last，默认）；'
                        'graph_ode=受控条件图 ODE（Heun RK2 同步积分 z/q，持续状态预测）。')
    p.add_argument('--ode_steps', type=int, default=2,
                   help='graph_ode 每个区间的固定 Heun RK2 步数（>=1；默认 2）。')
    p.add_argument('--dynamics_dt', type=float, default=1.0,
                   help='graph_ode 每个区间的积分步长 dt（>0，单位=TR；默认 1.0）。')
    p.add_argument('--graph_source', default='legacy_mask',
                   choices=['legacy_mask', 'history_fc'],
                   help='图来源：legacy_mask=外部解剖 Mask 图（旧路径）；'
                        'history_fc=当前可见历史的 signed Pearson 绝对值功能耦合图'
                        '（经 SoftAnatomicalPrior，不要求 Mask 文件）。')
    p.add_argument('--personalized', type=str2bool, default=False,
                   help='是否启用固定基线个体 profile（graph_ode 专用；需 --bold_source raw '
                        '并提供 calibration_bold）。False 时不创建 profile 模块。')
    p.add_argument('--profile_dim', type=int, default=32,
                   help='个体 profile 维度（personalized=True 时生效；默认 32）。')
    p.add_argument('--dynamics_history_memory', type=str2bool, default=False,
                   help='S2：向量场加入固定历史记忆 FiLM（编码 token 的 patch 维均值，'
                        'rollout 期间不变；零初始化，默认 False 不构建模块）。')
    p.add_argument('--calibration_length', type=int, default=0,
                   help='基线校准前缀长度（TR 数；bold_source=raw 时必须 >=1，'
                        '同时决定合法锚点下界与 calibration_bold 输出）。')
    p.add_argument('--control_manifest', default='',
                   help='控制序列 manifest CSV（表头 subject_id,controls_path；'
                        '相对路径相对 manifest 所在目录；NPZ 键 u，[T,F] 逐区间控制）。'
                        '空=无控制（controls=None，不伪造刺激标签）。')
    p.add_argument('--lambda_dynamics', type=float, default=0.5,
                   help='graph_ode 动态多步损失权重（h=2..H masked Huber+PCC；默认 0.5）。')
    p.add_argument('--lambda_variance', type=float, default=0.05,
                   help='方差读出高斯 NLL 损失权重（只更新 VarianceHead；默认 0.05）。')
    # ---- P0 损失扩展：时间差分 / 双头一致性 / horizon 权重策略 ----
    # 默认 λ=0、策略=gamma，与既有训练行为逐位一致；启用后为固定权重项，
    # 不进入 weighting='uncertainty' 聚合（只作用于 one/CPM/dynamics/variance）。
    p.add_argument('--loss_experiment_id', default='')
    p.add_argument('--temporal_mode', choices=['standard', 'adaptive', 'multiscale'], default='standard')
    p.add_argument('--temporal_lags', default='1')
    p.add_argument('--temporal_scale_weights', default='1.0')
    p.add_argument('--adaptive_delta_alpha', type=float, default=0.5)
    p.add_argument('--adaptive_delta_clip', type=float, default=3.0)
    p.add_argument('--adaptive_delta_scale', type=float, default=0.0,
                   help='Positive fixed train-only scale; 0 estimates mean absolute delta on train subjects.')
    p.add_argument('--dynamics_horizon_weight', type=str2bool, default=False)
    p.add_argument('--horizon_weight_power', type=float, default=1.0)
    p.add_argument('--s4_variogram_weight', type=float, default=0.0,
                   help='0=CRPS only; -1=fix lambda from four train pilot batches at 5% CRPS scale.')
    p.add_argument('--lambda_temporal_pcc', type=float, default=0.0,
                   help='Fixed weight for CPM+ODE future-trajectory temporal PCC; 0 disables. '
                        'At least 3 valid future points and nonconstant targets are required.')
    p.add_argument('--lambda_temporal_diff', type=float, default=0.0,
                   help='P0 一阶时间差分 Huber 权重（CPM + graph_ode 轨迹，'
                        '第一步参考真值历史末帧 x_t；0=禁用）。')
    p.add_argument('--lambda_cross_head', type=float, default=0.0,
                   help='P0 双预测头一致性权重（CPM 学生 ← sg(Graph-ODE) 教师；'
                        '0=禁用，仅 graph_ode 路线生效）。')
    p.add_argument('--cpm_horizon_weight', default='gamma',
                   choices=['gamma', 'increasing'],
                   help='CPM 逐 horizon 权重策略：gamma=γ^(h-1)（由 --cpm_gamma '
                        '控制，默认）；increasing=远期递增 1+α(h-1)/(H-1)。')
    p.add_argument('--cpm_horizon_alpha', type=float, default=0.5,
                   help='--cpm_horizon_weight=increasing 时的 α（默认 0.5）。')
    p.add_argument('--cross_head_start_epoch', type=int, default=0,
                   help='双头一致性预热门控：epoch < 该值时暂不启用（默认 0=全程启用）。')
    p.add_argument('--dynamics_metric_horizon', type=int,
                   default=DYNAMICS_CKPT_HORIZON,
                   help='graph_ode 选优指标覆盖的预测步数 N（H1..N，默认 8）；'
                        '指标键名随 N 变化（metric_mae_dynamics_h1_N_subject），'
                        '且只纳入具备完整 N 步真值的锚点。确定性 Graph-ODE 可自由推进'
                        '超过 cpm_horizon，训练监督范围仍由 cpm_horizon 决定。')

    # ---------- S4 概率轨迹（默认 off；开启时才构建新模块） ----------
    p.add_argument('--s4_variant', default='off', choices=['off', 's4_1', 's4_2'],
                   help="S4 概率化链：off=S2 原路径；s4_1=输出端低秩时空残差头；"
                        "s4_2=潜状态低秩随机创新（graph_ode 专用）。")
    p.add_argument('--s4_stage', default='head', choices=['head', 'joint'],
                   help="两阶段训练：head=前 --freeze_backbone_epochs 仅训 S4 模块，"
                        "之后联合微调；joint=从第 0 epoch 起联合训练。")
    p.add_argument('--s4_crps_weight', type=float, default=1.0,
                   help='S4 集合 CRPS 损失权重（固定权重，不进入 uncertainty 聚合）。')
    p.add_argument('--s4_lr_scale', type=float, default=1.0,
                   help='S4 新模块学习率倍率（相对 --lr_peak）。')
    p.add_argument('--s4_train_samples', type=int, default=32,
                   help='训练/验证时每个锚点采样的轨迹条数 K（默认 32）。')
    p.add_argument('--s4_residual_rank', type=int, default=8,
                   help='S4-1 低秩空间因子秩 r（默认 8）。')
    p.add_argument('--s4_residual_hidden', type=int, default=128,
                   help='S4-1 残差头隐层宽度（默认 128）。')
    p.add_argument('--s4_residual_init_scale', type=float, default=0.1,
                   help='S4-1 残差尺度初值（小初始化，默认 0.1；不是零方差）。')
    p.add_argument('--s4_noise_rank', type=int, default=8,
                   help='S4-2 潜状态创新秩 r（默认 8；禁止全 ROI 独立噪声）。')
    p.add_argument('--s4_noise_scale', type=float, default=1e-2,
                   help='S4-2 创新增益初值尺度（小初始化，默认 0.01）。')
    p.add_argument('--s4_ckpt_metric', default='point', choices=['point', 'crps'],
                   help='S4 实验的 checkpoint 判据：point=沿用 graph-ODE 多步 MAE；'
                        'crps=验证集 dynamics CRPS（越低越好）。')
    p.add_argument('--val_sampling_seed', type=int, default=2024,
                   help='S4 启用时验证期采样轨迹所用的随机种子（每个 epoch 重置并'
                        '在验证后恢复原随机流）。使 CRPS 选优基于同一批随机创新，'
                        '避免把采样运气当成模型改进；非 S4 运行不生效。')

    args = p.parse_args()
    if not math.isfinite(args.s4_variogram_weight) or (args.s4_variogram_weight < 0 and args.s4_variogram_weight != -1):
        p.error('--s4_variogram_weight must be nonnegative or -1 (train-only calibration)')
    if args.s4_variogram_weight != 0 and args.s4_variant != 's4_1':
        p.error('P1 temporal variogram requires the existing s4_1 head')
    if args.s4_variant != 'off':
        if args.dynamics_mode != 'graph_ode':
            p.error('--s4_variant 仅支持 --dynamics_mode graph_ode')
        if args.s4_stage == 'head' and args.freeze_backbone_epochs <= 0:
            p.error('--s4_stage head 需要 --freeze_backbone_epochs >= 1')
        if args.s4_train_samples < 2:
            p.error('--s4_train_samples 必须 >= 2（CRPS 需要集合）')
    # 历史入口行为：finetune 省略 --pretrained_weight 时回填默认主权重路径。
    # 本轮脚本始终显式传入起点，不依赖这个默认值。
    if args.mode == 'finetune' and not args.pretrained_weight:
        args.pretrained_weight = LEGACY_PRETRAINED_WEIGHT
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
