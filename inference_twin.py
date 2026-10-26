# coding=utf-8
"""NeuroTwin 孪生推理入口（方案 §10）。

封装 graph_ode 动态路线的全部推理路径：

- 使用现有 args/condition bundle 重建新结构并 **strict** 加载 checkpoint；
  不自动搜索"最新"权重，不临时随机创建临床头输出评分。
- Raw 前缀校准：每份扫描用其最早 ``calibration_length`` TR 计算固定
  ``mu0/s0`` 并标准化整个序列（与 ``utils/dataloader.py`` raw 分支同口径）；
  新原始观测先用**同一扫描**的固定 mu0/s0 转数据尺度。
- 个体 profile 一次计算后供 rollout / 观测更新复用；新访视重新校准扫描尺度
  并重新初始化 state，可显式沿用已有 p（``reuse_profile=True``），不拼接
  不同尺度的 history buffer。
- ``analyze_pathology`` 必须显式传 reference 文件；``predict_outcome`` 必须
  显式传 outcome checkpoint 文件。

边界声明：未经真实受控数据训练的控制分支只能输出**未校准的模拟响应**；
预测 std 不自动称校准置信区间；健康偏离不称诊断概率或因果病灶。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

#: 默认初始历史长度（训练 context_max=64；若 checkpoint 不同则取其值）
DEFAULT_HISTORY_LEN = 64
#: 健康参照 / outcome 基线固定的历史长度
REFERENCE_TR = 64


# ══════════════════════════════════════════════════════════════════════════
#  数据文件读取
# ══════════════════════════════════════════════════════════════════════════

def load_raw_series(path, num_rois: int) -> np.ndarray:
    """Raw ROI 序列 → ``[T,F]``（原始尺度，float64）。

    ``.mat``：键 ``ROISignals``/``Data``，按该来源明确的 ``[T,F]`` 布局解释
    （第 0 维 TR、第 1 维 ROI，不用 ``max(shape)`` 猜时间轴），取前
    ``num_rois`` 列（完整时间轴，不截断）；``.npz``：键 ``bold``；
    ``.npy``：直接数组。缺失/形状非法直接报路径，不回退旧缓存。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Raw 序列不存在: {p}')
    suffix = p.suffix.lower()
    if suffix == '.mat':
        from scipy.io import loadmat
        m = loadmat(str(p))
        for key in ('ROISignals', 'Data'):
            if key in m:
                arr = np.asarray(m[key], dtype=np.float64)
                break
        else:
            raise KeyError(f'{p} 缺少键 ROISignals/Data（现有键='
                           f'{[k for k in m if not k.startswith("__")]}）')
    elif suffix == '.npz':
        raw = np.load(str(p))
        if 'bold' not in raw:
            raise KeyError(f'{p} 缺少键 bold（现有键={list(raw.keys())}）')
        arr = np.asarray(raw['bold'], dtype=np.float64)
    elif suffix == '.npy':
        arr = np.asarray(np.load(str(p)), dtype=np.float64)
    else:
        raise ValueError(f'不支持的序列格式 {suffix}（{p}；支持 .mat/.npz/.npy）')
    if arr.ndim != 2:
        raise ValueError(f'{p} 期望二维 [T,F]，收到 {arr.shape}')
    if arr.shape[1] < num_rois:
        raise ValueError(f'{p} 仅 {arr.shape[1]} 列 ROI，少于 num_rois='
                         f'{num_rois}，无法取前 {num_rois} 列')
    arr = arr[:, :num_rois]
    if not np.isfinite(arr).all():
        raise ValueError(f'{p} 含非有限值')
    return arr


def load_controls_npz(path, num_rois: int, series_t: Optional[int] = None) -> np.ndarray:
    """控制 NPZ（键 ``u``，``[T,F]``）→ float64；与序列 T/F 匹配（方案 §4.3）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'控制文件不存在: {p}')
    raw = np.load(str(p))
    if 'u' not in raw:
        raise KeyError(f'{p} 缺少键 u（现有键={list(raw.keys())}）')
    u = np.asarray(raw['u'], dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != num_rois:
        raise ValueError(f'{p} u 期望 [T,{num_rois}]，收到 {u.shape}')
    if not np.isfinite(u).all():
        raise ValueError(f'{p} u 含非有限值')
    if series_t is not None and u.shape[0] != series_t:
        raise ValueError(f'{p} u 长度 {u.shape[0]} 与序列 T={series_t} 不一致'
                         '（完整时间轴口径，方案 §4.3）')
    return u


def future_control_slice(u: np.ndarray, anchor_t: int, horizon: int) -> np.ndarray:
    """锚点 t 的未来控制切片 ``u[t+1:t+H+1]``；缺尾区间按方案 §4.3 填零。"""
    f_dim = u.shape[1]
    out = np.zeros((horizon, f_dim), dtype=u.dtype)
    lo, hi = anchor_t + 1, min(anchor_t + 1 + horizon, u.shape[0])
    if lo < hi:
        out[:hi - lo] = u[lo:hi]
    return out


# ══════════════════════════════════════════════════════════════════════════
#  会话：Raw 校准、profile、持续状态
# ══════════════════════════════════════════════════════════════════════════

class TwinSession:
    """单被试推理会话：扫描校准 → profile → 持续状态推进 / 观测更新。

    所有 B=1；张量在 ``self.device``。每次 :meth:`calibrate_scan` / 
    :meth:`new_visit` 更新当前扫描的固定 mu0/s0；跨访视不共用扫描尺度。
    """

    def __init__(self, checkpoint: str, device: str = '',
                 condition_file: Optional[str] = None):
        from analysis.evaluate import (build_model_from_args,
                                                  load_condition_bundle,
                                                  load_train_args)
        from analysis.checkpoint import load_checkpoint_compat

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f'checkpoint 不存在: {checkpoint}'
                                    '（不自动搜索"最新"权重，请显式指定）')
        self.device = torch.device(device if device else
                                   ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.args = load_train_args(checkpoint)
        self.bundle = load_condition_bundle(checkpoint)
        self.model = build_model_from_args(self.args, self.device,
                                           condition_bundle=self.bundle)
        self.model, compat = load_checkpoint_compat(self.model, checkpoint,
                                                    self.device, strict=True)
        if compat.get('missing') or compat.get('unexpected'):
            raise RuntimeError(f'权重加载不完整（strict）：{compat}')
        self.model.eval()
        for p_ in self.model.parameters():
            p_.requires_grad_(False)
        if self.model.dynamics_mode != 'graph_ode':
            raise ValueError(
                f'inference_twin 只服务 graph_ode 路线，checkpoint 的 '
                f"dynamics_mode={self.model.dynamics_mode}（discrete 无持续状态）。")

        self.num_rois = int(self.args.num_rois)
        self.calibration_length = int(getattr(self.args, 'calibration_length', 0) or 0)
        if self.calibration_length < 1:
            raise ValueError('checkpoint 的 calibration_length 必须 >= 1'
                             '（Raw 前缀校准口径，方案 §4.1）。')
        context_max = int(getattr(self.args, 'context_max', DEFAULT_HISTORY_LEN))
        context_min = int(getattr(self.args, 'context_min', 16))
        self.default_history_len = max(context_min,
                                       min(DEFAULT_HISTORY_LEN, context_max))
        self.context_min, self.context_max = context_min, context_max
        self.graph_source = str(getattr(self.args, 'graph_source', 'legacy_mask'))
        self.personalized = bool(self.model.personalized)

        # 条件表（八字段）与训练统计量；condition_file 显式覆盖训练时记录
        self._table = None
        self._cond_stats = None
        self._demo_idx = None
        self._cond_cache: Dict[str, dict] = {}
        self._demo_cache: Dict[str, dict] = {}
        self._setup_condition(condition_file)

        # 当前扫描状态
        self.mu0: Optional[np.ndarray] = None        # [F]
        self.s0: Optional[np.ndarray] = None         # [F]
        self.series: Optional[np.ndarray] = None     # [T,F] 校准后数据尺度
        self.profile: Optional[torch.Tensor] = None  # [1,P]
        self._last_cond = None                       # init_state 缓存（forward 用）
        self._last_cal_bold = None
        self._last_sc = None

    # ---- 条件 / 人口学 ----
    def _setup_condition(self, condition_file: Optional[str]) -> None:
        if self.model.conditioner is None and condition_file is None:
            return
        from utils.dataloader import load_condition_table
        from utils.condition_schema import ConditionStats, schema_for
        from analysis.twin_reference import _demo_field_indices

        group = ((self.bundle or {}).get('config') or {}).get('group')
        cond_file = condition_file or getattr(self.args, 'condition_file', None)
        if not group or not cond_file:
            raise ValueError(
                '需要八字段条件表：bundle.config.group 与 --condition_file'
                '（缺省取 checkpoint 训练值）均需存在。')
        data_root = getattr(self.args, 'data_root')
        self._table = load_condition_table(data_root, cond_file, group)
        self._cond_stats = ConditionStats.from_dict(
            schema_for(group), (self.bundle or {}).get('stats') or {})
        self._demo_idx = _demo_field_indices(self._table.specs)

    def condition_for(self, subject_id: Optional[str]) -> Optional[dict]:
        """subject_id → cond dict（字段编码一次后缓存）；无 conditioner 返回 None。"""
        if self.model.conditioner is None:
            return None
        if not subject_id:
            raise ValueError('该 checkpoint 带条件编码器，必须提供 --subject_id'
                             ' 以从条件表对齐八字段基线条件。')
        sid = str(subject_id)
        if sid not in self._cond_cache:
            values, mask = self._table.align([sid])
            z, emb = self._cond_stats.encode(values, mask)
            self._cond_cache[sid] = {
                'cond_z': torch.from_numpy(z).to(self.device),
                'cond_code': torch.from_numpy(emb).to(self.device),
                'cond_mask': torch.from_numpy(mask.astype(np.float32)).to(self.device),
            }
        return self._cond_cache[sid]

    def demo_for(self, subject_id: str) -> Dict[str, Optional[float]]:
        """subject_id → 参照分析人口学 {Age, education, Sex}（只用这三字段）。"""
        sid = str(subject_id)
        if sid not in self._demo_cache:
            if self._table is None:
                raise ValueError('健康参照需要人口学（Age/education/Sex）：'
                                 '请提供 --condition_file 以对齐条件表。')
            values, mask = self._table.align([sid])
            out: Dict[str, Optional[float]] = {}
            for name in ('Age', 'education', 'Sex'):
                j = self._demo_idx[name]
                out[name] = (float(values[0, j]) if mask[0, j] > 0
                             and np.isfinite(values[0, j]) else None)
            self._demo_cache[sid] = out
        return self._demo_cache[sid]

    # ---- Raw 校准 ----
    def calibrate_scan(self, raw_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """原始 ``[T,F]`` 序列 → 前缀校准（固定 mu0/s0）并替换当前扫描状态。

        每 ROI 只用最早 ``calibration_length`` 个 TR 计算统计量
        （s0=sqrt(var+1e-6)，与训练 loader raw 分支同口径），再标准化整个
        可用序列。返回 ``(X [T,F] 数据尺度, mu0 [F], s0 [F])``。
        """
        x = np.asarray(raw_series, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.num_rois:
            raise ValueError(f'raw 期望 [T,{self.num_rois}]，收到 {x.shape}')
        if x.shape[0] < self.calibration_length:
            raise ValueError(f'序列只有 {x.shape[0]} 个 TR，少于 calibration_length='
                             f'{self.calibration_length}，无法前缀校准')
        cal = x[:self.calibration_length]
        mu0 = cal.mean(axis=0)                                     # [F]
        s0 = np.sqrt(cal.var(axis=0) + 1e-6)                       # [F]
        self.mu0, self.s0 = mu0, s0
        self.series = (x - mu0) / s0
        return self.series, mu0, s0

    def to_data_scale(self, raw_obs: np.ndarray) -> np.ndarray:
        """新原始观测（标量 [F] 或序列 [T,F]）用**同一扫描**固定 mu0/s0 转数据尺度。"""
        if self.mu0 is None:
            raise ValueError('尚未 calibrate_scan：新观测必须用同一扫描的固定 mu0/s0'
                             ' 转数据尺度，不能临时重估统计量。')
        obs = np.asarray(raw_obs, dtype=np.float64)
        return (obs - self.mu0) / self.s0

    # ---- profile ----
    @torch.no_grad()
    def compute_profile(self) -> torch.Tensor:
        """当前扫描最早 calibration_length TR（数据尺度）→ p ``[1,P]``，一次计算后复用。"""
        if self.series is None:
            raise ValueError('尚未 calibrate_scan，无 profile 可算。')
        cal = self.series[:self.calibration_length]                # [L,F]
        cal_t = torch.from_numpy(cal.T.astype(np.float32))         # [F,L]
        self.profile = self.model.calibrate_profile(
            cal_t[None, :, None, :].to(self.device))
        return self.profile

    # ---- state ----
    def _history_len(self, history_len: Optional[int], t_avail: int) -> int:
        k = int(history_len or self.default_history_len)
        k = min(k, t_avail)
        if k < self.context_min or k > self.context_max:
            raise ValueError(f'history_len={k} 超出训练 context 范围 '
                             f'[{self.context_min},{self.context_max}]（可用 TR={t_avail}）')
        return k

    def _load_sc(self, sc_path: Optional[str]) -> Optional[torch.Tensor]:
        if self.graph_source != 'legacy_mask':
            return None
        if not sc_path:
            raise ValueError('graph_source=legacy_mask 需要 --sc 提供外部图；'
                             'history_fc 路线从当前可见历史计算图。')
        from scipy.io import loadmat
        m = loadmat(str(sc_path))
        for key in ('SC', 'Mask', 'sc'):
            if key in m:
                arr = np.asarray(m[key], dtype=np.float32)
                break
        else:
            arr = np.asarray([v for k, v in m.items() if not k.startswith('__')][0])
        if arr.shape != (self.num_rois, self.num_rois):
            raise ValueError(f'外部图期望 [{self.num_rois},{self.num_rois}]，'
                             f'收到 {arr.shape}（{sc_path}）')
        return torch.from_numpy(arr)[None].to(self.device)

    @torch.no_grad()
    def init_state(self, history_len: Optional[int] = None,
                   profile: Optional[torch.Tensor] = None,
                   reuse_profile: bool = False,
                   sc_path: Optional[str] = None) -> dict:
        """当前扫描最早 K 个 TR（数据尺度）→ 持续状态（方案 §6.3）。

        Args:
            profile: 显式传入已计算的 p（跨访视复用时由 :meth:`new_visit` 转交）；
            reuse_profile: True 时沿用 ``self.profile``，不重新校准。
        """
        if self.series is None:
            raise ValueError('尚未 calibrate_scan，无法初始化状态。')
        k = self._history_len(history_len, self.series.shape[0])
        hist = self.series[:k]                                     # [K,F] 数据尺度
        hist_t = torch.from_numpy(hist.T.astype(np.float32))[None].to(self.device)
        cond = self.condition_for(getattr(self, 'subject_id', None))

        if self.personalized:
            if profile is None and reuse_profile:
                profile = self.profile
            if profile is None:
                profile = self.compute_profile()
            cal_t = torch.from_numpy(
                self.series[:self.calibration_length].T.astype(np.float32))
            cal_bold = cal_t[None, :, None, :].to(self.device)
        else:
            cal_bold = None
        sc_t = self._load_sc(sc_path)
        # 缓存供 forward_dynamics 使用（与 state 同一编码来源）
        self._last_cond, self._last_cal_bold, self._last_sc = cond, cal_bold, sc_t
        state = self.model.initialize_state(
            hist_t, sc_matrix=sc_t, pathology_score=None,
            condition=cond, profile=profile, calibration_bold=cal_bold)
        return state

    @torch.no_grad()
    def advance(self, state: dict, control: Optional[np.ndarray] = None
                ) -> Tuple[dict, np.ndarray]:
        """推进一个区间；control ``[F]`` 逐 ROI 有效驱动。返回 (新 state, 下一帧 [F])。"""
        c_t = None
        if control is not None:
            c_t = torch.from_numpy(np.asarray(control, dtype=np.float32)
                                   .reshape(1, -1)).to(self.device)
        new_state = self.model.advance_state(state, control=c_t)
        return new_state, self.model.readout_state(new_state)[0].cpu().numpy()

    @torch.no_grad()
    def observe(self, state: dict, raw_obs: np.ndarray) -> dict:
        """真实观测（原始尺度 [F]）→ 同一扫描固定 mu0/s0 数据尺度 → update_state。"""
        obs = self.to_data_scale(raw_obs)
        if obs.shape != (self.num_rois,):
            raise ValueError(f'观测期望 [{self.num_rois}]，收到 {obs.shape}')
        obs_t = torch.from_numpy(obs.astype(np.float32))[None].to(self.device)
        return self.model.update_state(state, obs_t)

    @torch.no_grad()
    def rollout(self, state: dict, steps: int,
                controls: Optional[np.ndarray] = None) -> np.ndarray:
        """从已初始化 state 持续推进 ``steps`` 步（不回填重编码）→ ``[steps,F]``。

        controls ``[steps,F]``：第 h 行对应第 h 个区间的有效驱动；None 与
        全零严格等价。图、已编码 c、profile 在无观测推进期间固定。
        """
        if controls is not None and np.asarray(controls).shape != (steps, self.num_rois):
            raise ValueError(f'controls 期望 [{steps},{self.num_rois}]，'
                             f'收到 {np.asarray(controls).shape}')
        cur = state
        frames: List[np.ndarray] = []
        for h in range(int(steps)):
            u = None if controls is None else np.asarray(controls)[h]
            cur, x = self.advance(cur, control=u)
            frames.append(x)
        return np.stack(frames, axis=0)

    @torch.no_grad()
    def simulate(self, state: dict, controls: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """同初始状态分别用 U 与零 U 推进 → ``(x_ctrl, x_zero, diff)`` 各 ``[H,F]``。

        差值由两条数据尺度轨迹相减；未经真实受控数据训练时只能称
        **未校准的模拟响应**。
        """
        # v2 §8（B8）：运行时显式声明边界，防止模拟差值被当作结论引用
        log.warning('simulate(): 干预通路未经真实受控数据校准，输出仅是未校准的'
                    '模拟响应，不可用于模拟结论或因果解释')
        u = np.asarray(controls, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != self.num_rois:
            raise ValueError(f'controls 期望 [H,{self.num_rois}]，收到 {u.shape}')
        ctrl_state, zero_state = state, state       # advance_state 不原位改写
        xs_ctrl: List[np.ndarray] = []
        xs_zero: List[np.ndarray] = []
        for h in range(u.shape[0]):
            ctrl_state, xc = self.advance(ctrl_state, control=u[h])
            zero_state, xz = self.advance(zero_state, control=None)
            xs_ctrl.append(xc)
            xs_zero.append(xz)
        x_ctrl = np.stack(xs_ctrl, axis=0)
        x_zero = np.stack(xs_zero, axis=0)
        return x_ctrl, x_zero, x_ctrl - x_zero

    @torch.no_grad()
    def forward_dynamics(self, state: dict, horizon: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """训练同路的 forward 多步动态预测 → ``(dynamics_pred [F,H], dynamics_std [F,H], one_step [F])``。

        使用与训练一致的 VarianceHead 读出（数据尺度）；std 是模型误差方差的
        点估计，**不是校准置信区间**。
        """
        hist = state['history'][0].cpu().numpy()                   # [K,F] 数据尺度
        x4 = torch.from_numpy(hist.T.astype(np.float32))[None, :, None, :].to(self.device)
        outputs, aux = self.model(
            x4, sc_matrix=self._last_sc, pathology_score=None,
            condition=self._last_cond, condition_epoch=None,
            calibration_bold=self._last_cal_bold,
            profile=state.get('profile'), dynamics_horizon=int(horizon))
        one_step = outputs[0, :, 0, 0].cpu().numpy()               # [F] 下一 TR 预测
        return (aux['dynamics_pred'][0].cpu().numpy(),
                aux['dynamics_std'][0].cpu().numpy(), one_step)

    # ---- 新访视 ----
    def new_visit(self, raw_series: np.ndarray, subject_id: Optional[str] = None,
                  reuse_profile: bool = False) -> dict:
        """新访视：重新校准**该扫描**的尺度并初始化 state。

        不拼接不同尺度的 history buffer；``reuse_profile=True`` 显式沿用已有
        个人基线码 p（跨访视稳定性是待验证性质）。
        """
        self.calibrate_scan(raw_series)
        if subject_id is not None:
            self.subject_id = str(subject_id)
        if not reuse_profile:
            self.profile = None
        return self.init_state()


# ══════════════════════════════════════════════════════════════════════════
#  健康参照与临床结局（显式文件）
# ══════════════════════════════════════════════════════════════════════════

def analyze_pathology(history: np.ndarray, demo: Dict[str, Optional[float]],
                      reference_path: str) -> Dict[str, np.ndarray]:
    """固定 64 TR 历史（数据尺度 ``[B,64,F]``）+ 人口学 → 健康参照偏离。

    reference 文件必须显式传入（``analysis.twin_reference fit`` 产出）；
    输出 ``normal_residual [B,F,6]`` / ``normal_distance [B,F]``——**不称
    诊断概率或因果病灶**。
    """
    from analysis.twin_reference import load_reference, score_reference
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim != 3 or hist.shape[1] != REFERENCE_TR:
        raise ValueError(f'history 期望 [B,{REFERENCE_TR},F]，收到 {hist.shape}')
    if not reference_path or not Path(reference_path).exists():
        raise FileNotFoundError(
            f'--reference 必须显式提供且存在（当前: {reference_path!r}）；'
            '健康偏离分析不使用默认或隐式参照。')
    payload = load_reference(reference_path)
    return score_reference(hist, demo, payload)


def predict_outcome(raw_bold: np.ndarray, controls_u: np.ndarray,
                    followup_days: float, subject_id: str,
                    dynamics_checkpoint: str, outcome_path: str,
                    baseline_scores: Optional[np.ndarray] = None,
                    condition_file: Optional[str] = None,
                    device: str = '') -> Dict[str, np.ndarray]:
    """独立临床结局读出推理（方案 §8）：基线信息 + 基线模拟 → HAMD/HAMA 变化。

    outcome checkpoint 必须显式传入且 ``arch='twin_outcome_v1'``；未拟合的
    读出（文件缺失/格式不符）直接报错，**不输出疗效评分**。只使用基线信息
    与基线模拟，不输入真实随访 BOLD/随访条件。
    """
    p = Path(outcome_path)
    if not p.exists():
        raise FileNotFoundError(
            f'--outcome 必须显式提供且存在（当前: {outcome_path!r}）；'
            '未拟合的 outcome 读出不输出任何评分。')
    ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get('arch') != 'twin_outcome_v1':
        raise ValueError(f'{p} 不是 twin_outcome_v1 读出 checkpoint'
                         f'（arch={ckpt.get("arch") if isinstance(ckpt, dict) else "?"}）；'
                         '请先用 python -m train.twin_outcome 拟合。')
    cfg = ckpt.get('config') or {}
    from train.twin_outcome import OutcomeFeatureExtractor
    # 参照取 outcome 训练时保存的文件，保证特征与训练逐位同源
    extractor = OutcomeFeatureExtractor(
        dynamics_checkpoint, cfg['reference'],
        condition_file=(condition_file or cfg.get('condition_file') or None),
        device=(torch.device(device) if device else None))
    feat = extractor.extract_from_arrays(raw_bold, controls_u,
                                         float(followup_days), str(subject_id))
    from models.twin_outcome import (FEATURE_DIM, ClinicalOutcomeHead,
                                     OutcomeStandardizer, predict_with_baseline)
    standardizer = OutcomeStandardizer.from_dict(ckpt['standardizer'])
    head = ClinicalOutcomeHead(in_dim=int(cfg.get('in_dim', FEATURE_DIM)),
                               hidden_dim=int(cfg.get('hidden_dim', 64)))
    head.load_state_dict(ckpt['state_dict'], strict=True)
    feats_std = standardizer.transform_features(feat[None])        # [1,83]
    base = None
    if baseline_scores is not None:
        base = np.asarray(baseline_scores, dtype=np.float64).reshape(1, 2)
    out = predict_with_baseline(head, torch.from_numpy(feats_std).to(extractor.device),
                                baseline_scores=base, standardizer=standardizer)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def _to_list(a: np.ndarray) -> list:
    return np.asarray(a, dtype=np.float64).tolist()


def main():
    ap = argparse.ArgumentParser(
        description='NeuroTwin 孪生推理（graph_ode 路线；方案 §10）')
    ap.add_argument('--checkpoint', required=True,
                    help='graph_ode 动态 checkpoint（显式路径，不搜索"最新"）')
    ap.add_argument('--raw', required=True,
                    help='Raw ROI 序列（.mat 键 ROISignals/Data / .npz 键 bold / .npy）[T,F]')
    ap.add_argument('--subject_id', default='',
                    help='条件表/参照人口学对齐用（checkpoint 带条件编码器时必需）')
    ap.add_argument('--condition_file', default='',
                    help='八字段条件 merged 表；缺省用 checkpoint 训练值')
    ap.add_argument('--sc', dest='sc_path', default='',
                    help='legacy_mask 路线的外部图；history_fc 路线不需要')
    ap.add_argument('--history_len', type=int, default=0,
                    help='初始历史 K（默认 min(64, context_max)，须在训练范围内）')
    ap.add_argument('--horizon', type=int, default=8, help='无观测推进步数 H')
    ap.add_argument('--controls', default='',
                    help='控制 NPZ（键 u [T,F]；锚点 t 切片 u[t+1:t+H+1]）')
    ap.add_argument('--zero_controls', action='store_true',
                    help='显式声明无治疗（全零控制），用于 outcome 特征')
    ap.add_argument('--observe', type=int, default=0,
                    help='用 raw 中历史之后的真实 TR 做 N 步观测更新（教师强制演示）')
    ap.add_argument('--reference', default='',
                    help='健康参照 NPZ（twin_reference fit 产出）；analyze_pathology 必需')
    ap.add_argument('--outcome', default='',
                    help='twin_outcome checkpoint；predict_outcome 必需')
    ap.add_argument('--followup_days', type=float, default=0.0,
                    help='outcome 特征：随访天数')
    ap.add_argument('--baseline_hamd', type=float, default=float('nan'))
    ap.add_argument('--baseline_hama', type=float, default=float('nan'))
    ap.add_argument('--device', default='')
    ap.add_argument('--output', default='', help='JSON 摘要输出路径')
    ap.add_argument('--out_npz', default='', help='数组输出 NPZ 路径')
    args = ap.parse_args()

    session = TwinSession(args.checkpoint, device=args.device,
                          condition_file=(args.condition_file or None))
    if args.subject_id:
        session.subject_id = args.subject_id

    raw = load_raw_series(args.raw, session.num_rois)
    session.calibrate_scan(raw)
    log.info('扫描校准：T=%d, calibration_length=%d（固定 mu0/s0）',
             raw.shape[0], session.calibration_length)
    state = session.init_state(history_len=(args.history_len or None),
                               sc_path=(args.sc_path or None))
    k = int(state['history'].shape[1])

    result: Dict[str, object] = {'checkpoint': args.checkpoint, 'raw': args.raw,
                                 'history_len': k, 'horizon': args.horizon}
    arrays: Dict[str, np.ndarray] = {}

    # ---- 无控制 rollout（持续状态预测）----
    pred = session.rollout(state, args.horizon)                    # [H,F]
    arrays['pred_rollout'] = pred
    log.info('rollout [%d,%d] 完成', *pred.shape)

    # ---- 训练同路 forward 多步预测 + std（非校准置信区间）----
    try:
        dyn, dyn_std, one_step = session.forward_dynamics(state, args.horizon)
        arrays.update({'pred_dynamics': dyn, 'pred_dynamics_std': dyn_std,
                       'pred_one_step': one_step})
    except (ValueError, KeyError, TypeError) as exc:
        log.warning('forward 多步动态预测不可用（%s）；仅报告 rollout 轨迹。', exc)

    # ---- 控制双轨迹 ----
    controls_full: Optional[np.ndarray] = None
    if args.controls:
        controls_full = load_controls_npz(args.controls, session.num_rois,
                                          series_t=raw.shape[0])
    elif args.zero_controls:
        controls_full = np.zeros_like(raw)
    if controls_full is not None:
        u_h = future_control_slice(controls_full, anchor_t=k - 1,
                                   horizon=args.horizon)           # [H,F]
        x_ctrl, x_zero, diff = session.simulate(state, u_h)
        arrays.update({'pred_controlled': x_ctrl, 'pred_zero_control': x_zero,
                       'control_diff': diff})
        log.info('控制双轨迹完成（差值 RMS=%.4g；未经受控数据训练=未校准模拟）',
                 float(np.sqrt((diff ** 2).mean())))

    # ---- 观测更新（教师强制演示：推进 1 步 → 真实观测替换末帧重编码）----
    if args.observe > 0:
        n = int(args.observe)
        if k + n > raw.shape[0]:
            raise ValueError(f'观测更新需要 {k + n} 个 TR，序列只有 {raw.shape[0]}')
        errs = []
        cur = state
        for i in range(n):
            cur, x_pred = session.advance(cur)                     # 预测第 K+i 帧
            obs_raw = raw[k + i]                                   # 真实原始观测
            cur = session.observe(cur, obs_raw)                    # 同一 mu0/s0
            obs_data = session.to_data_scale(obs_raw)
            errs.append(float(np.sqrt(((x_pred - obs_data) ** 2).mean())))
        arrays['observe_rmse_per_step'] = np.asarray(errs)
        result['observe_rmse'] = errs
        log.info('观测更新 %d 步完成（逐步 RMSE=%s）', n,
                 np.array2string(np.asarray(errs), precision=4))

    # ---- 健康参照偏离（显式 reference）----
    if args.reference:
        hist64 = session.series[:REFERENCE_TR]                     # [64,F] 数据尺度
        ref = analyze_pathology(hist64[None], session.demo_for(args.subject_id),
                                args.reference)
        arrays.update({'normal_residual': ref['normal_residual'][0],
                       'normal_distance': ref['normal_distance'][0],
                       'descriptors': ref['descriptors'][0]})
        result['normal_distance_mean'] = float(ref['normal_distance'][0].mean())
        log.info('健康参照偏离：mean|distance|=%.4f（不构成诊断）',
                 result['normal_distance_mean'])

    # ---- 独立临床结局读出（显式 outcome）----
    if args.outcome:
        if controls_full is None:
            raise ValueError('predict_outcome 需要 --controls（或显式 --zero_controls）：'
                             '不把缺控制路径猜成 sham。')
        u8 = future_control_slice(controls_full, anchor_t=k - 1, horizon=8)
        base = np.array([args.baseline_hamd, args.baseline_hama], dtype=np.float64)
        base = None if np.isnan(base).all() else base
        if not args.followup_days > 0:
            raise ValueError('--followup_days 必须为正（outcome 特征必需）')
        out = predict_outcome(raw, u8, args.followup_days,
                              subject_id=(args.subject_id or 'NA'),
                              dynamics_checkpoint=args.checkpoint,
                              outcome_path=args.outcome, baseline_scores=base,
                              condition_file=(args.condition_file or None),
                              device=args.device)
        result['outcome'] = {kk: _to_list(v) for kk, v in out.items()}
        log.info('outcome: delta_mean=%s delta_std=%s（未拟合不输出评分）',
                 np.array2string(out['delta_mean'][0], precision=3),
                 np.array2string(out['delta_std'][0], precision=3))

    # ---- 落盘 ----
    if args.out_npz:
        np.savez(args.out_npz, **arrays)
        result['out_npz'] = args.out_npz
    payload = json.dumps(result, ensure_ascii=False, indent=2,
                         default=lambda o: float(o))
    if args.output:
        Path(args.output).write_text(payload, encoding='utf-8')
        log.info('摘要已写入 %s', args.output)
    else:
        print(payload)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    main()
