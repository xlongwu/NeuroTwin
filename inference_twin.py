"""方案 B 单被试推理：固定扫描校准、字段条件、直接 H32 和观测刷新预测。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict,Optional,Tuple
import numpy as np
import torch

from model import ROOT,load_model_checkpoint,load_model_config

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


class TwinSession:
    """同一扫描固定校准尺度；每次从真实历史直接预测，不构造 ODE 状态。"""

    def __init__(self,checkpoint:str,device:str='',condition_file:Optional[str]=None):
        self.device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model,self.args,self.bundle=load_model_checkpoint(checkpoint,self.device)
        self.num_rois=self.args.num_rois
        self.calibration_length=self.args.calibration_length
        self.context_min,self.context_max=self.args.context_min,self.args.context_max
        self.default_history_len=self.context_max
        self.series=self.mu0=self.s0=None
        self.subject_id=None
        self._cond_cache,self._demo_cache={},{}
        from utils.dataloader import load_condition_table
        from utils.condition_schema import ConditionStats,schema_for
        group=self.bundle['config']['group']
        filename=condition_file or ('Rest-meta-MDD-V1V2-Merged-HC.xlsx' if group=='hc3'
                                   else 'Rest-meta-MDD-V1V2-Merged-MDD.xlsx')
        self._table=load_condition_table(ROOT/self.args.data_root,filename,group)
        self._cond_stats=ConditionStats.from_dict(schema_for(group),self.bundle['stats'])
        names=[s.name for s in self._table.specs]
        self._demo_idx={key:names.index(key) for key in ('Age','education','Sex')}

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


    def _history_len(self, history_len: Optional[int], t_avail: int) -> int:
        k = int(history_len or self.default_history_len)
        k = min(k, t_avail)
        if k < self.context_min or k > self.context_max:
            raise ValueError(f'history_len={k} 超出训练 context 范围 '
                             f'[{self.context_min},{self.context_max}]（可用 TR={t_avail}）')
        return k


    @torch.no_grad()
    def forecast(self,cutoff:Optional[int]=None,horizon:Optional[int]=None):
        if self.series is None:raise ValueError('请先 calibrate_scan')
        k=self._history_len(None,len(self.series))
        t=k-1 if cutoff is None else int(cutoff)
        if t<k-1 or t>=len(self.series):raise ValueError('cutoff 不在可用历史范围')
        h=self.model.cpm_horizon if horizon is None else int(horizon)
        if not 1<=h<=self.model.cpm_horizon:raise ValueError('预测范围须位于 [1,H32]')
        history=torch.from_numpy(self.series[t-k+1:t+1].astype(np.float32))[None].to(self.device)
        pred=self.model.forecast_horizon(history,self.condition_for(self.subject_id))
        return pred[0,:h].cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='')
    parser.add_argument('--mode',choices=['pretrain','finetune'],default='finetune')
    parser.add_argument('--raw',required=True)
    parser.add_argument('--subject_id',required=True)
    parser.add_argument('--condition_file',default='')
    parser.add_argument('--horizon',type=int,default=32)
    parser.add_argument('--cutoff',type=int,default=None)
    parser.add_argument('--device',default='')
    parser.add_argument('--out_npz',default='')
    parser.add_argument('--output',default='')
    ns=parser.parse_args()
    checkpoint=ns.checkpoint or str(ROOT/load_model_config()['weights'][ns.mode]['path'])
    session=TwinSession(checkpoint,ns.device,ns.condition_file or None)
    session.subject_id=ns.subject_id
    session.calibrate_scan(load_raw_series(ns.raw,session.num_rois))
    prediction=session.forecast(ns.cutoff,ns.horizon)
    result=dict(checkpoint=checkpoint,subject_id=ns.subject_id,horizon=ns.horizon,
                prediction_shape=list(prediction.shape),scale='first-32TR calibrated BOLD')
    if ns.out_npz:np.savez_compressed(ns.out_npz,prediction=prediction)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if ns.output:Path(ns.output).write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':main()
