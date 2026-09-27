# coding=utf-8
"""checkpoint 兼容加载与模型/数据构建。"""
import logging
import warnings
from typing import Dict, List, Tuple

import torch

from models.neurotwin import NeuroTwin
from utils.common import parse_pred_quantiles
from utils.dataloader import NeuroTwinDataLoader

log = logging.getLogger(__name__)

def load_checkpoint_compat(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    """兼容加载 checkpoint"""
    raw = torch.load(ckpt_path, map_location=device)
    # 兼容三种格式：train.optim.save_backbone_weights / main.py 保存的
    # {'state_dict', 'arch_version', 'config'}、旧格式 {'model': state_dict}、裸 state_dict。
    # 取错层级时（例如把整份元数据 dict 当 state_dict）load_state_dict 会把全部权重
    # 记为 missing，strict=False 下静默跳过、模型停在随机初始化——必须显式报错，
    # 不允许用兜底掩盖权重未加载。
    if isinstance(raw, dict) and isinstance(raw.get('state_dict'), dict):
        state_dict = raw['state_dict']
    elif isinstance(raw, dict) and isinstance(raw.get('model'), dict):
        state_dict = raw['model']
    else:
        state_dict = raw
    if not isinstance(state_dict, dict) or not any('.' in str(k) for k in state_dict):
        raise ValueError(
            f'无法从 {ckpt_path} 解析出参数 state_dict；'
            f'顶层键={list(raw) if isinstance(raw, dict) else type(raw)}')
    
    # 检测版本
    keys = set(state_dict.keys())
    has_new_wta = any('mta.' in k for k in keys)
    has_old_wta = any('attn.' in k and 'window_temporal_attn' in k for k in keys)
    
    version = 'mta' if has_new_wta and not has_old_wta else 'pre'
    log.info(f'Checkpoint version: [{version}]')
    
    if version == 'pre' and strict:
        strict = False
        warnings.warn('Detected old checkpoint, using strict=False', UserWarning)
    
    result = model.load_state_dict(state_dict, strict=strict)
    compat_info = {
        'version': version,
        'missing': list(result.missing_keys) if not strict else [],
        'unexpected': list(result.unexpected_keys) if not strict else [],
    }
    
    return model, compat_info


def build_model_and_loader(args, device: torch.device):
    """构建模型与评估用 DataLoader。

    评估划分由 ``args.eval_split`` 选择（``val`` / ``test``，均为被试级 8:1:1 内部划分）。
    划分参数（seed / val_ratio / test_ratio / stratify_bins）**必须与训练时一致**，
    否则 ``test`` 不再是训练时留出的那批被试。
    """
    eval_split = getattr(args, 'eval_split', 'val')
    if eval_split not in ('val', 'test'):
        raise ValueError(f"eval_split 仅支持 val/test，收到 '{eval_split}'")

    data_loader = NeuroTwinDataLoader(
        data_root=args.data_root,
        mode='finetune',
        batch_size=args.batch_size,
        in_window=args.in_window,
        pred_window=args.pred_window,
        pathology_input_dim=args.pathology_input_dim,
        clinical_file=args.clinical_file,
        total_windows=args.total_windows,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=getattr(args, 'test_ratio', 0.1),
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    if eval_split == 'test':
        eval_data = data_loader.get_test()
        eval_subjects = data_loader.get_test_subjects()
    else:
        eval_data = data_loader.get_val()
        eval_subjects = data_loader.get_val_subjects()
    log.info(f'评估划分: [{eval_split}] | 被试数={len(eval_subjects)} | '
             f'样本数={len(eval_data.dataset)}')
    
    model = NeuroTwin(
        features=args.num_rois,
        in_window=args.in_window,
        in_seq_len=args.seq_len,
        pred_window=args.pred_window,
        pred_seq_len=args.seq_len,
        n_block=args.n_block,
        dropout=args.dropout,
        pathology_input_dim=args.pathology_input_dim,
        pathology_dim=args.pathology_dim,
        adapter_alpha=args.alpha,
        norm=args.norm,
        pretrain_mode=False,
        ode_steps=args.ode_steps,
        ode_hidden_dim=args.ode_hidden_dim,
        num_scales=args.num_scales,
        num_experts=args.num_experts,
        top_k=args.top_k,
        stochastic_depth_rate=args.stochastic_depth_rate,
        moe_gate_temperature=args.moe_gate_temp_end,
        moe_expert_hidden_dim=args.moe_expert_hidden_dim,
        moe_use_shared_expert=args.moe_use_shared_expert,
        moe_router_cond_only=args.moe_router_cond_only,
        moe_use_argmax=args.moe_use_argmax,
        moe_inference_temperature=args.moe_inference_temperature,
        # 预测头需与训练时一致，否则概率头会以随机初始化权重参与校准指标计算
        pred_head=args.pred_head,
        pred_quantiles=parse_pred_quantiles(args.pred_quantiles),
    ).to(device)
    
    model, compat_info = load_checkpoint_compat(
        model=model,
        ckpt_path=args.finetuned_weight,
        device=device,
        strict=args.strict_load,
    )

    return model, eval_data, compat_info
    
