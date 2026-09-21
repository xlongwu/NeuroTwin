# coding=utf-8
"""checkpoint 兼容加载与模型/数据构建。"""
import logging
import warnings
from typing import Dict, List, Tuple

import torch

from models.neurotwin import NeuroTwin
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
    state_dict: dict = raw.get('model', raw) if isinstance(raw, dict) and 'model' in raw else raw
    
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
    """构建模型和数据加载器"""
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
        stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_data = data_loader.get_val()
    
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
    ).to(device)
    
    model, compat_info = load_checkpoint_compat(
        model=model,
        ckpt_path=args.finetuned_weight,
        device=device,
        strict=args.strict_load,
    )
    
