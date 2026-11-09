# coding=utf-8
"""唯一V2配置的权重身份核对与严格模型加载。"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from models.tfm import NeuroTwinTFM
from train.optim import read_checkpoint

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = ROOT / 'configs/neurotwin.json'
MODEL_ALIASES = dict(features='num_rois',patch_len='tfm_patch_len',dim='tfm_dim',
                     num_layers='tfm_layers',num_heads='tfm_heads',ff_ratio='tfm_ff_ratio',
                     use_revin='norm')


def load_model_config() -> dict:
    with open(MODEL_CONFIG,encoding='utf-8') as f:
        return json.load(f)


def config_value(cfg: dict, path):
    value=cfg
    for key in path:value=value[key]
    return value


# 只核对影响有效计算、优化与选模的配方；路径及运行并发不参与身份约束。
TRAINING_CONTRACT = ('seed','sampling_seed','cond_dropout_seed','batch_size','train_epochs',
                    'lr_init','lr_peak','lr_final','warmup_epochs','weight_decay','grad_clip',
                    'patience','freeze_backbone_epochs','backbone_lr_scale',
                    'ema_decay','amp','selection_corr_weight','train_samples_per_subject',
                    'random_context','random_cutoff','eval_context_length','eval_anchors_per_subject')


def contract_fields(cfg):
    return ([(('architecture',k),k) for k in cfg['architecture']]
            + [(('loss',k),k) for k in cfg['loss']]
            + [(('training',k),k) for k in TRAINING_CONTRACT]
            + [(('training',k),k) for k in ('unfreeze_last_layers','cpm_lr_scale')
               if k in cfg['training']])


def verify_checkpoint_contract(args,cfg) -> None:
    mismatches=[]
    for path,key in contract_fields(cfg):
        expected=config_value(cfg,path)
        actual=getattr(args,key,'<missing>')
        if actual!=expected:mismatches.append(f'{key}: checkpoint={actual!r}, config={expected!r}')
    expected=cfg['data_and_split']['conditioning'][args.mode]
    for key,val in expected.items():
        if getattr(args,key,None)!=val:mismatches.append(f'{key}: 角色条件不匹配')
    if 'augmentation' in cfg and getattr(args, 'augmentation', None) != cfg['augmentation']:
        mismatches.append('augmentation: checkpoint 与实验增强配置不一致')
    if mismatches:raise ValueError('正式配置与权重不一致:\n'+'\n'.join(mismatches))


def verify_checkpoint_hash(path,expected_sha256):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual!=expected_sha256:
        raise ValueError(f'权重 SHA256 不匹配: {path}; expected={expected_sha256}, actual={actual}')


def build_model_from_args(args,device,condition_bundle):
    kwargs={}
    for name,param in inspect.signature(NeuroTwinTFM.__init__).parameters.items():
        if name in ('self','condition_config','condition_stats_payload'):continue
        key=MODEL_ALIASES.get(name,name)
        if hasattr(args,key):kwargs[name]=getattr(args,key)
        elif param.default is inspect.Parameter.empty:raise ValueError(f'缺少模型参数: {key}')
    if condition_bundle:
        kwargs['condition_config']=condition_bundle['config']
        kwargs['condition_stats_payload']=condition_bundle['stats']
    model=NeuroTwinTFM(**kwargs).to(device)
    if condition_bundle:
        expected=condition_bundle['meta'].get('layout_hash')
        if expected and model.conditioner.layout()['layout_hash']!=expected:
            raise ValueError('权重的条件布局与重建模型不一致')
    return model


def load_model_checkpoint(path,device='cpu'):
    payload=read_checkpoint(path)
    args=SimpleNamespace(**payload['config']['args'])
    bundle=payload['config']['condition']
    model=build_model_from_args(args,torch.device(device),bundle)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval(),args,bundle


def build_model(device='cpu',verify_hash=True,mode='finetune',checkpoint_path=None):
    cfg=load_model_config()
    weight=cfg['weights'][mode]
    path=Path(checkpoint_path) if checkpoint_path is not None else ROOT/weight['path']
    if verify_hash:verify_checkpoint_hash(path,weight['sha256'])
    model,args,_=load_model_checkpoint(path,device)
    verify_checkpoint_contract(args,cfg)
    return model,cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pretrain','finetune'],default='finetune')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--ckpt',default='',help='显式指定已移动的权重，仍按所选配置核对SHA256和结构')
    args=parser.parse_args()
    model,cfg=build_model(args.device,mode=args.mode,checkpoint_path=args.ckpt or None)
    label=cfg['_meta']['display_name']
    print(f'[{label}] {args.mode}: {args.ckpt or cfg["weights"][args.mode]["path"]}')
    print(f'[{label}] strict load + SHA256 通过；参数量={sum(p.numel() for p in model.parameters())}；H={model.cpm_horizon}')


if __name__=='__main__':main()
