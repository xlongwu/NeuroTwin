#!/usr/bin/env python
"""一次性将已验证 B 和原 HC 初始化权重投影到精简结构，不修改源文件。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from main import parse_args
from model import MODEL_CONFIG, build_model_from_args, load_model_config
from train.optim import save_checkpoint

ACTIVE_PREFIXES=('rev_norm.','sc_prior.','conditioner.','patch_embed.','blocks.','final_norm.','cpm_head.')
REMOVED_PREFIXES=('one_step_head.','profile_encoder.','initial_condition_film.',
                  'spectral_encoder.','spectral_memory_gate.','spectral_memory_proj.')


def migrate(source:Path,destination:Path,mode:str,initializer=False):
    if destination.exists():raise FileExistsError(f'拒绝覆盖已有权重: {destination}')
    raw=torch.load(source,map_location='cpu',weights_only=False)
    if raw.get('arch_version')!=6:
        raise ValueError('此显式迁移器只接受本次原始 v6 权重')
    original=raw['config']
    if not initializer and original['args'].get('forecast_mode')!='cpm':
        raise ValueError('只允许迁移已训练方案 B，不能将 A/C 权重当作 B')
    args=parse_args(['--mode',mode])
    bundle=original['condition']
    model=build_model_from_args(args,torch.device('cpu'),bundle)
    active,excluded={},[]
    for key,value in raw['state_dict'].items():
        if key.startswith(ACTIVE_PREFIXES):active[key]=value
        elif key.startswith(REMOVED_PREFIXES):excluded.append(key)
        else:raise ValueError(f'未登记的源权重键: {key}')
    expanded=[]
    key='cpm_head.horizon_embed'
    expected=model.state_dict()
    if initializer and active[key].shape!=expected[key].shape:
        old=active[key]
        if old.shape[1]!=expected[key].shape[1] or not 0<old.shape[0]<expected[key].shape[0]:
            raise ValueError('H8→H32 horizon embedding 无法显式扩展')
        value=old[-1:].expand_as(expected[key]).clone()
        value[:old.shape[0]].copy_(old)
        active[key]=value
        expanded=[dict(key=key,source_horizon=old.shape[0],target_horizon=value.shape[0],policy='repeat_last')]
    model.load_state_dict(active,strict=True)
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    report=dict(source=str(source.relative_to(ROOT)),source_sha256=source_sha,
                copied_keys=sorted(active),removed_keys=sorted(excluded),expanded=expanded)
    meta={k:v for k,v in original.items() if k not in ('args','pretrained_migration','params')}
    meta.update(args=vars(args),condition=bundle,migration=report,
                role='hc_h8_initializer' if initializer else mode,
                original_training_args=original['args'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(destination,model,meta)
    (destination.parent/'权重迁移记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=['initializer','pretrain','finetune'],action='append')
    ns=parser.parse_args()
    cfg=load_model_config()
    for role in ns.role or ('initializer','pretrain','finetune'):
        item=cfg['weights'][role]
        if hashlib.sha256((ROOT/item['source']).read_bytes()).hexdigest()!=item['source_sha256']:
            raise ValueError(f'{role} 原权重身份与已验证记录不一致')
        digest=migrate(ROOT/item['source'],ROOT/item['path'],
                       'pretrain' if role=='initializer' else role,role=='initializer')
        item['sha256']=digest
        MODEL_CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
        print(role,item['path'],digest,flush=True)


if __name__=='__main__':main()
