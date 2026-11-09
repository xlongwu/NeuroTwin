"""从V2 checkpoint 严格重建，执行相同划分和统计量的 H32 评价。"""
import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import torch

from model import (ROOT,load_model_config,load_model_checkpoint)
from train.optim import read_checkpoint
from utils.common import set_seed


def load_train_args(path):
    return Namespace(**read_checkpoint(path)['config']['args'])


def load_condition_bundle(path):
    return read_checkpoint(path)['config']['condition']


def condition_stats_from_bundle(bundle,group):
    from utils.condition_schema import ConditionStats,schema_for
    return ConditionStats.from_dict(schema_for(group),bundle['stats'])


def evaluate(ckpt_path,out_dir,eval_splits=('test',),seed=2024,device=None,
             save_arrays=False,compute_feature_importance=False,importance_max_tasks=256):
    from analysis import next_point_eval as npe
    set_seed(seed)
    device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model,args,bundle=load_model_checkpoint(ckpt_path,device)
    torch.set_num_threads(args.num_threads)
    args.data_root=str(ROOT/args.data_root)
    result={}
    for split in eval_splits:
        loader,_,owner=npe.build_next_point_loader(args,split,condition_bundle=bundle)
        from main import split_hash
        actual_split_hash=split_hash(owner,args)
        if bundle.get('split_hash') and bundle['split_hash']!=actual_split_hash:
            raise ValueError('评价被试划分与checkpoint不一致')
        ar1=npe.fit_ar1_from_train(owner.base_dataset,owner.get_train_subjects())
        report=npe.evaluate_split(model,loader,args,device,out_dir,split,ar1=ar1,
            save_arrays=save_arrays,compute_importance=compute_feature_importance,
            importance_max_tasks=importance_max_tasks,
            checkpoint_identity={'path':str(ckpt_path),
                                 'sha256':hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()},
            split_identity=actual_split_hash)
        result[split]=npe.summarize(report)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt',default='')
    parser.add_argument('--mode',choices=['pretrain','finetune'],default='finetune')
    parser.add_argument('--out_dir',required=True)
    parser.add_argument('--splits',default='test')
    parser.add_argument('--device',default='')
    parser.add_argument('--seed',type=int,default=2024)
    parser.add_argument('--save_arrays',action='store_true')
    parser.add_argument('--feature_importance',action='store_true')
    parser.add_argument('--importance_max_tasks',type=int,default=256)
    ns=parser.parse_args()
    checkpoint=ns.ckpt or str(ROOT/load_model_config()['weights'][ns.mode]['path'])
    result=evaluate(checkpoint,ns.out_dir,tuple(ns.splits.split(',')),ns.seed,ns.device,
                    ns.save_arrays,ns.feature_importance,ns.importance_max_tasks)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
