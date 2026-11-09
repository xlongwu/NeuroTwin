# coding=utf-8
"""V2 唯一训练入口：HC CPM 预训练 → MDD 条件微调。"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from model import build_model_from_args, load_model_config
from models.conditioning import ConditioningConfig
from train.losses import ForecastFidelityLoss
from train.forecast_metrics import H32_STAGE_LABELS, h32_stage_metrics, horizon_metrics
from train.optim import (ModelEMA, build_optimizer, build_scheduler, load_initial_weights,
                         read_checkpoint, save_checkpoint, set_trainability,
                         set_training_mode, unfreeze_while_preserving_state)
from train.training_state import (TrainingState, load_training_state,
                                  read_training_state_metadata, restore_rng_state,
                                  save_training_state)
from utils.common import set_seed, str2bool
from utils.dataloader import NeuroTwinDataLoader
from utils.condition_schema import DEMO_GROUP, ConditionStats, load_inherited_stats, schema_for

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


def condition_loader_kwargs(args, bundle=None) -> dict:
    group = args.condition_group
    base = (ConditionStats.from_dict(schema_for(group), bundle['stats'])
            if bundle is not None else None)
    return dict(condition_group=group, condition_file=args.condition_file or None,
                split_file=args.split_file or None, condition_base_stats=base,
                condition_stats_mode='inherit' if base is not None else 'fit')


def forecast_batch(batch, device):
    return (batch['x'].to(device, non_blocking=True),
            batch['future'].to(device, non_blocking=True),
            batch['future_mask'].to(device, non_blocking=True),
            batch['x_last'].to(device, non_blocking=True),
            condition_batch(batch, device))


def build_data_loader(args, base_stats=None, inherit_all=False):
    return NeuroTwinDataLoader(
        data_root=args.data_root, mode=args.mode, batch_size=args.batch_size,
        condition_group=args.condition_group, condition_file=args.condition_file or None,
        split_file=args.split_file or None, condition_base_stats=base_stats,
        condition_stats_mode='inherit' if inherit_all else 'fit',
        clinical_file=args.clinical_file, num_workers=args.num_workers,
        pin_memory=args.pin_memory, seed=args.seed, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, stratify_bins=args.stratify_bins,
        cache_in_memory=args.cache_in_memory, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor, eval_batch_size=args.eval_batch_size,
        task_mode='next_timepoint', bold_source='raw', graph_source='history_fc',
        num_rois=args.num_rois, calibration_length=args.calibration_length,
        random_context=args.random_context, random_cutoff=args.random_cutoff,
        train_samples_per_subject=args.train_samples_per_subject,
        subject_cache_size=args.subject_cache_size, sampling_seed=args.sampling_seed,
        context_min=args.context_min, context_max=args.context_max, forecast_offsets=[1],
        eval_context_length=args.eval_context_length,
        eval_anchors_per_subject=args.eval_anchors_per_subject,
        train_rollout_steps=args.cpm_horizon, train_anchor_max_offset=args.cpm_horizon,
        eval_all_future_steps=args.cpm_horizon, eval_anchor_min_future_steps=args.cpm_horizon,
        eval_rollout_tasks_per_subject=0, eval_rollout_steps=0,
        augmentation_config=getattr(args, 'augmentation', None))


@torch.no_grad()
def evaluate_tfm(model, data_loader, criterion, device, args) -> dict:
    model.eval()
    values = defaultdict(lambda: defaultdict(list))
    losses = defaultdict(list)
    for batch in data_loader:
        x, target, mask, anchor, cond = forecast_batch(batch, device)
        pred, _ = model(x, condition=cond)
        loss, stats = criterion(pred, target, mask, anchor)
        for name, val in {**stats, 'metric_total': loss.detach()}.items():
            losses[name].append(float(val))
        stages = h32_stage_metrics(pred, target, mask, anchor, include_local=True)
        steps = horizon_metrics(pred, target, mask, anchor)
        # 小批量搬到 CPU，避免逐指标触发设备同步。
        stages = {k:v.cpu().tolist() for k,v in stages.items()}
        steps = {k:v.cpu().tolist() for k,v in steps.items()}
        for i, sid in enumerate(batch['subj_id']):
            for metric, rows in stages.items():
                for j, label in enumerate(H32_STAGE_LABELS):
                    if label == 'H1':
                        continue
                    val = rows[i][j]
                    if math.isfinite(val):
                        values[f'metric_{metric}_{label.lower()}'][str(sid)].append(val)
            for metric, rows in steps.items():
                for j, val in enumerate(rows[i]):
                    if math.isfinite(val):
                        values[f'metric_{metric}_h{j+1}'][str(sid)].append(val)
    metrics = {key:sum(rows)/len(rows) for key,rows in losses.items()}
    for name, subjects in values.items():
        means = [sum(rows)/len(rows) for rows in subjects.values()]
        metrics[name] = sum(means)/len(means)
        metrics[f'{name}_n_subjects'] = len(means)
    metrics['metric_selection_h32'] = (
        metrics['metric_mae_h1_32'] + args.selection_corr_weight *
        (1 - metrics['metric_temporal_pcc_h1_32']))
    return metrics


def _resume_argument_mismatches(saved: dict, current: dict) -> dict:
    ignored = {'checkpoint_dir','name','init_weight','pretrained_weight',
               'resume_training_state','save_training_state','eval_after_train','device'}
    if current['train_epochs'] >= saved['train_epochs']:
        ignored.add('train_epochs')
    return {k:(saved.get(k),current.get(k)) for k in sorted(current)
            if k not in ignored and saved.get(k)!=current.get(k)}


def parse_args(argv=None):
    cfg = load_model_config()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['pretrain','finetune'], default='pretrain')
    for section in ('architecture','loss','training','data_and_split'):
        for key, val in cfg[section].items():
            if key in ('conditioning',):
                continue
            if key in ('forecast_offsets',):
                continue
            kind = str2bool if isinstance(val,bool) else int if val is None else type(val)
            parser.add_argument('--'+key, type=kind, default=val)
    condition = cfg['data_and_split']['conditioning']
    for key,val in condition.items():
        if key not in ('pretrain','finetune'):
            kind = str2bool if isinstance(val,bool) else type(val)
            parser.add_argument('--'+key, type=kind, default=val)
    parser.add_argument('--condition_group', default='')
    parser.add_argument('--condition_arm', choices=['demo','cat'], default='')
    parser.add_argument('--condition_file', default='')
    parser.add_argument('--split_file', default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--device', default='')
    parser.add_argument('--init_weight', default='')
    parser.add_argument('--pretrained_weight', default='')
    parser.add_argument('--resume_training_state', default='')
    parser.add_argument('--augmentation', type=json.loads, default=cfg.get('augmentation'))
    parser.add_argument('--save_training_state', action='store_true')
    parser.add_argument('--no_eval_after_train', dest='eval_after_train', action='store_false')
    parser.set_defaults(eval_after_train=cfg['training'].get('eval_after_train', True))
    args = parser.parse_args(argv)
    args.name = args.name or f'neurotwin_{args.mode}'
    args.condition_group = args.condition_group or condition[args.mode]['condition_group']
    args.condition_arm = args.condition_arm or condition[args.mode]['condition_arm']
    args.split_file = args.split_file or str(Path(args.data_root)/f'subject_split_{args.mode}.json')
    args.init_weight = args.init_weight or cfg['weights']['initializer']['path']
    args.pretrained_weight = args.pretrained_weight or cfg['weights']['pretrain']['path']
    args.model_arch, args.task_mode, args.bold_source, args.graph_source = 'tfm','next_timepoint','raw','history_fc'
    args.forecast_offsets = '1'
    return args


def main(args):
    check_seed_separation(args)
    if args.cpm_horizon != 32 or args.context_min != 64 or args.context_max != 64 or args.random_context:
        raise ValueError('V2使用固定K64 / H32')
    torch.set_num_threads(args.num_threads)
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    amp = device.type == 'cuda' and args.amp
    if device.type == 'cuda' and args.tf32:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    resume_meta = read_training_state_metadata(args.resume_training_state) if args.resume_training_state else None
    progress = resume_meta['progress'] if resume_meta else None
    source = args.init_weight if args.mode == 'pretrain' else args.pretrained_weight
    if progress:
        mismatch = _resume_argument_mismatches(progress['study']['args'], vars(args))
        if mismatch:
            raise ValueError(f'续训配置不匹配: {mismatch}')
        base = ConditionStats.from_dict(schema_for(args.condition_group), progress['study']['condition_stats'])
    else:
        payload = read_checkpoint(source)['config']['condition']['stats']
        base = load_inherited_stats(payload, schema_for(args.condition_group), DEMO_GROUP)
    loader = build_data_loader(args, base, inherit_all=bool(progress) or args.mode=='pretrain')
    train_data, val_data = loader.get_train(), loader.get_val()
    stats = loader.get_condition_stats().to_dict()
    bundle = dict(config=ConditioningConfig(
        group=args.condition_group, arm=args.condition_arm, d_demo=args.cond_d_demo,
        d_new=args.cond_d_new, cond_dim=args.cond_dim, h_delta=args.cond_h_delta,
        delta_style=args.cond_delta_style, dropout_p=args.cond_dropout_p,
        dropout_seed=args.cond_dropout_seed).to_dict(),
        stats=stats, meta=loader.get_condition_meta(), split_hash=split_hash(loader,args))
    if progress and (progress['study']['condition_stats'] != stats or
                     progress['study']['split_hash'] != bundle['split_hash']):
        raise ValueError('续训条件统计量或被试划分不一致')
    model = build_model_from_args(args, device, bundle)
    migration = load_initial_weights(model, source, args) if not progress else progress['study']['initialization']
    criterion = ForecastFidelityLoss(**{k:getattr(args,k) for k in load_model_config()['loss']}).to(device)
    unfrozen = args.freeze_backbone_epochs == 0
    last_layers = args.unfreeze_last_layers
    set_trainability(model,args.mode,unfrozen,last_layers)
    optimizer = build_optimizer(model,criterion,args)
    scheduler = build_scheduler(optimizer,args)
    if progress and progress['backbone_unfrozen'] and not unfrozen:
        set_trainability(model,args.mode,True,last_layers)
        unfreeze_while_preserving_state(model,optimizer,args)
        scheduler.sync_param_groups()
        unfrozen = True
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    ema = ModelEMA(model,args.ema_decay)
    if args.compile and device.type=='cuda':
        model = torch.compile(model,dynamic=True)
    selector = CheckpointSelector('metric_selection_h32',patience=args.patience)
    if progress:
        run_dir = Path(args.resume_training_state).resolve().parent
        restored = load_training_state(args.resume_training_state,model=model,criterion=criterion,
            optimizer=optimizer,scheduler=scheduler,scaler=scaler,ema=ema,
            loader_generator=loader.loader_generator,restore_rng=False)
        start, global_step = restored.epoch, restored.global_step
        selector.best,selector.counter = restored.best_metric,restored.patience_counter
    else:
        run_dir = Path(args.checkpoint_dir)/args.name
        if run_dir.exists():
            index=2
            while Path(f'{run_dir}{index}').exists():index+=1
            run_dir=Path(f'{run_dir}{index}')
        run_dir.mkdir(parents=True)
        start,global_step=0,0
    best_name = 'base_best.pt' if args.mode=='pretrain' else 'finetuned_best.pt'
    if progress and not (run_dir/best_name).is_file():
        raise FileNotFoundError('续训目录缺少原最佳权重')
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir/'tensorboard'),purge_step=start if progress else None)
    except ImportError:
        writer = None
    if resume_meta and not selector.should_stop:
        restore_rng_state(resume_meta['rng'])
    label = load_model_config()['_meta']['display_name']
    print(f'[{label}] {args.mode} | run={run_dir} | K64 H32 | batch={args.batch_size}',flush=True)
    epoch_limit = start if progress and progress['early_stopped'] else args.train_epochs
    for epoch in range(start,epoch_limit):
        if not unfrozen and epoch>=args.freeze_backbone_epochs:
            set_trainability(model,args.mode,True,last_layers)
            added,names=unfreeze_while_preserving_state(model,optimizer,args)
            scheduler.sync_param_groups()
            unfrozen=True;selector.counter=0
            print(f'[unfreeze] epoch={epoch+1} added={added} groups={names}',flush=True)
        set_training_mode(model,unfrozen,last_layers)
        train_data.batch_sampler.set_epoch(epoch)
        train_values=defaultdict(list)
        for batch in tqdm(train_data,desc=f'Epoch {epoch+1}/{args.train_epochs}',leave=False):
            optimizer.zero_grad(set_to_none=True)
            x,target,mask,anchor,cond=forecast_batch(batch,device)
            with torch.amp.autocast('cuda',enabled=amp):
                pred,_=model(x,condition=cond,condition_epoch=epoch)
                loss,terms=criterion(pred,target,mask,anchor)
            scaler.scale(loss).backward()
            if args.grad_clip>0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.grad_clip)
            scaler.step(optimizer);scaler.update()
            ema.update(model)
            global_step+=1
            for k,v in terms.items():train_values[k].append(float(v))
        eval_model=ema.ema
        metrics=evaluate_tfm(eval_model,val_data,criterion,device,args)
        raw_metrics = evaluate_tfm(model,val_data,criterion,device,args)
        scheduler.step()
        is_best,score=selector.step(metrics)
        if not unfrozen:selector.counter=0
        train_metrics={k:sum(v)/len(v) for k,v in train_values.items()}
        meta=dict(args=vars(args),epoch=epoch,epoch_1based=epoch+1,
                  val_metrics=metrics,val_metric=score,condition=bundle,
                  initialization=migration,backbone_unfrozen=unfrozen,
                  loss_weights=criterion.get_loss_weights())
        meta['val_metrics_raw'] = raw_metrics
        if is_best:save_checkpoint(run_dir/best_name,eval_model,meta)
        stopped=selector.should_stop
        if stopped or epoch==args.train_epochs-1:
            save_checkpoint(run_dir/('base_last.pt' if args.mode=='pretrain' else 'finetuned_last.pt'),eval_model,meta)
        with (run_dir/'验证训练记录.jsonl').open('a',encoding='utf-8') as f:
            record = dict(epoch=epoch+1,train=train_metrics,val=metrics,
                          loss_weights=criterion.get_loss_weights())
            record.update(val_raw=raw_metrics, val_ema=metrics)
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if writer:
            for k,v in metrics.items():writer.add_scalar('Val/'+k,v,epoch)
            for k,v in raw_metrics.items():writer.add_scalar('ValRaw/'+k,v,epoch)
            for k,v in train_metrics.items():writer.add_scalar('Train/'+k,v,epoch)
            for g in optimizer.param_groups:writer.add_scalar('LR/'+g['name'],g['lr'],epoch)
            writer.flush()
        print(f'[Val] epoch={epoch+1} MAE={metrics["metric_mae_h1_32"]:.6f} '
              f'TemporalPCC={metrics["metric_temporal_pcc_h1_32"]:.6f} score={score:.6f} best={selector.best:.6f}',flush=True)
        print(f'[Val raw] epoch={epoch+1} MAE={raw_metrics["metric_mae_h1_32"]:.6f} '
              f'TemporalPCC={raw_metrics["metric_temporal_pcc_h1_32"]:.6f} '
              f'score={raw_metrics["metric_selection_h32"]:.6f}', flush=True)
        print('[Val stages] ' + ' | '.join(
            f'{stage}: MAE={metrics["metric_mae_"+stage]:.6f} '
            f'RMSE={metrics["metric_rmse_"+stage]:.6f} '
            f'TemporalPCC={metrics["metric_temporal_pcc_"+stage]:.6f}'
            for stage in ('h1_8', 'h9_16', 'h17_32', 'h1_32')), flush=True)
        if args.save_training_state or progress:
            state=TrainingState(epoch=epoch+1,global_step=global_step,best_metric=selector.best,
                best_metric_name=selector.metric_key,patience_counter=selector.counter,
                backbone_unfrozen=unfrozen,early_stopped=stopped,
                study=dict(args=vars(args),mode=args.mode,condition_stats=stats,
                           split_hash=bundle['split_hash'],initialization=migration))
            save_training_state(run_dir/'training_state.pt',model=model,criterion=criterion,
                optimizer=optimizer,scheduler=scheduler,scaler=scaler,ema=ema,
                loader_generator=loader.loader_generator,progress=state)
        if stopped:break
    if writer:writer.close()
    print(f'[done] best={run_dir/best_name}',flush=True)
    if args.mode=='finetune' and args.eval_after_train and start<epoch_limit:
        from analysis.evaluate import evaluate
        evaluate(run_dir/best_name,run_dir/'评估',device=device)
    return run_dir


if __name__=='__main__':
    main(parse_args())
