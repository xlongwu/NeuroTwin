"""IDG/SDRL/PDD 的辅助训练目标；原有三项主损失保持原接口。"""
from __future__ import annotations

import torch
from torch.nn import functional as F


DYNAMICS_MODEL_OPTIONS = (
    'personal_dynamics', 'personal_generator_hidden_dim', 'personal_rank',
    'personal_residual_scale', 'regional_diffusion', 'regional_decay',
    'personalized_graph', 'personal_graph_rank', 'stable_representation',
    'pathology_dynamics', 'pathology_dim', 'pathology_rank', 'pathology_residual_scale',
    'dynamics_readout_init_std',
)


def temporal_difference_l1(pred: torch.Tensor, target: torch.Tensor,
                           context_last: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """包括第一步与末次真实观测的差分；相邻两步均有效时才计入。"""
    p = torch.cat((context_last.unsqueeze(-1), pred), dim=-1).diff(dim=-1)
    t = torch.cat((context_last.unsqueeze(-1), target), dim=-1).diff(dim=-1)
    valid = mask.to(pred.dtype)
    valid = valid * torch.cat((torch.ones_like(valid[:, :1]), valid[:, :-1]), dim=-1)
    return ((p - t).abs() * valid.unsqueeze(1)).sum() / (
        valid.sum().clamp_min(1) * pred.shape[1])


def subject_representation_loss(first: torch.Tensor, second: torch.Tensor,
                                subject_ids: list[str], temperature: float = 0.1) -> tuple:
    """跨窗口一致性与对称主体对比；同一被试的重复样本均为正样本。"""
    if temperature <= 0:
        raise ValueError('contrastive_temperature 必须为正')
    a, b = F.normalize(first.float(), dim=-1), F.normalize(second.float(), dim=-1)
    consistency = (1 - (a * b).sum(-1)).mean()
    # 被试等权，避免重复采样的身份获得更高权重。
    unique = list(dict.fromkeys(subject_ids))
    a = torch.stack([a[[i for i, sid in enumerate(subject_ids) if sid == key]].mean(0)
                     for key in unique])
    b = torch.stack([b[[i for i, sid in enumerate(subject_ids) if sid == key]].mean(0)
                     for key in unique])
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(unique), device=first.device)
    contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return consistency, contrastive


def stable_loss_factor(epoch: int | None, warmup: int = 0, ramp: int = 0) -> float:
    """epoch 从零计数；预测任务先学习，再逐步加入身份约束。"""
    if epoch is None:
        return 1.0
    if epoch < warmup:
        return 0.0
    return min(1.0, (epoch - warmup + 1) / max(1, ramp))


def auxiliary_dynamics_loss(model, aux: dict, batch: dict, target: torch.Tensor,
                            mask: torch.Tensor, args, disease_label: float | None = None,
                            epoch: int | None = None) -> tuple:
    inner = getattr(model, '_orig_mod', model)
    pred = aux['dynamics_pred']
    total = pred.new_zeros(())
    stats = {}

    def add(name: str, value: torch.Tensor, weight: float) -> None:
        nonlocal total
        total = total + weight * value
        stats[name] = value.detach()

    if args.lambda_delta > 0:
        add('loss_delta', temporal_difference_l1(
            pred, target, batch['x_last'].to(pred.device), mask), args.lambda_delta)
    personal = aux.get('personal_parameters')
    if personal is not None and args.lambda_personal_reg > 0:
        add('loss_personal_reg', inner.one_step_head.personal_dynamics.regularization(
            personal, inner.one_step_head.kappa_raw), args.lambda_personal_reg)
    stable_factor = stable_loss_factor(epoch, getattr(args, 'stable_warmup_epochs', 0),
                                      getattr(args, 'stable_ramp_epochs', 0))
    if inner.stable_representation:
        stats['stable_loss_factor'] = pred.new_tensor(stable_factor)
    if inner.stable_representation and args.lambda_subject > 0 and stable_factor > 0:
        valid = batch['stable_pair_valid'].to(pred.device)
        if valid.any():
            second = inner.calibrate_profile(batch['stable_bold_b'].to(pred.device)[valid])
            first = aux['personal_profile'][valid]
            ids = [str(sid) for sid, hit in zip(batch['subj_id'], valid.tolist()) if hit]
            consistency, contrast = subject_representation_loss(
                first, second, ids, args.contrastive_temperature)
            add('loss_profile_consistency', consistency, args.lambda_subject * stable_factor)
            add('loss_subject', contrast, args.lambda_subject * stable_factor)
        stats['stable_pair_count'] = valid.sum().to(pred.dtype).detach()
    pathology = aux.get('pathology_parameters')
    if pathology is not None and disease_label is not None:
        labels = torch.full_like(pathology['disease_logit'], disease_label)
        add('loss_disease', F.binary_cross_entropy_with_logits(
            pathology['disease_logit'], labels), args.lambda_disease)
        # 健康参考额外约束逐样本病理算子接近零。
        energy = pathology['operator_energy'].mean()
        add('loss_path_reg', energy, args.lambda_path_reg)
        if disease_label == 0:
            add('loss_hc_path_reference', energy, args.lambda_hc_reference)
        if args.lambda_clinical > 0 and disease_label == 1:
            observed = batch['clinical_target_mask'].to(pred.device).bool()
            if observed.any():
                add('loss_clinical', F.huber_loss(
                    pathology['clinical_prediction'][observed],
                    batch['clinical_target'].to(pred.device)[observed]), args.lambda_clinical)
    return total, stats


def build_hc_reference(args, condition_stats):
    """仅使用原 HC train 划分；条件统计量继承冻结健康模型的统计量。"""
    from utils.common import parse_int_list
    from utils.dataloader import NeuroTwinDataLoader

    return NeuroTwinDataLoader(
        data_root=args.data_root, mode='pretrain', batch_size=args.batch_size,
        condition_group='hc3', condition_stats_mode='inherit',
        condition_base_stats=condition_stats, split_file=args.hc_reference_split or None,
        seed=args.seed, sampling_seed=args.sampling_seed, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, num_workers=args.num_workers,
        pin_memory=args.pin_memory, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor, cache_in_memory=args.cache_in_memory,
        bold_source=args.bold_source, context_min=args.context_min, context_max=args.context_max,
        context_lengths=parse_int_list(args.context_lengths, 'context_lengths'),
        forecast_offsets=[1], graph_source=args.graph_source, num_rois=args.num_rois,
        calibration_length=args.calibration_length,
        train_samples_per_subject=args.train_samples_per_subject,
        train_rollout_steps=args.cpm_horizon, train_anchor_max_offset=args.cpm_horizon,
        eval_context_length=args.eval_context_length or args.context_max,
        eval_rollout_tasks_per_subject=0, eval_rollout_steps=0,
        eval_all_future_steps=args.cpm_horizon,
        eval_anchor_min_future_steps=args.cpm_horizon,
    )
