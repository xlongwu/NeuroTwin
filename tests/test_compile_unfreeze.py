"""冻结期编译后解冻，主干必须实际获得梯度并进入 optimizer 更新。"""
from copy import deepcopy
from types import SimpleNamespace

import torch

from models.tfm import NeuroTwinTFM
from train.optim import (build_optimizer, set_finetune_trainability,
                         unfreeze_while_preserving_state)


def test_compiled_unfreeze_matches_eager_gradients():
    torch.manual_seed(2024)
    raw = NeuroTwinTFM(
        features=6, context_max=8, patch_len=2, dim=8,
        num_layers=1, num_heads=2, ff_ratio=2, dropout=0.0,
        one_step_hidden_dim=8, cpm_horizon=2, cpm_layers=1,
        pretrain_mode=False, tfm_patho_mode='film',
        dynamics_mode='graph_ode', graph_source='history_fc',
        personalized=True, profile_dim=4, dynamics_history_memory=True,
        ode_steps=1, sc_sinkhorn_iters=2)
    # 初始读出为零时会使上游梯度为零；模拟已预训练的非零读出。
    with torch.no_grad():
        raw.one_step_head.decode_rate.weight.normal_(std=0.1)
    eager = deepcopy(raw)
    args = SimpleNamespace(mode='finetune', lr_peak=1e-4,
                           backbone_lr_scale=0.2, adapter_lr_scale=1.0,
                           weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    set_finetune_trainability(raw, False)
    optimizer = build_optimizer(raw, criterion, args)
    compiled = torch.compile(raw, backend='aot_eager', dynamic=True)
    x, sc = torch.randn(2, 6, 1, 8), torch.eye(6)
    pathology = torch.randn(2, 1)

    def loss(model):
        out, aux = model(x, sc, pathology, calibration_bold=x)
        return (out.square().mean() + aux['cpm_pred'].square().mean()
                + aux['dynamics_pred'].square().mean()
                + aux['dynamics_logvar_norm'].square().mean())

    try:
        loss(compiled).backward()  # 缓存冻结期的前向/反向图。
        optimizer.step()
        condition_param = next(raw.cpm_head.film.parameters())
        condition_state = optimizer.state[condition_param]['exp_avg'].clone()
        eager.load_state_dict(raw.state_dict())
        set_finetune_trainability(compiled, True)
        unfreeze_while_preserving_state(compiled, optimizer, args)
        assert torch.equal(optimizer.state[condition_param]['exp_avg'], condition_state)
        in_optimizer = {id(p) for g in optimizer.param_groups for p in g['params']}
        assert all(id(p) in in_optimizer for p in raw.parameters() if p.requires_grad)

        optimizer.zero_grad(set_to_none=True)
        set_finetune_trainability(eager, True)
        loss(eager).backward()
        loss(compiled).backward()
        grads = dict(eager.named_parameters())
        for name, param in raw.named_parameters():
            reference = grads[name].grad
            if reference is None:
                assert param.grad is None, name
            else:
                assert param.grad is not None, name
                assert torch.isfinite(param.grad).all(), name
                torch.testing.assert_close(param.grad, reference)
        tracked = [raw.patch_embed.patch_proj.weight,
                   raw.one_step_head.w1.weight,
                   raw.profile_encoder.fc1.weight,
                   raw.cpm_head.out_proj.weight]
        before = [p.detach().clone() for p in tracked]
        assert all(p.grad.count_nonzero() > 0 for p in tracked)
        optimizer.step()
        assert all(not torch.equal(p, old) for p, old in zip(tracked, before))
    finally:
        torch.compiler.reset()
