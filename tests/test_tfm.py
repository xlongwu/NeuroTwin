"""Current Graph-ODE, field-condition and checkpoint contracts."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from experiments.evaluate_variant import (
    build_model_from_args,
    load_condition_bundle,
    load_train_args,
)
from models.conditioning import ConditioningConfig
from models.tfm import NeuroTwinTFM
from models.tfm_dynamics import GraphODEDynamicsHead
from train.optim import save_backbone_weights, load_backbone_weights, unwrap_state_dict
from utils.condition_schema import fit_condition_stats, schema_for


def make_model(group="mdd8"):
    torch.manual_seed(17)
    specs = schema_for(group)
    values = np.array(
        [
            [
                float(i % 2) if not s.is_continuous else float(10 + i + j)
                for j, s in enumerate(specs)
            ]
            for i in range(6)
        ]
    )
    mask = np.ones_like(values, dtype=np.int64)
    stats = fit_condition_stats(specs, values, mask, np.arange(6))
    z, codes = stats.encode(values[:2], mask[:2])
    cfg = ConditioningConfig(
        group=group, arm="demo" if group == "hc3" else "cat"
    ).to_dict()
    model = NeuroTwinTFM(
        features=4,
        context_max=12,
        patch_len=2,
        dim=16,
        num_layers=1,
        num_heads=4,
        ff_ratio=2,
        dropout=0.0,
        cpm_horizon=3,
        cpm_layers=1,
        one_step_hidden_dim=16,
        sc_prior_rank=4,
        profile_dim=8,
        condition_config=cfg,
        condition_stats_payload=stats.to_dict(),
    )
    cond = dict(
        cond_z=torch.tensor(z),
        cond_code=torch.tensor(codes),
        cond_mask=torch.tensor(mask[:2], dtype=torch.float32),
    )
    return model, cond, cfg, stats


def inputs():
    torch.manual_seed(29)
    return torch.randn(2, 4, 1, 10), torch.randn(2, 4, 1, 8)


@pytest.mark.parametrize("group", ["hc3", "mdd8"])
def test_current_heads_profile_shapes_and_condition_gradients(group):
    model, cond, _, _ = make_model(group)
    x, cal = inputs()
    pred, aux = model(x, None, condition=cond, calibration_bold=cal)
    assert isinstance(model.one_step_head, GraphODEDynamicsHead)
    assert pred.shape == (2, 4, 1, 1)
    for key in ("cpm_pred", "dynamics_pred", "dynamics_std"):
        assert aux[key].shape == (2, 4, 3)
        assert torch.isfinite(aux[key]).all()
    assert aux["personal_profile"].shape == (2, 8)
    normalized, _ = model.rev_norm(x, "norm")
    torch.testing.assert_close(
        aux["dynamics_mean_norm"],
        normalized[:, :, 0, -1:].expand(-1, -1, 3),
        rtol=0,
        atol=0,
    )
    # Open the zero-initialized readout and FiLM so a single backward exercises the field encoder.
    with torch.no_grad():
        model.one_step_head.decode_rate.weight.fill_(0.02)
        model.cpm_head.film.proj.weight.fill_(0.01)
        model.one_step_head.personal_film.proj.weight.fill_(0.01)
        model.one_step_head.kappa_personal.weight.fill_(0.01)
    pred, aux = model(x, None, condition=cond, calibration_bold=cal)
    (pred.square().mean() + aux["cpm_pred"].square().mean()).backward()
    assert any(
        p.grad is not None and bool(p.grad.abs().sum())
        for p in model.conditioner.parameters()
    )
    assert any(
        p.grad is not None and bool(p.grad.abs().sum())
        for p in model.profile_encoder.parameters()
    )


def test_persistent_state_rollout_and_observation_update():
    model, cond, _, _ = make_model()
    model.eval()
    x, cal = inputs()
    hist = x[:, :, 0, :].transpose(1, 2)
    with torch.no_grad():
        profile = model.calibrate_profile(cal)
        state = model.initialize_state(hist, condition=cond, profile=profile)
        states = []
        for _ in range(3):
            state = model.advance_state(state)
            states.append(model.readout_state(state))
        expected = torch.stack(states, dim=1)
        actual = model.rollout_next_states(
            hist, None, steps=3, condition=cond, profile=profile
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        one = model.predict_next_state(hist, None, condition=cond, profile=profile)
        torch.testing.assert_close(one["pred_next"].squeeze(-1), expected[:, 0])
        assert model.forecast_horizon(
            hist, None, condition=cond, profile=profile
        ).shape == (2, 3, 4)
        observation = torch.full((2, 4), 0.25)
        updated = model.update_state(state, observation)
        torch.testing.assert_close(updated["history"][:, -1], observation)
        torch.testing.assert_close(updated["profile"], profile)


def test_current_checkpoint_strict_roundtrip(tmp_path):
    model, cond, cfg, stats = make_model()
    model.eval()
    x, cal = inputs()
    args = dict(
        num_rois=4,
        context_max=12,
        tfm_patch_len=2,
        tfm_dim=16,
        tfm_layers=1,
        tfm_heads=4,
        tfm_ff_ratio=2,
        dropout=0.0,
        cpm_horizon=3,
        cpm_layers=1,
        tfm_one_step_hidden_dim=16,
        sc_prior_rank=4,
        profile_dim=8,
        norm=True,
        dynamics_mode="graph_ode",
        graph_source="history_fc",
        personalized=True,
        mode="finetune",
        condition_arm="cat",
        condition_group="mdd8",
    )
    path = tmp_path / "current.pt"
    save_backbone_weights(
        path,
        model,
        meta={
            "args": args,
            "condition": {
                "config": cfg,
                "stats": stats.to_dict(),
                "meta": {},
                "split_hash": "test",
            },
        },
    )
    rebuilt = build_model_from_args(
        load_train_args(path), "cpu", load_condition_bundle(path)
    ).eval()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    load_backbone_weights(rebuilt, path, "cpu")
    retired = dict(raw, config={"args": dict(args, dynamics_mode="discrete")})
    with pytest.raises(ValueError, match="graph_ode"):
        unwrap_state_dict(retired)
    with torch.no_grad():
        expected = model(x, None, condition=cond, calibration_bold=cal)
        actual = rebuilt(x, None, condition=cond, calibration_bold=cal)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for key in ("cpm_pred", "dynamics_pred", "dynamics_std"):
        torch.testing.assert_close(actual[1][key], expected[1][key], rtol=0, atol=0)


def test_missing_profile_and_retired_interfaces_are_rejected():
    model, cond, _, _ = make_model()
    x, _ = inputs()
    with pytest.raises(ValueError, match="profile"):
        model(x, None, condition=cond)
    with pytest.raises(ValueError, match="graph_ode"):
        NeuroTwinTFM(dynamics_mode="discrete")
    with pytest.raises(ValueError, match="history_fc"):
        NeuroTwinTFM(graph_source="legacy_mask")
    with pytest.raises(TypeError):
        NeuroTwinTFM(tfm_patho_mode="moe")
    with pytest.raises(TypeError):
        model(x, None, pathology_score=torch.ones(2))
    with pytest.raises(ValueError, match="Graph-ODE"):
        build_model_from_args(
            Namespace(num_rois=4, context_max=12, dynamics_mode="discrete"), "cpu"
        )
