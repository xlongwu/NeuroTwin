"""Model capability boundary; no checkpoint migration or untrained fallback model."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from analysis.study_data import sha256


@dataclass
class Prediction:
    mean: np.ndarray  # [H, F], in prefix-calibrated data scale
    # Standard deviation in the same data scale, not variance or log variance.
    std: object = None


class NativeAdapter:
    def __init__(self, checkpoint, device="cpu"):
        from analysis.checkpoint import (
            build_model_from_args,
            load_condition_bundle,
            load_train_args,
        )

        self.args = load_train_args(checkpoint)
        self.bundle = load_condition_bundle(checkpoint)
        self.device = torch.device(device)
        self.model = build_model_from_args(self.args, self.device, self.bundle)
        raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(raw["state_dict"], strict=True)
        self.model.eval()
        self._profiles = {}
        self.calibration = int(getattr(self.args, "calibration_length", 0))
        self.context = int(self.args.context_max)
        self.variance_trained = (
            float(getattr(self.args, "lambda_variance", 0)) > 0
            and raw["config"].get("epoch", -1) >= 0
        )
        self.metadata = dict(
            checkpoint=str(checkpoint),
            sha256=sha256(checkpoint),
            architecture="NeuroTwinTFM",
            dynamics_mode=self.model.dynamics_mode,
            graph_source=self.args.graph_source,
            condition_group=self.args.condition_group,
            variance_training_configured=self.variance_trained,
            output_scale="prefix-calibrated BOLD",
            inverse_transform="native model RevIN inverse; optional raw inverse uses subject prefix stats",
            epoch=raw["config"].get("epoch"),
            context=self.context,
            capabilities=[
                "one_step",
                "direct_cpm",
                "autoregressive",
                "ode_integration",
                "std",
            ],
        )

    def tensors(self, subject, anchor, context, history=None, conditions=None):
        ds = subject.dataset
        sample = ds.make_sample(ds._subject_pos[subject.native_id], context, anchor)
        x = sample["x"].unsqueeze(0).to(self.device)
        if history is not None:
            x = torch.as_tensor(
                history.T[None, :, None, :], dtype=torch.float32, device=self.device
            )
        kwargs = {}
        if "cond_z" in sample:
            kwargs["condition"] = {
                "cond_" + k: sample["cond_" + k].unsqueeze(0).to(self.device)
                for k in ("z", "code", "mask")
            }
        if "calibration_bold" in sample:
            kwargs["calibration_bold"] = (
                sample["calibration_bold"].unsqueeze(0).to(self.device)
            )
        if conditions is not None:
            kwargs["condition"] = conditions
        return x, sample["sc"].unsqueeze(0).to(self.device), kwargs

    def forward_tensor(self, x, sc, kwargs, horizon, mode):
        if mode == "autoregressive":
            history = x.squeeze(2).transpose(1, 2)
            preds = []
            for _ in range(horizon):
                out = self.model.predict_next_state(history, sc, **kwargs)[
                    "pred_next"
                ].squeeze(-1)
                preds.append(out)
                history = torch.cat([history[:, 1:], out[:, None]], dim=1)
            return torch.stack(preds, dim=1), None
        extra = {"dynamics_horizon": horizon} if mode == "ode_integration" else {}
        one, aux = self.model(x, sc, **kwargs, **extra)
        if mode == "one_step":
            if horizon != 1:
                raise ValueError("One-step mode only supports H=1")
            return one[:, :, 0, 0][:, None], None
        if mode == "ode_integration":
            return aux["dynamics_pred"].transpose(1, 2), aux["dynamics_std"].transpose(
                1, 2
            )
        if mode != "direct_cpm" or horizon > self.model.cpm_horizon:
            raise NotImplementedError("Direct CPM horizon unsupported")
        return aux["cpm_pred"][:, :, :horizon].transpose(1, 2), None

    def predict(
        self,
        subject,
        anchor,
        horizon,
        mode,
        context=None,
        history=None,
        conditions=None,
    ):
        x, sc, kwargs = self.tensors(
            subject, anchor, context or self.context, history, conditions
        )
        with torch.no_grad():
            if self.model.personalized:
                if subject.subject_id not in self._profiles:
                    self._profiles[subject.subject_id] = self.model.calibrate_profile(
                        kwargs["calibration_bold"]
                    )
                kwargs["profile"] = self._profiles[subject.subject_id]
                kwargs.pop("calibration_bold")
            mean, std = self.forward_tensor(x, sc, kwargs, horizon, mode)
        mean = mean[0].detach().cpu().numpy()
        if (
            mean.shape != (horizon, len(subject.roi_labels))
            or not np.isfinite(mean).all()
        ):
            raise ValueError("Invalid model output shape or nonfinite prediction")
        return Prediction(mean, None if std is None else std[0].cpu().numpy())

    def inverse_raw(self, subject, prediction):
        """Explicit optional inverse of dataset prefix calibration (not physiology validation)."""
        mu, scale = subject.dataset.get_calibration_stats(subject.native_id)
        return Prediction(
            prediction.mean * scale.T + mu.T,
            None if prediction.std is None else prediction.std * scale.T,
        )

    def randomize(self, seed):
        """Reset registered modules on a copy; direct free parameters remain fixed.

        This is a partial module-reset sanity check, not a fully random network.
        """
        other = deepcopy(self)
        other._profiles = {}
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for module in other.model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        other.model.eval()
        return other


class RidgeAdapter:
    """Per-ROI AR(1) ridge; equal subject contribution; train subjects only."""

    def fit(self, subjects):
        if not subjects or any(s.split != "train" for s in subjects):
            raise ValueError("Ridge baseline only accepts nonempty train subjects")
        f = subjects[0].bold.shape[1]
        lhs = np.zeros((f, 2, 2))
        rhs = np.zeros((f, 2))
        for s in subjects:
            x, y = s.bold[:-1], s.bold[1:]
            design = np.stack([np.ones_like(x), x], axis=-1)
            lhs += np.einsum("tfi,tfj->fij", design, design) / len(x)
            rhs += np.einsum("tfi,tf->fi", design, y) / len(x)
        self.weights = np.linalg.solve(
            lhs + np.diag([0.0, 1.0]) * 1e-3, rhs[..., None]
        )[..., 0]
        return self

    def predict(self, history, horizon):
        x = history[-1].copy()
        out = []
        for _ in range(horizon):
            x = self.weights[:, 0] + self.weights[:, 1] * x
            out.append(x.copy())
        return Prediction(np.asarray(out))


class SyntheticAdapter:
    """Fixed test fixture, explicitly not a trained NeuroTwin checkpoint."""

    context = 64
    calibration = 32
    variance_trained = False
    metadata = dict(
        architecture="synthetic AR fixture",
        dynamics_mode="synthetic",
        checkpoint=None,
        output_scale="prefix-calibrated BOLD",
        capabilities=["autoregressive"],
    )

    def predict(
        self,
        subject,
        anchor,
        horizon,
        mode,
        context=None,
        history=None,
        conditions=None,
    ):
        if mode != "autoregressive":
            raise NotImplementedError("Synthetic fixture only supports autoregressive")
        x = (subject.bold[anchor] if history is None else history[-1]).copy()
        return Prediction(np.stack([x * 0.5**h for h in range(1, horizon + 1)]))
