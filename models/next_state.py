"""Next whole-brain TR prediction using NeuroTwin's SC-conditioned encoder."""
import torch
import torch.nn as nn

from models.backbone import BrainDynamicsBackbone


class NextStateHead(nn.Module):
    """Read the current latent ROI state without flattening a fixed time axis."""

    def __init__(self, hidden_dim=32, forecast_offsets=(1,)):
        super().__init__()
        self.forecast_offsets = tuple(int(offset) for offset in forecast_offsets)
        if self.forecast_offsets != (1,):
            raise ValueError('Parallel MTP is reserved; this head currently supports forecast_offsets=[1]')
        self.net = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent, history_norm):
        features = torch.stack((latent[:, :, 0, -1], latent[:, :, 0].mean(-1),
                                history_norm[:, :, 0, -1]), dim=-1)
        return self.net(features).squeeze(-1)


class NextBrainStateModel(nn.Module):
    """Public interface: history [B,K,F], SC [B,F,F], HAMD [B,1]."""

    def __init__(self, backbone, context_max, prediction_target='delta',
                 forecast_offsets=(1,), head_hidden_dim=32, normalize=True):
        super().__init__()
        if prediction_target not in ('delta', 'absolute'):
            raise ValueError('prediction_target must be delta or absolute')
        self.backbone = backbone
        self.next_head = NextStateHead(head_hidden_dim, forecast_offsets)
        self.context_max = int(context_max)
        self.prediction_target = prediction_target
        self.use_history_norm = bool(normalize)

    @staticmethod
    def history_statistics(history, mask=None):
        if mask is None:
            mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        if mask.shape != history.shape[:2] or not mask.any(dim=1).all():
            raise ValueError('Invalid history mask')
        weight = mask.unsqueeze(-1).to(history.dtype)
        count = weight.sum(dim=1, keepdim=True)
        mean = (history * weight).sum(dim=1, keepdim=True) / count
        variance = ((history - mean).square() * weight).sum(dim=1, keepdim=True) / count
        return mean.detach(), (variance + 1e-5).sqrt().detach()

    def forward(self, bold_history, sc, hamd=None, history_mask=None):
        if bold_history.ndim != 3 or bold_history.shape[-1] != self.backbone.features:
            raise ValueError(f'Expected history [B,K,{self.backbone.features}]')
        b, k, f = bold_history.shape
        if not 1 <= k <= self.context_max or sc.shape != (b, f, f):
            raise ValueError('Context length or SC shape mismatch')
        if not torch.isfinite(bold_history).all() or not torch.isfinite(sc).all():
            raise ValueError('Nonfinite history or SC')
        if not self.backbone.pretrain_mode and hamd is None:
            raise ValueError('MDD next-state model requires HAMD')
        if history_mask is None:
            history_mask = torch.ones(b, k, dtype=torch.bool, device=bold_history.device)
        if not history_mask[:, -1].all():
            raise ValueError('Current state must be a valid history point')
        if self.use_history_norm:
            mean, std = self.history_statistics(bold_history, history_mask)
        else:
            mean = torch.zeros_like(bold_history[:, :1])
            std = torch.ones_like(mean)
        normalized = (bold_history - mean) / std
        normalized = normalized * history_mask.unsqueeze(-1)
        if k < self.context_max:
            normalized = torch.nn.functional.pad(normalized, (0, 0, self.context_max - k, 0))
        encoded_input = normalized.transpose(1, 2).unsqueeze(2)
        cond = self.backbone._prepare_pathology_condition(hamd)
        latent, sc_info, _ = self.backbone._encode(encoded_input, sc, cond)
        base = self.next_head(latent, encoded_input)
        correction = torch.zeros_like(base)
        aux = {'latent': latent}
        if self.backbone.moe is not None and self.backbone.patho_cond_layer != 'feature_only':
            adjacency = sc_info['A_eff'] if sc_info is not None else None
            residual, moe_aux = self.backbone.moe(
                latent, encoded_input, base[:, :, None, None], cond, sc,
                sc_adj=adjacency, revin_stats=None)
            correction = residual[:, :, 0, 0]
            aux.update(moe_aux)
        normalized_output = base + correction
        current = bold_history[:, -1]
        if self.prediction_target == 'delta':
            pred_delta = normalized_output * std[:, 0]
            pred_next = current + pred_delta
        else:
            pred_next = normalized_output * std[:, 0] + mean[:, 0]
            pred_delta = pred_next - current
        hc_delta = (base * std[:, 0] if self.prediction_target == 'delta'
                    else base * std[:, 0] + mean[:, 0] - current)
        return {'pred_next': pred_next, 'pred_delta': pred_delta,
                'delta_hc': hc_delta,
                'delta_mdd': correction * std[:, 0], **aux}


def build_next_model(args, device):
    backbone = BrainDynamicsBackbone(args)
    return NextBrainStateModel(backbone, args.context_max, args.prediction_target,
                               args.forecast_offsets, normalize=args.norm).to(device)
