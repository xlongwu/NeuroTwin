"""Fixed-anchor H32 trajectory diagnostics shared by validation and evaluation."""
import torch


STAGES = ((0, 1), (0, 8), (8, 16), (16, 32), (0, 32))
STAGE_LABELS = tuple('H1' if end == 1 else f'H{start + 1}_{end}' for start, end in STAGES)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.square().sum(-1) * b.square().sum(-1)).sqrt().clamp_min(1e-8)


@torch.no_grad()
def h32_stage_metrics(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor, anchor: torch.Tensor,
                      direction_deadzone: float = 0.05) -> dict:
    """Return [B, stages] metrics; use only anchors with all 32 real targets.

    Temporal metrics are computed per ROI then averaged. Undefined target
    variance is excluded; a constant prediction against varying truth has PCC 0.
    PSD diagnostics use all non-DC rFFT bins within each stage.
    """
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 32:
        raise ValueError('H32 metrics require matching [B,F,32] predictions and targets')
    if mask.shape != (pred.shape[0], 32) or anchor.shape != pred.shape[:2]:
        raise ValueError('H32 metrics require mask [B,32] and anchor [B,F]')
    complete = (mask > 0).all(-1)
    metrics = {name: [] for name in ('mae', 'rmse', 'temporal_pcc', 'std_ratio',
               'delta_pcc', 'direction_accuracy', 'acf1_error', 'psd_shape_pcc', 'log_psd_mae')}
    delta_p = torch.diff(torch.cat((anchor.unsqueeze(-1), pred), -1), dim=-1)
    delta_t = torch.diff(torch.cat((anchor.unsqueeze(-1), target), -1), dim=-1)
    for start, end in STAGES:
        p, t = pred[..., start:end], target[..., start:end]
        err = p - t
        values = {'mae': err.abs().mean((1, 2)), 'rmse': err.square().mean((1, 2)).sqrt()}
        dp, dt = delta_p[..., start:end], delta_t[..., start:end]
        meaningful = dt.abs() > direction_deadzone
        direction = (dp.sign() == dt.sign()).to(p.dtype)
        values['direction_accuracy'] = (
            (direction * meaningful).sum((1, 2)) / meaningful.sum((1, 2)).clamp_min(1))
        values['direction_accuracy'] = values['direction_accuracy'].masked_fill(
            ~meaningful.flatten(1).any(-1), float('nan'))
        if end - start > 1:
            varying = t.std(-1, correction=0) > 1e-8
            def roi_mean(value):
                return value.masked_fill(~varying, float('nan')).nanmean(-1)
            values['temporal_pcc'] = roi_mean(_pcc(p, t))
            values['std_ratio'] = roi_mean(p.std(-1, correction=0) / t.std(-1, correction=0).clamp_min(1e-8))
            values['delta_pcc'] = _pcc(dp, dt).masked_fill(
                dt.std(-1, correction=0) <= 1e-8, float('nan')).nanmean(-1)
            # Biased lag-1 ACF on centered signals, with a shared T denominator.
            pc, tc = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
            acp = (pc[..., 1:] * pc[..., :-1]).sum(-1) / pc.square().sum(-1).clamp_min(1e-8)
            act = (tc[..., 1:] * tc[..., :-1]).sum(-1) / tc.square().sum(-1).clamp_min(1e-8)
            values['acf1_error'] = roi_mean((acp - act).abs())
            pp = torch.fft.rfft(pc, dim=-1, norm='ortho').abs().square()[..., 1:]
            tp = torch.fft.rfft(tc, dim=-1, norm='ortho').abs().square()[..., 1:]
            values['psd_shape_pcc'] = roi_mean(_pcc(pp, tp))
            values['log_psd_mae'] = roi_mean((pp.clamp_min(1e-8).log() - tp.clamp_min(1e-8).log()).abs().mean(-1))
        for key in metrics:
            val = values.get(key, pred.new_full((pred.shape[0],), float('nan')))
            metrics[key].append(val.masked_fill(~complete, float('nan')))
    return {key: torch.stack(values, -1) for key, values in metrics.items()}
