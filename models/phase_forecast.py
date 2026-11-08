"""以历史末帧为相位原点的有界振荡先验；仍由 CPM 预测完整未来。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PhaseAwareForecast(nn.Module):
    """历史 [B,F,K]、多尺度状态 [B,F,D] → 动态特征及轨迹校正。

    rFFT 的相位原点为历史首帧。先旋转 K-1 帧到预测锚点，再外推 h=1..H；
    频移仅作用于锚点之后，幅值调制与模式可靠性均有界。
    """
    is_prediction_adapter = True

    def __init__(self, context_length: int = 64, horizon: int = 32,
                 dim: int = 256, modes: int = 13, tr_seconds: float = 2.):
        super().__init__()
        if not 1 <= modes < context_length // 2:
            raise ValueError('phase_modes 必须为非DC且非Nyquist的历史FFT频点')
        if tr_seconds <= 0:
            raise ValueError('tr_seconds 必须为正')
        self.context_length, self.horizon = context_length, horizon
        self.modes, self.tr_seconds = modes, float(tr_seconds)
        # 整数网格由EMA直接复制，避免对固定频率/时间坐标进行浮点滑动平均。
        self.register_buffer('bins', torch.arange(1, modes+1, dtype=torch.long))
        self.register_buffer('future_steps', torch.arange(1, horizon+1, dtype=torch.long))
        self.spectral_proj = nn.Linear(6, 64)
        self.state_proj = nn.Linear(dim, 64)
        # weight、frequency shift、amplitude slope、decay。
        self.mode_parameters = nn.Linear(64, 4)
        nn.init.normal_(self.mode_parameters.weight, std=1e-3)
        nn.init.zeros_(self.mode_parameters.bias)
        with torch.no_grad():
            self.mode_parameters.bias[3] = -2.
        self.feature_proj = nn.Sequential(nn.Linear(modes*4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.gate_state = nn.Linear(dim, 1, bias=False)
        self.gate_phase = nn.Linear(dim, 1)
        self.gate_horizon = nn.Linear(1, 1, bias=False)
        for layer in (self.gate_state, self.gate_phase, self.gate_horizon):
            nn.init.normal_(layer.weight, std=1e-3)
        nn.init.constant_(self.gate_phase.bias, -4.)

    def history_coefficients(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-1] != self.context_length:
            raise ValueError(f'相位预测需要 K{self.context_length} 当前历史窗口')
        with torch.autocast(device_type=history.device.type, enabled=False):
            return torch.fft.rfft(history.float(), dim=-1, norm='forward')[..., 1:self.modes+1]

    def extrapolate_modes(self, coeff: torch.Tensor, frequency_shift: torch.Tensor) -> torch.Tensor:
        """[B,F,M] → [B,F,H,M] 复数模式；频移单位为 Hz。"""
        with torch.autocast(device_type=coeff.device.type, enabled=False):
            freq = self.bins / (self.context_length*self.tr_seconds)
            anchor_phase = 2*math.pi*self.bins*(self.context_length-1)/self.context_length
            anchor_coeff = coeff * torch.polar(torch.ones_like(anchor_phase), anchor_phase)
            theta = 2*math.pi*(freq + frequency_shift.float()).unsqueeze(-2) \
                * (self.tr_seconds*self.future_steps[:, None])
            return 2*anchor_coeff.unsqueeze(-2)*torch.polar(torch.ones_like(theta), theta)

    def forward(self, history: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.history_coefficients(history)
        magnitude = coeff.abs()
        phase_unit = coeff / magnitude.clamp_min(1e-8)
        spectral = torch.stack((coeff.real, coeff.imag, magnitude,
                                phase_unit.real, phase_unit.imag,
                                (self.bins/self.modes).expand_as(magnitude)), -1)
        hidden = F.silu(self.spectral_proj(spectral) + self.state_proj(state).unsqueeze(-2))
        weight, shift, slope, decay = self.mode_parameters(hidden).unbind(-1)
        # 最大频移为半个历史FFT bin；不放大为无界指数振荡。
        shift_hz = .5 / (self.context_length*self.tr_seconds) * shift.tanh()
        oscillations = self.extrapolate_modes(coeff, shift_hz)
        progress = (self.future_steps/self.horizon)[:, None]
        reliability = torch.sigmoid(weight.unsqueeze(-2) - F.softplus(decay).unsqueeze(-2)*progress)
        amplitude = 1 + .5*slope.tanh().unsqueeze(-2)*progress
        modes = oscillations * reliability * amplitude
        features = torch.stack((modes.real, modes.imag, modes.abs(), reliability.expand_as(modes.real)), -1)
        phase_features = self.feature_proj(features.flatten(-2))
        return phase_features, modes.real.sum(-1).to(history.dtype)

    def correction_gate(self, state: torch.Tensor, phase_features: torch.Tensor) -> torch.Tensor:
        h = (self.future_steps/self.horizon).view(1, 1, -1, 1)
        return torch.sigmoid(self.gate_state(state).unsqueeze(-2)
                             + self.gate_phase(phase_features) + self.gate_horizon(h)).squeeze(-1)
