"""V2局部—全局频率状态及随时距变化的相位、幅值与可靠性。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LocalGlobalPhaseForecast(nn.Module):
    """16/32/64 TR末端复数状态融合，只由历史预测未来振荡。"""
    is_prediction_adapter = True

    def __init__(self, context_length: int = 64, horizon: int = 32,
                 dim: int = 256, modes: int = 13, tr_seconds: float = 2.):
        if context_length != 64:
            raise ValueError("V2使用固定K64，频率窗口为16/32/64 TR")
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
        nn.init.constant_(self.gate_phase.bias, -2.)

        self.windows = (16, 32, 64)
        self.window_embed = nn.Parameter(torch.zeros(3, 64))
        nn.init.normal_(self.window_embed, std=.02)
        self.scale_weight = nn.Linear(64, 1)
        self.horizon_proj = nn.Linear(3, 64)
        self.horizon_parameters = nn.Linear(64, 3)
        nn.init.normal_(self.horizon_parameters.weight, std=.01)
        nn.init.zeros_(self.horizon_parameters.bias)

    def correction_gate(self, state: torch.Tensor, phase_features: torch.Tensor) -> torch.Tensor:
        h = (self.future_steps/self.horizon).view(1, 1, -1, 1)
        return torch.sigmoid(self.gate_state(state).unsqueeze(-2)
                             + self.gate_phase(phase_features) + self.gate_horizon(h)).squeeze(-1)

    def window_coefficients(self, history: torch.Tensor) -> torch.Tensor:
        """[B,F,S,M]，均在K64频率网格上投影并对齐到同一历史末帧。"""
        if history.shape[-1] != self.context_length:
            raise ValueError('局部—全局相位预测需要 K64 当前历史窗口')
        coefficients = []
        with torch.autocast(device_type=history.device.type, enabled=False):
            for length in self.windows:
                # 时间坐标以末帧为零，避免不同窗口的复数相位原点错位。
                t = torch.arange(1-length, 1, device=history.device, dtype=torch.float32)
                angle = -2*math.pi*t[:, None]*self.bins/self.context_length
                basis = torch.polar(torch.ones_like(angle), angle)
                x = history[..., -length:].float()
                x = x - x.mean(-1, keepdim=True)
                coefficients.append((x.to(basis.dtype) @ basis)/length)
        return torch.stack(coefficients, -2)

    def forward(self, history: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.window_coefficients(history)
        magnitude = coeff.abs()
        unit = coeff/magnitude.clamp_min(1e-8)
        spectral = torch.stack((coeff.real, coeff.imag, magnitude, unit.real, unit.imag,
                                (self.bins/self.modes).expand_as(magnitude)), -1)
        hidden = F.silu(self.spectral_proj(spectral)
                        + self.state_proj(state)[..., None, None, :]
                        + self.window_embed[:, None, :])
        logits = self.scale_weight(hidden).squeeze(-1)
        lengths = self.bins.new_tensor(self.windows)
        # 至少覆盖一个周期才允许该窗口贡献；慢模式只能由长窗口估计。
        resolvable = lengths[:, None]*self.bins >= self.context_length
        weights = logits.masked_fill(~resolvable, -torch.inf).softmax(-2)
        anchor_coeff = (coeff*weights).sum(-2)
        hidden = (hidden*weights[..., None]).sum(-3)
        weight, shift, slope, decay = self.mode_parameters(hidden).unbind(-1)
        progress = self.future_steps/self.horizon
        coordinates = torch.stack((progress, torch.sin(math.pi*progress),
                                   torch.cos(math.pi*progress)), -1)
        dynamic = F.silu(hidden.unsqueeze(-3) + self.horizon_proj(coordinates)[:, None, :])
        amplitude_delta, phase_delta, reliability_delta = self.horizon_parameters(dynamic).unbind(-1)
        p = progress[:, None]
        phase_correction = (math.pi/2*shift.tanh().unsqueeze(-2)
                            + math.pi*phase_delta.tanh())*p
        theta = 2*math.pi*self.future_steps[:, None]*self.bins/self.context_length + phase_correction
        with torch.autocast(device_type=history.device.type, enabled=False):
            oscillations = 2*anchor_coeff.unsqueeze(-2)*torch.polar(torch.ones_like(theta.float()), theta.float())
        amplitude = 1 + .5*(slope.unsqueeze(-2) + amplitude_delta).tanh()
        reliability = torch.sigmoid(weight.unsqueeze(-2)
                                    - F.softplus(decay).unsqueeze(-2)*p + reliability_delta)
        modes = oscillations*amplitude*reliability
        features = torch.stack((modes.real, modes.imag, modes.abs(), reliability), -1)
        return self.feature_proj(features.flatten(-2)), modes.real.sum(-1).to(history.dtype)
