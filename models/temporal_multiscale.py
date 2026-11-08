"""共享于所有 ROI 的多尺度因果历史编码；不预先分解整条扫描。"""
import torch
from torch import nn
from torch.nn import functional as F

from models.tfm_backbone import RMSNorm


class CausalTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.left_pad = 2 * dilation
        self.norm = RMSNorm(channels)
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化只沿通道，不能使用会混合未来时间点的 GroupNorm。
        z = self.norm(x.transpose(1, 2))
        k, dilation = z.shape[1], self.conv.dilation[0]
        padded = F.pad(z, (0, 0, self.left_pad, 0))
        # 与左填充的扩张Conv1d完全相同；避开CuDNN确定性的大batch慢路径。
        # 保留Conv1d参数名/形状，使已有实验状态可严格恢复。
        windows = torch.cat([padded[:, j*dilation:j*dilation+k] for j in range(3)], -1)
        weight = self.conv.weight.permute(0, 2, 1).reshape(self.conv.out_channels, -1)
        z = F.silu(F.linear(windows, weight, self.conv.bias))
        z = F.linear(z, self.mix.weight[..., 0], self.mix.bias)
        return x + z.transpose(1, 2)


class MultiScaleTemporalEncoder(nn.Module):
    """[B,F,K] → [B,F,D]，联合各尺度最新状态与完整可见历史聚合。"""
    is_prediction_adapter = True

    def __init__(self, channels: int = 64, dim: int = 256):
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList([CausalTemporalBlock(channels, d) for d in (1, 2, 4, 8)])
        self.out_proj = nn.Linear(6 * channels, dim)
        self.out_norm = RMSNorm(dim)

    def temporal_features(self, history: torch.Tensor) -> list[torch.Tensor]:
        """逐时刻因果特征 [B*F,C,K]，供编码和因果性检查共用。"""
        b, f, k = history.shape
        z = F.linear(history.reshape(b*f, k, 1), self.input_proj.weight[..., 0],
                     self.input_proj.bias).transpose(1, 2)
        features = [z]
        for block in self.blocks:
            z = block(z)
            features.append(z)
        return features

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        b, f, _ = history.shape
        features = self.temporal_features(history)
        summary = torch.cat([z[..., -1] for z in features] + [features[-1].mean(-1)], -1)
        return self.out_norm(self.out_proj(summary)).reshape(b, f, -1)
