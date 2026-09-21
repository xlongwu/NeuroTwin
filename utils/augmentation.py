# coding=utf-8
"""
脑信号数据增强（优化版）
"""
import torch


class BrainSignalAugmentation:
    """
    训练时对单样本脑信号 x:[F, W, S] 做轻量增强

    改进策略：
    1. additive noise：模拟扫描噪声
    2. multiplicative scaling：模拟个体幅值波动
    3. channel dropout：减轻对单一 ROI 的过依赖
    4. contiguous time mask：增强对局部缺失/扰动的鲁棒性
    """
    def __init__(
        self,
        noise_std: float = 0.03,
        scale_std: float = 0.05,
        channel_drop_prob: float = 0.03,
        time_mask_prob: float = 0.15,
        max_mask_ratio: float = 0.10,
        enable_noise: bool = True,
        enable_scale: bool = True,
        enable_channel_drop: bool = True,
        enable_time_mask: bool = True,
    ):
        self.noise_std = noise_std
        self.scale_std = scale_std
        self.channel_drop_prob = channel_drop_prob
        self.time_mask_prob = time_mask_prob
        self.max_mask_ratio = max_mask_ratio
        self.enable_noise = enable_noise
        self.enable_scale = enable_scale
        self.enable_channel_drop = enable_channel_drop
        self.enable_time_mask = enable_time_mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [F, W, S], got {tuple(x.shape)}")

        x = x.clone()
        f, _, s = x.shape

        if self.enable_scale and self.scale_std > 0:
            scale = 1.0 + torch.randn(f, 1, 1, device=x.device, dtype=x.dtype) * self.scale_std
            scale = scale.clamp(0.8, 1.2)
            x = x * scale

        if self.enable_noise and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if self.enable_channel_drop and self.channel_drop_prob > 0:
            mask = (torch.rand(f, 1, 1, device=x.device) > self.channel_drop_prob).to(dtype=x.dtype)
            x = x * mask

        if self.enable_time_mask and self.time_mask_prob > 0 and torch.rand(1).item() < self.time_mask_prob:
            max_mask = max(1, int(round(s * self.max_mask_ratio)))
            mask_len = int(torch.randint(1, max_mask + 1, (1,)).item())
            start = int(torch.randint(0, max(1, s - mask_len + 1), (1,)).item())
            fill_value = x.mean(dim=-1, keepdim=True)
            x[:, :, start:start + mask_len] = fill_value.expand(-1, -1, mask_len)

        return x
