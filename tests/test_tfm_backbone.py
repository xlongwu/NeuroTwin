# coding=utf-8
"""NeuroTwin-TFM 主干模块单元测试（方案 §4–§6）。

重点验证：
- 形状契约（[B,F,K] → [B,F,P,D] → [B,F,P,D]）；
- Causal Temporal Attention 的因果性：修改未来 patch 不得影响过去 patch 的输出；
- SC-guided ROI Attention 的结构偏置语义：β=0 时与 A_eff 无关，β>0 时受 A_eff 调制；
- Block 端到端梯度流。
"""
import math

import pytest
import torch

from models.tfm_backbone import (
    CausalTemporalAttention,
    RMSNorm,
    SCGuidedROIAttention,
    TFMEncoderBlock,
    TemporalPatchEmbed,
)


# ---------------------------------------------------------------- RMSNorm
def test_rmsnorm_normalizes_and_prefix_slices():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 3, 8) * 5 + 1
    y = norm(x)
    # RMS 归一化后逐位置均方 ≈ 1（affine=1 时）
    rms = y.pow(2).mean(-1)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    # 前缀宽度：最后一维 4 < 8 时用 weight 前缀，不报错且数值有限
    y4 = norm(x[..., :4])
    assert y4.shape == (2, 3, 4) and torch.isfinite(y4).all()
    with pytest.raises(ValueError):
        norm(torch.randn(2, 3, 9))


# ------------------------------------------------------ TemporalPatchEmbed
def test_patch_embed_shapes_and_padding():
    torch.manual_seed(0)
    embed = TemporalPatchEmbed(features=6, context_max=16, patch_len=4, dim=12)
    assert embed.num_patches(16) == 4
    assert embed.num_patches(5) == 2          # ceil(5/4)
    x = torch.randn(3, 6, 16)
    h = embed(x)
    assert h.shape == (3, 6, 4, 12)
    x5 = torch.randn(3, 6, 5)
    h5 = embed(x5)
    assert h5.shape == (3, 6, 2, 12) and torch.isfinite(h5).all()
    with pytest.raises(ValueError):
        embed.num_patches(17)
    with pytest.raises(ValueError):
        embed(torch.randn(3, 7, 16))          # ROI 数不一致


def test_patch_embed_position_semantics():
    """变长 K：位置嵌入按**起点对齐**（第 i 个 patch ↔ 嵌入表第 i 行）。

    短序列（丢掉最早 TR）补零后，相同时间内容的 patch 投影结果一致；
    位置嵌入激活时的差异必须恰好等于位置嵌入之差。
    """
    torch.manual_seed(0)
    embed = TemporalPatchEmbed(features=4, context_max=8, patch_len=2, dim=8)
    embed.eval()
    x_full = torch.randn(1, 4, 8)
    x_short = x_full[:, :, 4:]                # 丢掉最早 4 个 TR（保留后 4 个）
    # 位置嵌入置零：内容相同 → 输出逐位一致（短序列 K=4 恰好无需补零）
    embed.pos_embed.data.zero_()
    h_full = embed(x_full)                    # [1,4,4,8]
    h_short = embed(x_short)                  # [1,4,2,8]
    assert torch.allclose(h_short[:, :, 1], h_full[:, :, 3], atol=1e-6)
    # 位置嵌入恢复：差异 = pos_embed[1] - pos_embed[3]（起点对齐语义）
    embed.pos_embed.data.normal_(0, 0.02)
    h_full = embed(x_full)
    h_short = embed(x_short)
    delta = h_short[:, :, 1] - h_full[:, :, 3]
    expect = (embed.pos_embed[1] - embed.pos_embed[3]).view(1, 1, -1)
    assert torch.allclose(delta, expect.expand_as(delta), atol=1e-6)


# ------------------------------------------------- CausalTemporalAttention
def test_temporal_attention_causality():
    """修改未来 patch 不得改变过去 patch 的输出（causal mask 的核心约束）。"""
    torch.manual_seed(0)
    attn = CausalTemporalAttention(dim=16, num_heads=4, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 5, 8, 16)
    out1 = attn(x)
    x2 = x.clone()
    x2[:, :, -1] = torch.randn_like(x2[:, :, -1])          # 只改最后一个 patch
    x2[:, :, -2] = torch.randn_like(x2[:, :, -2])          # 以及倒数第二个
    out2 = attn(x2)
    assert torch.allclose(out1[:, :, :-2], out2[:, :, :-2], atol=1e-5), \
        "causal temporal attention 泄漏了未来信息"
    assert not torch.allclose(out1[:, :, -1], out2[:, :, -1])


def test_temporal_attention_grad_flow():
    torch.manual_seed(0)
    attn = CausalTemporalAttention(dim=8, num_heads=2, dropout=0.1)
    x = torch.randn(2, 3, 6, 8, requires_grad=True)
    attn(x).sum().backward()
    assert attn.qkv.weight.grad is not None
    assert torch.isfinite(attn.qkv.weight.grad).all()


# ---------------------------------------------------- SCGuidedROIAttention
def _roi_attention_reference(attn: SCGuidedROIAttention, x, adj, use_bias):
    """与 forward 相同权重的手写参考实现，用于核对偏置语义。"""
    b, f, p, d = x.shape
    qkv = attn.qkv(x).permute(0, 2, 1, 3).reshape(
        b * p, f, 3, attn.num_heads, attn.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
    scores = scores.view(b, p, attn.num_heads, f, f)          # [B,P,h,F,F]
    if use_bias:
        bias = torch.log(adj + SCGuidedROIAttention.LOG_EPS)
        scores = scores + bias.view(b, 1, 1, f, f) \
            * attn.beta.view(1, 1, attn.num_heads, 1, 1)
    a = torch.softmax(scores.flatten(0, 1), dim=-1)
    out = torch.matmul(a, v).transpose(1, 2).reshape(b * p, f, d)
    out = attn.out_proj(out)
    return out.view(b, p, f, d).permute(0, 2, 1, 3)


def test_roi_attention_bias_semantics():
    """β=0 → 输出与 A_eff 无关（退化为普通 attention）；β>0 → 输出受 A_eff 调制。"""
    torch.manual_seed(0)
    attn = SCGuidedROIAttention(dim=8, num_heads=2, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 7, 3, 8)
    adj_a = torch.rand(2, 7, 7)
    adj_b = torch.rand(2, 7, 7)

    attn.beta.data.zero_()                    # β=0：无结构偏置
    out_a0 = attn(x, adj_a)
    out_b0 = attn(x, adj_b)
    assert torch.allclose(out_a0, out_b0, atol=1e-6), \
        "β=0 时 ROI attention 不应依赖 A_eff"
    assert torch.allclose(out_a0, _roi_attention_reference(attn, x, adj_a, False), atol=1e-5)

    attn.beta.data.fill_(1.0)                 # β>0：受 A_eff 调制
    out_a1 = attn(x, adj_a)
    out_b1 = attn(x, adj_b)
    assert not torch.allclose(out_a1, out_b1)
    assert torch.allclose(out_a1, _roi_attention_reference(attn, x, adj_a, True), atol=1e-5)

    with pytest.raises(ValueError):
        attn(x, None)


def test_roi_attention_shape_and_grad():
    torch.manual_seed(0)
    attn = SCGuidedROIAttention(dim=12, num_heads=3, dropout=0.1)
    x = torch.randn(2, 9, 4, 12, requires_grad=True)
    adj = torch.rand(2, 9, 9)
    out = attn(x, adj)
    assert out.shape == x.shape
    out.sum().backward()
    assert attn.beta.grad is not None and torch.isfinite(attn.beta.grad).all()


# ---------------------------------------------------------- TFMEncoderBlock
def test_encoder_block_end_to_end():
    """Block 前向形状 + 梯度回传 + 端到端因果性。"""
    torch.manual_seed(0)
    block = TFMEncoderBlock(dim=16, num_heads=4, dropout=0.0)
    block.eval()
    adj = torch.rand(2, 6, 6)
    x = torch.randn(2, 6, 5, 16)
    out = block(x, adj)
    assert out.shape == x.shape

    x2 = x.clone()
    x2[:, :, -1] = torch.randn_like(x2[:, :, -1])
    out2 = block(x2, adj)
    assert torch.allclose(out[:, :, :-1], out2[:, :, :-1], atol=1e-5), \
        "TFMEncoderBlock 泄漏了未来 patch 信息（temporal 轴必须 causal）"

    x.requires_grad_(True)
    block(x, adj).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"参数 {name} 梯度含 NaN/Inf"


def test_encoder_block_stack():
    """两层堆叠 + 变长 K 的完整 patch→block 链路冒烟。"""
    torch.manual_seed(0)
    embed = TemporalPatchEmbed(features=5, context_max=16, patch_len=4, dim=16)
    blocks = torch.nn.ModuleList([TFMEncoderBlock(16, 4, dropout=0.1) for _ in range(2)])
    x = torch.randn(2, 5, 10)                 # K=10：补零到 12 → P=3
    h = embed(x)
    adj = torch.rand(2, 5, 5)
    for blk in blocks:
        h = blk(h, adj)
    assert h.shape == (2, 5, 3, 16)
    h.mean().backward()
    assert embed.patch_proj.weight.grad is not None
