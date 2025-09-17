import pytest
import torch
from torch.testing import assert_close

from vit.attention import AttentivePool, CrossAttention, SelfAttention


class TestSelfAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("gated", [False, True])
    def test_forward(self, dtype, device, gated):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, gated=gated).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = SelfAttention(D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_weights(self, dtype, device):
        B, L, D = 16, 128, 128
        H = D // 16
        multihead_attention = SelfAttention(D, H).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention.forward_weights(x)
        assert y.shape == (B, H, L, L)


class TestCrossAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("gated", [False, True])
    def test_forward(self, dtype, device, gated):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, gated=gated).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, kv)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = CrossAttention(D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)

        layer.eval()
        y1 = layer(x, kv)
        y2 = layer(x, kv)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x, kv)
        y4 = layer(x, kv)
        assert not torch.allclose(y3, y4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_weights(self, dtype, device):
        B, L, D = 16, 128, 128
        H = D // 16
        layer = CrossAttention(D, H).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = layer.forward_weights(x, kv)
        assert y.shape == (B, H, L, L // 2)


class TestAttentivePool:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("gated", [False, True])
    def test_forward(self, dtype, device, gated):
        B, L, D = 16, 128, 128
        layer = AttentivePool(D, D // 16, gated=gated).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = layer(x)
        assert y.shape == (B, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, L, D = 16, 128, 128
        layer = AttentivePool(D, D // 16).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_weights(self, dtype, device):
        B, L, D = 16, 128, 128
        H = D // 16
        layer = AttentivePool(D, H).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = layer.forward_weights(x)
        assert y.shape == (B, L, H)
