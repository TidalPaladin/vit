import pytest
import torch
from torch.testing import assert_close

from vit.attention import AttentivePool, CrossAttention, SelfAttention


class TestSelfAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_forward(self, dtype, device, use_fourier_features):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, use_fourier_features=use_fourier_features).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        pos = torch.randn(B, L, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, pos)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_backward(self, dtype, device, use_fourier_features):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, use_fourier_features=use_fourier_features).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        pos = torch.randn(B, L, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, pos)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_forward_determinstic(self, device, use_fourier_features):
        B, L, D = 16, 128, 128
        layer = SelfAttention(D, D // 16, use_fourier_features=use_fourier_features).to(device)
        x = torch.randn(B, L, D, device=device)
        pos = torch.randn(B, L, 2, device=device)

        layer.eval()
        y1 = layer(x, pos)
        y2 = layer(x, pos)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x, pos)
        y4 = layer(x, pos)
        assert not torch.allclose(y3, y4)


class TestCrossAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_forward(self, dtype, device, use_fourier_features):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, use_fourier_features=use_fourier_features).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        pos_q = torch.randn(B, L, 2, dtype=dtype, device=device)
        pos_k = torch.randn(B, L // 2, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, kv, pos_q, pos_k)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_backward(self, dtype, device, use_fourier_features):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, use_fourier_features=use_fourier_features).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        pos_q = torch.randn(B, L, 2, dtype=dtype, device=device)
        pos_k = torch.randn(B, L // 2, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, kv, pos_q, pos_k)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_forward_determinstic(self, device, use_fourier_features):
        B, L, D = 16, 128, 128
        layer = CrossAttention(D, D // 16, use_fourier_features=use_fourier_features).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        pos_q = torch.randn(B, L, 2, device=device)
        pos_k = torch.randn(B, L // 2, 2, device=device)

        layer.eval()
        y1 = layer(x, kv, pos_q=pos_q, pos_k=pos_k)
        y2 = layer(x, kv, pos_q=pos_q, pos_k=pos_k)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x, kv, pos_q=pos_q, pos_k=pos_k)
        y4 = layer(x, kv, pos_q=pos_q, pos_k=pos_k)
        assert not torch.allclose(y3, y4)


class TestAttentivePool:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype, device):
        B, L, D = 16, 128, 128
        layer = AttentivePool(D, D // 16).to(device)
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
