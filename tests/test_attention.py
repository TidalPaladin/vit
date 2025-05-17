from timeit import timeit

import pytest
import torch
from torch.testing import assert_close

from vit.attention import AttentivePool, CrossAttention, PolarApprox, SelfAttention, separable_polar_approx


@pytest.mark.parametrize("k,n", [(3, 1), (1, 3), (3, 3)])
def test_separable_polar_approx(k, n):
    B, H, L = 2, 4, 32
    r = torch.rand(B, L)
    theta = torch.rand(B, L) * 2 * torch.pi
    b = torch.randn(H, k + 1, n + 1)
    c = torch.randn(H, k + 1, n + 1)
    result = separable_polar_approx(r, theta, b, c)
    assert tuple(result.shape) == (B, H, L)


@pytest.mark.cuda
def test_separable_polar_approx_time():
    k = 2
    n = 4
    B, H, L = 32, 8, 16384
    r = torch.rand(B, L, device="cuda")
    theta = torch.rand(B, L, device="cuda") * 2 * torch.pi
    b = torch.randn(H, k + 1, n + 1, device="cuda")
    c = torch.randn(H, k + 1, n + 1, device="cuda")
    separable_polar_approx(r, theta, b, c)  # warmup compile

    def fn():
        separable_polar_approx(r, theta, b, c)
        torch.cuda.synchronize()

    t1 = timeit(fn, number=100) / 100
    assert t1 < 0.1
    assert False


class TestPolarApprox:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype, device):
        B, H, L = 2, 4, 32
        r = torch.rand(B, L, device=device)
        theta = torch.rand(B, L, device=device) * 2 * torch.pi
        layer = PolarApprox(radial_degree=2, angular_degree=4, nhead=H).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = layer(r, theta)
        assert tuple(y.shape) == (B, H, L)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, H, L = 2, 4, 32
        r = torch.rand(B, L, device=device)
        theta = torch.rand(B, L, device=device) * 2 * torch.pi
        layer = PolarApprox(radial_degree=2, angular_degree=4, nhead=H).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = layer(r, theta)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None


class TestSelfAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16).to(device)
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


class TestCrossAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16).to(device)
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


class TestSelfAttentionWithBiases:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, attn_bias=True).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        pos = torch.randn(B, L, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, pos)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, attn_bias=True).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        pos = torch.randn(B, L, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, pos)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = SelfAttention(D, D // 16, attn_bias=True).to(device)
        x = torch.randn(B, L, D, device=device)
        pos = torch.randn(B, L, 2, device=device)
        layer.eval()
        y1 = layer(x, pos)
        y2 = layer(x, pos)
        assert_close(y1, y2)

        layer.train()
        layer(x, pos)
        layer(x, pos)


class TestCrossAttentionWithBiases:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, attn_bias=True).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        posq = torch.randn(B, L, 2, dtype=dtype, device=device)
        posk = torch.randn(B, L // 2, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, kv, posq, posk)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, attn_bias=True).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        posq = torch.randn(B, L, 2, dtype=dtype, device=device)
        posk = torch.randn(B, L // 2, 2, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = multihead_attention(x, kv, posq, posk)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = CrossAttention(D, D // 16, attn_bias=True).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        posq = torch.randn(B, L, 2, device=device)
        posk = torch.randn(B, L // 2, 2, device=device)
        layer.eval()
        y1 = layer(x, kv, posq, posk)
        y2 = layer(x, kv, posq, posk)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x, kv, posq, posk)
        y4 = layer(x, kv, posq, posk)
        assert not torch.allclose(y3, y4)
