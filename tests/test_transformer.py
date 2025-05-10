import pytest
import torch

from vit.pos_enc import create_grid
from vit.transformer import TransformerDecoderLayer, TransformerEncoderLayer


class TestTransformerEncoderLayer:

    @pytest.mark.parametrize("bias", [True, False])
    def test_forward(self, bias):
        B, H, W, D = 2, 8, 8, 128
        Nh = D // 16
        layer = TransformerEncoderLayer(D, Nh, 4 * D, bias=bias)
        x = torch.randn(B, H * W, D)
        grid = create_grid((H, W), device=x.device, dtype=x.dtype).expand(B, -1, -1)
        y = layer(x, grid)
        assert x.shape == y.shape

    @pytest.mark.parametrize("bias", [True, False])
    def test_backward(self, bias):
        B, H, W, D = 2, 8, 8, 128
        Nh = D // 16
        layer = TransformerEncoderLayer(D, Nh, 4 * D, bias=bias)
        x = torch.randn(B, H * W, D, requires_grad=True)
        grid = create_grid((H, W), device=x.device, dtype=x.dtype).expand(B, -1, -1)
        y = layer(x, grid)
        y.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None
            assert not p.grad.isnan().any()


class TestTransformerDecoderLayer:

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("self_attention", [True, False])
    def test_forward(self, bias, self_attention):
        B, H, W, D = 2, 8, 8, 128
        Nh = D // 16
        layer = TransformerDecoderLayer(D, Nh, 4 * D, bias=bias, self_attention=self_attention)
        q = torch.randn(B, H * W, D)
        kv = torch.randn(B, H * W, D)
        qgrid = create_grid((H, W), device=q.device, dtype=q.dtype).expand(B, -1, -1)
        kvgrid = create_grid((H, W), device=kv.device, dtype=kv.dtype).expand(B, -1, -1)
        y = layer(q, kv, qgrid, kvgrid)
        assert q.shape == y.shape

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("self_attention", [True, False])
    def test_backward(self, bias, self_attention):
        B, H, W, D = 2, 8, 8, 128
        Nh = D // 16
        layer = TransformerDecoderLayer(D, Nh, 4 * D, bias=bias, self_attention=self_attention)
        q = torch.randn(B, H * W, D, requires_grad=True)
        kv = torch.randn(B, H * W, D, requires_grad=True)
        qgrid = create_grid((H, W), device=q.device, dtype=q.dtype).expand(B, -1, -1)
        kvgrid = create_grid((H, W), device=kv.device, dtype=kv.dtype).expand(B, -1, -1)
        y = layer(q, kv, qgrid, kvgrid)
        y.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None
            assert not p.grad.isnan().any()
