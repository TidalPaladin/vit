import pytest
import torch
from torch.testing import assert_close

from vit.transformer import CrossAttentionTransformer, TransformerDecoderLayer, TransformerEncoderLayer


class TestTransformerEncoderLayer:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_forward(self, dtype, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerEncoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = transformer_layer(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_backward(self, dtype, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerEncoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = transformer_layer(x)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = TransformerEncoderLayer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)


class TestTransformerDecoderLayer:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_forward(self, dtype, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerDecoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = transformer_layer(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_backward(self, dtype, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerDecoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = transformer_layer(x, kv)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = TransformerDecoderLayer(D, D, D // 16).to(device)
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


class TestCrossAttentionTransformer:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_forward(self, dtype, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = CrossAttentionTransformer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = transformer_layer(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype, device):
        B, L, D = 16, 128, 128
        transformer_layer = CrossAttentionTransformer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, dtype=dtype, device=device)
        kv = torch.randn(B, L // 2, D, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = transformer_layer(x, kv)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = CrossAttentionTransformer(D, D, D // 16).to(device)
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
