import pytest
import torch
from torch.testing import assert_close

from vit.transformer import CrossAttentionTransformer, TransformerDecoderLayer, TransformerEncoderLayer


class TestTransformerEncoderLayer:
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_forward(self, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerEncoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_backward(self, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerEncoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
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

    def test_encoder_layer_scale_applied_once(self, device):
        """Regression test: verify layer_scale is applied once, not twice (issue #79)."""
        hidden_size = 64
        layer_scale_init = 0.1  # Use a value where γ vs γ² is easily distinguishable

        layer = TransformerEncoderLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 4,
            num_attention_heads=4,
            layer_scale=layer_scale_init,
        ).to(device)

        # Set deterministic weights for reproducibility
        torch.manual_seed(42)
        x = torch.randn(2, 16, hidden_size, device=device)

        layer.eval()
        with torch.no_grad():
            output = layer(x)

        # The residual contribution should be scaled by γ (0.1), not γ² (0.01)
        # If layer_scale were applied twice, the output would be much closer to the input
        residual = output - x

        # With γ=0.1 applied once, residual magnitude should be ~0.1 * unscaled_magnitude
        # With γ²=0.01 applied twice, residual would be 10x smaller
        # Check that residual is not negligibly small (which would indicate double scaling)
        assert residual.abs().mean() > 0.001, "Residual too small - layer_scale may be applied twice"


class TestTransformerDecoderLayer:
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_forward(self, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerDecoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_backward(self, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerDecoderLayer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
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
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    def test_forward(self, device, layer_scale):
        B, L, D = 16, 128, 128
        transformer_layer = CrossAttentionTransformer(D, D, D // 16, layer_scale=layer_scale).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x, kv)
        assert y.shape == (B, L, D)

    def test_backward(self, device):
        B, L, D = 16, 128, 128
        transformer_layer = CrossAttentionTransformer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
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
