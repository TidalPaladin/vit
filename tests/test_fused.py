import pytest
import torch
from torch.testing import assert_close

from vit.fused import LayerNormLinear, LayerNormMLP


class TestLayerNormLinear:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype):
        layer_norm_linear = LayerNormLinear(10, 20)
        x = torch.randn(10)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_linear(x)
        assert y.shape == (20,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype):
        layer_norm_linear = LayerNormLinear(10, 20)
        x = torch.randn(10, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_linear(x)
        y.sum().backward()
        for param in layer_norm_linear.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()


class TestLayerNormMLP:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "srelu"])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, activation, normalization):
        layer_norm_mlp = LayerNormMLP(10, 20, activation=activation, normalization=normalization)
        x = torch.randn(10)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_mlp(x)
        assert y.shape == (10,)

    def test_determinstic(self):
        torch.random.manual_seed(0)
        layer = LayerNormMLP(10, 20, dropout=0.1)
        x = torch.randn(10)

        layer.train()
        y1 = layer(x)
        y2 = layer(x)
        assert not torch.allclose(y1, y2)

        layer.eval()
        y3 = layer(x)
        y4 = layer(x)
        assert_close(y3, y4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype):
        layer_norm_mlp = LayerNormMLP(10, 20)
        x = torch.randn(10, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_mlp(x)
        y.sum().backward()
        for param in layer_norm_mlp.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()
