import pytest
import torch
from torch.testing import assert_close

from vit.fused import NormLinear, NormMLP


class TestNormLinear:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, device, dtype):
        layer_norm_linear = NormLinear(10, 20).to(device)
        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y = layer_norm_linear(x)
        assert y.shape == (20,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, device, dtype):
        layer_norm_linear = NormLinear(10, 20).to(device)
        x = torch.randn(10, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y = layer_norm_linear(x)
        y.sum().backward()
        for param in layer_norm_linear.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()


class TestNormMLP:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "srelu", "reglu", "swiglu", "geglu"])
    def test_forward(self, device, dtype, activation):
        layer_norm_mlp = NormMLP(10, 20, activation=activation).to(device)
        x = torch.randn(10, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y = layer_norm_mlp(x)
        assert y.shape == (10,)

    def test_determinstic(self, device):
        torch.random.manual_seed(0)
        layer = NormMLP(10, 20, dropout=0.1).to(device)
        x = torch.randn(10, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, device, dtype):
        layer_norm_mlp = NormMLP(10, 20).to(device)
        x = torch.randn(10, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y = layer_norm_mlp(x)
        y.sum().backward()
        for param in layer_norm_mlp.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_forward_matryoshka(self, device, dtype, activation):
        D_hidden, D_feedforward = 10, 20
        layer_norm_mlp = NormMLP(D_hidden, D_feedforward, activation=activation).to(device)
        x = torch.randn(D_hidden, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y1 = layer_norm_mlp(x)
            y2 = layer_norm_mlp(x, D_feedforward // 2)
        assert y1.shape == y2.shape
        assert not torch.allclose(y1, y2)
