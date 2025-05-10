import pytest
import torch
from torch.testing import assert_close

from vit.fused import NormLinear, NormMLP


class TestNormLinear:

    @pytest.mark.parametrize("bias", [True, False])
    def test_forward(self, bias):
        layer = NormLinear(10, 10, bias=bias)
        x = torch.randn(1, 10)
        y = layer(x)
        assert x.shape == y.shape

    @pytest.mark.parametrize("bias", [True, False])
    def test_backward(self, bias):
        layer = NormLinear(10, 10, bias=bias)
        x = torch.randn(1, 10)
        y = layer(x)
        y.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None
            assert not p.grad.isnan().any()


class TestNormMLP:

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("activation", ["gelu", "srelu", "silu"])
    def test_forward(self, bias, activation):
        layer = NormMLP(10, 10, bias=bias, activation=activation)
        x = torch.randn(1, 10)
        y = layer(x)
        assert x.shape == y.shape

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("activation", ["gelu", "srelu", "silu"])
    def test_backward(self, bias, activation):
        layer = NormMLP(10, 10, bias=bias, activation=activation)
        x = torch.randn(1, 10)
        y = layer(x)
        y.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None
            assert not p.grad.isnan().any()

    def test_deterministic(self):
        layer = NormMLP(10, 10, dropout=0.5)
        x = torch.randn(1, 10)
        layer.eval()
        assert_close(layer(x), layer(x))
        layer.train()
        assert not torch.allclose(layer(x), layer(x))
