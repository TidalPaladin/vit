from copy import deepcopy

import pytest
import torch
from torch.testing import assert_close
from torchao.dtypes import AffineQuantizedTensor
from torchao.quantization import Int8WeightOnlyConfig

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

    def test_determinstic(self, device):
        torch.random.manual_seed(0)
        layer = NormLinear(10, 20, dropout=0.5).to(device)
        x = torch.randn(10, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    def test_quantization(self, device):
        torch.random.manual_seed(0)
        layer_norm_linear = NormLinear(10, 20).to(device)
        layer_norm_linear.eval()
        quantized_layer_norm_linear = deepcopy(layer_norm_linear)
        quantized_layer_norm_linear.apply_quantization(Int8WeightOnlyConfig())
        weight = quantized_layer_norm_linear.linear.weight
        assert isinstance(weight, AffineQuantizedTensor)

        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer_norm_linear(x)
            y_quant = quantized_layer_norm_linear(x)
        assert_close(y, y_quant, atol=1e-2, rtol=0)


class TestNormMLP:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "srelu", "reglu", "swiglu", "geglu", "openswiglu"])
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

    def test_quantization(self, device):
        torch.random.manual_seed(0)
        layer_norm_mlp = NormMLP(10, 20).to(device)
        layer_norm_mlp.eval()
        quantized_layer_norm_mlp = deepcopy(layer_norm_mlp)
        quantized_layer_norm_mlp.apply_quantization(Int8WeightOnlyConfig())
        weight1 = quantized_layer_norm_mlp.fc1.weight
        weight2 = quantized_layer_norm_mlp.fc2.weight
        assert isinstance(weight1, AffineQuantizedTensor)
        assert isinstance(weight2, AffineQuantizedTensor)

        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer_norm_mlp(x)
            y_quant = quantized_layer_norm_mlp(x)
        assert_close(y, y_quant, atol=1e-2, rtol=0)
