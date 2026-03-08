from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close
from torchao.dtypes import AffineQuantizedTensor
from torchao.quantization import Int8WeightOnlyConfig

from vit.fused import AdaNormMLP, NormLinear, NormMLP


def _apply_norm_manual(
    x: torch.Tensor,
    norm: torch.nn.LayerNorm | torch.nn.RMSNorm,
    *,
    scale_delta: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
) -> torch.Tensor:
    bias = norm.bias if isinstance(norm, torch.nn.LayerNorm) else None
    if isinstance(norm, torch.nn.LayerNorm):
        x = F.layer_norm(x, x.shape[-1:], norm.weight, None, norm.eps)
    else:
        x = F.rms_norm(x, x.shape[-1:], norm.weight, norm.eps)
    if scale_delta is not None:
        x = x * (1 + scale_delta)
    if bias is not None:
        x = x + bias
    if shift is not None:
        x = x + shift
    return x


def _enable_adaln_gate(layer: AdaNormMLP) -> None:
    with torch.no_grad():
        assert layer.modulation.bias is not None
        hidden_size = layer.fc2.out_features
        layer.modulation.bias[2 * hidden_size :].fill_(1.0)


class TestNormLinear:
    def test_reset_parameters_zeros_bias(self):
        layer = NormLinear(10, 20)
        assert layer.linear.bias is not None
        assert torch.count_nonzero(layer.linear.bias) == 0

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, norm_type):
        layer_norm_linear = NormLinear(10, 20, norm_type=norm_type).to(device)
        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            y = layer_norm_linear(x)
        assert y.shape == (20,)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        layer_norm_linear = NormLinear(10, 20, norm_type=norm_type).to(device)
        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
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
    def test_reset_parameters_zeros_biases(self):
        layer = NormMLP(10, 20)
        assert layer.fc1.bias is not None
        assert layer.fc2.bias is not None
        assert torch.count_nonzero(layer.fc1.bias) == 0
        assert torch.count_nonzero(layer.fc2.bias) == 0

    @pytest.mark.parametrize("activation", ["relu", "swiglu", "srelu"])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, activation, norm_type):
        layer_norm_mlp = NormMLP(10, 20, activation=activation, norm_type=norm_type).to(device)
        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
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

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        layer_norm_mlp = NormMLP(10, 20, norm_type=norm_type).to(device)
        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            y = layer_norm_mlp(x)
        y.sum().backward()
        for param in layer_norm_mlp.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_with_explicit_none_matches_default(self, device):
        layer = NormMLP(10, 20, activation="relu", dropout=0.0).to(device)
        layer.eval()
        x = torch.randn(2, 4, 10, device=device)

        y_default = layer(x)
        y_none = layer(x, norm_scale_delta=None, norm_shift=None, output_gate=None)
        assert_close(y_default, y_none)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward_with_modulation_matches_manual_reference(self, device, norm_type):
        layer = NormMLP(10, 20, activation="relu", dropout=0.0, norm_type=norm_type).to(device)
        layer.eval()
        x = torch.randn(2, 4, 10, device=device)
        norm_scale_delta = torch.randn(2, 1, 10, device=device)
        norm_shift = torch.randn(2, 1, 10, device=device)
        output_gate = torch.randn(2, 1, 10, device=device)

        y = layer(x, norm_scale_delta=norm_scale_delta, norm_shift=norm_shift, output_gate=output_gate)

        expected = _apply_norm_manual(x, layer.norm, scale_delta=norm_scale_delta, shift=norm_shift)
        expected = F.linear(expected, layer.fc1.weight, layer.fc1.bias)
        expected = layer.activation(expected)
        expected = F.linear(expected, layer.fc2.weight, layer.fc2.bias)
        expected = expected * output_gate
        assert_close(y, expected)

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


class TestAdaNormMLP:
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_zero_init_outputs_zero(self, device, norm_type):
        layer = AdaNormMLP(10, 20, norm_type=norm_type).to(device)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)

        y = layer(x, conditioning=conditioning)
        assert_close(y, torch.zeros_like(y))

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, activation, norm_type):
        layer = AdaNormMLP(10, 20, activation=activation, norm_type=norm_type).to(device)
        _enable_adaln_gate(layer)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            y = layer(x, conditioning=conditioning)
        assert y.shape == x.shape

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        layer = AdaNormMLP(10, 20, norm_type=norm_type).to(device)
        _enable_adaln_gate(layer)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            y = layer(x, conditioning=conditioning)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_quantization(self, device):
        torch.random.manual_seed(0)
        layer = AdaNormMLP(10, 20).to(device)
        _enable_adaln_gate(layer)
        layer.eval()
        quantized_layer = deepcopy(layer)
        quantized_layer.apply_quantization(Int8WeightOnlyConfig())
        weight1 = quantized_layer.fc1.weight
        weight2 = quantized_layer.fc2.weight
        assert isinstance(weight1, AffineQuantizedTensor)
        assert isinstance(weight2, AffineQuantizedTensor)

        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer(x, conditioning=conditioning)
            y_quant = quantized_layer(x, conditioning=conditioning)
        assert_close(y, y_quant, atol=1e-2, rtol=0)
