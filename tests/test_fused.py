import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close
from torchao.quantization import Int8Tensor, Int8WeightOnlyConfig

from vit.fused import (
    MIN_GLU_AUTOTUNE_HIDDEN_SIZE,
    VALID_ADALN_GATE_INITS,
    AdaNormMLP,
    NormLinear,
    NormMLP,
)


TORCHAO_QUANTIZATION_CONFIG_VERSION = 2
COMPILE_TEST_TIMEOUT_SECONDS = 120
MLP_COMPILE_CASES = (
    ("srelu", "grad"),
    ("srelu", "no_grad"),
    ("swiglu", "grad"),
    ("swiglu", "no_grad"),
)


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
        quantized_layer_norm_linear.apply_quantization(
            Int8WeightOnlyConfig(version=TORCHAO_QUANTIZATION_CONFIG_VERSION)
        )
        weight = quantized_layer_norm_linear.linear.weight
        assert isinstance(weight, Int8Tensor)

        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer_norm_linear(x)
            y_quant = quantized_layer_norm_linear(x)
        # TorchAO v2 accumulates directly quantized linear outputs in float32.
        assert_close(y.float(), y_quant.float(), atol=1e-2, rtol=0)


class TestNormMLP:
    @pytest.mark.compile
    @pytest.mark.parametrize(("activation", "gradient_mode"), MLP_COMPILE_CASES)
    def test_compiles_across_dynamic_flat_token_counts(self, activation, gradient_mode):
        environment = os.environ.copy()
        environment.pop("TORCHDYNAMO_DISABLE", None)
        check_script = Path(__file__).with_name("norm_mlp_dynamic_shapes_compile_check.py")

        subprocess.run(
            [sys.executable, str(check_script), activation, gradient_mode],
            check=True,
            env=environment,
            timeout=COMPILE_TEST_TIMEOUT_SECONDS,
        )

    def test_reset_parameters_zeros_biases(self):
        layer = NormMLP(10, 20)
        assert layer.fc1.bias is not None
        assert layer.fc2.bias is not None
        assert torch.count_nonzero(layer.fc1.bias) == 0
        assert torch.count_nonzero(layer.fc2.bias) == 0

    def test_glu_max_autotune_gemm_requires_glu_activation(self):
        with pytest.raises(ValueError, match="glu_max_autotune_gemm requires a GLU activation"):
            NormMLP(10, 20, activation="srelu", glu_max_autotune_gemm=True)

    def test_glu_max_autotune_gemm_rejects_quantization(self):
        layer = NormMLP(10, 20, activation="swiglu", glu_max_autotune_gemm=True)

        with pytest.raises(ValueError, match="glu_max_autotune_gemm is not supported with quantization"):
            layer.apply_quantization(Int8WeightOnlyConfig(version=TORCHAO_QUANTIZATION_CONFIG_VERSION))

    def test_glu_max_autotune_gemm_falls_back_on_cpu(self, mocker):
        layer = NormMLP(
            MIN_GLU_AUTOTUNE_HIDDEN_SIZE,
            2 * MIN_GLU_AUTOTUNE_HIDDEN_SIZE,
            activation="swiglu",
            glu_max_autotune_gemm=True,
        )
        x = torch.randn(1, 2, MIN_GLU_AUTOTUNE_HIDDEN_SIZE)
        default_output = torch.ones_like(x)
        default_glu = mocker.patch("vit.fused.norm_mlp_glu", return_value=default_output)
        autotuned_glu = mocker.patch("vit.fused.norm_mlp_glu_max_autotune_gemm")

        output = layer(x)

        assert output is default_output
        default_glu.assert_called_once()
        autotuned_glu.assert_not_called()

    @pytest.mark.cuda
    def test_glu_max_autotune_gemm_falls_back_below_hidden_threshold(self, mocker):
        hidden_size = MIN_GLU_AUTOTUNE_HIDDEN_SIZE - 1
        layer = NormMLP(hidden_size, 2 * hidden_size, activation="swiglu", glu_max_autotune_gemm=True)
        x = torch.randn(1, 2, hidden_size, device="cuda")
        default_output = torch.ones_like(x)
        default_glu = mocker.patch("vit.fused.norm_mlp_glu", return_value=default_output)
        autotuned_glu = mocker.patch("vit.fused.norm_mlp_glu_max_autotune_gemm")

        output = layer(x)

        assert output is default_output
        default_glu.assert_called_once()
        autotuned_glu.assert_not_called()

    @pytest.mark.cuda
    def test_glu_max_autotune_gemm_selects_cuda_path_at_hidden_threshold(self, mocker):
        layer = NormMLP(
            MIN_GLU_AUTOTUNE_HIDDEN_SIZE,
            2 * MIN_GLU_AUTOTUNE_HIDDEN_SIZE,
            activation="swiglu",
            glu_max_autotune_gemm=True,
        )
        x = torch.randn(1, 2, MIN_GLU_AUTOTUNE_HIDDEN_SIZE, device="cuda")
        autotuned_output = torch.ones_like(x)
        default_glu = mocker.patch("vit.fused.norm_mlp_glu")
        autotuned_glu = mocker.patch(
            "vit.fused.norm_mlp_glu_max_autotune_gemm",
            return_value=autotuned_output,
        )

        output = layer(x)

        assert output is autotuned_output
        default_glu.assert_not_called()
        autotuned_glu.assert_called_once()

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

    def test_eager_intermediates_follow_dropout_mode(self, device):
        torch.random.manual_seed(0)
        layer = NormMLP(16, 64, activation="swiglu", dropout=0.5).to(device)
        x = torch.randn(8, 4, 16, device=device)

        layer.eval()
        eval_first = layer._forward_with_intermediates(x)
        eval_second = layer._forward_with_intermediates(x)
        assert_close(eval_first.output, layer(x))
        assert_close(eval_first.hidden, eval_second.hidden)
        assert_close(eval_first.output, eval_second.output)

        layer.train()
        train_first = layer._forward_with_intermediates(x)
        train_second = layer._forward_with_intermediates(x)
        assert not torch.allclose(train_first.hidden, train_second.hidden)
        assert not torch.allclose(train_first.output, train_second.output)
        assert layer.training

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
        quantized_layer_norm_mlp.apply_quantization(Int8WeightOnlyConfig(version=TORCHAO_QUANTIZATION_CONFIG_VERSION))
        weight1 = quantized_layer_norm_mlp.fc1.weight
        weight2 = quantized_layer_norm_mlp.fc2.weight
        assert isinstance(weight1, Int8Tensor)
        assert isinstance(weight2, Int8Tensor)

        x = torch.randn(10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer_norm_mlp(x)
            y_quant = quantized_layer_norm_mlp(x)
        # TorchAO v2 accumulates directly quantized linear outputs in float32.
        assert_close(y.float(), y_quant.float(), atol=1e-2, rtol=0)


class TestAdaNormMLP:
    def test_invalid_adaln_gate_init_raises(self):
        with pytest.raises(ValueError, match="adaln_gate_init must be one of"):
            AdaNormMLP(10, 20, adaln_gate_init=0.5)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_zero_init_outputs_zero(self, device, norm_type):
        layer = AdaNormMLP(10, 20, norm_type=norm_type).to(device)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)

        y = layer(x, conditioning=conditioning)
        assert_close(y, torch.zeros_like(y))

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_gate_init_one_matches_unconditioned_mlp_after_weight_load(self, device, norm_type):
        base = NormMLP(10, 20, norm_type=norm_type, dropout=0.0).to(device)
        conditioned = AdaNormMLP(10, 20, norm_type=norm_type, dropout=0.0, adaln_gate_init=1.0).to(device)
        conditioned.load_state_dict(base.state_dict(), strict=False)

        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)

        assert conditioned.modulation.bias is not None
        hidden_size = conditioned.fc2.out_features
        expected_gate = torch.full((hidden_size,), VALID_ADALN_GATE_INITS[1], device=device)
        assert_close(conditioned.modulation.bias[2 * hidden_size :], expected_gate)
        assert_close(conditioned(x, conditioning=conditioning), base(x))

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, activation, norm_type):
        layer = AdaNormMLP(10, 20, activation=activation, norm_type=norm_type).to(device)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            y = layer(x, conditioning=conditioning)
        assert y.shape == x.shape

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        layer = AdaNormMLP(10, 20, norm_type=norm_type).to(device)
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
        layer.eval()
        quantized_layer = deepcopy(layer)
        quantized_layer.apply_quantization(Int8WeightOnlyConfig(version=TORCHAO_QUANTIZATION_CONFIG_VERSION))
        weight1 = quantized_layer.fc1.weight
        weight2 = quantized_layer.fc2.weight
        assert isinstance(weight1, Int8Tensor)
        assert isinstance(weight2, Int8Tensor)

        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer(x, conditioning=conditioning)
            y_quant = quantized_layer(x, conditioning=conditioning)
        # TorchAO v2 accumulates directly quantized linear outputs in float32.
        assert_close(y.float(), y_quant.float(), atol=1e-2, rtol=0)


@pytest.mark.compile
@pytest.mark.cuda
@pytest.mark.parametrize("conditioned", [False, True], ids=["norm_mlp", "ada_norm_mlp"])
def test_glu_max_autotune_gemm_matches_default_cuda_forward_and_backward(conditioned):
    torch.manual_seed(0)
    hidden_size = MIN_GLU_AUTOTUNE_HIDDEN_SIZE
    ffn_hidden_size = 2 * hidden_size
    common_kwargs = {
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "activation": "swiglu",
        "dropout": 0.0,
        "device": torch.device("cuda"),
        "dtype": torch.bfloat16,
    }
    if conditioned:
        baseline = AdaNormMLP(**common_kwargs, conditioning_size=hidden_size, adaln_gate_init=1.0)
        candidate = AdaNormMLP(
            **common_kwargs,
            conditioning_size=hidden_size,
            adaln_gate_init=1.0,
            glu_max_autotune_gemm=True,
        )
    else:
        baseline = NormMLP(**common_kwargs)
        candidate = NormMLP(**common_kwargs, glu_max_autotune_gemm=True)
    candidate.load_state_dict(baseline.state_dict())

    baseline_input = torch.randn(2, 16, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_input = baseline_input.detach().clone().requires_grad_(True)
    conditioning = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16) if conditioned else None

    if conditioned:
        assert isinstance(baseline, AdaNormMLP)
        assert isinstance(candidate, AdaNormMLP)
        baseline_output = baseline(baseline_input, conditioning=conditioning)
        candidate_output = candidate(candidate_input, conditioning=conditioning)
    else:
        baseline_output = baseline(baseline_input)
        candidate_output = candidate(candidate_input)

    baseline_output.float().square().mean().backward()
    candidate_output.float().square().mean().backward()

    assert_close(candidate_output, baseline_output, rtol=1e-2, atol=1e-2)
    assert baseline_input.grad is not None
    assert candidate_input.grad is not None
    assert_close(candidate_input.grad, baseline_input.grad, rtol=1e-2, atol=1e-2)
    baseline_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert candidate_parameters.keys() == baseline_parameters.keys()
    for name, baseline_parameter in baseline_parameters.items():
        candidate_parameter = candidate_parameters[name]
        assert baseline_parameter.grad is not None
        assert candidate_parameter.grad is not None
        assert_close(candidate_parameter.grad, baseline_parameter.grad, rtol=1e-2, atol=1e-2)
