from collections import OrderedDict
from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from torchao.quantization import FqnToConfig, Int8DynamicActivationIntxWeightConfig, quantize_
from torchao.quantization.qat import FakeQuantizedLinear, QATConfig
from torchao.quantization.qat.linear import disable_linear_fake_quant

from vit.attention import CrossAttention, SelfAttention
from vit.explain.trace import eager_self_attention
from vit.fused import AdaNormMLP, NormLinear, NormMLP


CASES = ("linear", "mlp", "glu", "adaln", "self_attention", "specialized_attention", "cross_attention")
Layer = NormLinear | NormMLP | SelfAttention | CrossAttention


def _qat_config() -> QATConfig:
    return QATConfig(Int8DynamicActivationIntxWeightConfig(weight_dtype=getattr(torch, "int4")))


@pytest.fixture(params=CASES)
def layer_case(
    request: pytest.FixtureRequest, device: torch.device
) -> tuple[Layer, tuple[Tensor, ...], dict[str, Tensor], tuple[str, ...]]:
    torch.manual_seed(0)
    options = {"device": device, "dropout": 0.0}
    x = torch.randn(2, 4, 32, device=device, requires_grad=True)
    match request.param:
        case "linear":
            return NormLinear(32, 32, **options), (x,), {}, ("linear",)
        case "mlp" | "glu":
            activation = "gelu" if request.param == "mlp" else "swiglu"
            return NormMLP(32, 64, activation=activation, **options), (x,), {}, ("fc1", "fc2")
        case "adaln":
            layer = AdaNormMLP(32, 64, activation="swiglu", adaln_gate_init=1.0, **options)
            return layer, (x,), {"conditioning": torch.randn(2, 32, device=device)}, ("fc1", "fc2")
        case "cross_attention":
            layer = CrossAttention(
                32, 4, hidden_dropout=0.0, attention_dropout=0.0, qk_normalization=True, device=device
            )
            return layer, (x, torch.randn(2, 6, 32, device=device)), {}, ("q_proj", "kv_proj", "out_proj")
        case _:
            specialized = request.param == "specialized_attention"
            layer = SelfAttention(
                32,
                4,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                qk_normalization=True,
                num_global_tokens=2 if specialized else 0,
                specialize_norms=specialized,
                specialize_qkv=specialized,
                device=device,
            )
            names = ("qkv_proj", "visual_qkv_proj", "out_proj") if specialized else ("qkv_proj", "out_proj")
            return layer, (x,), {}, names


def test_qat_replaces_projection_modules(layer_case) -> None:
    layer, _, _, names = layer_case
    config = _qat_config()
    if isinstance(layer, (SelfAttention, CrossAttention)):
        layer.apply_quantization(config, config)
    else:
        layer.apply_quantization(config)

    for name in names:
        assert isinstance(layer.get_submodule(name), FakeQuantizedLinear), name


def test_qat_executes_fake_quantizers_and_backpropagates(layer_case) -> None:
    layer, args, kwargs, names = layer_case
    # Install the replacements independently to isolate the functional-forward bug.
    quantize_(layer, FqnToConfig(OrderedDict.fromkeys(names, _qat_config())), filter_fn=None)
    called: set[str] = set()
    handles = []
    for name in names:
        projection = layer.get_submodule(name)
        assert isinstance(projection, FakeQuantizedLinear)
        for kind in ("activation", "weight"):
            quantizer = getattr(projection, f"{kind}_fake_quantizer")
            handles.append(quantizer.register_forward_hook(_record_call(called, f"{name}.{kind}")))

    output = layer(*args, **kwargs)
    output.square().mean().backward()
    assert called == {f"{name}.{kind}" for name in names for kind in ("activation", "weight")}
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    for handle in handles:
        handle.remove()
    layer.apply(disable_linear_fake_quant)
    without_fake_quantization = layer(*args, **kwargs)
    assert not torch.equal(output, without_fake_quantization)


def _record_call(called: set[str], name: str) -> Callable[..., None]:
    def record(*_args: object) -> None:
        called.add(name)

    return record


@pytest.mark.parametrize("layer_case", ["mlp", "glu", "adaln"], indirect=True)
def test_qat_mlp_trace_matches_forward(layer_case) -> None:
    layer, args, kwargs, names = layer_case
    quantize_(layer, FqnToConfig(OrderedDict.fromkeys(names, _qat_config())), filter_fn=None)

    expected = layer(*args, **kwargs)
    traced = layer._forward_with_intermediates(*args, **kwargs)

    torch.testing.assert_close(traced.output, expected)


@pytest.mark.parametrize("layer_case", ["self_attention", "specialized_attention"], indirect=True)
def test_qat_attention_trace_matches_forward(layer_case) -> None:
    layer, args, _, names = layer_case
    quantize_(layer, FqnToConfig(OrderedDict.fromkeys(names, _qat_config())), filter_fn=None)

    expected = layer(*args)
    traced, probabilities, _ = eager_self_attention(layer, args[0], None)

    torch.testing.assert_close(traced, expected)
    torch.testing.assert_close(layer.forward_weights(*args), probabilities)


@pytest.mark.compile
@pytest.mark.parametrize("layer_case", ["mlp", "glu", "self_attention", "cross_attention"], indirect=True)
def test_qat_compiled_forward_and_backward(layer_case) -> None:
    layer, args, kwargs, names = layer_case
    quantize_(layer, FqnToConfig(OrderedDict.fromkeys(names, _qat_config())), filter_fn=None)
    expected = layer(*args, **kwargs)
    compiled = torch.compile(layer, fullgraph=True)

    actual = compiled(*args, **kwargs)
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected)
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
