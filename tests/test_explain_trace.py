from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from vit import ViT, ViTConfig
from vit.explain import ForwardArgs, MLPTrace, TraceConfig, ViTExplainer
from vit.fused import AdaNormMLP, NormMLP
from vit.norm import apply_norm, get_norm_bias


def make_tiny_config(**overrides) -> ViTConfig:
    config = ViTConfig(
        in_channels=3,
        patch_size=(4, 4),
        img_size=(9, 10),
        depth=2,
        hidden_size=16,
        ffn_hidden_size=32,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        num_cls_tokens=1,
        num_register_tokens=2,
        pos_enc="rope",
        dtype=torch.float32,
    )
    return replace(config, **overrides)


def output_fn(features):
    return features.visual_tokens.mean(dim=1)[..., :3]


def direct_glu_internals(mlp: NormMLP, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    normalized_input = apply_norm(
        inputs,
        mlp.norm.weight,
        get_norm_bias(mlp.norm),
        mlp.norm.eps or 1e-5,
        use_layer_norm=mlp._use_layer_norm,
    )
    fc1_output = F.linear(normalized_input, mlp.fc1.weight, mlp.fc1.bias)
    linear_branch, gate_branch = fc1_output.chunk(2, dim=-1)
    if mlp.limit is not None:
        linear_branch = linear_branch.clamp(min=-mlp.limit, max=mlp.limit)
        gate_branch = gate_branch.clamp(min=None, max=mlp.limit)
    if mlp.extra_bias is not None:
        linear_branch = linear_branch + mlp.extra_bias
    activation_output = mlp.activation(gate_branch)
    hidden = F.dropout(activation_output * linear_branch, p=mlp.dropout.p, training=mlp.training)
    output = F.linear(hidden, mlp.fc2.weight, mlp.fc2.bias)
    output = F.dropout(output, p=mlp.dropout.p, training=mlp.training, inplace=True)
    intermediates = {
        "normalized_input": normalized_input,
        "fc1_output": fc1_output,
        "linear_branch": linear_branch,
        "gate_branch": gate_branch,
        "activation_output": activation_output,
        "hidden": hidden,
        "output": output,
    }
    for tensor in intermediates.values():
        tensor.retain_grad()
    return intermediates


def test_prefix_length_counts_cls_and_register_tokens() -> None:
    model = ViT(make_tiny_config())
    assert model.prefix_length == 3


def test_trace_matches_eval_forward_and_preserves_full_visual_layout() -> None:
    torch.manual_seed(7)
    model = ViT(make_tiny_config()).eval()
    inputs = torch.randn(2, 3, 9, 10)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
    forward_args = ForwardArgs(mask=mask, rope_seed=13)
    explainer = ViTExplainer(model, output_fn)

    expected = model(inputs, mask=mask, rope_seed=13)
    trace = explainer.trace(inputs, config=TraceConfig(retain_gradients=True), forward_args=forward_args)

    assert_close(trace.features.dense_features, expected.dense_features)
    assert trace.layout.grid_size == (2, 2)
    assert trace.layout.modeled_size == (8, 8)
    assert trace.layout.original_size == (9, 10)
    assert_close(trace.layout.visual_validity, mask)
    assert len(trace.layers) == model.config.depth
    assert trace.layers[0].attention_probabilities.shape == (2, 2, 5, 5)
    assert trace.layers[0].attention_probabilities.requires_grad


def test_trace_restores_model_and_existing_gradients() -> None:
    model = ViT(make_tiny_config()).train()
    inputs = torch.randn(1, 3, 9, 10, requires_grad=True)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 2.0)
    original_gradient = parameter.grad.clone()
    original_requires_grad = {name: value.requires_grad for name, value in model.named_parameters()}
    mlp = model.get_block(0).mlp
    hook = mlp.register_forward_hook(lambda *_args: None)
    original_hooks = tuple(mlp._forward_hooks.items())

    try:
        ViTExplainer(model, output_fn).trace(inputs, config=TraceConfig(mlp_internals=True))

        assert model.training
        assert {name: value.requires_grad for name, value in model.named_parameters()} == original_requires_grad
        assert parameter.grad is not None
        assert_close(parameter.grad, original_gradient)
        assert tuple(mlp._forward_hooks.items()) == original_hooks
        assert inputs.grad is None
    finally:
        hook.remove()


def test_explainability_rejects_3d_inputs() -> None:
    config = make_tiny_config(patch_size=(2, 4, 4), img_size=(4, 8, 8))
    model = ViT(config)
    inputs = torch.randn(1, 3, 4, 8, 8)

    with pytest.raises(ValueError, match="supports 2D ViT inputs"):
        ViTExplainer(model, output_fn).trace(inputs)


@pytest.mark.parametrize(
    ("pos_enc", "norm_type", "dtype", "masked", "conditioned"),
    [
        ("none", "rmsnorm", torch.float32, False, False),
        ("fourier", "layernorm", torch.float32, True, False),
        ("learnable", "rmsnorm", torch.bfloat16, False, True),
        ("rope", "layernorm", torch.bfloat16, True, True),
    ],
)
def test_trace_forward_parity_across_model_variants(pos_enc, norm_type, dtype, masked, conditioned) -> None:
    torch.manual_seed(11)
    config = make_tiny_config(
        pos_enc=pos_enc,
        norm_type=norm_type,
        dtype=dtype,
        conditioning_size=5 if conditioned else None,
        adaln_gate_init=1.0 if conditioned else 0.0,
    )
    model = ViT(config).eval()
    inputs = torch.randn(2, 3, 9, 10, dtype=dtype)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]]) if masked else None
    conditioning = torch.randn(2, 5, dtype=dtype) if conditioned else None
    forward_args = ForwardArgs(mask=mask, rope_seed=19, output_norm=False, conditioning=conditioning)

    expected = model(
        inputs,
        mask=mask,
        rope_seed=19,
        output_norm=False,
        conditioning=conditioning,
    )
    actual = ViTExplainer(model, output_fn).trace(inputs, forward_args=forward_args).features

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert_close(actual.dense_features, expected.dense_features, atol=tolerance, rtol=tolerance)


@pytest.mark.cuda
def test_trace_forward_parity_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = ViT(make_tiny_config(dtype=torch.bfloat16), device=torch.device("cuda")).eval()
    inputs = torch.randn(1, 3, 9, 10, device="cuda", dtype=torch.bfloat16)
    expected = model(inputs)
    actual = ViTExplainer(model, output_fn).trace(inputs).features
    assert_close(actual.dense_features, expected.dense_features, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("activation", ["swiglu", "srelu", "gelu", "relu"])
def test_trace_captures_selected_mlp_internals(activation: str) -> None:
    torch.manual_seed(23)
    glu_options = {"glu_limit": 0.75, "glu_extra_bias": 0.2} if activation == "swiglu" else {}
    model = ViT(make_tiny_config(activation=activation, **glu_options)).eval()
    inputs = torch.randn(2, 3, 9, 10)

    expected = model(inputs)
    trace = ViTExplainer(model, output_fn).trace(
        inputs,
        config=TraceConfig(layers=(1,), mlp_internals=True),
    )

    assert_close(trace.features.dense_features, expected.dense_features)
    assert len(trace.layers) == 1
    layer_trace = trace.layers[0]
    mlp_trace = layer_trace.mlp
    assert isinstance(mlp_trace, MLPTrace)
    assert mlp_trace.output is layer_trace.mlp_output

    mlp = model.get_block(1).mlp
    expected_fc1 = F.linear(mlp_trace.normalized_input, mlp.fc1.weight, mlp.fc1.bias)
    assert_close(mlp_trace.fc1_output, expected_fc1)
    if activation == "swiglu":
        assert mlp.limit is not None
        assert mlp.extra_bias is not None
        expected_linear, expected_gate = expected_fc1.chunk(2, dim=-1)
        expected_linear = expected_linear.clamp(min=-mlp.limit, max=mlp.limit)
        expected_gate = expected_gate.clamp(min=None, max=mlp.limit)
        expected_linear = expected_linear + mlp.extra_bias
        assert_close(mlp_trace.linear_branch, expected_linear)
        assert_close(mlp_trace.gate_branch, expected_gate)
        assert_close(mlp_trace.activation_output, mlp.activation(expected_gate))
        assert_close(mlp_trace.hidden, mlp_trace.activation_output * expected_linear)
    else:
        assert mlp_trace.linear_branch is None
        assert mlp_trace.gate_branch is None
        assert_close(mlp_trace.activation_output, mlp.activation(expected_fc1))
        assert_close(mlp_trace.hidden, mlp_trace.activation_output)


def test_trace_captures_conditioned_mlp_internals() -> None:
    torch.manual_seed(29)
    model = ViT(
        make_tiny_config(
            activation="swiglu",
            conditioning_size=5,
            adaln_gate_init=1.0,
            glu_limit=0.8,
            glu_extra_bias=0.1,
        )
    ).eval()
    mlp = model.get_block(0).mlp
    assert isinstance(mlp, AdaNormMLP)
    with torch.no_grad():
        mlp.modulation.weight.normal_(std=0.05)
        mlp.modulation.bias.normal_(std=0.05)
    inputs = torch.randn(2, 3, 9, 10)
    conditioning = torch.randn(2, 5)
    forward_args = ForwardArgs(conditioning=conditioning)

    expected = model(inputs, conditioning=conditioning)
    trace = ViTExplainer(model, output_fn).trace(
        inputs,
        config=TraceConfig(layers=(0,), mlp_internals=True),
        forward_args=forward_args,
    )

    assert_close(trace.features.dense_features, expected.dense_features)
    assert trace.layers[0].mlp is not None
    assert trace.layers[0].mlp.output is trace.layers[0].mlp_output


def test_trace_retains_mlp_internal_gradients_matching_direct_eager_forward() -> None:
    torch.manual_seed(31)
    model = ViT(
        make_tiny_config(
            depth=1,
            activation="swiglu",
            glu_limit=0.9,
            glu_extra_bias=0.15,
        )
    ).eval()
    mlp = model.get_block(0).mlp
    with torch.no_grad():
        mlp.fc2.weight.normal_(std=0.1)
    inputs = torch.randn(1, 3, 9, 10, requires_grad=True)
    trace = ViTExplainer(model, output_fn).trace(
        inputs,
        config=TraceConfig(mlp_internals=True, retain_gradients=True),
    )
    mlp_trace = trace.layers[0].mlp
    assert mlp_trace is not None

    trace.features.dense_features.square().mean().backward()

    traced_tensors = {
        "normalized_input": mlp_trace.normalized_input,
        "fc1_output": mlp_trace.fc1_output,
        "linear_branch": mlp_trace.linear_branch,
        "gate_branch": mlp_trace.gate_branch,
        "activation_output": mlp_trace.activation_output,
        "hidden": mlp_trace.hidden,
        "output": mlp_trace.output,
    }
    assert all(tensor is not None and tensor.grad is not None for tensor in traced_tensors.values())
    assert mlp_trace.output.grad is not None

    reference_input = trace.layers[0].residual_post_attention.detach().clone().requires_grad_(True)
    reference_tensors = direct_glu_internals(mlp, reference_input)
    reference_tensors["output"].backward(mlp_trace.output.grad.detach())

    for name, traced_tensor in traced_tensors.items():
        assert traced_tensor is not None
        assert traced_tensor.grad is not None
        assert reference_tensors[name].grad is not None
        assert_close(traced_tensor.grad, reference_tensors[name].grad)


def test_default_trace_does_not_capture_mlp_internals(mocker) -> None:
    model = ViT(make_tiny_config()).eval()
    inputs = torch.randn(1, 3, 9, 10)
    eager_forward = mocker.patch("vit.explain.trace._forward_mlp_with_intermediates")

    trace = ViTExplainer(model, output_fn).trace(inputs)

    assert all(layer.mlp is None for layer in trace.layers)
    eager_forward.assert_not_called()


def test_ragged_mask_layout_distinguishes_padding_from_visual_tokens() -> None:
    model = ViT(make_tiny_config()).eval()
    inputs = torch.randn(2, 3, 9, 10)
    mask = torch.tensor([[True, True, True, False], [False, True, False, False]])

    expected = model(inputs, mask=mask)
    trace = ViTExplainer(model, output_fn).trace(inputs, forward_args=ForwardArgs(mask=mask))

    assert_close(trace.features.dense_features, expected.dense_features)
    assert trace.layout.visual_indices.tolist() == [[0, 1, 2], [1, -1, -1]]
    assert trace.layout.sequence_validity.tolist() == [[True, True, True], [True, False, False]]
