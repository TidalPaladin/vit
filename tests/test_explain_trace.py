from dataclasses import replace

import pytest
import torch
from torch.testing import assert_close

from vit import ViT, ViTConfig
from vit.explain import ForwardArgs, TraceConfig, ViTExplainer


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

    ViTExplainer(model, output_fn).trace(inputs)

    assert model.training
    assert {name: value.requires_grad for name, value in model.named_parameters()} == original_requires_grad
    assert parameter.grad is not None
    assert_close(parameter.grad, original_gradient)
    assert inputs.grad is None


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


def test_ragged_mask_layout_distinguishes_padding_from_visual_tokens() -> None:
    model = ViT(make_tiny_config()).eval()
    inputs = torch.randn(2, 3, 9, 10)
    mask = torch.tensor([[True, True, True, False], [False, True, False, False]])

    expected = model(inputs, mask=mask)
    trace = ViTExplainer(model, output_fn).trace(inputs, forward_args=ForwardArgs(mask=mask))

    assert_close(trace.features.dense_features, expected.dense_features)
    assert trace.layout.visual_indices.tolist() == [[0, 1, 2], [1, -1, -1]]
    assert trace.layout.sequence_validity.tolist() == [[True, True, True], [True, False, False]]
