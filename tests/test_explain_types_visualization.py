import pytest
import torch
import torch.nn.functional as F
from test_explain_trace import make_tiny_config, output_fn
from torch.testing import assert_close

from vit import ViT
from vit.explain import (
    Explanation,
    ForwardArgs,
    ViTExplainer,
    interpolate_token_attribution,
    normalize_attribution,
)
from vit.explain.types import TokenLayout, select_targets


def test_target_selection_supports_shared_per_example_dense_and_callable_targets() -> None:
    outputs = torch.arange(24).view(2, 3, 4)
    class_outputs = torch.arange(6).view(2, 3)

    assert_close(select_targets(class_outputs, 1), torch.tensor([1, 4]))
    assert_close(select_targets(class_outputs, torch.tensor([0, 2])), torch.tensor([0, 5]))
    assert_close(select_targets(outputs, (1, 2)), torch.tensor([6, 18]))
    assert_close(select_targets(outputs, torch.tensor([[0, 1], [2, 3]])), torch.tensor([1, 23]))
    assert_close(select_targets(class_outputs[:, 0], None), torch.tensor([0, 3]))
    assert_close(select_targets(class_outputs, lambda values: values[:, 2]), torch.tensor([2, 5]))


@pytest.mark.parametrize("target", [None, torch.ones(2, 2, 2, dtype=torch.long), "bad"])
def test_target_selection_rejects_ambiguous_targets(target) -> None:
    outputs = torch.ones(2, 3)
    with pytest.raises((TypeError, ValueError)):
        select_targets(outputs, target)


def test_token_layout_scatter_validates_batch_and_token_dimensions() -> None:
    model = ViT(make_tiny_config()).eval()
    trace = ViTExplainer(model, output_fn).trace(torch.randn(1, 3, 9, 10))

    with pytest.raises(ValueError, match="batch"):
        trace.layout.scatter_visual(torch.ones(2, 4))
    with pytest.raises(ValueError, match="token"):
        trace.layout.scatter_visual(torch.ones(1, 3))


@pytest.mark.parametrize("mode", ["none", "minmax", "symmetric", "absolute"])
def test_normalization_modes_preserve_nan_locations(mode) -> None:
    values = torch.tensor([[[-2.0, float("nan")], [1.0, 2.0]]])
    result = normalize_attribution(values, mode)
    assert torch.isnan(result[0, 0, 1])
    assert torch.isfinite(result[0, 0, 0])


def test_minmax_normalization_ignores_nan_locations() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, float("nan")]])

    result = normalize_attribution(values, "minmax")

    assert_close(result[:, :3], torch.tensor([[0.0, 0.5, 1.0]]))
    assert torch.isnan(result[:, 3]).all()


def test_interpolation_marks_ignored_borders_and_supports_modeled_size() -> None:
    model = ViT(make_tiny_config()).eval()
    trace = ViTExplainer(model, output_fn).trace(torch.randn(1, 3, 9, 10))
    explanation = Explanation("fixed", torch.ones(1, 4), None, torch.zeros(1), trace.layout)

    original = interpolate_token_attribution(explanation, normalization="minmax")
    modeled = interpolate_token_attribution(explanation, size=trace.layout.modeled_size)

    assert original.shape == (1, 9, 10)
    assert torch.isnan(original[:, 8:, :]).all()
    assert torch.isnan(original[:, :, 8:]).all()
    assert modeled.shape == (1, 8, 8)


def test_interpolation_preserves_masked_patches_without_nan_spread() -> None:
    model = ViT(make_tiny_config()).eval()
    inputs = torch.randn(1, 3, 9, 10)
    mask = torch.tensor([[True, False, True, False]])
    trace = ViTExplainer(model, output_fn).trace(inputs, forward_args=ForwardArgs(mask=mask))
    token_attributions = torch.tensor([[1.0, float("nan"), 3.0, float("nan")]])
    explanation = Explanation("fixed", token_attributions, None, torch.zeros(1), trace.layout)

    rendered = interpolate_token_attribution(
        explanation,
        size=trace.layout.modeled_size,
        normalization="minmax",
    )

    patch_height, patch_width = trace.layout.patch_size
    assert torch.isfinite(rendered[:, :, :patch_width]).all()
    assert torch.isnan(rendered[:, :patch_height, patch_width:]).all()
    assert torch.isnan(rendered[:, patch_height:, patch_width:]).all()


def test_interpolation_resizes_directly_to_arbitrary_requested_size() -> None:
    layout = TokenLayout(
        grid_size=(2, 2),
        patch_size=(4, 4),
        original_size=(8, 8),
        modeled_size=(8, 8),
        num_cls_tokens=0,
        num_register_tokens=0,
        visual_indices=torch.arange(4).view(1, 4),
        visual_validity=torch.ones(1, 4, dtype=torch.bool),
    )
    token_attributions = torch.arange(4, dtype=torch.float32).view(1, 4)
    explanation = Explanation("fixed", token_attributions, None, torch.zeros(1), layout)

    rendered = interpolate_token_attribution(explanation, size=(4, 4))
    expected = F.interpolate(
        token_attributions.view(1, 1, 2, 2),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

    assert_close(rendered, expected)


def test_normalization_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="normalization"):
        normalize_attribution(torch.ones(1, 2), "unknown")  # type: ignore[arg-type]
