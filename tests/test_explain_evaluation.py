import pytest
import torch
from torch.testing import assert_close

import vit.explain.evaluation as evaluation_module
from vit import ViT, ViTConfig
from vit.explain import (
    Completeness,
    DeletionInsertion,
    Explanation,
    ForwardArgs,
    Infidelity,
    InputXGradient,
    Localization,
    ParameterRandomizationSanity,
    SaCo,
    Sensitivity,
    ViTExplainer,
)
from vit.explain.evaluation import saco_score


def make_explainer() -> tuple[ViTExplainer, torch.Tensor]:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=0,
        hidden_size=4,
        ffn_hidden_size=8,
        num_attention_heads=1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        norm_type="layernorm",
        dtype=torch.float32,
    )
    model = ViT(config).eval()
    inputs = torch.randn(2, 1, 4, 4)
    return ViTExplainer(model, lambda features: features.visual_tokens.sum((1, 2))), inputs


def test_saco_pairwise_concordance_has_analytical_extremes() -> None:
    importance = torch.tensor([[3.0, 2.0, 1.0]])
    assert_close(saco_score(importance, torch.tensor([[3.0, 2.0, 1.0]])), torch.ones(1))
    assert_close(saco_score(importance, torch.tensor([[1.0, 2.0, 3.0]])), -torch.ones(1))


def test_localization_reports_pointing_and_relevance_mass() -> None:
    explainer, inputs = make_explainer()
    trace = explainer.trace(inputs)
    attribution = torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    explanation = Explanation("fixed", attribution, None, torch.zeros(2), trace.layout)
    region = torch.tensor([[False, False, True, False], [False, False, False, True]])

    result = Localization(region).evaluate(explainer, inputs, explanation, None, trace.forward_args)

    assert_close(result.values, torch.ones(2, 2))


def test_completeness_is_exact_for_known_attribution_sum() -> None:
    explainer, inputs = make_explainer()
    baseline = torch.zeros_like(inputs)
    trace = explainer.trace(inputs)
    input_score = explainer.output_fn(explainer.model(inputs))
    baseline_score = explainer.output_fn(explainer.model(baseline))
    difference = input_score - baseline_score
    attribution = difference[:, None].expand(-1, 4) / 4
    explanation = Explanation("fixed", attribution, None, input_score, trace.layout)

    result = Completeness(baseline=baseline).evaluate(explainer, inputs, explanation, None, trace.forward_args)

    assert_close(result.values, torch.zeros_like(difference), atol=1e-6, rtol=0)


def test_completeness_excludes_pixels_from_masked_patches() -> None:
    explainer, inputs = make_explainer()
    inputs = inputs[:1]
    baseline = torch.zeros_like(inputs)
    mask = torch.tensor([[True, False, True, False]])
    forward_args = ForwardArgs(mask=mask)
    trace = explainer.trace(inputs, forward_args=forward_args)
    difference = explainer.output_fn(explainer.model(inputs, mask=mask)) - explainer.output_fn(
        explainer.model(baseline, mask=mask)
    )
    pixel_attributions = torch.full_like(inputs, 100.0)
    pixel_attributions[..., :, :2] = difference.view(1, 1, 1, 1) / 8
    explanation = Explanation(
        "fixed",
        torch.tensor([[0.0, float("nan"), 0.0, float("nan")]]),
        pixel_attributions,
        difference,
        trace.layout,
    )

    result = Completeness(baseline=baseline).evaluate(explainer, inputs, explanation, None, forward_args)

    assert_close(result.values, torch.zeros_like(difference), atol=1e-6, rtol=0)


def test_evaluate_collects_named_metrics() -> None:
    explainer, inputs = make_explainer()
    trace = explainer.trace(inputs)
    explanation = Explanation("fixed", torch.ones(2, 4), None, torch.zeros(2), trace.layout)
    region = torch.ones(2, 4, dtype=torch.bool)

    report = explainer.evaluate(inputs, explanation, metrics=[Localization(region), SaCo(groups=2)])

    assert set(report.metrics) == {"localization", "saco"}


def test_deletion_insertion_returns_both_curves_and_auc() -> None:
    explainer, inputs = make_explainer()
    trace = explainer.trace(inputs)
    explanation = Explanation("fixed", torch.arange(8).view(2, 4).float(), None, torch.zeros(2), trace.layout)

    result = DeletionInsertion(steps=2).evaluate(explainer, inputs, explanation, None, trace.forward_args)

    assert result.values.shape == (2, 2, 3)
    assert set(result.metadata) == {"deletion_auc", "insertion_auc"}


def test_infidelity_supports_pixel_and_token_attributions() -> None:
    explainer, inputs = make_explainer()
    trace = explainer.trace(inputs)
    token_values = torch.ones(2, 4)
    pixel_values = torch.ones_like(inputs)
    token_explanation = Explanation("fixed", token_values, None, torch.zeros(2), trace.layout)
    pixel_explanation = Explanation("fixed", token_values, pixel_values, torch.zeros(2), trace.layout)
    metric = Infidelity(samples=2, noise_scale=0.001)

    token_result = metric.evaluate(explainer, inputs, token_explanation, None, trace.forward_args)
    pixel_result = metric.evaluate(explainer, inputs, pixel_explanation, None, trace.forward_args)

    assert torch.isfinite(token_result.values).all()
    assert torch.isfinite(pixel_result.values).all()


def test_masked_metrics_use_only_valid_visual_tokens(mocker) -> None:
    explainer, inputs = make_explainer()
    inputs = inputs[:1]
    mask = torch.tensor([[True, False, True, False]])
    forward_args = ForwardArgs(mask=mask, output_norm=False)
    trace = explainer.trace(inputs, forward_args=forward_args)
    explanation = Explanation(
        "fixed",
        torch.tensor([[2.0, float("nan"), 1.0, float("nan")]]),
        None,
        torch.zeros(1),
        trace.layout,
    )
    replaced_indices: set[int] = set()
    original_replace = evaluation_module._replace_patches

    def record_replacement(*args, **kwargs) -> None:
        patch_indices = args[2]
        replaced_indices.update(int(index) for index in patch_indices.tolist())
        original_replace(*args, **kwargs)

    mocker.patch.object(evaluation_module, "_replace_patches", side_effect=record_replacement)

    deletion = DeletionInsertion(steps=2).evaluate(explainer, inputs, explanation, None, forward_args)
    saco = SaCo(groups=2).evaluate(explainer, inputs, explanation, None, forward_args)
    infidelity = Infidelity(samples=2).evaluate(explainer, inputs, explanation, None, forward_args)
    sensitivity = Sensitivity(InputXGradient(), samples=2).evaluate(
        explainer,
        inputs,
        explanation,
        None,
        forward_args,
    )

    assert replaced_indices == {0, 2}
    for result in (deletion, saco, infidelity, sensitivity):
        assert torch.isfinite(result.values).all()
    assert torch.isfinite(saco.metadata["group_importance"]).all()


def test_sensitivity_and_parameter_randomization_restore_model() -> None:
    explainer, inputs = make_explainer()
    method = InputXGradient()
    explanation = explainer.attribute(inputs, method=method)
    state = {name: value.clone() for name, value in explainer.model.state_dict().items()}
    forward_args = explainer.trace(inputs).forward_args

    sensitivity = Sensitivity(method, samples=2).evaluate(explainer, inputs, explanation, None, forward_args)
    sanity = ParameterRandomizationSanity(method).evaluate(explainer, inputs, explanation, None, forward_args)

    assert torch.isfinite(sensitivity.values).all()
    assert torch.isfinite(sanity.values).all()
    assert (sanity.values < 0.999).all()
    for name, value in explainer.model.state_dict().items():
        assert_close(value, state[name])


def test_evaluate_rejects_duplicate_metric_names() -> None:
    explainer, inputs = make_explainer()
    trace = explainer.trace(inputs)
    explanation = Explanation("fixed", torch.ones(2, 4), None, torch.zeros(2), trace.layout)
    region = torch.ones(2, 4, dtype=torch.bool)

    try:
        explainer.evaluate(inputs, explanation, metrics=[Localization(region), Localization(region)])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate metric names must fail")


def test_evaluate_rejects_inputs_with_different_spatial_layout() -> None:
    explainer, inputs = make_explainer()
    trace = explainer.trace(inputs)
    explanation = Explanation("fixed", torch.ones(2, 4), None, torch.zeros(2), trace.layout)

    with pytest.raises(ValueError, match="token layout"):
        explainer.evaluate(
            torch.randn(2, 1, 6, 6),
            explanation,
            metrics=[DeletionInsertion(steps=1)],
        )


def test_evaluate_rejects_forward_mask_that_differs_from_explanation() -> None:
    explainer, inputs = make_explainer()
    inputs = inputs[:1]
    explanation_mask = torch.tensor([[True, False, True, False]])
    trace = explainer.trace(inputs, forward_args=ForwardArgs(mask=explanation_mask))
    explanation = Explanation(
        "fixed",
        torch.tensor([[1.0, float("nan"), 2.0, float("nan")]]),
        None,
        torch.zeros(1),
        trace.layout,
    )

    with pytest.raises(ValueError, match="token layout"):
        explainer.evaluate(inputs, explanation, metrics=[DeletionInsertion(steps=1)])
