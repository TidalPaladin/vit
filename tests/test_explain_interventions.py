import pytest
import torch
from torch.testing import assert_close

from vit import ViT, ViTConfig
from vit.explain import ForwardArgs, Intervention, ViTExplainer, interventions as intervention_module


def make_model(image_size=(4, 4)) -> ViT:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=image_size,
        depth=2,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        pos_enc="learnable",
        dtype=torch.float32,
    )
    model = ViT(config).eval()
    with torch.no_grad():
        for index in range(model.config.depth):
            block = model.get_block(index)
            block.self_attention.out_proj.weight.normal_(std=0.2)
            block.mlp.fc2.weight.normal_(std=0.2)
    return model


def output_fn(features):
    return features.visual_tokens.mean(1)[:, :2]


def test_noop_constant_intervention_is_identity() -> None:
    model = make_model()
    inputs = torch.randn(2, 1, 4, 4)
    explainer = ViTExplainer(model, output_fn)
    trace = explainer.trace(inputs)
    value = trace.layers[0].residual_pre.detach()

    result = explainer.intervene(
        inputs,
        target=0,
        interventions=[Intervention(site="residual_pre", layer=0, mode="constant", value=value)],
    )

    assert_close(result.intervened_scores, result.baseline_scores)
    assert_close(result.absolute_change, torch.zeros_like(result.absolute_change))


def test_head_ablation_changes_only_selected_execution() -> None:
    torch.manual_seed(3)
    model = make_model()
    inputs = torch.randn(2, 1, 4, 4)
    explainer = ViTExplainer(model, output_fn)

    result = explainer.intervene(
        inputs,
        target=1,
        interventions=[Intervention(site="head_output", layer=0, heads=[0], mode="zero")],
    )

    assert torch.isfinite(result.absolute_change).all()
    assert (result.absolute_change.abs() > 0).any()


def test_reference_patching_reproduces_reference_score_when_replacing_first_residual() -> None:
    model = make_model()
    clean = torch.randn(2, 1, 4, 4)
    reference = torch.randn(2, 1, 4, 4)
    explainer = ViTExplainer(model, output_fn)

    result = explainer.intervene(
        clean,
        target=0,
        interventions=[Intervention(site="residual_pre", layer=0, mode="reference")],
        reference_inputs=reference,
    )
    reference_scores = output_fn(model(reference))[:, 0]

    assert_close(result.intervened_scores, reference_scores)


def test_reference_patching_rejects_incompatible_layout() -> None:
    model = make_model()
    clean = torch.randn(1, 1, 4, 4)
    reference = torch.randn(1, 1, 5, 4)

    with pytest.raises(ValueError, match="matching token layouts"):
        ViTExplainer(model, output_fn).intervene(
            clean,
            target=0,
            interventions=[Intervention(site="residual_pre", layer=0, mode="reference")],
            reference_inputs=reference,
        )


def test_activation_atlas_preserves_ids_and_top_patch_coordinates() -> None:
    model = make_model()
    batches = [
        (torch.zeros(1, 1, 4, 4), ["zero"]),
        (torch.ones(1, 1, 4, 4), ["one"]),
    ]

    atlas = ViTExplainer(model, output_fn).scan_activations(
        batches,
        site="residual_pre",
        layer=0,
        top_k=1,
    )

    assert atlas.channels
    record = atlas.channels[0][0]
    assert record.sample_id in {"zero", "one"}
    assert len(record.patch_coordinate) == 2


def test_activation_atlas_creates_each_patch_thumbnail_at_most_once() -> None:
    model = make_model()
    calls: list[tuple[int, int]] = []

    atlas = ViTExplainer(model, output_fn).scan_activations(
        [(torch.randn(1, 1, 4, 4), ["sample"])],
        site="residual_pre",
        layer=0,
        top_k=4,
        thumbnail=lambda _image, coordinate: calls.append(coordinate) or coordinate,
    )

    assert atlas.channels
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("site", ["residual_pre", "post_attention", "mlp_output", "residual_post"])
def test_token_channel_interventions_cover_each_residual_site(site) -> None:
    model = make_model()
    inputs = torch.randn(1, 1, 4, 4)

    result = ViTExplainer(model, output_fn).intervene(
        inputs,
        target=0,
        interventions=[Intervention(site=site, layer=0, tokens=[0, 1], channels=[0, 1], mode="zero")],
    )

    assert torch.isfinite(result.intervened_scores).all()


def test_user_supplied_mean_intervention() -> None:
    model = make_model()
    inputs = torch.randn(1, 1, 4, 4)
    channel_mean = torch.arange(8, dtype=inputs.dtype)

    result = ViTExplainer(model, output_fn).intervene(
        inputs,
        target=0,
        interventions=[Intervention(site="residual_pre", layer=0, mode="mean", value=channel_mean)],
    )

    assert torch.isfinite(result.intervened_scores).all()


def test_sweep_batches_interventions_and_reuses_baseline(mocker) -> None:
    model = make_model()
    inputs = torch.randn(2, 1, 4, 4)
    mask = torch.tensor([[True, True, False, True], [False, True, True, True]])
    target = torch.tensor([0, 1])
    requested = (
        Intervention(site="residual_pre", layer=0, tokens=[0], mode="zero"),
        Intervention(site="head_output", layer=0, heads=[0], mode="zero"),
        Intervention(site="mlp_output", layer=1, channels=[0, 1], mode="zero"),
    )
    grad_states: list[bool] = []

    def recording_output(features):
        grad_states.append(torch.is_grad_enabled())
        return output_fn(features)

    explainer = ViTExplainer(model, recording_output)
    expected = tuple(
        explainer.intervene(
            inputs,
            target=target,
            interventions=[item],
            forward_args=ForwardArgs(mask=mask),
        )
        for item in requested
    )
    grad_states.clear()
    trace_spy = mocker.spy(intervention_module, "trace_vit")

    actual = explainer.sweep(
        inputs,
        target=target,
        interventions=requested,
        forward_args=ForwardArgs(mask=mask),
    )

    assert trace_spy.call_count == 2
    assert trace_spy.call_args_list[0].args[1].shape[0] == inputs.shape[0]
    assert trace_spy.call_args_list[1].args[1].shape[0] == len(requested) * inputs.shape[0]
    assert grad_states and not any(grad_states)
    for expected_result, actual_result in zip(expected, actual, strict=True):
        assert_close(actual_result.baseline_scores, expected_result.baseline_scores)
        assert_close(actual_result.intervened_scores, expected_result.intervened_scores)


def test_sweep_reuses_one_reference_trace(mocker) -> None:
    model = make_model()
    clean = torch.randn(2, 1, 4, 4)
    reference = torch.randn(2, 1, 4, 4)
    requested = (
        Intervention(site="residual_pre", layer=0, mode="reference"),
        Intervention(site="residual_post", layer=0, mode="reference"),
    )
    explainer = ViTExplainer(model, output_fn)
    expected = tuple(
        explainer.intervene(
            clean,
            target=0,
            interventions=[item],
            reference_inputs=reference,
        )
        for item in requested
    )
    trace_spy = mocker.spy(intervention_module, "trace_vit")

    actual = explainer.sweep(
        clean,
        target=0,
        interventions=requested,
        reference_inputs=reference,
    )

    assert trace_spy.call_count == 3
    for expected_result, actual_result in zip(expected, actual, strict=True):
        assert_close(actual_result.intervened_scores, expected_result.intervened_scores)
