from dataclasses import replace

import pytest
import torch
from torch.testing import assert_close

from vit import AttentivePoolHeadConfig, HeadConfig, TransposedConv2dHeadConfig, ViT, ViTConfig
from vit.explain import (
    AttentionRollout,
    ForwardArgs,
    GradientAttentionRollout,
    InputXGradient,
    IntegratedGradients,
    LayerGradCAM,
    LeGrad,
    PatchOcclusion,
    RawAttention,
    Saliency,
    SmoothGrad,
    ViTExplainer,
)
from vit.explain.methods import compose_rollout


def make_model(*, head=None, conditioning_size: int | None = None) -> ViT:
    heads = {} if head is None else {"prediction": head}
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=2,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        num_cls_tokens=1,
        num_register_tokens=1,
        pos_enc="learnable",
        conditioning_size=conditioning_size,
        adaln_gate_init=1.0,
        dtype=torch.float32,
        heads=heads,
    )
    model = ViT(config).eval()
    with torch.no_grad():
        for index in range(model.config.depth):
            block = model.get_block(index)
            block.self_attention.out_proj.weight.normal_(std=0.1)
            block.mlp.fc2.weight.normal_(std=0.1)
    return model


def output_fn(features):
    return features.visual_tokens.mean(1)[:, :3]


def test_rollout_formula_adds_identity_normalizes_and_composes() -> None:
    layer_0 = torch.tensor([[[[0.75, 0.25], [0.5, 0.5]]]])
    layer_1 = torch.tensor([[[[0.25, 0.75], [1.0, 0.0]]]])

    rollout = compose_rollout((layer_0, layer_1))

    normalized_0 = (layer_0[:, 0] + torch.eye(2)) / 2
    normalized_1 = (layer_1[:, 0] + torch.eye(2)) / 2
    assert_close(rollout, normalized_1 @ normalized_0)


def test_attention_methods_require_explicit_query_selection() -> None:
    model = make_model()
    inputs = torch.randn(1, 1, 4, 4)
    explainer = ViTExplainer(model, output_fn)

    for method in (RawAttention(), AttentionRollout()):
        with pytest.raises(ValueError, match="query"):
            explainer.attribute(inputs, method=method)


@pytest.mark.parametrize(
    "method",
    [
        RawAttention(query=0),
        AttentionRollout(query=0),
        GradientAttentionRollout(query=0),
        LeGrad(),
        LayerGradCAM(layer=0),
    ],
)
def test_native_methods_return_raw_full_grid_attributions(method) -> None:
    torch.manual_seed(5)
    model = make_model()
    inputs = torch.randn(2, 1, 4, 4)

    explanation = ViTExplainer(model, output_fn).attribute(inputs, target=1, method=method)

    assert explanation.token_attributions.shape == (2, 4)
    assert torch.isfinite(explanation.token_attributions).all()
    assert explanation.pixel_attributions is None
    assert explanation.target_scores.shape == (2,)
    assert explanation.layout.grid_size == (2, 2)


def test_integrated_gradients_satisfies_completeness_for_linearized_tiny_model() -> None:
    model = make_model()
    model.blocks = torch.nn.ModuleList()
    model._config = replace(model.config, depth=0)
    inputs = torch.randn(2, 1, 4, 4)
    baseline = torch.zeros_like(inputs)
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.sum((1, 2)))

    explanation = explainer.attribute(
        inputs,
        method=IntegratedGradients(baseline=baseline, n_steps=16),
    )
    input_scores = explainer.output_fn(model(inputs))
    baseline_scores = explainer.output_fn(model(baseline))

    assert explanation.pixel_attributions is not None
    assert_close(
        explanation.pixel_attributions.flatten(1).sum(1),
        input_scores - baseline_scores,
        atol=1e-4,
        rtol=1e-4,
    )
    assert explanation.configuration["baseline"] == {
        "kind": "tensor",
        "shape": list(baseline.shape),
        "dtype": str(baseline.dtype),
    }


def test_from_head_requires_explicit_pooling_for_plain_head() -> None:
    model = make_model(head=HeadConfig(out_features=3))
    with pytest.raises(ValueError, match="pool"):
        ViTExplainer.from_head(model, "prediction")

    explainer = ViTExplainer.from_head(model, "prediction", pool=lambda features: features.visual_tokens.mean(1))
    assert explainer.output_fn(model(torch.randn(1, 1, 4, 4))).shape == (1, 3)


def test_from_head_preserves_stateful_pool_module() -> None:
    class StatefulPool(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(8)
            self.training_states: list[bool] = []

        def forward(self, features) -> torch.Tensor:
            self.training_states.append(self.training)
            return self.norm(features.visual_tokens.mean(1))

    model = make_model(head=HeadConfig(out_features=3))
    pool = StatefulPool().train()
    assert pool.norm.running_mean is not None
    running_mean = pool.norm.running_mean.clone()
    explainer = ViTExplainer.from_head(model, "prediction", pool=pool)

    explainer.attribute(torch.randn(2, 1, 4, 4), target=0, method=InputXGradient())

    assert pool.training_states
    assert not any(pool.training_states)
    assert pool.training
    assert_close(pool.norm.running_mean, running_mean)


def test_from_head_adapts_attentive_pooling() -> None:
    model = make_model(head=AttentivePoolHeadConfig(out_features=3))
    explainer = ViTExplainer.from_head(model, "prediction")
    assert explainer.output_fn(model(torch.randn(1, 1, 4, 4))).shape == (1, 3)


def test_from_head_adapts_dense_2d_head() -> None:
    model = make_model(head=TransposedConv2dHeadConfig(out_features=3, kernel_size=2, stride=2, padding=0))
    explainer = ViTExplainer.from_head(model, "prediction")
    assert explainer.output_fn(model(torch.randn(1, 1, 4, 4))).shape == (1, 3, 4, 4)


@pytest.mark.parametrize(
    "method",
    [Saliency(), InputXGradient(), SmoothGrad(samples=2), PatchOcclusion()],
)
def test_captum_gradient_and_perturbation_methods(method) -> None:
    model = make_model()
    inputs = torch.randn(1, 1, 4, 4)

    explanation = ViTExplainer(model, output_fn).attribute(inputs, target=0, method=method)

    assert explanation.pixel_attributions is not None
    assert explanation.pixel_attributions.shape == inputs.shape
    assert explanation.token_attributions.shape == (1, 4)


@pytest.mark.parametrize("method", [IntegratedGradients(n_steps=4), SmoothGrad(samples=3)])
def test_captum_methods_expand_mask_with_internal_batches(method) -> None:
    model = make_model()
    inputs = torch.randn(2, 1, 4, 4)
    mask = torch.tensor([[True, True, False, True], [False, True, True, True]])

    explanation = ViTExplainer(model, output_fn).attribute(
        inputs,
        target=0,
        method=method,
        forward_args=ForwardArgs(mask=mask),
    )

    assert explanation.pixel_attributions is not None
    assert explanation.pixel_attributions.shape == inputs.shape
    assert torch.isfinite(explanation.pixel_attributions).all()


@pytest.mark.parametrize("method", [IntegratedGradients(n_steps=4), SmoothGrad(samples=3)])
def test_captum_methods_expand_conditioning_with_internal_batches(method) -> None:
    conditioning_size = 5
    model = make_model(conditioning_size=conditioning_size)
    inputs = torch.randn(2, 1, 4, 4)
    conditioning = torch.randn(2, conditioning_size)

    explanation = ViTExplainer(model, output_fn).attribute(
        inputs,
        target=0,
        method=method,
        forward_args=ForwardArgs(conditioning=conditioning),
    )

    assert explanation.pixel_attributions is not None
    assert explanation.pixel_attributions.shape == inputs.shape
    assert torch.isfinite(explanation.pixel_attributions).all()


def test_legrad_supports_frozen_inference_model() -> None:
    model = make_model().requires_grad_(False)
    inputs = torch.randn(1, 1, 4, 4)

    explanation = ViTExplainer(model, output_fn).attribute(inputs, target=0, method=LeGrad())

    assert torch.isfinite(explanation.token_attributions).all()
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_smoothgrad_is_seeded_and_preserves_caller_rng_state() -> None:
    torch.manual_seed(17)
    model = make_model()
    inputs = torch.randn(1, 1, 4, 4)
    explainer = ViTExplainer(model, output_fn)
    method = SmoothGrad(samples=4, stdev=0.2, seed=23)
    initial_rng_state = torch.random.get_rng_state().clone()

    first = explainer.attribute(inputs, target=0, method=method)
    after_first = torch.random.get_rng_state().clone()
    second = explainer.attribute(inputs, target=0, method=method)

    assert first.pixel_attributions is not None
    assert second.pixel_attributions is not None
    assert torch.equal(first.pixel_attributions, second.pixel_attributions)
    assert torch.equal(after_first, initial_rng_state)
    assert torch.equal(torch.random.get_rng_state(), initial_rng_state)
    assert first.configuration["seed"] == 23


def test_external_output_module_runs_in_eval_mode_and_restores_state() -> None:
    class RecordingOutput(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(8)
            self.linear = torch.nn.Linear(8, 3)
            self.training_states: list[bool] = []

        def forward(self, features) -> torch.Tensor:
            self.training_states.append(self.training)
            pooled = features.visual_tokens.mean(1)
            return self.linear(self.norm(pooled))

    model = make_model()
    output_module = RecordingOutput().train()
    inputs = torch.randn(2, 1, 4, 4)
    assert output_module.norm.running_mean is not None
    running_mean = output_module.norm.running_mean.clone()
    explainer = ViTExplainer(model, output_module, output_modules=output_module)

    explainer.attribute(inputs, target=0, method=InputXGradient())

    assert output_module.training_states
    assert not any(output_module.training_states)
    assert output_module.training
    assert_close(output_module.norm.running_mean, running_mean)


@pytest.mark.parametrize(
    "method",
    [
        RawAttention(query=[0, 1]),
        AttentionRollout(query=[0, 1]),
        GradientAttentionRollout(query=[0, 1]),
    ],
)
def test_attention_method_configuration_serializes_query_selector(method) -> None:
    explanation = ViTExplainer(make_model(), output_fn).attribute(
        torch.randn(1, 1, 4, 4),
        target=0,
        method=method,
    )

    assert explanation.configuration["query"] == [0, 1]
