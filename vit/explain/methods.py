"""Native and Captum-backed attribution methods for ViTs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import Tensor

from vit.vit import ViTFeatures

from .trace import make_token_layout, preserve_explainer_state, trace_vit
from .types import Explanation, ForwardArgs, Target, TraceConfig, select_targets, tensor_configuration


if TYPE_CHECKING:
    from .explainer import ViTExplainer


Query = int | Sequence[int] | Tensor | None


class AttributionMethod(Protocol):
    """Extension protocol implemented by all attribution methods."""

    @property
    def name(self) -> str: ...

    def attribute(
        self,
        explainer: "ViTExplainer",
        inputs: Tensor,
        target: Target | None,
        forward_args: ForwardArgs,
    ) -> Explanation: ...


def _queries(values: Tensor, query: Query) -> Tensor:
    if query is None:
        raise ValueError("attention explanations require an explicit query selector")
    if isinstance(query, int):
        return values[..., query, :]
    indices = torch.as_tensor(query, device=values.device, dtype=torch.long)
    selected = values.index_select(-2, indices)
    return selected.mean(dim=-2)


def _query_configuration(query: Query) -> int | list[int]:
    if query is None:  # pragma: no cover - callers validate query first
        raise ValueError("attention explanations require an explicit query selector")
    if isinstance(query, int):
        return query
    if isinstance(query, Tensor):
        value = query.detach().cpu()
        return int(value.item()) if value.ndim == 0 else [int(index) for index in value.tolist()]
    return [int(index) for index in query]


def _visual_attribution(values: Tensor, trace, *, fill_value: float = float("nan")) -> Tensor:
    visual = values[..., trace.layout.prefix_length :]
    return trace.layout.scatter_visual(visual, fill_value=fill_value).flatten(1)


def _scores(explainer: "ViTExplainer", features: ViTFeatures, target: Target | None) -> Tensor:
    return select_targets(explainer.output_fn(features), target)


def _result(
    method: str,
    attribution: Tensor,
    scores: Tensor,
    trace,
    *,
    layer_attributions: tuple[Tensor, ...] = (),
    configuration: dict[str, Any] | None = None,
) -> Explanation:
    return Explanation(
        method=method,
        token_attributions=attribution.detach(),
        pixel_attributions=None,
        target_scores=scores.detach(),
        layout=trace.layout,
        layer_attributions=tuple(value.detach() for value in layer_attributions),
        configuration=configuration or {},
    )


def compose_rollout(attention_layers: Sequence[Tensor]) -> Tensor:
    """Average heads, add residual identity, row-normalize, and compose layers."""
    if not attention_layers:
        raise ValueError("attention rollout requires at least one layer")
    rollout: Tensor | None = None
    for attention in attention_layers:
        if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
            raise ValueError("attention layers must have shape (batch, heads, query, key) with square maps")
        fused = attention.mean(dim=1)
        identity = torch.eye(fused.shape[-1], device=fused.device, dtype=fused.dtype)
        fused = fused + identity
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(fused.dtype).eps)
        rollout = fused if rollout is None else fused @ rollout
    assert rollout is not None
    return rollout


@dataclass(frozen=True)
class RawAttention:
    """Per-head attention from one layer for an explicit query selector."""

    query: Query = None
    layer: int = -1
    head: int | None = None
    name: str = "raw_attention"

    def attribute(self, explainer, inputs, target, forward_args) -> Explanation:
        if self.query is None:
            raise ValueError("raw attention requires an explicit query selector")
        layer = self.layer % explainer.model.config.depth
        with preserve_explainer_state(explainer):
            trace = trace_vit(explainer.model, inputs, TraceConfig(layers=(layer,)), forward_args)
            probabilities = trace.layers[0].attention_probabilities
            fused = probabilities.mean(1) if self.head is None else probabilities[:, self.head]
            attribution = _visual_attribution(_queries(fused, self.query), trace)
            scores = _scores(explainer, trace.features, target)
        return _result(
            self.name,
            attribution,
            scores,
            trace,
            configuration={"query": _query_configuration(self.query), "layer": layer, "head": self.head},
        )


@dataclass(frozen=True)
class AttentionRollout:
    """Target-independent attention-flow rollout for an explicit query."""

    query: Query = None
    layers: tuple[int, ...] | None = None
    name: str = "attention_rollout"

    def attribute(self, explainer, inputs, target, forward_args) -> Explanation:
        if self.query is None:
            raise ValueError("attention rollout requires an explicit query selector")
        with preserve_explainer_state(explainer):
            trace = trace_vit(explainer.model, inputs, TraceConfig(layers=self.layers), forward_args)
            rollout = compose_rollout(tuple(layer.attention_probabilities for layer in trace.layers))
            attribution = _visual_attribution(_queries(rollout, self.query), trace)
            scores = _scores(explainer, trace.features, target)
        return _result(
            self.name,
            attribution,
            scores,
            trace,
            configuration={"query": _query_configuration(self.query), "layers": self.layers},
        )


@dataclass(frozen=True)
class GradientAttentionRollout:
    """Class-specific rollout using positive gradient-times-attention relevance."""

    query: Query = None
    layers: tuple[int, ...] | None = None
    name: str = "gradient_attention_rollout"

    def attribute(self, explainer, inputs, target, forward_args) -> Explanation:
        if self.query is None:
            raise ValueError("gradient attention rollout requires an explicit query selector")
        with preserve_explainer_state(explainer):
            trace = trace_vit(explainer.model, inputs, TraceConfig(layers=self.layers), forward_args)
            scores = _scores(explainer, trace.features, target)
            probabilities = tuple(layer.attention_probabilities for layer in trace.layers)
            gradients = torch.autograd.grad(scores.sum(), probabilities, retain_graph=False)
            relevance = tuple(
                (gradient * probability).clamp_min(0) for gradient, probability in zip(gradients, probabilities)
            )
            rollout = compose_rollout(relevance)
            attribution = _visual_attribution(_queries(rollout, self.query), trace)
        return _result(
            self.name,
            attribution,
            scores,
            trace,
            configuration={"query": _query_configuration(self.query), "layers": self.layers},
        )


@dataclass(frozen=True)
class LeGrad:
    """ViT feature-formation attribution from positive attention gradients."""

    layers: tuple[int, ...] | None = None
    name: str = "legrad"

    def attribute(self, explainer, inputs, target, forward_args) -> Explanation:
        with preserve_explainer_state(explainer):
            trace = trace_vit(explainer.model, inputs, TraceConfig(layers=self.layers), forward_args)
            final_scores = _scores(explainer, trace.features, target)
            layer_values: list[Tensor] = []
            for layer in trace.layers:
                intermediate = (
                    explainer.model.output_norm(layer.residual_post)
                    if forward_args.output_norm
                    else layer.residual_post
                )
                features = ViTFeatures(
                    intermediate,
                    explainer.model.config.num_register_tokens,
                    explainer.model.config.num_cls_tokens,
                    trace.layout.grid_size,
                )
                intermediate_scores = _scores(explainer, features, target)
                gradient = torch.autograd.grad(
                    intermediate_scores.sum(),
                    layer.attention_probabilities,
                    retain_graph=True,
                )[0]
                relevance = gradient.clamp_min(0).mean(dim=(1, 2))
                layer_values.append(_visual_attribution(relevance, trace))
            attribution = torch.stack(layer_values).mean(0)
        return _result(
            self.name,
            attribution,
            final_scores,
            trace,
            layer_attributions=tuple(layer_values),
            configuration={"layers": self.layers},
        )


@dataclass(frozen=True)
class LayerGradCAM:
    """Grad-CAM over one block's post-block visual-token grid."""

    layer: int = -1
    relu: bool = True
    name: str = "layer_grad_cam"

    def attribute(self, explainer, inputs, target, forward_args) -> Explanation:
        layer = self.layer % explainer.model.config.depth
        with preserve_explainer_state(explainer):
            trace = trace_vit(explainer.model, inputs, TraceConfig(), forward_args)
            scores = _scores(explainer, trace.features, target)
            residual_post = trace.layers[layer].residual_post
            gradients = torch.autograd.grad(scores.sum(), residual_post, retain_graph=False)[0]
            activations = residual_post[:, trace.layout.prefix_length :]
            gradients = gradients[:, trace.layout.prefix_length :]
            channel_weights = gradients.mean(dim=1, keepdim=True)
            relevance = (activations * channel_weights).sum(dim=-1)
            if self.relu:
                relevance = relevance.clamp_min(0)
            attribution = trace.layout.scatter_visual(relevance).flatten(1)
        return _result(self.name, attribution, scores, trace, configuration={"layer": layer, "relu": self.relu})


def _captum() -> Any:
    try:
        import captum.attr
    except ModuleNotFoundError as error:
        if error.name == "captum" or (error.name and error.name.startswith("captum.")):
            raise ModuleNotFoundError(
                "Captum attribution methods require the explainability extra: pip install 'vit[explainability]'",
                name="captum",
            ) from None
        raise
    return captum.attr


def _input_to_tokens(attributions: Tensor, layout) -> Tensor:
    height, width = layout.modeled_size
    patch_height, patch_width = layout.patch_size
    cropped = attributions[..., :height, :width]
    patches = cropped.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    return patches.sum(dim=(1, -1, -2)).flatten(1)


class _CaptumMethod:
    name = "captum"

    def _attribute(
        self,
        captum_attr: Any,
        forward,
        inputs: Tensor,
        explainer: "ViTExplainer",
        additional_forward_args: tuple[Any, ...],
    ) -> Tensor:
        raise NotImplementedError

    def attribute(self, explainer, inputs, target, forward_args) -> Explanation:
        captum_attr = _captum()
        with preserve_explainer_state(explainer):
            layout = make_token_layout(explainer.model, inputs, forward_args.mask)

            def forward(
                tensor: Tensor,
                mask: Tensor | None,
                conditioning: Tensor | None,
                selected_target: Target | None,
            ) -> Tensor:
                features = explainer.model(
                    tensor,
                    mask=mask,
                    rope_seed=forward_args.rope_seed,
                    output_norm=forward_args.output_norm,
                    conditioning=conditioning,
                )
                return _scores(explainer, features, selected_target)

            additional_forward_args = (forward_args.mask, forward_args.conditioning, target)
            pixel_attributions = self._attribute(
                captum_attr,
                forward,
                inputs,
                explainer,
                additional_forward_args,
            )
            scores = forward(inputs, *additional_forward_args)
            token_attributions = _input_to_tokens(pixel_attributions, layout)
            token_attributions = token_attributions.masked_fill(~layout.visual_validity, torch.nan)
        return Explanation(
            method=self.name,
            token_attributions=token_attributions.detach(),
            pixel_attributions=pixel_attributions.detach(),
            target_scores=scores.detach(),
            layout=layout,
            configuration=tensor_configuration(**self.__dict__),
        )


@dataclass(frozen=True)
class Saliency(_CaptumMethod):
    absolute: bool = False
    name: str = "saliency"

    def _attribute(self, captum_attr, forward, inputs, explainer, additional_forward_args):
        _ = explainer
        return captum_attr.Saliency(forward).attribute(
            inputs,
            abs=self.absolute,
            additional_forward_args=additional_forward_args,
        )


@dataclass(frozen=True)
class InputXGradient(_CaptumMethod):
    name: str = "input_x_gradient"

    def _attribute(self, captum_attr, forward, inputs, explainer, additional_forward_args):
        _ = explainer
        return captum_attr.InputXGradient(forward).attribute(
            inputs,
            additional_forward_args=additional_forward_args,
        )


@dataclass(frozen=True)
class IntegratedGradients(_CaptumMethod):
    baseline: Tensor | float = 0.0
    n_steps: int = 50
    method: str = "gausslegendre"
    name: str = "integrated_gradients"

    def _attribute(self, captum_attr, forward, inputs, explainer, additional_forward_args):
        _ = explainer
        return captum_attr.IntegratedGradients(forward).attribute(
            inputs,
            baselines=self.baseline,
            n_steps=self.n_steps,
            method=self.method,
            additional_forward_args=additional_forward_args,
        )


@dataclass(frozen=True)
class SmoothGrad(_CaptumMethod):
    samples: int = 25
    stdev: float = 0.1
    absolute: bool = False
    seed: int = 0
    name: str = "smoothgrad"

    def _attribute(self, captum_attr, forward, inputs, explainer, additional_forward_args):
        _ = explainer
        saliency = captum_attr.Saliency(forward)
        devices = [inputs.device] if inputs.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if inputs.device.type == "cuda":
                with torch.cuda.device(inputs.device):
                    torch.cuda.manual_seed(self.seed)
            else:
                generator = torch.Generator(device="cpu").manual_seed(self.seed)
                torch.random.set_rng_state(generator.get_state())
            return captum_attr.NoiseTunnel(saliency).attribute(
                inputs,
                nt_type="smoothgrad",
                nt_samples=self.samples,
                stdevs=self.stdev,
                abs=self.absolute,
                additional_forward_args=additional_forward_args,
            )


@dataclass(frozen=True)
class PatchOcclusion(_CaptumMethod):
    baseline: Tensor | float = 0.0
    patch_size: tuple[int, int] | None = None
    strides: tuple[int, int] | None = None
    name: str = "patch_occlusion"

    def _attribute(self, captum_attr, forward, inputs, explainer, additional_forward_args):
        configured_patch_size = tuple(explainer.model.config.patch_size)
        patch_size = self.patch_size or (int(configured_patch_size[0]), int(configured_patch_size[1]))
        strides = self.strides or patch_size
        return captum_attr.Occlusion(forward).attribute(
            inputs,
            baselines=self.baseline,
            sliding_window_shapes=(inputs.shape[1], *patch_size),
            strides=(inputs.shape[1], *strides),
            additional_forward_args=additional_forward_args,
        )
