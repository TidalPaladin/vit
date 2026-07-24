"""Typed public data structures for ViT explainability."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

from vit.vit import ViTFeatures


TargetCallable: TypeAlias = Callable[[Tensor], Tensor]
DenseTarget: TypeAlias = tuple[int, ...]
Target: TypeAlias = int | Tensor | DenseTarget | TargetCallable


@dataclass(frozen=True)
class ForwardArgs:
    """Arguments that must be reproduced for every explanatory forward pass."""

    mask: Tensor | None = None
    rope_seed: int | None = None
    output_norm: bool = True
    conditioning: Tensor | None = None


@dataclass(frozen=True)
class TraceConfig:
    """Control which trace tensors retain gradients and which layers are returned."""

    layers: tuple[int, ...] | None = None
    retain_gradients: bool = False
    mlp_internals: bool = False


@dataclass(frozen=True)
class TokenLayout:
    """Map model sequence positions to the original two-dimensional patch grid."""

    grid_size: tuple[int, int]
    patch_size: tuple[int, int]
    original_size: tuple[int, int]
    modeled_size: tuple[int, int]
    num_cls_tokens: int
    num_register_tokens: int
    visual_indices: Tensor
    visual_validity: Tensor

    @property
    def prefix_length(self) -> int:
        return self.num_cls_tokens + self.num_register_tokens

    @property
    def sequence_validity(self) -> Tensor:
        return self.visual_indices >= 0

    @property
    def visual_token_count(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def spatially_matches(self, other: "TokenLayout") -> bool:
        """Return whether two layouts describe the same modeled image geometry."""
        return (
            self.grid_size == other.grid_size
            and self.patch_size == other.patch_size
            and self.original_size == other.original_size
            and self.modeled_size == other.modeled_size
        )

    def matches(self, other: "TokenLayout") -> bool:
        """Return whether two layouts have identical token and spatial semantics."""
        return (
            self.spatially_matches(other)
            and self.num_cls_tokens == other.num_cls_tokens
            and self.num_register_tokens == other.num_register_tokens
            and torch.equal(self.visual_indices, other.visual_indices)
            and torch.equal(self.visual_validity, other.visual_validity)
        )

    def scatter_visual(self, values: Tensor, *, fill_value: float = float("nan")) -> Tensor:
        """Scatter a final visual-sequence axis back to the complete patch grid."""
        if values.shape[0] != self.visual_indices.shape[0]:
            raise ValueError("attribution batch size does not match token layout")
        if values.shape[-1] != self.visual_indices.shape[-1]:
            raise ValueError("attribution token count does not match token layout")
        output = values.new_full((*values.shape[:-1], self.visual_token_count), fill_value)
        for batch_index in range(values.shape[0]):
            valid = self.sequence_validity[batch_index]
            indices = self.visual_indices[batch_index, valid]
            output[batch_index, ..., indices] = values[batch_index, ..., valid]
        return output.view(*values.shape[:-1], *self.grid_size)


@dataclass(frozen=True)
class MLPTrace:
    """Graph-connected tensors from one eager MLP forward."""

    normalized_input: Tensor
    fc1_output: Tensor
    linear_branch: Tensor | None
    gate_branch: Tensor | None
    activation_output: Tensor
    hidden: Tensor
    output: Tensor


@dataclass(frozen=True)
class LayerTrace:
    """Captured tensors at one transformer encoder block."""

    layer: int
    residual_pre: Tensor
    attention_probabilities: Tensor
    head_outputs: Tensor
    attention_output: Tensor
    residual_post_attention: Tensor
    mlp_output: Tensor
    residual_post: Tensor
    mlp: MLPTrace | None = None


@dataclass(frozen=True)
class ViTTrace:
    """Eager forward result with graph-connected internal transformer tensors."""

    features: ViTFeatures
    layout: TokenLayout
    layers: tuple[LayerTrace, ...]
    forward_args: ForwardArgs


@dataclass(frozen=True)
class Explanation:
    """Raw, unnormalized attribution values and the layout needed to render them."""

    method: str
    token_attributions: Tensor
    pixel_attributions: Tensor | None
    target_scores: Tensor
    layout: TokenLayout
    layer_attributions: tuple[Tensor, ...] = ()
    configuration: Mapping[str, Any] = field(default_factory=dict)


InterventionSite: TypeAlias = Literal[
    "residual_pre",
    "head_output",
    "post_attention",
    "mlp_output",
    "residual_post",
]


@dataclass(frozen=True)
class Intervention:
    """Replace selected activations at a causal intervention site."""

    site: InterventionSite
    layer: int
    mode: Literal["zero", "constant", "mean", "reference"] = "zero"
    tokens: Sequence[int] | slice | None = None
    channels: Sequence[int] | slice | None = None
    heads: Sequence[int] | slice | None = None
    value: float | Tensor | None = None


@dataclass(frozen=True)
class InterventionResult:
    """Target-score changes caused by one or more interventions."""

    baseline_scores: Tensor
    intervened_scores: Tensor
    absolute_change: Tensor
    relative_change: Tensor
    interventions: tuple[Intervention, ...]


@dataclass(frozen=True)
class ActivationRecord:
    sample_id: str
    value: float
    patch_coordinate: tuple[int, int]
    thumbnail: Any | None = None


@dataclass(frozen=True)
class ActivationAtlas:
    """Top activating dataset patches for channels at one trace site."""

    site: InterventionSite
    layer: int
    top_k: int
    channels: Mapping[int, tuple[ActivationRecord, ...]]
    layout: TokenLayout | None = None


@dataclass(frozen=True)
class MetricResult:
    name: str
    values: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    """Faithfulness, robustness, and localization measurements."""

    metrics: Mapping[str, MetricResult]


def select_targets(outputs: Tensor, target: Target | None) -> Tensor:
    """Select exactly one scalar output per batch item."""
    if callable(target):
        selected = target(outputs)
    elif target is None:
        if outputs.ndim != 1:
            raise ValueError("target is required when output_fn does not return one scalar per example")
        selected = outputs
    elif isinstance(target, int):
        selected = outputs[:, target]
    elif isinstance(target, tuple):
        selected = outputs[(slice(None), *target)]
    elif isinstance(target, Tensor):
        if target.ndim == 0:
            selected = outputs[:, int(target.item())]
        elif target.ndim == 1 and outputs.ndim == 2 and target.shape[0] == outputs.shape[0]:
            selected = outputs.gather(1, target.to(device=outputs.device, dtype=torch.long)[:, None]).squeeze(1)
        elif target.ndim == 2 and target.shape[0] == outputs.shape[0]:
            batch = torch.arange(outputs.shape[0], device=outputs.device)
            selected = outputs[(batch, *target.to(device=outputs.device, dtype=torch.long).unbind(1))]
        else:
            raise ValueError("target tensor must contain one class index or coordinate per example")
    else:
        raise TypeError(f"unsupported target type: {type(target).__name__}")
    if selected.shape != (outputs.shape[0],):
        raise ValueError(f"target must select one scalar per example, got shape {tuple(selected.shape)}")
    return selected


def tensor_configuration(**values: Any) -> dict[str, Any]:
    """Serialize method configuration without retaining tensor payloads."""
    return {
        name: {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if isinstance(value, torch.Tensor)
        else value
        for name, value in values.items()
    }
