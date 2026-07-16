"""High-level ViT explanation orchestration."""

from collections.abc import Callable, Sequence

import torch.nn as nn
from torch import Tensor

from vit.head import AttentivePoolHead, Head, TransposedConv2dHead, UpsampleHead
from vit.vit import ViT, ViTFeatures

from .evaluation import EvaluationMetric, evaluate as run_evaluation
from .interventions import (
    intervene as run_interventions,
    scan_activations as run_activation_scan,
    sweep_interventions as run_intervention_sweep,
)
from .methods import AttributionMethod
from .trace import preserve_explainer_state, trace_vit
from .types import (
    ActivationAtlas,
    EvaluationReport,
    Explanation,
    ForwardArgs,
    Intervention,
    InterventionResult,
    InterventionSite,
    Target,
    TraceConfig,
    ViTTrace,
)


class ViTExplainer:
    """Explain a native :class:`vit.ViT` using an explicit downstream output function."""

    def __init__(
        self,
        model: ViT,
        output_fn: Callable[[ViTFeatures], Tensor],
        *,
        output_modules: nn.Module | Sequence[nn.Module] = (),
    ):
        self.model = model
        self.output_fn = output_fn
        configured_modules = (output_modules,) if isinstance(output_modules, nn.Module) else tuple(output_modules)
        modules = ((output_fn,) if isinstance(output_fn, nn.Module) else ()) + configured_modules
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("output_modules must contain torch.nn.Module instances")
        self.output_modules = tuple(dict.fromkeys(modules))

    @classmethod
    def from_head(
        cls,
        model: ViT,
        head: str,
        *,
        pool: Callable[[ViTFeatures], Tensor] | None = None,
    ) -> "ViTExplainer":
        """Adapt a configured model head without guessing plain-head pooling semantics."""
        head_module = model.get_head(head)
        output_modules: tuple[nn.Module, ...] = ()
        if isinstance(head_module, AttentivePoolHead):

            def output_fn(features: ViTFeatures) -> Tensor:
                return head_module(features.visual_tokens)

        elif isinstance(head_module, (TransposedConv2dHead, UpsampleHead)):

            def output_fn(features: ViTFeatures) -> Tensor:
                return head_module(features.visual_tokens_as_grid.permute(0, 3, 1, 2))

        elif isinstance(head_module, Head):
            if pool is None:
                raise ValueError("plain Head explainers require an explicit pool callable")
            if isinstance(pool, nn.Module):
                output_modules = (pool,)

            def output_fn(features: ViTFeatures) -> Tensor:
                return head_module(pool(features))

        else:  # pragma: no cover - ViT.get_head currently narrows this union
            raise TypeError(f"unsupported head type: {type(head_module).__name__}")
        return cls(model, output_fn, output_modules=output_modules)

    def attribute(
        self,
        inputs: Tensor,
        *,
        target: Target | None = None,
        method: AttributionMethod,
        forward_args: ForwardArgs | None = None,
    ) -> Explanation:
        """Attribute a selected scalar output with a native or extension method."""
        return method.attribute(self, inputs, target, forward_args or ForwardArgs())

    def trace(
        self,
        inputs: Tensor,
        *,
        config: TraceConfig | None = None,
        forward_args: ForwardArgs | None = None,
    ) -> ViTTrace:
        """Capture a graph-connected eager trace and restore caller-owned model state."""
        with preserve_explainer_state(self):
            return trace_vit(self.model, inputs, config or TraceConfig(), forward_args or ForwardArgs())

    def intervene(
        self,
        inputs: Tensor,
        *,
        target: Target | None,
        interventions: list[Intervention] | tuple[Intervention, ...],
        reference_inputs: Tensor | None = None,
        forward_args: ForwardArgs | None = None,
        reference_forward_args: ForwardArgs | None = None,
    ) -> InterventionResult:
        """Measure target-score changes from simultaneous causal interventions."""
        return run_interventions(
            self,
            inputs,
            target,
            interventions,
            forward_args or ForwardArgs(),
            reference_inputs,
            reference_forward_args,
        )

    def sweep(
        self,
        inputs: Tensor,
        *,
        target: Target | None,
        interventions: list[Intervention] | tuple[Intervention, ...],
        reference_inputs: Tensor | None = None,
        forward_args: ForwardArgs | None = None,
        reference_forward_args: ForwardArgs | None = None,
    ) -> tuple[InterventionResult, ...]:
        """Evaluate interventions independently in a reproducible causal-effect sweep."""
        return run_intervention_sweep(
            self,
            inputs,
            target,
            interventions,
            forward_args or ForwardArgs(),
            reference_inputs,
            reference_forward_args,
        )

    def scan_activations(
        self,
        dataloader,
        *,
        site: InterventionSite,
        layer: int,
        top_k: int = 10,
        forward_args: ForwardArgs | None = None,
        thumbnail=None,
    ) -> ActivationAtlas:
        """Find dataset-level top activating patches without retaining source images."""
        return run_activation_scan(
            self,
            dataloader,
            site=site,
            layer=layer,
            top_k=top_k,
            forward_args=forward_args or ForwardArgs(),
            thumbnail=thumbnail,
        )

    def evaluate(
        self,
        inputs: Tensor,
        explanation: Explanation,
        *,
        target: Target | None = None,
        metrics: list[EvaluationMetric] | tuple[EvaluationMetric, ...],
        forward_args: ForwardArgs | None = None,
    ) -> EvaluationReport:
        """Evaluate an explanation with explicitly selected validation metrics."""
        return run_evaluation(
            self,
            inputs,
            explanation,
            target,
            metrics,
            forward_args or ForwardArgs(),
        )
