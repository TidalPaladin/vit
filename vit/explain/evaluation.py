"""Faithfulness, robustness, completeness, localization, and sanity metrics."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from .trace import make_token_layout, preserve_explainer_state
from .types import EvaluationReport, Explanation, ForwardArgs, MetricResult, Target, select_targets


if TYPE_CHECKING:
    from .explainer import ViTExplainer
    from .methods import AttributionMethod


class EvaluationMetric(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        explainer: "ViTExplainer",
        inputs: Tensor,
        explanation: Explanation,
        target: Target | None,
        forward_args: ForwardArgs,
    ) -> MetricResult: ...


def _scores(explainer: "ViTExplainer", inputs: Tensor, target: Target | None, forward_args: ForwardArgs) -> Tensor:
    features = explainer.model(
        inputs,
        mask=forward_args.mask,
        rope_seed=forward_args.rope_seed,
        output_norm=forward_args.output_norm,
        conditioning=forward_args.conditioning,
    )
    return select_targets(explainer.output_fn(features), target)


def _baseline(inputs: Tensor, value: Tensor | float) -> Tensor:
    baseline = torch.as_tensor(value, device=inputs.device, dtype=inputs.dtype)
    return baseline.expand_as(inputs) if baseline.numel() != inputs.numel() else baseline.reshape_as(inputs)


def _token_values_and_validity(explanation: Explanation) -> tuple[Tensor, Tensor]:
    values = explanation.token_attributions
    validity = explanation.layout.visual_validity.to(device=values.device)
    if validity.shape != values.shape:
        raise ValueError("token attribution shape does not match layout validity")
    if not validity.any(dim=1).all():
        raise ValueError("evaluation requires at least one valid visual token per example")
    if not torch.isfinite(values[validity]).all():
        raise ValueError("attributions for valid visual tokens must be finite")
    return values, validity


def _pixel_validity(explanation: Explanation, inputs: Tensor) -> Tensor:
    _, token_validity = _token_values_and_validity(explanation)
    grid_height, grid_width = explanation.layout.grid_size
    patch_height, patch_width = explanation.layout.patch_size
    modeled_height, modeled_width = explanation.layout.modeled_size
    patch_validity = token_validity.view(-1, 1, grid_height, grid_width)
    modeled_validity = patch_validity.repeat_interleave(patch_height, dim=2).repeat_interleave(patch_width, dim=3)
    validity = torch.zeros(
        (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
        dtype=torch.bool,
        device=inputs.device,
    )
    validity[..., :modeled_height, :modeled_width] = modeled_validity.to(device=inputs.device)
    return validity


def _replace_patches(
    destination: Tensor,
    source: Tensor,
    patch_indices: Tensor,
    batch_index: int,
    patch_size: tuple[int, int],
    grid_width: int,
) -> None:
    patch_height, patch_width = patch_size
    for index in patch_indices.tolist():
        row, column = divmod(int(index), grid_width)
        row_slice = slice(row * patch_height, (row + 1) * patch_height)
        column_slice = slice(column * patch_width, (column + 1) * patch_width)
        destination[batch_index, :, row_slice, column_slice] = source[batch_index, :, row_slice, column_slice]


def saco_score(importance: Tensor, score_changes: Tensor) -> Tensor:
    """Return pairwise signed concordance between group relevance and causal score changes."""
    if importance.shape != score_changes.shape or importance.ndim != 2:
        raise ValueError("SaCo importance and score changes must have matching (batch, groups) shapes")
    importance_delta = importance[:, :, None] - importance[:, None, :]
    score_delta = score_changes[:, :, None] - score_changes[:, None, :]
    upper = torch.triu(torch.ones_like(importance_delta, dtype=torch.bool), diagonal=1)
    comparable = upper & (importance_delta != 0) & (score_delta != 0)
    concordance = (importance_delta.sign() * score_delta.sign()).where(comparable, 0)
    count = comparable.sum(dim=(1, 2))
    return concordance.sum(dim=(1, 2)) / count.clamp_min(1)


@dataclass(frozen=True)
class DeletionInsertion:
    """Patch deletion and insertion curves ranked by attribution magnitude."""

    steps: int = 10
    baseline: Tensor | float = 0.0
    name: str = "deletion_insertion"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        if self.steps <= 0:
            raise ValueError("deletion/insertion steps must be positive")
        token_values, validity = _token_values_and_validity(explanation)
        ranking = token_values.abs().masked_fill(~validity, -torch.inf).argsort(dim=1, descending=True)
        baseline = _baseline(inputs, self.baseline)
        normalized_steps = torch.linspace(0, 1, self.steps + 1, device=inputs.device)
        valid_counts = validity.sum(dim=1)
        counts = (normalized_steps[:, None] * valid_counts.to(device=inputs.device)).round().long()
        deletion_values: list[Tensor] = []
        insertion_values: list[Tensor] = []
        with preserve_explainer_state(explainer), torch.no_grad():
            for step_counts in counts.tolist():
                deleted = inputs.clone()
                inserted = baseline.clone()
                for batch_index, count in enumerate(step_counts):
                    selected = ranking[batch_index, :count]
                    _replace_patches(
                        deleted,
                        baseline,
                        selected,
                        batch_index,
                        explanation.layout.patch_size,
                        explanation.layout.grid_size[1],
                    )
                    _replace_patches(
                        inserted,
                        inputs,
                        selected,
                        batch_index,
                        explanation.layout.patch_size,
                        explanation.layout.grid_size[1],
                    )
                deletion_values.append(_scores(explainer, deleted, target, forward_args))
                insertion_values.append(_scores(explainer, inserted, target, forward_args))
        curves = torch.stack((torch.stack(deletion_values, 1), torch.stack(insertion_values, 1)), dim=1)
        normalized_steps = normalized_steps.to(dtype=curves.dtype)
        metadata = {
            "deletion_auc": torch.trapezoid(curves[:, 0], normalized_steps, dim=1).detach(),
            "insertion_auc": torch.trapezoid(curves[:, 1], normalized_steps, dim=1).detach(),
        }
        return MetricResult(self.name, curves.detach(), metadata)


@dataclass(frozen=True)
class SaCo:
    """Signed causal-order concordance across independently ablated relevance groups."""

    groups: int = 10
    baseline: Tensor | float = 0.0
    name: str = "saco"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        token_values, validity = _token_values_and_validity(explanation)
        valid_counts = validity.sum(dim=1)
        group_count = min(self.groups, int(valid_counts.min().item()))
        if group_count < 2:
            raise ValueError("SaCo requires at least two groups")
        ranking = token_values.masked_fill(~validity, -torch.inf).argsort(dim=1, descending=True)
        batch_groups = tuple(
            torch.tensor_split(ranking[batch_index, : int(valid_counts[batch_index].item())], group_count)
            for batch_index in range(inputs.shape[0])
        )
        baseline = _baseline(inputs, self.baseline)
        importance = torch.stack(
            [
                torch.stack(
                    [
                        token_values[batch_index, batch_groups[batch_index][group_index]].sum()
                        for batch_index in range(inputs.shape[0])
                    ]
                )
                for group_index in range(group_count)
            ],
            dim=1,
        )
        with preserve_explainer_state(explainer), torch.no_grad():
            original_scores = _scores(explainer, inputs, target, forward_args)
            changes: list[Tensor] = []
            for group_index in range(group_count):
                ablated = inputs.clone()
                for batch_index in range(inputs.shape[0]):
                    _replace_patches(
                        ablated,
                        baseline,
                        batch_groups[batch_index][group_index],
                        batch_index,
                        explanation.layout.patch_size,
                        explanation.layout.grid_size[1],
                    )
                changes.append(original_scores - _scores(explainer, ablated, target, forward_args))
        score_changes = torch.stack(changes, dim=1)
        return MetricResult(
            self.name,
            saco_score(importance, score_changes).detach(),
            {"group_importance": importance.detach(), "score_changes": score_changes.detach()},
        )


@dataclass(frozen=True)
class Infidelity:
    """Expected squared mismatch between attribution response and output response."""

    samples: int = 20
    noise_scale: float = 0.01
    seed: int = 0
    name: str = "infidelity"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        if self.samples <= 0:
            raise ValueError("infidelity samples must be positive")
        token_values, validity = _token_values_and_validity(explanation)
        generator = torch.Generator(device=inputs.device).manual_seed(self.seed)
        errors: list[Tensor] = []
        with preserve_explainer_state(explainer), torch.no_grad():
            original = _scores(explainer, inputs, target, forward_args)
            for _ in range(self.samples):
                perturbation = torch.randn(inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype)
                perturbation = perturbation * self.noise_scale
                changed = _scores(explainer, inputs - perturbation, target, forward_args)
                if explanation.pixel_attributions is not None:
                    if explanation.pixel_attributions.shape != inputs.shape:
                        raise ValueError("pixel attribution shape must match inputs")
                    pixel_validity = _pixel_validity(explanation, inputs)
                    pixel_values = explanation.pixel_attributions.where(pixel_validity, 0)
                    estimated = (pixel_values * perturbation).flatten(1).sum(1)
                else:
                    patch_height, patch_width = explanation.layout.patch_size
                    patch_noise = (
                        perturbation[..., : explanation.layout.modeled_size[0], : explanation.layout.modeled_size[1]]
                        .unfold(2, patch_height, patch_height)
                        .unfold(3, patch_width, patch_width)
                    )
                    patch_noise = patch_noise.mean(dim=(1, -1, -2)).flatten(1)
                    estimated = (token_values.where(validity, 0) * patch_noise).sum(1)
                errors.append((estimated - (original - changed)).square())
        return MetricResult(self.name, torch.stack(errors).mean(0).detach())


@dataclass(frozen=True)
class Sensitivity:
    """Maximum attribution change under bounded Gaussian input perturbations."""

    method: "AttributionMethod"
    samples: int = 8
    radius: float = 0.01
    seed: int = 0
    name: str = "sensitivity"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        if self.samples <= 0:
            raise ValueError("sensitivity samples must be positive")
        token_values, validity = _token_values_and_validity(explanation)
        generator = torch.Generator(device=inputs.device).manual_seed(self.seed)
        changes: list[Tensor] = []
        for _ in range(self.samples):
            noise = torch.randn(inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype)
            perturbed = explainer.attribute(
                inputs + noise * self.radius,
                target=target,
                method=self.method,
                forward_args=forward_args,
            )
            perturbed_values, perturbed_validity = _token_values_and_validity(perturbed)
            if not torch.equal(validity, perturbed_validity):
                raise ValueError("sensitivity explanations must share one token validity layout")
            difference = (perturbed_values - token_values).abs().where(validity, 0)
            changes.append(difference.flatten(1).amax(1))
        return MetricResult(self.name, torch.stack(changes).amax(0).detach())


@dataclass(frozen=True)
class Completeness:
    """Absolute residual in the attribution-sum completeness equation."""

    baseline: Tensor | float = 0.0
    name: str = "completeness"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        baseline = _baseline(inputs, self.baseline)
        with preserve_explainer_state(explainer), torch.no_grad():
            score_difference = _scores(explainer, inputs, target, forward_args) - _scores(
                explainer, baseline, target, forward_args
            )
        if explanation.pixel_attributions is None:
            token_values, validity = _token_values_and_validity(explanation)
            attribution_sum = token_values.where(validity, 0).sum(1)
        else:
            pixel_validity = _pixel_validity(explanation, inputs)
            attribution_sum = explanation.pixel_attributions.where(pixel_validity, 0).flatten(1).sum(1)
        return MetricResult(self.name, (attribution_sum - score_difference).abs().detach())


@dataclass(frozen=True)
class Localization:
    """Pointing-game accuracy and positive relevance mass inside a supplied region."""

    region: Tensor
    name: str = "localization"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        _ = explainer, inputs, target, forward_args
        relevance, validity = _token_values_and_validity(explanation)
        region = self.region.to(device=relevance.device, dtype=torch.bool)
        if region.shape == (*relevance.shape[:-1], *explanation.layout.grid_size):
            region = region.flatten(1)
        if region.shape != relevance.shape:
            raise ValueError(f"localization region must have shape {tuple(relevance.shape)} or the token grid")
        peak = relevance.masked_fill(~validity, -torch.inf).argmax(1)
        pointing = region.gather(1, peak[:, None]).squeeze(1).to(relevance.dtype)
        positive = relevance.clamp_min(0).where(validity, 0)
        mass = (positive * region).sum(1) / positive.sum(1).clamp_min(torch.finfo(positive.dtype).eps)
        return MetricResult(self.name, torch.stack((pointing, mass), dim=1).detach())


@dataclass(frozen=True)
class ParameterRandomizationSanity:
    """Centered cosine similarity after deterministic model-parameter randomization."""

    method: "AttributionMethod"
    seed: int = 0
    name: str = "parameter_randomization"

    def evaluate(self, explainer, inputs, explanation, target, forward_args) -> MetricResult:
        state = deepcopy(explainer.model.state_dict())
        devices = [inputs.device] if inputs.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                for module in explainer.model.modules():
                    reset = getattr(module, "reset_parameters", None)
                    if callable(reset):
                        reset()
                randomized = explainer.attribute(
                    inputs,
                    target=target,
                    method=self.method,
                    forward_args=forward_args,
                )
        finally:
            explainer.model.load_state_dict(state)
        original_values, validity = _token_values_and_validity(explanation)
        randomized_values, randomized_validity = _token_values_and_validity(randomized)
        if not torch.equal(validity, randomized_validity):
            raise ValueError("randomized explanations must share one token validity layout")
        similarities: list[Tensor] = []
        for batch_index in range(original_values.shape[0]):
            valid = validity[batch_index]
            original = original_values[batch_index, valid]
            randomized_item = randomized_values[batch_index, valid]
            original = original - original.mean()
            randomized_item = randomized_item - randomized_item.mean()
            similarities.append(torch.nn.functional.cosine_similarity(original, randomized_item, dim=0))
        similarity = torch.stack(similarities)
        return MetricResult(self.name, similarity.detach())


def evaluate(
    explainer: "ViTExplainer",
    inputs: Tensor,
    explanation: Explanation,
    target: Target | None,
    metrics: list[EvaluationMetric] | tuple[EvaluationMetric, ...],
    forward_args: ForwardArgs,
) -> EvaluationReport:
    layout = make_token_layout(explainer.model, inputs, forward_args.mask)
    if not layout.matches(explanation.layout):
        raise ValueError("evaluation inputs and forward arguments must match the explanation token layout")
    results = {metric.name: metric.evaluate(explainer, inputs, explanation, target, forward_args) for metric in metrics}
    if len(results) != len(metrics):
        raise ValueError("evaluation metric names must be unique")
    return EvaluationReport(results)
