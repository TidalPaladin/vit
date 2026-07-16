"""Explicit normalization and interpolation operations for attribution maps."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import Explanation


Normalization = Literal["none", "minmax", "symmetric", "absolute"]


def normalize_attribution(values: Tensor, mode: Normalization = "minmax") -> Tensor:
    """Normalize a map only when the caller explicitly requests it."""
    if mode == "none":
        return values
    finite = torch.isfinite(values)
    safe = values.where(finite, 0)
    dimensions = tuple(range(1, values.ndim))
    has_finite = finite.any(dim=dimensions, keepdim=True)
    if mode == "absolute":
        safe = safe.abs()
        denominator = safe.amax(dim=dimensions, keepdim=True).clamp_min(torch.finfo(safe.dtype).eps)
        result = safe / denominator
    elif mode == "symmetric":
        denominator = safe.abs().amax(dim=dimensions, keepdim=True).clamp_min(torch.finfo(safe.dtype).eps)
        result = safe / denominator
    elif mode == "minmax":
        minimum = values.masked_fill(~finite, torch.inf).amin(dim=dimensions, keepdim=True)
        maximum = values.masked_fill(~finite, -torch.inf).amax(dim=dimensions, keepdim=True)
        minimum = minimum.where(has_finite, 0)
        maximum = maximum.where(has_finite, 0)
        result = (safe - minimum) / (maximum - minimum).clamp_min(torch.finfo(safe.dtype).eps)
    else:
        raise ValueError(f"unknown attribution normalization: {mode}")
    return result.where(finite, torch.nan)


def interpolate_token_attribution(
    explanation: Explanation,
    *,
    size: tuple[int, int] | None = None,
    normalization: Normalization = "none",
) -> Tensor:
    """Interpolate token scores while leaving unmodeled image borders as NaN."""
    target_size = size or explanation.layout.original_size
    if any(dimension <= 0 for dimension in target_size):
        raise ValueError("attribution interpolation dimensions must be positive")
    if size is not None and target_size == explanation.layout.modeled_size:
        modeled_target_size = target_size
    else:
        modeled_target_size = tuple(
            max(1, target * modeled // original)
            for target, modeled, original in zip(
                target_size,
                explanation.layout.modeled_size,
                explanation.layout.original_size,
                strict=True,
            )
        )
    values = explanation.token_attributions.view(-1, 1, *explanation.layout.grid_size)
    values = normalize_attribution(values, normalization)
    layout_validity = explanation.layout.visual_validity.to(device=values.device).view_as(values)
    validity = torch.isfinite(values) & layout_validity
    safe_values = values.where(validity, 0)
    support = validity.to(values.dtype)
    weighted_values = F.interpolate(
        safe_values,
        size=modeled_target_size,
        mode="bilinear",
        align_corners=False,
    )
    interpolated_support = F.interpolate(
        support,
        size=modeled_target_size,
        mode="bilinear",
        align_corners=False,
    )
    modeled = weighted_values / interpolated_support.clamp_min(torch.finfo(values.dtype).eps)
    modeled_validity = F.interpolate(support, size=modeled_target_size, mode="nearest").bool()
    modeled = modeled.where(modeled_validity, torch.nan)
    if target_size == modeled_target_size:
        return modeled[:, 0]
    result = modeled.new_full((modeled.shape[0], 1, *target_size), torch.nan)
    result[..., : modeled_target_size[0], : modeled_target_size[1]] = modeled
    return result[:, 0]
