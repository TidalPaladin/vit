# SPDX-License-Identifier: Apache-2.0
"""Axial two-dimensional rotary position embeddings."""

import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn


CoordinateNormalization = Literal["min", "max", "separate"]
_AXIS_COUNT = 2
_ROTARY_PAIR_SIZE = 2
_HEAD_DIMENSION_MULTIPLE = _AXIS_COUNT * _ROTARY_PAIR_SIZE
_FULL_ROTATION_RADIANS = 2 * math.pi


class RopePositionEmbedding(nn.Module):
    """Create axial 2D RoPE sine and cosine tables for an image token grid."""

    periods: Tensor

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: CoordinateNormalization = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (_HEAD_DIMENSION_MULTIPLE * num_heads) == 0

        has_period_range = min_period is not None and max_period is not None
        if (base is None and not has_period_range) or (base is not None and has_period_range):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        head_dim = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        periods = self._make_periods(head_dim, dtype=dtype, device=device)
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, *, H: int, W: int, rope_seed: int | None = None) -> Tensor:
        """Return stacked sine and cosine tables with shape ``(2, H * W, head_dim)``."""
        coordinates = self._make_coordinates(H, W)
        if self.training:
            generator = self._make_generator(rope_seed, coordinates.device)
            coordinates = self._augment_coordinates(coordinates, generator)

        radians_per_coordinate = _FULL_ROTATION_RADIANS / self.periods
        axis_angles = coordinates.unsqueeze(-1) * radians_per_coordinate
        half_angles = axis_angles.reshape(H * W, self.D_head // _ROTARY_PAIR_SIZE)
        angles = torch.cat((half_angles, half_angles), dim=-1)
        return torch.stack((angles.sin(), angles.cos()))

    if TYPE_CHECKING:

        def __call__(self, H: int, W: int, rope_seed: int | None = None) -> Tensor:
            return self.forward(H=H, W=W, rope_seed=rope_seed)

    def _make_periods(
        self,
        head_dim: int,
        *,
        dtype: torch.dtype | None,
        device: torch.device | None,
    ) -> Tensor:
        frequencies_per_axis = head_dim // _HEAD_DIMENSION_MULTIPLE
        if self.base is not None:
            exponents = torch.arange(frequencies_per_axis, dtype=dtype, device=device) / frequencies_per_axis
            return self.base**exponents

        assert self.min_period is not None and self.max_period is not None
        exponents = torch.linspace(0, 1, frequencies_per_axis, dtype=dtype, device=device)
        period_ratio = self.max_period / self.min_period
        return self.min_period * period_ratio**exponents

    def _make_coordinates(self, height: int, width: int) -> Tensor:
        height_denominator, width_denominator = self._coordinate_denominators(height, width)
        options = {"device": self.periods.device, "dtype": self.dtype}
        height_centers = (torch.arange(height, **options) + 0.5) / height_denominator
        width_centers = (torch.arange(width, **options) + 0.5) / width_denominator
        coordinates = torch.cartesian_prod(height_centers, width_centers).reshape(height * width, _AXIS_COUNT)
        return coordinates.mul(2).sub(1)

    def _coordinate_denominators(self, height: int, width: int) -> tuple[int, int]:
        match self.normalize_coords:
            case "separate":
                return height, width
            case "min":
                denominator = min(height, width)
                return denominator, denominator
            case "max":
                denominator = max(height, width)
                return denominator, denominator
            case _:
                raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

    def _augment_coordinates(self, coordinates: Tensor, generator: torch.Generator | None) -> Tensor:
        options = {"device": coordinates.device, "dtype": coordinates.dtype}

        if self.shift_coords is not None:
            shift = torch.empty(_AXIS_COUNT, **options).uniform_(
                -self.shift_coords,
                self.shift_coords,
                generator=generator,
            )
            coordinates = coordinates + shift

        if self.jitter_coords is not None:
            jitter = self._sample_log_uniform((_AXIS_COUNT,), self.jitter_coords, coordinates, generator)
            coordinates = coordinates * jitter

        if self.rescale_coords is not None:
            scale = self._sample_log_uniform((1,), self.rescale_coords, coordinates, generator)
            coordinates = coordinates * scale

        return coordinates

    @staticmethod
    def _sample_log_uniform(
        shape: tuple[int, ...],
        limit: float,
        reference: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        log_limit = math.log(limit)
        samples = torch.empty(shape, device=reference.device, dtype=reference.dtype)
        return samples.uniform_(-log_limit, log_limit, generator=generator).exp_()

    @staticmethod
    def _make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator


def rope_rotate_half(x: Tensor) -> Tensor:
    """Rotate the two feature halves by 90 degrees."""
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply a precomputed rotary table to a tensor."""
    return x * cos + rope_rotate_half(x) * sin


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """Apply RoPE to non-prefix tokens while preserving the input dtype."""
    sin, cos = rope
    original_dtype = x.dtype
    working = x.type_as(rope)

    prefix_length = working.shape[-2] - sin.shape[-2]
    assert prefix_length >= 0

    prefix = working[:, :, :prefix_length, :]
    rotated = rope_apply(working[:, :, prefix_length:, :], sin, cos)
    return torch.cat((prefix, rotated), dim=-2).to(original_dtype)
