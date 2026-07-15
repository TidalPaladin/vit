import math
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch.testing import assert_close

from vit.rope import RopePositionEmbedding


CoordinateMode = Literal["min", "max", "separate"]
PROJECT_ROOT = Path(__file__).parents[1]


def _reference_axial_rope(
    *,
    height: int,
    width: int,
    head_dim: int,
    base: float,
    normalize_coords: CoordinateMode,
    dtype: torch.dtype,
    device: torch.device,
    rope_seed: int | None = None,
    shift_coords: float | None = None,
    jitter_coords: float | None = None,
    rescale_coords: float | None = None,
) -> torch.Tensor:
    frequencies_per_axis = head_dim // 4
    exponents = torch.arange(frequencies_per_axis, dtype=dtype, device=device) / frequencies_per_axis
    periods = base**exponents

    if normalize_coords == "separate":
        height_denominator, width_denominator = height, width
    elif normalize_coords == "min":
        height_denominator = width_denominator = min(height, width)
    else:
        height_denominator = width_denominator = max(height, width)

    height_centers = (torch.arange(height, dtype=dtype, device=device) + 0.5) / height_denominator
    width_centers = (torch.arange(width, dtype=dtype, device=device) + 0.5) / width_denominator
    positions = torch.cartesian_prod(height_centers, width_centers).reshape(height * width, 2)
    positions = positions.mul(2).sub(1)

    generator = torch.Generator(device=device)
    if rope_seed is not None:
        generator.manual_seed(rope_seed)
    if shift_coords is not None:
        shift = torch.empty(2, dtype=dtype, device=device).uniform_(
            -shift_coords,
            shift_coords,
            generator=generator,
        )
        positions = positions + shift
    if jitter_coords is not None:
        log_jitter = math.log(jitter_coords)
        jitter = torch.empty(2, dtype=dtype, device=device).uniform_(
            -log_jitter,
            log_jitter,
            generator=generator,
        )
        positions = positions * jitter.exp()
    if rescale_coords is not None:
        log_rescale = math.log(rescale_coords)
        rescale = torch.empty(1, dtype=dtype, device=device).uniform_(
            -log_rescale,
            log_rescale,
            generator=generator,
        )
        positions = positions * rescale.exp()

    axis_angles = 2 * math.pi * positions.unsqueeze(-1) / periods
    angles = axis_angles.reshape(height * width, head_dim // 2).repeat(1, 2)
    return torch.stack((angles.sin(), angles.cos()))


class TestRopePositionEmbedding:
    @pytest.mark.parametrize("height,width", [(1, 1), (2, 3), (7, 4)])
    @pytest.mark.parametrize("normalize_coords", ["min", "max", "separate"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_numerical_parity_with_axial_reference(self, device, height, width, normalize_coords, dtype):
        embed_dim = 64
        num_heads = 4
        base = 100.0
        rope = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            normalize_coords=normalize_coords,
            dtype=dtype,
            device=device,
        )

        actual = rope(H=height, W=width)
        expected = _reference_axial_rope(
            height=height,
            width=width,
            head_dim=embed_dim // num_heads,
            base=base,
            normalize_coords=normalize_coords,
            dtype=dtype,
            device=device,
        )

        assert_close(actual, expected)

    def test_seeded_training_augmentation_matches_reference(self, device):
        embed_dim = 64
        num_heads = 4
        base = 100.0
        rope_seed = 314159
        shift_coords = 0.2
        jitter_coords = 1.3
        rescale_coords = 1.5
        rope = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            dtype=torch.float64,
            device=device,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
        )
        rope.train()

        actual = rope(H=3, W=5, rope_seed=rope_seed)
        expected = _reference_axial_rope(
            height=3,
            width=5,
            head_dim=embed_dim // num_heads,
            base=base,
            normalize_coords="separate",
            dtype=torch.float64,
            device=device,
            rope_seed=rope_seed,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
        )

        assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_configured_dtype_is_preserved(self, device, dtype):
        rope = RopePositionEmbedding(embed_dim=64, num_heads=4, dtype=dtype, device=device)

        result = rope(H=3, W=5)

        assert result.dtype == dtype

    def test_explicit_period_range_is_geometric_and_inclusive(self, device):
        min_period = 2.0
        max_period = 128.0
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=None,
            min_period=min_period,
            max_period=max_period,
            dtype=torch.float64,
            device=device,
        )

        assert_close(rope.periods[0], torch.tensor(min_period, dtype=torch.float64, device=device))
        assert_close(rope.periods[-1], torch.tensor(max_period, dtype=torch.float64, device=device))
        adjacent_ratios = rope.periods[1:] / rope.periods[:-1]
        assert_close(adjacent_ratios, adjacent_ratios[0].expand_as(adjacent_ratios))

    def test_source_uses_repository_license(self):
        source = (PROJECT_ROOT / "vit" / "rope.py").read_text()

        assert source.startswith("# SPDX-License-Identifier: Apache-2.0")
        assert "DINOv3" not in source

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_basic(self, device, dtype):
        """Test basic forward pass without augmentations."""
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=100.0,
        ).to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            result = rope(H=8, W=8)

        sin, cos = result
        assert sin.shape == (64, 16)  # HW=64, D=embed_dim//num_heads=16
        assert cos.shape == (64, 16)

    def test_deterministic_with_seed(self, device):
        """Test that providing the same seed produces identical results."""
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=100.0,
            shift_coords=0.1,
            jitter_coords=1.2,
            rescale_coords=1.5,
        ).to(device)

        rope.train()  # Enable training mode for augmentations

        result1 = rope(H=8, W=8, rope_seed=42)
        result2 = rope(H=8, W=8, rope_seed=42)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert_close(sin1, sin2, msg="Sin values should be identical with same seed")
        assert_close(cos1, cos2, msg="Cos values should be identical with same seed")

    def test_different_seeds_produce_different_results(self, device):
        """Test that different seeds produce different results."""
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=100.0,
            shift_coords=0.1,
            jitter_coords=1.2,
            rescale_coords=1.5,
        ).to(device)

        rope.train()  # Enable training mode for augmentations

        result1 = rope(H=8, W=8, rope_seed=42)
        result2 = rope(H=8, W=8, rope_seed=123)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert not torch.allclose(sin1, sin2), "Different seeds should produce different sin values"
        assert not torch.allclose(cos1, cos2), "Different seeds should produce different cos values"

    def test_non_deterministic_without_seed(self, device):
        """Test that multiple calls without seed produce different results."""
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=100.0,
            shift_coords=0.1,
            jitter_coords=1.2,
            rescale_coords=1.5,
        ).to(device)

        rope.train()  # Enable training mode for augmentations

        result1 = rope(H=8, W=8)
        result2 = rope(H=8, W=8)

        sin1, cos1 = result1
        sin2, cos2 = result2

        # With augmentations, results should be different without seed
        assert not torch.allclose(sin1, sin2), "Multiple calls without seed should produce different sin values"
        assert not torch.allclose(cos1, cos2), "Multiple calls without seed should produce different cos values"

    def test_eval_mode_consistent_without_seed(self, device):
        """Test that eval mode produces consistent results even without seed."""
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=100.0,
            shift_coords=0.1,
            jitter_coords=1.2,
            rescale_coords=1.5,
        ).to(device)

        rope.eval()  # Disable training mode - no augmentations

        result1 = rope(H=8, W=8)
        result2 = rope(H=8, W=8)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert_close(sin1, sin2, msg="Eval mode should produce identical results")
        assert_close(cos1, cos2, msg="Eval mode should produce identical results")

    @pytest.mark.parametrize("augmentation", ["shift_coords", "jitter_coords", "rescale_coords"])
    def test_individual_augmentations_deterministic(self, device, augmentation):
        """Test determinism for individual augmentation types."""
        kwargs = {
            "embed_dim": 64,
            "num_heads": 4,
            "base": 100.0,
        }
        kwargs[augmentation] = 0.2 if augmentation == "shift_coords" else 1.3

        rope = RopePositionEmbedding(**kwargs).to(device)
        rope.train()

        result1 = rope(H=8, W=8, rope_seed=42)
        result2 = rope(H=8, W=8, rope_seed=42)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert_close(sin1, sin2, msg=f"{augmentation} should be deterministic with seed")
        assert_close(cos1, cos2, msg=f"{augmentation} should be deterministic with seed")

    @pytest.mark.parametrize("normalize_coords", ["min", "max", "separate"])
    def test_coordinate_normalization_modes(self, device, normalize_coords):
        """Test different coordinate normalization modes work with deterministic seeds."""
        rope = RopePositionEmbedding(
            embed_dim=64,
            num_heads=4,
            base=100.0,
            normalize_coords=normalize_coords,
            shift_coords=0.1,
        ).to(device)

        rope.train()

        result1 = rope(H=8, W=8, rope_seed=42)
        result2 = rope(H=8, W=8, rope_seed=42)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert_close(sin1, sin2, msg=f"Normalization mode {normalize_coords} should be deterministic")
        assert_close(cos1, cos2, msg=f"Normalization mode {normalize_coords} should be deterministic")
