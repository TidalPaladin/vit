import pytest
import torch
from torch.testing import assert_close

from vit.rope import RopePositionEmbedding


class TestRopePositionEmbedding:
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_deterministic_with_seed(self, device, dtype):
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

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            result1 = rope(H=8, W=8, rope_seed=42)
            result2 = rope(H=8, W=8, rope_seed=42)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert_close(sin1, sin2, msg="Sin values should be identical with same seed")
        assert_close(cos1, cos2, msg="Cos values should be identical with same seed")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_different_seeds_produce_different_results(self, device, dtype):
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

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            result1 = rope(H=8, W=8, rope_seed=42)
            result2 = rope(H=8, W=8, rope_seed=123)

        sin1, cos1 = result1
        sin2, cos2 = result2

        assert not torch.allclose(sin1, sin2), "Different seeds should produce different sin values"
        assert not torch.allclose(cos1, cos2), "Different seeds should produce different cos values"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_non_deterministic_without_seed(self, device, dtype):
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

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
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
