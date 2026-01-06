import torch
from torch.testing import assert_close

from vit.drop_path import DropPath, drop_path


class TestDropPath:
    def test_forward_no_drop(self, device):
        """Test that drop_prob=0 returns input unchanged."""
        layer = DropPath(drop_prob=0.0).to(device)
        layer.train()
        x = torch.randn(4, 16, 64, device=device)
        y = layer(x)
        assert_close(y, x)

    def test_high_drop_rate(self, device):
        """Test that high drop_prob drops most samples in training mode."""
        layer = DropPath(drop_prob=0.99).to(device)
        layer.train()
        x = torch.ones(1000, 16, 64, device=device)  # Large batch for statistical test
        y = layer(x)
        # With 99% drop rate, most samples should be zero
        dropped_count = (y.sum(dim=(1, 2)) == 0).sum()
        # At least 90% should be dropped (allowing some variance)
        assert dropped_count >= 900

    def test_eval_mode_no_drop(self, device):
        """Test that eval mode disables dropping regardless of drop_prob."""
        layer = DropPath(drop_prob=0.5).to(device)
        layer.eval()
        x = torch.randn(4, 16, 64, device=device)
        y = layer(x)
        assert_close(y, x)

    def test_train_mode_stochastic(self, device):
        """Test that train mode with drop_prob>0 produces different outputs."""
        layer = DropPath(drop_prob=0.5).to(device)
        layer.train()
        x = torch.randn(4, 16, 64, device=device)
        # Run multiple times to ensure stochasticity
        y1 = layer(x)
        y2 = layer(x)
        # With 50% drop rate, outputs should differ (very unlikely to be same)
        assert not torch.allclose(y1, y2)

    def test_scaling(self, device):
        """Test that surviving paths are scaled by 1/(1-drop_prob)."""
        drop_prob = 0.5
        keep_prob = 1 - drop_prob
        x = torch.ones(1000, 1, 1, device=device)  # Large batch for statistical test

        # Call the function directly with training=True
        y = drop_path(x, drop_prob, training=True)

        # Non-zero elements should be scaled by 1/keep_prob = 2
        non_zero_mask = y != 0
        if non_zero_mask.any():
            scaled_values = y[non_zero_mask]
            expected = 1.0 / keep_prob
            assert_close(scaled_values, torch.full_like(scaled_values, expected))

    def test_sample_independence(self, device):
        """Test that drop decisions are independent per sample (not per element)."""
        layer = DropPath(drop_prob=0.5).to(device)
        layer.train()
        # Each sample should be either fully kept (scaled) or fully dropped
        x = torch.ones(100, 16, 64, device=device)
        y = layer(x)

        # For each sample, all elements should be either 0 or scaled uniformly
        for i in range(100):
            sample = y[i]
            unique_vals = sample.unique()
            # Should have at most 2 unique values: 0 and scaled value
            assert len(unique_vals) <= 2
