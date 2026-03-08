import pytest
import torch
from torch.testing import assert_close

from vit.norm import AdaNorm


class TestAdaNorm:
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_zero_init_matches_base_norm(self, device, norm_type):
        layer = AdaNorm(10, norm_type=norm_type).to(device)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)

        y = layer(x, conditioning)
        expected = layer.norm(x)
        assert_close(y, expected)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_batch_conditioning_broadcast_matches_expanded_conditioning(self, device, norm_type):
        layer = AdaNorm(10, norm_type=norm_type).to(device)
        with torch.no_grad():
            torch.nn.init.normal_(layer.modulation.weight, std=0.02)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)
        expanded_conditioning = conditioning[:, None, :].expand(-1, x.shape[1], -1)

        y = layer(x, conditioning)
        y_expanded = layer(x, expanded_conditioning)
        assert_close(y, y_expanded)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        layer = AdaNorm(10, norm_type=norm_type).to(device)
        x = torch.randn(2, 4, 10, device=device)
        conditioning = torch.randn(2, 10, device=device)

        y = layer(x, conditioning)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()
