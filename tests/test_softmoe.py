import pytest
import torch
from torch.testing import assert_close

from vit.soft_moe import SoftMoE


class TestNormMLP:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["srelu", "openswiglu"])
    @pytest.mark.parametrize("num_slots", [4, 10])
    @pytest.mark.parametrize("bias", [True, False])
    def test_forward(self, device, dtype, activation, num_slots, bias):
        B, L, D, D_hidden = 2, 10, 32, 32
        layer = SoftMoE(D, D_hidden, num_slots, bias=bias, activation=activation).to(device)
        x = torch.randn(B, L, D, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y = layer(x)
        assert y.shape == (B, L, D)

    def test_determinstic(self, device):
        torch.random.manual_seed(0)
        B, L, D, D_hidden = 2, 10, 32, 32
        num_slots = 4
        layer = SoftMoE(D, D_hidden, num_slots).to(device)
        x = torch.randn(B, L, D, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, device, dtype):
        B, L, D, D_hidden = 2, 10, 32, 32
        num_slots = 4
        layer = SoftMoE(D, D_hidden, num_slots).to(device)
        x = torch.randn(B, L, D, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()
