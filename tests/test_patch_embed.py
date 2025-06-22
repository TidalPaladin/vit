import math

import pytest
import torch

from vit.patch_embed import PatchEmbed2d, PatchEmbed3d


class TestPatchEmbed2d:

    @pytest.mark.parametrize("pos_enc", ["fourier", "learnable", "none"])
    def test_forward(self, device, pos_enc):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_enc=pos_enc).to(device)
        x = torch.randn(B, C, H, W, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    def test_backward(self, device):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W)).to(device)
        x = torch.randn(B, C, H, W, requires_grad=True, device=device)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()


class TestPatchEmbed3d:

    @pytest.mark.parametrize("pos_enc", ["fourier", "learnable", "none"])
    def test_forward(self, device, pos_enc):
        B, C, D, H, W = 2, 3, 4, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4), (D, H, W), pos_enc=pos_enc).to(device)
        x = torch.randn(B, C, D, H, W, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((D // 4, H // 4, W // 4)), D_model)

    def test_backward(self, device):
        B, C, D, H, W = 2, 3, 4, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4), (D, H, W)).to(device)
        x = torch.randn(B, C, D, H, W, requires_grad=True, device=device)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()
