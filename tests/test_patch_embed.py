import math

import pytest
import torch
from torch.testing import assert_close

from vit.patch_embed import PatchEmbed2d


class TestPatchEmbed2d:

    @pytest.mark.parametrize("pos_emb", ["factorized", "fourier", "none", "learnable"])
    def test_forward(self, device, pos_emb):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_emb=pos_emb).to(device)
        x = torch.randn(B, C, H, W, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    @pytest.mark.parametrize("pos_emb", ["factorized", "fourier", "none", "learnable"])
    def test_backward(self, device, pos_emb):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_emb=pos_emb).to(device)
        x = torch.randn(B, C, H, W, requires_grad=True, device=device)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_learnable_dropout_deterministic(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_emb="learnable")
        x = torch.randn(B, C, H, W)
        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)
