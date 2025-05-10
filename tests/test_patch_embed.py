import math

import torch

from vit.patch_embed import PatchEmbed2d


class TestPatchEmbed2d:

    def test_forward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4))
        x = torch.randn(B, C, H, W)
        y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    def test_backward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4))
        x = torch.randn(B, C, H, W, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()
