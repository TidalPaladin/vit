import math

import pytest
import torch
from torch.testing import assert_close

from vit.patch_embed import PatchEmbed2d, PatchEmbed3d


class TestPatchEmbed2d:

    @pytest.mark.parametrize("pos_emb", ["fourier", "none", "learnable"])
    def test_forward(self, device, pos_emb):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_emb=pos_emb).to(device)
        x = torch.randn(B, C, H, W, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    @pytest.mark.parametrize("pos_emb", ["fourier", "none", "learnable"])
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

    def test_backward_fourier_swiglu(self, device):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_emb="fourier", activation="swiglu").to(device)
        x = torch.randn(B, C, H, W, requires_grad=True, device=device)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("pos_emb", ["learnable", "fourier"])
    @pytest.mark.parametrize("dropout,pos_dropout", [(0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
    def test_dropout_deterministic(self, pos_emb, dropout, pos_dropout):
        torch.random.manual_seed(0)
        B, C, H, W = 16, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), (H, W), pos_emb=pos_emb, dropout=dropout, pos_dropout=pos_dropout)
        x = torch.randn(B, C, H, W)
        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y1 = layer(x)
        y2 = layer(x)
        assert not torch.allclose(y1, y2)


class TestPatchEmbed3d:

    @pytest.mark.parametrize("pos_emb", ["fourier", "none", "learnable"])
    def test_forward(self, device, pos_emb):
        B, C, D, H, W = 2, 3, 4, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4), (D, H, W), pos_emb=pos_emb).to(device)
        x = torch.randn(B, C, D, H, W, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((D // 4, H // 4, W // 4)), D_model)

    @pytest.mark.parametrize("pos_emb", ["fourier", "none", "learnable"])
    def test_backward(self, device, pos_emb):
        B, C, D, H, W = 2, 3, 4, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4), (D, H, W), pos_emb=pos_emb).to(device)
        x = torch.randn(B, C, D, H, W, requires_grad=True, device=device)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_backward_fourier_swiglu(self, device):
        B, C, D, H, W = 2, 3, 4, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4), (D, H, W), pos_emb="fourier", activation="swiglu").to(device)
        x = torch.randn(B, C, D, H, W, requires_grad=True, device=device)
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("pos_emb", ["learnable", "fourier"])
    @pytest.mark.parametrize("dropout,pos_dropout", [(0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
    def test_dropout_deterministic(self, pos_emb, dropout, pos_dropout):
        torch.random.manual_seed(0)
        B, C, D, H, W = 16, 3, 4, 64, 64
        D_model = 64
        layer = PatchEmbed3d(
            C, D_model, (4, 4, 4), (D, H, W), pos_emb=pos_emb, dropout=dropout, pos_dropout=pos_dropout
        )
        x = torch.randn(B, C, D, H, W)
        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)
