import pytest
import torch

from vit.pos_enc import create_grid
from vit.rope import MixedRoPE, init_random_nd_freqs


class TestMixedRoPE:

    @pytest.mark.parametrize(
        "dim,nhead,spatial_dims,theta",
        [
            (16, 2, 1, 100),
            (16, 2, 2, 100),
            (16, 2, 3, 100),
        ],
    )
    def test_init_random_nd_freqs(self, dim, nhead, spatial_dims, theta):
        freqs = init_random_nd_freqs(dim, nhead, spatial_dims, theta)
        assert freqs.shape == (spatial_dims, nhead, dim // 2)

    def test_forward_2d(self, device):
        B, H, W, D = 2, 8, 8, 64
        L = H * W
        nhead = 4
        layer = MixedRoPE(D, nhead, (H, W)).to(device)

        grid = create_grid((H, W), device=device).expand(B, -1, -1)
        x = torch.randn(B, nhead, L, D // nhead, device=device)
        out = layer(x, grid)
        assert out.shape == (B, nhead, L, D // nhead)
        assert not torch.allclose(x, out)

    def test_forward_3d(self, device):
        B, T, H, W, D = 2, 8, 8, 8, 64
        L = T * H * W
        nhead = 4
        layer = MixedRoPE(D, nhead, (T, H, W)).to(device)

        grid = create_grid((T, H, W), device=device).expand(B, -1, -1)
        x = torch.randn(B, nhead, L, D // nhead, device=device)
        out = layer(x, grid)
        assert out.shape == (B, nhead, L, D // nhead)
        assert not torch.allclose(x, out)

    def test_backward(self, device):
        B, H, W, D = 2, 8, 8, 64
        L = H * W
        nhead = 4
        layer = MixedRoPE(D, nhead, (H, W)).to(device)

        grid = create_grid((H, W), device=device).expand(B, -1, -1)
        x = torch.randn(B, nhead, L, D // nhead, device=device)
        out = layer(x, grid)
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()
