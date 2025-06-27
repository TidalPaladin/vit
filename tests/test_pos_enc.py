import pytest
import torch
from torch.testing import assert_close

from vit.pos_enc import FourierPosition, LearnablePosition, create_grid


class TestLearnablePosition:

    @pytest.mark.parametrize("fourier_init", [True, False])
    def test_forward(self, device, fourier_init):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8), fourier_init=fourier_init).to(device)
        out = layer((8, 8))
        assert out.shape == (1, 64, D)
        assert out.device == device

    def test_backward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_interpolate(self):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8))
        out = layer((12, 12))
        assert out.shape == (1, 144, D)

    def test_expand_positions(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8)).to(device)
        layer.expand_positions((12, 12))
        assert layer.positions.shape == (144, D)
        assert layer.positions.requires_grad is True
        assert layer.positions.device == device
        out = layer((12, 12))
        assert out.shape == (1, 144, D)

    def test_deterministic(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8), dropout=0.1).to(device)
        layer.eval()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert_close(out1, out2)

        layer.train()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert not torch.allclose(out1, out2)


class TestFourierPosition:

    def test_forward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = FourierPosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        assert out.shape == (1, 64, D)
        assert out.device == device

    def test_backward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = FourierPosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_interpolate(self):
        D = 16
        torch.random.manual_seed(0)
        layer = FourierPosition(D, (8, 8))
        out = layer((12, 12))
        assert out.shape == (1, 144, D)

    def test_deterministic(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = FourierPosition(D, (8, 8)).to(device)
        layer.eval()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert_close(out1, out2)

        layer.train()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert not torch.allclose(out1, out2)


@pytest.mark.parametrize("normalize", [True, False])
def test_create_grid(normalize, device):
    dims = (4, 4)
    grid = create_grid(dims, normalize=normalize, device=device)
    assert grid.shape == (1, 16, 2)
    if normalize:
        assert torch.all(grid[0, 0] == torch.tensor([-1.0, -1.0], device=device))
        assert torch.all(grid[0, -1] == torch.tensor([1.0, 1.0], device=device))
    else:
        assert torch.all(grid[0, 0] == torch.tensor([0, 0], device=device))
        assert torch.all(grid[0, -1] == torch.tensor([3, 3], device=device))
