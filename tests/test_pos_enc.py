import pytest
import torch

from vit.pos_enc import FourierPosition, HSirenPosition, HybridPosition, LearnablePosition, create_grid


class TestLearnablePosition:

    def test_forward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8)).to(device)
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


class TestHybridPosition:

    def test_forward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = HybridPosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        assert out.shape == (1, 64, D)
        assert out.device == device

    def test_backward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = HybridPosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_interpolate(self):
        D = 16
        torch.random.manual_seed(0)
        layer = HybridPosition(D, (8, 8))
        out = layer((12, 12))
        assert out.shape == (1, 144, D)

    def test_expand_positions(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = HybridPosition(D, (8, 8)).to(device)
        layer.expand_positions((12, 12))
        assert layer.learnable.positions.shape == (144, D)
        assert layer.learnable.positions.requires_grad is True
        assert layer.learnable.positions.device == device
        out = layer((12, 12))
        assert out.shape == (1, 144, D)


class TestHSirenPosition:

    def test_forward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = HSirenPosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        assert out.shape == (1, 64, D)
        assert out.device == device

    def test_backward(self, device):
        D = 16
        torch.random.manual_seed(0)
        layer = HSirenPosition(D, (8, 8)).to(device)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_interpolate(self):
        D = 16
        torch.random.manual_seed(0)
        layer = HSirenPosition(D, (8, 8))
        out = layer((12, 12))
        assert out.shape == (1, 144, D)
