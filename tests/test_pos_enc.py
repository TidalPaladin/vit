import pytest
import torch
from torch.testing import assert_close

from vit.pos_enc import LearnableFourierFeatures, LearnablePosition, RelativeFactorizedPosition, create_grid


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

    def test_deterministic(self):
        D = 16
        torch.random.manual_seed(0)
        layer = LearnablePosition(D, (8, 8), dropout=0.1)
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


class TestRelativeFactorizedPosition:

    def test_forward(self, device):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to(device)
        out = layer((8, 8))
        assert out.shape == (1, 64, D)
        assert out.device == device

    def test_backward(self, device):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to(device)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()


class TestLearnableFourierFeatures:

    def test_forward(self, device):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = LearnableFourierFeatures(C, D).to(device)
        out = layer((8, 8))
        assert out.shape == (1, 64, D)
        assert out.device == device

    def test_backward(self, device):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = LearnableFourierFeatures(C, D).to(device)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize(
        "dropout,jitter",
        [
            (0.0, True),
            (0.1, False),
            (0.1, True),
        ],
    )
    def test_deterministic(self, device, dropout, jitter):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = LearnableFourierFeatures(C, D, dropout=dropout, jitter=jitter).to(device)

        layer.eval()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert_close(out1, out2)

        layer.train()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert not torch.allclose(out1, out2)
