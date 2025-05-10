import pytest
import torch

from vit.pos_enc import create_grid


@pytest.mark.parametrize("normalize", [True, False])
def test_create_grid(normalize):
    dims = (4, 4)
    grid = create_grid(dims, normalize=normalize)
    assert grid.shape == (1, 16, 2)
    if normalize:
        assert torch.all(grid[0, 0] == torch.tensor([-1.0, -1.0]))
        assert torch.all(grid[0, -1] == torch.tensor([1.0, 1.0]))
    else:
        assert torch.all(grid[0, 0] == torch.tensor([0, 0]))
        assert torch.all(grid[0, -1] == torch.tensor([3, 3]))
