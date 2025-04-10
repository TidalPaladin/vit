import pytest
import torch
from torch.testing import assert_close

from vit.pos_enc import RelativeFactorizedPosition, create_grid


@pytest.fixture(params=["pytorch", pytest.param("te", marks=pytest.mark.cuda)])
def backend(request):
    return request.param


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


class TestRelativeFactorizedPosition:

    def test_forward(self, backend):
        C, D = 2, 16
        torch.random.manual_seed(0)
        device_type = "cuda" if backend == "te" else "cpu"
        layer = RelativeFactorizedPosition(C, D, backend=backend).to(device_type)
        out = layer((8, 8))
        L = 64
        assert out.shape == (1, L, D)

    def test_backward(self, backend):
        C, D = 2, 16
        torch.random.manual_seed(0)
        device_type = "cuda" if backend == "te" else "cpu"
        layer = RelativeFactorizedPosition(C, D, backend=backend).to(device_type)
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_baseline(self):
        C, D = 2, 16
        torch.random.manual_seed(0)
        baseline = RelativeFactorizedPosition(C, D, backend="pytorch").to("cuda")
        layer = RelativeFactorizedPosition(C, D, backend="te").to("cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer((8, 8))
            y_baseline = baseline((8, 8))
        assert_close(y, y_baseline, atol=1e-4, rtol=0)
