import pytest
import torch
from torch.testing import assert_close

from vit.pos_enc import RelativeFactorizedPosition as RelativeFactorizedPositionBaseline


try:
    from vit.te.pos_enc import RelativeFactorizedPosition
except ImportError:
    pytest.skip("Transformer Engine is not installed", allow_module_level=True)


class TestRelativeFactorizedPosition:

    def test_forward(self):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to("cuda")
        out = layer((8, 8))
        L = 64
        assert out.shape == (1, L, D)

    def test_backward(self):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to("cuda")
        out = layer((8, 8))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_baseline(self):
        C, D = 2, 16
        torch.random.manual_seed(0)
        baseline = RelativeFactorizedPositionBaseline(C, D).to("cuda")
        layer = RelativeFactorizedPosition(C, D).to("cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer((8, 8))
            y_baseline = baseline((8, 8))
        assert_close(y, y_baseline, atol=1e-4, rtol=0)
