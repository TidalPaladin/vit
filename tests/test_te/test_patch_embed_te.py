import math

import pytest
import torch
from torch.testing import assert_close

from vit.patch_embed import PatchEmbed2d as PatchEmbed2dBaseline


try:
    from vit.te.patch_embed import PatchEmbed2d
except ImportError:
    pytest.skip("Transformer Engine is not installed", allow_module_level=True)


class TestPatchEmbed2d:

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, normalization):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4), normalization=normalization).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    def test_forward_additional_features(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        additional_features = torch.randn(B, H // 4 * W // 4, D_model, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x, additional_features)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    def test_backward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_baseline(self, normalization):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        torch.random.manual_seed(0)
        baseline = PatchEmbed2dBaseline(C, D_model, (4, 4), normalization=normalization).to("cuda")
        layer = PatchEmbed2d(C, D_model, (4, 4), normalization=normalization).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline, atol=1e-4, rtol=0)
