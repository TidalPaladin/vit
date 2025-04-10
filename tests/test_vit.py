from dataclasses import replace

import pytest
import torch

from vit.vit import ViT, ViTConfig


@pytest.fixture
def config():
    config = ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
    )
    return config


class TestViT:

    def test_forward(self, config):
        x = torch.randn(1, 3, 224, 224)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out, cls_token = model(x)
        assert out.shape == (1, 196, 128)
        assert cls_token.shape == (1, 128)

    def test_forward_with_encoder_output(self, config):
        x = torch.randn(1, 3, 224, 224)
        encoder_output = torch.randn(1, 64, 128)
        config = replace(config, decoder=True)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out, cls_token = model(x, encoder_output=encoder_output)
        assert out.shape == (1, 196, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward_with_encoder_output(self, config, checkpoint):
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        encoder_output = torch.randn(1, 64, 128)
        config = replace(config, decoder=True, checkpoint=checkpoint)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, cls_token = model(x, encoder_output=encoder_output)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


#    @pytest.mark.skip(reason="Incomplete")
#    def test_baseline(self, config):
#        B, C, H, W = 2, 3, 64, 64
#        torch.random.manual_seed(0)
#        baseline = ViTBaseline(config).to("cuda")
#        layer = ViT(config).to("cuda")
#        x = torch.randn(B, C, H, W, device="cuda")
#
#        layer.eval()
#        baseline.eval()
#
#        # Sync weights
#        for name, param in baseline.named_parameters():
#            layer.get_parameter(name).data.copy_(param.data)
#
#        with torch.autocast(device_type="cuda", dtype=torch.float32):
#            y = layer(x)
#            y_baseline = baseline(x)
#        assert_close(y, y_baseline, atol=1e-4, rtol=0)
#
