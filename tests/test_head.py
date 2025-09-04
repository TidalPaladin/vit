import pytest
import torch
import torch.nn as nn

from vit.head import Head, HeadConfig, MLPHead
from vit.rope import RopePositionEmbedding
from vit.vit import ViTConfig


class TestHeadConfig:

    @pytest.mark.parametrize("head_type", ["linear", "mlp"])
    def test_instantiate(self, head_type):
        config = HeadConfig(head_type=head_type, pool_type="avg", out_dim=128, stop_gradient=False)
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        if head_type == "linear":
            assert isinstance(model, Head)
        elif head_type == "mlp":
            assert isinstance(model, MLPHead)
        else:
            raise ValueError(f"Invalid head type: {head_type}")


class TestHead:

    @pytest.mark.parametrize("pool_type", ["avg", "max", "attentive", "none"])
    @pytest.mark.parametrize("out_dim", [None, 128, 32])
    def test_forward(self, device, pool_type, out_dim):
        x = torch.randn(2, 196, 128, device=device)
        model = Head(128, pool_type, out_dim, 128 // 16, False).to(device)
        out = model(x)
        if pool_type == "none":
            assert out.shape == (2, 196, out_dim or 128)
        else:
            assert out.shape == (2, out_dim or 128)

    def test_backward(self, device):
        x = torch.randn(2, 196, 128, device=device, requires_grad=True)
        model = Head(128, "avg", 128, 128 // 16, False).to(device)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_stop_gradient(self, device):
        x = torch.randn(2, 196, 128, device=device, requires_grad=True)
        layer = nn.Linear(128, 128).to(device)
        model = Head(128, "avg", 128, 128 // 16, True).to(device)
        out = model(layer(x))
        out.sum().backward()
        assert layer.weight.grad is None
        assert layer.bias.grad is None

    def test_forward_rope(self, device):
        out_dim = 10
        pool_type = "attentive"
        dim = 128
        num_heads = dim // 16
        x = torch.randn(2, 144, 128, device=device)
        model = Head(dim, pool_type, out_dim, num_heads, False).to(device)
        rope = RopePositionEmbedding(dim, num_heads=num_heads, base=100).to(device)
        rope_angle = rope(H=12, W=12)
        out = model(x, rope_angle)
        if pool_type == "none":
            assert out.shape == (2, 196, out_dim or 128)
        else:
            assert out.shape == (2, out_dim or 128)


class TestMLPHead:

    @pytest.mark.parametrize("pool_type", ["avg", "max", "attentive", "none"])
    @pytest.mark.parametrize("out_dim", [None, 128, 32])
    def test_forward(self, device, pool_type, out_dim):
        x = torch.randn(2, 196, 128, device=device)
        model = MLPHead(128, 256, "gelu", pool_type, out_dim, 128 // 16, False).to(device)
        out = model(x)
        if pool_type == "none":
            assert out.shape == (2, 196, out_dim or 128)
        else:
            assert out.shape == (2, out_dim or 128)

    def test_backward(self, device):
        x = torch.randn(2, 196, 128, device=device, requires_grad=True)
        model = MLPHead(128, 256, "gelu", "avg", 128, 128 // 16, False).to(device)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_stop_gradient(self, device):
        x = torch.randn(2, 196, 128, device=device, requires_grad=True)
        layer = nn.Linear(128, 128).to(device)
        model = MLPHead(128, 256, "gelu", "avg", 128, 128 // 16, True).to(device)
        out = model(layer(x))
        out.sum().backward()
        assert layer.weight.grad is None
        assert layer.bias.grad is None

    def test_forward_rope(self, device):
        out_dim = 10
        pool_type = "attentive"
        dim = 128
        num_heads = dim // 16
        x = torch.randn(2, 144, 128, device=device)
        model = MLPHead(dim, 256, "gelu", pool_type, out_dim, num_heads, False).to(device)
        rope = RopePositionEmbedding(dim, num_heads=num_heads, base=100).to(device)
        rope_angle = rope(H=12, W=12)
        out = model(x, rope_angle)
        if pool_type == "none":
            assert out.shape == (2, 196, out_dim or 128)
        else:
            assert out.shape == (2, out_dim or 128)
