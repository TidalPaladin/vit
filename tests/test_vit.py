import math
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vit.head import HeadConfig
from vit.matryoshka import MatryoshkaConfig
from vit.vit import ViT, ViTConfig


@pytest.fixture(params=[pytest.param(False, id="2d"), pytest.param(True, id="3d")])
def config(request):
    is_3d = request.param
    config = ViTConfig(
        in_channels=3,
        patch_size=(16, 16) if not is_3d else (16, 16, 16),
        img_size=(224, 224) if not is_3d else (32, 224, 224),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
    )
    return config


def assert_all_requires_grad(module: Any):
    assert isinstance(module, nn.Module)
    assert all(p.requires_grad for p in module.parameters())


def assert_none_requires_grad(module: Any):
    assert isinstance(module, nn.Module)
    assert not any(p.requires_grad for p in module.parameters())


class TestViT:

    def test_config_from_yaml_str(self, config):
        config_str = config.to_yaml()
        config_from_str = ViTConfig.from_yaml(config_str)
        assert config == config_from_str

    def test_config_from_yaml_path(self, config, tmp_path):
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            f.write(config.to_yaml())
        config_from_path = ViTConfig.from_yaml(path)
        assert config == config_from_path

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, device, config, num_register_tokens, dtype):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.shape == (2, L, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_return_register_tokens(self, device, config, num_register_tokens, dtype):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x, return_register_tokens=True)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.shape == (2, L + num_register_tokens, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    def test_forward_masked(self, device, config, num_register_tokens):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        mask = model.create_mask(x, 0.5, 1)
        out = model(x, mask)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.shape == (2, L // 2, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, device, config, num_register_tokens, dtype):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
        )
        x = torch.randn(2, 3, *config.img_size, device=device, requires_grad=True)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_mlp_requires_grad(self, config):
        model = ViT(config)
        for block in model.blocks:
            assert_all_requires_grad(block.mlp)
            assert_all_requires_grad(block.self_attention)
        model.mlp_requires_grad_(False)
        for block in model.blocks:
            assert_none_requires_grad(block.mlp)
            assert_all_requires_grad(block.self_attention)

    def test_self_attention_requires_grad(self, config):
        model = ViT(config)
        for block in model.blocks:
            assert_all_requires_grad(block.mlp)
            assert_all_requires_grad(block.self_attention)
        model.self_attention_requires_grad_(False)
        for block in model.blocks:
            assert_all_requires_grad(block.mlp)
            assert_none_requires_grad(block.self_attention)

    def test_with_head(self, device, config):
        config = replace(
            config,
            heads={"cls": HeadConfig(head_type="linear", pool_type="avg", out_dim=128, stop_gradient=False)},
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)
        pred = model.heads["cls"](out)
        assert pred.shape == (2, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_attention_weights(self, device, config, num_register_tokens, dtype):
        torch.random.manual_seed(0)
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
        )
        B = 2
        H = config.num_attention_heads
        x = torch.randn(B, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        size = model.stem.tokenized_size(config.img_size)
        L = math.prod(size)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            weights = model.forward_attention_weights(x)

        for name, weight in weights.items():
            assert (weight >= 0).all(), f"{name} has negative weights"
            assert (weight <= 1).all(), f"{name} has weights greater than 1"
            assert weight.shape == (B, H, L + num_register_tokens, *size)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize(
        "feature_frac,feedforward_frac,heads_frac", [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 0.5)]
    )
    @pytest.mark.parametrize("layer_scale", [None, 0.1])
    def test_matryoshka_vit(
        self, device, config, num_register_tokens, feature_frac, feedforward_frac, heads_frac, layer_scale
    ):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            layer_scale=layer_scale,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        matryoshka = MatryoshkaConfig(
            feature_frac=feature_frac, feedforward_frac=feedforward_frac, heads_frac=heads_frac
        )
        model = ViT(config).to(device)
        out = model(x, matryoshka=matryoshka)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        D_out = int(model.config.hidden_size * feature_frac)
        assert out.shape == (2, L, D_out)
