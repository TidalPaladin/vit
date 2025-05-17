from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn as nn

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
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_forward(self, device, config, num_register_tokens, dtype, use_fourier_features):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            use_fourier_features=use_fourier_features,
        )
        x = torch.randn(2, 3, 224, 224, device=device)
        pos = torch.randn(2, 196, 2, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)
        assert out.shape == (2, 196, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_forward_masked(self, device, config, num_register_tokens, dtype, use_fourier_features):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            use_fourier_features=use_fourier_features,
        )
        x = torch.randn(2, 3, 224, 224, device=device)
        pos = torch.randn(2, 196, 2, device=device)
        model = ViT(config).to(device)
        mask = model.create_mask(x, 0.5, 1)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x, mask)
        assert out.shape == (2, 196 // 2, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fourier_features", [True, False])
    def test_backward(self, device, config, num_register_tokens, dtype, use_fourier_features):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            use_fourier_features=use_fourier_features,
        )
        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
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
