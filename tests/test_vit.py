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

    def test_forward(self, config):
        x = torch.randn(1, 3, 224, 224)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = model(x)
        assert out.shape == (1, 196, 128)

    def test_backward(self, config):
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"
