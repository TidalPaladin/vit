import math
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close
from torchao.quantization import Int8WeightOnlyConfig

from vit.head import HeadConfig
from vit.vit import ViT, ViTConfig, ViTFeatures


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
    @pytest.mark.parametrize(
        "activation,glu_limit,glu_extra_bias",
        [
            ("srelu", None, None),
            ("openswiglu", 7.0, 1.0),
        ],
    )
    def test_forward(self, device, config, num_register_tokens, dtype, activation, glu_limit, glu_extra_bias):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            activation=activation,
            glu_limit=glu_limit,
            glu_extra_bias=glu_extra_bias,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.visual_tokens.shape == (2, L, 128)

    @pytest.mark.parametrize("masked", [False, True])
    def test_forward_with_rope(self, device, config, masked):
        x = torch.randn(3, 3, *config.img_size, device=device)
        config = replace(config, pos_enc="rope")
        if len(config.patch_size) != 2:
            pytest.skip("RoPE not supported for non-2D input")
        model = ViT(config).to(device)
        assert model.rope is not None
        mask = model.create_mask(x, 0.5, 1) if masked else None
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            out = model(x, mask=mask)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.visual_tokens.shape == (3, L if not masked else L // 2, 128)

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
            out = model(x)
        math.prod(model.stem.tokenized_size(config.img_size))
        assert out.register_tokens.shape == (2, num_register_tokens, 128)

    @pytest.mark.parametrize("num_cls_tokens", [0, 1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_return_cls_tokens(self, device, config, num_cls_tokens, dtype):
        config = replace(
            config,
            num_cls_tokens=num_cls_tokens,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)
        math.prod(model.stem.tokenized_size(config.img_size))
        assert out.cls_tokens.shape == (2, num_cls_tokens, 128)

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
        assert out.visual_tokens.shape == (2, L // 2, 128)

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
        out.dense_features.sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
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

    def test_backbone_requires_grad(self, config):
        model = ViT(config)
        model.backbone_requires_grad_(False)
        for name, param in model.named_parameters():
            assert name.startswith("head") or not param.requires_grad, f"{name} is trainable"

    def test_with_head(self, device, config):
        config = replace(
            config,
            heads={"cls": HeadConfig(head_type="linear", pool_type="avg", out_dim=128, stop_gradient=False)},
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)
        pred = model.heads["cls"](out.visual_tokens)
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

    def test_quantization(self, device, config):
        torch.random.manual_seed(0)
        D = config.hidden_size
        H = D // 16
        config = replace(config, hidden_size=D, num_attention_heads=H)
        model = ViT(config).to(device)
        model.eval()
        quantized_model = deepcopy(model)
        quantized_model.apply_quantization(mlp_quantization_config=Int8WeightOnlyConfig())

        x = torch.randn(2, 3, *config.img_size, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = model(x)
            y_quant = quantized_model(x)
        assert_close(y.visual_tokens, y_quant.visual_tokens, atol=1e-2, rtol=0)


class TestViTFeatures:

    def test_iter(self):
        num_cls_tokens = 1
        num_register_tokens = 2
        num_visual_tokens = 128
        total_tokens = num_cls_tokens + num_register_tokens + num_visual_tokens
        features = ViTFeatures(torch.randn(2, total_tokens, 128), num_register_tokens, num_cls_tokens)
        cls_tokens, register_tokens, visual_tokens = features
        assert cls_tokens.shape == (2, 1, 128)
        assert register_tokens.shape == (2, num_register_tokens, 128)
        assert visual_tokens.shape == (2, num_visual_tokens, 128)

    def test_apply(self):
        features = ViTFeatures(torch.randn(2, 32, 128), 0, 0)
        features_plus_one = features.apply(lambda x: x + 1)
        assert_close(features_plus_one.dense_features, features.dense_features + 1)

    def test_repr(self):
        features = ViTFeatures(torch.randn(2, 33, 128), 0, 1)
        expected = f"ViTFeatures(cls_tokens=(2, 1, 128), register_tokens=(2, 0, 128), visual_tokens=(2, 32, 128))"
        assert isinstance(repr(features), str)
        assert repr(features) == expected

    def test_from_separate_features(self):
        cls_tokens = torch.randn(2, 1, 128)
        register_tokens = torch.randn(2, 2, 128)
        visual_tokens = torch.randn(2, 128, 128)
        features = ViTFeatures.from_separate_features(cls_tokens, register_tokens, visual_tokens)
        assert_close(features.dense_features, torch.cat([cls_tokens, register_tokens, visual_tokens], dim=1))
        assert features.num_cls_tokens == 1
        assert features.num_register_tokens == 2
        assert features.visual_tokens.shape == (2, 128, 128)
