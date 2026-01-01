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
        pos_enc="learnable",
        dtype=torch.float32,  # Use FP32 for test backward compatibility
    )
    return config


def assert_all_requires_grad(module: Any):
    assert isinstance(module, nn.Module)
    assert all(p.requires_grad for p in module.parameters())


def assert_none_requires_grad(module: Any):
    assert isinstance(module, nn.Module)
    assert not any(p.requires_grad for p in module.parameters())


class TestViT:
    def test_default_dtype_is_bfloat16(self):
        """Verify that the default master weight dtype is BF16."""
        config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            pos_enc="learnable",
        )
        assert config.dtype == torch.bfloat16
        model = ViT(config)
        # All parameters should be BF16
        for name, param in model.named_parameters():
            assert param.dtype == torch.bfloat16, f"{name} has dtype {param.dtype}, expected bfloat16"

    def test_custom_dtype_float32(self):
        """Verify that dtype can be overridden to FP32."""
        config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            pos_enc="learnable",
            dtype=torch.float32,
        )
        assert config.dtype == torch.float32
        model = ViT(config)
        # All parameters should be FP32
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, f"{name} has dtype {param.dtype}, expected float32"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_all_params_initialized_with_correct_dtype_and_device(self, device, dtype):
        """Verify all parameters (including heads) are initialized with correct dtype and device."""
        config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=2,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            pos_enc="learnable",
            num_register_tokens=2,
            num_cls_tokens=1,
            dtype=dtype,
            heads={"cls": HeadConfig(out_features=10), "seg": HeadConfig(out_features=64)},
        )
        model = ViT(config, device=device)

        # Check all parameters have correct dtype and device
        for name, param in model.named_parameters():
            assert param.dtype == dtype, f"Parameter {name} has dtype {param.dtype}, expected {dtype}"
            assert param.device == device, f"Parameter {name} on device {param.device}, expected {device}"

        # Check all buffers have correct device (buffers may have different dtypes, e.g. bool for masks)
        for name, buf in model.named_buffers():
            assert buf.device == device, f"Buffer {name} on device {buf.device}, expected {device}"
            # Only check dtype for float buffers
            if buf.dtype.is_floating_point:
                assert buf.dtype == dtype, f"Buffer {name} has dtype {buf.dtype}, expected {dtype}"

        # Verify specific components to ensure factory_kwargs propagated correctly
        assert model.stem.patch.weight.dtype == dtype, "Stem conv weight has wrong dtype"
        assert model.stem.patch.weight.device == device, "Stem conv weight on wrong device"
        assert model.register_tokens.dtype == dtype, "Register tokens have wrong dtype"
        assert model.cls_tokens.dtype == dtype, "CLS tokens have wrong dtype"
        assert model.output_norm.weight.dtype == dtype, "Output norm weight has wrong dtype"

        # Check heads explicitly
        for head_name in ["cls", "seg"]:
            head = model.heads[head_name]
            for name, param in head.named_parameters():
                assert param.dtype == dtype, f"Head {head_name} param {name} has dtype {param.dtype}, expected {dtype}"
                assert param.device == device, (
                    f"Head {head_name} param {name} on device {param.device}, expected {device}"
                )

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
            heads={"cls": HeadConfig(out_features=128)},
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)
        # Pool visual tokens before head projection
        pooled = out.visual_tokens.mean(dim=1)
        pred = model.heads["cls"](pooled)
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

    def test_forward_returns_tokenized_size(self, device, config):
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)
        expected_size = model.stem.tokenized_size(config.img_size)
        assert out.tokenized_size == tuple(expected_size)
        grid = out.visual_tokens_as_grid
        assert grid.shape == (2, *expected_size, config.hidden_size)


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
        expected = "ViTFeatures(cls_tokens=(2, 1, 128), register_tokens=(2, 0, 128), visual_tokens=(2, 32, 128))"
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

    @pytest.mark.parametrize(
        "tokenized_size",
        [
            pytest.param((8, 16), id="2d"),
            pytest.param((4, 8, 16), id="3d"),
        ],
    )
    def test_visual_tokens_as_grid(self, tokenized_size):
        B, C = 2, 64
        L = math.prod(tokenized_size)
        dense_features = torch.randn(B, L, C)
        features = ViTFeatures(dense_features, num_register_tokens=0, num_cls_tokens=0, tokenized_size=tokenized_size)
        grid = features.visual_tokens_as_grid
        assert grid.shape == (B, *tokenized_size, C)
        assert_close(grid.view(B, L, C), dense_features)

    def test_visual_tokens_as_grid_with_prefix_tokens(self):
        B, C = 2, 64
        tokenized_size = (8, 16)
        L = math.prod(tokenized_size)
        num_cls_tokens, num_register_tokens = 1, 2
        total_tokens = num_cls_tokens + num_register_tokens + L
        dense_features = torch.randn(B, total_tokens, C)
        features = ViTFeatures(dense_features, num_register_tokens, num_cls_tokens, tokenized_size)
        grid = features.visual_tokens_as_grid
        assert grid.shape == (B, *tokenized_size, C)
        assert_close(grid.view(B, L, C), features.visual_tokens)

    def test_visual_tokens_as_grid_no_tokenized_size_raises(self):
        features = ViTFeatures(torch.randn(2, 32, 128), 0, 0)
        with pytest.raises(ValueError, match="tokenized_size is not set"):
            _ = features.visual_tokens_as_grid

    def test_tokenized_size_property(self):
        tokenized_size = (14, 14)
        features = ViTFeatures(torch.randn(2, 196, 128), 0, 0, tokenized_size=tokenized_size)
        assert features.tokenized_size == tokenized_size

    def test_tokenized_size_none_by_default(self):
        features = ViTFeatures(torch.randn(2, 32, 128), 0, 0)
        assert features.tokenized_size is None

    def test_apply_preserves_tokenized_size(self):
        tokenized_size = (8, 8)
        features = ViTFeatures(torch.randn(2, 64, 128), 0, 0, tokenized_size=tokenized_size)
        features_plus_one = features.apply(lambda x: x + 1)
        assert features_plus_one.tokenized_size == tokenized_size

    def test_from_separate_features_with_tokenized_size(self):
        cls_tokens = torch.randn(2, 1, 128)
        register_tokens = torch.randn(2, 2, 128)
        visual_tokens = torch.randn(2, 64, 128)
        tokenized_size = (8, 8)
        features = ViTFeatures.from_separate_features(
            cls_tokens, register_tokens, visual_tokens, tokenized_size=tokenized_size
        )
        assert features.tokenized_size == tokenized_size
        grid = features.visual_tokens_as_grid
        assert grid.shape == (2, 8, 8, 128)
