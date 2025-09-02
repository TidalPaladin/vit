import math
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vit.head import HeadConfig
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
    @pytest.mark.parametrize("cls_token", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "activation,glu_limit,glu_extra_bias",
        [
            ("srelu", None, None),
            ("openswiglu", 7.0, 1.0),
        ],
    )
    def test_forward(
        self, device, config, num_register_tokens, cls_token, dtype, activation, glu_limit, glu_extra_bias
    ):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            cls_token=cls_token,
            activation=activation,
            glu_limit=glu_limit,
            glu_extra_bias=glu_extra_bias,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)

        # Test dictionary output format
        assert isinstance(out, dict)
        assert set(out.keys()) == {"x_reg", "x_cls", "x_visual", "x_pre_norm"}

        # Test shapes
        L = math.prod(model.stem.tokenized_size(config.img_size))
        batch_size = 2
        hidden_size = 128

        assert out["x_reg"].shape == (batch_size, num_register_tokens, hidden_size)
        assert out["x_cls"].shape == (batch_size, 1 if cls_token else 0, hidden_size)
        assert out["x_visual"].shape == (batch_size, L, hidden_size)
        assert out["x_pre_norm"].shape == (batch_size, num_register_tokens + (1 if cls_token else 0) + L, hidden_size)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("cls_token", [True, False])
    def test_forward_with_rope(self, device, config, masked, cls_token):
        x = torch.randn(3, 3, *config.img_size, device=device)
        config = replace(config, pos_enc="rope", cls_token=cls_token)
        if len(config.patch_size) != 2:
            pytest.skip("RoPE not supported for non-2D input")
        model = ViT(config).to(device)
        assert model.rope is not None
        mask = model.create_mask(x, 0.5, 1) if masked else None
        with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            out = model(x, mask=mask)

        # Test dictionary output format
        assert isinstance(out, dict)
        assert set(out.keys()) == {"x_reg", "x_cls", "x_visual", "x_pre_norm"}

        # Test shapes
        L = math.prod(model.stem.tokenized_size(config.img_size))
        L_visual = L if not masked else L // 2
        batch_size = 3
        hidden_size = 128
        num_register_tokens = config.num_register_tokens

        assert out["x_reg"].shape == (batch_size, num_register_tokens, hidden_size)
        assert out["x_cls"].shape == (batch_size, 1 if cls_token else 0, hidden_size)
        assert out["x_visual"].shape == (batch_size, L_visual, hidden_size)
        assert out["x_pre_norm"].shape == (
            batch_size,
            num_register_tokens + (1 if cls_token else 0) + L_visual,
            hidden_size,
        )

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("cls_token", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_output_shapes_with_tokens(self, device, config, num_register_tokens, cls_token, dtype):
        """Test that output dictionary contains correct shapes for all token types."""
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            cls_token=cls_token,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)

        # Test dictionary structure
        assert isinstance(out, dict)
        assert set(out.keys()) == {"x_reg", "x_cls", "x_visual", "x_pre_norm"}

        # Test individual tensor shapes
        L = math.prod(model.stem.tokenized_size(config.img_size))
        batch_size = 2
        hidden_size = 128

        assert out["x_reg"].shape == (batch_size, num_register_tokens, hidden_size)
        assert out["x_cls"].shape == (batch_size, 1 if cls_token else 0, hidden_size)
        assert out["x_visual"].shape == (batch_size, L, hidden_size)
        assert out["x_pre_norm"].shape == (batch_size, num_register_tokens + (1 if cls_token else 0) + L, hidden_size)

        # Test that pre-norm contains all tokens
        expected_total_length = num_register_tokens + (1 if cls_token else 0) + L
        assert out["x_pre_norm"].shape[1] == expected_total_length

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("cls_token", [True, False])
    def test_forward_masked(self, device, config, num_register_tokens, cls_token):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            cls_token=cls_token,
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        mask = model.create_mask(x, 0.5, 1)
        out = model(x, mask)

        # Test dictionary output format
        assert isinstance(out, dict)
        assert set(out.keys()) == {"x_reg", "x_cls", "x_visual", "x_pre_norm"}

        # Test shapes with masking
        L = math.prod(model.stem.tokenized_size(config.img_size))
        L_masked = L // 2
        batch_size = 2
        hidden_size = 128

        assert out["x_reg"].shape == (batch_size, num_register_tokens, hidden_size)
        assert out["x_cls"].shape == (batch_size, 1 if cls_token else 0, hidden_size)
        assert out["x_visual"].shape == (batch_size, L_masked, hidden_size)
        assert out["x_pre_norm"].shape == (
            batch_size,
            num_register_tokens + (1 if cls_token else 0) + L_masked,
            hidden_size,
        )

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("cls_token", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, device, config, num_register_tokens, cls_token, dtype):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            cls_token=cls_token,
        )
        x = torch.randn(2, 3, *config.img_size, device=device, requires_grad=True)
        model = ViT(config).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            out = model(x)

        # Sum all outputs for backward pass
        loss = out["x_visual"].sum() + out["x_pre_norm"].sum()
        if out["x_reg"].numel() > 0:
            loss += out["x_reg"].sum()
        if out["x_cls"].numel() > 0:
            loss += out["x_cls"].sum()

        loss.backward()
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

    def test_backbone_requires_grad(self, config):
        model = ViT(config)
        model.backbone_requires_grad_(False)
        for name, param in model.named_parameters():
            assert name.startswith("head") or not param.requires_grad, f"{name} is trainable"

    @pytest.mark.parametrize("cls_token", [True, False])
    def test_with_head(self, device, config, cls_token):
        config = replace(
            config,
            cls_token=cls_token,
            heads={"cls": HeadConfig(head_type="linear", pool_type="avg", out_dim=128, stop_gradient=False)},
        )
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)

        # Test that output is a dictionary
        assert isinstance(out, dict)

        # Test head with visual tokens
        pred = model.heads["cls"](out["x_visual"])
        assert pred.shape == (2, 128)

    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("cls_token", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_attention_weights(self, device, config, num_register_tokens, cls_token, dtype):
        torch.random.manual_seed(0)
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            cls_token=cls_token,
        )
        B = 2
        H = config.num_attention_heads
        x = torch.randn(B, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        size = model.stem.tokenized_size(config.img_size)
        L = math.prod(size)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            weights = model.forward_attention_weights(x)

        # Expected total sequence length includes register tokens, cls token (if enabled), and visual tokens
        expected_seq_len = num_register_tokens + (1 if cls_token else 0) + L

        for name, weight in weights.items():
            assert (weight >= 0).all(), f"{name} has negative weights"
            assert (weight <= 1).all(), f"{name} has weights greater than 1"
            assert weight.shape == (B, H, expected_seq_len, *size)

    def test_cls_token_enabled(self, device, config):
        """Test that CLS token is properly included when enabled."""
        config = replace(config, cls_token=True)
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)

        # CLS token should have shape (batch_size, 1, hidden_size)
        assert out["x_cls"].shape == (2, 1, 128)
        assert out["x_cls"].numel() > 0  # Should not be empty

    def test_cls_token_disabled(self, device, config):
        """Test that CLS token is properly excluded when disabled."""
        config = replace(config, cls_token=False)
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)

        # CLS token should have shape (batch_size, 0, hidden_size)
        assert out["x_cls"].shape == (2, 0, 128)
        assert out["x_cls"].numel() == 0  # Should be empty

    def test_output_consistency(self, device, config):
        """Test that pre-norm output is consistent with separated tokens."""
        config = replace(config, cls_token=True, num_register_tokens=2)
        x = torch.randn(2, 3, *config.img_size, device=device)
        model = ViT(config).to(device)
        out = model(x)

        # Pre-norm should be concatenation of reg + cls + visual tokens
        expected_pre_norm = torch.cat([out["x_reg"], out["x_cls"], out["x_visual"]], dim=1)

        # Apply normalization to compare (since x_pre_norm is before norm, others are after)
        pre_norm_normalized = model.output_norm(out["x_pre_norm"])

        # Shapes should match
        assert pre_norm_normalized.shape == expected_pre_norm.shape
        assert out["x_pre_norm"].shape[1] == out["x_reg"].shape[1] + out["x_cls"].shape[1] + out["x_visual"].shape[1]
