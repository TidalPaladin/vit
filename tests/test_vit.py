from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from vit.helpers import try_import_te
from vit.vit import ViT, ViTConfig


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


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

    @pytest.mark.parametrize("decoder", [False, True])
    def test_non_causal_default(self, config, decoder):
        if te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend="te", decoder=decoder)
        model = ViT(config)
        assert model.create_encoder_layer().self_attn_mask_type == "no_mask"  # type: ignore
        assert model.create_decoder_layer().enc_dec_attn_mask_type == "no_mask"  # type: ignore

        for block in model.blocks:
            assert block.self_attn_mask_type == "no_mask"  # type: ignore
            if hasattr(block, "inter_attention"):
                assert block.enc_dec_attn_mask_type == "no_mask"  # type: ignore

    @pytest.mark.parametrize("convnext_patch_embed", [False, True])
    @pytest.mark.parametrize("num_register_tokens", [0, 1, 2])
    @pytest.mark.parametrize("num_cls_tokens", [0, 1, 2])
    def test_forward(self, config, num_register_tokens, num_cls_tokens, convnext_patch_embed):
        config = replace(
            config,
            num_register_tokens=num_register_tokens,
            num_cls_tokens=num_cls_tokens,
            convnext_patch_embed=convnext_patch_embed,
        )
        x = torch.randn(1, 3, 224, 224)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out, cls_tokens, register_tokens = model(x)
        assert out.shape == (1, 196, 128)
        if num_cls_tokens > 0:
            assert cls_tokens.shape == (1, num_cls_tokens, 128)
        else:
            assert cls_tokens is None
        if num_register_tokens > 0:
            assert register_tokens.shape == (1, num_register_tokens, 128)
        else:
            assert register_tokens is None

    def test_forward_with_encoder_output(self, config):
        x = torch.randn(1, 3, 224, 224)
        encoder_output = torch.randn(1, 64, 128)
        config = replace(config, decoder=True)
        model = ViT(config)
        assert model.blocks[0].inter_attention is not None
        assert model.blocks[1].inter_attention is not None
        assert model.blocks[2].inter_attention is not None
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out, cls_token, _ = model(x, encoder_output=encoder_output)
        assert out.shape == (1, 196, 128)
        assert cls_token.shape == (1, 1, 128)

    def test_forward_with_encoder_output_custom_decoder_layers(self, config):
        x = torch.randn(1, 3, 224, 224)
        encoder_output = torch.randn(1, 64, 128)
        config = replace(config, decoder=True, decoder_layers=[0, 2])
        model = ViT(config)
        assert model.blocks[0].inter_attention is not None
        assert model.blocks[1].inter_attention is None
        assert model.blocks[2].inter_attention is not None
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out, cls_token, _ = model(x, encoder_output=encoder_output)
        assert out.shape == (1, 196, 128)
        assert cls_token.shape == (1, 1, 128)

    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = ViT(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, cls_token, _ = model(x)
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
            out, cls_token, _ = model(x, encoder_output=encoder_output)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.cuda
    def test_baseline(self, config):
        if te is None:
            pytest.skip("Transformer Engine is not available")
        B, C, H, W = 2, 3, 64, 64
        torch.random.manual_seed(0)

        baseline_config = replace(config, backend="te")
        baseline = ViT(baseline_config).to("cuda")
        layer = ViT(config).to("cuda")
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

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    def test_mlp_requires_grad(self, config, backend):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend, decoder=True)
        model = ViT(config)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)
            assert_all_requires_grad(block.inter_attention)
        model.mlp_requires_grad_(False)
        for block in model.blocks:
            assert_none_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)
            assert_all_requires_grad(block.inter_attention)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    def test_self_attention_requires_grad(self, config, backend):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend, decoder=True)
        model = ViT(config)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)
            assert_all_requires_grad(block.inter_attention)
        model.self_attention_requires_grad_(False)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_none_requires_grad(block.self_attention)
            assert_all_requires_grad(block.inter_attention)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    def test_inter_attention_requires_grad(self, config, backend):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend, decoder=True)
        model = ViT(config)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)
            assert_all_requires_grad(block.inter_attention)
        model.inter_attention_requires_grad_(False)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)
            assert_none_requires_grad(block.inter_attention)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    def test_inter_attention_requires_grad_encoder_only(self, config, backend):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend)
        model = ViT(config)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)
        model.inter_attention_requires_grad_(False)
        for block in model.blocks:
            assert_all_requires_grad(block.layernorm_mlp)
            assert_all_requires_grad(block.self_attention)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    @pytest.mark.parametrize("mlp", [False, True])
    @pytest.mark.parametrize("out_dim", [1, None])
    def test_forward_head_no_pooling(self, backend, config, mlp, out_dim):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend)
        device = "cuda" if backend == "te" else "cpu"

        x = torch.randn(1, 3, 224, 224, device=device)
        model = ViT(config).to(device)
        head = model.create_head(out_dim, mlp=mlp)
        head = head.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            out, cls_token, _ = model(x)
            out = head(cls_token)
        assert out.shape == (1, 1, out_dim or config.isotropic_output_dim)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    @pytest.mark.parametrize("mlp", [False, True])
    @pytest.mark.parametrize("pool_type", ["avg", "max"])
    @pytest.mark.parametrize("out_dim", [1, None])
    def test_forward_head_pooling(self, backend, config, mlp, pool_type, out_dim):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend)
        device = "cuda" if backend == "te" else "cpu"

        x = torch.randn(1, 3, 224, 224, device=device)
        model = ViT(config).to(device)
        head = model.create_head(out_dim, mlp=mlp, pool_type=pool_type)
        head = head.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            out, _, _ = model(x)
            out = head(out)
        assert out.shape == (1, out_dim or config.isotropic_output_dim)
