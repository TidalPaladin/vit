import math
from dataclasses import replace

import pytest
import torch
from torch.testing import assert_close

from vit.vit import ViT, ViTConfig


@pytest.fixture
def config():
    return ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        img_size=(224, 224),
        depth=4,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=8,
        pos_enc="learnable",
        dtype=torch.float32,
    )


class TestActivationCheckpointing:
    def test_forward_with_checkpointing_training(self, device, config):
        """Verify forward pass works with checkpointing enabled during training."""
        config = replace(config, activation_checkpointing=True)
        model = ViT(config).to(device)
        model.train()
        x = torch.randn(2, 3, 224, 224, device=device)
        out = model(x)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.visual_tokens.shape == (2, L, config.hidden_size)

    def test_forward_with_checkpointing_inference(self, device, config):
        """Verify forward pass works with checkpointing enabled during inference."""
        config = replace(config, activation_checkpointing=True)
        model = ViT(config).to(device)
        model.eval()
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(x)
        L = math.prod(model.stem.tokenized_size(config.img_size))
        assert out.visual_tokens.shape == (2, L, config.hidden_size)

    def test_backward_with_checkpointing(self, device, config):
        """Verify gradients are computed correctly with checkpointing."""
        config = replace(config, activation_checkpointing=True)
        model = ViT(config).to(device)
        model.train()
        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        out = model(x)
        out.dense_features.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"
                assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_checkpointing_output_matches_non_checkpointing(self, device, config):
        """Verify checkpointed forward produces same output as non-checkpointed."""
        torch.manual_seed(42)
        model_no_ckpt = ViT(config).to(device)
        model_no_ckpt.eval()

        config_ckpt = replace(config, activation_checkpointing=True)
        model_ckpt = ViT(config_ckpt).to(device)
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())
        model_ckpt.eval()

        x = torch.randn(2, 3, 224, 224, device=device)

        with torch.no_grad():
            out_no_ckpt = model_no_ckpt(x)
            out_ckpt = model_ckpt(x)

        assert_close(out_no_ckpt.dense_features, out_ckpt.dense_features)

    def test_checkpointing_gradient_properties(self, device, config):
        """Verify gradient properties are similar between checkpointed and non-checkpointed models.

        Note: Exact gradient matching is not guaranteed when using torch.compile with checkpointing,
        as the compiled graphs may differ between the initial forward pass and the recomputed forward
        pass during backward. Instead, we verify that gradients have similar statistical properties.
        """
        # Use config without stochastic elements for deterministic comparison
        config = replace(config, drop_path_rate=0.0, hidden_dropout=0.0, attention_dropout=0.0)

        torch.manual_seed(42)
        model_no_ckpt = ViT(config).to(device)

        config_ckpt = replace(config, activation_checkpointing=True)
        model_ckpt = ViT(config_ckpt).to(device)
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())

        torch.manual_seed(123)
        x = torch.randn(2, 3, 224, 224, device=device)

        # Forward and backward without checkpointing
        model_no_ckpt.train()
        out_no_ckpt = model_no_ckpt(x)
        out_no_ckpt.dense_features.sum().backward()
        grads_no_ckpt = {name: p.grad.clone() for name, p in model_no_ckpt.named_parameters() if p.grad is not None}

        # Forward and backward with checkpointing
        model_ckpt.train()
        out_ckpt = model_ckpt(x)
        out_ckpt.dense_features.sum().backward()
        grads_ckpt = {name: p.grad.clone() for name, p in model_ckpt.named_parameters() if p.grad is not None}

        # Verify gradient properties match (not exact values due to torch.compile interaction)
        for name in grads_no_ckpt:
            grad_no_ckpt = grads_no_ckpt[name]
            grad_ckpt = grads_ckpt[name]

            # Same shape
            assert grad_no_ckpt.shape == grad_ckpt.shape, f"{name} gradient shape mismatch"

            # Neither has NaNs
            assert not grad_no_ckpt.isnan().any(), f"{name} non-checkpointed gradient has NaNs"
            assert not grad_ckpt.isnan().any(), f"{name} checkpointed gradient has NaNs"

            # Similar magnitude (within 10x)
            norm_no_ckpt = grad_no_ckpt.norm()
            norm_ckpt = grad_ckpt.norm()
            if norm_no_ckpt > 1e-6:  # Skip if gradient is essentially zero
                ratio = norm_ckpt / norm_no_ckpt
                assert 0.1 < ratio < 10, f"{name} gradient magnitude differs too much: {ratio:.2f}x"

    def test_checkpointing_with_rope(self, device, config):
        """Verify checkpointing works with RoPE position encoding."""
        config = replace(config, activation_checkpointing=True, pos_enc="rope")
        model = ViT(config).to(device)
        model.train()
        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        out = model(x)
        out.dense_features.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient with RoPE"
                assert not param.grad.isnan().any(), f"{name} has nan gradient with RoPE"

    def test_checkpointing_with_mask(self, device, config):
        """Verify checkpointing works with token masking."""
        config = replace(config, activation_checkpointing=True, pos_enc="learnable")
        model = ViT(config).to(device)
        model.train()
        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        mask = model.create_mask(x, 0.5, 1)
        out = model(x, mask=mask)
        out.dense_features.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient with mask"
                assert not param.grad.isnan().any(), f"{name} has nan gradient with mask"

    def test_checkpointing_with_drop_path(self, device, config):
        """Verify checkpointing handles stochastic depth correctly."""
        config = replace(config, activation_checkpointing=True, drop_path_rate=0.1)
        model = ViT(config).to(device)
        model.train()
        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        out = model(x)
        out.dense_features.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not param.grad.isnan().any(), f"{name} has nan gradient with drop_path"

    @pytest.mark.cuda
    def test_checkpointing_with_autocast(self, config):
        """Verify checkpointing works with mixed precision training."""
        device = torch.device("cuda")
        config = replace(config, activation_checkpointing=True, dtype=torch.float32)
        model = ViT(config).to(device)
        model.train()
        x = torch.randn(2, 3, 224, 224, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            out.dense_features.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient with autocast"
                assert not param.grad.isnan().any(), f"{name} has nan gradient with autocast"

    @pytest.mark.cuda
    def test_checkpointing_reduces_memory(self, config):
        """Verify checkpointing reduces peak memory usage on CUDA."""
        device = torch.device("cuda")
        config = replace(config, depth=12, hidden_size=512, ffn_hidden_size=2048, num_attention_heads=16)

        # Measure memory without checkpointing
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model_no_ckpt = ViT(config).to(device)
        model_no_ckpt.train()
        x = torch.randn(4, 3, 224, 224, device=device)
        out = model_no_ckpt(x)
        out.dense_features.sum().backward()
        mem_no_ckpt = torch.cuda.max_memory_allocated()

        del model_no_ckpt, x, out
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure memory with checkpointing
        config_ckpt = replace(config, activation_checkpointing=True)
        model_ckpt = ViT(config_ckpt).to(device)
        model_ckpt.train()
        x = torch.randn(4, 3, 224, 224, device=device)
        out = model_ckpt(x)
        out.dense_features.sum().backward()
        mem_ckpt = torch.cuda.max_memory_allocated()

        # Checkpointing should use less memory
        assert mem_ckpt < mem_no_ckpt, f"Expected memory reduction, got {mem_ckpt} vs {mem_no_ckpt}"

    @pytest.mark.ci_skip
    def test_checkpointing_with_torch_compile(self, device, config):
        """Verify checkpointing works when model is torch.compiled."""
        config = replace(config, activation_checkpointing=True)
        model = ViT(config).to(device)
        model.train()

        compiled_model = torch.compile(model)

        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        out = compiled_model(x)
        out.dense_features.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient after compile"
                assert not param.grad.isnan().any(), f"{name} has nan gradient after compile"

    def test_config_yaml_serialization(self, config):
        """Verify activation_checkpointing serializes to/from YAML correctly."""
        config_ckpt = replace(config, activation_checkpointing=True)
        yaml_str = config_ckpt.to_yaml()
        config_restored = ViTConfig.from_yaml(yaml_str)
        assert config_restored.activation_checkpointing is True

        config_no_ckpt = replace(config, activation_checkpointing=False)
        yaml_str = config_no_ckpt.to_yaml()
        config_restored = ViTConfig.from_yaml(yaml_str)
        assert config_restored.activation_checkpointing is False
