import pytest
import torch
from torch.testing import assert_close

from vit.transformer import CrossAttentionTransformer, TransformerDecoderLayer, TransformerEncoderLayer


class TestTransformerEncoderLayer:
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, layer_scale, norm_type):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerEncoderLayer(D, D, D // 16, norm_type=norm_type, layer_scale=layer_scale).to(
            device
        )
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, layer_scale, norm_type):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerEncoderLayer(D, D, D // 16, norm_type=norm_type, layer_scale=layer_scale).to(
            device
        )
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = TransformerEncoderLayer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    def test_encoder_layer_scale_applied_once(self, device):
        """Regression test: verify layer_scale is applied once, not twice (issue #79)."""
        hidden_size = 64
        layer_scale_init = 0.1  # Use a value where γ vs γ² is easily distinguishable

        layer = TransformerEncoderLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 4,
            num_attention_heads=4,
            layer_scale=layer_scale_init,
        ).to(device)

        torch.manual_seed(42)
        x = torch.randn(2, 16, hidden_size, device=device)

        layer.eval()
        with torch.no_grad():
            # Isolate MLP path: compute MLP output and expected scaled result directly
            # First get post-attention state (input to MLP)
            attn_out = layer.layer_scale_attn(layer.self_attention(x))
            post_attn = x + attn_out

            # Compute unscaled MLP output
            mlp_out_unscaled = layer.mlp(post_attn)

            # Compute what layer_scale_mlp produces (should be γ * mlp_out, not γ² * mlp_out)
            mlp_out_scaled = layer.layer_scale_mlp(mlp_out_unscaled.clone())

            # Expected: scaled = γ * unscaled
            expected_scaled = layer_scale_init * mlp_out_unscaled

        # If layer_scale were applied twice, mlp_out_scaled would be γ² * unscaled (0.01x)
        # With correct single application, it should be γ * unscaled (0.1x)
        assert_close(mlp_out_scaled, expected_scaled, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("moe_routing_mode", ["expert_choice", "token_choice"])
    def test_forward_with_moe_tensors(self, device, moe_routing_mode):
        B, L, D = 2, 64, 64
        layer = TransformerEncoderLayer(
            hidden_size=D,
            ffn_hidden_size=4 * D,
            num_attention_heads=4,
            use_moe=True,
            moe_num_experts=4,
            moe_routing_mode=moe_routing_mode,
            moe_token_top_k=2,
        ).to(device)
        x = torch.randn(B, L, D, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_moe_tensors(x)
        assert y.shape == (B, L, D)
        assert router_logits.shape == (B, L, 4)
        assert expert_token_counts.shape == (4,)
        assert dropped_token_count.ndim == 0
        assert capacity.ndim == 0

    def test_forward_with_token_choice_simple_experts(self, device):
        B, L, D = 2, 64, 64
        layer = TransformerEncoderLayer(
            hidden_size=D,
            ffn_hidden_size=4 * D,
            num_attention_heads=4,
            use_moe=True,
            moe_num_experts=4,
            moe_routing_mode="token_choice",
            moe_token_top_k=2,
            moe_use_simple_experts=True,
            moe_num_zero_experts=1,
            moe_num_copy_experts=1,
            moe_num_constant_experts=1,
        ).to(device)
        x = torch.randn(B, L, D, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_moe_tensors(x)
        assert y.shape == (B, L, D)
        assert router_logits.shape == (B, L, 4)
        assert expert_token_counts.shape == (4,)
        assert dropped_token_count.ndim == 0
        assert capacity.ndim == 0

    def test_forward_with_moe_tensors_raises_on_dense_layer(self, device):
        B, L, D = 2, 64, 64
        layer = TransformerEncoderLayer(
            hidden_size=D,
            ffn_hidden_size=4 * D,
            num_attention_heads=4,
        ).to(device)
        x = torch.randn(B, L, D, device=device)
        with pytest.raises(RuntimeError, match="non-MoE encoder layer"):
            layer.forward_with_moe_tensors(x)


class TestTransformerDecoderLayer:
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, layer_scale, norm_type):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerDecoderLayer(D, D, D // 16, norm_type=norm_type, layer_scale=layer_scale).to(
            device
        )
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, layer_scale, norm_type):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerDecoderLayer(D, D, D // 16, norm_type=norm_type, layer_scale=layer_scale).to(
            device
        )
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x, kv)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = TransformerDecoderLayer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)

        layer.eval()
        y1 = layer(x, kv)
        y2 = layer(x, kv)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x, kv)
        y4 = layer(x, kv)
        assert not torch.allclose(y3, y4)


class TestCrossAttentionTransformer:
    @pytest.mark.parametrize("layer_scale", [None, 1e-5])
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, layer_scale, norm_type):
        B, L, D = 16, 128, 128
        transformer_layer = CrossAttentionTransformer(D, D, D // 16, norm_type=norm_type, layer_scale=layer_scale).to(
            device
        )
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        B, L, D = 16, 128, 128
        transformer_layer = CrossAttentionTransformer(D, D, D // 16, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = transformer_layer(x, kv)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = CrossAttentionTransformer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)

        layer.eval()
        y1 = layer(x, kv)
        y2 = layer(x, kv)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x, kv)
        y4 = layer(x, kv)
        assert not torch.allclose(y3, y4)
