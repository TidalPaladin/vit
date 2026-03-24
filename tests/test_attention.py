from copy import deepcopy
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close
from torchao.dtypes import AffineQuantizedTensor
from torchao.quantization import Int8WeightOnlyConfig

from vit import AttentivePool as PublicAttentivePool
from vit.attention import AttentivePool, CrossAttention, SelfAttention


NormModule = torch.nn.LayerNorm | torch.nn.RMSNorm


def _apply_norm_manual(x: torch.Tensor, norm: torch.nn.LayerNorm | torch.nn.RMSNorm) -> torch.Tensor:
    if isinstance(norm, torch.nn.LayerNorm):
        return F.layer_norm(x, x.shape[-1:], norm.weight, norm.bias, norm.eps)
    return F.rms_norm(x, x.shape[-1:], norm.weight, norm.eps)


def _to_heads(x: torch.Tensor, batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _assert_qk_norm_type(
    layer: SelfAttention | CrossAttention,
    norm_cls: type[torch.nn.LayerNorm] | type[torch.nn.RMSNorm],
) -> None:
    assert isinstance(layer.q_norm, norm_cls)
    assert isinstance(layer.k_norm, norm_cls)


class TestSelfAttention:
    def test_reset_parameters_initializes_all_projection_biases(self):
        layer = SelfAttention(64, 4)
        assert layer.qkv_proj.bias is not None
        assert layer.out_proj.bias is not None
        assert torch.count_nonzero(layer.qkv_proj.bias) == 0
        assert torch.count_nonzero(layer.out_proj.bias) == 0

    def test_positional_eps_argument_is_backward_compatible(self, device):
        layer = SelfAttention(128, 8, 0.1, 0.1, True, 1e-6).to(device)
        assert layer.norm.eps == 1e-6

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, norm_type):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = multihead_attention(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        B, L, D = 16, 128, 128
        multihead_attention = SelfAttention(D, D // 16, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = multihead_attention(x)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = SelfAttention(D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward_weights(self, device, norm_type):
        B, L, D = 16, 128, 128
        H = D // 16
        multihead_attention = SelfAttention(D, H, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = multihead_attention.forward_weights(x)
        assert y.shape == (B, H, L, L)

    def test_quantization(self, device):
        torch.random.manual_seed(0)
        B, L, D = 10, 128, 128
        H = D // 16
        layer = SelfAttention(D, H).to(device)
        layer.eval()
        quantized_layer = deepcopy(layer)
        quantized_layer.apply_quantization(Int8WeightOnlyConfig())
        weight = quantized_layer.qkv_proj.weight
        assert isinstance(weight, AffineQuantizedTensor)

        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer(x)
            y_quant = quantized_layer(x)
        assert_close(y, y_quant, atol=1e-2, rtol=0)


class TestCrossAttention:
    def test_reset_parameters_initializes_all_projection_biases(self):
        layer = CrossAttention(64, 4)
        assert layer.q_proj.bias is not None
        assert layer.kv_proj.bias is not None
        assert layer.out_proj.bias is not None
        assert torch.count_nonzero(layer.q_proj.bias) == 0
        assert torch.count_nonzero(layer.kv_proj.bias) == 0
        assert torch.count_nonzero(layer.out_proj.bias) == 0

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward(self, device, norm_type):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = multihead_attention(x, kv)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward(self, device, norm_type):
        B, L, D = 16, 128, 128
        multihead_attention = CrossAttention(D, D // 16, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = multihead_attention(x, kv)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = CrossAttention(D, D // 16).to(device)
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

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_forward_weights(self, device, norm_type):
        B, L, D = 16, 128, 128
        H = D // 16
        layer = CrossAttention(D, H, norm_type=norm_type).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = layer.forward_weights(x, kv)
        assert y.shape == (B, H, L, L // 2)

    def test_quantization(self, device):
        # TODO: Investigate this. May be a PyTorch bug.
        if device == torch.device("cpu"):
            pytest.skip("CrossAttention quantization fails to compile on CPU.")
        torch.random.manual_seed(0)
        B, L, D = 10, 128, 128
        H = D // 16
        layer = CrossAttention(D, H).to(device)
        layer.eval()
        quantized_layer = deepcopy(layer)
        quantized_layer.apply_quantization(Int8WeightOnlyConfig())
        weight1 = quantized_layer.q_proj.weight
        weight2 = quantized_layer.kv_proj.weight
        assert isinstance(weight1, AffineQuantizedTensor)
        assert isinstance(weight2, AffineQuantizedTensor)

        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y = layer(x, kv)
            y_quant = quantized_layer(x, kv)
        assert_close(y, y_quant, atol=1e-2, rtol=0)


class TestAttentivePool:
    def test_top_level_export(self):
        assert PublicAttentivePool is AttentivePool

    def test_reset_parameters_initializes_query_and_biases(self):
        layer = AttentivePool(64, 4)
        assert layer.query.abs().max() <= 0.04
        assert layer.cross_attention.q_proj.bias is not None
        assert layer.cross_attention.kv_proj.bias is not None
        assert layer.cross_attention.out_proj.bias is not None
        assert torch.count_nonzero(layer.cross_attention.q_proj.bias) == 0
        assert torch.count_nonzero(layer.cross_attention.kv_proj.bias) == 0
        assert torch.count_nonzero(layer.cross_attention.out_proj.bias) == 0

    def test_forward(self, device):
        B, L, D = 16, 128, 128
        layer = AttentivePool(D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = layer(x)
        assert y.shape == (B, D)

    def test_forward_multi_query(self, device):
        B, L, D, Q = 16, 128, 128, 3
        layer = AttentivePool(D, D // 16, num_queries=Q).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = layer(x)
        assert y.shape == (B, Q, D)

    def test_backward(self, device):
        B, L, D = 16, 128, 128
        layer = AttentivePool(D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_weights(self, device):
        B, L, D = 16, 128, 128
        H = D // 16
        num_queries = 1  # Default num_queries for AttentivePool
        layer = AttentivePool(D, H).to(device)
        x = torch.randn(B, L, D, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            y = layer.forward_weights(x)
        # forward_weights returns attention weights: (B, H, num_queries, L)
        assert y.shape == (B, H, num_queries, L)

    @pytest.mark.parametrize(
        ("norm_type", "norm_cls"), [("rmsnorm", torch.nn.RMSNorm), ("layernorm", torch.nn.LayerNorm)]
    )
    def test_qk_normalization_uses_requested_norm_type(self, norm_type, norm_cls):
        layer = AttentivePool(64, 4, norm_type=norm_type, qk_normalization=True)
        assert isinstance(layer.cross_attention.q_norm, norm_cls)
        assert isinstance(layer.cross_attention.k_norm, norm_cls)

    def test_invalid_num_queries_raises(self):
        with pytest.raises(ValueError, match="num_queries must be positive"):
            AttentivePool(64, 4, num_queries=0)

    def test_hidden_size_not_divisible_by_num_attention_heads_raises(self):
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by.*num_attention_heads"):
            AttentivePool(65, 4)


class TestAttentionWeightsScaling:
    """Test that forward_weights matches scaled_dot_product_attention scaling."""

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_self_attention_weights_scaling(self, device, norm_type):
        """Verify SelfAttention.forward_weights matches SDPA scaling."""
        B, L, D, H = 2, 16, 64, 4
        head_dim = D // H
        layer = SelfAttention(D, H, hidden_dropout=0, attention_dropout=0, norm_type=norm_type).to(device)
        layer.eval()
        x = torch.randn(B, L, D, device=device)

        # Get weights from forward_weights
        weights = layer.forward_weights(x)

        # Manually compute expected weights using the same projections
        if norm_type == "layernorm":
            layer_norm = cast(torch.nn.LayerNorm, layer.norm)
            x_norm = F.layer_norm(x, (D,), layer_norm.weight, layer_norm.bias, layer_norm.eps)
        else:
            rms_norm = cast(torch.nn.RMSNorm, layer.norm)
            x_norm = F.rms_norm(x, (D,), rms_norm.weight, rms_norm.eps)
        qkv = F.linear(x_norm, layer.qkv_proj.weight, layer.qkv_proj.bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, H, head_dim).transpose(1, 2)
        k = k.view(B, L, H, head_dim).transpose(1, 2)
        expected = (q @ k.mT * (head_dim**-0.5)).softmax(dim=-1)

        assert_close(weights, expected, atol=1e-5, rtol=1e-5)


class TestQKNormalization:
    @pytest.mark.parametrize(
        ("norm_type", "norm_cls"), [("rmsnorm", torch.nn.RMSNorm), ("layernorm", torch.nn.LayerNorm)]
    )
    def test_self_attention_qk_normalizer_follows_norm_type(self, norm_type, norm_cls):
        layer = SelfAttention(64, 4, norm_type=norm_type, qk_normalization=True)
        _assert_qk_norm_type(layer, norm_cls)

    @pytest.mark.parametrize(
        ("norm_type", "norm_cls"), [("rmsnorm", torch.nn.RMSNorm), ("layernorm", torch.nn.LayerNorm)]
    )
    def test_cross_attention_qk_normalizer_follows_norm_type(self, norm_type, norm_cls):
        layer = CrossAttention(64, 4, norm_type=norm_type, qk_normalization=True)
        _assert_qk_norm_type(layer, norm_cls)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_self_attention_forward_weights_with_qk_normalization(self, device, norm_type):
        B, L, D, H = 2, 16, 64, 4
        head_dim = D // H
        layer = SelfAttention(
            D, H, hidden_dropout=0, attention_dropout=0, norm_type=norm_type, qk_normalization=True
        ).to(device)
        layer.eval()
        x = torch.randn(B, L, D, device=device)

        weights = layer.forward_weights(x)

        x_norm = _apply_norm_manual(x, cast(NormModule, layer.norm))
        qkv = F.linear(x_norm, layer.qkv_proj.weight, layer.qkv_proj.bias)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = _to_heads(q, B, L, H, head_dim)
        k = _to_heads(k, B, L, H, head_dim)
        q_norm = cast(NormModule, layer.q_norm)
        k_norm = cast(NormModule, layer.k_norm)
        q = _apply_norm_manual(q, q_norm)
        k = _apply_norm_manual(k, k_norm)
        expected = (q @ k.mT * (head_dim**-0.5)).softmax(dim=-1)

        assert_close(weights, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_cross_attention_forward_weights_with_qk_normalization(self, device, norm_type):
        B, L_q, L_kv, D, H = 2, 16, 8, 64, 4
        head_dim = D // H
        layer = CrossAttention(
            D, H, hidden_dropout=0, attention_dropout=0, norm_type=norm_type, qk_normalization=True
        ).to(device)
        layer.eval()
        q = torch.randn(B, L_q, D, device=device)
        kv = torch.randn(B, L_kv, D, device=device)

        weights = layer.forward_weights(q, kv)

        q_input = _apply_norm_manual(q, cast(NormModule, layer.norm))
        q_proj = F.linear(q_input, layer.q_proj.weight, layer.q_proj.bias)
        k_proj, _ = F.linear(kv, layer.kv_proj.weight, layer.kv_proj.bias).chunk(2, dim=-1)
        q_proj = _to_heads(q_proj, B, L_q, H, head_dim)
        k_proj = _to_heads(k_proj, B, L_kv, H, head_dim)
        q_norm = cast(NormModule, layer.q_norm)
        k_norm = cast(NormModule, layer.k_norm)
        q_proj = _apply_norm_manual(q_proj, q_norm)
        k_proj = _apply_norm_manual(k_proj, k_norm)
        expected = (q_proj @ k_proj.mT * (head_dim**-0.5)).softmax(dim=-1)

        assert_close(weights, expected, atol=1e-5, rtol=1e-5)
