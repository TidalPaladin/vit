from typing import TYPE_CHECKING

import pytest
import torch
from torch.testing import assert_close

from vit.helpers import try_import_te
from vit.transformer import CrossAttentionMLP, TransformerLayer


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


class TestTransformerLayer:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    def test_forward(self, dtype, num_gqa_groups):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerLayer(D, D, D // 16, num_gqa_groups=num_gqa_groups, attn_input_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = transformer_layer(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    def test_forward_with_encoder_output(self, dtype, num_gqa_groups):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerLayer(D, D, D // 16, num_gqa_groups=num_gqa_groups, attn_input_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        encoder_output = torch.randn(B, L // 2, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = transformer_layer(x, encoder_output)
        assert y.shape == (B, L, D)

    def test_permute(self):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerLayer(D, D, D // 16, attn_input_format="bshd")
        x = torch.randn(B, L, D)
        x[0] = float("nan")
        y = transformer_layer(x)
        assert y[0].isnan().any()
        assert not y[1:].isnan().any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, dtype, checkpoint):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerLayer(D, D, D // 16, attn_input_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = transformer_layer(x, checkpoint_core_attention=checkpoint)
        y.sum().backward()
        for param in transformer_layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self):
        B, L, D = 16, 128, 128
        layer = TransformerLayer(D, D, D // 16, attn_input_format="bshd")
        x = torch.randn(B, L, D)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        layer.train()
        y3 = layer(x)
        y4 = layer(x)
        assert not torch.allclose(y3, y4)

    @pytest.mark.cuda
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    @pytest.mark.parametrize("bias", [False, True])
    def test_baseline_self_attention(self, num_gqa_groups, normalization, activation, bias):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        B, L, D = 16, 128, 128
        layer = TransformerLayer(
            D,
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            attn_input_format="bshd",
            normalization=normalization,
            activation=activation,
            bias=bias,
        ).cuda()
        baseline = te.TransformerLayer(
            D,
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            attn_input_format="bshd",
            normalization=normalization,
            self_attn_mask_type="no_mask",
            activation=activation,
            bias=bias,
        ).cuda()

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        x = torch.randn(B, L, D, dtype=torch.float32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer(x)
            y_baseline = baseline(x)

        assert_close(y, y_baseline, atol=1e-3, rtol=0)

    @pytest.mark.cuda
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    @pytest.mark.parametrize("bias", [False, True])
    def test_baseline_cross_attention(self, num_gqa_groups, normalization, activation, bias):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        B, L, D = 16, 128, 128
        layer = TransformerLayer(
            D,
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            attn_input_format="bshd",
            normalization=normalization,
            activation=activation,
            layer_type="decoder",
            bias=bias,
        ).cuda()
        baseline = te.TransformerLayer(
            D,
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            attn_input_format="bshd",
            normalization=normalization,
            self_attn_mask_type="no_mask",
            activation=activation,
            layer_type="decoder",
            bias=bias,
        ).cuda()

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        x = torch.randn(B, L, D, dtype=torch.float32, device="cuda")
        encoder_output = torch.randn(B, L // 2, D, dtype=torch.float32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer(x, encoder_output=encoder_output)
            y_baseline = baseline(x, encoder_output=encoder_output)

        assert_close(y, y_baseline, atol=1e-3, rtol=0)


class TestCrossAttentionMLP:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    def test_forward(self, dtype, num_gqa_groups):
        B, L, D = 16, 128, 128
        cross_attention_mlp = CrossAttentionMLP(D, D, D // 16, num_gqa_groups=num_gqa_groups, attn_input_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        encoder_output = torch.randn(B, L // 2, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = cross_attention_mlp(x, encoder_output)
        assert y.shape == (B, L, D)

    def test_permute(self):
        B, L, D = 16, 128, 128
        cross_attention_mlp = CrossAttentionMLP(D, D, D // 16, attn_input_format="bshd")
        x = torch.randn(B, L, D)
        encoder_output = torch.randn(B, L // 2, D)
        x[0] = float("nan")
        y = cross_attention_mlp(x, encoder_output)
        assert y[0].isnan().any()
        assert not y[1:].isnan().any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, dtype, checkpoint):
        B, L, D = 16, 128, 128
        cross_attention_mlp = CrossAttentionMLP(D, D, D // 16, attn_input_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        encoder_output = torch.randn(B, L // 2, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = cross_attention_mlp(x, encoder_output, checkpoint_core_attention=checkpoint)
        y.sum().backward()
        for param in cross_attention_mlp.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self):
        B, L, D = 16, 128, 128
        cross_attention_mlp = CrossAttentionMLP(D, D, D // 16, attn_input_format="bshd")
        x = torch.randn(B, L, D)
        encoder_output = torch.randn(B, L // 2, D)

        cross_attention_mlp.eval()
        y1 = cross_attention_mlp(x, encoder_output)
        y2 = cross_attention_mlp(x, encoder_output)
        assert_close(y1, y2)

        cross_attention_mlp.train()
        y3 = cross_attention_mlp(x, encoder_output)
        y4 = cross_attention_mlp(x, encoder_output)
        assert not torch.allclose(y3, y4)

    @pytest.mark.cuda
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    @pytest.mark.parametrize("bias", [False, True])
    def test_baseline_cross_attention(self, num_gqa_groups, normalization, activation, bias):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        B, L, D = 16, 128, 128
        cross_attention_mlp = CrossAttentionMLP(
            D,
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            attn_input_format="bshd",
            normalization=normalization,
            activation=activation,
            bias=bias,
            backend="pytorch",
        ).cuda()
        baseline = CrossAttentionMLP(
            D,
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            attn_input_format="bshd",
            normalization=normalization,
            activation=activation,
            bias=bias,
            backend="te",
        ).cuda()

        cross_attention_mlp.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            cross_attention_mlp.get_parameter(name).data.copy_(param.data)

        x = torch.randn(B, L, D, dtype=torch.float32, device="cuda")
        encoder_output = torch.randn(B, L // 2, D, dtype=torch.float32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = cross_attention_mlp(x, encoder_output=encoder_output)
            y_baseline = baseline(x, encoder_output=encoder_output)

        assert_close(y, y_baseline, atol=1e-3, rtol=0)
