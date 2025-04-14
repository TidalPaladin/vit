from typing import TYPE_CHECKING

import pytest
import torch
from torch.testing import assert_close

from vit.attention import MultiheadAttention
from vit.helpers import try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


class TestMultiheadAttention:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, num_gqa_groups, normalization):
        B, L, D = 16, 128, 128
        multihead_attention = MultiheadAttention(
            D, D // 16, num_gqa_groups=num_gqa_groups, qkv_format="bshd", normalization=normalization
        )
        x = torch.randn(B, L, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = multihead_attention(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward_with_encoder_output(self, dtype, num_gqa_groups, normalization):
        B, L, D = 16, 128, 128
        multihead_attention = MultiheadAttention(
            D, D // 16, num_gqa_groups=num_gqa_groups, qkv_format="bshd", normalization=normalization
        )
        x = torch.randn(B, L, D, dtype=dtype)
        encoder_output = torch.randn(B, L // 2, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = multihead_attention(x, encoder_output)
        assert y.shape == (B, L, D)

    def test_permute(self):
        B, L, D = 16, 128, 128
        multihead_attention = MultiheadAttention(D, D // 16, qkv_format="bshd")
        x = torch.randn(B, L, D)
        x[0] = float("nan")
        y = multihead_attention(x)
        assert y[0].isnan().any()
        assert not y[1:].isnan().any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, dtype, checkpoint):
        B, L, D = 16, 128, 128
        multihead_attention = MultiheadAttention(D, D // 16, qkv_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = multihead_attention(x, checkpoint_core_attention=checkpoint)
        y.sum().backward()
        for param in multihead_attention.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_forward_determinstic(self):
        B, L, D = 16, 128, 128
        layer = MultiheadAttention(D, D // 16, qkv_format="bshd")
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
    @pytest.mark.parametrize("input_layernorm", [False, True])
    def test_baseline_self_attention(self, num_gqa_groups, normalization, input_layernorm):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        B, L, D = 16, 128, 128
        layer = MultiheadAttention(
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            qkv_format="bshd",
            normalization=normalization,
            input_layernorm=input_layernorm,
        ).cuda()
        baseline = te.MultiheadAttention(
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            qkv_format="bshd",
            normalization=normalization,
            input_layernorm=input_layernorm,
            attn_mask_type="no_mask",
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

        assert_close(y, y_baseline, atol=1e-4, rtol=0)

    @pytest.mark.cuda
    @pytest.mark.parametrize("num_gqa_groups", [8, 4])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("input_layernorm", [False, True])
    def test_baseline_cross_attention(self, num_gqa_groups, normalization, input_layernorm):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        B, L, D = 16, 128, 128
        layer = MultiheadAttention(
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            qkv_format="bshd",
            normalization=normalization,
            input_layernorm=input_layernorm,
            attention_type="cross",
        ).cuda()
        baseline = te.MultiheadAttention(
            D,
            D // 16,
            num_gqa_groups=num_gqa_groups,
            qkv_format="bshd",
            normalization=normalization,
            input_layernorm=input_layernorm,
            attn_mask_type="no_mask",
            attention_type="cross",
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
