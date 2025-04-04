import pytest
import torch
from torch.testing import assert_close

from vit.transformer import TransformerLayer


class TestTransformerLayer:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 16])
    def test_forward(self, dtype, num_gqa_groups):
        B, L, D = 16, 128, 128
        transformer_layer = TransformerLayer(D, D, D // 16, num_gqa_groups=num_gqa_groups, attn_input_format="bshd")
        x = torch.randn(B, L, D, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = transformer_layer(x)
        assert y.shape == (B, L, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 16])
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
