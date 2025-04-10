import pytest
import torch
from torch.testing import assert_close

from vit.transformer import TransformerLayer as TransformerLayerBaseline


try:
    from transformer_engine.pytorch import TransformerLayer
except ImportError:
    pytest.skip("Transformer Engine is not installed", allow_module_level=True)


@pytest.mark.skip(reason="Incomplete")
class TestTransformerLayer:

    def test_baseline_simple(self):
        B, L, D = 16, 128, 32
        layer = TransformerLayer(
            D, D, 1, attn_input_format="bshd", hidden_dropout=0.0, attention_dropout=0.0, self_attn_mask_type="no_mask"
        ).cuda()
        baseline = TransformerLayerBaseline(
            D, D, 1, attn_input_format="bshd", hidden_dropout=0.0, attention_dropout=0.0
        ).cuda()
        x = torch.randn(B, L, D, dtype=torch.float32, device="cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in layer.named_parameters():
            baseline.get_parameter(name).data.copy_(param.data)

        # MLP only
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y, _ = layer.layernorm_mlp(x)
            y_baseline = baseline.layernorm_mlp(x)
        assert_close(y, y_baseline, atol=1e-4, rtol=0)

        # QKV projection
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            q, k, v = torch.split(layer.self_attention.layernorm_qkv(x), D, dim=-1)
            q_baseline, k_baseline, v_baseline = baseline.self_attention.layernorm_qkv(x)
        assert_close(q, q_baseline.squeeze(), atol=1e-4, rtol=0)
        assert_close(k, k_baseline.squeeze(), atol=1e-4, rtol=0)
        assert_close(v, v_baseline.squeeze(), atol=1e-4, rtol=0)

        # Self-attention only
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y, _ = layer.self_attention(x, attn_mask_type="no_mask")
            y_baseline = baseline.self_attention(x)
        assert_close(y, y_baseline, atol=1e-4, rtol=0)

        # Then check the whole thing
        x = torch.randn(B, L, D, dtype=torch.float32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline, atol=1e-4, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_gqa_groups", [8, 16])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, num_gqa_groups, normalization):
        B, L, D = 16, 128, 128
        layer = TransformerLayer(
            D, D, D // 16, num_gqa_groups=num_gqa_groups, attn_input_format="bshd", normalization=normalization
        ).cuda()
        baseline = TransformerLayerBaseline(
            D, D, D // 16, num_gqa_groups=num_gqa_groups, attn_input_format="bshd", normalization=normalization
        ).cuda()

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        x = torch.randn(B, L, D, dtype=dtype, device="cuda")
        with torch.autocast(device_type="cuda", dtype=dtype):
            y = layer(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline, atol=1e-4, rtol=0)
