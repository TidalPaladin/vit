import pytest
import torch
from torch.testing import assert_close

from vit.transformer import (
    CrossAttentionTransformer,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    _select_residual_subset,
)


def _create_batched_rope(batch_size: int, seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(2, batch_size, 1, seq_len, head_dim, device=device)


def _expected_keep_count(batch_size: int, drop_path_rate: float) -> int:
    return max(1, int(batch_size * (1.0 - drop_path_rate)))


def _assert_parameter_grads_finite(module: torch.nn.Module) -> None:
    for param in module.parameters():
        assert param.grad is not None
        assert not param.grad.isnan().any()


def _assert_linear_zero_initialized(linear: torch.nn.Linear) -> None:
    assert torch.count_nonzero(linear.weight) == 0
    assert linear.bias is not None
    assert torch.count_nonzero(linear.bias) == 0


@torch.no_grad()
def _seed_linear_weights(*linears: torch.nn.Linear, std: float = 0.01) -> None:
    for linear in linears:
        linear.weight.normal_(std=std)


class TestTransformerEncoderLayer:
    def test_default_residual_outputs_are_zero_initialized(self):
        layer = TransformerEncoderLayer(64, 128, 4)
        _assert_linear_zero_initialized(layer.self_attention.out_proj)
        _assert_linear_zero_initialized(layer.mlp.fc2)

    def test_forward_is_identity_at_initialization(self, device):
        layer = TransformerEncoderLayer(64, 128, 4).to(device)
        x = torch.randn(2, 12, 64, device=device)
        layer.eval()
        y = layer(x)
        assert_close(y, x)

    @pytest.mark.parametrize(
        ("norm_type", "norm_cls"), [("rmsnorm", torch.nn.RMSNorm), ("layernorm", torch.nn.LayerNorm)]
    )
    def test_qk_normalization_wires_self_attention_norm_type(self, norm_type, norm_cls):
        layer = TransformerEncoderLayer(64, 128, 4, norm_type=norm_type, qk_normalization=True)
        assert isinstance(layer.self_attention.q_norm, norm_cls)
        assert isinstance(layer.self_attention.k_norm, norm_cls)

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
        _assert_parameter_grads_finite(transformer_layer)

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = TransformerEncoderLayer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)

        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

        _seed_linear_weights(layer.self_attention.out_proj, layer.mlp.fc2)

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

    def test_selective_stochastic_depth_uses_fixed_keep_count(self, device, monkeypatch):
        batch_size, seq_len, hidden_size = 8, 16, 64
        drop_path_rate = 0.25
        expected_keep_count = _expected_keep_count(batch_size, drop_path_rate)
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads

        layer = TransformerEncoderLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 2,
            num_attention_heads=num_attention_heads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=drop_path_rate,
        ).to(device)
        layer.train()

        attention_batches: list[int] = []
        attention_rope_batches: list[int] = []
        mlp_batches: list[int] = []

        original_attention_forward = layer.self_attention.forward
        original_mlp_forward = layer.mlp.forward

        def record_attention_batch(
            x: torch.Tensor, attn_mask: torch.Tensor | None = None, rope: torch.Tensor | None = None
        ) -> torch.Tensor:
            attention_batches.append(x.shape[0])
            if rope is not None and rope.ndim == 5:
                attention_rope_batches.append(rope.shape[1])
            return original_attention_forward(x, attn_mask=attn_mask, rope=rope)

        def record_mlp_batch(x: torch.Tensor) -> torch.Tensor:
            mlp_batches.append(x.shape[0])
            return original_mlp_forward(x)

        monkeypatch.setattr(layer.self_attention, "forward", record_attention_batch)
        monkeypatch.setattr(layer.mlp, "forward", record_mlp_batch)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        rope = _create_batched_rope(batch_size, seq_len, head_dim, device)
        for _ in range(4):
            y = layer(x, rope=rope)
            assert y.shape == (batch_size, seq_len, hidden_size)

        assert attention_batches
        assert mlp_batches
        assert set(attention_batches) == {expected_keep_count}
        assert set(mlp_batches) == {expected_keep_count}
        assert set(attention_rope_batches) == {expected_keep_count}

    def test_selective_stochastic_depth_scale_matches_keep_ratio(self, device):
        batch_size = 8
        seq_len = 4
        hidden_size = 16
        drop_path_rate = 0.2
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        _, keep_indices, residual_scale = _select_residual_subset(x, drop_path_rate=drop_path_rate, training=True)
        assert keep_indices is not None
        keep_count = _expected_keep_count(batch_size, drop_path_rate)
        assert residual_scale == pytest.approx(batch_size / keep_count)

    def test_backward_with_selective_stochastic_depth(self, device):
        batch_size, seq_len, hidden_size = 8, 16, 64
        layer = TransformerEncoderLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 2,
            num_attention_heads=4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=0.5,
        ).to(device)
        layer.train()

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
        y = layer(x)
        y.sum().backward()

        _assert_parameter_grads_finite(layer)


class TestTransformerDecoderLayer:
    def test_default_residual_outputs_are_zero_initialized(self):
        layer = TransformerDecoderLayer(64, 128, 4)
        _assert_linear_zero_initialized(layer.self_attention.out_proj)
        _assert_linear_zero_initialized(layer.cross_attention.out_proj)
        _assert_linear_zero_initialized(layer.mlp.fc2)

    def test_forward_is_identity_at_initialization(self, device):
        layer = TransformerDecoderLayer(64, 128, 4).to(device)
        x = torch.randn(2, 12, 64, device=device)
        kv = torch.randn(2, 8, 64, device=device)
        layer.eval()
        y = layer(x, kv)
        assert_close(y, x)

    @pytest.mark.parametrize(
        ("norm_type", "norm_cls"), [("rmsnorm", torch.nn.RMSNorm), ("layernorm", torch.nn.LayerNorm)]
    )
    def test_qk_normalization_wires_self_and_cross_attention_norm_type(self, norm_type, norm_cls):
        layer = TransformerDecoderLayer(64, 128, 4, norm_type=norm_type, qk_normalization=True)
        assert isinstance(layer.self_attention.q_norm, norm_cls)
        assert isinstance(layer.self_attention.k_norm, norm_cls)
        assert isinstance(layer.cross_attention.q_norm, norm_cls)
        assert isinstance(layer.cross_attention.k_norm, norm_cls)

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
        _assert_parameter_grads_finite(transformer_layer)

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = TransformerDecoderLayer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)

        layer.eval()
        y1 = layer(x, kv)
        y2 = layer(x, kv)
        assert_close(y1, y2)

        _seed_linear_weights(layer.self_attention.out_proj, layer.cross_attention.out_proj, layer.mlp.fc2)

        layer.train()
        y3 = layer(x, kv)
        y4 = layer(x, kv)
        assert not torch.allclose(y3, y4)

    def test_selective_stochastic_depth_subsets_cross_attention_inputs(self, device, monkeypatch):
        batch_size, seq_len, hidden_size, kv_len = 7, 16, 64, 10
        drop_path_rate = 0.4
        expected_keep_count = _expected_keep_count(batch_size, drop_path_rate)
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads

        layer = TransformerDecoderLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 2,
            num_attention_heads=num_attention_heads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=drop_path_rate,
        ).to(device)
        layer.train()

        self_attention_batches: list[int] = []
        cross_attention_batches: list[tuple[int, int]] = []
        cross_rope_batches: list[tuple[int, int]] = []
        mlp_batches: list[int] = []

        original_self_attention_forward = layer.self_attention.forward
        original_cross_attention_forward = layer.cross_attention.forward
        original_mlp_forward = layer.mlp.forward

        def record_self_attention_batch(
            x: torch.Tensor, attn_mask: torch.Tensor | None = None, rope: torch.Tensor | None = None
        ) -> torch.Tensor:
            self_attention_batches.append(x.shape[0])
            return original_self_attention_forward(x, attn_mask=attn_mask, rope=rope)

        def record_cross_attention_batch(
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            rope_q: torch.Tensor | None = None,
            rope_k: torch.Tensor | None = None,
        ) -> torch.Tensor:
            cross_attention_batches.append((q.shape[0], kv.shape[0]))
            if rope_q is not None and rope_k is not None and rope_q.ndim == 5 and rope_k.ndim == 5:
                cross_rope_batches.append((rope_q.shape[1], rope_k.shape[1]))
            return original_cross_attention_forward(q, kv, attn_mask=attn_mask, rope_q=rope_q, rope_k=rope_k)

        def record_mlp_batch(x: torch.Tensor) -> torch.Tensor:
            mlp_batches.append(x.shape[0])
            return original_mlp_forward(x)

        monkeypatch.setattr(layer.self_attention, "forward", record_self_attention_batch)
        monkeypatch.setattr(layer.cross_attention, "forward", record_cross_attention_batch)
        monkeypatch.setattr(layer.mlp, "forward", record_mlp_batch)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        kv = torch.randn(batch_size, kv_len, hidden_size, device=device)
        rope_q = _create_batched_rope(batch_size, seq_len, head_dim, device)
        rope_k = _create_batched_rope(batch_size, kv_len, head_dim, device)

        for _ in range(4):
            y = layer(x, kv, rope_q=rope_q, rope_k=rope_k)
            assert y.shape == (batch_size, seq_len, hidden_size)

        assert self_attention_batches
        assert cross_attention_batches
        assert mlp_batches
        assert set(self_attention_batches) == {expected_keep_count}
        assert set(mlp_batches) == {expected_keep_count}
        assert set(cross_attention_batches) == {(expected_keep_count, expected_keep_count)}
        assert set(cross_rope_batches) == {(expected_keep_count, expected_keep_count)}

    def test_selective_stochastic_depth_preserves_broadcast_kv_batch(self, device, monkeypatch):
        batch_size, seq_len, hidden_size, kv_len = 8, 16, 64, 10
        kv_batch_size = 1
        drop_path_rate = 0.5
        expected_keep_count = _expected_keep_count(batch_size, drop_path_rate)
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads

        layer = TransformerDecoderLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 2,
            num_attention_heads=num_attention_heads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=drop_path_rate,
        ).to(device)
        layer.train()

        cross_attention_batches: list[tuple[int, int]] = []
        original_cross_attention_forward = layer.cross_attention.forward

        def record_cross_attention_batch(
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            rope_q: torch.Tensor | None = None,
            rope_k: torch.Tensor | None = None,
        ) -> torch.Tensor:
            cross_attention_batches.append((q.shape[0], kv.shape[0]))
            return original_cross_attention_forward(q, kv, attn_mask=attn_mask, rope_q=rope_q, rope_k=rope_k)

        monkeypatch.setattr(layer.cross_attention, "forward", record_cross_attention_batch)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        kv = torch.randn(kv_batch_size, kv_len, hidden_size, device=device)
        rope_q = _create_batched_rope(batch_size, seq_len, head_dim, device)
        rope_k = _create_batched_rope(kv_batch_size, kv_len, head_dim, device)

        y = layer(x, kv, rope_q=rope_q, rope_k=rope_k)
        assert y.shape == (batch_size, seq_len, hidden_size)
        assert cross_attention_batches == [(expected_keep_count, kv_batch_size)]


class TestCrossAttentionTransformer:
    def test_default_residual_outputs_are_zero_initialized(self):
        layer = CrossAttentionTransformer(64, 128, 4)
        _assert_linear_zero_initialized(layer.cross_attention.out_proj)
        _assert_linear_zero_initialized(layer.mlp.fc2)

    def test_forward_is_identity_at_initialization(self, device):
        layer = CrossAttentionTransformer(64, 128, 4).to(device)
        x = torch.randn(2, 12, 64, device=device)
        kv = torch.randn(2, 8, 64, device=device)
        layer.eval()
        y = layer(x, kv)
        assert_close(y, x)

    @pytest.mark.parametrize(
        ("norm_type", "norm_cls"), [("rmsnorm", torch.nn.RMSNorm), ("layernorm", torch.nn.LayerNorm)]
    )
    def test_qk_normalization_wires_cross_attention_norm_type(self, norm_type, norm_cls):
        layer = CrossAttentionTransformer(64, 128, 4, norm_type=norm_type, qk_normalization=True)
        assert isinstance(layer.cross_attention.q_norm, norm_cls)
        assert isinstance(layer.cross_attention.k_norm, norm_cls)

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
        _assert_parameter_grads_finite(transformer_layer)

    def test_forward_determinstic(self, device):
        B, L, D = 16, 128, 128
        layer = CrossAttentionTransformer(D, D, D // 16).to(device)
        x = torch.randn(B, L, D, device=device)
        kv = torch.randn(B, L // 2, D, device=device)

        layer.eval()
        y1 = layer(x, kv)
        y2 = layer(x, kv)
        assert_close(y1, y2)

        _seed_linear_weights(layer.cross_attention.out_proj, layer.mlp.fc2)

        layer.train()
        y3 = layer(x, kv)
        y4 = layer(x, kv)
        assert not torch.allclose(y3, y4)

    def test_selective_stochastic_depth_cross_attention_alignment(self, device, monkeypatch):
        batch_size, seq_len, hidden_size, kv_len = 9, 12, 64, 6
        drop_path_rate = 0.5
        expected_keep_count = _expected_keep_count(batch_size, drop_path_rate)
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads

        layer = CrossAttentionTransformer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 2,
            num_attention_heads=num_attention_heads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=drop_path_rate,
        ).to(device)
        layer.train()

        cross_attention_batches: list[tuple[int, int]] = []
        mlp_batches: list[int] = []

        original_cross_attention_forward = layer.cross_attention.forward
        original_mlp_forward = layer.mlp.forward

        def record_cross_attention_batch(
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            rope_q: torch.Tensor | None = None,
            rope_k: torch.Tensor | None = None,
        ) -> torch.Tensor:
            cross_attention_batches.append((q.shape[0], kv.shape[0]))
            if rope_q is not None and rope_q.ndim == 5:
                assert rope_q.shape[1] == q.shape[0]
            if rope_k is not None and rope_k.ndim == 5:
                assert rope_k.shape[1] == kv.shape[0]
            return original_cross_attention_forward(q, kv, attn_mask=attn_mask, rope_q=rope_q, rope_k=rope_k)

        def record_mlp_batch(x: torch.Tensor) -> torch.Tensor:
            mlp_batches.append(x.shape[0])
            return original_mlp_forward(x)

        monkeypatch.setattr(layer.cross_attention, "forward", record_cross_attention_batch)
        monkeypatch.setattr(layer.mlp, "forward", record_mlp_batch)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        kv = torch.randn(batch_size, kv_len, hidden_size, device=device)
        rope_q = _create_batched_rope(batch_size, seq_len, head_dim, device)
        rope_k = _create_batched_rope(batch_size, kv_len, head_dim, device)

        for _ in range(4):
            y = layer(x, kv, rope_q=rope_q, rope_k=rope_k)
            assert y.shape == (batch_size, seq_len, hidden_size)

        assert cross_attention_batches
        assert mlp_batches
        assert set(cross_attention_batches) == {(expected_keep_count, expected_keep_count)}
        assert set(mlp_batches) == {expected_keep_count}

    def test_selective_stochastic_depth_preserves_broadcast_kv_batch(self, device, monkeypatch):
        batch_size, seq_len, hidden_size, kv_len = 8, 12, 64, 6
        kv_batch_size = 1
        drop_path_rate = 0.5
        expected_keep_count = _expected_keep_count(batch_size, drop_path_rate)
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads

        layer = CrossAttentionTransformer(
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size * 2,
            num_attention_heads=num_attention_heads,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=drop_path_rate,
        ).to(device)
        layer.train()

        cross_attention_batches: list[tuple[int, int]] = []
        original_cross_attention_forward = layer.cross_attention.forward

        def record_cross_attention_batch(
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            rope_q: torch.Tensor | None = None,
            rope_k: torch.Tensor | None = None,
        ) -> torch.Tensor:
            cross_attention_batches.append((q.shape[0], kv.shape[0]))
            return original_cross_attention_forward(q, kv, attn_mask=attn_mask, rope_q=rope_q, rope_k=rope_k)

        monkeypatch.setattr(layer.cross_attention, "forward", record_cross_attention_batch)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        kv = torch.randn(kv_batch_size, kv_len, hidden_size, device=device)
        rope_q = _create_batched_rope(batch_size, seq_len, head_dim, device)
        rope_k = _create_batched_rope(kv_batch_size, kv_len, head_dim, device)

        y = layer(x, kv, rope_q=rope_q, rope_k=rope_k)
        assert y.shape == (batch_size, seq_len, hidden_size)
        assert cross_attention_batches == [(expected_keep_count, kv_batch_size)]
