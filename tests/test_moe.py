import pytest
import torch
from torch.testing import assert_close

from vit.moe import (
    ROUTING_MODE_TOKEN_CHOICE,
    ExpertChoiceMoE,
    MoELayerStats,
    MoEStats,
    TokenChoiceMoE,
)


HIDDEN_SIZE = 32
FFN_HIDDEN_SIZE = 64
NUM_EXPERTS = 4
CAPACITY_FRACTION = 0.25
SMALL_PROB = 1e-4


def _logits_from_probs(probs: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    return probs.log().view(1, 1, -1).expand(batch_size, sequence_length, -1).clone()


class TestExpertChoiceMoE:
    def test_forward_with_aux(self, device):
        layer = ExpertChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            activation="swiglu",
            norm_type="rmsnorm",
        ).to(device)
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_aux(x)
        assert y.shape == x.shape
        assert router_logits.shape == (2, 16, NUM_EXPERTS)
        assert expert_token_counts.shape == (NUM_EXPERTS,)
        assert dropped_token_count.ndim == 0
        assert capacity.ndim == 0

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward_with_aux_loss(self, device, norm_type):
        layer = ExpertChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            activation="gelu",
            norm_type=norm_type,
        ).to(device)
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_aux(x)
        stats = MoEStats(
            layers={
                0: MoELayerStats(
                    router_logits=router_logits,
                    expert_token_counts=expert_token_counts,
                    dropped_token_count=dropped_token_count,
                    capacity=capacity,
                )
            }
        )

        loss = y.sum() + 0.1 * stats.load_balancing_loss()
        loss.backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_deterministic_in_eval(self, device):
        torch.manual_seed(0)
        layer = ExpertChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            dropout=0.2,
            norm_type="rmsnorm",
        ).to(device)
        layer.eval()
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

    def test_invalid_capacity_factor_raises(self):
        with pytest.raises(ValueError, match="capacity_factor must be > 0"):
            ExpertChoiceMoE(hidden_size=16, ffn_hidden_size=32, num_experts=2, capacity_factor=0.0)

    def test_simple_experts_not_supported(self):
        with pytest.raises(ValueError, match="simple experts are only supported for token_choice routing"):
            ExpertChoiceMoE(
                hidden_size=16,
                ffn_hidden_size=32,
                num_experts=2,
                use_simple_experts=True,
            )


class TestTokenChoiceMoE:
    def test_forward_with_aux(self, device):
        layer = TokenChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            token_top_k=2,
            activation="swiglu",
            norm_type="rmsnorm",
        ).to(device)
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_aux(x)
        assert y.shape == x.shape
        assert router_logits.shape == (2, 16, NUM_EXPERTS)
        assert expert_token_counts.shape == (NUM_EXPERTS,)
        assert dropped_token_count.ndim == 0
        assert capacity.ndim == 0

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward_with_aux_loss(self, device, norm_type):
        layer = TokenChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            token_top_k=2,
            activation="gelu",
            norm_type=norm_type,
        ).to(device)
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_aux(x)
        stats = MoEStats(
            layers={
                0: MoELayerStats(
                    router_logits=router_logits,
                    expert_token_counts=expert_token_counts,
                    dropped_token_count=dropped_token_count,
                    capacity=capacity,
                    routing_mode=ROUTING_MODE_TOKEN_CHOICE,
                )
            }
        )

        loss = y.sum() + 0.1 * stats.load_balancing_loss()
        loss.backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize(("drop_overflow_tokens", "expected_dropped"), [(True, True), (False, False)])
    def test_drop_overflow_tokens_controls_fallback(self, device, drop_overflow_tokens, expected_dropped):
        layer = TokenChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            token_top_k=1,
            capacity_factor=CAPACITY_FRACTION,
            drop_overflow_tokens=drop_overflow_tokens,
            norm_type="rmsnorm",
        ).to(device)
        with torch.no_grad():
            assert layer.router.bias is not None
            layer.router.weight.zero_()
            layer.router.bias.zero_()
            layer.router.bias[0] = 10.0

        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        _, _, _, dropped_token_count, _ = layer.forward_with_aux(x)
        dropped_tokens = int(dropped_token_count.item())
        if expected_dropped:
            assert dropped_tokens > 0
        else:
            assert dropped_tokens == 0

    @pytest.mark.parametrize(("token_top_k", "num_experts"), [(0, 4), (5, 4)])
    def test_invalid_token_top_k_raises(self, token_top_k, num_experts):
        with pytest.raises(ValueError, match="token_top_k must be"):
            TokenChoiceMoE(
                hidden_size=16,
                ffn_hidden_size=32,
                num_experts=num_experts,
                token_top_k=token_top_k,
            )

    def test_simple_experts_functional_behavior(self, device):
        hidden_size = 8
        layer = TokenChoiceMoE(
            hidden_size=hidden_size,
            ffn_hidden_size=16,
            num_experts=4,
            token_top_k=1,
            use_simple_experts=True,
            num_zero_experts=1,
            num_copy_experts=1,
            num_constant_experts=1,
            dropout=0.0,
            norm_type="rmsnorm",
        ).to(device)
        layer.eval()
        with torch.no_grad():
            layer.router.weight.zero_()
            if layer.router.bias is not None:
                layer.router.bias.zero_()
            layer.router.weight[0, 0] = 100.0
            layer.router.weight[1, 1] = 100.0
            layer.router.weight[2, 2] = 100.0
            layer.router.weight[3, 3] = 100.0
            assert layer.constant_expert_vectors is not None
            layer.constant_expert_vectors[0].fill_(2.5)

        x = torch.zeros(1, 4, hidden_size, device=device)
        x[0, 0, 0] = 1.0
        x[0, 1, 1] = 1.0
        x[0, 2, 2] = 1.0
        x[0, 3, 3] = 1.0
        x_norm = layer._apply_input_norm(x)
        y, _, expert_token_counts, dropped_token_count, _ = layer.forward_with_aux(x)
        assert int(dropped_token_count.item()) == 0
        assert_close(expert_token_counts, torch.ones(4, device=device, dtype=torch.int64))
        assert_close(y[0, 0], torch.zeros(hidden_size, device=device))
        assert_close(y[0, 1], x_norm[0, 1])
        assert_close(y[0, 2], torch.full((hidden_size,), 2.5, device=device))

    def test_constant_simple_experts_receive_gradients(self, device):
        layer = TokenChoiceMoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=4,
            token_top_k=2,
            use_simple_experts=True,
            num_constant_experts=2,
            norm_type="rmsnorm",
        ).to(device)
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y, _, _, _, _ = layer.forward_with_aux(x)
        y.sum().backward()
        assert layer.constant_expert_vectors is not None
        assert layer.constant_expert_vectors.grad is not None
        assert not layer.constant_expert_vectors.grad.isnan().any()

    def test_simple_expert_counts_require_flag(self):
        with pytest.raises(ValueError, match="simple expert counts require use_simple_experts=True"):
            TokenChoiceMoE(
                hidden_size=16,
                ffn_hidden_size=32,
                num_experts=4,
                token_top_k=2,
                num_constant_experts=1,
            )

    def test_simple_expert_count_cannot_exceed_total_experts(self):
        with pytest.raises(ValueError, match="total simple experts"):
            TokenChoiceMoE(
                hidden_size=16,
                ffn_hidden_size=32,
                num_experts=2,
                token_top_k=1,
                use_simple_experts=True,
                num_zero_experts=1,
                num_copy_experts=1,
                num_constant_experts=1,
            )

    def test_simple_experts_must_leave_mlp_expert(self):
        with pytest.raises(ValueError, match="at least one MLP expert is required"):
            TokenChoiceMoE(
                hidden_size=16,
                ffn_hidden_size=32,
                num_experts=2,
                token_top_k=1,
                use_simple_experts=True,
                num_zero_experts=1,
                num_copy_experts=1,
            )


class TestMoELoss:
    def test_expert_choice_balanced_distribution_has_lower_loss_than_collapsed_distribution(self):
        num_experts = 4
        num_tokens = 32

        # Balanced: uniform router probabilities + uniform expert utilization.
        balanced_logits = torch.zeros(2, num_tokens // 2, num_experts)
        balanced_counts = torch.full((num_experts,), num_tokens // num_experts, dtype=torch.int64)
        balanced_stats = MoELayerStats(
            router_logits=balanced_logits,
            expert_token_counts=balanced_counts,
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_tokens // num_experts, dtype=torch.int64),
        )

        # Collapsed: router and assignments both concentrated on one expert.
        collapsed_logits = torch.full((2, num_tokens // 2, num_experts), -20.0)
        collapsed_logits[..., 0] = 20.0
        collapsed_counts = torch.tensor([num_tokens, 0, 0, 0], dtype=torch.int64)
        collapsed_stats = MoELayerStats(
            router_logits=collapsed_logits,
            expert_token_counts=collapsed_counts,
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_tokens, dtype=torch.int64),
        )

        balanced_loss = balanced_stats.load_balancing_loss()
        collapsed_loss = collapsed_stats.load_balancing_loss()
        assert balanced_loss < collapsed_loss

    def test_token_choice_balanced_distribution_has_lower_loss_than_collapsed_distribution(self):
        num_experts = 4
        num_tokens = 32
        batch_size = 2
        sequence_length = num_tokens // batch_size

        balanced_probs = torch.full((num_experts,), 1.0 / num_experts)
        balanced_stats = MoELayerStats(
            router_logits=_logits_from_probs(balanced_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([16, 16, 16, 16], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(16, dtype=torch.int64),
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
        )

        collapsed_probs = torch.tensor([1.0 - 3 * SMALL_PROB, SMALL_PROB, SMALL_PROB, SMALL_PROB])
        collapsed_stats = MoELayerStats(
            router_logits=_logits_from_probs(collapsed_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([64, 0, 0, 0], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(64, dtype=torch.int64),
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
        )

        balanced_loss = balanced_stats.load_balancing_loss()
        collapsed_loss = collapsed_stats.load_balancing_loss()
        assert balanced_loss < collapsed_loss

    def test_token_choice_prefers_more_balanced_routing_distributions(self):
        num_experts = 4
        batch_size = 2
        sequence_length = 16
        num_assignments = 64

        balanced_probs = torch.full((num_experts,), 1.0 / num_experts)
        moderate_skew_probs = torch.tensor([0.62, 0.20, 0.12, 0.06])
        heavy_skew_probs = torch.tensor([0.95, 0.03, 0.01, 0.01])

        balanced_stats = MoELayerStats(
            router_logits=_logits_from_probs(balanced_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([16, 16, 16, 16], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(16, dtype=torch.int64),
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
        )
        moderate_skew_stats = MoELayerStats(
            router_logits=_logits_from_probs(moderate_skew_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([40, 12, 8, 4], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_assignments, dtype=torch.int64),
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
        )
        heavy_skew_stats = MoELayerStats(
            router_logits=_logits_from_probs(heavy_skew_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([58, 4, 1, 1], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_assignments, dtype=torch.int64),
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
        )

        balanced_loss = balanced_stats.load_balancing_loss()
        moderate_skew_loss = moderate_skew_stats.load_balancing_loss()
        heavy_skew_loss = heavy_skew_stats.load_balancing_loss()
        assert balanced_loss < moderate_skew_loss
        assert moderate_skew_loss < heavy_skew_loss

    def test_moe_stats_reports_mean_loss_across_layers(self):
        num_experts = 4
        layer0 = MoELayerStats(
            router_logits=torch.zeros(1, 8, num_experts),
            expert_token_counts=torch.tensor([2, 2, 2, 2], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(2, dtype=torch.int64),
        )
        layer1 = MoELayerStats(
            router_logits=torch.full((1, 8, num_experts), -20.0).index_fill_(-1, torch.tensor([0]), 20.0),
            expert_token_counts=torch.tensor([8, 0, 0, 0], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(8, dtype=torch.int64),
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
        )
        stats = MoEStats(layers={0: layer0, 1: layer1})

        expected = torch.stack([layer0.load_balancing_loss(), layer1.load_balancing_loss()]).mean()
        assert_close(stats.load_balancing_loss(), expected)
