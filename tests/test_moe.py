import pytest
import torch
from torch.testing import assert_close

from vit.moe import MoE, MoELayerStats, MoEStats


HIDDEN_SIZE = 32
FFN_HIDDEN_SIZE = 64
NUM_EXPERTS = 4
CAPACITY_FRACTION = 0.25
SMALL_PROB = 1e-4


def _logits_from_probs(probs: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    return probs.log().view(1, 1, -1).expand(batch_size, sequence_length, -1).clone()


class TestMoE:
    def test_forward_with_aux(self, device):
        layer = MoE(
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
        layer = MoE(
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
        layer = MoE(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            token_top_k=2,
            dropout=0.2,
            norm_type="rmsnorm",
        ).to(device)
        layer.eval()
        x = torch.randn(2, 16, HIDDEN_SIZE, device=device)
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

    @pytest.mark.parametrize(("drop_overflow_tokens", "expected_dropped"), [(True, True), (False, False)])
    def test_drop_overflow_tokens_controls_fallback(self, device, drop_overflow_tokens, expected_dropped):
        layer = MoE(
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

    def test_batch_prioritized_routing_applies_capacity_across_topk_ranks(self, device):
        hidden_size = 2
        token_top_k = 2
        num_experts = 2
        capacity_factor = 0.5  # capacity = ceil((num_tokens * token_top_k / num_experts) * factor) = 1
        layer = MoE(
            hidden_size=hidden_size,
            ffn_hidden_size=8,
            num_experts=num_experts,
            token_top_k=token_top_k,
            capacity_factor=capacity_factor,
            drop_overflow_tokens=True,
            norm_type="rmsnorm",
            dropout=0.0,
        ).to(device)
        layer.eval()
        with torch.no_grad():
            layer.router.weight.copy_(torch.tensor([[10.0, 20.0], [9.0, -20.0]], device=device))
            if layer.router.bias is not None:
                layer.router.bias.zero_()

        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device)
        _, _, expert_token_counts, dropped_token_count, capacity = layer.forward_with_aux(x)

        assert int(capacity.item()) == 1
        assert_close(expert_token_counts, torch.tensor([1, 1], device=device, dtype=torch.int64))
        assert int(dropped_token_count.item()) == 1

    def test_invalid_capacity_factor_raises(self):
        with pytest.raises(ValueError, match="capacity_factor must be > 0"):
            MoE(hidden_size=16, ffn_hidden_size=32, num_experts=2, token_top_k=1, capacity_factor=0.0)

    @pytest.mark.parametrize(("token_top_k", "num_experts"), [(0, 4), (5, 4)])
    def test_invalid_token_top_k_raises(self, token_top_k, num_experts):
        with pytest.raises(ValueError, match="token_top_k must be"):
            MoE(
                hidden_size=16,
                ffn_hidden_size=32,
                num_experts=num_experts,
                token_top_k=token_top_k,
            )


class TestMoELoss:
    def test_balanced_distribution_has_lower_loss_than_collapsed_distribution(self):
        num_experts = 4
        num_assignments = 64
        batch_size = 2
        sequence_length = 16

        balanced_probs = torch.full((num_experts,), 1.0 / num_experts)
        balanced_stats = MoELayerStats(
            router_logits=_logits_from_probs(balanced_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([16, 16, 16, 16], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(16, dtype=torch.int64),
        )

        collapsed_probs = torch.tensor([1.0 - 3 * SMALL_PROB, SMALL_PROB, SMALL_PROB, SMALL_PROB])
        collapsed_stats = MoELayerStats(
            router_logits=_logits_from_probs(collapsed_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([num_assignments, 0, 0, 0], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_assignments, dtype=torch.int64),
        )

        balanced_loss = balanced_stats.load_balancing_loss()
        collapsed_loss = collapsed_stats.load_balancing_loss()
        assert balanced_loss < collapsed_loss

    def test_prefers_more_balanced_routing_distributions(self):
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
        )
        moderate_skew_stats = MoELayerStats(
            router_logits=_logits_from_probs(moderate_skew_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([40, 12, 8, 4], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_assignments, dtype=torch.int64),
        )
        heavy_skew_stats = MoELayerStats(
            router_logits=_logits_from_probs(heavy_skew_probs, batch_size, sequence_length),
            expert_token_counts=torch.tensor([58, 4, 1, 1], dtype=torch.int64),
            dropped_token_count=torch.tensor(0, dtype=torch.int64),
            capacity=torch.tensor(num_assignments, dtype=torch.int64),
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
        )
        stats = MoEStats(layers={0: layer0, 1: layer1})

        expected = torch.stack([layer0.load_balancing_loss(), layer1.load_balancing_loss()]).mean()
        assert_close(stats.load_balancing_loss(), expected)
