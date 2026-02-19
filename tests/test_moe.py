import pytest
import torch
from torch.testing import assert_close

from vit.moe import ExpertChoiceMoE, MoELayerStats, MoEStats


class TestExpertChoiceMoE:
    def test_forward_with_aux(self, device):
        layer = ExpertChoiceMoE(
            hidden_size=32,
            ffn_hidden_size=64,
            num_experts=4,
            activation="swiglu",
            norm_type="rmsnorm",
        ).to(device)
        x = torch.randn(2, 16, 32, device=device)
        y, router_logits, expert_token_counts, dropped_token_count, capacity = layer.forward_with_aux(x)
        assert y.shape == x.shape
        assert router_logits.shape == (2, 16, 4)
        assert expert_token_counts.shape == (4,)
        assert dropped_token_count.ndim == 0
        assert capacity.ndim == 0

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_backward_with_aux_loss(self, device, norm_type):
        layer = ExpertChoiceMoE(
            hidden_size=32,
            ffn_hidden_size=64,
            num_experts=4,
            activation="gelu",
            norm_type=norm_type,
        ).to(device)
        x = torch.randn(2, 16, 32, device=device)
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
            hidden_size=32,
            ffn_hidden_size=64,
            num_experts=4,
            dropout=0.2,
            norm_type="rmsnorm",
        ).to(device)
        layer.eval()
        x = torch.randn(2, 16, 32, device=device)
        y1 = layer(x)
        y2 = layer(x)
        assert_close(y1, y2)

    def test_invalid_capacity_factor_raises(self):
        with pytest.raises(ValueError, match="capacity_factor must be > 0"):
            ExpertChoiceMoE(hidden_size=16, ffn_hidden_size=32, num_experts=2, capacity_factor=0.0)
