import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import get_activation
from .norm import NormType, apply_norm, get_norm_bias, is_layer_norm, make_norm


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "layout_optimization": True,
        "epilogue_fusion": True,
        "aggressive_fusion": True,
    },
)
def _expert_mlp(
    x: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    dropout: float,
    training: bool,
) -> Tensor:
    x = F.linear(x, fc1_weight, fc1_bias)
    x = activation(x)
    x = F.dropout(x, p=dropout, training=training)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    return x


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "layout_optimization": True,
        "epilogue_fusion": True,
        "aggressive_fusion": True,
    },
)
def _expert_mlp_glu(
    x: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    dropout: float,
    training: bool,
    limit: float | None,
    extra_bias: float | None,
) -> Tensor:
    x = F.linear(x, fc1_weight, fc1_bias)
    x_linear, x_glu = x.chunk(2, dim=-1)
    if limit is not None:
        x_linear = x_linear.clamp(min=-limit, max=limit)
        x_glu = x_glu.clamp(min=None, max=limit)
    if extra_bias is not None:
        x_linear = x_linear + extra_bias
    x = activation(x_glu) * x_linear
    x = F.dropout(x, p=dropout, training=training)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    return x


@dataclass
class MoELayerStats:
    router_logits: Tensor
    expert_token_counts: Tensor
    dropped_token_count: Tensor
    capacity: Tensor

    def load_balancing_loss(self) -> Tensor:
        router_probs = torch.softmax(self.router_logits, dim=-1)
        importance = router_probs.mean(dim=(0, 1))
        total_tokens = self.router_logits.shape[0] * self.router_logits.shape[1]
        total_tokens = max(total_tokens, 1)
        load = self.expert_token_counts.to(dtype=self.router_logits.dtype) / float(total_tokens)
        return router_probs.shape[-1] * (importance * load).sum()


@dataclass
class MoEStats:
    layers: dict[int, MoELayerStats]

    def load_balancing_loss(self) -> Tensor:
        if not self.layers:
            return torch.tensor(0.0)
        losses = [self.layers[i].load_balancing_loss() for i in sorted(self.layers)]
        return torch.stack(losses).mean()


class ExpertChoiceMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        bias: bool = True,
        activation: str = "gelu",
        eps: float = 1e-5,
        dropout: float = 0.1,
        limit: float | None = None,
        extra_bias: float | None = None,
        capacity_factor: float = 1.0,
        router_jitter_noise: float = 0.0,
        drop_overflow_tokens: bool = True,
        quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
    ):
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}")
        if router_jitter_noise < 0:
            raise ValueError(f"router_jitter_noise must be >= 0, got {router_jitter_noise}")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity_factor = capacity_factor
        self.router_jitter_noise = router_jitter_noise
        self.drop_overflow_tokens = drop_overflow_tokens

        self.norm = make_norm(hidden_size, norm_type, eps=eps, **factory_kwargs)
        self._use_layer_norm = is_layer_norm(norm_type)
        self.router = nn.Linear(hidden_size, num_experts, bias=bias, **factory_kwargs)

        if activation.endswith("glu"):
            self._is_glu = True
            fc1_out = 2 * ffn_hidden_size
        else:
            self._is_glu = False
            fc1_out = ffn_hidden_size
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.limit = limit
        self.extra_bias = extra_bias
        self.quantization_config = quantization_config

        self.fc1_weight = nn.Parameter(torch.empty(num_experts, fc1_out, hidden_size, **factory_kwargs))
        self.fc2_weight = nn.Parameter(torch.empty(num_experts, hidden_size, ffn_hidden_size, **factory_kwargs))
        if bias:
            self.fc1_bias = nn.Parameter(torch.empty(num_experts, fc1_out, **factory_kwargs))
            self.fc2_bias = nn.Parameter(torch.empty(num_experts, hidden_size, **factory_kwargs))
        else:
            self.fc1_bias = None
            self.fc2_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.router.weight, std=0.02)
        if self.router.bias is not None:
            nn.init.constant_(self.router.bias, 0.0)
        nn.init.trunc_normal_(self.fc1_weight, std=0.02)
        nn.init.trunc_normal_(self.fc2_weight, std=0.02)
        if self.fc1_bias is not None:
            nn.init.constant_(self.fc1_bias, 0.0)
        if self.fc2_bias is not None:
            nn.init.constant_(self.fc2_bias, 0.0)
        self.apply_quantization(self.quantization_config)

    def apply_quantization(self, quantization_config: Any | None) -> None:
        # Quantization for expert-packed parameters is not supported yet.
        _ = quantization_config

    def _compute_capacity(self, num_tokens: int) -> int:
        return max(1, math.ceil(self.expert_capacity_factor * num_tokens / self.num_experts))

    def _run_expert(self, expert_idx: int, x: Tensor) -> Tensor:
        fc1_bias = self.fc1_bias[expert_idx] if self.fc1_bias is not None else None
        fc2_bias = self.fc2_bias[expert_idx] if self.fc2_bias is not None else None
        if self._is_glu:
            return _expert_mlp_glu(
                x,
                self.fc1_weight[expert_idx],
                fc1_bias,
                self.fc2_weight[expert_idx],
                fc2_bias,
                self.activation,
                self.dropout.p,
                self.training,
                self.limit,
                self.extra_bias,
            )
        return _expert_mlp(
            x,
            self.fc1_weight[expert_idx],
            fc1_bias,
            self.fc2_weight[expert_idx],
            fc2_bias,
            self.activation,
            self.dropout.p,
            self.training,
        )

    def _dispatch_tokens_to_expert(
        self,
        *,
        expert_idx: int,
        token_indices: Tensor,
        x_flat: Tensor,
        router_probs_flat: Tensor,
        output: Tensor,
        gate_sums: Tensor,
        expert_token_counts: Tensor,
        overwrite: bool = False,
    ) -> None:
        expert_inputs = x_flat.index_select(0, token_indices)
        expert_outputs = self._run_expert(expert_idx, expert_inputs)
        expert_gates = router_probs_flat.index_select(0, token_indices)[:, expert_idx : expert_idx + 1]
        if overwrite:
            output.index_copy_(0, token_indices, expert_outputs * expert_gates)
            gate_sums.index_copy_(0, token_indices, expert_gates)
        else:
            output.index_add_(0, token_indices, expert_outputs * expert_gates)
            gate_sums.index_add_(0, token_indices, expert_gates)
        expert_token_counts[expert_idx] += token_indices.numel()

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        norm_bias = get_norm_bias(self.norm)
        x = apply_norm(
            x,
            self.norm.weight,
            norm_bias,
            self.norm.eps or 1e-5,
            use_layer_norm=self._use_layer_norm,
        )

        B, L, D = x.shape
        num_tokens = B * L
        capacity = self._compute_capacity(num_tokens)
        top_k = min(capacity, num_tokens)

        x_flat = x.reshape(num_tokens, D)
        router_logits_flat = self.router(x_flat)
        router_logits = router_logits_flat.view(B, L, self.num_experts)
        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits_flat = router_logits.reshape(num_tokens, self.num_experts)
        router_probs_flat = torch.softmax(router_logits_flat, dim=-1)

        output = torch.zeros_like(x_flat)
        gate_sums = torch.zeros(num_tokens, 1, device=x.device, dtype=x.dtype)
        expert_token_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.int64)

        for expert_idx in range(self.num_experts):
            expert_scores = router_logits_flat[:, expert_idx]
            _, token_indices = torch.topk(expert_scores, k=top_k, dim=0, sorted=False)
            self._dispatch_tokens_to_expert(
                expert_idx=expert_idx,
                token_indices=token_indices,
                x_flat=x_flat,
                router_probs_flat=router_probs_flat,
                output=output,
                gate_sums=gate_sums,
                expert_token_counts=expert_token_counts,
            )

        assigned = gate_sums.squeeze(-1) > 0
        if assigned.any():
            output[assigned] = output[assigned] / gate_sums[assigned]

        if not self.drop_overflow_tokens and (~assigned).any():
            fallback_indices = (~assigned).nonzero(as_tuple=False).squeeze(-1)
            fallback_experts = router_logits_flat.index_select(0, fallback_indices).argmax(dim=-1)
            for expert_idx in range(self.num_experts):
                expert_mask = fallback_experts == expert_idx
                if expert_mask.any():
                    token_indices = fallback_indices[expert_mask]
                    self._dispatch_tokens_to_expert(
                        expert_idx=expert_idx,
                        token_indices=token_indices,
                        x_flat=x_flat,
                        router_probs_flat=router_probs_flat,
                        output=output,
                        gate_sums=gate_sums,
                        expert_token_counts=expert_token_counts,
                        overwrite=True,
                    )
            assigned = gate_sums.squeeze(-1) > 0

        dropped_token_count = (~assigned).to(dtype=torch.int64).sum()
        capacity_tensor = torch.tensor(capacity, device=x.device, dtype=torch.int64)
        return output.view(B, L, D), router_logits, expert_token_counts, dropped_token_count, capacity_tensor

    def forward(self, x: Tensor) -> Tensor:
        out, _, _, _, _ = self.forward_with_aux(x)
        return out

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
