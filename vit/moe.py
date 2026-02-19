import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import get_activation
from .norm import NormType, apply_norm, get_norm_bias, is_layer_norm, make_norm


ROUTING_MODE_EXPERT_CHOICE = "expert_choice"
ROUTING_MODE_TOKEN_CHOICE = "token_choice"
RoutingMode = Literal["expert_choice", "token_choice"]
CV2_EPS = 1e-10


@torch.compile(
    fullgraph=True,
    dynamic=True,
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
    dynamic=True,
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
    routing_mode: RoutingMode = ROUTING_MODE_EXPERT_CHOICE

    def load_balancing_loss(self) -> Tensor:
        router_probs = torch.softmax(self.router_logits, dim=-1)
        if self.routing_mode == ROUTING_MODE_TOKEN_CHOICE:
            return self._vmoe_style_load_balancing_loss(router_probs)
        return self._switch_style_load_balancing_loss(router_probs)

    def _switch_style_load_balancing_loss(self, router_probs: Tensor) -> Tensor:
        importance = router_probs.mean(dim=(0, 1))
        total_tokens = self.router_logits.shape[0] * self.router_logits.shape[1]
        total_tokens = max(total_tokens, 1)
        load = self.expert_token_counts.to(dtype=self.router_logits.dtype) / float(total_tokens)
        return router_probs.shape[-1] * (importance * load).sum()

    def _vmoe_style_load_balancing_loss(self, router_probs: Tensor) -> Tensor:
        importance = router_probs.mean(dim=(0, 1))
        total_assignments = self.expert_token_counts.sum().clamp_min(1)
        load = self.expert_token_counts.to(dtype=self.router_logits.dtype) / total_assignments.to(
            dtype=self.router_logits.dtype
        )
        return _cv_squared(importance) + _cv_squared(load)


@dataclass
class MoEStats:
    layers: dict[int, MoELayerStats]

    def load_balancing_loss(self) -> Tensor:
        if not self.layers:
            return torch.tensor(0.0)
        losses = [self.layers[i].load_balancing_loss() for i in sorted(self.layers)]
        return torch.stack(losses).mean()


def _cv_squared(x: Tensor) -> Tensor:
    mean = x.mean()
    variance = x.var(unbiased=False)
    return variance / (mean.square() + CV2_EPS)


class _PackedExpertsMoE(nn.Module):
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
        routing_mode: RoutingMode = ROUTING_MODE_EXPERT_CHOICE,
        num_zero_experts: int = 0,
        num_copy_experts: int = 0,
        num_constant_experts: int = 0,
    ):
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}")
        if router_jitter_noise < 0:
            raise ValueError(f"router_jitter_noise must be >= 0, got {router_jitter_noise}")
        if num_zero_experts < 0:
            raise ValueError(f"num_zero_experts must be >= 0, got {num_zero_experts}")
        if num_copy_experts < 0:
            raise ValueError(f"num_copy_experts must be >= 0, got {num_copy_experts}")
        if num_constant_experts < 0:
            raise ValueError(f"num_constant_experts must be >= 0, got {num_constant_experts}")
        num_simple_experts = num_zero_experts + num_copy_experts + num_constant_experts
        if num_simple_experts > num_experts:
            raise ValueError(f"total simple experts ({num_simple_experts}) must be <= num_experts ({num_experts})")
        num_mlp_experts = num_experts - num_simple_experts
        if num_mlp_experts <= 0:
            raise ValueError("at least one MLP expert is required")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_experts = num_experts
        self.num_zero_experts = num_zero_experts
        self.num_copy_experts = num_copy_experts
        self.num_constant_experts = num_constant_experts
        self.num_simple_experts = num_simple_experts
        self.num_mlp_experts = num_mlp_experts
        self.expert_capacity_factor = capacity_factor
        self.router_jitter_noise = router_jitter_noise
        self.drop_overflow_tokens = drop_overflow_tokens
        self.routing_mode = routing_mode
        self.zero_expert_end = num_zero_experts
        self.copy_expert_end = self.zero_expert_end + num_copy_experts
        self.constant_expert_end = self.copy_expert_end + num_constant_experts
        self.mlp_expert_start = self.constant_expert_end

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

        self.fc1_weight = nn.Parameter(torch.empty(num_mlp_experts, fc1_out, hidden_size, **factory_kwargs))
        self.fc2_weight = nn.Parameter(torch.empty(num_mlp_experts, hidden_size, ffn_hidden_size, **factory_kwargs))
        if bias:
            self.fc1_bias = nn.Parameter(torch.empty(num_mlp_experts, fc1_out, **factory_kwargs))
            self.fc2_bias = nn.Parameter(torch.empty(num_mlp_experts, hidden_size, **factory_kwargs))
        else:
            self.fc1_bias = None
            self.fc2_bias = None
        if num_constant_experts > 0:
            self.constant_expert_vectors = nn.Parameter(
                torch.empty(num_constant_experts, hidden_size, **factory_kwargs)
            )
        else:
            self.constant_expert_vectors = None
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
        if self.constant_expert_vectors is not None:
            nn.init.constant_(self.constant_expert_vectors, 0.0)
        self.apply_quantization(self.quantization_config)

    def apply_quantization(self, quantization_config: Any | None) -> None:
        # Quantization for expert-packed parameters is not supported yet.
        _ = quantization_config

    def _apply_input_norm(self, x: Tensor) -> Tensor:
        norm_bias = get_norm_bias(self.norm)
        return apply_norm(
            x,
            self.norm.weight,
            norm_bias,
            self.norm.eps or 1e-5,
            use_layer_norm=self._use_layer_norm,
        )

    def _compute_router_tensors(
        self, x_flat: Tensor, batch_size: int, sequence_length: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        router_logits_flat = self.router(x_flat)
        router_logits = router_logits_flat.view(batch_size, sequence_length, self.num_experts)
        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits_flat = router_logits.reshape(x_flat.shape[0], self.num_experts)
        router_probs_flat = torch.softmax(router_logits_flat, dim=-1)
        return router_logits, router_logits_flat, router_probs_flat

    def _run_expert(self, expert_idx: int, x_tokens: Tensor) -> Tensor:
        if expert_idx < self.zero_expert_end:
            return torch.zeros_like(x_tokens)
        if expert_idx < self.copy_expert_end:
            return x_tokens
        if expert_idx < self.constant_expert_end:
            if self.constant_expert_vectors is None:
                raise RuntimeError("constant expert vectors are not initialized")
            constant_idx = expert_idx - self.copy_expert_end
            return self.constant_expert_vectors[constant_idx : constant_idx + 1].expand(x_tokens.shape[0], -1)

        mlp_expert_idx = expert_idx - self.mlp_expert_start
        fc1_bias = self.fc1_bias[mlp_expert_idx] if self.fc1_bias is not None else None
        fc2_bias = self.fc2_bias[mlp_expert_idx] if self.fc2_bias is not None else None
        if self._is_glu:
            return _expert_mlp_glu(
                x_tokens,
                self.fc1_weight[mlp_expert_idx],
                fc1_bias,
                self.fc2_weight[mlp_expert_idx],
                fc2_bias,
                self.activation,
                self.dropout.p,
                self.training,
                self.limit,
                self.extra_bias,
            )
        return _expert_mlp(
            x_tokens,
            self.fc1_weight[mlp_expert_idx],
            fc1_bias,
            self.fc2_weight[mlp_expert_idx],
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
        if token_indices.numel() == 0:
            return
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

    def _init_dispatch_buffers(self, x_flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        num_tokens = x_flat.shape[0]
        output = torch.zeros_like(x_flat)
        gate_sums = torch.zeros(num_tokens, 1, device=x_flat.device, dtype=x_flat.dtype)
        expert_token_counts = torch.zeros(self.num_experts, device=x_flat.device, dtype=torch.int64)
        return output, gate_sums, expert_token_counts

    def _normalize_dispatched_output(self, output: Tensor, gate_sums: Tensor) -> Tensor:
        assigned = gate_sums.squeeze(-1) > 0
        if assigned.any():
            output[assigned] = output[assigned] / gate_sums[assigned]
        return assigned

    def _route_fallback_tokens(
        self,
        *,
        assigned: Tensor,
        x_flat: Tensor,
        router_logits_flat: Tensor,
        router_probs_flat: Tensor,
        output: Tensor,
        gate_sums: Tensor,
        expert_token_counts: Tensor,
    ) -> Tensor:
        fallback_indices = (~assigned).nonzero(as_tuple=False).squeeze(-1)
        if fallback_indices.numel() == 0:
            return assigned
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
        return gate_sums.squeeze(-1) > 0

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        out, _, _, _, _ = self.forward_with_aux(x)
        return out

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)


class ExpertChoiceMoE(_PackedExpertsMoE):
    def __init__(self, *args: Any, **kwargs: Any):
        use_simple_experts = kwargs.pop("use_simple_experts", False)
        num_zero_experts = kwargs.pop("num_zero_experts", 0)
        num_copy_experts = kwargs.pop("num_copy_experts", 0)
        num_constant_experts = kwargs.pop("num_constant_experts", 0)
        if use_simple_experts or num_zero_experts > 0 or num_copy_experts > 0 or num_constant_experts > 0:
            raise ValueError("simple experts are only supported for token_choice routing")
        super().__init__(*args, routing_mode=ROUTING_MODE_EXPERT_CHOICE, **kwargs)

    def _compute_capacity(self, num_tokens: int) -> int:
        return max(1, math.ceil(self.expert_capacity_factor * num_tokens / self.num_experts))

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = self._apply_input_norm(x)
        batch_size, sequence_length, hidden_size = x.shape
        num_tokens = batch_size * sequence_length
        capacity = self._compute_capacity(num_tokens)
        top_k = min(capacity, num_tokens)

        x_flat = x.reshape(num_tokens, hidden_size)
        router_logits, router_logits_flat, router_probs_flat = self._compute_router_tensors(
            x_flat,
            batch_size,
            sequence_length,
        )
        output, gate_sums, expert_token_counts = self._init_dispatch_buffers(x_flat)

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

        assigned = self._normalize_dispatched_output(output, gate_sums)

        if not self.drop_overflow_tokens and (~assigned).any():
            assigned = self._route_fallback_tokens(
                assigned=assigned,
                x_flat=x_flat,
                router_logits_flat=router_logits_flat,
                router_probs_flat=router_probs_flat,
                output=output,
                gate_sums=gate_sums,
                expert_token_counts=expert_token_counts,
            )

        dropped_token_count = (~assigned).to(dtype=torch.int64).sum()
        capacity_tensor = torch.tensor(capacity, device=x.device, dtype=torch.int64)
        return (
            output.view(batch_size, sequence_length, hidden_size),
            router_logits,
            expert_token_counts,
            dropped_token_count,
            capacity_tensor,
        )


class TokenChoiceMoE(_PackedExpertsMoE):
    def __init__(
        self,
        *args: Any,
        token_top_k: int = 2,
        use_simple_experts: bool = False,
        num_zero_experts: int = 0,
        num_copy_experts: int = 0,
        num_constant_experts: int = 0,
        **kwargs: Any,
    ):
        num_experts = kwargs.get("num_experts")
        if num_experts is None and len(args) >= 3:
            num_experts = args[2]
        if token_top_k <= 0:
            raise ValueError(f"token_top_k must be > 0, got {token_top_k}")
        if isinstance(num_experts, int) and token_top_k > num_experts:
            raise ValueError(f"token_top_k must be <= num_experts ({num_experts}), got {token_top_k}")
        if not use_simple_experts and (num_zero_experts > 0 or num_copy_experts > 0 or num_constant_experts > 0):
            raise ValueError("simple expert counts require use_simple_experts=True")
        super().__init__(
            *args,
            routing_mode=ROUTING_MODE_TOKEN_CHOICE,
            num_zero_experts=num_zero_experts if use_simple_experts else 0,
            num_copy_experts=num_copy_experts if use_simple_experts else 0,
            num_constant_experts=num_constant_experts if use_simple_experts else 0,
            **kwargs,
        )
        self.token_top_k = token_top_k
        self.use_simple_experts = use_simple_experts

    def _compute_capacity(self, num_tokens: int, token_top_k: int) -> int:
        assignments = num_tokens * token_top_k
        return max(1, math.ceil(self.expert_capacity_factor * assignments / self.num_experts))

    def _batch_prioritized_assignments(
        self,
        *,
        router_probs_flat: Tensor,
        token_top_k: int,
        capacity: int,
    ) -> tuple[Tensor, Tensor]:
        top_scores, top_experts = torch.topk(router_probs_flat, k=token_top_k, dim=-1, sorted=True)
        token_priority = top_scores[:, 0]
        num_tokens = router_probs_flat.shape[0]
        priority_order = torch.argsort(token_priority, descending=True)
        ordered_token_indices = torch.arange(
            num_tokens, device=router_probs_flat.device, dtype=torch.long
        ).index_select(0, priority_order)
        ordered_experts = top_experts.index_select(0, priority_order)

        expert_load = torch.zeros(self.num_experts, device=router_probs_flat.device, dtype=torch.int64)
        selected_token_indices: list[Tensor] = []
        selected_expert_indices: list[Tensor] = []

        for rank_idx in range(token_top_k):
            candidate_experts = ordered_experts[:, rank_idx]
            one_hot = F.one_hot(candidate_experts, num_classes=self.num_experts).to(dtype=torch.int64)
            local_positions = torch.cumsum(one_hot, dim=0) - 1
            local_positions = local_positions.gather(1, candidate_experts.unsqueeze(-1)).squeeze(-1)
            global_positions = expert_load.index_select(0, candidate_experts) + local_positions
            keep_mask = global_positions < capacity
            if not keep_mask.any():
                continue

            kept_token_indices = ordered_token_indices[keep_mask]
            kept_expert_indices = candidate_experts[keep_mask]
            selected_token_indices.append(kept_token_indices)
            selected_expert_indices.append(kept_expert_indices)
            expert_load = expert_load + torch.bincount(kept_expert_indices, minlength=self.num_experts).to(
                dtype=torch.int64
            )

        if not selected_token_indices:
            return (
                torch.empty(0, device=router_probs_flat.device, dtype=torch.long),
                torch.empty(0, device=router_probs_flat.device, dtype=torch.long),
            )

        return torch.cat(selected_token_indices, dim=0), torch.cat(selected_expert_indices, dim=0)

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = self._apply_input_norm(x)
        batch_size, sequence_length, hidden_size = x.shape
        num_tokens = batch_size * sequence_length
        token_top_k = min(self.token_top_k, self.num_experts)
        capacity = self._compute_capacity(num_tokens, token_top_k)

        x_flat = x.reshape(num_tokens, hidden_size)
        router_logits, router_logits_flat, router_probs_flat = self._compute_router_tensors(
            x_flat,
            batch_size,
            sequence_length,
        )
        output, gate_sums, expert_token_counts = self._init_dispatch_buffers(x_flat)

        selected_token_indices, selected_expert_indices = self._batch_prioritized_assignments(
            router_probs_flat=router_probs_flat,
            token_top_k=token_top_k,
            capacity=capacity,
        )

        for expert_idx in range(self.num_experts):
            assignment_mask = selected_expert_indices == expert_idx
            if not assignment_mask.any():
                continue
            token_indices = selected_token_indices[assignment_mask]
            self._dispatch_tokens_to_expert(
                expert_idx=expert_idx,
                token_indices=token_indices,
                x_flat=x_flat,
                router_probs_flat=router_probs_flat,
                output=output,
                gate_sums=gate_sums,
                expert_token_counts=expert_token_counts,
            )

        assigned = self._normalize_dispatched_output(output, gate_sums)

        if not self.drop_overflow_tokens and (~assigned).any():
            assigned = self._route_fallback_tokens(
                assigned=assigned,
                x_flat=x_flat,
                router_logits_flat=router_logits_flat,
                router_probs_flat=router_probs_flat,
                output=output,
                gate_sums=gate_sums,
                expert_token_counts=expert_token_counts,
            )

        dropped_token_count = (~assigned).to(dtype=torch.int64).sum()
        capacity_tensor = torch.tensor(capacity, device=x.device, dtype=torch.int64)
        return (
            output.view(batch_size, sequence_length, hidden_size),
            router_logits,
            expert_token_counts,
            dropped_token_count,
            capacity_tensor,
        )
