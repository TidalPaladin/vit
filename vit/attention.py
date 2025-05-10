from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from .helpers import compile_is_disabled
from .pos_enc import compute_alibi_slopes


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def self_attention_input_projection(
    # fmt: off
    x: Tensor,
    weight: Tensor, bias: Tensor | None,
    norm_weight: Tensor,
    head_dim: int, num_attention_heads: int,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    x = F.rms_norm(x, x.shape[-1:], weight=norm_weight)
    x = F.linear(x, weight, bias)
    x = rearrange(x, "b l (p h d) -> p b h l d", h=num_attention_heads, d=head_dim)
    q = x[0]
    k = x[1]
    v = x[2]
    return q, k, v


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def cross_attention_input_projection(
    # fmt: off
    q: Tensor, kv: Tensor,
    weight_q: Tensor, bias_q: Tensor | None,
    weight_kv: Tensor, bias_kv: Tensor | None,
    norm_weight: Tensor,
    head_dim: int, num_attention_heads: int,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    q = F.rms_norm(q, q.shape[-1:], weight=norm_weight)
    q = F.linear(q, weight_q, bias_q)
    kv = F.linear(kv, weight_kv, bias_kv)
    q = rearrange(q, "b l (h d) -> b h l d", h=num_attention_heads, d=head_dim)
    kv = rearrange(kv, "b l (p h d) -> p b h l d", h=num_attention_heads, d=head_dim)
    k = kv[0]
    v = kv[1]
    return q, k, v


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def attention_output_projection(
    # fmt: off
    o: Tensor,
    proj_weight: Tensor, proj_bias: Tensor | None,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    o = rearrange(o, "b h l d -> b l (h d)")
    o = F.linear(o, proj_weight, proj_bias)
    o = F.dropout(o, p=dropout, training=training)
    return o


def create_alibi_score_mod(q_pos: Tensor, kv_pos: Tensor, slopes: Tensor) -> Callable:
    def alibi(score, b, h, q_idx, kv_idx):
        delta = q_pos[b, q_idx] - kv_pos[b, kv_idx]
        bias = slopes[h] * delta.norm(p=2, dim=-1)
        return score + bias.nan_to_num(nan=0.0)

    return alibi


flex_attention_alibi = torch.compile(
    flex_attention, fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled()
)


class SelfAttention(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        bias: bool = True,
        alibi_scale: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.has_bias = bias
        self.query = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.output = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.norm = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.alibi_slopes = nn.Parameter(compute_alibi_slopes(self.num_attention_heads, alibi_scale))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.output.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.query.weight, std=0.02)
        nn.init.trunc_normal_(self.key.weight, std=0.02)
        nn.init.trunc_normal_(self.value.weight, std=0.02)
        nn.init.trunc_normal_(self.output.weight, std=0.02)

    def track_attention_weights(self, track: bool = True) -> None:
        self._track_attention_weights = track

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        # Projections
        wqkv_packed = torch.cat([self.query.weight, self.key.weight, self.value.weight], dim=0)
        bqkv_packed = torch.cat([self.query.bias, self.key.bias, self.value.bias], dim=0) if self.has_bias else None
        q, k, v = self_attention_input_projection(
            # fmt: off
            x,
            wqkv_packed, bqkv_packed,
            self.norm.weight,
            self.head_dim, self.num_attention_heads,
            # fmt: on
        )

        # Attention
        score_mod = create_alibi_score_mod(pos, pos, self.alibi_slopes)
        o = flex_attention_alibi(
            # fmt: off
            q, k, v,
            score_mod=score_mod,
            # fmt: on
        )
        assert isinstance(o, Tensor)

        # Output projection
        o = attention_output_projection(
            # fmt: off
            o,
            self.output.weight, self.output.bias,
            self.dropout.p, self.training,
            # fmt: on
        )
        return o


class CrossAttention(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        bias: bool = True,
        alibi_scale: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.has_bias = bias
        self.query = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.output = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.norm = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.alibi_slopes = nn.Parameter(compute_alibi_slopes(self.num_attention_heads, alibi_scale))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.output.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.query.weight, std=0.02)
        nn.init.trunc_normal_(self.key.weight, std=0.02)
        nn.init.trunc_normal_(self.value.weight, std=0.02)
        nn.init.trunc_normal_(self.output.weight, std=0.02)

    def track_attention_weights(self, track: bool = True) -> None:
        self._track_attention_weights = track

    def forward(self, q: Tensor, kv: Tensor, qpos: Tensor, kvpos: Tensor) -> Tensor:
        # Projections
        wq = self.query.weight
        bq = self.query.bias if self.has_bias else None
        wkv_packed = torch.cat([self.key.weight, self.value.weight], dim=0)
        bkv_packed = torch.cat([self.key.bias, self.value.bias], dim=0) if self.has_bias else None

        q, k, v = cross_attention_input_projection(
            # fmt: off
            q, kv,
            wq, bq,
            wkv_packed, bkv_packed,
            self.norm.weight,
            self.head_dim, self.num_attention_heads,
            # fmt: on
        )

        # Attention
        score_mod = create_alibi_score_mod(qpos, kvpos, self.alibi_slopes)
        o = flex_attention_alibi(
            # fmt: off
            q, k, v,
            score_mod=score_mod,
            # fmt: on
        )
        assert isinstance(o, Tensor)

        # Output projection
        o = attention_output_projection(
            # fmt: off
            o,
            self.output.weight, self.output.bias,
            self.dropout.p, self.training,
            # fmt: on
        )
        return o
