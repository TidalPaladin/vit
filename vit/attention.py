import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from .fused import LayerNormLinear, Linear
from .helpers import compile_is_disabled


def apply_qkv_norm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    normalization: str,
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor | None,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    if q is k and k is v:
        if normalization == "LayerNorm":
            q = k = v = F.layer_norm(q, q.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
        elif normalization == "RMSNorm":
            q = k = v = F.rms_norm(q, q.shape[-1:], weight=layer_norm_weight, eps=eps)
    elif k is v:
        if normalization == "LayerNorm":
            q = F.layer_norm(q, q.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
            k = v = F.layer_norm(k, k.shape[-1:], weight=layer_norm_weight, eps=eps)
        elif normalization == "RMSNorm":
            q = F.rms_norm(q, q.shape[-1:], weight=layer_norm_weight, eps=eps)
            k = v = F.rms_norm(k, k.shape[-1:], weight=layer_norm_weight, eps=eps)
    else:
        if normalization == "LayerNorm":
            q = F.layer_norm(q, q.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
            k = F.layer_norm(k, k.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
            v = F.layer_norm(v, v.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
        elif normalization == "RMSNorm":
            q = F.rms_norm(q, q.shape[-1:], weight=layer_norm_weight, eps=eps)
            k = F.rms_norm(k, k.shape[-1:], weight=layer_norm_weight, eps=eps)
            v = F.rms_norm(v, v.shape[-1:], weight=layer_norm_weight, eps=eps)
    return q, k, v


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def forward_input_projection_fused(
    # fmt: off
    q: Tensor, k: Tensor, v: Tensor,
    query_weight: Tensor, key_weight: Tensor, value_weight: Tensor,
    query_bias: Tensor | None, key_bias: Tensor | None, value_bias: Tensor | None,
    normalization: str,
    layer_norm_weight: Tensor, layer_norm_bias: Tensor | None,
    qkv_format: Literal["sbhd", "bshd"],
    num_attention_heads: int,
    num_gqa_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    q, k, v = apply_qkv_norm(q, k, v, normalization, layer_norm_weight, layer_norm_bias, eps)

    q = F.linear(q, query_weight, query_bias)
    k = F.linear(k, key_weight, key_bias)
    v = F.linear(v, value_weight, value_bias)

    if qkv_format == "sbhd":
        q = rearrange(q, "s b (h d) -> b h s d", h=num_attention_heads)
        k = rearrange(k, "s b (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
        v = rearrange(v, "s b (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
    elif qkv_format == "bshd":
        q = rearrange(q, "b s (h d) -> b h s d", h=num_attention_heads)
        k = rearrange(k, "b s (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
        v = rearrange(v, "b s (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
    else:
        raise ValueError(f"Invalid qkv_format: {qkv_format}")

    return q, k, v


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def forward_input_projection(
    # fmt: off
    q: Tensor, k: Tensor, v: Tensor,
    query_weight: Tensor, key_weight: Tensor, value_weight: Tensor,
    query_bias: Tensor | None, key_bias: Tensor | None, value_bias: Tensor | None,
    normalization: str,
    layer_norm_weight: Tensor, layer_norm_bias: Tensor | None,
    qkv_format: Literal["sbhd", "bshd"],
    num_attention_heads: int,
    num_gqa_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    q, k, v = apply_qkv_norm(q, k, v, normalization, layer_norm_weight, layer_norm_bias, eps)

    q = F.linear(q, query_weight, query_bias)
    k = F.linear(k, key_weight, key_bias)
    v = F.linear(v, value_weight, value_bias)

    if qkv_format == "sbhd":
        q = rearrange(q, "s b (h d) -> b h s d", h=num_attention_heads)
        k = rearrange(k, "s b (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
        v = rearrange(v, "s b (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
    elif qkv_format == "bshd":
        q = rearrange(q, "b s (h d) -> b h s d", h=num_attention_heads)
        k = rearrange(k, "b s (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
        v = rearrange(v, "b s (g h d) -> b h (g s) d", d=q.shape[-1], g=num_gqa_groups)
    else:
        raise ValueError(f"Invalid qkv_format: {qkv_format}")

    return q, k, v


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def forward_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor | None,
    qkv_format: Literal["sbhd", "bshd"],
    dropout: float,
    training: bool,
) -> Tensor:
    dropout = 0.0 if not training else dropout
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False)

    if qkv_format == "sbhd":
        o = rearrange(o, "b h s d -> s b (h d)")
    elif qkv_format == "bshd":
        o = rearrange(o, "b h s d -> b s (h d)")

    o = F.linear(o, proj_weight, proj_bias)
    return o


def forward_attention_matrix(x: Tensor): ...


class MultiheadAttention(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: int | None = None,
        attention_dropout: float = 0.1,
        layer_number: int | None = None,
        num_gqa_groups: int | None = None,
        attention_type: Literal["self", "cross"] = "self",
        normalization: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
        bias: bool = True,
        qkv_format: Literal["sbhd", "bshd"] = "sbhd",
        input_layernorm: bool = False,
        fuse_qkv_params: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        kv_channels = kv_channels if kv_channels else (hidden_size // num_attention_heads)
        num_gqa_groups = num_gqa_groups or num_attention_heads
        self.attention_dropout = attention_dropout
        self.layer_number = layer_number
        self.attention_type = attention_type
        self.qkv_format = qkv_format
        self.num_attention_heads = num_attention_heads
        self.fuse_qkv_params = fuse_qkv_params
        self._track_attention_weights = False
        self.attention_weights = None

        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"
        self.hidden_size_per_attention_head = kv_channels
        self.hidden_size_q = self.hidden_size_per_attention_head * num_attention_heads
        self.hidden_size_kv = self.hidden_size_per_attention_head * self.num_gqa_groups

        parameters_split = {
            "query": self.hidden_size_q,
            "key": self.hidden_size_kv,
            "value": self.hidden_size_kv,
        }

        match (input_layernorm, attention_type):
            case (False, "self"):
                self.qkv = Linear(
                    hidden_size,
                    self.hidden_size_q + 2 * self.hidden_size_kv,
                    bias=bias,
                    parameters_split=parameters_split,
                )
            case (True, "self"):
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    self.hidden_size_q + 2 * self.hidden_size_kv,
                    bias=bias,
                    parameters_split=parameters_split,
                    normalization=normalization,
                    eps=eps,
                )
            case (False, "cross"):
                self.query_layer = Linear(
                    hidden_size,
                    self.hidden_size_q,
                    bias=bias,
                )
                self.key_value = Linear(
                    hidden_size,
                    2 * self.hidden_size_kv,
                    bias=bias,
                    parameters_split=("key", "value"),
                )
            case (True, "cross"):
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    self.hidden_size_q,
                    parameters_split=("query",) if not fuse_qkv_params else None,
                    bias=bias,
                )
                self.key_value = Linear(
                    hidden_size,
                    2 * self.hidden_size_kv,
                    bias=bias,
                    parameters_split=("key", "value"),
                )
            case _:
                raise ValueError(f"Invalid input_layernorm: {input_layernorm} and attention_type: {attention_type}")

        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, "layernorm_qkv"):
            self.layernorm_qkv.reset_parameters()
        if hasattr(self, "layernorm_query"):
            self.layernorm_query.reset_parameters()
        if hasattr(self, "key_value"):
            self.key_value.reset_parameters()
        if hasattr(self, "query_layer"):
            self.query_layer.reset_parameters()
        self.proj.reset_parameters()

    def track_attention_weights(self, track: bool = True) -> None:
        self._track_attention_weights = track

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        checkpoint_core_attention: bool = False,
        attn_mask_type: Literal["arbitrary", "causal", "no_mask"] | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        assert attn_mask_type in (None, "causal", "no_mask", "arbitrary")

        # Packed
        if hasattr(self, "qkv") and self.qkv is not None:
            q, k, v = self.qkv(x).split([self.hidden_size_q, self.hidden_size_kv, self.hidden_size_kv], dim=-1)
        elif hasattr(self, "layernorm_qkv") and self.layernorm_qkv is not None:
            q, k, v = self.layernorm_qkv(x).split(
                [self.hidden_size_q, self.hidden_size_kv, self.hidden_size_kv], dim=-1
            )
        elif hasattr(self, "query_layer") and self.query_layer is not None:
            q = self.query_layer(x)
            k, v = self.key_value(encoder_output).split([self.hidden_size_kv, self.hidden_size_kv], dim=-1)
        elif hasattr(self, "layernorm_query") and self.layernorm_query is not None:
            q = self.layernorm_query(x)
            k, v = self.key_value(encoder_output).split([self.hidden_size_kv, self.hidden_size_kv], dim=-1)
        else:
            raise AssertionError("Invalid configuration")

        if self.qkv_format == "sbhd":
            q = rearrange(q, "s b (h d) -> b h s d", h=self.num_attention_heads)
            k = rearrange(k, "s b (h d) -> b h s d", d=q.shape[-1])
            v = rearrange(v, "s b (h d) -> b h s d", d=q.shape[-1])
        elif self.qkv_format == "bshd":
            q = rearrange(q, "b s (h d) -> b h s d", h=self.num_attention_heads)
            k = rearrange(k, "b s (h d) -> b h s d", d=q.shape[-1])
            v = rearrange(v, "b s (h d) -> b h s d", d=q.shape[-1])
        else:
            raise ValueError(f"Invalid qkv_format: {self.qkv_format}")

        k = repeat(k, "b h s d -> b (h g) s d", g=q.shape[1] // k.shape[1])
        v = repeat(v, "b h s d -> b (h g) s d", g=q.shape[1] // v.shape[1])

        dropout = 0.0 if not self.training else self.attention_dropout
        self._q = q
        self._k = k
        self._v = v

        # Optional attention weight tracking
        if self._track_attention_weights:
            with torch.no_grad():
                scale = math.sqrt(self.hidden_size_per_attention_head)
                weights = (q @ k.mT).div(scale).softmax(dim=-1)
                self.attention_weights = weights

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=attn_mask_type == "causal"
        )

        if self.qkv_format == "sbhd":
            o = rearrange(o, "b h s d -> s b (h d)")
        elif self.qkv_format == "bshd":
            o = rearrange(o, "b h s d -> b s (h d)")

        o = F.linear(o, self.proj.weight, self.proj.bias)
        return o
