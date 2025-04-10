from typing import Literal, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .helpers import DEFAULT_TRUNC_STD, compile_is_disabled


def apply_qkv_norm(
    q: Tensor, k: Tensor, v: Tensor, normalization: str, layer_norm_weight: Tensor, layer_norm_bias: Tensor | None
) -> Tuple[Tensor, Tensor, Tensor]:
    if q is k and k is v:
        if normalization == "LayerNorm":
            q = k = v = F.layer_norm(q, q.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias)
        elif normalization == "RMSNorm":
            q = k = v = F.rms_norm(q, q.shape[-1:], weight=layer_norm_weight)
    elif k is v:
        if normalization == "LayerNorm":
            q = F.layer_norm(q, q.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias)
            k = v = F.layer_norm(k, k.shape[-1:], weight=layer_norm_weight)
        elif normalization == "RMSNorm":
            q = F.rms_norm(q, q.shape[-1:], weight=layer_norm_weight)
            k = v = F.rms_norm(k, k.shape[-1:], weight=layer_norm_weight)
    else:
        if normalization == "LayerNorm":
            q = F.layer_norm(q, q.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias)
            k = F.layer_norm(k, k.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias)
            v = F.layer_norm(v, v.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias)
        elif normalization == "RMSNorm":
            q = F.rms_norm(q, q.shape[-1:], weight=layer_norm_weight)
            k = F.rms_norm(k, k.shape[-1:], weight=layer_norm_weight)
            v = F.rms_norm(v, v.shape[-1:], weight=layer_norm_weight)
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
) -> Tuple[Tensor, Tensor, Tensor]:
    q, k, v = apply_qkv_norm(q, k, v, normalization, layer_norm_weight, layer_norm_bias)

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


class InputProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        normalization: str,
        bias: bool,
        num_gqa_groups: int,
        qkv_format: Literal["sbhd", "bshd"],
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.normalization = normalization
        self.num_gqa_groups = num_gqa_groups
        self.qkv_format = qkv_format

        self.layer_norm_weight = nn.Parameter(torch.empty(hidden_size))
        self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size)) if normalization == "LayerNorm" else None

        q_dim = hidden_size
        kv_dim = hidden_size // num_attention_heads * num_gqa_groups

        self.query_weight = nn.Parameter(torch.empty(q_dim, hidden_size))
        self.key_weight = nn.Parameter(torch.empty(kv_dim, hidden_size))
        self.value_weight = nn.Parameter(torch.empty(kv_dim, hidden_size))
        self.query_bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.key_bias = nn.Parameter(torch.zeros(kv_dim)) if bias else None
        self.value_bias = nn.Parameter(torch.zeros(kv_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.query_weight, std=DEFAULT_TRUNC_STD)
        nn.init.trunc_normal_(self.key_weight, std=DEFAULT_TRUNC_STD)
        nn.init.trunc_normal_(self.value_weight, std=DEFAULT_TRUNC_STD)
        nn.init.ones_(self.layer_norm_weight)
        if self.query_bias is not None:
            nn.init.zeros_(self.query_bias)
        if self.key_bias is not None:
            nn.init.zeros_(self.key_bias)
        if self.value_bias is not None:
            nn.init.zeros_(self.value_bias)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)

    def forward(
        self, x: Tensor, encoder_output: Tensor | None = None, checkpoint_core_attention: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if encoder_output is not None:
            q = x
            k = v = encoder_output
        else:
            q = k = v = x

        if self.training and checkpoint_core_attention:
            y = checkpoint(
                forward_input_projection,
                q,
                k,
                v,
                self.query_weight,
                self.key_weight,
                self.value_weight,
                self.query_bias,
                self.key_bias,
                self.value_bias,
                self.normalization,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.qkv_format,
                self.num_attention_heads,
                self.num_gqa_groups,
                use_reentrant=False,
            )
        else:
            y = forward_input_projection(
                q,
                k,
                v,
                self.query_weight,
                self.key_weight,
                self.value_weight,
                self.query_bias,
                self.key_bias,
                self.value_bias,
                self.normalization,
                self.layer_norm_weight,
                self.layer_norm_bias,
                cast(Literal["sbhd", "bshd"], self.qkv_format),
                self.num_attention_heads,
                self.num_gqa_groups,
            )

        assert isinstance(y, tuple) and len(y) == 3
        assert all(isinstance(t, Tensor) for t in y)
        return cast(Tuple[Tensor, Tensor, Tensor], y)


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


class MultiheadAttention(nn.Module):
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
    ):
        super().__init__()
        assert kv_channels is None or kv_channels == hidden_size, "kv_channels must be None or equal to hidden_size"
        num_gqa_groups = num_gqa_groups or num_attention_heads
        self.attention_dropout = attention_dropout
        self.layer_number = layer_number
        self.attention_type = attention_type

        self.layernorm_qkv = InputProjection(
            hidden_size, num_attention_heads, normalization, bias, num_gqa_groups, qkv_format
        )
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm_qkv.reset_parameters()
        self.proj.reset_parameters()

    @property
    def qkv_format(self) -> Literal["sbhd", "bshd"]:
        return cast(Literal["sbhd", "bshd"], self.layernorm_qkv.qkv_format)

    @property
    def num_attention_heads(self) -> int:
        return self.layernorm_qkv.num_attention_heads

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        checkpoint_core_attention: bool = False,
    ) -> Tensor:
        q, k, v = self.layernorm_qkv(x, encoder_output, checkpoint_core_attention)

        if self.training and checkpoint_core_attention:
            o = checkpoint(
                forward_attention,
                q,
                k,
                v,
                self.proj.weight,
                self.proj.bias,
                self.qkv_format,
                self.attention_dropout,
                self.training,
                use_reentrant=False,
            )
        else:
            o = forward_attention(
                q, k, v, self.proj.weight, self.proj.bias, self.qkv_format, self.attention_dropout, self.training
            )

        assert isinstance(o, Tensor)
        return o
