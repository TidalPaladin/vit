from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope import apply_rope


# torch.compile has difficulty with einops.rearrange, so we use our own implementation
def _unfold_head_and_permute(x: Tensor, head_dim: int) -> Tensor:
    B, S, _ = x.shape
    return x.view(B, S, -1, head_dim).transpose(1, 2)


def _permute_and_fold_head(x: Tensor) -> Tensor:
    B, H, S, D = x.shape
    return x.transpose(1, 2).reshape(B, S, H * D)


@torch.compile(fullgraph=True)
def project_qkv_packed(
    # fmt: off
    x: Tensor,
    w_in: Tensor, b_in: Tensor | None,
    head_dim: int,
    eps: float,
    rope: Tensor | None = None,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    q, k, v = F.linear(x, w_in, b_in).chunk(3, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    if rope is not None:
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
    v = _unfold_head_and_permute(v, head_dim)
    return q, k, v


@torch.compile(fullgraph=True)
def project_q_kv_packed(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_q: Tensor, b_q: Tensor | None,
    w_kv: Tensor, b_kv: Tensor | None,
    head_dim: int,
    eps: float,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    q = F.linear(q, w_q, b_q)
    k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    v = _unfold_head_and_permute(v, head_dim)
    if rope_q is not None:
        q = apply_rope(q, rope_q)
    if rope_k is not None:
        k = apply_rope(k, rope_k)
    return q, k, v


@torch.compile(fullgraph=True)
def project_q_kv_packed_static_query(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_kv: Tensor, b_kv: Tensor | None,
    head_dim: int,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    v = _unfold_head_and_permute(v, head_dim)
    if rope_q is not None:
        q = apply_rope(q, rope_q)
    if rope_k is not None:
        k = apply_rope(k, rope_k)
    return q, k, v


@torch.compile(fullgraph=True)
def attention_qkv_packed(
    # fmt: off
    x: Tensor,
    w_in: Tensor, b_in: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    w_out: Tensor, b_out: Tensor | None,
    attn_mask: Tensor | None,
    eps: float,
    attention_dropout: float,
    dropout: float,
    training: bool,
    rope: Tensor | None = None,
    w_gate: Tensor | None = None, b_gate: Tensor | None = None,
    # fmt: on
) -> Tensor:
    x = F.rms_norm(x, x.shape[-1:], w_norm, eps=eps)
    q, k, v = project_qkv_packed(x, w_in, b_in, head_dim, eps, rope)
    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
    if w_gate is not None:
        o = o * F.linear(x, w_gate, b_gate).sigmoid()
    o = F.dropout(o, p=dropout, training=training, inplace=True)
    return o


@torch.compile(fullgraph=True)
def attention_q_kv_packed(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_q: Tensor, b_q: Tensor | None,
    w_kv: Tensor, b_kv: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    w_out: Tensor, b_out: Tensor | None,
    attn_mask: Tensor | None,
    eps: float,
    attention_dropout: float,
    dropout: float,
    training: bool,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    w_gate: Tensor | None = None, b_gate: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q_norm = F.rms_norm(q, q.shape[-1:], w_norm, eps=eps)
    q, k, v = project_q_kv_packed(q_norm, kv, w_q, b_q, w_kv, b_kv, head_dim, eps, rope_q, rope_k)
    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
    if w_gate is not None:
        o = o * F.linear(q_norm, w_gate, b_gate).sigmoid()
    o = F.dropout(o, p=dropout, training=training, inplace=True)
    return o


@torch.compile(fullgraph=True)
def attention_q_kv_packed_static_query(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_kv: Tensor, b_kv: Tensor | None,
    head_dim: int,
    w_out: Tensor, b_out: Tensor | None,
    attn_mask: Tensor | None,
    attention_dropout: float,
    dropout: float,
    training: bool,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    w_gate: Tensor | None = None, b_gate: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q_original = q
    q, k, v = project_q_kv_packed_static_query(q, kv, w_kv, b_kv, head_dim, rope_q, rope_k)
    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
    if w_gate is not None:
        o = o * F.linear(q_original, w_gate, b_gate).sigmoid()
    o = F.dropout(o, p=dropout, training=training, inplace=True)
    return o


@torch.no_grad()
@torch.compile(fullgraph=True)
def attention_weights_qkv_packed(
    # fmt: off
    x: Tensor,
    w_in: Tensor, b_in: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    eps: float,
    rope: Tensor | None = None,
    # fmt: on
) -> Tensor:
    x = F.rms_norm(x, x.shape[-1:], w_norm, eps=eps)
    q, k, _ = project_qkv_packed(x, w_in, b_in, head_dim, eps, rope)
    return (q @ k.mT).softmax(dim=-1)


@torch.no_grad()
@torch.compile(fullgraph=True)
def attention_weights_q_kv_packed(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_q: Tensor, b_q: Tensor | None,
    w_kv: Tensor, b_kv: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    eps: float,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q_norm = F.rms_norm(q, q.shape[-1:], w_norm, eps=eps)
    q, k, _ = project_q_kv_packed(q_norm, kv, w_q, b_q, w_kv, b_kv, head_dim, eps, rope_q, rope_k)
    return (q @ k.mT).softmax(dim=-1)


@torch.no_grad()
@torch.compile(fullgraph=True)
def attention_weights_q_kv_packed_static_query(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_kv: Tensor, b_kv: Tensor | None,
    head_dim: int,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q, k, _ = project_q_kv_packed_static_query(q, kv, w_kv, b_kv, head_dim, rope_q, rope_k)
    return (q @ k.mT).softmax(dim=-1)


class SelfAttention(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True,
        eps: float = 1e-5,
        gated: bool = False,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads
        self.gate = nn.Linear(hidden_size, hidden_size) if gated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.qkv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.qkv_proj.weight, std=0.02)
        if self.gate is not None:
            nn.init.trunc_normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, rope: Tensor | None = None) -> Tensor:
        return attention_qkv_packed(
            # fmt: off
            x,
            self.qkv_proj.weight, self.qkv_proj.bias,
            self.norm.weight,
            self._head_dim,
            self.out_proj.weight, self.out_proj.bias,
            attn_mask,
            self.norm.eps or 1e-5,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            rope,
            w_gate=self.gate.weight if self.gate is not None else None,
            b_gate=self.gate.bias if self.gate is not None else None,
            # fmt: on
        )

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, attn_mask: Tensor | None = None, rope: Tensor | None = None) -> Tensor:
            return self.forward(x, attn_mask, rope)

    def forward_weights(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        return attention_weights_qkv_packed(
            # fmt: off
            x,
            self.qkv_proj.weight, self.qkv_proj.bias,
            self.norm.weight,
            self._head_dim,
            self.norm.eps or 1e-5,
            rope,
            # fmt: on
        )


class CrossAttention(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True,
        eps: float = 1e-5,
        gated: bool = False,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads
        self.gate = nn.Linear(hidden_size, hidden_size) if gated else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.kv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.kv_proj.weight, std=0.02)
        if self.gate is not None:
            nn.init.trunc_normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        rope_q: Tensor | None = None,
        rope_k: Tensor | None = None,
    ) -> Tensor:
        return attention_q_kv_packed(
            # fmt: off
            q, kv,
            self.q_proj.weight, self.q_proj.bias,
            self.kv_proj.weight, self.kv_proj.bias,
            self.norm.weight,
            self._head_dim,
            self.out_proj.weight, self.out_proj.bias,
            attn_mask,
            self.norm.eps or 1e-5,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            rope_q,
            rope_k,
            w_gate=self.gate.weight if self.gate is not None else None,
            b_gate=self.gate.bias if self.gate is not None else None,
            # fmt: on
        )

    if TYPE_CHECKING:

        def __call__(
            self,
            q: Tensor,
            kv: Tensor,
            attn_mask: Tensor | None = None,
            rope_q: Tensor | None = None,
            rope_k: Tensor | None = None,
        ) -> Tensor:
            return self.forward(q, kv, attn_mask, rope_q, rope_k)

    def forward_weights(
        self, q: Tensor, kv: Tensor, rope_q: Tensor | None = None, rope_k: Tensor | None = None
    ) -> Tensor:
        return attention_weights_q_kv_packed(
            # fmt: off
            q, kv,
            self.q_proj.weight, self.q_proj.bias,
            self.kv_proj.weight, self.kv_proj.bias,
            self.norm.weight,
            self._head_dim,
            self.norm.eps or 1e-5,
            rope_q,
            rope_k,
            # fmt: on
        )


class AttentivePool(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_queries: int = 1,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        super().__init__()
        self._head_dim = hidden_size // num_attention_heads
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.query = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.gate = nn.Linear(hidden_size, hidden_size) if gated else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, std=1.0)
        self.kv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        nn.init.trunc_normal_(self.kv_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        if self.gate is not None:
            nn.init.trunc_normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        y = attention_q_kv_packed_static_query(
            # fmt: off
            self.query, x,
            self.kv_proj.weight, self.kv_proj.bias,
            self._head_dim,
            self.out_proj.weight, self.out_proj.bias,
            None,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            rope_k=rope,
            w_gate=self.gate.weight if self.gate is not None else None,
            b_gate=self.gate.bias if self.gate is not None else None,
            # fmt: on
        )
        if y.shape[1] == 1:
            y = y.squeeze(1)
        return y

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
            return self.forward(x, rope)

    def forward_weights(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        w = attention_weights_q_kv_packed_static_query(
            # fmt: off
            self.query, x,
            self.kv_proj.weight, self.kv_proj.bias,
            self._head_dim,
            rope_k=rope,
            # fmt: on
        )
        w = w.movedim(1, -1)
        if w.shape[1] == 1:
            w = w.squeeze(1)
        return w
