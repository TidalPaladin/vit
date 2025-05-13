from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    w_norm: Tensor,
    head_dim: int,
    eps: float,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    x = F.rms_norm(x, x.shape[-1:], w_norm, eps=eps)
    q, k, v = F.linear(x, w_in, b_in).chunk(3, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    v = _unfold_head_and_permute(v, head_dim)
    return q, k, v


@torch.compile(fullgraph=True)
def project_q_kv_packed(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_q: Tensor, b_q: Tensor | None,
    w_kv: Tensor, b_kv: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    eps: float,
    # fmt: on
) -> Tuple[Tensor, Tensor, Tensor]:
    q = F.rms_norm(q, q.shape[-1:], w_norm, eps=eps)
    q = F.linear(q, w_q, b_q)
    k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    v = _unfold_head_and_permute(v, head_dim)
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
    # fmt: on
) -> Tensor:
    q, k, v = project_qkv_packed(x, w_in, b_in, w_norm, head_dim, eps)
    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
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
    # fmt: on
) -> Tensor:
    q, k, v = project_q_kv_packed(q, kv, w_q, b_q, w_kv, b_kv, w_norm, head_dim, eps)
    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=training, inplace=True)
    return o


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
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.qkv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.qkv_proj.weight, std=0.02)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
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
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.kv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.kv_proj.weight, std=0.02)

    def forward(self, q: Tensor, kv: Tensor, attn_mask: Tensor | None = None) -> Tensor:
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
            # fmt: on
        )
