from typing import Callable, Final, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention


DEFORMABLE_ALIBI_PARAMS: Final = 6
O_IDX: Final = 0
C_IDX: Final = 1
W_IDX: Final = 2
P_IDX: Final = 3
DELTA_IDX: Final = 4
M_IDX: Final = 5

O_INIT: Final = 0.0
C_INIT: Final = 1.0
W_INIT: Final = 1.0
P_INIT: Final = 0.0
DELTA_INIT: Final = 0.0
M_INIT: Final = 1.0


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


@torch.compile(fullgraph=True)
def attentive_pool(
    # fmt: off
    x: Tensor,
    w: Tensor, b: Tensor | None,
    w_v: Tensor, b_v: Tensor | None,
    head_dim: int,
    # fmt: on
) -> Tensor:
    B, S, D = x.shape
    weights = F.linear(x, w, b)  # B, S, H
    weights = F.softmax(weights, dim=-2)
    weights = weights.unsqueeze(-1)  # B, S, H, 1
    v = F.linear(x, w_v, b_v).view(B, S, -1, head_dim)  # B, S, D
    v = (v * weights).sum(dim=1)
    return v.view(B, D)


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


class AttentivePool(nn.Module):
    attention_weights: Tensor | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        self._head_dim = hidden_size // num_attention_heads
        self.weight = nn.Linear(hidden_size, num_attention_heads, bias=bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.reset_parameters()
        self.value.reset_parameters()
        nn.init.trunc_normal_(self.weight.weight, std=0.02)
        nn.init.trunc_normal_(self.value.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return attentive_pool(
            # fmt: off
            x,
            self.weight.weight, self.weight.bias,
            self.value.weight, self.value.bias,
            self._head_dim,
            # fmt: on
        )


@torch.compile(fullgraph=True)
def deformable_alibi(
    # fmt: off
    r: Tensor, theta: Tensor,
    o: Tensor, c: Tensor, 
    w: Tensor, p: Tensor,
    delta: Tensor, m: Tensor
    # fmt: on
) -> Tensor:
    term1 = (r - o).relu() ** c.abs()
    term2 = 1 - delta.tanh().abs() * torch.cos((theta - p) * w / 2).pow(2).pow(m.abs())
    return term1 * term2


@torch.compile(fullgraph=True, dynamic=False)
def compute_bias_grid(
    # fmt: off
    q: Tensor, k: Tensor,
    o: Tensor, c: Tensor, 
    w: Tensor, p: Tensor,
    delta: Tensor, m: Tensor
    # fmt: on
) -> Tensor:
    B, Lq, C = q.shape
    _, Lk, _ = k.shape
    q = q.view(B, Lq, 1, C)
    k = k.view(B, 1, Lk, C)
    dist = q - k
    theta = torch.atan2(dist[..., 1], dist[..., 0])
    r = torch.norm(dist, dim=-1)

    H = o.shape[0]
    theta = theta.view(B, 1, Lq, Lk)
    r = r.view(B, 1, Lq, Lk)
    o = o.view(1, H, 1, 1)
    c = c.view(1, H, 1, 1)
    w = w.view(1, H, 1, 1)
    p = p.view(1, H, 1, 1)
    delta = delta.view(1, H, 1, 1)
    m = m.view(1, H, 1, 1)
    return deformable_alibi(r, theta, o, c, w, p, delta, m)


def init_bias_params(params: Tensor) -> Tensor:
    nn.init.zeros_(params[:, O_IDX])
    nn.init.trunc_normal_(params[:, C_IDX], mean=1, std=0.1, a=0.5, b=1.5)
    nn.init.constant_(params[:, W_IDX], 1.0)
    nn.init.constant_(params[:, P_IDX], 0.0)
    nn.init.trunc_normal_(params[:, DELTA_IDX], std=1.0)
    nn.init.constant_(params[:, M_IDX], 1.0)
    

class SelfAttentionWithBiases(SelfAttention):
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
        self.bias_params = None
        super().__init__(hidden_size, num_attention_heads, hidden_dropout, attention_dropout, bias, eps)
        self.bias_params = nn.Parameter(torch.zeros(num_attention_heads, DEFORMABLE_ALIBI_PARAMS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.bias_params is not None:
            init_bias_params(self.bias_params)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        posq = posk = pos
        o = self.bias_params[:, O_IDX]
        c = self.bias_params[:, C_IDX]
        w = self.bias_params[:, W_IDX]
        p = self.bias_params[:, P_IDX]
        delta = self.bias_params[:, DELTA_IDX]
        m = self.bias_params[:, M_IDX]
        bias = compute_bias_grid(posq, posk, o, c, w, p, delta, m)
        return attention_qkv_packed(
            # fmt: off
            x,
            self.qkv_proj.weight, self.qkv_proj.bias,
            self.norm.weight,
            self._head_dim,
            self.out_proj.weight, self.out_proj.bias,
            bias.float(),
            self.norm.eps or 1e-5,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            # fmt: on
        )


class CrossAttentionWithBiases(CrossAttention):
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
        self.bias_params = None
        super().__init__(hidden_size, num_attention_heads, hidden_dropout, attention_dropout, bias, eps)
        self.bias_params = nn.Parameter(torch.zeros(num_attention_heads, DEFORMABLE_ALIBI_PARAMS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.bias_params is not None:
            init_bias_params(self.bias_params)

    def forward(self, q: Tensor, kv: Tensor, posq: Tensor, posk: Tensor) -> Tensor:
        o = self.bias_params[:, O_IDX]
        c = self.bias_params[:, C_IDX]
        w = self.bias_params[:, W_IDX]
        p = self.bias_params[:, P_IDX]
        delta = self.bias_params[:, DELTA_IDX]
        m = self.bias_params[:, M_IDX]
        bias = compute_bias_grid(posq, posk, o, c, w, p, delta, m)
        return attention_q_kv_packed(
            # fmt: off
            q, kv,
            self.q_proj.weight, self.q_proj.bias,
            self.kv_proj.weight, self.kv_proj.bias,
            self.norm.weight,
            self._head_dim,
            self.out_proj.weight, self.out_proj.bias,
            bias.float(),
            self.norm.eps or 1e-5,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            # fmt: on
        )