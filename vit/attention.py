import math
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


@torch.compile(fullgraph=True, dynamic=False)
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


@torch.compile(fullgraph=True, dynamic=False)
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


@torch.compile(fullgraph=True, dynamic=False)
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


@torch.compile(fullgraph=True, dynamic=False)
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


@torch.compile(fullgraph=True, dynamic=False)
def attention_qkv_packed_fourier(
    # fmt: off
    x: Tensor,
    w_in: Tensor, b_in: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    w_out: Tensor, b_out: Tensor | None,
    w_fourier: Tensor, b_fourier: Tensor | None,
    w_f_q: Tensor, b_f_q: Tensor | None, 
    w_f_k: Tensor, b_f_k: Tensor | None, 
    pos: Tensor,
    attn_mask: Tensor | None,
    eps: float,
    attention_dropout: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    q, k, v = project_qkv_packed(x, w_in, b_in, w_norm, head_dim, eps)

    # Add Fourier features
    fourier_features = F.linear(pos, w_fourier, b_fourier)
    fourier_features = torch.cat([fourier_features.sin(), fourier_features.cos()], dim=-1)
    fourier_features = fourier_features / math.sqrt(fourier_features.shape[-1])
    q = q + _unfold_head_and_permute(F.linear(fourier_features, w_f_q, b_f_q), head_dim)
    k = k + _unfold_head_and_permute(F.linear(fourier_features, w_f_k, b_f_k), head_dim)

    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=training, inplace=True)
    return o


@torch.compile(fullgraph=True, dynamic=False)
def attention_q_kv_packed_fourier(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_q: Tensor, b_q: Tensor | None,
    w_kv: Tensor, b_kv: Tensor | None,
    w_norm: Tensor,
    head_dim: int,
    w_out: Tensor, b_out: Tensor | None,
    w_fourier: Tensor, b_fourier: Tensor | None,
    w_f_q: Tensor, b_f_q: Tensor | None, 
    w_f_k: Tensor, b_f_k: Tensor | None,
    pos_q: Tensor, pos_k: Tensor,
    attn_mask: Tensor | None,
    eps: float,
    attention_dropout: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    q, k, v = project_q_kv_packed(q, kv, w_q, b_q, w_kv, b_kv, w_norm, head_dim, eps)

    # Add Fourier features
    fourier_features_q = F.linear(pos_q, w_fourier, b_fourier)
    fourier_features_k = F.linear(pos_k, w_fourier, b_fourier)
    fourier_features_q = torch.cat([fourier_features_q.sin(), fourier_features_q.cos()], dim=-1)
    fourier_features_k = torch.cat([fourier_features_k.sin(), fourier_features_k.cos()], dim=-1)
    fourier_features_q = fourier_features_q / math.sqrt(fourier_features_q.shape[-1])
    fourier_features_k = fourier_features_k / math.sqrt(fourier_features_k.shape[-1])
    q = q + _unfold_head_and_permute(F.linear(fourier_features_q, w_f_q, b_f_q), head_dim)
    k = k + _unfold_head_and_permute(F.linear(fourier_features_k, w_f_k, b_f_k), head_dim)

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
        use_fourier_features: bool = False,
        spatial_dims: int = 2,
        fourier_size: int = 384,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads

        self.gamma = gamma
        if use_fourier_features:
            self.fourier = nn.Linear(spatial_dims, fourier_size, bias=False)
            self.q_fourier = nn.Linear(2 * fourier_size, hidden_size)
            self.k_fourier = nn.Linear(2 * fourier_size, hidden_size)
        else:
            self.fourier = self.q_fourier = self.k_fourier = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.qkv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.qkv_proj.weight, std=0.02)
        if self.fourier is not None:
            nn.init.normal_(self.fourier.weight, std=self.gamma**-2.0)
            nn.init.trunc_normal_(self.q_fourier.weight, std=0.02)
            nn.init.trunc_normal_(self.k_fourier.weight, std=0.02)

    def forward(self, x: Tensor, pos: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor:
        if self.fourier is not None:
            if pos is None:
                raise ValueError("pos is required when using Fourier features")
            return attention_qkv_packed_fourier(
                # fmt: off
                x,
                self.qkv_proj.weight, self.qkv_proj.bias,
                self.norm.weight,
                self._head_dim,
                self.out_proj.weight, self.out_proj.bias,
                self.fourier.weight, self.fourier.bias,
                self.q_fourier.weight, self.q_fourier.bias,
                self.k_fourier.weight, self.k_fourier.bias,
                pos,
                attn_mask,
                self.norm.eps or 1e-5,
                self.attention_dropout.p,
                self.dropout.p,
                self.training,
                # fmt: on
            )

        else:
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
        use_fourier_features: bool = False,
        spatial_dims: int = 2,
        fourier_size: int = 384,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads

        self.gamma = gamma
        if use_fourier_features:
            self.fourier = nn.Linear(spatial_dims, fourier_size, bias=False)
            self.q_fourier = nn.Linear(2 * fourier_size, hidden_size)
            self.k_fourier = nn.Linear(2 * fourier_size, hidden_size)
        else:
            self.fourier = self.q_fourier = self.k_fourier = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.kv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.kv_proj.weight, std=0.02)
        if self.fourier is not None:
            nn.init.normal_(self.fourier.weight, std=self.gamma**-2.0)
            nn.init.trunc_normal_(self.q_fourier.weight, std=0.02)
            nn.init.trunc_normal_(self.k_fourier.weight, std=0.02)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        pos_q: Tensor | None = None,
        pos_k: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if self.fourier is not None:
            if pos_q is None or pos_k is None:
                raise ValueError("pos_q and pos_k are required when using Fourier features")
            return attention_q_kv_packed_fourier(
                # fmt: off
                q, kv,
                self.q_proj.weight, self.q_proj.bias,
                self.kv_proj.weight, self.kv_proj.bias,
                self.norm.weight,
                self._head_dim,
                self.out_proj.weight, self.out_proj.bias,
                self.fourier.weight, self.fourier.bias,
                self.q_fourier.weight, self.q_fourier.bias,
                self.k_fourier.weight, self.k_fourier.bias,
                pos_q, pos_k,
                attn_mask,
                self.norm.eps or 1e-5,
                self.attention_dropout.p,
                self.dropout.p,
                self.training,
                # fmt: on
            )
        else:
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
