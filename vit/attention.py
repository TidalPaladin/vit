import math
from typing import Final, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


# @torch.compile(fullgraph=True)
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


# @torch.compile(fullgraph=True)
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


# @torch.compile(fullgraph=True, dynamic=False)
def separable_polar_approx(
    # fmt: off
    r: Tensor, theta: Tensor,
    b: Tensor, c: Tensor,
    # fmt: on
) -> Tensor:
    assert r.ndim == 2, f"r.ndim: {r.ndim}"
    assert theta.shape == r.shape, f"theta.shape: {theta.shape}, r.shape: {r.shape}"
    assert b.ndim == 3, f"b.ndim: {b.ndim}"
    assert c.ndim == 3, f"c.ndim: {c.ndim}"
    K = b.shape[-2] - 1
    N = b.shape[-1] - 1
    H = b.shape[0]

    # Convert inputs to (B, H, L)
    r = r.unsqueeze(1).expand(-1, H, -1)
    theta = theta.unsqueeze(1).expand(-1, H, -1)

    # Compute polynomial terms: r^0, r^1, ..., r^K
    r_powers = torch.pow(r.unsqueeze(-1), torch.arange(K + 1, device=r.device, dtype=r.dtype))

    # Compute Fourier terms: cos(0*theta), cos(theta), ..., cos(N*theta)
    cos_terms = torch.cos(theta.unsqueeze(-1) * torch.arange(N + 1, device=theta.device, dtype=theta.dtype))
    sin_terms = torch.sin(theta.unsqueeze(-1) * torch.arange(N + 1, device=theta.device, dtype=theta.dtype))

    # Compute approximation using real Fourier basis
    cos_part = torch.einsum("hkn,bhlk,bhln->bhl", b, r_powers, cos_terms)
    sin_part = torch.einsum("hkn,bhlk,bhln->bhl", c, r_powers, sin_terms)

    return cos_part + sin_part


# @torch.compile(fullgraph=True, dynamic=False)
def compute_bias_grid(
    # fmt: off
    q: Tensor, k: Tensor,
    b: Tensor, c: Tensor,
    # fmt: on
) -> Tensor:
    B, Lq, C = q.shape
    _, Lk, _ = k.shape
    H = b.shape[0]
    q = q.view(B, Lq, 1, C)
    k = k.view(B, 1, Lk, C)
    dist = q - k
    theta = torch.atan2(dist[..., 1], dist[..., 0])
    r = torch.norm(dist, dim=-1)
    assert r.shape == (B, Lq, Lk), f"r.shape: {r.shape}, (B, Lq, Lk): {(B, Lq, Lk)}"
    assert theta.shape == (B, Lq, Lk), f"theta.shape: {theta.shape}, (B, Lq, Lk): {(B, Lq, Lk)}"
    bias = separable_polar_approx(r.view(B, -1), theta.view(B, -1), b, c).view(B, H, Lq, Lk)
    return bias


def num_extra_tokens(q: Tensor, posq: Tensor) -> int:
    if q.shape[1] < posq.shape[1]:
        raise ValueError(f"q tokens must be >= posq tokens, q.shape: {q.shape}, posq.shape: {posq.shape}")
    return q.shape[1] - posq.shape[1]


# @torch.compile(fullgraph=True)
def expand_bias_grid_for_extra_tokens(bias: Tensor, extra_tokens: int) -> Tensor:
    B, H, Lq, Lk = bias.shape
    result = bias.new_zeros(B, H, Lq + extra_tokens, Lk + extra_tokens)
    result[:, :, :Lq, :Lk] = bias
    return result


class PolarApprox(nn.Module):

    def __init__(self, radial_degree: int = 2, angular_degree: int = 4, nhead: int = 1):
        super(PolarApprox, self).__init__()
        self.b = nn.Parameter(torch.empty(nhead, radial_degree + 1, angular_degree + 1))
        self.c = nn.Parameter(torch.empty(nhead, radial_degree + 1, angular_degree + 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.b, std=0.01, a=-0.5, b=0.5)
        nn.init.trunc_normal_(self.c, std=0.01, a=-0.5, b=0.5)
        self.b[..., 1, 0].data.fill_(1.0)

    def forward(self, r: Tensor, theta: Tensor) -> Tensor:
        return separable_polar_approx(r, theta, self.b, self.c)

    @torch.no_grad()
    def plot(
        self,
        r_min: float = 0,
        r_max: float = 10,
        title=None,
        filename="",
        vmax=None,
        vmin=None,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required to plot the function")

        # Create grid
        r = torch.linspace(r_min, r_max, 100, device=self.b.device, dtype=self.b.dtype)
        theta = torch.linspace(0, 2 * math.pi, 360, device=self.b.device, dtype=self.b.dtype)
        r, theta = torch.meshgrid(r, theta, indexing="ij")

        # Compute the function values
        z = self.forward(r.reshape(1, -1), theta.reshape(1, -1)).view_as(r).contiguous().cpu().numpy()
        r = r.cpu().numpy()
        theta = theta.cpu().numpy()

        # 2D Polar Heatmap
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        cax = ax.pcolormesh(theta, r, z, cmap="viridis", vmax=vmax, vmin=vmin)
        if title is not None:
            ax.set_title(f"{title}")
        fig.colorbar(cax, ax=ax, label="f(r, θ)")
        plt.tight_layout()
        plt.savefig(filename)


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
        attn_bias: bool = False,
        radial_degree: int = 2,
        angular_degree: int = 4,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attn_bias = PolarApprox(radial_degree, angular_degree, num_attention_heads) if attn_bias else None
        self._head_dim = hidden_size // num_attention_heads
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.qkv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.qkv_proj.weight, std=0.02)
        if self.attn_bias is not None:
            self.attn_bias.reset_parameters()

    def forward(self, x: Tensor, pos: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor:
        if self.attn_bias is not None:
            if attn_mask is not None:
                raise ValueError("attn_mask is not supported when using attention biases")
            if pos is None:
                raise ValueError("pos is required when using attention biases")
            posq = posk = pos
            attn_mask = compute_bias_grid(posq, posk, self.attn_bias.b, self.attn_bias.c)
            if num_extra_tokens(x, pos) > 0:
                attn_mask = expand_bias_grid_for_extra_tokens(attn_mask, num_extra_tokens(x, pos))

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
        attn_bias: bool = False,
        radial_degree: int = 2,
        angular_degree: int = 4,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attn_bias = PolarApprox(radial_degree, angular_degree, num_attention_heads) if attn_bias else None
        self._head_dim = hidden_size // num_attention_heads
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.kv_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.kv_proj.weight, std=0.02)
        if self.attn_bias is not None:
            self.attn_bias.reset_parameters()

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        posq: Tensor | None = None,
        posk: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if self.attn_bias is not None:
            if attn_mask is not None:
                raise ValueError("attn_mask is not supported when using attention biases")
            if posq is None:
                raise ValueError("posq is required when using attention biases")
            if posk is None:
                raise ValueError("posk is required when using attention biases")
            attn_mask = compute_bias_grid(posq, posk, self.attn_bias.b, self.attn_bias.c)

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
