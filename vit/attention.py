from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchao.quantization import quantize_

from .initialization import init_linear, trunc_normal_
from .norm import NormModule, NormType, apply_norm, get_norm_bias, is_layer_norm, make_norm
from .rope import apply_rope


# torch.compile has difficulty with einops.rearrange, so we use our own implementation
def _unfold_head_and_permute(x: Tensor, head_dim: int) -> Tensor:
    B, S, _ = x.shape
    return x.view(B, S, -1, head_dim).transpose(1, 2)


def _permute_and_fold_head(x: Tensor) -> Tensor:
    B, H, S, D = x.shape
    return x.transpose(1, 2).reshape(B, S, H * D)


def _apply_qk_norm(
    q: Tensor,
    k: Tensor,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
) -> tuple[Tensor, Tensor]:
    if not qk_normalization:
        return q, k
    assert q_norm_weight is not None and k_norm_weight is not None
    q = apply_norm(q, q_norm_weight, q_norm_bias, qk_eps, use_layer_norm=qk_use_layer_norm)
    k = apply_norm(k, k_norm_weight, k_norm_bias, qk_eps, use_layer_norm=qk_use_layer_norm)
    return q, k


def _make_qk_norms(
    head_dim: int,
    norm_type: NormType,
    eps: float,
    qk_normalization: bool,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[NormModule | None, NormModule | None]:
    if not qk_normalization:
        return None, None
    return (
        make_norm(head_dim, norm_type, eps=eps, device=device, dtype=dtype),
        make_norm(head_dim, norm_type, eps=eps, device=device, dtype=dtype),
    )


def _get_qk_norm_inputs(
    q_norm: NormModule | None,
    k_norm: NormModule | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, float]:
    if q_norm is None or k_norm is None:
        assert q_norm is None and k_norm is None
        return None, None, None, None, 1e-5
    return q_norm.weight, get_norm_bias(q_norm), k_norm.weight, get_norm_bias(k_norm), q_norm.eps or 1e-5


@torch.compile(fullgraph=True)
def project_qkv_packed(
    # fmt: off
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor | None,
    w_norm: Tensor,
    b_norm: Tensor | None,
    use_layer_norm: bool,
    head_dim: int,
    eps: float,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
    rope: Tensor | None = None,
    # fmt: on
) -> tuple[Tensor, Tensor, Tensor]:
    x = apply_norm(x, w_norm, b_norm, eps, use_layer_norm=use_layer_norm)
    q, k, v = F.linear(x, w_in, b_in).chunk(3, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    q, k = _apply_qk_norm(
        q,
        k,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        qk_use_layer_norm,
        qk_eps,
        qk_normalization,
    )
    if rope is not None:
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
    v = _unfold_head_and_permute(v, head_dim)
    return q, k, v


@torch.compile(fullgraph=True)
def project_q_kv_packed(
    # fmt: off
    q: Tensor,
    kv: Tensor,
    w_q: Tensor,
    b_q: Tensor | None,
    w_kv: Tensor,
    b_kv: Tensor | None,
    w_norm: Tensor,
    b_norm: Tensor | None,
    use_layer_norm: bool,
    head_dim: int,
    eps: float,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> tuple[Tensor, Tensor, Tensor]:
    q = apply_norm(q, w_norm, b_norm, eps, use_layer_norm=use_layer_norm)
    q = F.linear(q, w_q, b_q)
    k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    q = _unfold_head_and_permute(q, head_dim)
    k = _unfold_head_and_permute(k, head_dim)
    v = _unfold_head_and_permute(v, head_dim)
    q, k = _apply_qk_norm(
        q,
        k,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        qk_use_layer_norm,
        qk_eps,
        qk_normalization,
    )
    if rope_q is not None:
        q = apply_rope(q, rope_q)
    if rope_k is not None:
        k = apply_rope(k, rope_k)
    return q, k, v


@torch.compile(fullgraph=True)
def attention_qkv_packed(
    # fmt: off
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor | None,
    w_norm: Tensor,
    b_norm: Tensor | None,
    use_layer_norm: bool,
    head_dim: int,
    w_out: Tensor,
    b_out: Tensor | None,
    attn_mask: Tensor | None,
    eps: float,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
    attention_dropout: float,
    dropout: float,
    training: bool,
    rope: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q, k, v = project_qkv_packed(
        x,
        w_in,
        b_in,
        w_norm,
        b_norm,
        use_layer_norm,
        head_dim,
        eps,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        qk_use_layer_norm,
        qk_eps,
        qk_normalization,
        rope,
    )
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
    q: Tensor,
    kv: Tensor,
    w_q: Tensor,
    b_q: Tensor | None,
    w_kv: Tensor,
    b_kv: Tensor | None,
    w_norm: Tensor,
    b_norm: Tensor | None,
    use_layer_norm: bool,
    head_dim: int,
    w_out: Tensor,
    b_out: Tensor | None,
    attn_mask: Tensor | None,
    eps: float,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
    attention_dropout: float,
    dropout: float,
    training: bool,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q, k, v = project_q_kv_packed(
        q,
        kv,
        w_q,
        b_q,
        w_kv,
        b_kv,
        w_norm,
        b_norm,
        use_layer_norm,
        head_dim,
        eps,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        qk_use_layer_norm,
        qk_eps,
        qk_normalization,
        rope_q,
        rope_k,
    )
    attention_dropout = 0.0 if not training else attention_dropout
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention_dropout, is_causal=False, enable_gqa=True
    )
    o = _permute_and_fold_head(o)
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=training, inplace=True)
    return o


@torch.no_grad()
@torch.compile(fullgraph=True)
def attention_weights_qkv_packed(
    # fmt: off
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor | None,
    w_norm: Tensor,
    b_norm: Tensor | None,
    use_layer_norm: bool,
    head_dim: int,
    eps: float,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
    rope: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q, k, _ = project_qkv_packed(
        x,
        w_in,
        b_in,
        w_norm,
        b_norm,
        use_layer_norm,
        head_dim,
        eps,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        qk_use_layer_norm,
        qk_eps,
        qk_normalization,
        rope,
    )
    return (q @ k.mT * (head_dim**-0.5)).softmax(dim=-1)


@torch.no_grad()
@torch.compile(fullgraph=True)
def attention_weights_q_kv_packed(
    # fmt: off
    q: Tensor,
    kv: Tensor,
    w_q: Tensor,
    b_q: Tensor | None,
    w_kv: Tensor,
    b_kv: Tensor | None,
    w_norm: Tensor,
    b_norm: Tensor | None,
    use_layer_norm: bool,
    head_dim: int,
    eps: float,
    q_norm_weight: Tensor | None,
    q_norm_bias: Tensor | None,
    k_norm_weight: Tensor | None,
    k_norm_bias: Tensor | None,
    qk_use_layer_norm: bool,
    qk_eps: float,
    qk_normalization: bool,
    rope_q: Tensor | None = None,
    rope_k: Tensor | None = None,
    # fmt: on
) -> Tensor:
    q, k, _ = project_q_kv_packed(
        q,
        kv,
        w_q,
        b_q,
        w_kv,
        b_kv,
        w_norm,
        b_norm,
        use_layer_norm,
        head_dim,
        eps,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        qk_use_layer_norm,
        qk_eps,
        qk_normalization,
        rope_q,
        rope_k,
    )
    return (q @ k.mT * (head_dim**-0.5)).softmax(dim=-1)


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
        qkv_quantization_config: Any | None = None,
        out_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        qk_normalization: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm = make_norm(hidden_size, norm_type, eps=eps, **factory_kwargs)
        self._use_layer_norm = is_layer_norm(norm_type)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads
        self._qk_normalization = qk_normalization
        self.q_norm, self.k_norm = _make_qk_norms(self._head_dim, norm_type, eps, qk_normalization, **factory_kwargs)
        self.qkv_quantization_config = qkv_quantization_config
        self.out_quantization_config = out_quantization_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear(self.qkv_proj)
        init_linear(self.out_proj)
        self.apply_quantization(self.qkv_quantization_config, self.out_quantization_config)

    def apply_quantization(
        self, qkv_quantization_config: Any | None = None, out_quantization_config: Any | None = None
    ) -> None:
        """Apply quantization to the linear layers using torchao."""
        if qkv_quantization_config is not None:
            quantize_(self.qkv_proj, qkv_quantization_config)
        if out_quantization_config is not None:
            quantize_(self.out_proj, out_quantization_config)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, rope: Tensor | None = None) -> Tensor:
        q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, qk_eps = _get_qk_norm_inputs(self.q_norm, self.k_norm)
        return attention_qkv_packed(
            # fmt: off
            x,
            self.qkv_proj.weight,
            self.qkv_proj.bias,
            self.norm.weight,
            get_norm_bias(self.norm),
            self._use_layer_norm,
            self._head_dim,
            self.out_proj.weight,
            self.out_proj.bias,
            attn_mask,
            self.norm.eps or 1e-5,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            self._use_layer_norm,
            qk_eps,
            self._qk_normalization,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            rope,
            # fmt: on
        )

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, attn_mask: Tensor | None = None, rope: Tensor | None = None) -> Tensor:
            return self.forward(x, attn_mask, rope)

    def forward_weights(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, qk_eps = _get_qk_norm_inputs(self.q_norm, self.k_norm)
        return attention_weights_qkv_packed(
            # fmt: off
            x,
            self.qkv_proj.weight,
            self.qkv_proj.bias,
            self.norm.weight,
            get_norm_bias(self.norm),
            self._use_layer_norm,
            self._head_dim,
            self.norm.eps or 1e-5,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            self._use_layer_norm,
            qk_eps,
            self._qk_normalization,
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
        qkv_quantization_config: Any | None = None,
        out_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        qk_normalization: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm = make_norm(hidden_size, norm_type, eps=eps, **factory_kwargs)
        self._use_layer_norm = is_layer_norm(norm_type)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._head_dim = hidden_size // num_attention_heads
        self._qk_normalization = qk_normalization
        self.q_norm, self.k_norm = _make_qk_norms(self._head_dim, norm_type, eps, qk_normalization, **factory_kwargs)
        self.qkv_quantization_config = qkv_quantization_config
        self.out_quantization_config = out_quantization_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear(self.q_proj)
        init_linear(self.kv_proj)
        init_linear(self.out_proj)
        self.apply_quantization(self.qkv_quantization_config, self.out_quantization_config)

    def apply_quantization(
        self, qkv_quantization_config: Any | None = None, out_quantization_config: Any | None = None
    ) -> None:
        """Apply quantization to the linear layers using torchao."""
        if qkv_quantization_config is not None:
            quantize_(self.q_proj, qkv_quantization_config)
            quantize_(self.kv_proj, qkv_quantization_config)
        if out_quantization_config is not None:
            quantize_(self.out_proj, out_quantization_config)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        rope_q: Tensor | None = None,
        rope_k: Tensor | None = None,
    ) -> Tensor:
        q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, qk_eps = _get_qk_norm_inputs(self.q_norm, self.k_norm)
        return attention_q_kv_packed(
            # fmt: off
            q,
            kv,
            self.q_proj.weight,
            self.q_proj.bias,
            self.kv_proj.weight,
            self.kv_proj.bias,
            self.norm.weight,
            get_norm_bias(self.norm),
            self._use_layer_norm,
            self._head_dim,
            self.out_proj.weight,
            self.out_proj.bias,
            attn_mask,
            self.norm.eps or 1e-5,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            self._use_layer_norm,
            qk_eps,
            self._qk_normalization,
            self.attention_dropout.p,
            self.dropout.p,
            self.training,
            rope_q,
            rope_k,
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
        q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, qk_eps = _get_qk_norm_inputs(self.q_norm, self.k_norm)
        return attention_weights_q_kv_packed(
            # fmt: off
            q,
            kv,
            self.q_proj.weight,
            self.q_proj.bias,
            self.kv_proj.weight,
            self.kv_proj.bias,
            self.norm.weight,
            get_norm_bias(self.norm),
            self._use_layer_norm,
            self._head_dim,
            self.norm.eps or 1e-5,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            self._use_layer_norm,
            qk_eps,
            self._qk_normalization,
            rope_q,
            rope_k,
            # fmt: on
        )


class AttentivePool(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_queries: int = 1,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        qk_normalization: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if num_queries < 1:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        self.query = nn.Parameter(torch.empty(1, num_queries, hidden_size, **factory_kwargs))
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias=bias,
            device=device,
            dtype=dtype,
            norm_type=norm_type,
            qk_normalization=qk_normalization,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.query)
        self.cross_attention.reset_parameters()

    def _expand_query(self, batch_size: int) -> Tensor:
        return self.query.expand(batch_size, -1, -1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.cross_attention(self._expand_query(x.shape[0]), x)
        if y.shape[1] == 1:
            y = y.squeeze(1)
        return y

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)

    def forward_weights(self, x: Tensor) -> Tensor:
        return self.cross_attention.forward_weights(self._expand_query(x.shape[0]), x)
