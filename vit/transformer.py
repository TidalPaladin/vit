from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention, TokenSpecializedAttentionCompileMode
from .fused import AdaNormMLP, NormMLP, _MLPIntermediates
from .initialization import zero_bias_if_present
from .layer_scale import LayerScale
from .norm import NormType
from .packed import PackedAttentionBackend, PackedSequence


def _select_residual_subset(x: Tensor, drop_path_rate: float, training: bool) -> tuple[Tensor, Tensor | None, float]:
    batch_size = x.shape[0]
    if not training or drop_path_rate <= 0.0 or batch_size <= 1:
        return x, None, 1.0

    keep_prob = 1.0 - drop_path_rate
    keep_count = int(batch_size * keep_prob)
    keep_count = max(1, min(batch_size, keep_count))
    if keep_count == batch_size:
        return x, None, 1.0

    keep_indices = torch.randperm(batch_size, device=x.device)[:keep_count]
    residual_scale = float(batch_size / keep_count)
    return x.index_select(0, keep_indices), keep_indices, residual_scale


def _merge_residual_subset(
    x: Tensor,
    residual: Tensor,
    keep_indices: Tensor | None,
    residual_scale: float,
) -> Tensor:
    if keep_indices is None:
        return x + residual
    return x.flatten(1).index_add(0, keep_indices, residual.flatten(1), alpha=residual_scale).view_as(x)


def _packed_drop_path_scale(x: PackedSequence, drop_path_rate: float, training: bool) -> float | Tensor:
    if not training or drop_path_rate <= 0.0:
        return 1.0
    if drop_path_rate >= 1.0:
        return x.values.new_zeros((x.values.shape[0], 1))
    keep_probability = 1.0 - drop_path_rate
    sequence_scale = torch.empty((x.batch_size, 1), device=x.values.device, dtype=x.values.dtype).bernoulli_(
        keep_probability
    )
    sequence_scale.div_(keep_probability)
    return torch.repeat_interleave(sequence_scale, x.lengths.to(torch.int64), dim=0)


def _subset_batched_rope(rope: Tensor | None, keep_indices: Tensor | None, full_batch_size: int) -> Tensor | None:
    if rope is None or keep_indices is None:
        return rope
    if rope.ndim == 5 and rope.shape[1] == full_batch_size:
        return rope.index_select(1, keep_indices)
    return rope


def _subset_batch(tensor: Tensor, keep_indices: Tensor | None, full_batch_size: int) -> Tensor:
    if keep_indices is None:
        return tensor
    if tensor.shape[0] != full_batch_size:
        return tensor
    return tensor.index_select(0, keep_indices)


def _subset_conditioning(tensor: Tensor, keep_indices: Tensor | None, full_batch_size: int) -> Tensor:
    if tensor.ndim == 1:
        return tensor
    return _subset_batch(tensor, keep_indices, full_batch_size)


def _forward_mlp(
    mlp: NormMLP,
    x: Tensor,
    conditioning: Tensor | None,
    keep_indices: Tensor | None,
    full_batch_size: int,
) -> Tensor:
    if isinstance(mlp, AdaNormMLP):
        if conditioning is None:
            raise ValueError("conditioning is required when transformer conditioning_size is set")
        conditioning = _subset_conditioning(conditioning, keep_indices, full_batch_size)
        return mlp(x, conditioning=conditioning)
    if conditioning is not None:
        raise ValueError("conditioning is not supported unless transformer conditioning_size is set")
    return mlp(x)


def _forward_mlp_with_intermediates(
    mlp: NormMLP,
    x: Tensor,
    conditioning: Tensor | None,
    keep_indices: Tensor | None,
    full_batch_size: int,
) -> _MLPIntermediates:
    if isinstance(mlp, AdaNormMLP):
        if conditioning is None:
            raise ValueError("conditioning is required when transformer conditioning_size is set")
        conditioning = _subset_conditioning(conditioning, keep_indices, full_batch_size)
        return mlp._forward_with_intermediates(x, conditioning=conditioning)
    if conditioning is not None:
        raise ValueError("conditioning is not supported unless transformer conditioning_size is set")
    return mlp._forward_with_intermediates(x)


def _make_mlp(
    hidden_size: int,
    ffn_hidden_size: int,
    *,
    bias: bool,
    activation: str,
    norm_type: NormType,
    eps: float,
    dropout: float,
    limit: float | None,
    extra_bias: float | None,
    quantization_config: Any | None,
    conditioning_size: int | None,
    adaln_gate_init: float,
    glu_max_autotune_gemm: bool,
    device: torch.device | None,
    dtype: torch.dtype | None,
    num_global_tokens: int = 0,
) -> NormMLP:
    if conditioning_size is not None:
        return AdaNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=dropout,
            limit=limit,
            extra_bias=extra_bias,
            quantization_config=quantization_config,
            conditioning_size=conditioning_size,
            adaln_gate_init=adaln_gate_init,
            glu_max_autotune_gemm=glu_max_autotune_gemm,
            device=device,
            dtype=dtype,
        )
    return NormMLP(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        bias=bias,
        activation=activation,
        norm_type=norm_type,
        eps=eps,
        dropout=dropout,
        limit=limit,
        extra_bias=extra_bias,
        quantization_config=quantization_config,
        glu_max_autotune_gemm=glu_max_autotune_gemm,
        num_global_tokens=num_global_tokens,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def _zero_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    zero_bias_if_present(module)


@torch.no_grad()
def _zero_residual_outputs(*modules: nn.Linear) -> None:
    for module in modules:
        _zero_linear(module)


@torch.no_grad()
def _zero_mlp_residual_output(mlp: NormMLP) -> None:
    if isinstance(mlp, AdaNormMLP):
        return
    _zero_linear(mlp.fc2)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        attention_bias: bool = True,
        mlp_bias: bool = True,
        activation: str = "gelu",
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
        layer_scale: float | None = None,
        glu_limit: float | None = None,
        glu_extra_bias: float | None = None,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        qk_normalization: bool = False,
        conditioning_size: int | None = None,
        adaln_gate_init: float = 0.0,
        glu_max_autotune_gemm: bool = False,
        num_global_tokens: int = 0,
        specialize_global_token_norms: bool = False,
        specialize_global_token_qkv: bool = False,
        token_specialized_attention_compile_mode: TokenSpecializedAttentionCompileMode = "auto",
        token_specialized_attention_static_batch_sizes: tuple[int, ...] | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if specialize_global_token_norms and conditioning_size is not None:
            raise ValueError("global-token normalization specialization is incompatible with conditioned MLPs")
        self.drop_path_rate = drop_path_rate
        self.self_attention = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias=attention_bias,
            norm_type=norm_type,
            eps=eps,
            qkv_quantization_config=None,
            out_quantization_config=None,
            qk_normalization=qk_normalization,
            num_global_tokens=num_global_tokens,
            specialize_norms=specialize_global_token_norms,
            specialize_qkv=specialize_global_token_qkv,
            token_specialized_attention_compile_mode=token_specialized_attention_compile_mode,
            token_specialized_attention_static_batch_sizes=token_specialized_attention_static_batch_sizes,
            **factory_kwargs,
        )
        self.mlp = _make_mlp(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=mlp_bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=hidden_dropout,
            limit=glu_limit,
            extra_bias=glu_extra_bias,
            quantization_config=None,
            conditioning_size=conditioning_size,
            adaln_gate_init=adaln_gate_init,
            glu_max_autotune_gemm=glu_max_autotune_gemm,
            num_global_tokens=num_global_tokens if specialize_global_token_norms else 0,
            device=device,
            dtype=dtype,
        )
        self.layer_scale_attn = (
            LayerScale(
                hidden_size,
                layer_scale,
                inplace=True,
                num_global_tokens=num_global_tokens if specialize_global_token_norms else 0,
                **factory_kwargs,
            )
            if layer_scale is not None
            else nn.Identity()
        )
        self.layer_scale_mlp = (
            LayerScale(
                hidden_size,
                layer_scale,
                inplace=True,
                num_global_tokens=num_global_tokens if specialize_global_token_norms else 0,
                **factory_kwargs,
            )
            if layer_scale is not None
            else nn.Identity()
        )
        _zero_residual_outputs(self.self_attention.out_proj)
        _zero_mlp_residual_output(self.mlp)
        self.apply_quantization(mlp_quantization_config, qkv_quantization_config, attn_quantization_config)

    def apply_quantization(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
    ) -> None:
        if mlp_quantization_config is not None:
            self.mlp.apply_quantization(mlp_quantization_config)
        if qkv_quantization_config is not None or attn_quantization_config is not None:
            self.self_attention.apply_quantization(qkv_quantization_config, attn_quantization_config)

    def forward(self, x: Tensor, rope: Tensor | None = None, conditioning: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        rope_residual = _subset_batched_rope(rope, keep_indices, batch_size)
        o = self.layer_scale_attn(self.self_attention(x_residual, rope=rope_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        o = self.layer_scale_mlp(_forward_mlp(self.mlp, x_residual, conditioning, keep_indices, batch_size))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)
        return x

    def forward_packed(
        self,
        x: PackedSequence,
        rope: Tensor | None = None,
        *,
        backend: PackedAttentionBackend = "auto",
    ) -> PackedSequence:
        """Apply an encoder block to flat packed values with sequence-level drop path."""
        if isinstance(self.mlp, AdaNormMLP):
            raise RuntimeError("packed transformer execution does not support conditioned MLPs")
        if self.mlp.visual_norm is not None:
            raise RuntimeError("packed transformer execution does not support token specialization")
        if self.mlp.quantization_config is not None:
            raise RuntimeError("packed transformer execution does not support quantized MLP projections")

        attention = self.self_attention.forward_packed(x, rope=rope, backend=backend).values
        attention = self.layer_scale_attn(attention)
        values = x.values + attention * _packed_drop_path_scale(x, self.drop_path_rate, self.training)

        mlp_output = self.mlp._forward_with_intermediates(values).output
        mlp_output = self.layer_scale_mlp(mlp_output)
        values = values + mlp_output * _packed_drop_path_scale(x, self.drop_path_rate, self.training)
        return x.with_values(values)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, rope: Tensor | None = None, conditioning: Tensor | None = None) -> Tensor:
            return self.forward(x, rope, conditioning)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        attention_bias: bool = True,
        mlp_bias: bool = True,
        activation: str = "gelu",
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
        layer_scale: float | None = None,
        glu_limit: float | None = None,
        glu_extra_bias: float | None = None,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        qk_normalization: bool = False,
        conditioning_size: int | None = None,
        adaln_gate_init: float = 0.0,
        glu_max_autotune_gemm: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.self_attention = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias=attention_bias,
            norm_type=norm_type,
            eps=eps,
            qkv_quantization_config=None,
            out_quantization_config=None,
            qk_normalization=qk_normalization,
            **factory_kwargs,
        )
        self.cross_attention = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias=attention_bias,
            norm_type=norm_type,
            eps=eps,
            qkv_quantization_config=None,
            out_quantization_config=None,
            qk_normalization=qk_normalization,
            **factory_kwargs,
        )
        self.mlp = _make_mlp(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=mlp_bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=hidden_dropout,
            limit=glu_limit,
            extra_bias=glu_extra_bias,
            quantization_config=None,
            conditioning_size=conditioning_size,
            adaln_gate_init=adaln_gate_init,
            glu_max_autotune_gemm=glu_max_autotune_gemm,
            device=device,
            dtype=dtype,
        )
        self.layer_scale_attn = (
            LayerScale(hidden_size, layer_scale, inplace=True, **factory_kwargs)
            if layer_scale is not None
            else nn.Identity()
        )
        self.layer_scale_mlp = (
            LayerScale(hidden_size, layer_scale, inplace=True, **factory_kwargs)
            if layer_scale is not None
            else nn.Identity()
        )
        self.layer_scale_cross = (
            LayerScale(hidden_size, layer_scale, inplace=True, **factory_kwargs)
            if layer_scale is not None
            else nn.Identity()
        )
        _zero_residual_outputs(self.self_attention.out_proj, self.cross_attention.out_proj)
        _zero_mlp_residual_output(self.mlp)
        self.apply_quantization(mlp_quantization_config, qkv_quantization_config, attn_quantization_config)

    def apply_quantization(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
    ) -> None:
        if mlp_quantization_config is not None:
            self.mlp.apply_quantization(mlp_quantization_config)
        if qkv_quantization_config is not None or attn_quantization_config is not None:
            self.self_attention.apply_quantization(qkv_quantization_config, attn_quantization_config)
        if qkv_quantization_config is not None or attn_quantization_config is not None:
            self.cross_attention.apply_quantization(qkv_quantization_config, attn_quantization_config)

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        rope_q: Tensor | None = None,
        rope_k: Tensor | None = None,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        batch_size = x.shape[0]

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        rope_q_residual = _subset_batched_rope(rope_q, keep_indices, batch_size)
        o = self.layer_scale_attn(self.self_attention(x_residual, rope=rope_q_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        kv_residual = _subset_batch(kv, keep_indices, batch_size)
        rope_q_residual = _subset_batched_rope(rope_q, keep_indices, batch_size)
        rope_k_residual = _subset_batched_rope(rope_k, keep_indices, batch_size)
        o = self.layer_scale_cross(
            self.cross_attention(x_residual, kv_residual, rope_q=rope_q_residual, rope_k=rope_k_residual)
        )
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        o = self.layer_scale_mlp(_forward_mlp(self.mlp, x_residual, conditioning, keep_indices, batch_size))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)
        return x

    if TYPE_CHECKING:

        def __call__(
            self,
            x: Tensor,
            kv: Tensor,
            rope_q: Tensor | None = None,
            rope_k: Tensor | None = None,
            conditioning: Tensor | None = None,
        ) -> Tensor:
            return self.forward(x, kv, rope_q, rope_k, conditioning)


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        attention_bias: bool = True,
        mlp_bias: bool = True,
        activation: str = "gelu",
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
        layer_scale: float | None = None,
        glu_limit: float | None = None,
        glu_extra_bias: float | None = None,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        qk_normalization: bool = False,
        conditioning_size: int | None = None,
        adaln_gate_init: float = 0.0,
        glu_max_autotune_gemm: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.cross_attention = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            bias=attention_bias,
            norm_type=norm_type,
            eps=eps,
            qkv_quantization_config=None,
            out_quantization_config=None,
            qk_normalization=qk_normalization,
            **factory_kwargs,
        )
        self.mlp = _make_mlp(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=mlp_bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=hidden_dropout,
            limit=glu_limit,
            extra_bias=glu_extra_bias,
            quantization_config=None,
            conditioning_size=conditioning_size,
            adaln_gate_init=adaln_gate_init,
            glu_max_autotune_gemm=glu_max_autotune_gemm,
            device=device,
            dtype=dtype,
        )
        self.layer_scale_cross = (
            LayerScale(hidden_size, layer_scale, inplace=True, **factory_kwargs)
            if layer_scale is not None
            else nn.Identity()
        )
        self.layer_scale_mlp = (
            LayerScale(hidden_size, layer_scale, inplace=True, **factory_kwargs)
            if layer_scale is not None
            else nn.Identity()
        )
        _zero_residual_outputs(self.cross_attention.out_proj)
        _zero_mlp_residual_output(self.mlp)
        self.apply_quantization(mlp_quantization_config, qkv_quantization_config, attn_quantization_config)

    def apply_quantization(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
    ) -> None:
        if mlp_quantization_config is not None:
            self.mlp.apply_quantization(mlp_quantization_config)
        if qkv_quantization_config is not None or attn_quantization_config is not None:
            self.cross_attention.apply_quantization(qkv_quantization_config, attn_quantization_config)

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        rope_q: Tensor | None = None,
        rope_k: Tensor | None = None,
        conditioning: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = x.shape[0]

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        kv_residual = _subset_batch(kv, keep_indices, batch_size)
        attn_mask_residual = _subset_batch(attn_mask, keep_indices, batch_size) if attn_mask is not None else None
        rope_q_residual = _subset_batched_rope(rope_q, keep_indices, batch_size)
        rope_k_residual = _subset_batched_rope(rope_k, keep_indices, batch_size)
        o = self.layer_scale_cross(
            self.cross_attention(
                x_residual,
                kv_residual,
                attn_mask=attn_mask_residual,
                rope_q=rope_q_residual,
                rope_k=rope_k_residual,
            )
        )
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        o = self.layer_scale_mlp(_forward_mlp(self.mlp, x_residual, conditioning, keep_indices, batch_size))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)
        return x

    if TYPE_CHECKING:

        def __call__(
            self,
            x: Tensor,
            kv: Tensor,
            rope_q: Tensor | None = None,
            rope_k: Tensor | None = None,
            conditioning: Tensor | None = None,
            attn_mask: Tensor | None = None,
        ) -> Tensor:
            return self.forward(x, kv, rope_q, rope_k, conditioning, attn_mask)
