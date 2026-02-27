from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention
from .fused import NormMLP
from .layer_scale import LayerScale
from .norm import NormType


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
    residual_scale = float(1.0 / keep_prob)
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


def _subset_batched_rope(rope: Tensor | None, keep_indices: Tensor | None, full_batch_size: int) -> Tensor | None:
    if rope is None or keep_indices is None:
        return rope
    if rope.ndim == 5 and rope.shape[1] == full_batch_size:
        return rope.index_select(1, keep_indices)
    return rope


def _subset_batch(tensor: Tensor, keep_indices: Tensor | None) -> Tensor:
    if keep_indices is None:
        return tensor
    return tensor.index_select(0, keep_indices)


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
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            **factory_kwargs,
        )
        self.mlp = NormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=mlp_bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=hidden_dropout,
            limit=glu_limit,
            extra_bias=glu_extra_bias,
            quantization_config=mlp_quantization_config,
            **factory_kwargs,
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

    def forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        rope_residual = _subset_batched_rope(rope, keep_indices, batch_size)
        o = self.layer_scale_attn(self.self_attention(x_residual, rope=rope_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        o = self.layer_scale_mlp(self.mlp(x_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)
        return x

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
            return self.forward(x, rope)


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
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
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
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            **factory_kwargs,
        )
        self.mlp = NormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=mlp_bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=hidden_dropout,
            limit=glu_limit,
            extra_bias=glu_extra_bias,
            quantization_config=mlp_quantization_config,
            **factory_kwargs,
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

    def forward(self, x: Tensor, kv: Tensor, rope_q: Tensor | None = None, rope_k: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        rope_q_residual = _subset_batched_rope(rope_q, keep_indices, batch_size)
        o = self.layer_scale_attn(self.self_attention(x_residual, rope=rope_q_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        kv_residual = _subset_batch(kv, keep_indices)
        rope_q_residual = _subset_batched_rope(rope_q, keep_indices, batch_size)
        rope_k_residual = _subset_batched_rope(rope_k, keep_indices, batch_size)
        o = self.layer_scale_cross(
            self.cross_attention(x_residual, kv_residual, rope_q=rope_q_residual, rope_k=rope_k_residual)
        )
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        o = self.layer_scale_mlp(self.mlp(x_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)
        return x

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, kv: Tensor, rope_q: Tensor | None = None, rope_k: Tensor | None = None) -> Tensor:
            return self.forward(x, kv, rope_q, rope_k)


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
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            **factory_kwargs,
        )
        self.mlp = NormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=mlp_bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=hidden_dropout,
            limit=glu_limit,
            extra_bias=glu_extra_bias,
            quantization_config=mlp_quantization_config,
            **factory_kwargs,
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

    def forward(self, x: Tensor, kv: Tensor, rope_q: Tensor | None = None, rope_k: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        kv_residual = _subset_batch(kv, keep_indices)
        rope_q_residual = _subset_batched_rope(rope_q, keep_indices, batch_size)
        rope_k_residual = _subset_batched_rope(rope_k, keep_indices, batch_size)
        o = self.layer_scale_cross(
            self.cross_attention(x_residual, kv_residual, rope_q=rope_q_residual, rope_k=rope_k_residual)
        )
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)

        x_residual, keep_indices, residual_scale = _select_residual_subset(x, self.drop_path_rate, self.training)
        o = self.layer_scale_mlp(self.mlp(x_residual))
        x = _merge_residual_subset(x, o, keep_indices, residual_scale)
        return x

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, kv: Tensor, rope_q: Tensor | None = None, rope_k: Tensor | None = None) -> Tensor:
            return self.forward(x, kv, rope_q, rope_k)
