from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention
from .drop_path import drop_path
from .fused import NormMLP
from .layer_scale import LayerScale


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
        init_std: float = 0.02,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.self_attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            attention_bias,
            eps,
            init_std,
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            device=device,
            dtype=attention_dtype,
        )
        self.mlp = NormMLP(
            hidden_size,
            ffn_hidden_size,
            mlp_bias,
            activation,
            eps,
            hidden_dropout,
            glu_limit,
            glu_extra_bias,
            init_std,
            mlp_quantization_config,
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

    @torch.compile
    def forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        o = self.layer_scale_attn(self.self_attention(x, rope=rope))
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.layer_scale_mlp(self.layer_scale_mlp(self.mlp(x)))
        x = x + drop_path(o, self.drop_path_rate, self.training)
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
        init_std: float = 0.02,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.self_attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            attention_bias,
            eps,
            init_std,
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            device=device,
            dtype=attention_dtype,
        )
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            attention_bias,
            eps,
            init_std,
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            **factory_kwargs,
        )
        self.mlp = NormMLP(
            hidden_size,
            ffn_hidden_size,
            mlp_bias,
            activation,
            eps,
            hidden_dropout,
            glu_limit,
            glu_extra_bias,
            init_std,
            mlp_quantization_config,
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
        o = self.layer_scale_attn(self.self_attention(x, rope=rope_q))
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.layer_scale_cross(self.cross_attention(x, kv, rope_q=rope_q, rope_k=rope_k))
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.layer_scale_mlp(self.mlp(x))
        x = x + drop_path(o, self.drop_path_rate, self.training)
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
        init_std: float = 0.02,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            attention_bias,
            eps,
            init_std,
            qkv_quantization_config=qkv_quantization_config,
            out_quantization_config=attn_quantization_config,
            device=device,
            dtype=attention_dtype,
        )
        self.mlp = NormMLP(
            hidden_size,
            ffn_hidden_size,
            mlp_bias,
            activation,
            eps,
            hidden_dropout,
            glu_limit,
            glu_extra_bias,
            init_std,
            mlp_quantization_config,
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
        o = self.layer_scale_cross(self.cross_attention(x, kv, rope_q=rope_q, rope_k=rope_k))
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.layer_scale_mlp(self.mlp(x))
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, kv: Tensor, rope_q: Tensor | None = None, rope_k: Tensor | None = None) -> Tensor:
            return self.forward(x, kv, rope_q, rope_k)
