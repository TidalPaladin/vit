from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention
from .drop_path import drop_path
from .fused import NormMLP
from .layer_scale import LayerScale
from .moe import MoE
from .norm import NormType


MLPModule = NormMLP | MoE


def _build_mlp(
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
    device: torch.device | None,
    dtype: torch.dtype | None,
    use_moe: bool = False,
    moe_num_experts: int = 0,
    moe_expert_capacity_factor: float = 1.0,
    moe_router_jitter_noise: float = 0.0,
    moe_drop_overflow_tokens: bool = True,
    moe_token_top_k: int = 2,
) -> MLPModule:
    factory_kwargs = {"device": device, "dtype": dtype}
    if use_moe:
        return MoE(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_experts=moe_num_experts,
            token_top_k=moe_token_top_k,
            bias=bias,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
            dropout=dropout,
            limit=limit,
            extra_bias=extra_bias,
            capacity_factor=moe_expert_capacity_factor,
            router_jitter_noise=moe_router_jitter_noise,
            drop_overflow_tokens=moe_drop_overflow_tokens,
            quantization_config=quantization_config,
            **factory_kwargs,
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
        **factory_kwargs,
    )


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
        use_moe: bool = False,
        moe_num_experts: int = 0,
        moe_expert_capacity_factor: float = 1.0,
        moe_router_jitter_noise: float = 0.0,
        moe_drop_overflow_tokens: bool = True,
        moe_token_top_k: int = 2,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.is_moe_layer = use_moe
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
        self.mlp = _build_mlp(
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
            device=device,
            dtype=dtype,
            use_moe=use_moe,
            moe_num_experts=moe_num_experts,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_router_jitter_noise=moe_router_jitter_noise,
            moe_drop_overflow_tokens=moe_drop_overflow_tokens,
            moe_token_top_k=moe_token_top_k,
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
        if mlp_quantization_config is not None and hasattr(self.mlp, "apply_quantization"):
            self.mlp.apply_quantization(mlp_quantization_config)
        if qkv_quantization_config is not None or attn_quantization_config is not None:
            self.self_attention.apply_quantization(qkv_quantization_config, attn_quantization_config)

    @torch.compile
    def forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        o = self.layer_scale_attn(self.self_attention(x, rope=rope))
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.layer_scale_mlp(self.mlp(x))
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x

    def forward_with_moe_tensors(
        self, x: Tensor, rope: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(self.mlp, MoE):
            raise RuntimeError("forward_with_moe_tensors called on a non-MoE encoder layer")

        o = self.layer_scale_attn(self.self_attention(x, rope=rope))
        x = x + drop_path(o, self.drop_path_rate, self.training)

        mlp_out, router_logits, expert_token_counts, dropped_token_count, capacity = self.mlp.forward_with_aux(x)
        o = self.layer_scale_mlp(mlp_out)
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x, router_logits, expert_token_counts, dropped_token_count, capacity

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
        self.mlp = _build_mlp(
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
        self.mlp = _build_mlp(
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
