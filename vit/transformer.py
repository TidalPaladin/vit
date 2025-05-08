from typing import TYPE_CHECKING, Literal

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import MultiheadAttention
from .drop_path import drop_path
from .fused import LayerNormMLP
from .helpers import DEFAULT_BACKEND, Backend, try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


class TransformerLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_gqa_groups: int | None = None,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_number: int | None = None,
        layer_type: Literal["encoder", "decoder"] = "encoder",
        normalization: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
        bias: bool = True,
        activation: str = "gelu",
        attn_input_format: Literal["sbhd", "bshd"] = "sbhd",
        drop_path_rate: float = 0.0,
        fuse_qkv_params: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate

        self.self_attention = MultiheadAttention(
            hidden_size,
            num_attention_heads,
            None,
            attention_dropout,
            layer_number,
            num_gqa_groups,
            "self",
            normalization,
            bias,
            attn_input_format,
            True,
            fuse_qkv_params,
            eps,
        )
        if layer_type == "decoder":
            self.inter_attention = MultiheadAttention(
                hidden_size,
                num_attention_heads,
                None,
                attention_dropout,
                layer_number,
                num_gqa_groups,
                "cross",
                normalization,
                bias,
                attn_input_format,
                True,
                fuse_qkv_params,
                eps,
            )
        else:
            self.inter_attention = None
        self.layernorm_mlp = LayerNormMLP(hidden_size, ffn_hidden_size, bias, normalization, activation, eps)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        checkpoint_core_attention: bool = False,
        checkpoint_core_mlp: bool = False,
        core_attention_bias_type: Literal["pre_scale_bias", "post_scale_bias"] = "post_scale_bias",
        core_attention_bias: Tensor | None = None,
    ) -> Tensor:
        o = self.self_attention(x, checkpoint_core_attention=checkpoint_core_attention, core_attention_bias_type=core_attention_bias_type, core_attention_bias=core_attention_bias)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        if self.inter_attention is not None:
            o = self.inter_attention(x, encoder_output, checkpoint_core_attention=checkpoint_core_attention)
            x = x + drop_path(o, self.drop_path_rate, self.training)

        if self.training and checkpoint_core_mlp:
            o = checkpoint(self.layernorm_mlp, x, use_reentrant=False)
        else:
            o = self.layernorm_mlp(x)
        assert isinstance(o, Tensor)
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x


class CrossAttentionMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_gqa_groups: int | None = None,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_number: int | None = None,
        normalization: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
        bias: bool = True,
        activation: str = "gelu",
        attn_input_format: Literal["sbhd", "bshd"] = "sbhd",
        drop_path_rate: float = 0.0,
        fuse_qkv_params: bool = False,
        eps: float = 1e-5,
        backend: Backend = DEFAULT_BACKEND,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        match backend:
            case "pytorch":
                self.inter_attention = MultiheadAttention(
                    hidden_size,
                    num_attention_heads,
                    None,
                    attention_dropout,
                    layer_number,
                    num_gqa_groups,
                    "cross",
                    normalization,
                    bias,
                    attn_input_format,
                    True,
                    fuse_qkv_params,
                    eps,
                )
                self.layernorm_mlp = LayerNormMLP(hidden_size, ffn_hidden_size, bias, normalization, activation, eps)
            case "te":
                self.inter_attention = te.MultiheadAttention(
                    hidden_size,
                    num_attention_heads,
                    None,
                    attention_dropout,
                    eps,
                    layer_number=layer_number,
                    num_gqa_groups=num_gqa_groups,
                    normalization=normalization,
                    bias=bias,
                    attn_mask_type="no_mask",
                    input_layernorm=True,
                    fuse_qkv_params=fuse_qkv_params,
                    qkv_format=attn_input_format,
                    attention_type="cross",
                )
                self.layernorm_mlp = te.LayerNormMLP(
                    hidden_size, ffn_hidden_size, eps, bias=bias, normalization=normalization, activation=activation
                )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        checkpoint_core_attention: bool = False,
        checkpoint_core_mlp: bool = False,
        core_attention_bias_type: Literal["pre_scale_bias", "post_scale_bias"] = "post_scale_bias",
        core_attention_bias: Tensor | None = None,
    ) -> Tensor:
        o = self.inter_attention(x, encoder_output=encoder_output, checkpoint_core_attention=checkpoint_core_attention, core_attention_bias_type=core_attention_bias_type, core_attention_bias=core_attention_bias)
        o = self.hidden_dropout(o)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        if self.training and checkpoint_core_mlp:
            o = checkpoint(self.layernorm_mlp, x, use_reentrant=False)
        else:
            o = self.layernorm_mlp(x)
        assert isinstance(o, Tensor)
        o = self.hidden_dropout(o)
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x
