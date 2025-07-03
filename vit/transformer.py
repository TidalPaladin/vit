import torch
import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention
from .drop_path import drop_path
from .fused import NormMLP
from .layer_scale import LayerScale
from .matryoshka import MatryoshkaConfig, unslice_matryoshka


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True,
        activation: str = "gelu",
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.self_attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            bias,
            eps,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.layer_scale_attn = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.layer_scale_mlp = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.mlp.reset_parameters()

    @torch.compile
    def forward(self, x: Tensor, matryoshka: MatryoshkaConfig = MatryoshkaConfig()) -> Tensor:
        o = self.layer_scale_attn(self.self_attention(x, matryoshka))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])

        o = self.layer_scale_mlp(self.layer_scale_mlp(self.mlp(x)))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True,
        activation: str = "gelu",
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.self_attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            bias,
            eps,
        )
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            bias,
            eps,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.layer_scale_attn = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.layer_scale_mlp = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.layer_scale_cross = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.cross_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, kv: Tensor, matryoshka: MatryoshkaConfig = MatryoshkaConfig()) -> Tensor:
        o = self.layer_scale_attn(self.self_attention(x, matryoshka))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])

        o = self.layer_scale_cross(self.cross_attention(x, kv))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])

        o = self.layer_scale_mlp(self.mlp(x))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])
        return x


class CrossAttentionTransformer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True,
        activation: str = "gelu",
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            bias,
            eps,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.layer_scale_cross = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.layer_scale_mlp = (
            LayerScale(hidden_size, layer_scale, inplace=True) if layer_scale is not None else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, kv: Tensor, matryoshka: MatryoshkaConfig = MatryoshkaConfig()) -> Tensor:
        o = self.layer_scale_cross(self.cross_attention(x, kv, matryoshka))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])

        o = self.layer_scale_mlp(self.mlp(x))
        x = x + unslice_matryoshka(drop_path(o, self.drop_path_rate, self.training), x.shape[-1])
        return x
