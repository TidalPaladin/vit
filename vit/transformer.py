import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention
from .drop_path import drop_path
from .fused import NormMLP


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
        attn_bias: bool = False,
        radial_degree: int = 2,
        angular_degree: int = 4,
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
            attn_bias,
            radial_degree,
            angular_degree,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        o = self.self_attention(x, pos)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.mlp(x)
        x = x + drop_path(o, self.drop_path_rate, self.training)
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
        attn_bias: bool = False,
        radial_degree: int = 2,
        angular_degree: int = 4,
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
            attn_bias,
            radial_degree,
            angular_degree,
        )
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            bias,
            eps,
            attn_bias,
            radial_degree,
            angular_degree,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.cross_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, kv: Tensor, posq: Tensor | None = None, posk: Tensor | None = None) -> Tensor:
        o = self.self_attention(x, posq)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.cross_attention(x, kv, posq, posk)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.mlp(x)
        x = x + drop_path(o, self.drop_path_rate, self.training)
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
        attn_bias: bool = False,
        radial_degree: int = 2,
        angular_degree: int = 4,
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
            attn_bias,
            radial_degree,
            angular_degree,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, kv: Tensor, posq: Tensor | None = None, posk: Tensor | None = None) -> Tensor:
        o = self.cross_attention(x, kv, posq, posk)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.mlp(x)
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x
