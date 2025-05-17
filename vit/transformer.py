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
        use_fourier_features: bool = False,
        spatial_dims: int = 2,
        fourier_size: int = 384,
        gamma: float = 1.0,
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
            use_fourier_features,
            spatial_dims,
            fourier_size,
            gamma,
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
        use_fourier_features: bool = False,
        spatial_dims: int = 2,
        fourier_size: int = 384,
        gamma: float = 1.0,
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
            use_fourier_features,
            spatial_dims,
            fourier_size,
            gamma,
        )
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout,
            attention_dropout,
            bias,
            eps,
            use_fourier_features,
            spatial_dims,
            fourier_size,
            gamma,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.cross_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, kv: Tensor, pos_q: Tensor | None = None, pos_k: Tensor | None = None) -> Tensor:
        o = self.self_attention(x, pos_q)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.cross_attention(x, kv, pos_q, pos_k)
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
        use_fourier_features: bool = False,
        spatial_dims: int = 2,
        fourier_size: int = 384,
        gamma: float = 1.0,
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
            use_fourier_features,
            spatial_dims,
            fourier_size,
            gamma,
        )
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, eps, hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, kv: Tensor, pos_q: Tensor | None = None, pos_k: Tensor | None = None) -> Tensor:
        o = self.cross_attention(x, kv, pos_q, pos_k)
        x = x + drop_path(o, self.drop_path_rate, self.training)

        o = self.mlp(x)
        x = x + drop_path(o, self.drop_path_rate, self.training)
        return x
