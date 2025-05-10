import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention
from .fused import NormMLP


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        bias: bool = True,
        alibi_scale: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_attention_heads, bias, alibi_scale, dropout)
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, dropout)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        x = x + self.self_attn(x, pos)
        x = x + self.mlp(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        bias: bool = True,
        alibi_scale: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        self_attention: bool = True,
    ):
        super().__init__()
        self.self_attn = (
            SelfAttention(hidden_size, num_attention_heads, bias, alibi_scale, dropout) if self_attention else None
        )
        self.cross_attn = CrossAttention(hidden_size, num_attention_heads, bias, alibi_scale, dropout)
        self.mlp = NormMLP(hidden_size, ffn_hidden_size, bias, activation, dropout)
        self.drop_path_rate = drop_path_rate

    def forward(self, q: Tensor, kv: Tensor, qpos: Tensor, kvpos: Tensor) -> Tensor:
        if self.self_attn is not None:
            q = q + self.self_attn(q, qpos)
        q = q + self.cross_attn(q, kv, qpos, kvpos)
        q = q + self.mlp(q)
        return q
