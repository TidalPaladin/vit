from typing import Literal

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import MultiheadAttention
from .drop_path import drop_path
from .fused import LayerNormMLP


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
        activation: str = "srelu",
        attn_input_format: Literal["sbhd", "bshd"] = "sbhd",
        drop_path_rate: float = 0.0,
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
            )
        else:
            self.inter_attention = None
        self.layernorm_mlp = LayerNormMLP(hidden_size, ffn_hidden_size, bias, normalization, activation, hidden_dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        checkpoint_core_attention: bool = False,
        checkpoint_core_mlp: bool = False,
    ) -> Tensor:
        o = self.self_attention(x, checkpoint_core_attention=checkpoint_core_attention)
        o = x + drop_path(o, self.drop_path_rate, self.training)

        if self.inter_attention is not None:
            o = self.inter_attention(o, encoder_output, checkpoint_core_attention=checkpoint_core_attention)
            o = x + drop_path(o, self.drop_path_rate, self.training)

        if self.training and checkpoint_core_mlp:
            o = checkpoint(self.layernorm_mlp, o, use_reentrant=False)
        else:
            o = self.layernorm_mlp(o)
        assert isinstance(o, Tensor)
        o = x + drop_path(o, self.drop_path_rate, self.training)
        return o
