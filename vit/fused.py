from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import DEFAULT_TRUNC_STD, compile_is_disabled, get_activation


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def norm_linear(x: Tensor, weight: Tensor, bias: Tensor | None, norm_weight: Tensor) -> Tensor:
    y = F.rms_norm(x, x.shape[-1:], weight=norm_weight)
    return F.linear(y, weight, bias)


class NormLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(in_features)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.linear.weight, std=0.023)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        self.norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return norm_linear(x, self.linear.weight, self.linear.bias, self.norm.weight)


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def forward_layer_norm_mlp(
    # fmt: off
    x: Tensor,
    fc1_weight: Tensor, fc1_bias: Tensor | None,
    fc2_weight: Tensor, fc2_bias: Tensor | None,
    norm_weight: Tensor,
    activation: Callable[[Tensor], Tensor],
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    x = F.rms_norm(x, x.shape[-1:], weight=norm_weight)
    x = F.linear(x, fc1_weight, fc1_bias)
    x = activation(x)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    return x


class NormMLP(nn.Module):

    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, bias: bool = True, activation: str = "gelu", dropout: float = 0.0
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)
        self.norm = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.fc1.weight, std=DEFAULT_TRUNC_STD)
        nn.init.trunc_normal_(self.fc2.weight, std=DEFAULT_TRUNC_STD)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        self.norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return forward_layer_norm_mlp(
            # fmt: off
            x,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.norm.weight,
            self.activation,
            self.dropout.p,
            self.training,
            # fmt: on
        )
