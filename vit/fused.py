from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import DEFAULT_TRUNC_STD, compile_is_disabled, get_activation


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def forward_layer_norm_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    normalization: str,
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor | None,
) -> Tensor:
    if normalization == "LayerNorm":
        x = F.layer_norm(x, x.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias)
    elif normalization == "RMSNorm":
        x = F.rms_norm(x, x.shape[-1:], weight=layer_norm_weight)
    else:
        raise ValueError(f"Invalid normalization: {normalization}")
    return F.linear(x, weight, bias)


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalization: str = "LayerNorm",
    ):
        super().__init__()
        self.normalization = normalization

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.layer_norm_weight = nn.Parameter(torch.empty(in_features))
        self.layer_norm_bias = nn.Parameter(torch.zeros(in_features)) if normalization == "LayerNorm" else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=DEFAULT_TRUNC_STD)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.layer_norm_weight is not None:
            nn.init.ones_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)

    def forward(self, x: Tensor) -> Tensor:
        return forward_layer_norm_linear(
            x, self.weight, self.bias, self.normalization, self.layer_norm_weight, self.layer_norm_bias
        )


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def forward_layer_norm_mlp(
    x: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
    normalization: str,
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    training: bool,
) -> Tensor:
    if normalization == "LayerNorm":
        x = F.layer_norm(x, x.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
    elif normalization == "RMSNorm":
        x = F.rms_norm(x, x.shape[-1:], weight=layer_norm_weight, eps=eps)
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    x = F.linear(x, fc1_weight, fc1_bias)
    x = activation(x)
    x = F.linear(x, fc2_weight, fc2_bias)
    return x


class LayerNormMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        bias: bool = True,
        normalization: str = "LayerNorm",
        activation: str = "srelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalization = normalization
        self.activation = get_activation(activation)
        self.eps = eps

        self.fc1_weight = nn.Parameter(torch.empty(ffn_hidden_size, hidden_size))
        self.fc1_bias = nn.Parameter(torch.zeros(ffn_hidden_size)) if bias else None
        self.fc2_weight = nn.Parameter(torch.empty(hidden_size, ffn_hidden_size))
        self.fc2_bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

        self.layer_norm_weight = nn.Parameter(torch.empty(hidden_size))
        self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size)) if normalization == "LayerNorm" else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.fc1_weight, std=DEFAULT_TRUNC_STD)
        nn.init.trunc_normal_(self.fc2_weight, std=DEFAULT_TRUNC_STD)
        nn.init.ones_(self.layer_norm_weight)
        if self.fc1_bias is not None:
            nn.init.zeros_(self.fc1_bias)
        if self.fc2_bias is not None:
            nn.init.zeros_(self.fc2_bias)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)

    def forward(self, x: Tensor) -> Tensor:
        return forward_layer_norm_mlp(
            x,
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.normalization,
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.activation,
            self.eps,
            self.training,
        )
