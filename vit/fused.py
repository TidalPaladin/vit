from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import get_activation
from .matryoshka import MatryoshkaConfig, slice_matryoshka, slice_matryoshka_weight


@torch.compile(fullgraph=True)
def norm_linear(
    # fmt: off
    x: Tensor,
    weight: Tensor, bias: Tensor | None,
    norm_weight: Tensor,
    eps: float,
    # fmt: on
) -> Tensor:
    x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)
    return F.linear(x, weight, bias)


class NormLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = nn.RMSNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.linear.reset_parameters()
        nn.init.trunc_normal_(self.linear.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return norm_linear(x, self.linear.weight, self.linear.bias, self.norm.weight, self.norm.eps or 1e-5)


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "layout_optimization": True,
        "epilogue_fusion": True,
        "aggressive_fusion": True,
    },
)
def norm_mlp(
    # fmt: off
    x: Tensor,
    fc1_weight: Tensor, fc1_bias: Tensor | None,
    fc2_weight: Tensor, fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)
    x = F.linear(x, fc1_weight, fc1_bias)
    x = activation(x)
    x = F.dropout(x, p=dropout, training=training)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    return x


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "layout_optimization": True,
        "epilogue_fusion": True,
        "aggressive_fusion": True,
    },
)
def norm_mlp_glu(
    # fmt: off
    x: Tensor,
    fc1_weight: Tensor, fc1_bias: Tensor | None,
    fc_lu_weight: Tensor, fc_lu_bias: Tensor | None,
    fc2_weight: Tensor, fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)
    x = activation(F.linear(x, fc1_weight, fc1_bias)) * F.linear(x, fc_lu_weight, fc_lu_bias)
    x = F.dropout(x, p=dropout, training=training)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    return x


class NormMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        bias: bool = True,
        activation: str = "gelu",
        eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.fc_lu = nn.Linear(hidden_size, ffn_hidden_size, bias=bias) if activation.endswith("glu") else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        if self.fc_lu is not None:
            self.fc_lu.reset_parameters()
            nn.init.trunc_normal_(self.fc_lu.weight, std=0.02)

    def forward(self, x: Tensor, matryoshka: MatryoshkaConfig = MatryoshkaConfig()) -> Tensor:
        x = slice_matryoshka(x, matryoshka.feature_frac)
        fc1_weight = slice_matryoshka_weight(self.fc1.weight, matryoshka.feature_frac, matryoshka.feedforward_frac)
        fc1_bias = slice_matryoshka(self.fc1.bias, matryoshka.feedforward_frac) if self.fc1.bias is not None else None
        fc2_weight = slice_matryoshka_weight(self.fc2.weight, matryoshka.feedforward_frac, matryoshka.feature_frac)
        fc2_bias = slice_matryoshka(self.fc2.bias, matryoshka.feature_frac) if self.fc2.bias is not None else None
        norm_weight = slice_matryoshka(self.norm.weight, matryoshka.feature_frac)

        if self.fc_lu is not None:
            fc_lu_weight = slice_matryoshka_weight(
                self.fc_lu.weight, matryoshka.feature_frac, matryoshka.feedforward_frac
            )
            fc_lu_bias = (
                slice_matryoshka(self.fc_lu.bias, matryoshka.feedforward_frac) if self.fc_lu.bias is not None else None
            )
            return norm_mlp_glu(
                # fmt: off
                x,
                fc1_weight, fc1_bias,
                fc_lu_weight, fc_lu_bias,
                fc2_weight, fc2_bias,
                norm_weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                # fmt: on
            )
        else:
            return norm_mlp(
                # fmt: off
                x,
                fc1_weight, fc1_bias,
                fc2_weight, fc2_bias,
                norm_weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                # fmt: on
            )
