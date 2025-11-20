from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchao.quantization import quantize_

from .helpers import get_activation


@torch.compile(fullgraph=True)
def norm_linear(
    # fmt: off
    x: Tensor,
    weight: Tensor, bias: Tensor | None,
    norm_weight: Tensor,
    eps: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)
    x = F.dropout(x, p=dropout, training=training)
    return F.linear(x, weight, bias)


class NormLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-5,
        dropout: float = 0.0,
        quantization_config: Any | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = nn.RMSNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.quantization_config = quantization_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.linear.reset_parameters()
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        self.apply_quantization(self.quantization_config)

    def apply_quantization(self, quantization_config: Any | None) -> None:
        """Apply quantization to the linear layer using torchao."""
        if quantization_config is None:
            return
        quantize_(self.linear, quantization_config)

    def forward(self, x: Tensor) -> Tensor:
        return norm_linear(
            x,
            self.linear.weight,
            self.linear.bias,
            self.norm.weight,
            self.norm.eps or 1e-5,
            self.dropout.p,
            self.training,
        )

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)


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
    fc2_weight: Tensor, fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    limit: float | None = None,
    extra_bias: float | None = None,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)

    # FC1 - GLU
    x = F.linear(x, fc1_weight, fc1_bias)
    x_linear, x_glu = x.chunk(2, dim=-1)
    if limit is not None:
        x_linear = x_linear.clamp(min=-limit, max=limit)
        x_glu = x_glu.clamp(min=None, max=limit)
    if extra_bias is not None:
        x_linear = x_linear + extra_bias
    x = activation(x_glu) * x_linear
    x = F.dropout(x, p=dropout, training=training)

    # FC2
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
        limit: float | None = None,
        extra_bias: float | None = None,
        quantization_config: Any | None = None,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        if activation.endswith("glu"):
            self._is_glu = True
            self.fc1 = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=bias)
        else:
            self._is_glu = False
            self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.limit = limit
        self.extra_bias = extra_bias
        self.quantization_config = quantization_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)

        # Apply quantization after weight initialization
        self.apply_quantization(self.quantization_config)

    def apply_quantization(self, quantization_config: Any | None) -> None:
        """Apply quantization to both linear layers using torchao."""
        if quantization_config is None:
            return
        quantize_(self.fc1, quantization_config)
        quantize_(self.fc2, quantization_config)

    def forward(self, x: Tensor) -> Tensor:
        if self._is_glu:
            return norm_mlp_glu(
                # fmt: off
                x,
                self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias,
                self.norm.weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                self.limit,
                self.extra_bias,
                # fmt: on
            )
        else:
            return norm_mlp(
                # fmt: off
                x,
                self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias,
                self.norm.weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                # fmt: on
            )

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
