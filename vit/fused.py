from dataclasses import dataclass
from typing import Callable, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import get_activation


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


def _check_ffn_size(ffn_size: int, desired_ffn_size: int) -> None:
    if desired_ffn_size > ffn_size:
        raise ValueError(f"FFN size {ffn_size} is smaller than desired FFN size {desired_ffn_size}")


@dataclass
class _MLPParams:
    fc1_weight: Tensor
    fc1_bias: Tensor | None
    fc2_weight: Tensor
    fc2_bias: Tensor | None
    fc_lu_weight: Tensor | None
    fc_lu_bias: Tensor | None

    @property
    def ffn_size(self) -> int:
        return self.fc1_weight.shape[0]

    def matryoshka_slice(self, ffn_size: int) -> "_MLPParams":
        if ffn_size == self.ffn_size:
            return self
        _check_ffn_size(self.ffn_size, ffn_size)
        fc1_weight = self.fc1_weight[:ffn_size, :]
        fc1_bias = self.fc1_bias[:ffn_size] if self.fc1_bias is not None else None
        fc2_weight = self.fc2_weight[:, :ffn_size]
        fc2_bias = self.fc2_bias if self.fc2_bias is not None else None
        fc_lu_weight = self.fc_lu_weight[:ffn_size, :] if self.fc_lu_weight is not None else None
        fc_lu_bias = self.fc_lu_bias[:ffn_size] if self.fc_lu_bias is not None else None
        return _MLPParams(fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc_lu_weight, fc_lu_bias)

    def named_parameters(self) -> Iterator[Tuple[str, Tensor | None]]:
        yield "fc1_weight", self.fc1_weight
        yield "fc1_bias", self.fc1_bias
        yield "fc2_weight", self.fc2_weight
        yield "fc2_bias", self.fc2_bias
        if self.fc_lu_weight is not None:
            yield "fc_lu_weight", self.fc_lu_weight
            yield "fc_lu_bias", self.fc_lu_bias


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

    @property
    def ffn_size(self) -> int:
        return self.fc1.weight.shape[0]

    def _is_registered(self, ffn_size: int) -> bool:
        return hasattr(self, f"fc1_weight_{ffn_size}")

    def _register_slice(self, ffn_size: int) -> None:
        params = _MLPParams(
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.fc_lu.weight if self.fc_lu is not None else None,
            self.fc_lu.bias if self.fc_lu is not None else None,
        )
        params = params.matryoshka_slice(ffn_size)
        for name, param in params.named_parameters():
            self.register_buffer(f"{name}_{ffn_size}", param, persistent=False)

    def forward(self, x: Tensor, ffn_size: int | None = None) -> Tensor:
        ffn_size = ffn_size or self.ffn_size

        # Register slice if not already registered
        if not self._is_registered(ffn_size):
            self._register_slice(ffn_size)
        fc1_weight = getattr(self, f"fc1_weight_{ffn_size}")
        fc1_bias = getattr(self, f"fc1_bias_{ffn_size}")
        fc2_weight = getattr(self, f"fc2_weight_{ffn_size}")
        fc2_bias = getattr(self, f"fc2_bias_{ffn_size}")

        if self.fc_lu is not None:
            fc_lu_weight = getattr(self, f"fc_lu_weight_{ffn_size}")
            fc_lu_bias = getattr(self, f"fc_lu_bias_{ffn_size}")
            return norm_mlp_glu(
                # fmt: off
                x,
                fc1_weight, fc1_bias,
                fc_lu_weight, fc_lu_bias,
                fc2_weight, fc2_bias,
                self.norm.weight,
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
                self.norm.weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                # fmt: on
            )
