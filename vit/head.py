from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

import torch.nn as nn
import yaml
from torch import Tensor

from .attention import AttentivePool
from .fused import NormLinear, NormMLP


if TYPE_CHECKING:
    from .vit import ViTConfig
else:
    ViTConfig = Any


def head_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return HeadConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:vit.head.HeadConfig",
        "tag:yaml.org,2002:python/object:vit.HeadConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, head_config_constructor)


@dataclass
class HeadConfig:
    head_type: Literal["linear", "mlp"] = "linear"
    pool_type: Literal["avg", "max", "attentive", "none"] = "avg"
    in_dim: int | None = None
    out_dim: int | None = None
    stop_gradient: bool = False
    num_attention_heads: int | None = None
    rope: bool = False
    output_norm: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    def instantiate(self, backbone_config: ViTConfig) -> Union["Head", "MLPHead"]:
        match self.head_type:
            case "linear":
                return Head(
                    self.in_dim or backbone_config.hidden_size,
                    self.pool_type,
                    self.out_dim,
                    self.num_attention_heads or backbone_config.num_attention_heads,
                    self.stop_gradient,
                    self.output_norm,
                )
            case "mlp":
                return MLPHead(
                    self.in_dim or backbone_config.hidden_size,
                    backbone_config.ffn_hidden_size,
                    backbone_config.activation,
                    self.pool_type,
                    self.out_dim,
                    self.num_attention_heads or backbone_config.num_attention_heads,
                    self.stop_gradient,
                    self.output_norm,
                    backbone_config.hidden_dropout,
                )
            case _:
                raise ValueError(f"Invalid head type: {self.head_type}")


class AveragePool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)


class MaxPool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.amax(dim=1)


class Head(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        pool_type: Literal["avg", "max", "attentive", "none"] = "avg",
        out_dim: int | None = None,
        num_attention_heads: int | None = None,
        stop_gradient: bool = False,
        output_norm: bool = False,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        out_dim = out_dim or hidden_size
        self.stop_gradient = stop_gradient
        match pool_type:
            case "avg":
                self.pool = AveragePool()
            case "max":
                self.pool = MaxPool()
            case "attentive":
                if num_attention_heads is None:
                    raise ValueError("num_attention_heads is required for attentive pooling")
                self.pool = AttentivePool(
                    hidden_size, num_attention_heads, hidden_dropout=hidden_dropout, attention_dropout=attention_dropout
                )
            case "none":
                self.pool = nn.Identity()
            case _:
                raise ValueError(f"Invalid pool type: {pool_type}")
        self.proj = NormLinear(hidden_size, out_dim, dropout=hidden_dropout)
        self.norm = nn.RMSNorm(out_dim, eps=1e-5) if output_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()
        if isinstance(self.pool, AttentivePool):
            self.pool.reset_parameters()

    def forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        if self.stop_gradient:
            x = x.detach()
        x = self.pool(x, rope) if isinstance(self.pool, AttentivePool) else self.pool(x)
        return self.norm(self.proj(x))

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
            return self.forward(x)


class MLPHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = "gelu",
        pool_type: Literal["avg", "max", "attentive", "none"] = "avg",
        out_dim: int | None = None,
        num_attention_heads: int | None = None,
        stop_gradient: bool = False,
        output_norm: bool = False,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        out_dim = out_dim or hidden_size
        self.stop_gradient = stop_gradient
        match pool_type:
            case "avg":
                self.pool = AveragePool()
            case "max":
                self.pool = MaxPool()
            case "attentive":
                if num_attention_heads is None:
                    raise ValueError("num_attention_heads is required for attentive pooling")
                self.pool = AttentivePool(
                    hidden_size, num_attention_heads, hidden_dropout=hidden_dropout, attention_dropout=attention_dropout
                )
            case "none":
                self.pool = nn.Identity()
            case _:
                raise ValueError(f"Invalid pool type: {pool_type}")
        self.neck = NormMLP(hidden_size, ffn_hidden_size, activation=activation, dropout=hidden_dropout)
        self.proj = NormLinear(hidden_size, out_dim, dropout=hidden_dropout)
        self.norm = nn.RMSNorm(out_dim, eps=1e-5) if output_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()
        self.neck.reset_parameters()
        if isinstance(self.pool, AttentivePool):
            self.pool.reset_parameters()

    def forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        if self.stop_gradient:
            x = x.detach()
        x = self.pool(x, rope) if isinstance(self.pool, AttentivePool) else self.pool(x)
        x = self.neck(x)
        return self.norm(self.proj(x))

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
            return self.forward(x, rope)
