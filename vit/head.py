from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn
import yaml
from torch import Tensor


if TYPE_CHECKING:
    from .vit import ViTConfig
else:
    ViTConfig = Any


def head_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return HeadConfig(**values)


def transposed_conv2d_head_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return TransposedConv2dHeadConfig(**values)


def register_constructors():
    head_tags = [
        "tag:yaml.org,2002:python/object:vit.head.HeadConfig",
        "tag:yaml.org,2002:python/object:vit.HeadConfig",
    ]
    transposed_conv2d_head_tags = [
        "tag:yaml.org,2002:python/object:vit.head.TransposedConv2dHeadConfig",
        "tag:yaml.org,2002:python/object:vit.TransposedConv2dHeadConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in head_tags:
        for loader in loaders:
            loader.add_constructor(tag, head_config_constructor)
    for tag in transposed_conv2d_head_tags:
        for loader in loaders:
            loader.add_constructor(tag, transposed_conv2d_head_config_constructor)


@dataclass
class HeadConfig:
    in_features: int | None = None
    out_features: int | None = None
    dropout: float = 0.0

    def instantiate(self, backbone_config: ViTConfig) -> "Head":
        return Head(
            self.in_features or backbone_config.hidden_size,
            self.out_features or backbone_config.hidden_size,
            self.dropout,
        )


@dataclass
class TransposedConv2dHeadConfig:
    """Config for TransposedConv2dHead.

    Args:
        in_features: Number of input features. If None, uses backbone hidden_size.
        out_features: Number of output features. If None, uses backbone hidden_size.
        kernel_size: Size of convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
        output_padding: Additional size added to output shape.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input to output.
        dropout: Dropout probability.
    """

    in_features: int | None = None
    out_features: int | None = None
    kernel_size: int | tuple[int, int] = 4
    stride: int | tuple[int, int] = 2
    padding: int | tuple[int, int] = 1
    output_padding: int | tuple[int, int] = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1
    dropout: float = 0.0

    def instantiate(self, backbone_config: ViTConfig) -> "TransposedConv2dHead":
        in_features = self.in_features or backbone_config.hidden_size
        out_features = self.out_features or backbone_config.hidden_size
        return TransposedConv2dHead(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            dropout=self.dropout,
        )


class Head(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self, bias: float = 0.0) -> None:
        self.proj.reset_parameters()
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        return self.proj(x)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)


class TransposedConv2dHead(nn.Module):
    """Head that applies transposed 2D convolution.

    Expects input of shape (B, D, H, W) and outputs (B, out_channels, H', W')
    where H' and W' depend on the convolution parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 4,
        stride: int | tuple[int, int] = 2,
        padding: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
        self.reset_parameters()

    def reset_parameters(self, bias: float = 0.0) -> None:
        nn.init.trunc_normal_(self.conv_transpose.weight, std=0.02)
        if self.conv_transpose.bias is not None:
            nn.init.constant_(self.conv_transpose.bias, bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, D, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        x = self.dropout(x)
        x = self.conv_transpose(x)
        return x

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
