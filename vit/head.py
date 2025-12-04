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


def upsample_head_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return UpsampleHeadConfig(**values)


def register_constructors():
    head_tags = [
        "tag:yaml.org,2002:python/object:vit.head.HeadConfig",
        "tag:yaml.org,2002:python/object:vit.HeadConfig",
    ]
    transposed_conv2d_head_tags = [
        "tag:yaml.org,2002:python/object:vit.head.TransposedConv2dHeadConfig",
        "tag:yaml.org,2002:python/object:vit.TransposedConv2dHeadConfig",
    ]
    upsample_head_tags = [
        "tag:yaml.org,2002:python/object:vit.head.UpsampleHeadConfig",
        "tag:yaml.org,2002:python/object:vit.UpsampleHeadConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in head_tags:
        for loader in loaders:
            loader.add_constructor(tag, head_config_constructor)
    for tag in transposed_conv2d_head_tags:
        for loader in loaders:
            loader.add_constructor(tag, transposed_conv2d_head_config_constructor)
    for tag in upsample_head_tags:
        for loader in loaders:
            loader.add_constructor(tag, upsample_head_config_constructor)


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


@dataclass
class UpsampleHeadConfig:
    """Config for UpsampleHead.

    Args:
        in_features: Number of input channels. If None, uses backbone hidden_size.
        out_features: Number of output channels.
        hidden_features: Hidden channel sizes. Can be an int (same for all stages) or
            a list of ints specifying the output channels for each stage except the last.
            If None, uses in_features. For N stages, a list should have N-1 elements.
        num_upsample_stages: Number of 2x upsample stages (e.g., 4 for 16x total).
        num_smooth_layers: Number of smoothing conv layers per upsample stage.
        dropout: Dropout probability.
    """

    in_features: int | None = None
    out_features: int | None = None
    hidden_features: int | list[int] | None = None
    num_upsample_stages: int = 4
    num_smooth_layers: int = 2
    dropout: float = 0.0

    def instantiate(self, backbone_config: ViTConfig) -> "UpsampleHead":
        in_features = self.in_features or backbone_config.hidden_size
        hidden_features = self.hidden_features if self.hidden_features is not None else in_features
        out_features = self.out_features or backbone_config.hidden_size
        return UpsampleHead(
            in_channels=in_features,
            out_channels=out_features,
            hidden_channels=hidden_features,
            num_upsample_stages=self.num_upsample_stages,
            num_smooth_layers=self.num_smooth_layers,
            dropout=self.dropout,
        )


class UpsampleHead(nn.Module):
    """Head with progressive upsampling and smoothing for artifact-free output.

    Uses interleaved transposed convolutions (2x2, stride 2) for upsampling and
    regular convolutions (3x3, stride 1) for smoothing to reduce tiling artifacts
    between ViT patches.

    Expects input of shape (B, C, H, W) and outputs (B, out_channels, H * scale, W * scale)
    where scale = 2^num_upsample_stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | list[int] | None = None,
        num_upsample_stages: int = 4,
        num_smooth_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_upsample_stages = num_upsample_stages

        # Resolve hidden_channels to a list of output channels for each stage
        if hidden_channels is None:
            hidden_channels_list = [in_channels] * (num_upsample_stages - 1)
        elif isinstance(hidden_channels, int):
            hidden_channels_list = [hidden_channels] * (num_upsample_stages - 1)
        else:
            hidden_channels_list = hidden_channels
            expected_len = num_upsample_stages - 1
            if len(hidden_channels_list) < expected_len:
                raise ValueError(
                    f"hidden_channels list has {len(hidden_channels_list)} elements, but "
                    f"{num_upsample_stages} upsample stages require {expected_len} hidden channel values "
                    f"(one for each stage except the last, which uses out_channels={out_channels})"
                )
            if len(hidden_channels_list) > expected_len:
                raise ValueError(
                    f"hidden_channels list has {len(hidden_channels_list)} elements, but "
                    f"{num_upsample_stages} upsample stages only need {expected_len} hidden channel values "
                    f"(one for each stage except the last, which uses out_channels={out_channels})"
                )

        # Build progressive upsample stages
        upsample_layers: list[nn.ConvTranspose2d] = []
        smooth_layers: list[nn.Sequential] = []

        for i in range(num_upsample_stages):
            if i == 0:
                stage_in = in_channels
            else:
                stage_in = hidden_channels_list[i - 1]

            if i == num_upsample_stages - 1:
                stage_out = out_channels
            else:
                stage_out = hidden_channels_list[i]

            # Transposed conv for 2x upsampling
            upsample_layers.append(nn.ConvTranspose2d(stage_in, stage_out, kernel_size=2, stride=2, padding=0))

            # Smoothing convolutions to blend across patch boundaries
            smooth_modules: list[nn.Module] = []
            for j in range(num_smooth_layers):
                smooth_modules.append(nn.Conv2d(stage_out, stage_out, kernel_size=3, stride=1, padding=1))
                # Add GELU activation except after the last layer of the last stage
                if not (i == num_upsample_stages - 1 and j == num_smooth_layers - 1):
                    smooth_modules.append(nn.GELU())
            smooth_layers.append(nn.Sequential(*smooth_modules))

        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.smooth_layers = nn.ModuleList(smooth_layers)
        self.reset_parameters()

    def reset_parameters(self, bias: float = 0.0) -> None:
        for upsample in self.upsample_layers:
            assert isinstance(upsample, nn.ConvTranspose2d)
            nn.init.trunc_normal_(upsample.weight, std=0.02)
            if upsample.bias is not None:
                nn.init.constant_(upsample.bias, bias)
        for smooth in self.smooth_layers:
            assert isinstance(smooth, nn.Sequential)
            for module in smooth:
                if isinstance(module, nn.Conv2d):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H * scale, W * scale).
        """
        x = self.dropout(x)
        for upsample, smooth in zip(self.upsample_layers, self.smooth_layers):
            x = upsample(x)
            x = x + smooth(x)
        return x

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
