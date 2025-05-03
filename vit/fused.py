from typing import Callable, Dict, Sequence

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
    eps: float,
) -> Tensor:
    if normalization == "LayerNorm":
        x = F.layer_norm(x, x.shape[-1:], weight=layer_norm_weight, bias=layer_norm_bias, eps=eps)
    elif normalization == "RMSNorm":
        x = F.rms_norm(x, x.shape[-1:], weight=layer_norm_weight, eps=eps)
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
        parameters_split: Dict[str, int] | Sequence[str] | None = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalization = normalization
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.layer_norm_weight = nn.Parameter(torch.empty(in_features))
        self.layer_norm_bias = nn.Parameter(torch.zeros(in_features)) if normalization == "LayerNorm" else None
        self.eps = eps

        # Contiguous buffers for params
        weight_tensor = torch.empty(self.out_features, self.in_features)
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(self.out_features)

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) " f"with split sizes {self.parameter_split_sizes}"
            )

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in LayerNormLinear.parameters(). This makes it
        # more likely that they will stay contiguous if the weights
        # are manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Construct weight parameter
            param = torch.nn.Parameter(weight_tensor[split_start:split_end])
            nn.init.trunc_normal_(param, std=DEFAULT_TRUNC_STD)
            self.register_parameter(self.weight_names[i], param)

        # Construct bias parameters if needed
        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                assert bias_tensor is not None
                param = torch.nn.Parameter(bias_tensor[split_start:split_end])
                nn.init.zeros_(param)
                self.register_parameter(self.bias_names[i], param)
        else:
            for name in self.bias_names:
                self.register_parameter(name, None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.layer_norm_weight is not None:
            nn.init.ones_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)
        for name in self.weight_names:
            nn.init.trunc_normal_(getattr(self, name), std=DEFAULT_TRUNC_STD)
        for name in self.bias_names:
            if hasattr(self, name) and getattr(self, name) is not None:
                nn.init.zeros_(getattr(self, name))

    def forward(self, x: Tensor) -> Tensor:
        weight = torch.cat([getattr(self, name) for name in self.weight_names], dim=0)
        bias = torch.cat([getattr(self, name) for name in self.bias_names], dim=0) if self.use_bias else None
        return forward_layer_norm_linear(
            x, weight, bias, self.normalization, self.layer_norm_weight, self.layer_norm_bias, self.eps
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
        activation: str = "gelu",
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
        )


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parameters_split: Dict[str, int] | Sequence[str] | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Contiguous buffers for params
        weight_tensor = torch.empty(out_features, in_features)
        bias_tensor = None
        if bias:
            bias_tensor = torch.empty(out_features)

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) " f"with split sizes {self.parameter_split_sizes}"
            )

        # Construct weight parameters
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Construct weight parameter
            param = torch.nn.Parameter(weight_tensor[split_start:split_end])
            nn.init.trunc_normal_(param, std=DEFAULT_TRUNC_STD)
            self.register_parameter(self.weight_names[i], param)

        # Construct bias parameters if needed
        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                assert bias_tensor is not None
                param = torch.nn.Parameter(bias_tensor[split_start:split_end])
                nn.init.zeros_(param)
                self.register_parameter(self.bias_names[i], param)
        else:
            for name in self.bias_names:
                self.register_parameter(name, None)

    def reset_parameters(self) -> None:
        for name in self.weight_names:
            nn.init.trunc_normal_(getattr(self, name), std=DEFAULT_TRUNC_STD)
        for name in self.bias_names:
            if hasattr(self, name) and getattr(self, name) is not None:
                nn.init.zeros_(getattr(self, name))

    def forward(self, x: Tensor) -> Tensor:
        weight = torch.cat([getattr(self, name) for name in self.weight_names], dim=0)
        bias = torch.cat([getattr(self, name) for name in self.bias_names], dim=0) if self.use_bias else None
        return F.linear(x, weight, bias)
