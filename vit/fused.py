from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchao.quantization import quantize_

from .helpers import get_activation
from .initialization import init_linear, zero_bias_if_present
from .norm import NormType, apply_norm, get_norm_bias, is_layer_norm, make_norm, reshape_modulation


@torch.compile(fullgraph=True)
def norm_linear(
    # fmt: off
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    use_layer_norm: bool,
    eps: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    x = apply_norm(x, norm_weight, norm_bias, eps, use_layer_norm=use_layer_norm)
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = make_norm(in_features, norm_type, eps=eps, **factory_kwargs)
        self._use_layer_norm = is_layer_norm(norm_type)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.quantization_config = quantization_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear(self.linear)
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
            get_norm_bias(self.norm),
            self._use_layer_norm,
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
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    norm_bias: Tensor | None,
    use_layer_norm: bool,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    norm_scale_delta: Tensor | None = None,
    norm_shift: Tensor | None = None,
    output_gate: Tensor | None = None,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = apply_norm(
            x,
            norm_weight,
            norm_bias,
            eps,
            use_layer_norm=use_layer_norm,
            scale_delta=norm_scale_delta,
            shift=norm_shift,
        )
    x = F.linear(x, fc1_weight, fc1_bias)
    x = activation(x)
    x = F.dropout(x, p=dropout, training=training)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=dropout, training=training, inplace=True)
    if output_gate is not None:
        x = x * output_gate
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
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    norm_bias: Tensor | None,
    use_layer_norm: bool,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    limit: float | None = None,
    extra_bias: float | None = None,
    norm_scale_delta: Tensor | None = None,
    norm_shift: Tensor | None = None,
    output_gate: Tensor | None = None,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = apply_norm(
            x,
            norm_weight,
            norm_bias,
            eps,
            use_layer_norm=use_layer_norm,
            scale_delta=norm_scale_delta,
            shift=norm_shift,
        )

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
    if output_gate is not None:
        x = x * output_gate
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm = make_norm(hidden_size, norm_type, eps=eps, **factory_kwargs)
        self._use_layer_norm = is_layer_norm(norm_type)
        if activation.endswith("glu"):
            self._is_glu = True
            self.fc1 = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=bias, **factory_kwargs)
        else:
            self._is_glu = False
            self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.limit = limit
        self.extra_bias = extra_bias
        self.quantization_config = quantization_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear(self.fc1)
        init_linear(self.fc2)

        # Apply quantization after weight initialization
        self.apply_quantization(self.quantization_config)

    def apply_quantization(self, quantization_config: Any | None) -> None:
        """Apply quantization to both linear layers using torchao."""
        if quantization_config is None:
            return
        quantize_(self.fc1, quantization_config)
        quantize_(self.fc2, quantization_config)

    def forward(
        self,
        x: Tensor,
        *,
        norm_scale_delta: Tensor | None = None,
        norm_shift: Tensor | None = None,
        output_gate: Tensor | None = None,
    ) -> Tensor:
        if self._is_glu:
            return norm_mlp_glu(
                # fmt: off
                x,
                self.fc1.weight,
                self.fc1.bias,
                self.fc2.weight,
                self.fc2.bias,
                self.norm.weight,
                get_norm_bias(self.norm),
                self._use_layer_norm,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                self.limit,
                self.extra_bias,
                norm_scale_delta,
                norm_shift,
                output_gate,
                # fmt: on
            )
        else:
            return norm_mlp(
                # fmt: off
                x,
                self.fc1.weight,
                self.fc1.bias,
                self.fc2.weight,
                self.fc2.bias,
                self.norm.weight,
                get_norm_bias(self.norm),
                self._use_layer_norm,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                norm_scale_delta,
                norm_shift,
                output_gate,
                # fmt: on
            )

    if TYPE_CHECKING:

        def __call__(
            self,
            x: Tensor,
            *,
            norm_scale_delta: Tensor | None = None,
            norm_shift: Tensor | None = None,
            output_gate: Tensor | None = None,
        ) -> Tensor:
            return self.forward(x, norm_scale_delta=norm_scale_delta, norm_shift=norm_shift, output_gate=output_gate)


class AdaNormMLP(NormMLP):
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_type: NormType = "rmsnorm",
        conditioning_size: int | None = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=bias,
            activation=activation,
            eps=eps,
            dropout=dropout,
            limit=limit,
            extra_bias=extra_bias,
            quantization_config=quantization_config,
            device=device,
            dtype=dtype,
            norm_type=norm_type,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conditioning_size = hidden_size if conditioning_size is None else conditioning_size
        self.conditioning_activation = nn.SiLU()
        self.modulation = nn.Linear(self.conditioning_size, 3 * hidden_size, **factory_kwargs)
        self.reset_modulation_parameters()

    @torch.no_grad()
    def reset_modulation_parameters(self) -> None:
        nn.init.zeros_(self.modulation.weight)
        zero_bias_if_present(self.modulation)

    def forward(
        self,
        x: Tensor,
        *,
        conditioning: Tensor | None = None,
        norm_scale_delta: Tensor | None = None,
        norm_shift: Tensor | None = None,
        output_gate: Tensor | None = None,
    ) -> Tensor:
        if conditioning is None:
            raise ValueError("conditioning is required for AdaNormMLP")
        if norm_scale_delta is not None or norm_shift is not None or output_gate is not None:
            raise ValueError("AdaNormMLP does not accept manual modulation tensors when conditioning is provided")
        modulation = self.modulation(self.conditioning_activation(conditioning))
        adaptive_shift, adaptive_scale_delta, adaptive_output_gate = modulation.chunk(3, dim=-1)
        return super().forward(
            x,
            norm_scale_delta=reshape_modulation(adaptive_scale_delta, x),
            norm_shift=reshape_modulation(adaptive_shift, x),
            output_gate=reshape_modulation(adaptive_output_gate, x),
        )

    if TYPE_CHECKING:

        def __call__(
            self,
            x: Tensor,
            *,
            conditioning: Tensor | None = None,
            norm_scale_delta: Tensor | None = None,
            norm_shift: Tensor | None = None,
            output_gate: Tensor | None = None,
        ) -> Tensor:
            return self.forward(
                x,
                conditioning=conditioning,
                norm_scale_delta=norm_scale_delta,
                norm_shift=norm_shift,
                output_gate=output_gate,
            )
