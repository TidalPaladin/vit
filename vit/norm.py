from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .initialization import zero_bias_if_present


NormType = Literal["rmsnorm", "layernorm"]
NORM_TYPE_CHOICES: tuple[NormType, NormType] = ("rmsnorm", "layernorm")
NormModule = nn.RMSNorm | nn.LayerNorm


def is_layer_norm(norm_type: NormType) -> bool:
    return norm_type == "layernorm"


def make_norm(
    hidden_size: int,
    norm_type: NormType,
    eps: float | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> NormModule:
    factory_kwargs = {"device": device, "dtype": dtype}
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_size, eps=eps, **factory_kwargs)
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps if eps is not None else 1e-5, **factory_kwargs)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def get_norm_bias(norm: NormModule) -> Tensor | None:
    if isinstance(norm, nn.LayerNorm):
        return norm.bias
    return None


def apply_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    eps: float,
    *,
    use_layer_norm: bool,
    scale_delta: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    if use_layer_norm:
        x = F.layer_norm(x, x.shape[-1:], weight=weight, bias=None, eps=eps)
    else:
        x = F.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)

    if scale_delta is not None:
        x = x * (1 + scale_delta)
    if bias is not None:
        x = x + bias
    if shift is not None:
        x = x + shift
    return x


def reshape_modulation(modulation: Tensor, x: Tensor) -> Tensor:
    if modulation.ndim > x.ndim:
        raise ValueError(f"modulation rank {modulation.ndim} exceeds input rank {x.ndim}")
    if modulation.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"modulation hidden size must match input hidden size, got {modulation.shape[-1]} and {x.shape[-1]}"
        )
    if modulation.ndim == 1:
        return modulation

    prefix = modulation.shape[:-1]
    expected_prefix = x.shape[: len(prefix)]
    if prefix != expected_prefix:
        raise ValueError(
            f"modulation shape {tuple(modulation.shape)} is not compatible with input shape {tuple(x.shape)}"
        )

    expand_shape = (*prefix, *([1] * (x.ndim - modulation.ndim)), modulation.shape[-1])
    return modulation.reshape(expand_shape)


@torch.compile(fullgraph=True)
def ada_norm(
    x: Tensor,
    conditioning: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    modulation_weight: Tensor,
    modulation_bias: Tensor | None,
    use_layer_norm: bool,
    eps: float,
) -> Tensor:
    modulation = F.linear(F.silu(conditioning), modulation_weight, modulation_bias)
    scale_delta, shift = modulation.chunk(2, dim=-1)
    return apply_norm(
        x,
        norm_weight,
        norm_bias,
        eps,
        use_layer_norm=use_layer_norm,
        scale_delta=reshape_modulation(scale_delta, x),
        shift=reshape_modulation(shift, x),
    )


class AdaNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int | None = None,
        *,
        norm_type: NormType = "rmsnorm",
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = hidden_size if conditioning_size is None else conditioning_size
        self.norm = make_norm(hidden_size, norm_type, eps=eps, **factory_kwargs)
        self._use_layer_norm = is_layer_norm(norm_type)
        self.modulation = nn.Linear(self.conditioning_size, 2 * hidden_size, **factory_kwargs)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.zeros_(self.modulation.weight)
        zero_bias_if_present(self.modulation)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        return ada_norm(
            x,
            conditioning,
            self.norm.weight,
            get_norm_bias(self.norm),
            self.modulation.weight,
            self.modulation.bias,
            self._use_layer_norm,
            self.norm.eps or 1e-5,
        )

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, conditioning: Tensor) -> Tensor:
            return self.forward(x, conditioning)
