from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
) -> Tensor:
    if use_layer_norm:
        return F.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    return F.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)
