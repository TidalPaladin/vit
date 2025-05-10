import os
from typing import Callable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEFAULT_TRUNC_STD: Final[float] = 0.023


def compile_is_disabled() -> bool:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE", "1").lower() == "0"


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def srelu(x: Tensor) -> Tensor:
    return x.relu().pow(2)


class SRelu(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return srelu(x)


def get_activation(activation: str) -> Callable[[Tensor], Tensor]:
    match activation:
        case "relu":
            return F.relu
        case "silu":
            return F.silu
        case "gelu":
            return F.gelu
        case "srelu":
            return srelu
        case _:
            raise ValueError(f"Activation {activation} not supported")


def get_activation_module(activation: str) -> nn.Module:
    match activation:
        case "relu":
            return nn.ReLU()
        case "silu":
            return nn.SiLU()
        case "gelu":
            return nn.GELU()
        case "srelu":
            return SRelu()
        case _:
            raise ValueError(f"Activation {activation} not supported")
