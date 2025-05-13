from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.compile(fullgraph=True)
def srelu(x: Tensor) -> Tensor:
    y = x.relu()
    return y.pow(2)


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
