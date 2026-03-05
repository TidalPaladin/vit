import torch
import torch.nn as nn
from torch import Tensor


INIT_STD = 0.02
TRUNC_STD_SCALE = 2.0


@torch.no_grad()
def trunc_normal_(tensor: Tensor, std: float = INIT_STD) -> Tensor:
    bound = TRUNC_STD_SCALE * std
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-bound, b=bound)


@torch.no_grad()
def zero_bias_if_present(module: nn.Module) -> None:
    bias = getattr(module, "bias", None)
    if isinstance(bias, Tensor):
        nn.init.zeros_(bias)


@torch.no_grad()
def init_linear(module: nn.Linear, std: float = INIT_STD, *, zero_bias: bool = True) -> None:
    trunc_normal_(module.weight, std=std)
    if zero_bias:
        zero_bias_if_present(module)


@torch.no_grad()
def init_conv(
    module: nn.Conv2d | nn.Conv3d | nn.ConvTranspose2d, std: float = INIT_STD, *, zero_bias: bool = True
) -> None:
    trunc_normal_(module.weight, std=std)
    if zero_bias:
        zero_bias_if_present(module)
