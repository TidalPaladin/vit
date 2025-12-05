from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor


@torch.compile(fullgraph=True)
def layer_scale(x: Tensor, gamma: Tensor, inplace: bool = False) -> Tensor:
    if inplace and not x.requires_grad:
        return x.mul_(gamma)
    else:
        return x * gamma


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_value: float = 1e-5,
        inplace: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, **factory_kwargs))
        self.reset_parameters(init_value)

    def reset_parameters(self, value: float = 1e-5):
        nn.init.constant_(self.gamma, value)

    def forward(self, x: Tensor) -> Tensor:
        return layer_scale(x, self.gamma, self.inplace)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
