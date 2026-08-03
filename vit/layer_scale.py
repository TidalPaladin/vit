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
        num_global_tokens: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_global_tokens < 0:
            raise ValueError(f"num_global_tokens must be non-negative, got {num_global_tokens}")
        self.inplace = inplace
        self.num_global_tokens = num_global_tokens
        self.gamma = nn.Parameter(torch.empty(dim, **factory_kwargs))
        self.visual_gamma = nn.Parameter(torch.empty(dim, **factory_kwargs)) if num_global_tokens > 0 else None
        self.reset_parameters(init_value)

    def reset_parameters(self, value: float = 1e-5):
        nn.init.constant_(self.gamma, value)
        if self.visual_gamma is not None:
            nn.init.constant_(self.visual_gamma, value)

    def forward(self, x: Tensor) -> Tensor:
        if self.visual_gamma is not None:
            global_features = layer_scale(x[:, : self.num_global_tokens], self.gamma, self.inplace)
            visual_features = layer_scale(x[:, self.num_global_tokens :], self.visual_gamma, self.inplace)
            return torch.cat((global_features, visual_features), dim=1)
        return layer_scale(x, self.gamma, self.inplace)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
