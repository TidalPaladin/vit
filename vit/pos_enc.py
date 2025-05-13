from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.compile(fullgraph=True)
def create_grid(
    dims: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
) -> Tensor:
    r"""Create a grid of coordinate values given the size of each dimension.

    Args:
        dims:
            The length of each dimension
        proto:
            If provided, a source tensor with which to match device / requires_grad
        normalize:
            If true, normalize coordinate values on the range :math:`\[-1, 1\]`

    Shapes:
        * Output - :math:`(1, L, C)` where :math:`C` is ``len(dims)`` and :math:`L` is ``product(dims)``
    """
    if normalize:
        lens = [torch.linspace(-1, 1, d, device=device, dtype=dtype) for d in dims]
    else:
        lens = [torch.arange(d, device=device, dtype=dtype) for d in dims]
    grid = torch.stack(torch.meshgrid(lens, indexing="ij"), dim=-1)
    return grid.view(1, -1, len(dims))


@torch.compile(fullgraph=True)
def relative_factorized_position(
    dims: Sequence[int],
    w_fc1: Tensor,
    b_fc1: Tensor | None,
    w_fc2: Tensor,
    b_fc2: Tensor | None,
) -> Tensor:
    grid = create_grid(dims, device=w_fc1.device)
    y = F.linear(grid, w_fc1, b_fc1)
    y = F.linear(y, w_fc2, b_fc2)
    return y


class RelativeFactorizedPosition(nn.Module):
    """
    Computes relative factorized position encodings.

    Args:
        d_in:
            Input dimension size
        hidden_size:
            Hidden dimension size

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(self, d_in: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)

    def forward(self, dims: Sequence[int]) -> Tensor:
        return relative_factorized_position(
            # fmt: off
            dims,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            # fmt: on
        )
