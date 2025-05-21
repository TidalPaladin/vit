import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helpers import get_activation


@torch.compile(fullgraph=True, dynamic=False)
def learnable_position(
    dims: Sequence[int],
    positions_size: Sequence[int],
    positions: Tensor,
    dropout: float,
    training: bool,
) -> Tensor:
    L = math.prod(dims)
    if dims != positions_size:
        positions = positions.view(1, *positions_size, -1).movedim(-1, 1)
        positions = F.interpolate(positions, size=dims, mode="bicubic", antialias=False)
        positions = positions.movedim(1, -1)
        positions = positions.view(1, L, -1)
    positions = F.dropout(positions, p=dropout, training=training)
    return positions.view(1, L, -1)


class LearnablePosition(nn.Module):

    def __init__(self, hidden_size: int, spatial_size: Sequence[int], dropout: float = 0.0):
        super().__init__()
        total_size = math.prod(spatial_size)
        self.positions = nn.Parameter(torch.empty(total_size, hidden_size))
        self.spatial_size = spatial_size
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positions, std=0.02)

    @torch.no_grad()
    def expand_positions(self, size: Sequence[int]) -> None:
        positions = learnable_position(size, self.spatial_size, self.positions, self.dropout.p, self.training)
        self.positions = nn.Parameter(positions.reshape(-1, self.positions.shape[-1]))

    def forward(self, dims: Sequence[int] | None) -> Tensor:
        dims = dims or self.spatial_size
        return learnable_position(dims, self.spatial_size, self.positions, self.dropout.p, self.training)


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
    # fmt: off
    dims: Sequence[int],
    w_fc1: Tensor, b_fc1: Tensor | None,
    w_fc2: Tensor, b_fc2: Tensor | None,
    # fmt: on
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


@torch.compile(fullgraph=True, dynamic=False)
def learnable_fourier_features(
    # fmt: off
    dims: Sequence[int],
    w_fourier: Tensor, b_fourier: Tensor | None,
    w_fc1: Tensor, b_fc1: Tensor | None,
    w_fc2: Tensor, b_fc2: Tensor | None,
    normalize_grid: bool,
    activation: Callable[[Tensor], Tensor],
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    # Input Fourier features
    grid = create_grid(dims, device=w_fourier.device, normalize=normalize_grid)
    y = F.linear(grid, w_fourier, b_fourier)
    y = torch.cat([y.sin(), y.cos()], dim=-1)
    f = y.shape[-1]
    y = y / math.sqrt(f)

    # MLP
    y = F.linear(y, w_fc1, b_fc1)
    y = activation(y)
    y = F.dropout(y, p=dropout, training=training)
    y = F.linear(y, w_fc2, b_fc2)
    return y


class LearnableFourierFeatures(nn.Module):
    """
    Computes learnable Fourier feature positional embeddings.

    Args:
        d_in:
            Input dimension size
        hidden_size:
            Hidden dimension size
        fourier_size:
            Number of Fourier features
        inner_size:
            Hidden dimension size of the inner MLP
        gamma:
            Scale parameter for the Fourier features at initialization
        dropout:
            Dropout rate
        activation:
            Activation function

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(
        self,
        d_in: int,
        hidden_size: int,
        fourier_size: int | None = None,
        inner_size: int | None = None,
        gamma: float = 1.0,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        fourier_size = fourier_size or hidden_size
        inner_size = inner_size or hidden_size
        assert fourier_size % 2 == 0
        self.gamma = gamma
        self.fourier = nn.Linear(d_in, fourier_size // 2, bias=False)
        self.fc1 = nn.Linear(fourier_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.reset_parameters()

    def reset_parameters(self, gamma: float | None = None) -> None:
        gamma = gamma or self.gamma
        self.fourier.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.normal_(self.fourier.weight, std=gamma**-2.0)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)

    def forward(self, dims: Sequence[int]) -> Tensor:
        return learnable_fourier_features(
            # fmt: off
            dims,
            self.fourier.weight, self.fourier.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            normalize_grid=True,
            activation=self.activation,
            dropout=self.dropout.p,
            training=self.training,
            # fmt: on
        )
