import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def fourier_features_(param: Tensor, std: float = 1.0) -> None:
    *dims, hidden_size = param.shape
    if hidden_size % 2 != 0:
        raise ValueError(f"Hidden size must be even, got {hidden_size}")

    w = param.new_empty(len(dims), hidden_size // 2)
    w.normal_(std=std)
    grid = create_grid(dims, device=param.device, normalize=True).squeeze_(0)
    features = grid @ w
    features = torch.cat([features.sin(), features.cos()], dim=-1)
    features = features / math.sqrt(hidden_size)
    param.data.copy_(features.view_as(param))


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


@torch.compile(fullgraph=True, dynamic=False)
def learnable_position(dims: Sequence[int], positions_size: Sequence[int], positions: Tensor) -> Tensor:
    L = math.prod(dims)
    if dims != positions_size:
        positions = positions.view(1, *positions_size, -1).movedim(-1, 1)
        positions = F.interpolate(positions, size=dims, mode="bicubic", antialias=False)
        positions = positions.movedim(1, -1)
        positions = positions.view(1, L, -1)
    return positions.view(1, L, -1)


class LearnablePosition(nn.Module):

    def __init__(self, hidden_size: int, spatial_size: Sequence[int]):
        super().__init__()
        total_size = math.prod(spatial_size)
        self.spatial_size = spatial_size
        self.positions = nn.Parameter(torch.empty(total_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fourier_features_(self.positions.view(*self.spatial_size, self.positions.shape[-1]))

    @torch.no_grad()
    def expand_positions(self, size: Sequence[int]) -> None:
        positions = learnable_position(size, self.spatial_size, self.positions)
        self.positions = nn.Parameter(positions.reshape(-1, self.positions.shape[-1]))
        self.spatial_size = size

    def forward(self, dims: Sequence[int] | None) -> Tensor:
        dims = dims or self.spatial_size
        return learnable_position(dims, self.spatial_size, self.positions)


@torch.compile(fullgraph=True, dynamic=False)
def fourier_position(dims: Sequence[int], w: Tensor, w_proj: Tensor, b_proj: Tensor | None) -> Tensor:
    grid = create_grid(dims, device=w.device, normalize=True)
    features = grid @ w
    features = torch.cat([features.sin(), features.cos()], dim=-1)
    features = features / math.sqrt(w.shape[-1])
    return F.linear(features, w_proj, b_proj)


class FourierPosition(nn.Module):

    def __init__(self, hidden_size: int, spatial_size: Sequence[int]):
        super().__init__()
        self.spatial_size = spatial_size
        self.w = nn.Parameter(torch.empty(len(spatial_size), hidden_size // 2))
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self, std: float = 1.0) -> None:
        nn.init.normal_(self.w, std=std)
        self.proj.reset_parameters()

    def forward(self, dims: Sequence[int] | None) -> Tensor:
        dims = dims or self.spatial_size
        return fourier_position(dims, self.w, self.proj.weight, self.proj.bias)


@torch.compile(fullgraph=True, dynamic=False)
def hybrid_position(
    # fmt: off
    dims: Sequence[int], 
    w: Tensor, w_proj: Tensor, b_proj: Tensor | None,
    positions_size: Sequence[int], positions: Tensor,
    # fmt: on
) -> Tensor:
    pos_fourier = fourier_position(dims, w, w_proj, b_proj)
    pos_learnable = learnable_position(dims, positions_size, positions)
    return pos_fourier + pos_learnable


class HybridPosition(nn.Module):

    def __init__(self, hidden_size: int, spatial_size: Sequence[int]):
        super().__init__()
        self.fourier = FourierPosition(hidden_size, spatial_size)
        self.learnable = LearnablePosition(hidden_size, spatial_size)
        self.reset_parameters()

    def reset_parameters(self, std: float = 1.0) -> None:
        self.fourier.reset_parameters(std)
        self.learnable.reset_parameters()
        nn.init.zeros_(self.learnable.positions)

    @torch.no_grad()
    def expand_positions(self, size: Sequence[int]) -> None:
        self.learnable.expand_positions(size)

    def forward(self, dims: Sequence[int] | None) -> Tensor:
        dims = dims or self.learnable.spatial_size
        return hybrid_position(
            dims,
            self.fourier.w,
            self.fourier.proj.weight,
            self.fourier.proj.bias,
            self.learnable.spatial_size,
            self.learnable.positions,
        )
