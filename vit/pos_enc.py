import math
from typing import TYPE_CHECKING, Literal, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


PositionEncoder = Literal["fourier", "learnable", "none", "rope"]


def create_position_encoder(
    pos_enc: PositionEncoder,
    hidden_size: int,
    spatial_size: Sequence[int],
    **kwargs,
) -> Union["LearnablePosition", "FourierPosition", None]:
    match pos_enc:
        case "learnable":
            kwargs.setdefault("dropout", 0.1)
            return LearnablePosition(hidden_size, spatial_size, **kwargs)
        case "fourier":
            return FourierPosition(hidden_size, spatial_size, **kwargs)
        case "none":
            return None
        case "rope":
            raise ValueError("Please create RoPE position encoder manually")
        case _:
            raise ValueError(f"Invalid position encoder: {pos_enc}")


@torch.no_grad()
def fourier_features_(param: Tensor, std: float = 1.0) -> None:
    *dims, hidden_size = param.shape
    if hidden_size % 2 != 0:
        raise ValueError(f"Hidden size must be even, got {hidden_size}")

    w = param.new_empty(len(dims), hidden_size // 2)
    w.normal_(std=std)
    grid = create_grid(dims, dtype=param.dtype, device=param.device, normalize=True).squeeze_(0)
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
def learnable_position(
    dims: Sequence[int], positions_size: Sequence[int], positions: Tensor, dropout: float = 0.0, training: bool = False
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

    def __init__(
        self,
        hidden_size: int,
        spatial_size: Sequence[int],
        fourier_init: bool = True,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        total_size = math.prod(spatial_size)
        self.spatial_size = spatial_size
        self.positions = nn.Parameter(torch.empty(total_size, hidden_size, **factory_kwargs))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters(fourier_init)

    @torch.no_grad()
    def reset_parameters(self, fourier_init: bool = True) -> None:
        if fourier_init:
            fourier_features_(self.positions.view(*self.spatial_size, self.positions.shape[-1]))
            shift = self.positions.mean()
            scale = 0.02 / self.positions.std()
            self.positions.data.sub_(shift).mul_(scale)
        else:
            nn.init.trunc_normal_(self.positions, std=0.02)

    @torch.no_grad()
    def expand_positions(self, size: Sequence[int]) -> None:
        positions = learnable_position(size, self.spatial_size, self.positions, training=False)
        self.positions = nn.Parameter(positions.reshape(-1, self.positions.shape[-1]))
        self.spatial_size = size

    def forward(self, dims: Sequence[int] | None) -> Tensor:
        dims = dims or self.spatial_size
        return learnable_position(dims, self.spatial_size, self.positions, self.dropout.p, self.training)

    if TYPE_CHECKING:

        def __call__(self, dims: Sequence[int] | None) -> Tensor:
            return self.forward(dims)


@torch.compile(fullgraph=True, dynamic=False)
def fourier_position(
    dims: Sequence[int],
    w_fourier: Tensor,
    w_fc1: Tensor,
    b_fc1: Tensor | None,
) -> Tensor:
    with torch.autocast(device_type=w_fourier.device.type, enabled=False):
        grid = create_grid(dims, device=w_fourier.device, normalize=True)
        features = grid @ w_fourier
        features = torch.cat([features.sin(), features.cos()], dim=-1)
        features = features / math.sqrt(w_fourier.shape[-1])
    return F.linear(features, w_fc1, b_fc1)


class FourierPosition(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        spatial_size: Sequence[int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.spatial_size = spatial_size
        self.w_fourier = nn.Parameter(torch.empty(len(spatial_size), hidden_size // 2, **factory_kwargs))
        self.w_fc1 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self, std: float = 1.0) -> None:
        nn.init.normal_(self.w_fourier, std=std)
        nn.init.zeros_(self.w_fc1.bias)

    def forward(self, dims: Sequence[int] | None) -> Tensor:
        dims = dims or self.spatial_size
        return fourier_position(
            dims,
            self.w_fourier,
            self.w_fc1.weight,
            self.w_fc1.bias,
        )

    if TYPE_CHECKING:

        def __call__(self, dims: Sequence[int] | None) -> Tensor:
            return self.forward(dims)
