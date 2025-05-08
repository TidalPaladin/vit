from typing import TYPE_CHECKING, Sequence, cast

import torch
import torch.nn as nn
from torch import Tensor

from .fused import LayerNormMLP
from .helpers import DEFAULT_BACKEND, Backend, check_te_installed, compile_is_disabled, try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


@torch.compile(fullgraph=True, disable=compile_is_disabled())
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


def compute_alibi_slopes(num_attention_heads: int, scale: float = 8.0) -> Tensor:
    r"""Compute AliBi slopes for a given number of attention heads.

    Args:
        num_attention_heads: Number of attention heads.
        scale: Scale of the mask.

    Shapes:
        - output: :math:`(N)` where :math:`N` is the number of attention heads.

    Returns:
        AliBi slopes.
    """
    i = torch.arange(num_attention_heads)
    exponent = -scale * (i + 1) / num_attention_heads
    return (2 ** exponent).float()


def create_2d_alibi_grid(
    q_pos: Tensor,
    k_pos: Tensor,
    distance_slopes: Tensor, 
    angular_slopes: Tensor
) -> Tensor:
    r"""Create AliBi bias grid for a 2D input.

    Args:
        q_pos: Positions of the queries.
        k_pos: Positions of the keys.
        distance_slopes: AliBi slopes for distance.
        angular_slopes: AliBi slopes for angular.

    Shapes:
        - output: :math:`(N)` where :math:`N` is the number of attention heads.

    Returns:
        AliBi slopes.
    """
    if q_pos.shape[0] != k_pos.shape[0]:
        raise ValueError("q_pos and k_pos must have the same batch size")
    if not q_pos.shape[-1] == k_pos.shape[-1] == 2:
        raise ValueError("q_pos and k_pos must have the same number of spatial dimensions")

    # Expand to B, H, Lq, Lk, C
    B, Lq, C = q_pos.shape
    _, Lk, _ = k_pos.shape
    q_pos = q_pos.view(B, 1, Lq, 1, C)
    k_pos = k_pos.view(B, 1, 1, Lk, C)

    # Compute the distance bias between the query and key positions
    with torch.no_grad():
        delta = (q_pos - k_pos)

    distance_bias = torch.norm(delta.abs() * distance_slopes.view(1, -1, 1, 1, 1), dim=-1)
    angular_bias = torch.norm((delta * angular_slopes.view(1, -1, 1, 1, 1)).relu(), dim=-1)
    return distance_bias + angular_bias


class RelativeFactorizedPosition(nn.Module):
    """
    Computes relative factorized position encodings.

    A grid of positions in the interval :math:`[-1, 1]` is first created.
    This grid is then projected into a higher-dimensional space using a single linear projection.

    Args:
        d_in:
            Input dimension size
        d_out:
            Output dimension size

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(
        self,
        d_in: int,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = "gelu",
        normalization: str = "LayerNorm",
        bias: bool = True,
        backend: Backend = DEFAULT_BACKEND,
    ):
        super().__init__()
        match backend:
            case "pytorch":
                self.linear = nn.Linear(d_in, hidden_size, bias=bias)
                self.mlp = LayerNormMLP(
                    hidden_size, ffn_hidden_size, activation=activation, normalization=normalization, bias=bias
                )
            case "te":
                check_te_installed(te)
                self.linear = te.Linear(d_in, hidden_size, bias=bias)
                self.mlp = te.LayerNormMLP(
                    hidden_size, ffn_hidden_size, activation=activation, normalization=normalization, bias=bias
                )
            case _:
                raise ValueError(f"Backend {backend} not supported")

    def forward(self, dims: Sequence[int]) -> Tensor:
        with torch.no_grad():
            grid = create_grid(dims, device=cast(Tensor, self.linear.weight).device)
        y = self.linear(grid)
        y = self.mlp(y)
        return y
