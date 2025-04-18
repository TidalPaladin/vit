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


class RelativeFactorizedPosition(nn.Module):
    """
    Computes relative factorized position encodings.

    A grid of positions in the interval :math:`[-1, 1]` is first created.
    This grid is then projected into a higher-dimensional space using a multi-layer perceptron (MLP).
    The output is then normalized using a layer normalization. This computation is performed in float32 precision
    to ensure stability at high resolution, and mamtul precision is set to 'high' for this step.

    Args:
        d_in:
            Input dimension size
        d_out:
            Output dimension size

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(self, d_in: int, d_out: int, backend: Backend = DEFAULT_BACKEND, eps: float = 1e-5):
        super().__init__()
        match backend:
            case "pytorch":
                self.linear = nn.Linear(d_in, d_out)
                self.mlp = LayerNormMLP(d_out, d_out, activation="srelu", eps=eps)
            case "te":
                check_te_installed(te)
                self.linear = te.Linear(d_in, d_out)
                self.mlp = te.LayerNormMLP(d_out, d_out, activation="srelu", eps=eps)
            case _:
                raise ValueError(f"Backend {backend} not supported")

    def forward(self, dims: Sequence[int]) -> Tensor:
        with torch.no_grad():
            grid = create_grid(dims, device=cast(Tensor, self.linear.weight).device)
        result = self.linear(grid)
        result = self.mlp(result)
        return result
