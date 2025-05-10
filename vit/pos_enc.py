from typing import Sequence

import torch
from torch import Tensor

from .helpers import compile_is_disabled


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def create_grid(
    dims: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    normalize: bool = False,
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
    return (2**exponent).float()
