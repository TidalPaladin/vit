from typing import Sequence, cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from torch import Tensor

from ..pos_enc import create_grid


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

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.mlp = te.LayerNormMLP(d_out, d_out, activation="srelu")

    def forward(self, dims: Sequence[int]) -> Tensor:
        with torch.no_grad():
            grid = create_grid(dims, device=cast(Tensor, self.linear.weight).device)
        result = self.linear(grid)
        result = self.mlp(result)
        return result
