import torch
import torch.nn as nn
from torch import Tensor


@torch.compile(fullgraph=True)
def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """Stochastic depth per sample.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode

    Returns:
        Output tensor after applying stochastic depth
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    noise = torch.empty(shape, dtype=x.dtype, device=x.device)
    noise = noise.bernoulli_(keep_prob)
    if keep_prob > 0:
        noise = noise.div_(keep_prob)
    return x * noise


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        """Initialize DropPath module.

        Args:
            drop_prob: Probability of dropping a path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)
