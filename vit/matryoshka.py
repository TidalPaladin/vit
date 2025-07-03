from dataclasses import dataclass

import torch
from torch import Tensor


@torch.compile(fullgraph=True, dynamic=False)
def slice_matryoshka(x: Tensor, frac: float = 1.0) -> Tensor:
    D = x.shape[-1]
    D_sliced = int(D * frac)
    return x[..., :D_sliced]


@torch.compile(fullgraph=True, dynamic=False)
def slice_matryoshka_weight(w: Tensor, input_frac: float = 1.0, output_frac: float = 1.0) -> Tensor:
    D_out, D_in = w.shape
    D_in_sliced = int(D_in * input_frac)
    D_out_sliced = int(D_out * output_frac)
    return w[..., :D_out_sliced, :D_in_sliced]


@torch.compile(fullgraph=True, dynamic=False)
def slice_matryoshka_heads(x: Tensor, frac: float = 1.0) -> Tensor:
    _, H, _, _ = x.shape
    H_sliced = int(H * frac)
    return x[..., :H_sliced, :, :]


@torch.compile(fullgraph=True, dynamic=False)
def unslice_matryoshka(x: Tensor, size: int) -> Tensor:
    if size == x.shape[-1]:
        return x
    out = x.new_zeros(*x.shape[:-1], size)
    out[..., : x.shape[-1]] = x
    return out


@dataclass
class MatryoshkaConfig:
    feature_frac: float | None = None
    feedforward_frac: float | None = None
    heads_frac: float | None = None

    def __post_init__(self):
        if self.feature_frac is not None and not 0 < self.feature_frac <= 1:
            raise ValueError("feature_frac must be between 0 and 1")
        if self.feedforward_frac is not None and not 0 < self.feedforward_frac <= 1:
            raise ValueError("feedforward_frac must be between 0 and 1")
        if self.heads_frac is not None and not 0 < self.heads_frac <= 1:
            raise ValueError("heads_frac must be between 0 and 1")

    def __call__(self, x: Tensor) -> Tensor:
        if self.feature_frac is not None:
            x = slice_matryoshka(x, self.feature_frac)
        return x
