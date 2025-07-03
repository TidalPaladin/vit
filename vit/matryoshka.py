from dataclasses import dataclass
from typing import Self, Type

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
    feature_frac: float
    feedforward_frac: float
    heads_frac: float

    def __post_init__(self):
        if not 0 < self.feature_frac <= 1:
            raise ValueError("feature_frac must be between 0 and 1")  # pragma: no cover
        if not 0 < self.feedforward_frac <= 1:
            raise ValueError("feedforward_frac must be between 0 and 1")  # pragma: no cover
        if not 0 < self.heads_frac <= 1:
            raise ValueError("heads_frac must be between 0 and 1")  # pragma: no cover

    @classmethod
    def default(cls: Type[Self]) -> Self:
        return cls(feature_frac=1.0, feedforward_frac=1.0, heads_frac=1.0)
