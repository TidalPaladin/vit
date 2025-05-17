from typing import Callable, Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pos_enc import (
    LearnableFourierFeatures,
    LearnablePosition,
    RelativeFactorizedPosition,
    learnable_fourier_features,
    learnable_position,
    relative_factorized_position,
)


@torch.compile(fullgraph=True, dynamic=False)
def patch_embed(
    # fmt: off
    x: Tensor,
    w_patch: Tensor, b_patch: Tensor | None,
    w_norm: Tensor,
    eps: float,
    # fmt: on
) -> Tensor:
    patch_size = w_patch.shape[2:]
    y = F.conv2d(x, w_patch, b_patch, stride=patch_size)
    y = y.flatten(2).transpose(1, 2)
    y = F.rms_norm(y, y.shape[-1:], w_norm, eps)
    return y


@torch.compile(fullgraph=True, dynamic=False)
def patch_embed_relative_factorized_pos(
    # fmt: off
    x: Tensor,
    w_patch: Tensor, b_patch: Tensor | None,
    w_fc1: Tensor, b_fc1: Tensor | None,
    w_fc2: Tensor, b_fc2: Tensor | None,
    w_norm: Tensor,
    eps: float,
    # fmt: on
) -> Tensor:
    dims = x.shape[2:]
    patch_size = w_patch.shape[2:]
    dims = tuple(s // p for s, p in zip(dims, patch_size))
    y = F.conv2d(x, w_patch, b_patch, stride=patch_size)
    y = y.flatten(2).transpose(1, 2)
    y = y + relative_factorized_position(dims, w_fc1, b_fc1, w_fc2, b_fc2)
    y = F.rms_norm(y, y.shape[-1:], w_norm, eps)
    return y


@torch.compile(fullgraph=True, dynamic=False)
def patch_embed_learnable_fourier_pos(
    # fmt: off
    x: Tensor,
    w_patch: Tensor, b_patch: Tensor | None,
    w_fourier: Tensor, b_fourier: Tensor | None,
    w_fc1: Tensor, b_fc1: Tensor | None,
    w_fc2: Tensor, b_fc2: Tensor | None,
    w_norm: Tensor,
    normalize_grid: bool,
    activation: Callable[[Tensor], Tensor],
    dropout: float,
    training: bool,
    eps: float,
    # fmt: on
) -> Tensor:
    dims = x.shape[2:]
    patch_size = w_patch.shape[2:]
    dims = tuple(s // p for s, p in zip(dims, patch_size))
    y = F.conv2d(x, w_patch, b_patch, stride=patch_size)
    y = y.flatten(2).transpose(1, 2)
    pos = learnable_fourier_features(
        dims,
        w_fourier,
        b_fourier,
        w_fc1,
        b_fc1,
        w_fc2,
        b_fc2,
        normalize_grid,
        activation,
        dropout,
        training,
    )
    y = y + pos
    y = F.rms_norm(y, y.shape[-1:], w_norm, eps)
    return y


# @torch.compile(fullgraph=True, dynamic=False)
def patch_embed_learnable_pos(
    # fmt: off
    x: Tensor,
    w_patch: Tensor, b_patch: Tensor | None,
    positions: Tensor,
    positions_size: Sequence[int],
    w_norm: Tensor,
    eps: float,
    # fmt: on
) -> Tensor:
    dims = x.shape[2:]
    patch_size = w_patch.shape[2:]
    dims = tuple(s // p for s, p in zip(dims, patch_size))
    y = F.conv2d(x, w_patch, b_patch, stride=patch_size)
    y = y.flatten(2).transpose(1, 2)
    y = y + learnable_position(dims, positions_size, positions)
    y = F.rms_norm(y, y.shape[-1:], w_norm, eps)
    return y


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        img_size: Sequence[int],
        eps: float = 1e-5,
        pos_emb: Literal["factorized", "fourier", "none", "learnable"] = "learnable",
        **kwargs,
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        self.pos_enc = (
            RelativeFactorizedPosition(2, hidden_size, **kwargs)
            if pos_emb == "factorized"
            else (
                LearnableFourierFeatures(2, hidden_size, **kwargs)
                if pos_emb == "fourier"
                else LearnablePosition(hidden_size, img_size) if pos_emb == "learnable" else None
            )
        )
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.patch.reset_parameters()
        self.norm.reset_parameters()
        if self.pos_enc is not None:
            self.pos_enc.reset_parameters()

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self.patch.weight.shape[2:]

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def original_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return ht, wt

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.pos_enc, RelativeFactorizedPosition):
            return patch_embed_relative_factorized_pos(
                x,
                self.patch.weight,
                self.patch.bias,
                self.pos_enc.fc1.weight,
                self.pos_enc.fc1.bias,
                self.pos_enc.fc2.weight,
                self.pos_enc.fc2.bias,
                self.norm.weight,
                self.norm.eps or 1e-5,
            )
        elif isinstance(self.pos_enc, LearnableFourierFeatures):
            return patch_embed_learnable_fourier_pos(
                x,
                self.patch.weight,
                self.patch.bias,
                self.pos_enc.fourier.weight,
                self.pos_enc.fourier.bias,
                self.pos_enc.fc1.weight,
                self.pos_enc.fc1.bias,
                self.pos_enc.fc2.weight,
                self.pos_enc.fc2.bias,
                self.norm.weight,
                True,
                self.pos_enc.activation,
                self.pos_enc.dropout.p,
                self.training,
                self.norm.eps or 1e-5,
            )
        elif isinstance(self.pos_enc, LearnablePosition):
            return patch_embed_learnable_pos(
                x,
                self.patch.weight,
                self.patch.bias,
                self.pos_enc.positions,
                self.pos_enc.spatial_size,
                self.norm.weight,
                self.norm.eps or 1e-5,
            )
        else:
            return patch_embed(
                x,
                self.patch.weight,
                self.patch.bias,
                self.norm.weight,
                self.norm.eps or 1e-5,
            )
