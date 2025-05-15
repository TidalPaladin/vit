from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pos_enc import RelativeFactorizedPosition, relative_factorized_position


@torch.compile(fullgraph=True)
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


@torch.compile(fullgraph=True)
def patch_embed_with_positional_encoding(
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


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        eps: float = 1e-5,
        pos_emb: bool = True,
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        self.pos_enc = RelativeFactorizedPosition(2, hidden_size) if pos_emb else None
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
        if self.pos_enc is not None:
            return patch_embed_with_positional_encoding(
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
        else:
            return patch_embed(
                x,
                self.patch.weight,
                self.patch.bias,
                self.norm.weight,
                self.norm.eps or 1e-5,
            )
