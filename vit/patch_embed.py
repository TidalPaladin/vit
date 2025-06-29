from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .pos_enc import PositionEncoder, create_position_encoder


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        img_size: Sequence[int],
        pos_enc: PositionEncoder = "fourier",
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        self.pos_enc = create_position_encoder(pos_enc, hidden_size, self.tokenized_size(tuple(img_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.patch.reset_parameters()
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

    @torch.compile(fullgraph=True, dynamic=False)
    def forward(self, x: Tensor) -> Tensor:
        y = self.patch(x).flatten(2).transpose(1, 2)
        if self.pos_enc is not None:
            pos = self.pos_enc(self.tokenized_size(x.shape[2:]))
            return y + pos
        else:
            return y


class PatchEmbed3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        img_size: Sequence[int],
        pos_enc: PositionEncoder = "fourier",
    ):
        super().__init__()
        self.patch = nn.Conv3d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        self.pos_enc = create_position_encoder(pos_enc, hidden_size, self.tokenized_size(tuple(img_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.patch.reset_parameters()
        if self.pos_enc is not None:
            self.pos_enc.reset_parameters()

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self.patch.weight.shape[2:]

    def tokenized_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def original_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    @torch.compile(fullgraph=True, dynamic=False)
    def forward(self, x: Tensor) -> Tensor:
        y = self.patch(x).flatten(2).transpose(1, 2)
        if self.pos_enc is not None:
            pos = self.pos_enc(self.tokenized_size(x.shape[2:]))
            return y + pos
        else:
            return y
