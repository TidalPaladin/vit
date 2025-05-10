from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .helpers import compile_is_disabled


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def patch_embed_2d(
    # fmt: off
    x: Tensor,
    patch_size: Tuple[int, int],
    w: Tensor, b: Tensor | None,
    w_norm: Tensor,
    # fmt: on
):
    y = F.conv2d(x, w, b, stride=patch_size)
    y = rearrange(y, "b c h w -> b (h w) c")
    y = F.rms_norm(y, y.shape[-1:], weight=w_norm)
    return y


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def patch_embed_3d(
    # fmt: off
    x: Tensor,
    patch_size: Tuple[int, int, int],
    w: Tensor, b: Tensor | None,
    w_norm: Tensor,
    # fmt: on
):
    y = F.conv3d(x, w, b, stride=patch_size)
    y = rearrange(y, "b c d h w -> b (d h w) c")
    y = F.rms_norm(y, y.shape[-1:], weight=w_norm)
    return y


class PatchEmbed2d(nn.Module):

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int | Sequence[int]):
        super().__init__()
        self._patch_size = tuple(patch_size) if isinstance(patch_size, Sequence) else (patch_size, patch_size)
        self.patch = nn.Conv2d(in_channels, hidden_size, self.patch_size, stride=self.patch_size)
        self.norm = nn.RMSNorm(hidden_size)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def original_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return ht, wt

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed_2d(x, self.patch_size, self.patch.weight, self.patch.bias, self.norm.weight)


class PatchEmbed3d(nn.Module):

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int | Sequence[int]):
        super().__init__()
        self._patch_size = (
            tuple(patch_size) if isinstance(patch_size, Sequence) else (patch_size, patch_size, patch_size)
        )
        self.patch = nn.Conv3d(in_channels, hidden_size, self.patch_size, stride=self.patch_size)
        self.norm = nn.RMSNorm(hidden_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def tokenized_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def original_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed_3d(x, self.patch_size, self.patch.weight, self.patch.bias, self.norm.weight)
