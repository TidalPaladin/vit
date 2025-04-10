from typing import Sequence, Tuple

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .pos_enc import RelativeFactorizedPosition


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Sequence[int],
        normalization: str = "LayerNorm",
    ):
        super().__init__()
        self._patch_size = tuple(patch_size)
        self.patch = nn.Conv2d(in_channels, embed_dim, self.patch_size, stride=self.patch_size)
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim)
        match normalization:
            case "LayerNorm":
                self.norm = nn.LayerNorm(embed_dim)
            case "RMSNorm":
                self.norm = nn.RMSNorm(embed_dim)
            case _:
                raise ValueError(f"Invalid normalization: {normalization}")

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def original_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return ht, wt

    def forward(self, x: Tensor, additional_features: Tensor | None = None) -> Tensor:
        y = self.patch(x)
        y = rearrange(y, "b c h w -> b (h w) c")

        H, W = x.shape[2:]
        dims = self.tokenized_size((H, W))
        pos = self.pos_enc(dims)
        if additional_features is not None:
            y = y + additional_features
        y = y + pos
        return self.norm(y)
