from typing import Any, Sequence, Tuple, cast

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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels,
            hidden_size,
            cast(Any, tuple(patch_size)),
            stride=cast(Any, tuple(patch_size)),
            **factory_kwargs,
        )
        self.pos_enc = create_position_encoder(
            pos_enc, hidden_size, self.tokenized_size(tuple(img_size)), **factory_kwargs
        )

    def reset_parameters(self, std: float = 0.02) -> None:
        nn.init.trunc_normal_(self.patch.weight, std=std)
        if self.patch.bias is not None:
            nn.init.zeros_(self.patch.bias)
        if self.pos_enc is not None and hasattr(self.pos_enc, "reset_parameters"):
            self.pos_enc.reset_parameters(std=std)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return cast(Tuple[int, int], tuple(self.patch.weight.shape[2:]))

    def tokenized_size(self, size: Sequence[int]) -> Tuple[int, int]:
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch = nn.Conv3d(
            in_channels,
            hidden_size,
            cast(Any, tuple(patch_size)),
            stride=cast(Any, tuple(patch_size)),
            **factory_kwargs,
        )
        self.pos_enc = create_position_encoder(
            pos_enc, hidden_size, self.tokenized_size(tuple(img_size)), **factory_kwargs
        )

    def reset_parameters(self, std: float = 0.02) -> None:
        nn.init.trunc_normal_(self.patch.weight, std=std)
        if self.patch.bias is not None:
            nn.init.zeros_(self.patch.bias)
        if self.pos_enc is not None and hasattr(self.pos_enc, "reset_parameters"):
            self.pos_enc.reset_parameters(std=std)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return cast(Tuple[int, int, int], tuple(self.patch.weight.shape[2:]))

    def tokenized_size(self, size: Sequence[int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def original_size(self, size: Sequence[int]) -> Tuple[int, int, int]:
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
