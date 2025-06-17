from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pos_enc import FourierPosition, fourier_position


@torch.compile(fullgraph=True, dynamic=False)
def patch_embed(
    # fmt: off
    x: Tensor,
    w_patch: Tensor, b_patch: Tensor | None,
    w_fourier: Tensor,
    w_fourier_proj: Tensor, b_fourier_proj: Tensor | None,
    w_norm: Tensor,
    eps: float,
    is_3d: bool = False,
    # fmt: on
) -> Tensor:
    dims = x.shape[2:]
    patch_size = w_patch.shape[2:]
    dims = tuple(s // p for s, p in zip(dims, patch_size))
    if is_3d:
        y = F.conv3d(x, w_patch, b_patch, stride=patch_size)
    else:
        y = F.conv2d(x, w_patch, b_patch, stride=patch_size)
    y = y.flatten(2).transpose(1, 2)
    pos = fourier_position(dims, w_fourier, w_fourier_proj, b_fourier_proj)
    y = y + pos
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
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        self.pos_enc = FourierPosition(hidden_size, self.tokenized_size(tuple(img_size)))
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.patch.reset_parameters()
        self.norm.reset_parameters()
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
        return patch_embed(
            x,
            self.patch.weight,
            self.patch.bias,
            self.pos_enc.w,
            self.pos_enc.proj.weight,
            self.pos_enc.proj.bias,
            self.norm.weight,
            self.norm.eps or 1e-5,
        )


class PatchEmbed3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        img_size: Sequence[int],
        eps: float = 1e-5,
    ):
        super().__init__()
        self.patch = nn.Conv3d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        self.pos_enc = FourierPosition(hidden_size, self.tokenized_size(tuple(img_size)))
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.patch.reset_parameters()
        self.norm.reset_parameters()
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

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed(
            x,
            self.patch.weight,
            self.patch.bias,
            self.pos_enc.w,
            self.pos_enc.proj.weight,
            self.pos_enc.proj.bias,
            self.norm.weight,
            self.norm.eps or 1e-5,
            is_3d=True,
        )
