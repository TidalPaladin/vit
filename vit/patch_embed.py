from typing import Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pos_enc import LearnableFourierFeatures, LearnablePosition


@torch.compile(fullgraph=True, dynamic=False)
def patch_embed(x: Tensor, w_patch: Tensor, b_patch: Tensor | None) -> Tensor:
    patch_size = w_patch.shape[2:]
    if w_patch.ndim == 5:
        y = F.conv3d(x, w_patch, b_patch, stride=patch_size)
    elif w_patch.ndim == 4:
        y = F.conv2d(x, w_patch, b_patch, stride=patch_size)
    elif w_patch.ndim == 3:
        y = F.conv1d(x, w_patch, b_patch, stride=patch_size)
    else:
        raise ValueError(f"Invalid patch weight shape: {w_patch.shape}")
    y = y.flatten(2).transpose(1, 2)
    return y


def _check_mask(with_pos: bool | Tensor, with_image: bool | Tensor) -> None:
    if isinstance(with_pos, bool) and isinstance(with_image, bool) and not (with_pos or with_image):
        raise ValueError("At least one of with_pos or with_image must be True")


def _to_mask(val: bool | Tensor, n: int, device: torch.device) -> Tensor:
    if isinstance(val, bool):
        return torch.full((n, 1, 1), val, dtype=torch.bool, device=device)
    elif isinstance(val, Tensor):
        if val.shape != (n,):
            raise ValueError(f"Invalid mask shape: {val.shape}")
        return val.bool().view(n, 1, 1)
    else:
        raise ValueError(f"Invalid mask value: {val}")  # pragma: no cover


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        img_size: Sequence[int],
        pos_emb: Literal["fourier", "none", "learnable"] = "learnable",
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        match pos_emb:
            case "fourier":
                self.pos_enc = LearnableFourierFeatures(2, hidden_size, **kwargs)
            case "learnable":
                self.pos_enc = LearnablePosition(hidden_size, self.tokenized_size(tuple(img_size)), **kwargs)
            case "none":
                self.pos_enc = None
            case _:
                raise ValueError(f"Invalid pos_emb: {pos_emb}")
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

    def forward(self, x: Tensor, with_pos: bool = True, with_image: bool = True) -> Tensor:
        _check_mask(with_pos, with_image)
        y = patch_embed(x, self.patch.weight, self.patch.bias)
        pos = (
            self.pos_enc(self.tokenized_size(x.shape[2:])).type_as(y)
            if self.pos_enc is not None
            else torch.zeros_like(y)
        )
        y_mask = _to_mask(with_image, y.shape[0], device=y.device)
        pos_mask = _to_mask(with_pos, y.shape[0], device=y.device)
        y = y * y_mask + pos * pos_mask
        return self.norm(y)


class PatchEmbed3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: Sequence[int],
        img_size: Sequence[int],
        eps: float = 1e-5,
        pos_emb: Literal["fourier", "none", "learnable"] = "learnable",
        **kwargs,
    ):
        super().__init__()
        self.patch = nn.Conv3d(in_channels, hidden_size, tuple(patch_size), stride=tuple(patch_size))
        match pos_emb:
            case "fourier":
                self.pos_enc = LearnableFourierFeatures(3, hidden_size, **kwargs)
            case "learnable":
                self.pos_enc = LearnablePosition(hidden_size, self.tokenized_size(tuple(img_size)), **kwargs)
            case "none":
                self.pos_enc = None
            case _:
                raise ValueError(f"Invalid pos_emb: {pos_emb}")
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.patch.reset_parameters()
        self.norm.reset_parameters()
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

    def forward(self, x: Tensor, with_pos: bool | Tensor = True, with_image: bool | Tensor = True) -> Tensor:
        _check_mask(with_pos, with_image)
        y = patch_embed(x, self.patch.weight, self.patch.bias)
        pos = (
            self.pos_enc(self.tokenized_size(x.shape[2:])).type_as(y)
            if self.pos_enc is not None
            else torch.zeros_like(y)
        )
        y_mask = _to_mask(with_image, y.shape[0], device=y.device)
        pos_mask = _to_mask(with_pos, y.shape[0], device=y.device)
        y = y * y_mask + pos * pos_mask
        return self.norm(y)
