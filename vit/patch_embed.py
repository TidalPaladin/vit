from typing import TYPE_CHECKING, Sequence, Tuple

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .helpers import DEFAULT_BACKEND, Backend, check_te_installed, try_import_te
from .pos_enc import RelativeFactorizedPosition


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Sequence[int],
        normalization: str = "LayerNorm",
        backend: Backend = DEFAULT_BACKEND,
        eps: float = 1e-5,
    ):
        super().__init__()
        self._patch_size = tuple(patch_size)
        self.patch = nn.Conv2d(in_channels, embed_dim, self.patch_size, stride=self.patch_size)
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim, backend=backend, eps=eps)
        match (normalization, backend):
            case ("LayerNorm", "pytorch"):
                self.norm = nn.LayerNorm(embed_dim, eps=eps)
            case ("RMSNorm", "pytorch"):
                self.norm = nn.RMSNorm(embed_dim, eps=eps)
            case ("LayerNorm", "te"):
                check_te_installed(te)
                self.norm = te.LayerNorm(embed_dim, eps=eps)
            case ("RMSNorm", "te"):
                check_te_installed(te)
                self.norm = te.RMSNorm(embed_dim, eps=eps)
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
