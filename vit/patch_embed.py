import math
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence, Tuple

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .helpers import (
    DEFAULT_BACKEND,
    Backend,
    check_convnext_installed,
    check_te_installed,
    try_import_convnext,
    try_import_te,
)
from .pos_enc import RelativeFactorizedPosition


if TYPE_CHECKING:
    import convnext  # type: ignore[reportMissingImports]
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()
    convnext = try_import_convnext()


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


class ConvNextPatchEmbed2d(PatchEmbed2d):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Sequence[int],
        normalization: str = "LayerNorm",
        backend: Backend = DEFAULT_BACKEND,
        eps: float = 1e-5,
        depth: int = 2,
        convnext_patch_size: Sequence[int] = [2, 2],
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim, patch_size, normalization, backend, eps)
        check_convnext_installed(convnext)
        assert len(set(patch_size)) == 1, "Patch size must be the same for all dimensions"
        assert all(p % 2 == 0 for p in patch_size), "Patch size must be even"
        assert len(set(convnext_patch_size)) == 1, "ConvNext patch size must be the same for all dimensions"
        assert all(p % 2 == 0 for p in convnext_patch_size), "ConvNext patch size must be even"
        assert convnext_patch_size[0] < patch_size[0], "ConvNext patch size must be less than the input patch size"
        assert convnext_patch_size[1] < patch_size[1], "ConvNext patch size must be less than the input patch size"

        # Determine how many levels we need to match the expected patch size on the output
        # NOTE: We use a single conv on the last level since that level is ready for the transformer
        needed_levels = round(math.log2(max(patch_size))) - round(math.log2(max(convnext_patch_size)))
        depths = [depth] * needed_levels

        # Increment the width of each level by 2x, matching embed_dim on the last level
        hidden_sizes = list(reversed([embed_dim // (2 ** (i + 1)) for i in range(needed_levels)]))
        ffn_hidden_sizes = [d * 4 for d in hidden_sizes]

        # NOTE: This approach allows leakage between adjacent tokens, which may impact masked image modeling tasks.
        config = convnext.ConvNextConfig(
            in_channels=in_channels,
            patch_size=convnext_patch_size,
            kernel_size=[3, 3],
            depths=depths,
            hidden_sizes=hidden_sizes,
            ffn_hidden_sizes=ffn_hidden_sizes,
            activation="srelu",
            backend=backend,
            checkpoint=True,
            drop_path_rate=0.0,
            normalization=normalization,
        )
        config = replace(config, **kwargs)
        self.patch = config.instantiate()
        self.final_conv = nn.Conv2d(hidden_sizes[-1], embed_dim, 2, stride=2)

    def forward(self, x: Tensor, additional_features: Tensor | None = None) -> Tensor:
        P1, P2 = self.patch_size
        B, _, H, W = x.shape
        Ht, Wt = self.tokenized_size((H, W))

        y = rearrange(x, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=P1, p2=P2)
        y = self.patch(y)
        y = self.final_conv(y)
        y = rearrange(y, "(b h w) c () () -> b (h w) c", b=B, h=Ht, w=Wt)

        pos = self.pos_enc((Ht, Wt))
        if additional_features is not None:
            y = y + additional_features
        y = y + pos
        return self.norm(y)
