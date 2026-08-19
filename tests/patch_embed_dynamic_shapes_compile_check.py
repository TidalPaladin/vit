"""Isolated regression check for patch embedding across sparse subgroup sizes."""

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch._dynamo


CHANNELS = 1
HIDDEN_SIZE = 8
PATCH_SIZE = 4
IMAGE_SIZE = 16
DEPTH = 4
RECOMPILE_LIMIT = 4
RANDOM_SEED = 0
BATCH_SIZES = (1, 2, 3, 4, 5)


def _configure_eager_compile() -> None:
    original_compile = cast(Any, torch.compile)

    def eager_compile(*args: object, **kwargs: object) -> Callable[..., object]:
        kwargs["backend"] = "eager"
        kwargs.pop("options", None)
        return cast(Callable[..., object], original_compile(*args, **kwargs))

    torch.compile = cast(Any, eager_compile)


def _run_shape_sweep(layer: torch.nn.Module, feature_shape: tuple[int, ...]) -> None:
    for batch_size in BATCH_SIZES:
        for track_gradients in (True, False):
            context = nullcontext() if track_gradients else torch.no_grad()
            features = torch.randn(
                batch_size,
                *feature_shape,
                requires_grad=track_gradients,
            )
            with context:
                output = layer(features)
                assert output.shape[0] == batch_size
                assert torch.isfinite(output).all()
                if track_gradients:
                    output.sum().backward()
                    assert features.grad is not None
            layer.zero_grad(set_to_none=True)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.cache_size_limit = RECOMPILE_LIMIT
    _configure_eager_compile()

    from vit.patch_embed import PatchEmbed2d, PatchEmbed3d

    patch_embed_2d = PatchEmbed2d(
        CHANNELS,
        HIDDEN_SIZE,
        (PATCH_SIZE, PATCH_SIZE),
        (IMAGE_SIZE, IMAGE_SIZE),
    ).train()
    _run_shape_sweep(patch_embed_2d, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    torch._dynamo.reset()
    patch_embed_3d = PatchEmbed3d(
        CHANNELS,
        HIDDEN_SIZE,
        (DEPTH, PATCH_SIZE, PATCH_SIZE),
        (DEPTH, IMAGE_SIZE, IMAGE_SIZE),
    ).train()
    _run_shape_sweep(patch_embed_3d, (CHANNELS, DEPTH, IMAGE_SIZE, IMAGE_SIZE))


if __name__ == "__main__":
    main()
