"""Isolated CPU regression check for patch-embedding dynamic shapes."""

import sys
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch._dynamo


CHANNELS = 1
HIDDEN_SIZE = 8
PATCH_SIZE = 4
BASE_IMAGE_SIZE = 16
DEPTH = 4
RECOMPILE_LIMIT = 4
RANDOM_SEED = 0
BATCH_SIZES = (1, 2, 3, 4, 5)
IMAGE_SIZES = (16, 20, 24, 28, 32)


def _configure_eager_compile() -> None:
    original_compile = cast(Any, torch.compile)

    def eager_compile(*args: object, **kwargs: object) -> Callable[..., object]:
        kwargs["backend"] = "eager"
        kwargs.pop("options", None)
        return cast(Callable[..., object], original_compile(*args, **kwargs))

    torch.compile = cast(Any, eager_compile)


def _run_shape_sweep(layer: torch.nn.Module, input_shapes: tuple[tuple[int, ...], ...]) -> None:
    for input_shape in input_shapes:
        for track_gradients in (True, False):
            context = nullcontext() if track_gradients else torch.no_grad()
            features = torch.randn(*input_shape, requires_grad=track_gradients)
            with context:
                output = layer(features)
                assert output.shape[0] == input_shape[0]
                assert torch.isfinite(output).all()
                if track_gradients:
                    output.sum().backward()
                    assert features.grad is not None
            layer.zero_grad(set_to_none=True)


def _make_case(dimensions: str, shape_axis: str) -> tuple[torch.nn.Module, tuple[tuple[int, ...], ...]]:
    from vit.patch_embed import PatchEmbed2d, PatchEmbed3d

    pos_enc = "none" if shape_axis == "resolution" else "fourier"
    if dimensions == "2d":
        layer = PatchEmbed2d(
            CHANNELS,
            HIDDEN_SIZE,
            (PATCH_SIZE, PATCH_SIZE),
            (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE),
            pos_enc=pos_enc,
        ).train()
        if shape_axis == "batch":
            input_shapes = tuple((batch_size, CHANNELS, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE) for batch_size in BATCH_SIZES)
        else:
            input_shapes = tuple((1, CHANNELS, image_size, image_size) for image_size in IMAGE_SIZES)
    else:
        layer = PatchEmbed3d(
            CHANNELS,
            HIDDEN_SIZE,
            (DEPTH, PATCH_SIZE, PATCH_SIZE),
            (DEPTH, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE),
            pos_enc=pos_enc,
        ).train()
        if shape_axis == "batch":
            input_shapes = tuple(
                (batch_size, CHANNELS, DEPTH, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE) for batch_size in BATCH_SIZES
            )
        else:
            input_shapes = tuple((1, CHANNELS, DEPTH, image_size, image_size) for image_size in IMAGE_SIZES)
    return layer, input_shapes


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.cache_size_limit = RECOMPILE_LIMIT
    _configure_eager_compile()

    layer, input_shapes = _make_case(sys.argv[1], sys.argv[2])
    _run_shape_sweep(layer, input_shapes)


if __name__ == "__main__":
    main()
