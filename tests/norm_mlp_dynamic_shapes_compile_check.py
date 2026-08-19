"""Isolated regression check for NormMLP across sparse-bucket shapes."""

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch._dynamo


HIDDEN_SIZE = 8
FFN_HIDDEN_SIZE = 16
RECOMPILE_LIMIT = 4
RANDOM_SEED = 0
SHAPES = ((1, 3), (2, 5), (3, 7), (4, 9), (5, 11))


def _configure_eager_compile() -> None:
    original_compile = cast(Any, torch.compile)

    def eager_compile(*args: object, **kwargs: object) -> Callable[..., object]:
        kwargs["backend"] = "eager"
        kwargs.pop("options", None)
        return cast(Callable[..., object], original_compile(*args, **kwargs))

    torch.compile = cast(Any, eager_compile)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.cache_size_limit = RECOMPILE_LIMIT
    _configure_eager_compile()

    from vit.fused import NormMLP

    layer = NormMLP(HIDDEN_SIZE, FFN_HIDDEN_SIZE, activation="swiglu", dropout=0.0).train()
    for batch_size, token_count in SHAPES:
        for track_gradients in (True, False):
            context = nullcontext() if track_gradients else torch.no_grad()
            features = torch.randn(
                batch_size,
                token_count,
                HIDDEN_SIZE,
                requires_grad=track_gradients,
            )
            with context:
                output = layer(features)
                assert output.shape == features.shape
                assert torch.isfinite(output).all()
                if track_gradients:
                    output.sum().backward()
                    assert features.grad is not None
            layer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
