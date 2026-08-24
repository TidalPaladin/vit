"""Isolated CPU regression check for fused MLP flat-token shapes."""

import sys
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch._dynamo


HIDDEN_SIZE = 8
FFN_HIDDEN_SIZE = 16
RECOMPILE_LIMIT = 4
RANDOM_SEED = 0
FLAT_TOKEN_COUNTS = (7, 11, 19, 31, 43)


def _configure_eager_compile() -> None:
    original_compile = cast(Any, torch.compile)

    def eager_compile(*args: object, **kwargs: object) -> Callable[..., object]:
        kwargs["backend"] = "eager"
        kwargs.pop("options", None)
        return cast(Callable[..., object], original_compile(*args, **kwargs))

    torch.compile = cast(Any, eager_compile)


def _run_flat_token_sweep(layer: torch.nn.Module, *, track_gradients: bool) -> None:
    for token_count in FLAT_TOKEN_COUNTS:
        context = nullcontext() if track_gradients else torch.no_grad()
        features = torch.randn(
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


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.cache_size_limit = RECOMPILE_LIMIT
    _configure_eager_compile()

    from vit.fused import NormMLP

    activation = sys.argv[1]
    track_gradients = sys.argv[2] == "grad"
    layer = NormMLP(HIDDEN_SIZE, FFN_HIDDEN_SIZE, activation=activation, dropout=0.0).train()
    _run_flat_token_sweep(layer, track_gradients=track_gradients)


if __name__ == "__main__":
    main()
