"""Isolated CPU regression check for dynamic token-specialized attention shapes."""

from collections.abc import Callable

import torch
import torch._dynamo

from vit.attention import _dynamic_token_specialized_attention, attention_token_specialized_qkv_packed


HIDDEN_SIZE = 8
HEAD_SIZE = 4
NUM_GLOBAL_TOKENS = 1
BATCH_SIZE = 1
RANDOM_SEED = 0


def _run_shape_sweep(attention: Callable[..., torch.Tensor]) -> None:
    qkv_weight = torch.randn(3 * HIDDEN_SIZE, HIDDEN_SIZE)
    visual_qkv_weight = torch.randn_like(qkv_weight)
    norm_weight = torch.ones(HIDDEN_SIZE)
    visual_norm_weight = torch.ones_like(norm_weight)
    out_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)
    shape_count = torch._dynamo.config.recompile_limit + 1

    with torch.inference_mode():
        for num_visual_tokens in range(1, shape_count + 1):
            features = torch.randn(BATCH_SIZE, NUM_GLOBAL_TOKENS + num_visual_tokens, HIDDEN_SIZE)
            output = attention(
                features,
                NUM_GLOBAL_TOKENS,
                qkv_weight,
                None,
                visual_qkv_weight,
                None,
                norm_weight,
                None,
                visual_norm_weight,
                None,
                True,
                HEAD_SIZE,
                out_weight,
                None,
                None,
                1e-5,
                None,
                None,
                None,
                None,
                True,
                1e-5,
                False,
                0.0,
                0.0,
                False,
                None,
            )
            assert output.shape == features.shape
            assert torch.isfinite(output).all()


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    assert torch._dynamo.config.disable is False
    _run_shape_sweep(attention_token_specialized_qkv_packed)
    torch._dynamo.reset()
    _run_shape_sweep(_dynamic_token_specialized_attention)


if __name__ == "__main__":
    main()
