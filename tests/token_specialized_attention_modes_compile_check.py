"""Isolated CPU parity check for every public token-specialized attention compile mode."""

import torch
import torch._dynamo
from torch.testing import assert_close

from vit import TokenSpecializedAttentionCompileMode
from vit.attention import SelfAttention


HIDDEN_SIZE = 16
NUM_HEADS = 4
NUM_GLOBAL_TOKENS = 2
BATCH_SIZE = 2
SEQUENCE_LENGTH = 8
RANDOM_SEED = 0
COMPILE_MODES: tuple[TokenSpecializedAttentionCompileMode, ...] = (
    "auto",
    "dynamic",
    "static",
    "static_max_autotune",
)


def _make_attention(compile_mode: TokenSpecializedAttentionCompileMode) -> SelfAttention:
    static_batch_sizes = (BATCH_SIZE,) if compile_mode != "dynamic" else None
    return SelfAttention(
        HIDDEN_SIZE,
        NUM_HEADS,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_global_tokens=NUM_GLOBAL_TOKENS,
        specialize_norms=True,
        specialize_qkv=True,
        token_specialized_attention_compile_mode=compile_mode,
        token_specialized_attention_static_batch_sizes=static_batch_sizes,
        dtype=torch.float32,
    ).eval()


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    assert torch._dynamo.config.disable is False
    baseline_attention = _make_attention("auto")
    state_dict = baseline_attention.state_dict()
    features = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE)
    baseline_output: torch.Tensor | None = None
    baseline_input_gradient: torch.Tensor | None = None

    for compile_mode in COMPILE_MODES:
        attention = _make_attention(compile_mode)
        attention.load_state_dict(state_dict)
        mode_features = features.clone().requires_grad_()
        output = attention(mode_features)
        output.square().mean().backward()
        assert mode_features.grad is not None
        assert torch.isfinite(output).all()
        assert torch.isfinite(mode_features.grad).all()

        if baseline_output is None:
            baseline_output = output.detach()
            baseline_input_gradient = mode_features.grad.detach()
        else:
            assert baseline_input_gradient is not None
            assert_close(output, baseline_output)
            assert_close(mode_features.grad, baseline_input_gradient)


if __name__ == "__main__":
    main()
