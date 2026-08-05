"""Isolated CUDA regression check for token-specialized compiled attention."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch._dynamo
from torch.testing import assert_close

from vit import TokenSpecializedAttentionCompileMode
from vit.attention import (
    _STATIC_TOKEN_SPECIALIZED_ATTENTION_MIN_BATCH_SIZE,
    _attention_token_specialized_qkv_packed_impl,
    _select_token_specialized_attention,
    attention_token_specialized_qkv_packed,
)


IMAGE_SIZE = 32
PATCH_SIZE = 4
HIDDEN_SIZE = 384
NUM_HEADS = 12
NUM_CLS_TOKENS = 1
NUM_REGISTER_TOKENS = 7
NUM_GLOBAL_TOKENS = NUM_CLS_TOKENS + NUM_REGISTER_TOKENS
NUM_VISUAL_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2
NUM_MASKED_VISUAL_TOKENS = NUM_VISUAL_TOKENS // 2
SMALL_BATCH_SIZE = 2
TRAINING_BATCH_SIZE = _STATIC_TOKEN_SPECIALIZED_ATTENTION_MIN_BATCH_SIZE
ATTENTION_DROPOUT = 0.1
PROJECTION_DROPOUT = 0.1
RANDOM_SEED = 0
PARITY_WEIGHT_SCALE = HIDDEN_SIZE**-0.5
PARITY_RTOL = 1e-2
PARITY_ATOL = 1e-2


@dataclass(frozen=True)
class _CaseResult:
    inference_output: torch.Tensor
    training_output: torch.Tensor
    gradients: tuple[torch.Tensor, ...]


def _assert_finite_gradient(tensor: torch.Tensor) -> None:
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


def _run_case(
    *,
    separate_norms: bool,
    separate_qkv: bool,
    batch_size: int = SMALL_BATCH_SIZE,
    num_global_tokens: int = NUM_GLOBAL_TOKENS,
    compile_mode: TokenSpecializedAttentionCompileMode = "auto",
    attention_override: Callable[..., torch.Tensor] | None = None,
    attention_dropout: float = ATTENTION_DROPOUT,
    projection_dropout: float = PROJECTION_DROPOUT,
    weight_scale: float = 1.0,
) -> _CaseResult:
    sequence_length = num_global_tokens + NUM_MASKED_VISUAL_TOKENS
    features = torch.randn(batch_size, sequence_length, HIDDEN_SIZE, device="cuda", requires_grad=True)
    with torch.inference_mode():
        full_features = torch.randn(
            batch_size,
            num_global_tokens + NUM_VISUAL_TOKENS,
            HIDDEN_SIZE,
            device="cuda",
        )
    qkv_output_size = 3 * HIDDEN_SIZE
    global_qkv_weight = (torch.randn(qkv_output_size, HIDDEN_SIZE, device="cuda") * weight_scale).requires_grad_()
    visual_qkv_weight = (
        (torch.randn(qkv_output_size, HIDDEN_SIZE, device="cuda") * weight_scale).requires_grad_()
        if separate_qkv
        else None
    )
    global_norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", requires_grad=True)
    visual_norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", requires_grad=True) if separate_norms else None
    out_weight = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda") * weight_scale).requires_grad_()
    head_size = HIDDEN_SIZE // NUM_HEADS
    q_norm_weight = torch.ones(head_size, device="cuda", requires_grad=True)
    k_norm_weight = torch.ones(head_size, device="cuda", requires_grad=True)
    with torch.inference_mode():
        full_rope = torch.randn(2, NUM_VISUAL_TOKENS, head_size, device="cuda")
    masked_rope = torch.randn(
        2,
        batch_size,
        1,
        NUM_MASKED_VISUAL_TOKENS,
        head_size,
        device="cuda",
    )

    def run_attention(input_features: torch.Tensor, rope: torch.Tensor, *, training: bool) -> torch.Tensor:
        if attention_override is None:
            static_batch_sizes = (batch_size,) if compile_mode != "dynamic" else None
            attention = _select_token_specialized_attention(
                input_features,
                training,
                compile_mode,
                static_batch_sizes,
            )
            assert hasattr(attention, "_torchdynamo_orig_callable")
        else:
            attention = attention_override
        return attention(
            input_features,
            num_global_tokens,
            global_qkv_weight,
            None,
            visual_qkv_weight,
            None,
            global_norm_weight,
            None,
            visual_norm_weight,
            None,
            True,
            head_size,
            out_weight,
            None,
            None,
            1e-5,
            q_norm_weight,
            None,
            k_norm_weight,
            None,
            True,
            1e-5,
            True,
            attention_dropout,
            projection_dropout,
            training,
            rope,
        )

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_output = run_attention(full_features, full_rope, training=False)
    assert full_output.shape == full_features.shape
    assert torch.isfinite(full_output).all()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = run_attention(features, masked_rope, training=True)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    parameters = [
        features,
        global_qkv_weight,
    ]
    if separate_qkv:
        assert visual_qkv_weight is not None
        parameters.append(visual_qkv_weight)
    parameters.append(global_norm_weight)
    if separate_norms:
        assert visual_norm_weight is not None
        parameters.append(visual_norm_weight)
    parameters.extend((out_weight, q_norm_weight, k_norm_weight))
    for parameter in parameters:
        _assert_finite_gradient(parameter)

    return _CaseResult(
        inference_output=full_output.detach(),
        training_output=output.detach(),
        gradients=tuple(parameter.grad.detach().clone() for parameter in parameters if parameter.grad is not None),
    )


def _assert_case_parity(actual: _CaseResult, expected: _CaseResult) -> None:
    assert_close(actual.inference_output, expected.inference_output, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    assert_close(actual.training_output, expected.training_output, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    assert len(actual.gradients) == len(expected.gradients)
    for actual_gradient, expected_gradient in zip(actual.gradients, expected.gradients, strict=True):
        assert_close(actual_gradient, expected_gradient, rtol=PARITY_RTOL, atol=PARITY_ATOL)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    assert torch._dynamo.config.disable is False
    assert hasattr(attention_token_specialized_qkv_packed, "_torchdynamo_orig_callable")
    _run_case(separate_norms=True, separate_qkv=False)
    torch._dynamo.reset()
    _run_case(separate_norms=False, separate_qkv=True)
    torch._dynamo.reset()
    _run_case(separate_norms=True, separate_qkv=True)
    torch._dynamo.reset()
    _run_case(separate_norms=True, separate_qkv=True, batch_size=TRAINING_BATCH_SIZE)
    torch._dynamo.reset()
    _run_case(
        separate_norms=True,
        separate_qkv=True,
        batch_size=TRAINING_BATCH_SIZE,
        num_global_tokens=NUM_CLS_TOKENS,
    )
    torch.manual_seed(RANDOM_SEED)
    reference = _run_case(
        separate_norms=True,
        separate_qkv=True,
        attention_override=_attention_token_specialized_qkv_packed_impl,
        attention_dropout=0.0,
        projection_dropout=0.0,
        weight_scale=PARITY_WEIGHT_SCALE,
    )
    for compile_mode in ("dynamic", "static", "static_max_autotune"):
        torch._dynamo.reset()
        torch.manual_seed(RANDOM_SEED)
        actual = _run_case(
            separate_norms=True,
            separate_qkv=True,
            compile_mode=compile_mode,
            attention_dropout=0.0,
            projection_dropout=0.0,
            weight_scale=PARITY_WEIGHT_SCALE,
        )
        _assert_case_parity(actual, reference)


if __name__ == "__main__":
    main()
