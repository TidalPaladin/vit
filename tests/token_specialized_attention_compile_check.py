"""Isolated CUDA regression check for token-specialized compiled attention."""

import torch
import torch._dynamo

from vit.attention import attention_token_specialized_qkv_packed


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
TRAINING_BATCH_SIZE = 512
ATTENTION_DROPOUT = 0.1
PROJECTION_DROPOUT = 0.1
RANDOM_SEED = 0


def _assert_finite_gradient(tensor: torch.Tensor) -> None:
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


def _run_case(
    *,
    separate_norms: bool,
    separate_qkv: bool,
    batch_size: int = SMALL_BATCH_SIZE,
    num_global_tokens: int = NUM_GLOBAL_TOKENS,
) -> None:
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
    global_qkv_weight = torch.randn(qkv_output_size, HIDDEN_SIZE, device="cuda", requires_grad=True)
    visual_qkv_weight = (
        torch.randn(qkv_output_size, HIDDEN_SIZE, device="cuda", requires_grad=True) if separate_qkv else None
    )
    global_norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", requires_grad=True)
    visual_norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", requires_grad=True) if separate_norms else None
    out_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=True)
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
        return attention_token_specialized_qkv_packed(
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
            ATTENTION_DROPOUT,
            PROJECTION_DROPOUT,
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
        global_norm_weight,
        out_weight,
        q_norm_weight,
        k_norm_weight,
    ]
    if separate_norms:
        assert visual_norm_weight is not None
        parameters.append(visual_norm_weight)
    for parameter in parameters:
        _assert_finite_gradient(parameter)
    if separate_qkv:
        assert visual_qkv_weight is not None
        _assert_finite_gradient(visual_qkv_weight)


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


if __name__ == "__main__":
    main()
