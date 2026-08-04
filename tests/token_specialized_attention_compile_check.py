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
BATCH_SIZE = 2
ATTENTION_DROPOUT = 0.1
PROJECTION_DROPOUT = 0.1


def main() -> None:
    assert torch._dynamo.config.disable is False
    assert hasattr(attention_token_specialized_qkv_packed, "_torchdynamo_orig_callable")
    sequence_length = NUM_GLOBAL_TOKENS + NUM_MASKED_VISUAL_TOKENS
    features = torch.randn(BATCH_SIZE, sequence_length, HIDDEN_SIZE, device="cuda", requires_grad=True)
    with torch.inference_mode():
        full_features = torch.randn(
            BATCH_SIZE,
            NUM_GLOBAL_TOKENS + NUM_VISUAL_TOKENS,
            HIDDEN_SIZE,
            device="cuda",
        )
    qkv_output_size = 3 * HIDDEN_SIZE
    global_qkv_weight = torch.randn(qkv_output_size, HIDDEN_SIZE, device="cuda", requires_grad=True)
    visual_qkv_weight = global_qkv_weight
    global_norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", requires_grad=True)
    visual_norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", requires_grad=True)
    out_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=True)
    head_size = HIDDEN_SIZE // NUM_HEADS
    q_norm_weight = torch.ones(head_size, device="cuda", requires_grad=True)
    k_norm_weight = torch.ones(head_size, device="cuda", requires_grad=True)
    with torch.inference_mode():
        full_rope = torch.randn(2, NUM_VISUAL_TOKENS, head_size, device="cuda")
    masked_rope = torch.randn(
        2,
        BATCH_SIZE,
        1,
        NUM_MASKED_VISUAL_TOKENS,
        head_size,
        device="cuda",
    )

    def run_attention(input_features: torch.Tensor, rope: torch.Tensor, *, training: bool) -> torch.Tensor:
        return attention_token_specialized_qkv_packed(
            input_features,
            NUM_GLOBAL_TOKENS,
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
        run_attention(full_features, full_rope, training=False)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = run_attention(features, masked_rope, training=True)
    output.square().mean().backward()

    assert global_qkv_weight.grad is not None
    assert torch.isfinite(global_qkv_weight.grad).all()


if __name__ == "__main__":
    main()
