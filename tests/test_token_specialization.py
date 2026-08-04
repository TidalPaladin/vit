import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from vit import ViT, ViTConfig
from vit.layer_scale import LayerScale
from vit.transformer import TransformerEncoderLayer


HIDDEN_SIZE = 32
NUM_HEADS = 4
NUM_CLS_TOKENS = 1
NUM_REGISTER_TOKENS = 2
NUM_GLOBAL_TOKENS = NUM_CLS_TOKENS + NUM_REGISTER_TOKENS
DEPTH = 3
QKV_SPECIALIZATION_BLOCKS = 1
TEST_PROJECTION_STD = 0.02
COMPILE_TEST_TIMEOUT_SECONDS = 120


def _config(**overrides: object) -> ViTConfig:
    values: dict[str, object] = {
        "in_channels": 3,
        "patch_size": (4, 4),
        "img_size": (8, 8),
        "depth": DEPTH,
        "hidden_size": HIDDEN_SIZE,
        "ffn_hidden_size": HIDDEN_SIZE * 2,
        "num_attention_heads": NUM_HEADS,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "num_cls_tokens": NUM_CLS_TOKENS,
        "num_register_tokens": NUM_REGISTER_TOKENS,
        "pos_enc": "rope",
        "layer_scale": 1e-5,
        "norm_type": "layernorm",
        "dtype": torch.float32,
    }
    values.update(overrides)
    return ViTConfig(**values)  # type: ignore[arg-type]


def _specialized_config() -> ViTConfig:
    return _config(
        specialize_global_token_norms=True,
        specialize_global_token_qkv_blocks=QKV_SPECIALIZATION_BLOCKS,
    )


def test_specialization_requires_global_tokens() -> None:
    with pytest.raises(ValueError, match="at least one CLS or register token"):
        _config(
            num_cls_tokens=0,
            num_register_tokens=0,
            specialize_global_token_norms=True,
        )


def test_qkv_specialization_depth_must_fit_backbone() -> None:
    with pytest.raises(ValueError, match="cannot exceed depth"):
        replace(_specialized_config(), specialize_global_token_qkv_blocks=DEPTH + 1)


def test_specialization_clones_norms_scales_and_early_qkv() -> None:
    model = ViT(_specialized_config())

    for block_index, block in enumerate(model.blocks):
        assert isinstance(block, TransformerEncoderLayer)
        assert isinstance(block.layer_scale_attn, LayerScale)
        assert isinstance(block.layer_scale_mlp, LayerScale)
        attention = block.self_attention
        assert attention.num_global_tokens == NUM_GLOBAL_TOKENS
        assert attention.visual_norm is not None
        assert_close(attention.visual_norm.weight, attention.norm.weight)
        assert block.mlp.visual_norm is not None
        assert_close(block.mlp.visual_norm.weight, block.mlp.norm.weight)
        assert block.layer_scale_attn.visual_gamma is not None
        assert_close(block.layer_scale_attn.visual_gamma, block.layer_scale_attn.gamma)
        assert block.layer_scale_mlp.visual_gamma is not None
        assert_close(block.layer_scale_mlp.visual_gamma, block.layer_scale_mlp.gamma)

        if block_index < QKV_SPECIALIZATION_BLOCKS:
            assert attention.visual_qkv_proj is not None
            assert_close(attention.visual_qkv_proj.weight, attention.qkv_proj.weight)
        else:
            assert attention.visual_qkv_proj is None


def test_specialization_is_functionally_identical_at_initialization() -> None:
    torch.manual_seed(7)
    baseline = ViT(_config()).eval()
    torch.manual_seed(7)
    specialized = ViT(_specialized_config()).eval()
    images = torch.randn(2, 3, 8, 8)

    baseline_features = baseline(images)
    specialized_features = specialized(images)

    assert_close(specialized_features.dense_features, baseline_features.dense_features)


def test_specialized_paths_receive_distinct_gradients() -> None:
    model = ViT(_specialized_config())
    first_block = model.get_block(0)
    torch.nn.init.normal_(first_block.self_attention.out_proj.weight, std=TEST_PROJECTION_STD)
    images = torch.randn(2, 3, 8, 8)

    features = model(images)
    loss = features.cls_tokens.square().mean() + 2.0 * features.visual_tokens.square().mean()
    loss.backward()

    attention = first_block.self_attention
    assert attention.norm.weight.grad is not None
    assert attention.visual_norm is not None
    assert attention.visual_norm.weight.grad is not None
    assert not torch.equal(attention.norm.weight.grad, attention.visual_norm.weight.grad)
    assert attention.qkv_proj.weight.grad is not None
    assert attention.visual_qkv_proj is not None
    assert attention.visual_qkv_proj.weight.grad is not None
    assert not torch.equal(attention.qkv_proj.weight.grad, attention.visual_qkv_proj.weight.grad)


def test_attention_weight_tracing_supports_specialized_paths() -> None:
    model = ViT(_specialized_config()).eval()
    images = torch.randn(2, 3, 8, 8)

    weights = model.forward_attention_weights(images)

    assert tuple(weights) == tuple(f"layer_{index}" for index in range(DEPTH))
    assert weights["layer_0"].shape[:3] == (2, NUM_HEADS, NUM_GLOBAL_TOKENS + 4)


@pytest.mark.compile
@pytest.mark.cuda
def test_specialized_attention_backward_compiles_for_masked_sequence() -> None:
    environment = os.environ.copy()
    environment.pop("TORCHDYNAMO_DISABLE", None)
    check_script = Path(__file__).with_name("token_specialized_attention_compile_check.py")

    subprocess.run(
        [sys.executable, str(check_script)],
        check=True,
        env=environment,
        timeout=COMPILE_TEST_TIMEOUT_SECONDS,
    )
