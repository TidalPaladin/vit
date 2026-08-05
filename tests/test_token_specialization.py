import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
import torch
from torch.testing import assert_close
from torchao.quantization import Int8Tensor, Int8WeightOnlyConfig

from vit import TokenSpecializedAttentionCompileMode, ViT, ViTConfig, ViTFeatures
from vit.attention import (
    _STATIC_TOKEN_SPECIALIZED_ATTENTION_MIN_BATCH_SIZE,
    _attention_token_specialized_qkv_packed_impl,
    _dynamic_token_specialized_attention,
    _select_token_specialized_attention,
    _static_max_autotune_token_specialized_attention,
    _static_token_specialized_attention,
    attention_token_specialized_qkv_packed,
)
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
COMPILE_TEST_TIMEOUT_SECONDS = 600
TORCHAO_QUANTIZATION_CONFIG_VERSION = 2
LEGACY_DROPOUT = 0.1
LAYER_SCALE_INIT = 1e-5
LEGACY_POSITION_ENCODING = "fourier"
MODEL_INITIALIZATION_SEED = 7
RESIDUAL_INITIALIZATION_SEED = 11
CUSTOM_STATIC_BATCH_SIZES = (2, 4)
COMPILE_MODES = ("auto", "dynamic", "static", "static_max_autotune")


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
        "layer_scale": LAYER_SCALE_INIT,
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


@torch.no_grad()
def _seed_residual_outputs(model: ViT) -> None:
    for block in model.blocks:
        assert isinstance(block, TransformerEncoderLayer)
        torch.nn.init.normal_(block.self_attention.out_proj.weight, std=TEST_PROJECTION_STD)
        torch.nn.init.normal_(block.mlp.fc2.weight, std=TEST_PROJECTION_STD)


def test_specialization_is_disabled_by_default() -> None:
    config = _config()
    model = ViT(config)

    assert config.specialize_global_token_norms is False
    assert config.specialize_global_token_qkv_blocks == 0
    assert config.token_specialization_enabled is False
    assert config.token_specialized_attention_compile_mode == "auto"
    assert config.token_specialized_attention_static_batch_sizes is None
    assert config.num_global_tokens == NUM_GLOBAL_TOKENS
    assert all("visual_" not in name for name in model.state_dict())


def test_compile_mode_type_is_exported_from_public_api() -> None:
    assert get_args(TokenSpecializedAttentionCompileMode) == COMPILE_MODES


@pytest.mark.parametrize(
    ("compile_mode", "static_batch_sizes"),
    [
        ("dynamic", None),
        ("static", None),
        ("static_max_autotune", None),
        ("auto", CUSTOM_STATIC_BATCH_SIZES),
    ],
)
def test_compile_customization_requires_token_specialization(
    compile_mode: str,
    static_batch_sizes: tuple[int, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="requires token specialization"):
        _config(
            token_specialized_attention_compile_mode=compile_mode,
            token_specialized_attention_static_batch_sizes=static_batch_sizes,
        )


def test_compile_mode_must_be_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported token-specialized attention compile mode"):
        _config(
            specialize_global_token_norms=True,
            token_specialized_attention_compile_mode="unsupported",
        )


@pytest.mark.parametrize(
    "static_batch_sizes",
    [(), (0,), (-1,), (2, 2)],
    ids=["empty", "zero", "negative", "duplicate"],
)
def test_static_batch_sizes_must_be_nonempty_positive_and_unique(static_batch_sizes: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="static batch sizes"):
        _config(
            specialize_global_token_norms=True,
            token_specialized_attention_static_batch_sizes=static_batch_sizes,
        )


def test_dynamic_mode_rejects_static_batch_sizes() -> None:
    with pytest.raises(ValueError, match="cannot be used with dynamic"):
        _config(
            specialize_global_token_norms=True,
            token_specialized_attention_compile_mode="dynamic",
            token_specialized_attention_static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
        )


def test_default_config_preserves_encoder_factory_call_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    original_factory = ViT.create_encoder_layer
    factory_calls = 0

    def legacy_factory(
        self: ViT,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TransformerEncoderLayer:
        nonlocal factory_calls
        factory_calls += 1
        return original_factory(
            self,
            mlp_quantization_config,
            qkv_quantization_config,
            attn_quantization_config,
            device,
            dtype,
        )

    monkeypatch.setattr(ViT, "create_encoder_layer", legacy_factory)

    model = ViT(_config())

    assert len(model.blocks) == DEPTH
    assert factory_calls == DEPTH


def test_new_config_options_do_not_shift_legacy_positional_arguments() -> None:
    config = ViTConfig(
        3,
        (4, 4),
        (8, 8),
        DEPTH,
        HIDDEN_SIZE,
        HIDDEN_SIZE * 2,
        NUM_HEADS,
        LEGACY_DROPOUT,
        LEGACY_DROPOUT,
        True,
        True,
        "srelu",
        0.0,
        NUM_REGISTER_TOKENS,
        NUM_CLS_TOKENS,
        LEGACY_POSITION_ENCODING,
    )

    assert config.pos_enc == LEGACY_POSITION_ENCODING
    assert config.specialize_global_token_norms is False
    assert config.specialize_global_token_qkv_blocks == 0


def test_layer_scale_preserves_legacy_positional_factory_arguments() -> None:
    layer_scale = LayerScale(
        HIDDEN_SIZE,
        LAYER_SCALE_INIT,
        False,
        torch.device("cpu"),
        torch.float64,
    )

    assert layer_scale.gamma.dtype == torch.float64
    assert layer_scale.visual_gamma is None


@pytest.mark.parametrize(
    "specialization",
    [
        {"specialize_global_token_norms": True},
        {"specialize_global_token_qkv_blocks": 1},
    ],
)
def test_specialization_requires_global_tokens(specialization: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="at least one CLS or register token"):
        _config(
            num_cls_tokens=0,
            num_register_tokens=0,
            **specialization,
        )


@pytest.mark.parametrize("qkv_blocks", [-1, DEPTH + 1])
def test_qkv_specialization_depth_must_fit_backbone(qkv_blocks: int) -> None:
    with pytest.raises(ValueError, match="must be non-negative|cannot exceed depth"):
        replace(_specialized_config(), specialize_global_token_qkv_blocks=qkv_blocks)


def test_specialization_config_round_trips_through_yaml() -> None:
    configured = replace(
        _specialized_config(),
        token_specialized_attention_compile_mode="static",
        token_specialized_attention_static_batch_sizes=list(CUSTOM_STATIC_BATCH_SIZES),  # type: ignore[arg-type]
    )

    restored = ViTConfig.from_yaml(configured.to_yaml())

    assert restored.specialize_global_token_norms is True
    assert restored.specialize_global_token_qkv_blocks == QKV_SPECIALIZATION_BLOCKS
    assert restored.token_specialization_enabled is True
    assert restored.token_specialized_attention_compile_mode == "static"
    assert restored.token_specialized_attention_static_batch_sizes == CUSTOM_STATIC_BATCH_SIZES


def test_norm_and_qkv_specialization_can_be_configured_independently() -> None:
    norms_only = ViT(_config(specialize_global_token_norms=True))
    qkv_only = ViT(_config(specialize_global_token_qkv_blocks=QKV_SPECIALIZATION_BLOCKS))

    for block in norms_only.blocks:
        assert isinstance(block, TransformerEncoderLayer)
        assert block.self_attention.visual_norm is not None
        assert block.self_attention.visual_qkv_proj is None
        assert block.mlp.visual_norm is not None
    for block_index, block in enumerate(qkv_only.blocks):
        assert isinstance(block, TransformerEncoderLayer)
        assert block.self_attention.visual_norm is None
        assert (block.self_attention.visual_qkv_proj is not None) is (block_index < QKV_SPECIALIZATION_BLOCKS)
        assert block.mlp.visual_norm is None


def test_compile_policy_is_forwarded_to_specialized_encoder_attention() -> None:
    config = replace(
        _specialized_config(),
        token_specialized_attention_compile_mode="static",
        token_specialized_attention_static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
    )

    model = ViT(config)
    factory_layer = model.create_encoder_layer()

    for block in [*model.blocks, factory_layer]:
        assert isinstance(block, TransformerEncoderLayer)
        assert block.self_attention.token_specialized_attention_compile_mode == "static"
        assert block.self_attention.token_specialized_attention_static_batch_sizes == CUSTOM_STATIC_BATCH_SIZES


def test_compile_policy_only_applies_to_blocks_with_specialized_attention() -> None:
    config = _config(
        specialize_global_token_qkv_blocks=QKV_SPECIALIZATION_BLOCKS,
        token_specialized_attention_compile_mode="static",
        token_specialized_attention_static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
    )

    model = ViT(config)

    first_attention = model.get_block(0).self_attention
    assert first_attention.token_specialized_attention_compile_mode == "static"
    assert first_attention.token_specialized_attention_static_batch_sizes == CUSTOM_STATIC_BATCH_SIZES
    for block_index in range(QKV_SPECIALIZATION_BLOCKS, DEPTH):
        shared_attention = model.get_block(block_index).self_attention
        assert shared_attention.token_specialized_attention_compile_mode == "auto"
        assert shared_attention.token_specialized_attention_static_batch_sizes is None


def test_norm_specialization_rejects_conditioned_mlp() -> None:
    with pytest.raises(ValueError, match="incompatible with conditioned MLPs"):
        _config(specialize_global_token_norms=True, conditioning_size=HIDDEN_SIZE)


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


@pytest.mark.parametrize(
    "specialization",
    [
        {"specialize_global_token_norms": True},
        {"specialize_global_token_qkv_blocks": QKV_SPECIALIZATION_BLOCKS},
        {
            "specialize_global_token_norms": True,
            "specialize_global_token_qkv_blocks": QKV_SPECIALIZATION_BLOCKS,
        },
    ],
    ids=["norms", "qkv", "norms-and-qkv"],
)
@pytest.mark.parametrize("activation", ["srelu", "swiglu"])
@pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
def test_specialization_is_functionally_identical_at_initialization(
    specialization: dict[str, object],
    activation: str,
    norm_type: str,
) -> None:
    baseline_config = _config(activation=activation, norm_type=norm_type)
    specialized_config = replace(baseline_config, **specialization)
    torch.manual_seed(MODEL_INITIALIZATION_SEED)
    baseline = ViT(baseline_config).eval()
    torch.manual_seed(MODEL_INITIALIZATION_SEED)
    specialized = ViT(specialized_config).eval()
    torch.manual_seed(RESIDUAL_INITIALIZATION_SEED)
    _seed_residual_outputs(baseline)
    torch.manual_seed(RESIDUAL_INITIALIZATION_SEED)
    _seed_residual_outputs(specialized)
    images = torch.randn(2, 3, 8, 8)

    baseline_features = baseline(images)
    specialized_features = specialized(images)

    assert_close(specialized_features.dense_features, baseline_features.dense_features)


def test_specialization_preserves_forward_output_contract_with_masking() -> None:
    model = ViT(_specialized_config()).eval()
    images = torch.randn(2, 3, 8, 8)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    features = model(images, mask=mask)

    assert isinstance(features, ViTFeatures)
    assert features.cls_tokens.shape == (2, NUM_CLS_TOKENS, HIDDEN_SIZE)
    assert features.register_tokens.shape == (2, NUM_REGISTER_TOKENS, HIDDEN_SIZE)
    assert features.visual_tokens.shape == (2, 2, HIDDEN_SIZE)
    assert features.tokenized_size == (2, 2)


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


def test_specialized_paths_support_activation_checkpointing() -> None:
    model = ViT(replace(_specialized_config(), activation_checkpointing=True)).train()
    _seed_residual_outputs(model)
    images = torch.randn(2, 3, 8, 8, requires_grad=True)

    model(images).dense_features.square().mean().backward()

    first_block = model.get_block(0)
    assert first_block.self_attention.visual_norm is not None
    assert first_block.self_attention.visual_norm.weight.grad is not None
    assert first_block.self_attention.visual_qkv_proj is not None
    assert first_block.self_attention.visual_qkv_proj.weight.grad is not None
    assert first_block.mlp.visual_norm is not None
    assert first_block.mlp.visual_norm.weight.grad is not None


def test_qkv_specialization_quantizes_both_projection_paths() -> None:
    quantization = Int8WeightOnlyConfig(version=TORCHAO_QUANTIZATION_CONFIG_VERSION)
    model = ViT(_specialized_config(), qkv_quantization_config=quantization).eval()
    first_attention = model.get_block(0).self_attention

    assert isinstance(first_attention.qkv_proj.weight, Int8Tensor)
    assert first_attention.visual_qkv_proj is not None
    assert isinstance(first_attention.visual_qkv_proj.weight, Int8Tensor)
    assert model(torch.randn(2, 3, 8, 8)).dense_features.shape == (2, NUM_GLOBAL_TOKENS + 4, HIDDEN_SIZE)


def test_attention_weight_tracing_supports_specialized_paths() -> None:
    model = ViT(_specialized_config()).eval()
    images = torch.randn(2, 3, 8, 8)

    weights = model.forward_attention_weights(images)

    assert tuple(weights) == tuple(f"layer_{index}" for index in range(DEPTH))
    assert weights["layer_0"].shape[:3] == (2, NUM_HEADS, NUM_GLOBAL_TOKENS + 4)


def test_auto_mode_uses_isolated_static_attention_for_large_training_batches() -> None:
    features = torch.empty(_STATIC_TOKEN_SPECIALIZED_ATTENTION_MIN_BATCH_SIZE, 1, HIDDEN_SIZE)

    static_attention = _select_token_specialized_attention(features, training=True)

    assert static_attention is not attention_token_specialized_qkv_packed
    assert _select_token_specialized_attention(features[:-1], training=True) is attention_token_specialized_qkv_packed
    assert _select_token_specialized_attention(features, training=False) is attention_token_specialized_qkv_packed
    with torch.no_grad():
        assert _select_token_specialized_attention(features, training=True) is attention_token_specialized_qkv_packed


def test_auto_mode_static_batch_allowlist_replaces_default_threshold() -> None:
    features = torch.empty(CUSTOM_STATIC_BATCH_SIZES[0], 1, HIDDEN_SIZE)

    assert (
        _select_token_specialized_attention(
            features,
            training=True,
            compile_mode="auto",
            static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
        )
        is _static_token_specialized_attention
    )
    assert (
        _select_token_specialized_attention(
            features[:1],
            training=True,
            compile_mode="auto",
            static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
        )
        is attention_token_specialized_qkv_packed
    )
    with torch.no_grad():
        assert (
            _select_token_specialized_attention(
                features,
                training=True,
                compile_mode="auto",
                static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
            )
            is attention_token_specialized_qkv_packed
        )


def test_forced_compile_modes_select_isolated_helpers_on_cpu() -> None:
    features = torch.empty(CUSTOM_STATIC_BATCH_SIZES[0], 1, HIDDEN_SIZE)

    assert (
        _select_token_specialized_attention(features, training=False, compile_mode="dynamic")
        is _dynamic_token_specialized_attention
    )
    assert (
        _select_token_specialized_attention(features, training=False, compile_mode="static")
        is _static_token_specialized_attention
    )
    assert (
        _select_token_specialized_attention(features, training=False, compile_mode="static_max_autotune")
        is _static_token_specialized_attention
    )
    assert len(
        {
            id(attention_token_specialized_qkv_packed),
            id(_dynamic_token_specialized_attention),
            id(_static_token_specialized_attention),
            id(_static_max_autotune_token_specialized_attention),
        }
    ) == len(COMPILE_MODES)


def test_forced_static_mode_rejects_unlisted_batch() -> None:
    features = torch.empty(3, 1, HIDDEN_SIZE)

    with pytest.raises(ValueError, match="batch size 3.*allowed static batch sizes.*2, 4"):
        _select_token_specialized_attention(
            features,
            training=False,
            compile_mode="static",
            static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
        )


def test_static_batch_allowlist_supports_stochastic_depth_subset_size() -> None:
    input_batch_size = 8
    drop_path_rate = 0.25
    residual_batch_size = int(input_batch_size * (1.0 - drop_path_rate))
    layer = TransformerEncoderLayer(
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=HIDDEN_SIZE * 2,
        num_attention_heads=NUM_HEADS,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=drop_path_rate,
        num_global_tokens=NUM_GLOBAL_TOKENS,
        specialize_global_token_norms=True,
        token_specialized_attention_compile_mode="static",
        token_specialized_attention_static_batch_sizes=(residual_batch_size,),
    ).train()
    features = torch.randn(input_batch_size, NUM_GLOBAL_TOKENS + 4, HIDDEN_SIZE)

    output = layer(features)

    assert output.shape == features.shape


def test_export_bypasses_runtime_compile_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
    features = torch.empty(3, 1, HIDDEN_SIZE)

    attention = _select_token_specialized_attention(
        features,
        training=False,
        compile_mode="static",
        static_batch_sizes=CUSTOM_STATIC_BATCH_SIZES,
    )

    assert attention is _attention_token_specialized_qkv_packed_impl


@pytest.mark.cuda
def test_static_max_autotune_uses_cuda_helper() -> None:
    features = torch.empty(CUSTOM_STATIC_BATCH_SIZES[0], 1, HIDDEN_SIZE, device="cuda")

    assert (
        _select_token_specialized_attention(features, training=False, compile_mode="static_max_autotune")
        is _static_max_autotune_token_specialized_attention
    )


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


@pytest.mark.compile
def test_specialized_attention_compiles_across_dynamic_sequence_lengths() -> None:
    environment = os.environ.copy()
    environment.pop("TORCHDYNAMO_DISABLE", None)
    check_script = Path(__file__).with_name("token_specialized_attention_dynamic_shapes_compile_check.py")

    subprocess.run(
        [sys.executable, str(check_script)],
        check=True,
        env=environment,
        timeout=COMPILE_TEST_TIMEOUT_SECONDS,
    )


@pytest.mark.compile
def test_specialized_attention_compile_modes_on_cpu() -> None:
    environment = os.environ.copy()
    environment.pop("TORCHDYNAMO_DISABLE", None)
    check_script = Path(__file__).with_name("token_specialized_attention_modes_compile_check.py")

    subprocess.run(
        [sys.executable, str(check_script)],
        check=True,
        env=environment,
        timeout=COMPILE_TEST_TIMEOUT_SECONDS,
    )
