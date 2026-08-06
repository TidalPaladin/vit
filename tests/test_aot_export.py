from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from vit import TokenSpecializedAttentionCompileMode, ViT, ViTConfig, ViTFeatures


COMPILE_MODES: tuple[TokenSpecializedAttentionCompileMode, ...] = (
    "auto",
    "dynamic",
    "static",
    "static_max_autotune",
)
EXAMPLE_BATCH_SIZE = 2
DYNAMIC_BATCH_MAX = 4


def _export_model(compile_mode: TokenSpecializedAttentionCompileMode) -> tuple[ViT, torch.Tensor]:
    static_batch_sizes = (EXAMPLE_BATCH_SIZE,) if compile_mode != "dynamic" else None
    config = ViTConfig(
        in_channels=3,
        patch_size=(4, 4),
        img_size=(8, 8),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_cls_tokens=1,
        num_register_tokens=1,
        specialize_global_token_norms=True,
        specialize_global_token_qkv_blocks=1,
        token_specialized_attention_compile_mode=compile_mode,
        token_specialized_attention_static_batch_sizes=static_batch_sizes,
        dtype=torch.float32,
    )
    return ViT(config).eval(), torch.randn(EXAMPLE_BATCH_SIZE, 3, 8, 8)


def _assert_features_match(actual: ViTFeatures, expected: ViTFeatures) -> None:
    assert isinstance(actual, ViTFeatures)
    assert actual.num_cls_tokens == expected.num_cls_tokens
    assert actual.num_register_tokens == expected.num_register_tokens
    assert actual.tokenized_size == expected.tokenized_size
    assert_close(actual.dense_features, expected.dense_features)


@pytest.mark.parametrize("compile_mode", COMPILE_MODES)
def test_direct_vit_export_save_and_load(
    compile_mode: TokenSpecializedAttentionCompileMode,
    tmp_path: Path,
) -> None:
    model, example = _export_model(compile_mode)
    batch = torch.export.Dim("batch", min=1, max=DYNAMIC_BATCH_MAX)

    exported = torch.export.export(model, (example,), dynamic_shapes=({0: batch},))
    exported_path = tmp_path / f"vit-{compile_mode}.pt2"
    torch.export.save(exported, exported_path)
    loaded = torch.export.load(exported_path).module()
    dynamic_example = torch.randn(EXAMPLE_BATCH_SIZE + 1, 3, 8, 8)
    reference_model, _ = _export_model("dynamic")
    reference_model.load_state_dict(model.state_dict())

    with torch.no_grad():
        expected = reference_model(dynamic_example)
        actual = loaded(dynamic_example)

    _assert_features_match(actual, expected)


@pytest.mark.compile
@pytest.mark.parametrize("compile_mode", COMPILE_MODES)
def test_aot_inductor_package_roundtrip(
    compile_mode: TokenSpecializedAttentionCompileMode,
    tmp_path: Path,
) -> None:
    model, example = _export_model(compile_mode)
    batch = torch.export.Dim("batch", min=1, max=DYNAMIC_BATCH_MAX)
    exported = torch.export.export(model, (example,), dynamic_shapes=({0: batch},))
    package_path = tmp_path / f"vit-{compile_mode}.pt2"

    torch._inductor.aoti_compile_and_package(exported, package_path=str(package_path))
    loaded = torch._inductor.aoti_load_package(str(package_path))
    dynamic_example = torch.randn(EXAMPLE_BATCH_SIZE + 1, 3, 8, 8)
    reference_model, _ = _export_model("dynamic")
    reference_model.load_state_dict(model.state_dict())

    with torch.no_grad():
        expected = reference_model(dynamic_example)
        actual = loaded(dynamic_example)

    _assert_features_match(actual, expected)
