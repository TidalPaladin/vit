import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from vit import ViT, ViTConfig
from vit.explain import Explanation, ForwardArgs, ViTExplainer, load_explanation, save_explanation
from vit.explain.cli import main


def make_tiny_config() -> ViTConfig:
    return ViTConfig(
        in_channels=3,
        patch_size=(4, 4),
        img_size=(9, 10),
        depth=1,
        hidden_size=16,
        ffn_hidden_size=32,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="rope",
        dtype=torch.float32,
    )


def output_fn(features):
    return features.visual_tokens.mean(1)[:, :3]


def make_explanation() -> Explanation:
    model = ViT(make_tiny_config()).eval()
    inputs = torch.randn(1, 3, 9, 10)
    trace = ViTExplainer(model, output_fn).trace(inputs, forward_args=ForwardArgs(rope_seed=2))
    return Explanation(
        method="fixed",
        token_attributions=torch.arange(4, dtype=torch.float32).view(1, 4),
        pixel_attributions=None,
        target_scores=torch.tensor([1.25]),
        layout=trace.layout,
        layer_attributions=(torch.ones(1, 4),),
        configuration={"signed": True, "layers": [0]},
    )


def test_artifact_round_trip_is_non_pickle_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "explanation.npz"
    explanation = make_explanation()

    save_explanation(explanation, path)
    first_metadata = path.with_suffix(".json").read_bytes()
    loaded = load_explanation(path)
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays.files

    assert loaded.method == explanation.method
    assert torch.equal(loaded.token_attributions, explanation.token_attributions)
    assert loaded.configuration == explanation.configuration
    save_explanation(explanation, path, overwrite=True)
    assert path.with_suffix(".json").read_bytes() == first_metadata


def test_artifact_round_trip_preserves_bfloat16_tensors(tmp_path: Path) -> None:
    path = tmp_path / "bfloat16.npz"
    explanation = make_explanation()
    pixel_attributions = torch.randn(1, 3, 9, 10, dtype=torch.bfloat16)
    bfloat16_explanation = replace(
        explanation,
        token_attributions=explanation.token_attributions.to(torch.bfloat16),
        pixel_attributions=pixel_attributions,
        target_scores=explanation.target_scores.to(torch.bfloat16),
        layer_attributions=tuple(value.to(torch.bfloat16) for value in explanation.layer_attributions),
    )

    save_explanation(bfloat16_explanation, path)
    loaded = load_explanation(path)

    assert loaded.token_attributions.dtype == torch.bfloat16
    assert loaded.pixel_attributions is not None
    assert loaded.pixel_attributions.dtype == torch.bfloat16
    assert loaded.target_scores.dtype == torch.bfloat16
    assert loaded.layer_attributions[0].dtype == torch.bfloat16
    assert torch.equal(loaded.token_attributions, bfloat16_explanation.token_attributions)
    assert torch.equal(loaded.pixel_attributions, pixel_attributions)


def test_artifact_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "explanation.npz"
    save_explanation(make_explanation(), path)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_explanation(make_explanation(), path)


def test_cli_inspect_text_and_json_use_stdout(tmp_path: Path, capsys) -> None:
    path = tmp_path / "explanation.npz"
    save_explanation(make_explanation(), path)

    assert main(["--format", "json", "inspect", str(path)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["method"] == "fixed"
    assert captured.err == ""


def test_cli_runtime_failure_uses_stderr_and_exit_two(tmp_path: Path, capsys) -> None:
    assert main(["inspect", str(tmp_path / "missing.npz")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "vit-explain failed" in captured.err


def test_cli_compare_json(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_explanation(make_explanation(), first)
    save_explanation(make_explanation(), second)

    assert main(["--format", "json", "compare", str(first), str(second)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cosine_similarity"] == pytest.approx(1.0)


def test_cli_compare_rejects_equal_shapes_with_incompatible_layouts(tmp_path: Path, capsys) -> None:
    first_explanation = make_explanation()
    second_explanation = replace(
        first_explanation,
        layout=replace(first_explanation.layout, grid_size=(1, 4), patch_size=(8, 2)),
    )
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_explanation(first_explanation, first)
    save_explanation(second_explanation, second)

    assert main(["compare", str(first), str(second)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "different token layouts" in captured.err


def test_cli_help_does_not_import_optional_dependencies() -> None:
    blocker = """
import importlib.abc, importlib.util, sys
class Blocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'captum', 'matplotlib', 'PIL'}:
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec): return None
    def exec_module(self, module): raise ModuleNotFoundError(module.__name__)
sys.meta_path.insert(0, Blocker())
from vit.explain.cli import main
raise SystemExit(main(['--help']))
"""
    result = subprocess.run([sys.executable, "-c", blocker], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_cli_render_writes_image_and_enforces_overwrite(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "explanation.npz"
    output = tmp_path / "render.png"
    save_explanation(make_explanation(), artifact)

    assert main(["render", str(artifact), str(output), "--normalization", "symmetric"]) == 0
    assert output.is_file()
    assert "output" in capsys.readouterr().out
    assert main(["render", str(artifact), str(output)]) == 2
    assert "overwrite" in capsys.readouterr().err
    assert main(["--quiet", "render", str(artifact), str(output), "--overwrite"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_render_supports_bfloat16_artifact(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "bfloat16.npz"
    output = tmp_path / "bfloat16.png"
    explanation = make_explanation()
    save_explanation(
        replace(
            explanation,
            token_attributions=explanation.token_attributions.to(torch.bfloat16),
            target_scores=explanation.target_scores.to(torch.bfloat16),
            layer_attributions=tuple(value.to(torch.bfloat16) for value in explanation.layer_attributions),
        ),
        artifact,
    )

    assert main(["render", str(artifact), str(output)]) == 0
    assert output.is_file()
    assert "output" in capsys.readouterr().out


def test_cli_verbose_text_compare_and_shape_mismatch(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_explanation(make_explanation(), first)
    save_explanation(make_explanation(), second)

    assert main(["--verbose", "compare", str(first), str(second)]) == 0
    captured = capsys.readouterr()
    assert "cosine_similarity" in captured.out
    assert "completed compare" in captured.err


def test_artifact_validation_rejects_invalid_paths_and_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".npz"):
        save_explanation(make_explanation(), tmp_path / "bad.bin")

    path = tmp_path / "bad.npz"
    np.savez(path, token_attributions=np.ones(1))
    path.with_suffix(".json").write_text('{"artifact_version": 99}')
    with pytest.raises(ValueError, match="unsupported"):
        load_explanation(path)


def test_artifact_metadata_rejects_non_json_configuration(tmp_path: Path) -> None:
    explanation = make_explanation()
    invalid = Explanation(
        explanation.method,
        explanation.token_attributions,
        explanation.pixel_attributions,
        explanation.target_scores,
        explanation.layout,
        configuration={"invalid": object()},
    )
    path = tmp_path / "invalid.npz"
    with pytest.raises(TypeError, match="metadata"):
        save_explanation(invalid, path)
    assert not path.exists()
    assert not path.with_suffix(".json").exists()


def test_artifact_save_rejects_attributions_incompatible_with_layout(tmp_path: Path) -> None:
    explanation = make_explanation()
    invalid = replace(explanation, token_attributions=explanation.token_attributions[:, :-1])
    path = tmp_path / "invalid-shape.npz"

    with pytest.raises(ValueError, match="token_attributions"):
        save_explanation(invalid, path)

    assert not path.exists()
    assert not path.with_suffix(".json").exists()


def test_cli_rejects_malformed_artifact_shapes(tmp_path: Path, capsys) -> None:
    path = tmp_path / "malformed.npz"
    save_explanation(make_explanation(), path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays["token_attributions"] = arrays["token_attributions"][:, :-1]
    np.savez_compressed(path, **arrays)

    assert main(["inspect", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "malformed artifact" in captured.err


def test_cli_rejects_corrupt_npz_archive(tmp_path: Path, capsys) -> None:
    path = tmp_path / "corrupt.npz"
    save_explanation(make_explanation(), path)
    path.write_bytes(b"PK\x03\x04not-a-valid-archive")

    assert main(["inspect", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "vit-explain failed" in captured.err


def test_failed_artifact_overwrite_preserves_existing_pair(tmp_path: Path) -> None:
    path = tmp_path / "explanation.npz"
    explanation = make_explanation()
    save_explanation(explanation, path)
    original_arrays = path.read_bytes()
    original_metadata = path.with_suffix(".json").read_bytes()
    invalid = Explanation(
        explanation.method,
        explanation.token_attributions + 1,
        explanation.pixel_attributions,
        explanation.target_scores,
        explanation.layout,
        configuration={"invalid": object()},
    )

    with pytest.raises(TypeError, match="metadata"):
        save_explanation(invalid, path, overwrite=True)

    assert path.read_bytes() == original_arrays
    assert path.with_suffix(".json").read_bytes() == original_metadata


def test_cli_applies_color_and_progress_policies(tmp_path: Path, capsys) -> None:
    path = tmp_path / "explanation.npz"
    save_explanation(make_explanation(), path)

    assert main(["--color", "always", "--progress", "always", "inspect", str(path)]) == 0
    captured = capsys.readouterr()
    assert "\x1b[" in captured.out
    assert "vit-explain: inspect" in captured.err

    assert main(["--format", "json", "--color", "always", "--progress", "never", "inspect", str(path)]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "\x1b[" not in captured.out
    assert captured.err == ""

    assert main(["--no-color", "--color", "always", "inspect", str(path)]) == 0
    assert "\x1b[" not in capsys.readouterr().out
