"""Safe, deterministic-metadata explanation artifacts."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from zipfile import BadZipFile

import numpy as np
import torch

from .types import Explanation, TokenLayout


ARTIFACT_VERSION = 1
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"artifact metadata does not support {type(value).__name__}")


def _temporary_path(directory: Path, prefix: str) -> Path:
    with NamedTemporaryFile(dir=directory, prefix=prefix, suffix=".tmp", delete=False) as temporary_file:
        return Path(temporary_file.name)


def _replace_artifact_pair(temporary_paths: tuple[Path, Path], destination_paths: tuple[Path, Path]) -> None:
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination in destination_paths:
            if destination.exists():
                backup = _temporary_path(destination.parent, f".{destination.name}.backup-")
                backup.unlink()
                destination.replace(backup)
                backups[destination] = backup
        for temporary, destination in zip(temporary_paths, destination_paths, strict=True):
            temporary.replace(destination)
            installed.append(destination)
    except Exception:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _validate_explanation(explanation: Explanation, *, loaded: bool = False) -> None:
    prefix = "malformed artifact: " if loaded else ""

    def fail(message: str) -> None:
        raise ValueError(prefix + message)

    layout = explanation.layout
    if any(value <= 0 for value in (*layout.grid_size, *layout.patch_size, *layout.original_size)):
        fail("grid_size, patch_size, and original_size must be positive")
    expected_modeled_size = (
        layout.grid_size[0] * layout.patch_size[0],
        layout.grid_size[1] * layout.patch_size[1],
    )
    if layout.modeled_size != expected_modeled_size:
        fail(f"modeled_size must equal grid_size * patch_size, got {layout.modeled_size}")
    if any(modeled > original for modeled, original in zip(layout.modeled_size, layout.original_size, strict=True)):
        fail("modeled_size cannot exceed original_size")
    if layout.num_cls_tokens < 0 or layout.num_register_tokens < 0:
        fail("prefix token counts must be nonnegative")

    token_count = layout.visual_token_count
    token_attributions = explanation.token_attributions
    if token_attributions.ndim != 2 or token_attributions.shape[1] != token_count:
        fail(f"token_attributions must have shape (batch, {token_count})")
    if not token_attributions.is_floating_point():
        fail("token_attributions must use a floating-point dtype")
    batch_size = token_attributions.shape[0]
    if batch_size <= 0:
        fail("explanation batch size must be positive")
    if explanation.target_scores.shape != (batch_size,):
        fail(f"target_scores must have shape ({batch_size},)")
    if not explanation.target_scores.is_floating_point():
        fail("target_scores must use a floating-point dtype")

    visual_validity = layout.visual_validity
    visual_indices = layout.visual_indices
    if visual_validity.dtype != torch.bool or visual_validity.shape != (batch_size, token_count):
        fail(f"visual_validity must be bool with shape ({batch_size}, {token_count})")
    if visual_indices.dtype not in _INTEGER_DTYPES or visual_indices.ndim != 2 or visual_indices.shape[0] != batch_size:
        fail("visual_indices must be a two-dimensional integer tensor with the explanation batch size")
    maximum_sequence_length = int(visual_validity.sum(dim=1).max().item())
    if visual_indices.shape[1] != maximum_sequence_length:
        fail("visual_indices length must equal the largest valid visual-token count")
    for batch_index in range(batch_size):
        expected_indices = torch.nonzero(visual_validity[batch_index], as_tuple=False).flatten().cpu()
        padding = torch.full(
            (maximum_sequence_length - expected_indices.numel(),),
            -1,
            dtype=expected_indices.dtype,
        )
        expected_row = torch.cat((expected_indices, padding))
        if not torch.equal(visual_indices[batch_index].cpu().to(expected_row.dtype), expected_row):
            fail("visual_indices do not match visual_validity")

    if explanation.pixel_attributions is not None:
        pixel_attributions = explanation.pixel_attributions
        expected_pixel_shape = (batch_size, *layout.original_size)
        if (
            pixel_attributions.ndim != 4
            or pixel_attributions.shape[0] != batch_size
            or pixel_attributions.shape[-2:] != layout.original_size
        ):
            fail(
                f"pixel_attributions must have shape (batch, channels, {expected_pixel_shape[-2]}, {expected_pixel_shape[-1]})"
            )
        if not pixel_attributions.is_floating_point():
            fail("pixel_attributions must use a floating-point dtype")
    for index, attribution in enumerate(explanation.layer_attributions):
        if attribution.shape != token_attributions.shape or not attribution.is_floating_point():
            fail(f"layer_attributions[{index}] must match token_attributions shape and dtype kind")


def _artifact_arrays(explanation: Explanation) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    tensors = {
        "target_scores": explanation.target_scores,
        "token_attributions": explanation.token_attributions,
        "visual_indices": explanation.layout.visual_indices,
        "visual_validity": explanation.layout.visual_validity,
    }
    if explanation.pixel_attributions is not None:
        tensors["pixel_attributions"] = explanation.pixel_attributions
    tensors.update(
        {f"layer_attributions_{index:04d}": value for index, value in enumerate(explanation.layer_attributions)}
    )
    arrays: dict[str, np.ndarray] = {}
    stored_dtypes: dict[str, str] = {}
    for name, tensor in tensors.items():
        stored = tensor.detach().cpu()
        if stored.dtype == torch.bfloat16:
            stored_dtypes[name] = str(stored.dtype)
            stored = stored.float()
        try:
            arrays[name] = stored.numpy()
        except TypeError as error:
            raise TypeError(f"artifact array {name} uses unsupported dtype {tensor.dtype}") from error
    return arrays, stored_dtypes


def save_explanation(explanation: Explanation, path: str | Path, *, overwrite: bool = False) -> None:
    """Write numeric arrays to NPZ and deterministic, non-pickle metadata to JSON."""
    array_path = Path(path)
    if array_path.suffix != ".npz":
        raise ValueError("explanation artifact path must end in .npz")
    metadata_path = _metadata_path(array_path)
    existing = [candidate for candidate in (array_path, metadata_path) if candidate.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"artifact exists; pass overwrite=True to replace {existing[0]}")
    _validate_explanation(explanation)
    arrays, stored_dtypes = _artifact_arrays(explanation)
    metadata = {
        "array_dtypes": stored_dtypes,
        "artifact_version": ARTIFACT_VERSION,
        "configuration": _json_value(dict(explanation.configuration)),
        "kind": "vit.explain.Explanation",
        "layer_count": len(explanation.layer_attributions),
        "layout": {
            "grid_size": list(explanation.layout.grid_size),
            "modeled_size": list(explanation.layout.modeled_size),
            "num_cls_tokens": explanation.layout.num_cls_tokens,
            "num_register_tokens": explanation.layout.num_register_tokens,
            "original_size": list(explanation.layout.original_size),
            "patch_size": list(explanation.layout.patch_size),
        },
        "method": explanation.method,
    }
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_array = _temporary_path(array_path.parent, f".{array_path.name}.")
    temporary_metadata = _temporary_path(metadata_path.parent, f".{metadata_path.name}.")
    try:
        with temporary_array.open("wb") as artifact_file:
            save_arrays = cast(Any, np.savez_compressed)
            save_arrays(artifact_file, **dict(sorted(arrays.items())))
        temporary_metadata.write_text(metadata_text)
        _replace_artifact_pair(
            (temporary_array, temporary_metadata),
            (array_path, metadata_path),
        )
    finally:
        temporary_array.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def _pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(type(item) is int for item in value):
        raise ValueError(f"malformed artifact: {name} must contain two integers")
    return int(value[0]), int(value[1])


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"malformed artifact: {name} must be a nonnegative integer")
    return value


def load_explanation(path: str | Path) -> Explanation:
    """Load an explanation with NumPy pickle loading explicitly disabled."""
    array_path = Path(path)
    metadata_path = _metadata_path(array_path)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read artifact metadata {metadata_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("unsupported or malformed explanation artifact metadata")
    if metadata.get("artifact_version") != ARTIFACT_VERSION or metadata.get("kind") != "vit.explain.Explanation":
        raise ValueError("unsupported or malformed explanation artifact metadata")
    try:
        with np.load(array_path, allow_pickle=False) as stored:
            arrays = {name: torch.from_numpy(stored[name].copy()) for name in stored.files}
    except (OSError, TypeError, ValueError, KeyError, BadZipFile) as error:
        raise ValueError(f"could not read artifact arrays {array_path}: {error}") from error
    required = {"target_scores", "token_attributions", "visual_indices", "visual_validity"}
    if not required <= arrays.keys():
        raise ValueError(f"malformed artifact: missing arrays {sorted(required - arrays.keys())}")
    layer_count = metadata.get("layer_count", 0)
    layer_count = _nonnegative_int(layer_count, "layer_count")
    expected_arrays = required | {f"layer_attributions_{index:04d}" for index in range(layer_count)}
    if "pixel_attributions" in arrays:
        expected_arrays.add("pixel_attributions")
    if unexpected := arrays.keys() - expected_arrays:
        raise ValueError(f"malformed artifact: unexpected arrays {sorted(unexpected)}")

    stored_dtypes = metadata.get("array_dtypes", {})
    if not isinstance(stored_dtypes, dict) or not all(
        isinstance(name, str) and isinstance(dtype, str) for name, dtype in stored_dtypes.items()
    ):
        raise ValueError("malformed artifact: array_dtypes must map array names to dtype strings")
    for name, dtype in stored_dtypes.items():
        if name not in arrays or dtype != str(torch.bfloat16) or arrays[name].dtype != torch.float32:
            raise ValueError(f"malformed artifact: invalid stored dtype for {name}")
        arrays[name] = arrays[name].to(torch.bfloat16)

    layout_metadata = metadata.get("layout")
    if not isinstance(layout_metadata, dict):
        raise ValueError("malformed artifact: layout metadata is missing")
    layout = TokenLayout(
        grid_size=_pair(layout_metadata.get("grid_size"), "grid_size"),
        patch_size=_pair(layout_metadata.get("patch_size"), "patch_size"),
        original_size=_pair(layout_metadata.get("original_size"), "original_size"),
        modeled_size=_pair(layout_metadata.get("modeled_size"), "modeled_size"),
        num_cls_tokens=_nonnegative_int(layout_metadata.get("num_cls_tokens", 0), "num_cls_tokens"),
        num_register_tokens=_nonnegative_int(
            layout_metadata.get("num_register_tokens", 0),
            "num_register_tokens",
        ),
        visual_indices=arrays["visual_indices"],
        visual_validity=arrays["visual_validity"],
    )
    try:
        layer_attributions = tuple(arrays[f"layer_attributions_{index:04d}"] for index in range(layer_count))
    except KeyError as error:
        raise ValueError(f"malformed artifact: missing {error.args[0]}") from error
    configuration = metadata.get("configuration", {})
    if not isinstance(configuration, dict) or not isinstance(metadata.get("method"), str):
        raise ValueError("malformed artifact: method or configuration is invalid")
    explanation = Explanation(
        method=metadata["method"],
        token_attributions=arrays["token_attributions"],
        pixel_attributions=arrays.get("pixel_attributions"),
        target_scores=arrays["target_scores"],
        layout=layout,
        layer_attributions=layer_attributions,
        configuration=configuration,
    )
    _validate_explanation(explanation, loaded=True)
    return explanation
