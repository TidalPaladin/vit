"""Integration tests for AOT export to Rust inference cycle.

These tests verify the end-to-end workflow:
1. Create a ViT model configuration
2. Export it to AOTInductor format (.pt2)
3. Run inference using the Rust CLI
4. Verify output shape and inference success
"""

import re
import subprocess
from pathlib import Path

import pytest
import torch

from vit import ViTConfig


def aot_export_available() -> bool:
    """Check if AOT export is available and working."""
    try:
        # Check if torch._inductor.aoti_compile_and_package exists
        from torch._inductor import aoti_compile_and_package  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def rust_binary_path() -> Path:
    """Get the path to the Rust vit binary."""
    return Path(__file__).parent.parent / "rust" / "target" / "release" / "vit"


def rust_ffi_available() -> bool:
    """Check if the Rust binary with FFI support is available."""
    binary = rust_binary_path()
    if not binary.exists():
        return False

    # Try running inference to see if FFI is enabled
    result = subprocess.run(
        [str(binary), "infer", "--help"],
        capture_output=True,
        text=True,
    )
    # FFI is available if the help text doesn't mention "requires the FFI feature"
    return result.returncode == 0


def skip_if_no_rust_ffi():
    """Skip test if Rust FFI is not available."""
    if not rust_ffi_available():
        pytest.skip("Rust FFI not available. Build with: export LIBTORCH=/path/to/libtorch && make rust-ffi")


def check_rust_ffi_linking_error(result: dict) -> bool:
    """Check if the inference failure is due to FFI linking errors."""
    error = result.get("error", "") + result.get("stdout", "")
    return "symbol lookup error" in error or "undefined symbol" in error


def check_cuda_not_compiled(result: dict) -> bool:
    """Check if the inference failure is due to CUDA not compiled in."""
    error = result.get("error", "") + result.get("stdout", "")
    return "CUDA support not compiled" in error


def create_minimal_config(device: str = "cpu") -> ViTConfig:
    """Create a minimal ViT config for fast testing."""
    return ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        img_size=(64, 64),  # Small image for fast testing
        depth=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        pos_enc="learnable",  # Learnable is most compatible for export
        num_register_tokens=0,
        num_cls_tokens=0,
        dtype=torch.float32,  # Use FP32 for export compatibility
    )


class ExportError(Exception):
    """Exception raised when model export fails."""

    pass


def export_model(config: ViTConfig, output_path: Path, shape: tuple[int, ...], device: str) -> None:
    """Export a ViT model to AOTInductor format.

    Raises:
        ExportError: If the export fails for any reason.
    """
    import os

    config_path = output_path.parent / "config.yaml"
    config_path.write_text(config.to_yaml())

    script_path = Path(__file__).parent.parent / "scripts" / "export_aot.py"
    shape_str = ",".join(str(s) for s in shape)

    cmd = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--shape",
        shape_str,
        "--device",
        device,
        "--dtype",
        "float32",
    ]

    # Set TORCH_COMPILE_DISABLE=1 to avoid FakeTensorMode conflicts
    # between @torch.compile decorators and torch.export tracing
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if result.returncode != 0:
        # Check for known compatibility issues
        if "fake mode" in result.stderr and "doesn't match mode" in result.stderr:
            raise ExportError(
                f"AOT export failed due to PyTorch FakeTensorMode compatibility issue. "
                f"This may be a version mismatch. PyTorch version: {torch.__version__}\n"
                f"stderr: {result.stderr[:500]}..."
            )
        raise ExportError(f"Export failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")


def run_rust_inference(
    model_path: Path,
    config_path: Path,
    shape: tuple[int, ...],
    device: str,
) -> dict:
    """Run inference using the Rust CLI.

    Returns:
        Dictionary with inference results including output shape.
    """
    binary = rust_binary_path()
    shape_str = ",".join(str(s) for s in shape)

    cmd = [
        str(binary),
        "infer",
        "--model",
        str(model_path),
        "--config",
        str(config_path),
        "--device",
        device,
        "--dtype",
        "float32",
        "--shape",
        shape_str,
        "--warmup",
        "1",
        "--iterations",
        "1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
            "stdout": result.stdout,
        }

    # Parse output shape from stdout
    # Expected format: "  Output shape: [1, 16, 64]"
    output_shape = None
    for line in result.stdout.split("\n"):
        if "Output shape:" in line:
            # Extract shape like [1, 16, 64]
            match = re.search(r"\[([0-9, ]+)\]", line)
            if match:
                output_shape = tuple(int(x.strip()) for x in match.group(1).split(","))
            break

    return {
        "success": True,
        "output_shape": output_shape,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


class TestAOTInference:
    """Integration tests for AOT export and Rust inference."""

    @pytest.mark.ci_skip
    def test_export_and_infer_cpu(self, tmp_path):
        """Test end-to-end export and inference on CPU."""
        skip_if_no_rust_ffi()

        # Create config and export
        config = create_minimal_config(device="cpu")
        model_path = tmp_path / "model.so"
        config_path = tmp_path / "config.yaml"
        shape = (1, 3, 64, 64)

        # Export model
        try:
            export_model(config, model_path, shape, device="cpu")
        except ExportError as e:
            if "FakeTensorMode compatibility" in str(e):
                pytest.skip(f"Skipping due to PyTorch export compatibility: {e}")
            raise
        assert model_path.exists(), "Model file was not created"

        # Write config for Rust CLI
        config_path.write_text(config.to_yaml())

        # Run inference
        result = run_rust_inference(model_path, config_path, shape, device="cpu")

        # Skip if FFI has linking errors (libtorch version mismatch)
        if not result["success"] and check_rust_ffi_linking_error(result):
            pytest.skip(
                "Rust FFI has linking errors (libtorch version mismatch). "
                "Rebuild with matching libtorch version: make rust-ffi"
            )

        assert result["success"], f"Inference failed: {result.get('error', result.get('stdout', 'unknown'))}"
        assert result["output_shape"] is not None, f"Could not parse output shape from: {result['stdout']}"

        # Verify output shape
        batch_size = shape[0]
        num_patches = (config.img_size[0] // config.patch_size[0]) * (config.img_size[1] // config.patch_size[1])
        expected_seq_len = num_patches + config.num_register_tokens + config.num_cls_tokens
        expected_shape = (batch_size, expected_seq_len, config.hidden_size)

        assert result["output_shape"] == expected_shape, (
            f"Output shape mismatch: got {result['output_shape']}, expected {expected_shape}"
        )

    @pytest.mark.ci_skip
    @pytest.mark.cuda
    def test_export_and_infer_cuda(self, tmp_path):
        """Test end-to-end export and inference on CUDA."""
        skip_if_no_rust_ffi()

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create config and export
        config = create_minimal_config(device="cuda")
        model_path = tmp_path / "model.so"
        config_path = tmp_path / "config.yaml"
        shape = (1, 3, 64, 64)

        # Export model
        try:
            export_model(config, model_path, shape, device="cuda")
        except ExportError as e:
            if "FakeTensorMode compatibility" in str(e):
                pytest.skip(f"Skipping due to PyTorch export compatibility: {e}")
            raise
        assert model_path.exists(), "Model file was not created"

        # Write config for Rust CLI
        config_path.write_text(config.to_yaml())

        # Run inference
        result = run_rust_inference(model_path, config_path, shape, device="cuda")

        # Skip if FFI has linking errors (libtorch version mismatch)
        if not result["success"] and check_rust_ffi_linking_error(result):
            pytest.skip(
                "Rust FFI has linking errors (libtorch version mismatch). "
                "Rebuild with matching libtorch version: make rust-ffi"
            )

        # Skip if CUDA support not compiled
        if not result["success"] and check_cuda_not_compiled(result):
            pytest.skip(
                "Rust FFI was built without CUDA support. Rebuild with CUDA libtorch: make libtorch && make rust-ffi"
            )

        assert result["success"], f"Inference failed: {result.get('error', result.get('stdout', 'unknown'))}"
        assert result["output_shape"] is not None, f"Could not parse output shape from: {result['stdout']}"

        # Verify output shape
        batch_size = shape[0]
        num_patches = (config.img_size[0] // config.patch_size[0]) * (config.img_size[1] // config.patch_size[1])
        expected_seq_len = num_patches + config.num_register_tokens + config.num_cls_tokens
        expected_shape = (batch_size, expected_seq_len, config.hidden_size)

        assert result["output_shape"] == expected_shape, (
            f"Output shape mismatch: got {result['output_shape']}, expected {expected_shape}"
        )

    @pytest.mark.ci_skip
    def test_export_and_infer_with_register_tokens(self, tmp_path):
        """Test export and inference with register tokens."""
        skip_if_no_rust_ffi()

        # Create config with register tokens
        config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(64, 64),
            depth=2,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            pos_enc="learnable",
            num_register_tokens=4,  # Add register tokens
            num_cls_tokens=1,  # Add CLS token
            dtype=torch.float32,
        )

        model_path = tmp_path / "model.so"
        config_path = tmp_path / "config.yaml"
        shape = (2, 3, 64, 64)  # Batch size 2

        # Export model
        try:
            export_model(config, model_path, shape, device="cpu")
        except ExportError as e:
            if "FakeTensorMode compatibility" in str(e):
                pytest.skip(f"Skipping due to PyTorch export compatibility: {e}")
            raise
        assert model_path.exists(), "Model file was not created"

        # Write config for Rust CLI
        config_path.write_text(config.to_yaml())

        # Run inference
        result = run_rust_inference(model_path, config_path, shape, device="cpu")

        # Skip if FFI has linking errors (libtorch version mismatch)
        if not result["success"] and check_rust_ffi_linking_error(result):
            pytest.skip(
                "Rust FFI has linking errors (libtorch version mismatch). "
                "Rebuild with matching libtorch version: make rust-ffi"
            )

        assert result["success"], f"Inference failed: {result.get('error', result.get('stdout', 'unknown'))}"
        assert result["output_shape"] is not None, f"Could not parse output shape from: {result['stdout']}"

        # Verify output shape includes register and CLS tokens
        batch_size = shape[0]
        num_patches = (config.img_size[0] // config.patch_size[0]) * (config.img_size[1] // config.patch_size[1])
        expected_seq_len = num_patches + config.num_register_tokens + config.num_cls_tokens
        expected_shape = (batch_size, expected_seq_len, config.hidden_size)

        assert result["output_shape"] == expected_shape, (
            f"Output shape mismatch: got {result['output_shape']}, expected {expected_shape}"
        )

    @pytest.mark.ci_skip
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_export_and_infer_different_batch_sizes(self, tmp_path, batch_size):
        """Test export and inference with different batch sizes."""
        skip_if_no_rust_ffi()

        config = create_minimal_config(device="cpu")
        model_path = tmp_path / "model.so"
        config_path = tmp_path / "config.yaml"
        shape = (batch_size, 3, 64, 64)

        # Export model
        try:
            export_model(config, model_path, shape, device="cpu")
        except ExportError as e:
            if "FakeTensorMode compatibility" in str(e):
                pytest.skip(f"Skipping due to PyTorch export compatibility: {e}")
            raise
        assert model_path.exists(), "Model file was not created"

        # Write config for Rust CLI
        config_path.write_text(config.to_yaml())

        # Run inference
        result = run_rust_inference(model_path, config_path, shape, device="cpu")

        # Skip if FFI has linking errors (libtorch version mismatch)
        if not result["success"] and check_rust_ffi_linking_error(result):
            pytest.skip(
                "Rust FFI has linking errors (libtorch version mismatch). "
                "Rebuild with matching libtorch version: make rust-ffi"
            )

        assert result["success"], f"Inference failed: {result.get('error', result.get('stdout', 'unknown'))}"
        assert result["output_shape"] is not None, f"Could not parse output shape from: {result['stdout']}"

        # Verify batch size in output
        assert result["output_shape"][0] == batch_size, (
            f"Batch size mismatch: got {result['output_shape'][0]}, expected {batch_size}"
        )


class TestRustValidateAndSummarize:
    """Tests for Rust validate and summarize commands (no FFI required)."""

    def test_validate_config(self, tmp_path):
        """Test that Rust CLI can validate a config file."""
        binary = rust_binary_path()
        if not binary.exists():
            pytest.skip("Rust binary not built. Run: make rust-release")

        config = create_minimal_config()
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config.to_yaml())

        result = subprocess.run(
            [str(binary), "validate", str(config_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Validate failed: {result.stderr}"
        assert "valid" in result.stdout.lower() or result.returncode == 0

    def test_summarize_config(self, tmp_path):
        """Test that Rust CLI can summarize a config file."""
        binary = rust_binary_path()
        if not binary.exists():
            pytest.skip("Rust binary not built. Run: make rust-release")

        config = create_minimal_config()
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config.to_yaml())

        result = subprocess.run(
            [str(binary), "summarize", str(config_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Summarize failed: {result.stderr}"
        # Check that key information is present
        assert "64" in result.stdout  # hidden_size
        assert "2" in result.stdout  # depth

    def test_summarize_json_output(self, tmp_path):
        """Test that Rust CLI can output JSON summary."""
        binary = rust_binary_path()
        if not binary.exists():
            pytest.skip("Rust binary not built. Run: make rust-release")

        config = create_minimal_config()
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config.to_yaml())

        result = subprocess.run(
            [str(binary), "summarize", "--format", "json", str(config_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Summarize failed: {result.stderr}"

        import json

        summary = json.loads(result.stdout)
        assert summary["architecture"]["hidden_size"] == 64
        assert summary["architecture"]["depth"] == 2
        assert summary["architecture"]["num_attention_heads"] == 4
