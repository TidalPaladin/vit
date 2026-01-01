#!/usr/bin/env python
"""Tests for benchmarking functionality."""

from pathlib import Path

import pytest
import torch

from benchmark import (
    benchmark_latency,
    benchmark_memory,
    compute_gflops,
    create_input_from_config,
    plot_benchmark_results,
    run_full_benchmark,
    warmup_model,
)
from vit import ViTConfig


@pytest.fixture
def small_config() -> ViTConfig:
    """Create a small ViT configuration for testing."""
    return ViTConfig(
        in_channels=3,
        patch_size=[8, 8],
        img_size=[32, 32],
        depth=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        dtype=torch.float32,
    )


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_create_input_from_config(small_config: ViTConfig, device: torch.device) -> None:
    """Test input tensor creation from config."""
    batch_size = 2
    input_tensor = create_input_from_config(small_config, batch_size, device)

    expected_shape = (batch_size, small_config.in_channels, *small_config.img_size)
    assert input_tensor.shape == expected_shape
    assert input_tensor.device.type == device.type
    assert input_tensor.dtype == small_config.dtype


@pytest.mark.parametrize("pass_mode", ["forward", "backward", "forward_backward"])
def test_warmup_model(small_config: ViTConfig, device: torch.device, pass_mode: str) -> None:
    """Test model warmup with different pass modes."""
    model = small_config.instantiate(device=device)
    input_tensor = create_input_from_config(small_config, batch_size=1, device=device)

    # Should not raise any errors
    warmup_model(model, input_tensor, num_warmup_iters=2, pass_mode=pass_mode)  # type: ignore


def test_compute_gflops(small_config: ViTConfig, device: torch.device) -> None:
    """Test GFLOPs computation."""
    model = small_config.instantiate(device=device)
    input_shape = (1, small_config.in_channels, *small_config.img_size)

    gflops = compute_gflops(model, input_shape)

    assert isinstance(gflops, float)
    assert gflops > 0


@pytest.mark.parametrize("activation", ["gelu", "swiglu", "geglu"])
def test_compute_gflops_glu_variants(small_config: ViTConfig, device: torch.device, activation: str) -> None:
    """Test that GFLOPs computation accounts for GLU variants correctly."""
    # Create configs with different activations
    config_dict = small_config.__dict__.copy()
    config_dict["activation"] = activation
    config = ViTConfig(**config_dict)

    model = config.instantiate(device=device)
    input_shape = (1, config.in_channels, *config.img_size)

    gflops = compute_gflops(model, input_shape)

    assert isinstance(gflops, float)
    assert gflops > 0

    # GLU variants should have ~1.5x more FLOPs than non-GLU
    # (The first FC layer is 2x, but second FC and attention are the same)
    if activation.endswith("glu"):
        # Compute non-GLU version for comparison
        config_dict_no_glu = small_config.__dict__.copy()
        config_dict_no_glu["activation"] = "gelu"
        config_no_glu = ViTConfig(**config_dict_no_glu)
        model_no_glu = config_no_glu.instantiate(device=device)
        gflops_no_glu = compute_gflops(model_no_glu, input_shape)

        # GLU should have more FLOPs
        assert gflops > gflops_no_glu


@pytest.mark.parametrize("pass_mode", ["forward", "backward", "forward_backward"])
def test_benchmark_latency(small_config: ViTConfig, device: torch.device, pass_mode: str) -> None:
    """Test latency benchmarking."""
    model = small_config.instantiate(device=device)
    input_tensor = create_input_from_config(small_config, batch_size=1, device=device)

    # Warmup first
    warmup_model(model, input_tensor, num_warmup_iters=2, pass_mode=pass_mode)  # type: ignore

    # Benchmark
    latency_ms = benchmark_latency(model, input_tensor, num_iters=5, pass_mode=pass_mode, device=device)  # type: ignore

    assert isinstance(latency_ms, float)
    assert latency_ms > 0


def test_benchmark_memory(small_config: ViTConfig, device: torch.device) -> None:
    """Test memory benchmarking."""
    model = small_config.instantiate(device=device)
    input_tensor = create_input_from_config(small_config, batch_size=1, device=device)

    # Warmup first
    warmup_model(model, input_tensor, num_warmup_iters=2, pass_mode="forward")

    # Benchmark
    memory_mb = benchmark_memory(model, input_tensor, num_iters=2, pass_mode="forward", device=device)

    assert isinstance(memory_mb, float)
    assert memory_mb >= 0  # May be 0 for CPU


@pytest.mark.parametrize("pass_mode", ["forward", "backward", "forward_backward"])
def test_run_full_benchmark(small_config: ViTConfig, device: torch.device, pass_mode: str) -> None:
    """Test full benchmark suite."""
    result = run_full_benchmark(
        config=small_config,
        batch_size=1,
        device=device,
        pass_mode=pass_mode,  # type: ignore
        num_warmup_iters=2,
        num_latency_iters=5,
        num_memory_iters=2,
        config_name="test_config",
        show_progress=False,
    )

    assert result.config_name == "test_config"
    assert result.batch_size == 1
    assert result.image_size == tuple(small_config.img_size)
    assert result.pass_mode == pass_mode
    assert result.latency_ms > 0
    assert result.memory_mb >= 0
    assert result.gflops > 0


def test_plot_benchmark_results(small_config: ViTConfig, device: torch.device, tmp_path: Path) -> None:
    """Test plotting functionality."""
    # Create some benchmark results
    results = []
    for img_size in [(32, 32), (64, 64)]:
        config_dict = small_config.__dict__.copy()
        config_dict["img_size"] = img_size
        config = ViTConfig(**config_dict)

        result = run_full_benchmark(
            config=config,
            batch_size=1,
            device=device,
            pass_mode="forward",
            num_warmup_iters=1,
            num_latency_iters=2,
            num_memory_iters=1,
            config_name="test",
            show_progress=False,
        )
        results.append(result)

    # Test plotting
    output_paths = plot_benchmark_results(
        results=results,
        output_dir=tmp_path,
        metric="latency",
        plot_format=["png"],
        dpi=100,
    )

    assert len(output_paths) == 1
    assert output_paths[0].exists()
    assert output_paths[0].suffix == ".png"


def test_config_yaml_loading(tmp_path: Path) -> None:
    """Test loading config from YAML."""
    yaml_content = """
in_channels: 3
patch_size: [8, 8]
img_size: [32, 32]
depth: 2
hidden_size: 64
ffn_hidden_size: 128
num_attention_heads: 4
dtype: float32
"""

    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml_content)

    config = ViTConfig.from_yaml(yaml_path)

    assert config.in_channels == 3
    assert config.patch_size == [8, 8]
    assert config.img_size == [32, 32]
    assert config.depth == 2
    assert config.hidden_size == 64
