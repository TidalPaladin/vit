#!/usr/bin/env python
"""Tests for benchmarking functionality."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from benchmark import (
    BenchmarkResult,
    PassMode,
    benchmark_latency,
    benchmark_memory,
    compute_gflops,
    create_input_from_config,
    plot_benchmark_results,
    run_full_benchmark,
    warmup_model,
)
from benchmark.cli import main as benchmark_cli_main, parse_resolution
from vit import ViTConfig


FORWARD_GFLOPS = 4.0
SQUARE_RESOLUTION = 224
SMALL_CONFIG_YAML = """\
in_channels: 3
patch_size: [8, 8]
img_size: [32, 32]
depth: 2
hidden_size: 64
ffn_hidden_size: 128
num_attention_heads: 4
dtype: float32
"""


class RecordingModel(nn.Module):
    """Record forward and backward work relative to benchmark timer calls."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.events = events

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.events.append("forward")
        output = input_tensor * self.weight

        def record_backward(gradient: torch.Tensor) -> None:
            self.events.append("backward")

        output.register_hook(record_backward)
        return output


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


@pytest.mark.parametrize(
    ("pass_mode", "expected_events"),
    [
        ("backward", ["forward", "timer", "backward", "timer"]),
        ("forward_backward", ["timer", "forward", "backward", "timer"]),
    ],
)
def test_benchmark_latency_times_the_named_workload(
    mocker,
    pass_mode: PassMode,
    expected_events: list[str],
) -> None:
    """Backward-only timing excludes graph construction while combined timing includes it."""
    events: list[str] = []
    model = RecordingModel(events)

    def record_timer_call() -> float:
        events.append("timer")
        return float(events.count("timer"))

    mocker.patch("benchmark.benchmark.time.perf_counter", side_effect=record_timer_call)

    benchmark_latency(
        model,
        torch.ones(1),
        num_iters=1,
        pass_mode=pass_mode,
        device=torch.device("cpu"),
    )

    assert events == expected_events


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


def test_backward_memory_excludes_forward_peak(mocker) -> None:
    """Backward-only memory tracking must reset peak stats after graph construction."""
    events: list[str] = []
    model = RecordingModel(events)
    mocker.patch(
        "benchmark.benchmark.torch.cuda.reset_peak_memory_stats", side_effect=lambda _device: events.append("reset")
    )
    mocker.patch("benchmark.benchmark.torch.cuda.empty_cache", side_effect=lambda: events.append("empty_cache"))
    mocker.patch(
        "benchmark.benchmark.torch.cuda.max_memory_allocated",
        side_effect=lambda _device: events.append("peak") or 0,
    )

    benchmark_memory(
        model,
        torch.ones(1),
        num_iters=1,
        pass_mode="backward",
        device=torch.device("cuda"),
    )

    assert events == ["forward", "reset", "empty_cache", "backward", "peak"]


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


@pytest.mark.parametrize(
    ("pass_mode", "expected_gflops"),
    [
        ("forward", FORWARD_GFLOPS),
        ("backward", FORWARD_GFLOPS * 2),
        ("forward_backward", FORWARD_GFLOPS * 3),
    ],
)
def test_run_full_benchmark_reports_named_workload_operations(
    mocker,
    small_config: ViTConfig,
    pass_mode: PassMode,
    expected_gflops: float,
) -> None:
    """Reported operation counts match the selected pass workload."""
    mocker.patch("benchmark.benchmark.warmup_model")
    mocker.patch("benchmark.benchmark.benchmark_latency", return_value=1.0)
    mocker.patch("benchmark.benchmark.benchmark_memory", return_value=0.0)
    mocker.patch("benchmark.benchmark.compute_gflops", return_value=FORWARD_GFLOPS)

    result = run_full_benchmark(
        config=small_config,
        batch_size=1,
        device="cpu",
        pass_mode=pass_mode,
        show_progress=False,
    )

    assert result.gflops == expected_gflops


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
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(SMALL_CONFIG_YAML)

    config = ViTConfig.from_yaml(yaml_path)

    assert config.in_channels == 3
    assert config.patch_size == [8, 8]
    assert config.img_size == [32, 32]
    assert config.depth == 2
    assert config.hidden_size == 64


def test_parse_resolution_scalar_is_square() -> None:
    """A scalar resolution follows the documented two-dimensional square form."""
    assert parse_resolution(str(SQUARE_RESOLUTION)) == (SQUARE_RESOLUTION, SQUARE_RESOLUTION)


def test_cli_invalid_device_uses_runtime_error_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid device is a concise runtime error, not a successful empty run."""
    invalid_device = "invalid-device"
    exit_code = benchmark_cli_main(
        [
            "--configs",
            "unused.yaml",
            "--resolutions",
            str(SQUARE_RESOLUTION),
            "--device",
            invalid_device,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert invalid_device not in captured.out
    assert f"invalid device '{invalid_device}'" in captured.err


def test_cli_success_returns_zero_and_writes_primary_report(
    mocker,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A successful run emits the report on stdout and returns zero."""
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "results"
    config_path.write_text(SMALL_CONFIG_YAML)
    mocker.patch(
        "benchmark.cli.run_full_benchmark",
        return_value=BenchmarkResult(
            latency_ms=1.0,
            memory_mb=0.0,
            gflops=FORWARD_GFLOPS,
            config_name="config",
            batch_size=1,
            image_size=(32, 32),
            pass_mode="forward",
        ),
    )

    exit_code = benchmark_cli_main(
        [
            "--configs",
            str(config_path),
            "--resolutions",
            "32,32",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Completed 1 benchmark(s)" in captured.out
    assert (output_dir / "benchmark_results.csv").exists()


def test_cli_failure_uses_stderr_and_nonzero_exit(
    mocker,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A run with no successful cases reports diagnostics separately and fails."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SMALL_CONFIG_YAML)
    mocker.patch("benchmark.cli.run_full_benchmark", side_effect=RuntimeError("device failure"))

    exit_code = benchmark_cli_main(
        [
            "--configs",
            str(config_path),
            "--resolutions",
            "32,32",
            "--output-dir",
            str(tmp_path / "results"),
            "--no-plots",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "device failure" not in captured.out
    assert "vit-benchmark failed:" in captured.err
    assert "device failure" in captured.err
