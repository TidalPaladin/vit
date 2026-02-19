#!/usr/bin/env python
"""Criterion-style component benchmarks for transformer primitives."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from vit.attention import SelfAttention
from vit.drop_path import drop_path
from vit.fused import NormMLP
from vit.layer_scale import LayerScale
from vit.norm import NORM_TYPE_CHOICES, NormType


ComponentKind = Literal["mlp", "self_attention", "layer_scale_residual", "drop_path_residual"]
PassMode = Literal["forward", "backward", "forward_backward"]
ComparisonMetric = Literal["mean_ms", "median_ms", "p95_ms", "std_ms", "memory_mb"]


DEFAULT_COMPONENTS: tuple[ComponentKind, ...] = ("mlp", "self_attention")
DEFAULT_PASS_MODES: tuple[PassMode, ...] = ("forward",)
DEFAULT_FFN_MULTS: tuple[int, ...] = (4,)
DEFAULT_PRESETS: tuple[str, ...] = ("default",)
DEFAULT_SHAPE_TIERS: tuple[tuple[int, int, int, int], ...] = (
    (8, 64, 256, 4),
    (4, 256, 512, 8),
    (2, 1024, 768, 12),
)
DEFAULT_HEAD_BY_HIDDEN: dict[int, int] = {256: 4, 512: 8, 768: 12}


_DTYPE_NAME_TO_TORCH: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

_TORCH_DTYPE_TO_NAME: dict[torch.dtype, str] = {dtype: name for name, dtype in _DTYPE_NAME_TO_TORCH.items()}


@dataclass(frozen=True)
class PresetConfig:
    """Preset for component benchmark behavior."""

    activation: str = "swiglu"
    use_rope: bool = True
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    train_mode: bool = False
    drop_path_prob: float = 0.1
    layer_scale_init: float = 1e-5
    bias: bool = True
    eps: float = 1e-5


PRESET_CONFIGS: dict[str, PresetConfig] = {
    "default": PresetConfig(),
    "train": PresetConfig(hidden_dropout=0.1, attention_dropout=0.1, train_mode=True),
    "rope": PresetConfig(use_rope=True),
}


@dataclass(frozen=True)
class ComponentBenchmarkCase:
    """Single benchmark case configuration."""

    component: ComponentKind
    preset: str
    pass_mode: PassMode
    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int
    ffn_hidden_size: int
    activation: str
    use_rope: bool
    hidden_dropout: float
    attention_dropout: float
    train_mode: bool
    drop_path_prob: float
    layer_scale_init: float
    bias: bool
    eps: float
    norm_type: NormType

    @property
    def case_id(self) -> str:
        return (
            f"{self.component}|{self.preset}|{self.pass_mode}|"
            f"b{self.batch_size}s{self.seq_len}d{self.hidden_size}"
            f"h{self.num_heads}ffn{self.ffn_hidden_size}"
        )


@dataclass(frozen=True)
class ComponentBenchmarkStats:
    """Measured metrics for a benchmark case."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    sample_count: int
    measurement_seconds: float
    memory_mb: float | None


@dataclass(frozen=True)
class ComponentBenchmarkResult:
    """Result payload for a benchmark case."""

    case: ComponentBenchmarkCase
    stats: ComponentBenchmarkStats
    device: str
    param_dtype: str
    input_dtype: str
    autocast_dtype: str | None


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison of one metric between candidate and baseline."""

    case_id: str
    metric: ComparisonMetric
    baseline_value: float
    candidate_value: float
    delta_abs: float
    delta_pct: float
    is_regression: bool


@dataclass(frozen=True)
class ComparisonSummary:
    """Summary metadata for a comparison run."""

    baseline_cases: int
    candidate_cases: int
    compared_cases: int
    missing_in_baseline: int
    missing_in_candidate: int
    regressions: int
    metric: ComparisonMetric
    regression_threshold_pct: float
    noise_floor_pct: float


@dataclass(frozen=True)
class _BenchmarkTarget:
    forward: Callable[[], Tensor]
    zero_grad: Callable[[], None]


def resolve_dtype(dtype_name: str | None, *, allow_none: bool = False) -> torch.dtype | None:
    """Resolve a string dtype name to a torch dtype."""
    if dtype_name is None:
        if allow_none:
            return None
        raise ValueError("dtype cannot be None")

    lowered = dtype_name.lower()
    if lowered == "none" and allow_none:
        return None

    if lowered not in _DTYPE_NAME_TO_TORCH:
        valid = ", ".join(sorted(_DTYPE_NAME_TO_TORCH))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Valid values: {valid}")
    return _DTYPE_NAME_TO_TORCH[lowered]


def dtype_name(dtype: torch.dtype | None) -> str | None:
    """Return stable dtype name for serialization."""
    if dtype is None:
        return None
    return _TORCH_DTYPE_TO_NAME.get(dtype, str(dtype).removeprefix("torch."))


def configure_runtime(seed: int, deterministic: bool, threads: int | None = None) -> None:
    """Configure runtime determinism and threading for reproducible benchmarks."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    if threads is not None:
        torch.set_num_threads(threads)


def collect_run_metadata(
    *,
    seed: int,
    deterministic: bool,
    threads: int | None,
    device: torch.device,
    include_memory: bool,
) -> dict[str, Any]:
    """Collect benchmark runtime metadata for reproducibility."""
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "include_memory": include_memory,
        "seed": seed,
        "deterministic": deterministic,
        "threads": threads,
        "num_threads": torch.get_num_threads(),
        "git_sha": _resolve_git_sha(),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability(device))

    return metadata


def _resolve_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    sha = completed.stdout.strip()
    return sha or None


def build_component_benchmark_cases(
    *,
    components: list[ComponentKind] | tuple[ComponentKind, ...] = DEFAULT_COMPONENTS,
    presets: list[str] | tuple[str, ...] = DEFAULT_PRESETS,
    pass_modes: list[PassMode] | tuple[PassMode, ...] = DEFAULT_PASS_MODES,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
    hidden_sizes: list[int] | None = None,
    num_heads: list[int] | None = None,
    ffn_mults: list[int] | None = None,
    activation: str | None = None,
    use_rope: bool | None = None,
    hidden_dropout: float | None = None,
    attention_dropout: float | None = None,
    train_mode: bool | None = None,
    drop_path_prob: float | None = None,
    layer_scale_init: float | None = None,
    bias: bool | None = None,
    eps: float | None = None,
    norm_type: NormType = "rmsnorm",
) -> list[ComponentBenchmarkCase]:
    """Generate benchmark cases from sweep configuration and presets."""
    _validate_sweeps(batch_sizes, seq_lens, hidden_sizes, num_heads, ffn_mults)
    if norm_type not in NORM_TYPE_CHOICES:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Available norm types: {', '.join(NORM_TYPE_CHOICES)}")

    for preset in presets:
        if preset not in PRESET_CONFIGS:
            available = ", ".join(sorted(PRESET_CONFIGS))
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")

    shape_grid = _build_shape_grid(batch_sizes, seq_lens, hidden_sizes)
    ffn_mult_values = ffn_mults or list(DEFAULT_FFN_MULTS)
    cases: list[ComponentBenchmarkCase] = []

    for component, preset, pass_mode in product(components, presets, pass_modes):
        preset_config = PRESET_CONFIGS[preset]
        resolved_activation = activation if activation is not None else preset_config.activation
        resolved_use_rope = use_rope if use_rope is not None else preset_config.use_rope
        resolved_hidden_dropout = hidden_dropout if hidden_dropout is not None else preset_config.hidden_dropout
        resolved_attention_dropout = (
            attention_dropout if attention_dropout is not None else preset_config.attention_dropout
        )
        resolved_train_mode = train_mode if train_mode is not None else preset_config.train_mode
        resolved_drop_path_prob = drop_path_prob if drop_path_prob is not None else preset_config.drop_path_prob
        resolved_layer_scale_init = layer_scale_init if layer_scale_init is not None else preset_config.layer_scale_init
        resolved_bias = bias if bias is not None else preset_config.bias
        resolved_eps = eps if eps is not None else preset_config.eps

        for batch_size, seq_len, hidden_size in shape_grid:
            common_case_fields: dict[str, Any] = {
                "component": component,
                "preset": preset,
                "pass_mode": pass_mode,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "activation": resolved_activation,
                "hidden_dropout": resolved_hidden_dropout,
                "attention_dropout": resolved_attention_dropout,
                "train_mode": resolved_train_mode,
                "drop_path_prob": resolved_drop_path_prob,
                "layer_scale_init": resolved_layer_scale_init,
                "bias": resolved_bias,
                "eps": resolved_eps,
                "norm_type": norm_type,
            }

            if component == "self_attention":
                candidate_heads = _resolve_head_candidates(hidden_size, num_heads)
                for head_count in candidate_heads:
                    if hidden_size % head_count != 0:
                        continue
                    cases.append(
                        ComponentBenchmarkCase(
                            **common_case_fields,
                            num_heads=head_count,
                            ffn_hidden_size=hidden_size * ffn_mult_values[0],
                            use_rope=resolved_use_rope,
                        )
                    )
                continue

            if component == "mlp":
                for ffn_mult in ffn_mult_values:
                    cases.append(
                        ComponentBenchmarkCase(
                            **common_case_fields,
                            num_heads=_default_head_count(hidden_size),
                            ffn_hidden_size=hidden_size * ffn_mult,
                            use_rope=False,
                        )
                    )
                continue

            # Residual-only ops do not depend on heads/ffn size for runtime behavior.
            cases.append(
                ComponentBenchmarkCase(
                    **common_case_fields,
                    num_heads=_default_head_count(hidden_size),
                    ffn_hidden_size=hidden_size * ffn_mult_values[0],
                    use_rope=False,
                )
            )

    if not cases:
        raise ValueError("No benchmark cases generated. Check sweep values and divisibility constraints.")

    return cases


def run_component_benchmark_suite(
    *,
    device: torch.device,
    param_dtype: torch.dtype,
    input_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
    components: list[ComponentKind] | tuple[ComponentKind, ...] = DEFAULT_COMPONENTS,
    presets: list[str] | tuple[str, ...] = DEFAULT_PRESETS,
    pass_modes: list[PassMode] | tuple[PassMode, ...] = DEFAULT_PASS_MODES,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
    hidden_sizes: list[int] | None = None,
    num_heads: list[int] | None = None,
    ffn_mults: list[int] | None = None,
    activation: str | None = None,
    use_rope: bool | None = None,
    hidden_dropout: float | None = None,
    attention_dropout: float | None = None,
    train_mode: bool | None = None,
    drop_path_prob: float | None = None,
    layer_scale_init: float | None = None,
    bias: bool | None = None,
    eps: float | None = None,
    num_warmup_iters: int = 10,
    min_samples: int = 30,
    min_measurement_seconds: float = 1.0,
    max_measurement_seconds: float = 10.0,
    include_memory: bool = False,
    num_memory_iters: int = 5,
    norm_type: NormType = "rmsnorm",
) -> list[ComponentBenchmarkResult]:
    """Run all benchmark cases in a configured suite."""
    cases = build_component_benchmark_cases(
        components=components,
        presets=presets,
        pass_modes=pass_modes,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        ffn_mults=ffn_mults,
        activation=activation,
        use_rope=use_rope,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        train_mode=train_mode,
        drop_path_prob=drop_path_prob,
        layer_scale_init=layer_scale_init,
        bias=bias,
        eps=eps,
        norm_type=norm_type,
    )

    return [
        run_component_benchmark_case(
            case,
            device=device,
            param_dtype=param_dtype,
            input_dtype=input_dtype,
            autocast_dtype=autocast_dtype,
            num_warmup_iters=num_warmup_iters,
            min_samples=min_samples,
            min_measurement_seconds=min_measurement_seconds,
            max_measurement_seconds=max_measurement_seconds,
            include_memory=include_memory,
            num_memory_iters=num_memory_iters,
        )
        for case in cases
    ]


def run_component_benchmark_case(
    case: ComponentBenchmarkCase,
    *,
    device: torch.device,
    param_dtype: torch.dtype,
    input_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None = None,
    num_warmup_iters: int = 10,
    min_samples: int = 30,
    min_measurement_seconds: float = 1.0,
    max_measurement_seconds: float = 10.0,
    include_memory: bool = False,
    num_memory_iters: int = 5,
) -> ComponentBenchmarkResult:
    """Run a benchmark for one component case."""
    _validate_timing_args(min_samples, min_measurement_seconds, max_measurement_seconds)
    _validate_autocast_dtype(device, autocast_dtype)

    target = _build_target(case, device, param_dtype, input_dtype, autocast_dtype)

    for _ in range(num_warmup_iters):
        _execute_iteration(target, case.pass_mode, device, timed=False)

    latencies_ms: list[float] = []
    measurement_start = time.perf_counter()

    while True:
        latency_ms = _execute_iteration(target, case.pass_mode, device, timed=True)
        latencies_ms.append(latency_ms)

        elapsed_seconds = time.perf_counter() - measurement_start
        enough_samples = len(latencies_ms) >= min_samples
        reached_min_duration = elapsed_seconds >= min_measurement_seconds
        reached_max_duration = elapsed_seconds >= max_measurement_seconds

        if reached_max_duration:
            break
        if enough_samples and reached_min_duration:
            break

    measurement_seconds = time.perf_counter() - measurement_start
    stats = _compute_stats(latencies_ms, measurement_seconds)

    memory_mb = None
    if include_memory and device.type == "cuda":
        memory_mb = _measure_memory_mb(target, case.pass_mode, device, num_iters=num_memory_iters)

    stats = ComponentBenchmarkStats(
        mean_ms=stats.mean_ms,
        median_ms=stats.median_ms,
        p95_ms=stats.p95_ms,
        std_ms=stats.std_ms,
        sample_count=stats.sample_count,
        measurement_seconds=stats.measurement_seconds,
        memory_mb=memory_mb,
    )

    return ComponentBenchmarkResult(
        case=case,
        stats=stats,
        device=str(device),
        param_dtype=dtype_name(param_dtype) or "unknown",
        input_dtype=dtype_name(input_dtype) or "unknown",
        autocast_dtype=dtype_name(autocast_dtype),
    )


def benchmark_result_to_record(result: ComponentBenchmarkResult) -> dict[str, Any]:
    """Flatten benchmark result into a serialization-friendly record."""
    case = asdict(result.case)
    stats = asdict(result.stats)
    record: dict[str, Any] = {
        "case_id": result.case.case_id,
        **case,
        **stats,
        "device": result.device,
        "param_dtype": result.param_dtype,
        "input_dtype": result.input_dtype,
        "autocast_dtype": result.autocast_dtype,
    }
    return record


def benchmark_results_to_records(results: list[ComponentBenchmarkResult]) -> list[dict[str, Any]]:
    """Flatten benchmark results to records."""
    return [benchmark_result_to_record(result) for result in results]


def save_benchmark_run(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    results: list[ComponentBenchmarkResult],
) -> tuple[Path, Path]:
    """Persist benchmark run artifacts to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records = benchmark_results_to_records(results)
    payload = {
        "run_metadata": metadata,
        "results": records,
    }

    json_path = output_dir / "component_benchmark_results.json"
    csv_path = output_dir / "component_benchmark_results.csv"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_records_csv(records, csv_path)

    return json_path, csv_path


def load_benchmark_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load benchmark run artifacts from a run directory or JSON file path."""
    json_path = path if path.is_file() else path / "component_benchmark_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    metadata = payload.get("run_metadata", {})
    results = payload.get("results", [])
    if not isinstance(metadata, dict) or not isinstance(results, list):
        raise ValueError(f"Invalid benchmark payload in {json_path}")

    return metadata, results


def compare_benchmark_runs(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    metric: ComparisonMetric,
    regression_threshold_pct: float,
    noise_floor_pct: float,
) -> tuple[list[ComparisonResult], ComparisonSummary]:
    """Compare baseline/candidate benchmark records by case id."""
    baseline_map = {record["case_id"]: record for record in baseline_records}
    candidate_map = {record["case_id"]: record for record in candidate_records}

    baseline_ids = set(baseline_map)
    candidate_ids = set(candidate_map)

    shared_ids = sorted(baseline_ids & candidate_ids)
    missing_in_baseline = len(candidate_ids - baseline_ids)
    missing_in_candidate = len(baseline_ids - candidate_ids)

    comparisons: list[ComparisonResult] = []
    for case_id in shared_ids:
        baseline_value = _resolve_metric_value(baseline_map[case_id], metric)
        candidate_value = _resolve_metric_value(candidate_map[case_id], metric)

        if baseline_value is None or candidate_value is None:
            continue

        delta_abs = candidate_value - baseline_value
        if baseline_value == 0:
            delta_pct = math.inf if delta_abs > 0 else 0.0
        else:
            delta_pct = (delta_abs / baseline_value) * 100.0

        is_regression = (
            delta_pct > noise_floor_pct and delta_pct > regression_threshold_pct and _larger_is_worse(metric)
        )

        comparisons.append(
            ComparisonResult(
                case_id=case_id,
                metric=metric,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                delta_abs=delta_abs,
                delta_pct=delta_pct,
                is_regression=is_regression,
            )
        )

    summary = ComparisonSummary(
        baseline_cases=len(baseline_records),
        candidate_cases=len(candidate_records),
        compared_cases=len(comparisons),
        missing_in_baseline=missing_in_baseline,
        missing_in_candidate=missing_in_candidate,
        regressions=sum(1 for comparison in comparisons if comparison.is_regression),
        metric=metric,
        regression_threshold_pct=regression_threshold_pct,
        noise_floor_pct=noise_floor_pct,
    )

    return comparisons, summary


def save_comparison_report(
    *,
    output_path: Path,
    baseline_name: str,
    candidate_name: str,
    comparisons: list[ComparisonResult],
    summary: ComparisonSummary,
) -> Path:
    """Write comparison report as JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "summary": asdict(summary),
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def list_baseline_dirs(output_root: Path) -> list[Path]:
    """List baseline directories sorted by modification time descending."""
    if not output_root.exists():
        return []

    dirs = [path for path in output_root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_run_path(output_root: Path, identifier: str | None) -> Path:
    """Resolve a named run or direct path to a benchmark run directory."""
    if identifier is None:
        baselines = list_baseline_dirs(output_root)
        if not baselines:
            raise FileNotFoundError(f"No benchmark runs found in {output_root}")
        return baselines[0]

    direct = Path(identifier)
    if direct.exists():
        return direct

    named = output_root / identifier
    if named.exists():
        return named

    raise FileNotFoundError(f"Benchmark run '{identifier}' not found in {output_root}")


def _build_shape_grid(
    batch_sizes: list[int] | None,
    seq_lens: list[int] | None,
    hidden_sizes: list[int] | None,
) -> list[tuple[int, int, int]]:
    # Keep default concise and representative: paired small/medium/large points.
    if batch_sizes is None and seq_lens is None and hidden_sizes is None:
        return [(batch_size, seq_len, hidden_size) for batch_size, seq_len, hidden_size, _ in DEFAULT_SHAPE_TIERS]

    batch_values = batch_sizes or [tier[0] for tier in DEFAULT_SHAPE_TIERS]
    seq_values = seq_lens or [tier[1] for tier in DEFAULT_SHAPE_TIERS]
    hidden_values = hidden_sizes or [tier[2] for tier in DEFAULT_SHAPE_TIERS]

    return list(product(batch_values, seq_values, hidden_values))


def _resolve_head_candidates(hidden_size: int, num_heads: list[int] | None) -> list[int]:
    if num_heads is not None:
        return num_heads
    return [_default_head_count(hidden_size)]


def _default_head_count(hidden_size: int) -> int:
    if hidden_size in DEFAULT_HEAD_BY_HIDDEN:
        return DEFAULT_HEAD_BY_HIDDEN[hidden_size]

    head_count = max(1, hidden_size // 64)
    while hidden_size % head_count != 0 and head_count > 1:
        head_count -= 1
    return max(1, head_count)


def _build_target(
    case: ComponentBenchmarkCase,
    device: torch.device,
    param_dtype: torch.dtype,
    input_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
) -> _BenchmarkTarget:
    if case.component == "mlp":
        return _build_mlp_target(case, device, param_dtype, input_dtype, autocast_dtype)
    if case.component == "self_attention":
        return _build_self_attention_target(case, device, param_dtype, input_dtype, autocast_dtype)
    if case.component == "layer_scale_residual":
        return _build_layer_scale_target(case, device, param_dtype, input_dtype)
    if case.component == "drop_path_residual":
        return _build_drop_path_target(case, device, input_dtype)
    raise ValueError(f"Unsupported component: {case.component}")


def _build_mlp_target(
    case: ComponentBenchmarkCase,
    device: torch.device,
    param_dtype: torch.dtype,
    input_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
) -> _BenchmarkTarget:
    module = NormMLP(
        hidden_size=case.hidden_size,
        ffn_hidden_size=case.ffn_hidden_size,
        bias=case.bias,
        activation=case.activation,
        norm_type=case.norm_type,
        eps=case.eps,
        dropout=case.hidden_dropout,
        device=device,
        dtype=param_dtype,
    )
    module.train(case.train_mode or case.pass_mode != "forward")

    x = torch.randn(case.batch_size, case.seq_len, case.hidden_size, device=device, dtype=input_dtype)

    def forward() -> Tensor:
        return _call_with_autocast(lambda: module(x), device=device, autocast_dtype=autocast_dtype)

    def zero_grad() -> None:
        module.zero_grad(set_to_none=True)

    return _BenchmarkTarget(forward=forward, zero_grad=zero_grad)


def _build_self_attention_target(
    case: ComponentBenchmarkCase,
    device: torch.device,
    param_dtype: torch.dtype,
    input_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
) -> _BenchmarkTarget:
    if case.hidden_size % case.num_heads != 0:
        raise ValueError(
            f"Invalid attention case: hidden_size={case.hidden_size} must be divisible by num_heads={case.num_heads}"
        )

    module = SelfAttention(
        hidden_size=case.hidden_size,
        num_attention_heads=case.num_heads,
        hidden_dropout=case.hidden_dropout,
        attention_dropout=case.attention_dropout,
        bias=case.bias,
        norm_type=case.norm_type,
        eps=case.eps,
        device=device,
        dtype=param_dtype,
    )
    module.train(case.train_mode or case.pass_mode != "forward")

    x = torch.randn(case.batch_size, case.seq_len, case.hidden_size, device=device, dtype=input_dtype)
    rope = None
    if case.use_rope:
        head_dim = case.hidden_size // case.num_heads
        rope = _create_rope(case.seq_len, head_dim, device, input_dtype)

    def forward() -> Tensor:
        return _call_with_autocast(lambda: module(x, rope=rope), device=device, autocast_dtype=autocast_dtype)

    def zero_grad() -> None:
        module.zero_grad(set_to_none=True)

    return _BenchmarkTarget(forward=forward, zero_grad=zero_grad)


def _build_layer_scale_target(
    case: ComponentBenchmarkCase,
    device: torch.device,
    param_dtype: torch.dtype,
    input_dtype: torch.dtype,
) -> _BenchmarkTarget:
    layer = LayerScale(
        dim=case.hidden_size,
        init_value=case.layer_scale_init,
        inplace=False,
        device=device,
        dtype=param_dtype,
    )
    layer.train(case.train_mode or case.pass_mode != "forward")
    x = torch.randn(case.batch_size, case.seq_len, case.hidden_size, device=device, dtype=input_dtype)

    def forward() -> Tensor:
        return x + layer(x)

    def zero_grad() -> None:
        layer.zero_grad(set_to_none=True)

    return _BenchmarkTarget(forward=forward, zero_grad=zero_grad)


def _build_drop_path_target(
    case: ComponentBenchmarkCase,
    device: torch.device,
    input_dtype: torch.dtype,
) -> _BenchmarkTarget:
    requires_grad = case.pass_mode != "forward"
    x = torch.randn(
        case.batch_size,
        case.seq_len,
        case.hidden_size,
        device=device,
        dtype=input_dtype,
        requires_grad=requires_grad,
    )
    train_mode = case.train_mode or case.pass_mode != "forward"

    def forward() -> Tensor:
        return x + drop_path(x, case.drop_path_prob, train_mode)

    def zero_grad() -> None:
        if x.grad is not None:
            x.grad = None

    return _BenchmarkTarget(forward=forward, zero_grad=zero_grad)


def _create_rope(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE benchmark, got {head_dim}")

    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(0, head_dim // 2, device=device, dtype=torch.float32)
    frequencies = torch.pow(10000.0, -(2 * frequencies) / max(head_dim, 1))
    angles = positions * frequencies.unsqueeze(0)

    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype=dtype)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype=dtype)
    return torch.stack((sin, cos), dim=0)


def _call_with_autocast(
    fn: Callable[[], Tensor],
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> Tensor:
    if autocast_dtype is None:
        return fn()
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        return fn()


def _execute_iteration(target: _BenchmarkTarget, pass_mode: PassMode, device: torch.device, *, timed: bool) -> float:
    if pass_mode == "forward":
        target.zero_grad()
        elapsed_ms = _time_work(
            device,
            timed=timed,
            work=lambda: _forward_inference(target),
        )
        target.zero_grad()
        return elapsed_ms

    if pass_mode == "backward":
        target.zero_grad()
        output = target.forward()
        loss = output.mean()
        elapsed_ms = _time_work(device, timed=timed, work=loss.backward)
        target.zero_grad()
        return elapsed_ms

    if pass_mode == "forward_backward":
        target.zero_grad()
        elapsed_ms = _time_work(device, timed=timed, work=lambda: _forward_and_backward(target))
        target.zero_grad()
        return elapsed_ms

    raise ValueError(f"Unsupported pass mode: {pass_mode}")


def _forward_inference(target: _BenchmarkTarget) -> None:
    with torch.inference_mode():
        _ = target.forward()


def _forward_and_backward(target: _BenchmarkTarget) -> None:
    output = target.forward()
    output.mean().backward()


def _time_work(device: torch.device, *, timed: bool, work: Callable[[], None]) -> float:
    _synchronize_device(device)
    start = time.perf_counter() if timed else 0.0
    work()
    _synchronize_device(device)
    if not timed:
        return 0.0
    return (time.perf_counter() - start) * 1000.0


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_memory_mb(target: _BenchmarkTarget, pass_mode: PassMode, device: torch.device, num_iters: int) -> float:
    max_memory_mb = 0.0
    for _ in range(num_iters):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        _execute_iteration(target, pass_mode, device, timed=False)

        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        max_memory_mb = max(max_memory_mb, peak_memory_mb)

    return max_memory_mb


def _compute_stats(latencies_ms: list[float], measurement_seconds: float) -> ComponentBenchmarkStats:
    if not latencies_ms:
        raise ValueError("latencies_ms cannot be empty")

    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, min(len(sorted_latencies) - 1, math.ceil(len(sorted_latencies) * 0.95) - 1))

    return ComponentBenchmarkStats(
        mean_ms=float(statistics.fmean(latencies_ms)),
        median_ms=float(statistics.median(latencies_ms)),
        p95_ms=float(sorted_latencies[p95_index]),
        std_ms=float(statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0),
        sample_count=len(latencies_ms),
        measurement_seconds=measurement_seconds,
        memory_mb=None,
    )


def _resolve_metric_value(record: dict[str, Any], metric: ComparisonMetric) -> float | None:
    value = record.get(metric)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _larger_is_worse(metric: ComparisonMetric) -> bool:
    return metric in {"mean_ms", "median_ms", "p95_ms", "std_ms", "memory_mb"}


def _write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _validate_autocast_dtype(device: torch.device, autocast_dtype: torch.dtype | None) -> None:
    if autocast_dtype is None:
        return

    if device.type not in {"cpu", "cuda"}:
        return

    if autocast_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"{device.type.upper()} autocast dtype must be float16 or bfloat16")


def _validate_sweeps(
    batch_sizes: list[int] | None,
    seq_lens: list[int] | None,
    hidden_sizes: list[int] | None,
    num_heads: list[int] | None,
    ffn_mults: list[int] | None,
) -> None:
    for name, values in {
        "batch_sizes": batch_sizes,
        "seq_lens": seq_lens,
        "hidden_sizes": hidden_sizes,
        "num_heads": num_heads,
        "ffn_mults": ffn_mults,
    }.items():
        if values is None:
            continue
        if not values:
            raise ValueError(f"{name} cannot be empty")
        if any(value <= 0 for value in values):
            raise ValueError(f"{name} values must be positive")


def _validate_timing_args(min_samples: int, min_measurement_seconds: float, max_measurement_seconds: float) -> None:
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if min_measurement_seconds < 0:
        raise ValueError("min_measurement_seconds must be non-negative")
    if max_measurement_seconds <= 0:
        raise ValueError("max_measurement_seconds must be positive")
    if max_measurement_seconds < min_measurement_seconds:
        raise ValueError("max_measurement_seconds must be >= min_measurement_seconds")
