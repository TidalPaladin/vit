"""Decision benchmarks for packed variable-length self-attention."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from types import FunctionType
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from vit.attention import SelfAttention, _get_qk_norm_inputs, attention_qkv_packed
from vit.norm import get_norm_bias
from vit.packed import PackedSequence


PackedMethod = Literal["dense_masking", "padded_bucketing", "per_sequence", "pytorch", "flash_attention"]
PackedPassMode = Literal["forward", "forward_backward"]
RaggedProfile = Literal["mild", "high", "outlier"]

BASELINE_METHODS: tuple[PackedMethod, ...] = ("dense_masking", "padded_bucketing", "per_sequence")
PACKED_METHODS: tuple[PackedMethod, ...] = ("pytorch", "flash_attention")
DEFAULT_METHODS: tuple[PackedMethod, ...] = (*BASELINE_METHODS, "pytorch")
DEFAULT_PASS_MODES: tuple[PackedPassMode, ...] = ("forward", "forward_backward")
VIT_SMALL_HIDDEN_SIZE = 384
VIT_SMALL_NUM_HEADS = 6
VIT_SMALL_DEPTH = 12
PACKED_LATENCY_WIN_RATIO = 0.90
PACKED_LATENCY_PARITY_RATIO = 1.03
PACKED_MEMORY_WIN_RATIO = 0.85
PACKED_MAX_CASE_REGRESSION_RATIO = 1.05


@dataclass(frozen=True)
class PackedAttentionSurface:
    name: str
    batch_size: int
    max_seqlen: int


PACKED_ATTENTION_SURFACES: tuple[PackedAttentionSurface, ...] = (
    PackedAttentionSurface("b64_l196", 64, 196),
    PackedAttentionSurface("b16_l1024", 16, 1024),
    PackedAttentionSurface("b4_l4096", 4, 4096),
    PackedAttentionSurface("b2_l10000", 2, 10000),
)


@dataclass(frozen=True)
class PackedAttentionCase:
    surface: str
    profile: RaggedProfile
    pass_mode: PackedPassMode
    lengths: tuple[int, ...]
    hidden_size: int = VIT_SMALL_HIDDEN_SIZE
    num_heads: int = VIT_SMALL_NUM_HEADS

    @property
    def case_id(self) -> str:
        return f"{self.surface}|{self.profile}|{self.pass_mode}"

    @property
    def useful_tokens(self) -> int:
        return sum(self.lengths)


@dataclass(frozen=True)
class PackedAttentionBenchmarkResult:
    run_index: int
    case: PackedAttentionCase
    method: PackedMethod
    median_ms: float | None
    p95_ms: float | None
    useful_tokens_per_second: float | None
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    packing_median_ms: float | None
    compile_count: int
    error: str | None = None


@dataclass(frozen=True)
class PackedBackendDecision:
    backend: PackedMethod
    ship: bool
    latency_ratio: float | None
    memory_ratio: float | None
    worst_case_latency_ratio: float | None
    reason: str


@dataclass
class _Target:
    forward: Callable[[], Tensor]
    zero_grad: Callable[[], None]


_dense_attention_impl = getattr(cast(Any, attention_qkv_packed), "__wrapped__", attention_qkv_packed)


def _dense_attention_boundary(*args: Any) -> Tensor:
    return _dense_attention_impl(*args)


def _make_dense_attention_boundary(name: str) -> Callable[..., Tensor]:
    """Create a unique code object so one benchmark case cannot exhaust another's cache."""
    code = _dense_attention_boundary.__code__.replace(co_name=name, co_qualname=name)
    return cast(Callable[..., Tensor], FunctionType(code, _dense_attention_boundary.__globals__, name))


@cache
def _compiled_dense_attention_boundary(case_id: str, method: PackedMethod) -> Callable[..., Tensor]:
    name = f"_packed_benchmark_{method}_{case_id}"
    return torch.compile(fullgraph=True, dynamic=True)(_make_dense_attention_boundary(name))


def make_length_vector(surface: PackedAttentionSurface, profile: RaggedProfile) -> tuple[int, ...]:
    """Return an exact deterministic length vector for one benchmark profile."""
    batch_size = surface.batch_size
    maximum = surface.max_seqlen
    if profile == "mild":
        fractions = (1.0, 0.96, 0.92, 0.88, 0.84)
        return tuple(max(1, round(maximum * fractions[index % len(fractions)])) for index in range(batch_size))
    if profile == "high":
        fractions = (1.0, 0.75, 0.55, 0.4, 0.3, 0.2, 0.125, 0.0625)
        return tuple(max(1, round(maximum * fractions[index % len(fractions)])) for index in range(batch_size))
    if profile == "outlier":
        short_length = max(1, maximum // 4)
        return (maximum, *(short_length for _ in range(batch_size - 1)))
    raise ValueError(f"unsupported ragged profile: {profile}")


def build_packed_attention_cases(
    *,
    surfaces: Sequence[PackedAttentionSurface] = PACKED_ATTENTION_SURFACES,
    profiles: Sequence[RaggedProfile] = ("mild", "high", "outlier"),
    pass_modes: Sequence[PackedPassMode] = DEFAULT_PASS_MODES,
) -> list[PackedAttentionCase]:
    return [
        PackedAttentionCase(surface.name, profile, pass_mode, make_length_vector(surface, profile))
        for surface in surfaces
        for profile in profiles
        for pass_mode in pass_modes
    ]


def run_packed_attention_suite(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    methods: Sequence[PackedMethod] = DEFAULT_METHODS,
    cases: Sequence[PackedAttentionCase] | None = None,
    independent_runs: int = 3,
    warmup_iters: int = 3,
    samples: int = 10,
    memory_iters: int = 3,
    seed: int = 1337,
    on_result: Callable[[Sequence[PackedAttentionBenchmarkResult]], None] | None = None,
) -> list[PackedAttentionBenchmarkResult]:
    """Run deterministic packed and padded candidates with identical weights and inputs."""
    if device.type != "cuda":
        raise RuntimeError("packed attention decision benchmarks require CUDA")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("packed attention decision benchmarks require BF16 or FP16")
    selected_cases = build_packed_attention_cases() if cases is None else list(cases)
    results: list[PackedAttentionBenchmarkResult] = []
    for run_index in range(1, independent_runs + 1):
        for case_index, case in enumerate(selected_cases):
            case_seed = seed + run_index * 10_000 + case_index
            torch.manual_seed(case_seed)
            module = SelfAttention(
                case.hidden_size,
                case.num_heads,
                hidden_dropout=0.1,
                attention_dropout=0.1,
                device=device,
                dtype=dtype,
            )
            module.train(case.pass_mode == "forward_backward")
            for method in methods:
                torch.manual_seed(case_seed + 1)
                padded, validity, packed = _make_inputs(case, method, device, dtype)
                result = run_packed_attention_case(
                    case,
                    method,
                    module,
                    padded,
                    validity,
                    packed,
                    run_index=run_index,
                    device=device,
                    warmup_iters=warmup_iters,
                    samples=samples,
                    memory_iters=memory_iters,
                )
                results.append(result)
                if on_result is not None:
                    on_result(results)
    return results


def run_packed_attention_case(
    case: PackedAttentionCase,
    method: PackedMethod,
    module: SelfAttention,
    padded: Tensor | None,
    validity: Tensor | None,
    packed: PackedSequence | None,
    *,
    run_index: int,
    device: torch.device,
    warmup_iters: int,
    samples: int,
    memory_iters: int,
) -> PackedAttentionBenchmarkResult:
    """Measure one backend/baseline, recording OOM as evidence instead of aborting."""
    if method == "flash_attention" and not _flash_attention_installed():
        return _error_result(run_index, case, method, "optional flash-attn package is not installed")
    target = _build_target(case, method, module, padded, validity, packed)
    padded = None
    validity = None
    packed = None
    torch.cuda.empty_cache()
    torch._dynamo.utils.counters.clear()
    try:
        for _ in range(warmup_iters):
            _execute(target, case.pass_mode)
        latencies = [_time_iteration(target, case.pass_mode, device) for _ in range(samples)]
        peak_allocated_mb, peak_reserved_mb = _measure_memory(
            target,
            case.pass_mode,
            device,
            memory_iters,
        )
        packing_median_ms = None
        if method in PACKED_METHODS:
            packing_padded, packing_validity, _ = _make_inputs(
                case,
                "dense_masking",
                device,
                module.qkv_proj.weight.dtype,
            )
            assert packing_padded is not None and packing_validity is not None
            packing_median_ms = _measure_packing(packing_padded, packing_validity, device, samples)
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        return _error_result(run_index, case, method, f"CUDA OOM: {error}")

    median_ms = float(statistics.median(latencies))
    p95_ms = _percentile(latencies, 0.95)
    useful_tokens_per_second = case.useful_tokens / (median_ms / 1000.0)
    return PackedAttentionBenchmarkResult(
        run_index=run_index,
        case=case,
        method=method,
        median_ms=median_ms,
        p95_ms=p95_ms,
        useful_tokens_per_second=useful_tokens_per_second,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        packing_median_ms=packing_median_ms,
        compile_count=int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
    )


def decide_packed_backends(results: Sequence[PackedAttentionBenchmarkResult]) -> tuple[PackedBackendDecision, ...]:
    """Apply the approved balanced ship gate against the best baseline per case."""
    grouped: dict[tuple[str, PackedMethod], list[PackedAttentionBenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.case.case_id, result.method), []).append(result)

    decisions: list[PackedBackendDecision] = []
    for backend in PACKED_METHODS:
        latency_ratios: list[float] = []
        memory_ratios: list[float] = []
        missing_or_failed = False
        case_ids = sorted({result.case.case_id for result in results})
        for case_id in case_ids:
            candidate = grouped.get((case_id, backend), [])
            baseline_groups = [grouped.get((case_id, baseline), []) for baseline in BASELINE_METHODS]
            candidate_execution_latency = _median_metric(candidate, "median_ms")
            candidate_packing_latency = _median_metric(candidate, "packing_median_ms")
            candidate_latency = (
                None
                if candidate_execution_latency is None or candidate_packing_latency is None
                else candidate_execution_latency + candidate_packing_latency / VIT_SMALL_DEPTH
            )
            candidate_memory = _median_metric(candidate, "peak_allocated_mb")
            valid_baselines = [group for group in baseline_groups if _median_metric(group, "median_ms") is not None]
            if candidate_latency is None or candidate_memory is None or not valid_baselines:
                missing_or_failed = True
                continue
            best_latency_group = min(valid_baselines, key=lambda group: _required_median_metric(group, "median_ms"))
            baseline_latency = _required_median_metric(best_latency_group, "median_ms")
            baseline_memory = _required_median_metric(best_latency_group, "peak_allocated_mb")
            latency_ratios.append(candidate_latency / baseline_latency)
            memory_ratios.append(candidate_memory / baseline_memory)

        if missing_or_failed or not latency_ratios:
            decisions.append(
                PackedBackendDecision(backend, False, None, None, None, "candidate is unavailable or failed a case")
            )
            continue
        aggregate_latency = statistics.geometric_mean(latency_ratios)
        aggregate_memory = statistics.geometric_mean(memory_ratios)
        worst_latency = max(latency_ratios)
        latency_win = aggregate_latency <= PACKED_LATENCY_WIN_RATIO
        balanced_win = aggregate_latency <= PACKED_LATENCY_PARITY_RATIO and aggregate_memory <= PACKED_MEMORY_WIN_RATIO
        no_regression = worst_latency <= PACKED_MAX_CASE_REGRESSION_RATIO
        ship = no_regression and (latency_win or balanced_win)
        reason = (
            "passes latency gate"
            if ship and latency_win
            else "passes latency-parity and memory gate"
            if ship
            else "does not pass the balanced performance gate"
        )
        decisions.append(
            PackedBackendDecision(backend, ship, aggregate_latency, aggregate_memory, worst_latency, reason)
        )
    return tuple(decisions)


def save_packed_attention_run(
    output_dir: Path,
    results: Sequence[PackedAttentionBenchmarkResult],
    decisions: Sequence[PackedBackendDecision],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [_result_record(result) for result in results]
    json_path = output_dir / "packed_attention_results.json"
    csv_path = output_dir / "packed_attention_results.csv"
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "gate": {
                    "latency_win_ratio": PACKED_LATENCY_WIN_RATIO,
                    "latency_parity_ratio": PACKED_LATENCY_PARITY_RATIO,
                    "memory_win_ratio": PACKED_MEMORY_WIN_RATIO,
                    "max_case_regression_ratio": PACKED_MAX_CASE_REGRESSION_RATIO,
                    "packing_amortization_depth": VIT_SMALL_DEPTH,
                },
                "decisions": [asdict(decision) for decision in decisions],
                "results": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def load_packed_attention_run(path: Path) -> tuple[dict[str, Any], list[PackedAttentionBenchmarkResult]]:
    """Load a packed benchmark JSON artifact for aggregation or review."""
    json_path = path if path.is_file() else path / "packed_attention_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    records = payload.get("results")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise ValueError(f"invalid packed attention benchmark artifact: {json_path}")
    results = []
    for record in records:
        case = PackedAttentionCase(
            surface=record["surface"],
            profile=record["profile"],
            pass_mode=record["pass_mode"],
            lengths=tuple(json.loads(record["lengths"])),
            hidden_size=int(record["hidden_size"]),
            num_heads=int(record["num_heads"]),
        )
        results.append(
            PackedAttentionBenchmarkResult(
                run_index=int(record["run_index"]),
                case=case,
                method=record["method"],
                median_ms=record["median_ms"],
                p95_ms=record["p95_ms"],
                useful_tokens_per_second=record["useful_tokens_per_second"],
                peak_allocated_mb=record["peak_allocated_mb"],
                peak_reserved_mb=record["peak_reserved_mb"],
                packing_median_ms=record["packing_median_ms"],
                compile_count=int(record["compile_count"]),
                error=record["error"],
            )
        )
    return metadata, results


def _make_inputs(
    case: PackedAttentionCase,
    method: PackedMethod,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor | None, Tensor | None, PackedSequence | None]:
    maximum = max(case.lengths)
    padded = torch.randn(len(case.lengths), maximum, case.hidden_size, device=device, dtype=dtype, requires_grad=True)
    positions = torch.arange(maximum, device=device)
    validity = positions.unsqueeze(0) < torch.tensor(case.lengths, device=device).unsqueeze(1)
    if method in ("dense_masking", "padded_bucketing"):
        return padded, validity, None
    packed_values = padded.detach()[validity].clone().requires_grad_()
    packed = PackedSequence.from_lengths(packed_values, case.lengths)
    return None, None, packed


def _build_target(
    case: PackedAttentionCase,
    method: PackedMethod,
    module: SelfAttention,
    padded: Tensor | None,
    validity: Tensor | None,
    packed: PackedSequence | None,
) -> _Target:
    buckets: tuple[tuple[Tensor, Tensor], ...] = ()
    if method == "dense_masking":
        if padded is None or validity is None:
            raise ValueError("dense masking requires padded values and validity")
        attention_mask = validity[:, None, None, :]
        attention = _compiled_dense_attention_boundary(case.case_id, method)
        input_tensors = (padded,)

        def forward() -> Tensor:
            return _benchmark_dense_attention(module, padded, attention_mask, attention)[validity]

    elif method == "padded_bucketing":
        if padded is None or validity is None:
            raise ValueError("padded bucketing requires padded values and validity")
        buckets = _make_buckets(padded, validity.sum(dim=1).tolist())
        attention = _compiled_dense_attention_boundary(case.case_id, method)
        input_tensors = tuple(bucket_values for bucket_values, _ in buckets)

        def forward() -> Tensor:
            outputs = []
            for bucket_values, bucket_validity in buckets:
                mask = bucket_validity[:, None, None, :]
                outputs.append(_benchmark_dense_attention(module, bucket_values, mask, attention)[bucket_validity])
            return torch.cat(outputs)

    elif method == "per_sequence":
        if packed is None:
            raise ValueError("per-sequence execution requires packed values")
        sequences = packed.unbind()
        attention = _compiled_dense_attention_boundary(case.case_id, method)
        input_tensors = (packed.values,)

        def forward() -> Tensor:
            return torch.cat(
                [
                    _benchmark_dense_attention(module, sequence.unsqueeze(0), None, attention).squeeze(0)
                    for sequence in sequences
                ]
            )

    elif method in PACKED_METHODS:
        if packed is None:
            raise ValueError("packed attention requires packed values")
        backend = "pytorch" if method == "pytorch" else "flash_attention"
        input_tensors = (packed.values,)

        def forward() -> Tensor:
            return module._forward_packed_candidate(packed, backend=backend).values

    else:
        raise ValueError(f"unsupported packed attention benchmark method: {method}")

    def zero_grad() -> None:
        module.zero_grad(set_to_none=True)
        for input_tensor in input_tensors:
            input_tensor.grad = None

    return _Target(forward, zero_grad)


def _benchmark_dense_attention(
    module: SelfAttention,
    values: Tensor,
    attention_mask: Tensor | None,
    attention: Callable[..., Tensor],
) -> Tensor:
    q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, qk_eps = _get_qk_norm_inputs(module.q_norm, module.k_norm)
    return attention(
        values,
        module.qkv_proj.weight,
        module.qkv_proj.bias,
        module.norm.weight,
        get_norm_bias(module.norm),
        module._use_layer_norm,
        module._head_dim,
        module.out_proj.weight,
        module.out_proj.bias,
        attention_mask,
        module.norm.eps or 1e-5,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        module._use_layer_norm,
        qk_eps,
        module._qk_normalization,
        module.attention_dropout.p,
        module.dropout.p,
        module.training,
        None,
    )


def _make_buckets(padded: Tensor, lengths: Sequence[int]) -> tuple[tuple[Tensor, Tensor], ...]:
    grouped: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        bucket_length = 1 << (length - 1).bit_length()
        grouped.setdefault(bucket_length, []).append(index)
    buckets = []
    for bucket_length, indices in sorted(grouped.items()):
        index_tensor = torch.tensor(indices, device=padded.device)
        values = padded.detach().index_select(0, index_tensor)
        values = F.pad(values, (0, 0, 0, bucket_length - values.shape[1])).requires_grad_()
        sequence_lengths = torch.tensor([lengths[index] for index in indices], device=padded.device)
        validity = torch.arange(bucket_length, device=padded.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        buckets.append((values, validity))
    return tuple(buckets)


def _execute(target: _Target, pass_mode: PackedPassMode) -> None:
    target.zero_grad()
    if pass_mode == "forward":
        with torch.inference_mode():
            target.forward()
    else:
        target.forward().float().mean().backward()
    target.zero_grad()


def _time_iteration(target: _Target, pass_mode: PackedPassMode, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    _execute(target, pass_mode)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def _measure_memory(
    target: _Target,
    pass_mode: PackedPassMode,
    device: torch.device,
    iterations: int,
) -> tuple[float, float]:
    peak_allocated = 0
    peak_reserved = 0
    for _ in range(iterations):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        _execute(target, pass_mode)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
    scale = 1024**2
    return peak_allocated / scale, peak_reserved / scale


def _measure_packing(padded: Tensor, validity: Tensor, device: torch.device, samples: int) -> float:
    latencies = []
    for _ in range(samples):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        PackedSequence.from_padded(padded, validity)
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(latencies))


def _flash_attention_installed() -> bool:
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func  # pyright: ignore[reportMissingImports] # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return float(ordered[index])


def _median_metric(results: Sequence[PackedAttentionBenchmarkResult], name: str) -> float | None:
    values = [getattr(result, name) for result in results if result.error is None and getattr(result, name) is not None]
    return float(statistics.median(values)) if values else None


def _required_median_metric(results: Sequence[PackedAttentionBenchmarkResult], name: str) -> float:
    value = _median_metric(results, name)
    if value is None:
        raise ValueError(f"missing required benchmark metric: {name}")
    return value


def _error_result(
    run_index: int,
    case: PackedAttentionCase,
    method: PackedMethod,
    error: str,
) -> PackedAttentionBenchmarkResult:
    return PackedAttentionBenchmarkResult(run_index, case, method, None, None, None, None, None, None, 0, error)


def _result_record(result: PackedAttentionBenchmarkResult) -> dict[str, Any]:
    return {
        "run_index": result.run_index,
        "case_id": result.case.case_id,
        "surface": result.case.surface,
        "profile": result.case.profile,
        "pass_mode": result.case.pass_mode,
        "lengths": json.dumps(result.case.lengths),
        "useful_tokens": result.case.useful_tokens,
        "hidden_size": result.case.hidden_size,
        "num_heads": result.case.num_heads,
        "method": result.method,
        "median_ms": result.median_ms,
        "p95_ms": result.p95_ms,
        "useful_tokens_per_second": result.useful_tokens_per_second,
        "peak_allocated_mb": result.peak_allocated_mb,
        "peak_reserved_mb": result.peak_reserved_mb,
        "packing_median_ms": result.packing_median_ms,
        "compile_count": result.compile_count,
        "error": result.error,
    }
