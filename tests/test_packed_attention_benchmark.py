from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmark.packed_attention_benchmark import (
    PACKED_ATTENTION_SURFACES,
    VIT_SMALL_DEPTH,
    PackedAttentionBenchmarkResult,
    PackedAttentionCase,
    _make_dense_attention_boundary,
    _make_inputs,
    build_packed_attention_cases,
    decide_packed_backends,
    load_packed_attention_run,
    make_length_vector,
    run_packed_attention_suite,
    save_packed_attention_run,
)


def _result(
    case: PackedAttentionCase,
    method: str,
    latency: float,
    memory: float,
    *,
    run_index: int = 1,
    packing_latency: float = 0.0,
) -> PackedAttentionBenchmarkResult:
    return PackedAttentionBenchmarkResult(
        run_index=run_index,
        case=case,
        method=method,  # type: ignore[arg-type]
        median_ms=latency,
        p95_ms=latency,
        useful_tokens_per_second=case.useful_tokens * 1000 / latency,
        peak_allocated_mb=memory,
        peak_reserved_mb=memory,
        packing_median_ms=packing_latency if method == "pytorch" else None,
        compile_count=1,
    )


def test_default_cases_cover_four_surfaces_three_profiles_and_two_passes() -> None:
    cases = build_packed_attention_cases()

    assert len(cases) == 24
    assert {case.surface for case in cases} == {surface.name for surface in PACKED_ATTENTION_SURFACES}
    assert {case.profile for case in cases} == {"mild", "high", "outlier"}
    assert {case.pass_mode for case in cases} == {"forward", "forward_backward"}


@pytest.mark.parametrize("profile", ["mild", "high", "outlier"])
def test_length_vectors_are_exact_bounded_and_nonempty(profile) -> None:
    for surface in PACKED_ATTENTION_SURFACES:
        lengths = make_length_vector(surface, profile)
        assert len(lengths) == surface.batch_size
        assert min(lengths) >= 1
        assert max(lengths) == surface.max_seqlen


def test_decision_gate_accepts_latency_win_without_case_regression() -> None:
    cases = [
        PackedAttentionCase("a", "mild", "forward", (8, 7)),
        PackedAttentionCase("b", "high", "forward_backward", (12, 3)),
    ]
    results = []
    for run_index in range(1, 4):
        for case in cases:
            results.extend(
                (
                    _result(case, "dense_masking", 10, 100, run_index=run_index),
                    _result(case, "pytorch", 8.5, 95, run_index=run_index),
                )
            )

    pytorch_decision = decide_packed_backends(results)[0]

    assert pytorch_decision.ship is True
    assert pytorch_decision.latency_ratio == pytest.approx(0.85)


def test_decision_gate_rejects_one_representative_regression() -> None:
    fast_case = PackedAttentionCase("a", "high", "forward", (8, 2))
    slow_case = PackedAttentionCase("b", "mild", "forward", (8, 7))
    results = [
        _result(fast_case, "dense_masking", 10, 100),
        _result(fast_case, "pytorch", 5, 70),
        _result(slow_case, "dense_masking", 10, 100),
        _result(slow_case, "pytorch", 10.6, 70),
    ]

    pytorch_decision = decide_packed_backends(results)[0]

    assert pytorch_decision.ship is False
    assert pytorch_decision.worst_case_latency_ratio == pytest.approx(1.06)


def test_decision_gate_amortizes_one_pack_across_vit_small_blocks() -> None:
    case = PackedAttentionCase("a", "high", "forward", (8, 2))
    packing_latency = 24.0
    results = [
        _result(case, "dense_masking", 10, 100),
        _result(case, "pytorch", 8.5, 95, packing_latency=packing_latency),
    ]

    pytorch_decision = decide_packed_backends(results)[0]

    expected_ratio = (8.5 + packing_latency / VIT_SMALL_DEPTH) / 10
    assert pytorch_decision.ship is False
    assert pytorch_decision.latency_ratio == pytest.approx(expected_ratio)


def test_dense_benchmark_boundaries_use_distinct_compiler_code_objects() -> None:
    first = _make_dense_attention_boundary("first")
    second = _make_dense_attention_boundary("second")

    assert first.__code__ is not second.__code__


def test_packed_benchmark_artifact_round_trip(tmp_path: Path) -> None:
    case = PackedAttentionCase("a", "mild", "forward", (8, 7))
    results = [_result(case, "dense_masking", 10, 100), _result(case, "pytorch", 8, 80)]
    decisions = decide_packed_backends(results)

    _, csv_path = save_packed_attention_run(tmp_path, results, decisions, {"complete": True})
    metadata, restored = load_packed_attention_run(tmp_path)

    assert metadata == {"complete": True}
    assert restored == results
    assert b"\r\n" not in csv_path.read_bytes()


@pytest.mark.cuda
def test_small_cuda_suite_records_packed_and_dense_metrics() -> None:
    case = PackedAttentionCase("test", "high", "forward_backward", (9, 5, 2), hidden_size=64, num_heads=4)

    results = run_packed_attention_suite(
        device=torch.device("cuda"),
        methods=("dense_masking", "pytorch"),
        cases=(case,),
        independent_runs=1,
        warmup_iters=1,
        samples=2,
        memory_iters=1,
    )

    assert {result.method for result in results} == {"dense_masking", "pytorch"}
    assert all(result.error is None for result in results)
    assert all(result.median_ms is not None and result.median_ms > 0 for result in results)
    assert results[1].packing_median_ms is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("method", "expects_padded", "expects_packed"),
    [
        ("dense_masking", True, False),
        ("padded_bucketing", True, False),
        ("per_sequence", False, True),
        ("pytorch", False, True),
    ],
)
def test_benchmark_allocates_only_the_selected_input_representation(method, expects_padded, expects_packed) -> None:
    case = PackedAttentionCase("test", "high", "forward", (9, 5, 2), hidden_size=64, num_heads=4)

    padded, validity, packed = _make_inputs(case, method, torch.device("cuda"), torch.bfloat16)

    assert (padded is not None) is expects_padded
    assert (validity is not None) is expects_padded
    assert (packed is not None) is expects_packed
