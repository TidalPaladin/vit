from copy import deepcopy
from pathlib import Path

import torch

import benchmark.component_benchmark as component_benchmark
from benchmark.component_benchmark import (
    build_component_benchmark_cases,
    compare_benchmark_runs,
    load_benchmark_run,
    run_component_benchmark_case,
    run_component_benchmark_suite,
    save_benchmark_run,
)
from benchmark.component_cli import main as component_benchmark_main


def test_build_component_benchmark_cases_default_grid() -> None:
    cases = build_component_benchmark_cases()

    assert len(cases) == 6  # 3 shape tiers * 2 default components
    assert {case.component for case in cases} == {"mlp", "self_attention"}
    assert {case.pass_mode for case in cases} == {"forward"}
    assert {case.norm_type for case in cases} == {"rmsnorm"}


def test_run_component_benchmark_case_forward_mlp_cpu() -> None:
    case = build_component_benchmark_cases(
        components=["mlp"],
        pass_modes=["forward"],
        batch_sizes=[1],
        seq_lens=[8],
        hidden_sizes=[32],
        ffn_mults=[2],
        norm_type="layernorm",
    )[0]

    result = run_component_benchmark_case(
        case,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        input_dtype=torch.float32,
        autocast_dtype=None,
        num_warmup_iters=0,
        min_samples=1,
        min_measurement_seconds=0.0,
        max_measurement_seconds=0.05,
        include_memory=False,
    )

    assert result.stats.sample_count >= 1
    assert result.stats.mean_ms > 0
    assert result.stats.memory_mb is None


def test_run_component_benchmark_case_backward_drop_path_cpu() -> None:
    case = build_component_benchmark_cases(
        components=["drop_path_residual"],
        pass_modes=["backward"],
        batch_sizes=[1],
        seq_lens=[8],
        hidden_sizes=[32],
    )[0]

    result = run_component_benchmark_case(
        case,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        input_dtype=torch.float32,
        autocast_dtype=None,
        num_warmup_iters=0,
        min_samples=1,
        min_measurement_seconds=0.0,
        max_measurement_seconds=0.05,
        include_memory=False,
    )

    assert result.stats.sample_count >= 1
    assert result.stats.mean_ms > 0


def test_run_component_benchmark_case_honors_max_measurement_seconds(monkeypatch) -> None:
    case = build_component_benchmark_cases(
        components=["mlp"],
        pass_modes=["forward"],
        batch_sizes=[1],
        seq_lens=[8],
        hidden_sizes=[32],
        ffn_mults=[2],
    )[0]

    monkeypatch.setattr(component_benchmark, "_build_target", lambda *_args, **_kwargs: object())

    state = {"count": 0}

    def fake_execute_iteration(*_args, **_kwargs):
        state["count"] += 1
        return 1.0

    monkeypatch.setattr(component_benchmark, "_execute_iteration", fake_execute_iteration)

    fake_time = {"value": 0.0}

    def fake_perf_counter() -> float:
        current = fake_time["value"]
        fake_time["value"] += 0.03
        return current

    monkeypatch.setattr(component_benchmark.time, "perf_counter", fake_perf_counter)

    min_samples = 100
    result = run_component_benchmark_case(
        case,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        input_dtype=torch.float32,
        autocast_dtype=None,
        num_warmup_iters=0,
        min_samples=min_samples,
        min_measurement_seconds=0.0,
        max_measurement_seconds=0.05,
        include_memory=False,
    )

    assert result.stats.sample_count == state["count"]
    assert result.stats.sample_count < min_samples


def test_save_load_and_compare_component_benchmarks(tmp_path: Path) -> None:
    results = run_component_benchmark_suite(
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        input_dtype=torch.float32,
        autocast_dtype=None,
        components=["mlp"],
        pass_modes=["forward"],
        batch_sizes=[1],
        seq_lens=[8],
        hidden_sizes=[32],
        ffn_mults=[2],
        num_warmup_iters=0,
        min_samples=1,
        min_measurement_seconds=0.0,
        max_measurement_seconds=0.05,
        include_memory=False,
    )

    json_path, csv_path = save_benchmark_run(
        output_dir=tmp_path / "baseline",
        metadata={"suite": "component"},
        results=results,
    )

    assert json_path.exists()
    assert csv_path.exists()

    metadata, records = load_benchmark_run(tmp_path / "baseline")
    assert metadata["suite"] == "component"
    assert len(records) == 1

    candidate_records = deepcopy(records)
    candidate_records[0]["mean_ms"] = records[0]["mean_ms"] * 1.2

    comparisons, summary = compare_benchmark_runs(
        records,
        candidate_records,
        metric="mean_ms",
        regression_threshold_pct=5.0,
        noise_floor_pct=1.0,
    )

    assert summary.compared_cases == 1
    assert summary.regressions == 1
    assert comparisons[0].is_regression


def test_component_cli_run_and_compare(tmp_path: Path) -> None:
    baseline_args = [
        "run",
        "--components",
        "mlp",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "8",
        "--hidden-sizes",
        "32",
        "--ffn-mults",
        "2",
        "--warmup-iters",
        "0",
        "--min-samples",
        "1",
        "--min-measurement-seconds",
        "0",
        "--max-measurement-seconds",
        "0.05",
        "--save-as",
        "baseline",
        "--output-root",
        str(tmp_path),
    ]
    assert component_benchmark_main(baseline_args) == 0

    candidate_args = baseline_args.copy()
    candidate_args[candidate_args.index("baseline")] = "candidate"
    candidate_args.extend(["--norm-type", "layernorm"])
    assert component_benchmark_main(candidate_args) == 0

    compare_args = [
        "compare",
        "--baseline",
        "baseline",
        "--candidate",
        "candidate",
        "--output-root",
        str(tmp_path),
    ]
    assert component_benchmark_main(compare_args) == 0
    assert (tmp_path / "candidate" / "comparison_vs_baseline.json").exists()
