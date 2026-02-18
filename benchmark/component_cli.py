#!/usr/bin/env python
"""CLI for criterion-style component benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from vit.norm import NORM_TYPE_CHOICES

from .component_benchmark import (
    DEFAULT_COMPONENTS,
    DEFAULT_PASS_MODES,
    PRESET_CONFIGS,
    ComparisonMetric,
    ComparisonResult,
    ComparisonSummary,
    ComponentKind,
    PassMode,
    benchmark_results_to_records,
    collect_run_metadata,
    compare_benchmark_runs,
    configure_runtime,
    list_baseline_dirs,
    load_benchmark_run,
    resolve_dtype,
    resolve_run_path,
    run_component_benchmark_suite,
    save_benchmark_run,
    save_comparison_report,
)


DEFAULT_OUTPUT_ROOT = Path("benchmark_results/components")
DTYPE_CHOICES = ["float16", "float32", "float64", "bfloat16"]
AUTOCAST_CHOICES = ["none", "float16", "bfloat16"]
METRIC_CHOICES: list[ComparisonMetric] = ["mean_ms", "median_ms", "p95_ms", "std_ms", "memory_mb"]


def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser for component benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark transformer components with criterion-style sampling and baseline comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run component benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("--components", nargs="+", choices=_component_choices(), default=list(DEFAULT_COMPONENTS))
    run_parser.add_argument("--presets", nargs="+", choices=sorted(PRESET_CONFIGS), default=["default"])
    run_parser.add_argument("--pass-modes", nargs="+", choices=_pass_mode_choices(), default=list(DEFAULT_PASS_MODES))

    run_parser.add_argument("--batch-sizes", nargs="+", type=int)
    run_parser.add_argument("--seq-lens", nargs="+", type=int)
    run_parser.add_argument("--hidden-sizes", nargs="+", type=int)
    run_parser.add_argument("--num-heads", nargs="+", type=int)
    run_parser.add_argument("--ffn-mults", nargs="+", type=int)

    run_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    run_parser.add_argument("--param-dtype", choices=DTYPE_CHOICES, default="float32")
    run_parser.add_argument("--input-dtype", choices=DTYPE_CHOICES, default="float32")
    run_parser.add_argument("--autocast-dtype", choices=AUTOCAST_CHOICES, default="none")

    run_parser.add_argument("--activation", type=str)
    run_parser.add_argument("--norm-type", choices=list(NORM_TYPE_CHOICES), default="rmsnorm")
    run_parser.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--hidden-dropout", type=float)
    run_parser.add_argument("--attention-dropout", type=float)
    run_parser.add_argument("--train-mode", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--drop-path-prob", type=float)
    run_parser.add_argument("--layer-scale-init", type=float)
    run_parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--eps", type=float)

    run_parser.add_argument("--warmup-iters", type=int, default=10)
    run_parser.add_argument("--min-samples", type=int, default=30)
    run_parser.add_argument("--min-measurement-seconds", type=float, default=1.0)
    run_parser.add_argument("--max-measurement-seconds", type=float, default=10.0)

    run_parser.add_argument("--include-memory", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--memory-iters", type=int, default=5)

    run_parser.add_argument("--seed", type=int, default=1337)
    run_parser.add_argument("--deterministic", action="store_true")
    run_parser.add_argument("--threads", type=int)

    run_parser.add_argument("--save-as", type=str)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare candidate run with baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare_parser.add_argument("--baseline", required=True, help="Baseline run name or path")
    compare_parser.add_argument("--candidate", help="Candidate run name or path; defaults to latest run")
    compare_parser.add_argument("--metric", choices=METRIC_CHOICES, default="mean_ms")
    compare_parser.add_argument("--regression-threshold-pct", type=float, default=5.0)
    compare_parser.add_argument("--noise-floor-pct", type=float, default=1.0)
    compare_parser.add_argument("--fail-on-regression", action="store_true")
    compare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    list_parser = subparsers.add_parser(
        "list-baselines",
        help="List available benchmark runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    list_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "compare":
        return _compare_command(args)
    if args.command == "list-baselines":
        return _list_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_command(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    param_dtype = resolve_dtype(args.param_dtype)
    input_dtype = resolve_dtype(args.input_dtype)
    autocast_dtype = resolve_dtype(args.autocast_dtype, allow_none=True)
    assert param_dtype is not None
    assert input_dtype is not None

    include_memory = args.include_memory if args.include_memory is not None else device.type == "cuda"

    configure_runtime(seed=args.seed, deterministic=args.deterministic, threads=args.threads)

    metadata = collect_run_metadata(
        seed=args.seed,
        deterministic=args.deterministic,
        threads=args.threads,
        device=device,
        include_memory=include_memory,
    )

    suite_kwargs = {
        "components": args.components,
        "presets": args.presets,
        "pass_modes": args.pass_modes,
        "batch_sizes": args.batch_sizes,
        "seq_lens": args.seq_lens,
        "hidden_sizes": args.hidden_sizes,
        "num_heads": args.num_heads,
        "ffn_mults": args.ffn_mults,
        "activation": args.activation,
        "norm_type": args.norm_type,
        "use_rope": args.use_rope,
        "hidden_dropout": args.hidden_dropout,
        "attention_dropout": args.attention_dropout,
        "train_mode": args.train_mode,
        "drop_path_prob": args.drop_path_prob,
        "layer_scale_init": args.layer_scale_init,
        "bias": args.bias,
        "eps": args.eps,
    }

    results = run_component_benchmark_suite(
        device=device,
        param_dtype=param_dtype,
        input_dtype=input_dtype,
        autocast_dtype=autocast_dtype,
        num_warmup_iters=args.warmup_iters,
        min_samples=args.min_samples,
        min_measurement_seconds=args.min_measurement_seconds,
        max_measurement_seconds=args.max_measurement_seconds,
        include_memory=include_memory,
        num_memory_iters=args.memory_iters,
        **suite_kwargs,
    )

    run_name = args.save_as or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    output_dir = args.output_root / run_name
    json_path, csv_path = save_benchmark_run(output_dir=output_dir, metadata=metadata, results=results)

    records = benchmark_results_to_records(results)
    _print_run_summary(records)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")

    return 0


def _compare_command(args: argparse.Namespace) -> int:
    baseline_path = resolve_run_path(args.output_root, args.baseline)
    candidate_path = resolve_run_path(args.output_root, args.candidate)

    _, baseline_records = load_benchmark_run(baseline_path)
    _, candidate_records = load_benchmark_run(candidate_path)

    comparisons, summary = compare_benchmark_runs(
        baseline_records,
        candidate_records,
        metric=args.metric,
        regression_threshold_pct=args.regression_threshold_pct,
        noise_floor_pct=args.noise_floor_pct,
    )

    _print_compare_summary(summary)
    _print_compare_table(comparisons)

    report_dir = candidate_path if candidate_path.is_dir() else candidate_path.parent
    report_path = report_dir / f"comparison_vs_{baseline_path.name}.json"
    save_comparison_report(
        output_path=report_path,
        baseline_name=baseline_path.name,
        candidate_name=candidate_path.name,
        comparisons=comparisons,
        summary=summary,
    )
    print(f"\nSaved comparison report: {report_path}")

    if args.fail_on_regression and summary.regressions > 0:
        print("Regression gate failed.")
        return 1

    return 0


def _list_command(args: argparse.Namespace) -> int:
    baselines = list_baseline_dirs(args.output_root)
    if not baselines:
        print(f"No benchmark runs found in {args.output_root}")
        return 0

    print(f"Available benchmark runs in {args.output_root}:")
    for run_dir in baselines:
        timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC).isoformat()
        print(f"- {run_dir.name} ({timestamp})")
    return 0


def _print_run_summary(records: list[dict[str, Any]]) -> None:
    print(f"Ran {len(records)} benchmark case(s):")
    print(
        f"{'component':<22} {'preset':<10} {'pass':<16} {'shape':<22} "
        f"{'mean(ms)':>10} {'p95(ms)':>10} {'memory(MB)':>12}"
    )
    print("-" * 112)

    for record in sorted(records, key=lambda item: str(item["case_id"])):
        shape = f"B{record['batch_size']}/S{record['seq_len']}/D{record['hidden_size']}/H{record['num_heads']}"
        memory_value = record.get("memory_mb")
        memory = "-" if memory_value is None else f"{_as_float(memory_value):.2f}"
        print(
            f"{record['component']:<22} {record['preset']:<10} {record['pass_mode']:<16} {shape:<22} "
            f"{_as_float(record['mean_ms']):>10.4f} {_as_float(record['p95_ms']):>10.4f} {memory:>12}"
        )


def _print_compare_summary(summary: ComparisonSummary) -> None:
    print("Comparison summary:")
    print(f"- Metric: {summary.metric}")
    print(f"- Compared cases: {summary.compared_cases}")
    print(f"- Regressions: {summary.regressions}")
    print(f"- Missing in baseline: {summary.missing_in_baseline}")
    print(f"- Missing in candidate: {summary.missing_in_candidate}")


def _print_compare_table(comparisons: list[ComparisonResult]) -> None:
    if not comparisons:
        print("No overlapping comparable cases found.")
        return

    print(f"\n{'case_id':<70} {'baseline':>11} {'candidate':>11} {'delta%':>9} {'regress':>8}")
    print("-" * 120)

    for comparison in sorted(comparisons, key=lambda item: item.delta_pct, reverse=True):
        delta_pct = "inf" if comparison.delta_pct == float("inf") else f"{comparison.delta_pct:.2f}"
        print(
            f"{comparison.case_id:<70} {comparison.baseline_value:>11.4f} {comparison.candidate_value:>11.4f} "
            f"{delta_pct:>9} {str(comparison.is_regression):>8}"
        )


def _component_choices() -> list[ComponentKind]:
    return ["mlp", "self_attention", "layer_scale_residual", "drop_path_residual"]


def _pass_mode_choices() -> list[PassMode]:
    return ["forward", "backward", "forward_backward"]


def _as_float(value: Any) -> float:
    if not isinstance(value, (float, int)):
        raise TypeError(f"Expected numeric value, got: {type(value)!r}")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
