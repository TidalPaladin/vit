#!/usr/bin/env python
"""Benchmarking tools for ViT models."""

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .benchmark import (
        BenchmarkResult as BenchmarkResult,
        PassMode as PassMode,
        benchmark_latency as benchmark_latency,
        benchmark_memory as benchmark_memory,
        compute_gflops as compute_gflops,
        create_input_from_config as create_input_from_config,
        run_full_benchmark as run_full_benchmark,
        warmup_model as warmup_model,
    )
    from .component_benchmark import (
        ComparisonResult as ComparisonResult,
        ComparisonSummary as ComparisonSummary,
        ComponentBenchmarkCase as ComponentBenchmarkCase,
        ComponentBenchmarkResult as ComponentBenchmarkResult,
        ComponentBenchmarkStats as ComponentBenchmarkStats,
        compare_benchmark_runs as compare_benchmark_runs,
        configure_runtime as configure_runtime,
        load_benchmark_run as load_benchmark_run,
        run_component_benchmark_case as run_component_benchmark_case,
        run_component_benchmark_suite as run_component_benchmark_suite,
        save_benchmark_run as save_benchmark_run,
    )
    from .plotting import (
        PlotFormat as PlotFormat,
        plot_benchmark_results as plot_benchmark_results,
        plot_multi_metric_comparison as plot_multi_metric_comparison,
        plot_throughput_analysis as plot_throughput_analysis,
    )


_LAZY_EXPORTS = {
    "BenchmarkResult": ("benchmark", "BenchmarkResult"),
    "PassMode": ("benchmark", "PassMode"),
    "benchmark_latency": ("benchmark", "benchmark_latency"),
    "benchmark_memory": ("benchmark", "benchmark_memory"),
    "compute_gflops": ("benchmark", "compute_gflops"),
    "create_input_from_config": ("benchmark", "create_input_from_config"),
    "run_full_benchmark": ("benchmark", "run_full_benchmark"),
    "warmup_model": ("benchmark", "warmup_model"),
    "ComparisonResult": ("component_benchmark", "ComparisonResult"),
    "ComparisonSummary": ("component_benchmark", "ComparisonSummary"),
    "ComponentBenchmarkCase": ("component_benchmark", "ComponentBenchmarkCase"),
    "ComponentBenchmarkResult": ("component_benchmark", "ComponentBenchmarkResult"),
    "ComponentBenchmarkStats": ("component_benchmark", "ComponentBenchmarkStats"),
    "compare_benchmark_runs": ("component_benchmark", "compare_benchmark_runs"),
    "configure_runtime": ("component_benchmark", "configure_runtime"),
    "load_benchmark_run": ("component_benchmark", "load_benchmark_run"),
    "run_component_benchmark_case": ("component_benchmark", "run_component_benchmark_case"),
    "run_component_benchmark_suite": ("component_benchmark", "run_component_benchmark_suite"),
    "save_benchmark_run": ("component_benchmark", "save_benchmark_run"),
    "PlotFormat": ("plotting", "PlotFormat"),
    "plot_benchmark_results": ("plotting", "plot_benchmark_results"),
    "plot_multi_metric_comparison": ("plotting", "plot_multi_metric_comparison"),
    "plot_throughput_analysis": ("plotting", "plot_throughput_analysis"),
}

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "ComparisonSummary",
    "ComponentBenchmarkCase",
    "ComponentBenchmarkResult",
    "ComponentBenchmarkStats",
    "PassMode",
    "PlotFormat",
    "benchmark_latency",
    "benchmark_memory",
    "compare_benchmark_runs",
    "compute_gflops",
    "configure_runtime",
    "create_input_from_config",
    "load_benchmark_run",
    "plot_benchmark_results",
    "plot_multi_metric_comparison",
    "plot_throughput_analysis",
    "run_component_benchmark_case",
    "run_component_benchmark_suite",
    "run_full_benchmark",
    "save_benchmark_run",
    "warmup_model",
]


def __getattr__(name: str) -> Any:
    """Load public benchmark symbols only when requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    attribute = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = attribute
    return attribute


def __dir__() -> list[str]:
    """Include lazily loaded public symbols in interactive discovery."""
    return sorted({*globals(), *__all__})
