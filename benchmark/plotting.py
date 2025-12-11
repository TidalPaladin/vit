#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for benchmark results."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .benchmark import BenchmarkResult


PlotFormat = Literal["png", "svg"]


def plot_benchmark_results(
    results: list[BenchmarkResult],
    output_dir: Path,
    metric: Literal["latency", "memory", "gflops"] = "latency",
    plot_format: PlotFormat | list[PlotFormat] = ["png", "svg"],
    dpi: int = 300,
) -> list[Path]:
    """Create plots of benchmark results as function of image size.

    Args:
        results: List of benchmark results to plot.
        output_dir: Directory to save plots.
        metric: Which metric to plot.
        plot_format: Output format(s) for plots.
        dpi: DPI for raster formats.

    Returns:
        List of paths to created plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure plot_format is a list
    if isinstance(plot_format, str):
        plot_format = [plot_format]

    # Group results by config name and pass mode
    grouped_results: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.config_name, result.pass_mode)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)

    # Sort results by image size (use product of dimensions)
    for key in grouped_results:
        grouped_results[key].sort(key=lambda r: np.prod(r.image_size))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Metric configuration
    metric_config = {
        "latency": {
            "ylabel": "Latency (ms)",
            "title": "Inference Latency vs Image Size",
            "accessor": lambda r: r.latency_ms,
        },
        "memory": {
            "ylabel": "Memory (MB)",
            "title": "Peak Memory Usage vs Image Size",
            "accessor": lambda r: r.memory_mb,
        },
        "gflops": {
            "ylabel": "GFLOPs",
            "title": "Computational Cost vs Image Size",
            "accessor": lambda r: r.gflops,
        },
    }

    config = metric_config[metric]

    # Plot each config
    for (config_name, pass_mode), group_results in grouped_results.items():
        # Extract data
        image_sizes = [np.prod(r.image_size) for r in group_results]
        values = [config["accessor"](r) for r in group_results]

        # Create label
        label = f"{config_name} ({pass_mode})"

        # Plot line with markers
        ax.plot(image_sizes, values, marker="o", label=label, linewidth=2, markersize=6)

    # Configure plot
    ax.set_xlabel("Image Size (total pixels)", fontsize=12)
    ax.set_ylabel(config["ylabel"], fontsize=12)
    ax.set_title(config["title"], fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale if values span multiple orders of magnitude
    values_all = [config["accessor"](r) for r in results]
    if values_all and min(values_all) > 0 and max(values_all) / min(values_all) > 100:
        ax.set_yscale("log")

    plt.tight_layout()

    # Save in requested formats
    saved_paths = []
    for fmt in plot_format:
        output_path = output_dir / f"benchmark_{metric}.{fmt}"
        plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)

    plt.close(fig)

    return saved_paths


def plot_multi_metric_comparison(
    results: list[BenchmarkResult],
    output_dir: Path,
    plot_format: PlotFormat | list[PlotFormat] = ["png", "svg"],
    dpi: int = 300,
) -> list[Path]:
    """Create multi-panel plot comparing all metrics.

    Args:
        results: List of benchmark results to plot.
        output_dir: Directory to save plots.
        plot_format: Output format(s) for plots.
        dpi: DPI for raster formats.

    Returns:
        List of paths to created plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure plot_format is a list
    if isinstance(plot_format, str):
        plot_format = [plot_format]

    # Group results by config name and pass mode
    grouped_results: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.config_name, result.pass_mode)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)

    # Sort results by image size
    for key in grouped_results:
        grouped_results[key].sort(key=lambda r: np.prod(r.image_size))

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Metric configurations
    metrics = [
        {
            "ax": axes[0],
            "ylabel": "Latency (ms)",
            "title": "Inference Latency",
            "accessor": lambda r: r.latency_ms,
        },
        {
            "ax": axes[1],
            "ylabel": "Memory (MB)",
            "title": "Peak Memory Usage",
            "accessor": lambda r: r.memory_mb,
        },
        {
            "ax": axes[2],
            "ylabel": "GFLOPs",
            "title": "Computational Cost",
            "accessor": lambda r: r.gflops,
        },
    ]

    # Plot each metric
    for metric in metrics:
        ax = metric["ax"]

        for (config_name, pass_mode), group_results in grouped_results.items():
            # Extract data
            image_sizes = [np.prod(r.image_size) for r in group_results]
            values = [metric["accessor"](r) for r in group_results]

            # Create label
            label = f"{config_name} ({pass_mode})"

            # Plot line with markers
            ax.plot(image_sizes, values, marker="o", label=label, linewidth=2, markersize=6)

        # Configure subplot
        ax.set_xlabel("Image Size (total pixels)", fontsize=11)
        ax.set_ylabel(metric["ylabel"], fontsize=11)
        ax.set_title(metric["title"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save in requested formats
    saved_paths = []
    for fmt in plot_format:
        output_path = output_dir / f"benchmark_comparison.{fmt}"
        plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)

    plt.close(fig)

    return saved_paths


def plot_throughput_analysis(
    results: list[BenchmarkResult],
    output_dir: Path,
    plot_format: PlotFormat | list[PlotFormat] = ["png", "svg"],
    dpi: int = 300,
) -> list[Path]:
    """Create throughput analysis plot (samples/second).

    Args:
        results: List of benchmark results to plot.
        output_dir: Directory to save plots.
        plot_format: Output format(s) for plots.
        dpi: DPI for raster formats.

    Returns:
        List of paths to created plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure plot_format is a list
    if isinstance(plot_format, str):
        plot_format = [plot_format]

    # Group results
    grouped_results: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.config_name, result.pass_mode)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)

    # Sort results by image size
    for key in grouped_results:
        grouped_results[key].sort(key=lambda r: np.prod(r.image_size))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each config
    for (config_name, pass_mode), group_results in grouped_results.items():
        # Extract data
        image_sizes = [np.prod(r.image_size) for r in group_results]
        # Convert latency to throughput: (batch_size / latency_ms) * 1000 = samples/sec
        throughputs = [(r.batch_size / r.latency_ms) * 1000 for r in group_results]

        # Create label
        label = f"{config_name} ({pass_mode})"

        # Plot line with markers
        ax.plot(image_sizes, throughputs, marker="o", label=label, linewidth=2, markersize=6)

    # Configure plot
    ax.set_xlabel("Image Size (total pixels)", fontsize=12)
    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Inference Throughput vs Image Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save in requested formats
    saved_paths = []
    for fmt in plot_format:
        output_path = output_dir / f"benchmark_throughput.{fmt}"
        plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)

    plt.close(fig)

    return saved_paths
