#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for benchmark results."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .benchmark import BenchmarkResult


PlotFormat = Literal["png", "svg"]


def _format_image_size(image_size: tuple[int, ...]) -> str:
    """Format image size tuple as string (e.g., (224, 224) -> '224×224')."""
    return "×".join(map(str, image_size))


def plot_benchmark_results(
    results: list[BenchmarkResult],
    output_dir: Path,
    metric: Literal["latency", "memory", "gflops"] = "latency",
    plot_format: PlotFormat | list[PlotFormat] = ["png", "svg"],
    dpi: int = 300,
    log_scale: bool = False,
) -> list[Path]:
    """Create plots of benchmark results as function of image size.

    Args:
        results: List of benchmark results to plot.
        output_dir: Directory to save plots.
        metric: Which metric to plot.
        plot_format: Output format(s) for plots.
        dpi: DPI for raster formats.
        log_scale: Whether to use log scale for y-axis.

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
        grouped_results[key].sort(key=lambda r: int(np.prod(r.image_size)))

    # Collect all unique image sizes (as tuples) for x-axis
    all_image_sizes = sorted(set(r.image_size for r in results), key=lambda s: int(np.prod(s)))
    x_positions = list(range(len(all_image_sizes)))
    x_labels = [_format_image_size(size) for size in all_image_sizes]

    # Create mapping from image_size to x position
    size_to_pos = {size: i for i, size in enumerate(all_image_sizes)}

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Metric configuration
    metric_config = {
        "latency": {
            "ylabel": "Latency (ms)",
            "title": "Inference Latency vs Image Size",
            "accessor": lambda r: r.latency_ms,
            "auto_log": True,
        },
        "memory": {
            "ylabel": "Memory (MB)",
            "title": "Peak Memory Usage vs Image Size",
            "accessor": lambda r: r.memory_mb,
            "auto_log": True,
        },
        "gflops": {
            "ylabel": "GFLOPs",
            "title": "Computational Cost vs Image Size",
            "accessor": lambda r: r.gflops,
            "auto_log": False,  # Linear scale by default for GFLOPs
        },
    }

    config = metric_config[metric]

    # Plot each config
    for (config_name, pass_mode), group_results in grouped_results.items():
        # Extract data with x positions matching image sizes
        x_vals = [size_to_pos[r.image_size] for r in group_results]
        y_vals = [config["accessor"](r) for r in group_results]

        # Create label
        label = f"{config_name} ({pass_mode})"

        # Plot line with markers
        ax.plot(x_vals, y_vals, marker="o", label=label, linewidth=2, markersize=6)

    # Configure plot
    ax.set_xlabel("Image Size", fontsize=12)
    ax.set_ylabel(config["ylabel"], fontsize=12)
    ax.set_title(config["title"], fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Use log scale if requested or auto-determined (but not for gflops by default)
    if log_scale:
        ax.set_yscale("log")
    elif config["auto_log"]:
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
        grouped_results[key].sort(key=lambda r: int(np.prod(r.image_size)))

    # Collect all unique image sizes for x-axis
    all_image_sizes = sorted(set(r.image_size for r in results), key=lambda s: int(np.prod(s)))
    x_positions = list(range(len(all_image_sizes)))
    x_labels = [_format_image_size(size) for size in all_image_sizes]
    size_to_pos = {size: i for i, size in enumerate(all_image_sizes)}

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
            # Extract data with x positions matching image sizes
            x_vals = [size_to_pos[r.image_size] for r in group_results]
            y_vals = [metric["accessor"](r) for r in group_results]

            # Create label
            label = f"{config_name} ({pass_mode})"

            # Plot line with markers
            ax.plot(x_vals, y_vals, marker="o", label=label, linewidth=2, markersize=6)

        # Configure subplot
        ax.set_xlabel("Image Size", fontsize=11)
        ax.set_ylabel(metric["ylabel"], fontsize=11)
        ax.set_title(metric["title"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

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
        grouped_results[key].sort(key=lambda r: int(np.prod(r.image_size)))

    # Collect all unique image sizes for x-axis
    all_image_sizes = sorted(set(r.image_size for r in results), key=lambda s: int(np.prod(s)))
    x_positions = list(range(len(all_image_sizes)))
    x_labels = [_format_image_size(size) for size in all_image_sizes]
    size_to_pos = {size: i for i, size in enumerate(all_image_sizes)}

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each config
    for (config_name, pass_mode), group_results in grouped_results.items():
        # Extract data with x positions matching image sizes
        x_vals = [size_to_pos[r.image_size] for r in group_results]
        # Convert latency to throughput: (batch_size / latency_ms) * 1000 = samples/sec
        throughputs = [(r.batch_size / r.latency_ms) * 1000 for r in group_results]

        # Create label
        label = f"{config_name} ({pass_mode})"

        # Plot line with markers
        ax.plot(x_vals, throughputs, marker="o", label=label, linewidth=2, markersize=6)

    # Configure plot
    ax.set_xlabel("Image Size", fontsize=12)
    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Inference Throughput vs Image Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    plt.tight_layout()

    # Save in requested formats
    saved_paths = []
    for fmt in plot_format:
        output_path = output_dir / f"benchmark_throughput.{fmt}"
        plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)

    plt.close(fig)

    return saved_paths
