#!/usr/bin/env python
"""CLI tool for running ViT benchmarks."""

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from vit import ViTConfig

from .benchmark import run_full_benchmark
from .plotting import plot_benchmark_results, plot_multi_metric_comparison, plot_throughput_analysis


def parse_resolution(resolution_str: str) -> tuple[int, ...]:
    """Parse resolution string to tuple of integers.

    Args:
        resolution_str: Resolution string like "224" or "224,224" or "224,224,224".

    Returns:
        Tuple of resolution dimensions.
    """
    return tuple(int(x.strip()) for x in resolution_str.split(","))


def save_results_to_csv(results: list, output_path: Path) -> None:
    """Save benchmark results to CSV file.

    Args:
        results: List of BenchmarkResult objects.
        output_path: Path to output CSV file.
    """
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "config_name",
                "batch_size",
                "image_size",
                "pass_mode",
                "latency_ms",
                "memory_mb",
                "gflops",
            ]
        )

        # Write results
        for result in results:
            writer.writerow(
                [
                    result.config_name,
                    result.batch_size,
                    "x".join(map(str, result.image_size)),
                    result.pass_mode,
                    f"{result.latency_ms:.4f}",
                    f"{result.memory_mb:.2f}",
                    f"{result.gflops:.2f}",
                ]
            )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark ViT models with various configurations and resolutions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Path(s) to ViTConfig YAML file(s)",
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        required=True,
        help="Resolution(s) to benchmark (e.g., '224' or '224,224' for 2D, '64,224,224' for 3D)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmarks on",
    )

    parser.add_argument(
        "--pass-mode",
        type=str,
        choices=["forward", "backward", "forward_backward"],
        default="forward",
        help="Type of pass to benchmark",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save results",
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )

    parser.add_argument(
        "--latency-iters",
        type=int,
        default=100,
        help="Number of iterations for latency measurement",
    )

    parser.add_argument(
        "--memory-iters",
        type=int,
        default=10,
        help="Number of iterations for memory measurement",
    )

    parser.add_argument(
        "--plot-formats",
        nargs="+",
        choices=["png", "svg"],
        default=["png", "svg"],
        help="Output format(s) for plots",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster plot formats",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    # Parse resolutions
    resolutions = [parse_resolution(r) for r in args.resolutions]

    # Load configs
    configs = []
    for config_path in args.configs:
        config_path = Path(config_path)
        config = ViTConfig.from_yaml(config_path)
        config_name = config_path.stem
        configs.append((config_name, config))

    print(f"Loaded {len(configs)} configuration(s)")
    print(f"Testing {len(resolutions)} resolution(s)")
    print(f"Device: {args.device}")
    print(f"Pass mode: {args.pass_mode}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Run benchmarks
    all_results = []

    total_benchmarks = len(configs) * len(resolutions)
    with tqdm(total=total_benchmarks, desc="Overall progress") as overall_pbar:
        for config_name, config in configs:
            for resolution in resolutions:
                # Create modified config with new resolution
                config_dict = config.__dict__.copy()
                config_dict["img_size"] = resolution
                modified_config = ViTConfig(**config_dict)

                # Run benchmark
                try:
                    result = run_full_benchmark(
                        config=modified_config,
                        batch_size=args.batch_size,
                        device=args.device,
                        pass_mode=args.pass_mode,
                        num_warmup_iters=args.warmup_iters,
                        num_latency_iters=args.latency_iters,
                        num_memory_iters=args.memory_iters,
                        config_name=config_name,
                        show_progress=False,
                    )
                    all_results.append(result)

                    # Update progress bar with latest result
                    overall_pbar.set_postfix_str(
                        f"{config_name} @ {resolution}: "
                        f"{result.latency_ms:.2f}ms, "
                        f"{result.memory_mb:.0f}MB, "
                        f"{result.gflops:.1f} GFLOPs"
                    )
                except Exception as e:
                    print(f"\nError benchmarking {config_name} @ {resolution}: {e}")

                overall_pbar.update(1)

    print(f"\nCompleted {len(all_results)} benchmark(s)")

    # Save results to CSV
    csv_path = args.output_dir / "benchmark_results.csv"
    save_results_to_csv(all_results, csv_path)
    print(f"Saved results to {csv_path}")

    # Generate plots
    if not args.no_plots and all_results:
        print("\nGenerating plots...")

        try:
            # Individual metric plots
            for metric in ["latency", "memory", "gflops"]:
                paths = plot_benchmark_results(
                    all_results,
                    args.output_dir,
                    metric=metric,  # type: ignore
                    plot_format=args.plot_formats,
                    dpi=args.dpi,
                )
                print(f"  Created {metric} plot(s): {', '.join(str(p) for p in paths)}")

            # Multi-metric comparison
            paths = plot_multi_metric_comparison(
                all_results,
                args.output_dir,
                plot_format=args.plot_formats,
                dpi=args.dpi,
            )
            print(f"  Created comparison plot(s): {', '.join(str(p) for p in paths)}")

            # Throughput analysis
            paths = plot_throughput_analysis(
                all_results,
                args.output_dir,
                plot_format=args.plot_formats,
                dpi=args.dpi,
            )
            print(f"  Created throughput plot(s): {', '.join(str(p) for p in paths)}")

        except Exception as e:
            print(f"Error generating plots: {e}")

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
