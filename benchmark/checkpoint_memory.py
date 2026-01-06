#!/usr/bin/env python
"""Benchmark activation checkpointing memory savings.

This script compares memory usage and latency between models with and without
activation checkpointing, reporting the memory savings and compute overhead.

Usage:
    uv run python -m benchmark.checkpoint_memory
    uv run python -m benchmark.checkpoint_memory --batch-sizes 1 2 4 8 --depths 6 12 24
"""

import argparse
from dataclasses import dataclass

import torch

from vit import ViT, ViTConfig


@dataclass
class CheckpointBenchmarkResult:
    """Results from checkpointing benchmark."""

    depth: int
    hidden_size: int
    batch_size: int
    memory_no_ckpt_mb: float
    memory_ckpt_mb: float
    memory_savings_mb: float
    memory_savings_pct: float
    latency_no_ckpt_ms: float
    latency_ckpt_ms: float
    latency_overhead_pct: float


def measure_memory_and_latency(
    config: ViTConfig,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_iters: int = 5,
) -> tuple[float, float]:
    """Measure peak memory and average latency for a forward+backward pass.

    Args:
        config: ViT configuration.
        batch_size: Batch size.
        device: CUDA device.
        num_warmup: Number of warmup iterations.
        num_iters: Number of measurement iterations.

    Returns:
        Tuple of (peak_memory_mb, avg_latency_ms).
    """
    model = ViT(config, device=device)
    model.train()

    x = torch.randn(batch_size, config.in_channels, *config.img_size, device=device, dtype=config.dtype)

    # Warmup
    for _ in range(num_warmup):
        out = model(x)
        out.dense_features.sum().backward()
        model.zero_grad()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Measure
    latencies = []
    for _ in range(num_iters):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = model(x)
        out.dense_features.sum().backward()
        end.record()

        torch.cuda.synchronize(device)
        latencies.append(start.elapsed_time(end))
        model.zero_grad()

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    avg_latency_ms = sum(latencies) / len(latencies)

    del model, x
    torch.cuda.empty_cache()

    return peak_memory_mb, avg_latency_ms


def run_checkpoint_benchmark(
    depth: int,
    hidden_size: int,
    batch_size: int,
    device: torch.device,
    img_size: tuple[int, int] = (224, 224),
    num_warmup: int = 3,
    num_iters: int = 5,
) -> CheckpointBenchmarkResult:
    """Run benchmark comparing checkpointed vs non-checkpointed model.

    Args:
        depth: Number of transformer layers.
        hidden_size: Hidden dimension size.
        batch_size: Batch size.
        device: CUDA device.
        img_size: Input image size.
        num_warmup: Number of warmup iterations.
        num_iters: Number of measurement iterations.

    Returns:
        Benchmark results.
    """
    base_config = ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        img_size=img_size,
        depth=depth,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=hidden_size // 64,
        pos_enc="rope",
        activation_checkpointing=False,
        dtype=torch.bfloat16,
    )

    ckpt_config = ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        img_size=img_size,
        depth=depth,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=hidden_size // 64,
        pos_enc="rope",
        activation_checkpointing=True,
        dtype=torch.bfloat16,
    )

    # Measure without checkpointing
    mem_no_ckpt, lat_no_ckpt = measure_memory_and_latency(base_config, batch_size, device, num_warmup, num_iters)

    # Measure with checkpointing
    mem_ckpt, lat_ckpt = measure_memory_and_latency(ckpt_config, batch_size, device, num_warmup, num_iters)

    # Calculate savings/overhead
    memory_savings_mb = mem_no_ckpt - mem_ckpt
    memory_savings_pct = (memory_savings_mb / mem_no_ckpt) * 100 if mem_no_ckpt > 0 else 0
    latency_overhead_pct = ((lat_ckpt - lat_no_ckpt) / lat_no_ckpt) * 100 if lat_no_ckpt > 0 else 0

    return CheckpointBenchmarkResult(
        depth=depth,
        hidden_size=hidden_size,
        batch_size=batch_size,
        memory_no_ckpt_mb=mem_no_ckpt,
        memory_ckpt_mb=mem_ckpt,
        memory_savings_mb=memory_savings_mb,
        memory_savings_pct=memory_savings_pct,
        latency_no_ckpt_ms=lat_no_ckpt,
        latency_ckpt_ms=lat_ckpt,
        latency_overhead_pct=latency_overhead_pct,
    )


def print_results_table(results: list[CheckpointBenchmarkResult]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print("ACTIVATION CHECKPOINTING BENCHMARK RESULTS")
    print("=" * 120)

    # Header
    print(
        f"{'Depth':>6} {'Hidden':>7} {'Batch':>6} | "
        f"{'Mem (no ckpt)':>14} {'Mem (ckpt)':>12} {'Savings':>10} {'Savings %':>10} | "
        f"{'Lat (no ckpt)':>14} {'Lat (ckpt)':>12} {'Overhead %':>11}"
    )
    print("-" * 120)

    for r in results:
        print(
            f"{r.depth:>6} {r.hidden_size:>7} {r.batch_size:>6} | "
            f"{r.memory_no_ckpt_mb:>11.1f} MB {r.memory_ckpt_mb:>9.1f} MB "
            f"{r.memory_savings_mb:>7.1f} MB {r.memory_savings_pct:>9.1f}% | "
            f"{r.latency_no_ckpt_ms:>11.2f} ms {r.latency_ckpt_ms:>9.2f} ms "
            f"{r.latency_overhead_pct:>10.1f}%"
        )

    print("=" * 120)

    # Summary statistics
    if results:
        avg_savings = sum(r.memory_savings_pct for r in results) / len(results)
        avg_overhead = sum(r.latency_overhead_pct for r in results) / len(results)
        max_savings = max(r.memory_savings_pct for r in results)
        print("\nSummary:")
        print(f"  Average memory savings: {avg_savings:.1f}%")
        print(f"  Maximum memory savings: {max_savings:.1f}%")
        print(f"  Average latency overhead: {avg_overhead:.1f}%")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark activation checkpointing memory savings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[6, 12, 24],
        help="Transformer depths to benchmark",
    )

    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[384, 768],
        help="Hidden sizes to benchmark",
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Batch sizes to benchmark",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (square)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use",
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )

    parser.add_argument(
        "--measure-iters",
        type=int,
        default=5,
        help="Number of measurement iterations",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for memory benchmarking")
        return

    device = torch.device(args.device)
    img_size = (args.img_size, args.img_size)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Image size: {img_size}")
    print(f"Depths: {args.depths}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Batch sizes: {args.batch_sizes}")
    print()

    results = []
    total = len(args.depths) * len(args.hidden_sizes) * len(args.batch_sizes)
    current = 0

    for depth in args.depths:
        for hidden_size in args.hidden_sizes:
            for batch_size in args.batch_sizes:
                current += 1
                print(
                    f"[{current}/{total}] depth={depth}, hidden={hidden_size}, batch={batch_size}...",
                    end=" ",
                    flush=True,
                )

                try:
                    result = run_checkpoint_benchmark(
                        depth=depth,
                        hidden_size=hidden_size,
                        batch_size=batch_size,
                        device=device,
                        img_size=img_size,
                        num_warmup=args.warmup_iters,
                        num_iters=args.measure_iters,
                    )
                    results.append(result)
                    print(f"savings={result.memory_savings_pct:.1f}%, overhead={result.latency_overhead_pct:.1f}%")
                except torch.cuda.OutOfMemoryError:
                    print("OOM (skipped)")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"ERROR: {e}")

    print_results_table(results)


if __name__ == "__main__":
    main()
