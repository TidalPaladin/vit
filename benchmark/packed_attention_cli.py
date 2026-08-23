#!/usr/bin/env python
"""CLI for packed variable-length attention decision benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch

from .packed_attention_benchmark import (
    DEFAULT_METHODS,
    DEFAULT_PASS_MODES,
    PACKED_ATTENTION_SURFACES,
    VIT_SMALL_DEPTH,
    PackedAttentionSurface,
    build_packed_attention_cases,
    decide_packed_backends,
    run_packed_attention_suite,
    save_packed_attention_run,
)


DEFAULT_OUTPUT_ROOT = Path("benchmark_results/components/packed_attention")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark packed attention against padded and per-sequence baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--surfaces", nargs="+", choices=[surface.name for surface in PACKED_ATTENTION_SURFACES])
    parser.add_argument(
        "--profiles", nargs="+", choices=["mild", "high", "outlier"], default=["mild", "high", "outlier"]
    )
    parser.add_argument("--pass-modes", nargs="+", choices=list(DEFAULT_PASS_MODES), default=list(DEFAULT_PASS_MODES))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dense_masking", "padded_bucketing", "per_sequence", "pytorch", "flash_attention"],
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--independent-runs", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--memory-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--save-as")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    surface_names = set(args.surfaces or [surface.name for surface in PACKED_ATTENTION_SURFACES])
    surfaces: list[PackedAttentionSurface] = [
        surface for surface in PACKED_ATTENTION_SURFACES if surface.name in surface_names
    ]
    cases = build_packed_attention_cases(surfaces=surfaces, profiles=args.profiles, pass_modes=args.pass_modes)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    run_name = args.save_as or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    output_dir = args.output_root / run_name
    metadata = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "dtype": args.dtype,
        "independent_runs": args.independent_runs,
        "warmup_iters": args.warmup_iters,
        "samples": args.samples,
        "memory_iters": args.memory_iters,
        "seed": args.seed,
        "input_memory_scope": "method_specific",
        "packing_amortization_depth": VIT_SMALL_DEPTH,
        "complete": False,
    }

    def save_checkpoint(partial_results):
        save_packed_attention_run(
            output_dir,
            partial_results,
            decide_packed_backends(partial_results),
            metadata,
        )

    results = run_packed_attention_suite(
        device=device,
        dtype=dtype,
        methods=args.methods,
        cases=cases,
        independent_runs=args.independent_runs,
        warmup_iters=args.warmup_iters,
        samples=args.samples,
        memory_iters=args.memory_iters,
        seed=args.seed,
        on_result=save_checkpoint,
    )
    decisions = decide_packed_backends(results)
    metadata["complete"] = True
    json_path, csv_path = save_packed_attention_run(output_dir, results, decisions, metadata)

    for decision in decisions:
        status = "SHIP" if decision.ship else "DO NOT SHIP"
        print(f"{decision.backend}: {status}: {decision.reason}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0 if any(decision.ship for decision in decisions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
