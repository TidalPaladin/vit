"""Measure explainability latency and CUDA peak memory without fixed thresholds."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vit import ViT, ViTConfig
from vit.explain import IntegratedGradients, Intervention, LeGrad, PatchOcclusion, ViTExplainer


@dataclass(frozen=True)
class Result:
    fixture: str
    method: str
    latency_ms: float
    peak_memory_mib: float | None


def make_model(fixture: str, device: torch.device) -> tuple[ViT, torch.Tensor]:
    if fixture == "tiny":
        config = ViTConfig(
            in_channels=3,
            patch_size=(4, 4),
            img_size=(32, 32),
            depth=2,
            hidden_size=32,
            ffn_hidden_size=64,
            num_attention_heads=4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pos_enc="rope",
            dtype=torch.float32,
        )
    else:
        config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=12,
            hidden_size=768,
            ffn_hidden_size=3072,
            num_attention_heads=12,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pos_enc="rope",
            dtype=torch.float32,
        )
    model = config.instantiate(device=device).eval()
    inputs = torch.randn(1, config.in_channels, *config.img_size, device=device, dtype=config.dtype)
    return model, inputs


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(name: str, fixture: str, operation: Callable[[], Any], device: torch.device, repeats: int) -> Result:
    operation()
    synchronize(device)
    latencies: list[float] = []
    peak_memory: float | None = None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1_000)
        if device.type == "cuda":
            measured = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_memory = measured if peak_memory is None else max(peak_memory, measured)
    return Result(fixture, name, statistics.median(latencies), peak_memory)


def benchmark_fixture(fixture: str, device: torch.device, repeats: int, ig_steps: int) -> list[Result]:
    model, inputs = make_model(fixture, device)
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean(1)[:, :3])
    interventions = [
        Intervention(site="head_output", layer=layer, heads=[0], mode="zero")
        for layer in range(min(4, model.config.depth))
    ]
    operations = {
        "trace": lambda: explainer.trace(inputs),
        "legrad": lambda: explainer.attribute(inputs, target=0, method=LeGrad()),
        "integrated_gradients": lambda: explainer.attribute(
            inputs,
            target=0,
            method=IntegratedGradients(n_steps=ig_steps),
        ),
        "patch_occlusion": lambda: explainer.attribute(inputs, target=0, method=PatchOcclusion()),
        "causal_sweep": lambda: explainer.sweep(inputs, target=0, interventions=interventions),
    }
    return [measure(name, fixture, operation, device, repeats) for name, operation in operations.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("tiny", "vit-b", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    if args.repeats <= 0 or args.ig_steps <= 0:
        parser.error("--repeats and --ig-steps must be positive")
    device = torch.device(args.device)
    fixtures = ("tiny", "vit-b") if args.fixture == "all" else (args.fixture,)
    results = [
        result for fixture in fixtures for result in benchmark_fixture(fixture, device, args.repeats, args.ig_steps)
    ]
    if args.format == "json":
        print(json.dumps([asdict(result) for result in results], sort_keys=True))
        return
    print("fixture method latency_ms peak_memory_mib")
    for result in results:
        memory = "n/a" if result.peak_memory_mib is None else f"{result.peak_memory_mib:.1f}"
        print(f"{result.fixture} {result.method} {result.latency_ms:.2f} {memory}")


if __name__ == "__main__":
    main()
