#!/usr/bin/env python
"""Core benchmarking functions for ViT models."""

import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from vit import ViT, ViTConfig


PassMode = Literal["forward", "backward", "forward_backward"]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    latency_ms: float
    memory_mb: float
    gflops: float
    config_name: str
    batch_size: int
    image_size: tuple[int, ...]
    pass_mode: PassMode


def create_input_from_config(
    config: ViTConfig,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Create appropriate input tensor from config.

    Args:
        config: ViT configuration.
        batch_size: Number of samples in batch.
        device: Device to create tensor on.

    Returns:
        Input tensor with shape (B, C, *img_size).
    """
    shape = (batch_size, config.in_channels, *config.img_size)
    return torch.randn(shape, device=device, dtype=config.dtype)


def warmup_model(
    model: nn.Module,
    input_tensor: Tensor,
    num_warmup_iters: int = 10,
    pass_mode: PassMode = "forward",
) -> None:
    """Warm up model to trigger torch.compile and populate caches.

    Args:
        model: Model to warm up.
        input_tensor: Sample input tensor.
        num_warmup_iters: Number of warmup iterations.
        pass_mode: Type of pass to warm up.
    """
    if pass_mode == "forward":
        model.eval()
        with torch.inference_mode():
            for _ in range(num_warmup_iters):
                _ = model(input_tensor)
    else:
        model.train()
        for _ in range(num_warmup_iters):
            output = model(input_tensor)

            # Compute a simple loss for backward pass
            if isinstance(output, torch.Tensor):
                loss = output.mean()
            else:
                # For ViTFeatures, use the dense features
                loss = output.dense_features.mean()

            if pass_mode in ["backward", "forward_backward"]:
                loss.backward()
                model.zero_grad()


def compute_gflops(model: ViT, input_shape: tuple[int, ...]) -> float:
    """Compute GFLOPs for a single forward pass.

    Args:
        model: ViT model.
        input_shape: Shape of input tensor (B, C, *img_size).

    Returns:
        GFLOPs for forward pass.
    """
    config = model.config
    batch_size = input_shape[0]

    # Calculate tokenized sequence length
    img_size = input_shape[2:]
    patch_size = config.patch_size
    num_patches = 1
    for img_dim, patch_dim in zip(img_size, patch_size):
        num_patches *= img_dim // patch_dim

    seq_len = num_patches + config.num_register_tokens + config.num_cls_tokens
    hidden_size = config.hidden_size
    ffn_hidden_size = config.ffn_hidden_size
    num_heads = config.num_attention_heads
    depth = config.depth

    # Patch embedding: Conv operation
    # Input: (B, C, *img_size) -> Output: (B, hidden_size, *num_patches_per_dim)
    patch_embed_flops = batch_size * config.in_channels * hidden_size
    for img_dim, patch_dim in zip(img_size, patch_size):
        patch_embed_flops *= (img_dim // patch_dim) * patch_dim

    # Per-layer calculations
    # Self-attention: Q, K, V projections
    qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size

    # Attention computation: QK^T
    attn_scores_flops = batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)

    # Attention weighted sum: Attn @ V
    attn_output_flops = batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)

    # Attention output projection
    attn_proj_flops = batch_size * seq_len * hidden_size * hidden_size

    # FFN: Two linear layers
    # GLU variants (swiglu, geglu, reglu, etc.) use 2x projection for first layer
    is_glu = config.activation.endswith("glu")
    # First layer: hidden -> ffn_hidden (or 2*ffn_hidden for GLU)
    ffn1_flops = batch_size * seq_len * hidden_size * ffn_hidden_size * (2 if is_glu else 1)
    # Second layer: ffn_hidden -> hidden
    ffn2_flops = batch_size * seq_len * ffn_hidden_size * hidden_size

    # Total per layer
    layer_flops = qkv_flops + attn_scores_flops + attn_output_flops + attn_proj_flops + ffn1_flops + ffn2_flops

    # Total for all layers
    transformer_flops = depth * layer_flops

    # Total FLOPs
    total_flops = patch_embed_flops + transformer_flops

    # Convert to GFLOPs
    return total_flops / 1e9


def benchmark_latency(
    model: nn.Module,
    input_tensor: Tensor,
    num_iters: int = 100,
    pass_mode: PassMode = "forward",
    device: torch.device | None = None,
) -> float:
    """Benchmark inference latency.

    Args:
        model: Model to benchmark.
        input_tensor: Input tensor.
        num_iters: Number of iterations for averaging.
        pass_mode: Type of pass to benchmark.
        device: Device (used for synchronization).

    Returns:
        Average latency in milliseconds.
    """
    if device is None:
        device = next(model.parameters()).device

    latencies = []

    if pass_mode == "forward":
        model.eval()
        with torch.inference_mode():
            for _ in range(num_iters):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                start_time = time.perf_counter()
                _ = model(input_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
    else:
        model.train()
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()

            output = model(input_tensor)

            # Compute loss
            if isinstance(output, torch.Tensor):
                loss = output.mean()
            else:
                loss = output.dense_features.mean()

            if pass_mode in ["backward", "forward_backward"]:
                loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

            model.zero_grad()

    return sum(latencies) / len(latencies)


def benchmark_memory(
    model: nn.Module,
    input_tensor: Tensor,
    num_iters: int = 10,
    pass_mode: PassMode = "forward",
    device: torch.device | None = None,
) -> float:
    """Benchmark maximum memory utilization.

    Args:
        model: Model to benchmark.
        input_tensor: Input tensor.
        num_iters: Number of iterations for averaging.
        pass_mode: Type of pass to benchmark.
        device: Device (used for memory tracking).

    Returns:
        Maximum memory used in MB.
    """
    if device is None:
        device = next(model.parameters()).device

    if device.type != "cuda":
        return 0.0  # Memory tracking only supported for CUDA

    max_memory_mb = 0.0

    for _ in range(num_iters):
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        if pass_mode == "forward":
            model.eval()
            with torch.inference_mode():
                _ = model(input_tensor)
        else:
            model.train()
            output = model(input_tensor)

            # Compute loss
            if isinstance(output, torch.Tensor):
                loss = output.mean()
            else:
                loss = output.dense_features.mean()

            if pass_mode in ["backward", "forward_backward"]:
                loss.backward()

            model.zero_grad()

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # Convert to MB
        max_memory_mb = max(max_memory_mb, peak_memory)

    return max_memory_mb


def run_full_benchmark(
    config: ViTConfig,
    batch_size: int,
    device: torch.device | str,
    pass_mode: PassMode = "forward",
    num_warmup_iters: int = 10,
    num_latency_iters: int = 100,
    num_memory_iters: int = 10,
    config_name: str = "default",
    show_progress: bool = True,
) -> BenchmarkResult:
    """Run complete benchmark suite on a ViT configuration.

    Args:
        config: ViT configuration to benchmark.
        batch_size: Batch size for benchmarking.
        device: Device to run benchmark on.
        pass_mode: Type of pass to benchmark.
        num_warmup_iters: Number of warmup iterations.
        num_latency_iters: Number of iterations for latency measurement.
        num_memory_iters: Number of iterations for memory measurement.
        config_name: Name identifier for this config.
        show_progress: Whether to show progress bars.

    Returns:
        Benchmark results.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Create model and input
    with tqdm(total=5, desc=f"Benchmarking {config_name}", disable=not show_progress) as pbar:
        pbar.set_postfix_str("Creating model")
        model = config.instantiate(device=device)
        pbar.update(1)

        pbar.set_postfix_str("Creating input")
        input_tensor = create_input_from_config(config, batch_size, device)
        pbar.update(1)

        # Warmup
        pbar.set_postfix_str("Warming up")
        warmup_model(model, input_tensor, num_warmup_iters, pass_mode)
        pbar.update(1)

        # Benchmark latency
        pbar.set_postfix_str("Measuring latency")
        latency_ms = benchmark_latency(model, input_tensor, num_latency_iters, pass_mode, device)
        pbar.update(1)

        # Benchmark memory (run separately to avoid interference)
        pbar.set_postfix_str("Measuring memory")
        memory_mb = benchmark_memory(model, input_tensor, num_memory_iters, pass_mode, device)
        pbar.update(1)

    # Compute GFLOPs
    gflops = compute_gflops(model, input_tensor.shape)
    if pass_mode == "backward":
        gflops *= 2  # Backward pass is approximately 2x forward
    elif pass_mode == "forward_backward":
        gflops *= 3  # Forward + backward is approximately 3x forward

    return BenchmarkResult(
        latency_ms=latency_ms,
        memory_mb=memory_mb,
        gflops=gflops,
        config_name=config_name,
        batch_size=batch_size,
        image_size=tuple(config.img_size),
        pass_mode=pass_mode,
    )
