# ViT Benchmarking Tools

Comprehensive benchmarking suite for Vision Transformer models.

## Installation

Install the benchmarking dependencies:

```bash
uv sync --group benchmarking
```

## Features

- **Comprehensive Metrics**: Track inference latency, peak memory usage, and computational cost (GFLOPs)
- **Multiple Pass Modes**: Benchmark forward-only, backward-only, or forward-backward cycles
- **Automatic Warmup**: Triggers torch.compile and populates caches before measurement
- **Visualization**: Generate publication-quality plots in PNG and SVG formats
- **Progress Tracking**: Built-in progress bars for long-running benchmarks
- **Batch Processing**: Benchmark multiple configurations and resolutions in one run

## Quick Start

### Using the CLI

```bash
# Basic benchmark with a single config and resolution
vit-benchmark --configs config.yaml --resolutions 224,224 --output-dir results/

# Multiple configurations and resolutions
vit-benchmark \
    --configs config1.yaml config2.yaml \
    --resolutions 224,224 384,384 512,512 \
    --batch-size 8 \
    --device cuda \
    --output-dir results/

# Benchmark backward pass with custom iterations
vit-benchmark \
    --configs config.yaml \
    --resolutions 224,224 \
    --pass-mode forward_backward \
    --warmup-iters 20 \
    --latency-iters 200 \
    --output-dir results/
```

### Using the Python API

```python
from pathlib import Path
import torch
from vit import ViTConfig
from benchmark import run_full_benchmark, plot_benchmark_results

# Load config
config = ViTConfig.from_yaml("config.yaml")

# Run benchmark
result = run_full_benchmark(
    config=config,
    batch_size=1,
    device=torch.device("cuda"),
    pass_mode="forward",
    num_warmup_iters=10,
    num_latency_iters=100,
    num_memory_iters=10,
    config_name="my_model",
)

print(f"Latency: {result.latency_ms:.2f} ms")
print(f"Memory: {result.memory_mb:.2f} MB")
print(f"GFLOPs: {result.gflops:.2f}")

# Generate plots
output_dir = Path("results")
plot_benchmark_results([result], output_dir, metric="latency")
```

## CLI Options

### Required Arguments

- `--configs`: Path(s) to ViTConfig YAML file(s)
- `--resolutions`: Resolution(s) to benchmark
  - 2D: `224,224` or just `224` (assumes square)
  - 3D: `64,224,224`

### Optional Arguments

- `--batch-size`: Batch size for benchmarking (default: 1)
- `--device`: Device to run on (default: "cuda" if available, else "cpu")
- `--pass-mode`: Type of pass to benchmark (default: "forward")
  - `forward`: Forward pass only (uses torch.inference_mode())
  - `backward`: Backward pass only
  - `forward_backward`: Full training cycle
- `--output-dir`: Directory to save results (default: "benchmark_results")
- `--warmup-iters`: Number of warmup iterations (default: 10)
- `--latency-iters`: Number of iterations for latency measurement (default: 100)
- `--memory-iters`: Number of iterations for memory measurement (default: 10)
- `--plot-formats`: Output format(s) for plots (default: png svg)
- `--dpi`: DPI for raster formats (default: 300)
- `--no-plots`: Skip generating plots

## Output

The tool generates:

1. **CSV Results**: `benchmark_results.csv` with all measurements
2. **Individual Metric Plots**:
   - `benchmark_latency.{png,svg}`: Latency vs image size
   - `benchmark_memory.{png,svg}`: Memory usage vs image size
   - `benchmark_gflops.{png,svg}`: Computational cost vs image size
3. **Comparison Plot**: `benchmark_comparison.{png,svg}` with all metrics
4. **Throughput Plot**: `benchmark_throughput.{png,svg}` showing samples/second

## Example Config Files

### Minimal Config

```yaml
in_channels: 3
patch_size: [16, 16]
img_size: [224, 224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
dtype: bfloat16
```

### Advanced Config

```yaml
in_channels: 3
patch_size: [16, 16]
img_size: [224, 224]

# Transformer architecture
depth: 24
hidden_size: 1024
ffn_hidden_size: 4096
num_attention_heads: 16

# Regularization
hidden_dropout: 0.1
attention_dropout: 0.1
drop_path_rate: 0.1

# Special tokens
num_register_tokens: 4
num_cls_tokens: 1

# Position encoding
pos_enc: rope
rope_base: 10000
rope_normalize_coords: separate

# Master dtype
dtype: bfloat16
```

## API Reference

### Core Functions

#### `run_full_benchmark()`

Run complete benchmark suite on a ViT configuration.

**Parameters:**
- `config`: ViTConfig to benchmark
- `batch_size`: Batch size for benchmarking
- `device`: Device to run benchmark on
- `pass_mode`: "forward", "backward", or "forward_backward"
- `num_warmup_iters`: Number of warmup iterations
- `num_latency_iters`: Number of iterations for latency measurement
- `num_memory_iters`: Number of iterations for memory measurement
- `config_name`: Name identifier for this config
- `show_progress`: Whether to show progress bars

**Returns:** `BenchmarkResult` with all metrics

#### `plot_benchmark_results()`

Create plots of benchmark results as function of image size.

**Parameters:**
- `results`: List of BenchmarkResult objects
- `output_dir`: Directory to save plots
- `metric`: "latency", "memory", or "gflops"
- `plot_format`: "png", "svg", or list of formats
- `dpi`: DPI for raster formats

**Returns:** List of paths to created plot files

## Performance Tips

1. **GPU Warmup**: Always use at least 10 warmup iterations for CUDA
2. **Memory Tracking**: Memory benchmarks are only available on CUDA
3. **Compilation**: First run may be slower due to torch.compile
4. **Batch Size**: Larger batches give better throughput but may hit memory limits
5. **Iterations**: Use more iterations for more stable measurements (100+ for latency)

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch-size`
- Use smaller `--memory-iters` (default 10 is usually sufficient)
- Test with smaller resolutions first

### Slow Benchmarks

- Reduce `--latency-iters` (but results will be less stable)
- Use `--no-plots` if you only need CSV data
- Ensure CUDA is available with `--device cuda`

### Import Errors

Make sure benchmarking dependencies are installed:

```bash
uv sync --group benchmarking
```
