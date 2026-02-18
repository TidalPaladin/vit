---
name: vit-component-benchmark
description: Use this skill for low-level transformer component benchmarking and regression testing with `vit-component-benchmark` (run, compare, and baseline workflows), including device/pass/dtype controls and artifact interpretation.
---

# Vit Component Benchmark

Use this skill when you need fast, repeatable performance checks for transformer primitives instead of full end-to-end ViT benchmarking.

## Scope

- Components: `mlp`, `self_attention`, `layer_scale_residual`, `drop_path_residual`
- Modes: `forward`, `backward`, `forward_backward`
- Dtypes: `--param-dtype`, `--input-dtype`, `--autocast-dtype`
- Devices: `cpu`, `cuda`, `cuda:<idx>`
- Outputs: CSV + JSON run artifacts and JSON compare reports

## Command Surface

- `vit-component-benchmark run`
- `vit-component-benchmark compare`
- `vit-component-benchmark list-baselines`

Output root defaults to:

- `benchmark_results/components/`

## Standard Workflow

### 1) Create a baseline

```bash
uv run vit-component-benchmark run \
  --components mlp self_attention \
  --presets default \
  --pass-modes forward backward \
  --device cuda \
  --param-dtype float32 \
  --input-dtype float32 \
  --autocast-dtype bfloat16 \
  --save-as baseline
```

### 2) Run candidate after code changes

```bash
uv run vit-component-benchmark run \
  --components mlp self_attention \
  --presets default \
  --pass-modes forward backward \
  --device cuda \
  --param-dtype float32 \
  --input-dtype float32 \
  --autocast-dtype bfloat16 \
  --save-as candidate
```

### 3) Compare baseline vs candidate

```bash
uv run vit-component-benchmark compare \
  --baseline baseline \
  --candidate candidate \
  --metric mean_ms \
  --regression-threshold-pct 5 \
  --noise-floor-pct 1
```

### 4) Optional CI gate

```bash
uv run vit-component-benchmark compare \
  --baseline baseline \
  --candidate candidate \
  --metric mean_ms \
  --regression-threshold-pct 5 \
  --noise-floor-pct 1 \
  --fail-on-regression
```

## Common Recipes

### Single SwiGLU MLP case (hidden=768, ffn=3072), bf16 autocast

```bash
uv run vit-component-benchmark run \
  --components mlp \
  --activation swiglu \
  --pass-modes forward backward \
  --batch-sizes 4 \
  --seq-lens 256 \
  --hidden-sizes 768 \
  --ffn-mults 4 \
  --autocast-dtype bfloat16 \
  --save-as swiglu-mlp-768-3072-bf16
```

### Attention-focused sweep

```bash
uv run vit-component-benchmark run \
  --components self_attention \
  --presets rope \
  --pass-modes forward \
  --batch-sizes 2 4 \
  --seq-lens 256 1024 \
  --hidden-sizes 512 768 \
  --num-heads 8 12 \
  --device cuda \
  --autocast-dtype bfloat16 \
  --save-as attn-rope-sweep
```

### Residual-op regression check

```bash
uv run vit-component-benchmark run \
  --components layer_scale_residual drop_path_residual \
  --pass-modes forward backward \
  --batch-sizes 8 \
  --seq-lens 256 \
  --hidden-sizes 768 \
  --save-as residual-ops
```

## Reproducibility Controls

Use these for stable comparisons:

- `--seed <int>`
- `--deterministic`
- `--threads <int>`
- fixed `--device`
- fixed dtype flags
- fixed shape flags

## Artifact Layout and Interpretation

Each run creates:

- `benchmark_results/components/<run-name>/component_benchmark_results.csv`
- `benchmark_results/components/<run-name>/component_benchmark_results.json`

Comparison creates:

- `benchmark_results/components/<candidate>/comparison_vs_<baseline>.json`

Important fields:

- `case_id`: stable key for case matching
- `mean_ms`, `median_ms`, `p95_ms`, `std_ms`: latency summary statistics
- `memory_mb`: CUDA-only memory metric

## Practical Notes

- Prefer comparing runs with identical flags except the code change under test.
- For local iteration, lower `--min-samples` and durations; for decision runs, increase them.
- If `compare` reports missing cases, ensure baseline and candidate used matching case grids.
