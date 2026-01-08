# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vision Transformer (ViT) implementation in native PyTorch, optimized with `torch.compile`. This implementation does not use a CLS token - features are returned directly from the transformer. Supports modern enhancements: RMSNorm, SwiGLU, Squared ReLU, register tokens, and multiple position encodings (RoPE, Fourier, Learnable).

## Common Commands

```bash
# Setup
make init              # Initialize environment and install all dependencies (uses uv)

# Development
make check             # Full check: style + quality + types + tests
make style             # Auto-format code with ruff
make quality           # Check formatting without changes
make types             # Static type checking with basedpyright
make test              # Run unit tests with coverage
make test-<pattern>    # Run tests matching pattern (e.g., make test-attention)
make test-pdb-<pattern> # Debug tests with PDB

# Benchmarking
uv sync --group benchmarking
vit-benchmark --configs config.yaml --resolutions 224,224 --device cuda
```

## Architecture

**Core flow**: Images → PatchEmbed → Transformer Encoder → ViTFeatures → Heads

Key modules in `vit/`:
- `vit.py` - Main `ViT` model class and `ViTConfig` dataclass (YAML-serializable)
- `transformer.py` - `TransformerEncoderLayer` and `TransformerDecoderLayer`
- `attention.py` - `Attention`, `AttentionResidual`, `CrossAttention`
- `head.py` - Output heads (`Head`, `TransposedConv2dHead`, `UpsampleHead`) with configs
- `pos_enc.py` - Position encoders (`RopePositionEmbedding`, `FourierPosition`, `LearnablePosition`)
- `fused.py` - Fused operations (`NormMLP` combines norm + linear + activation)
- `tokens.py` - Register token handling

**Output structure**: `ViT.forward()` returns a `ViTFeatures` dataclass, not raw tensors.

**Heads system**: Heads are configured via `HeadConfig` dict in `ViTConfig.heads` and accessed via `model.heads["name"]`.

## Code Style

- Line length: 120 characters (ruff)
- Type checking: basedpyright with Python 3.14 target
- Linting and formatting: ruff (replaces black, isort, flake8, autopep8, autoflake)
- Always run `make style` before committing
- Test markers: `@pytest.mark.ci_skip` (skip in CI), `@pytest.mark.cuda` (requires GPU)

## Key Patterns

- **Config-driven instantiation**: Use `ViTConfig.instantiate()` and `HeadConfig.instantiate()` rather than direct class construction
- **torch.compile friendly**: Custom implementations avoid einops; designed for compilation
- **No CLS token**: All token pooling happens in heads, not the transformer
- **Position encoding**: Default is RoPE; configurable via `pos_enc` parameter

## Rust CLI (rust/)

Rust workspace for ViT model validation, summarization, and inference. Uses AOTInductor for compiled model inference.

**Why a custom bridge instead of tch-rs?**

The bridge (`rust/bridge/`) exists specifically to load AOTInductor-compiled models (`.so` shared libraries), which tch-rs does not support. The bridge wraps `torch::inductor::AOTIModelContainerRunner` - a specialized runtime for executing models compiled via `torch.export` + AOTInductor. tch-rs provides general PyTorch bindings but lacks AOTInductor model loading, `.so` artifact support, and the specialized inference timing/memory tracking APIs we need.

<!-- TODO: Check if tch-rs adds AOTInductor support in the future (https://github.com/LaurentMazare/tch-rs) -->

```bash
# Build without inference (no external dependencies)
make rust-release             # Build release binary
make rust-test                # Run Rust tests

# Build with inference support (requires libtorch)
make libtorch                 # Download libtorch with CUDA support
make libtorch-cpu             # Download CPU-only libtorch
make libtorch-rocm            # Download libtorch with ROCm support (AMD GPUs)
export LIBTORCH=$(pwd)/libtorch
make rust-ffi                 # Build with FFI/inference support (CUDA)
make rust-ffi-rocm            # Build with FFI/inference support (ROCm)

# Create portable distribution (self-contained, ~471MB)
make rust-install             # Creates dist/vit/ with all dependencies bundled

# CLI commands
./rust/target/release/vit validate config.yaml
./rust/target/release/vit summarize config.yaml
./rust/target/release/vit infer --model model.so --config config.yaml --shape 1,3,224,224

# Export model to AOTInductor format (.so shared library)
make export-model CONFIG=config.yaml OUTPUT=model.so DEVICE=cpu
```

**Crates:**
- `vit-core` - Config parsing and model summary (mirrors Python `ViTConfig`)
- `vit-ffi` - FFI bindings to C++ AOTInductor runtime (optional, requires libtorch)
- `vit-cli` - CLI binary with validate/summarize/infer commands

**Build Notes:**
- CUDA builds require CUDA toolkit with nvcc in PATH and GCC ≤13 (CUDA 12.x limitation)
- ROCm builds require ROCm toolkit (Linux only, ROCm 5.7+)
- Use CPU-only libtorch if you encounter compiler compatibility issues
- The Makefile automatically sets `LIBTORCH_CXX11_ABI=1` for correct C++ ABI compatibility
- PyTorch uses `cuda` device strings for both NVIDIA and AMD GPUs

## AOT Export Gotchas

When exporting models for Rust inference, be aware of these issues:

1. **torch.compile conflicts with torch.export**: The `@torch.compile` decorators throughout the codebase conflict with `torch.export` tracing, causing `FakeTensorMode` mismatch errors. The Makefile's `export-model` target automatically sets `TORCH_COMPILE_DISABLE=1` to resolve this. If running manually:
   ```bash
   TORCH_COMPILE_DISABLE=1 python scripts/export_aot.py --config config.yaml --output model.so
   ```

2. **Output format is .so, not .pt2**: The export produces a shared library (`.so`) file, not a ZIP package. The C++ AOTInductor runtime loads these directly.

3. **C++ ABI compatibility**: libtorch uses the new CXX11 ABI (`std::__cxx11::basic_string`). The FFI build must use `LIBTORCH_CXX11_ABI=1` to match. Symbol errors like `undefined symbol: ...basic_string...` indicate ABI mismatch.

4. **Static shapes**: Dynamic batch sizes may fail during export if the model specializes shapes. The export script automatically falls back to static shapes when this occurs.
