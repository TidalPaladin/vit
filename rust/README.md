# ViT Rust CLI

Rust workspace for ViT model validation, summarization, and inference using AOTInductor-compiled models.

## Overview

This workspace provides a Rust CLI (`vit`) for working with ViT models:

- **validate**: Validate a YAML configuration file
- **summarize**: Show model architecture and parameter counts
- **infer**: Run inference on an AOTInductor-compiled model (requires C++ bridge)

## Quick Start

```bash
# Build the CLI
cargo build --release

# Validate a config
./target/release/vit validate config.yaml

# Summarize model architecture
./target/release/vit summarize config.yaml
```

## Crates

| Crate | Description |
|-------|-------------|
| `vit-core` | Core types and config parsing (mirrors Python `ViTConfig`) |
| `vit-ffi` | FFI bindings to C++ AOTInductor runtime |
| `vit-cli` | CLI binary |

## Inference Setup

To run inference, you need to build with the `ffi` feature which requires libtorch.

### For NVIDIA GPUs (CUDA)

1. **Download libtorch with CUDA** (from project root):

   ```bash
   make libtorch CUDA_VERSION=12.8
   export LIBTORCH=$(pwd)/libtorch
   ```

2. **Export the model** from Python using AOTInductor:

   ```bash
   python scripts/export_aot.py \
       --config config.yaml \
       --weights weights.safetensors \
       --output model.so \
       --shape 1,3,224,224 \
       --device cuda
   ```

3. **Build with FFI support**:

   ```bash
   make rust-ffi
   ```

4. **Run inference**:

   ```bash
   ./rust/target/release/vit infer \
       --model model.so \
       --config config.yaml \
       --device cuda:0 \
       --shape 1,3,224,224
   ```

### For AMD GPUs (ROCm)

ROCm support is available on Linux with AMD GPUs (gfx900+).

1. **Download libtorch with ROCm** (from project root):

   ```bash
   make libtorch-rocm ROCM_VERSION=6.2
   export LIBTORCH=$(pwd)/libtorch
   ```

2. **Export the model** from Python using AOTInductor:

   ```bash
   # Ensure PyTorch is installed with ROCm support
   python scripts/export_aot.py \
       --config config.yaml \
       --weights weights.safetensors \
       --output model.so \
       --shape 1,3,224,224 \
       --device cuda  # PyTorch uses "cuda" for ROCm devices too
   ```

3. **Build with FFI support for ROCm**:

   ```bash
   make rust-ffi-rocm
   ```

4. **Run inference**:

   ```bash
   ./rust/target/release/vit infer \
       --model model.so \
       --config config.yaml \
       --device cuda:0 \
       --shape 1,3,224,224
   ```

**Note:** PyTorch uses `cuda` device strings for both NVIDIA and AMD GPUs. The actual backend (CUDA or HIP) is determined by how libtorch was built.

## Requirements

### For validate/summarize (no external dependencies)
- Rust 1.75+

### For inference (requires C++ bridge)
- Rust 1.75+
- libtorch (matching your PyTorch version)
- CMake 3.18+
- C++17 compiler
- **For NVIDIA GPUs**: CUDA toolkit (nvcc in PATH, GCC ≤13 for CUDA 12.x)
- **For AMD GPUs**: ROCm toolkit (Linux only, ROCm 5.7+)

## Configuration

The CLI can parse ViT YAML configuration files, including those with Python object tags:

```yaml
# config.yaml
in_channels: 3
patch_size: [16, 16]
img_size: [224, 224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
activation: srelu
pos_enc: rope
dtype: bfloat16
```

## Development

```bash
# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```
