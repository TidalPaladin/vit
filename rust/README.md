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

To run inference, you need to build with the `ffi` feature which requires libtorch:

1. **Install libtorch** (must match your PyTorch version):

   ```bash
   # Download from https://pytorch.org/get-started/locally/
   # Or use the libtorch bundled with your PyTorch installation
   export LIBTORCH=/path/to/libtorch
   ```

2. **Export the model** from Python using AOTInductor:

   ```bash
   python scripts/export_aot.py \
       --config config.yaml \
       --weights weights.safetensors \
       --output model.pt2 \
       --shape 1,3,224,224 \
       --device cuda
   ```

3. **Build with FFI support**:

   ```bash
   cd rust
   cargo build --release --features ffi
   ```

4. **Run inference**:

   ```bash
   ./target/release/vit infer \
       --model model.pt2 \
       --config config.yaml \
       --device cuda:0 \
       --shape 1,3,224,224
   ```

## Requirements

### For validate/summarize (no external dependencies)
- Rust 1.75+

### For inference (requires C++ bridge)
- Rust 1.75+
- libtorch (matching your PyTorch version)
- CMake 3.18+
- C++17 compiler
- CUDA toolkit (optional, for GPU support)

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
