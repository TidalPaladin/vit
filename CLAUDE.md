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
make style             # Auto-format code (autoflake, isort, autopep8, black)
make quality           # Check formatting without changes
make types             # Static type checking with pyright
make test              # Run unit tests with coverage (~370 tests, ~30s)
make test-<pattern>    # Run tests matching pattern (e.g., make test-attention) - PREFER THIS
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
- Test markers: `@pytest.mark.cuda` (requires GPU, skipped in CI), `@pytest.mark.compile` (requires torch.compile, skipped in CI)

## Key Patterns

- **Config-driven instantiation**: Use `ViTConfig.instantiate()` and `HeadConfig.instantiate()` rather than direct class construction
- **torch.compile friendly**: Custom implementations avoid einops; designed for compilation
- **No CLS token**: All token pooling happens in heads, not the transformer
- **Position encoding**: Default is RoPE; configurable via `pos_enc` parameter
- **Activation checkpointing**: Enable via `ViTConfig(activation_checkpointing=True)` to reduce memory ~50% at cost of ~40% latency overhead (for deep models with large batches)
