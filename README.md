# ViT

Implementation of Vision Transformer (ViT) in native PyTorch, accelerated by `torch.compile`.
Supports modern enhancements like RMSNorm, SwiGLU, Squared ReLU, register tokens, optional `CLS` tokens, and different positional encodings.

## Installation

This library can be installed with the following command

```bash
pip install vit @ git+https://github.com/TidalPaladin/vit.git
```

For benchmarking capabilities, install with the benchmarking extras:

```bash
pip install "vit[benchmarking] @ git+https://github.com/TidalPaladin/vit.git"
```

## Usage

```python
import torch
from vit import ViTConfig, HeadConfig

# Create a ViT backbone
config = ViTConfig(
    in_channels=3,
    patch_size=(14, 14),
    img_size=(224, 224),
    depth=12,
    hidden_size=768,
    ffn_hidden_size=3072,
    num_attention_heads=12,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    attention_bias=False,
    mlp_bias=False,
    activation="swiglu",  # or srelu, gelu, etc.
    drop_path_rate=0.1,
    num_register_tokens=16,
    pos_enc="fourier",
    layer_scale=1e-5,
    heads={
        "cls": HeadConfig(out_features=10),
    },
)
model = config.instantiate()

# Forward pass for features
B, C, H, W = 1, 3, 224, 224
x = torch.randn(B, C, H, W)
features = model(x)

# Apply a classification head (pool first)
pooled = features.visual_tokens.mean(dim=1)
logits = model.heads["cls"](pooled)  # (B, 10)
```

`model(x)` returns a `ViTFeatures` object with:
- `dense_features`
- `visual_tokens`
- `register_tokens`
- `cls_tokens`
- `tokenized_size`

## Mixture of Experts (MoE)

You can convert selected encoder MLP blocks to MoE by index via `moe_block_indices`.
Routing is controlled globally with `moe_routing_mode`:
- `"expert_choice"` (default)
- `"token_choice"`

```python
import torch
from vit import ViTConfig

config = ViTConfig(
    in_channels=3,
    patch_size=(16, 16),
    img_size=(224, 224),
    depth=12,
    hidden_size=768,
    ffn_hidden_size=3072,
    num_attention_heads=12,
    activation="swiglu",
    moe_block_indices=(2, 5, 8, 11),  # encoder layer indices to convert
    moe_num_experts=8,
    moe_routing_mode="token_choice",  # or "expert_choice"
    moe_token_top_k=2,                # used for token-choice routing
    moe_use_simple_experts=True,      # token-choice only
    moe_num_zero_experts=1,           # output 0
    moe_num_copy_experts=1,           # output x (skip)
    moe_num_constant_experts=1,       # output learned c (pure replace)
    moe_expert_capacity_factor=1.0,
    moe_router_jitter_noise=0.01,
    moe_drop_overflow_tokens=True,
    moe_aux_loss_weight=0.01,
    dtype=torch.float32,
)
model = config.instantiate()

x = torch.randn(4, 3, 224, 224)
features = model(x)
```

When MoE is enabled, routing diagnostics are exposed on `features.moe`:
- `features.moe.layers[layer_idx].router_logits`
- `features.moe.layers[layer_idx].expert_token_counts`
- `features.moe.layers[layer_idx].dropped_token_count`
- `features.moe.layers[layer_idx].capacity`
- `features.moe.layers[layer_idx].routing_mode`

### Applying the load-balancing loss

`MoEStats` includes a built-in helper for balancing loss aggregation:

```python
# Example training step
features = model(images)
pooled = features.visual_tokens.mean(dim=1)
task_loss = criterion(model.heads["cls"](pooled), labels)

moe_loss = features.moe.load_balancing_loss() if features.moe is not None else task_loss.new_zeros(())
loss = task_loss + config.moe_aux_loss_weight * moe_loss
loss.backward()
```

Balancing loss semantics are routing-mode aware:
- `expert_choice`: Switch-style importance/load dot-product loss (`N * sum(importance * load)`), which can naturally sit near `1.0` for balanced routing.
- `token_choice`: V-MoE-style coefficient-of-variation loss (`cv^2(importance) + cv^2(load)`), which is non-negative with minimum `0`.
  Token-choice routing uses batch-prioritized dispatch with per-expert capacity limits.

### Token-choice simple experts (MoE++)

When `moe_use_simple_experts=True`, token-choice MoE can reserve some experts for simple behavior:
- zero expert: outputs all zeros
- copy expert: outputs the normalized input token (`x`)
- constant expert: outputs a learned vector (`c`, pure replace)

Notes:
- simple experts are only supported for `moe_routing_mode="token_choice"`.
- `moe_num_experts` is the total expert count, including simple experts and MLP experts.
- `moe_num_zero_experts + moe_num_copy_experts + moe_num_constant_experts` must be `<= moe_num_experts`.
- at least one MLP expert must remain.

MoE MLP compute paths are designed to follow the same compile-first pattern used by dense MLP blocks.

## Activation Checkpointing

Enable activation checkpointing to reduce memory usage during training at the cost of additional compute:

```python
config = ViTConfig(
    # ... other params ...
    activation_checkpointing=True,  # Enable gradient checkpointing
)
```

Memory savings scale with batch size and model depth:

| Depth | Hidden | Batch | Memory Savings | Latency Overhead |
|-------|--------|-------|----------------|------------------|
| 12    | 768    | 4     | 24%            | 132%             |
| 12    | 768    | 8     | 49%            | 114%             |
| 24    | 768    | 4     | 26%            | 36%              |
| 24    | 768    | 8     | 52%            | 44%              |

Run the checkpointing benchmark to measure savings on your hardware:

```bash
uv run python -m benchmark.checkpoint_memory --depths 12 24 --hidden-sizes 768 --batch-sizes 4 8
```

## Benchmarking

The library includes a comprehensive benchmarking suite for measuring model performance:

```bash
# Install benchmarking dependencies
uv sync --group benchmarking

# Run benchmarks
vit-benchmark \
    --configs config.yaml \
    --resolutions 224,224 384,384 \
    --batch-size 8 \
    --device cuda \
    --output-dir results/
```

The benchmarking tool tracks:
- **Inference latency** (milliseconds per batch)
- **Peak memory usage** (MB)
- **Computational cost** (GFLOPs)

Results are saved as CSV files and visualized with publication-quality plots (PNG/SVG).

For low-level optimization regression testing of core transformer components, use
`vit-component-benchmark` (`run`, `compare`, `list-baselines`).
Detailed workflow and recipes live in:
- [`benchmark/README.md`](benchmark/README.md)
- `.agents/skills/vit-component-benchmark/SKILL.md` (contributor skill documentation)

See [`benchmark/README.md`](benchmark/README.md) for detailed documentation.

## References
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

* [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/abs/2106.02795)

* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)


* [ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs](https://arxiv.org/abs/2402.03804)

* [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)

* [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

* [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
