# ViT

Implementation of Vision Transformer (ViT) in native PyTorch, accelerated by `torch.compile`.
Supports RMSNorm, SwiGLU, Squared ReLU, optional CLS and register tokens, and several positional encodings.

Transformer residual output projections are zero-initialized by default (`attn out_proj` and `mlp fc2`) for stable deep-stack initialization.

## Installation

This library can be installed with the following command

```bash
pip install "vit @ git+https://github.com/TidalPaladin/vit.git"
```

For benchmarking capabilities, install with the benchmarking extras:

```bash
pip install "vit[benchmarking] @ git+https://github.com/TidalPaladin/vit.git"
```

Install Captum and artifact-rendering dependencies for the explainability toolbox:

```bash
pip install "vit[explainability] @ git+https://github.com/TidalPaladin/vit.git"
```

## Usage

```python
import torch
from vit import AttentivePoolHeadConfig, ViTConfig

# Create ViT-B/14, RMSNorm + SwiGLU, no biases
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
    activation="swiglu", # or srelu, gelu, etc.
    patch_embed_normalization=True,  # Apply backbone norm after patch embedding
    drop_path_rate=0.1,
    num_register_tokens=16,
    pos_enc="fourier",
    layer_scale=1e-5,
    heads={
        "cls": AttentivePoolHeadConfig(out_features=10),
    },
)
model = config.instantiate()

# Forward pass for features
B, C, H, W = 1, 3, 224, 224
x = torch.randn(B, C, H, W)
features = model(x)

# Apply classification head
logits = model.heads["cls"](features.visual_tokens)  # B, 10
```

### Global and visual token pathways

By default, CLS, register, and visual tokens share every encoder parameter. Token specialization can give the
CLS/register prefix and visual tokens separate normalization parameters, plus separate LayerScale parameters when
LayerScale is configured. Attention still connects every token. A configurable number of leading blocks can also use
separate QKV projections:

This pathway adapts the [CLS]-patch specialization proposed by
[Marouani et al. (2026)](https://arxiv.org/abs/2602.08626). This repository additionally treats register tokens as
global tokens and makes QKV specialization configurable over a chosen number of leading blocks.

```python
config = ViTConfig(
    # ... other parameters ...
    depth=12,
    num_cls_tokens=1,
    num_register_tokens=7,
    specialize_global_token_norms=True,
    specialize_global_token_qkv_blocks=4,
)
model = config.instantiate()
features = model(x)  # ViTFeatures, unchanged from the shared-path model
```

Both specialization options default to disabled. Specialized parameters are cloned from their shared counterparts at
initialization, so enabling the feature does not change the initial function before the paths receive separate updates.

Token-specialized attention exposes four compilation policies:

| Mode | Runtime behavior |
|------|------------------|
| `auto` | Default. During gradient-enabled training, batches of at least 512 use an isolated static graph; other calls use the adapting compiled path. |
| `dynamic` | Always use a separate `dynamic=True` compiled graph. |
| `static` | Always compile concrete shapes. |
| `static_max_autotune` | Use the static path with CUDA GEMM autotuning and no CUDA graphs. CPU inputs use `static`. |

Use an allowlist when the training batches that should use static compilation are known:

```python
config = ViTConfig(
    # ... token specialization and model parameters ...
    token_specialized_attention_compile_mode="auto",
    token_specialized_attention_static_batch_sizes=(128, 256, 512),
)
```

In `auto`, the allowlist replaces the default batch-size threshold. Unlisted batches remain on the adapting path. In
`static` and `static_max_autotune`, an allowlist bounds the accepted batches and unlisted batches raise `ValueError`.
Omit the allowlist to permit any observed batch size, with one concrete graph per distinct full input shape. `dynamic`
rejects a static allowlist. Allowlist values refer to the batch reaching specialized attention. Selective stochastic
depth can reduce that batch below the model input batch, so include the resulting residual-subset sizes when it is
enabled. Non-default compilation settings require token specialization to be enabled.

The same settings can be stored in YAML:

```yaml
token_specialized_attention_compile_mode: static
token_specialized_attention_static_batch_sizes: [128, 256, 512]
```

`static_max_autotune` has a substantial one-time CUDA compile cost. Prefer `auto` unless repeated measurements on the
target workload justify another mode. See [`docs/aot-export.md`](docs/aot-export.md) for `torch.export` and
AOTInductor support for every mode.

## CUDA GLU GEMM Autotuning

Compiled GLU MLPs can opt into PyTorch Inductor GEMM autotuning when steady-state CUDA throughput is more important
than compilation time:

```python
config = ViTConfig(
    # ... other params ...
    activation="swiglu",
    glu_max_autotune_gemm=True,
)
```

The option requires a GLU activation and is not supported with MLP quantization. It applies only to CUDA inputs when
the MLP input width is at least 512. CPU inputs and smaller MLPs use the default compiled GLU path.

On an RTX 3090 with PyTorch 2.13.0, native BF16, and `B=4, S=256, D=768, FFN=3072`, the opt-in changed the measured
steady-state latency as follows:

| Pass | Default | Autotuned | Change |
|------|--------:|----------:|-------:|
| Forward | 0.413 ms | 0.331 ms | 19.9% faster |
| Backward | 0.853 ms | 0.752 ms | 11.9% faster |
| Forward + backward | 1.157 ms | 1.016 ms | 12.2% faster |

A fresh-cache forward-backward compile increased from 4.53 seconds to 21.40 seconds. Results vary by GPU, PyTorch
version, shape, and dtype, so keep the option disabled unless steady-state measurements on the target workload justify
the compile-time cost.

## Explainability

`vit.explain` traces the native 2D `ViT`, attributes selected outputs, runs causal interventions, and evaluates
explanations. The caller supplies the downstream prediction through `output_fn`; the toolbox never guesses how a
plain linear head should pool tokens.

```python
from vit.explain import LeGrad, ViTExplainer

explainer = ViTExplainer.from_head(model, "cls")
explanation = explainer.attribute(x, target=3, method=LeGrad())

print(explanation.token_attributions.shape)
print(explanation.layout.visual_validity)
```

LeGrad is the recommended class-specific ViT method and is based on
[Bousselham et al. (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Bousselham_LeGrad_An_Explainability_Method_for_Vision_Transformers_via_Feature_Formation_ICCV_2025_paper.html).
Raw attention and attention rollout require an explicit query selector because they describe attention structure, not
a class prediction. Attribution arrays remain unnormalized; normalization and image interpolation are separate
visualization operations.

See [`docs/explainability.md`](docs/explainability.md) for methods, targets, masking, interventions, metrics,
artifacts, the experimental sparse autoencoder, and a method-by-method citation guide. Copy-ready references are in
[`docs/explainability-references.bib`](docs/explainability-references.bib). Run the synthetic example with:

```bash
uv run python examples/explain_synthetic.py
```

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

The library includes benchmarks for latency, peak memory, and floating-point operations:

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

* [Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](https://arxiv.org/abs/2602.08626)

* [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

* [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
