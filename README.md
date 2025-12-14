# ViT

Implementation of Vision Transformer (ViT) in native PyTorch, accelerated by `torch.compile`.
Supports modern enhancements like RMSNorm, SwiGLU, Squared ReLU, register tokens, and different positional encodings. This implementation does not incorporate a `CLS` token.

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

# Create ViT-B/14, RMSNorm + SwiGLU, no biases
config = ViTConfig(
    in_channels=3,
    patch_size=(14, 14),
    img_size=(224, 224)
    depth=12,
    hidden_size=768,
    ffn_hidden_size=3072,
    num_attention_heads=12,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    bias=False,
    activation="swiglu", # or srelu, gelu, etc.
    drop_path_rate=0.1,
    num_register_tokens=16,
    pos_enc="fourier",
    layer_scale=1e-5,
    heads={
        "cls": HeadConfig(pool_type="attentive", out_dim=10)
    }
)
model = config.instantiate()

# Forward pass for features
B, C, H, W = 1, 3, 224, 224
x = torch.randn(B, C, H, W)
features = model(x) # B, L, D
features_with_register_tokens = model(x, return_register_tokens=True)

# Apply classification head
logits = model.heads["cls"](features) # B, 10
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