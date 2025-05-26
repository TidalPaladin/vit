# ViT

Implementation of Vision Transformer (ViT) in native PyTorch, accelerated by `torch.compile`.
Supports modern enhancements like RMSNorm, SwiGLU, register tokens, and different positional encodings.

## Installation

The PyTorch backend is easily installable like a normal repository

```bash
pip install vit @ git+https://github.com/TidalPaladin/vit.git
```

## Usage

```python
from vit import ViTConfig

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
    pos_emb="fourier",
)
model = config.instantiate()
```

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