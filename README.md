# ViT

Implementation of Vision Transformer (ViT) that supports both native PyTorch and Transformer Engine backends.

## Installation

The PyTorch backend is easily installable like a normal repository

```bash
pip install convnext @ git+https://github.com/TidalPaladin/vit.git
```

To enable the Transformer Engine backend, manually install Transformer Engine with PyTorch support

```bash
pip install --no-build-isolation "transformer-engine[pytorch]"
```

The Transformer Engine backend is optional, and an `ImportError` will be raised when passing
`backend='te'` without installing Transformer Engine.

## Usage

```python
from vit import ViTConfig

config = ViTConfig(
    in_channels=3,
    patch_size=(16, 16),
    depth=8,
    hidden_size=384,
    ffn_hidden_size=384*4,
    num_attention_heads=384 // 32,
    activation="gelu",
    normalization="LayerNorm",
    hidden_dropout=0.1,
    drop_path_rate=0.1,
    backend="pytorch", # or 'te' for Transformer Engine
)
model = config.instantiate()
```
