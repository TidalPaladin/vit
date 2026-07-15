# 2D RoPE provenance

- Status: repository-owned replacement
- Authored: July 15, 2026
- License: Apache-2.0

The implementation in `vit/rope.py` was written for this repository from the rotary-position-embedding equation and
the package's established public contract. It assigns one frequency band to each image axis, maps cell centers to the
configured coordinate range, and applies the resulting angles to paired attention features.

The mathematical basis is the rotation-matrix formulation described in [RoFormer: Enhanced Transformer with Rotary
Position Embedding](https://arxiv.org/abs/2104.09864). The axial 2D layout, coordinate normalization modes, period
configuration, and training-time coordinate augmentation preserve this package's established behavior.

## Replacement review

- The replacement preserves the public API, state-dict key, and observable behavior that callers rely on.
- The implementation was authored from the mathematical contract and parity tests. No DINOv3 source text was copied,
  adapted, or retained.
- Mathematical parity tests cover multiple rectangular image sizes, all coordinate normalization modes, floating-point
  dtypes, and available devices.
- Existing tests cover deterministic seeded augmentation, unseeded training augmentation, and evaluation behavior.
- `vit/rope.py` carries the same Apache-2.0 identifier as the repository and distribution metadata.
