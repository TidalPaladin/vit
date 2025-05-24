from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


# Built upon implementation from https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py


def init_random_nd_freqs(dim: int, num_heads: int, spatial_dims: int = 2, theta: float = 10.0):
    """
    Initialize random N-dimensional frequency bases for RoPE.
    Only requires dim % 4 == 0 (same as original 2D version).
    """
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"

    # Same magnitude calculation as original
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    all_freqs = []

    for dim_idx in range(spatial_dims):
        dim_freqs = []

        for head in range(num_heads):
            # Generate random unit vector for this head and spatial dimension
            if spatial_dims == 1:
                # 1D case: just use random phase
                angle = torch.rand(1) * 2 * torch.pi
                cos_comp, sin_comp = torch.cos(angle), torch.sin(angle)
            elif spatial_dims == 2:
                # 2D case: random angle (same as original)
                angle = torch.rand(1) * 2 * torch.pi
                cos_comp = torch.cos(angle) if dim_idx == 0 else torch.sin(angle)
                sin_comp = torch.cos(angle + torch.pi / 2) if dim_idx == 0 else torch.sin(angle + torch.pi / 2)
            else:
                # N-D case: sample from unit sphere and use coordinates
                # Generate random point on unit sphere
                random_vec = torch.randn(spatial_dims)
                random_vec = random_vec / torch.norm(random_vec)
                cos_comp = random_vec[dim_idx]
                sin_comp = random_vec[(dim_idx + 1) % spatial_dims]

            # Create frequency components (same structure as original)
            freq_components = torch.cat([mag * cos_comp, mag * sin_comp], dim=-1)

            dim_freqs.append(freq_components)

        all_freqs.append(torch.stack(dim_freqs, dim=0))

    # Output: [spatial_dims, num_heads, dim//2]
    freqs = torch.stack(all_freqs, dim=0)
    return freqs


def generate_random_orthonormal_basis(n_dims: int):
    """Generate a random orthonormal basis using QR decomposition."""
    # Generate random matrix
    random_matrix = torch.randn(n_dims, n_dims)

    # QR decomposition gives us an orthonormal basis
    Q, R = torch.linalg.qr(random_matrix)

    # Ensure positive diagonal elements for consistency
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)

    return Q


# @torch.compile(fullgraph=True, dynamic=False)
def compute_mixed_cis_nd(freqs: Tensor, positions: Tensor, num_heads: int):
    """
    Compute mixed complex exponentials for N-dimensional RoPE with batch support.

    Args:
        freqs: [spatial_dims, num_heads, freq_dim] frequency tensor
        positions: [B, L, spatial_dims] or [L, spatial_dims] position coordinates
        num_heads: number of attention heads

    Returns:
        freqs_cis: [B, num_heads, L, freq_dim] or [num_heads, L, freq_dim] complex tensor
    """
    spatial_dims, num_heads_freq, freq_dim = freqs.shape

    # Handle both batched and non-batched positions
    if positions.ndim == 3:
        # Batched case: [B, L, spatial_dims]
        B, L, pos_dims = positions.shape
        is_batched = True
    else:
        # Non-batched case: [L, spatial_dims]
        L, pos_dims = positions.shape
        is_batched = False
        positions = positions.unsqueeze(0)  # Add batch dim: [1, L, spatial_dims]

    assert spatial_dims == pos_dims, f"Spatial dims mismatch: freqs has {spatial_dims}, positions has {pos_dims}"
    assert num_heads == num_heads_freq, f"Head count mismatch: expected {num_heads}, got {num_heads_freq}"

    # No float 16 for this range
    with torch.autocast(device_type=freqs.device.type, enabled=False):
        total_freqs = None

        for dim_idx in range(spatial_dims):
            # positions[:, :, dim_idx]: [B, L] - position coordinates for this spatial dimension
            # freqs[dim_idx]: [num_heads, freq_dim] - frequencies for this spatial dimension

            # Compute frequency values: [B, L, num_heads, freq_dim]
            pos_dim = positions[:, :, dim_idx]  # [B, L]
            freq_dim_tensor = freqs[dim_idx]  # [num_heads, freq_dim]

            # Broadcast and multiply: [B, L, 1, 1] * [1, 1, num_heads, freq_dim] = [B, L, num_heads, freq_dim]
            dim_freqs = pos_dim.unsqueeze(-1).unsqueeze(-1) * freq_dim_tensor.unsqueeze(0).unsqueeze(0)

            # Sum across all spatial dimensions
            if total_freqs is None:
                total_freqs = dim_freqs
            else:
                total_freqs = total_freqs + dim_freqs

        # Convert to complex exponentials: [B, L, num_heads, freq_dim]
        assert isinstance(total_freqs, Tensor)
        freqs_cis = torch.polar(torch.ones_like(total_freqs), total_freqs)

        # Transpose to match Q/K layout: [B, num_heads, L, freq_dim]
        freqs_cis = freqs_cis.transpose(-2, -3)

    # Remove batch dimension if input wasn't batched
    if not is_batched:
        freqs_cis = freqs_cis.squeeze(0)  # [num_heads, L, freq_dim]

    return freqs_cis


# @torch.compile(fullgraph=True, dynamic=False)
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to input tensor.
    Works for any number of spatial dimensions and supports batch dimensions.

    Args:
        x: Input tensor [B, H, L, D] or [H, L, D]
        freqs_cis: Complex frequencies [B, H, L, D//2] or [H, L, D//2]

    Returns:
        Rotated tensor with same shape as input
    """
    # Convert to complex representation
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Apply rotation
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)

    return x_out.type_as(x).to(x.device)


class MixedRoPE(nn.Module):
    freqs: nn.Parameter

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tokenized_size: Sequence[int],
        rope_theta: float = 100.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tokenized_size = tokenized_size
        self.rope_theta = rope_theta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        head_dim = self.hidden_size // self.num_heads
        freqs = init_random_nd_freqs(head_dim, self.num_heads, len(self.tokenized_size), self.rope_theta)
        self.freqs = nn.Parameter(freqs)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        if x.shape[0] != pos.shape[0]:
            raise ValueError(f"x and pos must have the same batch dimension, got {x.shape} and {pos.shape}")
        if x.shape[1] != self.num_heads:
            raise ValueError(
                f"x must have the same number of heads as the RoPE module, got {x.shape[1]} and {self.num_heads}"
            )
        if x.shape[2] != pos.shape[1]:
            raise ValueError(f"x must have the same sequence length as pos, got {x.shape} and {pos.shape}")
        if x.shape[3] != self.hidden_size // self.num_heads:
            raise ValueError(
                f"x must have the same head dimension as the RoPE module, got {x.shape} and {self.hidden_size // self.num_heads}"
            )
        freqs_cis = compute_mixed_cis_nd(self.freqs, pos, self.num_heads)
        return apply_rotary_emb(x, freqs_cis)
