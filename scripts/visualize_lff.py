#!/usr/bin/env python3
import math
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from vit.pos_enc import LearnableFourierFeatures, create_grid, fourier_features_only


def create_position_grid(grid_size: List[int], spatial_dims: int) -> torch.Tensor:
    """Create a grid of positions for the given spatial dimensions."""
    if spatial_dims == 2:
        h, w = grid_size
        y_coords = torch.arange(h).float()
        x_coords = torch.arange(w).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
    elif spatial_dims == 3:
        d, h, w = grid_size
        z_coords = torch.arange(d).float()
        y_coords = torch.arange(h).float()
        x_coords = torch.arange(w).float()
        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        positions = torch.stack([grid_z.flatten(), grid_y.flatten(), grid_x.flatten()], dim=-1)
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    return positions


def compute_similarity_heatmap(
    grid_size: List[int],
    query_pos: List[float],
    embed_dim: int,
    spatial_dims: int,
    gamma: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    """
    Compute similarity heatmap between a query position and all positions in the grid.

    Args:
        positions: [L, spatial_dims] position coordinates
        query_pos: Query position coordinates
        embed_dim: Dimension per attention head
        num_heads: Number of attention heads
        spatial_dims: Number of spatial dimensions
        gamma: Gamma parameter for LFF
        seed: Random seed for reproducibility

    Returns:
        similarities: [L] similarity scores between query and all positions
    """
    # Initialize RoPE frequencies
    torch.random.manual_seed(seed)
    layer = LearnableFourierFeatures(spatial_dims, embed_dim, gamma=gamma)
    layer.eval()

    # Apply RoPE to get position-dependent embeddings
    with torch.no_grad():
        embeddings = fourier_features_only(grid_size, layer.fourier.weight, layer.fourier.bias, True)

    # Get query position embedding
    query_pos_tensor = embeddings.view(*grid_size, embed_dim)
    for p in query_pos:
        query_pos_tensor = query_pos_tensor[int(p)]

    # Compute inner products (similarities)
    similarities = (embeddings @ query_pos_tensor).flatten()
    return similarities


def plot_2d_heatmap(similarities: torch.Tensor, grid_size: List[int], query_pos: List[float], save_path: str = None):
    """Plot 2D heatmap of similarities."""
    h, w = grid_size
    heatmap_data = similarities.reshape(h, w).numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, cmap="coolwarm", center=0, annot=False, fmt=".2f", cbar_kws={"label": "Similarity"}, vmin=0
    )

    # Mark query position
    query_y, query_x = query_pos
    plt.scatter(query_x + 0.5, query_y + 0.5, color="red", s=100, marker="x", linewidth=3)

    plt.title(f"LFF Embedding Similarities (Query at {query_pos})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 2D heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_3d_heatmap(similarities: torch.Tensor, grid_size: List[int], query_pos: List[float], save_path: str = None):
    """Plot 3D heatmap using cross-sections."""
    d, h, w = grid_size
    heatmap_data = similarities.reshape(d, h, w).numpy()

    # Calculate global min/max for consistent color scale across all slices
    global_vmin = heatmap_data.min()
    global_vmax = heatmap_data.max()

    # Calculate grid layout: 4 slices per row
    cols_per_row = 4
    rows = math.ceil(d / cols_per_row)
    cols = min(d, cols_per_row)

    # Create subplots for all z-slices
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Handle case where we have only one row or one subplot
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        # axes is already a 2D array
        pass

    query_z, query_y, query_x = query_pos

    # Plot all slices
    for i in range(d):
        row = i // cols_per_row
        col = i % cols_per_row

        if rows == 1:
            ax = axes[col] if cols > 1 else axes[0]
        else:
            ax = axes[row][col] if cols > 1 else axes[row][0]

        slice_data = heatmap_data[i]
        im = ax.imshow(slice_data, cmap="coolwarm", aspect="auto", vmin=global_vmin, vmax=global_vmax)
        ax.set_title(f"Z-slice {i}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Mark query position if it's in this slice
        if abs(i - query_z) < 0.5:
            ax.scatter(query_x, query_y, color="red", s=100, marker="x", linewidth=3)

        plt.colorbar(im, ax=ax, label="Similarity")

    # Hide any unused subplots
    total_subplots = rows * cols
    for i in range(d, total_subplots):
        row = i // cols_per_row
        col = i % cols_per_row

        if rows == 1:
            ax = axes[col] if cols > 1 else axes[0]
        else:
            ax = axes[row][col] if cols > 1 else axes[row][0]
        ax.set_visible(False)

    plt.suptitle(f"RoPE Embedding Similarities - 3D Cross-sections (Query at {query_pos})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 3D cross-sections to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_distance_relationship(
    similarities: torch.Tensor,
    positions: torch.Tensor,
    query_pos: List[float],
    spatial_dims: int,
    save_path: str = None,
):
    """Analyze how similarity relates to distance from query position."""
    query_tensor = torch.tensor(query_pos).float()

    # Compute distances from query position
    distances = torch.norm(positions - query_tensor, dim=1)

    # Sort by distance for cleaner plot
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_similarities = similarities[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_distances.numpy(), sorted_similarities.numpy(), alpha=0.6, s=20)
    plt.xlabel("Distance from Query Position")
    plt.ylabel("Similarity")
    plt.title(f"LFF Similarity vs Distance ({spatial_dims}D)")
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(sorted_distances.numpy(), sorted_similarities.numpy(), 2)
    p = np.poly1d(z)
    x_trend = np.linspace(sorted_distances.min(), sorted_distances.max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label="Trend")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved distance analysis to {save_path}")
    else:
        plt.show()

    plt.close()


@click.command()
@click.option(
    "--spatial-dims", type=click.Choice(["2", "3"]), default="2", help="Number of spatial dimensions (2 or 3)"
)
@click.option(
    "--grid-size",
    type=str,
    default="16,16",
    help='Grid size as comma-separated values (e.g., "16,16" for 2D or "8,16,16" for 3D)',
)
@click.option(
    "--query-pos",
    type=str,
    default="8,8",
    help='Query position as comma-separated values (e.g., "8,8" for 2D or "4,8,8" for 3D)',
)
@click.option("--embed-dim", type=int, default=64, help="Embedding dimension")
@click.option("--gamma", type=float, default=1.0, help="Gamma parameter for LFF")
@click.option(
    "--output-dir", type=str, default=None, help="Directory to save plots (if not provided, plots will be displayed)"
)
@click.option("--analyze-distance", is_flag=True, help="Also generate distance vs similarity analysis")
@click.option("--seed", type=int, default=0, help="Random seed for reproducibility")
def main(spatial_dims, grid_size, query_pos, embed_dim, gamma, output_dir, analyze_distance, seed):
    """
    Visualize LFF embeddings to sanity check the implementation.

    This tool creates visualizations showing how LFF embeddings preserve
    positional relationships by computing similarities between a query position
    and all positions in a grid.
    """
    # Parse parameters
    spatial_dims = int(spatial_dims)
    grid_size = [int(x) for x in grid_size.split(",")]
    query_pos = [float(x) for x in query_pos.split(",")]

    # Validate parameters
    if len(grid_size) != spatial_dims:
        raise ValueError(f"Grid size must have {spatial_dims} dimensions, got {len(grid_size)}")
    if len(query_pos) != spatial_dims:
        raise ValueError(f"Query position must have {spatial_dims} dimensions, got {len(query_pos)}")
    if embed_dim % 4 != 0:
        raise ValueError(f"Head dimension must be divisible by 4, got {embed_dim}")

    print(f"Visualizing {spatial_dims}D LFF embeddings")
    print(f"Grid size: {grid_size}")
    print(f"Query position: {query_pos}")
    print(f"Gamma: {gamma}")

    # Compute similarities
    print("Computing similarities...")
    similarities = compute_similarity_heatmap(grid_size, query_pos, embed_dim, spatial_dims, gamma, seed)

    # Generate visualizations
    if spatial_dims == 2:
        save_path = f"{output_dir}/lff_2d_heatmap.png" if output_dir else None
        plot_2d_heatmap(similarities, grid_size, query_pos, save_path)
    else:
        save_path = f"{output_dir}/lff_3d_heatmap.png" if output_dir else None
        plot_3d_heatmap(similarities, grid_size, query_pos, save_path)

    if analyze_distance:
        save_path = f"{output_dir}/lff_distance_analysis.png" if output_dir else None
        analyze_distance_relationship(similarities, positions, query_pos, spatial_dims, save_path)

    # Print some statistics
    print(f"\nSimilarity statistics:")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std:  {similarities.std():.4f}")
    print(f"  Min:  {similarities.min():.4f}")
    print(f"  Max:  {similarities.max():.4f}")


if __name__ == "__main__":
    main()
