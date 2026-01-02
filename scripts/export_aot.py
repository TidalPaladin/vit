#!/usr/bin/env python
"""Export a ViT model to AOTInductor format for Rust inference.

This script takes a ViT configuration file and optional weights, then exports
the model using torch.export and AOTInductor to create a .pt2 artifact that
can be loaded from C++/Rust.

IMPORTANT: Set TORCH_COMPILE_DISABLE=1 to avoid FakeTensorMode conflicts
between @torch.compile decorators and torch.export tracing.

Example usage:
    TORCH_COMPILE_DISABLE=1 python scripts/export_aot.py \
        --config config.yaml \
        --weights weights.safetensors \
        --output model.pt2 \
        --shape 1,3,224,224 \
        --device cuda

The exported model can then be loaded using the Rust CLI:
    vit infer --model model.pt2 --config config.yaml --device cuda:0 --shape 1,3,224,224
"""

import argparse
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
from torch import Tensor


# Reset dynamo cache before importing vit modules to avoid FakeTensorMode conflicts
# This is necessary because @torch.compile decorators in the vit package create
# cached compiled graphs that conflict with torch.export's tracing
torch._dynamo.reset()

from vit import ViTConfig  # noqa: E402


class ViTExportWrapper(nn.Module):
    """Wrapper that flattens ViTFeatures output to a single tensor.

    AOTInductor requires simple tensor outputs, so we extract just the
    dense_features tensor from the ViTFeatures dataclass.
    """

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning only dense features.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Dense features tensor of shape (B, L, hidden_size) where
            L = num_patches + num_register_tokens + num_cls_tokens.
        """
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            features = self.vit(x, mask=None, rope_seed=None, output_norm=True)
        return features.dense_features


def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse a comma-separated shape string to a tuple of ints."""
    return tuple(int(x.strip()) for x in shape_str.split(","))


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse a dtype string to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Options: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def main():
    parser = argparse.ArgumentParser(
        description="Export ViT model to AOTInductor format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to safetensors weights file (optional, uses random weights if not provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for .pt2 file",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="1,3,224,224",
        help="Input shape as comma-separated values (default: 1,3,224,224)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export (cpu or cuda, default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type for export (float32, bfloat16, default: float32)",
    )
    args = parser.parse_args()

    # Parse arguments
    shape = parse_shape(args.shape)
    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)

    print(f"Loading config from {args.config}")
    config = ViTConfig.from_yaml(args.config)

    print(f"Instantiating model on {device} with dtype {dtype}")
    model = config.instantiate(device=device)

    # Load weights if provided
    if args.weights is not None:
        print(f"Loading weights from {args.weights}")
        try:
            import safetensors.torch

            state_dict = safetensors.torch.load_file(args.weights)
            model.load_state_dict(state_dict)
        except ImportError:
            raise ImportError("safetensors is required to load weights. Install with: pip install safetensors")

    # Set to eval mode - this disables dropout, drop_path, and RoPE augmentations
    model.eval()

    # Wrap for export
    wrapper = ViTExportWrapper(model)
    wrapper.to(device=device, dtype=dtype)

    # Create example input
    print(f"Creating example input with shape {shape} and dtype {dtype}")
    example_input = torch.randn(shape, device=device, dtype=dtype)

    # Validate the input shape matches config
    expected_channels = config.in_channels
    expected_spatial = tuple(config.img_size)
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input shape (B, C, H, W), got {len(shape)}D")
    if shape[1] != expected_channels:
        raise ValueError(f"Input channels {shape[1]} doesn't match config in_channels {expected_channels}")
    if shape[2:] != expected_spatial:
        raise ValueError(f"Input spatial size {shape[2:]} doesn't match config img_size {expected_spatial}")

    # Test forward pass (with dynamo disabled to avoid compilation during test)
    print("Testing forward pass...")

    @torch.compiler.disable(recursive=True)
    def test_forward(model, x):
        with torch.no_grad():
            return model(x)

    output = test_forward(wrapper, example_input)
    print(f"Output shape: {output.shape}")

    # Reset dynamo cache to clear any state from previous runs
    print("Resetting dynamo cache...")
    torch._dynamo.reset()

    # Re-create example input after reset to ensure clean state
    example_input = torch.randn(shape, device=device, dtype=dtype)

    # Export using torch.export
    # The @torch.compile decorators in the model conflict with torch.export's tracing,
    # so we set TORCH_COMPILE_DISABLE=1 when running this script
    print("Exporting model with torch.export...")

    # Try with dynamic batch dimension first, fall back to static shapes if it fails
    batch_dim = torch.export.Dim("batch", min=1, max=32)
    dynamic_shapes = {"x": {0: batch_dim}}

    @torch.compiler.disable(recursive=True)
    def do_export(model, example, dyn_shapes=None):
        try:
            return torch.export.export(model, (example,), dynamic_shapes=dyn_shapes)
        except Exception as e:
            if dyn_shapes is not None and "specialized" in str(e).lower():
                print(f"Dynamic shapes failed (batch specialized): {e}")
                print("\nRetrying with static shapes...")
                return torch.export.export(model, (example,), dynamic_shapes=None)
            print(f"torch.export failed: {e}")
            print("\nTrying with strict=False...")
            return torch.export.export(model, (example,), strict=False, dynamic_shapes=dyn_shapes)

    exported = do_export(wrapper, example_input, dynamic_shapes)

    # Compile with AOTInductor to produce a .so file
    # We use aot_compile which produces a shared library that can be loaded by C++
    print(f"Compiling with AOTInductor to {args.output}...")

    # Get the GraphModule from the ExportedProgram
    gm = exported.module()

    # Use aot_compile to create a .so file
    so_path = torch._inductor.aot_compile(
        gm,
        args=(example_input,),
        options={"aot_inductor.output_path": str(args.output)},
    )
    print(f"Successfully exported to {so_path}")

    # Print summary
    print("\nExport Summary:")
    print(f"  Config: {args.config}")
    print(f"  Weights: {args.weights or 'random'}")
    print(f"  Output: {args.output}")
    print(f"  Shape: {shape}")
    print(f"  Device: {device}")
    print(f"  DType: {dtype}")
    print(f"  Output shape: {output.shape}")


if __name__ == "__main__":
    main()
