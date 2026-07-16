"""Inspect, render, and compare saved explainability artifacts."""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .artifacts import load_explanation
from .visualization import Normalization, interpolate_token_attribution


@dataclass(frozen=True)
class _Styles:
    enabled: bool

    def label(self, value: str) -> str:
        return f"\x1b[36m{value}\x1b[0m" if self.enabled else value

    def value(self, value: Any) -> str:
        rendered = str(value)
        return f"\x1b[33m{rendered}\x1b[0m" if self.enabled else rendered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect, render, and compare vit.explain artifacts.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Report output format")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="Color policy")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", action="store_true", help="Suppress nonessential status output")
    verbosity.add_argument("--verbose", action="store_true", help="Write diagnostic details to stderr")
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Progress display policy",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Summarize an explanation artifact")
    inspect_parser.add_argument("artifact", type=Path)
    render_parser = subparsers.add_parser("render", help="Render one attribution map")
    render_parser.add_argument("artifact", type=Path)
    render_parser.add_argument("output", type=Path)
    render_parser.add_argument("--batch-index", type=int, default=0)
    render_parser.add_argument(
        "--normalization",
        choices=("none", "minmax", "symmetric", "absolute"),
        default="minmax",
    )
    render_parser.add_argument("--overwrite", action="store_true")
    compare_parser = subparsers.add_parser("compare", help="Compare two attribution artifacts")
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)
    return parser


def _summary(explanation) -> dict[str, Any]:
    return {
        "configuration": dict(explanation.configuration),
        "grid_size": list(explanation.layout.grid_size),
        "has_pixel_attributions": explanation.pixel_attributions is not None,
        "layer_count": len(explanation.layer_attributions),
        "method": explanation.method,
        "modeled_size": list(explanation.layout.modeled_size),
        "original_size": list(explanation.layout.original_size),
        "target_scores_shape": list(explanation.target_scores.shape),
        "token_attributions_shape": list(explanation.token_attributions.shape),
    }


def _color_enabled(output_format: str, color: str, no_color: bool) -> bool:
    if no_color or output_format != "text" or color == "never":
        return False
    return color == "always" or sys.stdout.isatty()


def _progress_enabled(output_format: str, progress: str, quiet: bool) -> bool:
    if quiet or progress == "never":
        return False
    if progress == "always":
        return True
    return output_format == "text" and sys.stderr.isatty()


def _write_progress(command: str, enabled: bool) -> None:
    if enabled:
        print(f"vit-explain: {command} [1/1]", file=sys.stderr)


def _write_report(payload: dict[str, Any], output_format: str, styles: _Styles) -> None:
    if output_format == "json":
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return
    for name, value in payload.items():
        print(f"{styles.label(name)}: {styles.value(value)}")


def _compare(first, second) -> dict[str, Any]:
    if not first.layout.matches(second.layout):
        raise ValueError("artifacts have different token layouts")
    if first.token_attributions.shape != second.token_attributions.shape:
        raise ValueError("artifacts have different token-attribution shapes")
    first_values = first.token_attributions.nan_to_num().flatten().float()
    second_values = second.token_attributions.nan_to_num().flatten().float()
    cosine = float(torch.nn.functional.cosine_similarity(first_values[None], second_values[None]).item())
    difference = first_values - second_values
    return {
        "cosine_similarity": cosine if math.isfinite(cosine) else 0.0,
        "max_absolute_difference": float(difference.abs().max().item()),
        "mean_absolute_difference": float(difference.abs().mean().item()),
    }


def _render(explanation, output: Path, batch_index: int, normalization: str, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace {output}")
    if batch_index < 0 or batch_index >= explanation.token_attributions.shape[0]:
        raise ValueError(f"batch index must be in [0, {explanation.token_attributions.shape[0]})")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "rendering requires the explainability extra: pip install 'vit[explainability]'",
            name="matplotlib",
        ) from None
    output.parent.mkdir(parents=True, exist_ok=True)
    image = (
        interpolate_token_attribution(explanation, normalization=cast(Normalization, normalization))[batch_index]
        .float()
        .numpy()
    )
    color_map = plt.get_cmap("coolwarm" if normalization == "symmetric" else "viridis").with_extremes(bad="#777777")
    plt.imsave(output, image, cmap=color_map)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact-only explainability CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    styles = _Styles(_color_enabled(args.format, args.color, args.no_color))
    show_progress = _progress_enabled(args.format, args.progress, args.quiet)
    try:
        _write_progress(args.command, show_progress)
        if args.command == "inspect":
            explanation = load_explanation(args.artifact)
            _write_report(_summary(explanation), args.format, styles)
        elif args.command == "compare":
            first = load_explanation(args.first)
            second = load_explanation(args.second)
            _write_report(_compare(first, second), args.format, styles)
        elif args.command == "render":
            explanation = load_explanation(args.artifact)
            _render(explanation, args.output, args.batch_index, args.normalization, args.overwrite)
            if not args.quiet:
                _write_report({"output": str(args.output)}, args.format, styles)
        else:  # pragma: no cover - argparse enforces a subcommand
            parser.error(f"unknown command {args.command}")
        if args.verbose:
            print(f"vit-explain: completed {args.command}", file=sys.stderr)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, ModuleNotFoundError) as error:
        print(f"vit-explain failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
