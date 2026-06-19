"""Generate the five canonical primitive 2D sticker PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tbp.compositional_datasets.sticker_assets import (
    DEFAULT_PRIMITIVE_PADDING_RATIO,
    DEFAULT_STICKER_SIZE,
    generate_primitive_stickers,
)

DEFAULT_BACKGROUND_COLOR_HEX = "#ffffff"
DEFAULT_SHAPE_COLOR_HEX = "#2f2b5c"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse primitive sticker generation arguments.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=Path("assets/2D_stickers"),
        type=Path,
        help="Directory where primitive PNG sticker files are written.",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_STICKER_SIZE,
        type=int,
        help="Square output size in pixels.",
    )
    parser.add_argument(
        "--padding-ratio",
        default=DEFAULT_PRIMITIVE_PADDING_RATIO,
        type=float,
        help="Fraction of the canvas reserved around the primitive.",
    )
    parser.add_argument(
        "--background-color",
        default=DEFAULT_BACKGROUND_COLOR_HEX,
        help="Circular sticker backing color.",
    )
    parser.add_argument(
        "--shape-color",
        default=DEFAULT_SHAPE_COLOR_HEX,
        help="Primitive shape fill color.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate primitive sticker PNG files.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    outputs = generate_primitive_stickers(
        args.out_dir,
        size=args.size,
        padding_ratio=args.padding_ratio,
        background_color=args.background_color,
        shape_color=args.shape_color,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
