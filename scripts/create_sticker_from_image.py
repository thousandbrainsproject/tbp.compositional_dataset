"""Convert a custom transparent 2D image into a circular 2D sticker PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from tbp.compositional_datasets.sticker_assets import (
    DEFAULT_PADDING_RATIO,
    DEFAULT_STICKER_SIZE,
    create_sticker_from_image,
    save_png,
)

DEFAULT_BACKGROUND_COLOR_HEX = "#ffffff"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse custom sticker conversion arguments.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source image path.")
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path.")
    parser.add_argument(
        "--size",
        default=DEFAULT_STICKER_SIZE,
        type=int,
        help="Square output size in pixels.",
    )
    parser.add_argument(
        "--padding-ratio",
        default=DEFAULT_PADDING_RATIO,
        type=float,
        help="Fraction of the canvas reserved around the artwork.",
    )
    parser.add_argument(
        "--background-color",
        default=DEFAULT_BACKGROUND_COLOR_HEX,
        help="Circular sticker backing color.",
    )
    parser.add_argument(
        "--preserve-canvas",
        action="store_true",
        help="Preserve the full source canvas instead of cropping alpha bounds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Convert a source image into a circular sticker.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    with Image.open(args.input) as source:
        sticker = create_sticker_from_image(
            source,
            size=args.size,
            padding_ratio=args.padding_ratio,
            background_color=args.background_color,
            preserve_canvas=args.preserve_canvas,
        )
    save_png(sticker, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
