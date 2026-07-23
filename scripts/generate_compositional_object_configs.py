"""Generate deterministic six-sticker object configs for a compositional dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from tbp.compositional_datasets.batch_configs import generate_batch_configs

DEFAULT_OUTDIR = Path("~/tbp/data/compositional_objects_1.2")
DEFAULT_NUM_OBJECTS = 100
DEFAULT_BASE_CONFIG = ROOT / "configs" / "six_sticker_stamp.json"
DEFAULT_PARENT_DIR = ROOT / "assets" / "3D_objects"
DEFAULT_STICKER_DIR = ROOT / "assets" / "2D_stickers"


def _default_parents() -> list[Path]:
    """Return default parent GLB assets sorted by path.

    Returns:
        Sorted parent GLB paths.
    """
    return sorted(DEFAULT_PARENT_DIR.glob("*.glb"))


def _default_stickers() -> list[Path]:
    """Return default sticker PNG assets sorted by path.

    Returns:
        Sorted sticker PNG paths.
    """
    return sorted(DEFAULT_STICKER_DIR.glob("*.png"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse batch config generation arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTDIR,
        type=Path,
        help=f"Output dataset directory. Defaults to {DEFAULT_OUTDIR}.",
    )
    parser.add_argument(
        "--count",
        default=DEFAULT_NUM_OBJECTS,
        type=int,
        help="Number of object configs to generate.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Required random seed for deterministic generation.",
    )
    parser.add_argument(
        "--parents",
        nargs="*",
        type=Path,
        default=None,
        help="Parent GLB paths. Defaults to assets/3D_objects/*.glb.",
    )
    parser.add_argument(
        "--stickers",
        nargs="*",
        type=Path,
        default=None,
        help="Sticker PNG paths. Defaults to assets/2D_stickers/*.png.",
    )
    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        type=Path,
        help=(
            "Base six-sticker layout config whose slot geometry is copied "
            "while stickers and rotations are randomized."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate batch configs and print the manifest path.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    parents = _default_parents() if args.parents is None else args.parents
    stickers = _default_stickers() if args.stickers is None else args.stickers
    manifest_path = generate_batch_configs(
        args.out_dir,
        args.count,
        args.seed,
        parents,
        stickers,
        args.base_config,
        ROOT / "scripts" / "stamp_object_from_config.py",
        ROOT,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
