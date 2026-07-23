"""Append position-perturbed object pairs to a scene dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tbp.compositional_datasets.perturbed_scene_dataset import (
    PerturbationBounds,
    append_perturbed_scene_objects,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse perturbed object-pair append arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--source-start", type=int, default=101)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--id-offset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-displacement", type=float, default=0.008)
    parser.add_argument("--max-displacement", type=float, default=0.016)
    parser.add_argument("--max-attempts", type=int, default=10_000)
    parser.add_argument(
        "--configured-stamping-script",
        type=Path,
        default=ROOT / "scripts" / "stamp_object_from_config.py",
    )
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--preview-texture-max-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Append perturbed pairs and print the installed target range.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    out_dir = args.source_dataset if args.out_dir is None else args.out_dir
    bounds = PerturbationBounds(
        args.min_displacement,
        args.max_displacement,
        args.max_attempts,
    )
    pairs = append_perturbed_scene_objects(
        args.source_dataset,
        out_dir,
        source_start=args.source_start,
        count=args.count,
        id_offset=args.id_offset,
        seed=args.seed,
        bounds=bounds,
        stamping_script=args.configured_stamping_script,
        blender_executable=args.blender,
        preview_texture_max_size=args.preview_texture_max_size,
    )
    print(
        f"Installed {args.count} perturbed pairs: "
        f"{pairs[0].target_id} through {pairs[-1].target_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
