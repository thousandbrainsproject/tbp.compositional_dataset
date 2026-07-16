"""Pure helpers for producing paired, perturbed sticker layouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any

from tbp.compositional_datasets.generation_configs import blender_stamp_command_args
from tbp.compositional_datasets.scene_dataset import (
    _normalize_texture_sidecar,
    _run_render_command,
    _write_json,
    scene_dataset_config,
)

MIN_STAMP_COVERAGE = 0.95
RENDERED_ANCHOR_PROVENANCE_ABS_TOLERANCE = 1e-6
GEOMETRY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ObjectPair:
    """Paths for one source object and its offset-ID target object."""

    source_id: str
    target_id: str
    source_generation_config: Path
    source_object_config: Path
    source_mesh_dir: Path
    target_generation_config: Path
    target_object_config: Path
    target_mesh_dir: Path


@dataclass(frozen=True)
class PerturbationBounds:
    """Bounds controlling independent sticker-position perturbations."""

    min_displacement: float = 0.008
    max_displacement: float = 0.016
    max_attempts: int = 10_000

    def __post_init__(self) -> None:
        """Validate perturbation bounds after initialization."""
        if not math.isfinite(self.min_displacement) or self.min_displacement <= 0:
            raise ValueError("min_displacement must be positive and finite")
        if not math.isfinite(self.max_displacement):
            raise ValueError("max_displacement must be finite")
        if self.max_displacement < self.min_displacement:
            raise ValueError("max_displacement must be at least min_displacement")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


def target_object_id(source_id: str, id_offset: int) -> str:
    """Offset a three-digit source object ID while preserving its suffix.

    Args:
        source_id: Source object ID beginning with three digits and an underscore.
        id_offset: Positive integer added to the numeric ID prefix.

    Returns:
        Target object ID with the offset numeric prefix.

    Raises:
        ValueError: If the source ID, offset, or resulting ID is invalid.
    """
    match = re.fullmatch(r"(\d{3})_(.+)", source_id)
    if match is None or id_offset <= 0:
        raise ValueError(f"invalid source ID or offset: {source_id}, {id_offset}")
    target_number = int(match.group(1)) + id_offset
    if target_number > 999:
        raise ValueError("target object ID exceeds three digits")
    return f"{target_number:03d}_{match.group(2)}"


def discover_object_pairs(
    source_dataset: Path,
    out_dir: Path,
    *,
    source_start: int,
    count: int,
    id_offset: int,
) -> tuple[ObjectPair, ...]:
    """Discover complete source objects and preflight their target paths.

    Args:
        source_dataset: Dataset containing source generation, object, and mesh files.
        out_dir: Dataset directory where target artifacts will be written.
        source_start: First three-digit numeric source prefix to discover.
        count: Number of consecutive numeric source prefixes to discover.
        id_offset: Required offset of exactly 100 for the protected pair domain.

    Returns:
        Immutable source-target records in numeric source order.

    Raises:
        FileNotFoundError: If a requested source or required source artifact is missing.
        FileExistsError: If any target artifact already exists.
        ValueError: If IDs leave the protected pair domain or a numeric prefix
            matches multiple generation configs.
    """
    source_end = source_start + count - 1
    if count <= 0 or source_start < 101 or source_end > 200 or id_offset != 100:
        raise ValueError(
            "protected object ID domain requires sources 101-200 and id_offset 100"
        )

    pairs = []
    for source_number in range(source_start, source_start + count):
        matches = sorted(
            (source_dataset / "generation_configs").glob(f"{source_number:03d}_*.json")
        )
        if not matches:
            raise FileNotFoundError(f"source object {source_number:03d} was not found")
        if len(matches) != 1:
            raise ValueError(f"source object {source_number:03d} is ambiguous")

        source_generation_config = matches[0]
        source_id = source_generation_config.stem
        source_object_config = (
            source_dataset / "configs" / f"{source_id}.object_config.json"
        )
        source_mesh_dir = source_dataset / "meshes" / source_id
        required_paths = (
            source_object_config,
            source_mesh_dir / "textured.glb",
            source_mesh_dir / "textured.json",
        )
        for required_path in required_paths:
            if not required_path.exists():
                raise FileNotFoundError(
                    f"source object {source_id} is missing {required_path.name}"
                )

        target_id = target_object_id(source_id, id_offset)
        pairs.append(
            ObjectPair(
                source_id=source_id,
                target_id=target_id,
                source_generation_config=source_generation_config,
                source_object_config=source_object_config,
                source_mesh_dir=source_mesh_dir,
                target_generation_config=out_dir
                / "generation_configs"
                / f"{target_id}.json",
                target_object_config=out_dir
                / "configs"
                / f"{target_id}.object_config.json",
                target_mesh_dir=out_dir / "meshes" / target_id,
            )
        )

    for pair in pairs:
        target_paths = (
            pair.target_generation_config,
            pair.target_object_config,
            pair.target_mesh_dir,
        )
        if any(path.exists() for path in target_paths):
            raise FileExistsError(f"target object {pair.target_id} already exists")
    return tuple(pairs)


def target_object_config(source: dict[str, Any], target_id: str) -> dict[str, Any]:
    """Copy a Habitat object config and point it at a target mesh.

    Args:
        source: Source Habitat object configuration.
        target_id: Target object identifier used in the mesh path.

    Returns:
        Deep-copied object configuration with its render asset updated.
    """
    target = deepcopy(source)
    target["render_asset"] = f"../meshes/{target_id}/textured.glb"
    return target


def verify_rendered_pair(
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    source_metadata: dict[str, Any],
    target_metadata: dict[str, Any],
    bounds: PerturbationBounds,
) -> None:
    """Verify a rendered target preserves its source except for slot offsets.

    Args:
        source_config: Original six-slot generation configuration.
        target_config: Perturbed six-slot generation configuration.
        source_metadata: Metadata emitted when rendering the source object.
        target_metadata: Metadata emitted when rendering the target object.
        bounds: Allowed configured displacement bounds.

    Raises:
        ValueError: If the configs or rendered metadata do not form a valid pair.
    """
    validate_perturbed_config(source_config, target_config, bounds)
    if source_metadata["parent_mesh_path"] != target_metadata["parent_mesh_path"]:
        raise ValueError("rendered parent mesh changed")

    source_stickers = source_metadata["stickers"]
    target_stickers = target_metadata["stickers"]
    configured_slots = source_config["slots"]
    if len(source_stickers) != len(configured_slots) or len(target_stickers) != len(
        configured_slots
    ):
        raise ValueError("rendered sticker slots do not match configuration")

    source_records = {record["slot_name"]: record for record in source_stickers}
    target_records = {record["slot_name"]: record for record in target_stickers}
    slot_names = [slot["name"] for slot in configured_slots]
    expected_slot_names = set(slot_names)
    if (
        set(source_records) != expected_slot_names
        or set(target_records) != expected_slot_names
    ):
        raise ValueError("rendered sticker slots do not match configuration")

    source_slots = {slot["name"]: slot for slot in source_config["slots"]}
    target_slots = {slot["name"]: slot for slot in target_config["slots"]}
    axes = {"x": 0, "y": 1, "z": 2}
    right_axis = axes[target_config["placement"]["right_axis"]]
    up_axis = axes[target_config["placement"]["up_axis"]]
    for slot_name in slot_names:
        source_record = source_records[slot_name]
        target_record = target_records[slot_name]
        if source_record["side"] != target_record["side"]:
            raise ValueError("rendered sticker side changed")
        if source_record["sticker_path"] != target_record["sticker_path"]:
            raise ValueError("rendered sticker path changed")
        if source_record["rotation_deg"] != target_record["rotation_deg"]:
            raise ValueError("rendered sticker rotation changed")
        try:
            sizes_match = all(
                math.isclose(source_size, target_size, abs_tol=1e-9)
                for source_size, target_size in zip(
                    source_record["size"], target_record["size"], strict=True
                )
            )
        except ValueError:
            sizes_match = False
        if not sizes_match:
            raise ValueError("rendered sticker size changed")
        if (
            target_record["texture_stamp_coverage_ratio"]
            < target_config["min_coverage"]
        ):
            raise ValueError("rendered sticker coverage is below minimum")

        source_anchor = source_record["world_space_anchor_point"]
        target_anchor = target_record["world_space_anchor_point"]
        source_offset = source_slots[slot_name]["offset"]
        target_offset = target_slots[slot_name]["offset"]
        rendered_displacement = (
            target_anchor[right_axis] - source_anchor[right_axis],
            target_anchor[up_axis] - source_anchor[up_axis],
        )
        configured_displacement = (
            target_offset[0] - source_offset[0],
            target_offset[1] - source_offset[1],
        )
        for component, rendered, configured in zip(
            ("right", "up"),
            rendered_displacement,
            configured_displacement,
            strict=True,
        ):
            if math.isclose(
                rendered,
                configured,
                rel_tol=0.0,
                abs_tol=RENDERED_ANCHOR_PROVENANCE_ABS_TOLERANCE,
            ):
                continue
            raise ValueError(
                "rendered anchor displacement changed for "
                f"slot {slot_name} component {component}: "
                f"expected {configured!r}, actual {rendered!r}"
            )


def _projected_square_bounds(
    slot: dict[str, Any], side_length: float
) -> tuple[float, float, float, float]:
    """Return conservative rotated-square bounds for one sticker slot.

    Args:
        slot: Sticker slot containing an offset and optional rotation in degrees.
        side_length: Side length of the conservative square footprint.

    Returns:
        Minimum right, maximum right, minimum up, and maximum up bounds.
    """
    angle = math.radians(float(slot.get("rotation_deg", 0.0)))
    half_extent = side_length / 2.0 * (
        abs(math.cos(angle)) + abs(math.sin(angle))
    )
    right, up = slot["offset"]
    return (
        right - half_extent,
        right + half_extent,
        up - half_extent,
        up + half_extent,
    )


def validate_perturbed_config(
    source: dict[str, Any],
    target: dict[str, Any],
    bounds: PerturbationBounds,
) -> None:
    """Validate a perturbed six-sticker configuration against its source.

    Args:
        source: Original sticker layout configuration.
        target: Candidate configuration with perturbed slot offsets.
        bounds: Allowed radial displacement and sampling bounds.

    Raises:
        ValueError: If non-offset data changed or a geometric invariant fails.
    """
    source_non_offsets = deepcopy(source)
    target_non_offsets = deepcopy(target)
    for config in (source_non_offsets, target_non_offsets):
        for slot in config["slots"]:
            slot.pop("offset", None)
    if source_non_offsets != target_non_offsets:
        raise ValueError("source and target differ outside slot offsets")

    for source_slot, target_slot in zip(source["slots"], target["slots"], strict=True):
        displacement = math.dist(source_slot["offset"], target_slot["offset"])
        if not (
            bounds.min_displacement - GEOMETRY_TOLERANCE
            <= displacement
            <= bounds.max_displacement + GEOMETRY_TOLERANCE
        ):
            raise ValueError("slot displacement is outside perturbation bounds")

    minimum_spacing = max(0.025, target["max_longest_side"] + target["gap"])
    sides = ("front", "back")
    if any(slot["side"] not in sides for slot in source["slots"]) or any(
        not any(slot["side"] == side for slot in source["slots"]) for side in sides
    ):
        raise ValueError("slots must use front and back sides")
    for side in sides:
        source_slots = {
            slot["name"].rsplit("_", maxsplit=1)[-1]: slot
            for slot in source["slots"]
            if slot["side"] == side
        }
        target_slots = {
            slot["name"].rsplit("_", maxsplit=1)[-1]: slot
            for slot in target["slots"]
            if slot["side"] == side
        }
        if set(source_slots) != {"top", "left", "right"}:
            raise ValueError(f"side {side} must contain top, left, and right slots")

        top = target_slots["top"]["offset"]
        left = target_slots["left"]["offset"]
        right = target_slots["right"]["offset"]
        if top[1] <= left[1] or top[1] <= right[1]:
            raise ValueError(f"top slot is not above both lower slots on side {side}")
        if left[0] >= right[0]:
            raise ValueError(f"left slot is not left of right slot on side {side}")

        source_top = source_slots["top"]["offset"]
        source_left = source_slots["left"]["offset"]
        source_right = source_slots["right"]["offset"]
        source_area = (source_left[0] - source_top[0]) * (
            source_right[1] - source_top[1]
        ) - (source_left[1] - source_top[1]) * (
            source_right[0] - source_top[0]
        )
        target_area = (left[0] - top[0]) * (right[1] - top[1]) - (
            left[1] - top[1]
        ) * (right[0] - top[0])
        if source_area == 0 or target_area == 0 or source_area * target_area < 0:
            raise ValueError(f"triangle orientation changed on side {side}")

        for first, second in combinations(target_slots.values(), 2):
            if math.dist(first["offset"], second["offset"]) < minimum_spacing:
                raise ValueError(f"slots overlap on side {side}")
            first_bounds = _projected_square_bounds(
                first, float(target["max_longest_side"])
            )
            second_bounds = _projected_square_bounds(
                second, float(target["max_longest_side"])
            )
            right_overlap = min(first_bounds[1], second_bounds[1]) - max(
                first_bounds[0], second_bounds[0]
            )
            up_overlap = min(first_bounds[3], second_bounds[3]) - max(
                first_bounds[2], second_bounds[2]
            )
            if (
                right_overlap > GEOMETRY_TOLERANCE
                and up_overlap > GEOMETRY_TOLERANCE
            ):
                raise ValueError(
                    f"sticker footprints overlap: {first['name']} and {second['name']}"
                )


def perturb_stamp_config(
    source: dict[str, Any],
    rng: random.Random,
    bounds: PerturbationBounds,
) -> dict[str, Any]:
    """Sample a valid independently perturbed offset for every sticker slot.

    Args:
        source: Original sticker layout configuration.
        rng: Seedable random-number generator.
        bounds: Allowed radial displacement and sampling bounds.

    Returns:
        Deep-copied configuration containing valid perturbed offsets.

    Raises:
        ValueError: If no valid candidate is found within the attempt limit.
    """
    last_error = "no candidate"
    for _attempt in range(bounds.max_attempts):
        candidate = deepcopy(source)
        for slot in candidate["slots"]:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(bounds.min_displacement, bounds.max_displacement)
            slot["offset"][0] += radius * math.cos(angle)
            slot["offset"][1] += radius * math.sin(angle)
        try:
            validate_perturbed_config(source, candidate, bounds)
        except ValueError as error:
            last_error = str(error)
            continue
        return candidate
    raise ValueError(
        f"could not sample a valid perturbed layout after {bounds.max_attempts} attempts: {last_error}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a file.

    Args:
        path: JSON file to read.

    Returns:
        Parsed JSON object.
    """
    return json.loads(path.read_text())


def append_perturbed_scene_objects(
    source_dataset: Path,
    out_dir: Path,
    *,
    source_start: int,
    count: int,
    id_offset: int,
    seed: int,
    bounds: PerturbationBounds,
    stamping_script: Path,
    blender_executable: str = "blender",
    preview_texture_max_size: int = 512,
) -> tuple[ObjectPair, ...]:
    """Render, verify, and install a staged batch of perturbed object pairs.

    Args:
        source_dataset: Dataset containing the complete source objects.
        out_dir: Dataset directory where verified target artifacts are installed.
        source_start: First three-digit numeric source prefix to render.
        count: Number of consecutive source objects to render.
        id_offset: Required offset of exactly 100 for protected target object IDs.
        seed: Random seed used for deterministic offset perturbations.
        bounds: Allowed displacement and perturbation sampling bounds.
        stamping_script: Blender stamping script used to render each target.
        blender_executable: Blender executable name or path.
        preview_texture_max_size: Maximum width or height for `textured.png`.

    Returns:
        Immutable source-target records in numeric source order.

    Raises:
        FileExistsError: If any requested target already exists.
        FileNotFoundError: If a source or rendered sidecar is missing.
        RuntimeError: If Blender rendering fails.
        ValueError: If discovery, perturbation, or rendered verification fails.
    """
    source_dataset = Path(source_dataset)
    out_dir = Path(out_dir)
    pairs = discover_object_pairs(
        source_dataset,
        out_dir,
        source_start=source_start,
        count=count,
        id_offset=id_offset,
    )
    rng = random.Random(seed)

    with tempfile.TemporaryDirectory(
        prefix=f".{out_dir.name}-perturbed-staging-",
        dir=out_dir.parent,
    ) as staging_directory:
        staging_dir = Path(staging_directory)
        for pair in pairs:
            staged_generation_config = (
                staging_dir / "generation_configs" / f"{pair.target_id}.json"
            )
            staged_object_config = (
                staging_dir / "configs" / f"{pair.target_id}.object_config.json"
            )
            staged_mesh_dir = staging_dir / "meshes" / pair.target_id
            source_config = _read_json(pair.source_generation_config)
            if source_config["min_coverage"] < MIN_STAMP_COVERAGE:
                raise ValueError("min_coverage must be at least 0.95")
            target_config = perturb_stamp_config(source_config, rng, bounds)
            _write_json(staged_generation_config, target_config)
            _write_json(
                staged_object_config,
                target_object_config(
                    _read_json(pair.source_object_config), pair.target_id
                ),
            )
            source_metadata = _read_json(pair.source_mesh_dir / "textured.json")
            command = blender_stamp_command_args(
                blender_executable,
                stamping_script.resolve(),
                Path(source_metadata["parent_mesh_path"]),
                staged_generation_config,
                staged_mesh_dir / "textured.glb",
            )
            _run_render_command(command)
            output_glb_path = staged_mesh_dir / "textured.glb"
            if not output_glb_path.exists():
                raise RuntimeError(
                    "Blender command completed without writing expected GLB: "
                    f"{output_glb_path}"
                )
            _normalize_texture_sidecar(
                staged_mesh_dir,
                preview_texture_max_size=preview_texture_max_size,
            )
            verify_rendered_pair(
                source_config,
                target_config,
                source_metadata,
                _read_json(staged_mesh_dir / "textured.json"),
                bounds,
            )

        for directory_name in ("generation_configs", "configs", "meshes"):
            (out_dir / directory_name).mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            shutil.move(
                staging_dir / "generation_configs" / f"{pair.target_id}.json",
                pair.target_generation_config,
            )
            shutil.move(
                staging_dir / "configs" / f"{pair.target_id}.object_config.json",
                pair.target_object_config,
            )
            shutil.move(
                staging_dir / "meshes" / pair.target_id,
                pair.target_mesh_dir,
            )

    scene_config_path = out_dir / "compositional_objects.scene_dataset_config.json"
    if not scene_config_path.exists():
        _write_json(scene_config_path, scene_dataset_config())
    return pairs
