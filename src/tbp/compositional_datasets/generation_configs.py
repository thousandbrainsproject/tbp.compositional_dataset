from __future__ import annotations

import json
import random
import shutil
import shlex
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tbp.compositional_datasets.sticker_layout import load_sticker_layout_config

ROTATION_CHOICES = [float(angle) for angle in range(0, 360, 15)]
MAX_UNIQUE_ASSIGNMENT_ATTEMPTS = 10_000


@dataclass
class StampGenerationRun:
    """Resolved state shared by stamp config generation workflows.

    Args:
        out_dir: Resolved output directory.
        count: Number of objects to generate.
        seed: Random seed used for deterministic generation.
        parents: Resolved parent GLB paths.
        stickers: Resolved sticker PNG paths.
        base_config_path: Resolved base sticker layout config path.
        base_config: Loaded base sticker layout config data.
        rng: Random number generator used for sticker and rotation sampling.
        used_parent_visible_layouts: Dataset-level set used to reject repeated
            visible sticker assignments for the same parent.
    """

    out_dir: Path
    count: int
    seed: int
    parents: list[Path]
    stickers: list[Path]
    base_config_path: Path
    base_config: dict[str, Any]
    rng: random.Random
    used_parent_visible_layouts: set[Any] = field(default_factory=set)


@dataclass(frozen=True)
class StampGenerationJob:
    """One generated stamp config and its matching render command.

    Args:
        object_id: Generated object identifier chosen by the caller.
        parent_path: Resolved parent GLB path for this job.
        config_path: Path to the randomized stamp config written for this job.
        output_glb_path: Path where Blender should write the stamped GLB.
        manifest_slots: Manifest slot records returned by the config writer.
        blender_command_args: Blender command arguments for configured stamping.
    """

    object_id: str
    parent_path: Path
    config_path: Path
    output_glb_path: Path
    manifest_slots: list[dict[str, Any]]
    blender_command_args: list[str]


def prepare_stamp_generation_run(
    out_dir: Path,
    count: int,
    seed: int,
    parents: list[Path],
    stickers: list[Path],
    base_config_path: Path,
) -> StampGenerationRun:
    """Prepare resolved state for deterministic stamp config generation.

    Args:
        out_dir: Output dataset directory.
        count: Number of object configs to generate.
        seed: Random seed used for sticker and rotation sampling.
        parents: Parent GLB paths to balance across generated configs.
        stickers: Sticker PNG paths to sample and assign to fixed slots.
        base_config_path: Base six-slot layout config path.

    Returns:
        Resolved run state for generating one or more stamp jobs.

    Raises:
        ValueError: If count or required inputs are invalid.
        FileNotFoundError: If any required path does not exist.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    resolved_out_dir = Path(out_dir).expanduser().resolve()
    if resolved_out_dir.exists():
        shutil.rmtree(resolved_out_dir)

    resolved_parents = resolve_existing_paths(parents, "parent GLB")
    resolved_stickers = resolve_existing_paths(stickers, "sticker PNG")
    resolved_base_config_path = Path(base_config_path).expanduser().resolve()
    base_config = load_base_config(resolved_base_config_path)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    return StampGenerationRun(
        out_dir=resolved_out_dir,
        count=int(count),
        seed=int(seed),
        parents=resolved_parents,
        stickers=resolved_stickers,
        base_config_path=resolved_base_config_path,
        base_config=base_config,
        rng=random.Random(seed),
    )


def resolve_existing_paths(
    paths: list[Path],
    description: str,
) -> list[Path]:
    """Resolve and validate input asset paths.

    Args:
        paths: Paths to validate.
        description: Human-readable asset type for error messages.

    Returns:
        Sorted absolute paths.

    Raises:
        ValueError: If no paths are provided.
        FileNotFoundError: If any path does not exist.
    """
    if not paths:
        raise ValueError(f"at least one {description} is required")
    resolved = []
    for path in paths:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        resolved.append(candidate.resolve())
    return sorted(resolved, key=str)


def load_base_config(path: Path) -> dict[str, Any]:
    """Load and validate a six-sticker layout base config.

    Args:
        path: Base sticker layout config path.

    Returns:
        JSON object from the base config.

    Raises:
        ValueError: If the config is not a JSON object.
    """
    load_sticker_layout_config(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("base config must be a JSON object")
    return data


def _randomized_stamp_config_data(
    base_config: dict[str, Any],
    stickers: list[Path],
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return randomized config data and matching manifest slots.

    Args:
        base_config: Base six-slot layout config.
        stickers: Candidate sticker PNG paths.
        rng: Random number generator used to choose stickers and rotations.

    Returns:
        Randomized config and manifest slot records.
    """
    config = deepcopy(base_config)
    base_slots = list(config["slots"])
    sampled_stickers = [rng.choice(stickers) for _ in base_slots]
    rng.shuffle(sampled_stickers)

    slots = []
    manifest_slots = []
    for base_slot, sticker in zip(base_slots, sampled_stickers, strict=True):
        rotation_deg = rng.choice(ROTATION_CHOICES)
        slot = deepcopy(base_slot)
        slot["sticker"] = str(sticker)
        slot["rotation_deg"] = rotation_deg
        slots.append(slot)
        manifest_slots.append(
            {
                "slot_name": str(slot["name"]),
                "sticker_path": str(sticker),
                "rotation_deg": rotation_deg,
            }
        )
    config["slots"] = slots
    return config, manifest_slots


def write_randomized_stamp_config(
    path: Path,
    base_config: dict[str, Any],
    stickers: list[Path],
    rng: random.Random,
    used_parent_visible_layouts: set | None = None,
    parent_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Write one randomized sticker layout config and return manifest slots.

    Args:
        path: Destination config path.
        base_config: Base six-slot layout config.
        stickers: Candidate sticker PNG paths.
        rng: Random number generator used to choose stickers and rotations.
        used_parent_visible_layouts: Optional dataset-level set used to reject
            repeated visible assignments.
        parent_path: Parent GLB path required when used_parent_visible_layouts
            is used.

    Returns:
        Manifest slot records matching the written config slots.

    Raises:
        ValueError: If a unique assignment cannot be sampled.
    """
    if used_parent_visible_layouts is not None and parent_path is None:
        raise ValueError("parent_path is required when used_parent_visible_layouts is used")
    resolved_parent_path = parent_path

    for _attempt in range(MAX_UNIQUE_ASSIGNMENT_ATTEMPTS):
        config, manifest_slots = _randomized_stamp_config_data(
            base_config,
            stickers,
            rng,
        )
        if used_parent_visible_layouts is None:
            break
        assert resolved_parent_path is not None
        visible_layout = (
            str(resolved_parent_path),
            tuple(
                (str(slot["slot_name"]), str(slot["sticker_path"]))
                for slot in manifest_slots
            ),
        )
        if visible_layout not in used_parent_visible_layouts:
            used_parent_visible_layouts.add(visible_layout)
            break
    else:
        raise ValueError(
            "not enough distinct sticker-location assignments for the available "
            "stickers, rotations, parents, and count"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return manifest_slots


def generate_stamp_job(
    run: StampGenerationRun,
    *,
    index: int,
    object_id: str,
    config_path: Path,
    output_glb_path: Path,
    stamping_script: Path,
    blender_executable: str = "blender",
) -> StampGenerationJob:
    """Write one randomized stamp config and return its render job.

    Args:
        run: Prepared stamp generation run state.
        index: Zero-based generated object index.
        object_id: Generated object identifier chosen by the caller.
        config_path: Destination path for the randomized stamp config.
        output_glb_path: Rendered GLB output path.
        stamping_script: Blender script used to render one stamped object.
        blender_executable: Blender executable name or path.

    Returns:
        Generated job data containing paths, manifest slots, and command args.

    Raises:
        ValueError: If a unique visible assignment cannot be sampled.
    """
    parent = run.parents[index % len(run.parents)]
    manifest_slots = write_randomized_stamp_config(
        config_path,
        run.base_config,
        run.stickers,
        run.rng,
        run.used_parent_visible_layouts,
        parent,
    )
    command_args = blender_stamp_command_args(
        blender_executable,
        Path(stamping_script).resolve(),
        parent,
        config_path,
        output_glb_path,
    )
    return StampGenerationJob(
        object_id=object_id,
        parent_path=parent,
        config_path=config_path,
        output_glb_path=output_glb_path,
        manifest_slots=manifest_slots,
        blender_command_args=command_args,
    )


def blender_stamp_command_args(
    blender_executable: str,
    stamping_script: Path,
    parent_path: Path,
    config_path: Path,
    output_glb_path: Path,
) -> list[str]:
    """Build Blender configured-stamping command arguments.

    Args:
        blender_executable: Blender command name or path.
        stamping_script: Texture stamp script path.
        parent_path: Parent GLB path.
        config_path: Generated stamp config path.
        output_glb_path: Rendered GLB output path.

    Returns:
        Command argument list.
    """
    return [
        blender_executable,
        "--background",
        "--python",
        str(stamping_script),
        "--",
        "--object",
        str(parent_path),
        "--config",
        str(config_path),
        "--out",
        str(output_glb_path),
    ]


def shell_command(command: list[str] | tuple[str, ...]) -> str:
    """Return a shell-ready command string from command arguments.

    Args:
        command: Command argument list.

    Returns:
        Shell-quoted command string.
    """
    return " ".join(shlex.quote(part) for part in command)
