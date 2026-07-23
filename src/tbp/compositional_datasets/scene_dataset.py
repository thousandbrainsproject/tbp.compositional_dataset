from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from tbp.compositional_datasets.generation_configs import (
    generate_stamp_job,
    prepare_stamp_generation_run,
    shell_command,
)


def scene_dataset_config() -> dict[str, Any]:
    """Return the Habitat scene dataset config used by v1.1.

    Returns:
        JSON-ready scene dataset configuration.
    """
    return {"objects": {"paths": {".json": ["configs/"]}}}


def object_config(object_id: str) -> dict[str, Any]:
    """Return a Habitat object config for a generated object.

    Args:
        object_id: Dataset object identifier matching a mesh directory.

    Returns:
        JSON-ready object config.
    """
    return {
        "friction_coefficient": 3.0,
        "render_asset": f"../meshes/{object_id}/textured.glb",
        "requires_lighting": True,
        "up": [0.0, 1.0, 0.0],
        "front": [0.0, 1.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def object_id_for_parent(index: int, parent_path: Path) -> str:
    """Return a stable object id for a generated parent object.

    Args:
        index: Zero-based generated object index.
        parent_path: Parent GLB path.

    Returns:
        Object id using one-based numbering and the parent stem.
    """
    return f"{index + 1:03d}_{parent_path.stem}_6x2d_stickers"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting.

    Args:
        path: Destination path.
        data: JSON-ready object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _run_render_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and fail on non-zero exit.

    Args:
        command: Command argument list.

    Returns:
        Completed process.

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            _render_failure_message(
                prefix=f"Blender command failed with exit status {error.returncode}.",
                command=command,
                stdout=error.stdout or "",
                stderr=error.stderr or "",
            )
        ) from error


def _render_failure_message(
    *,
    prefix: str,
    command: list[str],
    stdout: str,
    stderr: str,
    output_glb_path: Path | None = None,
) -> str:
    """Build a diagnostic render failure message.

    Args:
        prefix: First line describing the render failure.
        command: Blender command argument list.
        stdout: Captured Blender stdout.
        stderr: Captured Blender stderr.
        output_glb_path: Optional expected rendered GLB path.

    Returns:
        Human-readable failure message with command and captured output.
    """
    output_line = f"Expected output: {output_glb_path}\n" if output_glb_path is not None else ""
    return (
        f"{prefix}\n"
        f"{output_line}"
        f"Command: {shell_command(command)}\n"
        f"stdout:\n{stdout or '<empty>'}\n"
        f"stderr:\n{stderr or '<empty>'}"
    )


def _replace_string_values(value: Any, old: str, new: str) -> Any:
    """Recursively replace string values inside JSON-like data.

    Args:
        value: JSON-like value.
        old: String value to replace.
        new: Replacement string value.

    Returns:
        Updated JSON-like value.
    """
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, list):
        return [_replace_string_values(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_string_values(item, old, new) for key, item in value.items()}
    return value


def _resize_png_preview(path: Path, max_size: int) -> None:
    """Resize a PNG in place for lightweight visual inspection.

    Args:
        path: PNG path to resize.
        max_size: Maximum width or height in pixels.
    """
    with Image.open(path) as image:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image.save(path, optimize=True)


def _normalize_texture_sidecar(mesh_dir: Path, *, preview_texture_max_size: int) -> Path:
    """Rename and downsize the stamped texture PNG preview.

    Args:
        mesh_dir: Directory containing `textured.glb` render outputs.
        preview_texture_max_size: Maximum width or height for `textured.png`.

    Returns:
        Final `textured.png` path.

    Raises:
        FileNotFoundError: If the expected texture or metadata sidecar is missing.
    """
    old_texture_path = mesh_dir / "textured_stamped_texture.png"
    new_texture_path = mesh_dir / "textured.png"
    metadata_path = mesh_dir / "textured.json"
    if not old_texture_path.exists():
        raise FileNotFoundError(old_texture_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    old_texture_path.replace(new_texture_path)
    _resize_png_preview(new_texture_path, preview_texture_max_size)
    metadata = json.loads(metadata_path.read_text())
    metadata = _replace_string_values(metadata, str(old_texture_path), str(new_texture_path))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return new_texture_path


def generate_scene_dataset(
    out_dir: Path,
    count: int,
    seed: int,
    parents: list[Path],
    stickers: list[Path],
    base_config_path: Path,
    stamping_script: Path,
    blender_executable: str = "blender",
    preview_texture_max_size: int = 512,
) -> Path:
    """Generate a rendered 1.1-format compositional object dataset.

    Args:
        out_dir: Output dataset directory.
        count: Number of rendered objects to generate.
        seed: Random seed used for sticker and rotation sampling.
        parents: Parent GLB paths to balance across generated objects.
        stickers: Sticker PNG paths to sample and assign to fixed slots.
        base_config_path: Base six-slot layout config path.
        stamping_script: Blender script used to render each object.
        blender_executable: Blender executable name or path.
        preview_texture_max_size: Maximum width or height for `textured.png`.

    Returns:
        Path to the Habitat scene dataset config.

    Raises:
        ValueError: If count or input assets are invalid.
        subprocess.CalledProcessError: If Blender rendering fails.
        FileNotFoundError: If expected render outputs are missing.
    """
    if preview_texture_max_size <= 0:
        raise ValueError("preview_texture_max_size must be positive")

    run = prepare_stamp_generation_run(
        out_dir,
        count,
        seed,
        parents,
        stickers,
        base_config_path,
    )
    scene_config_path = run.out_dir / "compositional_objects.scene_dataset_config.json"
    _write_json(scene_config_path, scene_dataset_config())

    for index in range(run.count):
        parent = run.parents[index % len(run.parents)]
        object_id = object_id_for_parent(index, parent)
        mesh_dir = run.out_dir / "meshes" / object_id
        stamp_config_path = run.out_dir / "generation_configs" / f"{object_id}.json"
        object_config_path = run.out_dir / "configs" / f"{object_id}.object_config.json"
        output_glb_path = mesh_dir / "textured.glb"

        job = generate_stamp_job(
            run,
            index=index,
            object_id=object_id,
            config_path=stamp_config_path,
            output_glb_path=output_glb_path,
            stamping_script=stamping_script,
            blender_executable=blender_executable,
        )
        render_result = _run_render_command(job.blender_command_args)
        if not job.output_glb_path.exists():
            raise RuntimeError(
                _render_failure_message(
                    prefix="Blender command completed without writing expected GLB.",
                    command=job.blender_command_args,
                    output_glb_path=job.output_glb_path,
                    stdout=render_result.stdout,
                    stderr=render_result.stderr,
                )
            )
        _normalize_texture_sidecar(
            mesh_dir,
            preview_texture_max_size=preview_texture_max_size,
        )
        _write_json(object_config_path, object_config(object_id))

    return scene_config_path
