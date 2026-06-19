from __future__ import annotations

import json
from pathlib import Path

from tbp.compositional_datasets.generation_configs import (
    generate_stamp_job,
    prepare_stamp_generation_run,
    shell_command,
)


def generate_batch_configs(
    out_dir: Path,
    count: int,
    seed: int,
    parents: list[Path],
    stickers: list[Path],
    base_config_path: Path,
    script_path: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Generate deterministic per-object configs and a manifest.

    Args:
        out_dir: Output dataset directory.
        count: Number of object configs to generate.
        seed: Random seed used for sticker and rotation sampling.
        parents: Parent GLB paths to balance across generated configs.
        stickers: Sticker PNG paths to sample and assign to fixed slots.
        base_config_path: Six-sticker layout config to copy fixed
            placement fields from.
        script_path: Configured stamping script path used in manifest
            commands. Defaults to `scripts/stamp_object_from_config.py` under
            project_root.
        project_root: Repository or project root for resolving relative
            paths. Defaults to the current working directory.

    Returns:
        Path to the written manifest JSON.

    Raises:
        ValueError: If required inputs are empty or invalid.
    """
    root = Path.cwd().resolve() if project_root is None else project_root.resolve()
    stamp_script_path = (
        root / "scripts" / "stamp_object_from_config.py"
        if script_path is None
        else script_path
    )
    stamp_script_path = stamp_script_path.resolve()

    run = prepare_stamp_generation_run(
        out_dir,
        count,
        seed,
        parents,
        stickers,
        base_config_path,
    )
    objects = []
    for index in range(run.count):
        object_id = f"object_{index:03d}"
        object_dir = run.out_dir / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        config_path = object_dir / "config.json"
        output_glb_path = object_dir / f"{object_id}.glb"
        job = generate_stamp_job(
            run,
            index=index,
            object_id=object_id,
            config_path=config_path,
            output_glb_path=output_glb_path,
            stamping_script=stamp_script_path,
        )

        objects.append(
            {
                "object_id": job.object_id,
                "parent_glb_path": str(job.parent_path),
                "config_path": str(job.config_path),
                "output_glb_path": str(job.output_glb_path),
                "seed": int(seed),
                "slots": job.manifest_slots,
                "blender_command": shell_command(job.blender_command_args),
            }
        )

    manifest = {
        "dataset_name": run.out_dir.name,
        "out_dir": str(run.out_dir),
        "count": int(run.count),
        "seed": int(seed),
        "base_config_path": str(run.base_config_path),
        "objects": objects,
    }
    manifest_path = run.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
