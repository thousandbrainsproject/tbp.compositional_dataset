"""Headless Blender script for configured six-sticker texture stamping."""

from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    site.addsitedir(USER_SITE)

from tbp.compositional_datasets.arrays import to_blender_vector
from tbp.compositional_datasets.blender_cli import argv_after_blender_separator
from tbp.compositional_datasets.blender_scene import (
    clear_scene,
    count_degenerate_triangles,
    mesh_triangle,
    object_bounds,
)
from tbp.compositional_datasets.sticker_layout import (
    compute_common_slot_sizes,
    load_sticker_layout_config,
    placement_ray,
    rotation_aligning_bitangent_to_world_up,
    validate_same_side_footprints_do_not_overlap,
)
from tbp.compositional_datasets.blender_texture_baking import (
    TextureStampError,
    TextureStampPlacement,
    stamp_texture_placements_on_object,
    triangulate_mesh_object,
)
from tbp.compositional_datasets.geometry import (
    is_degenerate_triangle,
    sticker_quad_corners,
    surface_frame,
)
from tbp.compositional_datasets.metadata import multi_generation_metadata, write_metadata
from tbp.compositional_datasets.placements import SurfaceAttachment
from tbp.compositional_datasets.uv_stamping import barycentric_coordinates_3d

import bpy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse configured-stamping command-line arguments.

    Args:
        argv: Optional argument list. Uses sys.argv when None.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Bake six configured PNG stickers into a parent mesh texture."
    )
    parser.add_argument("--object", required=True, type=Path, help="Parent .glb asset path.")
    parser.add_argument("--config", required=True, type=Path, help="Sticker layout JSON config path.")
    parser.add_argument("--out", required=True, type=Path, help="Output GLB path.")
    return parser.parse_args(argv)


def _prepare_scene(bpy_module: Any, object_path: Path) -> Any:
    """Import and prepare a GLB parent asset for configured stamping."""
    if object_path.suffix.lower() != ".glb":
        raise ValueError("Configured texture stamping requires a .glb parent asset")

    clear_scene(bpy_module)
    bpy_module.ops.import_scene.gltf(filepath=str(object_path))
    obj = [obj for obj in bpy_module.context.scene.objects if obj.type == "MESH"][0]
    triangulate_mesh_object(obj)
    degenerate_count = count_degenerate_triangles(obj)
    if degenerate_count:
        print(
            f"Imported mesh contains {degenerate_count} degenerate triangle(s); "
            "those faces cannot receive stamps"
        )
    bpy_module.context.view_layer.objects.active = obj
    obj.select_set(True)
    if hasattr(bpy_module.context.view_layer, "update"):
        bpy_module.context.view_layer.update()
    return obj


def _resolve_slot_hit(
    bpy_module: Any,
    obj: Any,
    slot: Any,
    bounds_center: tuple[float, float, float],
    bounds_radius: float,
    profile: Any,
) -> tuple[Any, int, Any]:
    """Ray-cast one configured slot to a mesh triangle.

    Args:
        bpy_module: Blender bpy module.
        obj: Mesh object to hit.
        slot: Configured sticker slot.
        bounds_center: Object bounds center.
        bounds_radius: Object bounds radius.
        profile: Placement axis profile.

    Returns:
        Target object, face index, and barycentric hit coordinates.

    Raises:
        ValueError: If the ray misses or hits an invalid face.
    """
    origin, direction = placement_ray(slot, bounds_center, bounds_radius, profile)
    depsgraph = bpy_module.context.evaluated_depsgraph_get()
    hit, location, _normal, face_index, hit_obj, _matrix = bpy_module.context.scene.ray_cast(
        depsgraph,
        to_blender_vector(origin),
        to_blender_vector(direction),
    )
    if not hit or hit_obj is None or hit_obj.type != "MESH" or face_index < 0:
        raise ValueError(f"Configured slot {slot.name} did not hit a mesh face")
    if hit_obj.name != obj.name:
        raise ValueError(f"Configured slot {slot.name} hit unexpected object {hit_obj.name}")

    triangle = mesh_triangle(hit_obj, int(face_index))
    if is_degenerate_triangle(triangle):
        raise ValueError("Selected face is degenerate; pick another surface face")
    barycentric = barycentric_coordinates_3d(
        triangle,
        [float(location[0]), float(location[1]), float(location[2])],
    )
    return hit_obj, int(face_index), barycentric


def run_configured_stamping(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run configured texture stamping inside Blender."""
    config = load_sticker_layout_config(args.config)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path = output_path.with_name(f"{output_path.stem}_stamped_texture.png")

    obj = _prepare_scene(bpy, args.object)
    bounds_center, bounds_radius = object_bounds(obj)
    sizes = compute_common_slot_sizes(
        config.slots,
        gap=config.gap,
        max_longest_side=config.max_longest_side,
        min_longest_side=config.min_longest_side,
    )

    stamp_requests: list[TextureStampPlacement] = []
    metadata_records: list[dict[str, Any]] = []
    footprints = []
    for slot in config.slots:
        target_obj, face_index, barycentric = _resolve_slot_hit(
            bpy,
            obj,
            slot,
            bounds_center,
            bounds_radius,
            config.placement,
        )
        triangle = mesh_triangle(target_obj, face_index)
        world_up = [0.0, 0.0, 0.0]
        world_up[config.placement.up_axis] = 1.0
        rotation_deg = rotation_aligning_bitangent_to_world_up(
            triangle,
            world_up=world_up,
        ) + float(slot.rotation_deg)
        attachment = SurfaceAttachment(
            triangle_id=face_index,
            barycentric=barycentric,
            rotation_about_normal_deg=rotation_deg,
            size=sizes[slot.name],
        )
        frame = surface_frame(triangle, attachment.barycentric, attachment.rotation_about_normal_deg)
        quad_corners = sticker_quad_corners(
            frame.anchor,
            frame.rotated_tangent,
            frame.rotated_bitangent,
            attachment.size,
        )
        footprints.append((slot.name, slot.side, quad_corners))
        stamp_requests.append(
            TextureStampPlacement(
                slot_name=slot.name,
                polygon_index=face_index,
                attachment=attachment,
                sticker_path=slot.sticker,
            )
        )
        metadata_records.append(
            {
                "slot_name": slot.name,
                "side": slot.side,
                "sticker_path": slot.sticker,
                "attachment": attachment,
                "frame": frame,
                "quad_corners": quad_corners,
            }
        )

    validate_same_side_footprints_do_not_overlap(footprints, config.placement)
    texture_stamps = stamp_texture_placements_on_object(
        bpy,
        obj,
        placements=stamp_requests,
        texture_path=texture_path,
        min_coverage=config.min_coverage,
    )
    for record, texture_stamp in zip(metadata_records, texture_stamps, strict=True):
        record["texture_stamp"] = texture_stamp

    bpy.ops.export_scene.gltf(filepath=str(output_path), export_format="GLB")
    metadata = multi_generation_metadata(parent_mesh_path=args.object, placements=metadata_records)
    metadata_path = write_metadata(metadata, output_path)
    return output_path, metadata_path


def main() -> None:
    """Run the configured-stamping script."""
    args = parse_args(argv_after_blender_separator())
    try:
        output_path, metadata_path = run_configured_stamping(args)
    except (TextureStampError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Wrote {output_path} and {metadata_path}")


if __name__ == "__main__":
    main()
