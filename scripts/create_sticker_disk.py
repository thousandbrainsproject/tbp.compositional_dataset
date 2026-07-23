"""Create one circular sticker disk GLB from a PNG image."""

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

from tbp.compositional_datasets.blender_cli import argv_after_blender_separator
from tbp.compositional_datasets.blender_scene import clear_scene
from tbp.compositional_datasets.sticker_disk import (
    DEFAULT_DIAMETER,
    DEFAULT_SEGMENTS,
    DEFAULT_THICKNESS,
    StickerDiskMesh,
    build_sticker_disk_mesh,
)

import bpy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse sticker-disk generator arguments.

    Args:
        argv: Optional argument list. Uses `sys.argv` when None.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Create a circular sticker disk GLB from one PNG."
    )
    parser.add_argument("--image", required=True, type=Path, help="Source sticker PNG path.")
    parser.add_argument("--out", required=True, type=Path, help="New output GLB path.")
    parser.add_argument("--diameter", type=float, default=DEFAULT_DIAMETER)
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    return parser.parse_args(argv)


def validate_asset_paths(image_path: Path, output_path: Path) -> None:
    """Validate source and destination paths before changing Blender state.

    Args:
        image_path: Existing sticker PNG.
        output_path: New GLB destination.

    Raises:
        FileNotFoundError: If the source image does not exist.
        ValueError: If either suffix is unsupported.
        FileExistsError: If the destination already exists.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"sticker image does not exist: {image_path}")
    if image_path.suffix.lower() != ".png":
        raise ValueError("sticker image must be a .png")
    if output_path.suffix.lower() != ".glb":
        raise ValueError("output must be a .glb")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")


def _create_cap_material(bpy_module: Any, image_path: Path) -> Any:
    """Create the image-textured material used by both disk caps.

    Args:
        bpy_module: Imported Blender Python module.
        image_path: Absolute source sticker PNG path.

    Returns:
        Blender material with the image connected to base color.
    """
    material = bpy_module.data.materials.new(name="StickerCaps")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    texture = nodes.new(type="ShaderNodeTexImage")
    texture.image = bpy_module.data.images.load(str(image_path), check_existing=False)
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    return material


def _create_side_material(bpy_module: Any) -> Any:
    """Create the opaque white material used by the disk side wall.

    Args:
        bpy_module: Imported Blender Python module.

    Returns:
        Blender material with an opaque white base color.
    """
    material = bpy_module.data.materials.new(name="StickerEdge")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    return material


def _create_mesh_object(
    bpy_module: Any,
    mesh_data: StickerDiskMesh,
    image_path: Path,
) -> Any:
    """Create one Blender object from validated pure mesh data.

    Args:
        bpy_module: Imported Blender Python module.
        mesh_data: Deterministic vertices, faces, UVs, and material indices.
        image_path: Absolute source sticker PNG path.

    Returns:
        Linked Blender mesh object with cap and side materials.
    """
    mesh = bpy_module.data.meshes.new("StickerDiskMesh")
    mesh.from_pydata(mesh_data.vertices, [], mesh_data.faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, face_uvs in zip(mesh.polygons, mesh_data.face_uvs, strict=True):
        polygon.material_index = mesh_data.material_indices[polygon.index]
        for loop_index, uv in zip(polygon.loop_indices, face_uvs, strict=True):
            uv_layer.data[loop_index].uv = uv

    obj = bpy_module.data.objects.new("StickerDisk", mesh)
    bpy_module.context.collection.objects.link(obj)
    obj.data.materials.append(_create_cap_material(bpy_module, image_path))
    obj.data.materials.append(_create_side_material(bpy_module))
    return obj


def create_sticker_disk(args: argparse.Namespace, bpy_module: Any) -> Path:
    """Create and export one circular sticker disk asset.

    Args:
        args: Parsed image, output, and geometry arguments.
        bpy_module: Imported Blender Python module.

    Returns:
        Absolute path to the exported GLB.

    Raises:
        FileNotFoundError: If the source image does not exist.
        FileExistsError: If the output already exists.
        ValueError: If paths or geometry values are invalid.
    """
    validate_asset_paths(args.image, args.out)
    mesh_data = build_sticker_disk_mesh(
        diameter=args.diameter,
        thickness=args.thickness,
        segments=args.segments,
    )
    clear_scene(bpy_module)
    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj = _create_mesh_object(bpy_module, mesh_data, args.image.resolve())
    obj.select_set(True)
    bpy_module.context.view_layer.objects.active = obj
    bpy_module.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
    )
    return output_path


def main() -> None:
    """Run the Blender sticker-disk generator."""
    args = parse_args(argv_after_blender_separator())
    try:
        output_path = create_sticker_disk(args, bpy)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
