from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tbp.compositional_datasets.placements import SurfaceAttachment, TextureStampResult
from tbp.compositional_datasets.uv_stamping import (
    TextureStampError,
    stamp_rgba_texture,
    texture_stamp_uvs,
    uv_stamp_overlaps_existing,
    uv_polygons_overlap_in_stamp,
)

DEFAULT_MIN_COVERAGE = 0.95


@dataclass(frozen=True)
class BaseColorTexture:
    """Writable base-color texture selected for stamping.

    Attributes:
        image: Blender image data-block that receives stamped pixels.
        texture_path: Path where the image is saved.
        texture_size: Texture width and height in pixels.
        material_index: Mesh material slot index associated with the texture.
        material_name: Name of the material associated with the texture.
    """

    image: Any
    texture_path: Path
    texture_size: tuple[int, int]
    material_index: int
    material_name: str


@dataclass(frozen=True)
class TextureStampPlacement:
    """One requested texture stamp in a multi-sticker batch.

    Attributes:
        slot_name: Stable slot name for error messages and metadata.
        polygon_index: Target triangulated polygon index.
        attachment: Surface attachment for this stamp.
        sticker_path: Sticker image to bake into the texture.
    """

    slot_name: str
    polygon_index: int
    attachment: SurfaceAttachment
    sticker_path: Path


def _find_image_texture_node(material: Any) -> Any | None:
    """Find the first image texture node with an image on a material.

    Args:
        material: Blender material, or None.

    Returns:
        Image texture node when present; otherwise None.
    """
    nodes = material.node_tree.nodes
    material_nodes = (
        [*getattr(nodes, "nodes", {}).values(), *nodes.created]
        if hasattr(nodes, "created")
        else list(nodes)
    )
    for node in material_nodes:
        if (
            getattr(node, "type", "") in {"TEX_IMAGE", "ShaderNodeTexImage"}
            and getattr(node, "image", None) is not None
        ):
            return node
    return None


def _save_image(image: Any, path: Path) -> None:
    """Save a Blender image as PNG and pack the saved bytes when supported.

    Args:
        image: Blender image data-block to save.
        path: Destination PNG path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    if hasattr(image, "filepath"):
        image.filepath = str(path)
    if hasattr(image, "pack") and path.exists():
        data = path.read_bytes()
        image.pack(data=data, data_len=len(data))


def _image_to_rgba_array(image: Any) -> np.ndarray:
    """Read Blender float pixels into a uint8 RGBA array.

    Args:
        image: Blender image data-block with float pixels in 0..1 range.

    Returns:
        A uint8 array with shape (height, width, 4).

    Raises:
        TextureStampError: If the image pixel buffer size does not match the
            image dimensions.
    """
    width, height = image.size
    width, height = int(width), int(height)
    pixels = np.asarray(list(image.pixels), dtype=float)
    if pixels.size != width * height * 4:
        raise TextureStampError("Blender image pixel buffer has unexpected size")
    bottom_first = pixels.reshape((height, width, 4))
    top_first = np.flipud(bottom_first)
    rgba = np.round(np.clip(top_first, 0.0, 1.0) * 255.0)
    return rgba.astype(np.uint8)


def _write_rgba_to_image(image: Any, rgba: Any) -> None:
    """Write a uint8 RGBA array into a Blender image.

    Args:
        image: Blender image data-block to update.
        rgba: Array-like pixel data with shape (height, width, 4).

    Raises:
        ValueError: If rgba does not match the Blender image dimensions.
    """
    array = np.asarray(rgba, dtype=np.uint8)
    width, height = image.size
    width, height = int(width), int(height)
    if array.shape != (height, width, 4):
        raise ValueError("rgba array shape must match Blender image size")
    bottom_first = np.flipud(array)
    pixels = (bottom_first.astype(np.float32) / np.float32(255.0)).reshape(-1)
    image.pixels.foreach_set(pixels)
    if hasattr(image, "update"):
        image.update()


def load_base_color_texture(
    obj: Any,
    texture_path: Path | str,
) -> BaseColorTexture:
    """Copy an object's active material base-color texture for stamping.

    Args:
        obj: Blender mesh object whose active material owns the texture.
        texture_path: Destination path for the writable texture PNG.

    Returns:
        Information about the writable image texture and owning material.

    Raises:
        TextureStampError: If the object lacks an active material or usable
            image texture.
    """
    output_path = Path(texture_path)
    material = getattr(obj, "active_material", None)
    if material is None:
        raise TextureStampError("parent mesh must have an active material with a base-color image texture")
    materials = list(getattr(obj.data, "materials", []))
    if material not in materials:
        raise TextureStampError("parent active material must be assigned to a material slot")

    image_node = _find_image_texture_node(material)
    if image_node is None:
        raise TextureStampError("parent material must have a base-color image texture")

    image = image_node.image.copy()
    image_node.image = image
    try:
        _save_image(image, output_path)
    except RuntimeError as error:
        raise TextureStampError("parent base-color image texture has no readable image data") from error
    width, height = image.size
    texture_size = (int(width), int(height))

    return BaseColorTexture(
        image=image,
        texture_path=output_path,
        texture_size=texture_size,
        material_index=materials.index(material),
        material_name=material.name,
    )


def triangulate_mesh_object(obj: Any) -> None:
    """Triangulate a Blender mesh object in place using bmesh.

    Args:
        obj: Blender mesh object to triangulate.
    """
    import bmesh  # type: ignore[import-not-found]

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _polygon_triangle_and_uvs(
    obj: Any,
    polygon_index: int,
    uv_layer_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract world-space triangle vertices and UVs for one polygon.

    Args:
        obj: Blender mesh object containing the polygon.
        polygon_index: Index of the target polygon.
        uv_layer_name: Name of the UV layer to read.

    Returns:
        A pair of arrays: triangle vertices with shape (3, 3) and UVs with
        shape (3, 2).

    Raises:
        TextureStampError: If the target polygon is not triangulated.
    """
    mesh = obj.data
    polygon = mesh.polygons[polygon_index]
    if polygon.loop_total != 3:
        raise TextureStampError("texture stamp target polygon must be triangulated")

    matrix_world = getattr(obj, "matrix_world", None)
    vertices = []
    uvs = []
    uv_layer = (
        mesh.uv_layers[uv_layer_name].data
        if isinstance(uv_layer_name, str)
        else mesh.uv_layers.active.data
    )
    for loop_index in polygon.loop_indices:
        loop = mesh.loops[loop_index]
        vertex = mesh.vertices[loop.vertex_index].co
        world_vertex = matrix_world @ vertex if matrix_world is not None else vertex
        vertices.append([float(world_vertex[0]), float(world_vertex[1]), float(world_vertex[2])])
        uv = uv_layer[loop_index].uv
        uvs.append([float(uv[0]), float(uv[1])])
    return np.array(vertices, dtype=float), np.array(uvs, dtype=float)


def _material_uv_polygons(obj: Any, uv_layer_name: str, material_index: int) -> list[np.ndarray]:
    """Collect UV polygons using a material slot.

    Args:
        obj: Blender mesh object to inspect.
        uv_layer_name: Name of the UV layer to read.
        material_index: Material slot index to match.

    Returns:
        UV polygons for mesh faces assigned to material_index.
    """
    mesh = obj.data
    uv_layer = mesh.uv_layers[uv_layer_name].data
    uv_polygons: list[np.ndarray] = []
    for polygon in mesh.polygons:
        if int(getattr(polygon, "material_index", 0)) != material_index:
            continue
        polygon_uvs = []
        for loop_index in polygon.loop_indices:
            uv = uv_layer[loop_index].uv
            polygon_uvs.append([float(uv[0]), float(uv[1])])
        uv_polygons.append(np.array(polygon_uvs, dtype=float))
    return uv_polygons


def stamp_texture_placements_on_object(
    bpy_module: Any,
    obj: Any,
    *,
    placements: list[TextureStampPlacement] | tuple[TextureStampPlacement, ...],
    texture_path: Path | str,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> list[TextureStampResult]:
    """Bake multiple stickers into one Blender object's base-color texture.

    Args:
        bpy_module: Blender bpy module, or a test double.
        obj: Blender mesh object receiving the stickers.
        placements: Stamp placements to bake in order.
        texture_path: Destination path for the stamped base-color texture.
        min_coverage: Minimum required fraction of each sticker footprint
            inside the 0..1 UV square.

    Returns:
        Texture stamp metadata in placement order.

    Raises:
        ValueError: If image arrays, UVs, or attachment values are invalid.
        TextureStampError: If any target polygon, UV footprint, or coverage is
            unsafe for stamping.
    """
    uv_layer_name = obj.data.uv_layers.active.name
    texture_info = load_base_color_texture(
        obj,
        texture_path,
    )
    material_uv_polygons = _material_uv_polygons(obj, uv_layer_name, texture_info.material_index)
    base = _image_to_rgba_array(texture_info.image)
    results: list[TextureStampResult] = []
    stamped_uv_corners: list[np.ndarray] = []

    for placement in placements:
        triangle, triangle_uvs = _polygon_triangle_and_uvs(obj, placement.polygon_index, uv_layer_name)
        stamp = texture_stamp_uvs(
            triangle,
            triangle_uvs,
            barycentric=placement.attachment.barycentric,
            rotation_about_normal_deg=placement.attachment.rotation_about_normal_deg,
            size=placement.attachment.size,
        )
        if uv_polygons_overlap_in_stamp(material_uv_polygons, stamp.uv_corners):
            raise TextureStampError(
                f"texture stamp UV footprint overlaps another UV island for slot {placement.slot_name}"
            )
        if uv_stamp_overlaps_existing(stamped_uv_corners, stamp.uv_corners):
            raise TextureStampError(f"texture stamp UV footprint overlaps another sticker for slot {placement.slot_name}")

        sticker_image = bpy_module.data.images.load(str(placement.sticker_path), check_existing=False)
        try:
            sticker_rgba = _image_to_rgba_array(sticker_image)
        finally:
            try:
                bpy_module.data.images.remove(sticker_image)
            except Exception:
                pass
        stamped = stamp_rgba_texture(
            base,
            sticker_rgba,
            stamp.uv_corners,
            min_coverage=min_coverage,
        )
        base = stamped.image
        stamped_uv_corners.append(stamp.uv_corners)
        results.append(
            TextureStampResult(
                texture_path=texture_info.texture_path,
                uv_center=stamp.uv_center,
                uv_corners=stamp.uv_corners,
                texture_size=texture_info.texture_size,
                target_object_name=obj.name,
                target_material_index=texture_info.material_index,
                target_material_name=texture_info.material_name,
                target_uv_layer=uv_layer_name,
                coverage_ratio=stamped.coverage_ratio,
            )
        )

    _write_rgba_to_image(texture_info.image, base)
    _save_image(texture_info.image, texture_info.texture_path)
    return results
