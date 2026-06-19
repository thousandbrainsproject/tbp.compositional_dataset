from __future__ import annotations

from typing import Any

import numpy as np

from tbp.compositional_datasets.arrays import to_blender_vector
from tbp.compositional_datasets.geometry import is_degenerate_triangle


def clear_scene(bpy_module: Any) -> None:
    """Delete all objects from the current Blender scene."""
    bpy_module.ops.object.select_all(action="SELECT")
    bpy_module.ops.object.delete()


def object_bounds(obj: Any):
    """Return the world-space center and radius of an object bounding box.

    Args:
        obj: Blender object with `bound_box` and `matrix_world`.

    Returns:
        Tuple containing center coordinates and bounding radius.
    """
    corners = np.array(
        [np.asarray(obj.matrix_world @ to_blender_vector(corner), dtype=float) for corner in obj.bound_box],
        dtype=float,
    )
    center_array = np.mean(corners, axis=0)
    radius = float(np.max(np.linalg.norm(corners - center_array, axis=1)))
    center = tuple(float(coordinate) for coordinate in center_array)
    return center, max(radius, 0.5)


def mesh_triangle(obj: Any, polygon_index: int) -> np.ndarray:
    """Return one mesh polygon as world-space triangle vertices.

    Args:
        obj: Blender mesh object.
        polygon_index: Index of a triangulated polygon.

    Returns:
        Triangle vertices as a floating-point array with shape (3, 3).

    Raises:
        RuntimeError: If the polygon is not triangulated.
    """
    mesh = obj.data
    polygon = mesh.polygons[polygon_index]
    if polygon.loop_total != 3:
        raise RuntimeError("Picker target mesh must be triangulated")

    vertices = []
    for loop_index in polygon.loop_indices:
        vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        vertices.append(np.asarray(obj.matrix_world @ vertex, dtype=float))
    return np.asarray(vertices, dtype=float)


def count_degenerate_triangles(obj: Any) -> int:
    """Count degenerate triangulated polygons on a mesh object.

    Args:
        obj: Blender mesh object.

    Returns:
        Number of triangulated mesh polygons with near-zero area.
    """
    count = 0
    for polygon_index, polygon in enumerate(obj.data.polygons):
        if polygon.loop_total != 3:
            continue
        if is_degenerate_triangle(mesh_triangle(obj, polygon_index)):
            count += 1
    return count
