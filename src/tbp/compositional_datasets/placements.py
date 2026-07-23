from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SurfaceAttachment:
    """Sticker placement request on a single mesh triangle.

    Attributes:
        triangle_id: Index of the target triangle on the parent mesh.
        barycentric: Three barycentric coordinates for the sticker center.
        rotation_about_normal_deg: Rotation angle around the triangle normal in
            degrees.
        size: Sticker width and height in world-space units.
    """

    triangle_id: int
    barycentric: ArrayLike
    rotation_about_normal_deg: float
    size: ArrayLike


@dataclass(frozen=True)
class SurfaceFrame:
    """World-space frame for a sticker anchored on a triangle surface.

    Attributes:
        anchor: World-space center point for the sticker.
        normal: Unit surface normal.
        tangent: Unit tangent direction before user rotation.
        bitangent: Unit bitangent direction before user rotation.
        rotated_tangent: Unit tangent direction after rotation around normal.
        rotated_bitangent: Unit bitangent direction after rotation around normal.
    """

    anchor: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    bitangent: np.ndarray
    rotated_tangent: np.ndarray
    rotated_bitangent: np.ndarray


@dataclass(frozen=True)
class TextureStampResult:
    """Result of baking a sticker into a parent texture.

    Attributes:
        texture_path: Path to the generated stamped texture.
        uv_center: Sticker center in UV coordinates.
        uv_corners: Four sticker corners in UV coordinates.
        texture_size: Generated texture size as width and height in pixels.
        target_object_name: Name of the Blender object that received the stamp.
        target_material_index: Material slot index used for the stamp.
        target_material_name: Name of the material used for the stamp.
        target_uv_layer: Name of the UV layer used for placement.
        coverage_ratio: Fraction of the sticker covered by the target triangle
            in UV space.
    """

    texture_path: Path
    uv_center: np.ndarray
    uv_corners: np.ndarray
    texture_size: tuple[int, int]
    target_object_name: str
    target_material_index: int
    target_material_name: str
    target_uv_layer: str
    coverage_ratio: float
