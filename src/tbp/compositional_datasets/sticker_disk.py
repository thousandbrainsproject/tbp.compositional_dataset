"""Deterministic geometry for circular sticker disk assets."""

from __future__ import annotations

from dataclasses import dataclass
import math

DEFAULT_DIAMETER = 0.02
DEFAULT_THICKNESS = 0.001
DEFAULT_SEGMENTS = 64
CAP_MATERIAL_INDEX = 0
SIDE_MATERIAL_INDEX = 1

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Face = tuple[int, ...]


@dataclass(frozen=True)
class StickerDiskMesh:
    """Pure mesh data used to construct one Blender sticker disk.

    Attributes:
        vertices: Object-space XYZ coordinates.
        faces: Vertex indices for cap triangles and side quads.
        face_uvs: Per-face UV coordinates in face-loop order.
        material_indices: Material slot index for every face.
    """

    vertices: tuple[Vector3, ...]
    faces: tuple[Face, ...]
    face_uvs: tuple[tuple[Vector2, ...], ...]
    material_indices: tuple[int, ...]


def _validate_parameters(diameter: float, thickness: float, segments: int) -> None:
    """Validate requested disk geometry.

    Args:
        diameter: Disk diameter in Blender units.
        thickness: Disk thickness in Blender units.
        segments: Number of perimeter segments.

    Raises:
        ValueError: If a dimension is non-positive or segments is below three.
    """
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if segments < 3:
        raise ValueError("segments must be at least 3")


def _cap_uv(angle: float, *, mirror_u: bool = False) -> Vector2:
    """Return a circular cap UV for one perimeter angle.

    Args:
        angle: Perimeter angle in radians.
        mirror_u: Whether to mirror U for an upright back-face view.

    Returns:
        UV coordinate inside the unit square.
    """
    cosine = math.cos(angle)
    u = 0.5 - 0.5 * cosine if mirror_u else 0.5 + 0.5 * cosine
    return (u, 0.5 + 0.5 * math.sin(angle))


def _build_face_uvs(angles: tuple[float, ...]) -> tuple[tuple[Vector2, ...], ...]:
    """Build UVs in the same order as generated polygon loops.

    Args:
        angles: Ordered perimeter angles.

    Returns:
        Per-loop UV coordinates for front caps, back caps, and side quads.
    """
    segments = len(angles)
    front = tuple(
        (
            (0.5, 0.5),
            _cap_uv(angles[index]),
            _cap_uv(angles[(index + 1) % segments]),
        )
        for index in range(segments)
    )
    back = tuple(
        (
            (0.5, 0.5),
            _cap_uv(angles[(index + 1) % segments], mirror_u=True),
            _cap_uv(angles[index], mirror_u=True),
        )
        for index in range(segments)
    )
    side = tuple(
        (
            (index / segments, 1.0),
            (index / segments, 0.0),
            ((index + 1) / segments, 0.0),
            ((index + 1) / segments, 1.0),
        )
        for index in range(segments)
    )
    return (*front, *back, *side)


def build_sticker_disk_mesh(
    *,
    diameter: float = DEFAULT_DIAMETER,
    thickness: float = DEFAULT_THICKNESS,
    segments: int = DEFAULT_SEGMENTS,
) -> StickerDiskMesh:
    """Build circular cap and side-wall mesh data.

    Args:
        diameter: Disk diameter in Blender units.
        thickness: Disk thickness in Blender units.
        segments: Number of perimeter segments.

    Returns:
        Immutable mesh, UV, and material-assignment data.

    Raises:
        ValueError: If the requested geometry is invalid.
    """
    _validate_parameters(diameter, thickness, segments)
    radius = diameter / 2.0
    half_thickness = thickness / 2.0
    angles = tuple(2.0 * math.pi * index / segments for index in range(segments))
    front_ring = tuple(
        (radius * math.cos(angle), radius * math.sin(angle), half_thickness)
        for angle in angles
    )
    back_ring = tuple((x, y, -half_thickness) for x, y, _z in front_ring)
    vertices = (
        (0.0, 0.0, half_thickness),
        *front_ring,
        (0.0, 0.0, -half_thickness),
        *back_ring,
    )

    front_center = 0
    front_start = 1
    back_center = segments + 1
    back_start = segments + 2
    front_faces = tuple(
        (front_center, front_start + index, front_start + (index + 1) % segments)
        for index in range(segments)
    )
    back_faces = tuple(
        (back_center, back_start + (index + 1) % segments, back_start + index)
        for index in range(segments)
    )
    side_faces = tuple(
        (
            front_start + index,
            back_start + index,
            back_start + (index + 1) % segments,
            front_start + (index + 1) % segments,
        )
        for index in range(segments)
    )
    faces = (*front_faces, *back_faces, *side_faces)
    material_indices = (CAP_MATERIAL_INDEX,) * (2 * segments) + (
        SIDE_MATERIAL_INDEX,
    ) * segments
    return StickerDiskMesh(
        vertices=vertices,
        faces=faces,
        face_uvs=_build_face_uvs(angles),
        material_indices=material_indices,
    )
