from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike


def to_blender_vector(value: ArrayLike) -> Any:
    """Return a Blender Vector when available.

    Args:
        value: Array-like value containing three coordinates.

    Returns:
        A mathutils.Vector inside Blender, or a coordinate tuple when mathutils
        is unavailable in unit tests.
    """
    coordinates = tuple(float(coordinate) for coordinate in value)
    try:
        from mathutils import Vector  # type: ignore[import-not-found]

        return Vector(coordinates)
    except Exception:
        return coordinates
