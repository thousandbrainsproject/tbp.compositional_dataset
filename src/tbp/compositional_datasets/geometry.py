from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tbp.compositional_datasets.placements import SurfaceFrame

DEFAULT_TOLERANCE = 1e-12


def _normalize(v: ArrayLike, epsilon: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """Return a unit-length copy of a vector.

    Args:
        v: Array-like vector to normalize.
        epsilon: Minimum allowed vector norm.

    Returns:
        The normalized vector as a NumPy array.

    Raises:
        ValueError: If v has a norm smaller than epsilon.
    """
    vector = np.asarray(v)
    norm = np.linalg.norm(vector)
    if norm < epsilon:
        raise ValueError(f"Cannot normalize near-zero vector (norm={norm:.2e})")
    return vector / norm


def triangle_area(triangle: ArrayLike) -> float:
    """Compute the area of a 3D triangle.

    Args:
        triangle: Array-like value with shape (3, 3) containing triangle
            vertices.

    Returns:
        The triangle area.

    Raises:
        ValueError: If the triangle is invalid.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    edge_a = triangle_array[1] - triangle_array[0]
    edge_b = triangle_array[2] - triangle_array[0]
    return float(0.5 * np.linalg.norm(np.cross(edge_a, edge_b)))


def is_degenerate_triangle(
    triangle: ArrayLike,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return whether a triangle has a collapsed or ill-conditioned basis.

    Args:
        triangle: Array-like value with shape (3, 3) containing triangle
            vertices.
        tolerance: Relative tolerance for the triangle edge basis.

    Returns:
        True when the triangle basis is collapsed or numerically degenerate.

    Raises:
        ValueError: If the triangle is invalid.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    edge_a = triangle_array[1] - triangle_array[0]
    edge_b = triangle_array[2] - triangle_array[0]
    d00 = float(np.dot(edge_a, edge_a))
    d01 = float(np.dot(edge_a, edge_b))
    d11 = float(np.dot(edge_b, edge_b))
    scale = d00 * d11
    if scale <= tolerance:
        return True
    denominator = d00 * d11 - d01 * d01
    return abs(denominator) <= tolerance * scale


def triangle_normal(triangle: ArrayLike) -> np.ndarray:
    """Compute the unit normal vector for a triangle.

    Args:
        triangle: Array-like value with shape (3, 3) containing triangle
            vertices.

    Returns:
        A three-element unit normal vector following the triangle winding order.

    Raises:
        ValueError: If the triangle is invalid or degenerate.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    edge_a = triangle_array[1] - triangle_array[0]
    edge_b = triangle_array[2] - triangle_array[0]
    return _normalize(np.cross(edge_a, edge_b))


def tangent_bitangent_frame(triangle: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal surface frame from a triangle.

    Args:
        triangle: Array-like value with shape (3, 3) containing triangle
            vertices.

    Returns:
        A tuple (normal, tangent, bitangent) of three-element unit vectors.

    Raises:
        ValueError: If the triangle is invalid or cannot produce a
            non-degenerate tangent direction.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    normal = triangle_normal(triangle_array)
    candidate_edges = (
        triangle_array[1] - triangle_array[0],
        triangle_array[2] - triangle_array[1],
        triangle_array[0] - triangle_array[2],
    )
    tangent = None
    for edge in candidate_edges:
        projected = edge - normal * np.dot(edge, normal)
        if np.linalg.norm(projected) > DEFAULT_TOLERANCE:
            tangent = _normalize(projected)
            break
    if tangent is None:
        raise ValueError("triangle must contain at least one non-degenerate edge")
    bitangent = _normalize(np.cross(normal, tangent))
    return normal, tangent, bitangent


def rotate_tangent_frame(
    tangent: ArrayLike,
    bitangent: ArrayLike,
    normal: ArrayLike,
    rotation_about_normal_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate tangent and bitangent vectors around a surface normal.

    Args:
        tangent: Tangent vector in the surface plane.
        bitangent: Bitangent vector in the surface plane.
        normal: Surface normal vector.
        rotation_about_normal_deg: Rotation angle in degrees.

    Returns:
        A tuple (rotated_tangent, rotated_bitangent) of normalized vectors
        projected back into the plane orthogonal to normal.

    Raises:
        ValueError: If any input vector is near zero.
    """
    tangent_array = _normalize(tangent)
    bitangent_array = _normalize(bitangent)
    normal_array = _normalize(normal)
    radians = np.deg2rad(rotation_about_normal_deg)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)

    rotated_tangent = cos_theta * tangent_array + sin_theta * bitangent_array
    rotated_bitangent = -sin_theta * tangent_array + cos_theta * bitangent_array

    rotated_tangent = rotated_tangent - normal_array * np.dot(rotated_tangent, normal_array)
    rotated_bitangent = rotated_bitangent - normal_array * np.dot(rotated_bitangent, normal_array)
    return (
        _normalize(rotated_tangent),
        _normalize(rotated_bitangent),
    )


def surface_frame(
    triangle: ArrayLike,
    barycentric: ArrayLike,
    rotation_about_normal_deg: float,
) -> SurfaceFrame:
    """Create a placement frame anchored on a triangle surface.

    Args:
        triangle: Array-like value with shape (3, 3) containing triangle
            vertices.
        barycentric: Array-like value containing three barycentric coordinates
            for the frame anchor.
        rotation_about_normal_deg: Rotation angle in degrees for the tangent
            and bitangent vectors.

    Returns:
        A SurfaceFrame with anchor, unrotated frame vectors, and rotated
        tangent axes.

    Raises:
        ValueError: If the triangle, barycentric coordinates, or frame vectors
            are invalid.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    barycentric_array = np.asarray(barycentric, dtype=float)
    normal, tangent, bitangent = tangent_bitangent_frame(triangle_array)
    rotated_tangent, rotated_bitangent = rotate_tangent_frame(
        tangent,
        bitangent,
        normal,
        rotation_about_normal_deg,
    )
    return SurfaceFrame(
        anchor=barycentric_array @ triangle_array,
        normal=normal,
        tangent=tangent,
        bitangent=bitangent,
        rotated_tangent=rotated_tangent,
        rotated_bitangent=rotated_bitangent,
    )


def sticker_quad_corners(
    center: ArrayLike,
    tangent: ArrayLike,
    bitangent: ArrayLike,
    size: ArrayLike,
) -> np.ndarray:
    """Compute corners for a rectangular sticker in a tangent plane.

    Args:
        center: Center point of the sticker.
        tangent: Unit direction for the sticker width. The vector is normalized
            before use.
        bitangent: Unit direction for the sticker height. The vector is
            normalized before use.
        size: Array-like value containing (width, height).

    Returns:
        A (4, 3) array of corners ordered counter-clockwise in the
        tangent-bitangent plane.

    Raises:
        ValueError: If size is invalid or either frame vector is near zero.
    """
    center_array = np.asarray(center, dtype=float)
    width, height = np.asarray(size, dtype=float)
    tangent_array = _normalize(tangent)
    bitangent_array = _normalize(bitangent)
    half_tangent = tangent_array * (width / 2.0)
    half_bitangent = bitangent_array * (height / 2.0)
    return np.array(
        [
            center_array - half_tangent - half_bitangent,
            center_array + half_tangent - half_bitangent,
            center_array + half_tangent + half_bitangent,
            center_array - half_tangent + half_bitangent,
        ],
        dtype=float,
    )
