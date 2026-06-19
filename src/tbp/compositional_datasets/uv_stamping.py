from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

from tbp.compositional_datasets.geometry import (
    DEFAULT_TOLERANCE,
    is_degenerate_triangle,
    surface_frame,
)


class TextureStampError(ValueError):
    """Raised when a texture-stamp placement cannot be represented safely."""


@dataclass(frozen=True)
class TextureStampUVs:
    """UV footprint for a sticker stamp on a triangle.

    Attributes:
        uv_center: Sticker center in UV coordinates.
        uv_corners: Four sticker corners in UV coordinates.
    """

    uv_center: np.ndarray
    uv_corners: np.ndarray


@dataclass(frozen=True)
class StampedTexture:
    """RGBA texture produced by compositing a sticker.

    Attributes:
        image: Stamped RGBA image as a uint8 array.
        coverage_ratio: Fraction of the sticker footprint inside the 0..1 UV
            square.
    """

    image: np.ndarray
    coverage_ratio: float


def _polygon_area(points: np.ndarray, *, signed: bool = False) -> float:
    """Return the area of a 2D polygon.

    Args:
        points: Polygon vertices as an (n, 2) array.
        signed: Whether to preserve orientation in the returned area.

    Returns:
        The polygon area, or 0.0 for fewer than three points. The value is
        non-negative unless signed is true.
    """
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
    return area if signed else abs(area)


def _clip_polygon_axis(
    polygon: list[np.ndarray],
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[np.ndarray]:
    """Clip a polygon against one axis-aligned half-plane.

    Args:
        polygon: Polygon vertices as 2D points.
        axis: Coordinate axis to clip, with 0 for u or x and 1 for v or y.
        bound: Axis value defining the clipping line.
        keep_greater: Whether to keep points greater than or equal to bound.

    Returns:
        The clipped polygon vertices.
    """
    if not polygon:
        return []

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return point[axis] >= bound - DEFAULT_TOLERANCE
        return point[axis] <= bound + DEFAULT_TOLERANCE

    clipped: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(float(delta)) > DEFAULT_TOLERANCE:
                t = (bound - previous[axis]) / delta
                clipped.append(previous + t * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def _clip_polygon_to_unit_square(points: np.ndarray) -> np.ndarray:
    """Clip a 2D polygon to the 0..1 UV square.

    Args:
        points: Polygon vertices as an (n, 2) array.

    Returns:
        A clipped polygon array, or an empty (0, 2) array if no area remains.
    """
    polygon = [np.asarray(point, dtype=float) for point in points]
    for axis, bound, keep_greater in (
        (0, 0.0, True),
        (0, 1.0, False),
        (1, 0.0, True),
        (1, 1.0, False),
    ):
        polygon = _clip_polygon_axis(polygon, axis, bound, keep_greater)
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=float)
    return np.array(polygon, dtype=float)


def _line_intersection(
    point_a: np.ndarray,
    point_b: np.ndarray,
    clip_a: np.ndarray,
    clip_b: np.ndarray,
) -> np.ndarray:
    """Find where a segment intersects a clipping line.

    Args:
        point_a: First endpoint of the subject segment.
        point_b: Second endpoint of the subject segment.
        clip_a: First point on the clipping line.
        clip_b: Second point on the clipping line.

    Returns:
        The intersection point, or point_b when the lines are nearly parallel.
    """
    segment = point_b - point_a
    clip_segment = clip_b - clip_a
    denominator = float(segment[0] * clip_segment[1] - segment[1] * clip_segment[0])
    if abs(denominator) <= DEFAULT_TOLERANCE:
        return point_b
    offset = clip_a - point_a
    t = float(offset[0] * clip_segment[1] - offset[1] * clip_segment[0]) / denominator
    return point_a + t * segment


def _convex_intersection_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Return the intersection polygon for two convex 2D polygons.

    Args:
        subject: Convex polygon to clip.
        clip: Convex polygon used as the clipping boundary.

    Returns:
        The intersection polygon, or an empty (0, 2) array if there is no
        positive-area intersection.
    """
    if len(subject) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=float)

    clip_orientation = 1.0 if _polygon_area(clip, signed=True) >= 0.0 else -1.0
    output = [np.asarray(point, dtype=float) for point in subject]
    for index, clip_a in enumerate(clip):
        clip_b = clip[(index + 1) % len(clip)]
        edge = clip_b - clip_a
        input_points = output
        output = []
        if not input_points:
            break

        def inside(point: np.ndarray) -> bool:
            point_delta = point - clip_a
            cross = float(edge[0] * point_delta[1] - edge[1] * point_delta[0])
            return clip_orientation * cross >= -DEFAULT_TOLERANCE

        previous = input_points[-1]
        previous_inside = inside(previous)
        for current in input_points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                output.append(_line_intersection(previous, current, clip_a, clip_b))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside

    if len(output) < 3:
        return np.empty((0, 2), dtype=float)
    return np.array(output, dtype=float)


def uv_stamp_overlaps_existing(
    existing_uv_corners: list[np.ndarray],
    uv_corners: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return whether a stamp UV footprint overlaps an earlier stamp.

    Args:
        existing_uv_corners: Earlier stamp UV quads.
        uv_corners: Candidate stamp UV quad.
        tolerance: Minimum intersection area considered overlapping.

    Returns:
        True when the candidate overlaps an existing stamp footprint.
    """
    for existing in existing_uv_corners:
        if _polygon_area(_convex_intersection_polygon(existing, uv_corners)) > tolerance:
            return True
    return False


def uv_polygons_overlap_in_stamp(
    uv_polygons: list[ArrayLike],
    stamp_uv_corners: ArrayLike,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return whether any UV polygons overlap inside a stamp footprint.

    Args:
        uv_polygons: Convex UV polygons to test.
        stamp_uv_corners: Four UV corners of the stamp footprint.
        tolerance: Minimum area considered a real overlap.

    Returns:
        True when at least two polygons intersect the stamp and overlap each
        other with area greater than tolerance.

    Raises:
        ValueError: If the stamp or UV polygons have invalid shapes or
            non-finite values.
    """
    stamp = np.asarray(stamp_uv_corners, dtype=float)
    if stamp.shape != (4, 2) or not np.all(np.isfinite(stamp)):
        raise ValueError("stamp_uv_corners must have shape (4, 2) and be finite")

    intersecting: list[np.ndarray] = []
    for polygon in uv_polygons:
        uv_polygon = np.asarray(polygon, dtype=float)
        if uv_polygon.ndim != 2 or uv_polygon.shape[1] != 2:
            raise ValueError("uv polygons must have shape (n, 2)")
        if not np.all(np.isfinite(uv_polygon)):
            raise ValueError("uv polygons must contain only finite values")
        if _polygon_area(_convex_intersection_polygon(uv_polygon, stamp)) > tolerance:
            intersecting.append(uv_polygon)

    for index, polygon in enumerate(intersecting):
        for other in intersecting[index + 1 :]:
            if _polygon_area(_convex_intersection_polygon(polygon, other)) > tolerance:
                return True
    return False


def barycentric_coordinates_3d(triangle: ArrayLike, point: ArrayLike) -> np.ndarray:
    """Return barycentric coordinates for a point in a triangle plane.

    Args:
        triangle: Triangle vertices with shape (3, 3).
        point: 3D point to project onto the triangle basis.

    Returns:
        Three barycentric coordinates for point.

    Raises:
        ValueError: If triangle or point is invalid, or if the triangle is
            degenerate.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    try:
        point_array = np.asarray(point, dtype=float)
    except ValueError as error:
        raise ValueError("point must be a finite 3D coordinate") from error

    a, b, c = triangle_array
    v0 = b - a
    v1 = c - a
    v2 = point_array - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denominator = d00 * d11 - d01 * d01
    if is_degenerate_triangle(triangle_array):
        raise ValueError("triangle is degenerate")
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=float)


def _world_delta_to_uv_delta(
    triangle: np.ndarray,
    triangle_uvs: np.ndarray,
    world_delta: np.ndarray,
) -> np.ndarray:
    """Map a world-space triangle-plane delta into UV-space delta.

    Args:
        triangle: Triangle vertices with shape (3, 3).
        triangle_uvs: Triangle UV coordinates with shape (3, 2).
        world_delta: 3D vector lying in the triangle plane.

    Returns:
        A two-element UV-space delta.

    Raises:
        ValueError: If the triangle is degenerate.
    """
    edge_matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    coefficients, _, rank, _ = np.linalg.lstsq(edge_matrix, world_delta, rcond=None)
    if rank < 2:
        raise ValueError("triangle is degenerate")
    uv_matrix = np.column_stack(
        (triangle_uvs[1] - triangle_uvs[0], triangle_uvs[2] - triangle_uvs[0])
    )
    return uv_matrix @ coefficients


def texture_stamp_uvs(
    triangle: ArrayLike,
    triangle_uvs: ArrayLike,
    *,
    barycentric: ArrayLike,
    rotation_about_normal_deg: float,
    size: ArrayLike,
) -> TextureStampUVs:
    """Convert a world-space sticker placement on one triangle to UV corners.

    Args:
        triangle: Triangle vertices with shape (3, 3).
        triangle_uvs: Triangle UV coordinates with shape (3, 2).
        barycentric: Three barycentric coordinates for the sticker center.
        rotation_about_normal_deg: Rotation angle around the triangle normal in
            degrees.
        size: Sticker width and height in world-space units.

    Returns:
        UV center and four UV corners for the sticker footprint.

    Raises:
        ValueError: If the triangle, size, barycentric coordinates, or frame is
            invalid.
        TextureStampError: If the UV triangle is degenerate.
    """
    triangle_array = np.asarray(triangle, dtype=float)
    uv_array = np.asarray(triangle_uvs, dtype=float)
    if _polygon_area(uv_array) <= DEFAULT_TOLERANCE:
        raise TextureStampError("degenerate UV triangle cannot receive a texture stamp")
    width, height = np.asarray(size, dtype=float)
    frame = surface_frame(triangle_array, barycentric, rotation_about_normal_deg)

    half_width_uv = _world_delta_to_uv_delta(
        triangle_array,
        uv_array,
        frame.rotated_tangent * (width / 2.0),
    )
    half_height_uv = _world_delta_to_uv_delta(
        triangle_array,
        uv_array,
        frame.rotated_bitangent * (height / 2.0),
    )
    uv_center = np.asarray(barycentric, dtype=float) @ uv_array
    uv_corners = np.array(
        [
            uv_center - half_width_uv - half_height_uv,
            uv_center + half_width_uv - half_height_uv,
            uv_center + half_width_uv + half_height_uv,
            uv_center - half_width_uv + half_height_uv,
        ],
        dtype=float,
    )
    return TextureStampUVs(uv_center=uv_center, uv_corners=uv_corners)


def coverage_ratio(uv_corners: ArrayLike) -> float:
    """Return the fraction of a stamp footprint inside the 0..1 UV square.

    Args:
        uv_corners: Four UV corners of the stamp footprint.

    Returns:
        A value from 0.0 to 1.0.

    Raises:
        ValueError: If uv_corners does not have shape (4, 2) or contains
            non-finite values.
        TextureStampError: If the stamp footprint has near-zero area.
    """
    corners = np.asarray(uv_corners, dtype=float)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("uv_corners must have shape (4, 2) and be finite")
    stamp_area = _polygon_area(corners)
    if stamp_area <= DEFAULT_TOLERANCE:
        raise TextureStampError("texture stamp UV footprint has zero area")
    clipped_area = _polygon_area(_clip_polygon_to_unit_square(corners))
    return min(1.0, max(0.0, clipped_area / stamp_area))


def _uv_to_pixel_points(uv_corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert UV corners to pixel-space points.

    Args:
        uv_corners: Four UV corners.
        width: Texture width in pixels.
        height: Texture height in pixels.

    Returns:
        A (4, 2) array of pixel coordinates.
    """
    return np.column_stack(
        (
            uv_corners[:, 0] * width,
            (1.0 - uv_corners[:, 1]) * height,
        )
    )


def _alpha_over(destination: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Composite one RGBA source pixel over one destination pixel.

    Args:
        destination: Destination RGBA pixel.
        source: Source RGBA pixel.

    Returns:
        The composited uint8 RGBA pixel.
    """
    src = source.astype(float) / 255.0
    dst = destination.astype(float) / 255.0
    src_alpha = src[3]
    dst_alpha = dst[3]
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    if out_alpha <= 0.0:
        return np.zeros(4, dtype=np.uint8)
    out_rgb = (src[:3] * src_alpha + dst[:3] * dst_alpha * (1.0 - src_alpha)) / out_alpha
    return np.round(np.concatenate((out_rgb, [out_alpha])) * 255.0).astype(np.uint8)


def stamp_rgba_texture(
    base_rgba: ArrayLike,
    sticker_rgba: ArrayLike,
    uv_corners: ArrayLike,
    *,
    min_coverage: float = 0.95,
) -> StampedTexture:
    """Alpha-composite a sticker image into a base RGBA texture.

    Args:
        base_rgba: Base texture image with shape (height, width, 4).
        sticker_rgba: Sticker image with shape (height, width, 4).
        uv_corners: Four UV corners defining the sticker footprint.
        min_coverage: Minimum required fraction of the sticker footprint inside
            the 0..1 UV square.

    Returns:
        The stamped texture image and measured coverage ratio.

    Raises:
        ValueError: If either image or uv_corners is invalid.
        TextureStampError: If coverage is below min_coverage or the pixel
            footprint has near-zero area.
    """
    base = np.asarray(base_rgba)
    if base.ndim != 3 or base.shape[2] != 4:
        raise ValueError("base_rgba must have shape (height, width, 4)")
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)
    base = base.copy()

    sticker = np.asarray(sticker_rgba)
    if sticker.ndim != 3 or sticker.shape[2] != 4:
        raise ValueError("sticker_rgba must have shape (height, width, 4)")
    if sticker.dtype != np.uint8:
        sticker = np.clip(sticker, 0, 255).astype(np.uint8)
    corners = np.asarray(uv_corners, dtype=float)
    ratio = coverage_ratio(corners)
    if ratio < min_coverage:
        raise TextureStampError(
            f"texture stamp coverage {ratio:.3f} is below required {min_coverage:.3f}"
        )

    texture_height, texture_width, _ = base.shape
    sticker_height, sticker_width, _ = sticker.shape
    pixel_corners = _uv_to_pixel_points(corners, texture_width, texture_height)
    origin = pixel_corners[0]
    width_axis = pixel_corners[1] - pixel_corners[0]
    height_axis = pixel_corners[3] - pixel_corners[0]
    transform = np.column_stack((width_axis, height_axis))
    determinant = float(np.linalg.det(transform))
    if abs(determinant) <= DEFAULT_TOLERANCE:
        raise TextureStampError("texture stamp pixel footprint has zero area")
    inverse_transform = np.linalg.inv(transform)

    min_x = max(0, int(math.floor(float(np.min(pixel_corners[:, 0])))))
    max_x = min(texture_width, int(math.ceil(float(np.max(pixel_corners[:, 0])))))
    min_y = max(0, int(math.floor(float(np.min(pixel_corners[:, 1])))))
    max_y = min(texture_height, int(math.ceil(float(np.max(pixel_corners[:, 1])))))

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            local = inverse_transform @ (np.array([x + 0.5, y + 0.5], dtype=float) - origin)
            u, v = local
            if u < -DEFAULT_TOLERANCE or u > 1.0 + DEFAULT_TOLERANCE:
                continue
            if v < -DEFAULT_TOLERANCE or v > 1.0 + DEFAULT_TOLERANCE:
                continue
            sticker_x = min(sticker_width - 1, max(0, int(u * sticker_width)))
            sticker_y = min(sticker_height - 1, max(0, int((1.0 - v) * sticker_height)))
            base[y, x] = _alpha_over(base[y, x], sticker[sticker_y, sticker_x])

    return StampedTexture(image=base, coverage_ratio=ratio)
