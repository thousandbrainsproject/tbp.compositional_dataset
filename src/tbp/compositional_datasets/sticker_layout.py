from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from tbp.compositional_datasets.geometry import rotate_tangent_frame, tangent_bitangent_frame

Side = Literal["front", "back"]

DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_GAP = 0.004
DEFAULT_MAX_LONGEST_SIDE = 0.021
DEFAULT_MIN_LONGEST_SIDE = 0.01
DEFAULT_PLACEMENT_PROFILE = {"right_axis": "x", "up_axis": "z", "front_axis": "y"}
DEFAULT_TRIAD_OFFSETS = (
    ("front_top", "front", (0.0, 0.018)),
    ("front_left", "front", (-0.018, -0.012)),
    ("front_right", "front", (0.018, -0.012)),
    ("back_top", "back", (0.0, 0.018)),
    ("back_left", "back", (-0.018, -0.012)),
    ("back_right", "back", (0.018, -0.012)),
)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PlacementProfile:
    """Axis profile for automatic object-space placement.

    Attributes:
        right_axis: Coordinate axis used for horizontal slot offsets.
        up_axis: Coordinate axis used for vertical slot offsets.
        front_axis: Coordinate axis used for front/back ray casts.
    """

    right_axis: int
    up_axis: int
    front_axis: int


DEFAULT_PLACEMENT = PlacementProfile(right_axis=0, up_axis=2, front_axis=1)


@dataclass(frozen=True)
class StickerSlot:
    """One configured sticker slot.

    Attributes:
        name: Unique slot name.
        side: Object side to stamp, either `front` or `back`.
        offset: Object-space right/up offset from the object bounds center.
        sticker: PNG sticker path.
        rotation_deg: Extra rotation around the surface normal in degrees.
    """

    name: str
    side: Side
    offset: tuple[float, float]
    sticker: Path
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class StickerLayoutConfig:
    """Configuration for a multi-sticker layout.

    Attributes:
        min_coverage: Minimum allowed coverage for each stamp footprint.
        gap: Minimum requested center-spacing reserve in object units.
        max_longest_side: Upper bound for each sticker's longest side.
        min_longest_side: Lower bound below which the layout is rejected.
        placement: Axis profile for ray casting and footprint projection.
        slots: Six configured sticker slots.
    """

    min_coverage: float
    gap: float
    max_longest_side: float
    min_longest_side: float
    placement: PlacementProfile
    slots: tuple[StickerSlot, ...]


def _slot_from_mapping(raw_slot: Any, config_path: Path) -> StickerSlot:
    """Parse one config slot.

    Args:
        raw_slot: Raw JSON mapping.
        config_path: Config file path.

    Returns:
        Parsed sticker slot.

    Raises:
        ValueError: If the slot is invalid.
        FileNotFoundError: If the sticker path does not exist.
    """
    if not isinstance(raw_slot, dict):
        raise ValueError("each slot must be an object")
    name = str(raw_slot.get("name", ""))
    if not name:
        raise ValueError("slot name must be non-empty")
    side = raw_slot.get("side")
    if side not in {"front", "back"}:
        raise ValueError("side must be 'front' or 'back'")
    if "sticker" not in raw_slot:
        raise ValueError("slot sticker is required")

    offset = raw_slot.get("offset")
    if not isinstance(offset, list | tuple) or len(offset) != 2:
        raise ValueError("offset must contain exactly 2 values")
    right_offset = float(offset[0])
    if not math.isfinite(right_offset):
        raise ValueError("offset[0] must be finite")
    up_offset = float(offset[1])
    if not math.isfinite(up_offset):
        raise ValueError("offset[1] must be finite")

    raw_sticker_path = str(raw_slot["sticker"])
    sticker_path = Path(raw_sticker_path)
    sticker_candidates = [sticker_path]
    if not sticker_path.is_absolute():
        sticker_candidates.append(config_path.parent / sticker_path)
    for candidate in sticker_candidates:
        if candidate.exists():
            sticker_path = candidate
            break
    else:
        raise FileNotFoundError(f"Sticker image not found: {raw_sticker_path}")

    rotation_deg = float(raw_slot.get("rotation_deg", 0.0))
    if not math.isfinite(rotation_deg):
        raise ValueError("rotation_deg must be finite")

    return StickerSlot(
        name=name,
        side=side,
        offset=(right_offset, up_offset),
        sticker=sticker_path,
        rotation_deg=rotation_deg,
    )


def load_sticker_layout_config(path: Path) -> StickerLayoutConfig:
    """Load and validate a sticker layout JSON config.

    Args:
        path: Config JSON path.

    Returns:
        Parsed sticker layout config.

    Raises:
        ValueError: If config fields are invalid.
        FileNotFoundError: If a sticker path does not exist.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("sticker layout config must be a JSON object")
    slots = tuple(_slot_from_mapping(raw_slot, path) for raw_slot in data.get("slots", []))
    if len(slots) != 6:
        raise ValueError("sticker layout config must contain exactly six slots")
    names: set[str] = set()
    for slot in slots:
        if slot.name in names:
            raise ValueError(f"duplicate slot name: {slot.name}")
        names.add(slot.name)

    min_coverage = float(data.get("min_coverage", DEFAULT_MIN_COVERAGE))
    if not math.isfinite(min_coverage):
        raise ValueError("min_coverage must be finite")
    if min_coverage <= 0.0:
        raise ValueError("min_coverage must be positive")
    if min_coverage > 1.0:
        raise ValueError("min_coverage must be no greater than 1")

    gap = float(data.get("gap", DEFAULT_GAP))
    if not math.isfinite(gap):
        raise ValueError("gap must be finite")
    if gap <= 0.0:
        raise ValueError("gap must be positive")

    max_longest_side = float(data.get("max_longest_side", DEFAULT_MAX_LONGEST_SIDE))
    if not math.isfinite(max_longest_side):
        raise ValueError("max_longest_side must be finite")
    if max_longest_side <= 0.0:
        raise ValueError("max_longest_side must be positive")

    min_longest_side = float(data.get("min_longest_side", DEFAULT_MIN_LONGEST_SIDE))
    if not math.isfinite(min_longest_side):
        raise ValueError("min_longest_side must be finite")
    if min_longest_side <= 0.0:
        raise ValueError("min_longest_side must be positive")

    placement_data = data.get("placement", DEFAULT_PLACEMENT_PROFILE)
    if not isinstance(placement_data, dict):
        raise ValueError("placement must be an object")
    placement_axes = []
    for key in ("right_axis", "up_axis", "front_axis"):
        axis = str(placement_data.get(key)).lower()
        if axis not in AXIS_INDEX:
            raise ValueError(f"{key} must be one of x, y, or z")
        placement_axes.append(AXIS_INDEX[axis])
    placement = PlacementProfile(
        right_axis=placement_axes[0],
        up_axis=placement_axes[1],
        front_axis=placement_axes[2],
    )
    if len({placement.right_axis, placement.up_axis, placement.front_axis}) != 3:
        raise ValueError("placement axes must be unique")

    return StickerLayoutConfig(
        min_coverage=min_coverage,
        gap=gap,
        max_longest_side=max_longest_side,
        min_longest_side=min_longest_side,
        placement=placement,
        slots=slots,
    )


def default_slot_config(stickers: list[Path] | tuple[Path, ...]) -> StickerLayoutConfig:
    """Return the requested six-slot front/back triad config.

    Args:
        stickers: Six explicit sticker paths.

    Returns:
        Default sticker layout config.

    Raises:
        ValueError: If exactly six sticker paths are not provided.
    """
    if len(stickers) != 6:
        raise ValueError("default slot config requires exactly six stickers")
    slots = tuple(
        StickerSlot(name, side, offset, Path(stickers[index]), 0.0)
        for index, (name, side, offset) in enumerate(DEFAULT_TRIAD_OFFSETS)
    )
    return StickerLayoutConfig(
        min_coverage=DEFAULT_MIN_COVERAGE,
        gap=DEFAULT_GAP,
        max_longest_side=DEFAULT_MAX_LONGEST_SIDE,
        min_longest_side=DEFAULT_MIN_LONGEST_SIDE,
        placement=DEFAULT_PLACEMENT,
        slots=slots,
    )


def placement_ray(
    slot: StickerSlot,
    bounds_center: tuple[float, float, float],
    bounds_radius: float,
    profile: PlacementProfile,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the world-space ray for one automatic slot.

    Args:
        slot: Configured sticker slot.
        bounds_center: Object bounds center.
        bounds_radius: Object bounds radius.
        profile: Axis profile used to map slot offsets.

    Returns:
        Ray origin and direction.
    """
    origin = [float(value) for value in bounds_center]
    origin[profile.right_axis] = float(bounds_center[profile.right_axis]) + slot.offset[0]
    origin[profile.up_axis] = float(bounds_center[profile.up_axis]) + slot.offset[1]
    distance = max(float(bounds_radius) * 3.0, 0.5)
    direction = [0.0, 0.0, 0.0]
    if slot.side == "front":
        origin[profile.front_axis] = float(bounds_center[profile.front_axis]) - distance
        direction[profile.front_axis] = 1.0
        return (origin[0], origin[1], origin[2]), (direction[0], direction[1], direction[2])
    origin[profile.front_axis] = float(bounds_center[profile.front_axis]) + distance
    direction[profile.front_axis] = -1.0
    return (origin[0], origin[1], origin[2]), (direction[0], direction[1], direction[2])


def size_for_longest_side(path: Path, longest_side: float) -> tuple[float, float]:
    """Return aspect-preserving sticker size for a longest side.

    Args:
        path: Sticker image path.
        longest_side: Longest side in object units.

    Returns:
        Width and height in object units.
    """
    with Image.open(path) as image:
        width, height = image.size
    width = int(width)
    height = int(height)
    aspect_ratio = width / height
    if width >= height:
        return (float(longest_side), float(longest_side) / aspect_ratio)
    return (float(longest_side) * aspect_ratio, float(longest_side))


def compute_common_slot_sizes(
    slots: list[StickerSlot] | tuple[StickerSlot, ...],
    *,
    gap: float,
    max_longest_side: float,
    min_longest_side: float = DEFAULT_MIN_LONGEST_SIDE,
) -> dict[str, tuple[float, float]]:
    """Compute aspect-preserving sizes using one common longest side.

    Args:
        slots: Configured slots.
        gap: Reserved object-space gap between slot centers.
        max_longest_side: Upper bound for the common longest side.
        min_longest_side: Lower bound below which the layout is rejected.

    Returns:
        Mapping from slot name to sticker width and height.

    Raises:
        ValueError: If the configured centers cannot fit the minimum size.
    """
    longest_side = float(max_longest_side)
    for index, left in enumerate(slots):
        for right in slots[index + 1 :]:
            if left.side != right.side:
                continue
            dx = left.offset[0] - right.offset[0]
            dz = left.offset[1] - right.offset[1]
            center_distance = math.hypot(dx, dz)
            longest_side = min(longest_side, center_distance - float(gap))
    if longest_side < float(min_longest_side):
        raise ValueError("cannot fit stickers with the configured spacing and gap")
    return {slot.name: size_for_longest_side(slot.sticker, longest_side) for slot in slots}


def rotation_aligning_bitangent_to_world_up(triangle: Any, *, world_up: Any = (0.0, 0.0, 1.0)) -> float:
    """Return rotation that aligns sticker height with projected world up.

    Args:
        triangle: Triangle vertices with shape (3, 3).
        world_up: World up vector to project into the surface plane.

    Returns:
        Rotation around the surface normal in degrees.

    Raises:
        ValueError: If world up cannot be projected into the surface plane.
    """
    normal, tangent, bitangent = tangent_bitangent_frame(triangle)
    up = np.asarray(world_up, dtype=float)
    projected_up = up - normal * float(np.dot(up, normal))
    norm = np.linalg.norm(projected_up)
    if norm <= 1e-12:
        raise ValueError("world up is parallel to the target surface normal")
    target = projected_up / norm
    tangent_component = float(np.dot(tangent, target))
    bitangent_component = float(np.dot(bitangent, target))
    angle = math.degrees(math.atan2(-tangent_component, bitangent_component))
    _rotated_tangent, rotated_bitangent = rotate_tangent_frame(tangent, bitangent, normal, angle)
    if float(np.dot(rotated_bitangent, target)) < 0.0:
        angle += 180.0
    return angle


def _projected_bounds(quad: Any, profile: PlacementProfile) -> tuple[float, float, float, float]:
    """Return right/up bounds for a sticker quad.

    Args:
        quad: World-space quad corners with shape (4, 3).

    Returns:
        Minimum right, maximum right, minimum up, and maximum up.
    """
    array = np.asarray(quad, dtype=float)
    if array.shape != (4, 3):
        raise ValueError("sticker quad must have shape (4, 3)")
    return (
        float(np.min(array[:, profile.right_axis])),
        float(np.max(array[:, profile.right_axis])),
        float(np.min(array[:, profile.up_axis])),
        float(np.max(array[:, profile.up_axis])),
    )


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether two projected bounds overlap with positive area.

    Args:
        left: First bounds.
        right: Second bounds.
        tolerance: Minimum overlap considered positive.

    Returns:
        True when bounds overlap on both axes.
    """
    primary_overlap = min(left[1], right[1]) - max(left[0], right[0])
    secondary_overlap = min(left[3], right[3]) - max(left[2], right[2])
    return primary_overlap > tolerance and secondary_overlap > tolerance


def validate_same_side_footprints_do_not_overlap(
    footprints: list[tuple[str, Side, Any]] | tuple[tuple[str, Side, Any], ...],
    profile: PlacementProfile,
) -> None:
    """Reject overlapping world-space footprints on the same object side.

    Args:
        footprints: Tuples of slot name, side, and world-space quad corners.
        profile: Axis profile used to project sticker footprints.

    Raises:
        ValueError: If two same-side footprints overlap.
    """
    for index, (left_name, left_side, left_quad) in enumerate(footprints):
        left_bounds = _projected_bounds(left_quad, profile)
        for right_name, right_side, right_quad in footprints[index + 1 :]:
            if left_side != right_side:
                continue
            if _bounds_overlap(left_bounds, _projected_bounds(right_quad, profile)):
                raise ValueError(f"sticker footprints overlap: {left_name} and {right_name}")
