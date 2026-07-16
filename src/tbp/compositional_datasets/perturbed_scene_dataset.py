"""Pure helpers for producing paired, perturbed sticker layouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import math
import random
import re
from typing import Any


@dataclass(frozen=True)
class PerturbationBounds:
    """Bounds controlling independent sticker-position perturbations."""

    min_displacement: float = 0.002
    max_displacement: float = 0.006
    max_attempts: int = 10_000

    def __post_init__(self) -> None:
        """Validate perturbation bounds after initialization."""
        if not math.isfinite(self.min_displacement) or self.min_displacement <= 0:
            raise ValueError("min_displacement must be positive and finite")
        if not math.isfinite(self.max_displacement):
            raise ValueError("max_displacement must be finite")
        if self.max_displacement < self.min_displacement:
            raise ValueError("max_displacement must be at least min_displacement")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


def target_object_id(source_id: str, id_offset: int) -> str:
    """Offset a three-digit source object ID while preserving its suffix.

    Args:
        source_id: Source object ID beginning with three digits and an underscore.
        id_offset: Positive integer added to the numeric ID prefix.

    Returns:
        Target object ID with the offset numeric prefix.

    Raises:
        ValueError: If the source ID, offset, or resulting ID is invalid.
    """
    match = re.fullmatch(r"(\d{3})_(.+)", source_id)
    if match is None or id_offset <= 0:
        raise ValueError(f"invalid source ID or offset: {source_id}, {id_offset}")
    target_number = int(match.group(1)) + id_offset
    if target_number > 999:
        raise ValueError("target object ID exceeds three digits")
    return f"{target_number:03d}_{match.group(2)}"


def validate_perturbed_config(
    source: dict[str, Any],
    target: dict[str, Any],
    bounds: PerturbationBounds,
) -> None:
    """Validate a perturbed six-sticker configuration against its source.

    Args:
        source: Original sticker layout configuration.
        target: Candidate configuration with perturbed slot offsets.
        bounds: Allowed radial displacement and sampling bounds.

    Raises:
        ValueError: If non-offset data changed or a geometric invariant fails.
    """
    source_non_offsets = deepcopy(source)
    target_non_offsets = deepcopy(target)
    for config in (source_non_offsets, target_non_offsets):
        for slot in config["slots"]:
            slot.pop("offset", None)
    if source_non_offsets != target_non_offsets:
        raise ValueError("source and target differ outside slot offsets")

    tolerance = 1e-12
    for source_slot, target_slot in zip(source["slots"], target["slots"], strict=True):
        displacement = math.dist(source_slot["offset"], target_slot["offset"])
        if not (
            bounds.min_displacement - tolerance
            <= displacement
            <= bounds.max_displacement + tolerance
        ):
            raise ValueError("slot displacement is outside perturbation bounds")

    minimum_spacing = target["max_longest_side"] + target["gap"]
    sides = {slot["side"] for slot in source["slots"]}
    for side in sides:
        source_slots = {
            slot["name"].rsplit("_", maxsplit=1)[-1]: slot
            for slot in source["slots"]
            if slot["side"] == side
        }
        target_slots = {
            slot["name"].rsplit("_", maxsplit=1)[-1]: slot
            for slot in target["slots"]
            if slot["side"] == side
        }
        if set(source_slots) != {"top", "left", "right"}:
            raise ValueError(f"side {side} must contain top, left, and right slots")

        top = target_slots["top"]["offset"]
        left = target_slots["left"]["offset"]
        right = target_slots["right"]["offset"]
        if top[1] <= left[1] or top[1] <= right[1]:
            raise ValueError(f"top slot is not above both lower slots on side {side}")
        if left[0] >= right[0]:
            raise ValueError(f"left slot is not left of right slot on side {side}")

        source_top = source_slots["top"]["offset"]
        source_left = source_slots["left"]["offset"]
        source_right = source_slots["right"]["offset"]
        source_area = (source_left[0] - source_top[0]) * (
            source_right[1] - source_top[1]
        ) - (source_left[1] - source_top[1]) * (
            source_right[0] - source_top[0]
        )
        target_area = (left[0] - top[0]) * (right[1] - top[1]) - (
            left[1] - top[1]
        ) * (right[0] - top[0])
        if source_area == 0 or target_area == 0 or source_area * target_area < 0:
            raise ValueError(f"triangle orientation changed on side {side}")

        for first, second in combinations(target_slots.values(), 2):
            if math.dist(first["offset"], second["offset"]) < minimum_spacing:
                raise ValueError(f"slots overlap on side {side}")


def perturb_stamp_config(
    source: dict[str, Any],
    rng: random.Random,
    bounds: PerturbationBounds,
) -> dict[str, Any]:
    """Sample a valid independently perturbed offset for every sticker slot.

    Args:
        source: Original sticker layout configuration.
        rng: Seedable random-number generator.
        bounds: Allowed radial displacement and sampling bounds.

    Returns:
        Deep-copied configuration containing valid perturbed offsets.

    Raises:
        ValueError: If no valid candidate is found within the attempt limit.
    """
    last_error = "no candidate"
    for _attempt in range(bounds.max_attempts):
        candidate = deepcopy(source)
        for slot in candidate["slots"]:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(bounds.min_displacement, bounds.max_displacement)
            slot["offset"][0] += radius * math.cos(angle)
            slot["offset"][1] += radius * math.sin(angle)
        try:
            validate_perturbed_config(source, candidate, bounds)
        except ValueError as error:
            last_error = str(error)
            continue
        return candidate
    raise ValueError(
        f"could not sample a valid perturbed layout after {bounds.max_attempts} attempts: {last_error}"
    )
