from __future__ import annotations

import math
import random
from typing import Any

import pytest

from tbp.compositional_datasets.perturbed_scene_dataset import (
    PerturbationBounds,
    perturb_stamp_config,
    target_object_id,
    validate_perturbed_config,
)


@pytest.fixture
def source_config() -> dict[str, Any]:
    """Return a six-slot sticker layout on the front and back sides."""
    slots = []
    for side in ("front", "back"):
        slots.extend(
            [
                {"name": f"{side}_top", "side": side, "offset": [0.0, 0.018]},
                {"name": f"{side}_left", "side": side, "offset": [-0.018, -0.012]},
                {"name": f"{side}_right", "side": side, "offset": [0.018, -0.012]},
            ]
        )
    return {
        "min_coverage": 0.9,
        "gap": 0.004,
        "max_longest_side": 0.021,
        "placement": {"right_axis": "x", "up_axis": "z", "front_axis": "y"},
        "slots": slots,
    }


def test_target_id_adds_100_and_preserves_suffix():
    """Verify numeric ID offsets leave object-name suffixes unchanged."""
    assert target_object_id("101_cube_6x2d_stickers", 100) == "201_cube_6x2d_stickers"
    assert target_object_id("200_sphere_6x2d_stickers", 100) == "300_sphere_6x2d_stickers"


def test_seeded_perturbation_is_deterministic_and_changes_only_offsets(source_config):
    """Verify seeded perturbations are bounded and change only offsets."""
    bounds = PerturbationBounds(0.002, 0.006)
    first = perturb_stamp_config(source_config, random.Random(123), bounds)
    second = perturb_stamp_config(source_config, random.Random(123), bounds)
    assert first == second
    for source_slot, target_slot in zip(source_config["slots"], first["slots"], strict=True):
        assert {k: v for k, v in source_slot.items() if k != "offset"} == {
            k: v for k, v in target_slot.items() if k != "offset"
        }
        assert 0.002 <= math.dist(source_slot["offset"], target_slot["offset"]) <= 0.006
    validate_perturbed_config(source_config, first, bounds)


def test_impossible_spacing_reports_attempt_limit(source_config):
    """Verify impossible spacing reports the exhausted attempt limit."""
    source_config["max_longest_side"] = 1.0
    bounds = PerturbationBounds(0.002, 0.006, max_attempts=3)
    with pytest.raises(ValueError, match="after 3 attempts"):
        perturb_stamp_config(source_config, random.Random(7), bounds)
